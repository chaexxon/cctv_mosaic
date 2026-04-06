# db/sqlite.py
"""
SQLite DB access layer (A + C main, B reads as needed)

목표
- 스키마(models.sql) 기반 CRUD의 단일 진실
- 얼굴 임베딩(ArcFace) BLOB 저장/복원 정확성 보장 (float32, (D,))
- A: jobs/results CRUD 지원
- C: insert_face/list_faces 및 plate 등록여부 확인 지원
- 예외/트랜잭션/연결 안정성 확보 (종합설계용)

주의
- list_faces()는 IdentityMatcher가 기대하는 형태로 FaceRecord 리스트를 반환해야 함:
  FaceRecord.id / FaceRecord.name / FaceRecord.embedding(np.ndarray)
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple, Dict

import numpy as np

from core.config import DB_PATH, FACE_EMBED_DIM, PROJECT_ROOT


# =========================
# Errors
# =========================
class DBError(RuntimeError):
    """DB layer error (wrap sqlite3 errors / data corruption)."""


# =========================
# Records (typed return)
# =========================
@dataclass(frozen=True)
class FaceRecord:
    id: int
    name: str
    embedding: np.ndarray  # (D,), float32


@dataclass(frozen=True)
class PlateRecord:
    id: int
    owner: str
    plate_text_norm: str


@dataclass(frozen=True)
class JobRecord:
    id: int
    start_ts: str
    end_ts: str
    source: str
    status: str
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class ResultRecord:
    job_id: int
    output_path: str
    thumb_path: Optional[str]
    meta_json: Optional[str]
    created_at: str


# =========================
# Helpers: embedding <-> blob
# =========================
def _ensure_float32_1d(x: np.ndarray, embed_dim: int) -> np.ndarray:
    if not isinstance(x, np.ndarray):
        raise DBError(f"embedding must be np.ndarray, got {type(x)}")
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.size != int(embed_dim):
        raise DBError(f"embedding dim mismatch: got {x.size}, expected {embed_dim}")
    if not np.isfinite(x).all():
        raise DBError("embedding contains NaN/Inf")
    return x


def _embedding_to_blob(x: np.ndarray, embed_dim: int) -> bytes:
    x = _ensure_float32_1d(x, embed_dim)
    # float32 raw bytes (little-endian by numpy default)
    return x.tobytes(order="C")


def _blob_to_embedding(blob: bytes, embed_dim: int) -> np.ndarray:
    if blob is None:
        raise DBError("embedding_blob is NULL (corrupted record?)")
    arr = np.frombuffer(blob, dtype=np.float32)
    if arr.size != int(embed_dim):
        raise DBError(f"embedding_blob dim mismatch: got {arr.size}, expected {embed_dim}")
    # copy() to detach from sqlite buffer
    arr = arr.copy()
    if not np.isfinite(arr).all():
        raise DBError("embedding_blob contains NaN/Inf")
    return arr


# =========================
# SQLiteDB
# =========================
class SQLiteDB:
    """
    SQLite DB wrapper.

    - 파일 경로를 받으면 자동 생성/연결
    - 스키마 초기화는 init_db()에서 수행 (scripts/init_db.py에서 호출 권장)
    - 기본은 autocommit 아님: 각 메서드에서 트랜잭션 처리(원자성)
    """

    def __init__(self, db_path: Optional[str | Path] = None, *, embed_dim: int = FACE_EMBED_DIM):
        self.db_path = str(db_path) if db_path is not None else str(DB_PATH)
        self.embed_dim = int(embed_dim)

        # db 폴더 보장
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

    # ---------- connection ----------
    def _connect(self) -> sqlite3.Connection:
        try:
            conn = sqlite3.connect(
                self.db_path,
                detect_types=sqlite3.PARSE_DECLTYPES,
                check_same_thread=False,  # FastAPI 등 멀티스레드 대비(각 호출마다 새 커넥션)
            )
            conn.row_factory = sqlite3.Row
            # pragma
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.execute("PRAGMA journal_mode = WAL;")  # 동시성/안정성 개선(가능하면)
            conn.execute("PRAGMA synchronous = NORMAL;")
            return conn
        except sqlite3.Error as e:
            raise DBError(f"failed to connect sqlite: {e}") from e

    def ping(self) -> None:
        """연결/파일 접근 정상 여부 확인."""
        with self._connect() as conn:
            conn.execute("SELECT 1;")

    # ---------- schema ----------
    def init_db(self, models_sql_path: Optional[str | Path] = None) -> None:
        """
        models.sql을 실행해 스키마를 초기화/업데이트.
        (A의 scripts/init_db.py에서 호출하는 용도)
        """
        sql_path = Path(models_sql_path) if models_sql_path else (PROJECT_ROOT / "db" / "models.sql")
        if not sql_path.exists():
            raise DBError(f"models.sql not found: {sql_path}")

        sql_text = sql_path.read_text(encoding="utf-8")
        with self._connect() as conn:
            try:
                conn.executescript(sql_text)
                conn.commit()
            except sqlite3.Error as e:
                conn.rollback()
                raise DBError(f"failed to init schema: {e}") from e

    # =========================
    # Faces (C)
    # =========================
    def insert_face(self, *, name: str, embedding: np.ndarray) -> int:
        """
        얼굴 등록: (name, embedding_blob)
        returns: inserted face_id
        """
        nm = str(name).strip()
        if not nm:
            raise DBError("name is empty")

        blob = _embedding_to_blob(embedding, self.embed_dim)

        with self._connect() as conn:
            try:
                cur = conn.execute(
                    "INSERT INTO faces (name, embedding_blob) VALUES (?, ?);",
                    (nm, sqlite3.Binary(blob)),
                )
                conn.commit()
                return int(cur.lastrowid)
            except sqlite3.Error as e:
                conn.rollback()
                raise DBError(f"insert_face failed: {e}") from e

    def list_faces(self) -> List[FaceRecord]:
        """
        얼굴 목록 반환 (IdentityMatcher가 기대하는 인터페이스)
        """
        with self._connect() as conn:
            try:
                rows = conn.execute(
                    "SELECT id, name, embedding_blob FROM faces ORDER BY id ASC;"
                ).fetchall()
            except sqlite3.Error as e:
                raise DBError(f"list_faces failed: {e}") from e

        out: List[FaceRecord] = []
        for r in rows:
            fid = int(r["id"])
            nm = str(r["name"])
            emb = _blob_to_embedding(r["embedding_blob"], self.embed_dim)
            out.append(FaceRecord(id=fid, name=nm, embedding=emb))
        return out

    def delete_face(self, face_id: int) -> None:
        with self._connect() as conn:
            try:
                conn.execute("DELETE FROM faces WHERE id=?;", (int(face_id),))
                conn.commit()
            except sqlite3.Error as e:
                conn.rollback()
                raise DBError(f"delete_face failed: {e}") from e

    # =========================
    # Plates (B/C)
    # =========================
    def insert_plate(self, *, owner: str, plate_text_norm: str) -> int:
        ow = str(owner).strip()
        pt = str(plate_text_norm).strip()
        if not ow:
            raise DBError("owner is empty")
        if not pt:
            raise DBError("plate_text_norm is empty")

        with self._connect() as conn:
            try:
                cur = conn.execute(
                    "INSERT INTO plates (owner, plate_text_norm) VALUES (?, ?);",
                    (ow, pt),
                )
                conn.commit()
                return int(cur.lastrowid)
            except sqlite3.IntegrityError:
                # unique constraint: 이미 등록된 번호판
                conn.rollback()
                # 기존 id 반환(원하면)
                row = conn.execute(
                    "SELECT id FROM plates WHERE plate_text_norm=?;",
                    (pt,),
                ).fetchone()
                if row is not None:
                    return int(row["id"])
                raise DBError("plate already exists but failed to fetch id")
            except sqlite3.Error as e:
                conn.rollback()
                raise DBError(f"insert_plate failed: {e}") from e

    def is_plate_registered(self, *, plate_text_norm: str) -> bool:
        pt = str(plate_text_norm).strip()
        if not pt:
            return False
        with self._connect() as conn:
            try:
                row = conn.execute(
                    "SELECT 1 FROM plates WHERE plate_text_norm=? LIMIT 1;",
                    (pt,),
                ).fetchone()
                return row is not None
            except sqlite3.Error as e:
                raise DBError(f"is_plate_registered failed: {e}") from e

    def list_plates(self) -> List[PlateRecord]:
        with self._connect() as conn:
            try:
                rows = conn.execute(
                    "SELECT id, owner, plate_text_norm FROM plates ORDER BY id ASC;"
                ).fetchall()
            except sqlite3.Error as e:
                raise DBError(f"list_plates failed: {e}") from e

        return [
            PlateRecord(id=int(r["id"]), owner=str(r["owner"]), plate_text_norm=str(r["plate_text_norm"]))
            for r in rows
        ]

    # =========================
    # Jobs / Results (A)
    # =========================
    def create_job(self, *, start_ts: str, end_ts: str, source: str, status: str = "queued") -> int:
        st = str(start_ts).strip()
        et = str(end_ts).strip()
        src = str(source).strip()
        if not st or not et or not src:
            raise DBError("start_ts/end_ts/source must be non-empty")

        with self._connect() as conn:
            try:
                cur = conn.execute(
                    """
                    INSERT INTO jobs (start_ts, end_ts, source, status, created_at, updated_at)
                    VALUES (?, ?, ?, ?, datetime('now'), datetime('now'));
                    """,
                    (st, et, src, str(status)),
                )
                conn.commit()
                return int(cur.lastrowid)
            except sqlite3.Error as e:
                conn.rollback()
                raise DBError(f"create_job failed: {e}") from e

    def update_job_status(self, *, job_id: int, status: str) -> None:
        jid = int(job_id)
        st = str(status).strip()
        if not st:
            raise DBError("status is empty")

        with self._connect() as conn:
            try:
                cur = conn.execute(
                    """
                    UPDATE jobs
                    SET status=?, updated_at=datetime('now')
                    WHERE id=?;
                    """,
                    (st, jid),
                )
                conn.commit()
                if cur.rowcount == 0:
                    raise DBError(f"job not found: {jid}")
            except sqlite3.Error as e:
                conn.rollback()
                raise DBError(f"update_job_status failed: {e}") from e

    def get_job(self, job_id: int) -> JobRecord:
        jid = int(job_id)
        with self._connect() as conn:
            try:
                row = conn.execute(
                    "SELECT id, start_ts, end_ts, source, status, created_at, updated_at FROM jobs WHERE id=?;",
                    (jid,),
                ).fetchone()
            except sqlite3.Error as e:
                raise DBError(f"get_job failed: {e}") from e

        if row is None:
            raise DBError(f"job not found: {jid}")

        return JobRecord(
            id=int(row["id"]),
            start_ts=str(row["start_ts"]),
            end_ts=str(row["end_ts"]),
            source=str(row["source"]),
            status=str(row["status"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )

    def save_result(
        self,
        *,
        job_id: int,
        output_path: str,
        thumb_path: Optional[str] = None,
        meta_json: Optional[str] = None,
    ) -> None:
        jid = int(job_id)
        outp = str(output_path).strip()
        if not outp:
            raise DBError("output_path is empty")

        with self._connect() as conn:
            try:
                # 결과는 job_id PK라서 upsert로 처리(재처리/덮어쓰기 가능)
                conn.execute(
                    """
                    INSERT INTO results (job_id, output_path, thumb_path, meta_json, created_at)
                    VALUES (?, ?, ?, ?, datetime('now'))
                    ON CONFLICT(job_id) DO UPDATE SET
                        output_path=excluded.output_path,
                        thumb_path=excluded.thumb_path,
                        meta_json=excluded.meta_json;
                    """,
                    (jid, outp, thumb_path, meta_json),
                )
                conn.commit()
            except sqlite3.Error as e:
                conn.rollback()
                raise DBError(f"save_result failed: {e}") from e

    def get_result(self, job_id: int) -> Optional[ResultRecord]:
        jid = int(job_id)
        with self._connect() as conn:
            try:
                row = conn.execute(
                    "SELECT job_id, output_path, thumb_path, meta_json, created_at FROM results WHERE job_id=?;",
                    (jid,),
                ).fetchone()
            except sqlite3.Error as e:
                raise DBError(f"get_result failed: {e}") from e

        if row is None:
            return None

        return ResultRecord(
            job_id=int(row["job_id"]),
            output_path=str(row["output_path"]),
            thumb_path=str(row["thumb_path"]) if row["thumb_path"] is not None else None,
            meta_json=str(row["meta_json"]) if row["meta_json"] is not None else None,
            created_at=str(row["created_at"]),
        )
