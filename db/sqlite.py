# db/sqlite.py
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple


class DBError(RuntimeError):
    pass


# -------------------------
# Rows
# -------------------------
@dataclass
class PlateRow:
    id: int
    owner: str
    plate_text_norm: str
    plate_raw: str
    created_at: str


@dataclass
class FaceRow:
    id: int
    name: str
    embedding_blob: bytes
    created_at: str


class SQLiteDB:
    """
    Minimal SQLite helper for cctv_mosaic.
    - Assumes schema already created by scripts/init_db.py using db/models.sql
    """

    def __init__(self, db_path: str = "db/cctv_mosaic.sqlite3"):
        self.db_path = str(db_path)

    def _connect(self) -> sqlite3.Connection:
        p = Path(self.db_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def execute(self, sql: str, params: Sequence[Any] = ()) -> None:
        try:
            with self._connect() as conn:
                conn.execute(sql, params)
                conn.commit()
        except Exception as e:
            raise DBError(str(e))

    def fetchall(self, sql: str, params: Sequence[Any] = ()) -> List[sqlite3.Row]:
        try:
            with self._connect() as conn:
                cur = conn.execute(sql, params)
                return list(cur.fetchall())
        except Exception as e:
            raise DBError(str(e))

    def fetchone(self, sql: str, params: Sequence[Any] = ()) -> Optional[sqlite3.Row]:
        try:
            with self._connect() as conn:
                cur = conn.execute(sql, params)
                return cur.fetchone()
        except Exception as e:
            raise DBError(str(e))

    # -------------------------
    # Plates
    # -------------------------
    def insert_plate(self, owner: str, plate_text_norm: str, plate_raw: str) -> int:
        row = self.fetchone(
            "SELECT id FROM plates WHERE plate_text_norm = ?",
            (plate_text_norm,),
        )
        if row is not None:
            return int(row["id"])

        try:
            with self._connect() as conn:
                cur = conn.execute(
                    "INSERT INTO plates(owner, plate_text_norm, plate_raw) VALUES(?,?,?)",
                    (owner, plate_text_norm, plate_raw),
                )
                conn.commit()
                return int(cur.lastrowid)
        except Exception as e:
            raise DBError(str(e))

    def delete_plate(self, plate_text_norm: str) -> int:
        try:
            with self._connect() as conn:
                cur = conn.execute(
                    "DELETE FROM plates WHERE plate_text_norm = ?",
                    (plate_text_norm,),
                )
                conn.commit()
                return int(cur.rowcount)
        except Exception as e:
            raise DBError(str(e))

    def list_plates(self) -> List[PlateRow]:
        rows = self.fetchall(
            "SELECT id, owner, plate_text_norm, plate_raw, created_at "
            "FROM plates ORDER BY id ASC"
        )
        out: List[PlateRow] = []
        for r in rows:
            out.append(
                PlateRow(
                    id=int(r["id"]),
                    owner=str(r["owner"]),
                    plate_text_norm=str(r["plate_text_norm"]),
                    plate_raw=str(r["plate_raw"]),
                    created_at=str(r["created_at"]),
                )
            )
        return out

    def is_plate_registered(self, plate_text_norm: str) -> bool:
        row = self.fetchone(
            "SELECT 1 FROM plates WHERE plate_text_norm = ? LIMIT 1",
            (plate_text_norm,),
        )
        return row is not None

    def get_registered_plate_set(self) -> set[str]:
        rows = self.fetchall("SELECT plate_text_norm FROM plates")
        return {str(r["plate_text_norm"]) for r in rows}

    # -------------------------
    # Faces
    # -------------------------
    def insert_face(self, name: str, embedding_blob: bytes) -> int:
        """
        Insert a face embedding.
        - embedding_blob: np.ndarray(float32) -> .tobytes() 형태 권장
        """
        try:
            with self._connect() as conn:
                cur = conn.execute(
                    "INSERT INTO faces(name, embedding_blob) VALUES(?,?)",
                    (name, embedding_blob),
                )
                conn.commit()
                return int(cur.lastrowid)
        except Exception as e:
            raise DBError(str(e))

    def list_faces(self) -> List[FaceRow]:
        rows = self.fetchall(
            "SELECT id, name, embedding_blob, created_at "
            "FROM faces ORDER BY id ASC"
        )
        out: List[FaceRow] = []
        for r in rows:
            out.append(
                FaceRow(
                    id=int(r["id"]),
                    name=str(r["name"]),
                    embedding_blob=bytes(r["embedding_blob"]),
                    created_at=str(r["created_at"]),
                )
            )
        return out

    def list_faces_embeddings(self) -> List[Tuple[int, str, bytes]]:
        rows = self.fetchall("SELECT id, name, embedding_blob FROM faces ORDER BY id ASC")
        return [(int(r["id"]), str(r["name"]), bytes(r["embedding_blob"])) for r in rows]