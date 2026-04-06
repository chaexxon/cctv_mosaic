# scripts/init_db.py
"""
DB Initialize / Migration Runner (A 공통, B/C 선사용 가능)

목표
- db/models.sql 실행하여 스키마 생성
- 기존 DB가 "구버전 스키마"인 경우를 탐지하여 안내/자동 보정
- 팀 통합 시, 각자 로컬 DB가 달라도 init_db 한 번으로 정리되게 설계

사용:
  python -m scripts.init_db
  python -m scripts.init_db --db db/cctv_mosaic.sqlite3
  python -m scripts.init_db --models db/models.sql
  python -m scripts.init_db --auto_migrate   (구버전 스키마면 자동 보정)
  python -m scripts.init_db --reset_db       (DB 파일 삭제 후 새로 생성)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from db.sqlite import SQLiteDB, DBError


def _has_column(db: SQLiteDB, table: str, column: str) -> bool:
    import sqlite3

    try:
        with db._connect() as conn:  # 내부 커넥션 사용 (db.sqlite.py가 check_same_thread=False로 안전)
            rows = conn.execute(f"PRAGMA table_info({table});").fetchall()
    except sqlite3.Error as e:
        raise DBError(f"failed to inspect table_info({table}): {e}") from e

    cols = {str(r["name"]) for r in rows}
    return column in cols


def _auto_migrate_faces_add_embedding_blob(db: SQLiteDB) -> None:
    """
    구버전 faces 테이블(embedding_blob 없음)을 위한 최소 마이그레이션.
    - 기존 테이블 유지
    - 컬럼만 추가 (NULL 허용)
    - 이후부터 insert_face는 정상 동작
    """
    import sqlite3

    with db._connect() as conn:
        try:
            conn.execute("ALTER TABLE faces ADD COLUMN embedding_blob BLOB;")
            conn.commit()
        except sqlite3.Error as e:
            conn.rollback()
            raise DBError(f"auto migrate failed (ALTER TABLE): {e}") from e


def main() -> int:
    parser = argparse.ArgumentParser(description="Initialize SQLite DB schema from models.sql")
    parser.add_argument("--db", default=None, help="sqlite db path (default: core.config.DB_PATH)")
    parser.add_argument("--models", default=None, help="models.sql path (default: db/models.sql)")
    parser.add_argument(
        "--auto_migrate",
        action="store_true",
        help="if schema is outdated, apply safe auto-migration (recommended for teammates)",
    )
    parser.add_argument(
        "--reset_db",
        action="store_true",
        help="DELETE existing db file then re-init (cleanest in development)",
    )
    args = parser.parse_args()

    db_path = Path(args.db) if args.db else None
    models_path = Path(args.models) if args.models else None

    try:
        # reset 옵션: 개발 단계에서 가장 깔끔
        if args.reset_db:
            # SQLiteDB가 기본 경로를 쓰는 경우를 위해 먼저 인스턴스 생성해서 실제 경로 확정
            tmp = SQLiteDB(db_path)
            p = Path(tmp.db_path)
            if p.exists():
                p.unlink()
            # 이후 새로 생성
            db = SQLiteDB(db_path)
        else:
            db = SQLiteDB(db_path)

        # 1) models.sql 실행 (테이블 없으면 생성)
        db.init_db(models_sql_path=models_path)

        # 2) 스키마 검사 (faces.embedding_blob 필수)
        ok = _has_column(db, "faces", "embedding_blob")
        if not ok:
            msg = (
                "[WARN] Your DB has an old 'faces' schema: missing column 'embedding_blob'.\n"
                "       This happens when an old db file already existed (CREATE TABLE IF NOT EXISTS won't modify it).\n"
            )
            print(msg, file=sys.stderr)

            if args.auto_migrate:
                _auto_migrate_faces_add_embedding_blob(db)
                # 재검사
                if not _has_column(db, "faces", "embedding_blob"):
                    raise DBError("auto_migrate done but embedding_blob still missing (unexpected).")
                print("[OK] auto_migrate applied: faces.embedding_blob added.")
            else:
                print(
                    "Fix options:\n"
                    "  1) (Recommended) python -m scripts.init_db --reset_db\n"
                    "  2) (Keep db)      python -m scripts.init_db --auto_migrate\n",
                    file=sys.stderr,
                )
                return 2

        # 3) 연결 확인
        db.ping()

    except DBError as e:
        print(f"[ERR] DB init failed: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"[ERR] unexpected error: {e}", file=sys.stderr)
        return 3

    print("[OK] DB schema initialized and verified.")
    print(" - db:", db.db_path)
    if models_path:
        print(" - models:", str(models_path))
    else:
        print(" - models: db/models.sql (default)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())