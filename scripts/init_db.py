# scripts/init_db.py
from __future__ import annotations

import argparse
from pathlib import Path

from db.sqlite import init_db, DEFAULT_DB_PATH, DEFAULT_MODELS_SQL


def main():
    p = argparse.ArgumentParser(description="Init SQLite DB schema from db/models.sql")
    p.add_argument("--db", type=str, default=str(DEFAULT_DB_PATH))
    p.add_argument("--models", type=str, default=str(DEFAULT_MODELS_SQL))
    args = p.parse_args()

    init_db(db_path=Path(args.db), models_sql_path=Path(args.models))
    print("[init_db] OK")
    print("  db     :", args.db)
    print("  models :", args.models)


if __name__ == "__main__":
    main()