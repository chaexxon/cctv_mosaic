# scripts/init_db.py

import sqlite3
from pathlib import Path

DB_PATH = Path("db/cctv_mosaic.sqlite3")
MODELS_SQL = Path("db/models.sql")


def main():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    sql = MODELS_SQL.read_text(encoding="utf-8")

    conn = sqlite3.connect(DB_PATH)
    try:
        conn.executescript(sql)
        conn.commit()
    finally:
        conn.close()

    print("[init_db] OK")


if __name__ == "__main__":
    main()