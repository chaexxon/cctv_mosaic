# scripts/register_plate.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from db.sqlite import SQLiteDB
from utils.plate_text import PLATE_RE, normalize_and_fix


def _print_list(db: SQLiteDB) -> None:
    plates = db.list_plates()
    print(f"[register_plate] db: {db.db_path}")
    print(f"[register_plate] count: {len(plates)}")
    if not plates:
        return
    for i, p in enumerate(plates, 1):
        print(
            f"{i:>3}. id={p.id}  norm={p.plate_text_norm}  owner={p.owner}  raw='{p.plate_raw}'  at={p.created_at}"
        )


def _delete_plate(db: SQLiteDB, plate_text: str) -> None:
    norm = normalize_and_fix(plate_text)
    if not PLATE_RE.match(norm):
        raise SystemExit(f"ERROR: invalid plate format after normalize: '{norm}'")

    n = db.delete_plate(norm)
    if n <= 0:
        print(f"[register_plate] NOT FOUND: {norm}")
        print(f"[register_plate] db: {db.db_path}")
        return

    print(f"[register_plate] DELETED: {norm}  (removed {n})")
    print(f"[register_plate] db: {db.db_path}")


def main():
    p = argparse.ArgumentParser(description="B: register license plate into SQLite DB (exact-only)")

    p.add_argument("--db", type=str, default="db/cctv_mosaic.sqlite3", help="sqlite db path")
    p.add_argument("--owner", type=str, default=None, help="owner/name label (ex: hongmin)")
    p.add_argument("--plate", type=str, default=None, help="plate text (ex: 172허7410)")

    p.add_argument("--list", action="store_true", help="list plates and exit")
    p.add_argument("--delete", type=str, default=None, help="delete by plate text (exact norm match) and exit")

    args = p.parse_args()

    db = SQLiteDB(args.db)

    if args.list:
        _print_list(db)
        return

    if args.delete is not None:
        _delete_plate(db, args.delete)
        return

    if not args.owner:
        raise SystemExit("ERROR: --owner is required")
    if not args.plate:
        raise SystemExit("ERROR: --plate is required")

    plate_raw: str = args.plate
    plate_norm = normalize_and_fix(plate_raw)

    if not PLATE_RE.match(plate_norm):
        raise SystemExit(
            "ERROR: plate text invalid.\n"
            f"  raw  : '{plate_raw}'\n"
            f"  norm : '{plate_norm}'\n"
            "Expected examples: 25무9889, 123가4567, 172허7410"
        )

    # insert (idempotent)
    plate_id = db.insert_plate(owner=args.owner, plate_text_norm=plate_norm, plate_raw=plate_raw)

    print("[register_plate] OK")
    print(f"  id         : {plate_id}")
    print(f"  owner      : {args.owner}")
    print(f"  plate_raw  : {plate_raw}")
    print(f"  plate_norm : {plate_norm}")
    print(f"  db         : {Path(db.db_path)}")


if __name__ == "__main__":
    main()