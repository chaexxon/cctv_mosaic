# scripts/register_plate.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

# ✅ 단일 진실: normalize/regex/fixups는 여기만 사용
from utils.plate_text import PLATE_RE, normalize_and_fix
from utils.plate_store import DEFAULT_STORE, load_store, save_store, is_registered, add_plate


# =========================================================
# Optional: OCR placeholder (future)
# =========================================================
def ocr_plate_from_image(_image_path: str) -> Optional[str]:
    """
    TODO(B): services/ocr_engine.py 연결되면 여기 구현
    지금은 None 반환 (텍스트 직접 입력으로만 등록)
    """
    return None


def _print_list(store_path: Path) -> None:
    store = load_store(store_path)
    plates = store.get("plates", []) or []
    print(f"[register_plate] store: {store_path}")
    print(f"[register_plate] count: {len(plates)}")
    if not plates:
        return

    # 최신순(마지막 추가된게 아래) 그대로 출력
    for i, it in enumerate(plates, 1):
        owner = it.get("owner", "")
        raw = it.get("plate_raw", "")
        norm = it.get("plate_norm", "")
        created = it.get("created_at", "")
        print(f"{i:>3}. {norm}  owner={owner}  raw='{raw}'  at={created}")


def _delete_plate(store_path: Path, plate_text: str) -> None:
    store = load_store(store_path)
    plates = store.get("plates", []) or []

    norm = normalize_and_fix(plate_text)

    before = len(plates)
    plates2 = [it for it in plates if it.get("plate_norm") != norm]
    after = len(plates2)

    if after == before:
        print(f"[register_plate] NOT FOUND: {norm}")
        print(f"[register_plate] store: {store_path}")
        return

    store["plates"] = plates2
    save_store(store_path, store)
    print(f"[register_plate] DELETED: {norm}  (removed {before - after})")
    print(f"[register_plate] store: {store_path}")


def main():
    parser = argparse.ArgumentParser(
        description="B: register license plate (exact-only store, text first, OCR optional)",
    )

    parser.add_argument("--owner", type=str, default=None, help="owner/name label (ex: hongmin)")
    parser.add_argument("--plate", type=str, default=None, help="plate text (ex: 123가4567)")
    parser.add_argument("--image", type=str, default=None, help="optional: plate image path for OCR (future)")

    parser.add_argument(
        "--store",
        type=str,
        default=str(DEFAULT_STORE),
        help="json store path (default: data/plates/plates.json)",
    )

    parser.add_argument("--force", action="store_true", help="allow duplicate insert (not recommended)")
    parser.add_argument("--list", action="store_true", help="list registered plates and exit")
    parser.add_argument("--delete", type=str, default=None, help="delete by plate text (exact norm match) and exit")

    args = parser.parse_args()

    store_path = Path(args.store)

    # list / delete modes
    if args.list:
        _print_list(store_path)
        return

    if args.delete is not None:
        _delete_plate(store_path, args.delete)
        return

    # register mode
    if not args.owner:
        raise SystemExit("ERROR: --owner is required for register mode")

    store = load_store(store_path)

    plate_raw: Optional[str] = args.plate

    # image OCR mode (future)
    if plate_raw is None and args.image is not None:
        plate_raw = ocr_plate_from_image(args.image)

    if plate_raw is None:
        raise SystemExit("ERROR: provide --plate TEXT  (OCR mode via --image is not implemented yet)")

    # ✅ normalize + OCR fixups + strict regex validation
    plate_norm = normalize_and_fix(plate_raw)

    if not PLATE_RE.match(plate_norm):
        raise SystemExit(
            "ERROR: plate text invalid.\n"
            f"  raw  : '{plate_raw}'\n"
            f"  norm : '{plate_norm}'\n"
            "Expected format examples: 25무9889, 123가4567"
        )

    if (not args.force) and is_registered(store, plate_norm):
        print(f"[register_plate] already registered: {plate_norm} (owner={args.owner})")
        print(f"[register_plate] store: {store_path}")
        return

    rec = add_plate(store, args.owner, plate_raw, plate_norm)
    save_store(store_path, store)

    print("[register_plate] OK")
    print(f"  owner      : {rec.owner}")
    print(f"  plate_raw  : {rec.plate_raw}")
    print(f"  plate_norm : {rec.plate_norm}")
    print(f"  saved_to   : {store_path}")


if __name__ == "__main__":
    main()
