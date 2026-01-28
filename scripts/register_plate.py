# scripts/register_plate.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from utils.normalize import normalize_plate_text, is_plausible_plate
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


def main():
    parser = argparse.ArgumentParser(description="B: register license plate (text first, OCR optional)")

    parser.add_argument("--owner", type=str, required=True, help="owner/name label (ex: hongmin)")
    parser.add_argument("--plate", type=str, default=None, help="plate text (ex: 12가3456)")
    parser.add_argument("--image", type=str, default=None, help="optional: plate image path for OCR (future)")
    parser.add_argument(
        "--store",
        type=str,
        default=str(DEFAULT_STORE),
        help="json store path (default: data/plates/plates.json)",
    )
    parser.add_argument("--force", action="store_true", help="allow duplicate insert")

    args = parser.parse_args()

    store_path = Path(args.store)
    store = load_store(store_path)

    plate_raw: Optional[str] = args.plate

    # image OCR mode (future)
    if plate_raw is None and args.image is not None:
        plate_raw = ocr_plate_from_image(args.image)

    if plate_raw is None:
        raise SystemExit("ERROR: provide --plate TEXT  (OCR mode via --image is not implemented yet)")

    plate_norm = normalize_plate_text(plate_raw)

    if not is_plausible_plate(plate_norm):
        raise SystemExit(f"ERROR: plate text looks invalid after normalize: '{plate_norm}' (raw='{plate_raw}')")

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
