# scripts/register_face.py
"""
Face Registration CLI (C)

역할:
- 이미지 1장을 입력받아 얼굴 임베딩을 추출
- (선택) 임베딩 정규화/검증
- SQLite DB에 (name, embedding_blob) 저장

사용 예:
  python -m scripts.register_face --name hongmin --img data/faces/test.jpg
  python -m scripts.register_face --name hongmin --img data/faces/test.jpg --db db/cctv_mosaic.sqlite3
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import cv2

from services.face_recognizer import FaceRecognizer
from db.sqlite import SQLiteDB


def _resolve_db_path(arg_db: str | None) -> str:
    if arg_db:
        return arg_db
    # core.config에 DB_PATH가 있으면 사용 (없으면 fallback)
    try:
        from core.config import DB_PATH  # type: ignore
        return str(DB_PATH)
    except Exception:
        return "db/cctv_mosaic.sqlite3"


def _load_bgr_image(img_path: Path) -> np.ndarray:
    img = cv2.imread(str(img_path))
    if img is None or img.size == 0:
        raise ValueError(f"Failed to read image: {img_path}")
    return img


def _normalize_embedding(emb: np.ndarray) -> np.ndarray:
    emb = emb.astype(np.float32, copy=False).reshape(-1)
    if not np.isfinite(emb).all():
        raise ValueError("Embedding contains NaN/Inf")
    n = float(np.linalg.norm(emb))
    if n < 1e-12:
        raise ValueError("Embedding norm is too small (zero vector?)")
    return emb / n


def main() -> int:
    parser = argparse.ArgumentParser(description="Register face embedding into SQLite DB")
    parser.add_argument("--name", required=False, help="person name/label to register")
    parser.add_argument("--img", required=False, help="path to face image")
    parser.add_argument("--dir", default=None, help="directory of face images")
    parser.add_argument("--db", default=None, help="sqlite db path (default: from core.config.DB_PATH or db/cctv_mosaic.sqlite3)")
    parser.add_argument("--no_norm", action="store_true", help="do not L2-normalize embedding (default: normalize)")
    args = parser.parse_args()

    # 🔥 폴더 전체 등록 모드
    if args.dir:
        from pathlib import Path
        import cv2
        import numpy as np

        faces_dir = Path(args.dir)
        if not faces_dir.exists():
            print(f"[ERR] dir not found: {faces_dir}")
            return 2

        recognizer = FaceRecognizer()
        db = SQLiteDB(_resolve_db_path(args.db))

        img_files = list(faces_dir.glob("*.*"))

        print(f"[INFO] found {len(img_files)} images")

        for img_path in img_files:
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    print(f"[SKIP] cannot read: {img_path}")
                    continue

                out = recognizer.extract_from_frame(img, bbox=None)
                emb = out.embedding if hasattr(out, "embedding") else out

                emb = emb.astype(np.float32).reshape(-1)
                emb = emb / np.linalg.norm(emb)

                name = img_path.stem
                face_id = db.insert_face(name=name, embedding_blob=emb.tobytes())

                print(f"[OK] {name} 등록됨 (id={face_id})")

            except Exception as e:
                print(f"[FAIL] {img_path}: {e}")

        return 0

    name = str(args.name).strip()
    if not name:
        print("[ERR] name is empty", file=sys.stderr)
        return 2

    img_path = Path(args.img)
    if not img_path.exists():
        print(f"[ERR] image not found: {img_path}", file=sys.stderr)
        return 2

    db_path = _resolve_db_path(args.db)
    db = SQLiteDB(db_path)

    recognizer = FaceRecognizer()

    # ---- 임베딩 추출 ----
    # 1) recognizer가 파일경로 기반 함수를 제공하면 그걸 우선 사용
    emb: np.ndarray | None = None
    try:
        if hasattr(recognizer, "extract_from_image"):
            out = recognizer.extract_from_image(str(img_path))  # type: ignore
            # out이 객체(embedding field)거나 ndarray일 수 있음
            emb = out.embedding if hasattr(out, "embedding") else out  # type: ignore
        else:
            frame = _load_bgr_image(img_path)
            # bbox=None이면 recognizer 내부에서 얼굴 검출/정렬을 하거나,
            # 이미지 전체에서 얼굴을 찾는 구현이어야 함
            out = recognizer.extract_from_frame(frame, bbox=None)  # type: ignore
            emb = out.embedding if hasattr(out, "embedding") else out  # type: ignore
    except Exception as e:
        print(f"[ERR] face embedding extraction failed: {e}", file=sys.stderr)
        return 3

    if emb is None:
        print("[ERR] embedding is None", file=sys.stderr)
        return 3
    if not isinstance(emb, np.ndarray):
        print(f"[ERR] embedding must be np.ndarray, got {type(emb)}", file=sys.stderr)
        return 3

    if not args.no_norm:
        emb = _normalize_embedding(emb)

    # ---- DB 저장 ----
    try:
        face_id = db.insert_face(name=name, embedding_blob=emb.tobytes())  # 너희 db/sqlite.py 시그니처에 맞춰둠
    except TypeError:
        # 혹시 시그니처가 insert_face(name, embedding_blob) 등으로 다르면 여기서 바로 잡을 수 있게 메시지
        print("[ERR] db.insert_face signature mismatch. Check db/sqlite.py insert_face(...) parameters.", file=sys.stderr)
        return 4
    except Exception as e:
        print(f"[ERR] DB insert failed: {e}", file=sys.stderr)
        return 4

    print(f"[OK] registered face: name={name}, id={face_id}, db={db_path}, emb_dim={emb.size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())