# core/config.py
"""
Project-wide configuration (A + B + C common)

원칙:
- 설정은 여기 한 군데에서만 관리 (단일 진실)
- 경로는 Path로 관리
- 팀원 B/C 코드에서 import 해서 바로 쓸 수 있게 키 이름 고정
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

# =========================
# Project root / paths
# =========================
# core/ 폴더 기준으로 프로젝트 루트 추정
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
INPUT_DIR = DATA_DIR / "input"
RAW_DIR = INPUT_DIR / "raw"
SEGMENTS_DIR = INPUT_DIR / "segments"
OUTPUT_DIR = DATA_DIR / "output"
FACES_DIR = DATA_DIR / "faces"
LOGS_DIR = DATA_DIR / "logs"

DB_DIR = PROJECT_ROOT / "db"
DB_PATH = DB_DIR / "cctv_mosaic.sqlite3"

MODELS_DIR = PROJECT_ROOT / "models"
YOLO_FACE_DIR = MODELS_DIR / "yolo_face"
YOLO_PLATE_DIR = MODELS_DIR / "yolo_plate"
ARCFACE_MODEL_DIR = MODELS_DIR / "arcface"  # InsightFace root로 사용(환경별 조정 가능)

# =========================
# Detection / Sampling policy
# =========================
# B: N프레임마다 탐지(부하 절감)
DETECT_EVERY_N_FRAMES: int = int(os.getenv("DETECT_EVERY_N_FRAMES", "3"))

# =========================
# Face recognition (C)
# =========================
FACE_EMBED_DIM: int = int(os.getenv("FACE_EMBED_DIM", "512"))

# cosine similarity threshold (등록 판정)
FACE_SIM_THRESHOLD: float = float(os.getenv("FACE_SIM_THRESHOLD", "0.55"))

# Temporal embedding (C)
TEMPORAL_WINDOW: int = int(os.getenv("TEMPORAL_WINDOW", "5"))
TEMPORAL_MODE: Literal["mean", "median"] = os.getenv("TEMPORAL_MODE", "mean")  # type: ignore

# =========================
# OCR (B)
# =========================
OCR_CONF_THRESHOLD: float = float(os.getenv("OCR_CONF_THRESHOLD", "0.6"))

# =========================
# Mosaic (B)
# =========================
MOSAIC_MODE: Literal["blur", "pixelate"] = os.getenv("MOSAIC_MODE", "blur")  # type: ignore

# =========================
# Output encoding
# =========================
OUTPUT_FPS: float = float(os.getenv("OUTPUT_FPS", "30.0"))
OUTPUT_CODEC: str = os.getenv("OUTPUT_CODEC", "mp4v")  # OpenCV fourcc 기본값

# =========================
# (Optional) runtime options
# =========================
# InsightFace ctx_id: CPU=-1, GPU=0
INSIGHTFACE_CTX_ID: int = int(os.getenv("INSIGHTFACE_CTX_ID", "-1"))

# =========================
# Ensure dirs exist (safe)
# =========================
def ensure_dirs() -> None:
    for p in [
        DATA_DIR,
        INPUT_DIR,
        RAW_DIR,
        SEGMENTS_DIR,
        OUTPUT_DIR,
        FACES_DIR,
        LOGS_DIR,
        DB_DIR,
        MODELS_DIR,
    ]:
        p.mkdir(parents=True, exist_ok=True)
