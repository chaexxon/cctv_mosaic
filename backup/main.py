from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import List, Optional
from uuid import uuid4

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from db.sqlite import SQLiteDB
from scripts.process_video import process_video
from services.face_recognizer import FaceRecognizer
from utils.plate_text import normalize_and_fix


PROJECT_ROOT = Path(__file__).resolve().parents[1]

STATIC_DIR = PROJECT_ROOT / "static"
UPLOAD_DIR = STATIC_DIR / "uploads"
OUTPUT_DIR = STATIC_DIR / "outputs"
TEMPLATE_DIR = PROJECT_ROOT / "templates"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="CCTV Mosaic System")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

templates = Jinja2Templates(directory=str(TEMPLATE_DIR))
db = SQLiteDB("db/cctv_mosaic.sqlite3")


def _time_to_seconds(t: str) -> float:
    t = str(t).strip()

    if ":" not in t:
        return float(t)

    parts = [float(p) for p in t.split(":")]

    if len(parts) == 3:
        h, m, s = parts
        return h * 3600 + m * 60 + s

    if len(parts) == 2:
        m, s = parts
        return m * 60 + s

    raise ValueError(f"Invalid time format: {t}")


def _cut_video(input_path: Path, output_path: Path, start_time: str, end_time: str) -> None:
    start_sec = _time_to_seconds(start_time)
    end_sec = _time_to_seconds(end_time)

    if end_sec <= start_sec:
        raise ValueError("end_time must be greater than start_time")

    duration = end_sec - start_sec
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-ss", str(start_sec),
        "-t", str(duration),
        "-i", str(input_path),
        "-c:v", "mpeg4",
        "-q:v", "3",
        "-an",
        str(output_path),
    ]

    subprocess.run(cmd, check=True)

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(f"clip video was not created: {output_path}")

    cap = cv2.VideoCapture(str(output_path))
    ok = cap.isOpened()
    cap.release()

    if not ok:
        raise RuntimeError(f"created clip cannot be opened by OpenCV: {output_path}")


def _register_face(face_path: Path, name: str, recognizer: FaceRecognizer) -> None:
    frame = cv2.imread(str(face_path))
    if frame is None:
        raise RuntimeError(f"failed to read face image: {face_path}")

    fe = recognizer.extract_from_frame(frame, bbox=None)
    emb = fe.embedding if hasattr(fe, "embedding") else fe

    emb = np.asarray(emb, dtype=np.float32).reshape(-1)

    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm

    db.insert_face(
        name=name,
        embedding_blob=emb.astype(np.float32).tobytes(),
    )


def _register_plate(plate_number: str, owner: str) -> None:
    raw = plate_number.strip()
    if not raw:
        return

    norm = normalize_and_fix(raw)
    if not norm:
        norm = raw

    db.insert_plate(
        owner=owner,
        plate_text_norm=norm,
        plate_raw=raw,
    )


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "video_url": None,
            "message": None,
        },
    )


@app.post("/process", response_class=HTMLResponse)
async def process_video_web(
    request: Request,
    video: UploadFile = File(...),
    start_time: str = Form(...),
    end_time: str = Form(...),
    face_images: List[UploadFile] = File(default=[]),
    plate_number: str = Form(""),
):
    uid = str(uuid4())[:8]
    user_name = f"user_{uid}"

    original_video_path = UPLOAD_DIR / f"{uid}_{video.filename}"

    with open(original_video_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    clipped_video_path = UPLOAD_DIR / f"{uid}_clip.mp4"

    _cut_video(
        input_path=original_video_path,
        output_path=clipped_video_path,
        start_time=start_time,
        end_time=end_time,
    )

    if face_images:
        recognizer = FaceRecognizer()

        for idx, face_image in enumerate(face_images):
            if face_image is None or not face_image.filename:
                continue

            suffix = Path(face_image.filename).suffix or ".jpg"
            face_path = UPLOAD_DIR / f"{uid}_face_{idx}{suffix}"

            with open(face_path, "wb") as buffer:
                shutil.copyfileobj(face_image.file, buffer)

            try:
                _register_face(face_path, name=user_name, recognizer=recognizer)
                print(f"[face-register] success: {face_path}")
            except Exception as e:
                print(f"[face-register] failed: {face_path} / {e}")

    if plate_number.strip():
        try:
            _register_plate(plate_number, owner=user_name)
            print(f"[plate-register] success: {plate_number}")
        except Exception as e:
            print(f"[plate-register] failed: {plate_number} / {e}")

    output_video_path = OUTPUT_DIR / f"{uid}_result.mp4"

    process_video(
        input_video_path=str(clipped_video_path),
        output_video_path=str(output_video_path),
        mode="blur",
        enable_plate=True,
        overwrite=True,
        print_every=30,
        db_path="db/cctv_mosaic.sqlite3",
    )

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "video_url": f"/static/outputs/{output_video_path.name}",
            "message": f"{start_time} ~ {end_time} 구간 처리 완료",
        },
    )