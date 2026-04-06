# utils/video.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Tuple

import cv2


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    frame_count: int


def open_capture(video_path: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    return cap


def get_video_info(cap: cv2.VideoCapture) -> VideoInfo:
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 1e-6:
        fps = 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return VideoInfo(width, height, fps, frame_count)


def iter_frames(cap: cv2.VideoCapture) -> Iterator[Tuple[int, Any]]:
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield idx, frame
        idx += 1


def make_writer(
    *,
    output_path: Path,
    fps: float,
    frame_size: Tuple[int, int],
    codec: str = "XVID",
    is_color: bool = True,
) -> cv2.VideoWriter:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        frame_size,
        is_color,
    )

    if not writer.isOpened():
        raise RuntimeError(
            f"Failed to open VideoWriter: path={output_path}, codec={codec}, fps={fps}, size={frame_size}"
        )
    return writer


def safe_release(cap=None, writer=None):
    try:
        if writer is not None:
            writer.release()
    finally:
        if cap is not None:
            cap.release()
