# utils/video.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Iterable, Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    """Video metadata."""
    fps: float
    width: int
    height: int
    frame_count: int
    fourcc: str


def _fourcc_to_str(fourcc_int: int) -> str:
    try:
        return "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])
    except Exception:
        return "????"


def open_capture(video_path: str | Path) -> cv2.VideoCapture:
    """
    Open a video file and return a cv2.VideoCapture.
    Raises RuntimeError if it cannot be opened.
    """
    vp = str(video_path)
    cap = cv2.VideoCapture(vp)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {vp}")
    return cap


def get_video_info(cap: cv2.VideoCapture) -> VideoInfo:
    """
    Read video metadata from an opened cv2.VideoCapture.
    """
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC) or 0)
    fourcc = _fourcc_to_str(fourcc_int)

    # Some videos report 0 FPS; fallback to a sane default to avoid writer crash
    if fps <= 0:
        fps = 30.0

    if width <= 0 or height <= 0:
        raise RuntimeError("Invalid video size read from capture.")

    return VideoInfo(
        fps=fps,
        width=width,
        height=height,
        frame_count=frame_count,
        fourcc=fourcc,
    )


def iter_frames(
    cap: cv2.VideoCapture,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    *,
    convert_bgr: bool = True,
) -> Generator[Tuple[int, np.ndarray], None, None]:
    """
    Iterate frames from an opened capture.

    Args:
        cap: Opened cv2.VideoCapture
        start_frame: inclusive start index
        end_frame: exclusive end index (None = until end)
        convert_bgr: OpenCV returns BGR; keep as BGR (True). If False, returns raw frame.

    Yields:
        (frame_idx, frame) where frame is np.ndarray (H,W,3) uint8
    """
    if start_frame < 0:
        start_frame = 0

    # Seek (may be approximate depending on codec)
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))

    idx = start_frame
    while True:
        if end_frame is not None and idx >= end_frame:
            break

        ok, frame = cap.read()
        if not ok or frame is None:
            break

        if convert_bgr:
            # OpenCV already gives BGR. Keep it to avoid extra cost.
            pass

        yield idx, frame
        idx += 1


def make_writer(
    output_path: str | Path,
    fps: float,
    frame_size: Tuple[int, int],
    *,
    codec: str = "mp4v",
    is_color: bool = True,
) -> cv2.VideoWriter:
    """
    Create a cv2.VideoWriter with safe defaults.

    Args:
        output_path: output file path (e.g., data/output/test.mp4)
        fps: frames per second
        frame_size: (width, height)
        codec: FourCC string. Common: "mp4v"(mp4), "avc1"(H.264 if available), "XVID"(avi)
        is_color: True for BGR frames

    Returns:
        cv2.VideoWriter

    Raises:
        RuntimeError if writer cannot be opened.
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    w, h = int(frame_size[0]), int(frame_size[1])
    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid frame_size: {frame_size}")

    if fps <= 0:
        fps = 30.0

    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(out_path), fourcc, float(fps), (w, h), isColor=is_color)

    if not writer.isOpened():
        raise RuntimeError(
            f"Failed to open VideoWriter: path={out_path}, codec={codec}, fps={fps}, size={(w,h)}"
        )
    return writer


def safe_release(cap: Optional[cv2.VideoCapture] = None, writer: Optional[cv2.VideoWriter] = None) -> None:
    """
    Safely release OpenCV resources.
    """
    try:
        if cap is not None:
            cap.release()
    except Exception:
        pass

    try:
        if writer is not None:
            writer.release()
    except Exception:
        pass


def probe_video(video_path: str | Path) -> VideoInfo:
    """
    Convenience: open -> get info -> release.
    """
    cap = None
    try:
        cap = open_capture(video_path)
        info = get_video_info(cap)
        return info
    finally:
        safe_release(cap=cap)


# ---- Minimal self-test (optional) ----
if __name__ == "__main__":
    # Example:
    # python utils/video.py data/input/raw/sample.mp4
    import sys

    if len(sys.argv) < 2:
        print("Usage: python utils/video.py <video_path>")
        raise SystemExit(0)

    path = sys.argv[1]
    cap = None
    try:
        cap = open_capture(path)
        info = get_video_info(cap)
        print("[VideoInfo]", info)

        # Read first 5 frames
        for i, (idx, frame) in enumerate(iter_frames(cap, start_frame=0, end_frame=5)):
            print(f"frame {idx}: shape={frame.shape}, dtype={frame.dtype}")
    finally:
        safe_release(cap=cap)
