# services/mosaic.py
from __future__ import annotations

from typing import Iterable, List, Literal, Sequence, Tuple

import cv2
import numpy as np

BBox = Tuple[int, int, int, int]  # (x1, y1, x2, y2)
MosaicMode = Literal["blur", "pixelate"]


def _clip_bbox(b: BBox, w: int, h: int) -> BBox | None:
    """
    Clip bbox to image bounds. Return None if bbox is invalid after clipping.
    """
    x1, y1, x2, y2 = b
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w))
    y2 = max(0, min(int(y2), h))

    # Ensure proper ordering and non-empty area
    if x2 <= x1 + 1 or y2 <= y1 + 1:
        return None
    return (x1, y1, x2, y2)


def _apply_blur_roi(roi: np.ndarray) -> np.ndarray:
    """
    Fast blur for ROI. Kernel size adapts to ROI size.
    """
    rh, rw = roi.shape[:2]
    # Adaptive kernel: odd numbers, minimum 7
    kx = max(7, (rw // 12) | 1)
    ky = max(7, (rh // 12) | 1)
    return cv2.GaussianBlur(roi, (kx, ky), sigmaX=0, sigmaY=0)


def _apply_pixelate_roi(roi: np.ndarray, scale: float) -> np.ndarray:
    """
    Pixelate ROI by downscale->upscale.
    scale: 0.05 ~ 0.3 typically. smaller => more pixelated.
    """
    rh, rw = roi.shape[:2]
    # Downscale size (at least 1x1)
    dw = max(1, int(rw * scale))
    dh = max(1, int(rh * scale))
    small = cv2.resize(roi, (dw, dh), interpolation=cv2.INTER_AREA)
    pixel = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)
    return pixel


def apply_mosaic(
    frame: np.ndarray,
    bboxes: Sequence[BBox],
    *,
    mode: MosaicMode = "blur",
    pixelate_scale: float = 0.12,
) -> np.ndarray:
    """
    Apply mosaic effect to given bboxes on a frame.

    Args:
        frame: BGR image (H, W, 3), uint8
        bboxes: list/seq of (x1, y1, x2, y2) in pixel coordinates
        mode: "blur" or "pixelate"
        pixelate_scale: downscale ratio for pixelate (smaller => stronger mosaic)

    Returns:
        frame_mosaic: np.ndarray (same object modified in-place and returned)
    """
    if frame is None or frame.size == 0:
        return frame
    if not bboxes:
        return frame

    h, w = frame.shape[:2]
    # Work in-place for speed
    out = frame

    # Clamp scale
    if pixelate_scale <= 0:
        pixelate_scale = 0.12
    pixelate_scale = float(np.clip(pixelate_scale, 0.02, 0.5))

    for b in bboxes:
        cb = _clip_bbox(b, w, h)
        if cb is None:
            continue
        x1, y1, x2, y2 = cb

        roi = out[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        if mode == "blur":
            out[y1:y2, x1:x2] = _apply_blur_roi(roi)
        elif mode == "pixelate":
            out[y1:y2, x1:x2] = _apply_pixelate_roi(roi, pixelate_scale)
        else:
            raise ValueError(f"Unknown mosaic mode: {mode}")

    return out


# ---- Minimal self-test (optional) ----
if __name__ == "__main__":
    # Quick visual sanity test:
    # python services/mosaic.py
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.putText(frame, "MOSAIC TEST", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)

    boxes = [(40, 120, 600, 260)]
    out1 = apply_mosaic(frame.copy(), boxes, mode="blur")
    out2 = apply_mosaic(frame.copy(), boxes, mode="pixelate", pixelate_scale=0.08)

    cv2.imwrite("mosaic_blur_test.png", out1)
    cv2.imwrite("mosaic_pixel_test.png", out2)
    print("Saved: mosaic_blur_test.png, mosaic_pixel_test.png")
