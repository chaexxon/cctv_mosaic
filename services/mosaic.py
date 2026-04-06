# services/mosaic.py
from __future__ import annotations
from typing import Literal, Sequence, Tuple

import cv2
import numpy as np

BBox = Tuple[int, int, int, int]  # (x1, y1, x2, y2)
MosaicMode = Literal["blur", "pixelate"]


def _clip_bbox(b: BBox, w: int, h: int) -> BBox | None:
    """
    Clip bbox to image bounds.
    Return None if bbox is invalid after clipping.
    """
    x1, y1, x2, y2 = b
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w))
    y2 = max(0, min(int(y2), h))

    # invalid / too small
    if x2 <= x1 + 1 or y2 <= y1 + 1:
        return None

    return (x1, y1, x2, y2)


def _apply_blur_roi(roi: np.ndarray) -> np.ndarray:
    """
    Gaussian blur for ROI.
    Kernel size adapts to ROI size.
    """
    rh, rw = roi.shape[:2]
    kx = max(15, (rw // 6) | 1)  # odd
    ky = max(15, (rh // 6) | 1)
    return cv2.GaussianBlur(roi, (kx, ky), sigmaX=0, sigmaY=0)


def _apply_pixelate_roi(roi: np.ndarray, scale: float) -> np.ndarray:
    """
    Pixelate ROI by downscale -> upscale.
    scale: 0.05 ~ 0.3 (smaller = stronger)
    """
    rh, rw = roi.shape[:2]
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
        bboxes: list of (x1, y1, x2, y2)
        mode: "blur" or "pixelate"
        pixelate_scale: downscale ratio for pixelate

    Returns:
        frame (modified in-place and returned)
    """
    if frame is None or frame.size == 0:
        return frame
    if not bboxes:
        return frame

    h, w = frame.shape[:2]
    out = frame  # in-place

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


# ---------------------------
# optional quick self-test
# ---------------------------
if __name__ == "__main__":
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.putText(
        frame,
        "MOSAIC TEST",
        (50, 200),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (255, 255, 255),
        4,
    )
    boxes = [(40, 120, 600, 260)]

    out1 = apply_mosaic(frame.copy(), boxes, mode="blur")
    out2 = apply_mosaic(frame.copy(), boxes, mode="pixelate", pixelate_scale=0.08)

    cv2.imwrite("mosaic_blur_test.png", out1)
    cv2.imwrite("mosaic_pixel_test.png", out2)
    print("Saved: mosaic_blur_test.png, mosaic_pixel_test.png")


#####
def apply_blur_ellipse(frame: np.ndarray, bbox: BBox, *, feather: int = 12) -> np.ndarray:
    """
    bbox 영역을 블러 처리하되, 타원(ellipse) 모양으로만 적용.
    feather: 경계 부드럽게(픽셀). 클수록 자연스러움.
    """
    if frame is None or frame.size == 0:
        return frame

    h, w = frame.shape[:2]
    cb = _clip_bbox(bbox, w, h)
    if cb is None:
        return frame

    x1, y1, x2, y2 = cb
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return frame

    # 1) roi 전체를 블러한 버전 생성
    blurred = _apply_blur_roi(roi.copy())

    # 2) 타원 마스크 생성 (roi 크기 기준)
    mh, mw = roi.shape[:2]
    mask = np.zeros((mh, mw), dtype=np.uint8)

    center = (mw // 2, mh // 2)
    axes = (max(1, int(mw * 0.45)), max(1, int(mh * 0.48)))  # 타원 크기 비율
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)

    # 3) feather(경계 부드럽게)
    if feather and feather > 0:
        k = int(feather) | 1  # odd
        mask = cv2.GaussianBlur(mask, (k, k), 0)

    # 4) 마스크로 합성
    mask_f = (mask.astype(np.float32) / 255.0)[:, :, None]  # (H,W,1)
    out_roi = (blurred * mask_f + roi * (1.0 - mask_f)).astype(np.uint8)

    frame[y1:y2, x1:x2] = out_roi
    return frame
