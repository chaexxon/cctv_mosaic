# services/plate_detector.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None  # type: ignore

BBox = Tuple[int, int, int, int]


@dataclass
class PlateDet:
    bbox: BBox
    conf: float
    cls: str = "plate"


class PlateDetector:
    """
    YOLOv8 plate detector wrapper (robust + tuned for small/far plates).

    Output format (list of dict):
      { "bbox": (x1,y1,x2,y2), "conf": float, "cls": "plate" }

    Design goals:
    - Catch small plates: run at higher imgsz (default 1280) + lower model conf (default 0.08)
    - Avoid exploding false positives: apply post-filters (area/aspect/edge margin)
    - Be tolerant to different scenes: allow overriding via ctor args / CLI pass-through

    Notes:
    - If model weights are missing, raises FileNotFoundError (caller can disable plate gracefully).
    - Ultralytics expects BGR np.ndarray; this wrapper accepts BGR frames.
    """

    def __init__(
        self,
        *,
        model_path: str = "models/yolo_plate/yolov8n-plate.pt",
        # Model-level prefilter (keep low for recall; post-filters will clean up)
        conf_thres: float = 0.08,
        # Inference resolution: higher helps far plates (CPU will be slower)
        imgsz: int = 1280,
        max_dets: int = 300,
        device: Optional[str] = None,
        # Post filters (frame-relative)
        min_area_ratio: float = 0.00003,  # 0.003% of frame area (tune; far plates are tiny)
        max_area_ratio: float = 0.05,     # 5% of frame area (avoid huge false positives)
        # Typical Korean plate bbox aspect (w/h). Allow wide range but exclude nonsense.
        min_ar: float = 1.4,
        max_ar: float = 6.5,
        # Reject boxes hugging border (often false boxes / partial)
        border_margin_ratio: float = 0.0,  # set 0.01~0.03 if you see edge false positives
    ):
        if YOLO is None:
            raise ImportError("ultralytics is not installed. pip install ultralytics")

        self.model_path = str(model_path)
        self.conf_thres = float(conf_thres)
        self.imgsz = int(imgsz)
        self.max_dets = int(max_dets)
        self.device = device

        self.min_area_ratio = float(min_area_ratio)
        self.max_area_ratio = float(max_area_ratio)
        self.min_ar = float(min_ar)
        self.max_ar = float(max_ar)
        self.border_margin_ratio = float(border_margin_ratio)

        p = Path(self.model_path)
        if not p.exists():
            raise FileNotFoundError(
                f"Plate model not found: {p}\n"
                f"Put weights there OR pass --plate_model_path to scripts/process_video.py"
            )

        self.model = YOLO(str(p))

    @staticmethod
    def _clip_bbox(b: BBox, w: int, h: int) -> Optional[BBox]:
        x1, y1, x2, y2 = b
        x1 = max(0, min(int(x1), w - 1))
        y1 = max(0, min(int(y1), h - 1))
        x2 = max(0, min(int(x2), w))
        y2 = max(0, min(int(y2), h))
        if x2 <= x1 + 1 or y2 <= y1 + 1:
            return None
        return (x1, y1, x2, y2)

    def _passes_filters(self, bbox: BBox, conf: float, w: int, h: int) -> bool:
        # conf already prefiltered by model, but keep a second guard (in case)
        if conf < self.conf_thres:
            return False

        x1, y1, x2, y2 = bbox
        bw = max(0, x2 - x1)
        bh = max(0, y2 - y1)
        if bw <= 1 or bh <= 1:
            return False

        frame_area = float(w * h)
        area = float(bw * bh)
        if area < self.min_area_ratio * frame_area:
            return False
        if area > self.max_area_ratio * frame_area:
            return False

        ar = bw / float(bh)
        if ar < self.min_ar or ar > self.max_ar:
            return False

        # optional border margin filter
        if self.border_margin_ratio > 0:
            mx = int(w * self.border_margin_ratio)
            my = int(h * self.border_margin_ratio)
            if x1 <= mx or y1 <= my or x2 >= (w - mx) or y2 >= (h - my):
                return False

        return True

    def detect(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
        h, w = frame_bgr.shape[:2]

        # Ultralytics expects BGR np.ndarray OK
        results = self.model.predict(
            source=frame_bgr,
            conf=self.conf_thres,   # keep low for recall, post-filter later
            imgsz=self.imgsz,       # higher -> better for far plates
            device=self.device,
            verbose=False,
            max_det=self.max_dets,
        )

        out: List[Dict[str, Any]] = []
        if not results:
            return out

        r0 = results[0]
        boxes = getattr(r0, "boxes", None)
        if boxes is None or getattr(boxes, "xyxy", None) is None:
            return out

        xyxy = boxes.xyxy
        if xyxy is None:
            return out

        xyxy_np = xyxy.cpu().numpy()
        confs_np = (
            boxes.conf.cpu().numpy()
            if getattr(boxes, "conf", None) is not None
            else np.ones((len(xyxy_np),), dtype=np.float32)
        )

        for bb, cf in zip(xyxy_np, confs_np):
            try:
                x1, y1, x2, y2 = [int(v) for v in bb.tolist()]
            except Exception:
                continue

            cb = self._clip_bbox((x1, y1, x2, y2), w, h)
            if cb is None:
                continue

            conf = float(cf)
            if not self._passes_filters(cb, conf, w, h):
                continue

            out.append({"bbox": cb, "conf": conf, "cls": "plate"})

        return out
