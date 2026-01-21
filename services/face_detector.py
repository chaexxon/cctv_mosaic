# services/face_detector.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None  # type: ignore


BBox = Tuple[int, int, int, int]  # (x1,y1,x2,y2)


@dataclass
class FaceDet:
    bbox: BBox
    conf: float
    cls: str = "face"


class FaceDetector:
    """
    YOLOv8-face detector wrapper with post-filters to reduce false positives.

    Filters (in order):
    1) conf >= conf_thres
    2) bbox area >= min_area_ratio * (W*H)
    3) aspect ratio in [min_ar, max_ar]
    """

    def __init__(
        self,
        *,
        model_path: str = "models/yolo_face/yolov8n-face.pt",
        conf_thres: float = 0.55,
        min_area_ratio: float = 0.001,  # 0.2% of frame area (tune)
        min_ar: float = 0.55,          # w/h lower bound
        max_ar: float = 2.00,          # w/h upper bound
        max_dets: int = 50,
        imgsz: int = 640,
        device: Optional[str] = None,
    ):
        if YOLO is None:
            raise ImportError("ultralytics is not installed. pip install ultralytics")

        self.model_path = model_path
        self.conf_thres = conf_thres
        self.min_area_ratio = min_area_ratio
        self.min_ar = min_ar
        self.max_ar = max_ar
        self.max_dets = max_dets
        self.imgsz = imgsz
        self.device = device

        self.model = YOLO(self.model_path)

    @staticmethod
    def _clip_bbox(b: BBox, w: int, h: int) -> BBox:
        x1, y1, x2, y2 = b
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))
        # ensure proper order
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        return (x1, y1, x2, y2)

    def _passes_filters(self, bbox: BBox, conf: float, w: int, h: int) -> bool:
        if conf < self.conf_thres:
            return False

        x1, y1, x2, y2 = bbox
        bw = max(0, x2 - x1)
        bh = max(0, y2 - y1)
        if bw == 0 or bh == 0:
            return False

        area = bw * bh
        frame_area = w * h
        if area < self.min_area_ratio * frame_area:
            return False

        ar = bw / float(bh)
        if ar < self.min_ar or ar > self.max_ar:
            return False

        return True

    def detect(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """
        Returns list of dicts:
          { "bbox": (x1,y1,x2,y2), "conf": float, "cls": "face" }
        """
        h, w = frame_bgr.shape[:2]

        # Ultralytics expects BGR np.ndarray OK
        results = self.model.predict(
            source=frame_bgr,
            conf=self.conf_thres,   # pre-filter at model level
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
            max_det=self.max_dets,
        )

        out: List[Dict[str, Any]] = []

        if not results:
            return out

        r0 = results[0]
        boxes = getattr(r0, "boxes", None)
        if boxes is None or boxes.xyxy is None:
            return out

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.ones((len(xyxy),), dtype=np.float32)

        for bb, cf in zip(xyxy, confs):
            x1, y1, x2, y2 = [int(v) for v in bb.tolist()]
            bbox = self._clip_bbox((x1, y1, x2, y2), w, h)
            conf = float(cf)

            if self._passes_filters(bbox, conf, w, h):
                out.append({"bbox": bbox, "conf": conf, "cls": "face"})

        return out
