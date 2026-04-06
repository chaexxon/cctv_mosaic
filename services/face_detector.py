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
    YOLOv8-face detector wrapper.

    핵심 정책:
    - model.predict(conf=...)는 "너무 높게" 잡으면 박스가 아예 안 나옴.
      => predict_conf는 낮게 두고(post-filter로 정제), 최종 conf_thres로 거르는 걸 추천.

    Post-filters (optional):
      1) conf >= conf_thres
      2) bbox area >= min_area_ratio * (W*H)
      3) aspect ratio in [min_ar, max_ar]
    """

    def __init__(
        self,
        *,
        model_path: str = "models/yolo_face/yolov8n-face.pt",
        # ✅ 최종 conf (후처리 기준)
        conf_thres: float = 0.30,
        # ✅ predict 단계 conf (낮게 두기: 박스 생성 자체를 막지 않도록)
        predict_conf: float = 0.15,
        # ✅ 원거리 얼굴까지 살리려면 낮추기 (1920x1080 기준 0.00025 => 약 518px)
        min_area_ratio: float = 0.00025,
        min_ar: float = 0.45,
        max_ar: float = 2.50,
        max_dets: int = 50,
        imgsz: int = 640,
        device: Optional[str] = None,
        # ✅ 급할 때: 후처리 필터를 거의 끄고 “일단 잡히게”
        disable_post_filters: bool = False,
    ):
        if YOLO is None:
            raise ImportError("ultralytics is not installed. pip install ultralytics")

        mp = Path(model_path)
        if not mp.exists():
            raise FileNotFoundError(f"Face model not found: {mp}")

        self.model_path = str(mp)
        self.conf_thres = float(conf_thres)
        self.predict_conf = float(predict_conf)
        self.min_area_ratio = float(min_area_ratio)
        self.min_ar = float(min_ar)
        self.max_ar = float(max_ar)
        self.max_dets = int(max_dets)
        self.imgsz = int(imgsz)
        self.device = device
        self.disable_post_filters = bool(disable_post_filters)

        self.model = YOLO(self.model_path)

    @staticmethod
    def _clip_bbox(b: BBox, w: int, h: int) -> Optional[BBox]:
        x1, y1, x2, y2 = b

        # xyxy가 뒤집혀 들어오는 경우 방어
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1

        x1 = max(0, min(int(x1), w - 1))
        y1 = max(0, min(int(y1), h - 1))
        x2 = max(0, min(int(x2), w - 1))
        y2 = max(0, min(int(y2), h - 1))

        if x2 <= x1 or y2 <= y1:
            return None
        return (x1, y1, x2, y2)

    def _passes_filters(self, bbox: BBox, conf: float, w: int, h: int) -> bool:
        if self.disable_post_filters:
            return True

        if conf < self.conf_thres:
            return False

        x1, y1, x2, y2 = bbox
        bw = x2 - x1
        bh = y2 - y1
        if bw <= 0 or bh <= 0:
            return False

        area = bw * bh
        frame_area = max(1, w * h)
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

        results = self.model.predict(
            source=frame_bgr,
            conf=self.predict_conf,  # ✅ 낮게
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
            cb = self._clip_bbox((x1, y1, x2, y2), w, h)
            if cb is None:
                continue
            conf = float(cf)

            if self._passes_filters(cb, conf, w, h):
                out.append({"bbox": cb, "conf": conf, "cls": "face"})

        return out
