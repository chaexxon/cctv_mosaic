# services/face_detector.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

BBox = Tuple[int, int, int, int]  # (x1, y1, x2, y2)


@dataclass
class Detection:
    bbox: BBox
    conf: float
    cls: str  # "face" or "plate"


class FaceDetector:
    """
    B module: face detection wrapper (YOLO/RetinaFace etc.)
    For now: interface only (returns empty list).
    """

    def __init__(self, model_path: str | None = None, conf_thres: float = 0.5):
        self.model_path = model_path
        self.conf_thres = float(conf_thres)
        self._model = None  # TODO: load model later

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Returns:
            detections = [
              {"bbox": (x1,y1,x2,y2), "conf": 0.9, "cls": "face"},
              ...
            ]
        """
        # TODO: implement actual model inference
        return []
