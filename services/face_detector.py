# services/face_detector.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np

try:
    from insightface.app import FaceAnalysis
except Exception as e:
    FaceAnalysis = None


BBox = Tuple[int, int, int, int]


@dataclass(frozen=True)
class DetectedFace:
    bbox: BBox
    score: float  # detection confidence


class FaceDetector:
    """
    InsightFace(det_10g) 기반 얼굴 검출 (여러 명 지원)
    - OpenCV Haar처럼 배경/나무 오탐 거의 없음
    - GPU 있으면 CUDAProvider로 붙일 수 있음(지금은 CPU여도 OK)
    """

    def __init__(
        self,
        det_size: Tuple[int, int] = (640, 640),
        det_thresh: float = 0.55,   # ⭐ 낮추면 더 많이 잡지만 오탐↑ / 높이면 덜 잡고 안정↑
        providers: Optional[List[str]] = None,
    ):
        if FaceAnalysis is None:
            raise ImportError(
                "insightface is not available. Install insightface or use FaceRecognizer detector."
            )

        if providers is None:
            # 너 로그상 CPUExecutionProvider만 가능하니 기본 CPU로 둠
            providers = ["CPUExecutionProvider"]

        self.app = FaceAnalysis(name="buffalo_l", providers=providers)
        self.app.prepare(ctx_id=0, det_size=det_size, det_thresh=det_thresh)

    def detect(self, frame_bgr: np.ndarray) -> List[DetectedFace]:
        if frame_bgr is None or frame_bgr.size == 0:
            return []

        faces = self.app.get(frame_bgr)  # 여러 얼굴
        out: List[DetectedFace] = []

        for f in faces:
            # f.bbox: [x1,y1,x2,y2] float
            x1, y1, x2, y2 = map(int, f.bbox)
            score = float(getattr(f, "det_score", 1.0))
            out.append(DetectedFace(bbox=(x1, y1, x2, y2), score=score))

        return out
