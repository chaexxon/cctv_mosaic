# services/face_recognizer.py
"""
Face Recognizer (C)
- InsightFace(ArcFace) 기반 embedding 추출 모듈
- 안정성/정석/검증을 최우선으로 설계

핵심 원칙
1) 출력 embedding은 항상 (FACE_EMBED_DIM,) float32 + L2 normalize 보장
2) 실행 초기에 설정/모델/차원 검증해서 조용히 망하는 상황 방지
3) bbox가 주어지면: InsightFace가 프레임에서 찾은 얼굴들 중 bbox와 IoU가 가장 큰 얼굴의 embedding 선택
   (=> B의 detector 결과와 결합 가능 / 랜드마크 없이도 안전하게 동작)
4) bbox가 없으면: 프레임 내 가장 큰 얼굴(면적 최대) 선택 (등록/캡처 용도)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np

from core.config import ARCFACE_MODEL_DIR, FACE_EMBED_DIM
from utils.similarity import l2_normalize as _l2_normalize


# -------------------------
# Exceptions
# -------------------------
class FaceRecognizerError(RuntimeError):
    pass


class DependencyMissingError(FaceRecognizerError):
    pass


class FaceNotFoundError(FaceRecognizerError):
    pass


class EmbeddingDimError(FaceRecognizerError):
    pass


# -------------------------
# Types
# -------------------------
BBox = Tuple[int, int, int, int]  # (x1, y1, x2, y2)


@dataclass(frozen=True)
class FaceEmbedding:
    embedding: np.ndarray  # shape: (FACE_EMBED_DIM,), dtype: float32, L2-normalized
    bbox: Optional[BBox] = None
    det_score: Optional[float] = None


# -------------------------
# Utility
# -------------------------
def _clamp_bbox(b: BBox, w: int, h: int) -> BBox:
    x1, y1, x2, y2 = b
    x1 = int(max(0, min(x1, w - 1)))
    x2 = int(max(0, min(x2, w)))
    y1 = int(max(0, min(y1, h - 1)))
    y2 = int(max(0, min(y2, h)))
    if x2 <= x1 or y2 <= y1:
        x2 = min(w, x1 + 1)
        y2 = min(h, y1 + 1)
    return (x1, y1, x2, y2)


def _iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    iw = max(0, inter_x2 - inter_x1)
    ih = max(0, inter_y2 - inter_y1)
    inter = iw * ih

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    denom = area_a + area_b - inter
    if denom <= 0:
        return 0.0
    return float(inter / denom)


def _largest_face_idx(bboxes: List[BBox]) -> int:
    areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in bboxes]
    return int(np.argmax(np.asarray(areas))) if areas else -1


def _safe_l2_norm(x: np.ndarray) -> np.ndarray:
    """
    utils.similarity.l2_normalize()는 norm이 너무 작으면 예외를 던진다.
    recognizer 레벨에서는 영벡터/비정상 벡터를 '안전한 실패'로 처리하기 위해
    여기서 한 번 더 방어한다.
    """
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if not np.isfinite(x).all():
        raise FaceRecognizerError("Embedding contains NaN/Inf.")
    n = float(np.linalg.norm(x))
    if n < 1e-12:
        # 여기서 예외로 처리할지, 0벡터로 반환할지 선택 가능
        # 등록/매칭 로직에선 0벡터는 결국 unknown 처리되므로 안전하게 0벡터 반환
        return np.zeros((x.size,), dtype=np.float32)
    return _l2_normalize(x)


# -------------------------
# Main
# -------------------------
class FaceRecognizer:
    """
    InsightFace FaceAnalysis를 이용해 embedding을 얻는다.

    NOTE
    - OpenCV 프레임(BGR uint8)을 그대로 넣어도 된다.
    - bbox가 주어지면, InsightFace가 찾은 얼굴들 중 bbox와 IoU가 가장 큰 얼굴을 선택한다.
      (B detector와 결합할 때 가장 안전한 방식)
    """

    def __init__(
        self,
        model_root: Optional[str] = None,
        det_size: Tuple[int, int] = (640, 640),
        ctx_id: int = -1,  # CPU: -1, GPU: 0
        iou_threshold: float = 0.2,
        enforce_embed_dim: int = FACE_EMBED_DIM,
        debug_print_loaded_path: bool = False,
    ) -> None:
        self.model_root = model_root or str(ARCFACE_MODEL_DIR)
        self.det_size = det_size
        self.ctx_id = int(ctx_id)
        self.iou_threshold = float(iou_threshold)
        self.enforce_embed_dim = int(enforce_embed_dim)

        if debug_print_loaded_path:
            print("[DEBUG] FaceRecognizer loaded from:", __file__)

        if not (0.0 <= self.iou_threshold <= 1.0):
            raise ValueError("iou_threshold must be in [0, 1]")

        try:
            from insightface.app import FaceAnalysis  # type: ignore
        except Exception as e:
            raise DependencyMissingError(
                "insightface가 설치되어 있지 않거나 import에 실패했습니다. "
                "pip install insightface onnxruntime opencv-python 를 먼저 설치하세요."
            ) from e

        self._app = FaceAnalysis(name="buffalo_l", root=self.model_root)
        self._app.prepare(ctx_id=self.ctx_id, det_size=self.det_size)

    def extract_from_frame(
        self,
        frame_bgr: np.ndarray,
        bbox: Optional[BBox] = None,
    ) -> FaceEmbedding:
        """
        frame에서 얼굴 embedding 1개를 선택/추출.
        """
        if not isinstance(frame_bgr, np.ndarray):
            raise TypeError("frame_bgr must be a numpy.ndarray")
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("frame_bgr must have shape (H, W, 3)")
        if frame_bgr.dtype != np.uint8:
            frame_bgr = frame_bgr.astype(np.uint8, copy=False)

        h, w = frame_bgr.shape[:2]

        faces = self._app.get(frame_bgr)  # detect + embed + landmarks
        if not faces:
            raise FaceNotFoundError("프레임에서 얼굴을 찾지 못했습니다.")

        cand_bboxes: List[BBox] = []
        cand_scores: List[float] = []
        cand_embs: List[Optional[np.ndarray]] = []

        for f in faces:
            fb = f.bbox
            cb = _clamp_bbox((int(fb[0]), int(fb[1]), int(fb[2]), int(fb[3])), w, h)
            cand_bboxes.append(cb)
            cand_scores.append(float(getattr(f, "det_score", 0.0)))

            emb = getattr(f, "embedding", None)
            if emb is None:
                cand_embs.append(None)
                continue
            cand_embs.append(np.asarray(emb, dtype=np.float32))

        valid_idxs = [i for i, e in enumerate(cand_embs) if e is not None]
        if not valid_idxs:
            raise FaceNotFoundError("얼굴은 검출됐지만 embedding 추출에 실패했습니다.")

        def pick_idx_by_rule() -> int:
            if bbox is None:
                vb = [cand_bboxes[i] for i in valid_idxs]
                li = _largest_face_idx(vb)
                if li < 0:
                    return -1
                return valid_idxs[li]
            else:
                tb = _clamp_bbox(bbox, w, h)
                ious = [_iou(tb, cand_bboxes[i]) for i in valid_idxs]
                best = int(np.argmax(np.asarray(ious)))
                if ious[best] < self.iou_threshold:
                    return -1
                return valid_idxs[best]

        idx = pick_idx_by_rule()
        if idx < 0:
            if bbox is None:
                raise FaceNotFoundError("유효한 얼굴 bbox를 찾지 못했습니다.")
            raise FaceNotFoundError("bbox와 매칭되는 얼굴을 찾지 못했습니다.")

        emb = np.asarray(cand_embs[idx], dtype=np.float32).reshape(-1)  # type: ignore

        if self.enforce_embed_dim > 0 and emb.shape[0] != self.enforce_embed_dim:
            raise EmbeddingDimError(
                f"Embedding dim mismatch: got {emb.shape[0]}, expected {self.enforce_embed_dim}"
            )

        emb = _safe_l2_norm(emb)

        return FaceEmbedding(
            embedding=emb,
            bbox=cand_bboxes[idx],
            det_score=cand_scores[idx],
        )

    # ==========================================================
    # ✅ 호환용 API (테스트 코드/기존 코드가 기대하던 이름들)
    # ==========================================================
    def extract_from_crop(self, face_bgr: np.ndarray) -> np.ndarray:
        """
        얼굴 crop(BGR)만 넣으면 embedding (FACE_EMBED_DIM,) 반환.
        - 내부적으로 extract_from_frame을 재사용
        """
        fe = self.extract_from_frame(face_bgr, bbox=None)
        return fe.embedding

    def extract(self, face_bgr: np.ndarray) -> np.ndarray:
        """(호환용 alias)"""
        return self.extract_from_crop(face_bgr)
