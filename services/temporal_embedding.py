# services/temporal_embedding.py
"""
Temporal Embedding (C)

역할
- tracker가 준 track_id 단위로 얼굴 embedding을 누적
- 최근 N개(TEMPORAL_WINDOW)만 유지
- 대표 embedding을 mean/median으로 계산하여 matcher에 전달

설계 원칙
- 입력 embedding shape/dtype/NaN/Inf 검증
- track별 메모리 누수 방지(자동 정리 기능 제공)
- Python 3.8+ 호환 (팀원 환경 차이 방어)
- config 단일진실(core.config)만 참조

주의
- 여기서는 "대표 embedding"만 만든다.
- 등록 판정/정책 결정은 pipeline/identity_matcher에서 한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Deque, Dict, Optional, Literal, Tuple

import numpy as np

from core.config import FACE_EMBED_DIM, TEMPORAL_WINDOW, TEMPORAL_MODE


TemporalMode = Literal["mean", "median"]


@dataclass
class TrackEmbeddingState:
    """track별 임베딩 누적 상태."""
    buf: Deque[np.ndarray]
    last_seen_frame: int


class TemporalEmbedder:
    """
    트랙별 임베딩을 누적해서 대표 임베딩을 산출한다.

    사용 예:
        te = TemporalEmbedder()
        rep = te.update(track_id=7, embedding=emb, frame_idx=123)
        # rep는 shape (D,) float32
    """

    def __init__(
        self,
        window: int = TEMPORAL_WINDOW,
        mode: TemporalMode = TEMPORAL_MODE,
        embed_dim: int = FACE_EMBED_DIM,
        normalize_output: bool = True,
        max_idle_frames: int = 300,
    ):
        """
        Args:
            window: track당 유지할 최근 임베딩 개수
            mode: "mean" or "median"
            embed_dim: 임베딩 차원(ArcFace: 보통 512)
            normalize_output: 대표 벡터를 L2 normalize 할지 여부
            max_idle_frames: 이 프레임 수 이상 업데이트 없는 track은 자동 제거(메모리 누수 방지)
        """
        self.window = int(window)
        self.mode = mode
        self.embed_dim = int(embed_dim)
        self.normalize_output = bool(normalize_output)
        self.max_idle_frames = int(max_idle_frames)

        self._tracks: Dict[int, TrackEmbeddingState] = {}

        self._validate_params()

    def _validate_params(self) -> None:
        if self.window < 1:
            raise ValueError("window must be >= 1")
        if self.embed_dim < 1:
            raise ValueError("embed_dim must be >= 1")
        if self.mode not in ("mean", "median"):
            raise ValueError("mode must be 'mean' or 'median'")
        if self.max_idle_frames < 1:
            raise ValueError("max_idle_frames must be >= 1")

    # -------------------------
    # Public API
    # -------------------------
    def update(self, track_id: int, embedding: np.ndarray, frame_idx: int) -> np.ndarray:
        """
        track_id에 embedding을 추가하고, 현재 대표 임베딩을 반환.

        Args:
            track_id: tracker가 부여한 ID
            embedding: 얼굴 임베딩 (shape=(D,))
            frame_idx: 현재 프레임 인덱스(증가하는 정수)

        Returns:
            rep_embedding: 대표 임베딩 (shape=(D,), float32)
        """
        tid = int(track_id)
        fi = int(frame_idx)
        emb = self._validate_embedding(embedding)

        st = self._tracks.get(tid)
        if st is None:
            st = TrackEmbeddingState(
                buf=deque(maxlen=self.window),
                last_seen_frame=fi,
            )
            self._tracks[tid] = st

        st.buf.append(emb)
        st.last_seen_frame = fi

        # 업데이트 때마다 오래된 track 자동 정리
        self.cleanup(current_frame=fi)

        return self._compute_representation(st.buf)

    def get(self, track_id: int) -> Optional[np.ndarray]:
        """
        해당 track_id의 현재 대표 임베딩을 반환(없으면 None).
        frame_idx 없이도 조회만 가능.
        """
        tid = int(track_id)
        st = self._tracks.get(tid)
        if st is None or len(st.buf) == 0:
            return None
        return self._compute_representation(st.buf)

    def reset(self, track_id: int) -> None:
        """특정 track의 누적 상태 제거."""
        tid = int(track_id)
        if tid in self._tracks:
            del self._tracks[tid]

    def cleanup(self, current_frame: int) -> int:
        """
        오래된 track을 제거하여 메모리 누수를 방지.
        Args:
            current_frame: 현재 프레임 인덱스
        Returns:
            removed_count: 제거된 track 개수
        """
        cf = int(current_frame)
        to_del = []
        for tid, st in self._tracks.items():
            if (cf - st.last_seen_frame) >= self.max_idle_frames:
                to_del.append(tid)

        for tid in to_del:
            del self._tracks[tid]

        return len(to_del)

    def stats(self) -> Tuple[int, int]:
        """
        (track 수, 총 저장된 embedding 수) 반환.
        """
        n_tracks = len(self._tracks)
        total = 0
        for st in self._tracks.values():
            total += len(st.buf)
        return n_tracks, total

    # -------------------------
    # Internals
    # -------------------------
    def _validate_embedding(self, emb: np.ndarray) -> np.ndarray:
        if not isinstance(emb, np.ndarray):
            raise TypeError("embedding must be np.ndarray")
        if emb.ndim != 1 or emb.size != self.embed_dim:
            raise ValueError(
                "embedding must be shape (%d,), got %s" % (self.embed_dim, str(emb.shape))
            )

        x = emb.astype(np.float32, copy=False)

        if not np.isfinite(x).all():
            raise ValueError("embedding contains NaN/Inf")

        # 영벡터 방어(특이 케이스)
        if float(np.linalg.norm(x)) < 1e-12:
            # 완전히 0이면 대표값도 의미 없으니 0으로 유지
            return np.zeros((self.embed_dim,), dtype=np.float32)

        return x

    def _compute_representation(self, buf: Deque[np.ndarray]) -> np.ndarray:
        if len(buf) == 0:
            return np.zeros((self.embed_dim,), dtype=np.float32)

        mat = np.stack(list(buf), axis=0).astype(np.float32, copy=False)  # (N, D)

        if self.mode == "mean":
            rep = mat.mean(axis=0)
        else:
            # median은 outlier(흔들림/가림)에 조금 더 강함
            rep = np.median(mat, axis=0)

        rep = rep.astype(np.float32, copy=False)

        if self.normalize_output:
            n = float(np.linalg.norm(rep))
            if n >= 1e-12:
                rep = rep / n
            else:
                rep = np.zeros((self.embed_dim,), dtype=np.float32)

        return rep
