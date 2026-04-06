# utils/similarity.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np


class SimilarityError(ValueError):
    """Embedding shape/dtype 등의 입력 문제를 명확히 알리기 위한 예외."""


def _as_float32_1d(x: np.ndarray, name: str = "embedding") -> np.ndarray:
    """
    embedding을 안전한 표준 형태로 변환:
    - np.ndarray 강제
    - float32 변환
    - 1D shape (D,) 강제 (만약 (1,D)면 squeeze 허용)
    """
    if x is None:
        raise SimilarityError(f"{name} is None")

    if not isinstance(x, np.ndarray):
        raise SimilarityError(f"{name} must be np.ndarray, got {type(x)}")

    if x.ndim == 2 and 1 in x.shape:
        x = x.reshape(-1)  # (1,D) 또는 (D,1) -> (D,)

    if x.ndim != 1:
        raise SimilarityError(f"{name} must be 1D array (D,), got shape={x.shape}")

    if x.size == 0:
        raise SimilarityError(f"{name} is empty")

    return x.astype(np.float32, copy=False)


def l2_normalize(x: np.ndarray, eps: float = 1e-12, name: str = "embedding") -> np.ndarray:
    """
    L2 normalize (안정적):
    - norm이 0에 가까우면 예외 발생(조용히 NaN 만들지 않음)
    """
    x = _as_float32_1d(x, name=name)
    norm = float(np.linalg.norm(x))
    if norm < eps:
        raise SimilarityError(f"{name} norm too small ({norm}). Check embedding extraction.")
    return x / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    cosine similarity of two embeddings (float).
    """
    a_n = l2_normalize(a, name="a")
    b_n = l2_normalize(b, name="b")
    return float(np.dot(a_n, b_n))


def normalize_db_embeddings(db_embeddings: np.ndarray) -> np.ndarray:
    """
    DB embeddings 표준화:
    - shape: (N,D)
    - dtype: float32
    - 각 row L2 normalize
    """
    if db_embeddings is None:
        return np.empty((0, 0), dtype=np.float32)

    if not isinstance(db_embeddings, np.ndarray):
        raise SimilarityError(f"db_embeddings must be np.ndarray, got {type(db_embeddings)}")

    if db_embeddings.ndim != 2:
        raise SimilarityError(f"db_embeddings must be 2D array (N,D), got shape={db_embeddings.shape}")

    n, d = db_embeddings.shape
    if n == 0 or d == 0:
        return np.empty((0, d), dtype=np.float32)

    db = db_embeddings.astype(np.float32, copy=False)

    norms = np.linalg.norm(db, axis=1, keepdims=True)
    if np.any(norms < 1e-12):
        # 특정 row가 0벡터면 데이터 자체가 잘못된 것
        bad_idx = int(np.where(norms[:, 0] < 1e-12)[0][0])
        raise SimilarityError(f"db_embeddings has near-zero vector at row {bad_idx}")

    return db / norms


def top_k_cosine(query: np.ndarray, db_embeddings: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    query: (D,)
    db_embeddings: (N,D)
    return:
      idxs: (k,)
      scores: (k,)
    - db가 비었으면 빈 배열 반환
    - k는 자동으로 min(k,N) 처리
    """
    if k < 1:
        raise SimilarityError("k must be >= 1")

    q = l2_normalize(query, name="query")

    if db_embeddings is None or (isinstance(db_embeddings, np.ndarray) and db_embeddings.size == 0):
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

    db = normalize_db_embeddings(db_embeddings)
    if db.size == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

    # (N,D) @ (D,) -> (N,)
    scores = db @ q

    n = scores.shape[0]
    k_eff = min(k, n)

    # 큰 N에서 효율적인 top-k
    idx_part = np.argpartition(scores, -k_eff)[-k_eff:]
    idx_sorted = idx_part[np.argsort(scores[idx_part])[::-1]]

    return idx_sorted.astype(np.int64), scores[idx_sorted].astype(np.float32)


@dataclass(frozen=True)
class FaceMatch:
    idx: int
    score: float