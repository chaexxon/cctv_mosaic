# services/identity_matcher.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from core.config import FACE_EMBED_DIM, FACE_SIM_THRESHOLD
from db.sqlite import SQLiteDB, DBError
from utils.similarity import top_k_cosine, cosine_similarity, SimilarityError


@dataclass(frozen=True)
class FaceMatch:
    """단일 매칭 결과(1개)."""
    is_registered: bool
    label: str
    score: float
    face_id: Optional[int] = None


@dataclass(frozen=True)
class FaceMatchTopK:
    """디버깅/분석용 top-k 결과."""
    face_ids: List[int]
    labels: List[str]
    scores: List[float]


class IdentityMatcher:
    """
    얼굴 임베딩 매칭기.

    중요:
    - db.list_faces()가 두 가지 형태를 모두 지원하도록 방어
      1) C 원래 버전: f.embedding (np.ndarray)
      2) 현재 네 버전: f.embedding_blob (bytes)
    """

    def __init__(
        self,
        db: Optional[SQLiteDB] = None,
        threshold: float = FACE_SIM_THRESHOLD,
        embed_dim: int = FACE_EMBED_DIM,
        top_k: int = 5,
        enable_cache: bool = True,
    ):
        self.db = db if db is not None else SQLiteDB()
        self.threshold = float(threshold)
        self.embed_dim = int(embed_dim)
        self.top_k = int(top_k)
        self.enable_cache = bool(enable_cache)

        self._cached_ids: List[int] = []
        self._cached_labels: List[str] = []
        self._cached_matrix: Optional[np.ndarray] = None

        self._validate_params()

        if self.enable_cache:
            self.refresh_cache()

    def _validate_params(self) -> None:
        if self.embed_dim < 1:
            raise ValueError("embed_dim must be >= 1")
        if not (0.0 < self.threshold <= 1.0):
            raise ValueError("threshold must be in (0, 1]")
        if self.top_k < 1:
            raise ValueError("top_k must be >= 1")

    def _embedding_from_face_row(self, f) -> np.ndarray:
        """
        Face row에서 embedding을 안전하게 추출.
        지원 형태:
        - f.embedding : np.ndarray
        - f.embedding_blob : bytes
        """
        # 1) C 원래 버전
        if hasattr(f, "embedding"):
            emb = getattr(f, "embedding")
            if not isinstance(emb, np.ndarray):
                raise DBError("DB returned non-ndarray embedding")
            emb = np.asarray(emb, dtype=np.float32).reshape(-1)

        # 2) 현재 네 sqlite.py 버전
        elif hasattr(f, "embedding_blob"):
            blob = getattr(f, "embedding_blob")
            if blob is None:
                raise DBError("embedding_blob is None")
            if not isinstance(blob, (bytes, bytearray, memoryview)):
                raise DBError(f"embedding_blob must be bytes-like, got {type(blob)}")
            emb = np.frombuffer(bytes(blob), dtype=np.float32).copy().reshape(-1)

        else:
            raise DBError("Face row has neither 'embedding' nor 'embedding_blob'")

        if emb.ndim != 1 or emb.size != self.embed_dim:
            raise DBError(
                f"DB embedding dim mismatch: expected ({self.embed_dim},), got {emb.shape}"
            )

        if not np.isfinite(emb).all():
            raise DBError("DB embedding contains NaN/Inf")

        # cosine similarity용 정규화
        norm = float(np.linalg.norm(emb))
        if norm < 1e-12:
            raise DBError("DB embedding norm too small")
        emb = emb / norm

        return emb.astype(np.float32, copy=False)

    def _validate_query_embedding(self, query_embedding: np.ndarray) -> np.ndarray:
        if not isinstance(query_embedding, np.ndarray):
            raise TypeError(f"query_embedding must be np.ndarray, got {type(query_embedding)}")

        q = np.asarray(query_embedding, dtype=np.float32).reshape(-1)

        if q.ndim != 1 or q.size != self.embed_dim:
            raise ValueError(
                f"query_embedding must have shape ({self.embed_dim},), got {q.shape}"
            )

        if not np.isfinite(q).all():
            raise ValueError("query_embedding contains NaN/Inf")

        norm = float(np.linalg.norm(q))
        if norm < 1e-12:
            raise ValueError("query_embedding norm is too small")
        q = q / norm

        return q.astype(np.float32, copy=False)

    def refresh_cache(self) -> None:
        """
        DB 얼굴 임베딩을 다시 읽어 캐시에 반영.
        """
        faces = self.db.list_faces()

        if len(faces) == 0:
            self._cached_ids = []
            self._cached_labels = []
            self._cached_matrix = None
            return

        ids: List[int] = []
        labels: List[str] = []
        mats: List[np.ndarray] = []

        for f in faces:
            emb = self._embedding_from_face_row(f)
            ids.append(int(f.id))
            labels.append(str(f.name))
            mats.append(emb)

        self._cached_ids = ids
        self._cached_labels = labels
        self._cached_matrix = np.stack(mats, axis=0).astype(np.float32, copy=False)

    def _get_db_matrix(self) -> Tuple[List[int], List[str], Optional[np.ndarray]]:
        if self.enable_cache:
            return self._cached_ids, self._cached_labels, self._cached_matrix

        faces = self.db.list_faces()
        if len(faces) == 0:
            return [], [], None

        ids: List[int] = []
        labels: List[str] = []
        mats: List[np.ndarray] = []

        for f in faces:
            emb = self._embedding_from_face_row(f)
            ids.append(int(f.id))
            labels.append(str(f.name))
            mats.append(emb)

        mat = np.stack(mats, axis=0).astype(np.float32, copy=False)
        return ids, labels, mat

    def match_face(self, query_embedding: np.ndarray) -> FaceMatch:
        """
        입력 임베딩 1개를 DB와 비교해 최종 판정 반환.
        """
        query = self._validate_query_embedding(query_embedding)

        ids, labels, db_mat = self._get_db_matrix()
        if db_mat is None or db_mat.shape[0] == 0:
            return FaceMatch(is_registered=False, label="unknown", score=0.0, face_id=None)

        k = min(self.top_k, db_mat.shape[0])
        idxs, scores = top_k_cosine(query, db_mat, k=k)

        best_local = int(idxs[0])
        best_score = float(scores[0])

        if best_score >= self.threshold:
            return FaceMatch(
                is_registered=True,
                label=labels[best_local],
                score=best_score,
                face_id=ids[best_local],
            )

        return FaceMatch(
            is_registered=False,
            label="unknown",
            score=best_score,
            face_id=None,
        )

    def match_face_topk(self, query_embedding: np.ndarray, k: Optional[int] = None) -> FaceMatchTopK:
        """
        디버깅용 top-k 반환.
        """
        query = self._validate_query_embedding(query_embedding)

        ids, labels, db_mat = self._get_db_matrix()
        if db_mat is None or db_mat.shape[0] == 0:
            return FaceMatchTopK(face_ids=[], labels=[], scores=[])

        kk = min(int(k) if k is not None else self.top_k, db_mat.shape[0])
        idxs, scores = top_k_cosine(query, db_mat, k=kk)

        out_ids = [ids[int(i)] for i in idxs]
        out_labels = [labels[int(i)] for i in idxs]
        out_scores = [float(s) for s in scores]
        return FaceMatchTopK(face_ids=out_ids, labels=out_labels, scores=out_scores)

    def best_similarity(self, query_embedding: np.ndarray) -> float:
        """
        DB 전체와의 최고 cosine similarity만 반환.
        """
        query = self._validate_query_embedding(query_embedding)

        _, _, db_mat = self._get_db_matrix()
        if db_mat is None or db_mat.shape[0] == 0:
            return 0.0

        sims = db_mat @ query
        return float(np.max(sims))

    def __len__(self) -> int:
        ids, _, mat = self._get_db_matrix()
        return 0 if mat is None else len(ids)