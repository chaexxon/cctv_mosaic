# services/identity_matcher.py
"""
Identity Matcher (C)

역할
- 얼굴 임베딩(query)을 DB에 저장된 임베딩들과 비교
- cosine similarity 기반 top-k 검색
- 임계값(FACE_SIM_THRESHOLD) 이상이면 "등록됨" 판정

설계 원칙
- 입력/DB 임베딩 shape/dtype 검증 (조용히 망하는 상황 방지)
- DB 비어있으면 안전하게 unknown 반환
- 반복 호출 성능을 위해 DB 임베딩 캐싱 지원
- 팀원 B/A 코드와 충돌 없도록, config/db/utils만 참조 (단일 진실)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from core.config import FACE_EMBED_DIM, FACE_SIM_THRESHOLD
from db.sqlite import SQLiteDB, DBError  # 네 db/sqlite.py에 DBError가 있어야 함
from utils.similarity import top_k_cosine, cosine_similarity, SimilarityError


@dataclass(frozen=True)
class FaceMatch:
    """단일 매칭 결과(1개)."""
    is_registered: bool
    label: str                 # 등록자 name 또는 "unknown"
    score: float               # cosine similarity
    face_id: Optional[int] = None


@dataclass(frozen=True)
class FaceMatchTopK:
    """디버깅/분석을 위한 top-k 결과."""
    face_ids: List[int]
    labels: List[str]
    scores: List[float]


class IdentityMatcher:
    """
    얼굴 임베딩 매칭기.

    사용 예:
        matcher = IdentityMatcher()
        result = matcher.match_face(embedding)
        if result.is_registered:
            ...
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
        self._cached_matrix: Optional[np.ndarray] = None  # shape: (N, D), float32

        self._validate_params()

        # 초기 로딩(캐시 ON이면 최초 1회)
        if self.enable_cache:
            self.refresh_cache()

    def _validate_params(self) -> None:
        if self.embed_dim < 1:
            raise ValueError("embed_dim must be >= 1")
        if not (0.0 < self.threshold <= 1.0):
            raise ValueError("threshold must be in (0, 1]")
        if self.top_k < 1:
            raise ValueError("top_k must be >= 1")

    def refresh_cache(self) -> None:
        """
        DB의 얼굴 임베딩을 다시 읽어서 캐시에 반영.
        (얼굴 등록 후에는 호출 권장)
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
            emb = f.embedding
            if not isinstance(emb, np.ndarray):
                raise DBError("DB returned non-ndarray embedding (corrupted record?)")
            if emb.ndim != 1 or emb.size != self.embed_dim:
                raise DBError(
                    "DB embedding dim mismatch: expected (%d,), got %s"
                    % (self.embed_dim, str(emb.shape))
                )
            ids.append(int(f.id))
            labels.append(str(f.name))
            mats.append(emb.astype(np.float32, copy=False))

        # (N, D)
        self._cached_ids = ids
        self._cached_labels = labels
        self._cached_matrix = np.stack(mats, axis=0).astype(np.float32, copy=False)

    def _get_db_matrix(self) -> Tuple[List[int], List[str], Optional[np.ndarray]]:
        """
        캐시가 있으면 캐시 사용, 없으면 DB에서 즉시 로드.
        """
        if self.enable_cache:
            return self._cached_ids, self._cached_labels, self._cached_matrix

        # cache OFF: 매 호출마다 DB 로드
        faces = self.db.list_faces()
        if len(faces) == 0:
            return [], [], None

        ids = [int(f.id) for f in faces]
        labels = [str(f.name) for f in faces]
        mats = []
        for f in faces:
            emb = f.embedding
            if emb.ndim != 1 or emb.size != self.embed_dim:
                raise DBError(
                    "DB embedding dim mismatch: expected (%d,), got %s"
                    % (self.embed_dim, str(emb.shape))
                )
            mats.append(emb.astype(np.float32, copy=False))
        mat = np.stack(mats, axis=0).astype(np.float32, copy=False)
        return ids, labels, mat

    def match_face(self, query_embedding: np.ndarray) -> FaceMatch:
        """
        입력 임베딩 1개를 DB와 비교해 최종 판정(등록/미등록)을 반환.
        """
        query = self._validate_query_embedding(query_embedding)

        ids, labels, db_mat = self._get_db_matrix()
        if db_mat is None or db_mat.shape[0] == 0:
            return FaceMatch(is_registered=False, label="unknown", score=0.0, face_id=None)

        # top-k 검색
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
        else:
            return FaceMatch(is_registered=False, label="unknown", score=best_score, face_id=None)

    def match_face_topk(self, query_embedding: np.ndarray, k: Optional[int] = None) -> FaceMatchTopK:
        """
        디버깅/로그용: top-k 후보 전체를 반환.
        """
        query = self._validate_query_embedding(query_embedding)

        ids, labels, db_mat = self._get_db_matrix()
        if db_mat is None or db_mat.shape[0] == 0:
            return FaceMatchTopK(face_ids=[], labels=[], scores=[])

        kk = min(int(k) if k is not None else self.top_k, db_mat.shape[0])
        idxs, scores = top_k_cosine(query, db_mat, k=kk)

        out_ids: List[int] = []
        out_labels: List[str] = []
        out_scores: List[float] = []

        for local_i, sc in zip(idxs.tolist(), scores.tolist()):
            local_i = int(local_i)
            out_ids.append(ids[local_i])
            out_labels.append(labels[local_i])
            out_scores.append(float(sc))

        return FaceMatchTopK(face_ids=out_ids, labels=out_labels, scores=out_scores)

    def _validate_query_embedding(self, emb: np.ndarray) -> np.ndarray:
        if not isinstance(emb, np.ndarray):
            raise TypeError("query_embedding must be np.ndarray")
        if emb.ndim != 1 or emb.size != self.embed_dim:
            raise ValueError(
                "query_embedding must be shape (%d,), got %s" % (self.embed_dim, str(emb.shape))
            )
        # float32로 통일 (DB도 float32)
        q = emb.astype(np.float32, copy=False)

        # (선택) 매우 작은 노름/NaN 방어: similarity.py의 안전장치가 있지만 여기서도 선제 방어
        if not np.isfinite(q).all():
            raise ValueError("query_embedding contains NaN/Inf")
        norm = float(np.linalg.norm(q))
        if norm < 1e-12:
            # 영벡터면 어떤 것도 매칭될 수 없음
            return np.zeros((self.embed_dim,), dtype=np.float32)

        return q
