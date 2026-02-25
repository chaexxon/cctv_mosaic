# utils/plate_vote.py
from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Tuple

from utils.plate_text import PLATE_RE


class PlateVote:
    """
    update(cands) -> (best_norm, best_score, stable)

    stable 조건:
      - best_norm이 PLATE_RE를 만족하고
      - 최근 window 중 best_norm이 나온 프레임 수 >= stable_min_hits
      - best_norm의 점유율 >= stable_ratio
    """

    def __init__(
        self,
        window: int = 35,
        stable_ratio: float = 0.55,
        stable_min_hits: int = 6,
        min_len: int = 2,
    ):
        self.window = int(window)
        self.stable_ratio = float(stable_ratio)
        self.stable_min_hits = int(stable_min_hits)
        self.min_len = int(min_len)
        self._buf: Deque[List[Tuple[str, float]]] = deque(maxlen=self.window)

    def update(self, candidates: List[Tuple[str, float]]):
        cleaned: List[Tuple[str, float]] = []
        for it in candidates or []:
            try:
                norm, conf = it
            except Exception:
                continue

            norm = (norm or "").strip()
            try:
                conf = float(conf)
            except Exception:
                conf = 0.0

            if len(norm) < self.min_len or conf <= 0:
                continue

            cleaned.append((norm, conf))

        self._buf.append(cleaned)

        scores: Dict[str, float] = {}
        hits: Dict[str, int] = {}
        frames = len(self._buf)

        for frame_cands in self._buf:
            seen_in_frame = set()
            for norm, conf in frame_cands:
                scores[norm] = scores.get(norm, 0.0) + conf
                if norm not in seen_in_frame:
                    hits[norm] = hits.get(norm, 0) + 1
                    seen_in_frame.add(norm)

        if not scores:
            return "", 0.0, False

        best_norm = max(scores, key=scores.get)
        best_score = float(scores[best_norm])

        best_hits = hits.get(best_norm, 0)
        ratio = (best_hits / frames) if frames > 0 else 0.0
        stable = (
            bool(PLATE_RE.match(best_norm))
            and best_hits >= self.stable_min_hits
            and ratio >= self.stable_ratio
        )

        return best_norm, best_score, stable
