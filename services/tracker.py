# services/tracker.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import math

import cv2
import numpy as np

BBox = Tuple[int, int, int, int]  # xyxy (int)


# ---------- geometry ----------
def iou_xyxy(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter + 1e-9
    return float(inter / union)


def center(b: BBox) -> Tuple[float, float]:
    x1, y1, x2, y2 = b
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


def diag(b: BBox) -> float:
    x1, y1, x2, y2 = b
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    return math.hypot(w, h)


def norm_center_dist(a: BBox, b: BBox) -> float:
    ax, ay = center(a)
    bx, by = center(b)
    d = math.hypot(ax - bx, ay - by)
    return d / max(1.0, 0.5 * (diag(a) + diag(b)))


def ema_bbox(prev: BBox, new: BBox, alpha: float) -> BBox:
    px1, py1, px2, py2 = prev
    nx1, ny1, nx2, ny2 = new
    return (
        int(round(alpha * px1 + (1 - alpha) * nx1)),
        int(round(alpha * py1 + (1 - alpha) * ny1)),
        int(round(alpha * px2 + (1 - alpha) * nx2)),
        int(round(alpha * py2 + (1 - alpha) * ny2)),
    )


# ---------- hungarian ----------
def hungarian(cost: List[List[float]]) -> List[Tuple[int, int]]:
    n_rows = len(cost)
    n_cols = len(cost[0]) if n_rows else 0
    n = max(n_rows, n_cols)
    if n == 0:
        return []

    big = 1e6
    C = [row[:] + [big] * (n - n_cols) for row in cost]
    for _ in range(n - n_rows):
        C.append([big] * n)

    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)
    way = [0] * (n + 1)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [big] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = big
            j1 = 0
            for j in range(1, n + 1):
                if not used[j]:
                    cur = C[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break

        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    out = []
    for j in range(1, n + 1):
        i = p[j]
        if 1 <= i <= n_rows and 1 <= j <= n_cols:
            out.append((i - 1, j - 1))
    return out


# ---------- appearance (reid) ----------
def crop_hist(frame_bgr: np.ndarray, bbox: BBox, bins: int = 16) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w - 1))
    y2 = max(0, min(int(y2), h - 1))
    if x2 <= x1 or y2 <= y1:
        return np.zeros((bins * bins * bins,), dtype=np.float32)

    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((bins * bins * bins,), dtype=np.float32)

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [bins, bins, bins], [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten().astype(np.float32)
    return hist


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a) + 1e-9)
    nb = float(np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b) / (na * nb))


# ---------- raw tracker ----------
@dataclass
class RawTrack:
    raw_id: int
    bbox: BBox
    miss: int = 0
    hits: int = 1
    age: int = 1


class RawIoUTracker:
    """
    목적: raw track_id(tid)를 최대한 안 죽이고 유지
    update(dets) -> list[dict] (dets에 track_id 부여)
    """

    def __init__(
        self,
        *,
        iou_thres: float = 0.10,
        dist_thres: float = 0.90,
        dist_weight: float = 0.20,
        max_miss: int = 180,
        min_hits: int = 3,
        ema_alpha: float = 0.85,
    ):
        self.iou_thres = float(iou_thres)
        self.dist_thres = float(dist_thres)
        self.dist_weight = float(dist_weight)
        self.max_miss = int(max_miss)
        self.min_hits = int(min_hits)
        self.ema_alpha = float(ema_alpha)

        self._next_id = 1
        self._tracks: List[RawTrack] = []

    def reset(self) -> None:
        self._next_id = 1
        self._tracks = []

    def update(self, dets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not dets:
            for t in self._tracks:
                t.age += 1
                t.miss += 1
            self._tracks = [t for t in self._tracks if t.miss <= self.max_miss]
            return []

        det_b = [d["bbox"] for d in dets]

        for t in self._tracks:
            t.age += 1

        # init
        if not self._tracks:
            for d in dets:
                d["track_id"] = self._next_id
                self._tracks.append(RawTrack(self._next_id, d["bbox"]))
                self._next_id += 1
            return self._emit(dets)

        # cost
        cost: List[List[float]] = []
        for t in self._tracks:
            row = []
            for b in det_b:
                i = iou_xyxy(t.bbox, b)
                nd = norm_center_dist(t.bbox, b)
                row.append((1.0 - i) + self.dist_weight * nd)
            cost.append(row)

        pairs = hungarian(cost)
        matched_t, matched_d = set(), set()

        for ti, di in pairs:
            t = self._tracks[ti]
            b = det_b[di]
            i = iou_xyxy(t.bbox, b)
            nd = norm_center_dist(t.bbox, b)

            if (i >= self.iou_thres) or (nd <= self.dist_thres):
                t.bbox = ema_bbox(t.bbox, b, self.ema_alpha)
                t.miss = 0
                t.hits += 1
                dets[di]["track_id"] = t.raw_id
                matched_t.add(ti)
                matched_d.add(di)

        # unmatched tracks miss
        for idx, t in enumerate(self._tracks):
            if idx not in matched_t:
                t.miss += 1

        self._tracks = [t for t in self._tracks if t.miss <= self.max_miss]

        # new tracks
        for di, d in enumerate(dets):
            if di not in matched_d:
                d["track_id"] = self._next_id
                self._tracks.append(RawTrack(self._next_id, d["bbox"]))
                self._next_id += 1

        return self._emit(dets)

    def _emit(self, dets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if self.min_hits <= 1:
            return dets
        ok = {t.raw_id for t in self._tracks if t.hits >= self.min_hits}
        return [d for d in dets if d.get("track_id") in ok]


# ---------- stable id manager ----------
@dataclass
class StableState:
    stable_id: int
    last_bbox: BBox
    last_frame: int
    last_hist: Optional[np.ndarray] = None


class StableIdManager:
    def __init__(
        self,
        *,
        ttl_frames: int = 150,
        reid_iou_gate: float = 0.10,
        reid_dist_gate: float = 1.30,
        reid_hist_gate: float = 0.60,
        use_hist: bool = True,
        min_age_for_reid: int = 30,
        strict_reid_hist_gate: float = 0.70,
    ):
        self.ttl_frames = int(ttl_frames)
        self.reid_iou_gate = float(reid_iou_gate)
        self.reid_dist_gate = float(reid_dist_gate)
        self.reid_hist_gate = float(reid_hist_gate)
        self.use_hist = bool(use_hist)

        self.min_age_for_reid = int(min_age_for_reid)
        self.strict_reid_hist_gate = float(strict_reid_hist_gate)

        self._next_sid = 1
        self._active: Dict[int, StableState] = {}  # raw_id -> state
        self._lost: Dict[int, StableState] = {}    # stable_id -> state

    def reset(self) -> None:
        self._next_sid = 1
        self._active = {}
        self._lost = {}

    def _alloc(self) -> int:
        sid = self._next_sid
        self._next_sid += 1
        return sid

    def _purge(self, frame_idx: int) -> None:
        dead = [sid for sid, st in self._lost.items() if frame_idx - st.last_frame > self.ttl_frames]
        for sid in dead:
            self._lost.pop(sid, None)

    def _best_lost(self, frame: Optional[np.ndarray], bbox: BBox, frame_idx: int) -> Optional[int]:
        if not self._lost:
            return None

        new_hist = None
        if self.use_hist and frame is not None:
            new_hist = crop_hist(frame, bbox)

        best_sid = None
        best_score = -1e9

        for sid, st in self._lost.items():
            if frame_idx - st.last_frame > self.ttl_frames:
                continue

            i = iou_xyxy(bbox, st.last_bbox)
            nd = norm_center_dist(bbox, st.last_bbox)

            if i < self.reid_iou_gate and nd > self.reid_dist_gate:
                continue

            score = 0.0
            score += 1.7 * i
            score += 1.0 * max(0.0, 1.0 - (nd / (self.reid_dist_gate + 1e-9)))

            if self.use_hist and frame is not None and st.last_hist is not None and new_hist is not None:
                hs = cos_sim(new_hist, st.last_hist)

                hist_gate = self.reid_hist_gate
                if (frame_idx - st.last_frame) <= self.min_age_for_reid:
                    hist_gate = max(hist_gate, self.strict_reid_hist_gate)

                if hs < hist_gate:
                    continue

                score += 1.2 * hs

            if score > best_score:
                best_score = score
                best_sid = sid

        return best_sid

    def update(self, frame: Optional[np.ndarray], frame_idx: int, tracked: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        self._purge(frame_idx)

        current_raw = {t["track_id"] for t in tracked}
        gone_raw = [rid for rid in list(self._active.keys()) if rid not in current_raw]
        for rid in gone_raw:
            st = self._active.pop(rid)
            self._lost[st.stable_id] = st

        out: List[Dict[str, Any]] = []
        for t in tracked:
            rid = int(t["track_id"])
            bbox = t["bbox"]

            if rid in self._active:
                st = self._active[rid]
                st.last_bbox = bbox
                st.last_frame = frame_idx
                if self.use_hist and frame is not None:
                    st.last_hist = crop_hist(frame, bbox)
                t["stable_id"] = st.stable_id
                out.append(t)
                continue

            sid = self._best_lost(frame, bbox, frame_idx)
            if sid is not None:
                prev = self._lost.pop(sid)
                self._active[rid] = StableState(
                    stable_id=prev.stable_id,
                    last_bbox=bbox,
                    last_frame=frame_idx,
                    last_hist=crop_hist(frame, bbox) if (self.use_hist and frame is not None) else None,
                )
                t["stable_id"] = prev.stable_id
                out.append(t)
                continue

            new_sid = self._alloc()
            self._active[rid] = StableState(
                stable_id=new_sid,
                last_bbox=bbox,
                last_frame=frame_idx,
                last_hist=crop_hist(frame, bbox) if (self.use_hist and frame is not None) else None,
            )
            t["stable_id"] = new_sid
            out.append(t)

        return out

    def hold_boxes(self, frame_idx: int, hold_last: int) -> List[Tuple[int, BBox]]:
        if hold_last <= 0:
            return []
        out: List[Tuple[int, BBox]] = []
        for st in self._active.values():
            if (frame_idx - st.last_frame) <= hold_last:
                out.append((st.stable_id, st.last_bbox))
        return out


# ---------- unified tracker ----------
class UnifiedTracker:
    def __init__(
        self,
        *,
        # raw
        track_iou: float = 0.10,
        track_dist: float = 0.90,
        track_dist_weight: float = 0.20,
        track_max_miss: int = 180,
        min_hits: int = 3,
        ema_alpha: float = 0.85,
        # stable
        sid_ttl: int = 150,
        reid_iou_gate: float = 0.10,
        reid_dist_gate: float = 1.30,
        reid_hist_gate: float = 0.60,
        use_hist: bool = True,
        # anti-jitter
        min_age_for_reid: int = 30,
        strict_reid_hist_gate: float = 0.70,
    ):
        self._raw: Dict[str, RawIoUTracker] = {}
        self._sid: Dict[str, StableIdManager] = {}

        for k in ["face", "plate"]:
            self._raw[k] = RawIoUTracker(
                iou_thres=track_iou,
                dist_thres=track_dist,
                dist_weight=track_dist_weight,
                max_miss=track_max_miss,
                min_hits=min_hits,
                ema_alpha=ema_alpha,
            )
            self._sid[k] = StableIdManager(
                ttl_frames=sid_ttl,
                reid_iou_gate=reid_iou_gate,
                reid_dist_gate=reid_dist_gate,
                reid_hist_gate=reid_hist_gate,
                use_hist=use_hist,
                min_age_for_reid=min_age_for_reid,
                strict_reid_hist_gate=strict_reid_hist_gate,
            )

    def reset(self) -> None:
        for k in self._raw:
            self._raw[k].reset()
            self._sid[k].reset()

    def update(self, frame: Optional[np.ndarray], frame_idx: int, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        buckets = {"face": [], "plate": [], "other": []}
        for d in detections:
            cls = d.get("cls", "face")
            if cls in ("face", "plate"):
                buckets[cls].append(d)
            else:
                buckets["other"].append(d)

        out: List[Dict[str, Any]] = []
        for cls in ("face", "plate"):
            dets = buckets[cls]
            tracked = self._raw[cls].update(dets) if dets else []
            stable = self._sid[cls].update(frame, frame_idx, tracked) if tracked else []
            out.extend(stable)

        out.extend(buckets["other"])
        return out

    def hold_boxes(self, frame_idx: int, hold_last: int, *, cls: str = "face") -> List[Tuple[int, BBox]]:
        if cls not in self._sid:
            return []
        return self._sid[cls].hold_boxes(frame_idx, hold_last)


# ✅ process_video.py가 항상 안정적으로 잡는 "공식 엔트리"
Tracker = RawIoUTracker
IoUTracker = RawIoUTracker
StableTracker = UnifiedTracker
