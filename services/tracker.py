# services/tracker.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

BBox = Tuple[int, int, int, int]

def iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0, inter_x2 - inter_x1)
    ih = max(0, inter_y2 - inter_y1)
    inter = iw * ih
    a_area = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    b_area = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = a_area + b_area - inter + 1e-9
    return inter / union

@dataclass
class Track:
    track_id: int
    bbox: BBox
    age: int = 0
    miss: int = 0

class IoUTracker:
    def __init__(self, iou_thres: float = 0.3, max_miss: int = 10):
        self.iou_thres = float(iou_thres)
        self.max_miss = int(max_miss)
        self._next_id = 1
        self._tracks: Dict[int, Track] = {}

    def update(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        det_boxes = [d["bbox"] for d in detections]

        assigned_det = set()
        updated_tracks: Dict[int, Track] = {}

        # match existing tracks
        for tid, tr in self._tracks.items():
            best_j = -1
            best_iou = 0.0
            for j, b in enumerate(det_boxes):
                if j in assigned_det:
                    continue
                v = iou(tr.bbox, b)
                if v > best_iou:
                    best_iou = v
                    best_j = j

            if best_j >= 0 and best_iou >= self.iou_thres:
                tr.bbox = det_boxes[best_j]
                tr.age += 1
                tr.miss = 0
                assigned_det.add(best_j)
                updated_tracks[tid] = tr
            else:
                tr.miss += 1
                tr.age += 1
                if tr.miss <= self.max_miss:
                    updated_tracks[tid] = tr

        # create new tracks for unmatched detections
        for j, b in enumerate(det_boxes):
            if j in assigned_det:
                continue
            tid = self._next_id
            self._next_id += 1
            updated_tracks[tid] = Track(track_id=tid, bbox=b, age=1, miss=0)

        self._tracks = updated_tracks
        return [{"track_id": t.track_id, "bbox": t.bbox, "cls": "face"} for t in self._tracks.values()]
