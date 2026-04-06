# scripts/process_video.py
from __future__ import annotations

import argparse
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

from utils.video import open_capture, get_video_info, iter_frames, make_writer, safe_release
from services.mosaic import apply_mosaic, MosaicMode, BBox
from services.face_detector import FaceDetector

from services.plate_detector import PlateDetector as SPlateDetector
from services.ocr_engine import OCREngine as SOcrEngine
from utils.plate_text import PLATE_RE, normalize_and_fix
from utils.plate_vote import PlateVote
from db.sqlite import SQLiteDB

# C modules
from services.face_recognizer import FaceRecognizer
from services.temporal_embedding import TemporalEmbedder
from services.identity_matcher import IdentityMatcher


# ============================================================
# Robust tracker import (supports your services/tracker.py)
# ============================================================
def _load_tracker_class():
    import services.tracker as tm

    for name in ["UnifiedTracker", "RawIoUTracker", "IoUTracker", "StableTracker", "Tracker"]:
        if hasattr(tm, name):
            return getattr(tm, name)

    for name in dir(tm):
        if name.endswith("Tracker"):
            obj = getattr(tm, name)
            if isinstance(obj, type):
                return obj

    raise ImportError("No tracker class found in services/tracker.py")


TrackerCls = _load_tracker_class()


def _build_tracker(args) -> Any:
    import inspect

    sig = inspect.signature(TrackerCls.__init__)
    allowed = set(sig.parameters.keys())
    kw = {}

    mapping = {
        "track_iou": "track_iou",
        "track_dist": "track_dist",
        "track_dist_weight": "track_dist_weight",
        "track_max_miss": "track_max_miss",
        "min_hits": "min_hits",
        "ema_alpha": "ema_alpha",
        "sid_ttl": "sid_ttl",
        "reid_iou_gate": "reid_iou_gate",
        "reid_dist_gate": "reid_dist_gate",
        "reid_hist_gate": "reid_hist_gate",
        "use_hist": "use_hist",
        "min_age_for_reid": "min_age_for_reid",
        "strict_reid_hist_gate": "strict_reid_hist_gate",
        "iou_thres": "iou_thres",
        "dist_thres": "dist_thres",
        "dist_weight": "dist_weight",
        "max_miss": "max_miss",
    }

    if args is not None:
        for arg_name, ctor_name in mapping.items():
            if ctor_name in allowed and hasattr(args, arg_name):
                kw[ctor_name] = getattr(args, arg_name)

    return TrackerCls(**kw)


# ============================================================
# Helpers
# ============================================================
def _resolve_job_paths(job_id: str):
    jid = str(job_id)
    inp = Path("data/input/segments") / f"clip_{jid}.mp4"
    out = Path("data/output") / f"{jid}.mp4"
    return inp, out


def _encode_h264(tmp: Path, out: Path):
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(tmp),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-movflags",
        "+faststart",
        str(out),
    ]
    subprocess.run(cmd, check=True)


def _draw_box(frame: np.ndarray, bbox: BBox, color: Tuple[int, int, int], label: str = "") -> None:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    if label:
        cv2.putText(
            frame,
            label,
            (x1, max(0, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )


def _pad_bbox(pb: BBox, w: int, h: int, pad_ratio: float = 0.18) -> BBox:
    x1, y1, x2, y2 = pb
    bw = x2 - x1
    bh = y2 - y1
    px = int(bw * pad_ratio)
    py = int(bh * pad_ratio)
    x1 = max(0, x1 - px)
    y1 = max(0, y1 - py)
    x2 = min(w, x2 + px)
    y2 = min(h, y2 + py)
    return (x1, y1, x2, y2)


def _safe_imwrite(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def _ocr_unpack(o: Any) -> Tuple[str, float]:
    """
    Support common OCR return formats:
      - {"text": "...", "conf": 0.9}
      - ("...", 0.9)
      - "..."
    """
    if isinstance(o, dict):
        raw = str(o.get("text", "")).strip()
        conf = float(o.get("conf", 0.0) or 0.0)
        return raw, conf

    if isinstance(o, (tuple, list)):
        raw = str(o[0]).strip() if len(o) >= 1 else ""
        conf = float(o[1]) if len(o) >= 2 else 0.0
        return raw, conf

    raw = str(o).strip() if o is not None else ""
    return raw, 0.0


def _iter_plate_bboxes(plate_dets: Iterable[Any]) -> List[BBox]:
    out: List[BBox] = []
    for d in plate_dets:
        if isinstance(d, dict):
            b = d.get("bbox", None)
        else:
            b = getattr(d, "bbox", None)
        if b is None:
            continue
        x1, y1, x2, y2 = [int(v) for v in b]
        if x2 > x1 and y2 > y1:
            out.append((x1, y1, x2, y2))
    return out


def _lane_id_from_bbox(pb: BBox, frame_w: int, lanes: int = 3) -> int:
    lanes = max(1, int(lanes))
    cx = (pb[0] + pb[2]) / 2.0
    lane_w = max(1.0, frame_w / lanes)
    lane = int(cx // lane_w)
    return max(0, min(lane, lanes - 1))


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


def _nms_boxes(boxes: List[BBox], iou_th: float = 0.55) -> List[BBox]:
    if not boxes:
        return []

    kept: List[BBox] = []
    for b in boxes:
        if all(_iou(b, k) < iou_th for k in kept):
            kept.append(b)
    return kept


def _get_track_val(t: Any, key: str, default=None):
    if isinstance(t, dict):
        return t.get(key, default)
    return getattr(t, key, default)


# ============================================================
# Core
# ============================================================
def process_video(
    input_video_path: str,
    output_video_path: str,
    *,
    mode: MosaicMode = "blur",
    pixelate_scale: float = 0.12,
    # face
    det_conf_face: float = 0.30,
    predict_conf_face: float = 0.15,
    disable_face_filters: bool = False,
    hold_last: int = 30,
    # debug
    draw_boxes: bool = False,
    draw_hold: bool = False,
    print_every: int = 0,
    overwrite: bool = True,
    debug_dump_plate_crops: bool = False,
    debug_dump_every: int = 30,
    debug_dump_dir: str = "data/debug",
    debug_dump_frame_dir: str = "data/debug_plate",
    # plate
    enable_plate: bool = False,
    plate_model_path: str = "models/yolo_plate/yolov8n-plate.pt",
    det_conf_plate: float = 0.35,
    ocr_conf_thres: float = 0.25,
    plate_vote_window: int = 35,
    plate_stable_ratio: float = 0.55,
    plate_stable_min_hits: int = 6,
    plate_lanes: int = 3,
    plate_pad_ratio: float = 0.18,
    # DB
    db_path: str = "db/cctv_mosaic.sqlite3",
    # 등록판 TTL (frames)
    plate_reg_hold_ttl: int = 60,
    args=None,
) -> str:
    cap = None
    writer = None

    out_path = Path(output_video_path)
    tmp_path = out_path.with_suffix(".tmp.mp4")

    if out_path.exists() and not overwrite:
        print(f"[process_video] output exists, skip (use --overwrite to force): {out_path}")
        return str(out_path)

    # lane별 plate vote
    plate_votes: Dict[int, PlateVote] = defaultdict(
        lambda: PlateVote(
            window=plate_vote_window,
            stable_ratio=plate_stable_ratio,
            stable_min_hits=plate_stable_min_hits,
        )
    )

    # lane별 최근 등록판 확정 프레임
    plate_reg_last_ok_frame: Dict[int, int] = defaultdict(lambda: -10**9)

    db = SQLiteDB(db_path)
    regs_exact: set[str] = set()

    try:
        cap = open_capture(input_video_path)
        info = get_video_info(cap)

        writer = make_writer(
            output_path=tmp_path,
            fps=info.fps,
            frame_size=(info.width, info.height),
            codec="mp4v",
            is_color=True,
        )

        # face detect / track
        detector = FaceDetector(
            conf_thres=det_conf_face,
            predict_conf=predict_conf_face,
            disable_post_filters=disable_face_filters,
        )
        tracker = _build_tracker(args)

        # face recognition (C)
        recognizer = FaceRecognizer()
        temporal = TemporalEmbedder()
        matcher = IdentityMatcher(db=db)

        # plate init
        plate_detector: Optional[SPlateDetector] = None
        ocr_engine: Optional[SOcrEngine] = None
        plate_ok = False

        if enable_plate:
            try:
                plate_detector = SPlateDetector(model_path=plate_model_path, conf_thres=det_conf_plate)
                ocr_engine = SOcrEngine(lang="ko")

                regs_exact = db.get_registered_plate_set()

                plate_ok = True
                print(f"[plate] detector=ON model={plate_model_path} conf={det_conf_plate}")
                print(f"[plate] ocr_backend={ocr_engine.backend} db={db_path} regs={len(regs_exact)}")
                if getattr(ocr_engine, "backend", "none") == "none":
                    print("[WARN] OCR backend is none -> plate matching won't work (all plates will be mosaiced)")
            except Exception as e:
                print(f"[WARN] plate init failed -> disable plate: {e}")
                plate_detector = None
                ocr_engine = None
                regs_exact = set()
                plate_ok = False
                enable_plate = False

        # 얼굴 hold는 "미등록 얼굴 blur 박스" 기준으로 유지
        last_face_blur_boxes: List[BBox] = []
        face_hold_left = 0

        for idx, frame in iter_frames(cap):
            # 번호판은 모자이크 전 원본에서 detect/crop
            frame_raw = frame.copy()

            # -------------------------
            # face
            # -------------------------
            dets = detector.detect(frame)
            face_dets = [d for d in dets if d.get("cls") == "face" and "bbox" in d]

            try:
                tracks = tracker.update(frame, idx, face_dets)
            except TypeError:
                tracks = tracker.update(face_dets)

            track_boxes: List[BBox] = []
            face_blur_boxes: List[BBox] = []
            used_hold = False

            if tracks:
                for t in tracks:
                    bbox = _get_track_val(t, "bbox", None)
                    cls_name = _get_track_val(t, "cls", "face")
                    track_id = _get_track_val(t, "track_id", None)

                    if bbox is None or cls_name != "face":
                        continue
                    if track_id is None:
                        continue

                    x1, y1, x2, y2 = [int(v) for v in bbox]
                    cur_box = (x1, y1, x2, y2)
                    track_boxes.append(cur_box)

                    try:
                        # C-1) bbox 기준 embedding 추출
                        fe = recognizer.extract_from_frame(frame, cur_box)

                        # C-2) temporal embedding
                        rep_emb = temporal.update(
                            track_id=int(track_id),
                            embedding=fe.embedding,
                            frame_idx=int(idx),
                        )

                        # C-3) DB 매칭
                        match = matcher.match_face(rep_emb)

                        if print_every and idx % print_every == 0:
                            print(
                                f"[face-match] frame={idx} track={track_id} "
                                f"registered={match.is_registered} "
                                f"label={match.label} score={match.score:.4f}"
                            )

                        # 정책: 등록된 얼굴은 유지, 미등록만 blur
                        if not match.is_registered:
                            face_blur_boxes.append(cur_box)

                    except Exception as e:
                        # 인식 실패 시 안전하게 blur
                        if print_every and idx % print_every == 0:
                            print(f"[WARN] face recognition failed frame={idx} track={track_id}: {e}")
                        face_blur_boxes.append(cur_box)

                if face_blur_boxes:
                    last_face_blur_boxes = face_blur_boxes[:]
                    face_hold_left = hold_last
                else:
                    last_face_blur_boxes = []
                    face_hold_left = 0

            elif face_hold_left > 0 and last_face_blur_boxes:
                face_blur_boxes = last_face_blur_boxes[:]
                face_hold_left -= 1
                used_hold = True

            if face_blur_boxes:
                frame = apply_mosaic(frame, face_blur_boxes, mode=mode, pixelate_scale=pixelate_scale)

            if draw_boxes and track_boxes:
                for b in track_boxes:
                    _draw_box(frame, b, (255, 0, 255), "face")

            if draw_hold and used_hold:
                for (x1, y1, x2, y2) in last_face_blur_boxes:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # -------------------------
            # plate
            # -------------------------
            plates_mosaic = 0
            plate_boxes_n = 0
            ocr_valid_n = 0

            if enable_plate and plate_ok and plate_detector is not None and ocr_engine is not None:
                H, W = frame_raw.shape[:2]

                if debug_dump_plate_crops and (idx % max(1, debug_dump_every) == 0):
                    dbg = Path(debug_dump_frame_dir)
                    dbg.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(dbg / f"frame_{idx:06d}.jpg"), frame_raw)

                plate_dets = plate_detector.detect(frame_raw)
                plate_boxes = _iter_plate_bboxes(plate_dets)

                # 중복 제거
                plate_boxes = _nms_boxes(plate_boxes, iou_th=0.55)

                plate_boxes_n = len(plate_boxes)
                plate_boxes_sorted = sorted(plate_boxes, key=lambda b: (b[0] + b[2]) / 2.0)

                boxes_to_mosaic: List[BBox] = []
                debug_plate = bool(print_every and idx % print_every == 0)

                # lane별 vote input
                lane_votes_in: Dict[int, List[Tuple[str, float]]] = defaultdict(list)

                # 1) OCR 먼저 전부 수행
                per_plate_info: List[Tuple[int, BBox, str, float, str]] = []
                # (lane, pb, raw, conf, norm)

                for i, pb in enumerate(plate_boxes_sorted):
                    pb2 = _pad_bbox(pb, W, H, pad_ratio=plate_pad_ratio)
                    x1, y1, x2, y2 = pb2

                    x1 = max(0, min(x1, W - 1))
                    x2 = max(0, min(x2, W))
                    y1 = max(0, min(y1, H - 1))
                    y2 = max(0, min(y2, H))
                    if x2 <= x1 or y2 <= y1:
                        continue

                    crop = frame_raw[y1:y2, x1:x2]

                    if debug_dump_plate_crops and (idx % max(1, debug_dump_every) == 0):
                        _safe_imwrite(Path(debug_dump_dir) / f"plate_crop_f{idx:06d}_i{i}.jpg", crop)

                    o = ocr_engine.read_plate(crop)
                    raw, conf = _ocr_unpack(o)
                    norm = normalize_and_fix(raw)

                    lane = _lane_id_from_bbox(pb, W, lanes=plate_lanes)
                    per_plate_info.append((lane, pb, raw, conf, norm))

                    if conf >= ocr_conf_thres and PLATE_RE.match(norm):
                        lane_votes_in[lane].append((norm, conf))
                        ocr_valid_n += 1

                # 2) lane별 vote 업데이트
                lane_best: Dict[int, Tuple[str, float, bool]] = {}
                for lane in range(max(1, plate_lanes)):
                    best_norm, best_score, stable = plate_votes[lane].update(lane_votes_in.get(lane, []))
                    lane_best[lane] = (best_norm, best_score, stable)

                # 3) 모자이크 / 예외 판정
                for i, (lane, pb, raw, conf, norm) in enumerate(per_plate_info):
                    best_norm, best_score, stable = lane_best.get(lane, ("", 0.0, False))

                    best_is_plate = bool(best_norm and PLATE_RE.match(best_norm))
                    is_reg_now = bool(stable and best_is_plate and (best_norm in regs_exact))

                    if is_reg_now:
                        plate_reg_last_ok_frame[lane] = idx

                    is_reg = is_reg_now or ((idx - plate_reg_last_ok_frame[lane]) <= plate_reg_hold_ttl)

                    if debug_plate:
                        ttl_left = plate_reg_hold_ttl - (idx - plate_reg_last_ok_frame[lane])
                        print(
                            f"[plate@{idx}] i={i} lane={lane} raw='{raw}' norm='{norm}' conf={conf:.2f} "
                            f"best='{best_norm}' score={best_score:.2f} stable={stable} "
                            f"reg_exact={is_reg} ttl={ttl_left}"
                        )

                    if not is_reg:
                        boxes_to_mosaic.append(pb)

                    if draw_boxes:
                        shown = best_norm if best_norm else norm
                        lbl = f"plate:{shown}" if is_reg else f"plate*:{shown}"
                        _draw_box(frame, pb, (0, 0, 255), lbl)

                if boxes_to_mosaic:
                    frame = apply_mosaic(frame, boxes_to_mosaic, mode=mode, pixelate_scale=pixelate_scale)
                    plates_mosaic = len(boxes_to_mosaic)

            writer.write(frame)

            if print_every and idx % print_every == 0:
                print(
                    f"[frame {idx}] faces_det={len(face_dets)} "
                    f"faces_track={len(track_boxes)} "
                    f"faces_blur={len(face_blur_boxes)} "
                    f"face_hold={used_hold} "
                    f"plate_boxes={plate_boxes_n} "
                    f"ocr_valid={ocr_valid_n} "
                    f"plates_blur={plates_mosaic}"
                )

        safe_release(cap)
        safe_release(writer)

        _encode_h264(tmp_path, out_path)
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)

        print(f"[process_video] done: {out_path}")
        return str(out_path)

    finally:
        safe_release(cap)
        safe_release(writer)


# ============================================================
# CLI
# ============================================================
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()

    p.add_argument("--in", dest="input_video_path", type=str, default=None)
    p.add_argument("--out", dest="output_video_path", type=str, default=None)
    p.add_argument("--job_id", type=str, default=None)

    p.add_argument("--mode", type=str, default="blur", choices=["blur", "pixelate"])
    p.add_argument("--pixelate_scale", type=float, default=0.12)

    # face
    p.add_argument("--det_conf", dest="det_conf_face", type=float, default=0.30)
    p.add_argument("--predict_conf_face", type=float, default=0.15)
    p.add_argument("--disable_face_filters", action="store_true")
    p.add_argument("--hold_last", type=int, default=30)

    # tracker
    p.add_argument("--track_iou", type=float, default=0.10)
    p.add_argument("--track_dist", type=float, default=0.90)
    p.add_argument("--track_dist_weight", type=float, default=0.20)
    p.add_argument("--track_max_miss", type=int, default=180)
    p.add_argument("--min_hits", type=int, default=3)
    p.add_argument("--ema_alpha", type=float, default=0.85)
    p.add_argument("--sid_ttl", type=int, default=150)
    p.add_argument("--reid_iou_gate", type=float, default=0.10)
    p.add_argument("--reid_dist_gate", type=float, default=1.30)
    p.add_argument("--reid_hist_gate", type=float, default=0.60)
    p.add_argument("--use_hist", action="store_true")
    p.add_argument("--min_age_for_reid", type=int, default=30)
    p.add_argument("--strict_reid_hist_gate", type=float, default=0.70)

    # debug
    p.add_argument("--draw_boxes", action="store_true")
    p.add_argument("--draw_hold", action="store_true")
    p.add_argument("--print_every", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--debug_dump_plate_crops", action="store_true")
    p.add_argument("--debug_dump_every", type=int, default=30)
    p.add_argument("--debug_dump_dir", type=str, default="data/debug")
    p.add_argument("--debug_dump_frame_dir", type=str, default="data/debug_plate")

    # plate
    p.add_argument("--enable_plate", action="store_true")
    p.add_argument("--plate_model_path", type=str, default="models/yolo_plate/yolov8n-plate.pt")
    p.add_argument("--det_conf_plate", type=float, default=0.35)
    p.add_argument("--ocr_conf_thres", type=float, default=0.25)
    p.add_argument("--plate_vote_window", type=int, default=35)
    p.add_argument("--plate_stable_ratio", type=float, default=0.55)
    p.add_argument("--plate_stable_min_hits", type=int, default=6)
    p.add_argument("--plate_lanes", type=int, default=3)
    p.add_argument("--plate_pad_ratio", type=float, default=0.18)
    p.add_argument("--db_path", type=str, default="db/cctv_mosaic.sqlite3")
    p.add_argument("--plate_reg_hold_ttl", type=int, default=60)

    return p


def main():
    parser = build_argparser()
    args = parser.parse_args()

    if args.job_id:
        input_path, output_path = _resolve_job_paths(args.job_id)
    else:
        if not args.input_video_path or not args.output_video_path:
            raise ValueError("Use either --job_id or both --in and --out")
        input_path = Path(args.input_video_path)
        output_path = Path(args.output_video_path)

    process_video(
        input_video_path=str(input_path),
        output_video_path=str(output_path),
        mode=args.mode,
        pixelate_scale=args.pixelate_scale,
        det_conf_face=args.det_conf_face,
        predict_conf_face=args.predict_conf_face,
        disable_face_filters=args.disable_face_filters,
        hold_last=args.hold_last,
        draw_boxes=args.draw_boxes,
        draw_hold=args.draw_hold,
        print_every=args.print_every,
        overwrite=args.overwrite,
        debug_dump_plate_crops=args.debug_dump_plate_crops,
        debug_dump_every=args.debug_dump_every,
        debug_dump_dir=args.debug_dump_dir,
        debug_dump_frame_dir=args.debug_dump_frame_dir,
        enable_plate=args.enable_plate,
        plate_model_path=args.plate_model_path,
        det_conf_plate=args.det_conf_plate,
        ocr_conf_thres=args.ocr_conf_thres,
        plate_vote_window=args.plate_vote_window,
        plate_stable_ratio=args.plate_stable_ratio,
        plate_stable_min_hits=args.plate_stable_min_hits,
        plate_lanes=args.plate_lanes,
        plate_pad_ratio=args.plate_pad_ratio,
        db_path=args.db_path,
        plate_reg_hold_ttl=args.plate_reg_hold_ttl,
        args=args,
    )


if __name__ == "__main__":
    main()