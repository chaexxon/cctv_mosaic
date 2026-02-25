# scripts/process_video.py
from __future__ import annotations

import argparse
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Iterable

import cv2
import numpy as np

from utils.video import open_capture, get_video_info, iter_frames, make_writer, safe_release
from services.mosaic import apply_mosaic, MosaicMode, BBox
from services.face_detector import FaceDetector

# ✅ 단일 진실(services / utils)
from services.plate_detector import PlateDetector as SPlateDetector
from services.ocr_engine import OCREngine as SOcrEngine
from utils.plate_store import load_store
from utils.plate_text import PLATE_RE, normalize_and_fix
from utils.plate_vote import PlateVote


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
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    if label:
        cv2.putText(frame, label, (x1, max(0, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def _pad_bbox(pb: BBox, w: int, h: int, pad_ratio: float = 0.18) -> BBox:
    x1, y1, x2, y2 = pb
    bw = x2 - x1
    bh = y2 - y1
    if bw <= 0 or bh <= 0:
        return (x1, y1, x2, y2)
    px = int(bw * pad_ratio)
    py = int(bh * pad_ratio)
    nx1 = max(0, x1 - px)
    ny1 = max(0, y1 - py)
    nx2 = min(w, x2 + px)
    ny2 = min(h, y2 + py)
    return (nx1, ny1, nx2, ny2)


def _lane_id_from_bbox(pb: BBox, frame_w: int, lanes: int = 3) -> int:
    x1, _, x2, _ = pb
    cx = (x1 + x2) / 2.0
    lane = int(cx / (frame_w / max(1, lanes)))
    return max(0, min(lanes - 1, lane))


def _load_registered_exact_set(store_path: str) -> set[str]:
    try:
        store = load_store(Path(store_path))
    except Exception:
        return set()

    regs: set[str] = set()
    for item in store.get("plates", []):
        raw = item.get("plate_norm")
        if not raw:
            continue
        norm = normalize_and_fix(str(raw))
        if PLATE_RE.match(norm):
            regs.add(norm)
    return regs


def _iter_plate_bboxes(plate_dets: Iterable[Any]) -> List[BBox]:
    out: List[BBox] = []
    for d in plate_dets or []:
        bb = d.get("bbox") if isinstance(d, dict) else d
        if bb is None:
            continue
        if isinstance(bb, (list, tuple)) and len(bb) == 4:
            try:
                x1, y1, x2, y2 = [int(v) for v in bb]
                out.append((x1, y1, x2, y2))
            except Exception:
                continue
    return out


def _ocr_unpack(o: Any) -> Tuple[str, float]:
    if o is None:
        return "", 0.0

    if hasattr(o, "text") or hasattr(o, "conf"):
        try:
            text = (getattr(o, "text", "") or "").strip()
        except Exception:
            text = ""
        try:
            conf = float(getattr(o, "conf", 0.0) or 0.0)
        except Exception:
            conf = 0.0
        return text, conf

    if isinstance(o, dict):
        text = (o.get("text") or "").strip()
        try:
            conf = float(o.get("conf") or 0.0)
        except Exception:
            conf = 0.0
        return text, conf

    if isinstance(o, (list, tuple)) and len(o) >= 2:
        text = str(o[0] or "").strip()
        try:
            conf = float(o[1] or 0.0)
        except Exception:
            conf = 0.0
        return text, conf

    return "", 0.0


def _safe_imwrite(path: Path, img_bgr: np.ndarray) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(str(path), img_bgr)
        return bool(ok)
    except Exception:
        return False


# ---------- NMS / merge to prevent multi-mosaic ----------
def _iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a_area = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    b_area = max(0, bx2 - bx1) * max(0, by2 - by1)
    return inter / max(1.0, (a_area + b_area - inter))


def _nms_boxes(boxes: List[BBox], iou_th: float = 0.55) -> List[BBox]:
    boxes = sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    kept: List[BBox] = []
    for b in boxes:
        if all(_iou(b, k) < iou_th for k in kept):
            kept.append(b)
    return kept


def _merge_overlaps(boxes: List[BBox], iou_th: float = 0.25) -> List[BBox]:
    merged: List[BBox] = []
    for b in boxes:
        placed = False
        for j, m in enumerate(merged):
            if _iou(b, m) >= iou_th:
                x1 = min(b[0], m[0])
                y1 = min(b[1], m[1])
                x2 = max(b[2], m[2])
                y2 = max(b[3], m[3])
                merged[j] = (x1, y1, x2, y2)
                placed = True
                break
        if not placed:
            merged.append(b)
    return merged


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
    debug_dump_plate_crops: bool = True,
    debug_dump_every: int = 30,
    debug_dump_dir: str = "data/debug",
    debug_dump_frame_dir: str = "data/debug_plate",
    # plate
    enable_plate: bool = False,
    plate_model_path: str = "models/yolo_plate/yolov8n-plate.pt",
    det_conf_plate: float = 0.35,
    ocr_conf_thres: float = 0.25,
    plate_store_path: str = "data/plates/plates.json",
    plate_vote_window: int = 35,
    plate_stable_ratio: float = 0.55,
    plate_stable_min_hits: int = 6,
    plate_lanes: int = 3,
    plate_pad_ratio: float = 0.18,
    # stabilizer
    plate_allow_hold_frames: int = 30,  # 등록판으로 판정되면 N프레임 유지
    args=None,
) -> str:
    cap = None
    writer = None

    out_path = Path(output_video_path)
    tmp_path = out_path.with_suffix(".tmp.mp4")

    if out_path.exists() and not overwrite:
        print(f"[process_video] output exists, skip (use --overwrite to force): {out_path}")
        return str(out_path)

    # ✅ vote는 (lane) 단위로만 (가장 안정적)
    plate_votes: Dict[int, PlateVote] = defaultdict(
        lambda: PlateVote(
            window=plate_vote_window,
            stable_ratio=plate_stable_ratio,
            stable_min_hits=plate_stable_min_hits,
        )
    )

    # ✅ 등록판 hold: lane별로 “최근에 등록판이었던” 유지
    plate_allow_ttl: Dict[int, int] = defaultdict(int)

    plate_reg_hold: Dict[int, int] = defaultdict(lambda: -10**9)
    plate_reg_hold_ttl = 120

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

        detector = FaceDetector(
            conf_thres=det_conf_face,
            predict_conf=predict_conf_face,
            disable_post_filters=disable_face_filters,
        )
        tracker = _build_tracker(args)

        # plate init
        plate_detector: Optional[SPlateDetector] = None
        ocr_engine: Optional[SOcrEngine] = None
        regs_exact: set[str] = set()
        plate_ok = False

        if enable_plate:
            try:
                plate_detector = SPlateDetector(model_path=plate_model_path, conf_thres=det_conf_plate)
                ocr_engine = SOcrEngine(lang="ko")
                regs_exact = _load_registered_exact_set(plate_store_path)
                plate_ok = True

                print(f"[plate] detector=ON model={plate_model_path} conf={det_conf_plate}")
                print(f"[plate] ocr_backend={ocr_engine.backend} store={plate_store_path} regs={len(regs_exact)}")
                if getattr(ocr_engine, "backend", "none") == "none":
                    print("[WARN] OCR backend is none -> plate matching won't work (all plates will be mosaiced)")
            except Exception as e:
                print(f"[WARN] plate init failed -> disable plate: {e}")
                plate_detector = None
                ocr_engine = None
                regs_exact = set()
                plate_ok = False
                enable_plate = False

        last_boxes: List[BBox] = []
        hold_left = 0

        for idx, frame in iter_frames(cap):
            # keep raw for plate detect/crop (avoid “mosaic first” side effects)
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

            track_boxes = [t["bbox"] for t in tracks if "bbox" in t]

            use_boxes: List[BBox] = []
            used_hold = False

            if track_boxes:
                use_boxes = track_boxes
                last_boxes = track_boxes
                hold_left = hold_last
            elif hold_left > 0 and last_boxes:
                use_boxes = last_boxes
                hold_left -= 1
                used_hold = True

            if use_boxes:
                frame = apply_mosaic(frame, use_boxes, mode=mode, pixelate_scale=pixelate_scale)

            if draw_boxes and track_boxes:
                for t in tracks:
                    if "bbox" in t:
                        _draw_box(frame, t["bbox"], (255, 0, 255), "face")

            if draw_hold and used_hold:
                for (x1, y1, x2, y2) in last_boxes:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # -------------------------
            # plate
            # -------------------------
            plates_mosaic = 0
            plate_boxes_n = 0
            ocr_valid_n = 0

            if enable_plate and plate_ok and plate_detector is not None and ocr_engine is not None:
                H, W = frame_raw.shape[:2]

                # frame dump (debug)
                if print_every and idx % 30 == 0:
                    dbg = Path(debug_dump_frame_dir)
                    dbg.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(dbg / f"frame_{idx:06d}.jpg"), frame_raw)

                plate_dets = plate_detector.detect(frame_raw)
                plate_boxes = _iter_plate_bboxes(plate_dets)

                # ✅ 중복 plate box 제거
                plate_boxes = _nms_boxes(plate_boxes, iou_th=0.55)

                plate_boxes_n = len(plate_boxes)
                plate_boxes_sorted = sorted(plate_boxes, key=lambda b: (b[0] + b[2]) / 2.0)

                boxes_to_mosaic: List[BBox] = []
                debug_plate = bool(print_every and idx % print_every == 0)

                # lane별로 이번 프레임에 관측된 텍스트(투표 입력) 모으기
                lane_votes_in: Dict[int, List[Tuple[str, float]]] = defaultdict(list)

                # 1) OCR 먼저 다 돌림 (원본 crop에서)
                per_plate_info: List[Tuple[int, BBox, str, float, str]] = []
                # tuple: (lane, pb, raw, conf, norm)

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

                    # dump crops for debugging
                    if debug_dump_plate_crops and (idx % max(1, debug_dump_every) == 0):
                        _safe_imwrite(Path(debug_dump_dir) / f"plate_crop_f{idx:06d}_i{i}.jpg", crop)

                    o = ocr_engine.read_plate(crop)
                    raw, conf = _ocr_unpack(o)
                    norm = normalize_and_fix(raw)

                    lane = _lane_id_from_bbox(pb, W, lanes=plate_lanes)
                    per_plate_info.append((lane, pb, raw, conf, norm))

                    # vote input 조건
                    if conf >= ocr_conf_thres and PLATE_RE.match(norm):
                        lane_votes_in[lane].append((norm, conf))
                        ocr_valid_n += 1

                # 2) lane별 vote update 후 best_norm 확정
                lane_best: Dict[int, Tuple[str, float, bool]] = {}
                for lane in range(max(1, plate_lanes)):
                    best_norm, best_score, stable = plate_votes[lane].update(lane_votes_in.get(lane, []))
                    lane_best[lane] = (best_norm, best_score, stable)

                # 3) 모자이크/예외 판정 + draw
                for i, (lane, pb, raw, conf, norm) in enumerate(per_plate_info):
                    best_norm, best_score, stable = lane_best.get(lane, ("", 0.0, False))

                    # ✅ 예외판정은 best_norm 기반 + stable + regs_exact
                    best_is_plate = bool(best_norm and PLATE_RE.match(best_norm))
                    is_reg = bool(stable and best_is_plate and (best_norm in regs_exact))

                    if is_reg:
                        plate_reg_hold[lane] = idx
                    else:
                        if (idx - plate_reg_hold[lane]) <= plate_reg_hold_ttl:
                            is_reg = True

                    if debug_plate:
                        print(
                            f"[plate@{idx}] i={i} lane={lane} raw='{raw}' norm='{norm}' conf={conf:.2f} "
                            f"best='{best_norm}' score={best_score:.2f} stable={stable} reg_exact={is_reg} "
                            f"ttl={idx - plate_reg_hold[lane]}"
                        )

                    if not is_reg:
                        boxes_to_mosaic.append(pb)

                    if draw_boxes:
                        shown = best_norm if best_norm else norm
                        lbl = f"plate:{shown}" if is_reg else f"plate*:{shown}"
                        _draw_box(frame, pb, (0, 0, 255), lbl)

                # ✅ 모자이크는 “한 번만”
                if boxes_to_mosaic:
                    boxes_to_mosaic = _merge_overlaps(boxes_to_mosaic, iou_th=0.25)
                    plates_mosaic = len(boxes_to_mosaic)
                    frame = apply_mosaic(frame, boxes_to_mosaic, mode=mode, pixelate_scale=pixelate_scale)

            if print_every and idx % print_every == 0:
                print(
                    f"[frame {idx}] face_dets={len(face_dets)} tracks={len(tracks)} hold={hold_left} "
                    f"plate_boxes={plate_boxes_n} ocr_valid={ocr_valid_n} plates_mosaic={plates_mosaic} "
                    f"ocr_backend={(ocr_engine.backend if ocr_engine else 'none')}"
                )

            writer.write(frame)

        writer.release()
        _encode_h264(tmp_path, out_path)
        tmp_path.unlink(missing_ok=True)
        return str(out_path)

    finally:
        safe_release(cap=cap, writer=writer)


# ============================================================
# CLI
# ============================================================
def main():
    p = argparse.ArgumentParser()

    # input/output
    p.add_argument("--job_id", type=str, default=None)
    p.add_argument("--in", dest="inp", default=None)
    p.add_argument("--out", dest="out", default=None)
    p.add_argument("--overwrite", action="store_true", help="overwrite output if exists")

    # debug
    p.add_argument("--draw_boxes", action="store_true")
    p.add_argument("--draw_hold", action="store_true")
    p.add_argument("--print_every", type=int, default=0)

    # optional plate crop dump
    p.add_argument("--debug_dump_plate_crops", action="store_true", help="dump plate crops to data/debug")
    p.add_argument("--debug_dump_every", type=int, default=30, help="dump every N frames")
    p.add_argument("--debug_dump_dir", type=str, default="data/debug")

    # face
    p.add_argument("--det_conf_face", type=float, default=0.30)
    p.add_argument("--predict_conf_face", type=float, default=0.15)
    p.add_argument("--disable_face_filters", action="store_true")
    p.add_argument("--hold_last", type=int, default=30)

    # plate
    p.add_argument("--enable_plate", action="store_true")
    p.add_argument("--plate_model_path", type=str, default="models/yolo_plate/yolov8n-plate.pt")
    p.add_argument("--det_conf_plate", type=float, default=0.35)
    p.add_argument("--ocr_conf_thres", type=float, default=0.25)
    p.add_argument("--plate_store_path", type=str, default="data/plates/plates.json")

    p.add_argument("--plate_vote_window", type=int, default=35)
    p.add_argument("--plate_stable_ratio", type=float, default=0.55)
    p.add_argument("--plate_stable_min_hits", type=int, default=6)

    p.add_argument("--plate_lanes", type=int, default=3)
    p.add_argument("--plate_pad_ratio", type=float, default=0.18)

    # stabilizer
    p.add_argument("--plate_allow_hold_frames", type=int, default=30)

    # tracker tuning
    p.add_argument("--iou_thres", type=float, default=0.10)
    p.add_argument("--dist_thres", type=float, default=0.90)
    p.add_argument("--dist_weight", type=float, default=0.20)
    p.add_argument("--max_miss", type=int, default=180)
    p.add_argument("--min_hits", type=int, default=3)
    p.add_argument("--ema_alpha", type=float, default=0.85)

    args = p.parse_args()

    if args.job_id:
        inp, out = _resolve_job_paths(args.job_id)
    else:
        if not args.inp or not args.out:
            raise SystemExit("Provide --job_id or both --in and --out")
        inp, out = Path(args.inp), Path(args.out)

    # debug flag default: ON if print_every enabled or user requests
    debug_dump = bool(args.debug_dump_plate_crops or (args.print_every > 0))

    res = process_video(
        str(inp),
        str(out),
        draw_boxes=args.draw_boxes,
        draw_hold=args.draw_hold,
        print_every=args.print_every,
        overwrite=args.overwrite,
        # face
        det_conf_face=args.det_conf_face,
        predict_conf_face=args.predict_conf_face,
        disable_face_filters=args.disable_face_filters,
        hold_last=args.hold_last,
        # plate
        enable_plate=args.enable_plate,
        plate_model_path=args.plate_model_path,
        det_conf_plate=args.det_conf_plate,
        ocr_conf_thres=args.ocr_conf_thres,
        plate_store_path=args.plate_store_path,
        plate_vote_window=args.plate_vote_window,
        plate_stable_ratio=args.plate_stable_ratio,
        plate_stable_min_hits=args.plate_stable_min_hits,
        plate_lanes=args.plate_lanes,
        plate_pad_ratio=args.plate_pad_ratio,
        plate_allow_hold_frames=args.plate_allow_hold_frames,
        # debug dump
        debug_dump_plate_crops=debug_dump,
        debug_dump_every=args.debug_dump_every,
        debug_dump_dir=args.debug_dump_dir,
        args=args,
    )
    print("Saved:", res)


if __name__ == "__main__":
    main()
