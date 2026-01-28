# scripts/process_video.py
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Any, List

import cv2

from utils.video import open_capture, get_video_info, iter_frames, make_writer, safe_release
from services.mosaic import apply_mosaic, MosaicMode, BBox
from services.face_detector import FaceDetector


def _load_tracker_class():
    import services.tracker as tm
    for name in ["IoUTracker", "StableTracker", "Tracker"]:
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

    # RawIoUTracker keys
    if "iou_thres" in allowed:
        kw["iou_thres"] = args.iou_thres
    if "dist_thres" in allowed:
        kw["dist_thres"] = args.dist_thres
    if "dist_weight" in allowed:
        kw["dist_weight"] = args.dist_weight
    if "max_miss" in allowed:
        kw["max_miss"] = args.max_miss
    if "min_hits" in allowed:
        kw["min_hits"] = args.min_hits
    if "ema_alpha" in allowed:
        kw["ema_alpha"] = args.ema_alpha

    # UnifiedTracker mapping (optional)
    if "track_iou" in allowed:
        kw["track_iou"] = args.iou_thres
    if "track_dist" in allowed:
        kw["track_dist"] = args.dist_thres
    if "track_dist_weight" in allowed:
        kw["track_dist_weight"] = args.dist_weight
    if "track_max_miss" in allowed:
        kw["track_max_miss"] = args.max_miss

    return TrackerCls(**kw)


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


def process_video(
    input_video_path: str,
    output_video_path: str,
    *,
    mode: MosaicMode = "blur",
    pixelate_scale: float = 0.12,
    det_conf: float = 0.55,
    hold_last: int = 30,
    draw_boxes: bool = False,
    draw_hold: bool = False,
    print_every: int = 0,
    args=None,
) -> str:
    cap = None
    writer = None
    tmp_path = Path(output_video_path).with_suffix(".tmp.mp4")
    out_path = Path(output_video_path)

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

        detector = FaceDetector(conf_thres=det_conf)
        tracker = _build_tracker(args)

        last_boxes: List[BBox] = []
        hold_left = 0

        for idx, frame in iter_frames(cap):
            dets = detector.detect(frame)
            face_dets = [d for d in dets if d.get("cls") == "face" and "bbox" in d]

            # tracker API: update(dets) -> list[dict]
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
                        x1, y1, x2, y2 = t["bbox"]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)

            if draw_hold and used_hold:
                for (x1, y1, x2, y2) in last_boxes:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            if print_every and idx % print_every == 0:
                print(f"[frame {idx}] dets={len(dets)} tracks={len(tracks)} hold={hold_left}")

            writer.write(frame)

        writer.release()
        _encode_h264(tmp_path, out_path)
        tmp_path.unlink(missing_ok=True)

        return str(out_path)

    finally:
        safe_release(cap=cap, writer=writer)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--job_id", type=str, default=None)
    parser.add_argument("--input", dest="inp", default=None)
    parser.add_argument("--output", dest="out", default=None)

    parser.add_argument("--draw_boxes", action="store_true")
    parser.add_argument("--draw_hold", action="store_true")
    parser.add_argument("--print_every", type=int, default=0)

    # ✅ tracker tuning (RawIoUTracker 기본)
    parser.add_argument("--iou_thres", type=float, default=0.10)
    parser.add_argument("--dist_thres", type=float, default=0.90)
    parser.add_argument("--dist_weight", type=float, default=0.20)
    parser.add_argument("--max_miss", type=int, default=180)
    parser.add_argument("--min_hits", type=int, default=3)
    parser.add_argument("--ema_alpha", type=float, default=0.85)

    args = parser.parse_args()

    if args.job_id:
        inp, out = _resolve_job_paths(args.job_id)
    else:
        if not args.inp or not args.out:
            raise SystemExit("ERROR: provide --job_id OR (--input and --output)")
        inp, out = Path(args.inp), Path(args.out)

    res = process_video(
        str(inp),
        str(out),
        draw_boxes=args.draw_boxes,
        draw_hold=args.draw_hold,
        print_every=args.print_every,
        args=args,
    )
    print("Saved:", res)


if __name__ == "__main__":
    main()
