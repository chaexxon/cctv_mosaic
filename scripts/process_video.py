# scripts/process_video.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import cv2

from utils.video import open_capture, get_video_info, iter_frames, make_writer, safe_release
from services.mosaic import apply_mosaic, MosaicMode, BBox
from services.face_detector import FaceDetector
from services.tracker import IoUTracker


def _demo_targets(video_info, box_w_ratio=0.25, box_h_ratio=0.45) -> Dict[int, List[BBox]]:
    w, h = video_info.width, video_info.height
    bw = int(w * box_w_ratio)
    bh = int(h * box_h_ratio)

    cx, cy = w // 2, h // 2
    x1 = cx - bw // 2
    y1 = cy - bh // 2
    x2 = cx + bw // 2
    y2 = cy + bh // 2

    targets: Dict[int, List[BBox]] = {}
    for frame_idx in range(15, 120):
        targets[frame_idx] = [(x1, y1, x2, y2)]
    return targets


def _draw_tracks(frame, tracks, *, det_conf: float, color=(0, 0, 255)):
    """
    Draw tracked boxes and track_id on frame (debug).
    NOTE: Call this AFTER mosaic so the debug overlay remains sharp.
    """
    for t in tracks:
        if "bbox" not in t:
            continue
        x1, y1, x2, y2 = t["bbox"]
        tid = t.get("track_id", -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        cv2.putText(
            frame,
            f"id={tid}",
            (x1, max(0, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )
    cv2.putText(
        frame,
        f"tracks={len(tracks)} conf>={det_conf:.2f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        color,
        2,
        cv2.LINE_AA,
    )
    return frame


def process_video(
    input_video_path: str,
    output_video_path: str,
    *,
    mode: MosaicMode = "blur",
    pixelate_scale: float = 0.12,
    codec: str = "mp4v",
    use_demo_boxes: bool = False,
    det_conf: float = 0.60,
    track_iou: float = 0.30,
    track_max_miss: int = 10,
    draw_boxes: bool = False,
    print_every: int = 0,
) -> str:
    cap = None
    writer = None
    try:
        cap = open_capture(input_video_path)
        info = get_video_info(cap)

        out_path = Path(output_video_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        writer = make_writer(
            output_path=out_path,
            fps=info.fps,
            frame_size=(info.width, info.height),
            codec=codec,
            is_color=True,
        )

        # FaceDetector 내부에서:
        # - YOLO conf prefilter + 후처리(min_area_ratio, aspect ratio) 적용됨
        detector = FaceDetector(conf_thres=det_conf)
        tracker = IoUTracker(iou_thres=track_iou, max_miss=track_max_miss)

        demo_targets = _demo_targets(info) if use_demo_boxes else {}

        for idx, frame in iter_frames(cap):
            boxes: List[BBox] = []

            # 0) demo boxes (선택)
            if use_demo_boxes:
                boxes.extend(demo_targets.get(idx, []))

            # 1) detector -> tracker
            dets = detector.detect(frame)
            face_dets = [d for d in dets if d.get("cls") == "face" and "bbox" in d]
            tracks = tracker.update(face_dets)

            # 2) mosaic targets = tracker boxes (+ demo boxes)
            track_boxes = [t["bbox"] for t in tracks if "bbox" in t]
            boxes.extend(track_boxes)

            # 3) mosaic 먼저 적용 (debug overlay가 안 흐려지게)
            if boxes:
                frame = apply_mosaic(frame, boxes, mode=mode, pixelate_scale=pixelate_scale)

            # 4) debug overlay는 마지막에 (선명하게 보이게)
            if draw_boxes:
                _draw_tracks(frame, tracks, det_conf=det_conf)

            # 5) debug print
            if print_every > 0 and (idx % print_every == 0):
                print(f"[frame {idx}] dets={len(dets)} face_dets={len(face_dets)} tracks={len(tracks)}")

            writer.write(frame)

        return str(out_path)

    finally:
        safe_release(cap=cap, writer=writer)


def main():
    parser = argparse.ArgumentParser(description="B: process video (YOLO face + IoU tracker + mosaic)")
    parser.add_argument("--in", dest="inp", required=True, help="input video path")
    parser.add_argument("--out", dest="out", required=True, help="output video path")
    parser.add_argument("--mode", choices=["blur", "pixelate"], default="blur", help="mosaic mode")
    parser.add_argument("--pixel_scale", type=float, default=0.12, help="pixelate scale (0.02~0.5)")
    parser.add_argument("--codec", type=str, default="mp4v", help="fourcc codec (mp4v, avc1, XVID...)")

    # detector / tracker tuning
    parser.add_argument("--det_conf", type=float, default=0.60, help="YOLO face confidence threshold")
    parser.add_argument("--track_iou", type=float, default=0.30, help="IoU match threshold (tracker)")
    parser.add_argument("--track_max_miss", type=int, default=10, help="max missed frames to keep a track")

    # debug
    parser.add_argument("--use_demo", action="store_true", help="apply demo mosaic boxes (center box)")
    parser.add_argument("--draw_boxes", action="store_true", help="draw tracker boxes + ids (debug)")
    parser.add_argument("--print_every", type=int, default=0, help="print stats every N frames (0=off)")

    args = parser.parse_args()

    out_path = process_video(
        input_video_path=args.inp,
        output_video_path=args.out,
        mode=args.mode,
        pixelate_scale=args.pixel_scale,
        codec=args.codec,
        use_demo_boxes=args.use_demo,
        det_conf=args.det_conf,
        track_iou=args.track_iou,
        track_max_miss=args.track_max_miss,
        draw_boxes=args.draw_boxes,
        print_every=args.print_every,
    )
    print(f"Saved output: {out_path}")


if __name__ == "__main__":
    main()
