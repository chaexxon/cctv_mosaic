# scripts/process_video.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import cv2

from utils.video import open_capture, get_video_info, iter_frames, make_writer, safe_release
from services.mosaic import apply_mosaic, MosaicMode, BBox


def _demo_targets(video_info, box_w_ratio=0.25, box_h_ratio=0.45) -> Dict[int, List[BBox]]:
    """
    Create dummy mosaic targets for demo:
    - Apply a big mosaic box for MANY consecutive frames
    """
    w, h = video_info.width, video_info.height
    bw = int(w * box_w_ratio)
    bh = int(h * box_h_ratio)

    # 🔹 중앙 (원하면 왼쪽으로 바꿀 수 있음)
    cx, cy = w // 2, h // 2
    # cx, cy = w // 4, h // 2   # ← 왼쪽에 걸리게 하고 싶으면 이 줄로

    x1 = cx - bw // 2
    y1 = cy - bh // 2
    x2 = cx + bw // 2
    y2 = cy + bh // 2

    targets: Dict[int, List[BBox]] = {}

    # 🔥 핵심: 연속 프레임에 모자이크 적용
    # 대략 0.5초 ~ 4초 구간
    for frame_idx in range(15, 120):
        targets[frame_idx] = [(x1, y1, x2, y2)]

    return targets


def process_video(
    input_video_path: str,
    output_video_path: str,
    mosaic_targets: Dict[int, List[BBox]],
    *,
    mode: MosaicMode = "blur",
    pixelate_scale: float = 0.12,
    codec: str = "mp4v",
) -> str:
    """
    Minimal B-side processor:
    - Read input video
    - For frames listed in mosaic_targets, apply mosaic to given bboxes
    - Write output video

    Args:
        input_video_path: input mp4 path
        output_video_path: output mp4 path
        mosaic_targets: {frame_idx: [(x1,y1,x2,y2), ...], ...}
        mode: "blur" or "pixelate"
        pixelate_scale: used only for pixelate mode
        codec: output fourcc (default mp4v)

    Returns:
        output_video_path
    """
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

        for idx, frame in iter_frames(cap):
            boxes = mosaic_targets.get(idx, [])
            if boxes:
                frame = apply_mosaic(frame, boxes, mode=mode, pixelate_scale=pixelate_scale)
            writer.write(frame)

        return str(out_path)

    finally:
        safe_release(cap=cap, writer=writer)


def main():
    parser = argparse.ArgumentParser(description="B: process video with dummy mosaic targets")
    parser.add_argument("--in", dest="inp", required=True, help="input video path")
    parser.add_argument("--out", dest="out", required=True, help="output video path")
    parser.add_argument("--mode", choices=["blur", "pixelate"], default="blur", help="mosaic mode")
    parser.add_argument("--pixel_scale", type=float, default=0.12, help="pixelate scale (0.02~0.5)")
    parser.add_argument("--codec", type=str, default="mp4v", help="fourcc codec (mp4v, avc1, XVID...)")
    parser.add_argument("--demo", action="store_true", help="use demo targets (default)")
    args = parser.parse_args()

    cap = open_capture(args.inp)
    info = get_video_info(cap)
    cap.release()

    # For now: demo targets only
    mosaic_targets = _demo_targets(info)

    out_path = process_video(
        input_video_path=args.inp,
        output_video_path=args.out,
        mosaic_targets=mosaic_targets,
        mode=args.mode,
        pixelate_scale=args.pixel_scale,
        codec=args.codec,
    )
    print(f"Saved output: {out_path}")


if __name__ == "__main__":
    main()
