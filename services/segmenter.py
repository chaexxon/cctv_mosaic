# services/segmenter.py
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import cv2


# ============================================================
# Utils
# ============================================================
def _has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def _ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def _resolve_paths(job_id: str, raw_path: Optional[str]):
    jid = str(job_id)
    raw = Path(raw_path) if raw_path else Path("data") / "input" / "raw" / "raw_0.mp4"
    clip = Path("data") / "input" / "segments" / f"clip_{jid}.mp4"
    return raw, clip


def _run_ffmpeg(cmd: list[str]) -> None:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(
            "[ffmpeg failed]\n"
            + "CMD:\n  " + " ".join(cmd) + "\n\n"
            + "STDERR:\n" + p.stderr
        )


def _to_seconds(ts: str) -> float:
    """
    Accept:
      - "HH:MM:SS"
      - "HH:MM:SS.xxx"
      - "MM:SS"
      - "SS"
      - "12.34"
    """
    s = (ts or "").strip()
    if not s:
        raise ValueError("empty timestamp")

    # raw float seconds
    if ":" not in s:
        return float(s)

    parts = s.split(":")
    if len(parts) == 2:  # MM:SS
        mm = int(parts[0])
        ss = float(parts[1])
        return mm * 60.0 + ss
    if len(parts) == 3:  # HH:MM:SS(.xxx)
        hh = int(parts[0])
        mm = int(parts[1])
        ss = float(parts[2])
        return hh * 3600.0 + mm * 60.0 + ss

    raise ValueError(f"invalid timestamp: {ts}")


def _fmt_hhmmss(sec: float) -> str:
    # ffmpeg likes HH:MM:SS.mmm
    if sec < 0:
        sec = 0.0
    hh = int(sec // 3600)
    mm = int((sec % 3600) // 60)
    ss = sec % 60.0
    return f"{hh:02d}:{mm:02d}:{ss:06.3f}"


def _get_video_fps_and_frames(video_path: Path) -> tuple[float, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
    if fps <= 1e-6:
        fps = 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    cap.release()
    return fps, n


def _opencv_fallback_cut(
    raw_path: Path,
    clip_path: Path,
    start_sec: float,
    end_sec: float,
    *,
    codec: str = "mp4v",
) -> str:
    """
    OpenCV fallback:
      - 정확도는 ffmpeg보다 떨어질 수 있음(키프레임/디코딩 이슈)
      - 그래도 PATH에 ffmpeg가 없거나 ffmpeg 에러 시 최소 기능 보장
    """
    cap = cv2.VideoCapture(str(raw_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open raw video: {raw_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
    if fps <= 1e-6:
        fps = 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    _ensure_parent(clip_path)
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(clip_path), fourcc, fps, (w, h), True)
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open writer: {clip_path}")

    start_frame = max(0, int(start_sec * fps))
    end_frame = max(start_frame, int(end_sec * fps))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    idx = start_frame
    while idx < end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        idx += 1

    writer.release()
    cap.release()
    return str(clip_path)


# ============================================================
# Core
# ============================================================
def segment_clip(
    *,
    job_id: str,
    raw_video: str,
    start: str,
    end: str,
    prefer_copy: bool = False,
    force_opencv: bool = False,
) -> str:
    """
    raw_video 에서 [start, end] 구간을 잘라
    data/input/segments/clip_{job_id}.mp4 생성

    기본은 ffmpeg 사용(안정/호환성 우선).
    - 인코딩: libx264 + yuv420p + faststart
    - (옵션) prefer_copy=True면 stream copy 시도(빠름). 실패하면 재인코딩으로 자동 fallback.
    - force_opencv=True면 OpenCV로만 자름(비추, 최후 수단)
    """
    raw_path, clip_path = _resolve_paths(job_id, raw_video)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw video not found: {raw_path}")
    _ensure_parent(clip_path)

    start_sec = _to_seconds(start)
    end_sec = _to_seconds(end)

    # swap/guard
    if end_sec < start_sec:
        start_sec, end_sec = end_sec, start_sec

    # clamp end to video duration (best-effort)
    try:
        fps, nframes = _get_video_fps_and_frames(raw_path)
        if nframes > 0 and fps > 0:
            dur = nframes / fps
            if start_sec > dur:
                start_sec = max(0.0, dur - 0.1)
            if end_sec > dur:
                end_sec = dur
    except Exception:
        pass

    if (end_sec - start_sec) <= 0.01:
        raise ValueError(f"too short range: start={start} end={end}")

    if force_opencv:
        print("[segmenter] OpenCV fallback forced")
        return _opencv_fallback_cut(raw_path, clip_path, start_sec, end_sec)

    if not _has_ffmpeg():
        print("[segmenter] ffmpeg not found -> OpenCV fallback")
        return _opencv_fallback_cut(raw_path, clip_path, start_sec, end_sec)

    ss = _fmt_hhmmss(start_sec)
    to = _fmt_hhmmss(end_sec)

    # 1) optional fast path: stream copy (can be inaccurate around keyframes)
    if prefer_copy:
        cmd_copy = [
            "ffmpeg",
            "-y",
            "-ss", ss,
            "-to", to,
            "-i", str(raw_path),
            "-map", "0:v:0",
            "-c:v", "copy",
            "-movflags", "+faststart",
            str(clip_path),
        ]
        try:
            print("[segmenter] ffmpeg copy clip start")
            _run_ffmpeg(cmd_copy)
            print(f"[segmenter] saved (copy): {clip_path}")
            return str(clip_path)
        except Exception as e:
            print(f"[segmenter] copy failed -> re-encode fallback: {e}")

    # 2) stable path: re-encode
    cmd = [
        "ffmpeg",
        "-y",
        "-ss", ss,
        "-to", to,
        "-i", str(raw_path),
        "-map", "0:v:0",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "veryfast",
        "-crf", "23",
        "-movflags", "+faststart",
        str(clip_path),
    ]
    print("[segmenter] ffmpeg re-encode clip start")
    _run_ffmpeg(cmd)
    print(f"[segmenter] saved: {clip_path}")
    return str(clip_path)


# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="B: segment raw video by time range (ffmpeg + OpenCV fallback)")

    parser.add_argument("--job_id", required=True, type=str)
    parser.add_argument("--raw", required=True, help="raw video path (ex: data/input/raw/raw_0.mp4)")
    parser.add_argument("--start", required=True, help="start time (HH:MM:SS[.ms] or seconds)")
    parser.add_argument("--end", required=True, help="end time (HH:MM:SS[.ms] or seconds)")

    parser.add_argument("--prefer_copy", action="store_true", help="try ffmpeg stream copy first (fast, may be less accurate)")
    parser.add_argument("--force_opencv", action="store_true", help="force OpenCV fallback (last resort)")

    args = parser.parse_args()

    out = segment_clip(
        job_id=args.job_id,
        raw_video=args.raw,
        start=args.start,
        end=args.end,
        prefer_copy=args.prefer_copy,
        force_opencv=args.force_opencv,
    )
    print(f"OK: {out}")


if __name__ == "__main__":
    main()
