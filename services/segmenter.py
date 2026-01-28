# services/segmenter.py
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


# =========================
# Utils
# =========================
def _has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def _ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def _resolve_paths(job_id: str, raw_path: str | None):
    jid = str(job_id)

    raw = Path(raw_path) if raw_path else Path("data") / "input" / "raw" / "raw_0.mp4"
    clip = Path("data") / "input" / "segments" / f"clip_{jid}.mp4"

    return raw, clip


def _run_ffmpeg(cmd: list[str]) -> None:
    p = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if p.returncode != 0:
        raise RuntimeError(
            "[ffmpeg failed]\n"
            + "CMD:\n  " + " ".join(cmd) + "\n\n"
            + "STDERR:\n" + p.stderr
        )


# =========================
# Core
# =========================
def segment_clip(
    *,
    job_id: str,
    raw_video: str,
    start: str,
    end: str,
) -> str:
    """
    raw_video 에서 [start, end] 구간을 잘라
    data/input/segments/clip_{job_id}.mp4 생성

    - codec : h264
    - pix_fmt : yuv420p (OpenCV 호환)
    """

    if not _has_ffmpeg():
        raise RuntimeError("ffmpeg not found in PATH")

    raw_path, clip_path = _resolve_paths(job_id, raw_video)

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw video not found: {raw_path}")

    _ensure_parent(clip_path)

    # 🔒 안정성 최우선 ffmpeg 옵션
    cmd = [
        "ffmpeg",
        "-y",
        "-ss", start,
        "-to", end,
        "-i", str(raw_path),
        "-map", "0:v:0",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "veryfast",
        "-crf", "23",
        "-movflags", "+faststart",
        str(clip_path),
    ]

    print("[segmenter] ffmpeg clip start")
    _run_ffmpeg(cmd)
    print(f"[segmenter] saved: {clip_path}")

    return str(clip_path)


# =========================
# CLI
# =========================
def main():
    parser = argparse.ArgumentParser(description="B: segment raw video by time range (ffmpeg)")

    parser.add_argument("--job_id", required=True, type=str)
    parser.add_argument("--raw", required=True, help="raw video path (ex: data/input/raw/raw_0.mp4)")
    parser.add_argument("--start", required=True, help="start time (HH:MM:SS)")
    parser.add_argument("--end", required=True, help="end time (HH:MM:SS)")

    args = parser.parse_args()

    out = segment_clip(
        job_id=args.job_id,
        raw_video=args.raw,
        start=args.start,
        end=args.end,
    )

    print(f"OK: {out}")


if __name__ == "__main__":
    main()
