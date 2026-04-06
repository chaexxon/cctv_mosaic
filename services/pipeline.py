# services/pipeline.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from scripts.process_video import process_video


def _resolve_db_path(db: Optional[Union[str, Path, object]]) -> str:
    """
    db 인자를 다양한 형태로 받아 db_path 문자열로 변환한다.

    지원:
    - None -> 기본 DB 경로 사용
    - str / Path -> 해당 경로 사용
    - SQLiteDB 객체처럼 db_path 속성이 있는 객체 -> 그 경로 사용
    """
    if db is None:
        return "db/cctv_mosaic.sqlite3"

    if isinstance(db, (str, Path)):
        return str(db)

    if hasattr(db, "db_path"):
        return str(getattr(db, "db_path"))

    return "db/cctv_mosaic.sqlite3"


def run(
    video_path: str,
    db=None,
    *,
    output_path: Optional[str] = None,
    enable_plate: bool = True,
    mode: str = "blur",
    overwrite: bool = True,
    print_every: int = 30,
) -> str:
    """
    전체 파이프라인 진입점.

    역할:
    - 입력 영상 경로 확인
    - 출력 경로 자동 생성
    - DB 경로 정리
    - scripts/process_video.py의 process_video() 호출

    정책:
    - 얼굴: 등록 인물은 유지, 미등록 인물은 모자이크
    - 번호판: 기존 B 로직 유지
    """
    print("[pipeline] start")

    in_path = Path(video_path)
    if not in_path.exists():
        raise FileNotFoundError(f"[pipeline] input video not found: {in_path}")

    # output_path가 없으면 자동 생성
    if output_path is None:
        out_dir = Path("data/output")
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(out_dir / f"{in_path.stem}_out.mp4")

    db_path = _resolve_db_path(db)

    print(f"[pipeline] input   : {in_path}")
    print(f"[pipeline] output  : {output_path}")
    print(f"[pipeline] db_path : {db_path}")
    print(f"[pipeline] plate   : {enable_plate}")
    print(f"[pipeline] mode    : {mode}")

    result_path = process_video(
        input_video_path=str(in_path),
        output_video_path=str(output_path),
        mode=mode,
        enable_plate=enable_plate,
        overwrite=overwrite,
        print_every=print_every,
        db_path=db_path,
    )

    print(f"[pipeline] done -> {result_path}")
    return result_path