"""
scripts package

이 패키지는 cctv_mosaic 프로젝트의 실행용 스크립트 모음이다.

역할:
- CLI 기반 유틸 스크립트 제공
  (init_db, register_face, register_plate, process_video 등)
- `python -m scripts.xxx` 형태의 실행을 가능하게 함

주의:
- 이 파일은 직접 실행하지 않는다.
- 테스트/실행의 엔트리포인트는 각 scripts/*.py 파일이다.
"""

# 패키지 마커 역할만 수행
# (의도적으로 import를 두지 않음 — 실행 시 부작용 방지)

__all__ = []
