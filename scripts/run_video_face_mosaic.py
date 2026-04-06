import cv2
from pathlib import Path

from services.face_detector import FaceDetector
from services.face_recognizer import FaceRecognizer
from services.identity_matcher import IdentityMatcher
from services.mosaic import apply_mosaic  # ✅ B 코드(함수형)와 일치
from db.sqlite import SQLiteDB

from core.config import MOSAIC_MODE  # blur/pixelate 설정값 재사용 (팀 공통)


# ===== 입력/출력 경로 =====
VIDEO_IN = "data/input/raw/test.mp4"
VIDEO_OUT = "data/output/out_mosaic.mp4"


def main():
    in_path = Path(VIDEO_IN)
    assert in_path.exists(), f"video not found: {in_path}"

    cap = cv2.VideoCapture(str(in_path))
    assert cap.isOpened(), "failed to open video"

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = Path(VIDEO_OUT)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    # ---- 모듈 ----
    detector = FaceDetector()
    recognizer = FaceRecognizer()

    db = SQLiteDB("db/cctv_mosaic.sqlite3")
    matcher = IdentityMatcher(db=db, enable_cache=True)
    matcher.refresh_cache()  # 등록 직후 안전빵

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        faces = detector.detect(frame)

        for f in faces:
            bbox = tuple(map(int, f.bbox))  # (x1, y1, x2, y2)

            try:
                emb_obj = recognizer.extract_from_frame(frame, bbox=bbox)
                result = matcher.match_face(emb_obj.embedding)

                if not result.is_registered:
                    # ✅ B의 mosaic API: 여러 bbox를 받는 함수
                    frame = apply_mosaic(frame, [bbox], mode=MOSAIC_MODE)
                # else: 등록 사용자 -> 모자이크 안함

            except Exception:
                # 인식 실패 시 보수적으로 모자이크
                frame = apply_mosaic(frame, [bbox], mode=MOSAIC_MODE)

        writer.write(frame)
        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"[INFO] processed {frame_idx} frames...")

    cap.release()
    writer.release()
    print("[OK] saved:", out_path)


if __name__ == "__main__":
    main()
