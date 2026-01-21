import cv2
import numpy as np
from pathlib import Path

from services.face_detector import FaceDetector
from services.face_recognizer import FaceRecognizer
from core.config import FACE_EMBED_DIM


def main():
    img_path = Path("data/faces/test.jpg")
    assert img_path.exists(), f"Image not found: {img_path}"

    # 1. 이미지 로드
    img = cv2.imread(str(img_path))
    assert img is not None, "Failed to load image"

    # 2. 얼굴 검출
    detector = FaceDetector()
    faces = detector.detect(img)

    print(f"[INFO] detected faces: {len(faces)}")
    assert len(faces) > 0, "No face detected"

    # 3. 얼굴 인식 (임베딩)
    recognizer = FaceRecognizer()
    emb_obj = recognizer.extract_from_frame(img, bbox=faces[0].bbox)
    emb = emb_obj.embedding


    # 4. 검증
    print("embedding shape:", emb.shape)
    print("embedding norm:", np.linalg.norm(emb))

    assert emb.shape == (FACE_EMBED_DIM,)
    print("[OK] face embedding extracted correctly")


if __name__ == "__main__":
    main()
