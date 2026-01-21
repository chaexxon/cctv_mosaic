# services/face_detector.py
from ultralytics import YOLO
from pathlib import Path
import cv2

class FaceDetector:
    def __init__(self, conf_thres=0.5):
        self.conf_thres = conf_thres

        model_path = Path("models/yolo_face/yolov8n-face.pt")
        if not model_path.exists():
            raise FileNotFoundError(f"YOLO face model not found: {model_path}")

        self.model = YOLO(str(model_path))

    def detect(self, frame):
        """
        Returns:
            list of dicts:
            {
              "bbox": (x1, y1, x2, y2),
              "conf": float,
              "cls": "face"
            }
        """
        results = self.model(frame, conf=self.conf_thres, verbose=False)

        detections = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "conf": conf,
                    "cls": "face"
                })

        return detections
