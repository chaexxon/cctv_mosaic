# cctv_mosaic

CCTV Mosaic System

- Face recognition (ArcFace)
- Plate detection (YOLOv8)
- Mosaic processing
- Video segment processing

Usage:
python -m scripts.process_video --in data/input/raw/test.mp4 --out data/output/out.mp4 --enable_plate