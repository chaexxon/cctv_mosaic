# CCTV Face Mosaic System (Capstone Design)

본 프로젝트는 CCTV 영상에서 얼굴을 탐지하고,
일시적인 가림이나 측면 얼굴로 인해 검출이 끊기더라도
동일 인물에 대해 **일관된 모자이크(Stable ID 기반)**를 적용하는 시스템이다.

---

## B: Video Processing (Face Detection + Stable-ID Tracking + Mosaic)

### 개요
본 모듈은 입력 영상에 대해 다음 파이프라인을 수행한다.

1. **Face Detection**  
   - YOLO 기반 얼굴 검출기 사용
2. **Raw Tracking (tid)**  
   - IoU + 중심거리 기반 추적
   - 검출이 끊기면 `track_id(tid)`는 변경될 수 있음
3. **Stable ID Tracking (sid)**  
   - 공간 정보 + appearance(HSV histogram)를 이용해
     동일 인물을 재부착
   - `stable_id(sid)`는 모자이크 유지 기준으로 사용
4. **Mosaic Rendering**
   - Stable ID 기준으로 얼굴 모자이크 적용

---

### tid vs sid
- **tid (track_id)**  
  Raw tracker에서 부여되는 임시 ID  
  검출 실패, 가림, 측면 얼굴 등에서 변경될 수 있음

- **sid (stable_id)**  
  Stable-ID Manager에서 부여되는 영구 ID  
  동일 인물에 대해 영상 전체에서 일관되게 유지됨  
  → **실제 모자이크 및 로그 기준**

---

### 실행 방법 (Windows PowerShell)

기본 실행:
```powershell
python -m scripts.process_video --in data/input/segments/clip_test.mp4 --out data/output/clip_final.mp4
