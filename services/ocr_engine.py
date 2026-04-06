# services/ocr_engine.py
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class OCRResult:
    text: str
    conf: float


class OCREngine:
    """
    OCR wrapper with graceful fallback.
    backend:
      - "easyocr"      (if installed)
      - "pytesseract"  (if installed)
      - "none"         (if nothing installed) -> caller may choose safe behavior
    """

    def __init__(self, *, lang: str = "ko"):
        self.backend = "none"
        self.reader = None
        self.lang = lang

        # easyocr (recommended)
        try:
            import easyocr  # type: ignore

            self.reader = easyocr.Reader(["ko", "en"], gpu=False)
            self.backend = "easyocr"
            return
        except Exception:
            pass

        # pytesseract (optional)
        try:
            import pytesseract  # type: ignore

            self.reader = pytesseract
            self.backend = "pytesseract"
            return
        except Exception:
            pass

        self.backend = "none"

    @staticmethod
    def _prep(crop_bgr: np.ndarray) -> np.ndarray:
        """
        Simple preprocessing for plate OCR.
        """
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 7, 50, 50)
        th = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            7,
        )
        return th

    def read_plate(self, crop_bgr: np.ndarray) -> OCRResult:
        if crop_bgr is None or crop_bgr.size == 0:
            return OCRResult("", 0.0)

        img = self._prep(crop_bgr)

        if self.backend == "easyocr":
            try:
                results = self.reader.readtext(img)  # type: ignore
                # [(bbox, text, conf), ...]
                if not results:
                    return OCRResult("", 0.0)
                best = max(results, key=lambda x: float(x[2]) if len(x) >= 3 else 0.0)
                text = str(best[1]).strip()
                conf = float(best[2]) if len(best) >= 3 else 0.0
                return OCRResult(text, conf)
            except Exception:
                return OCRResult("", 0.0)

        if self.backend == "pytesseract":
            try:
                import pytesseract  # type: ignore

                config = "--psm 7"
                text = pytesseract.image_to_string(img, lang="kor+eng", config=config)
                text = (text or "").strip()
                conf = 0.5 if text else 0.0
                return OCRResult(text, conf)
            except Exception:
                return OCRResult("", 0.0)

        return OCRResult("", 0.0)
