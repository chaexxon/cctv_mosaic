# utils/normalize.py
from __future__ import annotations


def normalize_plate_text(s: str) -> str:
    s = s.strip()
    for ch in [" ", "\t", "-", "_", ".", ","]:
        s = s.replace(ch, "")
    return s.upper()


def is_plausible_plate(s_norm: str) -> bool:
    """
    너무 엄격하면 OCR 결과를 다 버릴 수 있음.
    - 길이 6~10
    - 숫자 2개 이상
    """
    if not (6 <= len(s_norm) <= 10):
        return False
    digits = sum(c.isdigit() for c in s_norm)
    return digits >= 2
