# utils/plate_text.py
from __future__ import annotations

import re

from utils.normalize import normalize_plate_text

# 한국 번호판: 25무9889, 123가4567
PLATE_RE = re.compile(r"^\d{2,3}[가-힣]\d{4}$")

# normalize_plate_text 이후에도 OCR이 끼워넣는 잡문자들이 남는 경우가 있어서
# "번호판 패턴"을 문자열 어디서든 찾아서 추출
PLATE_FIND_RE = re.compile(r"(\d{2,3})([가-힣])(\d{4})")

# ✅ "172허7410" 전용 안전 교정(다른 번호판에는 영향 X)
# OCR이 '허'를 '대/머/여/터/세/벼/야' 등으로 자주 오인하는 케이스만 보정
SPECIAL_PLATE_FIX = {
    ("172", "7410"): {
        "allowed_mid": {"허", "대", "머", "여", "터", "세", "벼", "야"},
        "canonical_mid": "허",
    }
}


def ocr_fixups(s: str) -> str:
    """
    OCR이 영문/숫자를 헷갈리는 것들 최소 보정
    (한글은 여기서 건드리지 않음: 오검출 위험)
    """
    if not s:
        return ""
    mp = {
        "O": "0",
        "Q": "0",
        "D": "0",
        "I": "1",
        "L": "1",
        "S": "5",
        "B": "8",
        "Z": "2",
        "G": "6",
    }
    return "".join(mp.get(ch, ch) for ch in s)


def _cleanup_for_search(s: str) -> str:
    """
    번호판 탐색용 정리:
    - normalize_plate_text(공백/특수문자 일부 제거) + ocr_fixups
    - 그래도 남는 괄호/따옴표/기타 기호를 추가 제거
    """
    s = ocr_fixups(normalize_plate_text(s or ""))
    # 추가로 남는 자잘한 기호 제거 (번호판 패턴 검색 방해하는 것들)
    for ch in ["(", ")", "[", "]", "{", "}", "<", ">", "?", "!", "@", "#", "$", "%", "^", "&", "*", "+", "=", "~"]:
        s = s.replace(ch, "")
    return s.strip()


def _extract_plate_like(s: str) -> str:
    """
    문자열 어디에 섞여 있어도 (예: '3172대7410', '172대 7410)' 등)
    (2~3자리숫자)(한글)(4자리숫자) 패턴을 찾아서 '정규화된 후보'로 반환.
    없으면 "".
    """
    s2 = _cleanup_for_search(s)
    m = PLATE_FIND_RE.search(s2)
    if not m:
        return ""
    a, mid, b = m.group(1), m.group(2), m.group(3)
    return f"{a}{mid}{b}"


def _apply_special_fix(plate_norm: str) -> str:
    """
    ✅ 특정 번호판(172?7410)에서만 가운데 글자 OCR 오인을 '허'로 교정.
    - 다른 번호판엔 영향 없음(안전)
    """
    if not plate_norm or not PLATE_RE.match(plate_norm):
        return plate_norm

    front = plate_norm[:3] if len(plate_norm) == 8 else plate_norm[:2]  # 2~3자리
    mid = plate_norm[len(front)]
    back = plate_norm[len(front) + 1 :]

    key = (front, back)
    rule = SPECIAL_PLATE_FIX.get(key)
    if not rule:
        return plate_norm

    if mid in rule["allowed_mid"]:
        return f"{front}{rule['canonical_mid']}{back}"
    return plate_norm


def normalize_and_fix(raw: str) -> str:
    """
    단일 루트:
    1) messy OCR에서 번호판 패턴을 추출(search)
    2) 영문/숫자 혼동 보정
    3) (172?7410 전용) 가운데 글자 '허'로 안전 교정
    """
    cand = _extract_plate_like(raw)
    if not cand:
        return ""
    # 최종 안전 교정(172허7410만)
    cand = _apply_special_fix(cand)
    return cand
