# utils/plate_store.py
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

DEFAULT_STORE = Path("data") / "plates" / "plates.json"


def _ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def load_store(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"plates": []}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        bak = path.with_suffix(".json.bak")
        try:
            path.replace(bak)
        except Exception:
            pass
        return {"plates": []}


def save_store(path: Path, data: Dict[str, Any]) -> None:
    _ensure_parent(path)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def is_registered(store: Dict[str, Any], plate_norm: str) -> bool:
    for item in store.get("plates", []):
        if item.get("plate_norm") == plate_norm:
            return True
    return False


@dataclass
class PlateRecord:
    owner: str
    plate_raw: str
    plate_norm: str
    created_at: str


def add_plate(store: Dict[str, Any], owner: str, plate_raw: str, plate_norm: str) -> PlateRecord:
    rec = PlateRecord(
        owner=owner,
        plate_raw=plate_raw,
        plate_norm=plate_norm,
        created_at=datetime.now().isoformat(timespec="seconds"),
    )
    store.setdefault("plates", []).append(
        {
            "owner": rec.owner,
            "plate_raw": rec.plate_raw,
            "plate_norm": rec.plate_norm,
            "created_at": rec.created_at,
        }
    )
    return rec
