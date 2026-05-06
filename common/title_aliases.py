import os
from typing import Optional


TITLE_ALIASES = {
    "s44443-025-00393-9": "HA-RRT: A heuristic and adaptive RRT algorithm for ship path planning",
    "1-s2.0-s002980182403244x-main": "HMA-RRT: A hybrid multi-strategy adaptive RRT for autonomous vessel path planning",
    "1-s2.0-s002980182403244x": "HMA-RRT: A hybrid multi-strategy adaptive RRT for autonomous vessel path planning",
}


def _normalize_key(value: str) -> str:
    text = (value or "").strip()
    text = os.path.basename(text)
    if text.lower().endswith(".pdf"):
        text = text[:-4]
    return text.strip().lower()


def get_title_alias(
    document_title: Optional[str],
    document_source: Optional[str] = None,
    document_id: Optional[str] = None,
) -> Optional[str]:
    candidates = [
        _normalize_key(document_title or ""),
        _normalize_key(document_source or ""),
        _normalize_key(document_id or ""),
    ]
    for key in candidates:
        if key and key in TITLE_ALIASES:
            return TITLE_ALIASES[key]
    return None
