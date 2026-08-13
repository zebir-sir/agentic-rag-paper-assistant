"""Pure data rules for graph-localization cache entries."""
import hashlib
import json
import re
from typing import Any, Dict

LOCALIZATION_SCHEMA_VERSION = 1


def graph_localization_hash(title: str, profile_text: str) -> str:
    value = f"{LOCALIZATION_SCHEMA_VERSION}\n{title}\n{profile_text}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def normalize_graph_card(value: Dict[str, Any], original_title: str) -> Dict[str, Any]:
    keywords = value.get("keywords_zh") if isinstance(value.get("keywords_zh"), list) else []
    return {
        "title_zh": str(value.get("title_zh") or original_title).strip(),
        "overview_zh": str(value.get("overview_zh") or "").strip()[:100],
        "problem_zh": str(value.get("problem_zh") or "").strip()[:120],
        "method_zh": str(value.get("method_zh") or "").strip()[:120],
        "innovation_zh": str(value.get("innovation_zh") or "").strip()[:120],
        "keywords_zh": [str(item).strip()[:24] for item in keywords if str(item).strip()][:5],
    }


def validate_graph_card(card: Dict[str, Any], title: str) -> Dict[str, Any]:
    required = ["title_zh", "problem_zh", "method_zh", "innovation_zh"]
    missing = [field for field in required if not str(card.get(field) or "").strip()]
    protected = sorted(set(re.findall(r"\b[A-Z]{2,}[A-Z0-9]*\*|\b[A-Z][A-Z0-9]*(?:-[A-Z0-9]+)+\b", title)))
    output = json.dumps(card, ensure_ascii=False)
    missing_protected = [item for item in protected if item not in output]
    return {"valid": not missing and not missing_protected, "missing_fields": missing, "missing_protected_tokens": missing_protected}
