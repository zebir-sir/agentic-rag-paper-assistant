import os
import re
from typing import Optional


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def localize_paper_terms(text: str) -> str:
    value = text or ""
    value = re.sub(r"\bSection\s+([0-9]+(?:[.\-][0-9]+)*)\b", r"Section \1", value, flags=re.IGNORECASE)
    value = re.sub(r"\bTable\s+([0-9]+(?:[.\-][0-9]+)*)\b", r"Table \1", value, flags=re.IGNORECASE)
    value = re.sub(r"\bAlgorithm\s+([0-9]+(?:[.\-][0-9]+)*)\b", r"Algorithm \1", value, flags=re.IGNORECASE)
    value = re.sub(r"\bIntroduction\b", "Introduction", value, flags=re.IGNORECASE)
    value = re.sub(r"\bConclusion\b", "Conclusion", value, flags=re.IGNORECASE)
    value = re.sub(r"\bRelated Work\b", "Related Work", value, flags=re.IGNORECASE)
    value = re.sub(r"\bExperiments?\b", "Experiments", value, flags=re.IGNORECASE)
    value = re.sub(r"\bAppendix\b", "Appendix", value, flags=re.IGNORECASE)
    return value


def clean_snippet_text(text: str, max_len: int = 280) -> str:
    value = text or ""
    value = re.sub(r"<!--.*?-->", " ", value, flags=re.DOTALL)
    value = re.sub(r"\bformula-not-decoded\b", " ", value, flags=re.IGNORECASE)
    value = _normalize_whitespace(value)
    value = localize_paper_terms(value)
    if len(value) > max_len:
        value = value[:max_len].rstrip() + "..."
    return value


def make_friendly_title(raw_title: Optional[str], raw_source: Optional[str] = None) -> str:
    title = _normalize_whitespace(raw_title or "")
    source = _normalize_whitespace(raw_source or "")
    candidate = title or source or "Paper"
    candidate = os.path.basename(candidate)
    candidate = re.sub(r"\.pdf$", "", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"-main$", "", candidate, flags=re.IGNORECASE)
    candidate = candidate.strip(" -_")
    candidate = _normalize_whitespace(candidate.replace("_", " "))
    if not candidate:
        candidate = "Paper"

    looks_like_code = bool(re.fullmatch(r"[A-Za-z0-9.\-]+", candidate)) and " " not in candidate
    if looks_like_code:
        candidate = f"Paper {candidate}"

    return localize_paper_terms(candidate)
