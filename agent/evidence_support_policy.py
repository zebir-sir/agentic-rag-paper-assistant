from __future__ import annotations

import re
from typing import Iterable


_CITATION_MARKER_RE = re.compile(r"\[(?:E|Evidence\s*)?\d+\]", flags=re.IGNORECASE)
_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?%?\b")


def normalize_support_text(value: str) -> str:
    """Keep lexical support checks deterministic across English and Chinese evidence."""
    return "".join(char.lower() for char in str(value or "") if char.isalnum())


def _char_ngrams(value: str, size: int = 2) -> set[str]:
    normalized = normalize_support_text(value)
    if not normalized:
        return set()
    if len(normalized) <= size:
        return {normalized}
    return {normalized[index : index + size] for index in range(len(normalized) - size + 1)}


def is_citation_claim_supported(claim: str, evidence_texts: Iterable[str]) -> bool:
    """Return whether a cited factual sentence has adequate lexical support.

    This is a deterministic guardrail, not a semantic entailment model. It catches
    invalid numeric attributions and obvious citation drift before a response is
    returned while keeping the LLM-based answer review independent.
    """
    claim_without_markers = _CITATION_MARKER_RE.sub("", str(claim or "")).strip()
    normalized_claim = normalize_support_text(claim_without_markers)
    support_text = " ".join(str(item or "") for item in evidence_texts)
    normalized_support = normalize_support_text(support_text)
    if not normalized_claim or not normalized_support:
        return False

    claim_numbers = set(_NUMBER_RE.findall(claim_without_markers))
    support_numbers = set(_NUMBER_RE.findall(support_text))
    if claim_numbers and not claim_numbers.issubset(support_numbers):
        return False
    if normalized_claim in normalized_support:
        return True

    claim_ngrams = _char_ngrams(claim_without_markers)
    support_ngrams = _char_ngrams(support_text)
    if not claim_ngrams or not support_ngrams:
        return False
    overlap_count = len(claim_ngrams & support_ngrams)
    overlap_ratio = overlap_count / len(claim_ngrams)
    return overlap_ratio >= 0.42 or (overlap_count >= 8 and overlap_ratio >= 0.24)
