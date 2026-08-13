"""Evidence-backed paper relationship extraction from already ingested chunks."""

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List


_TITLE_STOPWORDS = {"a", "an", "the", "of", "for", "and", "in", "on", "to", "with", "based"}
_METHOD_SECTION_CUES = ("method", "approach", "algorithm", "methodology", "方法", "算法", "模型")
_LINEAGE_CUES = ("based on", "builds on", "extends", "extended", "extend", "improves", "improved", "improve", "variant of", "基于", "扩展", "改进", "继承")


@dataclass(frozen=True)
class GraphCandidate:
    document_id: str
    title: str
    abbreviation: str = ""


def _tokens(value: str) -> List[str]:
    return [token for token in re.findall(r"[a-z0-9]{2,}|[\u4e00-\u9fff]{2,}", str(value or "").lower()) if token not in _TITLE_STOPWORDS]


def _title_matches(text: str, candidate: GraphCandidate) -> bool:
    normalized_text = " ".join(_tokens(text))
    title_tokens = _tokens(candidate.title)
    if len(title_tokens) >= 3:
        return len(set(title_tokens) & set(_tokens(text))) / len(set(title_tokens)) >= 0.84
    abbreviation = str(candidate.abbreviation or "").strip().lower()
    return bool(abbreviation and len(abbreviation) >= 3 and re.search(rf"(?<![a-z0-9]){re.escape(abbreviation)}(?![a-z0-9])", normalized_text))


def _excerpt(content: str, candidate: GraphCandidate) -> str:
    value = " ".join(str(content or "").split())
    needle = next((token for token in _tokens(candidate.title) if len(token) >= 5 and token in value.lower()), "")
    index = value.lower().find(needle) if needle else 0
    return value[max(0, index - 180): index + 420]


def extract_evidence_backed_relations(
    source_document_id: str,
    chunks: Iterable[Dict[str, Any]],
    candidates: Iterable[GraphCandidate],
) -> List[Dict[str, Any]]:
    """Return only relations whose source chunk can be shown to a user."""
    relations: List[Dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    candidates = [candidate for candidate in candidates if candidate.document_id != source_document_id]
    for chunk in chunks:
        content = str(chunk.get("content") or "")
        metadata = dict(chunk.get("metadata") or {})
        section = str(metadata.get("section_path_text") or metadata.get("section_title") or "")
        section_lower = section.lower()
        is_reference_section = "reference" in section_lower or "bibliograph" in section_lower or "参考文献" in section
        is_method_section = any(cue in section_lower or cue in section for cue in _METHOD_SECTION_CUES)
        lineage_cue = next((cue for cue in _LINEAGE_CUES if cue in content.lower()), "")
        for candidate in candidates:
            if not _title_matches(content, candidate):
                continue
            base_evidence = {
                "source_chunk_id": str(chunk.get("id") or ""),
                "source_section": section,
                "excerpt": _excerpt(content, candidate),
                "matched_title": candidate.title,
            }
            if is_reference_section and (candidate.document_id, "cites") not in seen:
                relations.append({"target_document_id": candidate.document_id, "relation_type": "cites", "score": 1.0, "evidence": {**base_evidence, "kind": "reference_title_match", "explanation": "该论文的 References 段明确列出库内论文。"}})
                seen.add((candidate.document_id, "cites"))
            if is_method_section and lineage_cue and (candidate.document_id, "method_lineage") not in seen:
                relations.append({"target_document_id": candidate.document_id, "relation_type": "method_lineage", "score": 0.9, "evidence": {**base_evidence, "kind": "method_section_explicit_cue", "cue": lineage_cue, "explanation": "方法段明确表述基于、扩展或改进该库内论文的方法。"}})
                seen.add((candidate.document_id, "method_lineage"))
    return relations
