from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass(frozen=True)
class EvidenceReference:
    ref_id: int
    document_id: str = ""
    chunk_id: str = ""
    document_title: str = ""
    section: str = ""
    artifact_type: str = ""
    source_type: str = "local"
    score: float | None = None
    snippet: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def marker(self) -> str:
        return f"[{self.ref_id}]"

    def model_dump(self) -> Dict[str, Any]:
        return {
            "ref_id": self.ref_id,
            "marker": self.marker,
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "document_title": self.document_title,
            "section": self.section,
            "artifact_type": self.artifact_type,
            "source_type": self.source_type,
            "score": self.score,
            "snippet": self.snippet,
            "metadata": dict(self.metadata or {}),
        }


@dataclass(frozen=True)
class CitationReviewResult:
    reviewed: bool
    citation_count: int
    invalid_ref_ids: List[int] = field(default_factory=list)
    missing_citation_claims: List[str] = field(default_factory=list)
    cited_ref_ids: List[int] = field(default_factory=list)
    reference_count: int = 0

    @property
    def risk(self) -> int:
        if self.invalid_ref_ids:
            return 2
        if self.missing_citation_claims:
            return 1
        return 0

    def model_dump(self) -> Dict[str, Any]:
        return {
            "reviewed": self.reviewed,
            "citation_count": self.citation_count,
            "invalid_ref_ids": list(self.invalid_ref_ids),
            "missing_citation_claims": list(self.missing_citation_claims),
            "cited_ref_ids": list(self.cited_ref_ids),
            "reference_count": self.reference_count,
            "risk": self.risk,
        }


def _clean_text(value: Any, limit: int = 900) -> str:
    text = " ".join(str(value or "").split()).strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def _metadata_from_item(item: Any) -> Dict[str, Any]:
    if hasattr(item, "metadata"):
        metadata = getattr(item, "metadata") or {}
    elif isinstance(item, dict):
        metadata = item.get("metadata") or {}
    else:
        metadata = {}
    return dict(metadata) if isinstance(metadata, dict) else {}


def _value_from_item(item: Any, key: str, default: Any = "") -> Any:
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def build_evidence_references(items: List[Any], *, limit: int = 8) -> List[EvidenceReference]:
    references: List[EvidenceReference] = []
    seen: set[tuple[str, str, str]] = set()
    for item in list(items or []):
        metadata = _metadata_from_item(item)
        document_id = str(_value_from_item(item, "document_id") or "").strip()
        chunk_id = str(_value_from_item(item, "chunk_id") or "").strip()
        snippet = _clean_text(
            _value_from_item(item, "content", "")
            or _value_from_item(item, "snippet", "")
        )
        key = (document_id, chunk_id, snippet[:80])
        if key in seen or not snippet:
            continue
        seen.add(key)
        references.append(
            EvidenceReference(
                ref_id=len(references) + 1,
                document_id=document_id,
                chunk_id=chunk_id,
                document_title=str(_value_from_item(item, "document_title") or "").strip(),
                section=str(
                    metadata.get("section_path_text")
                    or metadata.get("section_title")
                    or ""
                ).strip(),
                artifact_type=str(metadata.get("artifact_type") or "").strip(),
                source_type=str(_value_from_item(item, "source_type", "") or metadata.get("source_type") or "local").strip(),
                score=_value_from_item(item, "score", None),
                snippet=snippet,
                metadata=metadata,
            )
        )
        if len(references) >= limit:
            break
    return references


def format_evidence_references_for_prompt(references: List[EvidenceReference]) -> str:
    if not references:
        return "当前未检索到可用证据。"
    blocks: List[str] = []
    for ref in references:
        lines = [
            f"[Evidence {ref.ref_id}]",
            f"citation_marker: {ref.marker}",
            f"document_title: {ref.document_title or 'N/A'}",
            f"score: {ref.score}",
            f"section: {ref.section or 'N/A'}",
            f"artifact_type: {ref.artifact_type or 'N/A'}",
            f"chunk_id: {ref.chunk_id or 'N/A'}",
            "content:",
            ref.snippet or "N/A",
        ]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


_CITATION_RE = re.compile(r"\[(?:E|Evidence\s*)?(\d+)\]", flags=re.IGNORECASE)
_CLAIM_SPLIT_RE = re.compile(r"[\n。！？；;]+")
_NUMERIC_RE = re.compile(r"\b20\d{2}\b|\b\d+(?:\.\d+)?%\b|\b\d+(?:\.\d+)?\b")
_CLAIM_CUES = (
    "实验",
    "提升",
    "下降",
    "结果",
    "表明",
    "证明",
    "优于",
    "方法",
    "模块",
    "机制",
    "accuracy",
    "improvement",
    "experiment",
    "result",
    "outperform",
)


def extract_citation_ids(answer: str) -> List[int]:
    ids: List[int] = []
    for match in _CITATION_RE.finditer(str(answer or "")):
        try:
            ref_id = int(match.group(1))
        except ValueError:
            continue
        if ref_id not in ids:
            ids.append(ref_id)
    return ids


def _looks_like_evidence_claim(text: str) -> bool:
    value = str(text or "").strip()
    if not value:
        return False
    lowered = value.lower()
    return bool(_NUMERIC_RE.search(value)) or any(cue in lowered for cue in _CLAIM_CUES)


def _claim_units(answer: str) -> List[str]:
    return [unit.strip(" -*:：") for unit in _CLAIM_SPLIT_RE.split(str(answer or "")) if unit.strip(" -*:：")]


def review_answer_citations(
    *,
    answer: str,
    references: List[EvidenceReference],
) -> CitationReviewResult:
    if not str(answer or "").strip():
        return CitationReviewResult(reviewed=False, citation_count=0, reference_count=len(references or []))

    valid_ref_ids = {ref.ref_id for ref in references or []}
    cited_ref_ids = extract_citation_ids(answer)
    invalid_ref_ids = [ref_id for ref_id in cited_ref_ids if ref_id not in valid_ref_ids]
    missing_claims: List[str] = []
    for unit in _claim_units(answer):
        if _looks_like_evidence_claim(unit) and not extract_citation_ids(unit):
            missing_claims.append(_clean_text(unit, limit=180))
        if len(missing_claims) >= 5:
            break

    return CitationReviewResult(
        reviewed=True,
        citation_count=len(cited_ref_ids),
        invalid_ref_ids=invalid_ref_ids,
        missing_citation_claims=missing_claims,
        cited_ref_ids=cited_ref_ids,
        reference_count=len(references or []),
    )
