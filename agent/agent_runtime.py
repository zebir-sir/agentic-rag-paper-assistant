from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .display_utils import clean_snippet_text, make_friendly_title
from .models import EvidenceSource


@dataclass
class AgentDependencies:
    session_id: str
    user_id: Optional[str] = None
    use_web_search: bool = False
    search_preferences: Dict[str, Any] = None
    retrieved_sources: List[EvidenceSource] = field(default_factory=list)
    source_keys: set[str] = field(default_factory=set)

    def __post_init__(self):
        if self.search_preferences is None:
            self.search_preferences = {
                "default_search_type": "hybrid",
                "default_limit": 10,
            }


def _resolve_default_search_type(deps: AgentDependencies) -> str:
    requested = str((deps.search_preferences or {}).get("default_search_type", "hybrid")).lower()
    return "vector" if requested == "vector" else "hybrid"


def _build_source_key(
    source_type: str,
    chunk_id: Optional[str],
    document_id: Optional[str],
    metadata: Dict[str, Any],
    document_title: str,
    snippet: str,
) -> str:
    if chunk_id:
        return f"{source_type}:chunk:{chunk_id}"
    if source_type == "web":
        doi = str(metadata.get("doi") or "").strip().lower()
        openalex_id = str(metadata.get("openalex_id") or "").strip().lower()
        if doi:
            return f"web:doi:{doi}"
        if openalex_id:
            return f"web:openalex:{openalex_id}"
    if document_id:
        return f"{source_type}:docid:{document_id}"
    weak = snippet[:80]
    return f"{source_type}:doc:{document_title}|{weak}"


def _collect_evidence_hit(
    deps: AgentDependencies,
    hit: Dict[str, Any],
) -> None:
    snippet = clean_snippet_text(str(hit.get("content") or ""), max_len=280)
    if not snippet:
        return

    raw_document_title = str(hit.get("document_title") or "")
    document_source = str(hit.get("document_source") or "")
    document_title = make_friendly_title(raw_document_title, document_source)
    chunk_id = str(hit.get("chunk_id")) if hit.get("chunk_id") else None
    document_id = str(hit.get("document_id")) if hit.get("document_id") else None
    score = hit.get("score")
    metadata = hit.get("metadata") if isinstance(hit.get("metadata"), dict) else {}
    metadata = {"source_type": "local", **metadata}
    source_type = "web" if str(metadata.get("source_type", "local")).lower() == "web" else "local"

    key = _build_source_key(
        source_type=source_type,
        chunk_id=chunk_id,
        document_id=document_id,
        metadata=metadata,
        document_title=document_title,
        snippet=snippet,
    )
    if key in deps.source_keys:
        return

    deps.source_keys.add(key)
    deps.retrieved_sources.append(
        EvidenceSource(
            source_type=source_type,
            document_id=document_id,
            document_title=document_title,
            document_source=document_source,
            chunk_id=chunk_id,
            snippet=snippet,
            score=float(score) if isinstance(score, (int, float)) else None,
            metadata=metadata,
        )
    )
