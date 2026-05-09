import json
import os
import re
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple, TypedDict

from langgraph.graph import END, START, StateGraph

from .agent_langchain import get_langchain_chat_model
from .agent_runtime import AgentDependencies
from .intent_planner import (
    IntentPlan,
    PlannerCapabilities,
    build_fallback_intent_plan,
    plan_user_intent_debug,
)
from .langchain_tools import build_langchain_tools
from .models import EvidenceSource, ToolCall
from .planner_runtime import execute_intent_plan_steps, summarize_hits_for_planner
from .prompts import SYSTEM_PROMPT
from .tools import DocumentListInput, is_general_web_search_enabled, list_documents_tool
from .openalex_router import _is_openalex_enabled


class LangGraphAnalysisState(TypedDict, total=False):
    question: str
    context_prompt: str
    session_id: str
    user_id: Optional[str]
    deps: AgentDependencies
    documents: List[Dict[str, Any]]
    retrieval_results: List[Dict[str, Any]]
    draft_answer: str
    final_answer: str
    tools_used: List[ToolCall]
    sources: List[EvidenceSource]
    warnings: List[str]
    metadata: Dict[str, Any]
    progress_callback: Optional[Callable[[str], Awaitable[None]]]
    current_query: str
    retrieval_attempt_count: int
    retrieval_attempts: List[Dict[str, Any]]
    rewritten_queries: List[str]
    retrieval_sufficient: bool
    retrieval_insufficient_reason: Optional[str]
    retrieval_top_score: Optional[float]
    max_retrieval_attempts: int
    skip_rewrite: bool
    suggested_rewrite_query: str
    retrieval_evaluation: Dict[str, Any]
    target_document_id: str
    target_document_title: str
    answer_scope: Dict[str, Any]
    scope_policy: str
    target_documents: List[Dict[str, Any]]
    allow_supplemental: bool
    scope_resolver_used: bool


@dataclass
class LangGraphAnalysisResult:
    message: str
    raw_state: Dict[str, Any]
    tools_used: List[ToolCall]
    sources: List[EvidenceSource]
    metadata: Dict[str, Any]


def _append_tool_call(state: LangGraphAnalysisState, tool_name: str, args: Dict[str, Any]) -> None:
    calls = list(state.get("tools_used") or [])
    calls.append(ToolCall(tool_name=tool_name, args=args or {}, tool_call_id=None))
    state["tools_used"] = calls


def _append_warning(state: LangGraphAnalysisState, warning_text: str) -> None:
    warnings = list(state.get("warnings") or [])
    warnings.append(warning_text)
    state["warnings"] = warnings


async def _emit_progress(
    state: LangGraphAnalysisState,
    message: str,
    *,
    phase: str = "warning",
    user_visible: bool = True,
) -> None:
    callback = state.get("progress_callback")
    if callback is None:
        return
    await callback(
        {
            "content": str(message or "").strip(),
            "phase": str(phase or "warning"),
            "user_visible": bool(user_visible),
            "level": "info",
        }
    )


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _build_planner_capabilities(deps: Optional[AgentDependencies] = None) -> PlannerCapabilities:
    search_preferences = (deps.search_preferences or {}) if deps is not None else {}
    user_allow_web = bool(
        search_preferences.get(
            "allow_web_search",
            bool(getattr(deps, "use_web_search", False)),
        )
    )
    user_allow_openalex = bool(search_preferences.get("allow_openalex_search", True))
    provider_openalex = bool(_is_openalex_enabled())
    provider_web = bool(is_general_web_search_enabled())

    return PlannerCapabilities(
        local_search_enabled=True,
        vector_search_enabled=True,
        hybrid_search_enabled=True,
        section_search_enabled=True,
        artifact_search_enabled=True,
        openalex_search_enabled=bool(user_allow_openalex and provider_openalex),
        web_search_enabled=bool(user_allow_web and provider_web),
        direct_answer_enabled=True,
        max_tools=2,
    )


def clean_legacy_warning_text(text: str, drop_warning: bool = False) -> str:
    value = str(text or "")
    legacy_full = "当前没有检索到直接相关片段，以下内容更适合作为一般性分析参考。"
    legacy_tail = "以下内容更适合作为一般性分析参考。"
    neutral_full = "本轮没有可核对的检索片段；未被检索证据支持的结论请谨慎参考。"
    neutral_tail = "未被检索证据支持的结论请谨慎参考。"
    if drop_warning:
        return value.replace(legacy_full, "").replace(legacy_tail, "").strip()
    return value.replace(legacy_full, neutral_full).replace(legacy_tail, neutral_tail)


def _build_runtime_decision_summary(metadata: Dict[str, Any]) -> str:
    metadata = dict(metadata or {})
    planner_decision = metadata.get("planner_decision") or {}
    if not isinstance(planner_decision, dict):
        planner_decision = {}

    lines: List[str] = []
    if planner_decision:
        lines.append(f"- intent: {planner_decision.get('intent')}")
        lines.append(f"- needs_retrieval: {planner_decision.get('needs_retrieval')}")
        lines.append(f"- direct_answer_allowed: {planner_decision.get('direct_answer_allowed')}")
        lines.append(f"- allow_external_sources: {planner_decision.get('allow_external_sources')}")
        lines.append(f"- planned_tools: {planner_decision.get('planned_tools')}")
        lines.append(f"- evidence_policy: {planner_decision.get('evidence_policy')}")
        lines.append(f"- reason: {planner_decision.get('reason')}")
    else:
        lines.append("- planner_decision: unavailable")

    lines.append(f"- retrieval_skipped_by_planner: {metadata.get('retrieval_skipped_by_planner')}")
    lines.append(f"- retrieval_skip_reason: {metadata.get('retrieval_skip_reason')}")
    lines.append(f"- retrieval_sufficient: {metadata.get('retrieval_sufficient')}")
    lines.append(f"- retrieval_insufficient_reason: {metadata.get('retrieval_insufficient_reason')}")
    lines.append(f"- retrieval_result_count: {metadata.get('retrieval_result_count')}")
    lines.append(f"- retrieval_attempt_count: {metadata.get('retrieval_attempt_count')}")
    lines.append(f"- tools_planned: {metadata.get('tools_planned')}")
    lines.append(f"- tools_executed: {metadata.get('tools_executed')}")
    return "\n".join(lines)


async def initial_intent_planning_node(state: LangGraphAnalysisState) -> LangGraphAnalysisState:
    next_state = dict(state)
    metadata = dict(next_state.get("metadata") or {})
    question = str(next_state.get("question") or "").strip()
    context_prompt = str(next_state.get("context_prompt") or "").strip()
    deps = next_state.get("deps")
    capabilities = _build_planner_capabilities(deps)

    try:
        model = get_langchain_chat_model()
    except Exception:
        model = None

    planner_debug = await plan_user_intent_debug(
        question=question,
        context_hint=context_prompt,
        model=model,
        capabilities=capabilities,
    )
    plan_payload = dict(planner_debug.get("normalized_plan") or {})
    next_state["intent_plan"] = plan_payload
    planner_decision = {
        "intent": plan_payload.get("intent"),
        "needs_retrieval": bool(plan_payload.get("needs_retrieval")),
        "direct_answer_allowed": bool(plan_payload.get("direct_answer_allowed")),
        "allow_external_sources": bool(plan_payload.get("allow_external_sources")),
        "planned_tools": [
            step.get("tool")
            for step in list(plan_payload.get("retrieval_steps") or [])
            if isinstance(step, dict) and step.get("tool")
        ],
        "evidence_policy": plan_payload.get("evidence_policy"),
        "reason": plan_payload.get("reason"),
        "warnings": list(plan_payload.get("warnings") or []),
    }
    metadata["planner_decision"] = planner_decision
    metadata["intent"] = plan_payload.get("intent")
    metadata["direct_answer_allowed"] = bool(plan_payload.get("direct_answer_allowed"))
    metadata["retrieval_skipped_by_planner"] = bool(not plan_payload.get("needs_retrieval"))
    metadata["retrieval_skip_reason"] = str(plan_payload.get("reason") or "planner_decided_direct_answer")
    metadata["fallback_used"] = bool(planner_debug.get("fallback_used"))
    metadata["fallback_reason"] = str(planner_debug.get("fallback_reason") or "")
    metadata["fallback_decision"] = str(planner_debug.get("fallback_decision") or "")
    metadata["raw_model_content_preview"] = str(planner_debug.get("raw_model_content_preview") or "")
    metadata["source_requirements"] = dict(plan_payload.get("source_requirements") or {})
    metadata["answer_policy"] = dict(plan_payload.get("answer_policy") or {})
    metadata["intent_plan"] = plan_payload
    next_state["metadata"] = metadata
    return next_state


def route_after_initial_intent(state: LangGraphAnalysisState) -> str:
    raw_plan = state.get("intent_plan") or {}
    if isinstance(raw_plan, dict) and bool(raw_plan.get("needs_retrieval")):
        return "inspect_documents"
    return "generate_analysis"


def _doc_to_dict(doc: Any) -> Dict[str, Any]:
    if hasattr(doc, "model_dump"):
        payload = doc.model_dump()
    elif isinstance(doc, dict):
        payload = dict(doc)
    else:
        payload = {
            "id": getattr(doc, "id", None),
            "title": getattr(doc, "title", None),
            "source": getattr(doc, "source", None),
            "chunk_count": getattr(doc, "chunk_count", None),
            "created_at": (
                getattr(doc, "created_at", None).isoformat()
                if getattr(doc, "created_at", None) is not None
                and hasattr(getattr(doc, "created_at"), "isoformat")
                else getattr(doc, "created_at", None)
            ),
        }
    created_at = payload.get("created_at")
    if created_at is not None and hasattr(created_at, "isoformat"):
        payload["created_at"] = created_at.isoformat()
    updated_at = payload.get("updated_at")
    if updated_at is not None and hasattr(updated_at, "isoformat"):
        payload["updated_at"] = updated_at.isoformat()
    return payload


_SCOPE_POLICIES = {"strict_target", "prefer_target", "broad_kb", "external_allowed"}


def build_answer_scope_prompt(question: str, documents: List[Dict[str, Any]]) -> str:
    doc_lines: List[str] = []
    for d in documents[:20]:
        doc_lines.append(
            f"- id={d.get('id')} | title={d.get('title')} | source={d.get('source')}"
        )
    doc_block = "\n".join(doc_lines) if doc_lines else "- none"
    return (
        "You are an answer-scope resolver for a paper-reading assistant.\n"
        "Infer which document scope the user expects, from the question and document list.\n"
        "Do not answer the question. Do not use fixed keyword tricks.\n"
        "Choose one scope_policy from: strict_target, prefer_target, broad_kb, external_allowed.\n"
        "If user intent points to one specific uploaded paper/file/title, choose strict_target or prefer_target.\n"
        "If user asks for multi-paper comparison or broad topic synthesis, choose broad_kb.\n"
        "If user asks for latest papers/related work/outside sources, choose external_allowed.\n"
        "If uncertain, prefer prefer_target or broad_kb (avoid over-strict).\n"
        "Return strict JSON only:\n"
        '{"scope_policy":"strict_target|prefer_target|broad_kb|external_allowed",'
        '"target_documents":[{"document_id":"...","title":"...","confidence":0.0,"match_reason":"..."}],'
        '"allow_supplemental":true,"scope_reason":"...","answer_instruction":"..."}\n\n'
        f"User question:\n{question}\n\n"
        f"Documents:\n{doc_block}\n"
    )


def parse_answer_scope(raw_text: str, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
    fallback = {
        "scope_policy": "broad_kb",
        "target_documents": [],
        "allow_supplemental": True,
        "scope_resolver_used": False,
        "scope_reason": "",
        "answer_instruction": "",
    }
    text = (raw_text or "").strip()
    if not text:
        return fallback
    match = re.search(r"\{[\s\S]*\}", text)
    payload = match.group(0) if match else text
    try:
        data = json.loads(payload)
    except Exception:
        return fallback
    if not isinstance(data, dict):
        return fallback

    policy = str(data.get("scope_policy") or "broad_kb").strip()
    if policy not in _SCOPE_POLICIES:
        policy = "broad_kb"

    docs_by_id = {str(d.get("id")): d for d in documents if d.get("id") is not None}
    parsed_targets: List[Dict[str, Any]] = []
    raw_targets = data.get("target_documents")
    if isinstance(raw_targets, list):
        for item in raw_targets:
            if not isinstance(item, dict):
                continue
            did = str(item.get("document_id") or "").strip()
            if not did or did not in docs_by_id:
                continue
            conf_raw = item.get("confidence", 0.0)
            try:
                conf = float(conf_raw)
            except Exception:
                conf = 0.0
            conf = max(0.0, min(1.0, conf))
            parsed_targets.append(
                {
                    "document_id": did,
                    "title": str(item.get("title") or docs_by_id[did].get("title") or "").strip(),
                    "confidence": conf,
                    "match_reason": str(item.get("match_reason") or "").strip(),
                }
            )

    allow_supp = data.get("allow_supplemental")
    if isinstance(allow_supp, bool):
        allow_supplemental = allow_supp
    elif isinstance(allow_supp, str):
        value = allow_supp.strip().lower()
        if value in {"true", "1", "yes"}:
            allow_supplemental = True
        elif value in {"false", "0", "no"}:
            allow_supplemental = False
        else:
            allow_supplemental = True
    else:
        allow_supplemental = True
    if policy == "strict_target":
        allow_supplemental = False

    return {
        "scope_policy": policy,
        "target_documents": parsed_targets,
        "allow_supplemental": allow_supplemental,
        "scope_resolver_used": True,
        "scope_reason": str(data.get("scope_reason") or "").strip(),
        "answer_instruction": str(data.get("answer_instruction") or "").strip(),
    }


def _extract_explicit_target_documents(question: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    text = str(question or "")
    ids = re.findall(
        r"(?:目标文档 ID|文档 ID|document_id)[:：]\s*([A-Za-z0-9_-]{3,80})",
        text,
        flags=re.IGNORECASE,
    )
    if not ids:
        return []

    docs_by_id = {
        str(d.get("id")): d
        for d in documents
        if d.get("id") is not None
    }

    targets: List[Dict[str, Any]] = []
    seen = set()
    for did in ids:
        did = str(did).strip()
        if not did or did in seen:
            continue
        seen.add(did)
        doc = docs_by_id.get(did, {})
        targets.append(
            {
                "document_id": did,
                "title": str(doc.get("title") or "").strip(),
                "confidence": 1.0,
                "match_reason": "explicit document_id in user prompt",
            }
        )
    return targets


async def resolve_answer_scope_node(state: LangGraphAnalysisState) -> LangGraphAnalysisState:
    next_state = dict(state)
    question = str(next_state.get("question") or "").strip()
    documents = list(next_state.get("documents") or [])
    fallback = parse_answer_scope("", documents)
    explicit_targets = _extract_explicit_target_documents(question, documents)
    if explicit_targets:
        parsed = {
            "scope_policy": "strict_target",
            "target_documents": explicit_targets,
            "allow_supplemental": False,
            "scope_resolver_used": True,
            "scope_reason": "User prompt contains explicit target document ID.",
            "answer_instruction": "Answer using the explicitly selected target document(s).",
        }
    else:
        try:
            model = get_langchain_chat_model()
            prompt = build_answer_scope_prompt(question, documents)
            response = await model.ainvoke(
                [
                    {"role": "system", "content": "Return strict JSON only."},
                    {"role": "user", "content": prompt},
                ]
            )
            parsed = parse_answer_scope(_extract_response_text(response), documents)
        except Exception as exc:
            _append_warning(next_state, f"answer scope resolver failed: {exc}")
            parsed = fallback

    next_state["answer_scope"] = parsed
    next_state["scope_policy"] = str(parsed.get("scope_policy") or "broad_kb")
    next_state["target_documents"] = list(parsed.get("target_documents") or [])
    next_state["allow_supplemental"] = bool(parsed.get("allow_supplemental", True))
    next_state["scope_resolver_used"] = bool(parsed.get("scope_resolver_used", False))

    metadata = dict(next_state.get("metadata") or {})
    metadata["scope_policy"] = next_state["scope_policy"]
    metadata["target_documents"] = next_state["target_documents"]
    metadata["allow_supplemental"] = next_state["allow_supplemental"]
    metadata["scope_resolver_used"] = next_state["scope_resolver_used"]
    metadata["scope_reason"] = str(parsed.get("scope_reason") or "")
    next_state["metadata"] = metadata
    return next_state


async def inspect_documents_node(state: LangGraphAnalysisState) -> LangGraphAnalysisState:
    next_state = dict(state)
    await _emit_progress(
        next_state,
        "正在定位相关论文和章节范围...",
        phase="document_inspection",
    )
    try:
        documents = await list_documents_tool(DocumentListInput(limit=5, offset=0))
        next_state["documents"] = [_doc_to_dict(doc) for doc in documents]
        _append_tool_call(next_state, "list_documents", {"limit": 5, "offset": 0})
    except Exception as exc:
        next_state["documents"] = []
        _append_warning(next_state, f"list_documents failed: {exc}")
    return next_state


def _find_tool_by_name(tools: List[Any], name: str) -> Optional[Any]:
    for tool in tools:
        if getattr(tool, "name", "") == name:
            return tool
    return None


def _dedupe_key(hit: Dict[str, Any]) -> Tuple[str, str, str]:
    chunk_id = str(hit.get("chunk_id") or "").strip()
    document_id = str(hit.get("document_id") or "").strip()
    content_prefix = str(hit.get("content") or "").strip()[:120]
    return chunk_id, document_id, content_prefix


def dedupe_retrieval_hits(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    deduped: List[Dict[str, Any]] = []
    for hit in hits:
        key = _dedupe_key(hit)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(hit)
    return deduped


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).lower()


def _extract_question_candidates(question: str) -> List[str]:
    q = str(question or "").strip()
    candidates: List[str] = []
    for m in re.findall(r"([A-Za-z0-9._\-]+\.(?:pdf|docx?|txt))", q, flags=re.IGNORECASE):
        candidates.append(m.strip())
    for m in re.findall(r"[\"'“”‘’《》「」『』](.{3,120}?)[\"'“”‘’《》「」『』]", q):
        s = m.strip()
        if s:
            candidates.append(s)
    return candidates


def _match_target_document(question: str, documents: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    q_norm = _normalize_text(question)
    if not q_norm or not documents:
        return None

    candidate_terms = _extract_question_candidates(question)
    best_doc = None
    best_score = 0
    for doc in documents:
        title = str(doc.get("title") or "")
        source = str(doc.get("source") or "")
        did = str(doc.get("id") or "")
        if not did:
            continue
        title_norm = _normalize_text(title)
        source_norm = _normalize_text(source)
        score = 0
        if title_norm and title_norm in q_norm:
            score = max(score, 4)
        if source_norm and source_norm in q_norm:
            score = max(score, 4)
        source_base = _normalize_text(source.split("/")[-1].split("\\")[-1])
        if source_base and source_base in q_norm:
            score = max(score, 5)
        title_no_ext = re.sub(r"\.(pdf|docx?|txt)$", "", title_norm).strip()
        if title_no_ext and len(title_no_ext) >= 4 and title_no_ext in q_norm:
            score = max(score, 3)
        for term in candidate_terms:
            t = _normalize_text(term)
            if not t:
                continue
            if t in title_norm or t in source_norm or t in source_base:
                score = max(score, 5)
        if score > best_score:
            best_score = score
            best_doc = doc
    if best_doc is None or best_score <= 0:
        return None
    return best_doc


def _top_score_of(hits: List[Dict[str, Any]]) -> Optional[float]:
    scores = [float(h["score"]) for h in hits if isinstance(h.get("score"), (int, float))]
    return max(scores) if scores else None


def _prioritize_target_document_hits(
    hits: List[Dict[str, Any]],
    target_document_id: str,
    min_results: int,
    strong_single_score: float,
) -> Tuple[List[Dict[str, Any]], bool]:
    if not target_document_id:
        return hits, False

    target_hits: List[Dict[str, Any]] = []
    other_hits: List[Dict[str, Any]] = []
    for hit in hits:
        if str(hit.get("document_id") or "") == target_document_id:
            target_hits.append(hit)
        else:
            other_hits.append(hit)

    for hit in target_hits:
        meta = dict(hit.get("metadata") or {})
        meta["reference_role"] = "primary_target"
        hit["metadata"] = meta
    for hit in other_hits:
        meta = dict(hit.get("metadata") or {})
        meta["reference_role"] = "supplemental"
        hit["metadata"] = meta

    target_top = _top_score_of(target_hits)
    target_enough = bool(
        target_hits
        and (len(target_hits) >= min_results or (target_top is not None and target_top >= strong_single_score))
    )
    if target_enough:
        return target_hits, False
    return target_hits + other_hits, bool(other_hits)


def _prioritize_sources_by_target(
    sources: List[EvidenceSource],
    target_document_id: str,
) -> List[EvidenceSource]:
    if not target_document_id:
        return list(sources or [])
    matched = [s for s in (sources or []) if str(getattr(s, "document_id", "") or "") == target_document_id]
    others = [s for s in (sources or []) if str(getattr(s, "document_id", "") or "") != target_document_id]
    for s in matched:
        s.metadata = {"reference_role": "primary_target", **(s.metadata or {})}
    for s in others:
        s.metadata = {"reference_role": "supplemental", **(s.metadata or {})}
    return matched + others


def _extract_target_ids(target_documents: List[Dict[str, Any]]) -> List[str]:
    return [str(d.get("document_id") or "").strip() for d in (target_documents or []) if str(d.get("document_id") or "").strip()]


def _apply_scope_policy_to_hits(
    hits: List[Dict[str, Any]],
    scope_policy: str,
    target_ids: List[str],
    allow_supplemental: bool,
) -> Tuple[List[Dict[str, Any]], bool]:
    if not hits:
        return [], False
    if scope_policy not in {"strict_target", "prefer_target"} or not target_ids:
        return hits, False

    target_set = set(target_ids)
    target_hits: List[Dict[str, Any]] = []
    other_hits: List[Dict[str, Any]] = []
    for h in hits:
        if str(h.get("document_id") or "") in target_set:
            target_hits.append(h)
        else:
            other_hits.append(h)

    for h in target_hits:
        m = dict(h.get("metadata") or {})
        m["reference_role"] = "primary_target"
        h["metadata"] = m
    for h in other_hits:
        m = dict(h.get("metadata") or {})
        m["reference_role"] = "supplemental"
        h["metadata"] = m

    if scope_policy == "strict_target":
        return target_hits, False

    target_top = _top_score_of(target_hits)
    target_enough = bool(
        target_hits and (len(target_hits) >= _env_int("LANGGRAPH_TARGET_DOC_MIN_RESULTS", 2) or (target_top is not None and target_top >= _env_float("LANGGRAPH_TARGET_DOC_STRONG_SINGLE_SCORE", 0.55)))
    )
    if target_enough or not allow_supplemental:
        return target_hits, False
    return target_hits + other_hits, bool(other_hits)


def _apply_scope_policy_to_sources(
    sources: List[EvidenceSource],
    scope_policy: str,
    target_ids: List[str],
    allow_supplemental: bool,
    supplemental_used: bool,
) -> List[EvidenceSource]:
    if scope_policy not in {"strict_target", "prefer_target"} or not target_ids:
        return list(sources or [])
    target_set = set(target_ids)
    matched = [s for s in (sources or []) if str(getattr(s, "document_id", "") or "") in target_set]
    others = [s for s in (sources or []) if str(getattr(s, "document_id", "") or "") not in target_set]
    for s in matched:
        s.metadata = {"reference_role": "primary_target", **(s.metadata or {})}
    for s in others:
        s.metadata = {"reference_role": "supplemental", **(s.metadata or {})}
    if scope_policy == "strict_target":
        return matched
    if supplemental_used and allow_supplemental:
        return matched + others
    return matched


def _extract_top_score(results: List[Dict[str, Any]]) -> Optional[float]:
    scores: List[float] = []
    for hit in results:
        score = hit.get("score")
        if isinstance(score, (int, float)):
            scores.append(float(score))
    if not scores:
        return None
    return max(scores)


def build_retrieval_evaluation_prompt(
    question: str,
    results: List[Dict[str, Any]],
    documents: List[Dict[str, Any]],
    attempts: List[Dict[str, Any]],
) -> str:
    doc_summary = _summarize_documents(documents)
    attempt_summary = _build_retrieval_attempts_summary(attempts, [])
    hit_lines: List[str] = []
    for idx, hit in enumerate(results[:8], start=1):
        title = str(hit.get("document_title") or "Untitled").strip()
        score = hit.get("score")
        snippet = str(hit.get("content") or "").strip().replace("\n", " ")
        snippet = snippet[:320] + ("..." if len(snippet) > 320 else "")
        hit_lines.append(f"{idx}. document_title={title} | score={score} | snippet={snippet}")
    hit_block = "\n".join(hit_lines) if hit_lines else "None"
    return (
        "You are a retrieval coverage checker.\n"
        "Given the user question and retrieved snippets, decide whether another retrieval attempt is likely to add useful missing evidence.\n"
        "Do not answer the user question.\n"
        "If the snippets provide enough material for a useful answer, set answerable=true.\n"
        "For synthesis, comparison, pros/cons, method analysis, and multi-paper questions, snippets do not need to contain a complete ready-made answer.\n"
        "It is enough if they provide useful evidence that can be synthesized.\n"
        "Use missing_aspects only for important missing evidence.\n"
        "Use needs_retry=true only when another focused local retrieval is likely to improve the answer.\n"
        "If answerable=false, missing_aspects should extract concrete entities/scenarios/terms/constraints from the user question whenever possible.\n"
        "If answerable=false, provide suggested_query for the next local knowledge-base retrieval and preserve concrete keywords from the user question.\n"
        "Do not invent paper-specific facts that are not supported by snippets.\n"
        "Prefer compact keyword-style query, e.g., 'Hybrid-RRT 火星通信延迟 低重力 外太空部署 失败案例' or English equivalent.\n"
        "Return strict JSON only with schema:\n"
        '{"answerable": bool, "confidence": float, "needs_retry": bool, "reason": str, '
        '"missing_aspects": [str], "suggested_query": str, "supporting_hit_indices": [int]}\n\n'
        f"User question:\n{question}\n\n"
        f"Document summary:\n{doc_summary}\n\n"
        f"Retrieval attempt summary:\n{attempt_summary}\n\n"
        f"Retrieved snippets:\n{hit_block}\n"
    )


def parse_retrieval_evaluation(raw_text: str) -> Optional[Dict[str, Any]]:
    text = (raw_text or "").strip()
    if not text:
        return None
    match = re.search(r"\{[\s\S]*\}", text)
    payload = match.group(0) if match else text
    try:
        data = json.loads(payload)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None

    def _to_bool(v: Any, default: bool = False) -> bool:
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            s = v.strip().lower()
            if s in {"true", "1", "yes"}:
                return True
            if s in {"false", "0", "no"}:
                return False
        return default

    confidence_raw = data.get("confidence", 0.0)
    try:
        confidence = float(confidence_raw)
    except Exception:
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    missing_aspects_raw = data.get("missing_aspects", [])
    missing_aspects: List[str] = []
    if isinstance(missing_aspects_raw, list):
        for item in missing_aspects_raw:
            if isinstance(item, str):
                v = item.strip()
                if v:
                    missing_aspects.append(v)

    indices_raw = data.get("supporting_hit_indices", [])
    supporting_hit_indices: List[int] = []
    if isinstance(indices_raw, list):
        for item in indices_raw:
            try:
                supporting_hit_indices.append(int(item))
            except Exception:
                continue

    return {
        "answerable": _to_bool(data.get("answerable"), False),
        "confidence": confidence,
        "needs_retry": _to_bool(data.get("needs_retry"), False),
        "reason": str(data.get("reason") or "").strip(),
        "missing_aspects": missing_aspects,
        "suggested_query": str(data.get("suggested_query") or "").strip(),
        "supporting_hit_indices": supporting_hit_indices,
    }


def _is_generic_suggested_query(text: str) -> bool:
    q = (text or "").strip().lower()
    if not q:
        return True
    generic_phrases = [
        "specific content",
        "details from",
        "related to the user's question",
        "related to the user question",
        "direct answer",
        "user's question",
        "the paper",
    ]
    return any(p in q for p in generic_phrases)


async def evaluate_retrieval_with_llm(state: LangGraphAnalysisState) -> Optional[Dict[str, Any]]:
    results = list(state.get("retrieval_results") or [])
    if not results:
        return None
    question = str(state.get("question") or "").strip()
    documents = list(state.get("documents") or [])
    attempts = list(state.get("retrieval_attempts") or [])
    prompt = build_retrieval_evaluation_prompt(
        question=question,
        results=results,
        documents=documents,
        attempts=attempts,
    )
    try:
        model = get_langchain_chat_model()
        response = await model.ainvoke(
            [
                {"role": "system", "content": "Return strict JSON only."},
                {"role": "user", "content": prompt},
            ]
        )
        parsed = parse_retrieval_evaluation(_extract_response_text(response))
        if parsed is None:
            _append_warning(state, "retrieval evaluator parse failed; fallback to rule grading.")
        return parsed
    except Exception as exc:
        _append_warning(state, f"retrieval evaluator failed: {exc}")
        return None


def grade_retrieval_quality(
    results: List[Dict[str, Any]],
    min_results: int,
    min_top_score: float,
    single_hit_strong_score: float,
) -> Tuple[bool, str, Optional[float]]:
    if not results:
        return False, "no_results", None

    result_count = len(results)
    top_score = _extract_top_score(results)
    if top_score is not None:
        if result_count == 1 and top_score >= single_hit_strong_score:
            return True, "single_strong_hit", top_score
        if top_score < min_top_score:
            return False, f"low_top_score<{min_top_score}", top_score
        if result_count < min_results and top_score < single_hit_strong_score:
            return False, f"insufficient_results<{min_results}", top_score
        return True, "sufficient_scored_hits", top_score

    if result_count >= min_results:
        return True, "sufficient_unscored_hits", None
    return False, f"insufficient_unscored_results<{min_results}", None


async def local_retrieval_node(state: LangGraphAnalysisState) -> LangGraphAnalysisState:
    next_state = dict(state)
    deps = next_state.get("deps")
    question = str(next_state.get("question") or "").strip()
    attempt_count = int(next_state.get("retrieval_attempt_count") or 0) + 1
    next_state["retrieval_attempt_count"] = attempt_count
    metadata = dict(next_state.get("metadata") or {})

    if attempt_count <= 1:
        query = question
    else:
        query = str(next_state.get("current_query") or question).strip()
    next_state["current_query"] = query

    limit = 3 if attempt_count == 1 else 5
    await _emit_progress(
        next_state,
        "正在理解问题，并判断需要哪些论文证据...",
        phase="planning",
    )

    existing_results = list(next_state.get("retrieval_results") or [])
    attempts = list(next_state.get("retrieval_attempts") or [])

    if not deps or not query:
        if not query:
            _append_warning(next_state, "Empty query; skip retrieval.")
        attempt = {
            "attempt": attempt_count,
            "query": query,
            "limit": limit,
            "result_count": 0,
            "top_score": None,
        }
        attempts.append(attempt)
        next_state["retrieval_attempts"] = attempts
        next_state["retrieval_results"] = existing_results
        return next_state

    try:
        tools = build_langchain_tools(deps)
        capabilities = _build_planner_capabilities(deps)
        retry_plan_used = False
        planner_used = False
        planner_fallback_used = False
        fallback_reason = ""
        fallback_decision = ""
        planner_debug: Dict[str, Any] = {}

        try:
            try:
                planner_model = get_langchain_chat_model()
            except Exception:
                planner_model = None
            planner_debug = await plan_user_intent_debug(
                question=question,
                context_hint=str(next_state.get("context_prompt") or ""),
                model=planner_model,
                capabilities=capabilities,
            )
            planner_used = True
            plan = IntentPlan.model_validate(planner_debug.get("normalized_plan") or {})
        except Exception as exc:
            planner_fallback_used = True
            fallback_reason = str(exc)
            fallback_decision = "local_retrieval_fallback"
            plan = build_fallback_intent_plan(
                question=query or question,
                capabilities=capabilities,
                reason=fallback_reason or "planner_failed",
            )

        if plan.needs_retrieval and not list(plan.retrieval_steps or []):
            planner_fallback_used = True
            fallback_reason = fallback_reason or "planner returned retrieval intent without retrieval_steps"
            fallback_decision = "local_retrieval_fallback"
            plan = build_fallback_intent_plan(
                question=query or question,
                capabilities=capabilities,
                reason=fallback_reason,
            )

        plan_payload = plan.model_dump()
        next_state["intent_plan"] = plan_payload
        planner_decision = {
            "intent": plan.intent,
            "needs_retrieval": bool(plan.needs_retrieval),
            "direct_answer_allowed": bool(plan.direct_answer_allowed),
            "allow_external_sources": bool(plan.allow_external_sources),
            "planned_tools": [step.tool for step in plan.retrieval_steps],
            "evidence_policy": plan.evidence_policy,
            "reason": plan.reason,
            "warnings": list(plan.warnings or []),
        }
        metadata["planner_used"] = planner_used
        metadata["intent_planner_used"] = planner_used
        metadata["planner_fallback_used"] = planner_fallback_used
        metadata["retry_plan_used"] = retry_plan_used
        metadata["intent_plan"] = plan_payload
        metadata["planner_decision"] = planner_decision
        metadata["planner_capabilities"] = capabilities.model_dump()
        metadata["available_tools"] = capabilities.available_tools()
        metadata["intent"] = plan.intent
        metadata["direct_answer_allowed"] = bool(plan.direct_answer_allowed)
        metadata["tools_planned"] = [step.model_dump() for step in plan.retrieval_steps]
        metadata["planned_retrieval_steps"] = [step.model_dump() for step in plan.retrieval_steps]
        metadata["fallback_used"] = bool(planner_debug.get("fallback_used")) or planner_fallback_used
        metadata["fallback_reason"] = fallback_reason or str(planner_debug.get("fallback_reason") or "")
        metadata["fallback_decision"] = fallback_decision or str(planner_debug.get("fallback_decision") or "")
        metadata["raw_model_content_preview"] = str(planner_debug.get("raw_model_content_preview") or "")

        if not bool(plan.needs_retrieval):
            metadata["retrieval_skipped_by_planner"] = True
            metadata["retrieval_skip_reason"] = str(plan.reason or "planner_decided_direct_answer")
            metadata["tools_executed"] = []
            metadata["filtered_unavailable_tools"] = []
            metadata["source_count"] = 0
            metadata["sources_count"] = 0
            next_state["metadata"] = metadata
            next_state["sources"] = []
            next_state["retrieval_results"] = existing_results
            attempts.append(
                {
                    "attempt": attempt_count,
                    "query": query,
                    "limit": limit,
                    "result_count": 0,
                    "top_score": None,
                    "skipped_by_planner": True,
                    "skip_reason": metadata["retrieval_skip_reason"],
                }
            )
            next_state["retrieval_attempts"] = attempts
            await _emit_progress(
                next_state,
                "正在检查证据是否足够回答问题...",
                phase="retrieval",
            )
            await _emit_progress(
                next_state,
                "正在基于检索到的依据组织回答...",
                phase="generation",
            )
            return next_state

        retrieval_query_text = f"{question} {query}".lower()
        artifact_cues = (
            "table",
            "figure",
            "algorithm",
            "pseudo",
            "表格",
            "图示",
            "图",
            "算法",
            "伪代码",
            "指标",
            "实验",
        )
        if any(cue in retrieval_query_text for cue in artifact_cues):
            await _emit_progress(
                next_state,
                "正在补充查找相关表格、图示或算法片段...",
                phase="retrieval",
            )
        else:
            await _emit_progress(
                next_state,
                "正在检索相关章节内容...",
                phase="retrieval",
            )
        exec_out = await execute_intent_plan_steps(
            plan=plan,
            tools=tools,
            fallback_query=query,
            capabilities=capabilities,
        )
        round_results = list(exec_out.get("results") or [])
        tools_planned = list(exec_out.get("planned_steps") or [step.model_dump() for step in plan.retrieval_steps])
        tools_executed_raw = list(exec_out.get("tools_executed") or [])
        filtered_unavailable_tools = list(exec_out.get("filtered_unavailable_tools") or [])
        for warning_text in list(exec_out.get("warnings") or []):
            _append_warning(next_state, str(warning_text))

        tools_executed: List[str] = []
        for item in tools_executed_raw:
            if isinstance(item, dict):
                tool_name = str(item.get("tool") or "").strip()
                if tool_name:
                    tools_executed.append(tool_name)
                    _append_tool_call(next_state, tool_name, dict(item.get("args") or {}))
            elif isinstance(item, str) and item.strip():
                tools_executed.append(item.strip())

        metadata["tools_planned"] = tools_planned
        metadata["tools_executed"] = tools_executed
        metadata["filtered_unavailable_tools"] = filtered_unavailable_tools
        metadata["retrieval_skipped_by_planner"] = False
        metadata["retrieval_skip_reason"] = ""
        metadata["retrieval_summary_for_planner"] = summarize_hits_for_planner(round_results)

        scope_policy = str(next_state.get("scope_policy") or "broad_kb")
        target_ids = _extract_target_ids(list(next_state.get("target_documents") or []))
        allow_supplemental = bool(next_state.get("allow_supplemental", True))
        prioritized_round_results, used_supplemental = _apply_scope_policy_to_hits(
            round_results,
            scope_policy=scope_policy,
            target_ids=target_ids,
            allow_supplemental=allow_supplemental,
        )
        merged = dedupe_retrieval_hits(existing_results + prioritized_round_results)
        top_score = _extract_top_score(prioritized_round_results)
        attempt = {
            "attempt": attempt_count,
            "query": query,
            "limit": limit,
            "result_count": len(prioritized_round_results),
            "top_score": top_score,
        }
        if target_ids:
            attempt["target_document_ids"] = target_ids
            attempt["supplemental_used"] = used_supplemental
        attempts.append(attempt)
        next_state["retrieval_attempts"] = attempts
        next_state["retrieval_results"] = merged
        next_state["sources"] = _apply_scope_policy_to_sources(
            list(getattr(deps, "retrieved_sources", []) or []),
            scope_policy=scope_policy,
            target_ids=target_ids,
            allow_supplemental=allow_supplemental,
            supplemental_used=used_supplemental,
        )
        metadata["source_count"] = len(list(next_state.get("sources") or []))
        metadata["sources_count"] = len(list(next_state.get("sources") or []))
        next_state["metadata"] = metadata
        await _emit_progress(
            next_state,
            "正在检查证据是否足够回答问题...",
            phase="retrieval",
        )
    except Exception as exc:
        _append_warning(next_state, f"local retrieval failed: {exc}")
        next_state["sources"] = list(getattr(deps, "retrieved_sources", []) or [])
        attempts.append(
            {
                "attempt": attempt_count,
                "query": query,
                "limit": limit,
                "result_count": 0,
                "top_score": None,
                "error": str(exc),
            }
        )
        next_state["retrieval_attempts"] = attempts
        next_state["retrieval_results"] = existing_results
    return next_state


async def grade_retrieval_node(state: LangGraphAnalysisState) -> LangGraphAnalysisState:
    next_state = dict(state)
    metadata = dict(next_state.get("metadata") or {})
    planner_decision = metadata.get("planner_decision") or {}
    retrieval_skipped_by_planner = bool(metadata.get("retrieval_skipped_by_planner"))
    direct_answer_allowed = bool(
        metadata.get("direct_answer_allowed")
        or planner_decision.get("direct_answer_allowed")
    )

    if retrieval_skipped_by_planner and direct_answer_allowed:
        next_state["retrieval_sufficient"] = True
        next_state["retrieval_insufficient_reason"] = None
        next_state["retrieval_top_score"] = None
        next_state["skip_rewrite"] = True
        metadata["retrieval_sufficient"] = True
        metadata["retrieval_insufficient_reason"] = "retrieval_skipped_by_planner_direct_answer"
        metadata["retrieval_top_score"] = None
        metadata["retrieval_retry_trigger"] = "planner_direct_answer"
        next_state["metadata"] = metadata
        return next_state

    await _emit_progress(
        next_state,
        "正在检查证据是否足够回答问题...",
        phase="retrieval",
    )
    results = list(next_state.get("retrieval_results") or [])

    min_results = _env_int("LANGGRAPH_RETRIEVAL_MIN_RESULTS", 2)
    min_top_score = _env_float("LANGGRAPH_RETRIEVAL_MIN_TOP_SCORE", 0.25)
    single_hit_strong_score = _env_float("LANGGRAPH_RETRIEVAL_SINGLE_HIT_STRONG_SCORE", 0.55)

    rule_sufficient, rule_reason, top_score = grade_retrieval_quality(
        results=results,
        min_results=min_results,
        min_top_score=min_top_score,
        single_hit_strong_score=single_hit_strong_score,
    )
    eval_conf_threshold = _env_float("LANGGRAPH_RETRIEVAL_EVAL_CONFIDENCE_THRESHOLD", 0.6)
    evaluator_used = False
    evaluator = None
    retrieval_retry_trigger = "rule"
    suggested_rewrite_query = ""
    attempts = int(next_state.get("retrieval_attempt_count") or 0)
    max_attempts = int(next_state.get("max_retrieval_attempts") or _env_int("LANGGRAPH_MAX_RETRIEVAL_ATTEMPTS", 2))
    scope_policy = str(next_state.get("scope_policy") or "broad_kb")
    target_ids = set(_extract_target_ids(list(next_state.get("target_documents") or [])))

    if results:
        evaluator = await evaluate_retrieval_with_llm(next_state)
        evaluator_used = evaluator is not None

    if evaluator_used and evaluator is not None:
        answerable = bool(evaluator.get("answerable"))
        confidence = float(evaluator.get("confidence", 0.0))
        needs_retry = bool(evaluator.get("needs_retry"))
        has_target_hits = bool(
            [
                r
                for r in results
                if str((r or {}).get("document_id") or "").strip() in target_ids
            ]
        )
        if not results:
            sufficient = False
            reason = "no_results"
        elif scope_policy == "strict_target" and target_ids and not has_target_hits:
            sufficient = False
            reason = "strict_target_no_hits"
        elif answerable:
            sufficient = True
            reason = str(evaluator.get("reason") or "")
            if confidence < eval_conf_threshold:
                reason = reason or "partial_evidence"
        elif needs_retry and attempts < max_attempts:
            sufficient = False
            reason = str(evaluator.get("reason") or "") or "evaluator_retry_suggested"
        elif scope_policy != "strict_target" and results:
            sufficient = True
            reason = str(evaluator.get("reason") or "") or "partial_evidence"
        else:
            sufficient = False
            reason = str(evaluator.get("reason") or "") or "evaluator_insufficient"
        next_state["retrieval_evaluation"] = evaluator
        suggested_rewrite_query = str(evaluator.get("suggested_query") or "").strip()
        if suggested_rewrite_query and not _is_generic_suggested_query(suggested_rewrite_query):
            next_state["suggested_rewrite_query"] = suggested_rewrite_query
        else:
            suggested_rewrite_query = ""
        retrieval_retry_trigger = "llm_evaluator_retry" if (not sufficient and bool(evaluator.get("needs_retry"))) else ("llm_evaluator" if not sufficient else "none")
    else:
        if not results:
            sufficient = False
            reason = "no_results"
        elif scope_policy == "strict_target" and target_ids:
            has_target_hits = bool(
                [
                    r
                    for r in results
                    if str((r or {}).get("document_id") or "").strip() in target_ids
                ]
            )
            sufficient = bool(has_target_hits and rule_sufficient)
            reason = rule_reason if sufficient else "strict_target_no_hits"
        else:
            sufficient = True
            reason = rule_reason

    next_state["retrieval_sufficient"] = sufficient
    next_state["retrieval_insufficient_reason"] = None if sufficient else reason
    next_state["retrieval_top_score"] = top_score

    metadata = dict(next_state.get("metadata") or {})
    latest_attempt = (list(next_state.get("retrieval_attempts") or []) or [None])[-1]
    metadata["retrieval_sufficient"] = sufficient
    metadata["retrieval_insufficient_reason"] = next_state.get("retrieval_insufficient_reason")
    metadata["retrieval_top_score"] = top_score
    metadata["retrieval_attempt_count"] = int(next_state.get("retrieval_attempt_count") or 0)
    metadata["latest_retrieval_result_count"] = (
        latest_attempt.get("result_count") if isinstance(latest_attempt, dict) else None
    )
    metadata["latest_retrieval_top_score"] = (
        latest_attempt.get("top_score") if isinstance(latest_attempt, dict) else None
    )
    metadata["retrieval_evaluator_used"] = evaluator_used
    metadata["retrieval_answerable"] = bool((evaluator or {}).get("answerable")) if evaluator else None
    metadata["retrieval_confidence"] = (evaluator or {}).get("confidence") if evaluator else None
    metadata["retrieval_needs_retry"] = bool((evaluator or {}).get("needs_retry")) if evaluator else None
    metadata["retrieval_missing_aspects"] = list((evaluator or {}).get("missing_aspects") or []) if evaluator else []
    metadata["retrieval_evaluator_reason"] = str((evaluator or {}).get("reason") or "") if evaluator else ""
    metadata["retrieval_retry_trigger"] = retrieval_retry_trigger
    metadata["suggested_rewrite_query"] = suggested_rewrite_query
    next_state["metadata"] = metadata

    if not sufficient:
        await _emit_progress(
            next_state,
            "正在补充检索相关证据...",
            phase="retrieval",
        )
    return next_state


def _clean_fallback_query(question: str) -> str:
    cleaned = question
    for token in ["这篇论文", "该论文", "帮我", "请", "分析", "总结", "一下", "详细", "深度", "地", "进行"]:
        cleaned = cleaned.replace(token, " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ，,。！？!?:：;；")
    return cleaned


def _parse_rewrite_queries(raw_text: str) -> List[str]:
    text = (raw_text or "").strip()
    if not text:
        return []
    match = re.search(r"\{[\s\S]*\}", text)
    payload = match.group(0) if match else text
    try:
        data = json.loads(payload)
    except Exception:
        # Text fallback: accept a single concise query line only.
        cleaned = text
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()
        lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
        if len(lines) != 1:
            return []
        line = lines[0]
        line = re.sub(r"^[-*•\d\.\)\s]+", "", line).strip()
        line = re.sub(
            r"^(?:query|queries|检索词|查询|改写查询)\s*[:：]\s*",
            "",
            line,
            flags=re.IGNORECASE,
        ).strip()
        if not line or len(line) > 300:
            return []
        if re.search(r"(因为|所以|理由|解释|说明|建议|I suggest|because|therefore)", line, flags=re.IGNORECASE):
            return []
        return [line]
    queries = data.get("queries") if isinstance(data, dict) else None
    if not isinstance(queries, list):
        return []
    out: List[str] = []
    for q in queries:
        if isinstance(q, str):
            v = q.strip()
            if v:
                out.append(v)
    return out


def _summarize_documents(documents: List[Dict[str, Any]]) -> str:
    if not documents:
        return "当前知识库中未发现文档。"
    lines: List[str] = []
    for idx, doc in enumerate(documents[:5], start=1):
        title = str(doc.get("title") or "Untitled").strip()
        source = str(doc.get("source") or "").strip()
        chunk_count = doc.get("chunk_count")
        lines.append(f"{idx}. {title} | source={source or 'N/A'} | chunk_count={chunk_count}")
    return "\n".join(lines)


def _summarize_retrieval_results(results: List[Dict[str, Any]]) -> str:
    if not results:
        return "当前未检索到可用片段。"
    lines: List[str] = []
    for idx, hit in enumerate(results[:3], start=1):
        title = str(hit.get("document_title") or "Untitled").strip()
        snippet = str(hit.get("content") or "").strip().replace("\n", " ")
        snippet = snippet[:220] + ("..." if len(snippet) > 220 else "")
        score = hit.get("score")
        lines.append(f"{idx}. {title} | score={score} | snippet={snippet}")
    return "\n".join(lines)


def _clean_generation_content(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _truncate_generation_content(text: str, limit: int) -> str:
    value = _clean_generation_content(text)
    if not value:
        return ""
    if len(value) <= limit:
        return value
    return value[:limit].rstrip() + "\n...[truncated]"


def _format_hit_content_for_generation(hit: Dict[str, Any]) -> str:
    metadata = dict(hit.get("metadata") or {})
    content = _clean_generation_content(hit.get("content") or "")
    content_type = str(metadata.get("content_type") or "").strip().lower()
    artifact_type = str(metadata.get("artifact_type") or "").strip().lower()
    caption = _clean_generation_content(metadata.get("caption") or "")

    if content_type != "artifact":
        return _truncate_generation_content(content, 700)

    if artifact_type == "table":
        return _truncate_generation_content(content, 3200)

    if artifact_type == "algorithm":
        return _truncate_generation_content(content, 2400)

    if artifact_type == "figure":
        parts = []
        if caption:
            parts.append(f"Caption: {caption}")
        context_before = _clean_generation_content(metadata.get("context_before") or "")
        if context_before:
            parts.append(f"Context before:\n{context_before}")
        if content:
            parts.append(f"Artifact content:\n{content}")
        context_after = _clean_generation_content(metadata.get("context_after") or "")
        if context_after:
            parts.append(f"Context after:\n{context_after}")
        return _truncate_generation_content("\n\n".join(parts), 1200)

    parts = []
    if caption:
        parts.append(f"Caption: {caption}")
    if content:
        parts.append(content)
    return _truncate_generation_content("\n\n".join(parts), 1800)


def format_retrieval_results_for_generation(results: List[Dict[str, Any]]) -> str:
    if not results:
        return "当前未检索到可用证据。"

    blocks: List[str] = []
    for idx, hit in enumerate(results[:8], start=1):
        metadata = dict(hit.get("metadata") or {})
        document_title = str(hit.get("document_title") or "Untitled").strip()
        score = hit.get("score")
        section_path_text = str(metadata.get("section_path_text") or metadata.get("section_title") or "").strip()
        artifact_type = str(metadata.get("artifact_type") or "").strip()
        caption = str(metadata.get("caption") or "").strip()
        chunk_id = str(hit.get("chunk_id") or "").strip()
        content_type = str(metadata.get("content_type") or "").strip()
        content = _format_hit_content_for_generation(hit)

        lines = [
            f"[Evidence {idx}]",
            f"document_title: {document_title or 'N/A'}",
            f"score: {score}",
            f"section: {section_path_text or 'N/A'}",
            f"artifact_type: {artifact_type or 'N/A'}",
            f"caption: {caption or 'N/A'}",
            f"chunk_id: {chunk_id or 'N/A'}",
            f"content_type: {content_type or 'text'}",
            "content:",
            content or "N/A",
        ]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _extract_response_text(response: Any) -> str:
    content = getattr(response, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                text = item.strip()
                if text:
                    parts.append(text)
            elif isinstance(item, dict):
                text = str(item.get("text") or item.get("content") or "").strip()
                if text:
                    parts.append(text)
        return "\n".join(parts).strip()
    return str(content).strip()


async def rewrite_query_node(state: LangGraphAnalysisState) -> LangGraphAnalysisState:
    next_state = dict(state)
    question = str(next_state.get("question") or "").strip()
    current_query = str(next_state.get("current_query") or question).strip()
    reason = str(next_state.get("retrieval_insufficient_reason") or "insufficient")
    documents = list(next_state.get("documents") or [])
    results = list(next_state.get("retrieval_results") or [])
    rewritten_queries = list(next_state.get("rewritten_queries") or [])
    suggested_rewrite_query = str(next_state.get("suggested_rewrite_query") or "").strip()

    history = {question, current_query, *rewritten_queries}
    if suggested_rewrite_query and suggested_rewrite_query not in history:
        rewritten_queries.append(suggested_rewrite_query)
        next_state["rewritten_queries"] = rewritten_queries
        next_state["current_query"] = suggested_rewrite_query
        _append_tool_call(
            next_state,
            "rewrite_retrieval_query",
            {"from": current_query, "to": suggested_rewrite_query, "reason": f"{reason}|suggested_by_evaluator"},
        )
        return next_state

    prompt = (
        "你是检索查询改写器。仅输出用于本地知识库检索的 query，不要回答问题，不要联网，不要编造事实。\n"
        "不要编造论文标题、作者、年份、DOI；如果信息不在输入中，不要新增。\n"
        "优先保留并重组用户问题中的论文标题、方法名、指标名、任务名等关键词。\n"
        "请返回严格 JSON，格式为：{\"queries\": [\"query1\", \"query2\"]}。\n\n"
        f"用户原始问题：{question}\n"
        f"当前 query：{current_query}\n"
        f"文档列表摘要：\n{_summarize_documents(documents)}\n\n"
        f"已检索片段摘要：\n{_summarize_retrieval_results(results)}\n\n"
        f"检索不足原因：{reason}\n"
    )

    candidate_queries: List[str] = []
    try:
        model = get_langchain_chat_model()
        response = await model.ainvoke(
            [
                {"role": "system", "content": "你只输出 JSON，不输出额外解释。"},
                {"role": "user", "content": prompt},
            ]
        )
        candidate_queries = _parse_rewrite_queries(_extract_response_text(response))
    except Exception as exc:
        _append_warning(next_state, f"rewrite retrieval query failed: {exc}")

    if not candidate_queries:
        if current_query != question:
            candidate_queries = [question]
        else:
            fallback = _clean_fallback_query(question)
            if fallback:
                candidate_queries = [fallback]

    history = {question, current_query, *rewritten_queries}
    selected: Optional[str] = None
    for item in candidate_queries:
        if item not in history:
            selected = item
            break

    if not selected:
        next_state["skip_rewrite"] = True
        next_state["retrieval_sufficient"] = False
        next_state["retrieval_insufficient_reason"] = "rewrite_failed_or_duplicate"
        _append_tool_call(
            next_state,
            "rewrite_retrieval_query",
            {"from": current_query, "to": current_query, "reason": "rewrite_failed_or_duplicate"},
        )
        return next_state

    rewritten_queries.append(selected)
    next_state["rewritten_queries"] = rewritten_queries
    next_state["current_query"] = selected
    _append_tool_call(next_state, "rewrite_retrieval_query", {"from": current_query, "to": selected, "reason": reason})
    return next_state


def route_after_grade(state: LangGraphAnalysisState) -> str:
    if bool(state.get("retrieval_sufficient")):
        return "generate_analysis"
    if bool(state.get("skip_rewrite")):
        return "generate_analysis"
    attempts = int(state.get("retrieval_attempt_count") or 0)
    max_attempts = int(state.get("max_retrieval_attempts") or _env_int("LANGGRAPH_MAX_RETRIEVAL_ATTEMPTS", 2))
    if attempts < max_attempts:
        return "rewrite_query"
    return "generate_analysis"


def _build_retrieval_attempts_summary(attempts: List[Dict[str, Any]], rewritten_queries: List[str]) -> str:
    if not attempts:
        return "无检索尝试记录。"
    lines = []
    for att in attempts:
        lines.append(
            f"- 第{att.get('attempt')}轮: query={att.get('query')!r}, result_count={att.get('result_count')}, top_score={att.get('top_score')}"
        )
    if rewritten_queries:
        lines.append(f"- query rewrite: {rewritten_queries}")
    else:
        lines.append("- query rewrite: 无")
    return "\n".join(lines)


def _build_retrieval_evaluation_summary(evaluation: Optional[Dict[str, Any]]) -> str:
    if not evaluation:
        return "No retrieval-evaluation record."
    return (
        f"answerable={evaluation.get('answerable')}, "
        f"confidence={evaluation.get('confidence')}, "
        f"needs_retry={evaluation.get('needs_retry')}, "
        f"reason={evaluation.get('reason')}, "
        f"missing_aspects={evaluation.get('missing_aspects')}, "
        f"supporting_hit_indices={evaluation.get('supporting_hit_indices')}"
    )


def _humanize_warning(warning: str) -> Optional[str]:
    if "No retrieval evidence found" in warning:
        return "本轮没有可核对的检索片段；未被检索证据支持的结论请谨慎参考。"
    if "strong evidence wording but no sources" in warning:
        return "回答存在较强结论措辞，但未收集到可核对来源，请谨慎参考。"
    return None


async def generate_analysis_node(state: LangGraphAnalysisState) -> LangGraphAnalysisState:
    next_state = dict(state)
    metadata = dict(next_state.get("metadata") or {})
    if bool(metadata.get("retrieval_skipped_by_planner")) and bool(metadata.get("direct_answer_allowed")):
        await _emit_progress(
            next_state,
            "正在基于检索到的依据组织回答...",
            phase="generation",
        )
    else:
        await _emit_progress(
            next_state,
            "正在基于检索到的依据组织回答...",
            phase="generation",
        )
    question = str(next_state.get("question") or "").strip()
    context_prompt = str(next_state.get("context_prompt") or "").strip()
    documents = list(next_state.get("documents") or [])
    retrieval_results = list(next_state.get("retrieval_results") or [])
    warnings = list(next_state.get("warnings") or [])
    retrieval_attempts = list(next_state.get("retrieval_attempts") or [])
    rewritten_queries = list(next_state.get("rewritten_queries") or [])
    retrieval_evaluation = next_state.get("retrieval_evaluation")
    answer_scope = dict(next_state.get("answer_scope") or {})
    intent_plan = dict(metadata.get("intent_plan") or {})
    answer_policy = dict(
        metadata.get("answer_policy")
        or intent_plan.get("answer_policy")
        or {}
    )
    scope_policy = str(next_state.get("scope_policy") or "broad_kb")
    answer_instruction = str(answer_scope.get("answer_instruction") or "")
    runtime_decision_summary = _build_runtime_decision_summary(metadata)

    context_block = ""
    if context_prompt:
        context_block = f"会话上下文（仅用于指代消歧，不作为文档证据）：\n{context_prompt}\n\n"

    prompt = (
        f"{context_block}"
        f"用户问题：\n{question}\n\n"
        f"文档列表（最多5条）：\n{_summarize_documents(documents)}\n\n"
        f"检索证据：\n{format_retrieval_results_for_generation(retrieval_results)}\n\n"
        f"检索尝试摘要：\n{_build_retrieval_attempts_summary(retrieval_attempts, rewritten_queries)}\n\n"
        f"运行决策摘要：\n{runtime_decision_summary}\n\n"
        f"回答范围策略：{scope_policy}\n"
        f"回答范围说明：{answer_instruction or 'N/A'}\n\n"
        f"本轮回答策略：\n{json.dumps(answer_policy, ensure_ascii=False, indent=2) if answer_policy else 'N/A'}\n\n"
        f"当前告警：\n{chr(10).join(warnings) if warnings else '无'}\n\n"
        "请根据用户问题和检索片段自然组织中文回答。优先完成用户当前任务。"
        "可以综合多个片段进行总结、比较和分析。"
        "对片段明确支持的内容直接分析；对片段没有覆盖的具体点，可以自然说明边界。"
        "回答时请区分 planner 主动跳过检索、检索失败、检索不足和已有证据回答。"
        "如果 planner 主动跳过检索并允许 direct answer，请直接回答，不要声称检索失败。"
        "如果执行过检索但证据不足，只在相关结论处说明证据边界。"
        "不要把主动跳过检索说成检索失败。"
        "不要编造论文细节，不要把推断说成论文原文结论。"
        "具体数字、百分比、年份、作者、DOI、venue、论文标题等事实必须来自检索片段或 source metadata。"
        "如果检索证据中包含 artifact table 或 artifact algorithm，应优先依据 content 中的原始表格/算法内容回答；"
        "不得因为 UI snippet 或 caption 不完整而声称表格数据缺失。"
        "不要把证据中的精确数字改写成模糊范围。"
        "算法机制、模块名称、实验结论必须有片段直接支撑；没有支撑时写“当前检索片段未明确说明”。"
        "若属于合理推断，必须写“基于现有片段可推断，但仍需原文确认”。"
        "不要把通用知识伪装成特定论文证据。"
        "建议按三层表达：证据明确支持、基于片段可推断、当前证据不足。"
        "请严格遵守："
        "只能使用 allowed_source_types；不得使用 blocked_source_types；"
        "如果 must_disclose_limitations=true，必须说明能力边界；"
        "不得把一种来源伪装成另一种来源；"
        "如果 mode=direct_answer 或 answer_with_disclosure，不要声称执行了检索；"
        "如果 mode=retrieve_and_answer，回答必须依据 retrieval_results。"
    )
    if scope_policy == "strict_target":
        prompt += (
            "你必须只将 target_documents 作为论文证据。"
            "若目标文档证据不足，直接说明不足。"
            "不要用 supplemental 片段来替代目标论文结论。"
        )

    system_text = (
        SYSTEM_PROMPT
        + "\n\n你正在执行 LangGraph 深度分析流程。请根据用户问题和检索片段完成分析。"
    )

    try:
        model = get_langchain_chat_model()
        response = await model.ainvoke(
            [
                {"role": "system", "content": system_text},
                {"role": "user", "content": prompt},
            ]
        )
        next_state["draft_answer"] = _extract_response_text(response)
    except Exception as exc:
        _append_warning(next_state, f"analysis generation failed: {exc}")
        fallback = "当前分析生成阶段出现异常。我会基于已有检索片段继续完成回答。"
        if not retrieval_results:
            fallback += " 本轮没有可核对的检索片段。"
        next_state["draft_answer"] = clean_legacy_warning_text(fallback)
    return next_state


async def evidence_check_node(state: LangGraphAnalysisState) -> LangGraphAnalysisState:
    next_state = dict(state)
    await _emit_progress(
        next_state,
        "正在核对回答是否超出已有证据...",
        phase="generation",
    )
    sources = list(next_state.get("sources") or [])
    retrieval_results = list(next_state.get("retrieval_results") or [])
    draft_answer = str(next_state.get("draft_answer") or "")
    metadata = dict(next_state.get("metadata") or {})
    planner_decision = dict(metadata.get("planner_decision") or {})
    retrieval_skipped_by_planner = bool(metadata.get("retrieval_skipped_by_planner"))
    direct_answer_allowed = bool(
        metadata.get("direct_answer_allowed")
        or planner_decision.get("direct_answer_allowed")
    )

    if not retrieval_results and not (retrieval_skipped_by_planner and direct_answer_allowed):
        _append_warning(
            next_state,
            "No retrieval evidence found; answer should be treated as general guidance.",
        )

    strong_claim_terms = [
        "根据论文",
        "根据文献",
        "研究表明",
        "evidence shows",
        "studies show",
        "according to",
    ]
    lower_answer = draft_answer.lower()
    if not sources and any(term in draft_answer or term in lower_answer for term in strong_claim_terms):
        _append_warning(
            next_state,
            "Answer contains strong evidence wording but no sources were collected.",
        )

    metadata = dict(next_state.get("metadata") or {})
    metadata["evidence_checked"] = True
    metadata["source_count"] = len(sources)
    metadata["sources_count"] = len(sources)
    metadata["retrieval_result_count"] = len(retrieval_results)
    next_state["metadata"] = metadata
    return next_state


async def finalize_node(state: LangGraphAnalysisState) -> LangGraphAnalysisState:
    next_state = dict(state)
    retrieval_attempts_final = list(next_state.get("retrieval_attempts") or [])
    latest_attempt = retrieval_attempts_final[-1] if retrieval_attempts_final else {}
    retrieval_results = list(next_state.get("retrieval_results") or [])
    rewritten_queries = list(next_state.get("rewritten_queries") or [])
    target_documents = list(next_state.get("target_documents") or [])
    scope_policy = str(next_state.get("scope_policy") or "")
    allow_supplemental = bool(next_state.get("allow_supplemental", True))
    target_document_ids = {
        str(item.get("document_id") or item.get("id") or "").strip()
        for item in target_documents
        if isinstance(item, dict)
    }
    target_document_ids.discard("")
    target_document_hit_count = sum(
        1 for hit in retrieval_results if str((hit or {}).get("document_id") or "").strip() in target_document_ids
    )
    target_document_enough = target_document_hit_count > 0
    await _emit_progress(
        next_state,
        "正在基于检索到的依据组织回答...",
        phase="generation",
    )
    deps = next_state.get("deps")
    warnings = list(next_state.get("warnings") or [])
    answer = str(next_state.get("draft_answer") or "").strip()

    if not answer:
        answer = "当前无法生成有效分析结果。"

    if warnings:
        notes = [text for text in (_humanize_warning(item) for item in warnings[:2]) if text]
        if notes:
            answer = answer + "\n\n注：" + "；".join(notes)

    next_state["final_answer"] = clean_legacy_warning_text(answer)
    scope_policy = scope_policy or "broad_kb"
    target_ids = _extract_target_ids(target_documents)
    supplemental_used = any(bool((a or {}).get("supplemental_used")) for a in retrieval_attempts_final)
    next_state["sources"] = _apply_scope_policy_to_sources(
        list(getattr(deps, "retrieved_sources", []) or []),
        scope_policy=scope_policy,
        target_ids=target_ids,
        allow_supplemental=allow_supplemental,
        supplemental_used=supplemental_used,
    )

    retrieval_attempt_count = int(next_state.get("retrieval_attempt_count") or len(retrieval_attempts_final))
    top_score = next_state.get("retrieval_top_score")
    if top_score is None:
        top_score = _extract_top_score(retrieval_results)

    metadata = dict(next_state.get("metadata") or {})
    metadata["agent_backend"] = "langgraph"
    metadata["workflow"] = "deep_analysis"
    metadata["tool_count"] = len(list(next_state.get("tools_used") or []))
    metadata["retrieval_attempt_count"] = retrieval_attempt_count
    metadata["retrieval_retry_count"] = max(0, retrieval_attempt_count - 1)
    metadata["rewritten_queries"] = rewritten_queries
    metadata["retrieval_sufficient"] = bool(next_state.get("retrieval_sufficient"))
    metadata["retrieval_insufficient_reason"] = next_state.get("retrieval_insufficient_reason")
    metadata["retrieval_top_score"] = top_score
    metadata["retrieval_result_count"] = len(retrieval_results)
    metadata["latest_retrieval_result_count"] = latest_attempt.get("result_count")
    metadata["latest_retrieval_top_score"] = latest_attempt.get("top_score")
    metadata["retrieval_evaluation"] = next_state.get("retrieval_evaluation")
    metadata["suggested_rewrite_query"] = next_state.get("suggested_rewrite_query")
    metadata["scope_policy"] = scope_policy
    metadata["target_documents"] = target_documents
    metadata["allow_supplemental"] = allow_supplemental
    metadata["scope_resolver_used"] = bool(next_state.get("scope_resolver_used", False))
    metadata["scope_reason"] = str((next_state.get("answer_scope") or {}).get("scope_reason") or "")
    metadata["supplemental_reference_used"] = supplemental_used
    metadata["target_document_hit_count"] = target_document_hit_count
    metadata["target_document_enough"] = target_document_enough
    metadata["source_count"] = len(next_state["sources"])
    next_state["metadata"] = metadata
    return next_state

def build_langgraph_workflow():
    graph = StateGraph(LangGraphAnalysisState)
    graph.add_node("inspect_documents", inspect_documents_node)
    graph.add_node("resolve_answer_scope", resolve_answer_scope_node)
    graph.add_node("local_retrieval", local_retrieval_node)
    graph.add_node("grade_retrieval", grade_retrieval_node)
    graph.add_node("rewrite_query", rewrite_query_node)
    graph.add_node("generate_analysis", generate_analysis_node)
    graph.add_node("evidence_check", evidence_check_node)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "inspect_documents")
    graph.add_edge("inspect_documents", "resolve_answer_scope")
    graph.add_edge("resolve_answer_scope", "local_retrieval")
    graph.add_edge("local_retrieval", "grade_retrieval")
    graph.add_conditional_edges(
        "grade_retrieval",
        route_after_grade,
        {"generate_analysis": "generate_analysis", "rewrite_query": "rewrite_query"},
    )
    graph.add_edge("rewrite_query", "local_retrieval")
    graph.add_edge("generate_analysis", "evidence_check")
    graph.add_edge("evidence_check", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile()


async def run_langgraph_analysis(
    question: str,
    deps: AgentDependencies,
    context_prompt: Optional[str] = None,
    progress_callback: Optional[Callable[[str], Awaitable[None]]] = None,
) -> LangGraphAnalysisResult:
    graph = build_langgraph_workflow()
    initial_state: LangGraphAnalysisState = {
        "question": question,
        "context_prompt": context_prompt or "",
        "session_id": deps.session_id,
        "user_id": deps.user_id,
        "deps": deps,
        "documents": [],
        "retrieval_results": [],
        "tools_used": [],
        "sources": [],
        "warnings": [],
        "metadata": {},
        "progress_callback": progress_callback,
        "current_query": question,
        "retrieval_attempt_count": 0,
        "retrieval_attempts": [],
        "rewritten_queries": [],
        "retrieval_sufficient": False,
        "retrieval_insufficient_reason": None,
        "retrieval_top_score": None,
        "max_retrieval_attempts": max(1, _env_int("LANGGRAPH_MAX_RETRIEVAL_ATTEMPTS", 2)),
        "skip_rewrite": False,
        "suggested_rewrite_query": "",
        "retrieval_evaluation": {},
        "target_document_id": "",
        "target_document_title": "",
        "answer_scope": {
            "scope_policy": "broad_kb",
            "target_documents": [],
            "allow_supplemental": True,
            "scope_resolver_used": False,
            "scope_reason": "",
            "answer_instruction": "",
        },
        "scope_policy": "broad_kb",
        "target_documents": [],
        "allow_supplemental": True,
        "scope_resolver_used": False,
    }

    if progress_callback is not None:
        await progress_callback(
            {
                "content": "正在理解问题，并判断需要哪些论文证据...",
                "phase": "planning",
                "user_visible": True,
                "level": "info",
            }
        )
    final_state = await graph.ainvoke(initial_state)
    message = str(final_state.get("final_answer") or final_state.get("draft_answer") or "").strip()
    tools_used = list(final_state.get("tools_used") or [])
    if "sources" in final_state:
        sources = list(final_state.get("sources") or [])
    else:
        sources = list(deps.retrieved_sources)
    metadata = dict(final_state.get("metadata") or {})

    return LangGraphAnalysisResult(
        message=message,
        raw_state=dict(final_state),
        tools_used=tools_used,
        sources=sources,
        metadata=metadata,
    )
