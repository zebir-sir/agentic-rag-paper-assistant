import json
import re
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


IntentType = Literal[
    "direct_answer",
    "local_paper_qa",
    "local_section_qa",
    "local_artifact_qa",
    "multi_paper_compare",
    "external_paper_discovery",
    "web_information",
    "general_technical_explanation",
    "unclear",
]

RetrievalToolName = Literal[
    "none",
    "hybrid_search",
    "vector_search",
    "section_search",
    "artifact_search",
    "openalex_search",
    "web_search",
]

ArtifactType = Literal["table", "figure", "algorithm"]


class RetrievalStep(BaseModel):
    tool: RetrievalToolName
    query: str = ""
    limit: int = Field(default=10, ge=1, le=50)
    search_type: Optional[str] = None
    section_query: Optional[str] = None
    artifact_types: List[ArtifactType] = Field(default_factory=list)
    document_id: Optional[str] = None
    reason: str = ""


class IntentPlan(BaseModel):
    intent: IntentType
    needs_retrieval: bool
    retrieval_steps: List[RetrievalStep] = Field(default_factory=list)
    max_tools: int = Field(default=2, ge=0, le=2)
    allow_external_sources: bool = False
    evidence_policy: str = "answer_with_available_evidence_and_state_uncertainty"
    direct_answer_allowed: bool = False
    rewrite_allowed: bool = True
    reason: str = ""
    warnings: List[str] = Field(default_factory=list)
    source_requirements: Dict[str, Any] = Field(default_factory=dict)
    answer_policy: Dict[str, Any] = Field(default_factory=dict)


class PlannerCapabilities(BaseModel):
    local_search_enabled: bool = True
    vector_search_enabled: bool = True
    hybrid_search_enabled: bool = True
    section_search_enabled: bool = True
    artifact_search_enabled: bool = True
    openalex_search_enabled: bool = False
    web_search_enabled: bool = False
    direct_answer_enabled: bool = True
    max_tools: int = Field(default=2, ge=0, le=2)

    def available_tools(self) -> List[str]:
        tools: List[str] = []
        if self.hybrid_search_enabled:
            tools.append("hybrid_search")
        if self.vector_search_enabled:
            tools.append("vector_search")
        if self.section_search_enabled:
            tools.append("section_search")
        if self.artifact_search_enabled:
            tools.append("artifact_search")
        if self.openalex_search_enabled:
            tools.append("openalex_search")
        if self.web_search_enabled:
            tools.append("web_search")
        return tools


def build_intent_planner_prompt(
    question: str,
    context_hint: str = "",
    capabilities: Optional[PlannerCapabilities] = None,
) -> str:
    caps = capabilities or PlannerCapabilities()
    hint = str(context_hint or "").strip()
    available_tools = caps.available_tools()
    return (
        "You are a lightweight intent planner for a paper-assistant system.\n"
        "Output only JSON with fields matching the IntentPlan schema.\n"
        f"Available tools: {available_tools}\n"
        "You may only plan tools listed in Available tools.\n"
        "Decide whether retrieval is needed.\n"
        "Use minimal necessary retrieval and plan at most 2 tools.\n"
        "If one tool is clearly sufficient, do not add a second tool.\n"
        "Prefer local tools for local paper evidence.\n"
        "section_search is for section-scoped evidence.\n"
        "artifact_search is supplementary for non-prose artifacts (tables, figures, algorithms), not default.\n"
        "openalex_search is for papers outside local corpus, metadata, DOI, authors, year, related work discovery.\n"
        "web_search is for open-web or recent/non-paper information.\n"
        "Use retrieval only when the question depends on document-grounded, paper-grounded, artifact-grounded, external-source, "
        "current-information, or other tool-provided evidence.\n"
        "If the question can be answered naturally without additional evidence or tool calls, choose direct_answer with needs_retrieval=false.\n"
        "Evidence cues (document/section/table/figure/algorithm/citation/OpenAlex/web/latest) are examples, not an exhaustive rule list.\n"
        "Do not retrieve merely because retrieval tools are available.\n"
        "If external tools are unavailable, do not plan them and pick available fallback.\n"
        "If no retrieval is needed, set needs_retrieval=false and retrieval_steps=[].\n"
        "Never execute tools. Only plan.\n"
        f"Question: {question}\n"
        f"Context hint: {hint}\n"
    )


def build_retry_intent_planner_prompt(
    question: str,
    previous_plan: IntentPlan,
    missing_aspects: List[str],
    retrieval_summary: str,
    suggested_query: str = "",
) -> str:
    return (
        "You are a lightweight retrieval re-planner for a paper-assistant system.\n"
        "Output only JSON with fields matching IntentPlan schema.\n"
        "Goal: improve missing evidence coverage from previous retrieval.\n"
        "Plan at most 2 tools and use minimal necessary retrieval.\n"
        "Avoid repeating exactly the same tool+query unless no better option exists.\n"
        "artifact_search is supplementary for non-prose artifacts (tables/figures/algorithms).\n"
        "section_search is for section-focused gaps.\n"
        "Prefer local tools first; use openalex/web only when external sources are truly needed.\n"
        "Never execute tools.\n"
        f"Question: {question}\n"
        f"Previous plan: {previous_plan.model_dump_json()}\n"
        f"Missing aspects: {json.dumps(missing_aspects, ensure_ascii=False)}\n"
        f"Retrieval summary: {retrieval_summary}\n"
        f"Suggested query: {suggested_query}\n"
    )


def _safe_default_plan(question: str) -> IntentPlan:
    q = str(question or "").lower()
    generic_markers = ("what is ", "explain ", "difference between ", "how does ")
    if any(m in q for m in generic_markers) and "paper" not in q and "document" not in q:
        return IntentPlan(
            intent="general_technical_explanation",
            needs_retrieval=False,
            retrieval_steps=[],
            max_tools=0,
            allow_external_sources=False,
            direct_answer_allowed=True,
            reason="Fallback direct technical explanation.",
            warnings=["planner_fallback_used"],
        )
    return IntentPlan(
        intent="local_paper_qa",
        needs_retrieval=True,
        retrieval_steps=[
            RetrievalStep(
                tool="hybrid_search",
                query=question,
                limit=10,
                reason="Fallback local retrieval plan.",
            )
        ],
        max_tools=1,
        allow_external_sources=False,
        direct_answer_allowed=False,
        reason="Fallback safe retrieval plan.",
        warnings=["planner_fallback_used"],
    )


_LOCAL_EVIDENCE_CUES = (
    "paper",
    "papers",
    "document",
    "documents",
    "knowledge base",
    "kb",
    "pdf",
    "section",
    "sections",
    "chapter",
    "appendix",
    "table",
    "figure",
    "algorithm",
    "experiment",
    "citation",
    "reference",
    "uploaded",
    "upload",
    "manuscript",
    "according to the paper",
    "according to the document",
    "论文",
    "文档",
    "知识库",
    "上传",
    "本地",
    "根据论文",
    "根据文档",
    "根据资料",
    "根据上文",
    "pdf",
    "章节",
    "小节",
    "附录",
    "表",
    "图",
    "算法",
    "实验",
    "引用",
    "参考文献",
)

_EXPLICIT_LOCAL_EVIDENCE_CUES = (
    "knowledge base",
    "kb",
    "uploaded",
    "upload",
    "document",
    "documents",
    "according to the paper",
    "according to the document",
    "this paper",
    "that paper",
    "the paper",
    "local",
    "paper in the knowledge base",
    "知识库",
    "本地",
    "上传",
    "文档",
    "根据知识库",
    "根据论文",
    "根据文档",
    "这篇论文",
    "该论文",
    "依据片段",
)

_LOCAL_SECTION_CUES = (
    "section",
    "sections",
    "chapter",
    "appendix",
    "abstract",
    "method",
    "methods",
    "experiment",
    "experiments",
    "results",
    "章节",
    "小节",
    "附录",
    "摘要",
    "方法",
    "实验",
    "结果",
)

_EXTERNAL_INFO_CUES = (
    "latest",
    "recent",
    "current",
    "today",
    "news",
    "web",
    "website",
    "online",
    "internet",
    "search online",
    "browse",
    "source link",
    "doi",
    "author",
    "authors",
    "year",
    "venue",
    "related work",
    "related papers",
    "literature",
    "最新",
    "最近",
    "当前",
    "今天",
    "新闻",
    "网页",
    "网上",
    "联网",
    "查一下",
    "资料来源",
    "链接",
    "doi",
    "作者",
    "年份",
    "期刊",
    "会议",
    "相关论文",
    "相关工作",
    "文献",
)

_ACADEMIC_EXTERNAL_CUES = (
    "openalex",
    "outside the knowledge base",
    "outside kb",
    "find papers",
    "find paper",
    "recommend papers",
    "recommend paper",
    "doi",
    "author",
    "authors",
    "year",
    "venue",
    "related work",
    "related papers",
    "literature",
    "source link",
    "open access",
    "open-access",
    "doi",
    "openalex",
    "知识库外",
    "找论文",
    "推荐论文",
    "作者",
    "年份",
    "期刊",
    "会议",
    "相关论文",
    "相关工作",
    "文献",
    "来源链接",
    "开放获取",
)

_STRICT_LOCAL_EVIDENCE_CUES = (
    "knowledge base",
    "kb",
    "uploaded",
    "upload",
    "pdf",
    "section",
    "sections",
    "appendix",
    "according to the paper",
    "according to the document",
    "知识库",
    "上传",
    "本地",
    "根据论文",
    "根据文档",
    "根据知识库",
    "pdf",
    "章节",
    "小节",
    "附录",
)


def _contains_any_cue(text: str, cues: tuple[str, ...]) -> bool:
    normalized = str(text or "").strip().lower()
    if not normalized:
        return False
    return any(cue in normalized for cue in cues)


def _append_warning_once(warnings: List[str], warning: str) -> None:
    value = str(warning or "").strip()
    if value and value not in warnings:
        warnings.append(value)


_TOOL_SOURCE_TYPE: Dict[str, str] = {
    "hybrid_search": "local_kb",
    "vector_search": "local_kb",
    "section_search": "local_section",
    "artifact_search": "local_artifact",
    "openalex_search": "external_academic",
    "web_search": "general_web",
    "none": "model_knowledge",
}

_ALL_SOURCE_TYPES = {
    "local_kb",
    "local_section",
    "local_artifact",
    "external_academic",
    "general_web",
    "model_knowledge",
}


def _is_source_type_available(source_type: str, capabilities: PlannerCapabilities) -> bool:
    if source_type == "local_kb":
        return bool(capabilities.hybrid_search_enabled or capabilities.vector_search_enabled)
    if source_type == "local_section":
        return bool(capabilities.section_search_enabled)
    if source_type == "local_artifact":
        return bool(capabilities.artifact_search_enabled)
    if source_type == "external_academic":
        return bool(capabilities.openalex_search_enabled)
    if source_type == "general_web":
        return bool(capabilities.web_search_enabled)
    if source_type == "model_knowledge":
        return bool(capabilities.direct_answer_enabled)
    return False


def _step_source_types(steps: List[RetrievalStep]) -> List[str]:
    out: List[str] = []
    for step in steps:
        source_type = _TOOL_SOURCE_TYPE.get(str(step.tool), "local_kb")
        if source_type not in out:
            out.append(source_type)
    return out


def _derive_source_requirements(
    question: str,
    normalized_plan: IntentPlan,
) -> Dict[str, Any]:
    text = str(question or "").strip().lower()
    step_sources = _step_source_types(list(normalized_plan.retrieval_steps or []))
    explicit_local_required = _contains_any_cue(text, _EXPLICIT_LOCAL_EVIDENCE_CUES)
    local_section_required = _contains_any_cue(text, _LOCAL_SECTION_CUES)
    local_scope_required = bool(explicit_local_required or local_section_required)
    external_academic_required = _contains_any_cue(text, _ACADEMIC_EXTERNAL_CUES)
    general_web_required = _contains_any_cue(
        text,
        ("latest", "recent", "current", "today", "news", "web", "online", "internet", "联网", "最新", "网页"),
    )
    citation_required = _contains_any_cue(
        text,
        ("doi", "author", "authors", "year", "venue", "citation", "citations", "作者", "年份", "期刊", "会议", "引用"),
    )
    freshness_required = _contains_any_cue(
        text,
        ("latest", "recent", "current", "today", "news", "最新", "近期", "当前", "今天"),
    )

    required_source_types: List[str] = []
    preferred_source_types: List[str] = []
    forbidden_source_types: List[str] = []

    if local_scope_required:
        required_source_types.append("local_kb")
        if local_section_required:
            preferred_source_types.append("local_section")

    if external_academic_required:
        if "external_academic" not in required_source_types:
            required_source_types.append("external_academic")
        if "external_academic" not in preferred_source_types:
            preferred_source_types.append("external_academic")

    if general_web_required:
        if "general_web" not in required_source_types:
            required_source_types.append("general_web")
        if "general_web" not in preferred_source_types:
            preferred_source_types.append("general_web")

    if not required_source_types:
        for source in step_sources:
            if source not in preferred_source_types:
                preferred_source_types.append(source)

    if local_scope_required and not external_academic_required and not general_web_required:
        forbidden_source_types.extend(["general_web", "external_academic"])

    can_answer_without_retrieval = not bool(required_source_types) and bool(
        normalized_plan.direct_answer_allowed
    )
    return {
        "required_source_types": required_source_types,
        "preferred_source_types": preferred_source_types,
        "forbidden_source_types": forbidden_source_types,
        "freshness_required": freshness_required,
        "citation_required": citation_required,
        "local_scope_required": local_scope_required,
        "local_section_required": local_section_required,
        "external_scope_required": bool(external_academic_required or general_web_required),
        "can_answer_without_retrieval": can_answer_without_retrieval,
    }


def resolve_source_policy(
    question: str,
    normalized_plan: IntentPlan,
    capabilities: PlannerCapabilities,
) -> tuple[Dict[str, Any], Dict[str, Any], IntentPlan]:
    source_requirements = _derive_source_requirements(question, normalized_plan)
    plan = normalized_plan.model_copy(deep=True)
    required = list(source_requirements.get("required_source_types") or [])
    forbidden = set(source_requirements.get("forbidden_source_types") or [])

    unavailable_required = [s for s in required if not _is_source_type_available(s, capabilities)]
    allowed_source_types = [s for s in _ALL_SOURCE_TYPES if _is_source_type_available(s, capabilities)]

    mode = "retrieve_and_answer" if plan.needs_retrieval else "direct_answer"
    must_disclose = False
    answer_boundary = "use_available_sources_only"
    guidance = "Follow allowed/blocked source types and keep source attribution transparent."

    compatible_steps: List[RetrievalStep] = []
    blocked_source_types = set(forbidden)
    for step in list(plan.retrieval_steps or []):
        step_source = _TOOL_SOURCE_TYPE.get(str(step.tool), "local_kb")
        if step_source in forbidden:
            blocked_source_types.add(step_source)
            continue
        if not _is_source_type_available(step_source, capabilities):
            blocked_source_types.add(step_source)
            continue
        compatible_steps.append(step)

    if required:
        if unavailable_required:
            must_disclose = True
            answer_boundary = "required_sources_unavailable"
            _append_warning_once(plan.warnings, "required_source_unavailable")
            _append_warning_once(
                plan.warnings,
                f"unavailable_required_sources:{','.join(unavailable_required)}",
            )
        if compatible_steps:
            plan.retrieval_steps = compatible_steps[: capabilities.max_tools]
            plan.needs_retrieval = True
            mode = "retrieve_and_answer"
            if unavailable_required:
                mode = "answer_with_disclosure"
                if "local_kb" in required and "general_web" in unavailable_required:
                    guidance = (
                        "Answer the local knowledge-base portion with retrieved evidence, "
                        "and explicitly state that latest/general web materials are unavailable."
                    )
        else:
            plan.retrieval_steps = []
            plan.needs_retrieval = False
            if plan.direct_answer_allowed and capabilities.direct_answer_enabled:
                mode = "answer_with_disclosure" if unavailable_required else "direct_answer"
            else:
                mode = "ask_clarification"
            must_disclose = True
            if "general_web" in unavailable_required and "local_kb" not in required:
                guidance = (
                    "Do not substitute local knowledge-base evidence for unavailable web/latest materials; "
                    "answer only with a clear limitation disclosure."
                )
    else:
        # No explicit source requirement: allow either retrieval or direct answer.
        if not plan.retrieval_steps and plan.direct_answer_allowed:
            mode = "direct_answer"
        elif plan.retrieval_steps:
            mode = "retrieve_and_answer"
        else:
            mode = "answer_with_disclosure" if capabilities.direct_answer_enabled else "ask_clarification"

    # Prevent incompatible fallback tags from pretending substitution is valid.
    if unavailable_required and any("fallback_to_hybrid_after_filter" == w for w in plan.warnings):
        plan.warnings = [w for w in plan.warnings if w != "fallback_to_hybrid_after_filter"]
    if unavailable_required and any("fallback_to_vector_after_filter" == w for w in plan.warnings):
        plan.warnings = [w for w in plan.warnings if w != "fallback_to_vector_after_filter"]

    if "local_kb" in required:
        blocked_source_types.add("model_knowledge")
    if unavailable_required and not compatible_steps:
        if "general_web" in required or "external_academic" in required:
            blocked_source_types.update({"local_kb", "local_section", "local_artifact"})

    answer_policy = {
        "mode": mode,
        "allowed_source_types": sorted(set(allowed_source_types)),
        "blocked_source_types": sorted(set(blocked_source_types)),
        "unavailable_required_sources": unavailable_required,
        "must_disclose_limitations": bool(must_disclose),
        "answer_boundary": answer_boundary,
        "guidance_to_answer_agent": guidance,
    }
    return source_requirements, answer_policy, plan


def _collect_plan_cue_text(raw: Any, normalized_steps: List[RetrievalStep]) -> str:
    parts: List[str] = []
    if isinstance(raw, dict):
        for key in ("intent", "reason", "question", "query"):
            value = raw.get(key)
            if isinstance(value, str) and value.strip():
                parts.append(value.strip())
        raw_steps = raw.get("retrieval_steps") or []
        if isinstance(raw_steps, list):
            for step in raw_steps:
                if not isinstance(step, dict):
                    continue
                for key in ("tool", "query", "section_query", "reason"):
                    value = step.get(key)
                    if isinstance(value, str) and value.strip():
                        parts.append(value.strip())
    for step in normalized_steps:
        for value in (step.tool, step.query, step.section_query, step.reason):
            if isinstance(value, str) and value.strip():
                parts.append(value.strip())
    return " ".join(parts).strip().lower()


def _collect_policy_question_text(raw: Any, normalized_steps: List[RetrievalStep]) -> str:
    parts: List[str] = []
    if isinstance(raw, dict):
        for key in ("question", "query"):
            value = raw.get(key)
            if isinstance(value, str) and value.strip():
                parts.append(value.strip())
        raw_steps = raw.get("retrieval_steps") or []
        if isinstance(raw_steps, list):
            for step in raw_steps:
                if not isinstance(step, dict):
                    continue
                for key in ("query", "section_query"):
                    value = step.get(key)
                    if isinstance(value, str) and value.strip():
                        parts.append(value.strip())
    for step in normalized_steps:
        for value in (step.query, step.section_query):
            if isinstance(value, str) and value.strip():
                parts.append(value.strip())
    return " ".join(parts).strip().lower()


def _external_unavailable_direct_answer_plan(
    warning_codes: List[str],
    reason: str,
) -> IntentPlan:
    warnings = list(warning_codes)
    _append_warning_once(warnings, "retrieval_skipped_to_avoid_wrong_local_fallback")
    return IntentPlan(
        intent="direct_answer",
        needs_retrieval=False,
        retrieval_steps=[],
        max_tools=0,
        allow_external_sources=False,
        direct_answer_allowed=True,
        evidence_policy="answer_with_available_evidence_and_state_uncertainty",
        reason=reason,
        warnings=warnings,
    )


def _extract_json_object_from_text(text: str) -> str:
    source = str(text or "")
    start = source.find("{")
    if start < 0:
        raise ValueError("No JSON object found in planner output.")

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(source)):
        char = source[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return source[start : index + 1]
    raise ValueError("Incomplete JSON object in planner output.")


def _extract_json_candidate(text: str) -> str:
    source = str(text or "").strip()
    if not source:
        raise ValueError("Empty planner output.")

    fenced_blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)```", source, flags=re.IGNORECASE)
    for block in fenced_blocks:
        candidate = str(block or "").strip()
        if not candidate:
            continue
        try:
            return _extract_json_object_from_text(candidate)
        except ValueError:
            continue

    return _extract_json_object_from_text(source)


def _extract_model_text(raw: Any) -> str:
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, dict):
        return json.dumps(raw, ensure_ascii=False)
    content = getattr(raw, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                if item.strip():
                    parts.append(item.strip())
            elif isinstance(item, dict):
                text = str(item.get("text") or item.get("content") or "").strip()
                if text:
                    parts.append(text)
        return "\n".join(parts).strip()
    return str(raw)


def _preview_text(text: str, limit: int = 500) -> str:
    preview = str(text or "").strip()
    if len(preview) <= limit:
        return preview
    return preview[:limit] + "..."


def build_fallback_intent_plan(
    question: str,
    capabilities: Optional[PlannerCapabilities] = None,
    reason: str = "",
) -> IntentPlan:
    caps = capabilities or PlannerCapabilities()
    normalized_question = str(question or "").strip()
    lower_question = normalized_question.lower()
    has_local_evidence_cues = _contains_any_cue(lower_question, _LOCAL_EVIDENCE_CUES)
    has_external_info_cues = _contains_any_cue(lower_question, _EXTERNAL_INFO_CUES)
    has_academic_external_cues = _contains_any_cue(lower_question, _ACADEMIC_EXTERNAL_CUES)

    if has_external_info_cues:
        if has_academic_external_cues and caps.openalex_search_enabled:
            return IntentPlan(
                intent="external_paper_discovery",
                needs_retrieval=True,
                retrieval_steps=[
                    RetrievalStep(
                        tool="openalex_search",
                        query=normalized_question,
                        limit=5,
                        reason="Fallback external academic retrieval for evidence-dependent question.",
                    )
                ],
                max_tools=min(1, caps.max_tools),
                allow_external_sources=True,
                direct_answer_allowed=False,
                reason=reason or "Fallback external retrieval: question appears to require external scholarly evidence.",
                warnings=["planner_fallback_used"],
            )
        if caps.web_search_enabled:
            return IntentPlan(
                intent="web_information",
                needs_retrieval=True,
                retrieval_steps=[
                    RetrievalStep(
                        tool="web_search",
                        query=normalized_question,
                        limit=5,
                        reason="Fallback web retrieval for evidence-dependent question.",
                    )
                ],
                max_tools=min(1, caps.max_tools),
                allow_external_sources=True,
                direct_answer_allowed=False,
                reason=reason or "Fallback external retrieval: question appears to require external/current evidence.",
                warnings=["planner_fallback_used"],
            )
        if has_local_evidence_cues:
            local_warnings = ["planner_fallback_used", "external_retrieval_unavailable"]
            if not caps.web_search_enabled:
                local_warnings.append("web_search_unavailable")
            if not caps.openalex_search_enabled and has_academic_external_cues:
                local_warnings.append("openalex_search_unavailable")
            if caps.hybrid_search_enabled:
                return IntentPlan(
                    intent="local_paper_qa",
                    needs_retrieval=True,
                    retrieval_steps=[
                        RetrievalStep(
                            tool="hybrid_search",
                            query=normalized_question,
                            limit=10,
                            reason="Fallback local retrieval while external retrieval is unavailable.",
                        )
                    ],
                    max_tools=min(1, caps.max_tools),
                    allow_external_sources=False,
                    direct_answer_allowed=False,
                    reason=reason or "Fallback local retrieval: external sources unavailable, keep local knowledge-base retrieval.",
                    warnings=local_warnings,
                )
            if caps.vector_search_enabled:
                return IntentPlan(
                    intent="local_paper_qa",
                    needs_retrieval=True,
                    retrieval_steps=[
                        RetrievalStep(
                            tool="vector_search",
                            query=normalized_question,
                            limit=10,
                            reason="Fallback local retrieval while external retrieval is unavailable.",
                        )
                    ],
                    max_tools=min(1, caps.max_tools),
                    allow_external_sources=False,
                    direct_answer_allowed=False,
                    reason=reason or "Fallback local retrieval: external sources unavailable, keep local knowledge-base retrieval.",
                    warnings=local_warnings,
                )
        if caps.direct_answer_enabled:
            return IntentPlan(
                intent="direct_answer",
                needs_retrieval=False,
                retrieval_steps=[],
                max_tools=0,
                allow_external_sources=False,
                direct_answer_allowed=True,
                reason=reason or "Fallback direct answer: external evidence appears required but unavailable.",
                warnings=["planner_fallback_used", "fallback_external_retrieval_unavailable"],
            )

    if has_local_evidence_cues:
        if caps.hybrid_search_enabled:
            return IntentPlan(
                intent="local_paper_qa",
                needs_retrieval=True,
                retrieval_steps=[
                    RetrievalStep(
                        tool="hybrid_search",
                        query=normalized_question,
                        limit=10,
                        reason="Fallback local retrieval for evidence-dependent question.",
                    )
                ],
                max_tools=min(1, caps.max_tools),
                allow_external_sources=False,
                direct_answer_allowed=False,
                reason=reason or "Fallback local retrieval: question appears to require local/document evidence.",
                warnings=["planner_fallback_used"],
            )
        if caps.vector_search_enabled:
            return IntentPlan(
                intent="local_paper_qa",
                needs_retrieval=True,
                retrieval_steps=[
                    RetrievalStep(
                        tool="vector_search",
                        query=normalized_question,
                        limit=10,
                        reason="Fallback local retrieval for evidence-dependent question.",
                    )
                ],
                max_tools=min(1, caps.max_tools),
                allow_external_sources=False,
                direct_answer_allowed=False,
                reason=reason or "Fallback local retrieval: question appears to require local/document evidence.",
                warnings=["planner_fallback_used"],
            )
        if caps.direct_answer_enabled:
            return IntentPlan(
                intent="direct_answer",
                needs_retrieval=False,
                retrieval_steps=[],
                max_tools=0,
                allow_external_sources=False,
                direct_answer_allowed=True,
                reason=reason or "Fallback direct answer: document evidence requested but local retrieval unavailable.",
                warnings=["planner_fallback_used", "fallback_local_retrieval_unavailable"],
            )

    if caps.direct_answer_enabled:
        return IntentPlan(
            intent="direct_answer",
            needs_retrieval=False,
            retrieval_steps=[],
            max_tools=0,
            allow_external_sources=False,
            direct_answer_allowed=True,
                reason=reason or "Fallback direct answer: question appears answerable without additional evidence or tools.",
            warnings=["planner_fallback_used"],
        )
    return _safe_default_plan(question)


def _safe_default_plan_with_caps(question: str, capabilities: PlannerCapabilities) -> IntentPlan:
    if capabilities.direct_answer_enabled and not capabilities.available_tools():
        return IntentPlan(
            intent="direct_answer",
            needs_retrieval=False,
            retrieval_steps=[],
            max_tools=0,
            allow_external_sources=False,
            direct_answer_allowed=True,
            reason="Fallback: no retrieval tool available.",
            warnings=["planner_fallback_no_tools"],
        )
    return build_fallback_intent_plan(question, capabilities=capabilities)


def _safe_retry_plan(question: str, missing_aspects: List[str], suggested_query: str = "") -> IntentPlan:
    query = str(suggested_query or "").strip() or str(question or "").strip()
    missing_text = " ".join(str(x) for x in (missing_aspects or [])).lower()
    if any(k in missing_text for k in ["table", "figure", "fig", "algorithm", "pseudocode"]):
        return IntentPlan(
            intent="local_artifact_qa",
            needs_retrieval=True,
            retrieval_steps=[
                RetrievalStep(
                    tool="artifact_search",
                    query=query,
                    limit=6,
                    artifact_types=["table", "figure", "algorithm"],
                    reason="Fallback retry plan for potential artifact evidence gaps.",
                )
            ],
            max_tools=1,
            allow_external_sources=False,
            reason="Fallback retry artifact plan.",
            warnings=["planner_retry_fallback_used"],
        )
    return IntentPlan(
        intent="local_paper_qa",
        needs_retrieval=True,
        retrieval_steps=[
            RetrievalStep(
                tool="hybrid_search",
                query=query,
                limit=8,
                reason="Fallback retry local retrieval plan.",
            )
        ],
        max_tools=1,
        allow_external_sources=False,
        reason="Fallback retry plan.",
        warnings=["planner_retry_fallback_used"],
    )


def normalize_intent_plan(raw: Any, capabilities: Optional[PlannerCapabilities] = None) -> IntentPlan:
    caps = capabilities or PlannerCapabilities()
    cue_text = ""
    question_for_policy = ""
    if isinstance(raw, IntentPlan):
        plan = raw
        cue_text = _collect_plan_cue_text(raw.model_dump(), list(plan.retrieval_steps or []))
        question_for_policy = _collect_policy_question_text(raw.model_dump(), list(plan.retrieval_steps or []))
    else:
        if isinstance(raw, str):
            raw = json.loads(raw)
        if not isinstance(raw, dict):
            raise ValueError("Plan payload must be a dict-like object.")

        allowed_tools = {
            "none",
            "hybrid_search",
            "vector_search",
            "section_search",
            "artifact_search",
            "openalex_search",
            "web_search",
        }
        allowed_artifacts = {"table", "figure", "algorithm"}

        raw_steps = raw.get("retrieval_steps") or []
        normalized_steps: List[RetrievalStep] = []
        for step in raw_steps[:2]:
            if not isinstance(step, dict):
                continue
            tool = str(step.get("tool") or "hybrid_search").strip().lower()
            if tool not in allowed_tools:
                tool = "hybrid_search"
            limit_val = step.get("limit", 10)
            try:
                limit_val = int(limit_val)
            except Exception:
                limit_val = 10
            limit_val = max(1, min(50, limit_val))
            artifact_types = []
            for t in (step.get("artifact_types") or []):
                tv = str(t or "").strip().lower()
                if tv in allowed_artifacts:
                    artifact_types.append(tv)
            normalized_steps.append(
                RetrievalStep(
                    tool=tool,  # type: ignore[arg-type]
                    query=str(step.get("query") or ""),
                    limit=limit_val,
                    search_type=step.get("search_type"),
                    section_query=step.get("section_query"),
                    artifact_types=artifact_types,  # type: ignore[arg-type]
                    document_id=step.get("document_id"),
                    reason=str(step.get("reason") or ""),
                )
            )
        cue_text = _collect_plan_cue_text(raw, normalized_steps)
        question_for_policy = _collect_policy_question_text(raw, normalized_steps)

        needs_retrieval = bool(raw.get("needs_retrieval", True))
        direct_answer_allowed = bool(raw.get("direct_answer_allowed", False))
        raw_intent = raw.get("intent")
        inferred_intent = str(raw_intent).strip() if raw_intent is not None else ""
        if not needs_retrieval:
            inferred_intent = "direct_answer"
            direct_answer_allowed = bool(caps.direct_answer_enabled)
        if not needs_retrieval:
            normalized_steps = []
        if direct_answer_allowed and not needs_retrieval:
            normalized_steps = []

        allow_external = bool(raw.get("allow_external_sources", False))
        for step in normalized_steps:
            if step.tool in {"openalex_search", "web_search"}:
                allow_external = True

        max_tools = raw.get("max_tools", caps.max_tools)
        try:
            max_tools = int(max_tools)
        except Exception:
            max_tools = 2
        if not needs_retrieval:
            max_tools = 0
        max_tools = max(0, min(caps.max_tools, max_tools))
        normalized_steps = normalized_steps[:max_tools] if max_tools > 0 else []

        plan = IntentPlan(
            intent=(inferred_intent or "local_paper_qa"),
            needs_retrieval=needs_retrieval,
            retrieval_steps=normalized_steps,
            max_tools=max_tools,
            allow_external_sources=allow_external,
            evidence_policy=str(
                raw.get("evidence_policy")
                or "answer_with_available_evidence_and_state_uncertainty"
            ),
            direct_answer_allowed=direct_answer_allowed,
            rewrite_allowed=bool(raw.get("rewrite_allowed", True)),
            reason=str(raw.get("reason") or ""),
            warnings=list(raw.get("warnings") or []),
        )

    if not plan.needs_retrieval:
        plan.intent = "direct_answer"
        plan.retrieval_steps = []
        plan.max_tools = 0
        if caps.direct_answer_enabled:
            plan.direct_answer_allowed = True
    if len(plan.retrieval_steps) > caps.max_tools:
        plan.retrieval_steps = plan.retrieval_steps[: caps.max_tools]

    allowed = set(caps.available_tools())
    filtered_steps: List[RetrievalStep] = []
    filtered_unavailable: List[str] = []
    for step in plan.retrieval_steps:
        if step.tool not in allowed:
            filtered_unavailable.append(step.tool)
            continue
        filtered_steps.append(step)
    plan.retrieval_steps = filtered_steps[: caps.max_tools]
    if filtered_unavailable:
        plan.warnings.append(f"filtered_unavailable_tools:{','.join(filtered_unavailable)}")
    if "web_search" in filtered_unavailable:
        _append_warning_once(plan.warnings, "web_search_unavailable")
    if "openalex_search" in filtered_unavailable:
        _append_warning_once(plan.warnings, "openalex_search_unavailable")
    if any(tool in {"web_search", "openalex_search"} for tool in filtered_unavailable):
        _append_warning_once(plan.warnings, "external_retrieval_unavailable")

    if plan.retrieval_steps:
        plan.needs_retrieval = True
    elif plan.needs_retrieval:
        has_local_evidence_cues = _contains_any_cue(cue_text, _STRICT_LOCAL_EVIDENCE_CUES)
        has_external_info_cues = _contains_any_cue(cue_text, _EXTERNAL_INFO_CUES)
        has_external_unavailable = any(
            tool in {"web_search", "openalex_search"} for tool in filtered_unavailable
        )

        if (
            has_external_unavailable
            and has_external_info_cues
            and not has_local_evidence_cues
            and caps.direct_answer_enabled
        ):
            external_reason = (
                "Web retrieval was requested but unavailable; answering without local KB retrieval."
                if "web_search" in filtered_unavailable
                else "External retrieval was requested but unavailable; answering without local KB retrieval."
            )
            plan = _external_unavailable_direct_answer_plan(
                warning_codes=plan.warnings,
                reason=external_reason,
            )
        elif caps.direct_answer_enabled and plan.direct_answer_allowed:
            plan.needs_retrieval = False
        elif caps.hybrid_search_enabled:
            plan.retrieval_steps = [RetrievalStep(tool="hybrid_search", query="", limit=10, reason="Fallback when planned tools unavailable.")]
            plan.max_tools = min(1, caps.max_tools)
            _append_warning_once(plan.warnings, "fallback_to_hybrid_after_filter")
            plan.needs_retrieval = True
        elif caps.vector_search_enabled:
            plan.retrieval_steps = [RetrievalStep(tool="vector_search", query="", limit=10, reason="Fallback when planned tools unavailable.")]
            plan.max_tools = min(1, caps.max_tools)
            _append_warning_once(plan.warnings, "fallback_to_vector_after_filter")
            plan.needs_retrieval = True
        else:
            plan.needs_retrieval = False
            _append_warning_once(plan.warnings, "no_available_retrieval_tools")
    plan.allow_external_sources = bool(caps.openalex_search_enabled or caps.web_search_enabled) and bool(
        any(s.tool in {"openalex_search", "web_search"} for s in plan.retrieval_steps)
    )
    source_requirements, answer_policy, revised_plan = resolve_source_policy(
        question=(question_for_policy or cue_text or ""),
        normalized_plan=plan,
        capabilities=caps,
    )
    revised_plan.source_requirements = source_requirements
    revised_plan.answer_policy = answer_policy
    return revised_plan


def _finalize_plan_queries(plan: IntentPlan, question: str) -> IntentPlan:
    for step in plan.retrieval_steps:
        if not step.query:
            step.query = question
    return plan


def _serialize_planner_output(raw: Any) -> Any:
    if raw is None:
        return None
    if isinstance(raw, (str, int, float, bool)):
        return raw
    if isinstance(raw, list):
        return [_serialize_planner_output(item) for item in raw]
    if isinstance(raw, dict):
        return {str(key): _serialize_planner_output(value) for key, value in raw.items()}
    if hasattr(raw, "model_dump"):
        try:
            return raw.model_dump()
        except Exception:
            pass
    content = getattr(raw, "content", None)
    if content is not None:
        return {
            "type": type(raw).__name__,
            "content": _serialize_planner_output(content),
        }
    return {"type": type(raw).__name__, "repr": str(raw)}


def _normalize_plan_from_model_output(raw: Any, question: str, capabilities: PlannerCapabilities) -> IntentPlan:
    if isinstance(raw, dict):
        return _finalize_plan_queries(normalize_intent_plan(raw, capabilities=capabilities), question)
    text = _extract_model_text(raw)
    candidate = _extract_json_candidate(text)
    return _finalize_plan_queries(normalize_intent_plan(candidate, capabilities=capabilities), question)


async def plan_user_intent_debug(
    question: str,
    context_hint: str = "",
    model: Optional[Any] = None,
    capabilities: Optional[PlannerCapabilities] = None,
) -> Dict[str, Any]:
    caps = capabilities or PlannerCapabilities()
    prompt = build_intent_planner_prompt(question, context_hint=context_hint, capabilities=caps)
    if model is None:
        fallback_plan = build_fallback_intent_plan(
            question,
            capabilities=caps,
            reason="Fallback direct answer: planner model unavailable." if caps.direct_answer_enabled else "Fallback retrieval: planner model unavailable.",
        )
        plan = normalize_intent_plan(
            {
                **fallback_plan.model_dump(),
                "question": question,
            },
            capabilities=caps,
        )
        return {
            "prompt": prompt,
            "capabilities": caps.model_dump(),
            "raw_plan": plan.model_dump(),
            "normalized_plan": plan.model_dump(),
            "fallback_used": True,
            "fallback_reason": "model_unavailable",
            "fallback_decision": "direct_answer_fallback" if plan.intent == "direct_answer" else "local_retrieval_fallback",
            "raw_model_content": "",
            "raw_model_content_preview": "",
        }

    try:
        if hasattr(model, "ainvoke"):
            raw = await model.ainvoke(prompt)
        elif hasattr(model, "__call__"):
            maybe = model(prompt)
            if hasattr(maybe, "__await__"):
                raw = await maybe
            else:
                raw = maybe
        else:
            fallback_plan = build_fallback_intent_plan(
                question,
                capabilities=caps,
                reason="Fallback direct answer: planner model not invokable." if caps.direct_answer_enabled else "Fallback retrieval: planner model not invokable.",
            )
            plan = normalize_intent_plan(
                {
                    **fallback_plan.model_dump(),
                    "question": question,
                },
                capabilities=caps,
            )
            return {
                "prompt": prompt,
                "capabilities": caps.model_dump(),
                "raw_plan": plan.model_dump(),
                "normalized_plan": plan.model_dump(),
                "fallback_used": True,
                "fallback_reason": "model_not_invokable",
                "fallback_decision": "direct_answer_fallback" if plan.intent == "direct_answer" else "local_retrieval_fallback",
                "raw_model_content": "",
                "raw_model_content_preview": "",
            }

        raw_model_content = _extract_model_text(raw)
        plan = _normalize_plan_from_model_output(raw, question, caps)
        return {
            "prompt": prompt,
            "capabilities": caps.model_dump(),
            "raw_plan": _serialize_planner_output(raw),
            "normalized_plan": plan.model_dump(),
            "fallback_used": False,
            "fallback_reason": "",
            "fallback_decision": "",
            "raw_model_content": raw_model_content,
            "raw_model_content_preview": _preview_text(raw_model_content),
        }
    except Exception as exc:
        raw_model_content = _extract_model_text(locals().get("raw"))
        fallback_plan = build_fallback_intent_plan(
            question,
            capabilities=caps,
            reason="Fallback direct answer: no evidence-seeking cues detected." if caps.direct_answer_enabled else "Fallback retrieval after planner parse failure.",
        )
        plan = normalize_intent_plan(
            {
                **fallback_plan.model_dump(),
                "question": question,
            },
            capabilities=caps,
        )
        return {
            "prompt": prompt,
            "capabilities": caps.model_dump(),
            "raw_plan": {"error": str(exc)},
            "normalized_plan": plan.model_dump(),
            "fallback_used": True,
            "fallback_reason": str(exc),
            "fallback_decision": "direct_answer_fallback" if plan.intent == "direct_answer" else "local_retrieval_fallback",
            "raw_model_content": raw_model_content,
            "raw_model_content_preview": _preview_text(raw_model_content),
        }


async def plan_user_intent(
    question: str,
    context_hint: str = "",
    model: Optional[Any] = None,
    capabilities: Optional[PlannerCapabilities] = None,
) -> IntentPlan:
    debug_payload = await plan_user_intent_debug(
        question=question,
        context_hint=context_hint,
        model=model,
        capabilities=capabilities,
    )
    return IntentPlan.model_validate(debug_payload["normalized_plan"])


async def revise_intent_plan_for_retry(
    question: str,
    previous_plan: IntentPlan,
    missing_aspects: List[str],
    retrieval_summary: str,
    suggested_query: str = "",
    model: Optional[Any] = None,
    capabilities: Optional[PlannerCapabilities] = None,
) -> IntentPlan:
    caps = capabilities or PlannerCapabilities()
    if model is None:
        return normalize_intent_plan(
            _safe_retry_plan(question, missing_aspects, suggested_query=suggested_query),
            capabilities=caps,
        )

    prompt = build_retry_intent_planner_prompt(
        question=question,
        previous_plan=previous_plan,
        missing_aspects=missing_aspects,
        retrieval_summary=retrieval_summary,
        suggested_query=suggested_query,
    )
    try:
        if hasattr(model, "ainvoke"):
            raw = await model.ainvoke(prompt)
        elif hasattr(model, "__call__"):
            maybe = model(prompt)
            raw = await maybe if hasattr(maybe, "__await__") else maybe
        else:
            return normalize_intent_plan(
                _safe_retry_plan(question, missing_aspects, suggested_query=suggested_query),
                capabilities=caps,
            )

        plan = normalize_intent_plan(
            raw if isinstance(raw, (dict, str)) else getattr(raw, "content", ""),
            capabilities=caps,
        )
        if not plan.needs_retrieval:
            return normalize_intent_plan(
                _safe_retry_plan(question, missing_aspects, suggested_query=suggested_query),
                capabilities=caps,
            )
        return plan
    except Exception:
        return normalize_intent_plan(
            _safe_retry_plan(question, missing_aspects, suggested_query=suggested_query),
            capabilities=caps,
        )
