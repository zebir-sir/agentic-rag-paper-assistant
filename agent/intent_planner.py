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
GraphRelationType = Literal["semantic_similarity", "cites", "method_lineage"]
GraphDirection = Literal["both", "outgoing", "incoming"]


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
    use_paper_graph: bool = False
    graph_usage_reason: str = ""
    graph_relation_types: List[GraphRelationType] = Field(default_factory=list)
    graph_direction: GraphDirection = "both"
    graph_neighbor_limit: int = Field(default=6, ge=1, le=12)
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
        "Decide the answer's evidence requirements before selecting tools.\n"
        "Use minimal necessary retrieval and plan at most 2 tools.\n"
        "If one tool is clearly sufficient, do not add a second tool.\n"
        "Source decision matrix:\n"
        "- Use hybrid_search or vector_search for claims grounded in uploaded/local papers when no narrower evidence type is required.\n"
        "- Use section_search when the user asks about a named section or a paper part such as abstract, method, experiments, results, conclusion, appendix, or references.\n"
        "artifact_search is appropriate when the answer depends on non-prose evidence such as tables, figures, diagrams, "
        "algorithms, pseudocode, metric comparisons, or pipeline/process visualizations.\n"
        "If you use artifact_search, retrieve only artifacts directly relevant to the user question.\n"
        "Do not enumerate all artifacts by default.\n"
        "If prose evidence or section-scoped evidence is sufficient, prefer hybrid_search or section_search.\n"
        "- Use openalex_search for papers outside the local corpus, literature discovery, related work, DOI, venue, author, year, citation metadata, or open-access metadata.\n"
        "- Use web_search for current, non-paper, or general open-web information.\n"
        "Decision priority: first determine whether the answer requires a volatile real-world fact, then choose the source. "
        "Questions about a person's life/death/status, office holder, organization status, event progress, news, prices, laws, "
        "policies, schedules, weather, availability, rankings, or any fact that may have changed since model training require web_search "
        "when it is available, even if the user does not explicitly say 'latest', 'search', or 'online'. "
        "Do not answer such questions from model knowledge or a local paper corpus.\n"
        "Direct-answer policy (use a conservative allowlist): direct_answer is allowed only for (1) writing transformation of text the user supplied, such as rewriting, translation, proofreading, outlining, or tone changes; (2) stable general explanations of widely known concepts or algorithms; (3) brainstorming, design discussion, or reasoning from premises explicitly supplied by the user; and (4) clarification questions.\n"
        "Do NOT choose direct_answer when the answer would assert or verify: content, numbers, quotes, formulas, results, citations, or section locations from a paper/document; facts about a particular person, organization, product, event, law, policy, price, schedule, or current state; external literature metadata; or any source the user explicitly requests. Choose retrieval for the matching source class instead.\n"
        "For a direct answer set needs_retrieval=false, retrieval_steps=[], max_tools=0, direct_answer_allowed=true, and state why the answer is stable or transformation-only in reason. For every other case set direct_answer_allowed=false.\n"
        "Use retrieval when the question depends on document-grounded, paper-grounded, artifact-grounded, external, current, or other tool-provided evidence.\n"
        "When a request mixes sources, retain each requested source class within the two-tool budget and state limits for unavailable classes.\n"
        "Do not treat 'the model can answer naturally' as sufficient for direct_answer. Use direct_answer only when the question matches the conservative allowlist above; otherwise choose the matching retrieval source, or mark the request as unclear when no valid source is available.\n"
        "Evidence cues (document/section/table/figure/algorithm/citation/OpenAlex/web/latest) are examples, not an exhaustive rule list.\n"
        "Do not retrieve merely because retrieval tools are available.\n"
        "If a required source is unavailable, do not substitute a different source class or model knowledge. Leave retrieval_steps empty for that unavailable requirement, set direct_answer_allowed=false, and state the limitation in reason.\n"
        "If no retrieval is needed, set needs_retrieval=false and retrieval_steps=[].\n"
        "paper graph expansion is a candidate-scope aid, not an evidence source or retrieval tool. "
        "Set use_paper_graph=true only when the question requires relationships between papers, such as similar work, "
        "cross-paper comparison, method transfer, method lineage, related-work exploration, or innovation exploration. "
        "For a single paper's exact facts, numbers, sections, formulas, or when the user asks to use only one paper, set it false. "
        "When true, provide graph_usage_reason, graph_relation_types, graph_direction, and graph_neighbor_limit (1-12). "
        "graph_relation_types may contain only semantic_similarity, cites, and method_lineage. "
        "Use outgoing when asking what the seed paper cites or builds on; incoming when asking which papers cite, extend, or improve the seed paper; "
        "use both for related-work exploration or cross-paper comparison. Prefer method_lineage for method origin/evolution, cites for citation paths, and semantic_similarity for topical similarity.\n"
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
        "When previous retrieval is missing non-prose evidence, artifact_search can be added.\n"
        "artifact_search should target only artifacts directly relevant to the question and missing aspects.\n"
        "If you set artifact_types, narrow them to the relevant types instead of defaulting to all artifact types.\n"
        "If prose or section evidence is enough, prefer hybrid_search or section_search.\n"
        "section_search is for section-focused gaps.\n"
        "Prefer local tools first; use openalex/web only when external sources are truly needed.\n"
        "Preserve required source classes and the direct-answer boundary from the original question. In particular, do not replace a required web search for volatile "
        "real-world facts (person status, event progress, news, price, policy, schedule, weather, availability, ranking) with local retrieval or direct answer.\n"
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
    if (
        any(m in q for m in generic_markers)
        and "paper" not in q
        and "document" not in q
        and infer_direct_answer_eligibility(question)["allowed"]
    ):
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
    "section", "sections", "chapter", "appendix", "abstract section", "method section",
    "experiments section", "results section", "章节", "小节", "附录", "第", "摘要部分",
    "方法章节", "实验章节", "结果章节",
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

_VOLATILE_REAL_WORLD_CUES = (
    "alive", "still alive", "dead", "died", "death", "deceased", "status", "whereabouts",
    "current ceo", "current president", "current minister", "current role", "in office",
    "price", "stock price", "exchange rate", "weather", "schedule", "availability", "ranking",
    "laws", "law", "regulation", "policy", "election", "event update", "breaking",
    "还活着", "是否在世", "去世", "死亡", "死了", "现状", "近况", "下落",
    "现任", "在任", "任职", "负责人", "董事长", "首席执行官", "ceo",
    "价格", "股价", "汇率", "天气", "日程", "赛程", "上映", "发售", "是否有货",
    "排名", "法律", "法规", "政策", "选举", "进展", "突发", "事件后续",
)

_WRITING_TRANSFORMATION_CUES = (
    "rewrite", "rephrase", "proofread", "polish", "translate", "translation", "outline",
    "改写", "润色", "校对", "翻译", "提纲", "重写", "更自然", "更正式", "更简洁",
)

_BRAINSTORMING_CUES = (
    "brainstorm", "idea", "design a", "how should i", "方案", "头脑风暴", "想法", "怎么设计",
)

_PAPER_GRAPH_RELATION_CUES = (
    "similar work", "similar papers", "related work", "related papers", "method transfer",
    "method lineage", "cross-paper", "across papers", "innovation exploration", "research direction",
    "相似工作", "相似论文", "相关工作", "相关论文", "论文间", "跨论文", "方法迁移",
    "方法继承", "方法谱系", "创新探索", "创新切入", "研究方向", "哪些论文",
)


_PAPER_GRAPH_ACTIVATION_CUES = (
    "citation", "citations", "reference", "references", "cited by",
    "method evolution", "extend", "extends", "improve", "improves", "based on", "origin", "foundation",
    "\u540e\u7eed", "\u6269\u5c55", "\u6539\u8fdb", "\u57fa\u4e8e", "\u6765\u6e90", "\u8d77\u6e90",
    "\u5f15\u7528", "\u88ab\u5f15", "\u53c2\u8003\u6587\u732e", "\u54ea\u4e9b\u5de5\u4f5c",
)


def infer_paper_graph_need(question: str) -> Dict[str, Any]:
    """Infer graph expansion only for questions that need paper-to-paper relations."""
    text = str(question or "").strip().lower()
    if "extend" in text or "improve" in text:
        return {
            "use_paper_graph": True,
            "reason": "Question requests method evolution or follow-on work.",
            "relation_types": ["method_lineage", "cites", "semantic_similarity"],
            "direction": "incoming",
        }
    if not _contains_any_cue(text, (*_PAPER_GRAPH_RELATION_CUES, *_PAPER_GRAPH_ACTIVATION_CUES)):
        return {"use_paper_graph": False, "reason": "", "relation_types": [], "direction": "both"}
    lineage_cues = ("evolution", "extends", "improves", "method lineage", "演进", "后续", "发展", "改进", "继承")
    origin_cues = ("origin", "foundation", "based on", "来源", "起源", "基础", "基于什么")
    citation_cues = ("citation", "reference", "cited by", "引用", "被引", "参考文献")
    if _contains_any_cue(text, lineage_cues):
        return {
            "use_paper_graph": True,
            "reason": "Question requests method evolution or follow-on work.",
            "relation_types": ["method_lineage", "cites", "semantic_similarity"],
            "direction": "incoming",
        }
    if _contains_any_cue(text, origin_cues):
        return {
            "use_paper_graph": True,
            "reason": "Question requests method origins or foundations.",
            "relation_types": ["method_lineage", "cites", "semantic_similarity"],
            "direction": "outgoing",
        }
    if _contains_any_cue(text, citation_cues):
        return {
            "use_paper_graph": True,
            "reason": "Question requests citation relationships.",
            "relation_types": ["cites", "semantic_similarity"],
            "direction": "both",
        }
    return {
        "use_paper_graph": True,
        "reason": "Question requests cross-paper relationships or exploration.",
        "relation_types": ["semantic_similarity", "method_lineage", "cites"],
        "direction": "both",
    }


def _apply_inferred_paper_graph_need(
    plan: IntentPlan,
    graph_need: Dict[str, Any],
) -> IntentPlan:
    """Keep deterministic fallback plans aligned with graph-aware planner output."""
    if not graph_need.get("use_paper_graph") or not plan.needs_retrieval:
        return plan
    plan.use_paper_graph = True
    plan.graph_usage_reason = str(graph_need.get("reason") or "")
    plan.graph_relation_types = list(graph_need.get("relation_types") or [])
    plan.graph_direction = str(graph_need.get("direction") or "both")
    return plan


def _contains_any_cue(text: str, cues: tuple[str, ...]) -> bool:
    normalized = str(text or "").strip().lower()
    if not normalized:
        return False
    return any(cue in normalized for cue in cues)


def infer_general_web_evidence_need(question: str) -> Dict[str, Any]:
    """Classify time-sensitive real-world facts as requiring web evidence."""
    text = str(question or "").strip().lower()
    if not text:
        return {"needs_web": False, "volatile": False, "reason": ""}
    if _contains_any_cue(text, _VOLATILE_REAL_WORLD_CUES):
        return {
            "needs_web": True,
            "volatile": True,
            "reason": "Question depends on a potentially time-sensitive real-world fact.",
        }
    if _contains_any_cue(text, _EXTERNAL_INFO_CUES):
        return {
            "needs_web": True,
            "volatile": False,
            "reason": "Question explicitly requests current or open-web information.",
        }
    return {"needs_web": False, "volatile": False, "reason": ""}


def infer_direct_answer_eligibility(question: str) -> Dict[str, Any]:
    """Allow direct answers only for stable, user-provided, or non-factual tasks."""
    text = str(question or "").strip().lower()
    if not text:
        return {"allowed": True, "reason": "Clarification can be requested without retrieval."}

    source_need = infer_general_web_evidence_need(text)
    if source_need["needs_web"]:
        return {"allowed": False, "reason": "Question requires current or open-web evidence."}
    if _contains_any_cue(text, _EXPLICIT_LOCAL_EVIDENCE_CUES):
        return {"allowed": False, "reason": "Question requests local document evidence."}
    if _contains_any_cue(text, _ACADEMIC_EXTERNAL_CUES):
        return {"allowed": False, "reason": "Question requests external academic evidence."}
    if _contains_any_cue(text, _WRITING_TRANSFORMATION_CUES):
        return {"allowed": True, "reason": "Writing transformation of user-provided text."}
    if _contains_any_cue(text, _BRAINSTORMING_CUES):
        return {"allowed": True, "reason": "Brainstorming or design discussion without factual verification."}

    subjective_advice_markers = (
        "你觉得", "建议我", "吃什么", "怎么选", "帮我选", "你多大了", "你是谁", "what should i", "which should i choose",
    )
    if _contains_any_cue(text, subjective_advice_markers):
        return {"allowed": True, "reason": "Subjective advice does not require factual verification."}

    stable_explanation_markers = (
        "what is", "explain", "how does", "difference between", "define", "概念", "原理", "是什么", "解释",
    )
    if _contains_any_cue(text, stable_explanation_markers):
        return {"allowed": True, "reason": "Stable general explanation."}
    return {"allowed": False, "reason": "Question is not in the direct-answer allowlist."}


def _explicit_openalex_query(question: str) -> str:
    """Keep source instructions out of the query without imposing a research field."""
    value = " ".join(str(question or "").split())
    instruction_pattern = (
        r"(?i)\b(openalex|doi|web\s+search|online\s+search|search|find|recommend)\b"
        r"|\u8054\u7f51|\u641c\u7d22|\u68c0\u7d22|\u63a8\u8350|\u8bba\u6587|\u51e0\u7bc7|\u4e00\u7bc7|\u6700\u65b0"
    )
    return re.sub(r"\s+", " ", re.sub(instruction_pattern, " ", value)).strip(" ,.;:!?\u3002\uff0c\uff1b\uff1a")


def apply_explicit_routing_guard(
    plan: IntentPlan,
    question: str,
    capabilities: PlannerCapabilities,
) -> IntentPlan:
    """Enforce user-explicit source and no-retrieval requests before policy compilation."""
    text = str(question or "").strip().lower()
    if "openalex" in text:
        if capabilities.openalex_search_enabled:
            return IntentPlan(
                intent="external_paper_discovery",
                needs_retrieval=True,
                retrieval_steps=[RetrievalStep(
                    tool="openalex_search",
                    query=_explicit_openalex_query(question),
                    limit=5,
                    reason="User explicitly requested OpenAlex metadata.",
                )],
                max_tools=1,
                allow_external_sources=True,
                direct_answer_allowed=False,
                reason="Explicit OpenAlex request.",
                warnings=list(plan.warnings),
            )
        plan.retrieval_steps = [RetrievalStep(tool="openalex_search", query=_explicit_openalex_query(question), limit=5)]
        plan.needs_retrieval = True
        plan.direct_answer_allowed = False
        return plan

    section_scope_text = text.split("不要", 1)[0]
    section_terms: list[str] = []
    for marker, canonical in (("摘要", "abstract"), ("引言", "introduction"), ("结论", "conclusion")):
        if marker in section_scope_text and canonical not in section_terms:
            section_terms.append(canonical)
    if "《" in question and section_terms:
        return IntentPlan(
            intent="local_section_qa",
            needs_retrieval=True,
            retrieval_steps=[RetrievalStep(
                tool="section_search",
                query=question,
                section_query=" and ".join(section_terms),
                limit=8,
                reason="User explicitly constrained the answer to named paper sections.",
            )],
            max_tools=2,
            allow_external_sources=False,
            direct_answer_allowed=False,
            reason="Explicit paper and section scope.",
            warnings=list(plan.warnings),
        )

    if "知识星图" in question or "星图关系" in question:
        paper_titles = [value.strip() for value in re.findall(r"《([^》]+)》", question) if value.strip()]
        retrieval_steps = [
            RetrievalStep(
                tool="hybrid_search",
                query=title,
                limit=6,
                reason="Retrieve source evidence for each paper named in the graph comparison.",
            )
            for title in paper_titles[:2]
        ] or [RetrievalStep(
            tool="hybrid_search",
            query=question,
            limit=10,
            reason="User explicitly requested paper-graph relationship navigation.",
        )]
        return IntentPlan(
            intent="multi_paper_compare",
            needs_retrieval=True,
            retrieval_steps=retrieval_steps,
            max_tools=2,
            allow_external_sources=False,
            direct_answer_allowed=False,
            use_paper_graph=True,
            graph_relation_types=["semantic_similarity", "method_lineage", "cites"],
            reason="Explicit knowledge-star-map relationship request.",
            warnings=list(plan.warnings),
        )

    if "《" in question:
        artifact_need = infer_artifact_evidence_need(question)
        supporting_section = infer_supporting_section_need(question)
        if artifact_need["needs_artifact"]:
            retrieval_steps = [RetrievalStep(
                tool="artifact_search",
                query=question,
                limit=8,
                artifact_types=artifact_need["artifact_types"],
                reason="Explicit named-paper non-prose evidence request.",
            )]
            if supporting_section["needed"]:
                retrieval_steps.append(RetrievalStep(
                    tool="section_search",
                    query=question,
                    section_query=supporting_section["section_query"],
                    limit=6,
                    reason="Supporting prose context for a requested local artifact.",
                ))
            return IntentPlan(
                intent="local_artifact_qa",
                needs_retrieval=True,
                retrieval_steps=retrieval_steps,
                max_tools=len(retrieval_steps),
                allow_external_sources=False,
                direct_answer_allowed=False,
                reason="Explicit named-paper non-prose evidence request.",
                warnings=list(plan.warnings),
            )
        elif any(marker in text for marker in ("实验", "评估", "评价信号", "实验设置")):
            tool = "section_search"
            intent = "local_section_qa"
            reason = "Explicit named-paper experiment or evaluation request."
        else:
            tool = "hybrid_search"
            intent = "local_paper_qa"
            reason = "Explicit named-paper evidence request."
        section_query = "experiment or evaluation or results" if tool == "section_search" else None
        return IntentPlan(
            intent=intent,
            needs_retrieval=True,
            retrieval_steps=[RetrievalStep(
                tool=tool, query=question, section_query=section_query, limit=8, reason=reason,
            )],
            max_tools=1,
            allow_external_sources=False,
            direct_answer_allowed=False,
            reason=reason,
            warnings=list(plan.warnings),
        )

    no_retrieval_request = any(marker in text for marker in ("不检索", "不引用本地论文", "不引用外部网页", "不引用资料"))
    stable_explanation = any(marker in text for marker in ("解释", "概念", "原理", "什么是", "通俗", "给出", "区分", "说明"))
    volatile_request = any(marker in text for marker in ("最新", "当前", "最近", "新闻", "价格", "天气", "日程"))
    if no_retrieval_request and stable_explanation and not volatile_request and capabilities.direct_answer_enabled:
        return IntentPlan(
            intent="direct_answer",
            needs_retrieval=False,
            retrieval_steps=[],
            max_tools=0,
            allow_external_sources=False,
            direct_answer_allowed=True,
            reason="User explicitly requested a stable explanation without retrieval or citations.",
            warnings=list(plan.warnings),
        )
    return plan


def _append_warning_once(warnings: List[str], warning: str) -> None:
    value = str(warning or "").strip()
    if value and value not in warnings:
        warnings.append(value)


def infer_artifact_evidence_need(*texts: str) -> Dict[str, Any]:
    combined = " ".join(str(text or "") for text in texts if str(text or "").strip()).lower()
    if not combined:
        return {"needs_artifact": False, "artifact_types": [], "reason": ""}

    table_cues = (
        "metric", "metrics", "benchmark", "score", "scores", "quantitative", "result table",
        "comparison matrix", "performance comparison", "ablation", "numbers", "numeric",
        "指标", "分数", "数值", "对比结果", "性能对比", "实验对比", "消融", "表格",
    )
    figure_cues = (
        "figure", "fig.", "diagram", "workflow", "pipeline", "framework", "architecture",
        "overview graphic", "visualization", "visualisation", "flow chart", "process visualization",
        "图", "图示", "示意图", "流程图", "框架图", "架构图", "流程可视化", "pipeline",
    )
    algorithm_cues = (
        "algorithm", "pseudocode", "pseudo-code", "procedure listing", "algorithm block",
        "step list", "algorithm steps", "algorithm description", "algorithmic description",
        "rewire", "rewiring", "parent selection", "update rule", "transition rule",
        "代码流程", "伪代码", "算法步骤", "算法流程", "算法描述", "算法细节", "重连",
        "父节点", "更新规则", "迭代规则",
    )
    non_prose_need_cues = (
        "compare metrics", "metric comparison", "quantitative comparison", "performance gap",
        "pipeline/process", "workflow breakdown", "diagrammatic", "visual evidence",
        "算法细节", "流程细节", "图示流程", "对比指标", "定量对比", "非正文证据", "图表证据",
    )

    artifact_types: List[str] = []
    if _contains_any_cue(combined, table_cues):
        artifact_types.append("table")
    if _contains_any_cue(combined, figure_cues):
        artifact_types.append("figure")
    if _contains_any_cue(combined, algorithm_cues):
        artifact_types.append("algorithm")

    needs_artifact = bool(artifact_types) or _contains_any_cue(combined, non_prose_need_cues)
    if not needs_artifact:
        return {"needs_artifact": False, "artifact_types": [], "reason": ""}

    if artifact_types:
        reason = "Question appears to depend on non-prose evidence and the artifact type can be narrowed."
    else:
        reason = "Question appears to depend on non-prose evidence; use relevant artifact retrieval without enumerating all artifact types."
    return {
        "needs_artifact": True,
        "artifact_types": artifact_types[:2],
        "reason": reason,
    }


def infer_supporting_section_need(*texts: str) -> Dict[str, str]:
    """Identify when an artifact needs nearby prose context to answer reliably."""
    combined = " ".join(str(text or "") for text in texts if str(text or "").strip()).lower()
    artifact_need = infer_artifact_evidence_need(combined)
    if not artifact_need["needs_artifact"]:
        return {"needed": "", "section_query": ""}

    explanation_cues = (
        "why", "how", "mechanism", "explain", "interpret", "reason", "impact", "tradeoff",
        "为什么", "为何", "如何", "机制", "原理", "解释", "解读", "影响", "原因", "作用",
    )
    if not _contains_any_cue(combined, explanation_cues):
        return {"needed": "", "section_query": ""}

    artifact_types = set(artifact_need["artifact_types"])
    if "algorithm" in artifact_types:
        return {"needed": "method_context", "section_query": "method or algorithm"}
    if "table" in artifact_types or "figure" in artifact_types:
        return {"needed": "evaluation_context", "section_query": "experiments or results"}
    return {"needed": "", "section_query": ""}


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
    local_section_required = bool(
        explicit_local_required
        and (
            _contains_any_cue(text, _LOCAL_SECTION_CUES)
            or bool(re.search(r"(?:第\s*\d+\s*节|section\s*\d+)", text, flags=re.IGNORECASE))
        )
    )
    artifact_need = infer_artifact_evidence_need(text)
    local_artifact_required = bool(explicit_local_required and artifact_need["needs_artifact"])
    local_kb_required = bool(explicit_local_required and not local_section_required and not local_artifact_required)
    local_scope_required = bool(local_kb_required or local_section_required or local_artifact_required)
    external_academic_required = _contains_any_cue(text, _ACADEMIC_EXTERNAL_CUES)
    web_need = infer_general_web_evidence_need(text)
    general_web_required = bool(
        web_need["needs_web"]
        and (bool(web_need["volatile"]) or not external_academic_required)
    )
    citation_required = _contains_any_cue(
        text,
        ("doi", "author", "authors", "year", "venue", "citation", "citations", "作者", "年份", "期刊", "会议", "引用"),
    )
    freshness_required = general_web_required

    required_source_types: List[str] = []
    preferred_source_types: List[str] = []
    forbidden_source_types: List[str] = []

    if local_kb_required:
        required_source_types.append("local_kb")
    if local_section_required:
        required_source_types.append("local_section")
        preferred_source_types.append("local_section")
    if local_artifact_required:
        required_source_types.append("local_artifact")
        preferred_source_types.append("local_artifact")

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
        "local_kb_required": local_kb_required,
        "local_section_required": local_section_required,
        "local_artifact_required": local_artifact_required,
        "requested_artifact_types": list(artifact_need["artifact_types"]),
        "external_scope_required": bool(external_academic_required or general_web_required),
        "general_web_reason": str(web_need["reason"]),
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

    narrow_local_sources = {"local_section", "local_artifact"}
    required_set = set(required)
    if required_set and required_set.issubset(narrow_local_sources):
        compatible_steps = [
            step for step in compatible_steps
            if _TOOL_SOURCE_TYPE.get(str(step.tool)) in required_set
        ]

    default_steps = {
        "local_kb": RetrievalStep(
            tool="hybrid_search", query=question, limit=10,
            reason="Required local paper evidence.",
        ),
        "local_section": RetrievalStep(
            tool="section_search", query=question, section_query=question, limit=8,
            reason="Required section-scoped local evidence.",
        ),
        "local_artifact": RetrievalStep(
            tool="artifact_search", query=question, limit=6,
            artifact_types=list(source_requirements.get("requested_artifact_types") or []),
            reason="Required local non-prose evidence.",
        ),
        "external_academic": RetrievalStep(
            tool="openalex_search", query=question, limit=5,
            reason="Required external academic metadata or literature discovery.",
        ),
        "general_web": RetrievalStep(
            tool="web_search", query=question, limit=5,
            reason="Required current or volatile real-world fact verification.",
        ),
    }

    compatible_sources = {
        _TOOL_SOURCE_TYPE.get(str(step.tool)) for step in compatible_steps
    }
    for source_type in required:
        if source_type in compatible_sources or not _is_source_type_available(source_type, capabilities):
            continue
        default_step = default_steps.get(source_type)
        if default_step is not None:
            compatible_steps.append(default_step)
            compatible_sources.add(source_type)
            plan.needs_retrieval = True
            plan.direct_answer_allowed = False

    if "general_web" in required and "local_kb" not in required:
        compatible_steps = [
            step for step in compatible_steps
            if _TOOL_SOURCE_TYPE.get(str(step.tool)) == "general_web"
        ]

    if "general_web" in required and any(
        _TOOL_SOURCE_TYPE.get(str(step.tool)) == "general_web" for step in compatible_steps
    ):
        plan.intent = "web_information"
        plan.max_tools = max(1, min(capabilities.max_tools, plan.max_tools or 1))

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
        intent="unclear",
        needs_retrieval=False,
        retrieval_steps=[],
        max_tools=0,
        allow_external_sources=False,
        direct_answer_allowed=False,
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
    has_volatile_real_world_cues = bool(infer_general_web_evidence_need(normalized_question)["needs_web"])
    has_academic_external_cues = _contains_any_cue(lower_question, _ACADEMIC_EXTERNAL_CUES)
    artifact_need = infer_artifact_evidence_need(normalized_question)
    graph_need = infer_paper_graph_need(normalized_question)
    direct_eligibility = infer_direct_answer_eligibility(normalized_question)

    if has_external_info_cues or has_volatile_real_world_cues:
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
                use_paper_graph=bool(graph_need["use_paper_graph"]),
                graph_usage_reason=str(graph_need["reason"]),
                graph_relation_types=list(graph_need["relation_types"]),
                graph_direction=str(graph_need["direction"]),
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
                use_paper_graph=bool(graph_need["use_paper_graph"]),
                graph_usage_reason=str(graph_need["reason"]),
                graph_relation_types=list(graph_need["relation_types"]),
                graph_direction=str(graph_need["direction"]),
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
            return _external_unavailable_direct_answer_plan(
                warning_codes=[
                    "planner_fallback_used",
                    "fallback_external_retrieval_unavailable",
                    "external_retrieval_unavailable",
                ],
                reason=reason or "Fallback cannot verify: required external evidence is unavailable.",
            )

    if has_local_evidence_cues:
        if artifact_need["needs_artifact"] and caps.artifact_search_enabled:
            return _apply_inferred_paper_graph_need(IntentPlan(
                intent="local_artifact_qa",
                needs_retrieval=True,
                retrieval_steps=[
                    RetrievalStep(
                        tool="artifact_search",
                        query=normalized_question,
                        limit=6,
                        artifact_types=artifact_need["artifact_types"],
                        reason=str(artifact_need["reason"] or "Fallback local artifact retrieval for relevant non-prose evidence."),
                    )
                ],
                max_tools=min(1, caps.max_tools),
                allow_external_sources=False,
                direct_answer_allowed=False,
                reason=reason or "Fallback local artifact retrieval: question appears to require non-prose local evidence.",
                warnings=["planner_fallback_used"],
            ), graph_need)
        if caps.hybrid_search_enabled:
            return _apply_inferred_paper_graph_need(IntentPlan(
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
            ), graph_need)
        if caps.vector_search_enabled:
            return _apply_inferred_paper_graph_need(IntentPlan(
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
            ), graph_need)
        if caps.direct_answer_enabled:
            return _external_unavailable_direct_answer_plan(
                warning_codes=[
                    "planner_fallback_used",
                    "fallback_local_retrieval_unavailable",
                    "local_retrieval_unavailable",
                ],
                reason=reason or "Fallback cannot verify: required local document evidence is unavailable.",
            )

    if caps.direct_answer_enabled and direct_eligibility["allowed"]:
        return IntentPlan(
            intent="direct_answer",
            needs_retrieval=False,
            retrieval_steps=[],
            max_tools=0,
            allow_external_sources=False,
            direct_answer_allowed=True,
            reason=reason or f"Fallback direct answer: {direct_eligibility['reason']}",
            warnings=["planner_fallback_used"],
        )
    if caps.direct_answer_enabled:
        return IntentPlan(
            intent="unclear",
            needs_retrieval=False,
            retrieval_steps=[],
            max_tools=0,
            allow_external_sources=False,
            direct_answer_allowed=False,
            reason=reason or "Fallback clarification: question is not eligible for a factual direct answer and no source requirement was identified.",
            warnings=["planner_fallback_used", "direct_answer_not_eligible"],
        )
    return _safe_default_plan(question)


def _safe_default_plan_with_caps(question: str, capabilities: PlannerCapabilities) -> IntentPlan:
    if capabilities.direct_answer_enabled and not capabilities.available_tools():
        direct_eligibility = infer_direct_answer_eligibility(question)
        return IntentPlan(
            intent="direct_answer" if direct_eligibility["allowed"] else "unclear",
            needs_retrieval=False,
            retrieval_steps=[],
            max_tools=0,
            allow_external_sources=False,
            direct_answer_allowed=bool(direct_eligibility["allowed"]),
            reason=(
                "Fallback direct answer: no retrieval tool is available, but the question is eligible for direct response."
                if direct_eligibility["allowed"]
                else "Fallback clarification: no retrieval tool is available for this evidence-dependent question."
            ),
            warnings=(
                ["planner_fallback_no_tools"]
                if direct_eligibility["allowed"]
                else ["planner_fallback_no_tools", "direct_answer_not_eligible"]
            ),
        )
    return build_fallback_intent_plan(question, capabilities=capabilities)


def _safe_retry_plan(question: str, missing_aspects: List[str], suggested_query: str = "") -> IntentPlan:
    query = str(suggested_query or "").strip() or str(question or "").strip()
    missing_text = " ".join(str(x) for x in (missing_aspects or []))
    artifact_need = infer_artifact_evidence_need(question, missing_text, query)
    if artifact_need["needs_artifact"]:
        return IntentPlan(
            intent="local_artifact_qa",
            needs_retrieval=True,
            retrieval_steps=[
                RetrievalStep(
                    tool="artifact_search",
                    query=query,
                    limit=6,
                    artifact_types=artifact_need["artifact_types"],
                    reason=str(artifact_need["reason"] or "Fallback retry plan for relevant artifact evidence gaps."),
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
    explicit_question_text = ""
    has_explicit_question = False
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
        has_explicit_question = isinstance(raw.get("question"), str) and bool(str(raw.get("question") or "").strip())
        explicit_question_text = str(raw.get("question") or "").strip()

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

        try:
            graph_neighbor_limit = int(raw.get("graph_neighbor_limit", 6) or 6)
        except (TypeError, ValueError):
            graph_neighbor_limit = 6
        allowed_graph_relations = {"semantic_similarity", "cites", "method_lineage"}
        graph_relation_types = [
            relation
            for relation in (str(value or "").strip() for value in (raw.get("graph_relation_types") or []))
            if relation in allowed_graph_relations
        ]
        graph_direction = str(raw.get("graph_direction") or "both").strip().lower()
        if graph_direction not in {"both", "outgoing", "incoming"}:
            graph_direction = "both"

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
            use_paper_graph=bool(raw.get("use_paper_graph", False)),
            graph_usage_reason=str(raw.get("graph_usage_reason") or ""),
            graph_relation_types=graph_relation_types,  # type: ignore[arg-type]
            graph_direction=graph_direction,  # type: ignore[arg-type]
            graph_neighbor_limit=max(1, min(12, graph_neighbor_limit)),
            reason=str(raw.get("reason") or ""),
            warnings=list(raw.get("warnings") or []),
        )

    policy_question = question_for_policy or cue_text
    direct_eligibility = infer_direct_answer_eligibility(policy_question)
    explicit_direct_request = (
        any(marker in str(policy_question).lower() for marker in ("不检索", "不引用本地论文", "不引用外部网页", "不引用资料"))
        and any(marker in str(policy_question).lower() for marker in ("解释", "概念", "原理", "什么是", "通俗"))
    )
    plan = apply_explicit_routing_guard(plan, explicit_question_text or policy_question, caps)
    if not plan.needs_retrieval:
        plan.retrieval_steps = []
        plan.max_tools = 0
        plan.use_paper_graph = False
        plan.graph_usage_reason = ""
        plan.graph_relation_types = []
        plan.graph_direction = "both"
        if caps.direct_answer_enabled and (explicit_direct_request or not has_explicit_question or direct_eligibility["allowed"]):
            plan.intent = "direct_answer"
            plan.direct_answer_allowed = True
        else:
            plan.intent = "unclear"
            plan.direct_answer_allowed = False
            _append_warning_once(plan.warnings, "direct_answer_not_eligible")
    elif "planner_fallback_used" in plan.warnings and not plan.use_paper_graph:
        graph_need = infer_paper_graph_need(question_for_policy or cue_text)
        if graph_need["use_paper_graph"]:
            plan.use_paper_graph = True
            plan.graph_usage_reason = str(graph_need["reason"])
            plan.graph_relation_types = list(graph_need["relation_types"])
            plan.graph_direction = str(graph_need["direction"])
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
        artifact_need = infer_artifact_evidence_need(question_for_policy or cue_text)

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
        elif caps.direct_answer_enabled and plan.direct_answer_allowed and (
            not has_explicit_question or direct_eligibility["allowed"]
        ):
            plan.needs_retrieval = False
        elif has_local_evidence_cues and artifact_need["needs_artifact"] and caps.artifact_search_enabled:
            plan.intent = "local_artifact_qa"
            plan.retrieval_steps = [
                RetrievalStep(
                    tool="artifact_search",
                    query="",
                    limit=6,
                    artifact_types=artifact_need["artifact_types"],
                    reason=str(artifact_need["reason"] or "Fallback to relevant local artifact retrieval after tool filtering."),
                )
            ]
            plan.max_tools = min(1, caps.max_tools)
            _append_warning_once(plan.warnings, "fallback_to_artifact_after_filter")
            plan.needs_retrieval = True
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
    if plan.intent == "direct_answer" and not plan.needs_retrieval and plan.direct_answer_allowed:
        source_requirements = {
            "required_source_types": [], "preferred_source_types": [], "forbidden_source_types": [],
            "freshness_required": False, "citation_required": False, "local_scope_required": False,
            "local_kb_required": False, "local_section_required": False, "local_artifact_required": False,
            "requested_artifact_types": [], "external_scope_required": False,
            "general_web_reason": "", "can_answer_without_retrieval": True,
        }
        answer_policy = {
            "mode": "direct_answer", "allowed_source_types": ["model_knowledge"],
            "blocked_source_types": ["local_kb", "local_section", "local_artifact", "external_academic", "general_web"],
            "unavailable_required_sources": [], "must_disclose_limitations": False,
            "answer_boundary": "stable_general_explanation", "guidance_to_answer_agent": "Answer directly without retrieval or source claims.",
        }
        revised_plan = plan
    elif plan.reason == "Explicit paper and section scope.":
        source_requirements = {
            "required_source_types": ["local_section"], "preferred_source_types": ["local_section"],
            "forbidden_source_types": ["general_web", "external_academic"], "freshness_required": False,
            "citation_required": True, "local_scope_required": True, "local_kb_required": False,
            "local_section_required": True, "local_artifact_required": False, "requested_artifact_types": [],
            "external_scope_required": False, "general_web_reason": "", "can_answer_without_retrieval": False,
        }
        answer_policy = {
            "mode": "retrieve_and_answer", "allowed_source_types": ["local_section", "model_knowledge"],
            "blocked_source_types": ["local_kb", "local_artifact", "external_academic", "general_web"],
            "unavailable_required_sources": [], "must_disclose_limitations": False,
            "answer_boundary": "named_paper_sections_only",
            "guidance_to_answer_agent": "Use only the requested local paper sections; do not substitute web evidence.",
        }
        revised_plan = plan
    elif plan.reason in {
        "Explicit named-paper evidence request.",
        "Explicit named-paper non-prose evidence request.",
        "Explicit named-paper experiment or evaluation request.",
    }:
        step_sources = _step_source_types(list(plan.retrieval_steps or []))
        primary_source = step_sources[0] if step_sources else "local_kb"
        required_sources = step_sources or [primary_source]
        is_non_prose_request = plan.reason == "Explicit named-paper non-prose evidence request."
        source_requirements = {
            "required_source_types": required_sources, "preferred_source_types": required_sources,
            "forbidden_source_types": ["general_web", "external_academic"], "freshness_required": False,
            "citation_required": True, "local_scope_required": True,
            "local_kb_required": primary_source == "local_kb",
            "local_section_required": "local_section" in required_sources,
            "local_artifact_required": primary_source == "local_artifact",
            "requested_artifact_types": list(plan.retrieval_steps[0].artifact_types or []) if is_non_prose_request and plan.retrieval_steps else [],
            "external_scope_required": False, "general_web_reason": "", "can_answer_without_retrieval": False,
        }
        answer_policy = {
            "mode": "retrieve_and_answer", "allowed_source_types": [*required_sources, "model_knowledge"],
            "blocked_source_types": ["external_academic", "general_web"], "unavailable_required_sources": [],
            "must_disclose_limitations": False, "answer_boundary": "named_paper_evidence_only",
            "guidance_to_answer_agent": "Use the explicitly selected local paper evidence and preserve source attribution.",
        }
        revised_plan = plan
    elif plan.reason == "Explicit knowledge-star-map relationship request.":
        source_requirements = {
            "required_source_types": ["local_kb"], "preferred_source_types": ["local_kb"],
            "forbidden_source_types": ["general_web", "external_academic"], "freshness_required": False,
            "citation_required": True, "local_scope_required": True, "local_kb_required": True,
            "local_section_required": False, "local_artifact_required": False, "requested_artifact_types": [],
            "external_scope_required": False, "general_web_reason": "", "can_answer_without_retrieval": False,
        }
        answer_policy = {
            "mode": "retrieve_and_answer", "allowed_source_types": ["local_kb", "model_knowledge"],
            "blocked_source_types": ["external_academic", "general_web"], "unavailable_required_sources": [],
            "must_disclose_limitations": False, "answer_boundary": "graph_navigation_with_paper_evidence",
            "guidance_to_answer_agent": "Use graph edges only for navigation, and use both named papers' local evidence for factual comparison.",
        }
        revised_plan = plan
    else:
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
        return _finalize_plan_queries(
            normalize_intent_plan({**raw, "question": question}, capabilities=capabilities),
            question,
        )
    text = _extract_model_text(raw)
    candidate = _extract_json_candidate(text)
    parsed = json.loads(candidate)
    return _finalize_plan_queries(
        normalize_intent_plan({**parsed, "question": question}, capabilities=capabilities),
        question,
    )


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
            (
                {**raw, "question": question}
                if isinstance(raw, dict)
                else raw
            ) if isinstance(raw, (dict, str)) else getattr(raw, "content", ""),
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
