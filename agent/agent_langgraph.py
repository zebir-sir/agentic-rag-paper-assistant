from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from .agent_langchain import get_langchain_chat_model
from .agent_runtime import AgentDependencies
from .langchain_tools import build_langchain_tools
from .models import EvidenceSource, ToolCall
from .prompts import SYSTEM_PROMPT
from .tools import DocumentListInput, list_documents_tool


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


async def _emit_progress(state: LangGraphAnalysisState, message: str) -> None:
    callback = state.get("progress_callback")
    if callback is None:
        return
    await callback(message)


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


async def inspect_documents_node(state: LangGraphAnalysisState) -> LangGraphAnalysisState:
    next_state = dict(state)
    await _emit_progress(next_state, "正在检查知识库文档...")
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


async def local_retrieval_node(state: LangGraphAnalysisState) -> LangGraphAnalysisState:
    next_state = dict(state)
    await _emit_progress(next_state, "正在检索本地知识库...")
    deps = next_state.get("deps")
    question = str(next_state.get("question") or "").strip()
    if not deps or not question:
        next_state["retrieval_results"] = []
        if not question:
            _append_warning(next_state, "Empty question; skip retrieval.")
        await _emit_progress(next_state, "已找到 0 条相关片段")
        return next_state

    try:
        tools = build_langchain_tools(deps)
        search_tool = _find_tool_by_name(tools, "search_knowledge_base")
        if search_tool is None:
            next_state["retrieval_results"] = []
            _append_warning(next_state, "search_knowledge_base tool not found.")
            return next_state

        results = await search_tool.ainvoke({"query": question, "limit": 3})
        next_state["retrieval_results"] = list(results or [])
        next_state["sources"] = list(deps.retrieved_sources)
        _append_tool_call(next_state, "search_knowledge_base", {"query": question, "limit": 3})
        await _emit_progress(next_state, f"已找到 {len(next_state['retrieval_results'])} 条相关片段")
    except Exception as exc:
        next_state["retrieval_results"] = []
        next_state["sources"] = list(getattr(deps, "retrieved_sources", []) or [])
        _append_warning(next_state, f"local retrieval failed: {exc}")
        await _emit_progress(next_state, "已找到 0 条相关片段")
    return next_state


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
        return "未检索到相关片段。"
    lines: List[str] = []
    for idx, hit in enumerate(results[:3], start=1):
        title = str(hit.get("document_title") or "Untitled").strip()
        snippet = str(hit.get("content") or "").strip().replace("\n", " ")
        snippet = snippet[:220] + ("..." if len(snippet) > 220 else "")
        score = hit.get("score")
        lines.append(f"{idx}. {title} | score={score} | snippet={snippet}")
    return "\n".join(lines)


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


def _humanize_warning(warning: str) -> Optional[str]:
    if "No retrieval evidence found" in warning:
        return "当前没有检索到直接相关片段，以下内容更适合作为一般性分析参考。"
    if "strong evidence wording but no sources" in warning:
        return "当前来源不足以支撑过强的论文式表述，回答已按保守方式理解。"
    return None


async def generate_analysis_node(state: LangGraphAnalysisState) -> LangGraphAnalysisState:
    next_state = dict(state)
    await _emit_progress(next_state, "正在整合证据并生成分析...")
    question = str(next_state.get("question") or "").strip()
    context_prompt = str(next_state.get("context_prompt") or "").strip()
    documents = list(next_state.get("documents") or [])
    retrieval_results = list(next_state.get("retrieval_results") or [])
    warnings = list(next_state.get("warnings") or [])

    context_block = ""
    if context_prompt:
        context_block = (
            "会话上下文（仅用于指代消解，不作为文档证据）：\n"
            f"{context_prompt}\n\n"
        )

    prompt = (
        f"{context_block}"
        f"用户问题：\n{question}\n\n"
        f"文档列表（最多5条）：\n{_summarize_documents(documents)}\n\n"
        f"检索结果（最多3条）：\n{_summarize_retrieval_results(retrieval_results)}\n\n"
        f"当前告警：\n{chr(10).join(warnings) if warnings else '无'}\n\n"
        "请用中文给出简洁、准确、较严谨的深度分析回答。"
        "若某个具体问题缺少证据，只在对应部分简短说明不确定或文中未明确，并给出保守建议。"
        "会话上下文只能用于理解指代关系，不能作为论文证据。"
        "不要编造具体论文细节，不要输出完整思考过程。"
        "回答结构清晰即可，可使用标题和分点，但不必强制固定模板。"
        "不要在已有证据的部分反复说证据不足，也不要与已有证据矛盾。"
    )

    system_text = (
        SYSTEM_PROMPT
        + "\n\n你现在处于 LangGraph 深度分析工作流中。"
        + "请基于已提供的文档列表和检索结果进行较严谨的分析。"
        + "若某个具体点证据不足，只在对应部分简短说明；不要让严格证据风格污染整篇回答。"
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
        fallback = "当前分析生成失败。请稍后重试，或先使用普通问答模式。"
        if not retrieval_results:
            fallback += " 当前没有检索到直接相关片段。"
        next_state["draft_answer"] = fallback
    return next_state


async def evidence_check_node(state: LangGraphAnalysisState) -> LangGraphAnalysisState:
    next_state = dict(state)
    await _emit_progress(next_state, "正在核对证据是否支撑结论...")
    sources = list(next_state.get("sources") or [])
    retrieval_results = list(next_state.get("retrieval_results") or [])
    draft_answer = str(next_state.get("draft_answer") or "")

    if not retrieval_results:
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
    metadata["retrieval_result_count"] = len(retrieval_results)
    next_state["metadata"] = metadata
    return next_state


async def finalize_node(state: LangGraphAnalysisState) -> LangGraphAnalysisState:
    next_state = dict(state)
    await _emit_progress(next_state, "正在整理最终回答...")
    deps = next_state.get("deps")
    warnings = list(next_state.get("warnings") or [])
    answer = str(next_state.get("draft_answer") or "").strip()

    if not answer:
        answer = "当前无法生成有效分析结果。"

    if warnings:
        notes = [text for text in (_humanize_warning(item) for item in warnings[:2]) if text]
        if notes:
            answer = answer + "\n\n说明：" + "；".join(notes)

    next_state["final_answer"] = answer
    next_state["sources"] = list(getattr(deps, "retrieved_sources", []) or [])

    metadata = dict(next_state.get("metadata") or {})
    metadata["agent_backend"] = "langgraph"
    metadata["workflow"] = "deep_analysis"
    metadata["tool_count"] = len(list(next_state.get("tools_used") or []))
    next_state["metadata"] = metadata
    return next_state


def build_langgraph_workflow():
    graph = StateGraph(LangGraphAnalysisState)
    graph.add_node("inspect_documents", inspect_documents_node)
    graph.add_node("local_retrieval", local_retrieval_node)
    graph.add_node("generate_analysis", generate_analysis_node)
    graph.add_node("evidence_check", evidence_check_node)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "inspect_documents")
    graph.add_edge("inspect_documents", "local_retrieval")
    graph.add_edge("local_retrieval", "generate_analysis")
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
    }

    if progress_callback is not None:
        await progress_callback("正在识别任务类型...")
        await progress_callback("正在判断需要哪些工具...")
    final_state = await graph.ainvoke(initial_state)
    message = str(final_state.get("final_answer") or final_state.get("draft_answer") or "").strip()
    tools_used = list(final_state.get("tools_used") or [])
    sources = list(final_state.get("sources") or list(deps.retrieved_sources))
    metadata = dict(final_state.get("metadata") or {})

    return LangGraphAnalysisResult(
        message=message,
        raw_state=dict(final_state),
        tools_used=tools_used,
        sources=sources,
        metadata=metadata,
    )
