from typing import Any, Dict, List, Optional, Tuple

from .intent_planner import IntentPlan, PlannerCapabilities, RetrievalStep


def summarize_hits_for_planner(hits: List[Dict[str, Any]], max_items: int = 3) -> str:
    if not hits:
        return "No hits."
    lines: List[str] = []
    for i, hit in enumerate(hits[:max_items], start=1):
        title = str(hit.get("document_title") or "Unknown")
        score = hit.get("score")
        content = str(hit.get("content") or "").replace("\n", " ").strip()
        content = content[:180] + ("..." if len(content) > 180 else "")
        lines.append(f"{i}. title={title} score={score} content={content}")
    return "\n".join(lines)


def _normalize_step(step: RetrievalStep, fallback_query: str) -> RetrievalStep:
    q = str(step.query or "").strip() or str(fallback_query or "").strip()
    limit = max(1, min(int(step.limit or 10), 50))
    return RetrievalStep(
        tool=step.tool,
        query=q,
        limit=limit,
        search_type=step.search_type,
        section_query=step.section_query,
        artifact_types=list(step.artifact_types or []),
        document_id=step.document_id,
        reason=step.reason,
    )


def _step_to_tool_call(step: RetrievalStep) -> Tuple[str, Dict[str, Any]]:
    if step.tool == "hybrid_search":
        return "hybrid_search", {"query": step.query, "limit": step.limit}
    if step.tool == "vector_search":
        return "vector_search", {"query": step.query, "limit": step.limit}
    if step.tool == "section_search":
        return "section_search", {
            "query": step.query,
            "section_query": step.section_query or step.query,
            "document_id": step.document_id,
            "limit": step.limit,
        }
    if step.tool == "artifact_search":
        return "artifact_search", {
            "query": step.query,
            "limit": step.limit,
            "artifact_types": list(step.artifact_types or ["table", "figure", "algorithm"]),
            "document_id": step.document_id,
        }
    if step.tool == "openalex_search":
        return "search_openalex_papers", {"query": step.query, "limit": max(1, min(step.limit, 10))}
    if step.tool == "web_search":
        return "search_web", {"query": step.query, "limit": max(1, min(step.limit, 10))}
    return "none", {}


def _normalize_external_result(item: Dict[str, Any]) -> Dict[str, Any]:
    title = str(item.get("title") or "External Source")
    snippet = str(item.get("snippet") or item.get("abstract") or title)
    source = str(item.get("source") or item.get("provider") or "web")
    doc_id = str(item.get("openalex_id") or item.get("url") or title)
    return {
        "chunk_id": doc_id,
        "document_id": doc_id,
        "content": snippet,
        "score": 0.0,
        "metadata": {"source_type": "web", **({} if not isinstance(item, dict) else item)},
        "document_title": title,
        "document_source": source,
    }


async def execute_intent_plan_steps(
    plan: IntentPlan,
    tools: List[Any],
    fallback_query: str,
    warnings: Optional[List[str]] = None,
    capabilities: Optional[PlannerCapabilities] = None,
) -> Dict[str, Any]:
    warnings = list(warnings or [])
    caps = capabilities or PlannerCapabilities()
    planned = [_normalize_step(s, fallback_query) for s in list(plan.retrieval_steps or [])[: caps.max_tools]]
    planned_raw = [s.model_dump() for s in planned]
    tool_map = {getattr(t, "name", ""): t for t in tools}
    results: List[Dict[str, Any]] = []
    tools_executed: List[Dict[str, Any]] = []
    filtered_unavailable_tools: List[str] = []
    seen = set()
    allowed_internal = {
        "hybrid_search": caps.hybrid_search_enabled,
        "vector_search": caps.vector_search_enabled,
        "section_search": caps.section_search_enabled,
        "artifact_search": caps.artifact_search_enabled,
        "search_openalex_papers": caps.openalex_search_enabled,
        "search_web": caps.web_search_enabled,
    }

    for step in planned:
        tool_name, args = _step_to_tool_call(step)
        if tool_name == "none":
            continue
        if not allowed_internal.get(tool_name, False):
            warnings.append(f"tool_unavailable:{tool_name}")
            filtered_unavailable_tools.append(tool_name)
            continue
        dedupe_key = f"{tool_name}|{args.get('query','')}|{args.get('document_id','')}"
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        tool = tool_map.get(tool_name)
        if tool is None:
            warnings.append(f"planned tool unavailable: {tool_name}")
            continue
        try:
            out = list(await tool.ainvoke(args) or [])
            if tool_name in {"search_openalex_papers", "search_web"}:
                out = [_normalize_external_result(x) for x in out if isinstance(x, dict)]
            results.extend(out)
            tools_executed.append({"tool": tool_name, "args": args})
        except Exception as exc:
            warnings.append(f"planned tool failed: {tool_name}: {exc}")
            continue

    return {
        "results": results,
        "tools_executed": tools_executed,
        "planned_steps": planned_raw,
        "filtered_unavailable_tools": filtered_unavailable_tools,
        "warnings": warnings,
    }
