import os
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .agent_runtime import AgentDependencies
from .models import EvidenceSource, ToolCall
from .tools import OpenAlexSearchInput, openalex_search_tool


@dataclass
class OpenAlexAnswerPlan:
    query: str
    fetch_limit: int = 8
    display_count: Optional[int] = None
    selection_mode: str = "best"
    need_summary: bool = False


def _is_openalex_enabled() -> bool:
    return bool(os.getenv("OPENALEX_API_KEY", "").strip())


def _is_explicit_web_paper_request(message: str) -> bool:
    text = str(message or "").lower().strip()
    if not text:
        return False
    keywords = [
        "openalex",
        "联网",
        "网上",
        "网络",
        "不要使用本地",
        "不要用本地",
        "只使用 openalex",
        "必须使用 openalex",
        "检索一篇论文",
        "搜索一篇论文",
        "随机检索",
        "推荐论文",
        "最新论文",
        "related work",
        "external paper",
        "随机联网",
        "只用 openalex",
        "只用联网",
        "不要本地知识库",
        "不用本地知识库",
        "知识库外",
        "web search",
        "online search",
    ]
    return any(keyword in text for keyword in keywords)


def _build_openalex_query(message: str) -> str:
    text = str(message or "").lower()
    tokens: List[str] = []

    def add(token: str):
        if token not in tokens:
            tokens.append(token)

    if "usv" in text:
        add("USV")
    if "ship" in text or "船舶" in text:
        add("ship")
    if "auv" in text or "水下" in text:
        add("AUV")
    if "rrt" in text:
        add("RRT")
    if "path planning" in text or "路径规划" in text:
        add("path planning")
    if "colregs" in text:
        add("COLREGS")
    if "ocean current" in text or "海流" in text or "洋流" in text:
        add("ocean current")

    return " ".join(tokens) if tokens else "USV ship path planning RRT"


def _infer_openalex_answer_plan(message: str) -> OpenAlexAnswerPlan:
    query = _build_openalex_query(message)
    text = str(message or "").lower()
    plan = OpenAlexAnswerPlan(query=query, fetch_limit=8, display_count=None)

    if any(k in text for k in ["随机", "random"]):
        plan.selection_mode = "random"

    if any(k in message for k in ["总结", "分析", "介绍", "解读", "评价"]):
        plan.need_summary = True

    explicit_counts = [
        ("一篇", 1), ("1篇", 1), ("一个", 1),
        ("两篇", 2), ("2篇", 2),
        ("三篇", 3), ("3篇", 3),
        ("五篇", 5), ("5篇", 5),
    ]
    for key, count in explicit_counts:
        if key in message:
            plan.display_count = count
            break

    if plan.display_count is None:
        if any(k in text for k in ["related work", "几篇", "多篇", "推荐论文", "相关论文"]):
            plan.display_count = 5
        else:
            plan.display_count = 3

    return plan


def _select_openalex_results(
    results: List[Dict[str, Any]],
    plan: OpenAlexAnswerPlan,
    message: str,
) -> List[Dict[str, Any]]:
    display_count = min(int(plan.display_count or 0), len(results))
    if display_count <= 0:
        return []
    if plan.selection_mode == "random":
        rng = random.Random(hash(message) & 0xFFFFFFFF)
        shuffled = list(results)
        rng.shuffle(shuffled)
        return shuffled[:display_count]
    return list(results)[:display_count]


def _format_openalex_first_response(
    results: List[Dict[str, Any]],
    plan: OpenAlexAnswerPlan,
    candidate_count: int,
) -> str:
    lines: List[str] = []
    if len(results) == 1:
        if plan.selection_mode == "random":
            lines.append("我从 OpenAlex 检索候选中选出了一篇论文：")
        else:
            lines.append("下面是从 OpenAlex 候选结果中选出的一篇论文：")
    else:
        if plan.selection_mode == "random":
            lines.append("下面是从 OpenAlex 检索候选中随机选出的若干篇论文：")
        elif any(k in plan.query.lower() for k in ["latest", "recent"]):
            lines.append("下面是按 OpenAlex 检索候选顺序列出的若干篇论文：")
        else:
            lines.append("下面是从 OpenAlex 检索候选中选出的若干篇论文：")
    lines.append(f"检索候选数：{candidate_count}，展示数：{len(results)}。")
    for idx, item in enumerate(results, start=1):
        title = str(item.get("title") or "N/A")
        year = item.get("year")
        year_text = str(year) if year is not None else "N/A"
        authors = item.get("authors") if isinstance(item.get("authors"), list) else []
        authors_text = ", ".join([str(a) for a in authors[:8]]) if authors else "N/A"
        source = str(item.get("source") or "N/A")
        doi = str(item.get("doi") or "N/A")
        link = str(item.get("landing_page_url") or item.get("openalex_id") or "N/A")
        has_pdf = "是" if bool(item.get("has_pdf")) else "否"
        lines.extend(
            [
                f"",
                f"{idx}. {title}",
                f"- 年份：{year_text}",
                f"- 作者：{authors_text}",
                f"- 来源：{source}",
                f"- DOI：{doi}",
                f"- 来源链接：{link}",
                f"- 是否有 PDF：{has_pdf}",
            ]
        )
    return "\n".join(lines).strip()


async def _run_openalex_first_if_needed(
    message: str,
    deps: AgentDependencies,
) -> Optional[Tuple[str, List[ToolCall], List[EvidenceSource], Dict[str, Any]]]:
    if not _is_explicit_web_paper_request(message):
        return None
    if not _is_openalex_enabled():
        response = "你要求使用 OpenAlex 联网搜索，但后端未配置 OPENALEX_API_KEY。"
        metadata = {
            "agent_backend": "openalex_first",
            "openalex_forced": True,
            "openalex_query": "",
            "source_count": 0,
        }
        return response, [], [], metadata

    plan = _infer_openalex_answer_plan(message)
    results = await openalex_search_tool(OpenAlexSearchInput(query=plan.query, limit=plan.fetch_limit))
    candidate_count = len(results)
    if not results:
        response = f"已按要求只使用 OpenAlex 联网搜索，但没有检索到与 `{plan.query}` 匹配的论文。请换关键词。"
        tools_used = [
            ToolCall(
                tool_name="search_openalex_papers",
                args={"query": plan.query, "limit": plan.fetch_limit},
                tool_call_id=None,
            )
        ]
        metadata = {
            "agent_backend": "openalex_first",
            "openalex_forced": True,
            "openalex_query": plan.query,
            "fetch_limit": plan.fetch_limit,
            "display_count": int(plan.display_count or 0),
            "candidate_count": 0,
            "selection_mode": plan.selection_mode,
            "source_count": 0,
        }
        return response, tools_used, [], metadata

    selected_results = _select_openalex_results(results, plan, message)
    sources: List[EvidenceSource] = []
    for item in selected_results:
        title = str(item.get("title") or "OpenAlex Result").strip() or "OpenAlex Result"
        venue = str(item.get("source") or "OpenAlex").strip() or "OpenAlex"
        snippet = str(item.get("abstract") or "").strip()
        if not snippet:
            year = item.get("year")
            snippet = f"{title} ({year if year is not None else 'N/A'})"
        source_link = str(item.get("landing_page_url") or item.get("openalex_id") or "").strip()
        openalex_id = str(item.get("openalex_id") or "").strip()
        sources.append(
            EvidenceSource(
                source_type="web",
                document_id=openalex_id or None,
                document_title=title,
                document_source=venue,
                chunk_id=openalex_id or None,
                snippet=snippet,
                score=None,
                metadata={
                    "source_type": "web",
                    "source_kind": "openalex",
                    "venue": venue,
                    "year": item.get("year"),
                    "authors": item.get("authors") or [],
                    "doi": item.get("doi"),
                    "source": venue,
                    "landing_page_url": source_link,
                    "pdf_url": item.get("pdf_url"),
                    "openalex_id": item.get("openalex_id"),
                    "is_oa": item.get("is_oa"),
                    "has_pdf": bool(item.get("has_pdf")),
                    "has_fulltext": bool(item.get("has_fulltext")),
                    "cited_by_count": item.get("cited_by_count"),
                },
            )
        )

    deps.retrieved_sources = list(sources)
    tools_used = [
        ToolCall(
            tool_name="search_openalex_papers",
            args={"query": plan.query, "limit": plan.fetch_limit},
            tool_call_id=None,
        )
    ]
    metadata = {
        "agent_backend": "openalex_first",
        "openalex_forced": True,
        "openalex_query": plan.query,
        "fetch_limit": plan.fetch_limit,
        "display_count": len(selected_results),
        "candidate_count": candidate_count,
        "selection_mode": plan.selection_mode,
        "source_count": len(sources),
    }
    response = _format_openalex_first_response(selected_results, plan, candidate_count)
    return response, tools_used, sources, metadata


def _split_openalex_stream_chunks(text: str) -> List[str]:
    value = text or ""
    if not value.strip():
        return []
    lines = value.splitlines(keepends=True)
    chunks: List[str] = []
    current: List[str] = []
    for line in lines:
        if re.match(r"^\d+\.\s", line) and current:
            chunks.append("".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        chunks.append("".join(current))
    if len(chunks) <= 1 and len(value) > 800:
        chunks = [part + "\n\n" for part in value.split("\n\n") if part.strip()]
    return chunks or [value]
