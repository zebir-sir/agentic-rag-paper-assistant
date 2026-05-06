import logging
import asyncio
import os
import re
from typing import List

from .agent_runtime import AgentDependencies
from .langchain_tools import build_langchain_tools
from .models import EvidenceSource
from .openalex_router import _is_explicit_web_paper_request
from common.title_aliases import TITLE_ALIASES, get_title_alias

logger = logging.getLogger(__name__)
LOCAL_PREFLIGHT_TIMEOUT_SECONDS = float(os.getenv("LOCAL_PREFLIGHT_TIMEOUT_SECONDS", "20"))
LOCAL_PREFLIGHT_QUERY_TIMEOUT_SECONDS = min(8.0, LOCAL_PREFLIGHT_TIMEOUT_SECONDS)


def _is_general_algorithm_question(message: str) -> bool:
    text = str(message or "").lower()
    has_algorithm = any(
        k in text for k in ["rrt", "rrt*", "a*", "dijkstra", "prm", "apf", "人工势场", "算法区别", "区别"]
    )
    has_paper_intent = any(
        k in str(message or "")
        for k in ["论文", "文献", "这篇", "该论文", "HMA-RRT", "HA-RRT", "总结", "创新点", "实验", "局限"]
    )
    return has_algorithm and not has_paper_intent


def _is_local_kb_question(message: str) -> bool:
    if _is_explicit_web_paper_request(message):
        return False
    if _is_general_algorithm_question(message):
        return False
    text = str(message or "").lower()
    keywords = [
        "论文", "文献", "总结", "创新", "方法", "实验", "局限", "启发", "对比",
        "hma-rrt", "ha-rrt", "这篇", "该论文", "路径规划论文",
    ]
    return any(keyword in text for keyword in keywords)


def _extract_known_local_paper_queries(message: str) -> List[str]:
    text = str(message or "").lower()
    queries: List[str] = []

    hma_hit = bool(re.search(r"(?<![a-z0-9])hma-rrt(?![a-z0-9])", text))
    ha_hit = bool(re.search(r"(?<![a-z0-9])ha-rrt(?![a-z0-9])", text))

    if hma_hit:
        alias = get_title_alias("1-s2.0-s002980182403244x-main") or TITLE_ALIASES["1-s2.0-s002980182403244x-main"]
        queries.append(alias)
        queries.append("1-s2.0-s002980182403244x-main")
    if ha_hit:
        alias = get_title_alias("s44443-025-00393-9") or TITLE_ALIASES["s44443-025-00393-9"]
        queries.append(alias)
        queries.append("s44443-025-00393-9")
    return queries


def _may_need_general_web_search(message: str) -> bool:
    if _is_explicit_web_paper_request(message):
        return False
    text = str(message or "").lower()
    keywords = [
        "联网", "网上", "搜索", "查一下", "资料", "来源", "最新", "准确", "定义", "区别", "对比",
    ]
    return any(keyword in text for keyword in keywords)


def _build_tool_choice_instruction(
    is_general_question: bool,
    may_need_web: bool,
    has_local_evidence: bool,
) -> str:
    lines: List[str] = []
    if is_general_question:
        lines.append(
            "这是通用技术解释问题。可以直接回答；如果你对定义、区别或细节不确定，或者用户要求来源/联网，请调用 search_web 查证。不要使用本地知识库，除非用户明确问已上传论文。"
        )
    if may_need_web:
        lines.append(
            "用户可能需要外部网页来源。若需要联网查证，请使用 search_web；若是在找论文、related work 或论文元数据，则使用 search_openalex_papers。"
        )
    if has_local_evidence:
        lines.append(
            "本轮有本地知识库参考片段，回答论文相关问题时优先参考，但不要机械套模板。"
        )
    if not lines:
        lines.append(
            "请先判断用户是在问本地论文、学术论文检索、通用网页资料，还是通用知识解释，再决定是否调用工具。"
        )
    return "\n".join(lines)


def is_degenerate_answer(text: str) -> bool:
    value = str(text or "")
    if not value.strip():
        return False
    compact = re.sub(r"\s+", "", value)
    lower = compact.lower()

    # 1) Over-repeated short token (e.g. R* repeated many times)
    if len(compact) > 80:
        token_counts = {}
        for token in re.findall(r"[A-Za-z\*]{1,3}", compact):
            if len(token) <= 1:
                continue
            token_counts[token] = token_counts.get(token, 0) + 1
        if token_counts and max(token_counts.values()) > 20:
            return True

    # 2) Repeated fragment patterns
    if re.search(r"(.{2,6})\1{9,}", compact):
        return True
    if len(compact) > 60 and re.search(r"([\u4e00-\u9fff]{1,3})\1{2,}", compact):
        return True

    # 3) Obvious dirty markers
    dirty_markers = ("rrt（（", "算法**", "rapid**")
    if any(marker in lower for marker in dirty_markers):
        return True
    if compact.count("(R*") >= 4 or lower.count("(r*") >= 4:
        return True
    if re.search(r"[，。；、,.;:!?！？\)）]{3,}", value):
        return True

    # 4) Severe Chinese duplication
    duplicate_terms = ("的的", "地地", "快地地")
    if sum(compact.count(term) for term in duplicate_terms) >= 3:
        return True

    # 5) Very low effective Chinese sentence density + high symbol noise
    if len(compact) > 80:
        chinese_chars = re.findall(r"[\u4e00-\u9fff]", value)
        chinese_count = len(chinese_chars)
        symbol_count = len(re.findall(r"[\(\)\*\[\]\{\}_`~|\\/<>@#\$%\^&+=-]", value))
        sentence_count = len(re.findall(r"[。！？.!?；;]", value))
        symbol_ratio = symbol_count / max(len(value), 1)
        chinese_ratio = chinese_count / max(len(value), 1)
        if symbol_ratio > 0.28 and chinese_ratio < 0.35 and sentence_count <= 1:
            return True

    return False


def has_unverified_web_citations(text: str) -> bool:
    value = str(text or "")
    if not value.strip():
        return False
    if re.search(r"\[[^\]]+\]\(https?://[^\)]+\)", value):
        return True
    if re.search(r"https?://\S+", value):
        return True
    if re.search(r"\bdoi\.org/\S+", value, flags=re.IGNORECASE):
        return True
    if re.search(r"\bsearch_web\s*\(", value, flags=re.IGNORECASE):
        return True
    if re.search(r"来源[:：]\s*\[\s*search_web", value, flags=re.IGNORECASE):
        return True
    return False


def _clean_retrieval_snippet(text: str, max_len: int = 500) -> str:
    value = str(text or "")
    value = re.sub(r"<!--.*?-->", " ", value, flags=re.DOTALL)
    value = re.sub(r"\bformula-not-decoded\b", " ", value, flags=re.IGNORECASE)
    value = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", value)
    value = re.sub(r"\s+", " ", value).strip()
    if len(value) > max_len:
        value = value[:max_len].rstrip() + "..."
    return value


async def _run_local_kb_preflight_if_needed(message: str, deps: AgentDependencies) -> str:
    if _is_explicit_web_paper_request(message):
        return ""
    if not _is_local_kb_question(message):
        return ""

    async def _search_with_timeout(search_tool, query: str, limit: int) -> List[dict]:
        return list(
            await asyncio.wait_for(
                search_tool.ainvoke({"query": query, "limit": limit}),
                timeout=LOCAL_PREFLIGHT_QUERY_TIMEOUT_SECONDS,
            )
            or []
        )

    async def _inner() -> str:
        tools = build_langchain_tools(deps)
        search_tool = next((t for t in tools if getattr(t, "name", "") == "search_knowledge_base"), None)
        if search_tool is None:
            logger.warning("Local KB preflight skipped: search_knowledge_base tool not found")
            return ""

        target_queries = _extract_known_local_paper_queries(message)
        timeout_happened = False
        grouped_results: List[List[dict]] = []

        if target_queries:
            for idx in range(0, len(target_queries), 2):
                title_query = target_queries[idx]
                id_query = target_queries[idx + 1] if idx + 1 < len(target_queries) else None
                title_hits: List[dict] = []
                try:
                    title_hits = await _search_with_timeout(search_tool, title_query, 3)
                except TimeoutError:
                    timeout_happened = True
                    logger.warning("Local KB preflight title query timed out: %s", title_query)
                except Exception as exc:
                    logger.warning("Local KB preflight title query failed: %s", exc)
                grouped_results.append(title_hits)

                if title_hits or not id_query:
                    continue
                try:
                    grouped_results.append(await _search_with_timeout(search_tool, id_query, 3))
                except TimeoutError:
                    timeout_happened = True
                    logger.warning("Local KB preflight id query timed out: %s", id_query)
                except Exception as exc:
                    logger.warning("Local KB preflight id query failed: %s", exc)

        try:
            grouped_results.append(await _search_with_timeout(search_tool, message, 3))
        except TimeoutError:
            timeout_happened = True
            logger.warning("Local KB preflight message query timed out")
        except Exception as exc:
            logger.warning("Local KB preflight message query failed: %s", exc)

        merged: List[dict] = []
        seen = set()
        for group in grouped_results:
            for hit in group:
                key = (
                    str(hit.get("chunk_id") or ""),
                    str(hit.get("document_id") or ""),
                    str(hit.get("content") or "")[:100],
                )
                if key in seen:
                    continue
                seen.add(key)
                merged.append(hit)
        if not merged:
            return ""

        lines = [
            "[本地知识库预检索结果]",
            "以下片段来自本地知识库。回答论文相关问题时优先参考这些片段，不要把它们说成联网资料。",
            "若片段不足以支撑结论，请明确说明“当前检索片段不足”。",
        ]
        kept = []
        for hit in merged[:10]:
            score = hit.get("score")
            if isinstance(score, float) and score < 0.25:
                continue
            kept.append(hit)
        if not kept:
            return ""

        final_hits: List[dict] = []
        if target_queries:
            for query in target_queries:
                matched = [
                    h for h in kept
                    if query.lower() in str(h.get("document_title") or "").lower()
                    or query.lower() in str(h.get("document_source") or "").lower()
                ]
                final_hits.extend(matched[:2])
        for hit in kept:
            if hit not in final_hits:
                final_hits.append(hit)
        final_hits = final_hits[:6]

        for idx, hit in enumerate(final_hits, start=1):
            title = str(hit.get("document_title") or "Untitled").strip()
            source = str(hit.get("document_source") or "").strip()
            score = hit.get("score")
            content = _clean_retrieval_snippet(str(hit.get("content") or ""), max_len=500)
            lines.append(f"{idx}. 标题：{title}\n来源：{source}\n相似度：{score}\n片段：{content}")

        if target_queries:
            for query in target_queries:
                query_l = query.lower()
                found = any(
                    query_l in str(hit.get("document_title") or "").lower()
                    or query_l in str(hit.get("document_source") or "").lower()
                    for hit in final_hits
                )
                if not found:
                    lines.append(f"- 未在本地预检索结果中命中 `{query}`。")

        if timeout_happened:
            lines.append("- 部分本地预检索请求超时，结果可能不完整。")

        return "\n\n".join(lines)

    try:
        return await asyncio.wait_for(_inner(), timeout=LOCAL_PREFLIGHT_TIMEOUT_SECONDS)
    except TimeoutError:
        logger.warning("Local KB preflight timed out")
        return ""
    except Exception as exc:
        logger.warning("Local KB preflight failed: %s", exc)
        return ""


def _dedupe_sources(sources: List[EvidenceSource]) -> List[EvidenceSource]:
    seen = set()
    unique: List[EvidenceSource] = []
    for source in list(sources or []):
        key = (
            str(getattr(source, "source_type", "") or ""),
            str(getattr(source, "document_id", "") or ""),
            str(getattr(source, "chunk_id", "") or ""),
            str(getattr(source, "document_title", "") or ""),
            str(getattr(source, "snippet", "") or "")[:100],
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(source)
    return unique


def _build_format_instruction(has_local_evidence: bool, is_general_question: bool) -> str:
    base = """请使用清晰的中文回答。
- 优先使用“小标题 + bullet list”的格式。
- 严禁使用 Markdown 表格。
- 对比类问题也不要使用 `|` 表格语法。
- 不要输出任何包含 `| HA-RRT | HMA-RRT`、`| HA-RRT | HMA-RRT*` 的表格。
- 标题必须写成 `## 1. 标题`，`##` 后必须有空格。
- 不要输出 `##1.`、`##2.`、`###1.` 这种无空格标题。
- 每个 bullet 必须单独成行。
- 如果比较 HA-RRT 和 HMA-RRT，固定使用：
  简短结论
  ## 1. 研究对象不同
  ## 2. 改进策略不同
  ## 3. 约束建模不同
  ## 4. 适用环境不同
  ## 5. 一句话总结
- 算法名保持一致：如果论文标题是 HMA-RRT，不要整篇写成 HMA-RRT*；如需说明 RRT*，写成“基于 RRT* 框架”。
"""

    if is_general_question:
        return base + """- 这是通用算法解释问题，可基于通用知识作答。
- 不要强行声明知识库证据不足。
"""

    if has_local_evidence:
        return base + """- 本轮已检索到本地知识库片段，回答应优先基于这些证据。
- 仅在确实缺证据的小节说明“证据不足”。
"""

    return base + """- 若用户在问指定论文且本轮未检索到证据，请简短说明证据不足。
"""

