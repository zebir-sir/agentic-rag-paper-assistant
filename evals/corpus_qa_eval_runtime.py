"""Construct and run a source-grounded, corpus-specific QA evaluation set."""
from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from agent.agent_langgraph import run_langgraph_analysis
from agent.agent_runtime import AgentDependencies
from agent.db_utils import db_pool
from common.encoding_utils import write_json_utf8, write_text_utf8


def _pick(rows: list[dict[str, Any]], count: int, offset: int = 0) -> list[dict[str, Any]]:
    return [rows[(offset + index) % len(rows)] for index in range(count)] if rows else []


def _is_readable_sample(value: str) -> bool:
    text = str(value or "")
    if len(text) < 80:
        return False
    readable = len(re.findall(r"[A-Za-z0-9\u4e00-\u9fff\s.,;:()\-]", text))
    return readable / max(1, len(text)) >= 0.75


async def build_cases() -> list[dict[str, Any]]:
    async with db_pool.acquire() as conn:
        document_rows = [dict(row) for row in await conn.fetch("""
            SELECT d.id::text, d.title, string_agg(left(c.content, 500), E'\n' ORDER BY c.chunk_index) AS sample_text
            FROM documents d JOIN chunks c ON c.document_id=d.id
            WHERE coalesce(c.metadata->>'content_type','pdf') <> 'artifact'
            GROUP BY d.id, d.title ORDER BY d.title
        """)]
        documents = [{"id": row["id"], "title": row["title"]} for row in document_rows if _is_readable_sample(row.get("sample_text", ""))]
        artifact_rows = [dict(row) for row in await conn.fetch("""
            SELECT DISTINCT d.id::text, d.title FROM documents d JOIN chunks c ON c.document_id=d.id
            WHERE c.metadata->>'artifact_type'='figure' AND d.id::text = ANY($1::text[]) ORDER BY d.title
        """, [row["id"] for row in documents])]
        algorithm_rows = [dict(row) for row in await conn.fetch("""
            SELECT DISTINCT d.id::text, d.title FROM documents d JOIN chunks c ON c.document_id=d.id
            WHERE c.metadata->>'artifact_type'='algorithm' AND d.id::text = ANY($1::text[]) ORDER BY d.title
        """, [row["id"] for row in documents])]
        experiment_rows = [dict(row) for row in await conn.fetch("""
            SELECT DISTINCT d.id::text, d.title FROM documents d JOIN chunks c ON c.document_id=d.id
            WHERE lower(coalesce(c.metadata->>'section_path_text','')) ~ '(experiment|evaluation|result)'
              AND d.id::text = ANY($1::text[])
            ORDER BY d.title
        """, [row["id"] for row in documents])]
        edge_rows = [dict(row) for row in await conn.fetch("""
            SELECT e.source_document_id::text AS source_id, e.target_document_id::text AS target_id,
                   e.relation_type, ds.title AS source_title, dt.title AS target_title
            FROM paper_graph_edges e
            JOIN documents ds ON ds.id=e.source_document_id
            JOIN documents dt ON dt.id=e.target_document_id
            WHERE e.source_document_id::text = ANY($1::text[]) AND e.target_document_id::text = ANY($1::text[])
            ORDER BY e.score DESC
        """, [row["id"] for row in documents])]

    cases: list[dict[str, Any]] = []
    def add(category: str, question: str, docs: list[dict[str, Any]], routes: list[str], points: list[str], boundary: str) -> None:
        cases.append({
            "id": f"qa_{len(cases) + 1:03d}", "category": category, "question": question,
            "gold_documents": docs, "expected_retrieval_routes": routes,
            "gold_answer_points": points, "answer_boundary": boundary,
        })

    for row in _pick(documents, 20):
        add("single_paper_method", f"请基于本地论文《{row['title']}》说明研究问题、核心方法和主要贡献。", [row], ["hybrid", "section"], ["研究问题", "核心方法", "主要贡献"], "仅可依据该论文原文片段；证据不足时必须说明。")
    for row in _pick(documents, 15, 20):
        add("section_scoped_summary", f"只根据《{row['title']}》的摘要和引言，解释它要解决什么问题、为何需要该方法；不要把实验结果当作摘要结论。", [row], ["section", "hybrid"], ["摘要/引言范围", "问题", "方法动机"], "只允许摘要与引言证据。")
    for row in _pick(artifact_rows, 10):
        add("figure_evidence", f"请依据《{row['title']}》中与方法有关的图或图注，解释图展示的流程或结构；不要从图中推断未展示的定量结论。", [row], ["artifact", "hybrid"], ["图/图注", "流程或结构", "证据边界"], "图表是证据，不能据此虚构数值或未展示结论。")
    for row in _pick(algorithm_rows, 10, 4):
        add("algorithm_evidence", f"请依据《{row['title']}》的算法伪代码或算法描述，按步骤解释关键过程及每一步的作用。", [row], ["artifact", "section", "hybrid"], ["算法步骤", "关键过程", "作用"], "只能解释已检索到的算法内容。")
    for row in _pick(experiment_rows or documents, 10, 8):
        add("experiment_interpretation", f"根据《{row['title']}》的实验或评估章节，说明实验设置、评价信号和原文能支持的结果，不要补造具体数值。", [row], ["section", "artifact", "hybrid"], ["实验设置", "评价信号", "结果边界"], "定量结论必须有原文证据。")
    for row in _pick(edge_rows, 10):
        first = {"id": row["source_id"], "title": row["source_title"]}; second = {"id": row["target_id"], "title": row["target_title"]}
        add("graph_relation_compare", f"从知识星图关系和两篇原文证据出发，对比《{first['title']}》与《{second['title']}》的共同点和差异，并说明图中的 {row['relation_type']} 关系只用于检索导航、不等于因果或直接改进。", [first, second], ["graph", "hybrid", "section"], ["共同点", "差异", "关系边界"], "星图关系不能单独作为事实证据，必须回到两篇论文片段。")
    for index in range(10):
        add("openalex_metadata", f"请使用 OpenAlex 检索与 RRT* motion planning 相关的论文，给出 {index % 3 + 1} 篇，逐篇列出标题、作者、年份、期刊/会议和 DOI；只使用实际返回的元数据。", [], ["openalex"], ["标题", "作者", "年份", "DOI"], "外部论文元数据必须来自 OpenAlex 返回，不得编造。")
    for index in range(5):
        add("web_unavailable_boundary", f"请联网检索 RRT* 在 {2022 + index} 年之后的网页资料并给出链接；若通用网页检索不可用，明确说明能力边界，不得虚构链接。", [], ["web"], ["能力边界或网页来源"], "网页工具不可用时只允许说明限制与下一步建议。")
    direct_questions = [
        "用通俗语言解释什么是采样偏置，不引用本地论文或外部网页。",
        "解释为何研究问题需要先区分约束、目标和评价指标，不检索资料。",
        "给出阅读机器人路径规划论文时核对方法有效性的三个通用问题，不引用资料。",
        "用一个简单例子区分路径可行性与路径最优性，不检索资料。",
        "解释为什么相关性不等于方法改进关系，不检索资料。",
        "给出比较两种算法时避免过度结论的通用写作原则，不检索资料。",
        "说明什么是实验变量、控制变量和评价指标，不检索资料。",
        "解释论文阅读中为何应将作者主张与证据分开记录，不检索资料。",
        "给出把一个研究问题拆成可验证子问题的通用步骤，不检索资料。",
        "解释为什么图表标题和图注不能替代正文方法细节，不检索资料。",
    ]
    for question in direct_questions:
        add("direct_answer", question, [], ["direct"], ["通用解释"], "不要求或使用本地、外部或网页证据。")
    if len(cases) != 100:
        raise RuntimeError(f"expected 100 cases, got {len(cases)}")
    return cases


def _as_dict(item: Any) -> dict[str, Any]:
    return item.model_dump() if hasattr(item, "model_dump") else dict(item)


def _score_case(case: dict[str, Any], answer: str, sources: list[dict[str, Any]], tools: list[str], metadata: dict[str, Any]) -> dict[str, Any]:
    expected_ids = {str(doc["id"]) for doc in case["gold_documents"]}
    source_ids = {str(source.get("document_id") or "") for source in sources}
    source_titles = "\n".join(str(source.get("document_title") or "") for source in sources).lower()
    expected_titles = [str(doc["title"]).lower() for doc in case["gold_documents"]]
    document_hit_count = sum(doc_id in source_ids for doc_id in expected_ids)
    title_fallback_count = sum(title in source_titles for title in expected_titles)
    expected_doc_coverage = max(document_hit_count, title_fallback_count) / max(1, len(expected_ids)) if expected_ids else 1.0
    source_schema_valid = all(all(key in source for key in ("document_id", "document_title", "chunk_id", "snippet", "metadata")) for source in sources)
    answer_text = str(answer or "").strip()
    answer_nonempty = len(answer_text) >= 80
    route_hints = {str(tool).lower() for tool in tools}
    expected_routes = set(case["expected_retrieval_routes"])
    route_satisfied = any(route in " ".join(route_hints) for route in expected_routes)
    has_external_claim = any(token in answer_text.lower() for token in ("openalex", "according to web", "web search"))
    category = str(case["category"])
    direct_allowed = bool(metadata.get("direct_answer_allowed"))
    boundary_markers = ("能力边界", "无法联网", "网页检索", "不可用", "无法提供链接")
    has_boundary = any(marker in answer_text for marker in boundary_markers)
    external_metadata_complete = all(
        any(str((source.get("metadata") or {}).get(field) or "").strip() for source in sources)
        for field in ("authors", "year", "doi")
    ) if category == "openalex_metadata" else True
    if category == "direct_answer":
        contract_pass = bool(answer_nonempty and not sources and not tools)
    elif category == "web_unavailable_boundary":
        contract_pass = bool(answer_nonempty and has_boundary and not sources)
    elif category == "openalex_metadata":
        contract_pass = bool(answer_nonempty and route_satisfied and sources and external_metadata_complete)
    else:
        graph_used = bool(metadata.get("paper_graph_used") or metadata.get("paper_graph_expanded_document_count"))
        graph_contract = graph_used if category == "graph_relation_compare" else True
        contract_pass = bool(answer_nonempty and route_satisfied and graph_contract and expected_doc_coverage >= 1.0 and source_schema_valid and not has_external_claim)
    return {
        "expected_document_coverage": expected_doc_coverage,
        "answer_nonempty": answer_nonempty,
        "source_schema_valid": source_schema_valid,
        "retrieval_route_observed": sorted(route_hints),
        "route_satisfied": route_satisfied,
        "external_claim_without_request": has_external_claim,
        "direct_answer_allowed": direct_allowed,
        "boundary_disclosed": has_boundary,
        "external_metadata_complete": external_metadata_complete,
        "contract_pass": contract_pass,
    }


async def run_cases(cases: list[dict[str, Any]], output_dir: Path, timeout_seconds: int = 150) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        started = time.perf_counter()
        row = {**case, "error": None}
        try:
            selected_document_ids = [str(document["id"]) for document in case["gold_documents"]]
            preferences = {
                "selected_document_ids": selected_document_ids,
                "allow_supplemental": case["category"] == "graph_relation_compare",
            } if selected_document_ids else {}
            result = await asyncio.wait_for(
                run_langgraph_analysis(case["question"], AgentDependencies(session_id=f"overnight-eval-{uuid.uuid4().hex[:12]}", user_id="evaluation", search_preferences=preferences), context_prompt=""),
                timeout=timeout_seconds,
            )
            sources = [_as_dict(source) for source in result.sources]
            metadata = dict(result.metadata or {})
            tools = [str(item) for item in metadata.get("tools_executed") or []]
            row.update({"answer": str(result.message or ""), "sources": sources, "tools_executed": tools, "metadata": metadata, "score": _score_case(case, result.message, sources, tools, metadata)})
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        row["latency_seconds"] = round(time.perf_counter() - started, 3)
        rows.append(row)
        write_json_utf8(output_dir / "qa_100_progress.json", {"completed": len(rows), "total": len(cases), "cases": rows}, indent=2)
    valid = [row for row in rows if not row["error"]]
    score_rows = [row["score"] for row in valid]
    summary = {"cases": len(rows), "completed": len(valid), "errors": len(rows) - len(valid), "answer_contract_pass_rate": sum(score["contract_pass"] for score in score_rows) / max(1, len(score_rows)), "expected_document_coverage": sum(score["expected_document_coverage"] for score in score_rows) / max(1, len(score_rows)), "source_schema_coverage": sum(score["source_schema_valid"] for score in score_rows) / max(1, len(score_rows)), "answer_nonempty_rate": sum(score["answer_nonempty"] for score in score_rows) / max(1, len(score_rows)), "latency_seconds_p50": sorted(row["latency_seconds"] for row in rows)[len(rows)//2] if rows else 0}
    return {"protocol": {"annotation": "single-expert, corpus-specific generated gold", "categories": dict(Counter(case["category"] for case in cases)), "answer_boundary": "Contract pass requires correct-paper evidence, valid displayed source fields, nonempty answer, and no unrequested external-source claim. It does not claim semantic perfection without independent human review."}, "summary": summary, "cases": rows}


def markdown(report: dict[str, Any]) -> str:
    lines = ["# Corpus QA Evaluation (100 Cases)", "", "## Protocol", "", *(f"- {key}: {value}" for key, value in report["protocol"].items()), "", "## Summary", "", *(f"- {key}: {value}" for key, value in report["summary"].items()), "", "## Per-case Audit", "", "| ID | Category | Document coverage | Contract | Latency | Error |", "|---|---|---:|---|---:|---|"]
    for row in report["cases"]:
        score = row.get("score") or {}; lines.append(f"| {row['id']} | {row['category']} | {score.get('expected_document_coverage','-')} | {score.get('contract_pass','-')} | {row['latency_seconds']} | {row.get('error') or ''} |")
    return "\n".join(lines) + "\n"
