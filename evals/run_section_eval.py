from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.db_utils import close_database, db_pool, initialize_database, section_search
from agent.tools import HybridSearchInput, hybrid_search_tool


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()


def _contains_any(text: str, keywords: List[str]) -> bool:
    t = _norm(text)
    return any(_norm(k) in t for k in keywords if str(k or "").strip())


def _extract_section_text(hit: Dict[str, Any]) -> str:
    md = hit.get("metadata") or {}
    if not isinstance(md, dict):
        return ""
    return f"{md.get('section_title','')} {md.get('section_path_text','')}"


def _normalize_metadata(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            loaded = json.loads(value)
            return loaded if isinstance(loaded, dict) else {}
        except Exception:
            return {}
    if value is None:
        return {}
    try:
        casted = dict(value)
        return casted if isinstance(casted, dict) else {}
    except Exception:
        return {}


def _normalize_hit(hit: Any) -> Dict[str, Any]:
    if isinstance(hit, dict):
        document_title = hit.get("document_title", "")
        document_source = hit.get("document_source", "")
        content = hit.get("content", "")
        metadata = _normalize_metadata(hit.get("metadata"))
    else:
        document_title = getattr(hit, "document_title", "")
        document_source = getattr(hit, "document_source", "")
        content = getattr(hit, "content", "")
        metadata = _normalize_metadata(getattr(hit, "metadata", {}))
    return {
        "document_title": str(document_title or ""),
        "document_source": str(document_source or ""),
        "content": str(content or ""),
        "metadata": metadata,
    }


def _normalize_rows(items: List[Any]) -> List[Dict[str, Any]]:
    return [_normalize_hit(item) for item in items]


def _keyword_recall(results: List[Dict[str, Any]], expected_keywords: List[str]) -> float:
    keys = [k for k in (expected_keywords or []) if str(k or "").strip()]
    if not keys:
        return 0.0
    blob = "\n".join(str(r.get("content") or "") for r in results).lower()
    hit = sum(1 for k in keys if _norm(k) in blob)
    return hit / len(keys)


def _doc_hit(rows: List[Dict[str, Any]], expected_docs: List[str]) -> bool:
    for r in rows:
        if _contains_any(f"{r.get('document_title','')} {r.get('document_source','')}", expected_docs):
            return True
    return False


def _section_precision(rows: List[Dict[str, Any]], expected_sections: List[str]) -> float:
    if not rows:
        return 0.0
    hit = sum(1 for r in rows if _contains_any(_extract_section_text(r), expected_sections))
    return hit / len(rows)


def _order_ok(rows: List[Dict[str, Any]]) -> Optional[bool]:
    values = []
    for r in rows:
        md = r.get("metadata") or {}
        if not isinstance(md, dict):
            continue
        s = md.get("section_start_line")
        c = md.get("section_chunk_index")
        if isinstance(s, int) or isinstance(c, int):
            values.append((int(s or 0), int(c or 0)))
    if not values:
        return None
    return all(values[i] <= values[i + 1] for i in range(len(values) - 1))


def _as_plain_from_chunk_results(items: List[Any]) -> List[Dict[str, Any]]:
    return _normalize_rows(items)


def _top_hits_preview(rows: List[Dict[str, Any]], limit: int = 3) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in rows[: max(0, limit)]:
        md = r.get("metadata") or {}
        section = ""
        if isinstance(md, dict):
            section = str(md.get("section_path_text") or md.get("section_title") or "")
        out.append(
            {
                "document_title": str(r.get("document_title") or ""),
                "section": section,
                "content_preview": str(r.get("content") or "")[:160],
            }
        )
    return out


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _rate_true(values: List[Optional[bool]]) -> float:
    filtered = [v for v in values if v is not None]
    if not filtered:
        return 0.0
    return sum(1 for v in filtered if v) / len(filtered)


async def _resolve_document_id(expected_document_keywords: List[str]) -> tuple[Optional[str], str]:
    keywords = [str(k or "").strip().lower() for k in (expected_document_keywords or []) if str(k or "").strip()]
    if not keywords:
        return None, ""

    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id::text AS id, title, source
            FROM documents
            """
        )

    best_id: Optional[str] = None
    best_title = ""
    best_score = 0
    for row in rows:
        title = str(row["title"] or "")
        source = str(row["source"] or "")
        text = f"{title} {source}".lower()
        score = sum(1 for k in keywords if k in text)
        if score <= 0:
            continue
        if score > best_score:
            best_score = score
            best_id = str(row["id"] or "")
            best_title = title

    if not best_id:
        return None, ""
    return best_id, best_title


async def run_eval(cases: List[Dict[str, Any]], limit: int, text_weight: float) -> Dict[str, Any]:
    case_results: List[Dict[str, Any]] = []

    for case in cases:
        cid = case.get("id") or "unknown"
        query = str(case.get("query") or "")
        section_query = str(case.get("section_query") or "")
        expected_docs = list(case.get("expected_document_keywords") or [])
        expected_sections = list(case.get("expected_section_keywords") or [])
        expected_content = list(case.get("expected_content_keywords") or [])
        resolved_document_id, resolved_document_title = await _resolve_document_id(expected_docs)
        doc_resolve_warning = "" if resolved_document_id else "document_not_resolved"
        doc_hint = " ".join(expected_docs).strip()
        hybrid_query = f"{doc_hint} {section_query} {query}".strip()

        item: Dict[str, Any] = {
            "id": cid,
            "query": query,
            "section_query": section_query,
            "resolved_document_id": resolved_document_id,
            "resolved_document_title": resolved_document_title,
            "document_resolve_warning": doc_resolve_warning,
            "hybrid_query": hybrid_query,
            "A": {"error": None},
            "B": {"error": None},
        }

        # A: section_search
        try:
            a_raw_rows = await section_search(
                query_text=query,
                section_query=section_query,
                document_id=resolved_document_id,
                limit=limit,
            )
            a_rows = _normalize_rows(a_raw_rows)
            a_doc_hit = _doc_hit(a_rows, expected_docs)
            a_sec_prec = _section_precision(a_rows, expected_sections)
            a_kw = _keyword_recall(a_rows, expected_content)
            a_order = _order_ok(a_rows)
            item["A"].update(
                {
                    "result_count": len(a_rows),
                    "doc_hit_at_k": a_doc_hit,
                    "section_precision_at_k": a_sec_prec,
                    "keyword_recall_at_k": a_kw,
                    "order_ok": a_order,
                    "section_search_top_hits": _top_hits_preview(a_rows),
                }
            )
        except Exception as exc:
            item["A"].update(
                {
                    "result_count": 0,
                    "doc_hit_at_k": False,
                    "section_precision_at_k": 0.0,
                    "keyword_recall_at_k": 0.0,
                    "order_ok": None,
                    "section_search_top_hits": [],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

        # B: hybrid_search
        try:
            b_items = await hybrid_search_tool(
                HybridSearchInput(
                    query=hybrid_query,
                    limit=limit,
                    text_weight=text_weight,
                )
            )
            b_rows = _as_plain_from_chunk_results(b_items)
            b_doc_hit = _doc_hit(b_rows, expected_docs)
            b_sec_prec = _section_precision(b_rows, expected_sections)
            b_kw = _keyword_recall(b_rows, expected_content)
            b_order = _order_ok(b_rows)
            item["B"].update(
                {
                    "result_count": len(b_rows),
                    "doc_hit_at_k": b_doc_hit,
                    "section_precision_at_k": b_sec_prec,
                    "keyword_recall_at_k": b_kw,
                    "order_ok": b_order,
                    "hybrid_search_top_hits": _top_hits_preview(b_rows),
                }
            )
        except Exception as exc:
            item["B"].update(
                {
                    "result_count": 0,
                    "doc_hit_at_k": False,
                    "section_precision_at_k": 0.0,
                    "keyword_recall_at_k": 0.0,
                    "order_ok": None,
                    "hybrid_search_top_hits": [],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

        case_results.append(item)

    def collect(metric: str, group: str) -> List[float]:
        vals = []
        for c in case_results:
            vals.append(float(c[group].get(metric) or 0.0))
        return vals

    def collect_bool(metric: str, group: str) -> List[Optional[bool]]:
        return [c[group].get(metric) for c in case_results]

    summary = {
        "total_cases": len(case_results),
        "limit": limit,
        "text_weight": text_weight,
        "A_section_precision_at_k": _mean(collect("section_precision_at_k", "A")),
        "B_section_precision_at_k": _mean(collect("section_precision_at_k", "B")),
        "A_doc_hit_at_k": _mean([1.0 if x else 0.0 for x in collect_bool("doc_hit_at_k", "A")]),
        "B_doc_hit_at_k": _mean([1.0 if x else 0.0 for x in collect_bool("doc_hit_at_k", "B")]),
        "A_keyword_recall_at_k": _mean(collect("keyword_recall_at_k", "A")),
        "B_keyword_recall_at_k": _mean(collect("keyword_recall_at_k", "B")),
        "A_order_ok_rate": _rate_true(collect_bool("order_ok", "A")),
        "B_order_ok_rate": _rate_true(collect_bool("order_ok", "B")),
    }

    return {"summary": summary, "cases": case_results}


def to_markdown(report: Dict[str, Any]) -> str:
    s = report["summary"]
    lines = []
    lines.append("# Section Eval A/B Report")
    lines.append("")
    lines.append("| Metric | Section Search (A) | Hybrid Search (B) |")
    lines.append("|---|---:|---:|")
    lines.append(f"| Avg Section Precision@K | {s['A_section_precision_at_k']:.3f} | {s['B_section_precision_at_k']:.3f} |")
    lines.append(f"| Doc Hit@K | {s['A_doc_hit_at_k']:.3f} | {s['B_doc_hit_at_k']:.3f} |")
    lines.append(f"| Avg Keyword Recall@K | {s['A_keyword_recall_at_k']:.3f} | {s['B_keyword_recall_at_k']:.3f} |")
    lines.append(f"| Order OK Rate | {s['A_order_ok_rate']:.3f} | {s['B_order_ok_rate']:.3f} |")
    lines.append("")
    lines.append("| Case ID | A SecPrec | B SecPrec | A DocHit | B DocHit | A KWRec | B KWRec | A Order | B Order | A Err | B Err |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|---|---|---|")
    for c in report["cases"]:
        a = c["A"]
        b = c["B"]
        lines.append(
            "| {id} | {a_sp:.3f} | {b_sp:.3f} | {a_d} | {b_d} | {a_kw:.3f} | {b_kw:.3f} | {a_o} | {b_o} | {a_e} | {b_e} |".format(
                id=c["id"],
                a_sp=float(a.get("section_precision_at_k") or 0.0),
                b_sp=float(b.get("section_precision_at_k") or 0.0),
                a_d=1 if a.get("doc_hit_at_k") else 0,
                b_d=1 if b.get("doc_hit_at_k") else 0,
                a_kw=float(a.get("keyword_recall_at_k") or 0.0),
                b_kw=float(b.get("keyword_recall_at_k") or 0.0),
                a_o="N/A" if a.get("order_ok") is None else ("1" if a.get("order_ok") else "0"),
                b_o="N/A" if b.get("order_ok") is None else ("1" if b.get("order_ok") else "0"),
                a_e=a.get("error") or "",
                b_e=b.get("error") or "",
            )
        )
    return "\n".join(lines) + "\n"


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run section A/B eval")
    parser.add_argument("--cases", default="evals/section_cases.json")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--text-weight", type=float, default=0.3)
    parser.add_argument("--out-dir", default="evals/results")
    args = parser.parse_args()

    cases = json.loads(Path(args.cases).read_text(encoding="utf-8-sig"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    await initialize_database()
    try:
        report = await run_eval(cases=cases, limit=max(1, args.limit), text_weight=args.text_weight)
    finally:
        await close_database()

    out_json = out_dir / "section_eval.json"
    out_md = out_dir / "section_eval.md"
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(to_markdown(report), encoding="utf-8")

    print("[section-eval] done")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"json: {out_json}")
    print(f"md:   {out_md}")


if __name__ == "__main__":
    asyncio.run(main())
