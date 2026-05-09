﻿﻿from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.db_utils import close_database, initialize_database, section_search
from agent.tools import ArtifactSearchInput, HybridSearchInput, artifact_search_tool, hybrid_search_tool
from common.encoding_utils import read_json_robust, write_json_utf8, write_text_utf8


def _norm(v: Any) -> str: return str(v or "").strip().lower()
def _contains_any(text: str, keys: List[str]) -> bool: return any(_norm(k) in _norm(text) for k in keys if str(k or "").strip())

def _keyword_recall(rows: List[Dict[str, Any]], keys: List[str]) -> float:
    if not keys: return 0.0
    blob = "\n".join(str(r.get("content") or "") for r in rows).lower()
    return sum(1 for k in keys if _norm(k) in blob) / len(keys)

def _order_ok(rows: List[Dict[str, Any]]) -> bool:
    vals=[]
    for r in rows:
        md = r.get("metadata") or {}
        vals.append((int(md.get("section_start_line") or 0), int(md.get("section_chunk_index") or 0)))
    return all(vals[i] <= vals[i+1] for i in range(len(vals)-1)) if vals else False

async def _run_mode(mode: str, q: str, section_hint: str, k: int) -> tuple[List[Dict[str, Any]], float]:
    t0 = time.time()
    if mode == "hybrid":
        out = await hybrid_search_tool(HybridSearchInput(query=q, limit=k))
        rows = [{"document_title":r.document_title,"document_source":r.document_source,"content":r.content,"metadata":dict(r.metadata or {}),"score":r.score} for r in out]
    elif mode == "section":
        out = await section_search(query_text=q, section_query=section_hint or q, limit=k)
        rows = [{"document_title":r.get("document_title"),"document_source":r.get("document_source"),"content":r.get("content"),"metadata":dict(r.get("metadata") or {}),"score":r.get("combined_score",0.0)} for r in out]
    elif mode == "artifact":
        out = await artifact_search_tool(ArtifactSearchInput(query=q, limit=k))
        rows = [{"document_title":r.document_title,"document_source":r.document_source,"content":r.content,"metadata":dict(r.metadata or {}),"score":r.score} for r in out]
    else:
        rows = []
    return rows, (time.time()-t0)*1000.0

async def run_suite(cases: List[Dict[str, Any]], limit: int, top_k: int) -> Dict[str, Any]:
    out_cases=[]
    for c in (cases[:limit] if limit>0 else cases):
        modes = list(c.get("retrieval_modes_to_compare") or [])
        expected_docs = list(c.get("expected_document_keywords") or [])
        expected_secs = list(c.get("expected_section_keywords") or [])
        expected_arts = list(c.get("expected_artifact_types") or [])
        expected_kw = list(c.get("expected_content_keywords") or [])
        mode_rows = {}
        for m in modes:
            rows, latency = await _run_mode(m, str(c.get("question") or ""), " ".join(expected_secs), top_k)
            doc_blob = "\n".join(f"{r.get('document_title','')} {r.get('document_source','')}" for r in rows)
            sec_blob = "\n".join(str((r.get("metadata") or {}).get("section_path_text") or (r.get("metadata") or {}).get("section_title") or "") for r in rows)
            art_blob = "\n".join(str((r.get("metadata") or {}).get("artifact_type") or "") for r in rows)
            mode_rows[m] = {
                "Doc Hit@1": 1 if (rows and _contains_any(doc_blob.splitlines()[0], expected_docs)) else 0,
                "Doc Hit@5": 1 if _contains_any(doc_blob, expected_docs) else 0,
                "Section Precision@K": (sum(1 for r in rows if _contains_any(str((r.get("metadata") or {}).get("section_path_text") or (r.get("metadata") or {}).get("section_title") or ""), expected_secs))/len(rows)) if rows and expected_secs else None,
                "Artifact Hit@K": (1 if _contains_any(art_blob, expected_arts) else 0) if expected_arts else None,
                "Keyword Recall@K": _keyword_recall(rows, expected_kw),
                "Order OK": _order_ok(rows),
                "Average latency ms": latency,
                "count": len(rows),
            }
        out_cases.append({"id":c.get("id"),"question":c.get("question"),"modes":mode_rows})

    aggregates={}
    for m in {k for c in out_cases for k in c["modes"].keys()}:
        entries=[c["modes"][m] for c in out_cases if m in c["modes"]]
        if not entries: continue
        def avg(key):
            vals=[float(e[key]) for e in entries if e.get(key) is not None]
            return sum(vals)/len(vals) if vals else None
        aggregates[m] = {"Doc Hit@1":avg("Doc Hit@1"),"Doc Hit@5":avg("Doc Hit@5"),"Section Precision@K":avg("Section Precision@K"),"Artifact Hit@K":avg("Artifact Hit@K"),"Keyword Recall@K":avg("Keyword Recall@K"),"Order OK":avg("Order OK"),"Average latency ms":avg("Average latency ms")}

    return {"summary":{"total_cases":len(out_cases),"mode_metrics":aggregates,"note":"section 与 hybrid/artifact 是互补关系"},"cases":out_cases}


def to_markdown(r: Dict[str, Any]) -> str:
    lines=["# Retrieval Quality Eval v2","","- 说明: section_search 与 hybrid_search/artifact_search 是互补关系，不是单向替代。","",f"- total_cases: {r['summary']['total_cases']}","","## Mode Summary","","| Mode | Doc@1 | Doc@5 | SecPrec@K | ArtHit@K | KWRecall@K | OrderOK | Latency(ms) |","|---|---:|---:|---:|---:|---:|---:|---:|"]
    for m,v in r["summary"]["mode_metrics"].items():
        lines.append(f"| {m} | {(v['Doc Hit@1'] or 0):.3f} | {(v['Doc Hit@5'] or 0):.3f} | {('N/A' if v['Section Precision@K'] is None else f'{v['Section Precision@K']:.3f}')} | {('N/A' if v['Artifact Hit@K'] is None else f'{v['Artifact Hit@K']:.3f}')} | {(v['Keyword Recall@K'] or 0):.3f} | {(v['Order OK'] or 0):.3f} | {(v['Average latency ms'] or 0):.1f} |")
    return "\n".join(lines)+"\n"


async def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--cases",default="evals/cases/retrieval_quality_cases.json"); p.add_argument("--limit",type=int,default=0); p.add_argument("--top-k",type=int,default=5); p.add_argument("--output-dir",default="evals/results"); a=p.parse_args()
    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    cases=read_json_robust(Path(a.cases))
    await initialize_database()
    try: report=await run_suite(cases,int(a.limit or 0),int(a.top_k or 5))
    finally: await close_database()
    write_json_utf8(out/"retrieval_quality_eval.json",report,indent=2); write_text_utf8(out/"retrieval_quality_eval.md",to_markdown(report)); print(json.dumps(report["summary"],ensure_ascii=False,indent=2))

if __name__=="__main__": asyncio.run(main())
