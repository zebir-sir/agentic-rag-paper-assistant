from __future__ import annotations

from typing import Any, Dict, List

from agent.agent_langchain import get_langchain_chat_model
from agent.tools import HybridSearchInput, hybrid_search_tool


async def run_naive_hybrid_rag(question: str, top_k: int = 5) -> Dict[str, Any]:
    rows = await hybrid_search_tool(HybridSearchInput(query=question, limit=max(1, top_k)))
    hits: List[Dict[str, Any]] = [
        {
            "document_title": r.document_title,
            "document_source": r.document_source,
            "content": r.content,
            "metadata": dict(r.metadata or {}),
            "score": r.score,
        }
        for r in rows
    ]

    evidence_lines: List[str] = []
    for idx, hit in enumerate(hits[:top_k], start=1):
        md = hit.get("metadata") or {}
        section = str(md.get("section_path_text") or md.get("section_title") or "")
        evidence_lines.append(
            f"[{idx}] title={hit.get('document_title','')} section={section} score={hit.get('score')}\n{str(hit.get('content') or '')[:500]}"
        )

    prompt = (
        "你是一个论文助手。请仅基于下面检索证据回答用户问题。"
        "如果证据不足，明确说明不足。不要声称联网搜索，不要编造 DOI/作者/年份。\n\n"
        f"用户问题:\n{question}\n\n"
        "检索证据:\n"
        + "\n\n".join(evidence_lines)
    )

    model = get_langchain_chat_model()
    resp = await model.ainvoke([
        {"role": "system", "content": "回答要简洁，优先证据支撑。"},
        {"role": "user", "content": prompt},
    ])
    answer = getattr(resp, "content", "")
    if not isinstance(answer, str):
        answer = str(answer)

    return {
        "answer": answer.strip(),
        "retrieval_hits": hits,
        "metadata": {"backend": "naive_hybrid_rag", "top_k": top_k},
    }
