"""Asynchronous, cached Chinese research cards for paper graph nodes."""
import asyncio
import json
import os
import re
from typing import Any, Dict

from openai import AsyncOpenAI

from .db_utils import db_pool
from .graph_localization_schema import (
    LOCALIZATION_SCHEMA_VERSION,
    graph_localization_hash,
    normalize_graph_card,
    validate_graph_card,
)
_running_tasks: set[str] = set()


def _extract_json(text: str) -> Dict[str, Any]:
    raw = str(text or "").strip()
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        return {}
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


async def _generate_localization(document_id: str, source_hash: str) -> None:
    try:
        async with db_pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT d.title, n.profile_text FROM documents d
                JOIN paper_graph_nodes n ON n.document_id=d.id
                WHERE d.id=$1::uuid
                """, document_id,
            )
        if not row:
            return
        title, profile_text = str(row["title"]), str(row["profile_text"] or "")
        if graph_localization_hash(title, profile_text) != source_hash:
            return
        api_key, base_url, model = (os.getenv("OPENAI_API_KEY", "").strip(), os.getenv("OPENAI_BASE_URL", "").strip(), os.getenv("LLM_CHOICE", "").strip())
        if not api_key or not base_url or not model:
            raise RuntimeError("primary agent model configuration is incomplete")
        prompt = (
            "Create a faithful Simplified-Chinese research card for a paper knowledge graph. Return JSON only with "
            "title_zh, overview_zh, problem_zh, method_zh, innovation_zh, keywords_zh. "
            "Use only the supplied original title and paper passages. Do not infer experiments or claims not present. "
            "Preserve algorithm/model/dataset acronyms, mathematical symbols, numeric values, and proper names exactly. "
            "Each text field must be concise Chinese (overview/problem/method/innovation <= 120 Chinese characters); "
            "keywords_zh must contain 3-5 concise terms.\n\n"
            f"Original title: {title}\n\nPaper passages:\n{profile_text}"
        )
        response = await AsyncOpenAI(api_key=api_key, base_url=base_url).chat.completions.create(
            model=model, temperature=0, max_tokens=1200, messages=[{"role": "user", "content": prompt}],
        )
        card = normalize_graph_card(_extract_json(response.choices[0].message.content or ""), title)
        validation = validate_graph_card(card, title)
        if not validation["valid"]:
            raise RuntimeError(f"localization validation failed: {json.dumps(validation, ensure_ascii=False)}")
        async with db_pool.acquire() as conn:
            await conn.execute(
                """UPDATE paper_graph_localizations SET status='ready', model=$4, content=$5::jsonb,
                   validation=$6::jsonb, error_message=NULL, updated_at=CURRENT_TIMESTAMP
                   WHERE document_id=$1::uuid AND target_language='zh' AND source_sha256=$2 AND schema_version=$3""",
                document_id, source_hash, LOCALIZATION_SCHEMA_VERSION, model,
                json.dumps(card, ensure_ascii=False), json.dumps(validation, ensure_ascii=False),
            )
            await conn.execute(
                """UPDATE paper_graph_nodes SET graph_version=graph_version + 1, updated_at=CURRENT_TIMESTAMP
                   WHERE document_id=$1::uuid""",
                document_id,
            )
    except Exception as exc:
        async with db_pool.acquire() as conn:
            await conn.execute(
                """UPDATE paper_graph_localizations SET status='failed', error_message=$4, updated_at=CURRENT_TIMESTAMP
                   WHERE document_id=$1::uuid AND target_language='zh' AND source_sha256=$2 AND schema_version=$3""",
                document_id, source_hash, LOCALIZATION_SCHEMA_VERSION, str(exc)[:500],
            )
            await conn.execute(
                """UPDATE paper_graph_nodes SET graph_version=graph_version + 1, updated_at=CURRENT_TIMESTAMP
                   WHERE document_id=$1::uuid""",
                document_id,
            )
    finally:
        _running_tasks.discard(document_id)


async def schedule_pending_graph_localizations(limit: int = 2) -> None:
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT document_id::text, source_sha256 FROM paper_graph_localizations
               WHERE target_language='zh' AND schema_version=$1 AND status='pending'
               ORDER BY updated_at LIMIT $2""", LOCALIZATION_SCHEMA_VERSION, limit,
        )
    for row in rows:
        document_id = str(row["document_id"])
        if document_id in _running_tasks:
            continue
        _running_tasks.add(document_id)
        asyncio.create_task(_generate_localization(document_id, str(row["source_sha256"])))
