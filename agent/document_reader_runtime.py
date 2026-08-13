"""Full-context academic translation with a document-level cache."""

import hashlib
import json
import os
import re
from typing import Any, AsyncIterator, Dict, List

from openai import AsyncOpenAI

from .db_utils import get_document, get_document_translation, save_document_translation


def document_source_hash(markdown: str) -> str:
    return hashlib.sha256(str(markdown or "").encode("utf-8")).hexdigest()


def _split_markdown_sections(markdown: str, max_chars: int = 10000) -> List[str]:
    """Keep a paper's heading order while bounding one translation request."""
    value = str(markdown or "").strip()
    if not value:
        return []
    sections = re.split(r"(?m)(?=^#{1,6}\s+)", value)
    result: List[str] = []
    for section in (item.strip() for item in sections if item.strip()):
        if len(section) <= max_chars:
            result.append(section)
            continue
        heading_match = re.match(r"^(#{1,6}\s+[^\n]+)", section)
        heading = heading_match.group(1) if heading_match else ""
        paragraphs = re.split(r"\n\s*\n", section)
        current = ""
        for paragraph in paragraphs:
            candidate = f"{current}\n\n{paragraph}".strip()
            if current and len(candidate) > max_chars:
                result.append(current)
                current = f"{heading}\n\n{paragraph}".strip() if heading else paragraph
            else:
                current = candidate
        if current:
            result.append(current)
    return result


def _extract_json(text: str) -> Dict[str, Any]:
    raw = str(text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, flags=re.DOTALL)
    candidate = fenced.group(1) if fenced else raw
    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(raw[start : end + 1])
                return parsed if isinstance(parsed, dict) else {}
            except json.JSONDecodeError:
                pass
    return {}


async def _complete(client: AsyncOpenAI, model: str, prompt: str, max_tokens: int) -> str:
    response = await client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return str(response.choices[0].message.content or "").strip()


async def _build_document_brief(
    client: AsyncOpenAI,
    model: str,
    source: str,
    target: str,
) -> Dict[str, Any]:
    """Read the complete paper before translating its chapters in sequence."""
    target_name = "Simplified Chinese" if target == "zh" else "English"
    sections = _split_markdown_sections(source, max_chars=18000)
    section_notes: List[str] = []
    for index, section in enumerate(sections, start=1):
        note = await _complete(
            client,
            model,
            "You are preparing a faithful academic translation. Read this part of a single paper and return a compact "
            "JSON object with keys `section`, `purpose`, `terms`, `symbols`, `entities`, and `translation_risks`. "
            "Do not translate the paper and do not invent facts.\n\n"
            f"Part {index}/{len(sections)}:\n{section}",
            max_tokens=2200,
        )
        section_notes.append(note)

    merged_notes = "\n\n".join(section_notes)
    merged_notes = "\n\n".join(section_notes)
    brief = _extract_json(
        await _complete(
            client,
            model,
            "Based on the complete section-by-section reading notes below, produce a JSON translation brief for a paper. "
            f"The target language is {target_name}. Required keys: `paper_summary`, `terminology`, `notation`, "
            "`translation_style`, `section_flow`, `do_not_translate`. Preserve established technical names where appropriate. "
            "This brief will be used to translate every chapter from top to bottom.\n\n"
            f"Reading notes:\n{merged_notes}",
            max_tokens=4200,
        )
    )
    return brief or {"paper_summary": "Use faithful academic terminology and preserve all mathematical notation."}


async def stream_document_translation(document_id: str, target_language: str) -> AsyncIterator[Dict[str, Any]]:
    """Translate in source order and expose completed Markdown chapters as SSE-safe events."""
    target = str(target_language or "").strip().lower()
    if target not in {"zh", "en"}:
        raise ValueError("target_language must be zh or en")

    document = await get_document(document_id)
    if not document:
        raise LookupError("document not found")
    source = str(document.get("content") or "").strip()
    if not source:
        raise ValueError("document has no parsed Markdown content")

    source_hash = document_source_hash(source)
    cached = await get_document_translation(document_id, target, source_hash)
    if cached:
        yield {"type": "complete", **cached, "cached": True, "section_count": 0}
        return

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    base_url = os.getenv("OPENAI_BASE_URL", "").strip()
    model = os.getenv("LLM_CHOICE", "").strip()
    if not api_key or not base_url or not model:
        raise RuntimeError("primary agent model configuration is incomplete")

    reading_sections = _split_markdown_sections(source, max_chars=18000)
    translation_sections = _split_markdown_sections(source)
    if not reading_sections or not translation_sections:
        raise ValueError("document has no translatable Markdown sections")
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    yield {
        "type": "status",
        "phase": "reading",
        "completed": 0,
        "total": len(reading_sections),
        "message": "正在通读全文，建立术语与章节关系",
    }
    target_name = "Simplified Chinese" if target == "zh" else "English"
    section_notes: List[str] = []
    for index, section in enumerate(reading_sections, start=1):
        note = await _complete(
            client,
            model,
            "You are preparing a faithful academic translation. Read this part of a single paper and return a compact "
            "JSON object with keys `section`, `purpose`, `terms`, `symbols`, `entities`, and `translation_risks`. "
            "Do not translate the paper and do not invent facts.\n\n"
            f"Part {index}/{len(reading_sections)}:\n{section}",
            max_tokens=2200,
        )
        section_notes.append(note)
        yield {
            "type": "status",
            "phase": "reading",
            "completed": index,
            "total": len(reading_sections),
            "message": f"正在建立全文术语与章节关系（{index}/{len(reading_sections)}）",
        }

    merged_notes = "\n\n".join(section_notes)
    brief = _extract_json(
        await _complete(
            client,
            model,
            "Based on the complete section-by-section reading notes below, produce a JSON translation brief for a paper. "
            f"The target language is {target_name}. Required keys: `paper_summary`, `terminology`, `notation`, "
            "`translation_style`, `section_flow`, `do_not_translate`. Preserve established technical names where appropriate. "
            "This brief will be used to translate every chapter from top to bottom.\n\n"
            f"Reading notes:\n{merged_notes}",
            max_tokens=4200,
        )
    ) or {"paper_summary": "Use faithful academic terminology and preserve all mathematical notation."}

    translated_parts: List[str] = []
    previous_translation = ""
    yield {
        "type": "status",
        "phase": "translating",
        "completed": 0,
        "total": len(translation_sections),
        "message": "全文理解完成，正在按原章节顺序翻译",
    }
    for index, section in enumerate(translation_sections, start=1):
        prompt = (
            f"Translate chapter {index}/{len(translation_sections)} of this academic paper into {target_name}. "
            "The paper was read in full before translation; apply the document translation brief consistently. "
            "Preserve heading levels, chapter order, paragraphs, Markdown tables, code and pseudocode, equations, "
            "citations, figure/table labels, links and numeric values. Do not summarize, omit, explain, or invent. "
            "Return only this chapter's translated Markdown.\n\n"
            f"Document translation brief:\n{json.dumps(brief, ensure_ascii=False)}\n\n"
            f"Previous translated chapter ending (for cohesion only):\n{previous_translation[-1200:]}\n\n"
            f"Source chapter:\n{section}"
        )
        translated = await _complete(client, model, prompt, max_tokens=12000)
        if not translated:
            raise RuntimeError(f"translation model returned empty chapter {index}")
        translated_parts.append(translated)
        previous_translation = translated
        yield {
            "type": "section",
            "index": index,
            "total": len(translation_sections),
            "content_markdown": translated,
            "message": f"已完成第 {index}/{len(translation_sections)} 章",
        }

    translated = "\n\n".join(translated_parts).strip()
    saved = await save_document_translation(
        document_id=document_id,
        target_language=target,
        source_sha256=source_hash,
        content_markdown=translated,
        model=model,
    )
    yield {"type": "complete", **saved, "cached": False, "section_count": len(translation_sections)}


async def translate_document(document_id: str, target_language: str) -> Dict[str, Any]:
    async for event in stream_document_translation(document_id, target_language):
        if event["type"] == "complete":
            return event
    raise RuntimeError("translation stream ended without a completion event")
