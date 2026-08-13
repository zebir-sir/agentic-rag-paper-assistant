"""Context-aware translation for selected PDF text with two-level caching."""

import hashlib
import json
import os
import re
from typing import Any, Dict

from openai import AsyncOpenAI

from .db_utils import (
    get_document,
    get_selection_translation,
    get_translation_profile,
    save_selection_translation,
    save_translation_profile,
)
from .document_reader_runtime import document_source_hash


PROFILE_VERSION = "selection-profile-v1"


def _normalized_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _selection_hash(value: str) -> str:
    return hashlib.sha256(_normalized_text(value).encode("utf-8")).hexdigest()


async def _complete(client: AsyncOpenAI, model: str, prompt: str, max_tokens: int) -> str:
    response = await client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return str(response.choices[0].message.content or "").strip()


async def _get_or_create_profile(document_id: str, target: str, source: str, client: AsyncOpenAI, model: str) -> Dict[str, Any]:
    source_hash = document_source_hash(f"{PROFILE_VERSION}\n{source}")
    cached = await get_translation_profile(document_id, target, source_hash)
    if cached:
        return {**cached, "cached": True, "source_hash": source_hash}

    target_name = "Simplified Chinese" if target == "zh" else "English"
    profile_text = source[:30000]
    prompt = (
        "Read this academic paper to create a compact translation terminology profile. "
        f"The target language is {target_name}. Return JSON only with keys terminology, abbreviations, notation, "
        "method_names, and translation_rules. Preserve established technical names and do not invent paper facts.\n\n"
        f"Paper text:\n{profile_text}"
    )
    raw_profile = await _complete(client, model, prompt, max_tokens=3200)
    try:
        profile = json.loads(raw_profile)
        if not isinstance(profile, dict):
            profile = {}
    except json.JSONDecodeError:
        profile = {"translation_rules": "Preserve technical terminology, mathematical notation, citations, and abbreviations."}
    saved = await save_translation_profile(
        document_id=document_id,
        target_language=target,
        source_sha256=source_hash,
        profile=profile,
        model=model,
    )
    return {**saved, "cached": False, "source_hash": source_hash}


async def translate_selection(document_id: str, target_language: str, selection: str, context_before: str = "", context_after: str = "") -> Dict[str, Any]:
    target = str(target_language or "").strip().lower()
    source_text = _normalized_text(selection)
    if target not in {"zh", "en"}:
        raise ValueError("target_language must be zh or en")
    if len(source_text) < 2:
        raise ValueError("请选择至少两个字符的文本")
    if len(source_text) > 6000:
        raise ValueError("单次选择不能超过 6000 个字符")
    document = await get_document(document_id)
    if not document:
        raise LookupError("document not found")
    source = str(document.get("content") or "").strip()
    if not source:
        raise ValueError("document has no parsed Markdown content")

    source_hash = document_source_hash(f"{PROFILE_VERSION}\n{source}")
    selection_hash = _selection_hash(source_text)
    cached = await get_selection_translation(document_id, target, source_hash, selection_hash)
    if cached:
        return {**cached, "cached": True, "profile_cached": True}

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    base_url = os.getenv("OPENAI_BASE_URL", "").strip()
    model = os.getenv("LLM_CHOICE", "").strip()
    if not api_key or not base_url or not model:
        raise RuntimeError("primary agent model configuration is incomplete")
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    profile_result = await _get_or_create_profile(document_id, target, source, client, model)
    target_name = "Simplified Chinese" if target == "zh" else "English"
    prompt = (
        "Translate only the selected academic text faithfully. Use the document terminology profile for consistency. "
        f"Target language: {target_name}. Preserve equations, numbers, citations, units, code, and abbreviations. "
        "Do not summarize, add explanations, or output labels. Return only the translation.\n\n"
        f"Terminology profile:\n{json.dumps(profile_result['profile'], ensure_ascii=False)}\n\n"
        f"Previous context:\n{_normalized_text(context_before)[-1200:]}\n\n"
        f"Selected text:\n{source_text}\n\n"
        f"Following context:\n{_normalized_text(context_after)[:1200]}"
    )
    translated = await _complete(client, model, prompt, max_tokens=min(6000, max(700, len(source_text) * 2)))
    if not translated:
        raise RuntimeError("translation model returned empty content")
    saved = await save_selection_translation(
        document_id=document_id,
        target_language=target,
        source_sha256=source_hash,
        selection_sha256=selection_hash,
        source_text=source_text,
        translated_text=translated,
        context_before=_normalized_text(context_before)[-1200:],
        context_after=_normalized_text(context_after)[:1200],
        model=model,
    )
    return {**saved, "cached": False, "profile_cached": profile_result["cached"]}
