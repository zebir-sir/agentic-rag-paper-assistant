"""Cached query translation used to bridge multilingual retrieval indexes."""

import asyncio
import hashlib
import os

from openai import AsyncOpenAI

from .cache_utils import cache_get_json, cache_set_json, make_cache_key


QUERY_TRANSLATION_CACHE_TTL_SECONDS = int(os.getenv("QUERY_TRANSLATION_CACHE_TTL_SECONDS", "604800"))
QUERY_TRANSLATION_TIMEOUT_SECONDS = float(os.getenv("QUERY_TRANSLATION_TIMEOUT_SECONDS", "20"))


async def translate_query_to_english(query: str) -> str | None:
    """Translate a non-English search query without changing scientific identifiers."""
    source = str(query or "").strip()
    if not source:
        return None

    query_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
    cache_key = make_cache_key("retrieval_query_translation", "zh", "en", query_hash)
    cached = await cache_get_json(cache_key)
    if isinstance(cached, str) and cached.strip():
        return cached.strip()

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    base_url = os.getenv("OPENAI_BASE_URL", "").strip()
    model = os.getenv("LLM_CHOICE", "").strip()
    if not api_key or not base_url or not model:
        return None

    prompt = (
        "Translate this academic search query into concise English retrieval terms. "
        "Preserve paper titles, algorithm names, abbreviations, symbols, and numbers exactly. "
        "Return only the English query, without explanation.\n\n"
        f"Query: {source}"
    )
    try:
        response = await asyncio.wait_for(
            AsyncOpenAI(api_key=api_key, base_url=base_url).chat.completions.create(
                model=model,
                temperature=0,
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            ),
            timeout=QUERY_TRANSLATION_TIMEOUT_SECONDS,
        )
    except Exception:
        return None

    translated = str(response.choices[0].message.content or "").strip()
    if not translated or translated == source:
        return None
    await cache_set_json(cache_key, translated, QUERY_TRANSLATION_CACHE_TTL_SECONDS)
    return translated
