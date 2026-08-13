from unittest.mock import AsyncMock, patch

import pytest

from agent.document_reader_runtime import _split_markdown_sections, stream_document_translation, translate_document


def test_split_markdown_sections_preserves_heading_order_and_long_sections():
    source = "# Abstract\nIntro\n\n# Method\n" + ("detail paragraph\n\n" * 20) + "# Results\nOutcome"

    sections = _split_markdown_sections(source, max_chars=100)

    assert sections[0].startswith("# Abstract")
    assert any(section.startswith("# Method") for section in sections)
    assert sections[-1].startswith("# Results")


@pytest.mark.asyncio
async def test_translation_reuses_primary_agent_model_and_cache():
    cached = {"id": "translation-1", "content_markdown": "cached", "model": "deepseek-flash"}
    with patch("agent.document_reader_runtime.get_document", new=AsyncMock(return_value={"content": "# Title\nBody"})), \
         patch("agent.document_reader_runtime.get_document_translation", new=AsyncMock(return_value=cached)) as get_cached:
        result = await translate_document("00000000-0000-0000-0000-000000000001", "zh")

    assert result["cached"] is True
    assert result["model"] == "deepseek-flash"
    get_cached.assert_awaited_once()


@pytest.mark.asyncio
async def test_translation_stream_returns_cached_result_without_calling_model():
    cached = {"id": "translation-1", "content_markdown": "# \u7f13\u5b58", "model": "deepseek-flash"}
    with patch("agent.document_reader_runtime.get_document", new=AsyncMock(return_value={"content": "# Title\nBody"})), \
         patch("agent.document_reader_runtime.get_document_translation", new=AsyncMock(return_value=cached)):
        events = [event async for event in stream_document_translation("00000000-0000-0000-0000-000000000001", "zh")]

    assert events == [{"type": "complete", **cached, "cached": True, "section_count": 0}]


@pytest.mark.asyncio
async def test_translation_stream_emits_sections_in_source_order_and_saves_final_markdown(monkeypatch):
    source = "# Abstract\nFirst\n\n# Method\nSecond"
    complete = AsyncMock(side_effect=[
        "{\"section\": \"abstract\"}",
        "{\"section\": \"method\"}",
        "{\"terminology\": {}}",
        "\u6458\u8981\u8bd1\u6587",
        "\u65b9\u6cd5\u8bd1\u6587",
    ])
    save = AsyncMock(return_value={"id": "translation-1", "content_markdown": "saved", "model": "test-model"})
    monkeypatch.setenv("OPENAI_API_KEY", "key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test")
    monkeypatch.setenv("LLM_CHOICE", "test-model")
    with patch("agent.document_reader_runtime.get_document", new=AsyncMock(return_value={"content": source})), \
         patch("agent.document_reader_runtime.get_document_translation", new=AsyncMock(return_value=None)), \
         patch("agent.document_reader_runtime.AsyncOpenAI"), \
         patch("agent.document_reader_runtime._complete", new=complete), \
         patch("agent.document_reader_runtime.save_document_translation", new=save):
        events = [event async for event in stream_document_translation("00000000-0000-0000-0000-000000000001", "zh")]

    sections = [event["content_markdown"] for event in events if event["type"] == "section"]
    assert sections == ["\u6458\u8981\u8bd1\u6587", "\u65b9\u6cd5\u8bd1\u6587"]
    assert save.await_args.kwargs["content_markdown"] == "\u6458\u8981\u8bd1\u6587\n\n\u65b9\u6cd5\u8bd1\u6587"
