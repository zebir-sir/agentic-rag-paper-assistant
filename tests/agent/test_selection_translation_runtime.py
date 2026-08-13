from unittest.mock import AsyncMock, patch

import pytest

from agent.selection_translation_runtime import translate_selection


@pytest.mark.asyncio
async def test_selection_translation_reuses_exact_cached_text():
    cached = {"id": "selection-1", "source_text": "A* search", "translated_text": "A* 搜索", "model": "test"}
    with patch("agent.selection_translation_runtime.get_document", new=AsyncMock(return_value={"content": "# Paper\nA* search"})), \
         patch("agent.selection_translation_runtime.get_selection_translation", new=AsyncMock(return_value=cached)):
        result = await translate_selection("00000000-0000-0000-0000-000000000001", "zh", " A*   search ")

    assert result["cached"] is True
    assert result["translated_text"] == "A* 搜索"


@pytest.mark.asyncio
async def test_selection_translation_builds_profile_then_saves_result(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test")
    monkeypatch.setenv("LLM_CHOICE", "test-model")
    complete = AsyncMock(side_effect=['{"terminology": {"RRT*": "RRT*"}}', "选择译文"])
    with patch("agent.selection_translation_runtime.get_document", new=AsyncMock(return_value={"content": "# Paper\nOriginal source"})), \
         patch("agent.selection_translation_runtime.get_selection_translation", new=AsyncMock(return_value=None)), \
         patch("agent.selection_translation_runtime.get_translation_profile", new=AsyncMock(return_value=None)), \
         patch("agent.selection_translation_runtime.save_translation_profile", new=AsyncMock(return_value={"profile": {"terminology": {}}, "model": "test"})), \
         patch("agent.selection_translation_runtime.save_selection_translation", new=AsyncMock(return_value={"id": "selection-1", "source_text": "Original source", "translated_text": "选择译文", "model": "test"})) as save, \
         patch("agent.selection_translation_runtime.AsyncOpenAI"), \
         patch("agent.selection_translation_runtime._complete", new=complete):
        result = await translate_selection("00000000-0000-0000-0000-000000000001", "zh", "Original source")

    assert result["cached"] is False
    assert save.await_args.kwargs["translated_text"] == "选择译文"
