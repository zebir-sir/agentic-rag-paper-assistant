from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from agent import api as api_module


@pytest.mark.asyncio
async def test_pdf_page_image_is_rendered_off_the_event_loop():
    async def run_in_thread(function, *args):
        return function(*args)

    with patch("agent.api.get_document_pdf", new=AsyncMock(return_value={"content": b"pdf"})), \
         patch("agent.api.render_cached_pdf_page_png", return_value=b"png") as render, \
         patch("agent.api.asyncio.to_thread", new=AsyncMock(side_effect=run_in_thread)) as to_thread:
        response = await api_module.get_document_pdf_page_image_endpoint("document-1", 1)

    assert response.body == b"png"
    assert response.headers["content-type"] == "image/png"
    assert render.call_args.args == (b"pdf", "document-1", 1)
    assert to_thread.await_count == 1


@pytest.mark.asyncio
async def test_selection_translation_reports_provider_outage_without_provider_name():
    with patch("agent.api.translate_selection", new=AsyncMock(side_effect=RuntimeError("connection failed"))), \
         patch("agent.api.is_translation_service_unavailable", return_value=True):
        with pytest.raises(HTTPException) as captured:
            await api_module.translate_selection_endpoint("document-1", "zh", {"selection": "source"})

    assert captured.value.status_code == 503
    assert "翻译模型服务" in str(captured.value.detail)
    assert "DeepSeek" not in str(captured.value.detail)


@pytest.mark.asyncio
async def test_selection_translation_keeps_internal_failures_distinct_from_outages():
    with patch("agent.api.translate_selection", new=AsyncMock(side_effect=RuntimeError("database failed"))), \
         patch("agent.api.is_translation_service_unavailable", return_value=False):
        with pytest.raises(HTTPException) as captured:
            await api_module.translate_selection_endpoint("document-1", "zh", {"selection": "source"})

    assert captured.value.status_code == 500
    assert captured.value.detail == "翻译处理失败"
