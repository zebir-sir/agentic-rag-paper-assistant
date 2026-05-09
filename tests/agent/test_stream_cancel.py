import asyncio
import json

import pytest

from agent.stream_registry import (
    register_stream_run,
    unregister_stream_run,
    cancel_stream_run,
    get_stream_run,
)


@pytest.mark.asyncio
async def test_cancel_stream_run_not_found():
    result = await cancel_stream_run("missing-run-id")
    assert result["status"] == "not_found"
    assert result["run_id"] == "missing-run-id"


@pytest.mark.asyncio
async def test_cancel_stream_run_already_finished():
    run_id = "finished-run-id"
    task = asyncio.create_task(asyncio.sleep(0))
    await task
    await register_stream_run(run_id, "session-a", task)
    result = await cancel_stream_run(run_id)
    assert result["status"] == "already_finished"
    await unregister_stream_run(run_id)


@pytest.mark.asyncio
async def test_cancel_stream_run_running_task():
    run_id = "running-run-id"
    task = asyncio.create_task(asyncio.sleep(60))
    await register_stream_run(run_id, "session-b", task)

    result = await cancel_stream_run(run_id)
    assert result["status"] == "cancelled"

    run = await get_stream_run(run_id)
    assert run is not None
    assert run.cancelled_by_user is True

    await asyncio.sleep(0)
    assert task.cancelled() or task.done()
    await unregister_stream_run(run_id)


@pytest.mark.asyncio
async def test_cancel_endpoint_returns_not_found_without_500(monkeypatch):
    from agent import api as api_module

    async def fake_cancel(_run_id: str):
        return {"status": "not_found", "run_id": _run_id}

    monkeypatch.setattr(api_module, "cancel_stream_run", fake_cancel)
    payload = await api_module.cancel_chat_stream("r-1")
    assert payload["run_id"] == "r-1"
    assert payload["status"] == "not_found"


@pytest.mark.asyncio
async def test_chat_stream_cancel_end_to_end_saves_partial_metadata(monkeypatch):
    from agent import api as api_module
    from agent.api import ChatRequest, ChatRuntime

    saved_messages = []

    class FakeDeps:
        def __init__(self):
            self.user_id = "u1"
            self.use_web_search = False
            self.search_preferences = {}
            self.retrieved_sources = []

    runtime = ChatRuntime(
        session_id="session-stream",
        deps=FakeDeps(),
        requested_search_type="hybrid",
        effective_search_type="hybrid",
        explicit_web_request=False,
        effective_use_web_search=False,
        use_react=False,
        full_prompt="prompt",
        langgraph_context_prompt="",
        compression_used=False,
        context_payload={"summary_updated": False},
        is_general_question=False,
        may_need_general_web_search=False,
        explicit_general_web_request=False,
        is_local_question=False,
        has_local_evidence=False,
        workflow_metadata={},
    )

    async def fake_prepare_chat_runtime(_request):
        return runtime

    async def fake_add_message(session_id, role, content, metadata=None):
        saved_messages.append(
            {
                "session_id": session_id,
                "role": role,
                "content": content,
                "metadata": dict(metadata or {}),
            }
        )

    async def fake_refresh_session_metadata(_session_id):
        return None

    async def fake_iter_langchain_agent_stream(_full_prompt, deps=None):
        yield {"type": "text", "content": "partial answer "}
        await asyncio.sleep(10)

    monkeypatch.setattr(api_module, "get_or_create_session", lambda request: asyncio.sleep(0, result="session-stream"))
    monkeypatch.setattr(api_module, "prepare_chat_runtime", fake_prepare_chat_runtime)
    monkeypatch.setattr(api_module, "add_message", fake_add_message)
    monkeypatch.setattr(api_module, "refresh_session_metadata", fake_refresh_session_metadata)
    monkeypatch.setattr(api_module, "iter_langchain_agent_stream", fake_iter_langchain_agent_stream)
    monkeypatch.setattr(api_module, "get_stream_backend", lambda: "langchain")
    monkeypatch.setattr(api_module, "get_agent_backend", lambda: "langchain")
    monkeypatch.setattr(api_module, "_dedupe_sources", lambda items: items)
    monkeypatch.setattr(api_module, "_normalize_web_unavailable_reply", lambda response, **kwargs: response)
    monkeypatch.setattr(api_module, "clean_legacy_warning_text", lambda text, drop_warning=False: text)
    monkeypatch.setattr(api_module, "clean_markdown_spacing", lambda text: text)

    request = ChatRequest(message="test stream cancel", session_id="", user_id="u1", use_react=False)
    response = await api_module.chat_stream(request)

    session_ready = asyncio.Event()
    text_ready = asyncio.Event()
    run_id_box = {"run_id": ""}
    seen_events = []

    async def consume_stream():
        async for chunk in response.body_iterator:
            payload = str(chunk)
            if not payload.startswith("data: "):
                continue
            data = json.loads(payload[6:].strip())
            seen_events.append(data)
            if data.get("type") == "session":
                run_id_box["run_id"] = str(data.get("run_id") or "")
                session_ready.set()
            if data.get("type") == "text":
                text_ready.set()

    consumer = asyncio.create_task(consume_stream())
    await asyncio.wait_for(session_ready.wait(), timeout=2)
    await asyncio.wait_for(text_ready.wait(), timeout=2)

    cancel_payload = await api_module.cancel_chat_stream(run_id_box["run_id"])
    assert cancel_payload["status"] in {"cancelled", "already_finished"}

    await asyncio.wait_for(consumer, timeout=2)

    assistant_messages = [m for m in saved_messages if m["role"] == "assistant"]
    if assistant_messages:
        metadata = assistant_messages[-1]["metadata"]
        assert metadata.get("cancelled") is True
        assert metadata.get("cancelled_by_user") is True
        assert metadata.get("partial_response") is True
        assert metadata.get("run_id") == run_id_box["run_id"]
