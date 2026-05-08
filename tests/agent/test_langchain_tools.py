import pytest
from types import SimpleNamespace

from agent.agent_runtime import AgentDependencies
from agent.langchain_tools import build_langchain_tools
from agent.tool_payloads import run_artifact_search_payload


@pytest.mark.asyncio
async def test_build_langchain_tools_includes_artifact_search():
    deps = AgentDependencies(session_id="s1")
    tools = build_langchain_tools(deps)
    names = [getattr(t, "name", "") for t in tools]
    assert "artifact_search" in names


@pytest.mark.asyncio
async def test_artifact_search_tool_calls_payload(monkeypatch):
    captured = {}

    async def fake_run_artifact_search_payload(**kwargs):
        captured.update(kwargs)
        return [{"chunk_id": "c1", "content": "x"}]

    monkeypatch.setattr(
        "agent.langchain_tools.run_artifact_search_payload",
        fake_run_artifact_search_payload,
    )

    deps = AgentDependencies(session_id="s2")
    tools = build_langchain_tools(deps)
    artifact_tool = next(t for t in tools if getattr(t, "name", "") == "artifact_search")
    out = await artifact_tool.ainvoke(
        {
            "query": "compare table metrics",
            "limit": 7,
            "artifact_types": ["table"],
            "document_id": "11111111-1111-1111-1111-111111111111",
            "text_weight": 0.2,
        }
    )

    assert out == [{"chunk_id": "c1", "content": "x"}]
    assert captured["deps"] is deps
    assert captured["query"] == "compare table metrics"
    assert captured["limit"] == 7
    assert captured["artifact_types"] == ["table"]
    assert captured["document_id"] == "11111111-1111-1111-1111-111111111111"
    assert captured["text_weight"] == 0.2


@pytest.mark.asyncio
async def test_run_artifact_search_payload_collects_hits(monkeypatch):
    async def fake_artifact_search_tool(_input):
        return [
            SimpleNamespace(
                chunk_id="chunk-1",
                document_id="doc-1",
                content="artifact evidence",
                score=0.8,
                metadata={"artifact_type": "figure"},
                document_title="Doc 1",
                document_source="doc1.md",
            )
        ]

    monkeypatch.setattr("agent.tool_payloads.artifact_search_tool", fake_artifact_search_tool)

    deps = AgentDependencies(session_id="s3")
    payload = await run_artifact_search_payload(
        deps=deps,
        query="figure pipeline",
        limit=5,
        artifact_types=["figure"],
    )

    assert len(payload) == 1
    assert payload[0]["chunk_id"] == "chunk-1"
    assert len(deps.retrieved_sources) == 1
    assert deps.retrieved_sources[0].chunk_id == "chunk-1"


@pytest.mark.asyncio
async def test_run_artifact_search_payload_error_sets_retrieval_error(monkeypatch):
    async def fake_artifact_search_tool(_input):
        raise RuntimeError("boom")

    monkeypatch.setattr("agent.tool_payloads.artifact_search_tool", fake_artifact_search_tool)

    deps = AgentDependencies(session_id="s4")
    payload = await run_artifact_search_payload(deps=deps, query="algo")

    assert payload == []
    assert "artifact_search_failed:RuntimeError" == deps.search_preferences.get("retrieval_error")
