from types import SimpleNamespace

import pytest

from agent.agent_runtime import AgentDependencies
from agent.models import EvidenceSource
from agent.simple_chat_runtime import (
    choose_simple_chat_strategy,
    run_simple_chat_runtime,
)


def test_choose_simple_chat_strategy_prefers_artifact_mode_for_short_table_question():
    decision = choose_simple_chat_strategy(
        message="这个表说明什么？",
        resolved_query="AIT* 论文里的 Table 2；当前追问：这个表说明什么？",
        is_local_question=True,
        use_react=False,
        use_web_search=False,
    )

    assert decision.enabled is True
    assert decision.mode == "artifact"
    assert decision.artifact_types == ["table"]


def test_choose_simple_chat_strategy_skips_complex_compare_question():
    decision = choose_simple_chat_strategy(
        message="请详细对比这篇论文和 RRT* 的区别。",
        resolved_query="请详细对比这篇论文和 RRT* 的区别。",
        is_local_question=True,
        use_react=False,
        use_web_search=False,
    )

    assert decision.enabled is False
    assert decision.reason == "complex_question"


@pytest.mark.asyncio
async def test_run_simple_chat_runtime_returns_message_and_sources(monkeypatch):
    async def fake_run_artifact_search_payload(**kwargs):
        deps = kwargs["deps"]
        deps.retrieved_sources.append(
            EvidenceSource(
                source_type="local",
                document_id="doc-1",
                document_title="AIT*",
                document_source="aitstar.pdf",
                chunk_id="chunk-1",
                snippet="Table 2 compares success rate and planning time.",
                score=0.88,
                metadata={"artifact_type": "table", "section_title": "Experiments"},
            )
        )
        return [{"chunk_id": "chunk-1"}]

    class _FakeModel:
        async def ainvoke(self, _messages):
            return SimpleNamespace(content="Table 2 主要在比较成功率和规划时间。")

    monkeypatch.setattr(
        "agent.simple_chat_runtime.run_artifact_search_payload",
        fake_run_artifact_search_payload,
    )
    monkeypatch.setattr(
        "agent.simple_chat_runtime.get_simple_chat_model",
        lambda: _FakeModel(),
    )

    deps = AgentDependencies(session_id="simple-1")
    decision = choose_simple_chat_strategy(
        message="这个表说明什么？",
        resolved_query="AIT* 论文里的 Table 2；当前追问：这个表说明什么？",
        is_local_question=True,
        use_react=False,
        use_web_search=False,
    )
    result = await run_simple_chat_runtime(
        deps=deps,
        user_message="这个表说明什么？",
        decision=decision,
    )

    assert result is not None
    assert result.message == "Table 2 主要在比较成功率和规划时间。"
    assert result.metadata["simple_chat_used"] is True
    assert result.metadata["simple_chat_mode"] == "artifact"
    assert result.tools_used[0].tool_name == "artifact_search"
    assert len(result.sources) == 1
