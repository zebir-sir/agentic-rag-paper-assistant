from types import SimpleNamespace

import pytest

from agent.agent_langgraph import run_langgraph_analysis
from agent.agent_runtime import AgentDependencies


@pytest.mark.asyncio
async def test_run_langgraph_analysis_accepts_context_prompt(monkeypatch):
    captured = {}

    async def fake_list_documents(_):
        return []

    def fake_build_tools(_deps):
        class DummySearchTool:
            async def ainvoke(self, payload):
                captured["retrieval_query"] = payload["query"]
                return []

        return [SimpleNamespace(name="search_knowledge_base", ainvoke=DummySearchTool().ainvoke)]

    class DummyModel:
        async def ainvoke(self, _messages):
            return SimpleNamespace(content="ok")

    monkeypatch.setattr("agent.agent_langgraph.get_langchain_chat_model", lambda: DummyModel())
    monkeypatch.setattr("agent.agent_langgraph.list_documents_tool", fake_list_documents)
    monkeypatch.setattr("agent.agent_langgraph.build_langchain_tools", fake_build_tools)

    deps = AgentDependencies(session_id="s1", user_id="u1")
    result = await run_langgraph_analysis(
        question="当前问题",
        deps=deps,
        context_prompt="历史上下文",
    )

    assert result.message
    assert result.raw_state["context_prompt"] == "历史上下文"
    assert result.raw_state["question"] == "当前问题"
    assert captured["retrieval_query"] == "当前问题"
