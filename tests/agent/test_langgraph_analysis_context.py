from types import SimpleNamespace

import pytest
import langchain.agents as langchain_agents

if not hasattr(langchain_agents, "create_agent"):
    langchain_agents.create_agent = lambda *args, **kwargs: None

from agent.agent_langgraph import local_retrieval_node, run_langgraph_analysis
from agent.agent_runtime import AgentDependencies


@pytest.mark.asyncio
async def test_run_langgraph_analysis_accepts_context_prompt(monkeypatch):
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
    assert "metadata" in result.raw_state
    assert isinstance(result.raw_state["metadata"], dict)


@pytest.mark.asyncio
async def test_run_langgraph_analysis_direct_answer_skips_document_inspection(monkeypatch):
    async def fake_plan_user_intent_debug(**_kwargs):
        return {
            "normalized_plan": {
                "intent": "direct_answer",
                "needs_retrieval": False,
                "retrieval_steps": [],
                "max_tools": 0,
                "allow_external_sources": False,
                "evidence_policy": "answer_with_available_evidence_and_state_uncertainty",
                "direct_answer_allowed": True,
                "rewrite_allowed": True,
                "reason": "no retrieval needed",
                "warnings": [],
            },
            "fallback_used": False,
            "fallback_reason": "",
            "fallback_decision": "",
            "raw_model_content_preview": "",
        }

    async def fail_list_documents(_payload):
        raise AssertionError("inspect_documents should be skipped for direct answer path")

    class DummyModel:
        async def ainvoke(self, _messages):
            return SimpleNamespace(content="直接回答即可。")

    monkeypatch.setattr("agent.agent_langgraph.plan_user_intent_debug", fake_plan_user_intent_debug)
    monkeypatch.setattr("agent.agent_langgraph.get_langchain_chat_model", lambda: DummyModel())
    monkeypatch.setattr("agent.agent_langgraph.list_documents_tool", fail_list_documents)

    deps = AgentDependencies(session_id="s2", user_id="u2")
    result = await run_langgraph_analysis(
        question="简单介绍一下你的能力",
        deps=deps,
        context_prompt="",
    )

    assert result.message
    assert result.metadata["retrieval_skipped_by_planner"] is True


@pytest.mark.asyncio
async def test_graph_expansion_requires_planner_approval_and_respects_strict_scope(monkeypatch):
    async def fake_neighbors(_document_ids, limit, relation_types=None, direction="both"):
        assert limit == 4
        assert relation_types == ["cites"]
        assert direction == "outgoing"
        return ["neighbor-1"]

    monkeypatch.setattr("agent.agent_langgraph.get_graph_neighbor_document_ids", fake_neighbors)
    monkeypatch.setattr("agent.agent_langgraph.build_langchain_tools", lambda _deps: [])

    deps = AgentDependencies(
        session_id="s3",
        search_preferences={"use_paper_graph": True},
    )
    base_state = {
        "deps": deps,
        "question": "跨论文比较方法",
        "current_query": "跨论文比较方法",
        "metadata": {"planner_used": True},
        "intent_plan": {
            "intent": "multi_paper_compare",
            "needs_retrieval": True,
            "retrieval_steps": [{"tool": "hybrid_search", "query": "跨论文比较方法", "limit": 5}],
            "max_tools": 1,
            "use_paper_graph": True,
            "graph_usage_reason": "cross_paper_comparison",
            "graph_relation_types": ["cites"],
            "graph_direction": "outgoing",
            "graph_neighbor_limit": 4,
        },
        "target_documents": [{"document_id": "seed-1", "document_language": "en"}],
        "scope_policy": "prefer_target",
        "allow_supplemental": True,
        "retrieval_results": [],
        "retrieval_attempt_count": 0,
        "retrieval_attempts": [],
        "planning_only": True,
        "max_retrieval_attempts": 2,
    }

    planned = await local_retrieval_node(base_state)
    assert planned["metadata"]["paper_graph_used"] is True
    assert planned["metadata"]["graph_expanded_document_ids"] == ["neighbor-1"]
    assert planned["metadata"]["paper_graph_relation_policy_fallback"] is False

    strict_state = dict(base_state)
    strict_state["metadata"] = {"planner_used": True}
    strict_state["scope_policy"] = "strict_target"
    strict = await local_retrieval_node(strict_state)
    assert strict["metadata"]["paper_graph_used"] is False
    assert strict["metadata"]["paper_graph_usage_reason"] == "blocked_by_strict_target_scope"
