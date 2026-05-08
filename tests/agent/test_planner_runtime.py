import pytest

from agent.intent_planner import IntentPlan, PlannerCapabilities, RetrievalStep
from agent.planner_runtime import execute_intent_plan_steps


class _FakeTool:
    def __init__(self, name, output=None, error=None):
        self.name = name
        self._output = output or []
        self._error = error

    async def ainvoke(self, _args):
        if self._error is not None:
            raise self._error
        return self._output


@pytest.mark.asyncio
async def test_execute_intent_plan_steps_executes_at_most_two_steps():
    plan = IntentPlan(
        intent="local_paper_qa",
        needs_retrieval=True,
        retrieval_steps=[
            RetrievalStep(tool="hybrid_search", query="q1", limit=5),
            RetrievalStep(tool="section_search", query="q2", section_query="Method", limit=5),
            RetrievalStep(tool="artifact_search", query="q3", limit=5),
        ],
        max_tools=2,
    )
    tools = [
        _FakeTool("hybrid_search", [{"chunk_id": "c1", "content": "a"}]),
        _FakeTool("section_search", [{"chunk_id": "c2", "content": "b"}]),
        _FakeTool("artifact_search", [{"chunk_id": "c3", "content": "c"}]),
    ]
    out = await execute_intent_plan_steps(plan, tools, fallback_query="qq")
    assert len(out["tools_executed"]) == 2
    assert out["tools_executed"][0]["tool"] == "hybrid_search"
    assert out["tools_executed"][1]["tool"] == "section_search"


@pytest.mark.asyncio
async def test_execute_intent_plan_steps_handles_tool_failure_with_warning():
    plan = IntentPlan(
        intent="local_paper_qa",
        needs_retrieval=True,
        retrieval_steps=[RetrievalStep(tool="hybrid_search", query="q", limit=5)],
        max_tools=1,
    )
    tools = [_FakeTool("hybrid_search", error=RuntimeError("boom"))]
    out = await execute_intent_plan_steps(plan, tools, fallback_query="q")
    assert out["results"] == []
    assert out["tools_executed"] == []
    assert any("planned tool failed: hybrid_search" in w for w in out["warnings"])


@pytest.mark.asyncio
async def test_execute_intent_plan_steps_skips_unavailable_tool():
    plan = IntentPlan(
        intent="web_information",
        needs_retrieval=True,
        retrieval_steps=[RetrievalStep(tool="web_search", query="latest", limit=5)],
        max_tools=1,
    )
    caps = PlannerCapabilities(web_search_enabled=False)
    tools = [_FakeTool("search_web", [{"title": "x", "url": "u"}])]
    out = await execute_intent_plan_steps(plan, tools, fallback_query="q", capabilities=caps)
    assert out["tools_executed"] == []
    assert "search_web" in out["filtered_unavailable_tools"]
    assert any("tool_unavailable:search_web" in w for w in out["warnings"])


@pytest.mark.asyncio
async def test_execute_intent_plan_steps_skips_unavailable_openalex():
    plan = IntentPlan(
        intent="external_paper_discovery",
        needs_retrieval=True,
        retrieval_steps=[RetrievalStep(tool="openalex_search", query="related work", limit=5)],
        max_tools=1,
    )
    caps = PlannerCapabilities(openalex_search_enabled=False)
    tools = [_FakeTool("search_openalex_papers", [{"title": "p1", "openalex_id": "oa1"}])]
    out = await execute_intent_plan_steps(plan, tools, fallback_query="q", capabilities=caps)
    assert out["tools_executed"] == []
    assert "search_openalex_papers" in out["filtered_unavailable_tools"]
    assert any("tool_unavailable:search_openalex_papers" in w for w in out["warnings"])
