import asyncio

import pytest

from agent.intent_planner import IntentPlan, PlannerCapabilities, RetrievalStep
from agent.models import ChunkResult
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


class _RecordingFakeTool(_FakeTool):
    def __init__(self, name, output=None, error=None):
        super().__init__(name, output=output, error=error)
        self.calls = []

    async def ainvoke(self, args):
        self.calls.append(dict(args))
        return await super().ainvoke(args)


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


def test_execute_intent_plan_steps_keeps_external_provider_failure_out_of_evidence():
    plan = IntentPlan(
        intent="web_information",
        needs_retrieval=True,
        retrieval_steps=[RetrievalStep(tool="web_search", query="latest", limit=5)],
        max_tools=1,
    )
    tools = [
        _FakeTool(
            "search_web",
            [{"_external_retrieval_status": {
                "source_type": "general_web",
                "provider": "test-web",
                "state": "provider_error",
                "retry_count": 1,
            }}],
        )
    ]

    out = asyncio.run(
        execute_intent_plan_steps(
            plan,
            tools,
            fallback_query="latest",
            capabilities=PlannerCapabilities(web_search_enabled=True),
        )
    )

    assert out["results"] == []
    assert out["tools_executed"][0]["status"] == "provider_error"
    assert out["external_retrieval_statuses"][0]["source_type"] == "general_web"


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


@pytest.mark.asyncio
async def test_execute_intent_plan_steps_enforces_selected_document_scope_and_language():
    plan = IntentPlan(
        intent="local_paper_qa",
        needs_retrieval=True,
        retrieval_steps=[
            RetrievalStep(tool="hybrid_search", query="how does it work", limit=5),
            RetrievalStep(tool="artifact_search", query="curve", limit=5),
        ],
        max_tools=2,
    )
    hybrid = _RecordingFakeTool("hybrid_search")
    artifact = _RecordingFakeTool("artifact_search")

    await execute_intent_plan_steps(
        plan,
        [hybrid, artifact],
        fallback_query="q",
        target_document_ids=["doc-a", "doc-b"],
        target_embedding_language="en",
        enforce_target_scope=True,
    )

    assert hybrid.calls[0]["document_ids"] == ["doc-a", "doc-b"]
    assert hybrid.calls[0]["embedding_language"] == "en"
    assert artifact.calls[0]["document_ids"] == ["doc-a", "doc-b"]
    assert artifact.calls[0]["embedding_language"] == "en"


@pytest.mark.asyncio
async def test_execute_intent_plan_steps_keeps_pydantic_chunk_results_for_generation():
    plan = IntentPlan(
        intent="local_artifact_qa",
        needs_retrieval=True,
        retrieval_steps=[RetrievalStep(tool="artifact_search", query="rewiring", limit=5)],
        max_tools=1,
    )
    result = ChunkResult(
        chunk_id="alg-1",
        document_id="doc-1",
        content="Algorithm: Rewire",
        score=0.87,
        document_title="Target paper",
        document_source="paper.pdf",
        metadata={"artifact_type": "algorithm"},
    )
    out = await execute_intent_plan_steps(
        plan,
        [_FakeTool("artifact_search", [result])],
        fallback_query="rewiring",
    )
    assert len(out["results"]) == 1
    assert out["results"][0]["content"] == "Algorithm: Rewire"
    assert out["results"][0]["metadata"]["source_type"] == "local_artifact"


def test_langchain_tool_argument_schemas_keep_scoped_document_and_language_fields():
    from agent.langchain_tools import ArtifactSearchArgs, SectionSearchArgs

    artifact = ArtifactSearchArgs(
        query="rewiring",
        document_ids=["doc-1"],
        embedding_language="en",
    )
    section = SectionSearchArgs(
        query="rewiring",
        section_query="method",
        document_ids=["doc-1"],
    )
    assert artifact.document_ids == ["doc-1"]
    assert artifact.embedding_language == "en"
    assert section.document_ids == ["doc-1"]
