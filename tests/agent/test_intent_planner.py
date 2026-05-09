import pytest

from agent.intent_planner import (
    IntentPlan,
    PlannerCapabilities,
    RetrievalStep,
    build_fallback_intent_plan,
    build_intent_planner_prompt,
    build_retry_intent_planner_prompt,
    infer_artifact_evidence_need,
    normalize_intent_plan,
    plan_user_intent,
    plan_user_intent_debug,
    revise_intent_plan_for_retry,
)


def _policy(plan: IntentPlan) -> dict:
    return dict(plan.answer_policy or {})


def _req(plan: IntentPlan) -> dict:
    return dict(plan.source_requirements or {})


def test_normalize_plan_truncates_steps_and_filters_types_and_bounds_limit():
    raw = {
        "intent": "local_artifact_qa",
        "needs_retrieval": True,
        "max_tools": 2,
        "retrieval_steps": [
            {
                "tool": "artifact_search",
                "query": "metrics table",
                "limit": 999,
                "artifact_types": ["table", "bad_type", "figure"],
            },
            {
                "tool": "section_search",
                "query": "method",
                "section_query": "Method",
                "limit": -5,
            },
            {
                "tool": "web_search",
                "query": "extra",
                "limit": 10,
            },
        ],
    }
    plan = normalize_intent_plan(raw)
    assert len(plan.retrieval_steps) == 2
    assert plan.retrieval_steps[0].limit == 50
    assert plan.retrieval_steps[1].limit == 1
    assert plan.retrieval_steps[0].artifact_types == ["table", "figure"]


def test_normalize_local_artifact_plan_keeps_only_valid_artifact_types():
    raw = {
        "intent": "local_artifact_qa",
        "needs_retrieval": True,
        "retrieval_steps": [
            {
                "tool": "artifact_search",
                "query": "performance comparison and pipeline visualization",
                "limit": 8,
                "artifact_types": ["figure", "invalid", "table"],
            }
        ],
        "max_tools": 1,
    }
    plan = normalize_intent_plan(raw)
    assert plan.intent == "local_artifact_qa"
    assert len(plan.retrieval_steps) == 1
    assert plan.retrieval_steps[0].tool == "artifact_search"
    assert plan.retrieval_steps[0].artifact_types == ["figure", "table"]


def test_normalize_plan_needs_retrieval_false_clears_steps():
    raw = {
        "needs_retrieval": False,
        "retrieval_steps": [{"tool": "hybrid_search", "query": "x", "limit": 10}],
    }
    plan = normalize_intent_plan(raw)
    assert plan.intent == "direct_answer"
    assert plan.needs_retrieval is False
    assert plan.direct_answer_allowed is True
    assert plan.retrieval_steps == []


def test_normalize_direct_answer_plan_keeps_no_retrieval():
    raw = {
        "intent": "direct_answer",
        "needs_retrieval": False,
        "direct_answer_allowed": True,
        "retrieval_steps": [{"tool": "hybrid_search", "query": "unused", "limit": 10}],
    }
    plan = normalize_intent_plan(raw)
    assert plan.intent == "direct_answer"
    assert plan.needs_retrieval is False
    assert plan.direct_answer_allowed is True
    assert plan.retrieval_steps == []


def test_normalize_missing_intent_with_no_retrieval_becomes_direct_answer():
    raw = {
        "needs_retrieval": False,
        "retrieval_steps": [],
    }
    plan = normalize_intent_plan(raw)
    assert plan.intent == "direct_answer"
    assert plan.needs_retrieval is False
    assert plan.retrieval_steps == []
    assert plan.direct_answer_allowed is True
    assert plan.max_tools == 0


def test_normalize_extra_direct_answer_field_still_becomes_direct_answer():
    raw = {
        "needs_retrieval": False,
        "retrieval_steps": [],
        "direct_answer": "some answer text",
    }
    plan = normalize_intent_plan(raw)
    assert plan.intent == "direct_answer"
    assert plan.needs_retrieval is False
    assert plan.retrieval_steps == []
    assert plan.direct_answer_allowed is True


def test_normalize_missing_intent_with_retrieval_steps_keeps_retrieval():
    raw = {
        "needs_retrieval": True,
        "retrieval_steps": [
            {"tool": "hybrid_search", "query": "根据上传论文总结方法贡献", "limit": 10},
        ],
    }
    plan = normalize_intent_plan(raw)
    assert plan.needs_retrieval is True
    assert len(plan.retrieval_steps) == 1
    assert plan.retrieval_steps[0].tool == "hybrid_search"
    assert plan.intent == "local_paper_qa"


@pytest.mark.asyncio
async def test_plan_user_intent_parses_valid_json():
    class FakeModel:
        async def ainvoke(self, _prompt: str):
            return (
                '{"intent":"local_section_qa","needs_retrieval":true,'
                '"retrieval_steps":[{"tool":"section_search","query":"compare","section_query":"Results","limit":5}],'
                '"max_tools":1,"allow_external_sources":false}'
            )

    plan = await plan_user_intent("compare results", model=FakeModel())
    assert plan.intent == "local_section_qa"
    assert len(plan.retrieval_steps) == 1
    assert plan.retrieval_steps[0].tool == "section_search"


@pytest.mark.asyncio
async def test_plan_user_intent_fallback_to_direct_answer_for_writing_help():
    class FakeModel:
        async def ainvoke(self, _prompt: str):
            return ""

    payload = await plan_user_intent_debug("请帮我把这句话说得更自然一点", model=FakeModel())
    plan = payload["normalized_plan"]
    assert payload["fallback_used"] is True
    assert payload["fallback_decision"] == "direct_answer_fallback"
    assert plan["intent"] == "direct_answer"
    assert plan["needs_retrieval"] is False
    assert plan["retrieval_steps"] == []


@pytest.mark.asyncio
async def test_plan_user_intent_fallback_to_direct_answer_for_general_explanation():
    class FakeModel:
        async def ainvoke(self, _prompt: str):
            return "not-json"

    payload = await plan_user_intent_debug("RAG 是什么", model=FakeModel())
    plan = payload["normalized_plan"]
    assert payload["fallback_used"] is True
    assert payload["fallback_decision"] == "direct_answer_fallback"
    assert plan["intent"] == "direct_answer"
    assert plan["needs_retrieval"] is False
    assert plan["retrieval_steps"] == []


@pytest.mark.asyncio
async def test_plan_user_intent_fallback_to_local_hybrid_for_uploaded_paper_evidence():
    class FakeModel:
        async def ainvoke(self, _prompt: str):
            return "not-json"

    payload = await plan_user_intent_debug("根据上传论文总结方法贡献", model=FakeModel())
    plan = payload["normalized_plan"]
    assert payload["fallback_used"] is True
    assert payload["fallback_decision"] == "local_retrieval_fallback"
    assert plan["needs_retrieval"] is True
    assert len(plan["retrieval_steps"]) == 1
    assert plan["retrieval_steps"][0]["tool"] == "hybrid_search"


@pytest.mark.asyncio
async def test_plan_user_intent_keeps_valid_artifact_types_and_external_flag():
    class FakeModel:
        async def ainvoke(self, _prompt: str):
            return {
                "intent": "external_paper_discovery",
                "needs_retrieval": True,
                "retrieval_steps": [
                    {
                        "tool": "artifact_search",
                        "query": "algorithm pseudo code",
                        "limit": 8,
                        "artifact_types": ["algorithm", "x"],
                    },
                    {"tool": "openalex_search", "query": "related work", "limit": 5},
                ],
                "max_tools": 2,
                "allow_external_sources": False,
            }

    caps = PlannerCapabilities(
        openalex_search_enabled=True,
        web_search_enabled=True,
        artifact_search_enabled=True,
    )
    plan = await plan_user_intent("find related papers", model=FakeModel(), capabilities=caps)
    assert plan.retrieval_steps[0].artifact_types == ["algorithm"]
    assert plan.allow_external_sources is True
    assert any(step.tool == "openalex_search" for step in plan.retrieval_steps)


def test_build_fallback_intent_plan_prefers_direct_answer_without_evidence_dependency():
    plan = build_fallback_intent_plan("请帮我把这句话说得更自然一点")
    assert plan.intent == "direct_answer"
    assert plan.needs_retrieval is False
    assert plan.direct_answer_allowed is True


def test_build_fallback_intent_plan_uses_local_search_for_document_dependency():
    plan = build_fallback_intent_plan("请根据上传文档总结实验结果")
    assert plan.needs_retrieval is True
    assert len(plan.retrieval_steps) == 1
    assert plan.retrieval_steps[0].tool == "hybrid_search"


def test_build_fallback_intent_plan_uses_external_when_available():
    caps = PlannerCapabilities(openalex_search_enabled=True, web_search_enabled=True)
    plan = build_fallback_intent_plan("帮我找相关论文并给出 DOI", capabilities=caps)
    assert plan.needs_retrieval is True
    assert plan.retrieval_steps[0].tool in {"openalex_search", "web_search"}


def test_planner_prompt_is_generic_and_budgeted():
    prompt = build_intent_planner_prompt("test question")
    assert "at most 2 tools" in prompt.lower() or "max" in prompt.lower()
    assert "minimal necessary retrieval" in prompt.lower()
    assert "question depends on" in prompt.lower()
    assert "not an exhaustive rule list" in prompt.lower()
    assert "do not retrieve merely because retrieval tools are available" in prompt.lower()
    assert "directly relevant to the user question" in prompt.lower()
    assert "do not enumerate all artifacts by default" in prompt.lower()
    assert "prefer hybrid_search or section_search" in prompt.lower()
    for banned in ["HA-RRT", "HMA-RRT", "Table 4", "Fig. 6", "吃什么", "25岁", "老不老"]:
        assert banned not in prompt


def test_retry_planner_prompt_is_generic_and_relevant_only():
    previous = IntentPlan(
        intent="local_paper_qa",
        needs_retrieval=True,
        retrieval_steps=[RetrievalStep(tool="hybrid_search", query="method summary")],
        max_tools=1,
    )
    prompt = build_retry_intent_planner_prompt(
        question="请解释方法流程和实验对比",
        previous_plan=previous,
        missing_aspects=["缺少非正文证据"],
        retrieval_summary="Only prose chunks were found.",
        suggested_query="method process and metric comparison",
    )
    assert "directly relevant to the question and missing aspects" in prompt.lower()
    assert "instead of defaulting to all artifact types" in prompt.lower()
    for banned in ["HA-RRT", "HMA-RRT", "Table 2", "Fig. 3", "Algorithm 1"]:
        assert banned not in prompt


def test_infer_artifact_evidence_need_is_narrow_and_not_default_all():
    inferred = infer_artifact_evidence_need("请根据上传论文的流程图解释方法 pipeline")
    assert inferred["needs_artifact"] is True
    assert inferred["artifact_types"] == ["figure"]

    inferred = infer_artifact_evidence_need("请根据上传论文中的指标对比和消融结果解释性能差异")
    assert inferred["needs_artifact"] is True
    assert inferred["artifact_types"] == ["table"]


@pytest.mark.asyncio
async def test_revise_intent_plan_for_retry_fallback_artifact():
    previous = IntentPlan(
        intent="local_paper_qa",
        needs_retrieval=True,
        retrieval_steps=[RetrievalStep(tool="hybrid_search", query="q")],
    )
    plan = await revise_intent_plan_for_retry(
        question="analyze missing evidence",
        previous_plan=previous,
        missing_aspects=["table metrics missing"],
        retrieval_summary="No hits",
        suggested_query="metrics table",
        model=None,
    )
    assert plan.needs_retrieval is True
    assert len(plan.retrieval_steps) >= 1
    assert plan.retrieval_steps[0].tool == "artifact_search"
    assert plan.max_tools <= 2
    assert plan.retrieval_steps[0].artifact_types == ["table"]


def test_build_fallback_intent_plan_can_target_relevant_artifact_only():
    caps = PlannerCapabilities(
        hybrid_search_enabled=True,
        artifact_search_enabled=True,
        max_tools=2,
    )
    plan = build_fallback_intent_plan(
        "请根据上传论文里的流程图解释方法 pipeline",
        capabilities=caps,
    )
    assert plan.intent == "local_artifact_qa"
    assert plan.needs_retrieval is True
    assert len(plan.retrieval_steps) == 1
    assert plan.retrieval_steps[0].tool == "artifact_search"
    assert plan.retrieval_steps[0].artifact_types == ["figure"]
    assert plan.max_tools <= 2


def test_capabilities_filter_unavailable_external_tools_without_local_cues():
    caps = PlannerCapabilities(
        hybrid_search_enabled=True,
        web_search_enabled=False,
        openalex_search_enabled=False,
        artifact_search_enabled=False,
        max_tools=2,
    )
    raw = {
        "intent": "external_paper_discovery",
        "needs_retrieval": True,
        "retrieval_steps": [
            {"tool": "web_search", "query": "latest"},
            {"tool": "openalex_search", "query": "related work"},
        ],
        "max_tools": 2,
    }
    plan = normalize_intent_plan(raw, capabilities=caps)
    assert plan.intent == "direct_answer"
    assert plan.needs_retrieval is False
    assert plan.retrieval_steps == []
    assert plan.direct_answer_allowed is True
    assert any("filtered_unavailable_tools" in w for w in plan.warnings)
    assert "external_retrieval_unavailable" in plan.warnings
    assert "fallback_to_hybrid_after_filter" not in plan.warnings


def test_web_request_does_not_fallback_to_local_when_web_unavailable():
    caps = PlannerCapabilities(
        hybrid_search_enabled=True,
        web_search_enabled=False,
        openalex_search_enabled=False,
        direct_answer_enabled=True,
        max_tools=2,
    )
    raw = {
        "intent": "web_information",
        "needs_retrieval": True,
        "retrieval_steps": [
            {"tool": "web_search", "query": "联网查一下 RRT* 最新资料", "limit": 5},
        ],
        "max_tools": 1,
        "direct_answer_allowed": False,
    }
    plan = normalize_intent_plan(raw, capabilities=caps)
    assert plan.needs_retrieval is False
    assert plan.retrieval_steps == []
    assert plan.direct_answer_allowed is True
    assert "web_search_unavailable" in plan.warnings or "external_retrieval_unavailable" in plan.warnings
    assert "fallback_to_hybrid_after_filter" not in plan.warnings


def test_web_request_with_local_cue_can_keep_local_when_web_unavailable():
    caps = PlannerCapabilities(
        hybrid_search_enabled=True,
        web_search_enabled=False,
        openalex_search_enabled=False,
        direct_answer_enabled=True,
        max_tools=2,
    )
    raw = {
        "intent": "web_information",
        "needs_retrieval": True,
        "retrieval_steps": [
            {"tool": "web_search", "query": "根据知识库论文总结 RRT*，并联网查一下最新资料", "limit": 5},
        ],
        "max_tools": 1,
        "direct_answer_allowed": False,
    }
    plan = normalize_intent_plan(raw, capabilities=caps)
    assert plan.needs_retrieval is True
    assert len(plan.retrieval_steps) == 1
    assert plan.retrieval_steps[0].tool == "hybrid_search"
    assert "web_search_unavailable" in plan.warnings or "external_retrieval_unavailable" in plan.warnings


def test_local_paper_request_does_not_require_external_academic():
    caps = PlannerCapabilities(
        hybrid_search_enabled=True,
        section_search_enabled=True,
        openalex_search_enabled=False,
        web_search_enabled=False,
        direct_answer_enabled=True,
    )
    raw = {
        "intent": "local_paper_qa",
        "needs_retrieval": True,
        "retrieval_steps": [
            {
                "tool": "hybrid_search",
                "query": "总结知识库里 Hybrid-RRT 这篇论文的方法流程，并说明依据来自哪些章节",
                "limit": 5,
            }
        ],
        "max_tools": 1,
    }
    plan = normalize_intent_plan(raw, capabilities=caps)
    req = _req(plan)
    pol = _policy(plan)
    assert plan.needs_retrieval is True
    assert "local_kb" in req.get("required_source_types", [])
    assert "external_academic" not in req.get("required_source_types", [])
    assert "external_academic" not in pol.get("unavailable_required_sources", [])
    assert pol.get("must_disclose_limitations") is not True


def test_mixed_local_and_web_request_keeps_local_retrieval_when_web_unavailable():
    caps = PlannerCapabilities(
        hybrid_search_enabled=True,
        web_search_enabled=False,
        openalex_search_enabled=False,
        direct_answer_enabled=True,
    )
    raw = {
        "intent": "web_information",
        "needs_retrieval": True,
        "retrieval_steps": [
            {"tool": "web_search", "query": "根据知识库论文总结 RRT*，并联网查一下最新资料", "limit": 5},
        ],
        "max_tools": 1,
    }
    plan = normalize_intent_plan(raw, capabilities=caps)
    pol = _policy(plan)
    assert plan.needs_retrieval is True
    assert any(step.tool in {"hybrid_search", "vector_search"} for step in plan.retrieval_steps)
    assert "general_web" in pol.get("unavailable_required_sources", [])
    assert pol.get("must_disclose_limitations") is True
    assert "local_kb" not in pol.get("blocked_source_types", [])
    assert "latest/general web materials are unavailable" in str(pol.get("guidance_to_answer_agent", ""))


def test_web_only_unavailable_does_not_fallback_local():
    caps = PlannerCapabilities(
        hybrid_search_enabled=True,
        web_search_enabled=False,
        openalex_search_enabled=False,
        direct_answer_enabled=True,
    )
    raw = {
        "intent": "web_information",
        "needs_retrieval": True,
        "retrieval_steps": [
            {"tool": "web_search", "query": "联网查一下 RRT* 最新资料", "limit": 5},
        ],
        "max_tools": 1,
    }
    plan = normalize_intent_plan(raw, capabilities=caps)
    pol = _policy(plan)
    assert plan.needs_retrieval is False
    assert plan.retrieval_steps == []
    assert "general_web" in pol.get("unavailable_required_sources", [])
    assert "local_kb" in pol.get("blocked_source_types", [])
    assert "fallback_to_hybrid_after_filter" not in plan.warnings


def test_domain_technical_question_may_use_local_kb():
    caps = PlannerCapabilities(
        hybrid_search_enabled=True,
        direct_answer_enabled=True,
    )
    raw = {
        "intent": "local_paper_qa",
        "needs_retrieval": True,
        "retrieval_steps": [
            {"tool": "hybrid_search", "query": "RRT* 和 Informed RRT* 有什么区别", "limit": 5},
        ],
        "max_tools": 1,
    }
    plan = normalize_intent_plan(raw, capabilities=caps)
    assert (
        (plan.needs_retrieval is True and any(step.tool == "hybrid_search" for step in plan.retrieval_steps))
        or (plan.intent in {"direct_answer", "general_technical_explanation"} and plan.needs_retrieval is False)
    )


def test_requested_source_unavailable_does_not_use_incompatible_substitute():
    caps = PlannerCapabilities(
        hybrid_search_enabled=True,
        web_search_enabled=False,
        openalex_search_enabled=False,
        direct_answer_enabled=True,
    )
    raw = {
        "intent": "web_information",
        "needs_retrieval": True,
        "retrieval_steps": [{"tool": "web_search", "query": "latest public docs", "limit": 5}],
        "max_tools": 1,
    }
    plan = normalize_intent_plan(raw, capabilities=caps)
    pol = _policy(plan)
    assert plan.needs_retrieval is False
    assert "local_kb" in pol.get("blocked_source_types", [])
    assert "general_web" in pol.get("unavailable_required_sources", [])
    assert pol.get("mode") in {"direct_answer", "answer_with_disclosure"}
    assert "fallback_to_hybrid_after_filter" not in plan.warnings


def test_requested_local_kb_does_not_use_model_knowledge_as_paper_evidence():
    caps = PlannerCapabilities(
        hybrid_search_enabled=False,
        vector_search_enabled=False,
        section_search_enabled=False,
        artifact_search_enabled=False,
        direct_answer_enabled=True,
    )
    raw = {
        "intent": "local_paper_qa",
        "needs_retrieval": True,
        "retrieval_steps": [{"tool": "hybrid_search", "query": "based on uploaded papers", "limit": 5}],
    }
    plan = normalize_intent_plan(raw, capabilities=caps)
    pol = _policy(plan)
    assert plan.needs_retrieval is False
    assert pol.get("must_disclose_limitations") is True
    assert "model_knowledge" in pol.get("blocked_source_types", [])


def test_unspecified_source_can_use_domain_local_kb():
    caps = PlannerCapabilities(
        hybrid_search_enabled=True,
        direct_answer_enabled=True,
    )
    raw = {
        "intent": "general_technical_explanation",
        "needs_retrieval": False,
        "retrieval_steps": [],
        "direct_answer_allowed": True,
    }
    plan = normalize_intent_plan(raw, capabilities=caps)
    pol = _policy(plan)
    assert pol.get("mode") in {"direct_answer", "retrieve_and_answer", "answer_with_disclosure"}


def test_mixed_source_request_keeps_compatible_part_and_discloses_missing_part():
    caps = PlannerCapabilities(
        hybrid_search_enabled=True,
        openalex_search_enabled=False,
        web_search_enabled=False,
        direct_answer_enabled=True,
    )
    raw = {
        "intent": "external_paper_discovery",
        "needs_retrieval": True,
        "retrieval_steps": [
            {"tool": "hybrid_search", "query": "uploaded paper evidence", "limit": 5},
            {"tool": "openalex_search", "query": "related work doi authors year", "limit": 5},
        ],
        "max_tools": 2,
    }
    plan = normalize_intent_plan(raw, capabilities=caps)
    pol = _policy(plan)
    assert any(step.tool == "hybrid_search" for step in plan.retrieval_steps) or (plan.needs_retrieval is False)
    assert "external_academic" in pol.get("unavailable_required_sources", [])
    assert pol.get("must_disclose_limitations") is True


def test_casual_question_direct_answer():
    caps = PlannerCapabilities(direct_answer_enabled=True)
    raw = {
        "intent": "direct_answer",
        "needs_retrieval": False,
        "retrieval_steps": [],
        "direct_answer_allowed": True,
    }
    plan = normalize_intent_plan(raw, capabilities=caps)
    pol = _policy(plan)
    assert plan.needs_retrieval is False
    assert pol.get("mode") == "direct_answer"
