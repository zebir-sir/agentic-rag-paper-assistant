import pytest

from agent.tools import _expand_section_queries
from agent.intent_planner import (
    IntentPlan,
    PlannerCapabilities,
    RetrievalStep,
    build_fallback_intent_plan,
    build_intent_planner_prompt,
    build_retry_intent_planner_prompt,
    infer_artifact_evidence_need,
    infer_supporting_section_need,
    infer_direct_answer_eligibility,
    infer_general_web_evidence_need,
    infer_paper_graph_need,
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


@pytest.mark.asyncio
async def test_explicit_openalex_request_is_planned_not_short_circuited():
    class FakeModel:
        async def ainvoke(self, _prompt: str):
            return {
                "intent": "direct_answer",
                "needs_retrieval": False,
                "retrieval_steps": [],
                "direct_answer_allowed": True,
            }

    plan = await plan_user_intent(
        "Use OpenAlex to find papers about single-cell spatial transcriptomics",
        model=FakeModel(),
        capabilities=PlannerCapabilities(openalex_search_enabled=True),
    )

    assert plan.intent == "external_paper_discovery"
    assert [step.tool for step in plan.retrieval_steps] == ["openalex_search"]
    assert "single-cell spatial transcriptomics" in plan.retrieval_steps[0].query


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
    assert "person's life/death/status" in prompt.lower()
    assert "even if the user does not explicitly say" in prompt.lower()
    assert "conservative allowlist" in prompt.lower()
    assert "writing transformation" in prompt.lower()
    assert "do not choose direct_answer" in prompt.lower()
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


def test_algorithm_evidence_cues_include_descriptions_and_rewiring_without_overlapping_figures():
    inferred = infer_artifact_evidence_need("解释该论文的算法描述中 rewiring 重连步骤如何改善路径代价")
    assert inferred["needs_artifact"] is True
    assert inferred["artifact_types"] == ["algorithm"]

    context = infer_supporting_section_need("解释该论文的算法描述中 rewiring 重连步骤如何改善路径代价")
    assert context == {"needed": "method_context", "section_query": "method or algorithm"}


def test_explicit_named_paper_artifact_request_plans_typed_evidence_and_context():
    plan = normalize_intent_plan(
        {
            "question": "结合《某篇论文》的算法描述，解释重连步骤为什么有效，并优先给出伪代码证据。",
            "intent": "local_paper_qa",
            "needs_retrieval": True,
            "retrieval_steps": [{"tool": "hybrid_search", "query": "ignored"}],
        }
    )
    assert plan.intent == "local_artifact_qa"
    assert [step.tool for step in plan.retrieval_steps] == ["artifact_search", "section_search"]
    assert plan.retrieval_steps[0].artifact_types == ["algorithm"]
    assert plan.retrieval_steps[1].section_query == "method or algorithm"
    assert _req(plan)["required_source_types"] == ["local_artifact", "local_section"]


def test_infer_paper_graph_need_only_for_cross_paper_relationship_queries():
    assert infer_paper_graph_need("请结合相关论文分析该方法可以迁移到哪些场景")["use_paper_graph"] is True
    assert infer_paper_graph_need("请说明这篇论文第 3 节的实验设置")["use_paper_graph"] is False


def test_explicit_no_retrieval_chinese_explanation_stays_direct_answer():
    plan = normalize_intent_plan(
        {
            "question": "用通俗语言解释什么是采样偏置，不引用本地论文或外部网页。",
            "intent": "unclear",
            "needs_retrieval": True,
            "retrieval_steps": [{"tool": "hybrid_search", "query": "sampling bias"}],
        }
    )
    assert plan.intent == "direct_answer"
    assert plan.needs_retrieval is False
    assert plan.retrieval_steps == []


@pytest.mark.parametrize(
    ("question", "expected_direction", "expected_relations"),
    [
        ("这篇方法基于哪些基础工作？", "outgoing", ["method_lineage", "cites", "semantic_similarity"]),
        ("哪些后续论文扩展或改进了这篇方法？", "incoming", ["method_lineage", "cites", "semantic_similarity"]),
        ("这篇论文引用了哪些工作？", "both", ["cites", "semantic_similarity"]),
    ],
)
def test_infer_paper_graph_need_selects_relation_types_and_direction(question, expected_direction, expected_relations):
    inferred = infer_paper_graph_need(question)
    assert inferred["use_paper_graph"] is True
    assert inferred["direction"] == expected_direction
    assert inferred["relation_types"] == expected_relations


def test_normalize_plan_keeps_model_graph_decision_and_bounds_neighbor_limit():
    plan = normalize_intent_plan(
        {
            "intent": "multi_paper_compare",
            "needs_retrieval": True,
            "retrieval_steps": [{"tool": "hybrid_search", "query": "跨论文比较方法", "limit": 6}],
            "max_tools": 1,
            "use_paper_graph": True,
            "graph_usage_reason": "cross_paper_comparison",
            "graph_relation_types": ["cites", "invalid", "method_lineage"],
            "graph_direction": "incoming",
            "graph_neighbor_limit": 99,
        }
    )
    assert plan.use_paper_graph is True
    assert plan.graph_usage_reason == "cross_paper_comparison"
    assert plan.graph_relation_types == ["cites", "method_lineage"]
    assert plan.graph_direction == "incoming"
    assert plan.graph_neighbor_limit == 12


def test_direct_answer_plan_cannot_enable_paper_graph():
    plan = normalize_intent_plan(
        {
            "intent": "direct_answer",
            "needs_retrieval": False,
            "retrieval_steps": [],
            "use_paper_graph": True,
            "graph_usage_reason": "should_be_ignored",
        }
    )
    assert plan.use_paper_graph is False
    assert plan.graph_usage_reason == ""
    assert plan.graph_relation_types == []
    assert plan.graph_direction == "both"


def test_volatile_real_world_fact_requires_web_without_explicit_search_cue():
    need = infer_general_web_evidence_need("张雪峰还活着吗")
    assert need["needs_web"] is True

    caps = PlannerCapabilities(web_search_enabled=True, direct_answer_enabled=True)
    plan = normalize_intent_plan(
        {
            "question": "张雪峰还活着吗",
            "intent": "direct_answer",
            "needs_retrieval": False,
            "retrieval_steps": [],
            "direct_answer_allowed": True,
        },
        capabilities=caps,
    )
    assert plan.intent == "web_information"
    assert plan.needs_retrieval is True
    assert plan.direct_answer_allowed is False
    assert [step.tool for step in plan.retrieval_steps] == ["web_search"]
    assert "general_web" in _req(plan)["required_source_types"]


@pytest.mark.asyncio
async def test_model_direct_answer_for_volatile_fact_is_corrected_to_web_plan():
    class FakeModel:
        async def ainvoke(self, _prompt: str):
            return {
                "intent": "direct_answer",
                "needs_retrieval": False,
                "retrieval_steps": [],
                "direct_answer_allowed": True,
            }

    plan = await plan_user_intent(
        "张雪峰还活着吗",
        model=FakeModel(),
        capabilities=PlannerCapabilities(web_search_enabled=True, direct_answer_enabled=True),
    )
    assert plan.intent == "web_information"
    assert plan.needs_retrieval is True
    assert [step.tool for step in plan.retrieval_steps] == ["web_search"]


def test_volatile_real_world_fact_discloses_when_web_is_unavailable():
    plan = normalize_intent_plan(
        {
            "question": "张雪峰还活着吗",
            "intent": "direct_answer",
            "needs_retrieval": False,
            "retrieval_steps": [],
            "direct_answer_allowed": True,
        },
        capabilities=PlannerCapabilities(web_search_enabled=False, direct_answer_enabled=True),
    )
    policy = _policy(plan)
    assert plan.needs_retrieval is False
    assert plan.direct_answer_allowed is False
    assert plan.intent == "unclear"
    assert "general_web" in policy["unavailable_required_sources"]
    assert policy["must_disclose_limitations"] is True


@pytest.mark.parametrize(
    ("question", "expected_tool"),
    [
        ("根据上传论文总结方法贡献", "hybrid_search"),
        ("根据上传论文的实验章节说明评估设置", "section_search"),
        ("根据上传论文的消融表解释性能差异", "artifact_search"),
        ("找相关论文并给出 DOI 和发表会议", "openalex_search"),
    ],
)
def test_required_source_class_corrects_an_incompatible_direct_answer(question, expected_tool):
    caps = PlannerCapabilities(
        hybrid_search_enabled=True,
        section_search_enabled=True,
        artifact_search_enabled=True,
        openalex_search_enabled=True,
        direct_answer_enabled=True,
    )
    plan = normalize_intent_plan(
        {
            "question": question,
            "intent": "direct_answer",
            "needs_retrieval": False,
            "retrieval_steps": [],
            "direct_answer_allowed": True,
        },
        capabilities=caps,
    )
    assert plan.needs_retrieval is True
    assert [step.tool for step in plan.retrieval_steps] == [expected_tool]


def test_local_artifact_request_does_not_substitute_hybrid_search():
    caps = PlannerCapabilities(hybrid_search_enabled=True, artifact_search_enabled=True)
    plan = normalize_intent_plan(
        {
            "question": "根据上传论文的消融表解释性能差异",
            "intent": "local_paper_qa",
            "needs_retrieval": True,
            "retrieval_steps": [{"tool": "hybrid_search", "query": "性能差异", "limit": 5}],
            "max_tools": 1,
        },
        capabilities=caps,
    )
    assert [step.tool for step in plan.retrieval_steps] == ["artifact_search"]
    assert plan.retrieval_steps[0].artifact_types == ["table"]


def test_mixed_required_sources_are_not_silently_reclassified_when_tool_budget_is_two():
    caps = PlannerCapabilities(
        hybrid_search_enabled=True,
        openalex_search_enabled=True,
        web_search_enabled=True,
        direct_answer_enabled=True,
        max_tools=2,
    )
    plan = normalize_intent_plan(
        {
            "question": "根据上传论文对比方法，并找相关论文 DOI 和最新政策要求",
            "intent": "direct_answer",
            "needs_retrieval": False,
            "retrieval_steps": [],
            "direct_answer_allowed": True,
        },
        capabilities=caps,
    )
    assert set(_req(plan)["required_source_types"]) == {"local_kb", "external_academic", "general_web"}
    assert len(plan.retrieval_steps) == 2
    assert "general_web" in _req(plan)["required_source_types"]


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
    assert plan.intent == "unclear"
    assert plan.needs_retrieval is False
    assert plan.retrieval_steps == []
    assert plan.direct_answer_allowed is False
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
    assert plan.intent == "unclear"
    assert plan.direct_answer_allowed is False
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
    assert "local_section" in req.get("required_source_types", [])
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
    assert pol.get("mode") in {"ask_clarification", "answer_with_disclosure"}
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


@pytest.mark.parametrize(
    "question",
    [
        "Please rewrite this paragraph in a more formal tone.",
        "Translate this sentence into Chinese.",
        "What is retrieval-augmented generation?",
        "Explain the basic principle of RRT star.",
        "Brainstorm three evaluation dimensions for the user-provided proposal.",
    ],
)
def test_direct_answer_allowlist_accepts_stable_or_user_provided_tasks(question):
    assert infer_direct_answer_eligibility(question)["allowed"] is True


@pytest.mark.parametrize(
    "question",
    [
        "Is Zhang Xuefeng still alive?",
        "Summarize the results in the uploaded paper.",
        "Find related papers and provide their DOI and venue.",
        "What is the current price of this product?",
        "Give me the exact quote from the paper's conclusion.",
    ],
)
def test_direct_answer_allowlist_rejects_evidence_dependent_tasks(question):
    assert infer_direct_answer_eligibility(question)["allowed"] is False


def test_model_direct_answer_without_allowlist_eligibility_becomes_clarification():
    plan = normalize_intent_plan(
        {
            "question": "Introduce Zhang Xuefeng.",
            "intent": "direct_answer",
            "needs_retrieval": False,
            "retrieval_steps": [],
            "direct_answer_allowed": True,
        },
        capabilities=PlannerCapabilities(direct_answer_enabled=True),
    )
    assert plan.intent == "unclear"
    assert plan.needs_retrieval is False
    assert plan.direct_answer_allowed is False
    assert "direct_answer_not_eligible" in plan.warnings


def test_no_tool_fallback_does_not_direct_answer_an_evidence_dependent_question():
    plan = build_fallback_intent_plan(
        "Is Zhang Xuefeng still alive?",
        capabilities=PlannerCapabilities(
            hybrid_search_enabled=False,
            vector_search_enabled=False,
            section_search_enabled=False,
            artifact_search_enabled=False,
            openalex_search_enabled=False,
            web_search_enabled=False,
            direct_answer_enabled=True,
        ),
    )
    assert plan.intent == "unclear"
    assert plan.direct_answer_allowed is False


def test_local_fallback_preserves_paper_graph_relation_intent():
    plan = build_fallback_intent_plan("Which later papers extend or improve this method?")

    assert plan.needs_retrieval is True
    assert plan.use_paper_graph is True
    assert plan.graph_direction == "incoming"
    assert plan.graph_relation_types == ["method_lineage", "cites", "semantic_similarity"]


def test_explicit_paper_section_scope_forces_section_retrieval():
    plan = normalize_intent_plan(
        {
            "question": "只根据《A Motion Planning Paper》的摘要和引言解释问题。",
            "intent": "direct_answer",
            "needs_retrieval": False,
            "retrieval_steps": [],
        },
        capabilities=PlannerCapabilities(),
    )

    assert plan.needs_retrieval is True
    assert plan.retrieval_steps[0].tool == "section_search"


def test_section_query_expands_common_front_matter_aliases():
    queries = _expand_section_queries("Abstract or Introduction")

    assert "abstract" in queries
    assert "summary" in queries
    assert "introduction" in queries
    assert "motivation" in queries
