import asyncio

from agent.intent_planner import PlannerCapabilities, plan_user_intent_debug


def _planned_tools(payload: dict) -> list[str]:
    plan = payload.get("normalized_plan") or {}
    steps = plan.get("retrieval_steps") or []
    return [str(step.get("tool")) for step in steps if isinstance(step, dict) and step.get("tool")]


def _policy(payload: dict) -> dict:
    plan = payload.get("normalized_plan") or {}
    return dict(plan.get("answer_policy") or {})


def _run(coro):
    return asyncio.run(coro)


def test_source_policy_casual_chat_does_not_retrieve():
    payload = _run(
        plan_user_intent_debug(
            "你多大了",
            model=None,
            capabilities=PlannerCapabilities(direct_answer_enabled=True),
        )
    )
    plan = payload["normalized_plan"]

    assert plan["intent"] == "direct_answer"
    assert plan["needs_retrieval"] is False
    assert plan["retrieval_steps"] == []
    assert plan["direct_answer_allowed"] is True


def test_source_policy_food_advice_does_not_use_local_kb():
    payload = _run(
        plan_user_intent_debug(
            "你觉得我今晚吃什么比较好",
            model=None,
            capabilities=PlannerCapabilities(direct_answer_enabled=True),
        )
    )
    plan = payload["normalized_plan"]
    tools = _planned_tools(payload)

    assert plan["intent"] == "direct_answer"
    assert plan["needs_retrieval"] is False
    assert "hybrid_search" not in tools
    assert "vector_search" not in tools
    assert "section_search" not in tools


def test_source_policy_local_paper_question_requires_local_retrieval():
    payload = _run(
        plan_user_intent_debug(
            "总结知识库里 Hybrid-RRT 这篇论文的方法流程，并说明依据来自哪些章节",
            model=None,
            capabilities=PlannerCapabilities(
                hybrid_search_enabled=True,
                vector_search_enabled=True,
                section_search_enabled=True,
                artifact_search_enabled=True,
                direct_answer_enabled=True,
            ),
        )
    )
    plan = payload["normalized_plan"]
    tools = set(_planned_tools(payload))

    assert plan["intent"] in {"local_paper_qa", "local_section_qa", "multi_paper_compare"}
    assert plan["needs_retrieval"] is True
    assert tools & {"hybrid_search", "vector_search", "section_search", "artifact_search"}


def test_source_policy_web_only_unavailable_does_not_fake_local_web():
    class FakeModel:
        async def ainvoke(self, _prompt: str):
            return {
                "intent": "web_information",
                "needs_retrieval": True,
                "retrieval_steps": [
                    {"tool": "web_search", "query": "联网查一下 RRT* 最新资料", "limit": 5},
                ],
                "max_tools": 1,
            }

    payload = _run(
        plan_user_intent_debug(
            "联网查一下 RRT* 最新资料",
            model=FakeModel(),
            capabilities=PlannerCapabilities(
                hybrid_search_enabled=True,
                web_search_enabled=False,
                openalex_search_enabled=False,
                direct_answer_enabled=True,
            ),
        )
    )
    policy = _policy(payload)
    warnings = list((payload.get("normalized_plan") or {}).get("warnings") or [])

    assert "hybrid_search" not in _planned_tools(payload)
    assert policy.get("must_disclose_limitations") is True
    assert "general_web" in policy.get("unavailable_required_sources", [])
    assert any(
        warning in {"web_search_unavailable", "required_source_unavailable", "external_retrieval_unavailable"}
        or warning.startswith("unavailable_required_sources:")
        for warning in warnings
    )


def test_source_policy_mixed_local_web_keeps_local_and_discloses_web_gap():
    class FakeModel:
        async def ainvoke(self, _prompt: str):
            return {
                "intent": "web_information",
                "needs_retrieval": True,
                "retrieval_steps": [
                    {"tool": "hybrid_search", "query": "根据知识库论文总结 RRT*", "limit": 5},
                    {"tool": "web_search", "query": "联网查一下最新资料", "limit": 5},
                ],
                "max_tools": 2,
            }

    payload = _run(
        plan_user_intent_debug(
            "根据知识库论文总结 RRT*，并联网查一下最新资料",
            model=FakeModel(),
            capabilities=PlannerCapabilities(
                local_search_enabled=True,
                hybrid_search_enabled=True,
                vector_search_enabled=True,
                web_search_enabled=False,
                openalex_search_enabled=False,
                direct_answer_enabled=True,
            ),
        )
    )
    policy = _policy(payload)
    tools = _planned_tools(payload)

    assert any(tool in {"hybrid_search", "vector_search", "section_search", "artifact_search"} for tool in tools)
    assert "general_web" in policy.get("unavailable_required_sources", [])
    assert policy.get("must_disclose_limitations") is True
    assert "general_web" not in policy.get("allowed_source_types", [])
