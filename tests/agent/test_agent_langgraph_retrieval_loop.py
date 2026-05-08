import pytest

from agent.agent_langgraph import (
    _build_planner_capabilities,
    _build_runtime_decision_summary,
    _apply_scope_policy_to_hits,
    clean_legacy_warning_text,
    _humanize_warning,
    evidence_check_node,
    finalize_node,
    generate_analysis_node,
    initial_intent_planning_node,
    parse_answer_scope,
    route_after_initial_intent,
    resolve_answer_scope_node,
    _match_target_document,
    _prioritize_sources_by_target,
    _prioritize_target_document_hits,
    dedupe_retrieval_hits,
    grade_retrieval_node,
    grade_retrieval_quality,
    local_retrieval_node,
    parse_retrieval_evaluation,
    rewrite_query_node,
    route_after_grade,
)
from agent.agent_runtime import AgentDependencies
from agent.models import EvidenceSource


def test_grade_empty_results_insufficient():
    sufficient, reason, top_score = grade_retrieval_quality(
        results=[],
        min_results=2,
        min_top_score=0.25,
        single_hit_strong_score=0.55,
    )
    assert sufficient is False
    assert reason == "no_results"
    assert top_score is None


def test_grade_single_strong_hit_sufficient():
    sufficient, reason, top_score = grade_retrieval_quality(
        results=[{"score": 0.55, "content": "x"}],
        min_results=2,
        min_top_score=0.25,
        single_hit_strong_score=0.55,
    )
    assert sufficient is True
    assert reason == "single_strong_hit"
    assert top_score == 0.55


def test_grade_multi_hits_top_score_sufficient():
    sufficient, reason, top_score = grade_retrieval_quality(
        results=[{"score": 0.12}, {"score": 0.42}],
        min_results=2,
        min_top_score=0.25,
        single_hit_strong_score=0.55,
    )
    assert sufficient is True
    assert reason == "sufficient_scored_hits"
    assert top_score == 0.42


def test_grade_low_top_score_insufficient():
    sufficient, reason, top_score = grade_retrieval_quality(
        results=[{"score": 0.1}, {"score": 0.2}],
        min_results=2,
        min_top_score=0.25,
        single_hit_strong_score=0.55,
    )
    assert sufficient is False
    assert reason.startswith("low_top_score")
    assert top_score == 0.2


def test_dedupe_retrieval_hits():
    hits = [
        {"chunk_id": "c1", "document_id": "d1", "content": "abcdef"},
        {"chunk_id": "c1", "document_id": "d1", "content": "abcdef"},
        {"chunk_id": "c2", "document_id": "d1", "content": "abcdef"},
    ]
    deduped = dedupe_retrieval_hits(hits)
    assert len(deduped) == 2


def test_route_before_max_attempts_goes_rewrite():
    route = route_after_grade(
        {
            "retrieval_sufficient": False,
            "retrieval_attempt_count": 1,
            "max_retrieval_attempts": 2,
            "skip_rewrite": False,
        }
    )
    assert route == "rewrite_query"


def test_route_after_max_attempts_goes_generate():
    route = route_after_grade(
        {
            "retrieval_sufficient": False,
            "retrieval_attempt_count": 2,
            "max_retrieval_attempts": 2,
            "skip_rewrite": False,
        }
    )
    assert route == "generate_analysis"


def test_parse_retrieval_evaluation_normal_json():
    raw = """{"answerable": true, "confidence": 0.9, "needs_retry": false, "reason": "ok", "missing_aspects": ["x"], "suggested_query": "q", "supporting_hit_indices": [1,2]}"""
    parsed = parse_retrieval_evaluation(raw)
    assert parsed is not None
    assert parsed["answerable"] is True
    assert parsed["confidence"] == 0.9
    assert parsed["supporting_hit_indices"] == [1, 2]


def test_parse_retrieval_evaluation_defaults_when_missing():
    raw = """{"answerable": false}"""
    parsed = parse_retrieval_evaluation(raw)
    assert parsed is not None
    assert parsed["answerable"] is False
    assert parsed["confidence"] == 0.0
    assert parsed["needs_retry"] is False
    assert parsed["missing_aspects"] == []
    assert parsed["supporting_hit_indices"] == []


def test_parse_retrieval_evaluation_confidence_clamp():
    p1 = parse_retrieval_evaluation("""{"answerable": true, "confidence": 9}""")
    p2 = parse_retrieval_evaluation("""{"answerable": true, "confidence": -3}""")
    assert p1 is not None and p1["confidence"] == 1.0
    assert p2 is not None and p2["confidence"] == 0.0


@pytest.mark.asyncio
async def test_rewrite_uses_suggested_query_first():
    state = {
        "question": "what is method",
        "current_query": "what is method",
        "retrieval_insufficient_reason": "insufficient",
        "documents": [],
        "retrieval_results": [],
        "rewritten_queries": [],
        "suggested_rewrite_query": "paper method ablation",
        "tools_used": [],
    }
    out = await rewrite_query_node(state)
    assert out["current_query"] == "paper method ablation"
    assert out["rewritten_queries"] == ["paper method ablation"]


@pytest.mark.asyncio
async def test_grade_fallbacks_to_rule_when_evaluator_fails(monkeypatch):
    async def fake_eval(_state):
        return None

    monkeypatch.setattr("agent.agent_langgraph.evaluate_retrieval_with_llm", fake_eval)
    state = {
        "retrieval_results": [{"score": 0.8, "content": "x"}],
        "retrieval_attempt_count": 1,
        "retrieval_attempts": [{"attempt": 1, "query": "q", "result_count": 1, "top_score": 0.8}],
        "metadata": {},
    }
    out = await grade_retrieval_node(state)
    assert out["retrieval_sufficient"] is True
    assert out["metadata"]["retrieval_evaluator_used"] is False


def test_match_target_document_by_filename():
    question = "请分析 Hybrid-RRT.pdf 在火星场景下的表现"
    docs = [
        {"id": "d1", "title": "Random Paper", "source": "random.pdf"},
        {"id": "d2", "title": "Hybrid-RRT", "source": "/kb/Hybrid-RRT.pdf"},
    ]
    matched = _match_target_document(question, docs)
    assert matched is not None
    assert matched["id"] == "d2"


def test_prioritize_target_hits_without_supplemental_when_enough():
    hits = [
        {"document_id": "d2", "score": 0.8, "content": "a", "metadata": {}},
        {"document_id": "d2", "score": 0.7, "content": "b", "metadata": {}},
        {"document_id": "d3", "score": 0.95, "content": "c", "metadata": {}},
    ]
    out, used_supp = _prioritize_target_document_hits(
        hits, target_document_id="d2", min_results=2, strong_single_score=0.55
    )
    assert used_supp is False
    assert all(h["document_id"] == "d2" for h in out)


def test_prioritize_sources_target_first():
    s1 = EvidenceSource(
        source_type="local",
        document_id="d3",
        document_title="Other",
        document_source="x",
        snippet="x",
    )
    s2 = EvidenceSource(
        source_type="local",
        document_id="d2",
        document_title="Hybrid-RRT",
        document_source="y",
        snippet="y",
    )
    ordered = _prioritize_sources_by_target([s1, s2], "d2")
    assert ordered[0].document_id == "d2"
    assert ordered[0].metadata.get("reference_role") == "primary_target"


def test_parse_answer_scope_normal_json():
    docs = [{"id": "d1", "title": "A"}, {"id": "d2", "title": "B"}]
    raw = """{"scope_policy":"strict_target","target_documents":[{"document_id":"d2","title":"B","confidence":0.9,"match_reason":"x"}],"allow_supplemental":false,"scope_reason":"r","answer_instruction":"i"}"""
    parsed = parse_answer_scope(raw, docs)
    assert parsed["scope_policy"] == "strict_target"
    assert parsed["target_documents"][0]["document_id"] == "d2"
    assert parsed["allow_supplemental"] is False


def test_parse_answer_scope_filters_invalid_doc_id():
    docs = [{"id": "d1", "title": "A"}]
    raw = """{"scope_policy":"prefer_target","target_documents":[{"document_id":"d1","confidence":0.8},{"document_id":"dx","confidence":1.0}]}"""
    parsed = parse_answer_scope(raw, docs)
    assert len(parsed["target_documents"]) == 1
    assert parsed["target_documents"][0]["document_id"] == "d1"


def test_scope_strict_target_filters_only_target():
    hits = [
        {"document_id": "d1", "content": "a", "metadata": {}},
        {"document_id": "d2", "content": "b", "metadata": {}},
    ]
    out, supp = _apply_scope_policy_to_hits(hits, "strict_target", ["d1"], allow_supplemental=False)
    assert supp is False
    assert len(out) == 1 and out[0]["document_id"] == "d1"


def test_scope_prefer_target_allows_supplemental_when_insufficient():
    hits = [
        {"document_id": "d1", "score": 0.2, "content": "a", "metadata": {}},
        {"document_id": "d2", "score": 0.9, "content": "b", "metadata": {}},
    ]
    out, supp = _apply_scope_policy_to_hits(hits, "prefer_target", ["d1"], allow_supplemental=True)
    assert supp is True
    assert len(out) == 2


def test_scope_broad_kb_keeps_all():
    hits = [
        {"document_id": "d1", "content": "a", "metadata": {}},
        {"document_id": "d2", "content": "b", "metadata": {}},
    ]
    out, supp = _apply_scope_policy_to_hits(hits, "broad_kb", ["d1"], allow_supplemental=True)
    assert supp is False
    assert len(out) == 2


@pytest.mark.asyncio
async def test_resolver_failure_fallback_broad(monkeypatch):
    class BadModel:
        async def ainvoke(self, _messages):
            raise RuntimeError("boom")

    monkeypatch.setattr("agent.agent_langgraph.get_langchain_chat_model", lambda: BadModel())
    state = {"question": "q", "documents": [{"id": "d1", "title": "A"}], "metadata": {}, "warnings": []}
    out = await resolve_answer_scope_node(state)
    assert out["scope_policy"] == "broad_kb"
    assert out["allow_supplemental"] is True


@pytest.mark.asyncio
async def test_local_retrieval_uses_planner_steps(monkeypatch):
    class FakeTool:
        def __init__(self, name):
            self.name = name

        async def ainvoke(self, _args):
            return [{"chunk_id": "c1", "document_id": "d1", "content": "x", "score": 0.7, "metadata": {}, "document_title": "Doc", "document_source": "s"}]

    async def fake_plan_user_intent_debug(**_kwargs):
        return {
            "normalized_plan": {
                "intent": "local_paper_qa",
                "needs_retrieval": True,
                "retrieval_steps": [{"tool": "hybrid_search", "query": "q", "limit": 5}],
                "max_tools": 1,
                "allow_external_sources": False,
                "direct_answer_allowed": False,
                "rewrite_allowed": True,
                "evidence_policy": "answer_with_available_evidence_and_state_uncertainty",
                "reason": "",
                "warnings": [],
            },
            "fallback_used": False,
            "fallback_reason": "",
            "fallback_decision": "",
            "raw_model_content_preview": "",
        }

    async def fake_execute_intent_plan_steps(**_kwargs):
        return {
            "results": [{"chunk_id": "c1", "document_id": "d1", "content": "x", "score": 0.7, "metadata": {}, "document_title": "Doc", "document_source": "s"}],
            "tools_executed": [{"tool": "hybrid_search", "args": {"query": "q", "limit": 5}}],
            "planned_steps": [{"tool": "hybrid_search", "query": "q", "limit": 5}],
            "warnings": [],
        }

    monkeypatch.setattr("agent.agent_langgraph.plan_user_intent_debug", fake_plan_user_intent_debug)
    monkeypatch.setattr("agent.agent_langgraph.execute_intent_plan_steps", fake_execute_intent_plan_steps)
    monkeypatch.setattr("agent.agent_langgraph.get_langchain_chat_model", lambda: object())
    monkeypatch.setattr("agent.agent_langgraph.build_langchain_tools", lambda _deps: [FakeTool("hybrid_search"), FakeTool("search_knowledge_base")])
    monkeypatch.setattr("agent.agent_langgraph._is_openalex_enabled", lambda: False)
    monkeypatch.setattr("agent.agent_langgraph.is_general_web_search_enabled", lambda: False)

    state = {
        "deps": AgentDependencies(session_id="s1"),
        "question": "compare methods",
        "context_prompt": "",
        "retrieval_attempt_count": 0,
        "retrieval_results": [],
        "retrieval_attempts": [],
        "metadata": {},
        "target_documents": [],
        "scope_policy": "broad_kb",
        "allow_supplemental": True,
        "tools_used": [],
    }
    out = await local_retrieval_node(state)
    assert out["metadata"]["intent_planner_used"] is True
    assert "hybrid_search" in out["metadata"]["tools_executed"]
    assert len(out["retrieval_results"]) == 1
    assert "planner_capabilities" in out["metadata"]
    assert "available_tools" in out["metadata"]


@pytest.mark.asyncio
async def test_local_retrieval_planner_fallback_to_direct_answer_without_search(monkeypatch):
    class FakeSearchTool:
        name = "search_knowledge_base"

        async def ainvoke(self, _args):
            raise AssertionError("search_knowledge_base should not run for direct-answer fallback")

    async def bad_plan_debug(**_kwargs):
        raise RuntimeError("planner failed")

    monkeypatch.setattr("agent.agent_langgraph.plan_user_intent_debug", bad_plan_debug)
    monkeypatch.setattr("agent.agent_langgraph.get_langchain_chat_model", lambda: object())
    monkeypatch.setattr("agent.agent_langgraph.build_langchain_tools", lambda _deps: [FakeSearchTool()])
    monkeypatch.setattr("agent.agent_langgraph._is_openalex_enabled", lambda: False)
    monkeypatch.setattr("agent.agent_langgraph.is_general_web_search_enabled", lambda: False)

    state = {
        "deps": AgentDependencies(session_id="s2"),
        "question": "你觉得我今晚吃什么比较好",
        "context_prompt": "",
        "retrieval_attempt_count": 0,
        "retrieval_results": [],
        "retrieval_attempts": [],
        "metadata": {},
        "target_documents": [],
        "scope_policy": "broad_kb",
        "allow_supplemental": True,
        "tools_used": [],
    }
    out = await local_retrieval_node(state)
    assert out["metadata"]["planner_fallback_used"] is True
    assert out["metadata"]["intent"] == "direct_answer"
    assert out["metadata"]["retrieval_skipped_by_planner"] is True
    assert out["metadata"]["tools_executed"] == []
    assert out["metadata"]["sources_count"] == 0
    assert out["retrieval_results"] == []


@pytest.mark.asyncio
async def test_local_retrieval_skips_when_planner_direct_answer(monkeypatch):
    async def fake_plan_user_intent_debug(**_kwargs):
        return {
            "normalized_plan": {
                "intent": "direct_answer",
                "needs_retrieval": False,
                "retrieval_steps": [],
                "max_tools": 0,
                "allow_external_sources": False,
                "direct_answer_allowed": True,
                "rewrite_allowed": True,
                "evidence_policy": "answer_with_available_evidence_and_state_uncertainty",
                "reason": "",
                "warnings": [],
            },
            "fallback_used": False,
            "fallback_reason": "",
            "fallback_decision": "",
            "raw_model_content_preview": "",
        }

    called = {"execute": 0}

    async def fake_execute_intent_plan_steps(**_kwargs):
        called["execute"] += 1
        return {"results": [], "tools_executed": [], "planned_steps": [], "warnings": []}

    monkeypatch.setattr("agent.agent_langgraph.plan_user_intent_debug", fake_plan_user_intent_debug)
    monkeypatch.setattr("agent.agent_langgraph.execute_intent_plan_steps", fake_execute_intent_plan_steps)
    monkeypatch.setattr("agent.agent_langgraph.get_langchain_chat_model", lambda: object())
    monkeypatch.setattr("agent.agent_langgraph.build_langchain_tools", lambda _deps: [])

    state = {
        "deps": AgentDependencies(session_id="s3"),
        "question": "what is rrt",
        "context_prompt": "",
        "retrieval_attempt_count": 0,
        "retrieval_results": [],
        "retrieval_attempts": [],
        "metadata": {},
        "target_documents": [],
        "scope_policy": "broad_kb",
        "allow_supplemental": True,
        "tools_used": [],
    }
    out = await local_retrieval_node(state)
    assert out["metadata"]["retrieval_skipped_by_planner"] is True
    assert out["metadata"]["direct_answer_allowed"] is True
    assert out["metadata"]["tools_executed"] == []
    assert out["metadata"]["sources_count"] == 0
    assert out["metadata"]["planner_decision"]["needs_retrieval"] is False
    assert called["execute"] == 0


@pytest.mark.asyncio
async def test_initial_intent_planning_direct_answer_sets_skip_metadata(monkeypatch):
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
                "reason": "no extra evidence needed",
                "warnings": [],
            },
            "fallback_used": False,
        }

    monkeypatch.setattr("agent.agent_langgraph.plan_user_intent_debug", fake_plan_user_intent_debug)
    monkeypatch.setattr("agent.agent_langgraph.get_langchain_chat_model", lambda: object())
    out = await initial_intent_planning_node(
        {"deps": AgentDependencies(session_id="s"), "question": "hello", "context_prompt": "", "metadata": {}}
    )
    assert out["metadata"]["retrieval_skipped_by_planner"] is True
    assert out["metadata"]["direct_answer_allowed"] is True
    assert route_after_initial_intent(out) == "generate_analysis"


def test_route_after_initial_intent_local_retrieval_path():
    state = {
        "intent_plan": {
            "intent": "local_paper_qa",
            "needs_retrieval": True,
            "retrieval_steps": [{"tool": "hybrid_search", "query": "q", "limit": 5}],
            "max_tools": 1,
            "allow_external_sources": False,
            "evidence_policy": "answer_with_available_evidence_and_state_uncertainty",
            "direct_answer_allowed": False,
            "rewrite_allowed": True,
            "reason": "",
            "warnings": [],
        }
    }
    assert route_after_initial_intent(state) == "inspect_documents"


def test_humanize_warning_has_no_legacy_position_text():
    text = _humanize_warning("No retrieval evidence found; answer should be treated as general guidance.")
    assert text is not None
    assert "以下内容" not in text


def test_clean_legacy_warning_text_rewrites_legacy_sentence():
    legacy = "当前没有检索到直接相关片段，以下内容更适合作为一般性分析参考。"
    cleaned = clean_legacy_warning_text(legacy)
    assert "以下内容" not in cleaned
    assert "本轮没有可核对的检索片段" in cleaned


@pytest.mark.asyncio
async def test_local_retrieval_skips_when_planner_raw_plan_has_no_intent_but_no_retrieval(monkeypatch):
    async def fake_plan_user_intent_debug(**_kwargs):
        from agent.intent_planner import normalize_intent_plan

        raw_plan = {
            "needs_retrieval": False,
            "retrieval_steps": [],
            "direct_answer": "some answer text",
        }
        normalized = normalize_intent_plan(raw_plan)
        return {
            "raw_plan": raw_plan,
            "normalized_plan": normalized.model_dump(),
            "fallback_used": False,
            "fallback_reason": "",
            "fallback_decision": "",
            "raw_model_content_preview": "",
        }

    monkeypatch.setattr("agent.agent_langgraph.plan_user_intent_debug", fake_plan_user_intent_debug)
    monkeypatch.setattr("agent.agent_langgraph.get_langchain_chat_model", lambda: object())
    monkeypatch.setattr("agent.agent_langgraph.build_langchain_tools", lambda _deps: [])

    state = {
        "deps": AgentDependencies(session_id="s3b"),
        "question": "你觉得我今晚吃什么比较好",
        "context_prompt": "",
        "retrieval_attempt_count": 0,
        "retrieval_results": [],
        "retrieval_attempts": [],
        "metadata": {},
        "target_documents": [],
        "scope_policy": "broad_kb",
        "allow_supplemental": True,
        "tools_used": [],
    }
    out = await local_retrieval_node(state)
    assert out["metadata"]["intent"] == "direct_answer"
    assert out["metadata"]["retrieval_skipped_by_planner"] is True
    assert out["metadata"]["direct_answer_allowed"] is True
    assert out["metadata"]["tools_executed"] == []
    assert out["metadata"]["sources_count"] == 0
    assert out["retrieval_results"] == []


@pytest.mark.asyncio
async def test_local_retrieval_rebuilds_missing_steps_for_evidence_question(monkeypatch):
    class FakeTool:
        def __init__(self, name):
            self.name = name

        async def ainvoke(self, _args):
            return [{"chunk_id": "c1", "document_id": "d1", "content": "x", "score": 0.7, "metadata": {}, "document_title": "Doc", "document_source": "s"}]

    async def fake_plan_user_intent_debug(**_kwargs):
        return {
            "normalized_plan": {
                "intent": "local_paper_qa",
                "needs_retrieval": True,
                "retrieval_steps": [],
                "max_tools": 1,
                "allow_external_sources": False,
                "direct_answer_allowed": False,
                "rewrite_allowed": True,
                "evidence_policy": "answer_with_available_evidence_and_state_uncertainty",
                "reason": "",
                "warnings": [],
            },
            "fallback_used": False,
            "fallback_reason": "",
            "fallback_decision": "",
            "raw_model_content_preview": "",
        }

    async def fake_execute_intent_plan_steps(**_kwargs):
        return {
            "results": [{"chunk_id": "c1", "document_id": "d1", "content": "x", "score": 0.7, "metadata": {}, "document_title": "Doc", "document_source": "s"}],
            "tools_executed": [{"tool": "hybrid_search", "args": {"query": "根据上传论文总结方法贡献", "limit": 10}}],
            "planned_steps": [{"tool": "hybrid_search", "query": "根据上传论文总结方法贡献", "limit": 10}],
            "warnings": [],
        }

    monkeypatch.setattr("agent.agent_langgraph.plan_user_intent_debug", fake_plan_user_intent_debug)
    monkeypatch.setattr("agent.agent_langgraph.execute_intent_plan_steps", fake_execute_intent_plan_steps)
    monkeypatch.setattr("agent.agent_langgraph.get_langchain_chat_model", lambda: object())
    monkeypatch.setattr("agent.agent_langgraph.build_langchain_tools", lambda _deps: [FakeTool("hybrid_search")])
    monkeypatch.setattr("agent.agent_langgraph._is_openalex_enabled", lambda: False)
    monkeypatch.setattr("agent.agent_langgraph.is_general_web_search_enabled", lambda: False)

    state = {
        "deps": AgentDependencies(session_id="s4"),
        "question": "根据上传论文总结方法贡献",
        "context_prompt": "",
        "retrieval_attempt_count": 0,
        "retrieval_results": [],
        "retrieval_attempts": [],
        "metadata": {},
        "target_documents": [],
        "scope_policy": "broad_kb",
        "allow_supplemental": True,
        "tools_used": [],
    }
    out = await local_retrieval_node(state)
    assert out["metadata"]["planner_fallback_used"] is True
    assert out["metadata"]["fallback_decision"] == "local_retrieval_fallback"
    assert out["metadata"]["intent"] == "local_paper_qa"
    assert out["metadata"]["tools_executed"] == ["hybrid_search"]
    assert len(out["retrieval_results"]) == 1


@pytest.mark.asyncio
async def test_direct_answer_progress_does_not_emit_retrieval_round(monkeypatch):
    progress_messages = []

    async def fake_plan_user_intent_debug(**_kwargs):
        return {
            "normalized_plan": {
                "intent": "direct_answer",
                "needs_retrieval": False,
                "retrieval_steps": [],
                "max_tools": 0,
                "allow_external_sources": False,
                "direct_answer_allowed": True,
                "rewrite_allowed": True,
                "evidence_policy": "answer_with_available_evidence_and_state_uncertainty",
                "reason": "No evidence required.",
                "warnings": [],
            },
            "fallback_used": False,
            "fallback_reason": "",
            "fallback_decision": "",
            "raw_model_content_preview": "",
        }

    monkeypatch.setattr("agent.agent_langgraph.plan_user_intent_debug", fake_plan_user_intent_debug)
    monkeypatch.setattr("agent.agent_langgraph.get_langchain_chat_model", lambda: object())
    monkeypatch.setattr("agent.agent_langgraph.build_langchain_tools", lambda _deps: [])

    async def progress_callback(message: str) -> None:
        progress_messages.append(message)

    state = {
        "deps": AgentDependencies(session_id="s-progress-direct"),
        "question": "请简单介绍一下你的能力",
        "context_prompt": "",
        "retrieval_attempt_count": 0,
        "retrieval_results": [],
        "retrieval_attempts": [],
        "metadata": {},
        "target_documents": [],
        "scope_policy": "broad_kb",
        "allow_supplemental": True,
        "tools_used": [],
        "progress_callback": progress_callback,
    }
    await local_retrieval_node(state)
    joined = "\n".join(progress_messages)
    assert "正在规划回答..." in joined
    assert "已判断本轮可直接回答，跳过本地检索。" in joined
    assert "Local retrieval round" not in joined
    assert "Retrieval round" not in joined
    assert "正在执行第 1 轮检索" not in joined


@pytest.mark.asyncio
async def test_retrieval_progress_uses_chinese_status(monkeypatch):
    progress_messages = []

    class FakeTool:
        def __init__(self, name):
            self.name = name

        async def ainvoke(self, _args):
            return [{"chunk_id": "c1", "document_id": "d1", "content": "x", "score": 0.7, "metadata": {}, "document_title": "Doc", "document_source": "s"}]

    async def fake_plan_user_intent_debug(**_kwargs):
        return {
            "normalized_plan": {
                "intent": "local_paper_qa",
                "needs_retrieval": True,
                "retrieval_steps": [{"tool": "hybrid_search", "query": "q", "limit": 5}],
                "max_tools": 1,
                "allow_external_sources": False,
                "direct_answer_allowed": False,
                "rewrite_allowed": True,
                "evidence_policy": "answer_with_available_evidence_and_state_uncertainty",
                "reason": "Need evidence.",
                "warnings": [],
            },
            "fallback_used": False,
            "fallback_reason": "",
            "fallback_decision": "",
            "raw_model_content_preview": "",
        }

    async def fake_execute_intent_plan_steps(**_kwargs):
        return {
            "results": [{"chunk_id": "c1", "document_id": "d1", "content": "x", "score": 0.7, "metadata": {}, "document_title": "Doc", "document_source": "s"}],
            "tools_executed": [{"tool": "hybrid_search", "args": {"query": "q", "limit": 5}}],
            "planned_steps": [{"tool": "hybrid_search", "query": "q", "limit": 5}],
            "warnings": [],
        }

    monkeypatch.setattr("agent.agent_langgraph.plan_user_intent_debug", fake_plan_user_intent_debug)
    monkeypatch.setattr("agent.agent_langgraph.execute_intent_plan_steps", fake_execute_intent_plan_steps)
    monkeypatch.setattr("agent.agent_langgraph.get_langchain_chat_model", lambda: object())
    monkeypatch.setattr("agent.agent_langgraph.build_langchain_tools", lambda _deps: [FakeTool("hybrid_search")])

    async def progress_callback(message: str) -> None:
        progress_messages.append(message)

    state = {
        "deps": AgentDependencies(session_id="s-progress-retrieval"),
        "question": "请根据文档总结方法",
        "context_prompt": "",
        "retrieval_attempt_count": 0,
        "retrieval_results": [],
        "retrieval_attempts": [],
        "metadata": {},
        "target_documents": [],
        "scope_policy": "broad_kb",
        "allow_supplemental": True,
        "tools_used": [],
        "progress_callback": progress_callback,
    }
    await local_retrieval_node(state)
    joined = "\n".join(progress_messages)
    assert "正在规划回答..." in joined
    assert "正在执行第 1 轮检索..." in joined
    assert "第 1 轮检索完成" in joined
    assert "Local retrieval round" not in joined
    assert "Retrieval round" not in joined


@pytest.mark.asyncio
async def test_evidence_check_skips_no_evidence_warning_for_direct_answer():
    state = {
        "sources": [],
        "retrieval_results": [],
        "draft_answer": "这里是一个直接回答。",
        "warnings": [],
        "metadata": {
            "retrieval_skipped_by_planner": True,
            "direct_answer_allowed": True,
            "planner_decision": {
                "direct_answer_allowed": True,
            },
        },
    }
    out = await evidence_check_node(state)
    assert not any("No retrieval evidence found" in warning for warning in out.get("warnings", []))


@pytest.mark.asyncio
async def test_evidence_warning_and_finalize_note_for_empty_retrieval():
    checked = await evidence_check_node(
        {
            "sources": [],
            "retrieval_results": [],
            "draft_answer": "这是基于一般信息的回答。",
            "warnings": [],
            "metadata": {
                "retrieval_skipped_by_planner": False,
                "direct_answer_allowed": False,
                "planner_decision": {
                    "direct_answer_allowed": False,
                },
            },
        }
    )
    assert any("No retrieval evidence found" in warning for warning in checked.get("warnings", []))

    finalized = await finalize_node(
        {
            "retrieval_attempts": [],
            "retrieval_results": [],
            "rewritten_queries": [],
            "target_documents": [],
            "scope_policy": "broad_kb",
            "allow_supplemental": True,
            "deps": AgentDependencies(session_id="s-finalize-note"),
            "warnings": checked.get("warnings", []),
            "draft_answer": "这是基于一般信息的回答。",
            "metadata": {},
            "sources": [],
        }
    )
    assert "本轮没有可核对的检索片段" in finalized["final_answer"]
    assert "以下内容" not in finalized["final_answer"]


def test_humanize_warning_uses_position_independent_wording():
    text = _humanize_warning("No retrieval evidence found; answer should be treated as general guidance.")
    assert text is not None
    assert "以下内容" not in text
    assert "上述内容" not in text


def test_runtime_decision_summary_contains_planner_and_runtime_state():
    summary = _build_runtime_decision_summary(
        {
            "planner_decision": {
                "intent": "direct_answer",
                "needs_retrieval": False,
                "direct_answer_allowed": True,
                "planned_tools": [],
                "reason": "No evidence required.",
                "evidence_policy": "answer_with_available_evidence_and_state_uncertainty",
            },
            "retrieval_skipped_by_planner": True,
            "retrieval_sufficient": True,
            "retrieval_insufficient_reason": None,
            "tools_executed": [],
        }
    )
    assert "intent: direct_answer" in summary
    assert "needs_retrieval: False" in summary
    assert "retrieval_skipped_by_planner: True" in summary


@pytest.mark.asyncio
async def test_generate_analysis_prompt_includes_runtime_decision_summary(monkeypatch):
    captured = {"prompt": ""}

    class DummyModel:
        async def ainvoke(self, messages):
            captured["prompt"] = messages[1]["content"]
            return type("Resp", (), {"content": "ok"})()

    monkeypatch.setattr("agent.agent_langgraph.get_langchain_chat_model", lambda: DummyModel())
    state = {
        "question": "请简要介绍系统能力",
        "context_prompt": "",
        "documents": [],
        "retrieval_results": [],
        "warnings": [],
        "retrieval_attempts": [],
        "rewritten_queries": [],
        "retrieval_evaluation": {},
        "answer_scope": {},
        "scope_policy": "broad_kb",
        "metadata": {
            "planner_decision": {
                "intent": "direct_answer",
                "needs_retrieval": False,
                "direct_answer_allowed": True,
                "planned_tools": [],
                "reason": "No evidence required.",
                "evidence_policy": "answer_with_available_evidence_and_state_uncertainty",
            },
            "retrieval_skipped_by_planner": True,
            "direct_answer_allowed": True,
            "retrieval_sufficient": True,
            "retrieval_insufficient_reason": None,
            "tools_executed": [],
        },
    }
    out = await generate_analysis_node(state)
    assert out["draft_answer"] == "ok"
    assert "运行决策摘要" in captured["prompt"]
    assert "planner 主动跳过检索并允许 direct answer" in captured["prompt"]


@pytest.mark.asyncio
async def test_grade_retrieval_short_circuits_after_planner_direct_answer():
    state = {
        "retrieval_results": [],
        "retrieval_attempt_count": 1,
        "max_retrieval_attempts": 2,
        "scope_policy": "broad_kb",
        "target_documents": [],
        "metadata": {
            "retrieval_skipped_by_planner": True,
            "direct_answer_allowed": True,
        },
    }
    out = await grade_retrieval_node(state)
    assert out["retrieval_sufficient"] is True
    assert out["retrieval_insufficient_reason"] is None
    assert out["metadata"]["retrieval_retry_trigger"] == "planner_direct_answer"


def test_build_planner_capabilities_from_provider_status(monkeypatch):
    monkeypatch.setattr("agent.agent_langgraph._is_openalex_enabled", lambda: True)
    monkeypatch.setattr("agent.agent_langgraph.is_general_web_search_enabled", lambda: True)
    deps = AgentDependencies(
        session_id="s-caps",
        use_web_search=True,
        search_preferences={"allow_web_search": True, "allow_openalex_search": True},
    )
    caps = _build_planner_capabilities(deps)
    assert caps.openalex_search_enabled is True
    assert caps.web_search_enabled is True


def test_build_planner_capabilities_respects_user_toggle(monkeypatch):
    monkeypatch.setattr("agent.agent_langgraph._is_openalex_enabled", lambda: True)
    monkeypatch.setattr("agent.agent_langgraph.is_general_web_search_enabled", lambda: True)
    deps = AgentDependencies(
        session_id="s-caps-off",
        use_web_search=False,
        search_preferences={"allow_web_search": False, "allow_openalex_search": False},
    )
    caps = _build_planner_capabilities(deps)
    assert caps.web_search_enabled is False
    assert caps.openalex_search_enabled is False
