import pytest

from agent.agent_langgraph import (
    _apply_scope_policy_to_hits,
    parse_answer_scope,
    resolve_answer_scope_node,
    _match_target_document,
    _prioritize_sources_by_target,
    _prioritize_target_document_hits,
    dedupe_retrieval_hits,
    grade_retrieval_node,
    grade_retrieval_quality,
    parse_retrieval_evaluation,
    rewrite_query_node,
    route_after_grade,
)
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
