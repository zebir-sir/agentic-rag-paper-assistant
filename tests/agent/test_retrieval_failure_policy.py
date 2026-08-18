from agent.retrieval_failure_policy import (
    apply_external_retrieval_failure_policy,
    build_external_retrieval_disclosure,
)


def test_external_failure_allows_disclosed_model_knowledge_fallback():
    policy = apply_external_retrieval_failure_policy(
        {"allowed_source_types": ["general_web"], "blocked_source_types": []},
        [{"source_type": "general_web", "provider": "web", "state": "provider_error", "retry_count": 1}],
    )

    disclosure = build_external_retrieval_disclosure(policy["external_retrieval_failures"], policy)

    assert policy["mode"] == "answer_with_disclosure"
    assert policy["external_fallback_mode"] == "model_knowledge_with_disclosure"
    assert "model_knowledge" in policy["allowed_source_types"]
    assert "模型通用知识" in disclosure
    assert "自行辨别并核验" in disclosure


def test_external_failure_does_not_override_blocked_model_knowledge():
    policy = apply_external_retrieval_failure_policy(
        {"blocked_source_types": ["model_knowledge"]},
        [{"source_type": "external_academic", "provider": "openalex", "state": "circuit_open"}],
    )

    disclosure = build_external_retrieval_disclosure(policy["external_retrieval_failures"], policy)

    assert policy["external_fallback_mode"] == "disclosure_only"
    assert "模型通用知识" not in policy.get("allowed_source_types", [])
    assert "无法用模型知识替代" in disclosure
