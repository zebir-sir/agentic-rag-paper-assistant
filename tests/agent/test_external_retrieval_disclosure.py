from agent.agent_runtime import AgentDependencies
from agent.api import _apply_external_retrieval_disclosure


def test_api_appends_model_knowledge_disclosure_for_external_failure():
    deps = AgentDependencies(
        session_id="external-fallback",
        search_preferences={
            "external_retrieval_statuses": [
                {
                    "source_type": "general_web",
                    "provider": "test-web",
                    "state": "circuit_open",
                    "retry_after_seconds": 30,
                }
            ]
        },
    )

    answer, statuses, policy, disclosure = _apply_external_retrieval_disclosure(
        "这是基于已有通用知识的说明。",
        deps,
        {},
    )

    assert statuses[0]["state"] == "circuit_open"
    assert policy["external_fallback_mode"] == "model_knowledge_with_disclosure"
    assert disclosure in answer
    assert "模型通用知识" in answer


def test_api_does_not_enable_model_fallback_for_evidence_bound_local_question():
    deps = AgentDependencies(
        session_id="local-evidence-bound",
        search_preferences={
            "external_retrieval_statuses": [
                {"source_type": "general_web", "provider": "test-web", "state": "provider_error"}
            ]
        },
    )

    _, _, policy, disclosure = _apply_external_retrieval_disclosure(
        "当前没有本地论文证据。",
        deps,
        {},
        allow_model_knowledge=False,
    )

    assert policy["external_fallback_mode"] == "disclosure_only"
    assert "无法用模型知识替代" in disclosure
