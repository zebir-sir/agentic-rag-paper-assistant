from agent.retrieval_harness_runtime import build_retrieval_harness_trace_payload


def test_trace_payload_exposes_only_compact_retrieval_harness_metadata():
    payload = build_retrieval_harness_trace_payload(
        {
            "retrieval_contract": {
                "scope_policy": "strict_target",
                "required_source_types": ["local_kb"],
                "allowed_source_types": ["local_kb"],
                "max_tool_calls_per_round": 1,
            },
            "retrieval_contract_evaluation": {
                "required_sources_satisfied": True,
                "evidence_source_types": ["local_kb"],
            },
            "retrieval_execution_records": [
                {"tool": "hybrid_search", "args": {"query": "must not leak"}},
            ],
            "retrieval_attempt_count": 1,
        }
    )

    assert payload["available"] is True
    assert payload["tools_executed"] == ["hybrid_search"]
    assert payload["required_sources_satisfied"] is True
    assert "query" not in str(payload)


def test_trace_payload_is_absent_without_harness_metadata():
    assert build_retrieval_harness_trace_payload({}) == {"available": False}
