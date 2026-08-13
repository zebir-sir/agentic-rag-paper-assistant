from agent.tool_specs import validate_tool_arguments


def test_scoped_local_tool_specs_accept_document_scope_and_embedding_language():
    valid, reason = validate_tool_arguments(
        "artifact_search",
        {
            "query": "rewiring algorithm",
            "artifact_types": ["algorithm"],
            "document_ids": ["paper-1"],
            "embedding_language": "en",
        },
    )
    assert valid is True, reason

    valid, reason = validate_tool_arguments(
        "section_search",
        {
            "query": "rewiring",
            "section_query": "method",
            "document_ids": ["paper-1"],
        },
    )
    assert valid is True, reason


def test_scoped_local_tool_specs_reject_unknown_embedding_language():
    valid, reason = validate_tool_arguments(
        "hybrid_search",
        {"query": "method", "embedding_language": "fr"},
    )
    assert valid is False
    assert reason == "invalid_enum:embedding_language"
