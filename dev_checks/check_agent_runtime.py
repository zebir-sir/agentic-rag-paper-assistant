from pathlib import Path
import importlib
import sys


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))

    runtime = importlib.import_module("agent.agent_runtime")

    required_attrs = [
        "AgentDependencies",
        "_resolve_default_search_type",
        "_collect_evidence_hit",
        "_build_source_key",
    ]
    for name in required_attrs:
        assert hasattr(runtime, name), f"missing {name}"

    AgentDependencies = getattr(runtime, "AgentDependencies")
    resolve_search_type = getattr(runtime, "_resolve_default_search_type")
    collect_evidence_hit = getattr(runtime, "_collect_evidence_hit")

    deps = AgentDependencies(session_id="dev_check_session")
    assert deps.search_preferences.get("default_search_type") == "hybrid", "default search_type should be hybrid"
    assert deps.retrieved_sources == [], "retrieved_sources should be empty initially"
    assert deps.source_keys == set(), "source_keys should be empty initially"

    assert resolve_search_type(deps) == "hybrid", "default resolve should be hybrid"
    deps.search_preferences["default_search_type"] = "vector"
    assert resolve_search_type(deps) == "vector", "vector resolve failed"
    deps.search_preferences["default_search_type"] = "unknown_type"
    assert resolve_search_type(deps) == "hybrid", "unknown resolve should fallback to hybrid"

    hit = {
        "content": "This is a local evidence snippet about path planning.",
        "score": 0.88,
        "document_title": "test_paper.pdf",
        "document_source": "documents/test_paper.pdf",
        "chunk_id": "chunk-1",
        "document_id": "doc-1",
        "metadata": {},
    }
    collect_evidence_hit(deps, hit)
    assert len(deps.retrieved_sources) == 1, "expected one evidence source after first collect"
    first = deps.retrieved_sources[0]
    assert first.source_type == "local", "expected local source_type"
    assert str(first.document_title or "").strip() != "", "document_title should not be empty"
    assert first.chunk_id == "chunk-1", "chunk_id mismatch"

    collect_evidence_hit(deps, hit)
    assert len(deps.retrieved_sources) == 1, "duplicate hit should be de-duplicated"

    agent_text = Path("agent/agent.py").read_text(encoding="utf-8")
    assert "@dataclass\nclass AgentDependencies" not in agent_text, "agent.py still contains dataclass AgentDependencies"
    assert "class AgentDependencies:" not in agent_text, "agent.py still defines AgentDependencies"

    api_text = Path("agent/api.py").read_text(encoding="utf-8")
    forbidden_snippets = [
        "from .agent import rag_agent",
        "rag_agent.run",
        "rag_agent.iter",
    ]
    for forbidden in forbidden_snippets:
        assert forbidden not in api_text, (
            f"api.py still contains forbidden direct dependency: {forbidden}"
        )

    print("check_agent_runtime passed")


if __name__ == "__main__":
    main()
