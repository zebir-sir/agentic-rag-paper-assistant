from pathlib import Path
import importlib
import sys


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))

    runner = importlib.import_module("agent.agent_runner")
    required_attrs = [
        "run_agent",
        "iter_agent",
        "run_pydantic_agent",
        "iter_pydantic_agent",
        "AgentDependencies",
    ]
    for name in required_attrs:
        assert hasattr(runner, name), f"missing {name}"

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

    print("check_agent_runner passed")


if __name__ == "__main__":
    main()
