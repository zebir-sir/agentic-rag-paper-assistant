import importlib
import os
import sys
from pathlib import Path


def _assert_no_top_level_imports() -> None:
    runner_text = Path("agent/agent_runner.py").read_text(encoding="utf-8")
    api_text = Path("agent/api.py").read_text(encoding="utf-8")

    runner_top_level_lines = [
        line.strip()
        for line in runner_text.splitlines()
        if line and not line.startswith((" ", "\t"))
    ]
    api_top_level_lines = [
        line.strip()
        for line in api_text.splitlines()
        if line and not line.startswith((" ", "\t"))
    ]

    assert "from .agent import rag_agent" not in runner_top_level_lines, (
        "agent_runner.py should not import rag_agent at top level"
    )
    assert (
        "from pydantic_ai.messages import PartStartEvent, PartDeltaEvent, TextPartDelta"
        not in api_top_level_lines
    ), "api.py should not import pydantic stream event types at top level"

    assert "_get_pydantic_rag_agent" in runner_text, "agent_runner should provide lazy rag_agent loader"
    assert "_get_pydantic_stream_event_types" in api_text, "api.py should provide lazy stream event type loader"


def _assert_runtime_imports() -> None:
    importlib.import_module("agent.agent_runner")
    importlib.import_module("agent.api")


def _assert_default_backends() -> None:
    runner = importlib.import_module("agent.agent_runner")
    get_agent_backend = getattr(runner, "get_agent_backend")
    get_stream_backend = getattr(runner, "get_stream_backend")

    old_agent = os.environ.get("AGENT_BACKEND")
    old_stream = os.environ.get("STREAM_BACKEND")
    try:
        os.environ.pop("AGENT_BACKEND", None)
        os.environ.pop("STREAM_BACKEND", None)
        assert get_agent_backend() == "langchain", "default AGENT_BACKEND should be langchain"
        assert get_stream_backend() == "langchain", "default STREAM_BACKEND should be langchain"
    finally:
        if old_agent is None:
            os.environ.pop("AGENT_BACKEND", None)
        else:
            os.environ["AGENT_BACKEND"] = old_agent

        if old_stream is None:
            os.environ.pop("STREAM_BACKEND", None)
        else:
            os.environ["STREAM_BACKEND"] = old_stream


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))
    os.chdir(project_root)

    _assert_no_top_level_imports()
    _assert_runtime_imports()
    _assert_default_backends()

    print("check_pydantic_import_decoupling passed")


if __name__ == "__main__":
    main()
