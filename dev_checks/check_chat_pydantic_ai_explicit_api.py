import asyncio
import importlib
import os
import sys
from pathlib import Path


async def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))

    runner_module = importlib.import_module("agent.agent_runner")
    runtime_module = importlib.import_module("agent.agent_runtime")

    get_agent_backend = getattr(runner_module, "get_agent_backend")
    run_agent = getattr(runner_module, "run_agent")
    AgentDependencies = getattr(runtime_module, "AgentDependencies")

    original = os.environ.get("AGENT_BACKEND")
    try:
        os.environ["AGENT_BACKEND"] = "pydantic_ai"
        if get_agent_backend() != "pydantic_ai":
            raise AssertionError("get_agent_backend() should be pydantic_ai when explicitly set")

        deps = AgentDependencies(
            session_id="dev_check_explicit_pydantic",
            user_id="dev_check_user",
            use_web_search=False,
            search_preferences={"default_search_type": "hybrid", "default_limit": 3},
        )
        result = await run_agent("请用一句中文介绍你是什么系统，不要联网。", deps=deps)
        if not hasattr(result, "output"):
            raise AssertionError("pydantic_ai result should have output")
        if not str(result.output).strip():
            raise AssertionError("pydantic_ai output should be non-empty")
    finally:
        if original is None:
            os.environ.pop("AGENT_BACKEND", None)
        else:
            os.environ["AGENT_BACKEND"] = original

    print("check_chat_pydantic_ai_explicit_api passed")


if __name__ == "__main__":
    asyncio.run(main())
