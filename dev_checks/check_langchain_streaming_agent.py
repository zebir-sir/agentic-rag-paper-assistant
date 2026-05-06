import asyncio
import importlib
import sys
from pathlib import Path

DIRTY_MARKERS = [
    "NoneNone",
    "AIMessage(",
    "HumanMessage(",
    "ToolMessage(",
    "{'messages':",
    "{'model':",
]


async def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))

    db_utils_module = importlib.import_module("agent.db_utils")
    runtime_module = importlib.import_module("agent.agent_runtime")
    langchain_module = importlib.import_module("agent.agent_langchain")

    initialize_database = getattr(db_utils_module, "initialize_database")
    execute_init_sql = getattr(db_utils_module, "execute_init_sql")
    close_database = getattr(db_utils_module, "close_database")

    AgentDependencies = getattr(runtime_module, "AgentDependencies")
    stream_langchain_agent = getattr(langchain_module, "stream_langchain_agent")

    await initialize_database()
    try:
        await execute_init_sql("sql/schema.sql")

        deps = AgentDependencies(
            session_id="dev_check_langchain_streaming_agent",
            user_id="dev_check_user",
            use_web_search=False,
            search_preferences={
                "default_search_type": "hybrid",
                "default_limit": 3,
            },
        )

        result = await stream_langchain_agent(
            full_prompt="请用中文简单介绍你作为科研论文阅读助手能做什么。",
            deps=deps,
        )

        if not isinstance(result.message, str) or not result.message.strip():
            raise AssertionError("result.message should be non-empty string")
        if not isinstance(result.chunks, list):
            raise AssertionError("result.chunks should be list")
        for chunk in result.chunks:
            if not isinstance(chunk, str):
                raise AssertionError("every chunk should be string")
        if not "".join(result.chunks).strip() and not result.message.strip():
            raise AssertionError("chunks and message are both empty")
        joined = "".join(result.chunks)
        for marker in DIRTY_MARKERS:
            if marker in joined:
                raise AssertionError(f"chunks contain dirty marker: {marker}")
            if marker in result.message:
                raise AssertionError(f"message contains dirty marker: {marker}")
        if not isinstance(result.tools_used, list):
            raise AssertionError("result.tools_used should be list")
        if not isinstance(result.sources, list):
            raise AssertionError("result.sources should be list")
    finally:
        await close_database()

    print("check_langchain_streaming_agent passed")


if __name__ == "__main__":
    asyncio.run(main())
