import asyncio
import importlib
import sys
from pathlib import Path


async def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))

    db_utils_module = importlib.import_module("agent.db_utils")
    runtime_module = importlib.import_module("agent.agent_runtime")
    langchain_tools_module = importlib.import_module("agent.langchain_tools")

    initialize_database = getattr(db_utils_module, "initialize_database")
    execute_init_sql = getattr(db_utils_module, "execute_init_sql")
    close_database = getattr(db_utils_module, "close_database")

    AgentDependencies = getattr(runtime_module, "AgentDependencies")
    build_langchain_tools = getattr(langchain_tools_module, "build_langchain_tools")

    deps = AgentDependencies(
        session_id="dev_check_langchain_tools",
        user_id="dev_check_user",
        use_web_search=False,
        search_preferences={
            "default_search_type": "hybrid",
            "default_limit": 3,
        },
    )

    tools = build_langchain_tools(deps)
    assert isinstance(tools, list), "build_langchain_tools must return a list"

    expected_names = {
        "search_knowledge_base",
        "vector_search",
        "hybrid_search",
        "get_document",
        "list_documents",
        "search_openalex_papers",
    }
    found_names = {getattr(tool, "name", "") for tool in tools}
    missing = expected_names - found_names
    assert not missing, f"missing tools: {sorted(list(missing))}"

    for tool in tools:
        assert hasattr(tool, "ainvoke"), f"tool {getattr(tool, 'name', '<unknown>')} does not support ainvoke"

    assert deps.retrieved_sources == [], "deps.retrieved_sources should be empty before tool calls"

    tool_map = {tool.name: tool for tool in tools}
    list_documents_tool = tool_map["list_documents"]
    openalex_tool = tool_map["search_openalex_papers"]

    await initialize_database()
    try:
        await execute_init_sql("sql/schema.sql")

        list_result = await list_documents_tool.ainvoke({"limit": 1, "offset": 0})
        assert isinstance(list_result, list), "list_documents result should be a list"
        assert deps.retrieved_sources == [], "list_documents should not collect evidence"

        openalex_result = await openalex_tool.ainvoke({"query": "RRT path planning", "limit": 2})
        assert openalex_result == [], "openalex tool should return [] when use_web_search=False"
        assert deps.retrieved_sources == [], "openalex(use_web_search=False) should not collect evidence"
    finally:
        await close_database()

    api_text = Path("agent/api.py").read_text(encoding="utf-8")
    assert "build_langchain_tools" not in api_text, "api.py should not use build_langchain_tools in this stage"
    assert "langchain_tools" not in api_text, "api.py should not import langchain_tools in this stage"

    agent_text = Path("agent/agent.py").read_text(encoding="utf-8")
    assert "rag_agent = Agent" in agent_text, "agent.py must still define rag_agent = Agent"

    print("check_langchain_tools passed")


if __name__ == "__main__":
    asyncio.run(main())
