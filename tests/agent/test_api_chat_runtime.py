import pytest
import langchain.agents as langchain_agents
from unittest.mock import AsyncMock, patch

if not hasattr(langchain_agents, "create_agent"):
    langchain_agents.create_agent = lambda *args, **kwargs: None

from agent.api import openalex_status, prepare_chat_runtime
from agent.models import ChatRequest


@pytest.mark.asyncio
async def test_openalex_status_reports_capability_state():
    with patch("agent.api._is_openalex_enabled", return_value=True):
        assert await openalex_status() == {"enabled": True}


@pytest.mark.asyncio
async def test_prepare_chat_runtime_includes_carryover_metadata_and_prompt_block():
    request = ChatRequest(
        message="这个方法和 RRT* 的差别呢？",
        session_id="session-1",
    )

    with patch("agent.api._prepare_agent_prompt", new_callable=AsyncMock) as mock_prepare:
        mock_prepare.return_value = {
            "full_prompt": "Previous conversation:\n...\n\nCurrent question: 这个方法和 RRT* 的差别呢？",
            "compression_used": False,
            "memory_updated": False,
            "memory_state": None,
            "history_messages": [
                {"role": "user", "content": "请总结 AIT* 这篇论文的方法核心。"},
                {"role": "assistant", "content": "AIT* 主要把启发式搜索和增量采样结合起来。"},
            ],
        }

        runtime = await prepare_chat_runtime(request)

    assert runtime.workflow_metadata["dialog_act"] == "follow_up_reference"
    assert runtime.workflow_metadata["history_resolution_used"] is True
    assert runtime.workflow_metadata["resolved_query"].endswith("当前追问：这个方法和 RRT* 的差别呢？")
    assert runtime.workflow_metadata["simple_chat_candidate"] is False
    assert "[Conversation carry-over]" in runtime.full_prompt
    assert "resolved_query_for_retrieval" in runtime.full_prompt
    assert "[Conversation carry-over]" in runtime.langgraph_context_prompt
