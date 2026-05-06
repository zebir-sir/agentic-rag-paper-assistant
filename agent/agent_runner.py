import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from .agent_langchain import run_langchain_agent, stream_langchain_agent
from .agent_runtime import AgentDependencies

logger = logging.getLogger(__name__)


def get_agent_backend() -> str:
    value = os.getenv("AGENT_BACKEND", "langchain").strip().lower()
    if value in {"pydantic_ai", "langchain"}:
        return value
    logger.warning("Unsupported AGENT_BACKEND=%s, falling back to pydantic_ai", value)
    return "pydantic_ai"


def get_stream_backend() -> str:
    value = os.getenv("STREAM_BACKEND", "langchain").strip().lower()
    if value in {"pydantic_ai", "langchain"}:
        return value
    logger.warning("Unsupported STREAM_BACKEND=%s, falling back to pydantic_ai", value)
    return "pydantic_ai"


def _get_pydantic_rag_agent():
    from .agent import rag_agent

    return rag_agent


async def run_pydantic_agent(full_prompt: str, deps: AgentDependencies):
    rag_agent = _get_pydantic_rag_agent()
    return await rag_agent.run(full_prompt, deps=deps)


@asynccontextmanager
async def iter_pydantic_agent(full_prompt: str, deps: AgentDependencies):
    rag_agent = _get_pydantic_rag_agent()
    async with rag_agent.iter(full_prompt, deps=deps) as run:
        yield run


async def run_agent(full_prompt: str, deps: AgentDependencies):
    backend = get_agent_backend()
    if backend == "langchain":
        return await run_langchain_agent(full_prompt, deps)
    return await run_pydantic_agent(full_prompt, deps)


@asynccontextmanager
async def iter_agent(full_prompt: str, deps: AgentDependencies):
    # Streaming keeps pydantic_ai path in this phase.
    async with iter_pydantic_agent(full_prompt, deps) as run:
        yield run


def is_model_request_node(node: Any) -> bool:
    rag_agent = _get_pydantic_rag_agent()
    return rag_agent.is_model_request_node(node)
