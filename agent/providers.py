import os
import asyncio
from typing import Any, Dict, Optional
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.openai import OpenAIChatModel
import openai
from dotenv import load_dotenv

load_dotenv()


def get_llm_model(model_choice: Optional[str] = None) -> OpenAIChatModel:
    llm_choice = model_choice or os.getenv("LLM_CHOICE", "gpt-4o-mini")
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")

    provider = OpenAIProvider(
        api_key=api_key,
        base_url=base_url,
    )
    return OpenAIChatModel(llm_choice, provider=provider)


def get_embedding_client() -> openai.AsyncOpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")

    return openai.AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
    )


def get_embedding_model() -> str:
    return os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")


def get_embedding_dimensions(model_choice: Optional[str] = None) -> Optional[int]:
    raw = str(os.getenv("EMBEDDING_DIMENSIONS", "") or "").strip()
    if raw:
        value = int(raw)
        return value if value > 0 else None

    model_name = str(model_choice or get_embedding_model() or "").strip()
    if model_name.startswith("text-embedding-3"):
        return 1024
    if model_name.startswith("Qwen/Qwen3-Embedding-"):
        return 1024
    return None


def build_embedding_request_kwargs(
    *,
    model: str,
    input_value: Any,
    encoding_format: str = "float",
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": model,
        "input": input_value,
        "encoding_format": encoding_format,
    }
    dimensions = get_embedding_dimensions(model)
    if dimensions is not None:
        payload["dimensions"] = dimensions
    return payload


async def test_llm_connection() -> tuple[bool, str | None]:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    base_url = os.getenv("OPENAI_BASE_URL", "").strip()
    llm_choice = os.getenv("LLM_CHOICE", "").strip()

    if not api_key:
        return False, "OPENAI_API_KEY is missing"
    if not base_url:
        return False, "OPENAI_BASE_URL is missing"
    if not llm_choice:
        return False, "LLM_CHOICE is missing"

    client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
    try:
        await asyncio.wait_for(
            client.chat.completions.create(
                model=llm_choice,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
                temperature=0,
            ),
            timeout=8,
        )
        return True, None
    except Exception as exc:
        return False, str(exc)
