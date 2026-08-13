"""Embedding provider and language routing policy for the research corpus."""

import os
import re
from dataclasses import dataclass
from typing import Literal

import openai
from dotenv import load_dotenv

load_dotenv()

EmbeddingLanguage = Literal["zh", "en"]
_CJK_RE = re.compile(r"[\u3400-\u9fff]")


@dataclass(frozen=True)
class EmbeddingRoute:
    language: EmbeddingLanguage
    model: str
    api_key: str
    base_url: str


def detect_embedding_language(text: str) -> EmbeddingLanguage:
    """Use a conservative CJK ratio; formulas and identifiers do not affect routing."""
    value = str(text or "")
    letters = re.findall(r"[A-Za-z\u3400-\u9fff]", value)
    if not letters:
        return "en"
    cjk_count = len(_CJK_RE.findall(value))
    return "zh" if cjk_count / len(letters) >= 0.08 else "en"


def get_embedding_route(text: str = "", language: EmbeddingLanguage | None = None) -> EmbeddingRoute:
    selected_language = language or detect_embedding_language(text)
    model_var = "EMBEDDING_MODEL_ZH" if selected_language == "zh" else "EMBEDDING_MODEL_EN"
    fallback_model = "BAAI/bge-large-zh-v1.5" if selected_language == "zh" else "BAAI/bge-large-en-v1.5"
    return EmbeddingRoute(
        language=selected_language,
        model=os.getenv(model_var, fallback_model),
        api_key=os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY", ""),
        base_url=os.getenv("EMBEDDING_BASE_URL") or os.getenv("OPENAI_BASE_URL", ""),
    )


def get_embedding_client_for_route(route: EmbeddingRoute) -> openai.AsyncOpenAI:
    return openai.AsyncOpenAI(api_key=route.api_key, base_url=route.base_url)
