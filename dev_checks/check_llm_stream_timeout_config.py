from pathlib import Path


def assert_contains(text: str, needle: str, label: str) -> None:
    if needle not in text:
        raise AssertionError(f"Missing {label}: {needle}")


api_text = Path("agent/api.py").read_text(encoding="utf-8")
langchain_text = Path("agent/agent_langchain.py").read_text(encoding="utf-8")

assert_contains(api_text, "LLM_FIRST_TOKEN_TIMEOUT_SECONDS", "first-token-timeout config")
assert_contains(api_text, "已找到相关片段，正在生成回答", "pre-generation status")

if "async for event in iter_langchain_agent_stream" in api_text:
    raise AssertionError("LangChain stream branch still uses bare async-for without timeout wrapper")

if "__anext__" not in api_text and "_next_stream_event_with_timeout" not in api_text:
    raise AssertionError("No __anext__/wait_for timeout wrapper found for LangChain stream iterator")

assert_contains(langchain_text, "LLM_REQUEST_TIMEOUT_SECONDS", "LLM request timeout env")
if "timeout=request_timeout_seconds" not in langchain_text and "request_timeout=request_timeout_seconds" not in langchain_text:
    raise AssertionError("ChatOpenAI timeout/request_timeout not configured")

print("check_llm_stream_timeout_config: ok")
