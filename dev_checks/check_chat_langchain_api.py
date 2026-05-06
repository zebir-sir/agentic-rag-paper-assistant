import os
import sys
from typing import Any, Dict, List

import requests


API_URL = os.getenv("API_URL_FOR_CHECK", "http://localhost:8888").rstrip("/")


def fail(message: str) -> None:
    print(f"FAIL: {message}")
    sys.exit(1)


def info(message: str) -> None:
    print(f"INFO: {message}")


def warn(message: str) -> None:
    print(f"WARN: {message}")


def post_json(path: str, payload: Dict[str, Any], timeout: int = 180) -> Dict[str, Any]:
    response = requests.post(f"{API_URL}{path}", json=payload, timeout=timeout)
    if response.status_code != 200:
        fail(f"{path} returned {response.status_code}: {response.text}")
    try:
        return response.json()
    except ValueError as exc:
        fail(f"{path} did not return valid JSON: {exc}")
    return {}


def get_json(path: str, timeout: int = 60) -> Dict[str, Any]:
    response = requests.get(f"{API_URL}{path}", timeout=timeout)
    if response.status_code != 200:
        fail(f"{path} returned {response.status_code}: {response.text}")
    try:
        return response.json()
    except ValueError as exc:
        fail(f"{path} did not return valid JSON: {exc}")
    return {}


def require_chat_shape(data: Dict[str, Any], tag: str) -> None:
    message = data.get("message")
    session_id = data.get("session_id")
    sources = data.get("sources")
    tools_used = data.get("tools_used")
    metadata = data.get("metadata")

    if not isinstance(message, str) or not message.strip():
        fail(f"{tag}: message must be non-empty string")
    if not isinstance(session_id, str) or not session_id.strip():
        fail(f"{tag}: session_id must be non-empty string")
    if not isinstance(sources, list):
        fail(f"{tag}: sources must be list, got {type(sources).__name__}")
    if not isinstance(tools_used, list):
        fail(f"{tag}: tools_used must be list, got {type(tools_used).__name__}")
    if not isinstance(metadata, dict):
        fail(f"{tag}: metadata must be dict, got {type(metadata).__name__}")

    for key in [
        "search_type",
        "requested_search_type",
        "effective_search_type",
        "compression_used",
        "use_web_search",
        "use_react",
        "openalex_enabled",
        "agent_backend",
    ]:
        if key not in metadata:
            fail(f"{tag}: metadata missing key `{key}`")

    if metadata.get("agent_backend") != "langchain":
        fail(
            f"{tag}: metadata.agent_backend={metadata.get('agent_backend')!r}, expected 'langchain'. "
            "请确认 .env 中 AGENT_BACKEND=langchain 并已重启容器。"
        )


def short_text(value: Any, limit: int = 80) -> str:
    text = str(value or "").strip().replace("\n", " ")
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def assistant_tool_names(tools_used: List[Dict[str, Any]]) -> List[str]:
    names: List[str] = []
    for item in tools_used:
        if isinstance(item, dict):
            name = item.get("tool_name") or item.get("name")
            if isinstance(name, str) and name.strip():
                names.append(name.strip())
    return names


def main() -> None:
    info(f"API_URL={API_URL}")

    payload_1 = {
        "message": "你好，请用一句话介绍你是什么系统",
        "user_id": "dev_check_langchain_api",
        "search_type": "hybrid",
        "use_web_search": False,
        "use_react": False,
        "metadata": {},
    }
    chat_1 = post_json("/chat", payload_1)
    require_chat_shape(chat_1, "chat_1")
    session_id = chat_1["session_id"]
    info(f"first session_id={session_id}")
    info(f"first message preview={short_text(chat_1.get('message'))}")

    payload_2 = {
        "message": "请接着上一轮，用更短的一句话说明你的用途。",
        "session_id": session_id,
        "user_id": "dev_check_langchain_api",
        "search_type": "hybrid",
        "use_web_search": False,
        "use_react": False,
        "metadata": {},
    }
    chat_2 = post_json("/chat", payload_2)
    require_chat_shape(chat_2, "chat_2")
    if chat_2.get("session_id") != session_id:
        fail("chat_2: session_id is not the same as chat_1")
    info(f"second message preview={short_text(chat_2.get('message'))}")

    payload_3 = {
        "message": "请先查看当前知识库有哪些文档，然后用一句中文总结结果。如果没有文档，请说明当前知识库为空。",
        "user_id": "dev_check_langchain_api",
        "search_type": "hybrid",
        "use_web_search": False,
        "use_react": False,
        "metadata": {},
    }
    chat_3 = post_json("/chat", payload_3)
    require_chat_shape(chat_3, "chat_3")
    tools = chat_3.get("tools_used") or []
    tool_names = assistant_tool_names(tools)
    info(f"third tools_used names={tool_names}")
    if not tool_names:
        warn("third request returned empty tools_used (not a failure in this phase)")

    sessions_payload = get_json("/sessions")
    sessions = sessions_payload.get("sessions")
    if not isinstance(sessions, list):
        fail("/sessions: `sessions` must be list")
    info(f"sessions count={len(sessions)}")

    messages_payload = get_json(f"/sessions/{session_id}/messages")
    messages = messages_payload.get("messages")
    if not isinstance(messages, list):
        fail("/sessions/{session_id}/messages: `messages` must be list")
    if len(messages) < 2:
        fail("session messages less than 2, conversation was not saved correctly")

    roles = [m.get("role") for m in messages if isinstance(m, dict)]
    if "user" not in roles or "assistant" not in roles:
        fail(f"session messages missing required roles, got roles={roles}")

    assistant_messages = [m for m in messages if isinstance(m, dict) and m.get("role") == "assistant"]
    if not assistant_messages:
        fail("no assistant message found in history")

    latest_assistant_metadata = assistant_messages[-1].get("metadata") or {}
    if not isinstance(latest_assistant_metadata, dict):
        fail("assistant metadata in history is not dict")
    if latest_assistant_metadata.get("agent_backend") != "langchain":
        fail(
            "assistant metadata.agent_backend is not langchain in history; "
            "请确认当前 backend 为 langchain 并重启后重试。"
        )
    if "sources" not in latest_assistant_metadata:
        fail("assistant metadata missing `sources`")
    sources_meta = latest_assistant_metadata.get("sources")
    if not isinstance(sources_meta, list):
        fail("assistant metadata.sources must be list")

    info("history recovery check passed")
    print("check_chat_langchain_api passed")


if __name__ == "__main__":
    main()
