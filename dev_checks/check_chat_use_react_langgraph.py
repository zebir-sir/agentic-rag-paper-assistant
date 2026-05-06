import os
import sys
import time
from typing import Any, Dict, List

import requests


API_URL = os.getenv("API_URL", os.getenv("API_URL_FOR_CHECK", "http://localhost:8888")).rstrip("/")


def fail(message: str) -> None:
    print(f"FAIL: {message}")
    sys.exit(1)


def info(message: str) -> None:
    print(f"INFO: {message}")


def _candidate_base_urls() -> List[str]:
    candidates = [API_URL]
    for raw in [
        os.getenv("API_URL_FOR_CHECK"),
        "http://localhost:8888",
        "http://localhost:8059",
    ]:
        if raw:
            normalized = raw.rstrip("/")
            if normalized not in candidates:
                candidates.append(normalized)
    return candidates


def _wait_for_api(base_url: str, timeout_sec: int = 45) -> bool:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            resp = requests.get(f"{base_url}/health", timeout=4)
            if resp.status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(1.0)
    return False


def _resolve_api_url() -> str:
    for base_url in _candidate_base_urls():
        if _wait_for_api(base_url):
            return base_url
    fail(
        "API is not reachable on candidates: "
        + ", ".join(_candidate_base_urls())
        + " ; please check container networking and API port."
    )
    return API_URL


def post_chat(base_url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    resp = requests.post(f"{base_url}/chat", json=payload, timeout=240)
    if resp.status_code != 200:
        fail(f"/chat returned {resp.status_code}: {resp.text}")
    try:
        return resp.json()
    except ValueError as exc:
        fail(f"/chat returned invalid JSON: {exc}")
    return {}


def ensure_common_shape(data: Dict[str, Any], tag: str) -> None:
    if not isinstance(data.get("message"), str) or not str(data.get("message")).strip():
        fail(f"{tag}: message must be non-empty string")
    if not isinstance(data.get("session_id"), str) or not str(data.get("session_id")).strip():
        fail(f"{tag}: session_id must be non-empty string")
    if not isinstance(data.get("metadata"), dict):
        fail(f"{tag}: metadata must be dict")
    if not isinstance(data.get("tools_used"), list):
        fail(f"{tag}: tools_used must be list")
    if not isinstance(data.get("sources"), list):
        fail(f"{tag}: sources must be list")


def _tool_names(tools: List[Any]) -> List[str]:
    names: List[str] = []
    for item in tools:
        if isinstance(item, dict):
            name = item.get("tool_name") or item.get("name")
            if isinstance(name, str) and name.strip():
                names.append(name.strip())
    return names


def main() -> None:
    base_url = _resolve_api_url()
    info(f"API_URL={base_url}")

    react_payload = {
        "message": "请用中文说明你作为科研论文阅读助手能做什么；如果知识库证据不足，你会如何处理？",
        "user_id": "dev_check_react_langgraph_user",
        "search_type": "hybrid",
        "use_web_search": False,
        "use_react": True,
        "metadata": {},
    }
    react_data = post_chat(base_url, react_payload)
    ensure_common_shape(react_data, "use_react=true")

    react_meta = react_data["metadata"]
    if react_meta.get("use_react") is not True:
        fail("use_react=true call: metadata.use_react should be true")
    if react_meta.get("agent_backend") != "langgraph":
        fail(f"use_react=true call: metadata.agent_backend should be langgraph, got {react_meta.get('agent_backend')}")
    if react_meta.get("workflow") != "deep_analysis":
        fail(f"use_react=true call: metadata.workflow should be deep_analysis, got {react_meta.get('workflow')}")
    if react_meta.get("evidence_checked") is not True:
        fail("use_react=true call: metadata.evidence_checked should be true")

    names = _tool_names(react_data.get("tools_used") or [])
    if "list_documents" not in names:
        fail(f"use_react=true call: tools_used should include list_documents, got {names}")

    normal_payload = {
        "message": "你好，请用一句话介绍你是什么系统",
        "user_id": "dev_check_react_langgraph_user",
        "search_type": "hybrid",
        "use_web_search": False,
        "use_react": False,
        "metadata": {},
    }
    normal_data = post_chat(base_url, normal_payload)
    ensure_common_shape(normal_data, "use_react=false")

    normal_meta = normal_data["metadata"]
    if normal_meta.get("agent_backend") != "langchain":
        fail(
            f"use_react=false call: metadata.agent_backend should be langchain, got {normal_meta.get('agent_backend')}"
        )
    if normal_meta.get("workflow") == "deep_analysis":
        fail("use_react=false call: should not route to LangGraph deep_analysis workflow")

    print("check_chat_use_react_langgraph passed")


if __name__ == "__main__":
    main()
