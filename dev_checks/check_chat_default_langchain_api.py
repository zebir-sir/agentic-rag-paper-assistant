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
    for raw in [os.getenv("API_URL_FOR_CHECK"), "http://localhost:8888", "http://localhost:8059"]:
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
    fail("API is not reachable")
    return API_URL


def main() -> None:
    base_url = _resolve_api_url()
    info(f"API_URL={base_url}")

    payload: Dict[str, Any] = {
        "message": "你好，请用一句话介绍你是什么系统",
        "user_id": "dev_check_default_langchain_user",
        "search_type": "hybrid",
        "use_web_search": False,
        "use_react": False,
        "metadata": {},
    }
    resp = requests.post(f"{base_url}/chat", json=payload, timeout=240)
    if resp.status_code != 200:
        fail(f"/chat returned {resp.status_code}: {resp.text}")
    data = resp.json()

    if not isinstance(data.get("message"), str) or not data.get("message", "").strip():
        fail("message must be non-empty")
    if not isinstance(data.get("session_id"), str) or not data.get("session_id", "").strip():
        fail("session_id must be non-empty")
    if not isinstance(data.get("tools_used"), list):
        fail("tools_used must be list")
    if not isinstance(data.get("sources"), list):
        fail("sources must be list")
    metadata = data.get("metadata")
    if not isinstance(metadata, dict):
        fail("metadata must be dict")

    if metadata.get("agent_backend") != "langchain":
        fail(f"agent_backend should be langchain, got {metadata.get('agent_backend')}")
    if metadata.get("use_react") is not False:
        fail(f"use_react should be False, got {metadata.get('use_react')}")
    if metadata.get("workflow") == "deep_analysis":
        fail("workflow deep_analysis should not appear for use_react=false")

    print("check_chat_default_langchain_api passed")


if __name__ == "__main__":
    main()
