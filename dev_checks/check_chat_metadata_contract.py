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


def _post_chat(base_url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    resp = requests.post(f"{base_url}/chat", json=payload, timeout=240)
    if resp.status_code != 200:
        fail(f"/chat returned {resp.status_code}: {resp.text}")
    data = resp.json()
    if not isinstance(data.get("tools_used"), list):
        fail("tools_used must be list")
    if not isinstance(data.get("sources"), list):
        fail("sources must be list")
    if not isinstance(data.get("metadata"), dict):
        fail("metadata must be dict")
    return data


def _tool_names(items: List[Any]) -> List[str]:
    names: List[str] = []
    for item in items:
        if isinstance(item, dict):
            value = item.get("tool_name") or item.get("name")
            if isinstance(value, str) and value.strip():
                names.append(value.strip())
    return names


def main() -> None:
    base_url = _resolve_api_url()
    info(f"API_URL={base_url}")

    data_a = _post_chat(
        base_url,
        {
            "message": "你好，请用一句话介绍你是什么系统",
            "user_id": "dev_check_metadata_contract_user",
            "search_type": "hybrid",
            "use_web_search": False,
            "use_react": False,
            "metadata": {},
        },
    )
    meta_a = data_a["metadata"]
    if meta_a.get("agent_backend") != "langchain":
        fail(f"use_react=false should be langchain, got {meta_a.get('agent_backend')}")
    if meta_a.get("use_react") is not False:
        fail("use_react=false call metadata.use_react should be False")
    for key in [
        "requested_search_type",
        "effective_search_type",
        "compression_used",
        "use_web_search",
        "openalex_enabled",
    ]:
        if key not in meta_a:
            fail(f"use_react=false metadata missing `{key}`")

    data_b = _post_chat(
        base_url,
        {
            "message": "请用中文说明你作为科研论文阅读助手能做什么；如果知识库证据不足，你会如何处理？",
            "user_id": "dev_check_metadata_contract_user",
            "search_type": "hybrid",
            "use_web_search": False,
            "use_react": True,
            "metadata": {},
        },
    )
    meta_b = data_b["metadata"]
    if meta_b.get("agent_backend") != "langgraph":
        fail(f"use_react=true should be langgraph, got {meta_b.get('agent_backend')}")
    if meta_b.get("use_react") is not True:
        fail("use_react=true call metadata.use_react should be True")
    if meta_b.get("workflow") != "deep_analysis":
        fail(f"use_react=true workflow should be deep_analysis, got {meta_b.get('workflow')}")
    if meta_b.get("evidence_checked") is not True:
        fail("use_react=true metadata.evidence_checked should be True")
    for key in ["source_count", "retrieval_result_count", "tool_count"]:
        if key not in meta_b:
            fail(f"use_react=true metadata missing `{key}`")

    names = _tool_names(data_b.get("tools_used") or [])
    if "list_documents" not in names:
        fail(f"use_react=true tools_used should include list_documents, got {names}")

    print("check_chat_metadata_contract passed")


if __name__ == "__main__":
    main()
