import json
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


def _collect_stream_events(base_url: str) -> List[Dict[str, Any]]:
    payload = {
        "message": "请简单介绍你能做什么，并说明如果知识库没有证据你会如何处理。",
        "user_id": "dev_check_stream_contract_user",
        "search_type": "hybrid",
        "use_web_search": False,
        "use_react": False,
    }
    resp = requests.post(f"{base_url}/chat/stream", json=payload, stream=True, timeout=240)
    if resp.status_code != 200:
        fail(f"/chat/stream returned {resp.status_code}: {resp.text}")

    events: List[Dict[str, Any]] = []
    for raw in resp.iter_lines(decode_unicode=True):
        if not raw:
            continue
        if not raw.startswith("data: "):
            continue
        try:
            event = json.loads(raw[6:])
        except json.JSONDecodeError:
            continue
        events.append(event)
        if event.get("type") == "end":
            break
    return events


def main() -> None:
    base_url = _resolve_api_url()
    info(f"API_URL={base_url}")

    events = _collect_stream_events(base_url)
    if not events:
        fail("no SSE events received")

    types = [str(e.get("type") or "") for e in events]
    info(f"event types={types}")

    # Required event presence.
    if "session" not in types:
        fail("missing session event")
    if "text" not in types:
        fail("missing text event")
    if "sources" not in types:
        fail("missing sources event")
    if "end" not in types:
        fail("missing end event")
    if "error" in types:
        fail("stream contains error event")

    # Order checks.
    if types[0] != "session":
        fail(f"first event must be session, got {types[0]!r}")
    if types[-1] != "end":
        fail(f"last event must be end, got {types[-1]!r}")

    idx_session = types.index("session")
    idx_end = len(types) - 1
    idx_sources = types.index("sources")
    idx_text = types.index("text")

    if not (idx_session < idx_text < idx_end):
        fail("text event order invalid: must be after session and before end")
    if not (idx_sources < idx_end):
        fail("sources must appear before end")

    # Payload checks.
    session_event = events[idx_session]
    if not isinstance(session_event.get("session_id"), str) or not session_event.get("session_id", "").strip():
        fail("session event must contain non-empty session_id")

    text_chunks: List[str] = []
    for event in events:
        etype = event.get("type")
        if etype == "text":
            content = event.get("content")
            if not isinstance(content, str):
                fail("text event content must be string")
            text_chunks.append(content)
        elif etype == "sources":
            if "sources" not in event:
                fail("sources event missing sources field")
            if not isinstance(event.get("sources"), list):
                fail("sources event field `sources` must be list")
        elif etype == "tools":
            if "tools" not in event:
                fail("tools event missing tools field")
            if not isinstance(event.get("tools"), list):
                fail("tools event field `tools` must be list")

    full_text = "".join(text_chunks).strip()
    if len(full_text) == 0:
        fail("concatenated text content is empty")
    forbidden_markers = [
        "NoneNone",
        "AIMessage(",
        "HumanMessage(",
        "ToolMessage(",
        "{'messages':",
        "{'model':",
        "additional_kwargs=",
        "response_metadata=",
        "usage_metadata=",
    ]
    for marker in forbidden_markers:
        if marker in full_text:
            fail(f"concatenated text contains forbidden marker: {marker}")

    print("check_chat_stream_contract passed")


if __name__ == "__main__":
    main()
