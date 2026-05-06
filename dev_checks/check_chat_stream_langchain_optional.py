import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

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
        "user_id": "dev_check_stream_langchain_optional_user",
        "search_type": "hybrid",
        "use_web_search": False,
        "use_react": False,
    }
    resp = requests.post(f"{base_url}/chat/stream", json=payload, stream=True, timeout=240)
    if resp.status_code != 200:
        fail(f"/chat/stream returned {resp.status_code}: {resp.text}")

    events: List[Dict[str, Any]] = []
    for raw in resp.iter_lines(decode_unicode=True):
        if not raw or not raw.startswith("data: "):
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

    health = requests.get(f"{base_url}/health", timeout=10)
    if health.status_code != 200:
        fail(f"/health returned {health.status_code}")

    events = _collect_stream_events(base_url)
    if not events:
        fail("no SSE events received")

    types = [str(e.get("type") or "") for e in events]
    info(f"event types={types}")

    if types[0] != "session":
        fail(f"first event must be session, got {types[0]!r}")
    if types[-1] != "end":
        fail(f"last event must be end, got {types[-1]!r}")
    if "error" in types:
        fail("stream contains error event")
    if "text" not in types:
        fail("missing text event")
    if "sources" not in types:
        fail("missing sources event")

    text_parts: List[str] = []
    stream_backend_seen: Optional[str] = None
    for event in events:
        etype = event.get("type")
        if etype == "text":
            content = event.get("content")
            if not isinstance(content, str):
                fail("text event content must be string")
            text_parts.append(content)
        elif etype == "sources":
            if not isinstance(event.get("sources"), list):
                fail("sources event `sources` must be list")
        elif etype == "tools":
            if not isinstance(event.get("tools"), list):
                fail("tools event `tools` must be list")

    full_text = "".join(text_parts).strip()
    if not full_text:
        fail("stream text content is empty")

    session_id = events[0].get("session_id")
    if isinstance(session_id, str) and session_id.strip():
        msg_resp = requests.get(f"{base_url}/sessions/{session_id}/messages", timeout=30)
        if msg_resp.status_code == 200:
            body = msg_resp.json()
            messages = body.get("messages") if isinstance(body, dict) else None
            if isinstance(messages, list):
                assistants = [m for m in messages if isinstance(m, dict) and m.get("role") == "assistant"]
                if assistants:
                    meta = assistants[-1].get("metadata")
                    if isinstance(meta, dict):
                        candidate = meta.get("stream_backend")
                        if isinstance(candidate, str):
                            stream_backend_seen = candidate

    if stream_backend_seen is not None and stream_backend_seen != "langchain":
        fail(
            f"service stream backend is {stream_backend_seen!r}, expected 'langchain'. "
            "Please restart API with STREAM_BACKEND=langchain."
        )
    if stream_backend_seen is None:
        fail(
            "unable to confirm STREAM_BACKEND=langchain from persisted metadata. "
            "Please restart API with STREAM_BACKEND=langchain and retry."
        )

    print("check_chat_stream_langchain_optional passed")


if __name__ == "__main__":
    main()
