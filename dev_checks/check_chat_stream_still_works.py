import json
import os
import sys
import time
from typing import List

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

    payload = {
        "message": "请简单介绍你能做什么",
        "user_id": "dev_check_stream_user",
        "search_type": "hybrid",
        "use_web_search": False,
        "use_react": False,
        "metadata": {},
    }
    resp = requests.post(f"{base_url}/chat/stream", json=payload, stream=True, timeout=240)
    if resp.status_code != 200:
        fail(f"/chat/stream returned {resp.status_code}: {resp.text}")

    seen_session = False
    seen_text = False
    seen_sources = False
    seen_end = False

    for raw in resp.iter_lines(decode_unicode=True):
        if not raw or not raw.startswith("data: "):
            continue
        try:
            event = json.loads(raw[6:])
        except json.JSONDecodeError:
            continue
        etype = event.get("type")
        if etype == "session":
            seen_session = True
        elif etype == "text":
            if str(event.get("content") or "").strip():
                seen_text = True
        elif etype == "sources":
            seen_sources = True
        elif etype == "end":
            seen_end = True
            break
        elif etype == "error":
            fail(f"stream returned error: {event.get('content')}")

    if not seen_session:
        fail("missing session event")
    if not seen_text:
        fail("missing non-empty text event")
    if not seen_sources:
        fail("missing sources event")
    if not seen_end:
        fail("missing end event")

    print("check_chat_stream_still_works passed")


if __name__ == "__main__":
    main()
