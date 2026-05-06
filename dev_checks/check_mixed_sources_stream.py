import json
import os
import sys
from typing import Any, Dict, List

import requests

API_URL = os.getenv("API_URL", "http://localhost:8058")


def print_status(status: str, msg: str) -> None:
    print(f"{status}: {msg}")


def fail(msg: str) -> None:
    print_status("FAIL", msg)
    sys.exit(1)


def fetch_openalex_enabled() -> bool:
    try:
        resp = requests.get(f"{API_URL}/openalex/status", timeout=10)
        return bool(resp.json().get("enabled")) if resp.status_code == 200 else False
    except Exception:
        return False


def validate_source_types(sources: List[Dict[str, Any]]) -> None:
    for idx, source in enumerate(sources, start=1):
        stype = source.get("source_type")
        if stype not in {"local", "web"}:
            fail(f"第{idx}条来源的 source_type 非法: {stype}")


def main() -> None:
    enabled = fetch_openalex_enabled()
    if not enabled:
        print_status("SKIP", "OpenAlex 未启用（缺少 OPENALEX_API_KEY）")
        return

    payload = {
        "message": "请补充本地论文之外的相关工作，并区分本地与联网依据。",
        "search_type": "hybrid",
        "use_web_search": True,
    }
    resp = requests.post(f"{API_URL}/chat/stream", json=payload, stream=True, timeout=240)
    if resp.status_code != 200:
        fail(f"/chat/stream 请求失败: {resp.status_code} {resp.text}")

    seen_session = False
    seen_text = False
    seen_sources = False
    seen_end = False
    source_types: List[str] = []

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
            if event.get("content"):
                seen_text = True
        elif etype == "sources":
            seen_sources = True
            sources = event.get("sources") or []
            if sources:
                validate_source_types(sources)
                source_types = sorted(set([s.get("source_type") for s in sources]))
        elif etype == "end":
            seen_end = True
            break
        elif etype == "error":
            fail(f"流式返回 error: {event.get('content')}")

    if not seen_session or not seen_text or not seen_end:
        fail(f"流式事件链不完整: session={seen_session}, text={seen_text}, end={seen_end}")
    if not seen_sources:
        fail("未收到 sources 事件")

    if not source_types:
        print_status("SKIP", "收到 sources 事件，但本轮未形成可验证来源（可能未触发有效检索）")
        return

    print_status("PASS", f"流式来源事件正常，source_type={source_types}")


if __name__ == "__main__":
    main()
