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
        "message": "请结合本地知识库并补充最新 related work，给出简明对比并标注依据来源。",
        "search_type": "hybrid",
        "use_web_search": True,
    }
    resp = requests.post(f"{API_URL}/chat", json=payload, timeout=180)
    if resp.status_code != 200:
        fail(f"/chat 请求失败: {resp.status_code} {resp.text}")

    data = resp.json()
    session_id = data.get("session_id")
    sources = data.get("sources") or []
    if not session_id:
        fail("未返回 session_id")
    if not isinstance(sources, list) or len(sources) == 0:
        fail("未返回 sources")

    validate_source_types(sources)
    source_types = sorted(set([s.get("source_type") for s in sources]))
    print_status("INFO", f"/chat 返回来源类型: {source_types}")

    hist = requests.get(f"{API_URL}/sessions/{session_id}/messages", timeout=30)
    if hist.status_code != 200:
        fail(f"历史消息接口失败: {hist.status_code} {hist.text}")
    messages = hist.json().get("messages") or []
    assistants = [m for m in messages if m.get("role") == "assistant"]
    if not assistants:
        fail("历史中没有 assistant 消息")
    latest_meta = assistants[-1].get("metadata") or {}
    saved_sources = latest_meta.get("sources") or []
    if not isinstance(saved_sources, list) or not saved_sources:
        fail("assistant metadata 未保存 sources")
    validate_source_types(saved_sources)

    saved_types = sorted(set([s.get("source_type") for s in saved_sources]))
    if "local" in saved_types and "web" in saved_types:
        print_status("PASS", "本轮同时包含本地与联网来源，且已保存到历史 metadata")
        return

    print_status("PASS", f"本轮只包含单一来源类型 {saved_types}，结构校验通过")


if __name__ == "__main__":
    main()
