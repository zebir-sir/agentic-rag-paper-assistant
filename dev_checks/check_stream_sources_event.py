import json
import os
import sys
import requests

API_URL = os.getenv("API_URL", "http://localhost:8058")


def fail(msg: str):
    print(f"FAIL: {msg}")
    sys.exit(1)


def main():
    payload = {
        "message": "请给出文档结论并展示依据片段。",
        "search_type": "hybrid",
    }
    with requests.post(f"{API_URL}/chat/stream", json=payload, stream=True, timeout=180) as resp:
        if resp.status_code != 200:
            fail(f"/chat/stream 请求失败: {resp.status_code} {resp.text}")

        seen = {"session": False, "text": False, "sources": False, "end": False}
        sources_payload = []

        for raw in resp.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data: "):
                continue
            try:
                event = json.loads(raw[6:])
            except json.JSONDecodeError:
                continue

            et = event.get("type")
            if et in seen:
                seen[et] = True
            if et == "text" and event.get("content"):
                seen["text"] = True
            if et == "sources":
                sources_payload = event.get("sources") or []
            if et == "end":
                break

    missing = [k for k, v in seen.items() if not v]
    if missing:
        fail(f"缺少流事件: {missing}")

    if not isinstance(sources_payload, list) or len(sources_payload) == 0:
        fail("sources 事件为空")

    first = sources_payload[0]
    required = ["document_title", "snippet", "document_source"]
    for field in required:
        if not first.get(field):
            fail(f"sources 字段缺失: {field}")

    print("PASS: /chat/stream sources 事件校验通过")


if __name__ == "__main__":
    main()
