import os
import sys
import requests

API_URL = os.getenv("API_URL", "http://localhost:8058")


def fail(msg: str):
    print(f"FAIL: {msg}")
    sys.exit(1)


def main():
    payload = {
        "message": "请总结一下我上传文档的核心方法，并给出依据片段。",
        "search_type": "hybrid",
    }
    resp = requests.post(f"{API_URL}/chat", json=payload, timeout=120)
    if resp.status_code != 200:
        fail(f"/chat 请求失败: {resp.status_code} {resp.text}")

    data = resp.json()
    message = data.get("message")
    session_id = data.get("session_id")
    sources = data.get("sources") or []

    if not message:
        fail("message 为空")
    if not session_id:
        fail("session_id 为空")
    if not isinstance(sources, list) or len(sources) == 0:
        fail("sources 为空")

    first = sources[0]
    if not first.get("document_title"):
        fail("source.document_title 缺失")
    if not first.get("snippet"):
        fail("source.snippet 缺失")

    hist = requests.get(f"{API_URL}/sessions/{session_id}/messages", timeout=30)
    if hist.status_code != 200:
        fail(f"历史消息接口失败: {hist.status_code} {hist.text}")
    items = hist.json().get("messages") or []

    assistant_msgs = [m for m in items if m.get("role") == "assistant"]
    if not assistant_msgs:
        fail("历史中没有 assistant 消息")

    latest_meta = assistant_msgs[-1].get("metadata") or {}
    saved_sources = latest_meta.get("sources") or []
    if not isinstance(saved_sources, list) or len(saved_sources) == 0:
        fail("assistant metadata 中未保存 sources")

    print("PASS: /chat sources 返回与历史保存校验通过")


if __name__ == "__main__":
    main()
