import json
import os
import sys
import requests

API_URL = os.getenv("API_URL", "http://localhost:8058")
LONG_MESSAGE = "请继续深入讨论论文中的方法、实验与局限，并给出更细节推理。 " * 1200
TARGET_STATUS = "正在压缩历史对话..."


def fail(msg: str):
    print(f"FAIL: {msg}")
    sys.exit(1)


def send_chat(message: str, session_id: str | None = None):
    payload = {
        "message": message,
        "session_id": session_id,
        "search_type": "hybrid",
    }
    resp = requests.post(f"{API_URL}/chat", json=payload, timeout=120)
    if resp.status_code != 200:
        fail(f"/chat 调用失败: {resp.status_code} {resp.text}")
    return resp.json()


def trigger_stream(session_id: str) -> bool:
    payload = {
        "message": LONG_MESSAGE,
        "session_id": session_id,
        "search_type": "hybrid",
    }
    with requests.post(f"{API_URL}/chat/stream", json=payload, stream=True, timeout=180) as resp:
        if resp.status_code != 200:
            fail(f"/chat/stream 调用失败: {resp.status_code} {resp.text}")

        found_status = False
        for raw in resp.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data: "):
                continue
            try:
                event = json.loads(raw[6:])
            except json.JSONDecodeError:
                continue

            if event.get("type") == "status":
                content = str(event.get("content") or "")
                print(f"status event: {content}")
                if TARGET_STATUS in content:
                    found_status = True
            if event.get("type") in {"end", "error"}:
                break

        return found_status


def main():
    print(f"Running stream compression status check against {API_URL}")

    first = send_chat(LONG_MESSAGE, None)
    session_id = first.get("session_id")
    if not session_id:
        fail("首轮聊天未返回 session_id")
    print(f"Session created: {session_id}")

    for i in range(1, 4):
        send_chat(LONG_MESSAGE, session_id)
        print(f"Warmup round {i} done")

    found = trigger_stream(session_id)
    if not found:
        fail("未收到压缩状态事件：正在压缩历史对话...")

    print("PASS: /chat/stream 返回了压缩状态事件")


if __name__ == "__main__":
    main()
