import os
import sys
import requests

API_URL = os.getenv("API_URL", "http://localhost:8058")
LONG_MESSAGE = "请基于论文内容继续深入分析方法细节与实验设计。 " * 1200


def fail(msg: str):
    print(f"FAIL: {msg}")
    sys.exit(1)


def get_session_metadata(session_id: str):
    resp = requests.get(f"{API_URL}/sessions/{session_id}", timeout=20)
    if resp.status_code != 200:
        fail(f"获取会话失败: {resp.status_code} {resp.text}")
    payload = resp.json()
    return payload.get("metadata", {})


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


def main():
    print(f"Running summary compression check against {API_URL}")

    first = send_chat(LONG_MESSAGE, None)
    session_id = first.get("session_id")
    if not session_id:
        fail("首轮聊天未返回 session_id")

    print(f"Session created: {session_id}")

    triggered = False
    for i in range(1, 9):
        send_chat(LONG_MESSAGE, session_id)
        metadata = get_session_metadata(session_id)
        compression_count = int(metadata.get("compression_count") or 0)
        compacted_count = int(metadata.get("compacted_message_count") or 0)
        latest_summary = str(metadata.get("latest_summary") or "").strip()
        print(
            f"Round {i}: compression_count={compression_count}, "
            f"compacted_message_count={compacted_count}, summary_len={len(latest_summary)}"
        )
        if compression_count >= 1 and compacted_count > 0 and latest_summary:
            triggered = True
            break

    if not triggered:
        fail("未触发摘要压缩，metadata 未达到预期")

    follow_up = send_chat("请继续，总结当前讨论中最关键的3点。", session_id)
    answer = str(follow_up.get("message") or "").strip()
    if not answer:
        fail("压缩后续聊失败：返回消息为空")

    print("PASS: 摘要压缩触发成功，metadata 正确，且可继续聊天")


if __name__ == "__main__":
    main()
