import os
import sys
import requests


API_URL = os.getenv("API_URL", "http://localhost:8058")


def fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


def main() -> None:
    first = requests.post(
        f"{API_URL}/chat",
        json={"message": "第一条测试消息", "search_type": "hybrid"},
        timeout=30,
    )
    if first.status_code != 200:
        fail(f"/chat first call status={first.status_code} body={first.text}")

    first_data = first.json()
    session_id = first_data.get("session_id")
    if not session_id:
        fail("first /chat did not return session_id")

    second = requests.post(
        f"{API_URL}/chat",
        json={"message": "第二条测试消息", "search_type": "hybrid", "session_id": session_id},
        timeout=30,
    )
    if second.status_code != 200:
        fail(f"/chat second call status={second.status_code} body={second.text}")

    second_data = second.json()
    second_session_id = second_data.get("session_id")
    if second_session_id != session_id:
        fail(f"session_id changed: first={session_id} second={second_session_id}")

    messages_resp = requests.get(f"{API_URL}/sessions/{session_id}/messages", timeout=30)
    if messages_resp.status_code != 200:
        fail(f"/sessions/{{id}}/messages status={messages_resp.status_code} body={messages_resp.text}")
    messages = messages_resp.json().get("messages", [])
    if len(messages) < 4:
        fail(f"expected at least 4 messages in session, got {len(messages)}")

    print(f"PASS: session continuation works, session_id={session_id}, messages={len(messages)}")


if __name__ == "__main__":
    main()

