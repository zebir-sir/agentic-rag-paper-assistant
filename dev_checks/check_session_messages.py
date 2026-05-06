import os
import sys
from datetime import datetime
import requests


API_URL = os.getenv("API_URL", "http://localhost:8058")
REQUIRED_FIELDS = {"message_id", "role", "content", "created_at", "metadata"}


def fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def main() -> None:
    sessions_resp = requests.get(f"{API_URL}/sessions", timeout=30)
    if sessions_resp.status_code != 200:
        fail(f"/sessions status={sessions_resp.status_code} body={sessions_resp.text}")
    sessions = sessions_resp.json().get("sessions", [])
    if not sessions:
        fail("session list is empty")

    session_id = sessions[0]["session_id"]
    resp = requests.get(f"{API_URL}/sessions/{session_id}/messages", timeout=30)
    if resp.status_code != 200:
        fail(f"/sessions/{{id}}/messages status={resp.status_code} body={resp.text}")

    payload = resp.json()
    messages = payload.get("messages", [])
    if not messages:
        fail("messages list is empty")

    for idx, msg in enumerate(messages):
        missing = [f for f in REQUIRED_FIELDS if f not in msg]
        if missing:
            fail(f"message[{idx}] missing fields: {missing}")

    times = [parse_time(m["created_at"]) for m in messages]
    if times != sorted(times):
        fail("messages are not ordered by created_at ascending")

    print(f"PASS: session message recovery works, session_id={session_id}, messages={len(messages)}")


if __name__ == "__main__":
    main()

