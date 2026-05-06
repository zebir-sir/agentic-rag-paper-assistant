import os
import sys
import requests


API_URL = os.getenv("API_URL", "http://localhost:8058")
REQUIRED_FIELDS = {
    "session_id",
    "title",
    "created_at",
    "updated_at",
    "expires_at",
    "message_count",
}


def fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


def main() -> None:
    resp = requests.get(f"{API_URL}/sessions", timeout=30)
    if resp.status_code != 200:
        fail(f"/sessions status={resp.status_code} body={resp.text}")

    payload = resp.json()
    sessions = payload.get("sessions", [])
    if not sessions:
        fail("session list is empty")

    first = sessions[0]
    missing = [f for f in REQUIRED_FIELDS if f not in first]
    if missing:
        fail(f"missing fields in session list item: {missing}")
    if not first.get("title"):
        fail("title is empty in first session item")

    print(f"PASS: session list works, total={payload.get('total')}, first_session={first.get('session_id')}")


if __name__ == "__main__":
    main()

