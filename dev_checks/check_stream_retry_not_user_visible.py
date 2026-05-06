from pathlib import Path


def main() -> None:
    text = Path("agent/api.py").read_text(encoding="utf-8")
    forbidden = 'yield sse_event("status", content="检测到回答格式异常'
    if forbidden in text:
        raise AssertionError("retry status message is user-visible in SSE")

    for field in ["retry_attempted", "retry_failed", "retry_suppressed"]:
        if field not in text:
            raise AssertionError(f"missing retry metadata field: {field}")

    print("PASS: stream retry status is not user-visible")


if __name__ == "__main__":
    main()
