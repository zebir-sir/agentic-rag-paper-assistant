import json
import os
import sys

import requests

API_URL = os.getenv("API_URL", "http://localhost:8058")


def fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


def main() -> None:
    payload = {
        "message": "请进行深度分析：比较本地依据并给出结论。",
        "search_type": "hybrid",
        "use_react": True,
        "use_web_search": False,
    }
    resp = requests.post(f"{API_URL}/chat/stream", json=payload, stream=True, timeout=240)
    if resp.status_code != 200:
        fail(f"/chat/stream 请求失败: {resp.status_code} {resp.text}")

    seen_session = False
    seen_text = False
    seen_sources = False
    seen_end = False
    seen_plan_status = False
    seen_check_status = False

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
        elif etype == "status":
            content = str(event.get("content") or "")
            if "正在规划分析步骤" in content:
                seen_plan_status = True
            if "正在核对依据" in content:
                seen_check_status = True
        elif etype == "text":
            if event.get("content"):
                seen_text = True
        elif etype == "sources":
            seen_sources = True
        elif etype == "end":
            seen_end = True
            break
        elif etype == "error":
            fail(f"流式返回错误: {event.get('content')}")

    if not seen_session:
        fail("缺少 session 事件")
    if not seen_plan_status:
        fail("缺少“正在规划分析步骤...”状态事件")
    if not seen_text:
        fail("缺少 text 事件")
    if not seen_sources:
        fail("缺少 sources 事件")
    if not seen_end:
        fail("缺少 end 事件")

    if seen_check_status:
        print("PASS: 深度分析流式链路正常，已检测到规划与核对状态事件")
    else:
        print("PASS: 深度分析流式链路正常，已检测到规划状态事件")


if __name__ == "__main__":
    main()
