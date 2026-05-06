import os
import sys
from typing import Any, Dict

import requests

API_URL = os.getenv("API_URL", "http://localhost:8058")


def fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


def call_chat(use_react: bool) -> Dict[str, Any]:
    payload = {
        "message": "请简要总结当前知识库中的关键方法，并给出依据。",
        "search_type": "hybrid",
        "use_react": use_react,
        "use_web_search": False,
    }
    resp = requests.post(f"{API_URL}/chat", json=payload, timeout=180)
    if resp.status_code != 200:
        fail(f"/chat(use_react={use_react}) 失败: {resp.status_code} {resp.text}")
    data = resp.json()
    if not data.get("message"):
        fail(f"/chat(use_react={use_react}) message 为空")
    if not data.get("session_id"):
        fail(f"/chat(use_react={use_react}) session_id 为空")
    if "sources" not in data:
        fail(f"/chat(use_react={use_react}) 缺少 sources 字段")
    if "metadata" not in data:
        fail(f"/chat(use_react={use_react}) 缺少 metadata 字段")
    return data


def main() -> None:
    normal = call_chat(False)
    deep = call_chat(True)

    if not isinstance(normal.get("sources"), list):
        fail("普通模式 sources 不是列表")
    if not isinstance(deep.get("sources"), list):
        fail("深度模式 sources 不是列表")

    if not isinstance(deep.get("metadata"), dict):
        fail("深度模式 metadata 非字典")
    if deep["metadata"].get("use_react") is not True:
        fail("深度模式 metadata.use_react 不是 true")

    print("PASS: use_react=false/true 均可正常调用 /chat，且结构兼容")


if __name__ == "__main__":
    main()
