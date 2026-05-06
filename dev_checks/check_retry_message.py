import re
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    text = (root / "agent" / "agent_langchain.py").read_text(encoding="utf-8")
    m = re.search(r'GENERATION_RETRY_FAILED_MESSAGE\s*=\s*"([^"]+)"', text)
    assert m, "GENERATION_RETRY_FAILED_MESSAGE not found"
    msg = m.group(1)
    lowered = msg.lower()
    assert "联网" not in msg, "retry message should not suggest networking"
    assert "开启联网搜索" not in msg, "retry message should not suggest enable web search"
    assert "openalex" not in lowered, "retry message should not mention OpenAlex"
    print("PASS: retry message is local-safe")


if __name__ == "__main__":
    main()
