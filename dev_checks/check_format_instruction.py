import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.routing import _build_format_instruction


def main() -> None:
    text = _build_format_instruction(has_local_evidence=True, is_general_question=False)
    assert "## 1." in text, "missing markdown heading guidance"
    assert "Markdown 表格" in text, "missing markdown table guidance"
    assert "不要把多个编号挤在一行" in text, "missing list line-break guidance"
    print("PASS: format instruction contains markdown constraints")


if __name__ == "__main__":
    main()
