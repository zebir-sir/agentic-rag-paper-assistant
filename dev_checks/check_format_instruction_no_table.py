import ast
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    src = (root / "agent" / "routing.py").read_text(encoding="utf-8")
    module = ast.parse(src)
    fn_node = next(
        node for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "_build_format_instruction"
    )
    fn_module = ast.Module(body=[fn_node], type_ignores=[])
    ns = {}
    exec(compile(fn_module, filename="<routing_format>", mode="exec"), ns)
    text = ns["_build_format_instruction"](has_local_evidence=True, is_general_question=False)
    assert "默认不要使用 Markdown 表格" in text
    assert "## 1. 标题" in text
    assert "不要输出 `##1.`" in text
    print("PASS: format instruction no-table constraints present")


if __name__ == "__main__":
    main()
