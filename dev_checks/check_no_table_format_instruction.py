import ast
from pathlib import Path


def main() -> None:
    src = Path("agent/routing.py").read_text(encoding="utf-8")
    mod = ast.parse(src)
    fn = next(
        node for node in mod.body
        if isinstance(node, ast.FunctionDef) and node.name == "_build_format_instruction"
    )
    ns = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "<routing>", "exec"), ns)
    text = ns["_build_format_instruction"](has_local_evidence=True, is_general_question=False)

    assert "严禁使用 Markdown 表格" in text
    assert "不要输出 `##1.`" in text
    assert "对比类问题也不要使用 `|`" in text
    assert "## 1. 标题" in text
    print("PASS: no-table format instruction constraints present")


if __name__ == "__main__":
    main()
