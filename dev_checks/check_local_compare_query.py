import ast
import re
from pathlib import Path
import sys
from typing import List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.title_aliases import TITLE_ALIASES, get_title_alias


def main():
    src = (ROOT / "agent" / "routing.py").read_text(encoding="utf-8")
    module = ast.parse(src)
    fn_node = next(
        node for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "_extract_known_local_paper_queries"
    )
    fn_module = ast.Module(body=[fn_node], type_ignores=[])
    ns = {"re": re, "TITLE_ALIASES": TITLE_ALIASES, "get_title_alias": get_title_alias, "List": List}
    exec(compile(fn_module, filename="<routing_extract>", mode="exec"), ns)
    extract_fn = ns["_extract_known_local_paper_queries"]

    q1 = "总结 HA-RRT 和 HMA-RRT 的区别"
    r1 = extract_fn(q1)
    text1 = " ".join(r1)
    assert "HA-RRT" in text1, "HA-RRT query missing"
    assert "HMA-RRT" in text1, "HMA-RRT query missing"

    q2 = "请总结 HMA-RRT 的方法"
    r2 = extract_fn(q2)
    text2 = " ".join(r2)
    assert "HMA-RRT" in text2, "HMA-RRT query missing for single title"
    assert "HA-RRT: A heuristic and adaptive RRT algorithm for ship path planning" not in text2, "HMA-RRT should not trigger HA-RRT query"
    print("PASS: local compare query extraction works")


if __name__ == "__main__":
    main()
