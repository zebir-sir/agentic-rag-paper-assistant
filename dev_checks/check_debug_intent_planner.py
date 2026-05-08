import importlib
import sys
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))

    module = importlib.import_module("dev_checks.debug_intent_planner")
    parser = module.build_parser()
    args = parser.parse_args(["test question", "--allow-web", "--run-graph"])

    assert args.question == "test question"
    assert args.allow_web is True
    assert args.run_graph is True
    print("check_debug_intent_planner passed")


if __name__ == "__main__":
    main()
