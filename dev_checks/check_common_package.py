from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    text = Path("pyproject.toml").read_text(encoding="utf-8")
    marker = "[tool.hatch.build.targets.wheel]"
    if marker not in text:
        raise AssertionError("missing [tool.hatch.build.targets.wheel] section")
    section = text.split(marker, 1)[1]
    lines = []
    for line in section.splitlines():
        if line.startswith("[") and line.strip() != marker:
            break
        lines.append(line)
    section_text = "\n".join(lines)
    if '"common"' not in section_text:
        raise AssertionError("wheel packages must include common")

    import common.display_utils  # noqa: F401
    import common.title_aliases  # noqa: F401

    print("ok")


if __name__ == "__main__":
    main()
