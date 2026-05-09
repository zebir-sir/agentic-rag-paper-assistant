from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.encoding_utils import (
    read_json_robust,
    read_text_utf8,
    repair_mojibake_text,
    write_json_utf8,
    write_text_utf8,
)


DEFAULT_ROOTS = [
    Path("test_logs"),
    Path("evals/results"),
]
TEXT_SUFFIXES = {".txt", ".md"}
JSON_SUFFIXES = {".json"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Repair common mojibake in test_logs and eval result files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print files that would be updated without writing changes.",
    )
    parser.add_argument(
        "--path",
        nargs="*",
        default=None,
        help="Optional file or directory paths to scan. Defaults to test_logs and evals/results.",
    )
    return parser


def _iter_candidate_files(paths: list[str] | None) -> Iterable[Path]:
    targets = [Path(item) for item in paths] if paths else DEFAULT_ROOTS
    seen: set[Path] = set()
    for target in targets:
        if not target.exists():
            continue
        if target.is_file():
            resolved = target.resolve()
            if resolved not in seen:
                seen.add(resolved)
                yield target
            continue
        for path in sorted(target.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in JSON_SUFFIXES | TEXT_SUFFIXES:
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            yield path


def _repair_json_value(value: Any) -> Any:
    if isinstance(value, str):
        return repair_mojibake_text(value)
    if isinstance(value, list):
        return [_repair_json_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _repair_json_value(item) for key, item in value.items()}
    return value


def _looks_suspicious(text: str) -> bool:
    value = str(text or "")
    if not value:
        return False
    markers = ("鑱", "浣", "鏌", "璁", "鍒", "绗", "澶", "鏈", "鏂", "锛", "?", "\ufffd")
    return any(marker in value for marker in markers)


def repair_file(path: Path, *, dry_run: bool) -> tuple[str, str]:
    suffix = path.suffix.lower()
    if suffix in JSON_SUFFIXES:
        original = read_json_robust(path)
        repaired = _repair_json_value(original)
        if repaired == original:
            return ("skipped", "unresolved_mojibake" if _looks_suspicious(str(original)) else "unchanged")
        if dry_run:
            return ("modified", "dry_run")
        write_json_utf8(path, repaired, indent=2)
        return ("modified", "json_repaired")

    if suffix in TEXT_SUFFIXES:
        original_text = read_text_utf8(path)
        repaired_text = repair_mojibake_text(original_text)
        if repaired_text == original_text:
            return ("skipped", "unresolved_mojibake" if _looks_suspicious(original_text) else "unchanged")
        if dry_run:
            return ("modified", "dry_run")
        write_text_utf8(path, repaired_text)
        return ("modified", "text_repaired")

    return ("skipped", "unsupported_suffix")


def main() -> int:
    args = build_parser().parse_args()
    scanned = 0
    modified = 0
    skipped = 0

    for path in _iter_candidate_files(args.path):
        scanned += 1
        try:
            status, reason = repair_file(path, dry_run=args.dry_run)
        except Exception as exc:
            skipped += 1
            print(f"ERROR {path}: {exc}")
            continue

        if status == "modified":
            modified += 1
            action = "WOULD_FIX" if args.dry_run else "FIXED"
            print(f"{action} {path} [{reason}]")
        else:
            skipped += 1
            if reason == "unresolved_mojibake":
                print(f"SKIP {path} [{reason}]")

    print(
        "SUMMARY",
        f"scanned={scanned}",
        f"modified={modified}",
        f"skipped={skipped}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
