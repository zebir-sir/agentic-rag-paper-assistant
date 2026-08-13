"""Derive stable paper titles from parsed PDF content."""

import re
from pathlib import Path


_STAGING_PREFIX = re.compile(r"^[0-9a-f]{8}_", re.IGNORECASE)
_HEADING = re.compile(r"^#{1,2}\s+(.+?)\s*$")
_SECTION = re.compile(
    r"^(abstract|introduction|keywords?|authors?|references|contents?)\b",
    re.IGNORECASE,
)
_AUTHOR_MARKERS = re.compile(r"(?:@|\b(university|institute|department|laboratory|email)\b)", re.IGNORECASE)


def filename_title(file_path: str) -> str:
    """Return a readable file-derived fallback without upload staging prefixes."""
    stem = Path(file_path).stem.strip()
    stem = _STAGING_PREFIX.sub("", stem)
    return re.sub(r"[_\s]+", " ", stem).strip() or "Untitled paper"


def _clean_candidate(value: str) -> str:
    value = re.sub(r"^#{1,6}\s*", "", str(value or "")).strip()
    value = re.sub(r"\s+", " ", value)
    return value.strip(" -_\t")


def _looks_like_title(value: str) -> bool:
    if not 12 <= len(value) <= 360:
        return False
    if _SECTION.match(value) or _AUTHOR_MARKERS.search(value):
        return False
    readable = re.findall(r"[A-Za-z0-9\u4e00-\u9fff]", value)
    if len(readable) / len(value) < 0.6:
        return False
    words = re.findall(r"[A-Za-z][A-Za-z0-9*+/-]*", value)
    return len(words) >= 3 and len(words) <= 40


def extract_document_title(content: str, file_path: str) -> str:
    """Prefer the paper's first Markdown title, then a credible first-page line."""
    fallback = filename_title(file_path)
    lines = [_clean_candidate(line) for line in str(content or "").splitlines()[:80]]
    for line in lines:
        heading = _HEADING.match(line)
        if heading:
            candidate = _clean_candidate(heading.group(1))
            if _looks_like_title(candidate):
                return candidate

    for index, line in enumerate(lines[:30]):
        if not _looks_like_title(line):
            continue
        next_line = lines[index + 1] if index + 1 < len(lines) else ""
        if _looks_like_title(next_line) and not _AUTHOR_MARKERS.search(next_line):
            combined = f"{line} {next_line}"
            if _looks_like_title(combined):
                return combined
        return line
    return fallback
