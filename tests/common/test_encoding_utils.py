import json

from common.encoding_utils import (
    read_json_robust,
    read_text_utf8,
    repair_mojibake_text,
    write_json_utf8,
    write_text_utf8,
)


def test_write_json_utf8_keeps_chinese_readable(tmp_path):
    path = tmp_path / "sample.json"
    payload = {"question": "你多大了", "note": "联网查一下 RRT* 最新资料"}

    write_json_utf8(path, payload, indent=2)

    raw_text = path.read_text(encoding="utf-8")
    assert "你多大了" in raw_text
    assert "联网查一下 RRT* 最新资料" in raw_text
    assert "\\u4f60" not in raw_text
    assert json.loads(raw_text)["question"] == "你多大了"


def test_read_json_robust_supports_utf8_sig(tmp_path):
    path = tmp_path / "utf8sig.json"
    content = '\ufeff{"question": "你多大了"}'
    path.write_text(content, encoding="utf-8")

    payload = read_json_robust(path)

    assert payload["question"] == "你多大了"


def test_repair_mojibake_text_recovers_common_chinese():
    repaired = repair_mojibake_text("浣犲澶т簡")
    assert "你多大" in repaired or "你多大了" in repaired


def test_repair_mojibake_text_keeps_normal_chinese():
    original = "总结知识库里 Hybrid-RRT 这篇论文的方法流程"
    assert repair_mojibake_text(original) == original


def test_write_text_utf8_writes_chinese_in_utf8(tmp_path):
    path = tmp_path / "note.txt"

    write_text_utf8(path, "你多大了\n")

    assert path.read_bytes().decode("utf-8") == "你多大了\n"
    assert read_text_utf8(path) == "你多大了\n"
