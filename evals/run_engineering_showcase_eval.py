from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.encoding_utils import write_json_utf8, write_text_utf8


SUITES: Dict[str, Dict[str, object]] = {
    "source_display_and_citation": {
        "responsibility": "证据来源是否能生成可展示引用，并检查非法/缺失 citation",
        "tests": ["tests/agent/test_evidence_citation_runtime.py"],
        "showcase": "Evidence references + citation review",
    },
    "http_middleware_runtime": {
        "responsibility": "请求 ID、安全头、限流、请求大小限制和结构化错误是否可用",
        "tests": ["tests/agent/test_http_runtime.py"],
        "showcase": "HTTP middleware + runtime metrics",
    },
    "redis_cache_degrade": {
        "responsibility": "Redis 不可用时缓存层是否安全降级，不阻塞主流程",
        "tests": ["tests/agent/test_cache_utils.py", "tests/agent/test_tools_embedding_cache.py"],
        "showcase": "Redis cache hit/miss/degrade",
    },
    "session_memory": {
        "responsibility": "多轮会话摘要、历史过滤和最近消息预览是否稳定",
        "tests": ["tests/agent/test_memory_runtime.py", "tests/agent/test_history_resolver.py", "tests/agent/test_dialog_policy.py"],
        "showcase": "Multi-turn memory/runtime policy",
    },
    "chat_metrics": {
        "responsibility": "聊天请求指标、来源混合、工具调用和最近请求统计是否稳定",
        "tests": ["tests/agent/test_chat_metrics_runtime.py"],
        "showcase": "Chat observability metrics",
    },
    "runtime_config_and_providers": {
        "responsibility": "运行时配置诊断、Embedding 参数适配和 Prompt registry 是否稳定",
        "tests": [
            "tests/agent/test_runtime_config.py",
            "tests/agent/test_providers.py",
            "tests/agent/test_prompt_registry.py",
        ],
        "showcase": "Runtime diagnostics + provider/prompt registry",
    },
    "async_ingestion_contract": {
        "responsibility": "RabbitMQ 任务投递、worker 消费和入库任务错误处理是否稳定",
        "tests": [
            "tests/agent/test_rabbitmq_producer.py",
            "tests/agent/test_ingestion_worker.py",
        ],
        "showcase": "RabbitMQ producer + ingestion worker contract",
    },
}


def _run_pytest(test_paths: List[str], timeout_seconds: int) -> Dict[str, object]:
    cmd = [sys.executable, "-m", "pytest", *test_paths, "-q"]
    started = time.time()
    try:
        completed = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=max(1, timeout_seconds),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "status": "TIMEOUT",
            "runtime_seconds": round(time.time() - started, 3),
            "command": " ".join(cmd),
            "output": str(exc),
        }

    output = (completed.stdout or "") + ("\n" + completed.stderr if completed.stderr else "")
    return {
        "ok": completed.returncode == 0,
        "status": "PASS" if completed.returncode == 0 else "FAIL",
        "runtime_seconds": round(time.time() - started, 3),
        "command": " ".join(cmd),
        "output": output.strip(),
    }


def _extract_passed_count(output: str) -> int:
    import re

    match = re.search(r"(\d+)\s+passed", output or "")
    return int(match.group(1)) if match else 0


def build_report(timeout_seconds: int) -> Dict[str, object]:
    rows: List[Dict[str, object]] = []
    started = time.time()
    for name, spec in SUITES.items():
        result = _run_pytest(list(spec["tests"]), timeout_seconds=timeout_seconds)
        output = str(result.get("output") or "")
        rows.append(
            {
                "suite": name,
                "status": result["status"],
                "showcase": spec["showcase"],
                "responsibility": spec["responsibility"],
                "tests": list(spec["tests"]),
                "passed_tests": _extract_passed_count(output),
                "runtime_seconds": result["runtime_seconds"],
                "command": result["command"],
                "output_tail": output[-1200:],
            }
        )

    total = len(rows)
    passed = sum(1 for row in rows if row["status"] == "PASS")
    return {
        "summary": {
            "total_suites": total,
            "pass_count": passed,
            "fail_count": total - passed,
            "total_passed_tests": sum(int(row["passed_tests"]) for row in rows),
            "runtime_seconds": round(time.time() - started, 3),
            "public_status": "SHOWCASE_READY" if passed == total else "NEEDS_ATTENTION",
        },
        "suites": rows,
    }


def to_markdown(report: Dict[str, object]) -> str:
    summary = dict(report.get("summary") or {})
    suites = list(report.get("suites") or [])
    lines = [
        "# Engineering Showcase Eval",
        "",
        "这份报告用于展示项目的工程化链路：来源展示、请求中间件、缓存降级、多轮记忆和可观测性。",
        "",
        "## Summary",
        "",
        f"- public_status: {summary.get('public_status')}",
        f"- pass_count: {summary.get('pass_count')} / {summary.get('total_suites')}",
        f"- total_passed_tests: {summary.get('total_passed_tests')}",
        f"- runtime_seconds: {summary.get('runtime_seconds')}",
        "",
        "## Scorecard",
        "",
        "| Suite | Status | Showcase | Passed Tests | Responsibility |",
        "|---|---|---|---:|---|",
    ]
    for row in suites:
        lines.append(
            "| {suite} | {status} | {showcase} | {passed_tests} | {responsibility} |".format(
                suite=row.get("suite"),
                status=row.get("status"),
                showcase=row.get("showcase"),
                passed_tests=row.get("passed_tests"),
                responsibility=str(row.get("responsibility") or "").replace("|", "/"),
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run stable engineering showcase eval suites.")
    parser.add_argument("--output-dir", default="evals/results")
    parser.add_argument("--timeout-seconds", type=int, default=120)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(timeout_seconds=int(args.timeout_seconds or 120))
    write_json_utf8(output_dir / "engineering_showcase_eval.json", report, indent=2)
    write_text_utf8(output_dir / "engineering_showcase_eval.md", to_markdown(report))
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
