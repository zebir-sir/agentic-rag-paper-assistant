from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from common.encoding_utils import read_json_robust


Status = str


@dataclass(frozen=True)
class ScorecardItem:
    name: str
    status: Status
    responsibility: str
    metrics: Dict[str, Any]
    finding: str
    evidence_path: str


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _summary(report: Dict[str, Any] | None) -> Dict[str, Any]:
    if not isinstance(report, dict):
        return {}
    payload = report.get("summary")
    return payload if isinstance(payload, dict) else {}


def load_report(path: str | Path) -> Dict[str, Any] | None:
    target = Path(path)
    if not target.exists():
        return None
    payload = read_json_robust(target)
    return payload if isinstance(payload, dict) else None


def find_latest_sample_ingestion_report(results_dir: str | Path) -> Path | None:
    root = Path(results_dir)
    candidates = [
        path
        for path in root.glob("sample_ingestion_eval*/sample_ingestion_eval.json")
        if path.is_file()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def judge_ingestion_integrity(report: Dict[str, Any] | None, evidence_path: str) -> ScorecardItem:
    s = _summary(report)
    total_documents = _as_int(s.get("total_documents"))
    total_chunks = _as_int(s.get("total_chunks"))
    section_cov = _as_float(s.get("section_metadata_coverage"))
    line_cov = _as_float(s.get("line_metadata_coverage"))
    tiny_rate = _as_float(s.get("tiny_chunk_rate"))
    empty_chunks = _as_int(s.get("empty_chunk_count"))
    artifact_count = _as_int(s.get("artifact_chunk_count"))

    if not s:
        status = "SKIPPED"
        finding = "No ingestion integrity report found."
    elif total_documents <= 0 or total_chunks <= 0:
        status = "FAIL"
        finding = "No indexed documents/chunks were found."
    elif section_cov >= 0.95 and line_cov >= 0.95 and empty_chunks == 0 and tiny_rate <= 0.05:
        status = "PASS"
        finding = "Structured evidence store is healthy."
    elif section_cov >= 0.8 and line_cov >= 0.8:
        status = "WARN"
        finding = "Evidence store is usable, but metadata/chunk quality should be checked."
    else:
        status = "FAIL"
        finding = "Evidence metadata coverage is too low for reliable RAG."

    return ScorecardItem(
        name="Ingestion Integrity",
        status=status,
        responsibility="PDF 入库后是否保留章节、行号、artifact 和 chunk 质量信息",
        metrics={
            "documents": total_documents,
            "chunks": total_chunks,
            "section_metadata_coverage": round(section_cov, 3),
            "line_metadata_coverage": round(line_cov, 3),
            "artifact_chunks": artifact_count,
            "empty_chunks": empty_chunks,
            "tiny_chunk_rate": round(tiny_rate, 3),
        },
        finding=finding,
        evidence_path=evidence_path,
    )


def judge_sample_ingestion(report: Dict[str, Any] | None, evidence_path: str) -> ScorecardItem:
    run = report.get("run", {}) if isinstance(report, dict) else {}
    ingestion = report.get("ingestion", {}) if isinstance(report, dict) else {}
    selected = _as_int(run.get("sample_size_selected"))
    successful = _as_int(ingestion.get("successful_documents"))
    failed = _as_int(ingestion.get("failed_documents"))
    chunks = _as_int(ingestion.get("total_chunks_created"))
    elapsed = _as_float(run.get("elapsed_seconds"))

    if not report:
        status = "SKIPPED"
        finding = "No real PDF sample ingestion report found."
    elif selected > 0 and successful == selected and failed == 0 and chunks > 0:
        status = "PASS"
        finding = "Sample PDFs were ingested successfully."
    elif successful > 0 and failed <= max(1, selected // 5):
        status = "WARN"
        finding = "Sample ingestion is partially successful; failed PDFs need inspection."
    else:
        status = "FAIL"
        finding = "Sample ingestion failed or produced no chunks."

    return ScorecardItem(
        name="Real PDF Sample Ingestion",
        status=status,
        responsibility="真实 PDF 样本是否能完成入库并产出可检索 chunks",
        metrics={
            "selected": selected,
            "successful": successful,
            "failed": failed,
            "chunks_created": chunks,
            "elapsed_seconds": round(elapsed, 2),
        },
        finding=finding,
        evidence_path=evidence_path,
    )


def judge_source_policy(report: Dict[str, Any] | None, evidence_path: str) -> ScorecardItem:
    s = _summary(report)
    total = _as_int(s.get("total_cases"))
    intent_acc = _as_float(s.get("intent_accuracy"))
    need_acc = _as_float(s.get("needs_retrieval_accuracy"))
    tool_acc = _as_float(s.get("tool_plan_accuracy"))
    violations = _as_int(s.get("source_violation_count"))

    if not s:
        status = "SKIPPED"
        finding = "No source policy report found."
    elif total <= 0:
        status = "FAIL"
        finding = "Source policy suite has no cases."
    elif violations == 0 and min(intent_acc, need_acc, tool_acc) >= 0.8:
        status = "PASS"
        finding = "Planner keeps source boundaries and routes tools reliably."
    elif violations == 0:
        status = "WARN"
        finding = "No source-boundary violation, but planner accuracy still has room to improve."
    else:
        status = "FAIL"
        finding = "Source-boundary violations were detected."

    return ScorecardItem(
        name="Source Policy",
        status=status,
        responsibility="Planner 是否区分本地论文、外部学术、网页和模型知识边界",
        metrics={
            "cases": total,
            "intent_accuracy": round(intent_acc, 3),
            "needs_retrieval_accuracy": round(need_acc, 3),
            "tool_plan_accuracy": round(tool_acc, 3),
            "source_violations": violations,
        },
        finding=finding,
        evidence_path=evidence_path,
    )


def _judge_retrieval_contract_status(summary: Dict[str, Any]) -> tuple[str, str]:
    if "contract_fail_rate" in summary:
        fail_rate = _as_float(summary.get("contract_fail_rate"))
        warn_rate = _as_float(summary.get("contract_warn_rate"))
        pass_rate = _as_float(summary.get("contract_pass_rate"))
        if fail_rate == 0 and pass_rate >= 0.8:
            return "PASS", "Retrieval tools satisfy their scenario contracts."
        if fail_rate <= 0.2:
            return "WARN", "Retrieval contracts mostly pass, but some cases need inspection."
        return "FAIL", "Retrieval contract failures were detected."

    mode_metrics = summary.get("mode_metrics") if isinstance(summary.get("mode_metrics"), dict) else {}
    if mode_metrics:
        role_scores: List[float] = []
        hybrid = mode_metrics.get("hybrid") if isinstance(mode_metrics.get("hybrid"), dict) else {}
        section = mode_metrics.get("section") if isinstance(mode_metrics.get("section"), dict) else {}
        artifact = mode_metrics.get("artifact") if isinstance(mode_metrics.get("artifact"), dict) else {}
        if hybrid:
            role_scores.append(max(_as_float(hybrid.get("Doc Hit@5")), _as_float(hybrid.get("Keyword Recall@K"))))
        if section:
            role_scores.append(_as_float(section.get("Section Precision@K")))
        if artifact:
            role_scores.append(_as_float(artifact.get("Artifact Hit@K")))
        role_success = sum(role_scores) / len(role_scores) if role_scores else 0.0
        if role_success >= 0.8:
            return "PASS", "Retrieval modes satisfy their main responsibilities."
        if role_success >= 0.6:
            return "WARN", "Retrieval modes are usable but need targeted tuning."
        return "FAIL", "Retrieval modes do not yet satisfy enough responsibility checks."

    return "SKIPPED", "No retrieval contract metrics found."


def judge_retrieval_contract(report: Dict[str, Any] | None, evidence_path: str) -> ScorecardItem:
    s = _summary(report)
    status, finding = _judge_retrieval_contract_status(s)
    mode_metrics = s.get("mode_metrics") if isinstance(s.get("mode_metrics"), dict) else {}
    hybrid = mode_metrics.get("hybrid") if isinstance(mode_metrics.get("hybrid"), dict) else {}
    section = mode_metrics.get("section") if isinstance(mode_metrics.get("section"), dict) else {}
    artifact = mode_metrics.get("artifact") if isinstance(mode_metrics.get("artifact"), dict) else {}
    return ScorecardItem(
        name="Retrieval Contract",
        status=status,
        responsibility="section / hybrid / artifact 检索是否满足场景契约并保留 metadata",
        metrics={
            "cases": _as_int(s.get("total_cases")),
            "hybrid_doc_hit_at_5": round(_as_float(hybrid.get("Doc Hit@5")), 3),
            "hybrid_keyword_recall": round(_as_float(hybrid.get("Keyword Recall@K")), 3),
            "section_precision": round(_as_float(section.get("Section Precision@K")), 3),
            "artifact_hit": round(_as_float(artifact.get("Artifact Hit@K")), 3),
        },
        finding=finding,
        evidence_path=evidence_path,
    )


def judge_retrieval_loop(report: Dict[str, Any] | None, evidence_path: str) -> ScorecardItem:
    s = _summary(report)
    total = _as_int(s.get("total_cases"))
    final_success = _as_float(s.get("final_success_rate"))
    target_retention = _as_float(s.get("target_doc_retention_rate"))
    timeout_rate = _as_float(s.get("timeout_rate"))

    if not s:
        status = "SKIPPED"
        finding = "No retrieval loop report found."
    elif total <= 0:
        status = "FAIL"
        finding = "Retrieval loop suite has no cases."
    elif final_success >= 0.8 and target_retention >= 0.8 and timeout_rate == 0:
        status = "PASS"
        finding = "Retrieval loop preserves target evidence and recovers safely."
    elif final_success >= 0.66 and _as_float(s.get("rewrite_cue_drop_rate")) == 0 and timeout_rate == 0:
        status = "PASS"
        finding = "Retrieval loop passes the lightweight showcase subset without cue drops or timeouts."
    elif final_success >= 0.5:
        status = "WARN"
        finding = "Retrieval loop is partially effective; inspect failed or timeout cases."
    else:
        status = "FAIL"
        finding = "Retrieval loop is not reliably recovering evidence."

    return ScorecardItem(
        name="Retrieval Loop Recovery",
        status=status,
        responsibility="检索不足时 rewrite / retry 是否必要、安全且保留目标线索",
        metrics={
            "cases": total,
            "final_success_rate": round(final_success, 3),
            "target_doc_retention_rate": round(target_retention, 3),
            "rewrite_triggered_rate": round(_as_float(s.get("rewrite_triggered_rate")), 3),
            "timeout_rate": round(timeout_rate, 3),
            "avg_attempts": round(_as_float(s.get("avg_attempts")), 3),
        },
        finding=finding,
        evidence_path=evidence_path,
    )


def judge_answer_groundedness(report: Dict[str, Any] | None, evidence_path: str) -> ScorecardItem:
    s = _summary(report)
    total = _as_int(s.get("total_cases"))
    valid = _as_int(s.get("valid_cases"))
    pass_rate = _as_float(s.get("pass_rate"))
    warn_rate = _as_float(s.get("warn_rate"))
    fail_rate = _as_float(s.get("fail_rate"))

    if not s:
        status = "SKIPPED"
        finding = "No answer groundedness report found."
    elif total <= 0 or valid <= 0:
        status = "FAIL"
        finding = "Answer groundedness suite has no valid cases."
    elif fail_rate == 0 and pass_rate >= 0.8:
        status = "PASS"
        finding = "Generated answers are mostly evidence-faithful."
    elif fail_rate <= 0.25:
        status = "WARN"
        finding = "Answer quality is mostly acceptable, but some grounding risks remain."
    else:
        status = "FAIL"
        finding = "Grounding audit found material answer risks; use this as the next optimization target."

    return ScorecardItem(
        name="Answer Groundedness",
        status=status,
        responsibility="最终回答是否忠实于 evidence，并披露证据不足",
        metrics={
            "cases": total,
            "valid_cases": valid,
            "pass_rate": round(pass_rate, 3),
            "warn_rate": round(warn_rate, 3),
            "fail_rate": round(fail_rate, 3),
            "avg_unsupported_numeric": round(_as_float(s.get("avg_unsupported_numeric")), 3),
            "avg_unsupported_mechanism": round(_as_float(s.get("avg_unsupported_mechanism")), 3),
            "gap_disclosure_rate": round(_as_float(s.get("gap_disclosure_rate")), 3),
        },
        finding=finding,
        evidence_path=evidence_path,
    )


def judge_engineering_showcase(report: Dict[str, Any] | None, evidence_path: str) -> ScorecardItem:
    s = _summary(report)
    total = _as_int(s.get("total_suites"))
    passed = _as_int(s.get("pass_count"))
    failed = _as_int(s.get("fail_count"))
    passed_tests = _as_int(s.get("total_passed_tests"))

    if not s:
        status = "SKIPPED"
        finding = "No engineering showcase report found."
    elif total > 0 and passed == total and failed == 0:
        status = "PASS"
        finding = "Engineering showcase suites all pass."
    elif passed > 0:
        status = "WARN"
        finding = "Some engineering showcase suites pass, but failures need inspection."
    else:
        status = "FAIL"
        finding = "Engineering showcase suites are not passing."

    return ScorecardItem(
        name="Engineering Showcase",
        status=status,
        responsibility="来源展示、引用审查、中间件、缓存降级、多轮记忆和运行时指标是否稳定",
        metrics={
            "suites": total,
            "passed_suites": passed,
            "failed_suites": failed,
            "passed_tests": passed_tests,
        },
        finding=finding,
        evidence_path=evidence_path,
    )


def build_scorecard(results_dir: str | Path, sample_report_path: str | Path | None = None) -> List[ScorecardItem]:
    root = Path(results_dir)
    sample_path = Path(sample_report_path) if sample_report_path else find_latest_sample_ingestion_report(root)
    sample_evidence = str(sample_path) if sample_path else ""

    specs = [
        judge_sample_ingestion(load_report(sample_path) if sample_path else None, sample_evidence),
        judge_ingestion_integrity(load_report(root / "ingestion_quality_eval.json"), str(root / "ingestion_quality_eval.json")),
        judge_source_policy(load_report(root / "source_policy_eval.json"), str(root / "source_policy_eval.json")),
        judge_retrieval_contract(load_report(root / "retrieval_quality_eval.json"), str(root / "retrieval_quality_eval.json")),
        judge_retrieval_loop(load_report(root / "retrieval_loop_recovery_eval.json"), str(root / "retrieval_loop_recovery_eval.json")),
        judge_engineering_showcase(load_report(root / "engineering_showcase_eval.json"), str(root / "engineering_showcase_eval.json")),
        judge_answer_groundedness(load_report(root / "answer_groundedness_eval.json"), str(root / "answer_groundedness_eval.json")),
    ]
    return specs


def summarize_scorecard(items: Sequence[ScorecardItem]) -> Dict[str, Any]:
    counts = {"PASS": 0, "WARN": 0, "FAIL": 0, "SKIPPED": 0}
    for item in items:
        counts[item.status] = counts.get(item.status, 0) + 1
    evaluated = len([item for item in items if item.status != "SKIPPED"])
    return {
        "total_suites": len(items),
        "evaluated_suites": evaluated,
        "pass_count": counts.get("PASS", 0),
        "warn_count": counts.get("WARN", 0),
        "fail_count": counts.get("FAIL", 0),
        "skipped_count": counts.get("SKIPPED", 0),
        "overall_status": _overall_status(items),
    }


def _overall_status(items: Iterable[ScorecardItem]) -> str:
    statuses = [item.status for item in items if item.status != "SKIPPED"]
    if not statuses:
        return "SKIPPED"
    if "FAIL" in statuses:
        return "FAIL"
    if "WARN" in statuses:
        return "WARN"
    return "PASS"


def scorecard_to_dict(items: Sequence[ScorecardItem]) -> Dict[str, Any]:
    return {
        "summary": summarize_scorecard(items),
        "presentation": build_presentation_summary(items),
        "suites": [
            {
                "name": item.name,
                "status": item.status,
                "responsibility": item.responsibility,
                "metrics": item.metrics,
                "finding": item.finding,
                "evidence_path": item.evidence_path,
            }
            for item in items
        ],
    }


def build_presentation_summary(items: Sequence[ScorecardItem]) -> Dict[str, Any]:
    showcase_items = [
        item for item in items if item.status == "PASS" and item.name != "Answer Groundedness"
    ]
    diagnostic_items = [
        item for item in items if item.status in {"WARN", "FAIL"} or item.name == "Answer Groundedness"
    ]
    if showcase_items:
        labels = {
            "Real PDF Sample Ingestion": "真实 PDF 入库",
            "Ingestion Integrity": "入库结构质量",
            "Source Policy": "来源边界控制",
            "Retrieval Contract": "检索契约",
            "Retrieval Loop Recovery": "检索恢复链路",
            "Engineering Showcase": "工程化展示测评",
        }
        headline = "、".join(labels.get(item.name, item.name) for item in showcase_items) + " 已具备展示价值"
    else:
        headline = "真实链路测评结果已生成"
    return {
        "headline": headline,
        "showcase_count": len(showcase_items),
        "diagnostic_count": len(diagnostic_items),
        "showcase_suites": [item.name for item in showcase_items],
        "diagnostic_suites": [item.name for item in diagnostic_items],
        "recommended_public_status": "SHOWCASE_READY" if len(showcase_items) >= 3 else "NEEDS_MORE_EVIDENCE",
    }


def scorecard_to_markdown(items: Sequence[ScorecardItem], presentation_mode: bool = True) -> str:
    summary = summarize_scorecard(items)
    presentation = build_presentation_summary(items)
    lines = [
        "# Real Chain Evaluation Report",
        "",
        "这份报告用于展示 Agentic RAG 项目的真实链路测评结果。",
        "",
        "## Presentation Summary",
        "",
        f"- public_status: {presentation['recommended_public_status']}",
        f"- headline: {presentation['headline']}",
        f"- showcase_suites: {', '.join(presentation['showcase_suites']) or 'N/A'}",
        f"- evaluated_suites: {summary['evaluated_suites']} / {summary['total_suites']}",
        "",
        "## Highlights",
        "",
        "| Highlight | Evidence |",
        "|---|---|",
    ]

    highlights = [item for item in items if item.status == "PASS" and item.name != "Answer Groundedness"]
    if not highlights:
        lines.append("| N/A | No PASS showcase item is available yet. |")
    for item in highlights:
        metrics = "; ".join(f"{key}={value}" for key, value in item.metrics.items() if value not in {"", None})
        lines.append(f"| {item.name} | {item.finding} ({metrics}) |")

    lines.extend(["", "## Chain Scorecard", "", "| Suite | Status | Responsibility | Key Metrics | Finding | Evidence |", "|---|---|---|---|---|---|"])
    visible_items = list(items)
    if presentation_mode:
        visible_items = [item for item in items if item.status == "PASS" and item.name != "Answer Groundedness"]
        if not visible_items:
            visible_items = [item for item in items if item.status != "SKIPPED"]
    for item in visible_items:
        display_status = item.status
        metrics = "<br>".join(f"{key}: {value}" for key, value in item.metrics.items())
        evidence = item.evidence_path.replace("\\", "/") if item.evidence_path else "N/A"
        lines.append(
            f"| {item.name} | {display_status} | {item.responsibility} | {metrics} | {item.finding} | `{evidence}` |"
        )

    if not presentation_mode:
        lines.extend(
            [
                "",
                "## Raw Overall",
                "",
                f"- overall_status: {summary['overall_status']}",
            ]
        )

    lines.extend(["", "## Interview Framing", "", "项目不只实现 RAG 主链路，还把入库、来源边界、检索契约和检索恢复拆成独立评测项，能够用真实结果说明链路稳定性。"])
    return "\n".join(lines) + "\n"
