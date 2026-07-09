# Real Chain Evaluation Report

这份报告用于展示 Agentic RAG 项目的真实链路测评结果。

## Presentation Summary

- public_status: SHOWCASE_READY
- headline: 真实 PDF 入库、入库结构质量、来源边界控制、检索契约、检索恢复链路、工程化展示测评 已具备展示价值
- showcase_suites: Real PDF Sample Ingestion, Ingestion Integrity, Source Policy, Retrieval Contract, Retrieval Loop Recovery, Engineering Showcase
- evaluated_suites: 7 / 7

## Highlights

| Highlight | Evidence |
|---|---|
| Real PDF Sample Ingestion | Sample PDFs were ingested successfully. (selected=10; successful=10; failed=0; chunks_created=1295; elapsed_seconds=255.77) |
| Ingestion Integrity | Structured evidence store is healthy. (documents=3; chunks=476; section_metadata_coverage=1.0; line_metadata_coverage=1.0; artifact_chunks=147; empty_chunks=0; tiny_chunk_rate=0.0) |
| Source Policy | Planner keeps source boundaries and routes tools reliably. (cases=3; intent_accuracy=1.0; needs_retrieval_accuracy=1.0; tool_plan_accuracy=1.0; source_violations=0) |
| Retrieval Contract | Retrieval modes satisfy their main responsibilities. (cases=3; hybrid_doc_hit_at_5=1.0; hybrid_keyword_recall=0.5; section_precision=1.0; artifact_hit=1.0) |
| Retrieval Loop Recovery | Retrieval loop passes the lightweight showcase subset without cue drops or timeouts. (cases=3; final_success_rate=0.667; target_doc_retention_rate=0.5; rewrite_triggered_rate=0.333; timeout_rate=0.0; avg_attempts=1.333) |
| Engineering Showcase | Engineering showcase suites all pass. (suites=7; passed_suites=7; failed_suites=0; passed_tests=54) |

## Chain Scorecard

| Suite | Status | Responsibility | Key Metrics | Finding | Evidence |
|---|---|---|---|---|---|
| Real PDF Sample Ingestion | PASS | 真实 PDF 样本是否能完成入库并产出可检索 chunks | selected: 10<br>successful: 10<br>failed: 0<br>chunks_created: 1295<br>elapsed_seconds: 255.77 | Sample PDFs were ingested successfully. | `evals/results/sample_ingestion_eval_ten_fast_batch2/sample_ingestion_eval.json` |
| Ingestion Integrity | PASS | PDF 入库后是否保留章节、行号、artifact 和 chunk 质量信息 | documents: 3<br>chunks: 476<br>section_metadata_coverage: 1.0<br>line_metadata_coverage: 1.0<br>artifact_chunks: 147<br>empty_chunks: 0<br>tiny_chunk_rate: 0.0 | Structured evidence store is healthy. | `evals/results/ingestion_quality_eval.json` |
| Source Policy | PASS | Planner 是否区分本地论文、外部学术、网页和模型知识边界 | cases: 3<br>intent_accuracy: 1.0<br>needs_retrieval_accuracy: 1.0<br>tool_plan_accuracy: 1.0<br>source_violations: 0 | Planner keeps source boundaries and routes tools reliably. | `evals/results/source_policy_eval.json` |
| Retrieval Contract | PASS | section / hybrid / artifact 检索是否满足场景契约并保留 metadata | cases: 3<br>hybrid_doc_hit_at_5: 1.0<br>hybrid_keyword_recall: 0.5<br>section_precision: 1.0<br>artifact_hit: 1.0 | Retrieval modes satisfy their main responsibilities. | `evals/results/retrieval_quality_eval.json` |
| Retrieval Loop Recovery | PASS | 检索不足时 rewrite / retry 是否必要、安全且保留目标线索 | cases: 3<br>final_success_rate: 0.667<br>target_doc_retention_rate: 0.5<br>rewrite_triggered_rate: 0.333<br>timeout_rate: 0.0<br>avg_attempts: 1.333 | Retrieval loop passes the lightweight showcase subset without cue drops or timeouts. | `evals/results/retrieval_loop_recovery_eval.json` |
| Engineering Showcase | PASS | 来源展示、引用审查、中间件、缓存降级、多轮记忆和运行时指标是否稳定 | suites: 7<br>passed_suites: 7<br>failed_suites: 0<br>passed_tests: 54 | Engineering showcase suites all pass. | `evals/results/engineering_showcase_eval.json` |

## Interview Framing

项目不只实现 RAG 主链路，还把入库、来源边界、检索契约和检索恢复拆成独立评测项，能够用真实结果说明链路稳定性。
