# Evals

本目录包含 **Agentic RAG Paper Assistant** 的轻量工程评测，用于验证 Agentic RAG pipeline 中的关键责任：入库质量、来源边界、检索契约、检索循环和回答忠实度。

## Evaluation Suites

| Suite | Script | 目标 | README 定位 |
|---|---|---|---|
| Ingestion Integrity | `run_ingestion_quality_eval.py` | 检查 PDF 入库后的章节 metadata、行号 metadata、artifact chunks 和 chunk 质量 | 稳定指标 |
| Source Policy | `run_source_policy_eval.py` | 检查 Planner 意图识别、来源路由、不可用工具过滤和来源越界 | 稳定指标 |
| Retrieval Contract | `run_retrieval_quality_eval.py` | 检查检索工具是否满足特定场景契约，并保留必要 metadata | 诊断指标 |
| Retrieval Loop Diagnostics | `run_retrieval_loop_recovery_eval.py` | 检查 rewrite / retry、cue 保留和目标文档保留情况 | 诊断指标 |
| Answer Groundedness Audit | `run_answer_groundedness_eval.py` | 审计未支撑断言、证据差距披露和回答忠实度 | 质量门 |

## 快速运行

```bash
python evals/run_all_evals.py --limit 3
```

单独运行：

```bash
python evals/run_ingestion_quality_eval.py
python evals/run_source_policy_eval.py --limit 10
python evals/run_retrieval_quality_eval.py --limit 10
python evals/run_retrieval_loop_recovery_eval.py --limit 5 --timeout-seconds 120
python evals/run_answer_groundedness_eval.py --limit 5 --timeout-seconds 120
```

## 适合 README 展示的稳定指标

- 130 tests passed；
- 3 documents / 476 chunks indexed；
- 100% section metadata coverage；
- 100% line metadata coverage；
- 147 artifact chunks；
- 当前 Source Policy eval 中 0 source-boundary violations。

## Public Metrics vs. Internal Diagnostics

不是所有评测结果都适合直接作为公开展示指标。

- **Stable metrics**：适合放在 README 中展示工程质量；
- **Diagnostics**：用于开发时检查检索或 workflow 行为；
- **Quality gate**：用于发现回答忠实度风险，指导后续优化。

这样既能让 README 保持干净，也能保留详细评测结果的可追溯性。

## 输出文件

典型输出：

```text
evals/results/summary.md
evals/results/ingestion_quality_eval.md
evals/results/source_policy_eval.md
evals/results/retrieval_quality_eval.md
evals/results/retrieval_loop_recovery_eval.md
evals/results/answer_groundedness_eval.md
```

公开仓库中建议保留 `summary.md` 和稳定/诊断类 `.md` 摘要；过细的 JSON 结果可作为本地内部记录，不必公开展示。

## 推荐开发流程

| 修改内容 | 建议运行 |
|---|---|
| PDF parsing / chunking | Ingestion Integrity |
| Planner / source routing | Source Policy |
| Search SQL / metadata | Retrieval Contract |
| LangGraph retrieval loop | Retrieval Loop Diagnostics |
| Generation prompt | Answer Groundedness Audit |
