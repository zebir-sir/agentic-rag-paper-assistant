# Evals: 轻量评测与效果验证

本目录提供项目级轻量评测，用于验证检索与回答链路是否按预期工作，并用于展示工程验证能力。它不是大规模 benchmark。

## 文件说明
- `retrieval_cases.json`：普通检索评测样例
- `section_cases.json`：章节级检索 A/B 评测样例
- `retrieval_loop_cases.json`：LangGraph 多轮检索闭环评测样例
- `run_retrieval_eval.py`：普通检索评测脚本
- `run_section_eval.py`：章节级 A/B 评测脚本（Section Search vs Hybrid Search）
- `run_retrieval_loop_eval.py`：LangGraph 多轮检索闭环评测脚本
- `answer_quality_cases.md`：人工回答质量 Rubric 模板
- `results/`：评测输出目录（JSON + Markdown）

## 三类评测目标
- Retrieval Eval：验证普通 hybrid 检索是否命中目标论文与关键内容
- Section Eval A/B：对比 section_search 与 hybrid_search 在章节命中和顺序上的差异
- Retrieval Loop Eval：验证 LangGraph 多轮检索、query rewrite、retrieval confidence 的工作情况

## 指标解释
- `Doc Hit@K`：TopK 是否命中期望论文关键词
- `Section Hit@K`：TopK 是否命中期望章节关键词（retrieval eval）
- `Section Precision@K`：TopK 中命中目标章节的比例（section A/B）
- `Keyword Recall@K`：TopK 内容对期望关键词的覆盖比例
- `Order OK`：结果是否大致按章节顺序返回
- `Rewrite Used Rate`：发生 query rewrite 的比例
- `Avg Retrieval Attempts`：平均检索轮数
- `Avg Retrieval Confidence`：平均检索置信度

## 真实结果摘要

### Retrieval Eval (limit=5)
| Metric | Value |
|---|---:|
| Total Cases | 15 |
| Doc Hit@1 | 0.73 |
| Doc Hit@5 | 1.00 |
| Section Hit@5 | 0.73 |
| Avg Keyword Recall@5 | 0.78 |

### Section Eval A/B (limit=5)
| Metric | Section Search | Hybrid Search |
|---|---:|---:|
| Section Precision@5 | 0.60 | 0.42 |
| Doc Hit@5 | 0.60 | 1.00 |
| Keyword Recall@5 | 0.47 | 0.83 |
| Order OK Rate | 1.00 | 0.00 |

### Retrieval Loop Eval
| Metric | Value |
|---|---:|
| Total Cases | 5 |
| Doc Hit@K | 1.00 |
| Avg Keyword Recall@K | 0.44 |
| Rewrite Used Rate | 0.60 |
| Avg Retrieval Attempts | 1.60 |
| Avg Retrieval Confidence | 0.64 |

## Docker 运行命令
```bash
docker compose exec api python evals/run_retrieval_eval.py --limit 5
docker compose exec api python evals/run_section_eval.py --limit 5
docker compose exec api python evals/run_retrieval_loop_eval.py --max-cases 5 --timeout-seconds 120 --verbose
```

## 注意事项
- 评测集是轻量级项目评测，不是大规模 benchmark。
- 结果依赖当前已入库论文、数据库状态、模型与 embedding 配置。
- 评测分数应结合具体场景理解，不建议跨项目直接横向比较。
