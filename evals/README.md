# Evals

轻量评测模块，用于验证本项目的三条核心链路：
- 普通检索效果（Retrieval Eval）
- 章节级检索效果（Section Eval A/B）
- LangGraph 多轮检索闭环（Retrieval Loop Eval）

说明：这是项目级验证，不是大规模 benchmark。

## 文件说明
- `retrieval_cases.json`：普通检索样例
- `section_cases.json`：章节检索 A/B 样例
- `retrieval_loop_cases.json`：多轮检索闭环样例
- `run_retrieval_eval.py`：普通检索评测脚本
- `run_section_eval.py`：章节检索 A/B 脚本
- `run_retrieval_loop_eval.py`：多轮检索闭环脚本
- `answer_quality_cases.md`：人工评分模板
- `results/`：评测输出（JSON + Markdown）

## 运行命令
```bash
docker compose exec api python evals/run_retrieval_eval.py --limit 5
docker compose exec api python evals/run_section_eval.py --limit 5
docker compose exec api python evals/run_retrieval_loop_eval.py --max-cases 5 --timeout-seconds 120 --verbose
```

## 指标（简要）
- `Doc Hit@K`：是否命中目标论文
- `Section Hit@K` / `Section Precision@K`：章节命中效果
- `Keyword Recall@K`：关键词覆盖度
- `Order OK`：章节顺序保持
- `Rewrite Used Rate`：闭环中 query rewrite 使用比例
- `Avg Retrieval Attempts`：平均检索轮数

## 当前结果摘要

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

## 结果解读（简要）
- `section_search` 更适合章节限定问题（如只看 Abstract/Experiments/References）。
- `hybrid_search` 在全局命中与关键词覆盖上更强，适合开放式问答。
- 多轮检索闭环会在部分挑战问题中触发 rewrite，提升覆盖能力。
- 两类检索策略是互补关系，不是单向替代。

## 结果文件
- `evals/results/retrieval_eval.json`
- `evals/results/retrieval_eval.md`
- `evals/results/section_eval.json`
- `evals/results/section_eval.md`
- `evals/results/retrieval_loop_eval.json`
- `evals/results/retrieval_loop_eval.md`

## 注意事项
- 结果依赖当前数据库中已入库论文。
- 部分评测依赖模型与 embedding 配置。
- Web/OpenAlex 能力受外部配置影响，不可用时会影响对应评测链路。
