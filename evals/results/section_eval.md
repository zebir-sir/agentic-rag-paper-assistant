# Section Eval A/B Report

| Metric | Section Search (A) | Hybrid Search (B) |
|---|---:|---:|
| Section Precision@5 | 0.60 | 0.42 |
| Doc Hit@5 | 0.60 | 1.00 |
| Keyword Recall@5 | 0.47 | 0.83 |
| Order OK Rate | 1.00 | 0.00 |

解读：
- Section Search 更适合目标章节过滤与顺序保持（如只看 Abstract/Experiments/References）。
- Hybrid Search 在全局论文命中与关键词覆盖上更强，适合开放式问答。
- 两者适用场景不同，不是单向替代关系。
