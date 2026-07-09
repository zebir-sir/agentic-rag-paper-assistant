# Retrieval Quality Eval v2

- 说明: section_search 与 hybrid_search/artifact_search 是互补关系，不是单向替代。

- total_cases: 3

## Mode Summary

| Mode | Doc@1 | Doc@5 | SecPrec@K | ArtHit@K | KWRecall@K | OrderOK | Latency(ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| artifact | 1.000 | 1.000 | 0.400 | 1.000 | 0.500 | 0.000 | 86.8 |
| section | 0.000 | 0.000 | 1.000 | 0.000 | 0.556 | 0.667 | 73.4 |
| hybrid | 0.333 | 1.000 | 0.067 | 1.000 | 0.500 | 0.000 | 810.3 |
