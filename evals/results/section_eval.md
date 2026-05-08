﻿﻿﻿﻿﻿﻿﻿﻿﻿﻿﻿﻿﻿# Section Eval A/B Report

| Metric | Section Search (A) | Hybrid Search (B) |
|---|---:|---:|
| Avg Section Precision@K | 0.600 | 0.420 |
| Doc Hit@K | 0.600 | 1.000 |
| Avg Keyword Recall@K | 0.467 | 0.833 |
| Order OK Rate | 1.000 | 0.000 |

解读：
- Section Search 更偏向“目标章节证据块”的精准过滤与顺序保持。
- Hybrid Search 更偏向全局覆盖与论文命中，不保证章节顺序。
- 两者适用场景不同，不是单向替代关系。

| Case ID | A SecPrec | B SecPrec | A DocHit | B DocHit | A KWRec | B KWRec | A Order | B Order | A Err | B Err |
|---|---:|---:|---:|---:|---:|---:|---|---|---|---|
| section_001 | 1.000 | 0.200 | 1 | 1 | 1.000 | 1.000 | 1 | 0 |  |  |
| section_002 | 0.000 | 0.000 | 0 | 1 | 0.000 | 1.000 | N/A | 0 |  |  |
| section_003 | 1.000 | 0.800 | 1 | 1 | 0.667 | 1.000 | 1 | 0 |  |  |
| section_004 | 0.000 | 0.400 | 0 | 1 | 0.000 | 0.667 | N/A | 0 |  |  |
| section_005 | 0.000 | 0.000 | 0 | 1 | 0.000 | 1.000 | N/A | 0 |  |  |
| section_006 | 1.000 | 0.800 | 1 | 1 | 1.000 | 0.333 | 1 | 0 |  |  |
| section_007 | 1.000 | 0.600 | 1 | 1 | 1.000 | 1.000 | 1 | 0 |  |  |
| section_008 | 1.000 | 0.200 | 1 | 1 | 0.667 | 1.000 | 1 | 0 |  |  |
| section_009 | 0.000 | 0.400 | 0 | 1 | 0.000 | 0.667 | N/A | 0 |  |  |
| section_010 | 1.000 | 0.800 | 1 | 1 | 0.333 | 0.667 | 1 | 0 |  |  |
