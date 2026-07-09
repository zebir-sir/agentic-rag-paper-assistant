# Retrieval Loop Diagnostics

- 目标：诊断检索循环（重写/重试）是否有效保留了原始意图，并观察其对召回的影响。

- total_cases: 3
- rewrite_triggered_rate: 0.333
- avg_cue_preservation_ratio: 1.000
- target_doc_retention_rate: 0.500
- initial_retrieval_sufficient_rate: 0.667
- rewrite_necessary_rate: 0.333
- rewrite_unnecessary_triggered_rate: 0.333
- rewrite_cue_drop_rate: 0.000
- final_success_rate: 0.667
- avg_attempts: 1.333

## Diagnostic Details

| ID | Init Suff. | Rewrite Need | Rewrite | Cue Drop | Cue Pres. | Init Hit | Final Hit | Retained | Attempts | Failure Reason |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| loop_001 | 1 | 0 | 0 | 0 | 0.000 | 1 | 1 | 1 | 1 | None |
| loop_002 | 1 | 0 | 1 | 0 | 1.000 | 1 | 0 | 0 | 2 | No target document hit after retrieval loop |
| loop_003 | 0 | 1 | 0 | 0 | 0.000 | 0 | 1 | 0 | 1 | None |
