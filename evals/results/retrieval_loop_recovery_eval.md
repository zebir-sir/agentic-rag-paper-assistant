# Retrieval Loop Diagnostics

- 目标：诊断检索循环（重写/重试）是否有效保留了原始意图，并观察其对召回的影响。

- total_cases: 3
- rewrite_triggered_rate: 0.000
- avg_cue_preservation_ratio: 0.000
- target_doc_retention_rate: 1.000
- initial_retrieval_sufficient_rate: 1.000
- rewrite_necessary_rate: 0.000
- rewrite_unnecessary_triggered_rate: 0.000
- rewrite_cue_drop_rate: 0.000
- final_success_rate: 1.000
- avg_attempts: 1.000

## Diagnostic Details

| ID | Init Suff. | Rewrite Need | Rewrite | Cue Drop | Cue Pres. | Init Hit | Final Hit | Retained | Attempts | Failure Reason |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| loop_001 | 1 | 0 | 0 | 0 | 0.000 | 1 | 1 | 1 | 1 | None |
| loop_002 | 1 | 0 | 0 | 0 | 0.000 | 1 | 1 | 1 | 1 | None |
| loop_003 | 1 | 0 | 0 | 0 | 0.000 | 1 | 1 | 1 | 1 | None |
