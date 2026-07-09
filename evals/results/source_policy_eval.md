# Source Policy Eval

- total_cases: 3
- intent_accuracy: 1.000
- needs_retrieval_accuracy: 1.000
- tool_plan_accuracy: 1.000
- no_retrieval_accuracy: 1.000
- boundary_disclosure_expected_count: 0
- source_violation_count: 0
- unavailable_tool_filtered_count: 0

| Case | Intent | Needs Retrieval | Tools | Intent OK | Need OK | Tool OK | Violation |
|---|---|---:|---|---:|---:|---:|---:|
| source_001 | direct_answer | 0 |  | 1 | 1 | 1 | 0 |
| source_002 | direct_answer | 0 |  | 1 | 1 | 1 | 0 |
| source_003 | local_paper_qa | 1 | hybrid_search | 1 | 1 | 1 | 0 |
