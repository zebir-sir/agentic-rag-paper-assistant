# Source Policy Eval

- total_cases: 9
- intent_accuracy: 0.889
- needs_retrieval_accuracy: 0.889
- tool_plan_accuracy: 0.889
- no_retrieval_accuracy: 0.889
- boundary_disclosure_expected_count: 3
- source_violation_count: 0
- unavailable_tool_filtered_count: 0

| Case | Intent | Needs Retrieval | Tools | Intent OK | Need OK | Tool OK | Violation |
|---|---|---:|---|---:|---:|---:|---:|
| source_001 | direct_answer | 0 |  | 1 | 1 | 1 | 0 |
| source_002 | direct_answer | 0 |  | 1 | 1 | 1 | 0 |
| source_003 | local_paper_qa | 1 | hybrid_search | 1 | 1 | 1 | 0 |
| source_004 | local_paper_qa | 1 | hybrid_search | 1 | 1 | 0 | 0 |
| source_005 | local_paper_qa | 1 | hybrid_search | 1 | 1 | 1 | 0 |
| source_006 | direct_answer | 0 |  | 1 | 1 | 1 | 0 |
| source_007 | local_paper_qa | 1 | hybrid_search | 1 | 1 | 1 | 0 |
| source_008 | local_paper_qa | 1 | hybrid_search | 0 | 0 | 1 | 0 |
| source_009 | direct_answer | 0 |  | 1 | 1 | 1 | 0 |
