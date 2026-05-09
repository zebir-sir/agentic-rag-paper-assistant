# Evaluation Summary

| Suite | Responsibility | Status | Current Finding | README Usage |
|---|---|---|---|---|
| Ingestion Integrity | 验证入库后是否保留章节和 artifact evidence | PASS | 3 docs / 476 chunks / 100% section and line metadata coverage / 147 artifact chunks | 稳定指标 |
| Source Policy | 验证来源边界控制 | PASS | 当前 case 中 0 source-boundary violations | 稳定指标 |
| Retrieval Contract | 检查检索工具是否满足场景契约并保留 metadata | DIAGNOSTIC | 输出 PASS / WARN / FAIL 与 failure reason | 内部诊断 |
| Retrieval Loop Diagnostics | 检查 rewrite 必要性、cue 保留和目标文档保留 | DIAGNOSTIC | 追踪 retry / rewrite 行为是否必要且安全 | 内部诊断 |
| Answer Groundedness Audit | 审计未支撑断言、证据差距和来源边界 | QUALITY GATE | 用于发现回答忠实度风险 | 质量门 |

## README-facing Snapshot

- **130 tests passed**
- **3 documents indexed**
- **476 chunks indexed**
- **100% section metadata coverage**
- **100% line metadata coverage**
- **147 artifact chunks**
- 当前 Source Policy 评测中 **0 source-boundary violations**

## Interpretation

本项目将评测作为工程反馈闭环：

- Ingestion metrics 用来确认本地 evidence store 是否结构化可用；
- Source Policy metrics 用来确认 Planner 是否保持来源边界；
- Retrieval diagnostics 用来调试检索行为，而不是过度包装 benchmark；
- Groundedness audit 作为回答忠实度质量门，用来发现未被 evidence 支撑的断言风险。

详细单项报告保留在本目录中。
