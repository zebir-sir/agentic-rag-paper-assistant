# Evaluation Design

Agentic RAG Paper Assistant 使用轻量项目级评测体系验证 Agentic RAG pipeline 中各模块的工程责任。

评测目标不是做一个公开排行榜，也不是把系统压缩成单一分数，而是让入库、规划、检索、重试和回答忠实度都能被检查、回归和定位。

## 1. 评测理念

论文 RAG 系统的失败点通常来自多个层面：

- 入库阶段可能丢失章节结构；
- 检索阶段可能没有命中目标论文、章节或 artifact；
- Planner 可能选择错误来源；
- 外部工具不可用时，系统可能错误地用本地证据替代；
- 最终回答可能包含 evidence 未支撑的数字或机制断言。

因此，本项目将评测拆成多个 suite，分别验证每一层是否履行自己的职责。

## 2. Evaluation Suites

### 2.1 Ingestion Integrity Eval

**目标**  
验证 PDF 入库后是否生成可用的结构化 evidence。

**核心检查项**

- section metadata coverage；
- line metadata coverage；
- artifact chunk count；
- empty chunk count；
- tiny chunk rate；
- chunk size distribution。

**意义**  
如果入库阶段没有保留结构，后续检索和证据追踪都会不可靠。

**运行命令**

```bash
python evals/run_ingestion_quality_eval.py
```

### 2.2 Source Policy Eval

**目标**  
验证 Planner 是否遵守来源边界，并根据问题选择合理工具。

**核心检查项**

- intent accuracy；
- needs retrieval accuracy；
- tool plan accuracy；
- source-boundary violation；
- unavailable-tool filtering。

**意义**  
系统不能把本地论文内容说成是网页检索结果，也不能在 OpenAlex / Web 不可用时伪造外部来源。

**运行命令**

```bash
python evals/run_source_policy_eval.py --limit 10
```

### 2.3 Retrieval Contract Eval

**目标**  
验证检索工具在特定场景下是否满足自己的“契约”。

**典型检查**

- section question 是否返回章节 metadata；
- artifact question 是否返回 table / figure / algorithm chunk；
- known-document 场景是否保留目标文档身份；
- metadata 是否足够支撑 UI 的 evidence display。

**意义**  
这个 eval 不是粗暴比较 retrieval mode 谁更强，而是检查每个检索工具是否完成自己的职责。

**运行命令**

```bash
python evals/run_retrieval_quality_eval.py --limit 10
```

### 2.4 Retrieval Loop Diagnostics

**目标**  
诊断 LangGraph 检索循环在初次检索不足时是否合理触发 rewrite / retry。

**核心诊断项**

- rewrite triggered rate；
- cue preservation ratio；
- target document retention；
- initial retrieval sufficiency；
- rewrite necessity；
- final success。

**意义**  
系统支持 query rewrite 和 retry，但不应该无意义地重写，也不应该在重写时丢失原始问题线索。

**运行命令**

```bash
python evals/run_retrieval_loop_recovery_eval.py --limit 5 --timeout-seconds 120
```

### 2.5 Answer Groundedness Audit

**目标**  
审计最终回答是否忠实于检索证据。

**核心检查项**

- unsupported numeric claims；
- unsupported mechanism claims；
- unsupported external facts；
- source-boundary violations；
- evidence-gap disclosure；
- inference labeling。

**意义**  
RAG 的质量不只取决于检索是否命中，还取决于最终回答有没有越过 evidence 边界。

**运行命令**

```bash
python evals/run_answer_groundedness_eval.py --limit 5 --timeout-seconds 120
```

## 3. README 展示指标

适合放在公开 README 中的稳定指标：

| 指标 | 当前结果 |
|---|---:|
| 测试结果 | 130 passed |
| 入库文档数 | 3 |
| 索引 chunks | 476 |
| Section metadata coverage | 100% |
| Line metadata coverage | 100% |
| Artifact chunks | 147 |
| 当前 Source Policy 评测中的来源越界 | 0 |

这些是轻量工程指标，不是大规模 benchmark 宣称。

## 4. 如何理解结果

| Suite | 解读方式 |
|---|---|
| Ingestion Integrity | evidence store 的结构化健康检查 |
| Source Policy | 来源边界和工具规划的健康检查 |
| Retrieval Contract | 检索行为诊断 |
| Retrieval Loop Diagnostics | rewrite / retry 行为诊断 |
| Answer Groundedness Audit | 回答忠实度质量门 |

README 展示稳定指标和体系设计；详细诊断结果放在 `evals/results/` 中，作为开发和回归参考。

## 5. 一键运行

```bash
python evals/run_all_evals.py --limit 3
```

汇总结果输出到：

```text
evals/results/summary.md
```

## 6. 推荐使用方式

开发过程中可以按修改范围选择 eval：

| 修改范围 | 推荐运行 |
|---|---|
| PDF 解析 / chunking | Ingestion Integrity |
| Planner / 工具路由 | Source Policy |
| Search SQL / metadata | Retrieval Contract |
| LangGraph 检索循环 | Retrieval Loop Diagnostics |
| 生成 prompt / 回答策略 | Answer Groundedness Audit |

公开展示时，README 保持简洁；完整评测设计和运行方式放在本文档中。
