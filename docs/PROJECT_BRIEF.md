# Agentic RAG Paper Assistant — 项目简介

## 1. 项目概览

**Agentic RAG Paper Assistant** 是一个面向科研论文阅读场景的 Agentic RAG 系统。项目重点不是做一个普通聊天机器人，而是围绕长 PDF 论文中的结构化证据、多源检索、来源边界控制和流式深度分析，构建一套可解释、可回归、可展示的论文阅读工作台。

系统支持：

- PDF 论文上传与入库；
- 基于章节结构的 Section-aware Chunking；
- 表格、图示、算法/伪代码等 Artifact Evidence 抽取；
- PostgreSQL + pgvector 本地知识库检索；
- OpenAlex 学术论文检索；
- 可选通用 Web Search；
- LangGraph 深度分析工作流；
- 流式问答、依据片段展示和历史会话管理。

## 2. 项目要解决的问题

普通 RAG 系统在科研论文场景中经常遇到这些问题：

- 固定长度切块容易打断 Method、Experiments、Conclusion 等章节边界；
- 表格、图示、算法块往往承载关键信息，但普通文本检索难以稳定命中；
- 模型容易混淆本地论文证据、外部学术元数据、网页资料和自身通用知识；
- 初次检索失败后缺少可控的重试与改写机制；
- 多轮论文分析容易把调试字段、工具调用信息混入下一轮上下文；
- 回答缺少清晰的 evidence tracing，难以确认结论来自哪里。

本项目围绕这些问题设计了结构化入库、多源规划、检索评估、来源边界和证据追踪能力。

## 3. 核心能力

### 3.1 结构感知的 PDF 入库

PDF 通过 Docling 解析为 Markdown-like 结构文本后，系统会基于标题层级构建章节路径，并在 chunk metadata 中保留：

- `section_title`
- `section_path_text`
- section 行号范围
- section 内分片索引
- content type
- artifact 类型与上下文

相比普通固定长度切块，这种方式更适合论文问答，因为回答可以回到具体章节和证据片段。

### 3.2 Artifact-aware Evidence

系统会将论文中的非正文证据单独抽取为 artifact chunk，包括：

- table
- figure
- algorithm
- pseudo-code

artifact chunk 会保存 caption、上下文、章节路径和行号信息。对于实验结果、流程图、算法步骤类问题，这类证据比普通正文 chunk 更有价值。

### 3.3 Source-aware Intent Planning

Planner 会根据用户问题判断是否需要检索、需要什么来源、可用工具有哪些，以及最终回答应该遵守什么边界。

| Source | 典型用途 |
|---|---|
| Local Knowledge Base | 用户上传论文、本地知识库问答 |
| Section Search | Abstract、Method、Experiments、References 等章节限定问题 |
| Artifact Search | 表格、图示、算法、伪代码相关问题 |
| OpenAlex | related work、作者、年份、DOI、venue、开放获取链接 |
| General Web Search | 可选网页检索能力 |
| Model Knowledge | 不依赖论文证据的通用解释 |

这个设计的核心是：不同来源有不同职责，不能把本地论文证据、联网结果和模型知识混在一起。

### 3.4 LangGraph 深度分析工作流

复杂论文问题会进入 LangGraph workflow，流程包括：

1. 意图规划；
2. 文档与回答范围识别；
3. 本地 / 章节 / artifact / 外部来源检索；
4. 检索质量检查；
5. 必要时 query rewrite / retry；
6. 证据约束下生成回答；
7. 输出 answer + sources。

相比一次性 RAG chain，这种节点化流程更容易调试、评测和扩展。

### 3.5 科研论文分析工作台

Streamlit UI 提供了一个轻量研究工作台：

- 上传论文并入库；
- 单篇论文总结、创新点、方法、实验、局限性分析；
- 多篇论文对比；
- OpenAlex / Web / ReAct 工具开关；
- 流式生成状态展示；
- 依据片段展开；
- 历史会话管理；
- 可取消的流式生成。

## 4. 架构概览

```text
User
  ↓
Streamlit Research Workspace
  ↓
FastAPI Backend
  ↓
Intent Planner + Source Policy
  ↓
LangGraph Agentic RAG Workflow
  ├─ Local Vector / Hybrid Search
  ├─ Section Search
  ├─ Artifact Search
  ├─ OpenAlex Academic Search
  ├─ Optional Web Search
  ├─ Retrieval Quality Check
  └─ Evidence-aware Generation
  ↓
Streaming Answer + Evidence Sources
```

## 5. 技术栈

| 层级 | 技术 |
|---|---|
| Frontend | Streamlit |
| Backend | FastAPI, SSE |
| Agent Workflow | LangChain, LangGraph |
| Optional Agent Runtime | Pydantic AI |
| Vector Store | PostgreSQL + pgvector |
| Text Search | PostgreSQL hybrid search / pg_trgm 思路 |
| PDF Parsing | Docling |
| Embedding / LLM | OpenAI-compatible API |
| Academic Search | OpenAlex |
| Deployment | Docker Compose |
| Testing | pytest, pytest-asyncio, pytest-mock |

## 6. 工程快照

当前项目级工程快照：

| 指标 | 当前结果 |
|---|---:|
| 测试结果 | 130 tests passed |
| 入库论文 | 3 documents |
| 索引分块 | 476 chunks |
| 章节 metadata 覆盖率 | 100% |
| 行号 metadata 覆盖率 | 100% |
| Artifact chunks | 147 |
| Source Policy 当前评测 | 0 source-boundary violations |

这些指标用于展示项目的工程质量，不作为大规模公开 benchmark 宣称。

## 7. 评测体系概览

项目采用轻量工程评测体系，用来验证 Agentic RAG pipeline 中各模块是否履行自己的职责。

| Suite | 验证责任 | 展示口径 |
|---|---|---|
| Ingestion Integrity | 入库后是否保留章节、行号、artifact 证据 | 稳定指标 |
| Source Policy | Planner 是否遵守来源边界 | 稳定指标 |
| Retrieval Contract | 检索工具是否满足场景契约并保留 metadata | 诊断指标 |
| Retrieval Loop Diagnostics | 检索重试、改写、cue 保留是否合理 | 诊断指标 |
| Answer Groundedness Audit | 审计回答是否存在未支撑断言 | 质量门 |

评测设计不是为了把系统压缩成一个分数，而是为了让每一层工程责任可检查、可回归、可定位。

## 8. 适用场景

- 单篇论文总结；
- 方法流程拆解；
- 实验结果解读；
- 表格 / 图示 / 算法证据分析；
- 多篇论文方法对比；
- related work 检索；
- 私有论文知识库构建；
- 多轮科研阅读与讨论。

## 9. 项目定位

这个项目最适合被理解为一个 **面向科研论文阅读的 Agentic RAG 工程系统**。它的价值不在于“接了一个向量库”，而在于围绕论文场景做了：

- 结构化入库；
- artifact evidence 抽取；
- source-aware planning；
- LangGraph 深度分析；
- evidence tracing；
- 流式研究工作台；
- 工程化评测与测试保障。
