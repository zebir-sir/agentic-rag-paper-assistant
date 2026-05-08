# Design Notes

## 1. 项目定位
本项目不是简单聊天机器人，而是面向复杂文档场景的 Agentic RAG 系统。核心关注点是：
- 多源检索
- 检索不足时的自我修正
- 章节级证据定位
- 可评测的 RAG 工程链路

## 2. Source Selection Policy
不同来源的职责边界如下：
- Local Knowledge Base：用户上传 PDF 论文的本地全文证据
- Section Search：用户要求特定章节（如 Abstract/Method/Experiments/References）时使用
- OpenAlex：related work、作者、年份、DOI、venue、开放获取链接等学术元数据检索
- General Web Search：普通网页资料、最新信息、非论文来源问题

约束：不能把本地论文中提取的 References 片段冒充为联网实时结果；如使用本地 References，应明确标注来源。

## 3. Why LangGraph
相比简单链式 RAG，本项目采用 LangGraph 的原因是：
- 复杂问题下，初次检索可能不足
- 需要在流程中显式支持 retrieval evaluation / query rewrite / retry retrieval
- 工作流节点化后更可控，也更方便调试与评测

## 4. Why Section-aware Chunking
固定长度切块在论文场景常见问题：
- 章节边界被打断
- 方法、实验、结论内容容易混杂
- evidence 难以稳定定位

当前实现做法：
- 基于 Docling 产出的 Markdown heading 识别 section
- 记录 `section_title` / `section_path_text` / `section_chunk_index` 等 metadata
- 在 chunk content 注入 `[Section: ...]` 前缀，提升检索可见性

## 5. Retrieval Strategy
- vector search：更偏语义召回
- hybrid search：兼顾语义与关键词
- section_search：适合“只看 Abstract / Experiments / References”等章节限定问题
- artifact_search：补充表格、图、算法/伪代码等非正文证据（table/figure/algorithm）
- OpenAlex / Web Search：用于本地语料之外的外部扩展检索

### Capability-aware Intent Planner

- Planner 负责“是否检索、检索什么、检索多少”的轻量规划，不生成最终答案。
- Planner 只在可用工具集合内规划（capability-aware）：不可用工具会在 normalize/runtime 两层被过滤。
- 每轮最多 2 个检索步骤，优先最小必要检索。
- 如果外部工具未启用，Planner 不会执行 OpenAlex/Web；系统会回退到可用本地检索或直答。
- 该机制是通用决策框架，不依赖特定论文、图号、表号或算法名规则。

## 6. Evaluation
当前轻量评测结果（真实运行结果）：
- Retrieval Eval: `Doc Hit@5 = 1.00`, `Section Hit@5 = 0.73`, `Avg Keyword Recall@5 = 0.78`
- Section Eval A/B: `Section Search Precision@5 = 0.60`, `Hybrid Search = 0.42`, `Order OK Rate = 1.00 vs 0.00`
- Retrieval Loop Eval: `Rewrite Used Rate = 0.60`, `Avg Retrieval Attempts = 1.60`

解释：
- section_search 并不全面替代 hybrid_search
- 两者适用场景不同：章节定位 vs 全局覆盖
- 该 eval 是轻量项目评测，不是大规模 benchmark

## 7. Current Limitations & Mitigations

| Limitation | Impact | Mitigation / Future Work |
|---|---|---|
| PDF parsing depends on Docling output quality | Complex multi-column layouts, formulas or scanned PDFs may affect section detection and chunk quality. | Add page-level evidence mapping and consider LayoutParser / OCR fallback for difficult PDFs. |
| Deep analysis has timeout protection | Very broad multi-paper questions may timeout before completing all retrieval / rewrite / evidence-check steps. | Add dynamic retry budgeting and guide users to narrow scope by paper, section or question type. |
| Ingestion job state is stored in API process memory | Running ingestion jobs may lose state after service restart. | Move ingestion jobs to Redis / Celery or another persistent task queue. |
| Section search relies on section metadata | If the PDF lacks clear headings, section-level retrieval falls back to normal chunk retrieval quality. | Improve heading normalization and add manual section correction / page-level fallback. |
| General Web Search depends on external provider configuration | If API key/provider is not configured, web search is unavailable. | Expose tool availability in UI and fall back to local KB / OpenAlex when appropriate. |

## 8. Future Work
- page-level evidence（页级证据定位）
- Redis/Celery 持久任务队列
- 更大规模评测集
- API routes/services 拆分
- 多用户认证与数据隔离
