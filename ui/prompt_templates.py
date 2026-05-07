from typing import Any, List


def _format_single_target(title: str, document_id: str | None = None) -> str:
    lines = [f"目标论文标题：{title}"]
    if document_id:
        lines.append(f"目标文档 ID：{document_id}")
        lines.append("请优先围绕该文档 ID 对应的知识库文档作答；不要仅凭标题猜测目标论文。")
    return "\n".join(lines)


def _normalize_paper_refs(papers: List[Any]) -> List[dict]:
    normalized = []
    for item in papers:
        if isinstance(item, dict):
            title = str(item.get("title") or "").strip()
            document_id = str(item.get("document_id") or "").strip()
        else:
            title = str(item or "").strip()
            document_id = ""
        if title:
            normalized.append({"title": title, "document_id": document_id})
    return normalized


def _format_paper_refs(papers: List[Any]) -> str:
    refs = _normalize_paper_refs(papers)
    lines = []
    for idx, paper in enumerate(refs, start=1):
        if paper["document_id"]:
            lines.append(f"{idx}. 标题：{paper['title']}\n   文档 ID：{paper['document_id']}")
        else:
            lines.append(f"{idx}. 标题：{paper['title']}")
    return "\n".join(lines)


def build_single_summary_prompt(title: str, document_id: str | None = None) -> str:
    target_block = _format_single_target(title, document_id)
    return f"""请基于知识库中的目标论文作答。

{target_block}

请基于知识库中的论文《{title}》做一份严谨的中文阅读总结。
建议按下面结构组织；如果某些部分证据不足，请简短说明“未检索到足够信息”，不要编造：
1. 研究问题
2. 应用场景与假设
3. 核心方法
4. 主要创新点
5. 实验设置与主要结果
6. 局限性
7. 对后续研究或实践应用的启发

要求：优先基于论文依据作答，结论需要能回到知识库片段。"""


def build_single_innovation_prompt(title: str, document_id: str | None = None) -> str:
    target_block = _format_single_target(title, document_id)
    return f"""请基于知识库中的目标论文作答。

{target_block}

请对论文《{title}》做“创新点分析”，使用中文回答。
请重点分析：
1. 它改进了哪类基线方法
2. 创新发生在哪个环节
3. 为什么这个创新可能有效
4. 代价或副作用是什么
5. 这种创新更偏机制创新、启发式增强还是工程增强
6. 创新是否足以支撑论文价值
要求：优先结合论文证据，不要泛泛而谈；证据不足处请明确说明。"""


def build_single_method_prompt(title: str, document_id: str | None = None) -> str:
    target_block = _format_single_target(title, document_id)
    return f"""请基于知识库中的目标论文作答。

{target_block}

请系统拆解论文《{title}》的方法流程，使用中文回答。
建议面向研究生读者解释：
1. 方法主线
2. 输入与输出
3. 核心步骤
4. 关键决策点
5. 各步骤之间如何衔接
6. 每一步在整篇论文中的作用

要求：强调流程逻辑与步骤作用；如果细节证据不足，请说明“论文片段中未明确”。"""


def build_single_experiment_prompt(title: str, document_id: str | None = None) -> str:
    target_block = _format_single_target(title, document_id)
    return f"""请基于知识库中的目标论文作答。

{target_block}

请对论文《{title}》做“实验解读”，使用中文回答。
请重点分析：
1. 实验设置
2. 对比基线
3. 评价指标
4. 主要结果
5. 结果是否真正支撑论文结论
6. 实验是否充分
7. 是否缺少关键消融或对照
要求：对“证据是否充分”给出判断与理由；证据不足处请明确说明。"""


def build_single_limitation_prompt(title: str, document_id: str | None = None) -> str:
    target_block = _format_single_target(title, document_id)
    return f"""请基于知识库中的目标论文作答。

{target_block}

请对论文《{title}》做“局限性分析”，使用中文回答。
请重点分析：
1. 方法假设是否过强
2. 适用范围限制
3. 实验不足
4. 复杂度或实时性问题
5. 未考虑的约束或场景
6. 未来最值得改进的点

要求：指出具体风险，不要只做礼貌性评价；没有依据的点请不要硬写。"""


def build_single_inspiration_prompt(title: str, document_id: str | None = None) -> str:
    target_block = _format_single_target(title, document_id)
    return f"""请基于知识库中的目标论文作答。

{target_block}

请基于论文《{title}》，分析“对我的研究的启发”，使用中文回答。
请从科研选题、方法迁移和实践应用角度展开；若论文属于某个具体领域，请结合该领域背景分析：
1. 哪些思想值得迁移
2. 哪些模块、方法或实验设计可以借鉴
3. 哪些地方不宜直接照搬
4. 对后续研究或项目实践有哪些具体启发
5. 可以从哪些点继续改进或创新
要求：尽量给出可落地的研究切入点；优先基于论文证据，不确定时请说明。"""


def _format_titles(papers: List[Any]) -> str:
    refs = _normalize_paper_refs(papers)
    return "、".join([f"《{p['title']}》" for p in refs])


def build_multi_problem_compare_prompt(titles: List[Any]) -> str:
    papers = _format_titles(titles)
    paper_refs = _format_paper_refs(titles)
    return f"""目标论文：
{paper_refs}

如果存在文档 ID，请优先围绕这些文档 ID 对应的知识库文档作答；不要把其他相似标题论文混入主要分析。

请对以下论文做“核心问题对比”：{papers}。请使用中文回答。
请按对比结构分析：
1. 每篇论文要解决的研究问题
2. 应用场景是否一致
3. 问题设定与假设差异
4. 各自关注的优化目标
5. 本质上是在解决同类问题，还是不同层面的问题
要求：明确列出共性与差异；如果某项证据不足，请简短说明。"""


def build_multi_method_compare_prompt(titles: List[Any]) -> str:
    papers = _format_titles(titles)
    paper_refs = _format_paper_refs(titles)
    return f"""目标论文：
{paper_refs}

如果存在文档 ID，请优先围绕这些文档 ID 对应的知识库文档作答；不要把其他相似标题论文混入主要分析。

请对以下论文做“方法与创新点对比”：{papers}。请使用中文回答。
请重点比较：
1. 各篇论文的方法路线
2. 创新点分别落在哪个环节
3. 是理论假设、模型结构、算法机制、实验设计还是工程实现上的创新
4. 方法本质差异是什么
5. 哪篇更偏理论贡献，哪篇更偏实践可用
要求：结论要有依据，不要只做表面罗列，也不要为了填满模板而编造。"""


def build_multi_experiment_compare_prompt(titles: List[Any]) -> str:
    papers = _format_titles(titles)
    paper_refs = _format_paper_refs(titles)
    return f"""目标论文：
{paper_refs}

如果存在文档 ID，请优先围绕这些文档 ID 对应的知识库文档作答；不要把其他相似标题论文混入主要分析。

请对以下论文做“实验与结果对比”：{papers}。请使用中文回答。
请重点比较：
1. 实验设置
2. baseline 选择
3. 指标是否合理
4. 结果是否充分支撑结论
5. 哪篇实验更扎实
6. 哪篇更有说服力
要求：请给出判断依据；证据不足处请明确说明。"""


def build_multi_value_compare_prompt(titles: List[Any]) -> str:
    papers = _format_titles(titles)
    paper_refs = _format_paper_refs(titles)
    return f"""目标论文：
{paper_refs}

如果存在文档 ID，请优先围绕这些文档 ID 对应的知识库文档作答；不要把其他相似标题论文混入主要分析。

请对以下论文做“适用场景与借鉴价值”对比：{papers}。请使用中文回答。
请重点比较：
1. 各篇论文更适合什么场景
2. 各篇论文的适用条件和限制是什么
3. 哪篇更适合借鉴到后续研究或项目实践
4. 哪些思想、方法或实验设计值得迁移
5. 哪些方法受数据、场景、假设或工程条件限制较大
6. 如果时间有限，哪篇更值得优先精读以及为什么
要求：给出清晰的优先级建议；比较材料不足时请简短说明。"""
