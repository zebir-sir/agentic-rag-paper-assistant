import html
from typing import Any, Callable, Dict, List

import streamlit as st

try:
    from ui.display_utils import clean_snippet_text, make_friendly_title, localize_paper_terms
    from ui.prompt_templates import (
        build_multi_experiment_compare_prompt,
        build_multi_method_compare_prompt,
        build_multi_problem_compare_prompt,
        build_multi_value_compare_prompt,
        build_single_experiment_prompt,
        build_single_innovation_prompt,
        build_single_inspiration_prompt,
        build_single_limitation_prompt,
        build_single_method_prompt,
        build_single_summary_prompt,
    )
    from ui.title_aliases import get_title_alias
except ImportError:  # pragma: no cover - streamlit script mode
    from display_utils import clean_snippet_text, make_friendly_title, localize_paper_terms
    from prompt_templates import (
        build_multi_experiment_compare_prompt,
        build_multi_method_compare_prompt,
        build_multi_problem_compare_prompt,
        build_multi_value_compare_prompt,
        build_single_experiment_prompt,
        build_single_innovation_prompt,
        build_single_inspiration_prompt,
        build_single_limitation_prompt,
        build_single_method_prompt,
        build_single_summary_prompt,
    )
    from title_aliases import get_title_alias


def inject_styles() -> None:
    st.markdown(
        """
<style>
[data-testid="stAppViewContainer"] {
    background: #f3f6fb;
    color: #334155;
}
[data-testid="stHeader"] {
    background: rgba(243, 246, 251, 0.9);
}
.main .block-container {
    max-width: 1040px;
    margin-left: auto;
    margin-right: auto;
    padding-top: 1.35rem;
    padding-bottom: 1.1rem;
}
[data-testid="stChatInput"] {
    max-width: 1040px;
    margin-left: auto;
    margin-right: auto;
    background: #ffffff;
    border: 1px solid #d8e0ec;
    border-radius: 14px;
    box-shadow: 0 8px 20px rgba(15, 23, 42, 0.04);
    padding: 8px 10px;
}
[data-testid="stChatInput"]::before {
    display: none;
}
[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] form,
[data-testid="stChatInput"] [data-baseweb="textarea"],
[data-testid="stChatInput"] [data-baseweb="textarea"] > div {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
[data-testid="stChatInput"] textarea {
    background: transparent !important;
    border: none !important;
    border-radius: 0 !important;
    box-shadow: none !important;
    outline: none !important;
    color: #334155 !important;
    padding: 0.3rem 0 !important;
}
[data-testid="stChatInput"] textarea:focus,
[data-testid="stChatInput"] textarea:focus-visible {
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
}
[data-testid="stChatInput"] button {
    box-shadow: none !important;
    border-radius: 10px !important;
}
[data-testid="stChatInput"] button:not(:disabled) {
    background: #ef4444 !important;
    color: #ffffff !important;
    border: none !important;
}
[data-testid="stChatInput"] button:not(:disabled):hover {
    background: #dc2626 !important;
    color: #ffffff !important;
    border: none !important;
}
[data-testid="stChatInput"] button:disabled {
    background: #e2e8f0 !important;
    color: #94a3b8 !important;
    border: none !important;
    opacity: 1 !important;
}
[data-testid="stChatInput"] button svg {
    fill: currentColor !important;
}
.stMarkdown, .stText, p, li, label {
    color: #334155;
}
h1, h2, h3, h4, h5, h6 {
    color: #0f172a;
}
[data-testid="stSidebar"] {
    background: #eef2f7;
    border-right: 1px solid #dbe3ee;
}
[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
    background: #eef2f7;
}
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] .stText,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #334155;
}
.stChatMessage {
    max-width: 100%;
    padding: 0.2rem 0;
}
.stAlert {
    max-width: 100%;
    border: 1px solid #e5eaf2;
    border-radius: 14px;
    background: rgba(255, 255, 255, 0.88);
}
.stChatMessage [data-testid="stMarkdownContainer"] > p:first-child {
    margin-top: 0.1rem;
}
.stChatMessage {
    border-radius: 12px;
}
[data-testid="stChatMessageContent"] {
    border: 1px solid #e5eaf2;
    border-radius: 12px;
    padding: 0.9rem 1rem 0.95rem 1rem;
    background: #f8fbff;
    box-shadow: 0 4px 14px rgba(15, 23, 42, 0.04);
}
[data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) [data-testid="stChatMessageContent"] {
    background: #f1f5f9;
    border-color: #dbe4f0;
}
.workspace-hero {
    border: 1px solid #e5eaf2;
    border-radius: 16px;
    padding: 1.5rem 1.75rem 1.3rem 1.75rem;
    background: #ffffff;
    box-shadow: 0 12px 28px rgba(15, 23, 42, 0.05);
    margin-bottom: 1rem;
}
.workspace-hero h1 {
    font-size: 2rem;
    line-height: 1.2;
    margin: 0 0 0.4rem 0;
    color: #0f172a;
}
.workspace-hero p {
    color: #64748b;
    margin: 0;
}
.workspace-hero-examples {
    margin-top: 0.85rem;
    color: #64748b;
    font-size: 0.95rem;
    line-height: 1.65;
}
.workspace-toolbar-label {
    font-size: 0.78rem;
    letter-spacing: 0.02em;
    color: #64748b;
    margin-bottom: 0.2rem;
}
.workspace-toolbar-anchor + div {
    border-top: 1px solid #dbe3ee;
    background: transparent;
    border-radius: 0;
    box-shadow: none;
    padding: 0.5rem 0 0.2rem 0;
    margin-top: 0.15rem;
}
.workspace-toolbar-anchor + div [data-testid="stVerticalBlock"] {
    gap: 0.35rem;
}
.workspace-chat-stage {
    border: 1px solid #e5eaf2;
    border-radius: 18px;
    padding: 0.7rem 0.9rem 0.9rem 0.9rem;
    background: rgba(255, 255, 255, 0.72);
    box-shadow: 0 10px 28px rgba(15, 23, 42, 0.04);
    margin-bottom: 0.9rem;
}
[data-testid="stSidebar"] [data-testid="stButton"] button[kind="secondary"] {
    border-color: #d8e2f0;
    background: #ffffff;
}
[data-testid="stSidebar"] .history-list-anchor + div [data-testid="stVerticalBlock"] {
    gap: 0.12rem;
}
[data-testid="stSidebar"] .history-item-anchor + div {
    margin-top: 0 !important;
}
[data-testid="stSidebar"] .history-title-anchor + div [data-testid="stButton"] button,
[data-testid="stSidebar"] .history-delete-anchor + div [data-testid="stButton"] button {
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
    outline: none !important;
    min-height: 36px !important;
    border-radius: 10px !important;
}
[data-testid="stSidebar"] .history-title-anchor + div [data-testid="stButton"] button {
    padding: 6px 8px !important;
    justify-content: flex-start !important;
    text-align: left !important;
    width: 100%;
    color: #1f2937 !important;
    font-weight: 500;
    font-size: 13px !important;
}
[data-testid="stSidebar"] .history-title-anchor.active + div [data-testid="stButton"] button {
    background: #eaf1ff !important;
    box-shadow: inset 2px 0 0 #4f7cff !important;
}
[data-testid="stSidebar"] .history-title-anchor + div [data-testid="stButton"] button:hover {
    background: #f5f8fc !important;
    color: #111827 !important;
}
[data-testid="stSidebar"] .history-title-anchor + div [data-testid="stButton"] button p {
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    width: 100%;
    margin: 0;
}
[data-testid="stSidebar"] .history-delete-anchor + div [data-testid="stButton"] {
    display: flex;
    justify-content: flex-end;
}
[data-testid="stSidebar"] .history-delete-anchor + div [data-testid="stButton"] button {
    width: 28px !important;
    min-width: 28px !important;
    height: 28px !important;
    min-height: 28px !important;
    padding: 0 !important;
    margin-top: 4px;
    color: #94a3b8 !important;
    font-size: 1rem;
    opacity: 0.28;
}
[data-testid="stSidebar"] .history-item-anchor + div:hover .history-delete-anchor + div [data-testid="stButton"] button,
[data-testid="stSidebar"] .history-delete-anchor + div [data-testid="stButton"] button:focus,
[data-testid="stSidebar"] .history-delete-anchor + div [data-testid="stButton"] button:hover {
    opacity: 1;
}
[data-testid="stSidebar"] .history-delete-anchor + div [data-testid="stButton"] button:hover {
    background: #e5e7eb !important;
    color: #ef4444 !important;
}
[data-testid="stSidebar"] .history-confirm-anchor + div {
    margin: -0.1rem 0 0.35rem 0;
}
[data-testid="stSidebar"] .history-confirm-anchor + div [data-testid="stMarkdownContainer"] p {
    font-size: 12px;
    color: #64748b;
}
[data-testid="stSidebar"] .history-confirm-anchor + div [data-testid="stButton"] button {
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
    min-height: 28px !important;
    height: 28px !important;
    padding: 0 6px !important;
    border-radius: 8px !important;
    font-size: 12px !important;
}
[data-testid="stSidebar"] .history-confirm-anchor + div [data-testid="stButton"] button:hover {
    background: #eef2f7 !important;
}
[data-testid="stSidebar"] .history-confirm-anchor + div [data-testid="stButton"]:first-of-type button {
    color: #334155 !important;
}
[data-testid="stSidebar"] .history-confirm-anchor + div [data-testid="stButton"]:nth-of-type(2) button {
    color: #64748b !important;
}
.workspace-chat-stage .stChatMessage {
    margin-bottom: 0.35rem;
}
.stChatMessage h1 {font-size: 1.45rem !important; line-height: 1.35 !important;}
.stChatMessage h2 {font-size: 1.25rem !important; line-height: 1.35 !important;}
.stChatMessage h3 {font-size: 1.12rem !important; line-height: 1.35 !important;}
.stChatMessage p, .stChatMessage li {font-size: 1rem !important; line-height: 1.75 !important;}
.source-snippet {
    font-size: 0.92rem;
    line-height: 1.55;
    color: #4b5563;
    white-space: pre-wrap;
    word-break: break-word;
    background: #f8fafc;
    border-radius: 0.5rem;
    padding: 0.6rem 0.75rem;
    margin-top: 0.35rem;
}
</style>
""",
        unsafe_allow_html=True,
    )


def _normalize_sources(sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized = []
    for source in sources:
        metadata = source.get("metadata") or {}
        source_type = str(source.get("source_type") or metadata.get("source_type") or "local").lower()
        source_type = "web" if source_type == "web" else "local"
        normalized.append({**source, "source_type": source_type, "metadata": metadata})
    return normalized


def render_plain_snippet(snippet: str) -> None:
    safe = html.escape(snippet or "")
    st.markdown(f"<div class='source-snippet'>{safe}</div>", unsafe_allow_html=True)


def _source_title(source: Dict[str, Any]) -> str:
    raw_title = source.get("document_title") or ""
    raw_source = source.get("document_source") or ""
    alias_title = get_title_alias(raw_title, raw_source, source.get("document_id"))
    return alias_title or make_friendly_title(raw_title, raw_source) or "论文"


def _section_caption(metadata: Dict[str, Any]) -> str:
    if not isinstance(metadata, dict):
        return ""

    section_path_text = str(metadata.get("section_path_text") or "").strip()
    section_title = str(metadata.get("section_title") or "").strip()
    section_label = section_path_text or section_title
    if not section_label:
        return ""

    parts = [f"章节: {localize_paper_terms(section_label)}"]

    def _safe_int(value: Any) -> int | None:
        try:
            if value is None:
                return None
            return int(value)
        except (TypeError, ValueError):
            return None

    start_line = _safe_int(metadata.get("section_start_line"))
    end_line = _safe_int(metadata.get("section_end_line"))
    if start_line is not None and end_line is not None and start_line > 0 and end_line >= start_line:
        parts.append(f"行号: {start_line}-{end_line}")

    section_chunk_index = _safe_int(metadata.get("section_chunk_index"))
    section_chunk_count = _safe_int(metadata.get("section_chunk_count"))
    if (
        section_chunk_index is not None
        and section_chunk_count is not None
        and section_chunk_count > 1
        and section_chunk_index >= 0
    ):
        display_index = min(section_chunk_index + 1, section_chunk_count)
        parts.append(f"分片: {display_index}/{section_chunk_count}")

    return " · ".join(parts)


def render_sources(
    sources: List[Dict[str, Any]],
    base_url: str,
    key_prefix: str,
    add_openalex_source_to_kb: Callable[[str, Dict[str, Any]], tuple[bool, str]],
) -> None:
    if not sources:
        return

    normalized = _normalize_sources(sources)
    local_sources = [s for s in normalized if s["source_type"] == "local"]
    web_sources = [s for s in normalized if s["source_type"] == "web"]

    with st.expander("依据片段", expanded=False):
        if local_sources:
            st.markdown("#### 本地知识库")
            for idx, source in enumerate(local_sources, start=1):
                metadata = source.get("metadata") or {}
                title = _source_title(source)
                snippet = clean_snippet_text(source.get("snippet") or "", max_len=320)
                score = source.get("score")
                score_text = f"（相似度: {score:.3f}）" if isinstance(score, (int, float)) else ""

                st.markdown(f"**{idx}. {title}** {score_text}")
                raw_source = source.get("document_source") or ""
                if raw_source:
                    st.caption(f"来源: {localize_paper_terms(raw_source)}")
                section_caption = _section_caption(metadata)
                if section_caption:
                    st.caption(section_caption)
                render_plain_snippet(snippet)
                if metadata.get("doi"):
                    st.caption(f"DOI: {metadata.get('doi')}")
                st.divider()

        if web_sources:
            st.markdown("#### 联网搜索")
            for idx, source in enumerate(web_sources, start=1):
                metadata = source.get("metadata") or {}
                source_kind = str(metadata.get("source_kind") or "").lower()
                title = _source_title(source)
                snippet = clean_snippet_text(source.get("snippet") or "", max_len=320)

                st.markdown(f"**{idx}. {title}**")
                if source_kind == "general_web":
                    st.caption("来源类型：网页搜索")
                elif source_kind == "openalex":
                    st.caption("来源类型：OpenAlex 论文检索")
                    st.caption("说明：该依据通常基于论文元数据与摘要片段，可能不是全文。")
                else:
                    st.caption("来源类型：联网搜索")

                authors = metadata.get("authors") or []
                year = metadata.get("year")
                if authors:
                    text = "、".join(authors[:4]) + (" 等" if len(authors) > 4 else "")
                    st.caption(f"作者: {text}")
                if year:
                    st.caption(f"年份: {year}")
                if metadata.get("doi"):
                    st.markdown(f"DOI: `{metadata.get('doi')}`")
                if metadata.get("landing_page_url"):
                    st.markdown(f"[查看来源链接]({metadata.get('landing_page_url')})")
                if metadata.get("is_oa") is not None:
                    st.caption(f"开放获取: {'是' if bool(metadata.get('is_oa')) else '否'}")

                render_plain_snippet(snippet)

                pdf_url = metadata.get("pdf_url") or metadata.get("content_url")
                btn_key = f"{key_prefix}_add_{idx}_{source.get('document_id')}"
                if st.button("加入知识库", key=btn_key, disabled=not bool(pdf_url)):
                    ok, msg = add_openalex_source_to_kb(base_url, source)
                    if ok:
                        st.success(msg)
                    else:
                        st.error(msg)
                if not pdf_url:
                    st.caption("仅可查看来源：没有可用的 PDF/content 链接。")
                st.divider()


def render_analysis_panel(
    base_url: str,
    backend_health_ok: bool,
    fetch_documents: Callable[[str], List[Dict[str, Any]]],
    set_pending_prompt: Callable[[str], None],
) -> None:
    st.markdown("### 论文分析面板")
    mode = st.radio("阅读模式", options=["单篇分析", "多篇对比"], horizontal=True)
    if not backend_health_ok:
        st.warning("当前无法连接后端接口，请先检查接口地址或服务状态。")
        return

    documents = fetch_documents(base_url)
    if not documents:
        st.info("当前知识库暂无可用论文，请先导入文档。")
        return

    doc_map: Dict[str, Dict[str, Any]] = {}
    options: List[str] = []
    for doc in documents:
        doc_id = str(doc.get("id") or "")
        if doc_id:
            options.append(doc_id)
            doc_map[doc_id] = doc
    if not options:
        st.info("当前知识库暂无可用论文，请先导入文档。")
        return

    def fmt(doc_id: str) -> str:
        doc = doc_map.get(doc_id, {})
        alias = get_title_alias(doc.get("title", ""), doc.get("source", ""), doc.get("id", ""))
        return alias or make_friendly_title(doc.get("title", ""), doc.get("source", ""))

    if mode == "单篇分析":
        selected_doc_id = st.selectbox(
            "选择论文",
            options=[""] + options,
            format_func=lambda value: "请选择论文" if value == "" else fmt(value),
        )
        selected_title = fmt(selected_doc_id) if selected_doc_id else ""
        col1, col2, col3 = st.columns(3)
        col4, col5, col6 = st.columns(3)
        if col1.button("快速总结", use_container_width=True):
            st.warning("请先选择 1 篇论文。") if not selected_doc_id else set_pending_prompt(build_single_summary_prompt(selected_title, selected_doc_id))
        if col2.button("创新点分析", use_container_width=True):
            st.warning("请先选择 1 篇论文。") if not selected_doc_id else set_pending_prompt(build_single_innovation_prompt(selected_title, selected_doc_id))
        if col3.button("方法流程", use_container_width=True):
            st.warning("请先选择 1 篇论文。") if not selected_doc_id else set_pending_prompt(build_single_method_prompt(selected_title, selected_doc_id))
        if col4.button("实验解读", use_container_width=True):
            st.warning("请先选择 1 篇论文。") if not selected_doc_id else set_pending_prompt(build_single_experiment_prompt(selected_title, selected_doc_id))
        if col5.button("局限性分析", use_container_width=True):
            st.warning("请先选择 1 篇论文。") if not selected_doc_id else set_pending_prompt(build_single_limitation_prompt(selected_title, selected_doc_id))
        if col6.button("对我研究的启发", use_container_width=True):
            st.warning("请先选择 1 篇论文。") if not selected_doc_id else set_pending_prompt(build_single_inspiration_prompt(selected_title, selected_doc_id))
    else:
        selected_doc_ids = st.multiselect("选择论文（2~3 篇）", options=options, format_func=fmt, max_selections=3)
        selected_papers = [
            {"title": fmt(doc_id), "document_id": doc_id}
            for doc_id in selected_doc_ids
        ]
        c1, c2, c3, c4 = st.columns(4)
        if c1.button("核心问题对比", use_container_width=True):
            st.warning("多篇对比至少选择 2 篇论文。") if len(selected_papers) < 2 else set_pending_prompt(build_multi_problem_compare_prompt(selected_papers))
        if c2.button("方法与创新点对比", use_container_width=True):
            st.warning("多篇对比至少选择 2 篇论文。") if len(selected_papers) < 2 else set_pending_prompt(build_multi_method_compare_prompt(selected_papers))
        if c3.button("实验与结果对比", use_container_width=True):
            st.warning("多篇对比至少选择 2 篇论文。") if len(selected_papers) < 2 else set_pending_prompt(build_multi_experiment_compare_prompt(selected_papers))
        if c4.button("适用场景与借鉴价值", use_container_width=True):
            st.warning("多篇对比至少选择 2 篇论文。") if len(selected_papers) < 2 else set_pending_prompt(build_multi_value_compare_prompt(selected_papers))
