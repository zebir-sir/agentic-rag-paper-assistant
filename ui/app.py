import os
import time
from typing import Any, Dict
from urllib.parse import urlparse, urlunparse

import requests
import streamlit as st
from dotenv import load_dotenv

try:
    from ui.api_client import (
        add_openalex_source_to_kb,
        fetch_documents,
        fetch_openalex_status,
        fetch_session_messages,
        fetch_sessions,
        fetch_web_search_status,
        format_session_time,
        clean_assistant_display_text,
        stream_chat,
        cancel_ingestion_job,
        fetch_ingestion_job,
        start_pdf_ingestion,
        upload_pdf_to_kb,
    )
    from ui.components import inject_styles, render_analysis_panel, render_sources
except ImportError:  # pragma: no cover - streamlit script mode
    from api_client import (
        add_openalex_source_to_kb,
        fetch_documents,
        fetch_openalex_status,
        fetch_session_messages,
        fetch_sessions,
        fetch_web_search_status,
        format_session_time,
        clean_assistant_display_text,
        stream_chat,
        cancel_ingestion_job,
        fetch_ingestion_job,
        start_pdf_ingestion,
        upload_pdf_to_kb,
    )
    from components import inject_styles, render_analysis_panel, render_sources


load_dotenv()

APP_PORT = int(os.getenv("APP_PORT", 8000))
API_URL = os.getenv("API_URL", f"http://localhost:{APP_PORT}")
USER_ID = "user"

st.set_page_config(page_title="科研论文阅读助手", page_icon="📎", layout="wide")
inject_styles()


def ensure_app_state() -> None:
    defaults = {
        "active_session_id": None,
        "messages": [],
        "session_list": [],
        "restored_session_id": None,
        "pending_prompt": None,
        "use_web_search": False,
        "use_openalex_search": False,
        "use_general_web_search": False,
        "use_react": True,
        "search_type": "hybrid",
        "is_streaming": False,
        "stop_requested": False,
        "ingestion_job_id": None,
        "ingestion_job_filename": None,
        "ingestion_job_done": False,
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = list(default) if isinstance(default, list) else default


ensure_app_state()


def _normalize_base_url(raw_url: str) -> str:
    value = (raw_url or "").strip() or API_URL
    if "://" not in value:
        value = f"http://{value}"
    parsed = urlparse(value)
    scheme = parsed.scheme or "http"
    netloc = parsed.netloc or parsed.path
    normalized = urlunparse((scheme, netloc, "", "", "", ""))
    return normalized.rstrip("/")


def _build_fallback_base_url(primary_base_url: str) -> str | None:
    try:
        parsed = urlparse(primary_base_url)
        host = (parsed.hostname or "").lower()
        scheme = parsed.scheme or "http"
        port = parsed.port or APP_PORT
        if host == "api" and port == 8888:
            return f"{scheme}://localhost:8059"
        if host in {"localhost", "127.0.0.1"} and port == 8059:
            return f"{scheme}://api:8888"
        if host == "api":
            return f"{scheme}://localhost:{port}"
        if host in {"localhost", "127.0.0.1"}:
            return f"{scheme}://api:{port}"
    except Exception:
        return None
    return None


def _probe_health(base_url: str) -> tuple[bool, str | None]:
    live_url = f"{base_url}/health/live"
    deep_url = f"{base_url}/health"
    last_error = None

    for _ in range(3):
        try:
            resp = requests.get(live_url, timeout=5)
            if resp.status_code == 200 and resp.json().get("status") == "ok":
                return True, None
            last_error = f"{live_url} HTTP {resp.status_code}"
        except Exception as exc:
            last_error = f"{live_url} error: {exc}"
        time.sleep(0.5)

    for _ in range(2):
        try:
            resp = requests.get(deep_url, timeout=10)
            if resp.status_code == 200:
                return True, None
            last_error = f"{deep_url} HTTP {resp.status_code}"
        except Exception as exc:
            last_error = f"{deep_url} error: {exc}"
        time.sleep(0.5)

    return False, last_error


def resolve_backend_base_url(base_url: str) -> tuple[str, Dict[str, Any]]:
    primary = _normalize_base_url(base_url)
    fallback = _build_fallback_base_url(primary)

    primary_ok, primary_err = _probe_health(primary)
    if primary_ok:
        return primary, {
            "primary_base_url": primary,
            "resolved_base_url": primary,
            "used_fallback": False,
            "health_ok": True,
            "error_summary": None,
        }

    if fallback and fallback != primary:
        fallback_ok, fallback_err = _probe_health(fallback)
        if fallback_ok:
            return fallback, {
                "primary_base_url": primary,
                "resolved_base_url": fallback,
                "used_fallback": True,
                "health_ok": True,
                "error_summary": primary_err,
            }
        return primary, {
            "primary_base_url": primary,
            "resolved_base_url": primary,
            "used_fallback": False,
            "health_ok": False,
            "error_summary": f"primary: {primary_err or 'unreachable'}; fallback: {fallback_err or 'unreachable'}",
        }

    return primary, {
        "primary_base_url": primary,
        "resolved_base_url": primary,
        "used_fallback": False,
        "health_ok": False,
        "error_summary": primary_err,
    }


def _set_pending_prompt(prompt: str) -> None:
    st.session_state.pending_prompt = prompt
    st.rerun()


def send_user_message(
    message: str,
    base_url: str,
    search_type: str,
    use_web_search: bool,
    allow_web_search: bool,
    allow_openalex_search: bool,
    use_react: bool,
) -> None:
    ensure_app_state()
    ok, _ = _probe_health(base_url)
    if not ok:
        st.warning("后端不可用，请先检查服务状态。")
        return

    st.session_state.messages.append({"role": "user", "content": message, "metadata": {}})
    with st.chat_message("user"):
        st.write(message)
    with st.chat_message("assistant"):
        stream_chat(
            message=message,
            base_url=base_url,
            search_type=search_type,
            use_web_search=use_web_search,
            use_react=use_react,
            user_id=USER_ID,
            allow_web_search=allow_web_search,
            allow_openalex_search=allow_openalex_search,
        )
    st.session_state.session_list = fetch_sessions(base_url)


def _effective_use_web_search(openalex_enabled: bool, general_web_enabled: bool) -> bool:
    allow_openalex = bool(st.session_state.get("use_openalex_search", False))
    allow_web = bool(st.session_state.get("use_general_web_search", False))
    return bool((allow_openalex and openalex_enabled) or (allow_web and general_web_enabled))


def render_welcome_guide() -> None:
    st.markdown("### 开始阅读你的论文")
    st.caption("你可以直接提问，也可以使用下方输入框上方的「分析面板」选择论文总结、创新点分析或多篇对比。")
    st.markdown(
        "- `总结这篇论文的研究问题、核心方法和创新点`\n"
        "- `分析这篇论文的实验设置是否充分`\n"
        "- `对比两篇论文的方法差异和适用场景`\n"
        "- `帮我查找某个方向的 related work 并给出来源`\n"
        "- `例如：对比 RRT* 和 Informed RRT* 的区别`"
    )


def _render_upload_ingest_panel(base_url: str) -> None:
    st.markdown("#### 上传论文入库")
    uploaded_pdf = st.file_uploader("选择 PDF 文件", type=["pdf"], key="kb_pdf_uploader")
    fast_ingest = st.checkbox(
        "快速入库（推荐）",
        value=True,
        key="kb_fast_ingest",
        help="快速模式会跳过图片/表格解析和语义切分，适合先快速加入知识库。",
    )
    if st.button("开始入库", disabled=uploaded_pdf is None, key="kb_upload_submit"):
        with st.spinner("正在创建入库任务，可能需要几十秒..."):
            ok, msg, payload = start_pdf_ingestion(
                base_url,
                uploaded_pdf.name if uploaded_pdf is not None else "",
                uploaded_pdf.getvalue() if uploaded_pdf is not None else b"",
                fast=fast_ingest,
            )
        if ok:
            st.success(msg)
            st.session_state.ingestion_job_id = payload.get("job_id")
            st.session_state.ingestion_job_filename = payload.get("filename")
            st.session_state.ingestion_job_done = False
            st.caption(f"任务 ID：{payload.get('job_id')}")
            st.rerun()
        else:
            st.error(msg)
    st.divider()


def _render_ingestion_progress_body(base_url: str) -> None:
    job_id = st.session_state.get("ingestion_job_id")
    if not job_id:
        return

    ok, msg, payload = fetch_ingestion_job(base_url, str(job_id))
    if not ok:
        st.warning(msg)
        if st.button("关闭提示", key="close_ingest_status_error"):
            st.session_state.ingestion_job_id = None
            st.session_state.ingestion_job_filename = None
            st.session_state.ingestion_job_done = False
            st.rerun()
        return

    status = str(payload.get("status") or "")
    progress = int(payload.get("progress") or 0)
    message = str(payload.get("message") or "")
    filename = str(payload.get("filename") or st.session_state.get("ingestion_job_filename") or "")

    st.caption(f"入库文件：{filename}")
    st.caption(f"状态：{status} · {message}")
    st.progress(max(0, min(100, progress)))

    if status in {"queued", "running", "cancelling"}:
        if st.button("取消入库", key=f"cancel_ingest_{job_id}", type="secondary"):
            c_ok, c_msg, _ = cancel_ingestion_job(base_url, str(job_id))
            if c_ok:
                st.warning("已请求取消入库")
            else:
                st.error(c_msg)
            st.rerun()
        if not hasattr(st, "fragment"):
            if st.button("刷新进度", key=f"refresh_ingest_{job_id}"):
                st.rerun()
        return

    if status == "succeeded":
        st.success("入库完成")
    elif status == "failed":
        st.error("入库失败")
        err_text = str(payload.get("error") or payload.get("stderr_tail") or payload.get("stdout_tail") or "")
        if err_text:
            with st.expander("查看错误详情", expanded=False):
                st.code(err_text[-2000:])
    elif status == "cancelled":
        st.warning("入库已取消")

    st.session_state.ingestion_job_done = True
    if st.button("关闭提示", key=f"close_ingest_{job_id}"):
        st.session_state.ingestion_job_id = None
        st.session_state.ingestion_job_filename = None
        st.session_state.ingestion_job_done = False
        st.rerun()


if hasattr(st, "fragment"):
    @st.fragment(run_every="1s")
    def render_ingestion_progress(base_url: str) -> None:
        _render_ingestion_progress_body(base_url)
else:
    def render_ingestion_progress(base_url: str) -> None:
        _render_ingestion_progress_body(base_url)


def _render_analysis_panel_compact(base_url: str, backend_health_ok: bool) -> None:
    if hasattr(st, "popover"):
        with st.popover("📄 分析面板", use_container_width=False):
            _render_upload_ingest_panel(base_url)
            render_analysis_panel(base_url, backend_health_ok, fetch_documents, _set_pending_prompt)
    else:
        with st.expander("📄 分析面板", expanded=False):
            _render_upload_ingest_panel(base_url)
            render_analysis_panel(base_url, backend_health_ok, fetch_documents, _set_pending_prompt)


def _render_tools_compact(
    openalex_enabled: bool,
    general_web_enabled: bool,
    general_web_provider: str,
    backend_health_ok: bool,
) -> None:
    def _render_body() -> None:
        st.toggle(
            "OpenAlex 检索",
            key="use_openalex_search",
            help="开启仅表示允许系统使用 OpenAlex 学术元数据检索，是否调用由 Planner 自动判断。",
        )
        st.toggle(
            "Web 检索",
            key="use_general_web_search",
            help="开启仅表示允许系统联网检索网页信息，是否调用由 Planner 自动判断。",
        )
        st.toggle(
            "深度分析 / ReAct",
            key="use_react",
            help="开启后使用 Planner-guided 深度分析流程；关闭时走普通聊天路径。",
        )
        st.caption("系统会在你开启的能力范围内自动规划检索。")
        if not openalex_enabled:
            if backend_health_ok:
                st.caption("OpenAlex 当前不可用（未配置 OPENALEX_API_KEY）。")
            else:
                st.caption("OpenAlex 状态暂不可判断：后端不可达。")
        if backend_health_ok and not general_web_enabled:
            st.caption(f"通用网页搜索当前不可用（provider: {general_web_provider or '未配置'}）。")
        if backend_health_ok and general_web_enabled:
            st.caption(f"通用网页搜索已配置：{general_web_provider}")

    if hasattr(st, "popover"):
        with st.popover("🛠 工具", use_container_width=False):
            _render_body()
    else:
        with st.expander("🛠 工具", expanded=False):
            _render_body()


def render_input_toolbar(
    resolved_base_url: str,
    backend_health_ok: bool,
    openalex_enabled: bool,
    general_web_enabled: bool,
    general_web_provider: str,
) -> None:
    st.caption("系统会在你开启的能力范围内自动规划本地检索、章节检索、图表/算法检索以及可用的外部检索。")
    left, right = st.columns([2.1, 7.9])
    with left:
        c1, c2 = st.columns(2)
        with c1:
            _render_analysis_panel_compact(resolved_base_url, backend_health_ok)
        with c2:
            _render_tools_compact(openalex_enabled, general_web_enabled, general_web_provider, backend_health_ok)
    with right:
        if bool(st.session_state.get("is_streaming")):
            if st.button("■ 停止", type="secondary", use_container_width=True):
                st.session_state.stop_requested = True


with st.sidebar:
    st.header("设置")
    input_base_url = st.text_input("接口地址", value=API_URL)
    resolved_base_url, backend_diag = resolve_backend_base_url(input_base_url)
    backend_health_ok = bool(backend_diag.get("health_ok"))

    st.caption(f"当前后端：`{resolved_base_url}`")
    if backend_diag.get("used_fallback"):
        st.info(f"已自动切换到可用接口：`{resolved_base_url}`")
    if not backend_health_ok:
        st.warning("当前接口地址不可用，请检查 API_URL 或容器网络。")
        if backend_diag.get("error_summary"):
            st.caption(f"诊断：{backend_diag.get('error_summary')}")

    if st.button("检查服务状态"):
        now_ok, now_err = _probe_health(resolved_base_url)
        if now_ok:
            st.success("服务运行正常")
        else:
            st.error(f"服务不可用：{now_err}")
    if st.button("新建对话"):
        st.session_state.messages = []
        st.session_state.active_session_id = None
        st.session_state.restored_session_id = None
        st.session_state.pending_prompt = None
        st.rerun()
    if st.button("刷新会话列表"):
        st.session_state.session_list = fetch_sessions(resolved_base_url) if backend_health_ok else []
        st.rerun()
    if st.button("清空当前对话"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.subheader("历史会话")
    st.session_state.session_list = fetch_sessions(resolved_base_url) if backend_health_ok else []
    if not st.session_state.session_list:
        st.caption("暂无历史会话" if backend_health_ok else "后端不可达，无法加载历史会话")
    for session in st.session_state.session_list:
        sid = session.get("session_id")
        title = session.get("title") or "新对话"
        time_str = format_session_time(session.get("updated_at"))
        selected = sid == st.session_state.active_session_id
        label = f"{'● ' if selected else ''}{title}  {time_str}"
        if st.button(label, key=f"session_{sid}", use_container_width=True):
            st.session_state.active_session_id = sid
            st.session_state.messages = fetch_session_messages(resolved_base_url, sid)
            st.session_state.restored_session_id = sid
            st.session_state.pending_prompt = None
            st.rerun()


openalex_enabled = fetch_openalex_status(resolved_base_url) if backend_health_ok else False
web_search_status = (
    fetch_web_search_status(resolved_base_url)
    if backend_health_ok
    else {"enabled": False, "provider": ""}
)
general_web_enabled = bool(web_search_status.get("enabled"))
general_web_provider = str(web_search_status.get("provider") or "")

st.title("科研论文阅读助手")

if not st.session_state.messages:
    render_welcome_guide()

for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            st.markdown(clean_assistant_display_text(msg["content"]))
        else:
            st.write(msg["content"])
        if msg["role"] == "assistant":
            metadata = msg.get("metadata") or {}
            sources = metadata.get("sources") if isinstance(metadata, dict) else []
            if isinstance(sources, list) and sources:
                render_sources(sources, resolved_base_url, f"hist_{idx}", add_openalex_source_to_kb)

if st.session_state.pending_prompt:
    pending = st.session_state.pending_prompt
    st.session_state.pending_prompt = None
    send_user_message(
        pending,
        resolved_base_url,
        "hybrid",
        _effective_use_web_search(openalex_enabled, general_web_enabled),
        bool(st.session_state.get("use_general_web_search", False)),
        bool(st.session_state.get("use_openalex_search", False)),
        bool(st.session_state.get("use_react", True)),
    )
    st.rerun()

render_ingestion_progress(resolved_base_url)

render_input_toolbar(
    resolved_base_url=resolved_base_url,
    backend_health_ok=backend_health_ok,
    openalex_enabled=openalex_enabled,
    general_web_enabled=general_web_enabled,
    general_web_provider=general_web_provider,
)

if prompt := st.chat_input("请输入您的问题", disabled=not backend_health_ok):
    if not backend_health_ok:
        st.warning("后端不可用，请先检查服务状态。")
        st.stop()
    send_user_message(
        prompt,
        resolved_base_url,
        "hybrid",
        _effective_use_web_search(openalex_enabled, general_web_enabled),
        bool(st.session_state.get("use_general_web_search", False)),
        bool(st.session_state.get("use_openalex_search", False)),
        bool(st.session_state.get("use_react", True)),
    )
