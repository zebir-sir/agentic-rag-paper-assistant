import base64
import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests
import streamlit as st

STREAM_READ_TIMEOUT_SECONDS = float(os.getenv("STREAM_READ_TIMEOUT_SECONDS", "90"))
VISIBLE_STATUS_PHASES = {"planning", "document_inspection", "retrieval", "generation", "warning"}

try:
    from ui.components import render_sources
except ImportError:  # pragma: no cover - streamlit script mode
    from components import render_sources


def ensure_chat_state() -> None:
    defaults = {
        "active_session_id": None,
        "messages": [],
        "session_list": [],
        "restored_session_id": None,
        "pending_prompt": None,
        "use_web_search": False,
        "use_react": True,
        "search_type": "hybrid",
        "is_streaming": False,
        "stop_requested": False,
        "stop_button_visible": False,
        "cancel_requested": False,
        "cancel_status": "",
        "current_run_id": None,
        "cancelled_by_user": False,
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = list(default) if isinstance(default, list) else default


def clean_assistant_display_text(text: str) -> str:
    value = str(text or "")
    legacy = "当前没有检索到直接相关片段，以下内容更适合作为一般性分析参考。"
    neutral = "本轮没有可核对的检索片段；未被检索证据支持的结论请谨慎参考。"
    value = value.replace(legacy, neutral)
    value = value.replace("以下内容更适合作为一般性分析参考。", "")
    value = value.replace("以下内容", "")
    return value


def normalize_status_text(data: Dict[str, Any]) -> str:
    phase = str(data.get("phase") or "")
    content = str(data.get("content") or "")
    phase_key = phase.strip().lower()
    if phase_key == "planning":
        return "正在规划回答..."
    if phase_key == "retrieval":
        return content or "正在检索相关资料..."
    if phase_key == "generation":
        return "正在生成回答..."
    if phase_key == "document_inspection":
        return "正在读取知识库文档..."
    return content


def map_status_text(content: str, phase: str) -> str:
    phase_key = str(phase or "").strip().lower()
    if phase_key == "planning":
        return "正在规划回答..."
    if phase_key == "document_inspection":
        return "正在读取知识库文档..."
    if phase_key == "generation":
        return "正在生成回答..."
    if phase_key == "retrieval":
        text = str(content or "").strip()
        return text or "正在检索相关资料..."
    if phase_key == "warning":
        return str(content or "").strip()
    return ""


def should_show_status_event(data: Dict[str, Any]) -> bool:
    if not isinstance(data, dict):
        return False
    if data.get("user_visible") is False:
        return False
    phase = str(data.get("phase") or "").strip()
    if phase:
        return phase in {"planning", "retrieval", "generation", "warning", "document_inspection"}
    content = str(data.get("content") or "")
    hidden_markers = [
        "正在检查知识库文档",
        "Local retrieval round",
        "Retrieval round",
        "Compressing conversation history",
        "Searching local knowledge base",
        "Analyzing the question and generating an answer",
    ]
    return not any(marker in content for marker in hidden_markers)


def should_display_status_event(event: Dict[str, Any]) -> bool:
    phase = str(event.get("phase") or "").strip().lower()
    user_visible = bool(event.get("user_visible", True))
    if not user_visible:
        return False
    return phase in VISIBLE_STATUS_PHASES


def fetch_openalex_status(base_url: str) -> bool:
    try:
        resp = requests.get(f"{base_url}/openalex/status", timeout=5)
        return bool(resp.json().get("enabled")) if resp.status_code == 200 else False
    except Exception:
        return False


def fetch_web_search_status(base_url: str) -> Dict[str, Any]:
    try:
        resp = requests.get(f"{base_url}/web-search/status", timeout=5)
        if resp.status_code == 200:
            payload = resp.json()
            return {
                "enabled": bool(payload.get("enabled")),
                "provider": str(payload.get("provider") or ""),
            }
        return {"enabled": False, "provider": ""}
    except Exception:
        return {"enabled": False, "provider": ""}


def fetch_sessions(base_url: str) -> List[Dict[str, Any]]:
    try:
        resp = requests.get(f"{base_url}/sessions", params={"limit": 20, "days": 7}, timeout=10)
        return resp.json().get("sessions", []) if resp.status_code == 200 else []
    except Exception:
        return []


def fetch_documents(base_url: str) -> List[Dict[str, Any]]:
    try:
        resp = requests.get(f"{base_url}/documents", params={"limit": 200, "offset": 0}, timeout=15)
        return resp.json().get("documents", []) if resp.status_code == 200 else []
    except Exception:
        return []


def fetch_session_messages(base_url: str, session_id: str) -> List[Dict[str, Any]]:
    try:
        resp = requests.get(f"{base_url}/sessions/{session_id}/messages", timeout=10)
        if resp.status_code != 200:
            return []
        items = resp.json().get("messages", [])
        return [
            {
                "role": item["role"],
                "content": item["content"],
                "metadata": item.get("metadata") or {},
            }
            for item in items
        ]
    except Exception:
        return []


def delete_session(base_url: str, session_id: str) -> tuple[bool, str]:
    try:
        resp = requests.delete(f"{base_url}/sessions/{session_id}", timeout=10)
        if resp.status_code == 200:
            return True, "会话已删除"
        try:
            detail = (resp.json() or {}).get("detail")
        except Exception:
            detail = resp.text
        return False, f"删除会话失败：{detail}"
    except Exception as e:
        return False, f"删除会话失败：{e}"



def upload_pdf_to_kb(base_url: str, filename: str, file_bytes: bytes, fast: bool = True) -> tuple[bool, str, Dict[str, Any]]:
    try:
        content_base64 = base64.b64encode(file_bytes).decode("ascii")
        resp = requests.post(
            f"{base_url}/documents/upload",
            json={
                "filename": filename,
                "content_base64": content_base64,
                "fast": fast,
            },
            timeout=600,
        )
        if resp.status_code == 200:
            return True, "入库成功", resp.json()
        try:
            detail = (resp.json() or {}).get("detail")
        except Exception:
            detail = resp.text
        return False, f"入库失败：{detail}", {}
    except Exception as e:
        return False, f"入库失败：{e}", {}


def start_pdf_ingestion(base_url: str, filename: str, file_bytes: bytes, fast: bool = True) -> tuple[bool, str, Dict[str, Any]]:
    try:
        content_base64 = base64.b64encode(file_bytes).decode("ascii")
        resp = requests.post(
            f"{base_url}/documents/upload/start",
            json={
                "filename": filename,
                "content_base64": content_base64,
                "fast": fast,
            },
            timeout=120,
        )
        if resp.status_code == 200:
            payload = resp.json()
            if str(payload.get("status")) == "accepted" and payload.get("job_id"):
                return True, "已开始入库", payload
            return False, "入库任务创建失败：返回格式异常", {}
        try:
            detail = (resp.json() or {}).get("detail")
        except Exception:
            detail = resp.text
        return False, f"入库任务创建失败：{detail}", {}
    except Exception as e:
        return False, f"入库任务创建失败：{e}", {}


def fetch_ingestion_job(base_url: str, job_id: str) -> tuple[bool, str, Dict[str, Any]]:
    try:
        resp = requests.get(f"{base_url}/documents/upload/jobs/{job_id}", timeout=10)
        if resp.status_code == 200:
            return True, "ok", resp.json()
        try:
            detail = (resp.json() or {}).get("detail")
        except Exception:
            detail = resp.text
        return False, f"获取入库任务失败：{detail}", {}
    except Exception as e:
        return False, f"获取入库任务失败：{e}", {}


def cancel_ingestion_job(base_url: str, job_id: str) -> tuple[bool, str, Dict[str, Any]]:
    try:
        resp = requests.post(f"{base_url}/documents/upload/jobs/{job_id}/cancel", timeout=20)
        if resp.status_code == 200:
            return True, "已请求取消入库", resp.json()
        try:
            detail = (resp.json() or {}).get("detail")
        except Exception:
            detail = resp.text
        return False, f"取消入库失败：{detail}", {}
    except Exception as e:
        return False, f"取消入库失败：{e}", {}

def add_openalex_source_to_kb(base_url: str, source: Dict[str, Any]) -> tuple[bool, str]:
    metadata = source.get("metadata") or {}
    payload = {
        "title": source.get("document_title"),
        "openalex_id": metadata.get("openalex_id") or source.get("document_id"),
        "pdf_url": metadata.get("pdf_url"),
        "content_url": metadata.get("content_url"),
    }
    try:
        resp = requests.post(f"{base_url}/openalex/add-to-kb", json=payload, timeout=300)
        if resp.status_code == 200:
            return True, "已加入知识库并完成导入。"
        try:
            detail = resp.json().get("detail")
        except Exception:
            detail = resp.text
        return False, f"加入失败：{detail}"
    except Exception as e:
        return False, f"加入失败：{e}"


def cancel_chat_stream(base_url: str, run_id: str) -> tuple[bool, str, Dict[str, Any]]:
    if not str(run_id or "").strip():
        return False, "run_id 为空", {}
    try:
        resp = requests.post(f"{base_url}/chat/stream/{run_id}/cancel", timeout=5)
        payload = resp.json() if resp.status_code == 200 else {}
        status = str(payload.get("status") or "")
        if resp.status_code == 200:
            if status in {"cancelled", "already_finished", "not_found"}:
                return True, status, payload
            return True, "not_found", payload
        return False, f"HTTP {resp.status_code}", payload
    except Exception as e:
        return False, str(e), {}


def iter_chat_stream_events(base_url: str, request_data: Dict[str, Any]):
    buffer = ""
    with requests.post(
        f"{base_url}/chat/stream",
        json=request_data,
        stream=True,
        timeout=(10, STREAM_READ_TIMEOUT_SECONDS),
        headers={
            "Accept": "text/event-stream",
            "Accept-Encoding": "identity",
            "Cache-Control": "no-cache",
        },
    ) as resp:
        if resp.status_code != 200:
            body_text = ""
            try:
                body_text = resp.text
            except Exception:
                body_text = ""
            detail = body_text.strip()[:500] if body_text else f"HTTP {resp.status_code}"
            raise RuntimeError(f"/chat/stream 请求失败（{resp.status_code}）：{detail}")
        for raw in resp.iter_content(chunk_size=1, decode_unicode=True):
            if not raw:
                continue
            buffer += raw
            while "\n\n" in buffer:
                event_text, buffer = buffer.split("\n\n", 1)
                for line in event_text.splitlines():
                    line = line.strip()
                    if not line.startswith("data: "):
                        continue
                    try:
                        data = json.loads(line[6:])
                        yield data
                    except json.JSONDecodeError:
                        continue


def build_chat_request_data(
    message: str,
    search_type: str,
    use_web_search: bool,
    use_react: bool,
    user_id: str,
    allow_web_search: bool = False,
    allow_openalex_search: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    request_metadata: Dict[str, Any] = dict(metadata or {})
    request_metadata.update({
        "allow_web_search": bool(allow_web_search),
        "allow_openalex_search": bool(allow_openalex_search),
    })
    return {
        "message": message,
        "session_id": session_id,
        "user_id": user_id,
        "search_type": search_type,
        "use_web_search": use_web_search,
        "use_react": use_react,
        "metadata": request_metadata,
    }


def stream_chat(
    message: str,
    base_url: str,
    search_type: str,
    use_web_search: bool,
    use_react: bool,
    user_id: str,
    allow_web_search: bool = False,
    allow_openalex_search: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
    **_: Any,
):
    ensure_chat_state()
    request_data = build_chat_request_data(
        message=message,
        search_type=search_type,
        use_web_search=use_web_search,
        use_react=use_react,
        user_id=user_id,
        allow_web_search=allow_web_search,
        allow_openalex_search=allow_openalex_search,
        metadata=metadata,
        session_id=st.session_state.active_session_id,
    )
    status_box = st.empty()
    response_box = st.empty()
    sources_box = st.empty()
    stop_box = st.empty()

    status_box.info("正在连接后端流式接口...")
    st.session_state.is_streaming = True
    st.session_state.stop_requested = False
    st.session_state.stop_button_visible = True
    st.session_state.cancel_requested = False
    st.session_state.cancel_status = ""
    st.session_state.current_run_id = None
    st.session_state.cancelled_by_user = False
    st.session_state.stream_loop_exit_reason = ""

    full_response = ""
    current_sources: List[Dict[str, Any]] = []
    stream_error = ""
    stopped_by_user = False
    try:
        # Fallback stop control near assistant output area.
        # Streamlit's rerun model can make toolbar stop buttons timing-sensitive during streaming.
        with stop_box.container():
            if st.button("■ 停止", type="secondary", key=f"stop_stream_fallback_{int(time.time() * 1000)}"):
                st.session_state.stop_requested = True
                st.session_state.cancel_requested = True
                st.session_state.stop_button_visible = False
                st.session_state.cancel_status = "已停止生成"
        for data in iter_chat_stream_events(base_url, request_data):
            if bool(st.session_state.get("stop_requested")):
                stopped_by_user = True
                st.session_state.stop_button_visible = False
                st.session_state.cancel_requested = True
                st.session_state.cancelled_by_user = True
                st.session_state.stream_loop_exit_reason = "stream loop exited by user stop"
                run_id = str(st.session_state.get("current_run_id") or "")
                if run_id:
                    ok, cancel_state, _ = cancel_chat_stream(base_url, run_id)
                    if ok and cancel_state in {"cancelled", "already_finished", "not_found"}:
                        st.session_state.cancel_status = "已停止生成"
                    else:
                        st.session_state.cancel_status = "已停止生成"
                break
            dtype = data.get("type")
            if dtype == "session":
                st.session_state.active_session_id = data.get("session_id")
                st.session_state.current_run_id = data.get("run_id")
                status_box.info("已连接后端，正在等待回答...")
            elif dtype == "status":
                if not should_show_status_event(data):
                    continue
                mapped = normalize_status_text(data)
                if not mapped:
                    continue
                status_box.info(mapped)
            elif dtype == "text":
                status_box.empty()
                full_response += data.get("content", "")
                response_box.markdown(full_response + "▌")
                time.sleep(0.005)
            elif dtype == "replace":
                status_box.empty()
                full_response = data.get("content", "")
                response_box.markdown(full_response + "▌")
            elif dtype == "sources":
                current_sources = data.get("sources", []) or []
                with sources_box.container():
                    render_sources(current_sources, base_url, "stream", add_openalex_source_to_kb)
            elif dtype == "error":
                stream_error = str(data.get("content") or "流式请求失败")
                response_box.error(stream_error)
                break
            elif dtype == "cancelled":
                stopped_by_user = True
                st.session_state.cancelled_by_user = True
                st.session_state.cancel_status = str(data.get("message") or "已停止生成")
                st.session_state.stream_loop_exit_reason = "stream loop exited by server cancelled event"
                break
            elif dtype == "end":
                break
    except Exception as e:
        stream_error = str(e)
        response_box.error(f"流式请求失败：{e}")
    finally:
        if full_response.strip():
            status_box.empty()
        stop_box.empty()
        if full_response.strip():
            cleaned_response = clean_assistant_display_text(full_response)
            response_box.markdown(cleaned_response)
            full_response = cleaned_response
        elif stopped_by_user:
            status_box.empty()
            response_box.info("已停止生成")
        elif not stream_error:
            response_box.info("本轮没有收到有效回答，请稍后重试。")
        ensure_chat_state()
        if full_response.strip():
            metadata: Dict[str, Any] = {"sources": current_sources}
            if stopped_by_user:
                metadata["stopped_by_user"] = True
                metadata["cancelled"] = True
                metadata["cancelled_by_user"] = True
                metadata["partial_response"] = True
                metadata["run_id"] = st.session_state.get("current_run_id")
            if stream_error:
                metadata["stream_error"] = stream_error
            st.session_state.messages.append(
                {"role": "assistant", "content": clean_assistant_display_text(full_response), "metadata": metadata}
            )
        st.session_state.restored_session_id = st.session_state.active_session_id
        st.session_state.is_streaming = False
        st.session_state.stop_requested = False
        st.session_state.stop_button_visible = False
        st.session_state.cancel_requested = False
        st.session_state.current_run_id = None


def format_session_time(iso_value: str) -> str:
    if not iso_value:
        return ""
    try:
        dt = datetime.fromisoformat(iso_value.replace("Z", "+00:00"))
        return dt.strftime("%m-%d %H:%M")
    except Exception:
        return ""

