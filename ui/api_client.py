import base64
import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List

import requests
import streamlit as st

STREAM_READ_TIMEOUT_SECONDS = float(os.getenv("STREAM_READ_TIMEOUT_SECONDS", "90"))

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
        "use_react": False,
        "search_type": "hybrid",
        "is_streaming": False,
        "stop_requested": False,
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = list(default) if isinstance(default, list) else default


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


def stream_chat(
    message: str,
    base_url: str,
    search_type: str,
    use_web_search: bool,
    use_react: bool,
    user_id: str,
):
    ensure_chat_state()
    request_data = {
        "message": message,
        "session_id": st.session_state.active_session_id,
        "user_id": user_id,
        "search_type": search_type,
        "use_web_search": use_web_search,
        "use_react": use_react,
    }
    status_box = st.empty()
    response_box = st.empty()
    sources_box = st.empty()
    stop_box = st.empty()

    status_box.info("正在连接后端流式接口...")
    st.session_state.is_streaming = True
    st.session_state.stop_requested = False

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
        for data in iter_chat_stream_events(base_url, request_data):
            if bool(st.session_state.get("stop_requested")):
                stopped_by_user = True
                break
            dtype = data.get("type")
            if dtype == "session":
                st.session_state.active_session_id = data.get("session_id")
                status_box.info("已连接后端，正在等待回答...")
            elif dtype == "status":
                status_text = str(data.get("content", "") or "")
                if "检测到回答格式异常" in status_text or "正在重新生成" in status_text:
                    continue
                status_box.info(status_text)
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
            elif dtype == "end":
                break
    except Exception as e:
        stream_error = str(e)
        response_box.error(f"流式请求失败：{e}")

    if full_response.strip():
        status_box.empty()
    stop_box.empty()
    if full_response.strip():
        response_box.markdown(full_response)
    elif stopped_by_user:
        response_box.info("已停止继续生成，已保留当前输出内容。")
    elif not stream_error:
        response_box.info("本轮没有收到有效回答，请稍后重试。")
    ensure_chat_state()
    if full_response.strip():
        metadata: Dict[str, Any] = {"sources": current_sources}
        if stopped_by_user:
            metadata["stopped_by_user"] = True
        if stream_error:
            metadata["stream_error"] = stream_error
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response, "metadata": metadata}
        )
    st.session_state.restored_session_id = st.session_state.active_session_id
    st.session_state.is_streaming = False


def format_session_time(iso_value: str) -> str:
    if not iso_value:
        return ""
    try:
        dt = datetime.fromisoformat(iso_value.replace("Z", "+00:00"))
        return dt.strftime("%m-%d %H:%M")
    except Exception:
        return ""

