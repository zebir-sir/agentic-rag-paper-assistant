import asyncio
import base64
import binascii
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.request import Request, urlopen

from fastapi import HTTPException
from .ingestion_tasks_db import (
    create_ingestion_task,
    update_ingestion_task_status,
)
from .rabbitmq_producer import publish_ingestion_task

DOCUMENT_UPLOAD_MAX_BYTES = int(os.getenv("DOCUMENT_UPLOAD_MAX_BYTES", str(30 * 1024 * 1024)))
MAX_INGESTION_JOBS = int(os.getenv("MAX_INGESTION_JOBS", "50"))
_DOCUMENT_ID_PATTERN = re.compile(
    r"Saved document to PostgreSQL with ID:\s*([0-9a-fA-F-]{36})",
    flags=re.IGNORECASE,
)


@dataclass
class IngestionJob:
    job_id: str
    filename: str
    file_path: str
    import_dir: str
    fast: bool = True
    status: str = "queued"
    progress: int = 0
    message: str = "等待入库..."
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    returncode: Optional[int] = None
    error: Optional[str] = None
    process: Optional[Any] = None
    stdout_lines: deque[str] = field(default_factory=lambda: deque(maxlen=80))
    stderr_lines: deque[str] = field(default_factory=lambda: deque(maxlen=80))
    cancel_requested: bool = False


INGESTION_JOBS: Dict[str, IngestionJob] = {}
INGESTION_JOBS_LOCK = asyncio.Lock()


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tail_from_lines(lines: deque[str], max_chars: int = 4000) -> str:
    text = "".join(lines)
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _job_to_public_dict(job: IngestionJob) -> Dict[str, Any]:
    return {
        "job_id": job.job_id,
        "filename": job.filename,
        "file_path": job.file_path,
        "fast": job.fast,
        "status": job.status,
        "progress": int(max(0, min(100, job.progress))),
        "message": job.message,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
        "returncode": job.returncode,
        "error": job.error,
        "stdout_tail": _tail_from_lines(job.stdout_lines, max_chars=4000),
        "stderr_tail": _tail_from_lines(job.stderr_lines, max_chars=4000),
    }


def _cleanup_ingestion_jobs_locked() -> None:
    if len(INGESTION_JOBS) <= MAX_INGESTION_JOBS:
        return
    terminal_states = {"succeeded", "failed", "cancelled"}
    terminal_jobs = [job for job in INGESTION_JOBS.values() if job.status in terminal_states]
    terminal_jobs.sort(key=lambda j: j.created_at)
    while len(INGESTION_JOBS) > MAX_INGESTION_JOBS and terminal_jobs:
        oldest = terminal_jobs.pop(0)
        INGESTION_JOBS.pop(oldest.job_id, None)


def _safe_filename(seed: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", (seed or "").strip())
    cleaned = cleaned.strip("._-")
    return cleaned[:120] or f"openalex_{uuid.uuid4().hex[:8]}"


def _download_binary_file(url: str, target_path: Path) -> None:
    request = Request(url, headers={"User-Agent": "agentic-rag-openalex/1.0"})
    with urlopen(request, timeout=30) as response:
        data = response.read()
    target_path.write_bytes(data)


def _make_upload_paths(filename_raw: str) -> tuple[str, Path, Path]:
    safe_filename = _safe_filename(filename_raw)
    if not safe_filename.lower().endswith(".pdf"):
        safe_filename = f"{safe_filename}.pdf"
    stem = _safe_filename(Path(safe_filename).stem) or f"upload_{uuid.uuid4().hex[:8]}"
    unique_dir = f"{uuid.uuid4().hex[:8]}_{stem}"
    import_dir = Path("documents") / "ui_uploads" / unique_dir
    import_dir.mkdir(parents=True, exist_ok=True)
    target_path = import_dir / safe_filename
    return safe_filename, import_dir, target_path


def _validate_and_decode_upload_payload(payload: Dict[str, Any]) -> tuple[str, bytes, bool]:
    filename_raw = str(payload.get("filename") or "").strip()
    content_base64 = str(payload.get("content_base64") or "").strip()
    fast = bool(payload.get("fast", True))

    if not filename_raw:
        raise HTTPException(status_code=400, detail="filename is required")
    if not content_base64:
        raise HTTPException(status_code=400, detail="content_base64 is required")
    if not filename_raw.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are supported")

    try:
        data = base64.b64decode(content_base64, validate=True)
    except binascii.Error:
        raise HTTPException(status_code=400, detail="Invalid base64 content")

    if not data:
        raise HTTPException(status_code=400, detail="Empty file content")
    if len(data) > DOCUMENT_UPLOAD_MAX_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large (max {DOCUMENT_UPLOAD_MAX_BYTES} bytes)")
    return filename_raw, data, fast


def _build_ingestion_command(import_dir: Path, *, fast: bool = True) -> list[str]:
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "ingestion.ingest",
        "--documents",
        str(import_dir),
        "--verbose",
    ]
    if fast:
        cmd.extend(["--no-semantic", "--no-images", "--no-tables"])
    return cmd


async def _run_ingestion_for_directory(import_dir: Path, *, fast: bool = True) -> subprocess.CompletedProcess:
    cmd = _build_ingestion_command(import_dir, fast=fast)
    return await asyncio.to_thread(
        subprocess.run,
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )


def _extract_document_id_from_ingestion_output(stdout: str, stderr: str) -> Optional[str]:
    text = f"{stdout}\n{stderr}"
    match = _DOCUMENT_ID_PATTERN.search(text)
    if not match:
        return None
    return match.group(1)


async def ingest_saved_pdf_file(file_path: str, *, fast: bool = True) -> Dict[str, Any]:
    path = Path(file_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")
    # Isolate this task to a single staged file to avoid ingesting sibling PDFs.
    with tempfile.TemporaryDirectory(prefix="single_ingest_") as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        staged_name = f"{uuid.uuid4().hex[:8]}_{path.name}"
        staged_path = tmp_dir_path / staged_name
        shutil.copy2(path, staged_path)

        process = await _run_ingestion_for_directory(tmp_dir_path, fast=fast)
        stdout = process.stdout or ""
        stderr = process.stderr or ""
        if process.returncode != 0:
            error_text = (stderr or stdout or "Ingestion process failed").strip()
            raise RuntimeError(error_text[-2000:])

        return {
            "returncode": process.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "document_id": _extract_document_id_from_ingestion_output(stdout, stderr),
            "document_id_source": "ingestion_log",
            "staged_file_name": staged_name,
        }


async def _run_ingestion_job(job_id: str) -> None:
    async with INGESTION_JOBS_LOCK:
        job = INGESTION_JOBS.get(job_id)
        if not job:
            return
        job.status = "running"
        job.progress = max(job.progress, 10)
        job.message = "开始入库..."
        job.started_at = _now_utc_iso()

    cmd = _build_ingestion_command(Path(job.import_dir), fast=job.fast)
    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        async with INGESTION_JOBS_LOCK:
            if job.cancel_requested:
                job.status = "cancelling"
                job.message = "正在取消入库..."
            job.process = process

        fallback_progress = 15

        async def _read_stream(stream: asyncio.StreamReader, is_stderr: bool) -> None:
            nonlocal fallback_progress
            while True:
                raw = await stream.readline()
                if not raw:
                    break
                line = raw.decode("utf-8", errors="replace")
                lower = line.lower()
                async with INGESTION_JOBS_LOCK:
                    current = INGESTION_JOBS.get(job_id)
                    if not current:
                        continue
                    if is_stderr:
                        current.stderr_lines.append(line)
                    else:
                        current.stdout_lines.append(line)
                        if "extracting content from" in lower:
                            current.progress = max(current.progress, 25)
                            current.message = "正在解析 PDF..."
                        elif "total chunks created" in lower:
                            current.progress = max(current.progress, 45)
                            current.message = "正在分块..."
                        elif "saved document to postgresql" in lower:
                            current.progress = max(current.progress, 90)
                            current.message = "正在写入数据库..."
                        elif "ingestion summary" in lower:
                            current.progress = max(current.progress, 95)
                            current.message = "正在整理入库结果..."
                        else:
                            m = re.search(r"Embedding batch\s+(\d+)\s*/\s*(\d+)\s+processed", line, flags=re.IGNORECASE)
                            if m:
                                x = int(m.group(1))
                                y = max(1, int(m.group(2)))
                                p = 50 + int(35 * x / y)
                                current.progress = max(current.progress, p)
                                current.message = f"正在生成向量 {x}/{y}..."
                            else:
                                fallback_progress = min(85, fallback_progress + 1)
                                current.progress = max(current.progress, fallback_progress)

        await asyncio.gather(
            _read_stream(process.stdout, is_stderr=False),
            _read_stream(process.stderr, is_stderr=True),
        )
        returncode = await process.wait()

        async with INGESTION_JOBS_LOCK:
            current = INGESTION_JOBS.get(job_id)
            if not current:
                return
            current.returncode = returncode
            if current.cancel_requested or (returncode is not None and returncode < 0):
                current.status = "cancelled"
                current.message = "入库已取消"
            elif returncode == 0:
                current.status = "succeeded"
                current.progress = 100
                current.message = "入库完成"
            else:
                current.status = "failed"
                current.message = "入库失败"
                stderr_tail = _tail_from_lines(current.stderr_lines, max_chars=2000)
                stdout_tail = _tail_from_lines(current.stdout_lines, max_chars=2000)
                current.error = stderr_tail or stdout_tail or "Ingestion process failed"
            current.finished_at = _now_utc_iso()
            current.process = None
            _cleanup_ingestion_jobs_locked()
    except Exception as exc:
        async with INGESTION_JOBS_LOCK:
            current = INGESTION_JOBS.get(job_id)
            if not current:
                return
            current.status = "failed"
            current.message = "入库失败"
            current.error = str(exc)
            current.finished_at = _now_utc_iso()
            current.process = None
            _cleanup_ingestion_jobs_locked()


async def start_upload_ingestion_job(payload: Dict[str, Any]) -> Dict[str, Any]:
    filename_raw, data, fast = _validate_and_decode_upload_payload(payload)
    safe_filename, import_dir, target_path = _make_upload_paths(filename_raw)
    target_path.write_bytes(data)

    job_id = uuid.uuid4().hex
    job = IngestionJob(
        job_id=job_id,
        filename=safe_filename,
        file_path=str(target_path),
        import_dir=str(import_dir),
        fast=fast,
    )
    async with INGESTION_JOBS_LOCK:
        INGESTION_JOBS[job_id] = job
        _cleanup_ingestion_jobs_locked()
    asyncio.create_task(_run_ingestion_job(job_id))
    return {
        "status": "accepted",
        "job_id": job_id,
        "filename": safe_filename,
        "file_path": str(target_path),
        "fast": fast,
    }


async def run_sync_upload_ingestion(payload: Dict[str, Any]) -> Dict[str, Any]:
    filename_raw, data, fast = _validate_and_decode_upload_payload(payload)
    safe_filename, import_dir, target_path = _make_upload_paths(filename_raw)
    target_path.write_bytes(data)

    try:
        ingestion_result = await ingest_saved_pdf_file(str(target_path), fast=fast)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Ingestion failed",
                "stderr": str(exc),
            },
        )

    return {
        "status": "success",
        "filename": safe_filename,
        "file_path": str(target_path),
        "ingestion_output": str(ingestion_result.get("stdout") or "")[-2000:],
        "fast": fast,
    }


async def submit_async_ingestion_task(payload: Dict[str, Any]) -> Dict[str, Any]:
    filename_raw, data, _fast = _validate_and_decode_upload_payload(payload)
    _safe_filename, _import_dir, target_path = _make_upload_paths(filename_raw)
    target_path.write_bytes(data)

    task_id = uuid.uuid4().hex
    task = await create_ingestion_task(
        task_id=task_id,
        document_id=None,
        file_path=str(target_path),
        status="queued",
    )

    try:
        await publish_ingestion_task(
            task_id=task["task_id"],
            document_id=task.get("document_id"),
            file_path=task["file_path"],
        )
    except Exception as exc:
        updated = await update_ingestion_task_status(
            task_id=task_id,
            status="failed",
            error_message=str(exc),
        )
        raise HTTPException(
            status_code=503,
            detail={
                "message": "Failed to publish ingestion task to RabbitMQ",
                "task_id": task_id,
                "error": str(exc),
                "task": updated,
            },
        )

    return task


async def get_upload_ingestion_job(job_id: str) -> Dict[str, Any]:
    async with INGESTION_JOBS_LOCK:
        job = INGESTION_JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return _job_to_public_dict(job)


async def cancel_upload_ingestion_job(job_id: str) -> Dict[str, Any]:
    terminal_states = {"succeeded", "failed", "cancelled"}
    async with INGESTION_JOBS_LOCK:
        job = INGESTION_JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        if job.status in terminal_states:
            return _job_to_public_dict(job)
        job.cancel_requested = True
        job.status = "cancelling"
        job.message = "正在取消入库..."
        process = job.process

    if process and process.returncode is None:
        try:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=5)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
        except ProcessLookupError:
            pass
        except Exception:
            pass

    async with INGESTION_JOBS_LOCK:
        latest = INGESTION_JOBS.get(job_id)
        if not latest:
            raise HTTPException(status_code=404, detail="Job not found")
        if latest.status not in terminal_states:
            latest.status = "cancelled"
            latest.message = "入库已取消"
            latest.finished_at = _now_utc_iso()
            latest.process = None
            if latest.returncode is None:
                latest.returncode = -int(signal.SIGTERM)
        _cleanup_ingestion_jobs_locked()
        return _job_to_public_dict(latest)


async def add_openalex_file_to_kb(file_url: str, title: str) -> Dict[str, Any]:
    filename = _safe_filename(title)
    if not filename.lower().endswith(".pdf"):
        filename += ".pdf"

    import_dir = Path("documents") / "openalex_imports" / _safe_filename(title)
    import_dir.mkdir(parents=True, exist_ok=True)
    target_path = import_dir / filename

    await asyncio.to_thread(_download_binary_file, file_url, target_path)

    process = await _run_ingestion_for_directory(import_dir, fast=True)
    if process.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Ingestion failed",
                "stderr": process.stderr[-2000:],
            },
        )

    return {
        "status": "success",
        "file_path": str(target_path),
        "ingestion_output": process.stdout[-2000:],
    }
