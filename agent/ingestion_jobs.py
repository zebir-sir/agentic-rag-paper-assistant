"""Persistent ingestion task submission and worker-side PDF execution helpers."""

import asyncio
import base64
import binascii
import os
import re
import shutil
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.request import Request, urlopen

from fastapi import HTTPException

from .ingestion_tasks_db import create_ingestion_task, update_ingestion_task_status
from .rabbitmq_producer import publish_ingestion_task

DOCUMENT_UPLOAD_MAX_BYTES = int(os.getenv("DOCUMENT_UPLOAD_MAX_BYTES", str(30 * 1024 * 1024)))


def _safe_filename(seed: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", (seed or "").strip())
    cleaned = cleaned.strip("._-")
    return cleaned[:120] or f"upload_{uuid.uuid4().hex[:8]}"


def _make_upload_paths(filename_raw: str) -> tuple[str, Path, Path]:
    safe_filename = _safe_filename(filename_raw)
    if not safe_filename.lower().endswith(".pdf"):
        safe_filename = f"{safe_filename}.pdf"
    stem = _safe_filename(Path(safe_filename).stem)
    import_dir = Path("documents") / "ui_uploads" / f"{uuid.uuid4().hex[:8]}_{stem}"
    import_dir.mkdir(parents=True, exist_ok=True)
    target_path = import_dir / safe_filename
    return safe_filename, import_dir, target_path


def _validate_and_decode_upload_payload(payload: Dict[str, Any]) -> tuple[str, bytes, bool]:
    filename_raw = str(payload.get("filename") or "").strip()
    content_base64 = str(payload.get("content_base64") or "").strip()
    fast = bool(payload.get("fast", False))
    if not filename_raw:
        raise HTTPException(status_code=400, detail="filename is required")
    if not content_base64:
        raise HTTPException(status_code=400, detail="content_base64 is required")
    if not filename_raw.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are supported")
    try:
        data = base64.b64decode(content_base64, validate=True)
    except binascii.Error as exc:
        raise HTTPException(status_code=400, detail="Invalid base64 content") from exc
    if not data:
        raise HTTPException(status_code=400, detail="Empty file content")
    if len(data) > DOCUMENT_UPLOAD_MAX_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large (max {DOCUMENT_UPLOAD_MAX_BYTES} bytes)")
    return filename_raw, data, fast


async def _enqueue_file(file_path: Path, *, fast: bool) -> Dict[str, Any]:
    task = await create_ingestion_task(
        task_id=uuid.uuid4().hex,
        document_id=None,
        file_path=str(file_path),
        fast=fast,
        status="queued",
    )
    try:
        await publish_ingestion_task(
            task_id=task["task_id"],
            document_id=None,
            file_path=task["file_path"],
            fast=fast,
        )
    except Exception as exc:
        updated = await update_ingestion_task_status(
            task_id=str(task["task_id"]), status="failed", error_message=str(exc)
        )
        raise HTTPException(
            status_code=503,
            detail={"message": "Failed to publish ingestion task to RabbitMQ", "task": updated, "error": str(exc)},
        ) from exc
    return task


async def submit_async_ingestion_task(payload: Dict[str, Any]) -> Dict[str, Any]:
    filename_raw, data, fast = _validate_and_decode_upload_payload(payload)
    _, _, target_path = _make_upload_paths(filename_raw)
    target_path.write_bytes(data)
    return await _enqueue_file(target_path, fast=fast)


async def submit_async_ingestion_tasks(payloads: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    """Create one durable task per PDF; the worker executes them in queue order."""
    if not payloads:
        raise HTTPException(status_code=400, detail="At least one PDF is required")
    if len(payloads) > 100:
        raise HTTPException(status_code=400, detail="A batch can contain at most 100 PDFs")
    return [await submit_async_ingestion_task(payload) for payload in payloads]


def _build_ingestion_command(import_dir: Path, *, fast: bool = False) -> list[str]:
    cmd = [sys.executable, "-u", "-m", "ingestion.ingest", "--documents", str(import_dir), "--verbose"]
    if fast:
        cmd.extend(["--no-semantic", "--no-images", "--no-tables"])
    return cmd


async def ingest_saved_pdf_file(
    file_path: str,
    *,
    fast: bool = False,
    should_interrupt: Optional[callable] = None,
    progress_callback: Optional[callable] = None,
) -> Dict[str, Any]:
    """Execute exactly one queued PDF in an isolated temporary directory."""
    path = Path(file_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")
    with tempfile.TemporaryDirectory(prefix="single_ingest_") as tmp_dir:
        staged_path = Path(tmp_dir) / f"{uuid.uuid4().hex[:8]}_{path.name}"
        shutil.copy2(path, staged_path)
        process = await asyncio.create_subprocess_exec(
            *_build_ingestion_command(Path(tmp_dir), fast=fast),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_lines: list[str] = []
        stderr_lines: list[str] = []

        async def read_stream(stream: asyncio.StreamReader, target: list[str]) -> None:
            while raw := await stream.readline():
                line = raw.decode("utf-8", errors="replace")
                target.append(line)
                if progress_callback:
                    await progress_callback(line)

        readers = [asyncio.create_task(read_stream(process.stdout, stdout_lines)), asyncio.create_task(read_stream(process.stderr, stderr_lines))]
        try:
            while True:
                if should_interrupt and await should_interrupt():
                    process.terminate()
                    await process.wait()
                    raise IngestionInterrupted("Task was paused or deleted")
                try:
                    await asyncio.wait_for(process.wait(), timeout=0.5)
                    break
                except asyncio.TimeoutError:
                    continue
            await asyncio.gather(*readers)
        finally:
            if process.returncode is None:
                process.terminate()
                await process.wait()
            await asyncio.gather(*readers, return_exceptions=True)
        stdout, stderr = "".join(stdout_lines), "".join(stderr_lines)
        if process.returncode != 0:
            raise RuntimeError((stderr or stdout or "Ingestion process failed").strip()[-2000:])
        return {"returncode": process.returncode, "stdout": stdout, "stderr": stderr, "staged_file_name": staged_path.name}


class IngestionInterrupted(Exception):
    """Raised when a durable task is paused or deleted during processing."""


def _download_binary_file(url: str, target_path: Path) -> None:
    request = Request(url, headers={"User-Agent": "agentic-rag-openalex/1.0"})
    with urlopen(request, timeout=30) as response:
        content_type = str(response.headers.get("Content-Type") or "").lower()
        data = response.read()
    # OpenAlex may expose a publisher landing page even when no OA PDF exists.
    # Reject HTML and non-PDF payloads before they enter the ingestion queue.
    if not data.startswith(b"%PDF-") and "application/pdf" not in content_type:
        raise ValueError(f"URL did not return a PDF (content-type: {content_type or 'unknown'})")
    target_path.write_bytes(data)


async def add_openalex_file_to_kb(file_url: str, title: str, *, fast: bool = False) -> Dict[str, Any]:
    """Download an OpenAlex PDF and submit it to the same durable queue as local uploads."""
    filename = _safe_filename(title)
    if not filename.lower().endswith(".pdf"):
        filename += ".pdf"
    import_dir = Path("documents") / "openalex_imports" / f"{uuid.uuid4().hex[:8]}_{_safe_filename(title)}"
    import_dir.mkdir(parents=True, exist_ok=True)
    target_path = import_dir / filename
    try:
        await asyncio.to_thread(_download_binary_file, file_url, target_path)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"OpenAlex PDF download failed: {exc}") from exc
    task = await _enqueue_file(target_path, fast=fast)
    return task
