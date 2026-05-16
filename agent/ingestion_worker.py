import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict

from .app_config import get_ingestion_max_retries, get_ingestion_queue_name, get_rabbitmq_url
from .db_utils import close_database, initialize_database
from .ingestion_jobs import ingest_saved_pdf_file
from .ingestion_tasks_db import get_ingestion_task, update_ingestion_task_status
from .rabbitmq_producer import (
    build_ingestion_dlq_message,
    publish_ingestion_dlq_message,
    publish_ingestion_task,
)

logger = logging.getLogger(__name__)


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _short_error(exc: Exception, max_len: int = 500) -> str:
    text = str(exc).strip() or exc.__class__.__name__
    if len(text) <= max_len:
        return text
    return text[:max_len]


def parse_ingestion_message(body: bytes) -> Dict[str, Any]:
    try:
        payload = json.loads(body.decode("utf-8"))
    except Exception as exc:
        raise ValueError(f"Invalid JSON message: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError("Message payload must be a JSON object")

    task_id = str(payload.get("task_id") or "").strip()
    file_path = str(payload.get("file_path") or "").strip()
    document_id_raw = payload.get("document_id")
    document_id = str(document_id_raw).strip() if document_id_raw else None

    if not task_id:
        raise ValueError("Missing required field: task_id")

    return {
        "task_id": task_id,
        "file_path": file_path,
        "document_id": document_id,
    }


async def handle_ingestion_message(payload: Dict[str, Any]) -> Dict[str, Any]:
    task_id = str(payload.get("task_id") or "").strip()
    file_path = str(payload.get("file_path") or "").strip()
    message_document_id = payload.get("document_id")
    message_document_id = str(message_document_id).strip() if message_document_id else None
    max_retries = get_ingestion_max_retries()
    if not task_id:
        raise ValueError("Missing required field: task_id")

    current = await get_ingestion_task(task_id)
    if not current:
        logger.warning("Task not found, skip duplicated/orphan message: task_id=%s", task_id)
        return {"action": "skip_not_found", "task_id": task_id}

    current_status = str(current.get("status") or "")
    current_retry = int(current.get("retry_count") or 0)
    current_document_id = str(current.get("document_id") or "").strip() or None
    if current_status == "done":
        return {"action": "skip_done", "task_id": task_id}
    if current_status == "failed" and current_retry >= max_retries:
        return {"action": "skip_failed_max", "task_id": task_id}

    if not file_path:
        err = "Missing required field: file_path"
        new_retry = current_retry + 1
        if new_retry < max_retries:
            updated = await update_ingestion_task_status(
                task_id=task_id,
                status="queued",
                error_message=err,
                retry_count=new_retry,
                finished_at=None,
            )
            return {"action": "retry", "task": updated or current, "error_message": err}
        updated = await update_ingestion_task_status(
            task_id=task_id,
            status="failed",
            error_message=err,
            retry_count=new_retry,
            finished_at=_now_utc(),
        )
        return {"action": "failed_max", "task": updated or current, "error_message": err}

    await update_ingestion_task_status(
        task_id=task_id,
        status="processing",
        error_message=None,
        started_at=_now_utc(),
        finished_at=None,
    )

    try:
        result = await ingest_saved_pdf_file(file_path, fast=True)
        final_document_id = str(result.get("document_id") or current_document_id or message_document_id or "").strip() or None
        updated = await update_ingestion_task_status(
            task_id=task_id,
            status="done",
            document_id=final_document_id,
            error_message=None,
            finished_at=_now_utc(),
        )
        return {"action": "done", "task": updated or {"task_id": task_id, "status": "done"}}
    except Exception as exc:
        err = _short_error(exc)
        new_retry = current_retry + 1
        if new_retry < max_retries:
            updated = await update_ingestion_task_status(
                task_id=task_id,
                status="queued",
                error_message=err,
                retry_count=new_retry,
                finished_at=None,
            )
            return {"action": "retry", "task": updated or current, "error_message": err}
        updated = await update_ingestion_task_status(
            task_id=task_id,
            status="failed",
            error_message=err,
            retry_count=new_retry,
            finished_at=_now_utc(),
        )
        return {"action": "failed_max", "task": updated or current, "error_message": err}


async def process_rabbitmq_message(body: bytes) -> Dict[str, Any]:
    payload = parse_ingestion_message(body)
    result = await handle_ingestion_message(payload)
    action = str(result.get("action") or "")
    if action == "retry":
        task = result.get("task") or {}
        await publish_ingestion_task(
            task_id=str(task.get("task_id") or payload.get("task_id") or ""),
            document_id=(str(task.get("document_id") or "").strip() or None),
            file_path=str(task.get("file_path") or payload.get("file_path") or ""),
        )
        return result
    if action == "failed_max":
        task = result.get("task") or {}
        failed_at = _now_utc().isoformat()
        dlq_payload = build_ingestion_dlq_message(
            task_id=str(task.get("task_id") or payload.get("task_id") or ""),
            document_id=(str(task.get("document_id") or "").strip() or None),
            file_path=str(task.get("file_path") or payload.get("file_path") or ""),
            retry_count=int(task.get("retry_count") or 0),
            error_message=str(result.get("error_message") or task.get("error_message") or "ingestion failed"),
            failed_at=failed_at,
        )
        await publish_ingestion_dlq_message(dlq_payload)
        return result
    return result


async def process_queue_message_and_ack(message: Any) -> None:
    try:
        await process_rabbitmq_message(message.body)
    except Exception as exc:
        logger.exception("Failed to process ingestion message: %s", exc)
    finally:
        await message.ack()


async def consume_ingestion_tasks() -> None:
    rabbitmq_url = get_rabbitmq_url()
    if not rabbitmq_url:
        raise RuntimeError("RabbitMQ URL is not configured")

    try:
        import aio_pika
    except ImportError as exc:
        raise RuntimeError("aio-pika is not installed") from exc

    queue_name = get_ingestion_queue_name()
    await initialize_database()
    connection = await aio_pika.connect_robust(rabbitmq_url)
    channel = await connection.channel()
    await channel.set_qos(prefetch_count=1)
    queue = await channel.declare_queue(queue_name, durable=True)
    logger.info("Ingestion worker is consuming queue: %s", queue_name)

    try:
        async with queue.iterator() as queue_iter:
            async for message in queue_iter:
                await process_queue_message_and_ack(message)
    finally:
        await connection.close()
        await close_database()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    asyncio.run(consume_ingestion_tasks())


if __name__ == "__main__":
    main()
