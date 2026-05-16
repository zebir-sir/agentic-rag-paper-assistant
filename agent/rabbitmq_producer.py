import json
from typing import Any, Dict, Optional

from .app_config import get_ingestion_dlq_name, get_ingestion_queue_name, get_rabbitmq_url


def build_ingestion_task_message(
    task_id: str,
    document_id: Optional[str],
    file_path: str,
) -> Dict[str, Any]:
    return {
        "task_id": task_id,
        "document_id": document_id,
        "file_path": file_path,
    }


async def publish_ingestion_task(
    task_id: str,
    document_id: Optional[str],
    file_path: str,
) -> None:
    rabbitmq_url = get_rabbitmq_url()
    if not rabbitmq_url:
        raise RuntimeError("RabbitMQ URL is not configured")

    try:
        import aio_pika
    except ImportError as exc:
        raise RuntimeError("aio-pika is not installed") from exc

    queue_name = get_ingestion_queue_name()
    message_dict = build_ingestion_task_message(task_id, document_id, file_path)
    message_body = json.dumps(message_dict, ensure_ascii=False).encode("utf-8")

    connection = await aio_pika.connect_robust(rabbitmq_url)
    try:
        channel = await connection.channel()
        await channel.declare_queue(queue_name, durable=True)
        await channel.default_exchange.publish(
            aio_pika.Message(
                body=message_body,
                content_type="application/json",
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            ),
            routing_key=queue_name,
        )
    finally:
        await connection.close()


def build_ingestion_dlq_message(
    task_id: str,
    document_id: Optional[str],
    file_path: str,
    retry_count: int,
    error_message: str,
    failed_at: str,
) -> Dict[str, Any]:
    return {
        "task_id": task_id,
        "document_id": document_id,
        "file_path": file_path,
        "status": "failed",
        "retry_count": retry_count,
        "error_message": error_message,
        "failed_at": failed_at,
    }


async def publish_ingestion_dlq_message(payload: Dict[str, Any]) -> None:
    rabbitmq_url = get_rabbitmq_url()
    if not rabbitmq_url:
        raise RuntimeError("RabbitMQ URL is not configured")

    try:
        import aio_pika
    except ImportError as exc:
        raise RuntimeError("aio-pika is not installed") from exc

    queue_name = get_ingestion_dlq_name()
    message_body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    connection = await aio_pika.connect_robust(rabbitmq_url)
    try:
        channel = await connection.channel()
        await channel.declare_queue(queue_name, durable=True)
        await channel.default_exchange.publish(
            aio_pika.Message(
                body=message_body,
                content_type="application/json",
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            ),
            routing_key=queue_name,
        )
    finally:
        await connection.close()
