import os


def get_rabbitmq_url() -> str:
    rabbitmq_url = os.getenv("RABBITMQ_URL", "").strip()
    if rabbitmq_url:
        return rabbitmq_url
    user = os.getenv("RABBITMQ_DEFAULT_USER", "").strip()
    password = os.getenv("RABBITMQ_DEFAULT_PASS", "").strip()
    host = os.getenv("RABBITMQ_HOST", "").strip()
    port = os.getenv("RABBITMQ_PORT", "").strip()
    vhost = os.getenv("RABBITMQ_VHOST", "/").strip()
    if user and password and host and port:
        return f"amqp://{user}:{password}@{host}:{port}/{vhost}"
    return ""


def get_ingestion_queue_name() -> str:
    return str(os.getenv("INGESTION_QUEUE_NAME", "ingestion_tasks") or "ingestion_tasks").strip()


def get_ingestion_dlq_name() -> str:
    return str(os.getenv("INGESTION_DLQ_NAME", "ingestion_tasks_dlq") or "ingestion_tasks_dlq").strip()


def get_ingestion_max_retries() -> int:
    raw = str(os.getenv("INGESTION_MAX_RETRIES", "3") or "3").strip()
    try:
        value = int(raw)
    except ValueError:
        return 3
    return max(1, value)
