FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./
RUN pip install uv && uv pip install --system -r pyproject.toml

# RabbitMQ async client is installed separately to avoid invalidating the heavy base dependency layer
RUN uv pip install --system aio-pika
# Redis client is installed separately to avoid invalidating the heavy base dependency layer
RUN uv pip install --system redis

COPY . .

EXPOSE 8058
EXPOSE 8501
