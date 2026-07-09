FROM python:3.11-slim AS base

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY requirements/ ./requirements/

FROM base AS rag-core

RUN uv pip install --system -r requirements/requirements-worker.txt

FROM rag-core AS api

RUN uv pip install --system fastapi uvicorn redis

COPY . .

EXPOSE 8888

FROM rag-core AS worker

RUN uv pip install --system fastapi

COPY . .

FROM base AS ui

RUN uv pip install --system -r requirements/requirements-ui.txt

COPY . .

EXPOSE 8501
