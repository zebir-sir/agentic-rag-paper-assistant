FROM python:3.11-slim AS base

WORKDIR /app

RUN sed -i 's|deb.debian.org|mirrors.aliyun.com|g' /etc/apt/sources.list.d/debian.sources \
    && apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgl1 \
    libglib2.0-0 \
    libx11-6 \
    libxcb1 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY requirements/ ./requirements/

FROM base AS rag-core

RUN uv pip install --system -r requirements/requirements-worker.txt

FROM rag-core AS runtime

RUN uv pip install --system fastapi uvicorn redis

COPY . .

EXPOSE 8888

FROM runtime AS api

FROM runtime AS worker

FROM node:22-alpine AS web

WORKDIR /app/web

COPY web/package*.json ./
RUN npm ci

COPY web/ ./

EXPOSE 5173
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "5173"]
