FROM python:3.11-slim AS base

WORKDIR /app

ARG PYPI_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

ENV PIP_INDEX_URL=${PYPI_INDEX_URL} \
    UV_DEFAULT_INDEX=${PYPI_INDEX_URL}

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
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install --no-cache-dir --index-url "${PYPI_INDEX_URL}" uv

FROM base AS rag-core

COPY pyproject.toml README.md ./
COPY agent/ ./agent/
COPY common/ ./common/
COPY ingestion/ ./ingestion/

RUN uv pip install --system --no-cache .

FROM rag-core AS runtime

COPY . .

EXPOSE 8888

FROM runtime AS api

FROM runtime AS worker

FROM node:22-alpine AS web

WORKDIR /app/web

ARG NPM_REGISTRY=https://registry.npmmirror.com

COPY web/package*.json ./
RUN npm config set registry "${NPM_REGISTRY}" \
    && npm config set replace-registry-host always \
    && npm ci --registry="${NPM_REGISTRY}"

COPY web/ ./

EXPOSE 5173
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "5173"]
