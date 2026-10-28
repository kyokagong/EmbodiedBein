# syntax=docker/dockerfile:1
FROM node:20-alpine AS web
WORKDIR /web
COPY frontend/package.json ./
RUN npm install
COPY frontend/ ./
ENV EB_WEB_OUT=/app/dist
RUN mkdir -p /app/dist && npm run build

FROM python:3.11-slim-bookworm
RUN sed -i 's|http://deb.debian.org|https://mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list.d/debian.sources \
    && apt-get -o Acquire::https::Timeout=15 -o Acquire::Retries=1 update \
    && apt-get -o Acquire::https::Timeout=15 -o Acquire::Retries=1 install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv
WORKDIR /app
COPY backend /app/backend
COPY contracts /app/contracts
COPY plugins /app/plugins
COPY --from=web /app/dist /app/web/dist
WORKDIR /app/backend
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
RUN uv sync --no-dev
ENV PATH="/app/backend/.venv/bin:$PATH" \
    PYTHONPATH="/app/backend:/app/plugins" \
    EB_WEB_DIST=/app/web/dist \
    EB_CONTRACTS_DIR=/app/contracts
EXPOSE 8000
CMD ["uvicorn", "eb.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
