# syntax=docker/dockerfile:1
FROM node:20-alpine AS web
WORKDIR /web
COPY frontend/package.json ./
RUN npm install
COPY frontend/ ./
ENV EB_WEB_OUT=/app/dist
RUN mkdir -p /app/dist && npm run build

FROM rust:1.83-bookworm AS rust
WORKDIR /src
COPY rust/ ./
RUN cargo build --release -p ebd

FROM python:3.11-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates git \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv
WORKDIR /app
COPY backend /app/backend
COPY contracts /app/contracts
COPY --from=web /app/dist /app/web/dist
COPY --from=rust /src/target/release/ebctl /usr/local/bin/ebctl
WORKDIR /app/backend
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
RUN uv sync --no-dev
ENV PATH="/app/backend/.venv/bin:$PATH" \
    PYTHONPATH="/app/backend" \
    EB_WEB_DIST=/app/web/dist \
    EB_CONTRACTS_DIR=/app/contracts
EXPOSE 8000
CMD ["uvicorn", "eb.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
