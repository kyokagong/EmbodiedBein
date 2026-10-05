FROM eb-api:latest AS backend
RUN uv pip install --python /app/backend/.venv/bin/python pytest pytest-asyncio pyarrow==19.0.1 ruff
WORKDIR /workspace/backend
ENV PYTHONPATH=/workspace/backend
CMD ["python", "-m", "pytest", "-q", "-p", "no:cacheprovider"]
