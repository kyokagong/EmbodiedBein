FROM python:3.11-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends python3-pip \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir numpy pyyaml
WORKDIR /workspace
COPY deploy/eb_tools /opt/eb_tools
ENV PYTHONPATH=/opt/eb_tools
CMD ["python", "-m", "eb_tools.retarget", "--help"]
