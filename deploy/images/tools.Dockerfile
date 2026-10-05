FROM python:3.11-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends python3-pip \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir numpy pyyaml
WORKDIR /workspace
COPY deploy/eb_tools /opt/eb_tools
ENV PYTHONPATH=/opt/eb_tools
CMD ["python", "-m", "eb_tools.retarget", "--help"]
# Dataset parsing is isolated in task containers, never installed into the API.
RUN pip install --no-cache-dir pyarrow==19.0.1 huggingface-hub==0.34.4
COPY backend/eb/data_validation.py backend/eb/data_worker.py backend/eb/model_sources.py backend/eb/model_worker.py backend/eb/robotwin.py backend/eb/robotwin_worker.py /opt/eb_tools/eb/
COPY backend/eb/rpo_wbc.py /opt/eb_tools/eb/rpo_wbc.py
COPY backend/eb/rpo_amp_convert.py /opt/eb_tools/eb/rpo_amp_convert.py
COPY backend/eb/runners/rpo_amp_motion.py /opt/eb_tools/eb/runners/rpo_amp_motion.py
