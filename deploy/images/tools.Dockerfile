FROM python:3.11-slim-bookworm
RUN pip install --no-cache-dir --timeout 30 --retries 1 --index-url https://mirrors.aliyun.com/pypi/simple numpy==1.26.4 pyyaml==6.0.2
WORKDIR /workspace
COPY deploy/eb_tools /opt/eb_tools
ENV PYTHONPATH=/opt/eb_tools
CMD ["python", "-m", "eb_tools.retarget", "--help"]
# Dataset parsing is isolated in task containers, never installed into the API.
RUN pip install --no-cache-dir --timeout 30 --retries 1 --index-url https://mirrors.aliyun.com/pypi/simple pyarrow==19.0.1 huggingface-hub==0.34.4
COPY backend/eb/data_validation.py backend/eb/data_worker.py backend/eb/model_sources.py backend/eb/model_worker.py backend/eb/robotwin.py backend/eb/robotwin_worker.py /opt/eb_tools/eb/
COPY backend/eb/rpo_wbc.py /opt/eb_tools/eb/rpo_wbc.py
COPY backend/eb/rpo_amp_convert.py /opt/eb_tools/eb/rpo_amp_convert.py
COPY backend/eb/runners/rpo_amp_motion.py /opt/eb_tools/eb/runners/rpo_amp_motion.py

# Reuse the native deployment stack's fixed CPU ONNX versions.
RUN pip install --no-cache-dir --timeout 30 --retries 1 --index-url https://mirrors.aliyun.com/pypi/simple onnx==1.16.1 onnxruntime==1.22.1
COPY backend/eb/onnx_io.py /opt/eb_tools/eb/onnx_io.py
COPY backend/eb/bundle/worker.py /opt/eb_tools/eb/bundle/worker.py
RUN python -c "import onnx, onnxruntime as ort; assert 'CPUExecutionProvider' in ort.get_available_providers()"
COPY backend/eb/motiondecode.py /opt/eb_tools/eb/motiondecode.py
COPY backend/eb/motiondecode_view.py /opt/eb_tools/eb/motiondecode_view.py
