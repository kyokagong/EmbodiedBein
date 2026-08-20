FROM python:3.11-slim
RUN pip install --no-cache-dir mujoco onnxruntime numpy
WORKDIR /workspace
CMD ["python", "-c", "print('eb/mujoco ready')"]
