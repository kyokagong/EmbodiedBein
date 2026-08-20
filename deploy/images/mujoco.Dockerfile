FROM python:3.11-slim-bookworm
RUN pip install --no-cache-dir mujoco numpy pyyaml
WORKDIR /workspace
COPY deploy/mujoco_tools /opt/mujoco_tools
ENV PYTHONPATH=/opt/mujoco_tools
CMD ["python", "-m", "sim2sim_gate"]
