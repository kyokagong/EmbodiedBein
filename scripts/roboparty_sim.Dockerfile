FROM python:3.12-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends libglfw3 libgl1 libgl1-mesa-dri \
    && rm -rf /var/lib/apt/lists/*
RUN python -m pip install --no-cache-dir --only-binary=:all: numpy==2.2.6 onnxruntime==1.23.2 mujoco==3.3.7
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
WORKDIR /app
COPY roboparty_sim.py /app/roboparty_sim.py
ENTRYPOINT ["python", "/app/roboparty_sim.py"]
