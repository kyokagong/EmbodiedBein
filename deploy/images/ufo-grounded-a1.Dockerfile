# Build with make images-ufo-grounded-a1, which pins this local base to e122bb0... .
# Never retag or rebuild the frozen training image.
FROM eb/ufo-native:rpo-train-v2
WORKDIR /opt/eb-native
COPY backend/eb/ufo_grounded_reference.py ./eb/ufo_grounded_reference.py
COPY backend/eb/runners/ufo_grounded_reference.py ./eb/runners/ufo_grounded_reference.py
RUN python -m compileall -q eb/ufo_grounded_reference.py eb/runners/ufo_grounded_reference.py
ENV PYTHONPATH=/opt/eb-native PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
ENTRYPOINT []
CMD ["python", "-m", "eb.runners.ufo_grounded_reference", "--help"]
