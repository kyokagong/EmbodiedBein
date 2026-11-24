# Reuse the installed MuJoCo, CPU OSMesa and ffmpeg runtime; no new dependencies.
FROM eb/ufo-native:rpo-train-v5
COPY backend/eb/__init__.py /opt/eb-motiondecode/eb/__init__.py
COPY backend/eb/motiondecode.py backend/eb/motiondecode_view.py backend/eb/motiondecode_preview.py /opt/eb-motiondecode/eb/
ENV PYTHONPATH=/opt/eb-motiondecode MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
    OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 LP_NUM_THREADS=2 PYTHONUNBUFFERED=1
WORKDIR /opt/eb-motiondecode
CMD ["python", "-m", "eb.motiondecode_preview", "--help"]
