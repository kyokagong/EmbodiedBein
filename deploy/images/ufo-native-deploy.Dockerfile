FROM ufo-rpo:rpo-23dof-20261016
USER root
RUN apt-get update && apt-get install -y --no-install-recommends libosmesa6 && rm -rf /var/lib/apt/lists/*
RUN uv pip install --python /opt/ufo-venv/bin/python onnxruntime==1.22.1 matplotlib==3.10.6
# Runtime injects NVIDIA libraries but this base only ships the Mesa vendor entry.
COPY deploy/images/ufo-native-nvidia-egl.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json
WORKDIR /opt/eb-native
COPY backend/eb/__init__.py ./eb/__init__.py
COPY backend/eb/ufo_native.py ./eb/ufo_native.py
COPY backend/eb/ufo_live.py ./eb/ufo_live.py
COPY backend/eb/domain/environment_profile.py ./eb/domain/environment_profile.py
COPY backend/eb/runners/ufo_native_*.py ./eb/runners/
ENV PYTHONPATH=/opt/eb-native PYTHONDONTWRITEBYTECODE=1
RUN python - <<'PY'
import hashlib, importlib.metadata, importlib.util, json
from pathlib import Path
root = Path(importlib.util.find_spec('ufo').origin).parent
source = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*.py')}
runner = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('eb').rglob('*.py')}
packages = ('torch','mujoco','mujoco-warp','mjlab','numpy','onnx','onnxruntime','safetensors','joblib','matplotlib','pillow')
Path('runtime-receipt.json').write_text(json.dumps({'ufo_source':source,'platform_runner_source':runner,
 'base_image_id':'sha256:8733fd6e6ed852b4687d43e5499a5bd4f6dec58b12e5ff23a3bc92ff233f46fd',
 'dependencies':{name:importlib.metadata.version(name) for name in packages}},sort_keys=True))
PY
ENTRYPOINT []
CMD ["python", "-m", "eb.runners.ufo_native_package", "--plan", "/plan.json"]
