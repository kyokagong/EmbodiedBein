ARG BASE=eb/ufo-native:rpo-deploy-v1
FROM ${BASE} AS scheduler-runtime
USER root
WORKDIR /opt/eb-native
COPY deploy/images/ufo-native-train-requirements.txt ./scheduler-requirements.txt
COPY deploy/images/ufo-native-train-resolved.lock ./scheduler-resolved.lock
COPY deploy/images/ufo-native-train-metadata.py ./reconcile-torch-metadata.py
RUN python reconcile-torch-metadata.py && \
    uv pip freeze --python /opt/ufo-venv/bin/python | sed '/^-e /d' > base-constraints.txt && \
    uv pip install --python /opt/ufo-venv/bin/python -r scheduler-requirements.txt -c base-constraints.txt -c scheduler-resolved.lock \
      --extra-index-url https://download.pytorch.org/whl/cu128 && \
    uv pip check --python /opt/ufo-venv/bin/python && \
    uv pip freeze --python /opt/ufo-venv/bin/python > scheduler-resolved.txt
FROM scheduler-runtime
COPY backend/eb/runners/rlinf_ufo_*.py ./eb/runners/
COPY backend/eb/runners/ufo_training_data.py ./eb/runners/ufo_training_data.py
COPY backend/eb/runners/ufo_training_analysis.py ./eb/runners/ufo_training_analysis.py
COPY backend/eb/ufo_training_checkpoint.py ./eb/ufo_training_checkpoint.py
COPY backend/eb/ufo_training_prompt.py ./eb/ufo_training_prompt.py
COPY backend/eb/runners/ufo_training_prompt.py ./eb/runners/ufo_training_prompt.py
COPY backend/eb/ufo_native.py ./eb/ufo_native.py
COPY backend/eb/runners/ufo_native_codec.py backend/eb/runners/ufo_native_sim.py backend/eb/runners/ufo_native_package.py backend/eb/runners/ufo_native_live.py backend/eb/runners/ufo_native_stream.py ./eb/runners/
RUN python - <<'PY'
import hashlib, importlib.metadata as m, json, sys
from pathlib import Path
required = {'torch':'2.7.1+cu128','numpy':'1.26.4','wandb':'0.26.0','protobuf':'4.25.9','mujoco':'3.11.0','mjlab':'1.6.0','ray':'2.47.0','nvidia-nccl-cu12':'2.26.5'}
assert sys.version_info[:2] == (3,10)
assert all(m.version(k) == v for k,v in required.items())
assert set(Path('base-constraints.txt').read_text().splitlines()) <= set(Path('scheduler-resolved.txt').read_text().splitlines())
receipt = {'schema_version':'eb/ufo-native-train-image/v1', 'python':sys.version,
 'base_image_id':'sha256:79ab792eeec399e1b979c2f4ddebaaa0bceaeecf3055f62a84df659e424b7adf',
 'profile_sha256':hashlib.sha256(Path('scheduler-requirements.txt').read_bytes()).hexdigest(),
 'dependency_lock_sha256':hashlib.sha256(Path('scheduler-resolved.lock').read_bytes()).hexdigest(),
 'base_dependency_versions_preserved':True,
 'torch_metadata_reconciliation':json.loads(Path('torch-metadata-receipt.json').read_text()),
 'dependencies':{d.metadata['Name']:d.version for d in m.distributions()},
 'runner_source':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('eb').rglob('*.py')}}
Path('train-image-receipt.json').write_text(json.dumps(receipt, sort_keys=True, indent=2))
PY
RUN python -m compileall -q eb
ENV PYTHONPATH=/opt/eb-native PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=disabled
ENTRYPOINT []
CMD ["python", "-m", "eb.runners.rlinf_ufo_entry", "--plan", "/plan.json"]
