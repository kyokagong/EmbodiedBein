FROM eb/ufo-native:rpo-train-v3
COPY backend/eb/runners/rlinf_ufo_*.py /opt/eb-native/eb/runners/
COPY backend/eb/ufo_training_checkpoint.py /opt/eb-native/eb/ufo_training_checkpoint.py
RUN python - <<'PY'
import hashlib, json
from pathlib import Path
root = Path('/opt/eb-native')
receipt = json.loads((root / 'train-image-receipt.json').read_text())
receipt['parent_image_digest'] = 'sha256:7a0babe5e5b8cff1b63bedee6c1746cb4710710a4305ec93667d8a52df58cb3e'
receipt['capabilities'] = ['native_state', '2048_env_fresh_replay_reconfiguration']
receipt['runner_source'] = {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (root / 'eb').rglob('*.py')}
(root / 'train-image-receipt.json').write_text(json.dumps(receipt, sort_keys=True, indent=2))
PY
RUN python -m compileall -q /opt/eb-native/eb
