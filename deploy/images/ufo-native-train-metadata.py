"""Reconcile Torch packaging with UFO's explicit NCCL override, without code edits."""
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path

dist = metadata.distribution('torch')
assert dist.version == '2.7.1+cu128'
path = Path(dist._path) / 'METADATA'
original = path.read_bytes()
before = b'Requires-Dist: nvidia-nccl-cu12==2.26.2;'
after = b'Requires-Dist: nvidia-nccl-cu12==2.26.5;'
# Wheel metadata can include a space before the environment marker.
if before not in original:
    before = b'Requires-Dist: nvidia-nccl-cu12==2.26.2 ;'
    after = b'Requires-Dist: nvidia-nccl-cu12==2.26.5 ;'
assert original.count(before) == 1, 'Unexpected Torch NCCL metadata; review this profile'
assert metadata.version('nvidia-nccl-cu12') == '2.26.5'
patched = original.replace(before, after)
root = Path('/opt/eb-native')
(root / 'torch-METADATA.original').write_bytes(original)
path.write_bytes(patched)
(root / 'torch-metadata-receipt.json').write_text(json.dumps({
    'schema_version': 'eb/ufo-torch-metadata-reconciliation/v1',
    'reason': 'Preserve UFO pyproject tool.uv override nvidia-nccl-cu12==2.26.5',
    'original_sha256': hashlib.sha256(original).hexdigest(),
    'patched_sha256': hashlib.sha256(patched).hexdigest(),
    'replacement_from': before.decode(), 'replacement_to': after.decode(),
    'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'torch_version': dist.version, 'nccl_version': metadata.version('nvidia-nccl-cu12'),
}, sort_keys=True, indent=2))
