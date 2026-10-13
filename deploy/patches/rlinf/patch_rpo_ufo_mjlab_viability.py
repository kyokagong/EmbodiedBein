"""Derive the source receipt without reapplying the frozen v12 RLinf patch."""
import hashlib
import json
from pathlib import Path

root = Path('/workspace/eb_runner')
path = Path('/opt/ufo-auxiliary-source-files.json')
hashes = json.loads(path.read_text())
for name in hashes:
    file = Path('/workspace/tests/rpo_ufo_tracking_diagnostic.py') if name == 'rpo_ufo_tracking_diagnostic.py' else root/name
    hashes[name] = hashlib.sha256(file.read_bytes()).hexdigest()
path.write_text(json.dumps(hashes,indent=2))
path = Path('/opt/ufo-source-receipt.json')
receipt = json.loads(path.read_text())
receipt.update(physics_profile='native_mjcf_viability_v13', contact_profile='native_mjcf_viability_v13',
    scope='native UFO agent + RPO MJLab dynamic RSI and training-only position tolerance',
    viability_patch_hash='sha256:'+hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    auxiliary_source_sha256='sha256:'+hashlib.sha256(json.dumps(hashes,sort_keys=True,separators=(',',':')).encode()).hexdigest())
path.write_text(json.dumps(receipt,indent=2))
