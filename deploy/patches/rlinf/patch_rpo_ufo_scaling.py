"""Image-local time budget and measured owned sources for the frozen F recipe."""

import hashlib
import json
from pathlib import Path

runner = Path("/workspace/rlinf/rlinf/runners/embodied_runner.py")
source = runner.read_text()
old = 'time.time() - start_time >= 3300'
new = 'time.time() - start_time >= (int(os.environ.get("EB_UFO_WALL_SECONDS", "3300")) if self.cfg.actor.model.get("scaling_profile") == "env6144_batch2048_v6" else 3300)'
assert source.count(old) == 1
runner.write_text(source.replace(old, new))

files_path = Path("/opt/ufo-auxiliary-source-files.json")
hashes = json.loads(files_path.read_text())
for name in hashes:
    path = Path("/workspace/tests/rpo_ufo_tracking_diagnostic.py") if name == "rpo_ufo_tracking_diagnostic.py" else Path("/workspace/eb_runner") / name
    hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
files_path.write_text(json.dumps(hashes, indent=2))
receipt_path = Path("/opt/ufo-source-receipt.json")
receipt = json.loads(receipt_path.read_text())
receipt.update(
    scaling_profile="env6144_batch2048_v6",
    auxiliary_source_sha256="sha256:" + hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
    scaling_profile_patch_hash="sha256:" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
)
receipt_path.write_text(json.dumps(receipt, indent=2))
