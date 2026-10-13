"""Image-local environment alias and controller source binding; no native loss changes."""
import hashlib
import json
from pathlib import Path

registry = Path("/workspace/rlinf/rlinf/envs/sim/isaaclab/__init__.py")
source = registry.read_text()
original = '    "RPO-UFO-Isaac-RSI-v1": RPOUFOIsaacEnv,'
assert source.count(original) == 1
alias = '    "RPO-UFO-Isaac-RSI-slew-v4": RPOUFOIsaacEnv,'
assert alias not in source
registry.write_text(source.replace(original, original + "\n" + alias))
receipt_path = Path("/opt/ufo-source-receipt.json")
receipt = json.loads(receipt_path.read_text())
receipt["action_control_source_sha256"] = "sha256:" + hashlib.sha256(
    Path("/workspace/eb_runner/rpo_bfm_codec.py").read_bytes()).hexdigest()
receipt["action_control_profile_patch_hash"] = "sha256:" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
receipt_path.write_text(json.dumps(receipt, indent=2))
