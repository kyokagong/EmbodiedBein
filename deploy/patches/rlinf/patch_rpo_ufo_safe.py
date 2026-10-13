"""Image-local physical profile; native UFO mathematical source remains unchanged."""
import hashlib
import json
from pathlib import Path


root = Path("/workspace/rlinf")
path = root / "rlinf/envs/sim/isaaclab/__init__.py"
source = path.read_text()
old = '    "RPO-UFO-Isaac-RSI-v1": RPOUFOIsaacEnv,'
assert source.count(old) == 1
path.write_text(source.replace(old, old + '\n    "RPO-UFO-Isaac-RSI-safe-v2": RPOUFOIsaacEnv,'))
path = root / "rlinf/data/schema/embodied_types.py"
source = path.read_text()
old = '"ufo_aux_rewards", "episode_id", "step_id")'
assert source.count(old) == 1
path.write_text(source.replace(old, '"ufo_aux_rewards", "ufo_action_trace", "episode_id", "step_id")'))
receipt_path = Path("/opt/ufo-source-receipt.json")
receipt = json.loads(receipt_path.read_text())
receipt["learning_profile_patch_hash"] = "sha256:" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
receipt_path.write_text(json.dumps(receipt, indent=2))
