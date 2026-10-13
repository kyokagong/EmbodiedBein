"""Image-local alias and owned auxiliary source receipt; native UFO remains unchanged."""

import hashlib
import json
from pathlib import Path

registry = Path("/workspace/rlinf/rlinf/envs/sim/isaaclab/__init__.py")
source = registry.read_text()
original = '    "RPO-UFO-Isaac-RSI-v1": RPOUFOIsaacEnv,'
alias = '    "RPO-UFO-Isaac-RSI-slew-velocity-v5": RPOUFOIsaacEnv,'
assert source.count(original) == 1 and alias not in source
registry.write_text(source.replace(original, original + "\n" + alias))
owned = (
    "bfm_ufo_agent.py",
    "bfm_ufo_replay.py",
    "bfm_ufo_worker.py",
    "bfm_ufo_checkpoint.py",
    "bfm_ufo_policy.py",
    "bfm_ufo_artifact.py",
    "rpo_ufo_isaac.py",
    "rpo_bfm.py",
    "rpo_bfm_codec.py",
    "rpo_bfm_extension.py",
    "rpo_bfm_eval.py",
    "rpo_bfm_parallel_eval.py",
    "rpo_flat_artifact.py",
)
hashes = {
    name: hashlib.sha256((Path("/workspace/eb_runner") / name).read_bytes()).hexdigest()
    for name in owned
}
hashes["rpo_ufo_tracking_diagnostic.py"] = hashlib.sha256(
    Path("/workspace/tests/rpo_ufo_tracking_diagnostic.py").read_bytes()
).hexdigest()
receipt_path = Path("/opt/ufo-source-receipt.json")
receipt = json.loads(receipt_path.read_text())
receipt.update(
    auxiliary_profile="soft_velocity_v5",
    auxiliary_source_sha256="sha256:"
    + hashlib.sha256(
        json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest(),
    auxiliary_profile_patch_hash="sha256:"
    + hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
)
receipt_path.write_text(json.dumps(receipt, indent=2))
Path("/opt/ufo-auxiliary-source-files.json").write_text(json.dumps(hashes, indent=2))
