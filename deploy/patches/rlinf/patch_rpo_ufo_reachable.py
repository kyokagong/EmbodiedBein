"""Measured v7 image-local objective bounds and new Gym alias; upstream tree stays read-only."""

import hashlib
import json
from pathlib import Path
from code_snapshot_hash import code_snapshot_hash

registry = Path("/workspace/rlinf/rlinf/envs/sim/isaaclab/__init__.py")
source = registry.read_text()
old = '    "RPO-UFO-Isaac-RSI-v1": RPOUFOIsaacEnv,'
new = '    "RPO-UFO-Isaac-RSI-reachable-v7": RPOUFOIsaacEnv,'
assert source.count(old) == 1 and new not in source
registry.write_text(source.replace(old, old + "\n" + new))

actor = Path("/opt/ufo/humanoidverse/agents/fb_cpr_aux/agent.py")
source = actor.read_text()
old = "weight = Q_fb.abs().mean().detach() if self.cfg.train.scale_reg else 1.0"
new = "weight_raw = Q_fb.abs().mean().detach() if self.cfg.train.scale_reg else Q_fb.new_tensor(1.0)\n            weight = weight_raw.clamp(max=50.0)"
assert source.count(old) == 1
source = source.replace(old, new)
old = '                "actor_loss": actor_loss.detach(),'
assert source.count(old) == 1
actor.write_text(source.replace(old, old + '\n                "actor_reg_weight_raw": weight_raw,\n                "actor_reg_weight_applied": weight,'))

critic = Path("/opt/ufo/humanoidverse/agents/fb_cpr/agent.py")
source = critic.read_text()
old = "reward = self._model._discriminator.compute_reward(obs=obs, z=z)"
assert source.count(old) == 1
source = source.replace(old, "reward_raw = self._model._discriminator.compute_reward(obs=obs, z=z)\n                reward = reward_raw.clamp(-2.0, 2.0)")
old = '                "mean_disc_reward": reward.mean().detach(),'
assert source.count(old) == 1
critic.write_text(source.replace(old, old + '\n                "mean_disc_reward_raw": reward_raw.mean().detach(),\n                "disc_reward_clip_fraction": (reward_raw.abs() > 2.0).float().mean(),'))

files_path = Path("/opt/ufo-auxiliary-source-files.json")
hashes = json.loads(files_path.read_text())
hashes["rpo_ufo_motion.py"] = ""
for name in hashes:
    path = Path("/workspace/tests/rpo_ufo_tracking_diagnostic.py") if name == "rpo_ufo_tracking_diagnostic.py" else Path("/workspace/eb_runner") / name
    hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
files_path.write_text(json.dumps(hashes, indent=2))
receipt_path = Path("/opt/ufo-source-receipt.json")
receipt = json.loads(receipt_path.read_text())
receipt["ufo"] = {**receipt["ufo"], "upstream_content_hash": receipt["ufo"]["content_hash"],
    "content_hash": code_snapshot_hash(Path("/opt/ufo")), "scope": "pinned native agent plus measured v7 regularization/reward bounds"}
receipt.update(control_profile="reachable_v2", auxiliary_profile="soft_velocity_v7", optimizer_profile="stabilized_v7",
    auxiliary_source_sha256="sha256:" + hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
    stabilization_patch_hash="sha256:" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
receipt_path.write_text(json.dumps(receipt, indent=2))
