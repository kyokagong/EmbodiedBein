"""Record the native rollout option; keep pinned UFO algorithm source unchanged."""
import hashlib
import json
from pathlib import Path

from bfm_ufo_agent import agent_config

config = agent_config(
    {"architecture_profile": "native_fb_v1", "rollout_expert_trajectories": False}, device="cpu"
)
assert not config.train.rollout_expert_trajectories
assert config.train.use_mix_rollout and config.train.expert_asm_ratio == .6
assert config.train.batch_size == 1024
assert config.model.archi.f.hidden_dim == 2048 and config.model.archi.f.hidden_layers == 6
assert config.model.archi.discriminator.hidden_dim == 1024 and config.model.archi.b.hidden_dim == 256
receipt_path = Path("/opt/ufo-source-receipt.json")
receipt = json.loads(receipt_path.read_text())
receipt["rollout_profile_patch_hash"] = "sha256:" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
receipt_path.write_text(json.dumps(receipt, indent=2))
