from pathlib import Path
import sys


root = Path(sys.argv[1])
registry = root / "rlinf/envs/sim/isaaclab/__init__.py"
policy = root / "rlinf/models/embodiment/mlp_policy/mlp_policy.py"
model_factory = root / "rlinf/models/embodiment/mlp_policy/__init__.py"


def replace(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    if text.count(old) != 1:
        raise SystemExit(f"expected one patch site in {path}: {old[:60]!r}")
    path.write_text(text.replace(old, new))


replace(
    registry,
    "from .tasks.berkeley_humanoid import BerkeleyHumanoidEnv\nfrom .tasks.stack_cube import IsaaclabStackCubeEnv\n\nREGISTER_ISAACLAB_ENVS = {\n    \"Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Rewarded-v0\": IsaaclabStackCubeEnv,\n    \"Velocity-Berkeley-Humanoid-Lite-v0\": BerkeleyHumanoidEnv,\n    \"Velocity-Berkeley-Humanoid-Lite-Biped-v0\": BerkeleyHumanoidEnv,\n}\n",
    "from .tasks.stack_cube import IsaaclabStackCubeEnv\nfrom rpo_wbc_extension import RPOWBCEnv\n\nREGISTER_ISAACLAB_ENVS = {\n    \"Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Rewarded-v0\": IsaaclabStackCubeEnv,\n    \"RPO-WBC-WalkUpper-v0\": RPOWBCEnv,\n    \"RPO-WBC-StandWalk-v1\": RPOWBCEnv,\n}\ntry:\n    from .tasks.berkeley_humanoid import BerkeleyHumanoidEnv\nexcept ModuleNotFoundError:\n    BerkeleyHumanoidEnv = None\nif BerkeleyHumanoidEnv is not None:\n    REGISTER_ISAACLAB_ENVS.update({\n        \"Velocity-Berkeley-Humanoid-Lite-v0\": BerkeleyHumanoidEnv,\n        \"Velocity-Berkeley-Humanoid-Lite-Biped-v0\": BerkeleyHumanoidEnv,\n    })\n",
)
replace(policy, "self.value_head = ValueHead(\n                obs_dim,", "self.value_head = ValueHead(\n                self.critic_obs_dim,")
replace(
    model_factory,
    '            q_head_type=cfg.get("q_head_type", "default"),\n        )\n\n    return model',
    '            q_head_type=cfg.get("q_head_type", "default"),\n'
    '            critic_obs_dim=(cfg.get("critic_obs_dim", cfg.obs_dim)\n'
    '                            if cfg.get("policy_setup") == "rpo-v2-amp-locomotion-23dof" else cfg.obs_dim),\n'
    '        )\n\n    return model',
)
replace(
    policy,
    'return {"states": env_obs["states"].to(device)}',
    'return {"states": env_obs["states"].to(device), "critic_states": (env_obs.get("critic_states")'
    ' if env_obs.get("critic_states") is not None and env_obs["critic_states"].shape[-1] == self.critic_obs_dim'
    ' else env_obs["states"]).to(device)}',
)
replace(policy, "values = self.value_head(states)\n                output_dict.update(values=values)",
        'values = self.value_head(forward_inputs.get("critic_states", states))\n                output_dict.update(values=values)')
replace(policy, "        use_rsample: bool = False,\n    ) -> tuple", "        use_rsample: bool = False,\n        critic_states: torch.Tensor | None = None,\n    ) -> tuple")
replace(policy, "chunk_values = self.value_head(states)\n        else:", "chunk_values = self.value_head(critic_states if critic_states is not None else states)\n        else:")
replace(
    policy,
    'env_obs["states"], mode=mode, calculate_values=calculate_values\n        )',
    'env_obs["states"], mode=mode, calculate_values=calculate_values, critic_states=env_obs["critic_states"]\n        )',
)
replace(policy, 'forward_inputs["states"] = env_obs["states"]',
        'forward_inputs["states"] = env_obs["states"]\n            forward_inputs["critic_states"] = env_obs["critic_states"]')

patched = policy.read_text()
assert "self.value_head(self.critic_obs_dim" not in patched
assert 'forward_inputs.get("critic_states", states)' in patched
assert '"RPO-WBC-WalkUpper-v0": RPOWBCEnv' in registry.read_text()
assert '"RPO-WBC-StandWalk-v1": RPOWBCEnv' in registry.read_text()
