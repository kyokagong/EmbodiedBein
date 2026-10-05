"""Apply AMP-only hooks to the image copy of the pinned RLinf tree."""

import sys
from pathlib import Path


root = Path(sys.argv[1])


def replace(relative, old, new):
    path = root / relative
    source = path.read_text()
    if source.count(old) != 1:
        raise SystemExit(f"AMP patch expected exactly one site in {relative}: {old[:72]!r}")
    path.write_text(source.replace(old, new))


replace(
    "rlinf/data/schema/embodied_types.py",
    '            "states": states,\n',
    '            "states": states,\n'
    '            **({"critic_states": obs["critic_states"]} if "critic_states" in obs else {}),\n',
)
replace(
    "rlinf/models/embodiment/mlp_policy/mlp_policy.py",
    '        device = next(self.parameters()).device\n'
    '        return {"states": env_obs["states"].to(device), "critic_states": (env_obs.get("critic_states")',
    '        device = next(self.parameters()).device\n'
    '        if self.critic_obs_dim != self.obs_dim and (env_obs.get("critic_states") is None\n'
    '                or env_obs["critic_states"].shape[-1] != self.critic_obs_dim):\n'
    '            raise ValueError(f"critic_states must have {self.critic_obs_dim} features")\n'
    '        return {"states": env_obs["states"].to(device), "critic_states": (env_obs.get("critic_states")',
)
replace(
    "rlinf/models/embodiment/mlp_policy/mlp_policy.py",
    '        if self.independent_std:\n            self.actor_logstd = nn.Parameter(',
    '        import os\n'
    '        if os.getenv("EB_RPO_AMP_INITIAL_POSE") == "official_a1_stand_v1" and obs_dim == 780 and action_dim == 23:\n'
    '            from rpo_wbc import AMP_STAND_ACTION\n'
    '            with torch.no_grad():\n'
    '                self.actor_mean.bias.copy_(torch.tensor(AMP_STAND_ACTION))\n'
    '        if self.independent_std:\n            self.actor_logstd = nn.Parameter(',
)
replace(
    "rlinf/models/embodiment/mlp_policy/mlp_policy.py",
    '        if action_scale is not None:\n',
    '        initial_logstd = os.getenv("EB_RPO_AMP_INITIAL_LOGSTD")\n'
    '        if initial_logstd is not None and obs_dim == 780 and action_dim == 23:\n'
    '            value = float(initial_logstd)\n'
    '            if not -4.0 <= value <= 0.0:\n'
    '                raise ValueError("RPO AMP initial logstd is outside the approved range")\n'
    '            with torch.no_grad():\n'
    '                self.actor_logstd.fill_(value)\n'
    '        if action_scale is not None:\n',
)
replace(
    "rlinf/envs/sim/isaaclab/__init__.py",
    "from rpo_wbc_extension import RPOWBCEnv\n",
    "from rpo_wbc_extension import RPOWBCEnv\nfrom rpo_amp_extension import RPOAMPEnv\n",
)
replace(
    "rlinf/envs/sim/isaaclab/__init__.py",
    '    "RPO-WBC-StandWalk-v1": RPOWBCEnv,\n',
    '    "RPO-WBC-StandWalk-v1": RPOWBCEnv,\n'
    '    "RPO-AMP-StandWalk-v1": RPOAMPEnv,\n',
)
replace(
    "rlinf/envs/sim/isaaclab/isaaclab_env.py",
    "        infos = self._record_metrics(\n"
    "            step_reward, terminations, {}\n"
    "        )  # return infos is useless",
    "        infos = self._record_metrics(\n"
    "            step_reward, terminations,\n"
    '            infos if self.isaaclab_env_id == "RPO-AMP-StandWalk-v1" else {},\n'
    "        )",
)
replace(
    "rlinf/workers/env/env_worker.py",
    "            last_step_truncations = env_output.truncations[:, -1]\n",
    "            last_step_truncations = env_output.truncations[:, -1]\n"
    '            if self.model_cfg.get("policy_setup") == "rpo-v2-amp-locomotion-23dof":\n'
    "                last_step_truncations = last_step_truncations & ~env_output.terminations[:, -1]\n",
)
replace(
    "rlinf/workers/env/env_worker.py",
    "                    env_output, env_info, chunk_step_payload = self.env_interact_step(\n"
    "                        policy_output.actions,\n"
    "                        stage_id,\n"
    "                    )\n"
    "                    # Emulated observation latency",
    "                    env_output, env_info, chunk_step_payload = self.env_interact_step(\n"
    "                        policy_output.actions,\n"
    "                        stage_id,\n"
    "                    )\n"
    '                    if self.model_cfg.get("policy_setup") == "rpo-v2-amp-locomotion-23dof":\n'
    "                        amp_info = env_output.env_infos or {}\n"
    '                        amp_info = amp_info.get("final_info", amp_info)\n'
    '                        if "amp_window" not in amp_info or "amp_valid" not in amp_info:\n'
    '                            raise ValueError("RPO AMP environment did not return a motion window")\n'
    "                        self.trajectory_builders[stage_id].forward_inputs[-1].update({\n"
    '                            "amp_window": amp_info["amp_window"].clone().cpu(),\n'
    '                            "amp_valid": amp_info["amp_valid"].reshape(-1, 1).clone().cpu(),\n'
    "                        })\n"
    "                    # Emulated observation latency",
)
replace(
    "examples/embodiment/train_embodied_agent.py",
    "            actor_worker_cls = EmbodiedFSDPActor\n",
    "            actor_worker_cls = EmbodiedFSDPActor\n"
    '            if cfg.actor.model.get("policy_setup") == "rpo-v2-amp-locomotion-23dof":\n'
    '                if cfg.runner.get("use_training_pipeline", False) or cfg.runner.get("overlap_env_bootstrap", False):\n'
    '                    raise ValueError("RPO AMP requires synchronous rollout and bootstrap")\n'
    "                from rpo_amp_worker import RPOAMPActor\n"
    "                actor_worker_cls = RPOAMPActor\n",
)
