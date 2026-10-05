"""Scoped Flat hooks, applied after WBC/AMP to the image's RLinf copy."""

from pathlib import Path
import sys

root = Path(sys.argv[1])


def replace(relative, old, new):
    path = root / relative
    source = path.read_text()
    if source.count(old) != 1:
        raise SystemExit(f"Flat patch expected one site in {relative}: {old[:72]!r}")
    path.write_text(source.replace(old, new))


replace("rlinf/envs/sim/isaaclab/__init__.py",
        "from rpo_amp_extension import RPOAMPEnv\n",
        "from rpo_amp_extension import RPOAMPEnv\nfrom rpo_flat_extension import RPOFlatEnv\n")
replace("rlinf/envs/sim/isaaclab/__init__.py",
        '    "RPO-AMP-StandWalk-v1": RPOAMPEnv,\n',
        '    "RPO-AMP-StandWalk-v1": RPOAMPEnv,\n    "RPO-Flat-RLinf-v1": RPOFlatEnv,\n')
replace("rlinf/models/embodiment/mlp_policy/__init__.py",
        '    iql_config = cfg.get("iql_config", None)\n',
        '    if cfg.get("policy_setup") == "rpo-v2-flat-23dof":\n'
        '        from rpo_flat_policy import RPOFlatPolicy as MLPPolicy\n'
        '    iql_config = cfg.get("iql_config", None)\n')
replace("rlinf/models/embodiment/mlp_policy/__init__.py",
        'if cfg.get("policy_setup") == "rpo-v2-amp-locomotion-23dof" else cfg.obs_dim),',
        'if cfg.get("policy_setup") in {"rpo-v2-amp-locomotion-23dof", "rpo-v2-flat-23dof"} else cfg.obs_dim),')
replace("examples/embodiment/train_embodied_agent.py",
        '    use_training_pipeline = bool(cfg.runner.get("use_training_pipeline", False))\n',
        '    use_training_pipeline = bool(cfg.runner.get("use_training_pipeline", False))\n'
        '    if cfg.actor.model.get("policy_setup") == "rpo-v2-flat-23dof" and (\n'
        '        use_training_pipeline or cfg.runner.get("overlap_env_bootstrap", False)\n'
        '    ):\n'
        '        raise ValueError("RPO Flat requires synchronous rollout and bootstrap")\n')
replace("examples/embodiment/train_embodied_agent.py",
        '            actor_worker_cls = EmbodiedFSDPActor\n',
        '            actor_worker_cls = EmbodiedFSDPActor\n'
        '            if cfg.actor.model.get("policy_setup") == "rpo-v2-flat-23dof":\n'
        '                from rpo_flat_worker import RPOFlatActor\n'
        '                actor_worker_cls = RPOFlatActor\n')
replace("rlinf/workers/env/env_worker.py",
        '            if self.model_cfg.get("policy_setup") == "rpo-v2-amp-locomotion-23dof":\n'
        '                last_step_truncations = last_step_truncations & ~env_output.terminations[:, -1]\n',
        '            if self.model_cfg.get("policy_setup") in {"rpo-v2-amp-locomotion-23dof", "rpo-v2-flat-23dof"}:\n'
        '                last_step_truncations = last_step_truncations & ~env_output.terminations[:, -1]\n')
replace("rlinf/envs/sim/isaaclab/isaaclab_env.py",
        '            infos if self.isaaclab_env_id == "RPO-AMP-StandWalk-v1" else {},\n',
        '            infos if self.isaaclab_env_id in {"RPO-AMP-StandWalk-v1", "RPO-Flat-RLinf-v1"} else {},\n')
replace("rlinf/workers/actor/embodied_fsdp_actor_worker.py",
        '        loss, metrics_data = policy_loss(**loss_kwargs)\n',
        '        if hasattr(self, "_record_flat_kl"):\n'
        '            self._record_flat_kl(output_dict["flat_kl"], loss_mask)\n'
        '            loss_kwargs["rpo_flat_value_mse"] = True\n'
        '        loss, metrics_data = policy_loss(**loss_kwargs)\n')
replace("rlinf/algorithms/losses.py",
        '    value_loss_original = huber_loss(\n'
        '        returns - values, huber_delta\n'
        '    )  # [bsz, ] | [bsz, chunk-step]\n'
        '    value_loss_clipped = huber_loss(\n'
        '        returns - value_pred_clipped, huber_delta\n'
        '    )  # [bsz, ] | [bsz, chunk-step]\n'
        '    value_loss = torch.max(value_loss_original, value_loss_clipped)\n',
        '    if kwargs.get("rpo_flat_value_mse", False):\n'
        '        from rpo_flat_worker import clipped_value_mse\n'
        '        value_loss = clipped_value_mse(values, returns, prev_values, value_clip)\n'
        '    else:\n'
        '        value_loss_original = huber_loss(returns - values, huber_delta)\n'
        '        value_loss_clipped = huber_loss(returns - value_pred_clipped, huber_delta)\n'
        '        value_loss = torch.max(value_loss_original, value_loss_clipped)\n')
