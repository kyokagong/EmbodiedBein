"""Add RPO BFM hooks to the image-local RLinf source copy."""

from pathlib import Path
import sys

root = Path(sys.argv[1])


def replace(relative, old, new):
    path = root / relative
    source = path.read_text()
    if source.count(old) != 1:
        raise SystemExit(f"BFM patch expected one site in {relative}: {old[:72]!r}")
    path.write_text(source.replace(old, new))


replace("rlinf/envs/sim/isaaclab/__init__.py",
        "from rpo_flat_extension import RPOFlatEnv\n",
        "from rpo_flat_extension import RPOFlatEnv\nfrom rpo_bfm_extension import RPOBFMEnv\n"
        "from rpo_tracking_extension import RPOTrackingEnv\n")
replace("rlinf/envs/sim/isaaclab/__init__.py",
        '    "RPO-Flat-RLinf-v1": RPOFlatEnv,\n',
        '    "RPO-Flat-RLinf-v1": RPOFlatEnv,\n    "RPO-BFM-Pretrain-v1": RPOBFMEnv,\n'
        '    "RPO-Tracking-Teacher-v1": RPOTrackingEnv,\n')
replace("rlinf/models/embodiment/mlp_policy/__init__.py",
        '    iql_config = cfg.get("iql_config", None)\n',
        '    if cfg.get("policy_setup") == "rpo-v2-tracking-teacher":\n'
        '        from rpo_tracking_policy import RPOTrackingPolicy\n'
        '        return RPOTrackingPolicy(cfg)\n'
        '    if cfg.get("policy_setup") == "rpo-v2-bfm-zero":\n'
        '        from bfm_zero_policy import BFMZeroPolicy\n'
        '        return BFMZeroPolicy(cfg)\n'
        '    iql_config = cfg.get("iql_config", None)\n')
replace("rlinf/workers/rollout/hf/huggingface_worker.py",
        '            rollout_model_config.model_path = self.cfg.rollout.model.model_path\n',
        '            rollout_model_config.model_path = self.cfg.rollout.model.model_path\n'
        '            if rollout_model_config.get("policy_setup") == "rpo-v2-bfm-zero":\n'
        '                rollout_model_config.seed = int(self.cfg.actor.seed) + 1009 * self._rank\n')
replace("examples/embodiment/train_embodied_agent.py",
        '    if cfg.algorithm.loss_type == "embodied_sac":\n',
        '    if cfg.actor.model.get("policy_setup") == "rpo-v2-bfm-zero":\n'
        '        from bfm_zero_worker import RPOBFMActor\n'
        '        actor_worker_cls = RPOBFMActor\n'
        '    elif cfg.algorithm.loss_type == "embodied_sac":\n')
replace("rlinf/data/schema/embodied_types.py",
        '            "task_descriptions": task_descriptions,\n'
        '        }\n',
        '            "task_descriptions": task_descriptions,\n'
        '            **{key: obs[key] for key in ("last_action", "history_actor", '
        '"privileged_state", "commands", "bfm_aux_rewards", "episode_id", "step_id") '
        'if key in obs},\n'
        '        }\n')

replace("examples/embodiment/train_embodied_agent.py",
        'if cfg.actor.model.get("policy_setup") == "rpo-v2-flat-23dof":\n',
        'if cfg.actor.model.get("policy_setup") in {"rpo-v2-flat-23dof", "rpo-v2-tracking-teacher"}:\n')
replace("examples/embodiment/train_embodied_agent.py",
        'cfg.actor.model.get("policy_setup") == "rpo-v2-flat-23dof" and (',
        'cfg.actor.model.get("policy_setup") in {"rpo-v2-flat-23dof", "rpo-v2-tracking-teacher"} and (')
replace("rlinf/runners/embodied_runner.py",
        '            return self.run_pipeline()\n\n'
        '        start_step = self.global_step\n'
        '        start_time = time.time()\n'
        '        for _step in range(start_step, self.max_steps):\n'
        '            # set global step\n',
        '            return self.run_pipeline()\n\n'
        '        start_step = self.global_step\n'
        '        start_time = time.time()\n'
        '        for _step in range(start_step, self.max_steps):\n'
        '            if (self.cfg.actor.model.get("policy_setup") == "rpo-v2-tracking-teacher"\n'
        '                    and time.time() - start_time >= 2900):\n'
        '                if self.global_step % self.cfg.runner.save_interval:\n'
        '                    self._save_checkpoint()\n'
        '                print(f"Tracking teacher wall budget reached at update {self.global_step}", flush=True)\n'
        '                break\n'
        '            # set global step\n')
replace("rlinf/workers/env/env_worker.py",
        '{"rpo-v2-amp-locomotion-23dof", "rpo-v2-flat-23dof"}',
        '{"rpo-v2-amp-locomotion-23dof", "rpo-v2-flat-23dof", "rpo-v2-tracking-teacher"}')
replace("rlinf/envs/sim/isaaclab/isaaclab_env.py",
        '{"RPO-AMP-StandWalk-v1", "RPO-Flat-RLinf-v1"}',
        '{"RPO-AMP-StandWalk-v1", "RPO-Flat-RLinf-v1", "RPO-Tracking-Teacher-v1"}')
