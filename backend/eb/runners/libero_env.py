"""Bind the existing upstream timeout bootstrap helper in LIBERO env workers."""
import json
import os
from pathlib import Path

import numpy as np
from rlinf.envs.utils import get_env_attr
import rlinf.workers.env.env_worker as env_module
from rlinf.utils.metric_utils import add_time_limit_bootstrap_rewards


def install_env_compat():
    if not hasattr(env_module, "add_time_limit_bootstrap_rewards"):
        env_module.add_time_limit_bootstrap_rewards = add_time_limit_bootstrap_rewards


class LiberoEnvWorker(env_module.EnvWorker):
    def __init__(self, cfg):
        install_env_compat()
        super().__init__(cfg)

    def env_evaluate_step(self, raw_actions, stage_id):
        """Record actual completed LIBERO cases before upstream reduces them to means."""
        raw_plan = os.environ.get("EB_EVAL_PLAN")
        if not raw_plan:
            return super().env_evaluate_step(raw_actions, stage_id)
        plan_hash = json.loads(raw_plan)["hash"]
        if self.cfg.env.eval.auto_reset:
            raise ValueError("platform case evidence requires auto_reset=False")
        previous = self.eval_prev_done[stage_id].clone()
        env = self.eval_env_list[stage_id]
        reset_ids = np.asarray(get_env_attr(env, "reset_state_ids")).copy()
        output, metrics = super().env_evaluate_step(raw_actions, stage_id)
        newly_done = (self.eval_prev_done[stage_id] & ~previous).nonzero(as_tuple=True)[0].tolist()
        if not newly_done:
            return output, metrics
        success = metrics.get("success_once")
        if success is None or len(success) != len(newly_done):
            raise ValueError("LIBERO completed cases lack per-case success_once")
        task_ids, trial_ids = get_env_attr(env, "_get_task_and_trial_ids_from_reset_state_ids")(reset_ids[newly_done])
        suite = str(self.cfg.env.eval.task_suite_name)
        target = Path(os.environ["EB_OUTPUT_DIR"]) / f"eval-episodes-rank{self._rank}.jsonl"
        with target.open("a") as stream:
            for index, env_index in enumerate(newly_done):
                value = float(success[index])
                if value not in (0.0, 1.0):
                    raise ValueError("LIBERO case success must be binary")
                row = {"case_id": f"{suite}:task{int(task_ids[index])}:init{int(trial_ids[index])}",
                       "run_id": os.environ["EB_RUN_ID"], "plan_hash": plan_hash,
                       "task_id": int(task_ids[index]), "init_state_id": int(trial_ids[index]),
                       "reset_state_id": int(reset_ids[env_index]),
                       "env_seed": int(get_env_attr(env, "seed")), "policy_seed": int(self.cfg.actor.seed),
                       "success": bool(value), "status": "success" if value else "task_failure"}
                if self.cfg.env.eval.video_cfg.save_video:
                    video_count = get_env_attr(env, "video_cnt")
                    if type(video_count) is int and video_count >= 0:
                        row["video_rel"] = f"seed_{row['env_seed']}/{video_count}.mp4"
                for field in ("success_at_end", "episode_len", "return"):
                    if field in metrics and len(metrics[field]) == len(newly_done):
                        row[field] = float(metrics[field][index])
                stream.write(json.dumps(row, separators=(",", ":")) + "\n")
        return output, metrics
