"""RLInf RoboTwin EnvWorker with platform-owned per-case evidence."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from omegaconf import open_dict
from rlinf.envs.utils import get_env_attr
import rlinf.workers.env.env_worker as env_module

from robotwin_eval import _load_eval_plan, build_case_record


def pad_worker_wave(cases, slots, capacity):
    if not cases or len(cases) != len(slots) or len(cases) > capacity:
        raise ValueError("RoboTwin worker wave 容量无效")
    padding = capacity - len(cases)
    return list(cases) + [cases[-1]] * padding, list(slots) + [None] * padding


class RobotwinEnvWorker(env_module.EnvWorker):
    def init_worker(self):
        """Initialize the Ray worker without creating a task renderer yet."""
        self.broadcast(True, groups=[(self._group_name, list(range(self._world_size)))])
        self.update_env_cfg()
        self.eval_env_list = []
        self._eb_task_id = None
        self._eb_wave_slots = []
        self._eb_env_capacity = self.eval_num_envs_per_stage
        self._eb_plan = _load_eval_plan()
        if not self._eb_plan:
            raise ValueError("RoboTwin evaluation requires a frozen plan")

    @staticmethod
    def _base_env(env):
        while getattr(env, "env", None) is not None:
            env = env.env
        return env

    def configure_eval_wave(self, cases, active_slots, task_id, max_episode_steps):
        if (not cases or tuple(active_slots) != tuple(range(len(cases)))
                or any(case.get("task_id") != task_id for case in cases)):
            raise ValueError("RoboTwin wave case/slot mapping is invalid")
        total = len(cases)
        start, stop = self.eval_rank_slice(total, self._rank, self._world_size)
        local_count = stop - start
        local_cases = cases[start:stop]
        local_slots = list(active_slots[start:stop])
        local_cases, local_slots = pad_worker_wave(
            local_cases, local_slots, self._eb_env_capacity)
        seeds = [int(case["seed"]) for case in local_cases]
        with open_dict(self.cfg.env.eval):
            self.cfg.env.eval.total_num_envs = self._eb_env_capacity * self._world_size
            self.cfg.env.eval.max_episode_steps = max_episode_steps
            self.cfg.env.eval.max_steps_per_rollout_epoch = max_episode_steps
            self.cfg.env.eval.fixed_reset_state_ids = seeds
            self.cfg.env.eval.task_config.task_name = task_id
            self.cfg.env.eval.task_config.step_lim = max_episode_steps
        self.eval_num_envs_per_stage = self._eb_env_capacity
        self.eval_batch_size = self._eb_env_capacity * self._world_size
        self.n_eval_chunk_steps = (
            max_episode_steps + int(self.model_cfg.num_action_chunks) - 1
        ) // int(self.model_cfg.num_action_chunks)
        reusable = self._eb_task_id == task_id and self.eval_env_list and all(
            int(get_env_attr(env, "num_envs", -1)) == self._eb_env_capacity
            for env in self.eval_env_list
        )
        if reusable:
            for env in self.eval_env_list:
                base_env = self._base_env(env)
                base_env.reset_state_ids = torch.as_tensor(seeds, dtype=torch.long)
                base_env._reset_metrics()
                base_env.is_start = True
        else:
            self.close_eval_task()
            env_cls = env_module.get_env_cls(self.cfg.env.eval.env_type, self.cfg.env.eval)
            self.eval_env_list = self._setup_env_and_wrappers(
                env_cls=env_cls,
                env_cfg=self.cfg.env.eval,
                num_envs_per_stage=self._eb_env_capacity,
            )
        self.eval_prev_done = [torch.zeros(self._eb_env_capacity, dtype=torch.bool)
                               for _ in range(self.stage_num)]
        self._eb_task_id = task_id
        self._eb_wave_slots = local_slots
        self._eb_wave_id = int(cases[0]["wave_id"])
        self._eb_active_count = local_count
        self._eb_sim_steps = {}
        return {"rank":self._rank, "active_slots":local_slots, "envs":local_count}

    def close_eval_task(self, timeout_s=30.0):
        del timeout_s  # VectorEnv.close is synchronous.
        for env in self.eval_env_list:
            close = get_env_attr(env, "offload") or get_env_attr(env, "close")
            if callable(close):
                close()
        self.eval_env_list = []
        self._eb_task_id = None
        self._eb_wave_slots = []
        return {"rank":self._rank, "children_alive":0,
                "renderers_alive":0, "pending_messages":0}

    def env_evaluate_step(self, raw_actions, stage_id):
        plan = self._eb_plan
        if self.cfg.env.eval.auto_reset or self.cfg.env.eval.ignore_terminations:
            raise ValueError("platform RoboTwin evidence requires fixed reset without ignored termination")
        previous = self.eval_prev_done[stage_id].clone()
        env = self.eval_env_list[stage_id]
        reset_ids = np.asarray(get_env_attr(env, "reset_state_ids")).copy()
        output, metrics = super().env_evaluate_step(raw_actions, stage_id)
        infos = output.env_infos or {}
        for field in ("chunk_executed", "sim_steps_executed", "sim_timestep"):
            if field not in infos or len(infos[field]) != len(reset_ids):
                raise ValueError(f"RoboTwin 缺少精确 {field} 证据")
        if not hasattr(self, "_eb_sim_steps"):
            self._eb_sim_steps = {}
        totals = self._eb_sim_steps.setdefault(stage_id, np.zeros(len(reset_ids), dtype=np.int64))
        totals += np.asarray(infos["sim_steps_executed"], dtype=np.int64)
        newly_done = (self.eval_prev_done[stage_id] & ~previous).nonzero(as_tuple=True)[0].tolist()
        if not newly_done:
            return output, metrics
        active_metric_indexes = [index for index, env_index in enumerate(newly_done)
                                 if env_index < self._eb_active_count]
        for key, value in list(metrics.items()):
            if hasattr(value, "__len__") and len(value) == len(newly_done):
                metrics[key] = value[active_metric_indexes]
        newly_done = [newly_done[index] for index in active_metric_indexes]
        if not newly_done:
            return output, metrics
        success = metrics.get("success_once")
        lengths = metrics.get("episode_len")
        if success is None or lengths is None or len(success) != len(newly_done) or len(lengths) != len(newly_done):
            raise ValueError("RoboTwin completed cases lack per-case success or length")
        target = Path(os.environ["EB_OUTPUT_DIR"]) / f"eval-episodes-rank{self._rank}.jsonl"
        with target.open("a") as stream:
            for index, env_index in enumerate(newly_done):
                value = float(success[index])
                if value not in (0.0, 1.0):
                    raise ValueError("RoboTwin case success_once must be binary")
                requested = int(self.cfg.rollout.model.num_action_chunks)
                executed = int(infos["chunk_executed"][env_index])
                env_steps = int(lengths[index]) - requested + executed
                instruction = str(output.obs["task_descriptions"][env_index])
                sim_steps = int(totals[env_index])
                row = build_case_record(plan, int(reset_ids[env_index]), bool(value), env_steps,
                    requested, executed, sim_steps,
                    float(sim_steps) * float(infos["sim_timestep"][env_index]),
                    instruction, os.environ["EB_RUN_ID"], self._eb_task_id)
                row.update(worker_rank=self._rank,
                           env_slot=self._eb_wave_slots[env_index],
                           wave_id=self._eb_wave_id)
                for field in ("success_at_end", "return", "reward"):
                    if field in metrics and len(metrics[field]) == len(newly_done):
                        row[field] = float(metrics[field][index])
                stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        return output, metrics
