"""RLinf reset bridge around the unmodified official RoboLab RPO Flat task."""

from __future__ import annotations

import os

TASK_ID = "RPO-Flat-RLinf-v1"


class FlatResetCompat:
    """Return real terminal states; RLinf owns finished-environment resets."""

    def step(self, actions):
        self._flat_defer_reset = True
        try:
            return super().step(actions)
        finally:
            self._flat_defer_reset = False

    def _reset_idx(self, env_ids):
        if not getattr(self, "_flat_defer_reset", False):
            return super()._reset_idx(env_ids)

    def reset(self, seed=None, options=None, env_ids=None):
        if env_ids is None:
            return super().reset(seed=seed, options=options)
        import torch

        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if len(env_ids) == 0:
            return self.obs_buf, self.extras
        keep = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        keep[env_ids] = False
        old_obs = {key: value[keep].clone() for key, value in self.obs_buf.items()}
        buffers = (self.actor_obs_buffer, self.critic_obs_buffer)
        old_history = [(buf.buffer[keep].clone(), buf._num_pushes[keep].clone()) for buf in buffers]
        self._reset_idx(env_ids)
        self.obs_buf = self._get_observations()
        # Official _get_observations appends all rows. Preserve active histories
        # at the new ring pointer so a partial reset cannot append a duplicate.
        for buf, (history, pushes) in zip(buffers, old_history):
            buf._buffer[:, keep] = torch.roll(history.transpose(0, 1), buf._pointer + 1, dims=0)
            buf._num_pushes[keep] = pushes
        for key, value in old_obs.items():
            self.obs_buf[key][keep] = value
        return self.obs_buf, self.extras


def register_task():
    import gymnasium as gym

    if TASK_ID not in gym.registry:
        gym.register(id=TASK_ID, entry_point="rpo_flat_extension:RPOFlatIsaacEnv",
                     kwargs={"env_cfg_entry_point": "rpo_flat_extension:RPOFlatEnvCfg"},
                     disable_env_checker=True)


class RPOFlatEnv:
    def __new__(cls, *args, **kwargs):
        from rlinf.envs.sim.isaaclab.isaaclab_env import IsaaclabBaseEnv

        class Bridge(IsaaclabBaseEnv):
            def _make_env_function(self):
                seed, num_envs = self.seed, self.cfg.init_params.num_envs

                def make_env():
                    os.environ.pop("DISPLAY", None)
                    from isaaclab.app import AppLauncher
                    app = AppLauncher(headless=True, enable_cameras=False).app
                    import gymnasium as gym
                    from robolab.tasks.direct.base.rpo_env_cfg import RPOFlatEnvCfg

                    register_task()
                    cfg = RPOFlatEnvCfg()
                    cfg.seed = seed
                    cfg.scene.num_envs = num_envs
                    cfg.commands.debug_vis = False
                    cfg.scene.left_feet_scanner.debug_vis = False
                    cfg.scene.right_feet_scanner.debug_vis = False
                    return gym.make(TASK_ID, cfg=cfg, render_mode=None).unwrapped, app

                return make_env

            def _wrap_obs(self, obs):
                return {"states": obs["policy"], "critic_states": obs["critic"]}

        return Bridge(*args, **kwargs)


def __getattr__(name):
    if name == "RPOFlatEnvCfg":
        from robolab.tasks.direct.base.rpo_env_cfg import RPOFlatEnvCfg
        return RPOFlatEnvCfg
    if name == "RPOFlatIsaacEnv":
        from robolab.tasks.direct.base.base_env import BaseEnv

        class RPOFlatIsaacEnv(FlatResetCompat, BaseEnv):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                from rpo_wbc import RPO_LAB_JOINT_ORDER

                if tuple(self.robot.joint_names) != RPO_LAB_JOINT_ORDER:
                    raise ValueError("RPO Flat asset joint order differs from the frozen native contract")

        return RPOFlatIsaacEnv
    raise AttributeError(name)
