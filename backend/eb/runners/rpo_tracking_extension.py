"""Closed-loop residual teacher; training and resets remain owned by RLinf."""

import os
from pathlib import Path

from rpo_tracking_reference import GYM_ID


def register_task():
    import gymnasium as gym

    if GYM_ID not in gym.registry:
        gym.register(
            id=GYM_ID,
            entry_point="rpo_tracking_extension:RPOTrackingIsaacEnv",
            kwargs={"env_cfg_entry_point": "rpo_tracking_extension:RPOTrackingEnvCfg"},
            disable_env_checker=True,
        )


class RPOTrackingEnv:
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

                    register_task()
                    cfg = __getattr__("RPOTrackingEnvCfg")()
                    cfg.seed, cfg.scene.num_envs = seed, num_envs
                    return gym.make(GYM_ID, cfg=cfg, render_mode=None).unwrapped, app

                return make_env

            def _wrap_obs(self, obs):
                return {"states": obs["policy"], "critic_states": obs["critic"]}

            def _record_metrics(self, step_reward, terminations, infos):
                infos = super()._record_metrics(step_reward, terminations, infos)
                # RLinf's generic positive-reward flag is not tracking qualification.
                infos["episode"].pop("success_once", None)
                return infos

        return Bridge(*args, **kwargs)


def __getattr__(name):
    import rpo_bfm_extension as bfm

    if name == "RPOTrackingEnvCfg":
        return bfm.RPOBFMEnvCfg
    if name != "RPOTrackingIsaacEnv":
        raise AttributeError(name)
    import numpy as np
    import torch
    from rpo_bfm import ACTION_GAIN, DEFAULT_Q
    from rpo_bfm_codec import encode_state, quaternion_matrix
    from rpo_bfm_motion_probe import retarget_reference
    from rpo_tracking_reference import reference_receipt

    class RPOTrackingIsaacEnv(bfm.RPOBFMIsaacEnv):
        def _ensure_tracking(self):
            self._ensure_bfm_buffers()
            if hasattr(self, "reference_q"):
                return
            root = Path(os.environ["EB_TRACKING_REFERENCE_ROOT"])
            sha = os.environ["EB_TRACKING_ASSET_SHA256"]
            reference_receipt(root, os.environ["EB_TRACKING_REFERENCE_SHA256"], sha)
            _, arrays = retarget_reference(root, sha)
            self.reference_ids = list(arrays)

            def stack(key):
                return torch.as_tensor(
                    np.stack([a[key] for a in arrays.values()]), device=self.device, dtype=torch.float32
                )

            self.reference_q, self.reference_dq = stack("q"), stack("dq")
            self.reference_pos, self.reference_quat = stack("root_pos"), stack("root_quat")
            self.reference_vel = stack("body_lin_vel")[:, :, 0]
            self.reference_omega = stack("body_ang_vel")[:, :, 0]
            rotations = quaternion_matrix(self.reference_quat)
            gravity = -rotations[..., 2, :]
            omega_b = torch.einsum("...ji,...j->...i", rotations, self.reference_omega)
            self.reference_states = encode_state(self.reference_q, self.reference_dq, gravity, omega_b)
            self.reference_clip = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
            self.reference_frame = torch.zeros_like(self.reference_clip)
            self.reference_gain = 0.25 * torch.tensor(ACTION_GAIN, device=self.device)
            self.reference_default = torch.tensor(DEFAULT_Q, device=self.device)

        def _get_observations(self):
            obs = super()._get_observations()
            self._ensure_tracking()
            clip, frame = self.reference_clip, self.reference_frame
            future = (frame[:, None] + torch.tensor([1, 5, 10], device=self.device)).clamp_max(149)
            data = self.robot.data
            actual_rot = quaternion_matrix(data.root_quat_w)
            target_rot = quaternion_matrix(self.reference_quat[clip, frame])
            relative = actual_rot.transpose(-1, -2) @ target_rot
            actor = torch.cat(
                (
                    obs["states"],
                    obs["last_action"],
                    obs["history_actor"],
                    self.reference_states[clip[:, None], future].flatten(1),
                    self.reference_pos[clip, frame] - (data.root_pos_w - self.scene.env_origins),
                    relative[..., :2].flatten(1),
                    self.reference_vel[clip, frame] - data.root_lin_vel_w,
                    data.root_lin_vel_w,
                    frame[:, None].float() / 149,
                ),
                dim=-1,
            )
            assert actor.shape == (self.num_envs, 547)
            obs["policy"] = actor
            obs["critic"] = torch.cat((actor, obs["privileged_state"]), dim=-1)
            return obs

        def step(self, residual):
            self._ensure_tracking()
            residual = residual.to(self.device)
            self.reference_frame = (self.reference_frame + 1).clamp_max(149)
            q = self.reference_q[self.reference_clip, self.reference_frame]
            action = ((q - self.reference_default) / self.reference_gain + residual).clamp(-1, 1)
            return super().step(action)

        def _get_rewards(self):
            self._ensure_tracking()
            clip, frame = self.reference_clip, self.reference_frame
            data = self.robot.data
            q_error = (data.joint_pos - self.reference_q[clip, frame]).square().mean(-1)
            pos_error = (
                (data.root_pos_w - self.scene.env_origins - self.reference_pos[clip, frame]).square().sum(-1)
            )
            angle = 2 * torch.acos(
                (data.root_quat_w * self.reference_quat[clip, frame]).sum(-1).abs().clamp(0, 1)
            )
            vel_error = (data.root_lin_vel_w - self.reference_vel[clip, frame]).square().sum(-1)
            reward = (
                4 * torch.exp(-q_error / 0.09)
                + 4 * torch.exp(-pos_error / 0.04)
                + 2 * torch.exp(-angle.square() / 0.25)
                + 2 * torch.exp(-vel_error / 0.25)
                + 1
            ) * self.step_dt
            return reward + self._aux_components().sum(-1) - 2 * self.reset_terminated.float()

        def _get_dones(self):
            terminated, truncated = super()._get_dones()
            self._ensure_tracking()
            return terminated, truncated | (self.reference_frame >= 149)

        def _reset_idx(self, env_ids):
            if getattr(self, "_flat_defer_reset", False):
                return
            super()._reset_idx(env_ids)
            self._ensure_tracking()
            if env_ids is None:
                env_ids = torch.arange(self.num_envs, device=self.device)
            n = len(env_ids)
            self.reference_clip[env_ids] = torch.randint(4, (n,), device=self.device)
            self.reference_frame[env_ids] = torch.randint(75, (n,), device=self.device)
            self.initialize_reference(env_ids)

        def initialize_reference(self, env_ids):
            clip, frame = self.reference_clip[env_ids], self.reference_frame[env_ids]
            root = self.robot.data.default_root_state[env_ids].clone()
            root[:, :3] = self.scene.env_origins[env_ids] + self.reference_pos[clip, frame]
            root[:, 3:7] = self.reference_quat[clip, frame]
            root[:, 7:10] = self.reference_vel[clip, frame]
            root[:, 10:13] = self.reference_omega[clip, frame]
            self.robot.write_root_state_to_sim(root, env_ids=env_ids)
            self.robot.write_joint_state_to_sim(
                self.reference_q[clip, frame], self.reference_dq[clip, frame], env_ids=env_ids
            )
            self.scene.write_data_to_sim()
            self.sim.forward()
            self.scene.update(self.step_dt)

    return RPOTrackingIsaacEnv
