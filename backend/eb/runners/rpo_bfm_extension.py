"""RPO BFM stage-A environment; RLinf owns resets, policy and training."""

from __future__ import annotations

import os

try:
    from eb.rpo_bfm import ACTION_GAIN, DEFAULT_Q, BODY_ORDER, GYM_ID, OBS_DIMS, RPO_LAB_JOINT_ORDER
except ModuleNotFoundError:
    from rpo_bfm import ACTION_GAIN, DEFAULT_Q, BODY_ORDER, GYM_ID, OBS_DIMS, RPO_LAB_JOINT_ORDER


RLINF_OBSERVATION_KEYS = (*OBS_DIMS, "commands", "bfm_aux_rewards", "episode_id", "step_id")


def rlinf_observation(obs):
    return {key: obs[key] for key in RLINF_OBSERVATION_KEYS}


def register_task():
    import gymnasium as gym

    if GYM_ID not in gym.registry:
        gym.register(id=GYM_ID, entry_point="rpo_bfm_extension:RPOBFMIsaacEnv",
                     kwargs={"env_cfg_entry_point": "rpo_bfm_extension:RPOBFMEnvCfg"},
                     disable_env_checker=True)


class RPOBFMEnv:
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
                    cfg = __getattr__("RPOBFMEnvCfg")()
                    cfg.seed, cfg.scene.num_envs = seed, num_envs
                    return gym.make(GYM_ID, cfg=cfg, render_mode=None).unwrapped, app

                return make_env

            def _wrap_obs(self, obs):
                return rlinf_observation(obs)

        return Bridge(*args, **kwargs)


def __getattr__(name):
    if name == "RPOBFMEnvCfg":
        from isaaclab.utils import configclass
        from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
        from isaaclab.terrains.trimesh.mesh_terrains_cfg import MeshPlaneTerrainCfg
        from robolab.tasks.direct.base.rpo_env_cfg import RPOFlatEnvCfg
        from robolab.tasks.direct.base.scene_cfg import SceneCfg

        @configclass
        class RPOBFMEnvCfg(RPOFlatEnvCfg):
            def __post_init__(self):
                super().__post_init__()
                if os.environ.get("EB_UFO_CONTACT_PROFILE") == "contact_stable_v8":
                    from rpo_ufo_contact import configure_substeps
                    configure_substeps(self)
                # Procedural flat mesh stays offline; Isaac's built-in plane references remote USD.
                self.scene_context.terrain_type = "generator"
                size = (100., 100.)
                if os.environ.get("EB_UFO_GROUND_PROFILE") == "wide_plane_v9":
                    from rpo_ufo_ground import SIZE_M
                    size = SIZE_M
                self.scene_context.terrain_generator = TerrainGeneratorCfg(
                    seed=42, size=size, num_rows=1, num_cols=1,
                    sub_terrains={"flat": MeshPlaneTerrainCfg(proportion=1.)})
                for actuator in self.scene_context.robot.actuators.values():
                    actuator.min_delay = actuator.max_delay = 0
                    if os.environ.get("EB_UFO_ANKLE_CONTROL") in {"kd25","kd30"}:
                        damping=3. if os.environ['EB_UFO_ANKLE_CONTROL']=='kd30' else 2.5
                        if isinstance(actuator.damping, dict):
                            actuator.damping={name:(damping if "ankle" in name else value)
                                              for name,value in actuator.damping.items()}
                        elif any("ankle" in name for name in actuator.joint_names_expr):
                            actuator.damping=damping
                self.scene = SceneCfg(config=self.scene_context, physics_dt=self.sim.dt,
                                      step_dt=self.decimation * self.sim.dt)
                self.scene.terrain.use_terrain_origins = False
                if (os.environ.get('EB_UFO_DYNAMICS_PROFILE')=='contact_compliance_v11'
                        and os.environ.get('EB_UFO_CONTACT_MATERIAL')!='rigid'):
                    from rpo_ufo_compliance import configure_material
                    configure_material(self.scene.terrain.physics_material)
                self.scene.left_feet_scanner.debug_vis = False
                self.scene.right_feet_scanner.debug_vis = False
                self.commands.debug_vis = False
                self.noise.add_noise = False
                self.robot.terminate_base_height = .50
                self.robot.terminate_base_orientation = .80

        return RPOBFMEnvCfg
    if name != "RPOBFMIsaacEnv":
        raise AttributeError(name)
    import torch
    from robolab.tasks.direct.base.base_env import BaseEnv
    from rpo_flat_extension import FlatResetCompat
    from rpo_bfm_codec import History, action_anchor, encode_privileged, encode_state, limit_action_slew, policy_to_target

    class RPOBFMIsaacEnv(FlatResetCompat, BaseEnv):
        def __init__(self, *args, **kwargs):
            self.bfm_control_profile = os.environ.get("EB_UFO_CONTROL_PROFILE", "v1")
            if self.bfm_control_profile not in {"v1", "reachable_v2"}:
                raise ValueError("Unknown BFM control profile")
            self.bfm_slew = os.environ.get("EB_UFO_LEARNING_PROFILE") in {"physical_limits_slew_v4", "physical_limits_stable_slew_v4"}
            super().__init__(*args, **kwargs)
            if (tuple(self.robot.joint_names) != RPO_LAB_JOINT_ORDER
                    or tuple(self.robot.body_names) != BODY_ORDER):
                raise ValueError("RPO BFM asset differs from frozen joint/body order")
            if not hasattr(self.robot.data, "body_link_lin_vel_w"):
                raise ValueError("RPO BFM requires link-origin body velocities")
            self._ensure_bfm_buffers()
            if self.bfm_slew:
                from rpo_bfm import SLEW_CONTROL
                from rpo_flat_artifact import _hard_limits
                torch.testing.assert_close(torch.tensor(_hard_limits(self)["joint_vel"]),
                    torch.tensor(SLEW_CONTROL["joint_velocity_limits_rad_s"]), atol=0, rtol=0)
                if self.step_dt != SLEW_CONTROL["control_dt_s"]:
                    raise ValueError("Target-slew control frequency differs from frozen protocol")
            if self.bfm_control_profile == "reachable_v2":
                from rpo_bfm import JOINT_POSITION_MIN, JOINT_POSITION_MAX
                from rpo_flat_artifact import _hard_limits
                torch.testing.assert_close(torch.tensor(_hard_limits(self)["joint_pos"]),
                    torch.tensor(list(zip(JOINT_POSITION_MIN, JOINT_POSITION_MAX))), atol=1e-6, rtol=0)
                if not self.bfm_slew:
                    raise ValueError("Reachable v2 control requires the frozen physical target slew")

        def _ensure_bfm_buffers(self):
            if not hasattr(self, "bfm_history"):
                self.bfm_history = History(self.num_envs, device=self.device)
                self.bfm_last_action = torch.zeros(self.num_envs, 23, device=self.device)
                self.bfm_previous_action = self.bfm_last_action.clone()
                self.bfm_aux_rewards = torch.zeros(self.num_envs, 3, device=self.device)
                self.bfm_episode_id = torch.zeros(self.num_envs, 1, dtype=torch.long, device=self.device)
                self.bfm_step_id = torch.zeros_like(self.bfm_episode_id)

        def _get_observations(self):
            self._ensure_bfm_buffers()
            data = self.robot.data
            state = encode_state(data.joint_pos, data.joint_vel, data.projected_gravity_b,
                                 data.root_ang_vel_b)
            privileged = encode_privileged(data.body_pos_w, data.body_quat_w,
                                          data.body_link_lin_vel_w, data.body_ang_vel_w,
                                          self.scene.env_origins[:, 2])
            return {"states": state, "last_action": self.bfm_last_action.clone(),
                    "history_actor": self.bfm_history.snapshot(), "privileged_state": privileged,
                    "commands": self.command_generator.command.clone(),
                    "bfm_aux_rewards": self.bfm_aux_rewards.clone(),
                    "episode_id": self.bfm_episode_id.clone(), "step_id": self.bfm_step_id.clone()}

        def step(self, actions):
            actions = actions.to(self.device)
            if actions.shape != (self.num_envs, 23) or not bool(torch.isfinite(actions).all()):
                raise ValueError("RPO BFM policy action has invalid shape or nonfinite values")
            current = self._get_observations()
            self.bfm_history.append(current["states"], self.bfm_last_action)
            self.bfm_previous_action = self.bfm_last_action.clone()
            self.bfm_last_action = (limit_action_slew(actions, self.bfm_previous_action, control_profile=self.bfm_control_profile)
                                    if self.bfm_slew else actions.clamp(-1, 1))
            self.bfm_step_id += 1
            flat_action = ((policy_to_target(self.bfm_last_action, control_profile="reachable_v2") - actions.new_tensor(DEFAULT_Q)) / .25
                           if self.bfm_control_profile == "reachable_v2"
                           else self.bfm_last_action * actions.new_tensor(ACTION_GAIN))
            obs, reward, terminated, truncated, info = super().step(flat_action)
            auxiliary = self._aux_components().clone()
            self.bfm_aux_rewards = auxiliary
            obs["bfm_aux_rewards"] = auxiliary
            info["bfm_aux_rewards"] = auxiliary
            info["bfm_policy_action"] = actions.clone()
            info["bfm_executed_action"] = self.bfm_last_action.clone()
            info["bfm_flat_action"] = flat_action.clone()
            return obs, reward, terminated, truncated, info

        def _aux_components(self):
            data = self.robot.data
            limits = data.soft_joint_pos_limits
            violation = ((limits[..., 0] - data.joint_pos).clamp_min(0)
                         + (data.joint_pos - limits[..., 1]).clamp_min(0)).sum(-1)
            torque = data.applied_torque.square().sum(-1)
            action_rate = (self.bfm_last_action - self.bfm_previous_action).square().sum(-1)
            # Named, signed contributions, each integrated exactly once over policy dt.
            return torch.stack((-10 * violation, -1e-5 * torque, -.1 * action_rate), dim=-1) * self.step_dt

        def _get_rewards(self):
            return self._aux_components().sum(-1)

        def _reset_slew_anchor(self, env_ids):
            anchor = action_anchor(self.robot.data.joint_pos[env_ids], control_profile=self.bfm_control_profile)
            self.bfm_last_action[env_ids] = anchor
            self.bfm_previous_action[env_ids] = anchor
            self.robot.set_joint_position_target(policy_to_target(anchor, control_profile=self.bfm_control_profile), env_ids=env_ids)

        def _reset_idx(self, env_ids):
            if getattr(self, "_flat_defer_reset", False):
                return
            super()._reset_idx(env_ids)
            if hasattr(self, "bfm_history"):
                self.bfm_history.reset(env_ids)
                self.bfm_last_action[env_ids] = 0
                self.bfm_previous_action[env_ids] = 0
                self.bfm_step_id[env_ids] = 0
                self.bfm_aux_rewards[env_ids] = 0
                self.bfm_episode_id[env_ids] += 1
                if getattr(self, "bfm_slew", False):
                    self._reset_slew_anchor(env_ids)

        def reset(self, seed=None, options=None, env_ids=None):
            self._ensure_bfm_buffers()
            if env_ids is None:
                # BFM observation reads do not append Flat's legacy ring buffers.
                return super(FlatResetCompat, self).reset(seed=seed, options=options)
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
            if len(env_ids):
                self._reset_idx(env_ids)
            self.obs_buf = self._get_observations()
            return self.obs_buf, self.extras

    return RPOBFMIsaacEnv
