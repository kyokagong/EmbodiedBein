"""RPO V2 WBC task and RLinf bridge, loaded only inside the Isaac image."""

from __future__ import annotations

import os


TASK_ID = "RPO-WBC-WalkUpper-v0"
BASIC_TASK_ID = "RPO-WBC-StandWalk-v1"
AMP_TASK_ID = "RPO-AMP-StandWalk-v1"
FALL_BASE_HEIGHT_M = 0.50
FALL_MAX_TILT_RAD = 0.80


def training_settings_from_env() -> dict[str, float]:
    return {
        "target_speed_mps": float(os.getenv("EB_RPO_TARGET_SPEED_MPS", "0.3")),
        "stand_fraction": float(os.getenv("EB_RPO_STAND_FRACTION", "0.5")),
        "velocity_reward_weight": float(os.getenv("EB_RPO_VELOCITY_REWARD_WEIGHT", "3.0")),
        "velocity_tracking_std": float(os.getenv("EB_RPO_VELOCITY_TRACKING_STD", "0.5")),
        "upright_reward_weight": float(os.getenv("EB_RPO_UPRIGHT_REWARD_WEIGHT", "0.5")),
    }


class DirectRLEnvAutoResetCompat:
    """Reset finished Isaac environments when RLinf requests a partial reset."""

    def reset(self, seed=None, options=None, env_ids=None):
        if env_ids is not None:
            if len(env_ids) > 0:
                self._reset_idx(env_ids)
                self.obs_buf = self._get_observations()
            return self.obs_buf, self.extras
        return super().reset(seed=seed, options=options)


def register_task() -> None:
    import gymnasium as gym

    for task_id, module, cfg_name, env_name in (
        (TASK_ID, "rpo_wbc_extension", "RPOWBCEnvCfg", "RPOWBCIsaacEnv"),
        (BASIC_TASK_ID, "rpo_wbc_extension", "RPOBasicEnvCfg", "RPOBasicIsaacEnv"),
        (AMP_TASK_ID, "rpo_amp_extension", "RPOAMPEnvCfg", "RPOAMPIsaacEnv"),
    ):
        if task_id not in gym.registry:
            gym.register(id=task_id, entry_point=f"{module}:{env_name}",
                         kwargs={"env_cfg_entry_point": f"{module}:{cfg_name}"},
                         disable_env_checker=True)


class RPOWBCEnv:
    """RLinf wrapper preserving actor/critic observations and terminal state."""

    def __new__(cls, *args, **kwargs):
        from rlinf.envs.sim.isaaclab.isaaclab_env import IsaaclabBaseEnv

        class _Bridge(IsaaclabBaseEnv):
            def _make_env_function(self):
                seed = self.seed
                env_id = self.isaaclab_env_id
                num_envs = self.cfg.init_params.num_envs

                def make_env():
                    os.environ.pop("DISPLAY", None)
                    from isaaclab.app import AppLauncher
                    app = AppLauncher(headless=True, enable_cameras=False).app
                    register_task()
                    from isaaclab_tasks.utils import load_cfg_from_registry
                    import gymnasium as gym

                    cfg = load_cfg_from_registry(env_id, "env_cfg_entry_point")
                    cfg.seed = seed
                    cfg.scene.num_envs = num_envs
                    return gym.make(env_id, cfg=cfg, render_mode=None).unwrapped, app

                return make_env

            def _wrap_obs(self, obs):
                return {"states": obs["policy"], "critic_states": obs["critic"]}

        return _Bridge(*args, **kwargs)


def _runtime_classes(basic=False, amp=False):
    import torch
    from isaaclab.utils import configclass
    from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
    from robolab.tasks.direct.interrupt.interrupt_env import InterruptEnv
    from robolab.tasks.direct.interrupt.rpo_interrupt_env_cfg import RPOInterruptEnvCfg
    from robolab.tasks.direct.base import mdp
    from robolab.tasks.direct.base.scene_cfg import SceneCfg

    joint_order = [
        "left_thigh_yaw_joint", "left_thigh_roll_joint", "left_thigh_pitch_joint",
        "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
        "right_thigh_yaw_joint", "right_thigh_roll_joint", "right_thigh_pitch_joint",
        "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint", "torso_joint",
        "left_arm_pitch_joint", "left_arm_roll_joint", "left_arm_yaw_joint", "left_elbow_pitch_joint",
        "left_elbow_yaw_joint", "right_arm_pitch_joint", "right_arm_roll_joint", "right_arm_yaw_joint",
        "right_elbow_pitch_joint", "right_elbow_yaw_joint",
    ]

    @configclass
    class EnvCfg(RPOInterruptEnvCfg):
        def __post_init__(self):
            super().__post_init__()
            self.observation_space = 78 if amp else 89
            self.state_space = 139 if amp else 150
            self.robot.actor_obs_history_length = 10
            self.robot.critic_obs_history_length = 10
            self.interrupt.use_interrupt = False
            self.robot.terminate_base_height = FALL_BASE_HEIGHT_M
            self.robot.terminate_base_orientation = FALL_MAX_TILT_RAD
            if amp:
                from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
                from isaaclab.terrains.trimesh.mesh_terrains_cfg import MeshPlaneTerrainCfg

                fixed_delay = os.getenv("EB_RPO_AMP_ACTUATOR_DELAY_STEPS")
                if fixed_delay is not None:
                    if fixed_delay not in {"0", "1", "2"}:
                        raise ValueError("RPO AMP actuator delay must be 0–2 physics steps")
                    for actuator in self.scene_context.robot.actuators.values():
                        actuator.min_delay = actuator.max_delay = int(fixed_delay)
                self.scene_context.terrain_type = "generator"
                self.scene_context.terrain_generator = TerrainGeneratorCfg(
                    size=(100.0, 100.0), num_rows=1, num_cols=1,
                    sub_terrains={"flat": MeshPlaneTerrainCfg(proportion=1.0)})
            else:
                self.scene_context.terrain_type = "plane"
                self.scene_context.terrain_generator = None
            self.scene_context.height_scanner.enable_height_scan = False
            self.scene = SceneCfg(config=self.scene_context, physics_dt=self.sim.dt,
                                  step_dt=self.decimation * self.sim.dt)
            if amp:
                import isaaclab.sim as sim_utils

                self.scene.terrain.use_terrain_origins = False
                self.scene.terrain.visual_material = sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.2, 0.2, 0.2))
                self.scene.sky_light.spawn = sim_utils.DomeLightCfg(
                    intensity=750.0, color=(1.0, 1.0, 1.0))
                self.scene.left_feet_scanner.debug_vis = False
                self.scene.right_feet_scanner.debug_vis = False
                self.commands.debug_vis = False
            self.reward.joint_deviation_torso.weight = 0.0
            self.reward.joint_deviation_interrupt.weight = 0.0
            self.reward.stand_still.weight = 0.0
            self.reward.action_penalty.weight = 0.0
            if basic:
                settings = training_settings_from_env()
                self.commands.ranges.lin_vel_x = (settings["target_speed_mps"], settings["target_speed_mps"])
                self.commands.ranges.lin_vel_y = (0.0, 0.0)
                self.commands.ranges.ang_vel_z = (0.0, 0.0)
                self.commands.rel_standing_envs = settings["stand_fraction"]
                self.commands.heading_command = False
                self.commands.rel_heading_envs = 0.0
                self.commands.resampling_time_range = (20.0, 20.0)
                self.noise.add_noise = False
                for event in ("physics_material", "add_base_mass", "randomize_rigid_body_com",
                              "scale_link_mass", "scale_actuator_gains", "scale_joint_parameters"):
                    setattr(self.events, event, None)
                self.events.push_robot = None
                self.events.reset_base.params["velocity_range"] = {
                    key: (-0.05, 0.05) if key in ("x", "y", "yaw") else (0.0, 0.0)
                    for key in ("x", "y", "z", "roll", "pitch", "yaw")}
                self.events.reset_robot_joints.params["position_range"] = (-0.05, 0.05)
                self.reward.track_lin_vel_xy_exp.weight = settings["velocity_reward_weight"]
                if amp:
                    self.reward.track_lin_vel_xy_exp.params["std"] = settings["velocity_tracking_std"]
                self.reward.upward.weight = settings["upright_reward_weight"]
                self.reward.joint_deviation_torso.weight = -0.3
                self.reward.stand_still.func = mdp.stand_still
                self.reward.stand_still.params.pop("interrupt_cfg")
                self.reward.stand_still.weight = -0.2

    class Env(DirectRLEnvAutoResetCompat, InterruptEnv):
        reference_resample_steps = 30

        def init_buffers(self):
            self.codec_cfg = SceneEntityCfg(name="robot", joint_names=joint_order, preserve_order=True)
            self.codec_cfg.resolve(self.scene)
            self.upper_reference = torch.zeros((self.num_envs, 11), device=self.device)
            self.upper_reference_target = torch.zeros((self.num_envs, 11), device=self.device)
            self.reference_age = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            super().init_buffers()
            if amp:
                import importlib

                module = importlib.import_module(
                    f"{__package__}.rpo_amp_extension" if __package__ else "rpo_amp_extension"
                )
                self.amp_history = module.AMPFrameHistory(self.num_envs, self.device)
                self.amp_key_bodies = SceneEntityCfg(
                    "robot", body_names=[
                        "left_ankle_roll_link", "right_ankle_roll_link",
                        "left_knee_link", "right_knee_link",
                        "left_elbow_yaw_link", "right_elbow_yaw_link",
                    ], preserve_order=True
                )
                self.amp_key_bodies.resolve(self.scene)

        def _amp_frame(self):
            from isaaclab.utils.math import quat_apply_inverse

            robot = self.robot.data
            n = self.num_envs
            delta = robot.body_pos_w[:, self.amp_key_bodies.body_ids] - robot.root_pos_w[:, None]
            q = robot.root_quat_w[:, None].expand(n, 6, 4).reshape(-1, 4)
            key_body_pos_b = quat_apply_inverse(q, delta.reshape(-1, 3)).reshape(n, 18)
            ids = self.codec_cfg.joint_ids
            root_height = (robot.root_pos_w[:, 2] - self.scene.env_origins[:, 2]).unsqueeze(-1)
            frame = torch.cat((
                robot.root_ang_vel_b, robot.joint_pos[:, ids], robot.joint_vel[:, ids],
                key_body_pos_b, robot.projected_gravity_b, root_height, robot.root_lin_vel_b,
            ), dim=-1)
            if frame.shape != (n, 74) or not torch.isfinite(frame).all():
                raise ValueError("RPO AMP feature frame is invalid")
            return frame

        def _sample_reference(self, env_ids):
            if basic:
                self.upper_reference[env_ids] = 0.0
                self.upper_reference_target[env_ids] = 0.0
                self.reference_age[env_ids] = 0
                return
            phase = min(self.common_step_counter // 5000, 2)
            amplitude = (0.1, 0.2, 0.35)[phase]
            target = torch.empty((len(env_ids), 11), device=self.device).uniform_(-amplitude, amplitude)
            target[:, 0] *= 0.5  # torso remains conservative until hardware limits are approved.
            if phase == 0:
                target[:, (0, 5, 10)] = 0.0  # torso and elbow yaw enter in later curriculum phases.
            self.upper_reference_target[env_ids] = target
            self.reference_age[env_ids] = 0

        def compute_current_observations(self):
            ids = self.codec_cfg.joint_ids
            robot = self.robot
            forces = self.contact_sensor.data.net_forces_w_history
            joint_pos = robot.data.joint_pos[:, ids] - robot.data.default_joint_pos[:, ids]
            joint_vel = robot.data.joint_vel[:, ids] - robot.data.default_joint_vel[:, ids]
            actor_terms = (
                robot.data.root_ang_vel_b * self.obs_scales.ang_vel,
                robot.data.projected_gravity_b * self.obs_scales.projected_gravity,
                self.command_generator.command * self.obs_scales.commands,
                joint_pos * self.obs_scales.joint_pos,
                joint_vel * self.obs_scales.joint_vel,
                self.action_buffer.buffer[:, -1, :] * self.obs_scales.actions,
            )
            actor = torch.cat(actor_terms if amp else (*actor_terms, self.upper_reference), dim=-1)
            feet_contact = torch.max(torch.norm(forces[:, :, self.feet_cfg.body_ids], dim=-1), dim=1)[0] > 1.0
            feet_force = self.contact_sensor.data.net_forces_w[:, self.feet_cfg.body_ids, :]
            feet_air = self.contact_sensor.data.current_air_time[:, self.feet_cfg.body_ids]
            feet_height = torch.stack([
                self.scene[name].data.pos_w[:, 2] - self.scene[name].data.ray_hits_w[..., 2].mean(dim=-1)
                for name in ("left_feet_scanner", "right_feet_scanner")
            ], dim=-1)
            feet_height = torch.nan_to_num(torch.clamp(feet_height - 0.04, 0.0, 1.0), nan=1.0)
            critic = torch.cat((actor, robot.data.root_lin_vel_b * self.obs_scales.lin_vel,
                                feet_contact.float(), feet_force.flatten(1), feet_air.flatten(1), feet_height.flatten(1),
                                robot.data.joint_acc[:, ids], robot.data.applied_torque[:, ids]), dim=-1)
            return actor, critic

        def _get_rewards(self):
            reward = super()._get_rewards()
            if amp:
                return reward
            ids = self.codec_cfg.joint_ids[12:]
            upper = self.robot.data.joint_pos[:, ids] - self.robot.data.default_joint_pos[:, ids]
            return reward + self.step_dt * torch.exp(-4.0 * torch.square(upper - self.upper_reference).mean(dim=-1))

        def _pre_physics_step(self, actions):
            if not amp:
                due = self.reference_age >= self.reference_resample_steps
                if due.any():
                    self._sample_reference(due.nonzero(as_tuple=False).squeeze(-1))
                delta = torch.clamp(self.upper_reference_target - self.upper_reference, -0.01, 0.01)
                self.upper_reference += delta
                self.reference_age += 1
            canonical = torch.clip(actions, -self.clip_actions, self.clip_actions).to(self.device)
            self.action_buffer.append(canonical)
            self.actions = self.robot.data.default_joint_pos.clone()
            self.actions[:, self.codec_cfg.joint_ids] += canonical * self.action_scale

        def _reset_idx(self, env_ids):
            super()._reset_idx(env_ids)
            self._sample_reference(env_ids)
            if amp and os.getenv("EB_RPO_AMP_INITIAL_POSE") == "official_a1_stand_v1" and len(env_ids) > 0:
                try:
                    from eb.rpo_wbc import AMP_STAND_HEIGHT_M, AMP_STAND_Q
                except ImportError:
                    from rpo_wbc import AMP_STAND_HEIGHT_M, AMP_STAND_Q
                root = self.robot.data.root_state_w[env_ids].clone()
                root[:, 2] = self.scene.env_origins[env_ids, 2] + AMP_STAND_HEIGHT_M
                joints = self.robot.data.joint_pos[env_ids].clone()
                joints[:, self.codec_cfg.joint_ids] = torch.tensor(AMP_STAND_Q, device=self.device)
                self.robot.write_root_state_to_sim(root, env_ids=env_ids)
                self.robot.write_joint_state_to_sim(joints, torch.zeros_like(joints), env_ids=env_ids)
            if amp and hasattr(self, "amp_history") and len(env_ids) > 0:
                self.amp_history.reset(env_ids, self._amp_frame()[env_ids])

        def step(self, actions):
            """Leave reset ownership to RLinf so its final observation is the real terminal state."""
            self._pre_physics_step(actions.to(self.device))
            rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()
            for _ in range(self.cfg.decimation):
                self._sim_step_counter += 1
                self._apply_action()
                self.scene.write_data_to_sim()
                self.sim.step(render=False)
                if self._sim_step_counter % self.cfg.sim.render_interval == 0 and rendering:
                    self.sim.render()
                self.scene.update(dt=self.physics_dt)
            self.episode_length_buf += 1
            self.common_step_counter += 1
            self.command_generator.compute(self.step_dt)
            if "interval" in self.event_manager.available_modes:
                self.event_manager.apply(mode="interval", dt=self.step_dt)
            self.reset_terminated[:], self.reset_time_outs[:] = self._get_dones()
            self.reset_buf = self.reset_terminated | self.reset_time_outs
            self.reward_buf = self._get_rewards()
            self.obs_buf = self._get_observations()
            if amp:
                window, valid = self.amp_history.advance(self._amp_frame())
                self.extras["amp_window"] = window
                self.extras["amp_valid"] = valid
            return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    return EnvCfg, Env


def __getattr__(name):
    if name not in {"RPOWBCEnvCfg", "RPOWBCIsaacEnv", "RPOBasicEnvCfg", "RPOBasicIsaacEnv"}:
        raise AttributeError(name)
    cfg, env = _runtime_classes(name.startswith("RPOBasic"))
    if name.endswith("EnvCfg"):
        return cfg
    return env
