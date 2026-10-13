"""Native RPO MJCF/MuJoCo-Warp physics with the frozen platform UFO codec."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch

try:
    from eb.rpo_bfm import (BODY_ORDER, CODEC_V2_HASH, DEFAULT_Q, JOINT_POSITION_MIN,
                           JOINT_POSITION_MAX, RPO_LAB_JOINT_ORDER, SLEW_CONTROL)
    from eb.runners.rpo_bfm_codec import (History, action_anchor, encode_privileged,
        encode_state, limit_action_slew, policy_to_target, quaternion_matrix)
    from eb.runners.rpo_ufo_isaac import ObservationCache, PhysicsInterval, raw_aux_rewards, validate_integration
    from eb.runners.bfm_ufo_agent import VELOCITY_AUX_NAMES
except ModuleNotFoundError:
    from rpo_bfm import (BODY_ORDER, CODEC_V2_HASH, DEFAULT_Q, JOINT_POSITION_MIN,
                        JOINT_POSITION_MAX, RPO_LAB_JOINT_ORDER, SLEW_CONTROL)
    from rpo_bfm_codec import (History, action_anchor, encode_privileged,
        encode_state, limit_action_slew, policy_to_target, quaternion_matrix)
    from rpo_ufo_isaac import ObservationCache, PhysicsInterval, raw_aux_rewards, validate_integration
    from bfm_ufo_agent import VELOCITY_AUX_NAMES

GYM_ID = "RPO-UFO-MJLab-RSI-v12"
VIABILITY_GYM_ID = "RPO-UFO-MJLab-RSI-viability-v13"
try:
    from eb.rpo_bfm import MJLAB_PHYSICS_PROFILE as PHYSICS_PROFILE
except ModuleNotFoundError:
    from rpo_bfm import MJLAB_PHYSICS_PROFILE as PHYSICS_PROFILE


def actuator_parameters(name):
    if "torso" in name or "knee" in name:
        return 150., 5., 120., 25.
    if "thigh" in name:
        return 100., 3.3, 120., 25.
    if "ankle" in name:
        return 40., 2.5, 27., 8.
    if "elbow_pitch" in name:
        return 30., 1.5, 27., 8.
    if "elbow_yaw" in name:
        return 20., 1., 27., 8.
    if "arm" in name:
        return 40., 2., 27., 8.
    raise ValueError("Unknown RPO actuator")


def prepare_spec(asset_root, *, physics_dt=.0025):
    import mujoco
    path = Path(asset_root) / "rpo_description/mjcf/rpo.xml"
    spec = mujoco.MjSpec.from_file(str(path))
    for actuator in list(spec.actuators):
        spec.delete(actuator)
    for sensor in list(spec.sensors):
        spec.delete(sensor)
    for geom in list(spec.geoms):
        if geom.type == mujoco.mjtGeom.mjGEOM_PLANE:
            spec.delete(geom)
    spec.option.timestep = physics_dt
    spec.option.solver = mujoco.mjtSolver.mjSOL_NEWTON
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.iterations = 100
    for name, low, high in zip(RPO_LAB_JOINT_ORDER, JOINT_POSITION_MIN, JOINT_POSITION_MAX):
        spec.joint(name).range = (low, high)
    # Keep the original collision masks: enabling self-collision creates ankle/knee overlaps.
    return spec


def _joint_observation(env):
    return env.scene["robot"].data.joint_pos


def make_config(asset_root, *, num_envs, seed, substeps=8, physics_dt=.0025):
    validate_integration(substeps, physics_dt)
    from mjlab.actuator import DcMotorActuatorCfg
    from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
    from mjlab.envs import ManagerBasedRlEnvCfg
    from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
    from mjlab.scene import SceneCfg
    from mjlab.sensor.contact_sensor import ContactMatch, ContactSensorCfg
    from mjlab.sim import MujocoCfg, SimulationCfg
    from mjlab.terrains import TerrainEntityCfg

    actuators = []
    for name in RPO_LAB_JOINT_ORDER:
        kp, kd, effort, velocity = actuator_parameters(name)
        actuators.append(DcMotorActuatorCfg(target_names_expr=(name,), stiffness=kp,
            damping=kd, effort_limit=effort, saturation_effort=effort,
            velocity_limit=velocity, armature=.01, frictionloss=.01))
    robot = EntityCfg(spec_fn=lambda: prepare_spec(asset_root, physics_dt=physics_dt), sort_actuators=True,
        init_state=EntityCfg.InitialStateCfg(pos=(0., 0., .75),
            joint_pos=dict(zip(RPO_LAB_JOINT_ORDER, DEFAULT_Q)), joint_vel={".*": 0.}),
        articulation=EntityArticulationInfoCfg(actuators=tuple(actuators), soft_joint_pos_limit_factor=.9))
    return ManagerBasedRlEnvCfg(decimation=substeps, seed=seed, auto_reset=False,
        episode_length_s=20., scale_rewards_by_dt=False,
        scene=SceneCfg(num_envs=num_envs, env_spacing=2.5,
            terrain=TerrainEntityCfg(terrain_type="plane", env_spacing=2.5),
            entities={"robot": robot}, sensors=(ContactSensorCfg(name="body_contact",
                primary=ContactMatch(mode="body", pattern=BODY_ORDER, entity="robot"),
                fields=("found", "force"), reduce="netforce", history_length=8),)),
        observations={"actor": ObservationGroupCfg(terms={"joint_pos":
            ObservationTermCfg(func=_joint_observation)}, concatenate_terms=True, enable_corruption=False)},
        sim=SimulationCfg(nconmax=128, njmax=1024, contact_sensor_maxmatch=128,
            mujoco=MujocoCfg(timestep=physics_dt, solver="newton", iterations=100, integrator="implicitfast")))


class RPOUFOMJLabPhysics:
    """Explicit completed-frame control loop; RLinf owns auto-reset and final observations."""
    def __init__(self, *, num_envs, seed, device="cuda:0", asset_root=None, motion_root=None, asset_sha256=None,
                 physics_profile=None, substeps=8, physics_dt=.0025):
        validate_integration(substeps, physics_dt)
        import numpy as np
        from mjlab.envs import ManagerBasedRlEnv
        try:
            from eb.runners.rpo_ufo_contact import ContactResetPool
            from eb.runners.rpo_ufo_motion import load_robot_state
            from eb.runners.bfm_ufo_rollout_context import tracking_mask
        except ModuleNotFoundError:
            from rpo_ufo_contact import ContactResetPool
            from rpo_ufo_motion import load_robot_state
            from bfm_ufo_rollout_context import tracking_mask

        profile = physics_profile or os.environ.get('EB_UFO_PHYSICS_PROFILE', 'native_mjcf_v12')
        if profile not in {'native_mjcf_v12', 'native_mjcf_viability_v13'}:
            raise ValueError('Unknown native physics profile')
        self.physics_profile = PHYSICS_PROFILE
        if profile == 'native_mjcf_viability_v13':
            try:
                from eb.rpo_bfm import MJLAB_VIABILITY_PHYSICS_PROFILE
            except ModuleNotFoundError:
                from rpo_bfm import MJLAB_VIABILITY_PHYSICS_PROFILE
            self.physics_profile = MJLAB_VIABILITY_PHYSICS_PROFILE
        if (substeps, physics_dt) != (8, .0025):
            self.physics_profile = {**self.physics_profile, 'physics_dt_s':physics_dt, 'decimation':substeps}
        self.num_envs, self.seed, self._device = num_envs, seed, device
        asset_root = asset_root or os.environ["RPO_ASSET_ROOT"]
        motion_root = motion_root or os.environ["EB_UFO_MOTION_ROOT"]
        asset_sha256 = asset_sha256 or os.environ["EB_UFO_ASSET_SHA256"]
        self.lab = ManagerBasedRlEnv(make_config(asset_root, num_envs=num_envs, seed=seed,
            substeps=substeps, physics_dt=physics_dt), device=device)
        self.robot = self.lab.scene["robot"]
        self.joint_ids = [self.robot.joint_names.index(n) for n in RPO_LAB_JOINT_ORDER]
        self.body_ids = [self.robot.body_names.index(n) for n in BODY_ORDER]
        self.sensor = self.lab.scene.sensors["body_contact"]
        sensor_names = [n.split("/")[-1] for n in self.sensor.primary_names]
        if set(sensor_names) != set(BODY_ORDER) or len(sensor_names) != 24:
            raise ValueError("Native contact sensor body coverage differs")
        self.contact_ids = [sensor_names.index(n) for n in BODY_ORDER]
        self.feet = [BODY_ORDER.index(s + "_ankle_roll_link") for s in ("left", "right")]
        self.protected = [BODY_ORDER.index(n) for n in ("torso_link", "left_thigh_yaw_link",
            "right_thigh_yaw_link", "left_thigh_roll_link", "right_thigh_roll_link")]
        self.limits = {"joint_pos": torch.tensor(list(zip(JOINT_POSITION_MIN, JOINT_POSITION_MAX)), device=device),
            "joint_vel": torch.tensor(SLEW_CONTROL["joint_velocity_limits_rad_s"], device=device),
            "applied_torque": torch.tensor([actuator_parameters(n)[2] for n in RPO_LAB_JOINT_ORDER], device=device)}
        _, reference = load_robot_state(Path(motion_root), asset_sha256=asset_sha256,
            seed=seed, expected_codec_sha256=CODEC_V2_HASH)
        self.pool = ContactResetPool(reference, asset_root, tracking_contact=True, native_mjcf=True,
            joint_lookahead_s=self.physics_profile.get('rsi_joint_lookahead_s'))
        self.tracking = torch.as_tensor(tracking_mask(num_envs, seed), device=device)[:, None]
        self.history = History(num_envs, device=device)
        self.cache = ObservationCache()
        self.interval = PhysicsInterval(num_envs, self.limits, device=device, velocity_multiplier=1.3,
            position_tolerance=self.physics_profile.get('training_position_tolerance_rad', .001),
            substeps=substeps, physics_dt=physics_dt)
        self.last = torch.zeros(num_envs, 23, device=device)
        self.previous = self.last.clone()
        self.requested, self.target = self.last.clone(), self.last.clone()
        self.episode = torch.full((num_envs, 1), seed * 10**9, dtype=torch.long, device=device)
        self.steps = torch.zeros_like(self.episode)
        self.reference = torch.full((num_envs, 3), -1, dtype=torch.long, device=device)
        self.lift = torch.zeros(num_envs, device=device)
        self.peak_force = torch.zeros(num_envs, 2, device=device)
        self.macro = torch.zeros(num_envs, 3, dtype=torch.bool, device=device)
        self.interval_valid = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.reset()
        path = os.environ.get("EB_UFO_DIAGNOSTICS_DIR")
        if path:
            root = Path(path); root.mkdir(parents=True, exist_ok=True)
            model = self.lab.sim.mj_model
            receipt = {**self.physics_profile, "seed": seed, "num_envs": num_envs,
                "joint_order": list(RPO_LAB_JOINT_ORDER), "body_order": list(BODY_ORDER),
                "joint_ids": self.joint_ids, "body_ids": self.body_ids,
                "mass_kg": float(model.body_mass.sum()), "body_mass_kg": model.body_mass.tolist(),
                "body_inertia_kg_m2": model.body_inertia.tolist(), "dof_armature": model.dof_armature.tolist(),
                "actuator_parameters": {n: actuator_parameters(n) for n in RPO_LAB_JOINT_ORDER},
                "joint_position_limits_rad": self.limits["joint_pos"].cpu().tolist(),
                "soft_joint_position_limits_rad": self.robot.data.soft_joint_pos_limits[0, self.joint_ids].cpu().tolist(),
                "geom_contype": model.geom_contype.tolist(), "geom_conaffinity": model.geom_conaffinity.tolist(),
                "geom_solref": model.geom_solref.tolist(), "geom_friction": model.geom_friction.tolist(),
                "geom_solimp": model.geom_solimp.tolist(),
                "compiled_solver": {"timestep_s": float(model.opt.timestep),
                    "solver": int(model.opt.solver), "integrator": int(model.opt.integrator),
                    "iterations": int(model.opt.iterations), "cone": int(model.opt.cone),
                    "gravity_mps2": model.opt.gravity.tolist()},
                "integration_actual": {"physics_dt_s": physics_dt, "substeps": substeps,
                    "control_dt_s": physics_dt*substeps},
                "xml_sha256": hashlib.sha256((Path(asset_root)/"rpo_description/mjcf/rpo.xml").read_bytes()).hexdigest(),
                "compiled_xml_sha256": hashlib.sha256(self.lab.scene.spec.to_xml().encode()).hexdigest(),
                "tracking_count": int(self.tracking.sum()), "tracking_partition_sha256":
                    hashlib.sha256(self.tracking.cpu().numpy().tobytes()).hexdigest(),
                "rsi_pool": self.pool.rows, "qualification": "not_requested"}
            (root/f"physics-startup-{os.getpid()}.json").write_text(json.dumps(receipt, indent=2))

    def device(self):
        return self._device

    def _refresh(self, dt=.0025):
        self.lab.sim.forward()
        self.lab.scene.update(dt=dt)
        self.lab.sim.sense()

    def _completed_substep(self):
        self.lab.sim.step()
        data = self.lab.sim.data
        if data.nacon[0] >= data.naconmax or (data.nefc >= data.njmax).any():
            raise ValueError("Native contact/constraint capacity exhausted")
        self.lab.scene.update(dt=self.interval.physics_dt)
        self.lab.sim.sense()
        forces = self._forces().clone()
        torque = self.robot.data.qfrc_actuator[:, self.joint_ids].clone()
        # Forward refreshes post-integration geometry but replaces the force used for integration.
        self._refresh(dt=0.)
        return forces, torque

    def _state(self):
        data = self.robot.data
        return {"q": data.joint_pos[:, self.joint_ids], "dq": data.joint_vel[:, self.joint_ids],
            "torques": data.qfrc_actuator[:, self.joint_ids],
            "body_pos": data.body_link_pos_w[:, self.body_ids],
            "body_quat": data.body_link_quat_w[:, self.body_ids],
            "body_lin_vel": data.body_link_vel_w[:, self.body_ids, :3],
            "body_ang_vel": data.body_link_vel_w[:, self.body_ids, 3:]}

    def _forces(self):
        # Native sensors report primary→secondary; aux requires force on the robot (support +Z).
        return -self.sensor.data.force[:, self.contact_ids]

    def reset(self, seed=None, env_ids=None):
        if seed is not None and seed != self.seed:
            raise ValueError("Native sampler seed is frozen")
        ids = torch.arange(self.num_envs, device=self._device) if env_ids is None else env_ids
        self.lab.reset(env_ids=ids)
        root, q, dq, lift, reference = self.pool.sample(len(ids), self._device,
            tracking=self.tracking[ids, 0].cpu().numpy())
        root = root.clone(); root[:, :3] += self.lab.scene.env_origins[ids]
        self.robot.write_root_link_pose_to_sim(root[:, :7], env_ids=ids)
        self.robot.write_root_link_velocity_to_sim(root[:, 7:], env_ids=ids)
        self.robot.write_joint_state_to_sim(q, dq, joint_ids=self.joint_ids, env_ids=ids)
        self.last[ids] = action_anchor(q, control_profile="reachable_v2")
        self.previous[ids] = self.last[ids]; self.requested[ids] = self.last[ids]
        self.target[ids] = policy_to_target(self.last[ids], control_profile="reachable_v2")
        self.robot.set_joint_position_target(self.target[ids], joint_ids=self.joint_ids, env_ids=ids[:, None])
        self.reference[ids], self.lift[ids] = reference, lift
        self.episode[ids] += 1; self.steps[ids] = 0
        self.history.reset(ids); self.interval.reset(ids)
        self.interval_valid[ids] = False; self.macro[ids] = False; self.peak_force[ids] = 0
        self.lab.scene.write_data_to_sim(); self._refresh(dt=self.interval.physics_dt)
        self.cache.invalidate()
        return self.get_observations(), {}

    def get_observations(self):
        def read():
            s = self._state()
            rotation = quaternion_matrix(s["body_quat"][:, 0]).transpose(-1, -2)
            gravity = rotation @ rotation.new_tensor([0., 0., -1.])
            omega = (rotation @ s["body_ang_vel"][:, 0, :, None]).squeeze(-1)
            state = encode_state(s["q"], s["dq"], gravity, omega)
            forces = self._forces()
            mapping = {"feet": self.feet, "contact_feet": self.feet,
                "ankles": [RPO_LAB_JOINT_ORDER.index(side+"_ankle_roll_joint") for side in ("left","right")],
                "undesired_contacts": self.protected}
            raw = raw_aux_rewards({"q": s["q"], "dq": s["dq"], "torques": s["torques"],
                "joint_limits": self.robot.data.soft_joint_pos_limits[:, self.joint_ids], "torque_limits": self.limits["applied_torque"],
                "velocity_limits": self.limits["joint_vel"], "body_quat_w": s["body_quat"],
                "body_lin_vel_w": s["body_lin_vel"]}, self.last, self.previous,
                {"forces": forces, "body_names": BODY_ORDER}, mapping,
                aux_names=VELOCITY_AUX_NAMES, auxiliary_profile="soft_velocity_v7")
            interval = torch.zeros(self.num_envs, 223, device=self._device)
            if self.interval.samples == self.interval.substeps:
                raw[:, -1] = self.interval.velocity_penalty()
                interval = self.interval.trace(self.steps)
            raw[~self.interval_valid] = 0; interval[~self.interval_valid] = 0
            flags = self.interval.strict_flags.clone(); flags[~self.interval_valid] = False
            tilt = torch.acos((-gravity[:, 2:3]).clamp(-1, 1))
            trace = torch.cat((self.requested, self.last, self.target, s["dq"], s["torques"],
                flags.float(), self.peak_force, s["body_pos"][:, 0, 2:3], tilt, self.macro.float(),
                self.lift[:, None], self.interval.peak_velocity.amax(-1, keepdim=True),
                torch.full((self.num_envs, 1), self.interval.physics_dt, device=self._device),
                s["body_pos"][:, 0, :2], torch.full((self.num_envs, 1), 1e6, device=self._device)), -1)
            return {"states": state, "privileged_state": encode_privileged(s["body_pos"],
                s["body_quat"], s["body_lin_vel"], s["body_ang_vel"], self.lab.scene.env_origins[:, 2]),
                "history_actor": self.history.snapshot(), "last_action": self.last.clone(),
                "episode_id": self.episode.clone(), "step_id": self.steps.clone(),
                "ufo_aux_rewards": raw, "ufo_action_trace": trace,
                "ufo_interval_trace": interval, "ufo_reset_reference": self.reference.clone(),
                "ufo_tracking_mode": self.tracking.clone()}
        return self.cache.read(read)

    def step(self, actions):
        actions = torch.as_tensor(actions, dtype=torch.float32, device=self._device)
        if actions.shape != (self.num_envs, 23) or not torch.isfinite(actions).all():
            raise ValueError("Invalid native RPO action")
        self.history.append(self.get_observations()["states"], self.last)
        self.previous = self.last.clone(); self.requested = actions.to(self._device).clone()
        self.last = limit_action_slew(self.requested, self.previous, control_profile="reachable_v2")
        self.target = policy_to_target(self.last, control_profile="reachable_v2")
        self.steps += 1; self.interval.begin(); self.peak_force.zero_()
        for index in range(self.interval.substeps):
            self.robot.set_joint_position_target(self.target, joint_ids=self.joint_ids)
            self.lab.scene.write_data_to_sim()
            forces, applied_torque = self._completed_substep()
            s = self._state()
            foot_force = forces[:, self.feet, 2].abs()
            feet = torch.cat((s["body_pos"][:, self.feet], s["body_quat"][:, self.feet],
                s["body_lin_vel"][:, self.feet], s["body_ang_vel"][:, self.feet]), -1)
            self.interval.sample(index, s["q"], s["dq"], applied_torque, self.target,
                foot_force, forces[:, self.protected], foot_state=feet)
            self.peak_force = torch.maximum(self.peak_force, foot_force)
        rotation = quaternion_matrix(s["body_quat"][:, 0])
        height = s["body_pos"][:, 0, 2] < .53
        tilt = torch.acos(rotation[:, 2, 2].clamp(-1, 1)) > .70
        self.macro = torch.stack((self.interval.contact, height, tilt), -1)
        self.interval.finish(self.steps, self.macro.any(-1))
        self.interval_valid[:] = True; self.cache.invalidate()
        terminated = self.macro.any(-1) | self.interval.training_flags.any(-1)
        truncated = self.steps[:, 0] >= 1000
        return self.get_observations(), torch.zeros(self.num_envs, device=self._device), terminated, truncated, {}

    def close(self):
        self.lab.close()


class RPOUFOMJLabEnv:
    def __new__(cls, *args, **kwargs):
        from rlinf.envs.sim.isaaclab.isaaclab_env import IsaaclabBaseEnv

        class Bridge(IsaaclabBaseEnv):
            def _init_isaaclab_env(self):
                if self.cfg.init_params.id not in {GYM_ID, VIABILITY_GYM_ID}:
                    raise ValueError("Unknown native MJLab environment")
                profile = 'native_mjcf_viability_v13' if self.cfg.init_params.id == VIABILITY_GYM_ID else 'native_mjcf_v12'
                if os.environ.get('EB_UFO_PHYSICS_PROFILE', profile) != profile:
                    raise ValueError('Native Gym ID and physics profile differ')
                self.env = RPOUFOMJLabPhysics(num_envs=self.num_envs, seed=self.seed, physics_profile=profile)

            def _wrap_obs(self, obs):
                return obs

        return Bridge(*args, **kwargs)
