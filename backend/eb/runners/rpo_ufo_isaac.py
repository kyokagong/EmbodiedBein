"""RPO-specific raw aux, reference reset and terminal boundary for UFO/Isaac."""

from __future__ import annotations

import os

import torch

try:
    from eb.runners.bfm_ufo_agent import AUX_NAMES, VELOCITY_AUX_NAMES
    from eb.runners.rpo_bfm_codec import quaternion_matrix
except ModuleNotFoundError:
    from bfm_ufo_agent import AUX_NAMES, VELOCITY_AUX_NAMES
    from rpo_bfm_codec import quaternion_matrix

GYM_ID = "RPO-UFO-Isaac-RSI-v1"
SAFE_GYM_ID = "RPO-UFO-Isaac-RSI-safe-v2"
TRACKING_GYM_ID = "RPO-UFO-Isaac-RSI-tracking-contact-v10"
INTERVAL_TRACE_WIDTH = 223


def physical_limit_flags(q, dq, torque, limits):
    if any(not torch.isfinite(value).all() for value in (q, dq, torque)):
        raise ValueError("Nonfinite physical state")
    bounds = limits["joint_pos"]
    return torch.stack(
        (
            ((q < bounds[:, 0] - 0.001) | (q > bounds[:, 1] + 0.001)).any(-1),
            (dq.abs() > limits["joint_vel"] + 0.001).any(-1),
            (torque.abs() > limits["applied_torque"] + 0.001).any(-1),
        ),
        -1,
    )


def validate_integration(substeps, physics_dt):
    if type(substeps) is not int or (substeps, physics_dt) not in ((8, .0025), (16, .00125)):
        raise ValueError('Unapproved integration settings; control interval must remain 20ms')


class PhysicsInterval:
    """Only completed frames enter the current control interval and strict shadow."""

    def __init__(self, num_envs, limits, *, device, velocity_multiplier=1.3, compliance=False,
                 position_tolerance=.001, substeps=8, physics_dt=.0025):
        validate_integration(substeps, physics_dt)
        self.substeps, self.physics_dt = substeps, physics_dt
        if velocity_multiplier not in (1., 1.15, 1.3, 2.):
            raise ValueError("Unapproved training velocity multiplier")
        self.limits, self.multiplier = limits, velocity_multiplier
        if position_tolerance not in (.001, .02):
            raise ValueError("Unapproved training position tolerance")
        self.position_tolerance = position_tolerance
        self.compliance = compliance
        self.first_violation = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.peak_velocity = torch.zeros(num_envs, 23, device=device)
        self.peak_index = torch.zeros_like(self.peak_velocity)
        self.peak_q = torch.zeros_like(self.peak_velocity)
        self.peak_torque = torch.zeros_like(self.peak_velocity)
        self.peak_target = torch.zeros_like(self.peak_velocity)
        self.peak_signed_velocity = torch.zeros_like(self.peak_velocity)
        self.event_feet = torch.zeros(num_envs, 26, device=device)
        self.event_force = torch.zeros(num_envs, 2, device=device)
        self.event_joint = torch.zeros(num_envs, 1, device=device)
        self.penalty_sum = torch.zeros_like(self.peak_velocity)
        self.exposure = torch.zeros_like(self.peak_velocity)
        self.force_impulse = torch.zeros(num_envs, 2, device=device)
        self.strict_flags = torch.zeros(num_envs, 3, dtype=torch.bool, device=device)
        self.training_flags = self.strict_flags.clone()
        self.contact = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.samples = 0

    def begin(self):
        for value in (self.peak_velocity, self.peak_index, self.peak_q, self.peak_torque,
                      self.peak_target, self.penalty_sum, self.exposure, self.force_impulse,
                      self.strict_flags, self.training_flags, self.contact,
                      self.peak_signed_velocity, self.event_feet, self.event_force, self.event_joint):
            value.zero_()
        self.samples = 0

    def sample(self, index, q, dq, torque, target, foot_force, protected_force, *, foot_state=None):
        if index == self.samples - 1:
            return
        if index != self.samples or not 0 <= index < self.substeps:
            raise ValueError("Physics frames must be completed once in order")
        if any(not torch.isfinite(value).all() for value in (target,foot_force,protected_force)):
            raise ValueError("Nonfinite completed physics frame")
        if foot_state is not None and (foot_state.shape != (len(q),2,13) or not torch.isfinite(foot_state).all()):
            raise ValueError("Malformed event foot link state")
        flags = physical_limit_flags(q, dq, torque, self.limits)
        self.strict_flags |= flags
        training = flags.clone()
        bounds = self.limits['joint_pos']
        training[:, 0] = ((q < bounds[:, 0] - self.position_tolerance)
            | (q > bounds[:, 1] + self.position_tolerance)).any(-1)
        training[:, 1] = (dq.abs() > self.multiplier * self.limits["joint_vel"] + .001).any(-1)
        self.training_flags |= training
        relative = dq.abs() / self.limits["joint_vel"]
        if self.compliance:
            from rpo_ufo_compliance import velocity_penalty
            self.penalty_sum += velocity_penalty(relative)
        else:
            self.penalty_sum += (relative - .70).clamp_min(0).square()
        self.exposure += (dq.abs() > self.limits["joint_vel"] + .001).float() / self.substeps
        self.force_impulse += foot_force.abs() * self.physics_dt
        self.contact |= (protected_force.norm(dim=-1) > 1.).any(-1)
        higher = dq.abs() > self.peak_velocity
        event_higher = dq.abs().amax(-1) > self.peak_velocity.amax(-1)
        if foot_state is not None:
            self.event_feet[event_higher]=foot_state.reshape(len(q),26)[event_higher]
        self.event_force[event_higher]=foot_force[event_higher]
        self.event_joint[event_higher,0]=dq.abs().argmax(-1)[event_higher].float()
        for destination, incoming in ((self.peak_index, torch.full_like(dq, index)),
                                      (self.peak_q, q), (self.peak_torque, torque),
                                      (self.peak_target, target), (self.peak_signed_velocity, dq)):
            destination.copy_(torch.where(higher, incoming, destination))
        self.peak_velocity = torch.maximum(self.peak_velocity, dq.abs())
        self.samples += 1

    def velocity_penalty(self):
        if self.samples != self.substeps:
            raise ValueError("Velocity reward requires all completed frames")
        return self.penalty_sum.sum(-1) / self.substeps

    def finish(self, step_ids, macro_terminated):
        violation = self.strict_flags.any(-1) | self.contact | macro_terminated
        first = (self.first_violation == 0) & violation
        self.first_violation[first] = step_ids.reshape(-1)[first]

    def reset(self, ids):
        self.first_violation[ids] = 0
        for value in (self.peak_velocity,self.peak_index,self.peak_q,self.peak_torque,self.peak_target,
                      self.penalty_sum,self.exposure,self.force_impulse,self.strict_flags,self.training_flags,self.contact,
                      self.peak_signed_velocity,self.event_feet,self.event_force,self.event_joint):
            value[ids]=0

    def trace(self, step_ids):
        if self.samples != self.substeps:
            raise ValueError("Incomplete interval trace")
        strict_length = torch.where(self.first_violation > 0, self.first_violation, step_ids.reshape(-1))
        return torch.cat((self.peak_velocity, self.peak_index, self.peak_q, self.peak_torque,
                          self.peak_target, self.penalty_sum / self.substeps, self.exposure, self.force_impulse,
                          strict_length[:, None].float(),
                          torch.full_like(strict_length[:, None], self.samples).float(),
                          self.strict_flags.float(), self.training_flags.float(),self.peak_signed_velocity,
                          self.event_feet,self.event_force,self.event_joint), -1)


class ObservationCache:
    def __init__(self):
        self.value = None

    def invalidate(self):
        self.value = None

    def read(self, factory):
        if self.value is None:
            self.value = {key: value.clone() for key, value in factory().items()}
        return {key: value.clone() for key, value in self.value.items()}


def reference_state_batch(sampler, count, *, device, limits=None):
    samples = sampler.sample_numpy(count * 8)["sample_ids"].reshape(count, 8, 2)[:, 0]
    fields = {key: [] for key in ("root_pos", "root_quat", "root_lin_vel_w", "root_ang_vel_w", "q", "dq")}
    for clip_index, frame in samples:
        clip = sampler.clips[int(clip_index)]
        if clip["split"] != "train" or not clip["arrays"]["valid"][frame]:
            raise ValueError("Reference reset cannot use held-out/invalid state")
        for key in fields:
            fields[key].append(clip["arrays"][key][frame])
    import numpy as np

    fields = {key: torch.as_tensor(np.stack(value), device=device) for key, value in fields.items()}
    root = torch.cat(
        [fields[key] for key in ("root_pos", "root_quat", "root_lin_vel_w", "root_ang_vel_w")], -1
    )
    if limits is not None:
        unsafe = physical_limit_flags(fields["q"], fields["dq"], torch.zeros_like(fields["q"]), limits).any(
            -1
        )
        # Reject infeasible reset frames without changing immutable DatasetVersion validity.
        for _ in range(32):
            if not unsafe.any():
                break
            replacement = reference_state_batch(sampler, int(unsafe.sum()), device=device)
            root[unsafe], fields["q"][unsafe], fields["dq"][unsafe] = replacement
            unsafe = physical_limit_flags(
                fields["q"], fields["dq"], torch.zeros_like(fields["q"]), limits
            ).any(-1)
        if unsafe.any():
            raise ValueError("No physically feasible train reference reset within frozen draw budget")
    return root, fields["q"], fields["dq"]


def write_reference_state(robot, root, q, dq, env_ids, env_origins):
    root = root.clone()
    root[:, :3] += env_origins[env_ids]
    # RobotState velocities describe the link origin; the legacy API writes COM velocity.
    robot.write_root_link_state_to_sim(root, env_ids=env_ids)
    robot.write_joint_state_to_sim(q, dq, env_ids=env_ids)


def terminal_transition(next_obs, infos, terminated, truncated):
    done = (terminated | truncated).reshape(-1)
    result = {key: value.clone() for key, value in next_obs.items()}
    if done.any() and "final_observation" in infos:
        mask = infos.get("_final_observation")
        if mask is None or not torch.equal(mask.reshape(-1).bool(), done):
            raise ValueError("Terminal observation mask differs from reset environments")
        final = infos["final_observation"]
        if final.keys() != result.keys():
            raise ValueError("Terminal observation schema differs")
        for key in result:
            result[key][done] = final[key][done]
    return {
        "observation": result,
        "terminated": terminated.reshape(-1, 1).bool(),
        "truncated": truncated.reshape(-1, 1).bool(),
    }


def raw_aux_rewards(data, action, previous_action, contact, mapping, *, aux_names=AUX_NAMES, auxiliary_profile=None):
    if auxiliary_profile not in (None, "soft_velocity_v5", "soft_velocity_v7"):
        raise ValueError("Unknown UFO auxiliary profile")
    if tuple(aux_names) not in (AUX_NAMES, VELOCITY_AUX_NAMES):
        raise ValueError("Unknown UFO auxiliary reward layout")
    required = {"q", "torques", "joint_limits", "torque_limits", "body_quat_w", "body_lin_vel_w"}
    if (
        not required <= data.keys()
        or not {"forces", "body_names"} <= contact.keys()
        or not {"feet", "contact_feet", "ankles", "undesired_contacts"} <= mapping.keys()
    ):
        raise ValueError("UFO eight raw aux require frozen torque/limit/foot/contact fields")
    q, forces = data["q"], contact["forces"]
    limits, torques = data["joint_limits"], data["torques"]
    feet, contact_feet = mapping["feet"], mapping["contact_feet"]
    foot_contact = forces[:, contact_feet, 2] > 1
    gravity = quaternion_matrix(data["body_quat_w"][:, feet]).transpose(-1, -2)[..., :, 2].neg()
    undesired = mapping["undesired_contacts"]
    result = {
        "penalty_torques": torques.square().sum(-1),
        "penalty_action_rate": (action - previous_action).square().sum(-1),
        "limits_dof_pos": ((limits[..., 0] - q).clamp_min(0) + (q - limits[..., 1]).clamp_min(0)).sum(-1),
        "limits_torque": (torques.abs() - data["torque_limits"]).clamp_min(0).sum(-1),
        "penalty_undesired_contact": (forces[:, undesired].abs() > 1).any(dim=(1, 2)).float(),
        "penalty_feet_ori": (gravity[..., :2].square().sum(-1).sqrt() * foot_contact).sum(-1),
        "penalty_ankle_roll": q[:, mapping["ankles"]].square().sum(-1),
        "penalty_slippage": (
            data["body_lin_vel_w"][:, feet].norm(dim=-1) * (forces[:, contact_feet].norm(dim=-1) > 1)
        ).sum(-1),
    }
    if tuple(aux_names) == VELOCITY_AUX_NAMES:
        dq, velocity = data.get("dq"), data.get("velocity_limits")
        if (
            dq is None
            or velocity is None
            or dq.shape != q.shape
            or velocity.shape not in (q.shape, q.shape[-1:])
            or not torch.isfinite(dq).all()
            or not torch.isfinite(velocity).all()
            or (velocity <= 0).any()
        ):
            raise ValueError("Invalid UFO velocity state or rated velocity limits")
        result["limits_dof_vel"] = (dq.abs() / velocity - (0.7 if auxiliary_profile == "soft_velocity_v7" else 0.8)).clamp_min(0).square().sum(-1)
    if set(result) != set(aux_names) or any(not torch.isfinite(value).all() for value in result.values()):
        raise ValueError("Nonfinite raw UFO auxiliary reward")
    return torch.stack([result[key] for key in aux_names], -1)


def register_task():
    import gymnasium as gym

    for id_ in (GYM_ID, SAFE_GYM_ID, "RPO-UFO-Isaac-RSI-slew-v4", "RPO-UFO-Isaac-RSI-slew-velocity-v5", "RPO-UFO-Isaac-RSI-reachable-v7", "RPO-UFO-Isaac-RSI-contact-v8", "RPO-UFO-Isaac-RSI-ground-v9", TRACKING_GYM_ID):
        if id_ in gym.registry:
            continue
        gym.register(
            id=id_,
            entry_point="rpo_ufo_isaac:RPOUFOIsaacPhysics",
            kwargs={"env_cfg_entry_point": "rpo_bfm_extension:RPOBFMEnvCfg"},
            disable_env_checker=True,
        )


class RPOUFOIsaacEnv:
    def __new__(cls, *args, **kwargs):
        from rlinf.envs.sim.isaaclab.isaaclab_env import IsaaclabBaseEnv

        class Bridge(IsaaclabBaseEnv):
            def _make_env_function(self):
                def make_env():
                    from isaaclab.app import AppLauncher

                    app = AppLauncher(headless=True, enable_cameras=False).app
                    import gymnasium as gym
                    from rpo_bfm_extension import RPOBFMEnvCfg

                    register_task()
                    cfg = RPOBFMEnvCfg()
                    cfg.seed, cfg.scene.num_envs = self.seed, self.cfg.init_params.num_envs
                    return gym.make(self.cfg.init_params.id, cfg=cfg).unwrapped, app

                return make_env

            def _wrap_obs(self, obs):
                keys = (
                    "states",
                    "privileged_state",
                    "last_action",
                    "history_actor",
                    "episode_id",
                    "step_id",
                    "ufo_aux_rewards",
                )
                if self.cfg.init_params.id in {
                    SAFE_GYM_ID,
                    "RPO-UFO-Isaac-RSI-slew-v4",
                    "RPO-UFO-Isaac-RSI-slew-velocity-v5",
                    "RPO-UFO-Isaac-RSI-reachable-v7",
                    "RPO-UFO-Isaac-RSI-contact-v8",
                    "RPO-UFO-Isaac-RSI-ground-v9",
                    TRACKING_GYM_ID,
                }:
                    keys = (*keys, "ufo_action_trace")
                if self.cfg.init_params.id == TRACKING_GYM_ID:
                    keys = (*keys, "ufo_reset_reference", "ufo_interval_trace", "ufo_tracking_mode")
                return {key: obs[key] for key in keys}

        return Bridge(*args, **kwargs)


def __getattr__(name):
    if name != "RPOUFOIsaacPhysics":
        raise AttributeError(name)
    from rpo_bfm import ACTION_GAIN, DEFAULT_Q, CODEC_HASH, CODEC_V2_HASH, BODY_ORDER, RPO_LAB_JOINT_ORDER
    from rpo_bfm_codec import policy_to_target
    from rpo_bfm_extension import RPOBFMIsaacEnv
    from rpo_flat_extension import FlatResetCompat
    from rpo_ufo_motion import load_robot_state

    class Physics(RPOBFMIsaacEnv):
        def __init__(self, *args, **kwargs):
            self.ufo_cache = ObservationCache()
            self.ufo_safe = os.environ.get("EB_UFO_LEARNING_PROFILE") in {
                "physical_limits_v2",
                "physical_limits_slew_v4",
                "physical_limits_stable_slew_v4",
            }
            auxiliary = os.environ.get("EB_UFO_AUXILIARY_PROFILE")
            if auxiliary not in (None, "soft_velocity_v5", "soft_velocity_v7") or auxiliary and not self.ufo_safe:
                raise ValueError("Unknown or incompatible UFO auxiliary environment profile")
            self.ufo_auxiliary_profile = auxiliary
            self.ufo_tracking_contact = os.environ.get("EB_UFO_CONTEXT_PROFILE") == "environment_tracking_v10"
            self.ufo_ground = os.environ.get("EB_UFO_GROUND_PROFILE") == "wide_plane_v9"
            self.ufo_contact = os.environ.get("EB_UFO_CONTACT_PROFILE") == "contact_stable_v8"
            self.ufo_aux_names = VELOCITY_AUX_NAMES if auxiliary else AUX_NAMES
            super().__init__(*args, **kwargs)
            from pathlib import Path

            _, self.ufo_reference = load_robot_state(
                Path(os.environ["EB_UFO_MOTION_ROOT"]),
                asset_sha256=os.environ["EB_UFO_ASSET_SHA256"],
                seed=self.cfg.seed,
                expected_codec_sha256=CODEC_V2_HASH if self.bfm_control_profile == "reachable_v2" else CODEC_HASH,
            )
            contact_names = self.contact_sensor.body_names
            feet_names = [side + "_ankle_roll_link" for side in ("left", "right")]
            self.ufo_mapping = {
                "feet": [BODY_ORDER.index(n) for n in feet_names],
                "contact_feet": [contact_names.index(n) for n in feet_names],
                "ankles": [
                    RPO_LAB_JOINT_ORDER.index(side + "_ankle_roll_joint") for side in ("left", "right")
                ],
                "undesired_contacts": list(self.termination_contact_cfg.body_ids),
            }
            if self.ufo_safe:
                from rpo_flat_artifact import _hard_limits

                self.ufo_limits = {
                    key: torch.as_tensor(value, device=self.device)
                    for key, value in _hard_limits(self).items()
                }
                self.ufo_requested_action = self.bfm_last_action.clone()
            if self.ufo_contact:
                from rpo_ufo_contact import ContactResetPool
                self.ufo_reset_pool = ContactResetPool(self.ufo_reference, os.environ["RPO_ASSET_ROOT"],
                    tracking_contact=self.ufo_tracking_contact,
                    flexion=os.environ.get('EB_UFO_DYNAMICS_PROFILE')=='contact_compliance_v11')
                if self.ufo_ground:
                    from rpo_ufo_ground import SIZE_M, coverage_receipt
                    if tuple(self.cfg.scene.terrain.terrain_generator.size) != SIZE_M:
                        raise ValueError("Ground mesh differs from frozen v9 size")
                    self.ufo_ground_coverage = coverage_receipt(self.scene.env_origins.cpu().numpy(),
                        self.ufo_reset_pool.foot_bounds, SIZE_M)
                self.ufo_reset_lift = torch.zeros(self.num_envs, device=self.device)
                self.ufo_substep_hard = torch.zeros(self.num_envs, 3, dtype=torch.bool, device=self.device)
                self.ufo_substep_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                self.ufo_peak_velocity = torch.zeros(self.num_envs, device=self.device)
                self.ufo_peak_contact = torch.zeros(self.num_envs, 2, device=self.device)
                self.ufo_apply_count = 0
                if self.ufo_tracking_contact:
                    from bfm_ufo_rollout_context import tracking_mask
                    self.ufo_tracking_mask = tracking_mask(self.num_envs, int(self.cfg.seed))
                    self.ufo_reset_reference = torch.full((self.num_envs,3),-1,dtype=torch.long,device=self.device)
                    self.ufo_interval = PhysicsInterval(self.num_envs,self.ufo_limits,device=self.device,
                        velocity_multiplier=1.3,
                        compliance=os.environ.get('EB_UFO_DYNAMICS_PROFILE')=='contact_compliance_v11')
                import json
                path = os.environ.get("EB_UFO_DIAGNOSTICS_DIR")
                if path:
                    Path(path).mkdir(parents=True, exist_ok=True)
                    (Path(path) / f"rsi-pool-{os.getpid()}.json").write_text(json.dumps(self.ufo_reset_pool.rows))
                    if self.ufo_ground:
                        (Path(path) / f"ground-coverage-{os.getpid()}.json").write_text(json.dumps(self.ufo_ground_coverage))
                    if self.ufo_tracking_contact:
                        from rpo_ufo_contact import physics_startup_receipt
                        receipt=physics_startup_receipt(self.robot.root_physx_view.get_masses().cpu().numpy(),
                            self.ufo_tracking_mask,seed=self.cfg.seed)
                        receipt['nominal_total_mass_kg']=float(self.robot.data.default_mass[0].sum())
                        receipt['actuator_gains']={name:{key:[float(getattr(actuator,key).min()),
                            float(getattr(actuator,key).max())] for key in ('stiffness','damping','effort_limit')}
                            for name,actuator in self.robot.actuators.items()}
                        if os.environ.get('EB_UFO_DYNAMICS_PROFILE')=='contact_compliance_v11':
                            from rpo_ufo_compliance import material_receipt
                            receipt['contact_material']=material_receipt(self.sim.stage,
                                compliant=os.environ.get('EB_UFO_CONTACT_MATERIAL')!='rigid')
                        (Path(path)/f"physics-startup-{os.getpid()}.json").write_text(json.dumps(receipt))

        def reference_reset(self, env_ids):
            if getattr(self, "ufo_contact", False) and hasattr(self, "ufo_reset_pool"):
                if self.ufo_tracking_contact:
                    root,q,dq,lift,reference=self.ufo_reset_pool.sample(len(env_ids),self.device,
                        tracking=self.ufo_tracking_mask[env_ids.cpu().numpy()])
                    self.ufo_reset_reference[env_ids]=reference
                    self.ufo_interval.reset(env_ids)
                else:
                    root, q, dq, lift = self.ufo_reset_pool.sample(len(env_ids), self.device)
                self.ufo_reset_lift[env_ids] = lift
            else:
                root, q, dq = reference_state_batch(
                    self.ufo_reference, len(env_ids), device=self.device, limits=getattr(self, "ufo_limits", None))
            write_reference_state(self.robot, root, q, dq, env_ids, self.scene.env_origins)
            if self.bfm_slew:
                self._reset_slew_anchor(env_ids)
            if getattr(self, "ufo_contact", False):
                # Base reset forwarded its temporary state before RSI overwrote q/root.
                self.contact_sensor.reset(env_ids)
                self.scene.write_data_to_sim()
                self.sim.forward()
                self.robot.update(0.)
            self.ufo_cache.invalidate()

        def _sample_substep(self):
            data = self.robot.data
            self.ufo_substep_hard |= physical_limit_flags(data.joint_pos, data.joint_vel, data.applied_torque, self.ufo_limits)
            protected = self.contact_sensor.data.net_forces_w[:, self.termination_contact_cfg.body_ids]
            self.ufo_substep_contact |= (protected.norm(dim=-1)>1.).any(-1)
            self.ufo_peak_velocity = torch.maximum(self.ufo_peak_velocity, data.joint_vel.abs().amax(-1))
            force = self.contact_sensor.data.net_forces_w[:, self.ufo_mapping["contact_feet"], 2].abs()
            self.ufo_peak_contact = torch.maximum(self.ufo_peak_contact, force)
            if getattr(self, "ufo_tracking_contact", False):
                feet=self.ufo_mapping['feet']
                foot_state=torch.cat((data.body_link_pos_w[:,feet],data.body_link_quat_w[:,feet],
                    data.body_link_lin_vel_w[:,feet],data.body_ang_vel_w[:,feet]),-1)
                self.ufo_interval.sample(self.ufo_apply_count-1,data.joint_pos,data.joint_vel,
                    data.applied_torque,data.joint_pos_target,force,protected,foot_state=foot_state)

        def _apply_action(self):
            if getattr(self, "ufo_contact", False) and hasattr(self, "ufo_substep_hard"):
                # Contact force at interval start belongs to the previous physics frame, including after RSI.
                if self.ufo_apply_count:
                    self._sample_substep()
                self.ufo_apply_count += 1
            return super()._apply_action()

        def _reset_idx(self, env_ids):
            if getattr(self, "_flat_defer_reset", False):
                return
            super()._reset_idx(env_ids)
            if hasattr(self, "ufo_reference") and len(env_ids):
                self.reference_reset(env_ids)
            self.ufo_cache.invalidate()

        def _get_observations(self):
            def read():
                value = super(Physics, self)._get_observations()
                value["ufo_aux_rewards"] = (
                    self._ufo_aux()
                    if hasattr(self, "ufo_mapping")
                    else torch.zeros(self.num_envs, len(self.ufo_aux_names), device=self.device)
                )
                if self.ufo_safe:
                    data = self.robot.data
                    flags = (
                        physical_limit_flags(
                            data.joint_pos, data.joint_vel, data.applied_torque, self.ufo_limits
                        )
                        if hasattr(self, "ufo_limits")
                        else torch.zeros(self.num_envs, 3, device=self.device)
                    )
                    if self.ufo_contact and hasattr(self, "ufo_substep_hard"):
                        flags = flags | self.ufo_substep_hard
                    value["ufo_action_trace"] = torch.cat(
                        (
                            getattr(self, "ufo_requested_action", self.bfm_last_action),
                            self.bfm_last_action,
                            data.joint_pos_target,
                            data.joint_vel,
                            data.applied_torque,
                            flags.float(),
                        ),
                        -1,
                    )
                    if self.ufo_contact and hasattr(self, "ufo_substep_hard"):
                        contact, height, tilt = self._contact_termination_reasons()
                        extra = torch.cat((self.ufo_peak_contact, data.root_pos_w[:,2:3],
                            torch.acos((-data.projected_gravity_b[:,2:3]).clamp(-1,1)),
                            torch.stack((contact,height,tilt),-1).float(), self.ufo_reset_lift[:,None],
                            self.ufo_peak_velocity[:,None], torch.full((self.num_envs,1),self.physics_dt,device=self.device)), -1)
                        value["ufo_action_trace"] = torch.cat((value["ufo_action_trace"],extra),-1)
                if getattr(self, "ufo_ground", False) and hasattr(self, "ufo_reset_pool"):
                    from rpo_ufo_ground import SIZE_M
                    feet = data.body_link_pos_w[:,self.ufo_mapping["feet"],:2]
                    margin = (torch.as_tensor(SIZE_M,device=self.device)/2-feet.abs()-self.ufo_reset_pool.foot_radius).amin(dim=(1,2))
                    value["ufo_action_trace"] = torch.cat((value["ufo_action_trace"],data.root_pos_w[:,:2],margin[:,None]),-1)
                if getattr(self,"ufo_tracking_contact",False) and hasattr(self,"ufo_interval"):
                    value["ufo_reset_reference"]=self.ufo_reset_reference.clone()
                    value["ufo_tracking_mode"]=torch.as_tensor(self.ufo_tracking_mask,device=self.device)[:,None]
                    value["ufo_interval_trace"]=(self.ufo_interval.trace(self.bfm_step_id) if self.ufo_interval.samples==8
                        else torch.zeros(self.num_envs,INTERVAL_TRACE_WIDTH,device=self.device))
                return value

            return self.ufo_cache.read(read)

        def _ufo_aux(self):
            data = self.robot.data
            result = raw_aux_rewards(
                {
                    "q": data.joint_pos,
                    "torques": data.applied_torque,
                    "joint_limits": data.soft_joint_pos_limits,
                    "torque_limits": data.joint_effort_limits,
                    "body_quat_w": data.body_quat_w,
                    "body_lin_vel_w": data.body_link_lin_vel_w,
                    **(
                        {"dq": data.joint_vel, "velocity_limits": self.ufo_limits["joint_vel"]}
                        if self.ufo_aux_names == VELOCITY_AUX_NAMES
                        else {}
                    ),
                },
                self.bfm_last_action,
                self.bfm_previous_action,
                {
                    "forces": self.contact_sensor.data.net_forces_w,
                    "body_names": self.contact_sensor.body_names,
                },
                self.ufo_mapping,
                aux_names=self.ufo_aux_names, auxiliary_profile=self.ufo_auxiliary_profile,
            )
            if getattr(self,"ufo_tracking_contact",False) and hasattr(self,"ufo_interval"):
                result[:,-1]=(self.ufo_interval.velocity_penalty() if self.ufo_interval.samples
                    else torch.zeros(self.num_envs,device=self.device))
            return result

        def _get_rewards(self):
            return torch.zeros(self.num_envs, device=self.device)

        def _get_dones(self):
            terminated, truncated = super()._get_dones()
            if self.ufo_contact and hasattr(self, "ufo_substep_hard"):
                self._sample_substep()
                if getattr(self,"ufo_tracking_contact",False):
                    self.ufo_interval.finish(self.bfm_step_id,terminated)
                    terminated |= self.ufo_interval.training_flags.any(-1) | self.ufo_substep_contact
                else:
                    terminated |= self.ufo_substep_hard.any(-1) | self.ufo_substep_contact
            if self.ufo_safe and hasattr(self, "ufo_limits"):
                data = self.robot.data
                flags=physical_limit_flags(data.joint_pos,data.joint_vel,data.applied_torque,self.ufo_limits)
                if getattr(self,"ufo_tracking_contact",False):
                    flags[:,1]=(data.joint_vel.abs()>1.3*self.ufo_limits["joint_vel"]+.001).any(-1)
                terminated |= flags.any(-1)
            return terminated, truncated

        def _contact_termination_reasons(self):
            data = self.robot.data
            forces = self.contact_sensor.data.net_forces_w_history[:,:,self.termination_contact_cfg.body_ids]
            contact = (forces.norm(dim=-1).amax(1)>1.).any(-1)
            if hasattr(self, "ufo_substep_contact"):
                contact |= self.ufo_substep_contact
            height = data.root_pos_w[:,2]<self.cfg.robot.terminate_base_height
            tilt = torch.acos((-data.projected_gravity_b[:,2]).clamp(-1,1))>self.cfg.robot.terminate_base_orientation
            return contact,height,tilt

        def step(self, actions):
            if actions.shape != (self.num_envs, 23) or not torch.isfinite(actions).all():
                raise ValueError("Invalid UFO RPO action")
            current = self._get_observations()
            self.bfm_history.append(current["states"], self.bfm_last_action)
            self.bfm_previous_action = self.bfm_last_action.clone()
            if getattr(self, "ufo_safe", False):
                self.ufo_requested_action = actions.to(self.device).clone()
            if getattr(self, "bfm_slew", False):
                from rpo_bfm_codec import limit_action_slew
                requested=actions.to(self.device)
                if getattr(self,"ufo_tracking_contact",False) and os.environ.get("EB_UFO_ANKLE_CONTROL") == "lpf15":
                    from rpo_bfm_codec import filter_ankle_action
                    requested=filter_ankle_action(requested,self.bfm_previous_action)
                self.bfm_last_action = limit_action_slew(requested, self.bfm_previous_action, control_profile=self.bfm_control_profile)
            else:
                self.bfm_last_action = actions.to(self.device).clamp(-1, 1)
            self.bfm_step_id += 1
            self.ufo_cache.invalidate()
            if self.ufo_contact and hasattr(self, "ufo_substep_hard"):
                self.ufo_apply_count = 0
                self.ufo_substep_hard.zero_()
                self.ufo_substep_contact.zero_()
                self.ufo_peak_velocity.zero_()
                self.ufo_peak_contact.zero_()
                if getattr(self,"ufo_tracking_contact",False):
                    self.ufo_interval.begin()
            flat_action = ((policy_to_target(self.bfm_last_action, control_profile="reachable_v2") - self.bfm_last_action.new_tensor(DEFAULT_Q)) / .25
                           if getattr(self, "bfm_control_profile", "v1") == "reachable_v2"
                           else self.bfm_last_action * self.bfm_last_action.new_tensor(ACTION_GAIN))
            result = FlatResetCompat.step(
                self, flat_action
            )
            return result

    return Physics
