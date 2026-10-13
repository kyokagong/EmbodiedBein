"""Frozen RPO BFM metadata. No framework dependencies or runtime IO."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import PurePosixPath

try:
    from eb.rpo_wbc import RPO_LAB_JOINT_ORDER
except ModuleNotFoundError:
    from rpo_wbc import RPO_LAB_JOINT_ORDER


ENV_ID = "ee_rpo_v2_bfm_23dof_v1"
ENV_V2_ID = "ee_rpo_v2_bfm_23dof_v2"
REACHABLE_RECIPE = "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_reachable_v7"
GROUND_RECIPE = "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_ground_v9"
TRACKING_CONTACT_RECIPE = "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_tracking_contact_v10"
COMPLIANCE_RECIPE = "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_compliance_v11"
MJLAB_RECIPE = "rpo_bfm_ufo_agent_rlinf_mjlab_4gpu_v12"
MJLAB_VIABILITY_RECIPE = "rpo_bfm_ufo_agent_rlinf_mjlab_4gpu_viability_v13"
MJLAB_PHYSICS_PROFILE = {
    "schema_version": "eb/ufo-mjlab-physics/v1", "backend": "mjlab",
    "physics_dt_s": .0025, "decimation": 8, "control_dt_s": .02,
    "solver": "newton", "iterations": 100, "integrator": "implicitfast",
    "geometry": "original RPO MJCF collision meshes", "self_collision": False,
    "joint_position_limits": "frozen RPO interface limits; 90% soft position reward margin",
    "contact": "original MJCF solref/solimp/friction; native infinite plane",
    "actuator": "MJLab DcMotorActuator, zero delay, original RPO gains, ankle Kd2.5",
    "armature_kg_m2": .01, "velocity_limit": "DC motor torque-speed curve; no qvel clipping",
    "training_velocity_termination_multiplier": 1.3,
    "strict_shadow": "original rated position/velocity/torque and macro fall, all substeps",
    "codec_metadata_note": "published physics_hz=200 is historical; runtime=400Hz",
}
TRACKING_CONTACT_KD_RECIPE = "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_tracking_contact_kd_v10"
MJLAB_VIABILITY_PHYSICS_PROFILE = {**MJLAB_PHYSICS_PROFILE,
    "schema_version": "eb/ufo-mjlab-physics/v2", "rsi_joint_lookahead_s": .02,
    "rsi_joint_margin_rad": .001, "training_position_tolerance_rad": .02,
    "strict_position_tolerance_rad": .001,
    "armature_evidence": "original asset value; rotor inertia and reduction ratio unavailable"}
CONTACT_RECIPE = "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_contact_v8"
# Frozen rpo.urdf finite limits in native RPO_LAB_JOINT_ORDER; observation zero remains DEFAULT_Q.
JOINT_POSITION_MIN = (-1., -.2, -3.14, -.2, -1., -3.14, -3.14, -2.094, -2.094, -.25, -3.14, -.2, -.2, -1.57, -1.57, -.6, -.6, -.6, -.6, -.5, -.5, -1.57, -1.57)
JOINT_POSITION_MAX = (.2, 1., 3.14, 1., .2, 1.57, 1.57, .7854, .7854, 3.14, .25, 2.5, 2.5, 1.57, 1.57, .6, .6, 1.57, 1.57, .5, .5, 1.57, 1.57)
ACTION_MIDPOINT_V2 = tuple((lo + hi) / 2 for lo, hi in zip(JOINT_POSITION_MIN, JOINT_POSITION_MAX))
ACTION_HALFWIDTH_V2 = tuple((hi - lo) / 2 for lo, hi in zip(JOINT_POSITION_MIN, JOINT_POSITION_MAX))
TASK_ID = "tsk_rpo_bfm_pretrain"
GYM_ID = "RPO-BFM-Pretrain-v1"
TEMPLATE_ID = "rpo_v2_bfm_zero_pretrain"
MOTION_CONVERTER_REVISION = "isaac-fk-raw-wxyz-v2"
LOCAL_POOL_CONVERTER_REVISION = "isaac-fk-local-pool-wxyz-v3"
REFERENCE_POOL_CONVERTER_REVISION = "isaac-fk-reference-pool-wxyz-v5"
UNCHECKED_REFERENCE_REVISION = "isaac-fk-reference-pool-wxyz-v4"
OFFICIAL_REFERENCE_IMAGE = "eb/rlinf:rpo-v2-bfm-reference-v5"
G1_CONVERTER_REVISION = "isaac-fk-g1-calibrated-wxyz-v1"
MIXED_POOL_V1 = "isaac-fk-g1-rpo-mixed-pool-wxyz-v1"
MIXED_POOL_CONVERTER_REVISION = "isaac-fk-g1-rpo-mixed-pool-wxyz-v2"
G1_IMAGE = "eb/rlinf:rpo-v2-bfm-four-gpu-v6"
REFERENCE_IMAGE = G1_IMAGE
TRAIN_IMAGE="eb/rlinf:rpo-v2-bfm-walking-v10"
EVAL_IMAGE="eb/rlinf:rpo-v2-bfm-walking-eval-v11"
EVAL_TEMPLATE="rpo_v2_bfm_zero_eval"
EVAL_SUITE="suite_rpo_v2_bfm_isaac_velocity_commands_v2"
EVAL_BATCH_SUITE="suite_rpo_v2_bfm_isaac_velocity_commands_v3"
VALIDATION_SUITE="suite_rpo_v2_bfm_isaac_velocity_validation_v2"
UFO_VALIDATION_SUITE="suite_rpo_ufo_isaac_velocity_validation_v1"
UFO_ACCEPTANCE_SUITE="suite_rpo_ufo_isaac_velocity_acceptance_v1"
UFO_REPAIR_ACCEPTANCE_SUITE="suite_rpo_ufo_isaac_velocity_acceptance_v2"
ACCEPTANCE_SUITE="suite_rpo_v2_bfm_isaac_velocity_acceptance_v2"
EVAL_SUITE_V1="suite_rpo_v2_bfm_isaac_velocity_commands_v1"
LEGACY_CONVERTERS = {"isaac-fk-raw-resample-v1", "isaac-fk-local-pool-v2"}
BFM_REVISION = "318cf44a3262e5bdec5944f82f1a5f509b95d09b"
BODY_ORDER = ("base_link", *(name.replace("_joint", "_link") for name in RPO_LAB_JOINT_ORDER))
DEFAULT_Q = (0., 0., 0., 0., 0., .18, .18, -.1, -.1, .06, -.06, .3, .3,
             0., 0., -.2, -.2, .78, .78, 0., 0., 0., 0.)
# Measured diagnostic v4 offsets, ceil(1.1 * max(abs(q-default)) / .25), minimum 1.
ACTION_GAIN = (1., 2., 2., 1., 1., 2., 2., 6., 4., 2., 1., 9., 7., 3., 2., 2., 3., 2., 3., 1., 1., 1., 1.)
SLEW_CONTROL = {
    "schema_version": "eb/rpo-target-slew/v1", "control_dt_s": .02,
    "target_velocity_limit_fraction": .5,
    "joint_velocity_limits_rad_s": [25., 25., 25., 25., 25., 8., 8., 25., 25., 8., 8.,
        25., 25., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8.],
    "reset": "project measured q into v1 target box; last_action=anchor; clear history",
    "replay_action": "requested", "observation_last_action": "applied",
    "guarantee": "target increments only; actual dq and infeasible reset residual remain measured",
}
OBS_DIMS = {"states": 52, "last_action": 23, "history_actor": 300, "privileged_state": 358}


def action_control_for_execution(execution):
    if (execution or {}).get("recipe_id") == MJLAB_VIABILITY_RECIPE:
        return {**REACHABLE_CONTROL, "physics": MJLAB_VIABILITY_PHYSICS_PROFILE,
                "tracking_context": "environment_tracking_v10"}
    if (execution or {}).get("recipe_id") == MJLAB_RECIPE:
        return {**REACHABLE_CONTROL, "physics": MJLAB_PHYSICS_PROFILE,
                "tracking_context": "environment_tracking_v10"}
    if (execution or {}).get("recipe_id") == COMPLIANCE_RECIPE:
        return {**action_control_for_execution({"recipe_id":TRACKING_CONTACT_KD_RECIPE}),
            "dynamics_profile":"contact_compliance_v11","contact_stiffness_N_per_m":60000.,
            "contact_damping_Ns_per_m":600.,"min_rsi_knee_flexion_rad":.15,
            "max_rsi_hip_pitch_rad":.10,"rsi_flexion_scope":"near-ground support legs only",
            "velocity_penalty":"relu(r-.7)^2+8*relu(r-1)^2"}
    if (execution or {}).get("recipe_id") == TRACKING_CONTACT_KD_RECIPE:
        control=action_control_for_execution({"recipe_id":TRACKING_CONTACT_RECIPE})
        control.pop("ankle_filter_cutoff_hz")
        return {**control,"ankle_control_profile":"kd25","ankle_nominal_damping":2.5}
    if (execution or {}).get("recipe_id") == TRACKING_CONTACT_RECIPE:
        return {**action_control_for_execution({"recipe_id":GROUND_RECIPE}),
                "tracking_context":"environment_tracking_v10","training_velocity_termination_multiplier":1.3,
                "velocity_reward_aggregation":"mean_eight_completed_physics_frames",
                "rsi_support_normal_velocity_max_mps":.15,"ankle_filter_cutoff_hz":15.}
    if (execution or {}).get("recipe_id") == GROUND_RECIPE:
        try:
            from eb.runners.rpo_ufo_ground import PROFILE, SIZE_M
        except ModuleNotFoundError:
            from rpo_ufo_ground import PROFILE, SIZE_M
        return {**action_control_for_execution({"recipe_id": CONTACT_RECIPE}),
            "ground": {"profile": PROFILE, "size_m": list(SIZE_M), "motion_reserve_m": 20.}}
    if (execution or {}).get("recipe_id") == CONTACT_RECIPE:
        try:
            from eb.runners.rpo_ufo_contact import SETTINGS
        except ModuleNotFoundError:
            from rpo_ufo_contact import SETTINGS
        return {**REACHABLE_CONTROL, "physics": SETTINGS}
    if (execution or {}).get("recipe_id") == REACHABLE_RECIPE:
        return REACHABLE_CONTROL
    return SLEW_CONTROL if (execution or {}).get("recipe_id") in {
        "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_slew_v4", "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_slew_velocity_v5",
        "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_scale6144_v6",
    } else None


def auxiliary_profile_for_execution(execution):
    if (execution or {}).get("recipe_id") in {MJLAB_VIABILITY_RECIPE,MJLAB_RECIPE,COMPLIANCE_RECIPE,TRACKING_CONTACT_RECIPE,TRACKING_CONTACT_KD_RECIPE, REACHABLE_RECIPE, CONTACT_RECIPE, GROUND_RECIPE}:
        return "soft_velocity_v7"
    return "soft_velocity_v5" if (execution or {}).get("recipe_id") in {"rpo_bfm_ufo_agent_rlinf_isaac_4gpu_slew_velocity_v5", "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_scale6144_v6"} else None
PROMPT_COMMAND = {"field": "commands", "dim": 3, "units": ["m/s", "m/s", "rad/s"],
                  "minimum": [-.6, -.5, -1.57], "maximum": [1., .5, 1.57],
                  "consumer": "command_to_latent"}


def spec(interface_id=ENV_ID) -> dict:
    if interface_id not in {ENV_ID, ENV_V2_ID}:
        raise ValueError("Unknown frozen BFM interface")
    value = {
        "schema_version": "eb/rpo-bfm-codec/v1", "interface_id": ENV_ID,
        "joint_order": list(RPO_LAB_JOINT_ORDER), "body_order": list(BODY_ORDER),
        "observation_dims": dict(OBS_DIMS),
        "state_layout": ["q_minus_default", "dq_radps", "gravity_body", "omega_body_div4"],
        "privileged_layout": ["root_height", "heading_local_positions_without_root",
                              "heading_local_rotation_x_z", "heading_local_lin_vel", "heading_local_ang_vel"],
        "quaternion": "wxyz", "history_layout": "frame_major_oldest_first_zero_pad",
        "history_frame": ["states", "last_action"], "history_length": 4,
        "default_joint_pos": list(DEFAULT_Q), "action_gain": list(ACTION_GAIN),
        "action_scale": .25, "policy_action_bounds": [-1., 1.], "actuator_delay_steps": 0,
        "policy_hz": 50, "physics_hz": 200, "z_dim": 256,
        "privileged_velocity_point": "link_origin",
        "prompt_command": PROMPT_COMMAND,
    }
    if interface_id == ENV_V2_ID:
        value.update(schema_version="eb/rpo-bfm-codec/v2", interface_id=ENV_V2_ID,
                     action_offset=list(ACTION_MIDPOINT_V2), action_scale=list(ACTION_HALFWIDTH_V2),
                     action_gain=[1.] * 23, joint_position_min=list(JOINT_POSITION_MIN),
                     joint_position_max=list(JOINT_POSITION_MAX), action_mapping="urdf_finite_midpoint_halfwidth")
    return value


CODEC_HASH = hashlib.sha256(json.dumps(spec(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()
CODEC_V2_HASH = hashlib.sha256(json.dumps(spec(ENV_V2_ID), sort_keys=True, separators=(",", ":")).encode()).hexdigest()
REACHABLE_CONTROL = {**SLEW_CONTROL, "schema_version": "eb/rpo-target-slew/v2",
    "control_profile": "reachable_v2", "interface_id": ENV_V2_ID, "codec_sha256": CODEC_V2_HASH,
    "joint_position_min_rad": list(JOINT_POSITION_MIN), "joint_position_max_rad": list(JOINT_POSITION_MAX),
    "action_midpoint_rad": list(ACTION_MIDPOINT_V2), "action_halfwidth_rad": list(ACTION_HALFWIDTH_V2),
    "observation_default_joint_pos_rad": list(DEFAULT_Q),
    "reset": "clamp measured q to physical finite bounds; normalize anchor; clear history"}


def codec_hash_for_execution(execution):
    return CODEC_V2_HASH if (execution or {}).get("recipe_id") in {MJLAB_VIABILITY_RECIPE,MJLAB_RECIPE,COMPLIANCE_RECIPE,TRACKING_CONTACT_RECIPE,TRACKING_CONTACT_KD_RECIPE, REACHABLE_RECIPE, CONTACT_RECIPE, GROUND_RECIPE} else CODEC_HASH


def validate_motion_manifest(manifest: dict, *, asset_sha256: str) -> dict:
    if (manifest.get("schema_version") != "eb/rpo-bfm-motion/v1"
            or manifest.get("purpose") != "behavior_prior" or manifest.get("usage") != "train"
            or manifest.get("joint_order") != list(RPO_LAB_JOINT_ORDER)
            or manifest.get("body_order") != list(BODY_ORDER)
            or manifest.get("converter_revision") not in {MOTION_CONVERTER_REVISION, LOCAL_POOL_CONVERTER_REVISION, REFERENCE_POOL_CONVERTER_REVISION, UNCHECKED_REFERENCE_REVISION, G1_CONVERTER_REVISION, MIXED_POOL_CONVERTER_REVISION,MIXED_POOL_V1, *LEGACY_CONVERTERS}
            or manifest.get("codec_sha256") != CODEC_HASH
            or manifest.get("asset_sha256") != asset_sha256):
        raise ValueError("BFM motion schema, usage, joint/body order or frozen codec/asset mismatch")
    refs = manifest.get("source_refs")
    if (not isinstance(refs, list) or not refs or any(
            not isinstance(ref, dict) or not ref.get("uri") or not ref.get("license")
            or not re.fullmatch(r"[0-9a-f]{40}", str(ref.get("revision", "")))
            or not re.fullmatch(r"[0-9a-f]{64}", str(ref.get("sha256", ""))) for ref in refs)):
        raise ValueError("BFM motion requires pinned source hashes and license provenance")
    clips = manifest.get("clips")
    if not isinstance(clips, list) or not clips:
        raise ValueError("BFM motion has no clips")
    ids, files, groups = set(), set(), {}
    for clip in clips:
        if not isinstance(clip, dict):
            raise ValueError("BFM clip metadata is invalid")
        name = str(clip.get("file", ""))
        path = PurePosixPath(name)
        if (path.is_absolute() or len(path.parts) != 1 or "\\" in name
                or path.suffix != ".npz" or name in files
                or not re.fullmatch(r"[0-9a-f]{64}", str(clip.get("sha256", "")))
                or not re.fullmatch(r"[A-Za-z0-9_.-]+", str(clip.get("id", "")))
                or clip["id"] in ids or clip.get("fps") != 50
                or type(clip.get("frames")) is not int or clip["frames"] < 9
                or clip.get("split") not in {"train", "validation", "test"}
                or not isinstance(clip.get("source_group"), str) or not clip["source_group"]):
            raise ValueError("BFM clip path/hash/id/FPS/frames/split is invalid")
        group = clip["source_group"]
        if group in groups and groups[group] != clip["split"]:
            raise ValueError("BFM source group leaked across dataset splits")
        ids.add(clip["id"])
        files.add(name)
        groups[group] = clip["split"]
    if not any(clip["split"] == "train" for clip in clips):
        raise ValueError("BFM motion has no training clips")
    corrected = manifest["converter_revision"] not in LEGACY_CONVERTERS
    qualified = (corrected and manifest["converter_revision"] != G1_CONVERTER_REVISION and manifest.get("qualification") == "qualified"
                 and manifest.get("dynamic_replay_verified") is True
                 and manifest.get("source_rights") == "verified")
    candidate = (manifest["converter_revision"] in {REFERENCE_POOL_CONVERTER_REVISION,G1_CONVERTER_REVISION}
                 and manifest.get("qualification") == "reference_candidate")
    if manifest["converter_revision"] == G1_CONVERTER_REVISION:
        if (manifest.get("source_use") != "local_noncommercial_no_redistribution"
                or manifest.get("calibration_verified") is not True
                or manifest.get("calibration_applied_to_solver") is not True
                or not re.fullmatch(r"[0-9a-f]{64}",str(manifest.get("calibration_sha256","")))):
            raise ValueError("G1 BFM view requires hash-bound calibrated local research use")
    if manifest["converter_revision"] in {MIXED_POOL_CONVERTER_REVISION,MIXED_POOL_V1}:
        if (manifest.get("qualification") != "local_research_reference"
                or manifest.get("source_use") != "local_noncommercial_no_redistribution"
                or manifest.get("calibration_verified") is not True
                or manifest.get("calibration_applied_to_solver") is not True
                or any(not re.fullmatch(r"[0-9a-f]{64}",str(manifest.get(key,"")))
                       for key in ("source_pool_sha256","selection_sha256","official_manifest_sha256"))):
            raise ValueError("Mixed BFM pool requires frozen calibrated local research sources and splits")
        usable={split:set() for split in ("train","validation","test")}
        train_frames=0
        for clip in clips:
            quality=clip.get("reference_quality") or {}
            if (not isinstance(clip.get("valid_frames"),int) or not 0<=clip["valid_frames"]<=clip["frames"]
                    or not 0<=quality.get("max_body_position_error_m",1)<=.002):
                raise ValueError("Mixed BFM pool lacks independently verified reference quality")
            if quality.get("continuous_9frame_starts",0)>0 and clip["id"].startswith("g1_"):
                usable[clip["split"]].add(clip["source_group"])
                if clip["split"]=="train":train_frames+=clip["valid_frames"]
        if len(usable["train"])<5 or not usable["validation"] or not usable["test"] or train_frames<2000:
            raise ValueError("Mixed BFM pool lacks independent usable train/validation/test coverage")
        qualified=candidate=True
    return {"scope": "rpo_bfm_motion_v1", "status": "validated", "smoke_ready": corrected,
            "source_quaternion_verified": corrected,
            "prior_candidate_ready": candidate, "controller_evaluation_pass": False,
            "training_ready": qualified, "asset_sha256": asset_sha256, "codec_sha256": CODEC_HASH}


def learner_updates(rounds: int, updates_per_round: int = 1) -> int:
    if type(rounds) is not int or rounds < 1 or type(updates_per_round) is not int or updates_per_round not in {1, 8}:
        raise ValueError("BFM needs positive runner rounds and 1 or 8 updates per round")
    return rounds * updates_per_round


def validate_training_budget(updates:int,validation:dict, *, start_update:int=0)->str:
    if (type(updates) is not int or not 1<=updates<=100000
            or type(start_update) is not int or not 1<=updates-start_update<=1000):
        raise ValueError("BFM requires 1..1000 new learner updates per segment, cumulative target <=100000")
    if validation.get("smoke_ready") is not True:
        raise ValueError("BFM requires a validated motion view")
    if updates>2 and validation.get("training_ready") is not True:
        raise ValueError("BFM runs beyond two-update smoke require a training-ready frozen motion pool")
    return "local_research" if updates>2 else "smoke"


def eval_suite_spec()->dict:
    try:
        from eb.rpo_flat import flat_suite_spec
    except ModuleNotFoundError:
        from rpo_flat import flat_suite_spec
    return {**flat_suite_spec(),"benchmark":"rpo_bfm","policy":"deterministic_command_latent_onnx",
        "codec_sha256":CODEC_HASH,"actor_observation_dim":375,"latent_dim":256,
        "action_gain":list(ACTION_GAIN),"randomization":"native_RPOBFMEnvCfg",
        "sensor_noise_bounds":{"joint_pos_rad":.03,"joint_vel_radps":1.75,"gravity":.05,"angular_vel_div4":.05},
        "actuator_delay_steps":0,"sensor_noise_history":"same_measurement_cached_per_episode_step",
        "smoke_commands":[[0.,0.,0.],[.3,0.,0.],[0.,.2,0.],[.2,0.,.5]]}


def frozen_eval_suite_spec(suite_id,profile):
    if suite_id in {UFO_VALIDATION_SUITE, UFO_ACCEPTANCE_SUITE, UFO_REPAIR_ACCEPTANCE_SUITE}:
        eval_case_seeds(suite_id, profile)
        return {**batch_suite_spec(), "evaluation_role": profile, "required_video": True,
                "planned_cases": 16 if profile == "validation" else 100,
                "minimum_successes": None if profile == "validation" else 80,
                "parallel_protocol": "independent_cases_fixed_terrain_v2", "terrain_seed": 64000,
                "terrain_randomization": "fixed_official_grid_case_slot_seed_mod_200",
                "equivalence_tolerance": {"velocity_rmse": .005, "yaw_rmse": .005, "actual_steps": 1}}
    if suite_id in {VALIDATION_SUITE, ACCEPTANCE_SUITE}:
        eval_case_seeds(suite_id, profile)
        return {**batch_suite_spec(), "evaluation_role":profile, "required_video":True,
                "planned_cases":16 if profile == "validation" else 100,
                "minimum_successes":None if profile == "validation" else 80}
    if suite_id==EVAL_BATCH_SUITE:return batch_suite_spec()
    value=eval_suite_spec()
    if suite_id==EVAL_SUITE_V1 and profile=="smoke":
        for key in ("sensor_noise_bounds","actuator_delay_steps","sensor_noise_history"):value.pop(key)
    elif suite_id!=EVAL_SUITE:
        raise ValueError("BFM standard evaluation requires the complete v2 suite")
    return value


def batch_suite_spec():
    return {**eval_suite_spec(), "parallel_protocol":"independent_cases_shared_terrain_v1",
        "terrain_randomization":"shared_official_grid_per_batch_seeded_by_first_case",
        "case_randomization":"independent_startup_reset_command_noise_interval_rng",
        "inactive_case":"freeze_initial_state_exclude_from_metrics_no_autoreset"}


def eval_case_seeds(suite_id: str, profile: str) -> list[int]:
    if suite_id == UFO_REPAIR_ACCEPTANCE_SUITE and profile == "acceptance":
        return list(range(66000, 66100))
    if suite_id in {VALIDATION_SUITE, UFO_VALIDATION_SUITE} and profile == "validation":
        return list(range(61000, 61016))
    if suite_id in {ACCEPTANCE_SUITE, UFO_ACCEPTANCE_SUITE} and profile == "acceptance":
        return list(range(62000, 62100))
    if suite_id in {EVAL_SUITE, EVAL_BATCH_SUITE, EVAL_SUITE_V1} and profile in {"smoke", "standard"}:
        return list(range(60000, 60004) if profile == "smoke" else range(50000, 50100))
    raise ValueError("BFM suite and evaluation role must match frozen case seeds")
