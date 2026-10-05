"""Export and independently evaluate the official RPO-Flat deterministic actor."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import traceback
from pathlib import Path

import numpy as np

OBS_DIM = 780
CRITIC_DIM = 1390
ACTION_DIM = 23
REQUIRED_STEPS = 1000
VELOCITY_RMSE_MAX = 0.20
YAW_RMSE_MAX = 0.30
CONTROL_PERIOD_S = 0.02
ROBOLAB_REVISION = "6b1c3d9988497c8961dcba77892de32edc1770e1"
STANDARD_SEEDS = tuple(range(50000, 50100))
SMOKE_SEEDS = tuple(range(60000, 60003))
VIDEO_SETTINGS = {"standard": (320, 240, 10), "hd": (1920, 1080, 25)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def case_success(row: dict) -> bool:
    velocity, yaw = row.get("velocity_rmse"), row.get("yaw_rmse")
    return bool(
        row.get("completed") is True
        and row.get("actual_steps", row.get("episode_len")) == REQUIRED_STEPS
        and row.get("episode_len") == REQUIRED_STEPS
        and row.get("metric_steps") == REQUIRED_STEPS
        and not any(row.get(key, True) for key in ("fall", "nan", "hard_limit"))
        and isinstance(velocity, (int, float))
        and math.isfinite(velocity)
        and isinstance(yaw, (int, float))
        and math.isfinite(yaw)
        and 0 <= velocity <= VELOCITY_RMSE_MAX
        and 0 <= yaw <= YAW_RMSE_MAX
    )


def tracking_metrics(commands, velocities) -> dict:
    """Linear RMSE is the XY vector error, not the per-axis mean error."""
    commands = np.asarray(commands, dtype=np.float64).reshape(-1, 3)
    velocities = np.asarray(velocities, dtype=np.float64).reshape(-1, 3)
    if commands.shape != velocities.shape:
        raise ValueError("command and velocity traces must have matching shapes")
    finite = np.isfinite(commands).all(axis=1) & np.isfinite(velocities).all(axis=1)
    errors = velocities[finite] - commands[finite]
    count = len(errors)
    return {
        "metric_steps": count,
        "metrics_available": count > 0,
        "velocity_rmse": float(np.sqrt(np.mean(np.sum(errors[:, :2] ** 2, axis=1)))) if count else None,
        "yaw_rmse": float(np.sqrt(np.mean(errors[:, 2] ** 2))) if count else None,
    }


def summarize_cases(rows: list[dict], planned: int) -> dict:
    if type(planned) is not int or planned < 1 or len(rows) > planned:
        raise ValueError("invalid planned case count")
    if len({row["seed"] for row in rows}) != len(rows):
        raise ValueError("duplicate evaluation seeds")
    if any(row.get("success") is not case_success(row) for row in rows):
        raise ValueError("case success flag disagrees with actual acceptance metrics")
    completed = sum(row.get("completed") is True for row in rows)
    successes = sum(case_success(row) for row in rows)
    measured = [row for row in rows if row.get("metric_steps", 0) > 0]
    for row in measured:
        if not all(
            isinstance(row.get(key), (int, float)) and math.isfinite(row[key]) and row[key] >= 0
            for key in ("velocity_rmse", "yaw_rmse")
        ):
            raise ValueError("non-finite or missing measured tracking metric")
    count = sum(row["metric_steps"] for row in measured)
    metrics = {
        "n_trials": planned,
        "successes": successes,
        "success_rate": successes / planned,
        "fall_count": sum(bool(row.get("fall")) for row in rows),
        "nan_count": sum(bool(row.get("nan")) for row in rows),
        "hard_limit_count": sum(bool(row.get("hard_limit")) for row in rows),
        "metric_steps": count,
    }
    for key in ("velocity_rmse", "yaw_rmse"):
        metrics[key] = (
            math.sqrt(sum(row[key] ** 2 * row["metric_steps"] for row in measured) / count) if count else None
        )
    return {
        "cases": {
            "planned": planned,
            "completed": completed,
            "valid": completed,
            "invalid": 0,
            "infra_failed": planned - completed,
        },
        "metrics": metrics,
    }


def _checkpoint_weights(checkpoint: Path) -> Path:
    if checkpoint.is_file():
        return checkpoint
    candidates = [
        checkpoint / relative
        for relative in (
            "full_weights.pt",
            "model_state_dict/full_weights.pt",
            "actor/model_state_dict/full_weights.pt",
        )
    ]
    matches = [path for path in candidates if path.is_file()]
    if len(matches) != 1:
        raise ValueError("checkpoint must identify one complete full_weights.pt")
    return matches[0]


def _load_actor(checkpoint: Path):
    import torch
    from rpo_flat_policy import RPOFlatPolicy

    weights = _checkpoint_weights(checkpoint)
    state = torch.load(weights, map_location="cpu", weights_only=True)
    for wrapper in ("model", "model_state_dict", "state_dict", "module"):
        if isinstance(state, dict) and isinstance(state.get(wrapper), dict):
            state = state[wrapper]
    if not isinstance(state, dict):
        raise ValueError("checkpoint does not contain a state dict")
    cleaned = {}
    for key, value in state.items():
        while key.startswith(("module.", "_fsdp_wrapped_module.")):
            key = key.removeprefix("module.").removeprefix("_fsdp_wrapped_module.")
        if key in cleaned:
            raise ValueError("checkpoint has duplicate actor state keys")
        cleaned[key] = value
    policy = RPOFlatPolicy(
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        num_action_chunks=1,
        add_value_head=True,
        add_q_head=False,
        critic_obs_dim=CRITIC_DIM,
    )
    policy.load_state_dict(cleaned, strict=True)
    policy.eval()
    for name, value in policy.state_dict().items():
        if not torch.isfinite(value).all():
            raise ValueError(f"checkpoint has non-finite state: {name}")
    for norm in (policy.actor_norm, policy.critic_norm):
        if (norm._std <= 0).any() or (norm._var < 0).any() or norm.count < 0:
            raise ValueError("checkpoint normalization has invalid variance/std/count")
    if (policy.actor_std <= 0).any():
        raise ValueError("checkpoint actor has invalid action standard deviation")
    actor = torch.nn.Sequential(policy.actor_norm, policy.backbone, policy.actor_mean).eval()
    return actor, policy.actor_norm, weights


def golden_error_metrics(expected: np.ndarray, actual: np.ndarray) -> dict:
    delta = np.abs(expected - actual)
    metrics = {
        "error_atol": 1e-4, "error_rtol": 1e-5,
        "max_abs_error": float(np.max(delta)),
        "max_error_ratio": float(np.max(delta / (1e-4 + 1e-5 * np.abs(expected)))),
        "max_clipped_action_error": float(np.max(np.abs(
            np.clip(expected, -100, 100) - np.clip(actual, -100, 100)))),
    }
    metrics["max_joint_target_error_rad"] = 0.25 * metrics["max_clipped_action_error"]
    if (not np.isfinite(expected).all() or not np.isfinite(actual).all()
            or metrics["max_error_ratio"] > 1 or metrics["max_joint_target_error_rad"] > 1e-4):
        raise ValueError("exported ONNX failed golden numerical check")
    return metrics


def export(checkpoint: Path, output: Path) -> dict:
    import onnxruntime as ort
    import torch
    from rpo_wbc import RPO_LAB_JOINT_ORDER

    actor, norm, weights = _load_actor(checkpoint)
    output.mkdir(parents=True, exist_ok=True)
    onnx_path = output / "policy.onnx"
    ramp = np.linspace(-100.0, 100.0, OBS_DIM, dtype=np.float32)
    varied = np.random.default_rng(42).uniform(-100, 100, (4, OBS_DIM)).astype(np.float32)
    samples = np.vstack(
        (
            np.zeros(OBS_DIM),
            np.ones(OBS_DIM),
            -np.ones(OBS_DIM),
            ramp,
            -ramp,
            np.full(OBS_DIM, 100.0),
            np.full(OBS_DIM, -100.0),
            varied,
        )
    ).astype(np.float32)
    inputs = torch.from_numpy(samples)
    torch.onnx.export(
        actor,
        inputs[:1],
        onnx_path,
        input_names=["observation"],
        output_names=["action"],
        dynamic_axes={"observation": {0: "batch"}, "action": {0: "batch"}},
        opset_version=17,
    )
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    with torch.inference_mode():
        expected = actor(inputs).numpy()
        expected_single = np.vstack([actor(sample[None]).numpy() for sample in inputs])
    actual = session.run(None, {"observation": samples})[0]
    actual_single = np.vstack([
        session.run(None, {"observation": sample[None]})[0] for sample in samples])
    if (
        session.get_inputs()[0].shape[-1] != OBS_DIM
        or session.get_outputs()[0].shape[-1] != ACTION_DIM
    ):
        raise ValueError("exported ONNX failed measured I/O check")
    batch_errors = golden_error_metrics(expected, actual)
    single_errors = golden_error_metrics(expected_single, actual_single)
    errors = {key: max(value, single_errors[key]) for key, value in batch_errors.items()}
    np.savez_compressed(
        output / "golden-traces.npz", observation=samples, torch_action=expected, onnx_action=actual,
        torch_action_single=expected_single, onnx_action_single=actual_single,
    )
    np.savez_compressed(
        output / "actor-normalization.npz",
        **{key: value.detach().cpu().numpy() for key, value in norm.state_dict().items()},
    )
    receipt = {
        "schema_version": "eb/rpo-flat-export/v3",
        "status": "verified",
        "checkpoint_sha256": sha256(weights),
        "onnx": {
            "path": "policy.onnx",
            "sha256": sha256(onnx_path),
            "obs": OBS_DIM,
            "act": ACTION_DIM,
            "opset": 17,
            **errors,
        },
        "codec": {
            "history_length": 10,
            "frame_dim": 78,
            "action_scale": 0.25,
            "joint_order": list(RPO_LAB_JOINT_ORDER),
            "normalization": "embedded",
            "normalization_epsilon": float(norm.eps),
        },
        "golden": {
            "path": "golden-traces.npz",
            "sha256": sha256(output / "golden-traces.npz"),
            "n_inputs": len(samples),
            "batch_sizes": [1, len(samples)],
            "input_range": [-100.0, 100.0],
        },
        "normalization": {
            "path": "actor-normalization.npz",
            "sha256": sha256(output / "actor-normalization.npz"),
        },
    }
    _write_json(output / "export-manifest.json", receipt)
    return receipt


def _numpy(value) -> np.ndarray:
    return value.detach().cpu().numpy().copy()


def _trace_array(value) -> list:
    array = np.asarray(value)
    return np.where(np.isfinite(array), array, None).tolist()


def _state(env) -> dict[str, np.ndarray]:
    data = env.robot.data
    return {
        name: _numpy(getattr(data, name)[0])
        for name in (
            "root_pos_w",
            "root_quat_w",
            "root_lin_vel_b",
            "root_ang_vel_b",
            "root_lin_vel_w",
            "root_ang_vel_w",
            "projected_gravity_b",
            "joint_pos",
            "joint_vel",
            "applied_torque",
        )
    }


def command_frame_velocity(state: dict) -> np.ndarray:
    """Match official rewards: yaw-aligned linear XY and world angular Z."""
    w, x, y, z = state["root_quat_w"]
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    vx, vy = state["root_lin_vel_w"][:2]
    return np.array(
        [
            np.cos(yaw) * vx + np.sin(yaw) * vy,
            -np.sin(yaw) * vx + np.cos(yaw) * vy,
            state["root_ang_vel_w"][2],
        ],
        dtype=np.float64,
    )


def _hard_limits(env) -> dict[str, np.ndarray]:
    data = env.robot.data
    limits = {
        "joint_pos": _numpy(data.joint_pos_limits[0]),
        "joint_vel": _numpy(data.joint_vel_limits[0]),
        "applied_torque": _numpy(data.joint_effort_limits[0]),
    }
    for actuator in env.robot.actuators.values():
        # Explicit actuators can have lower limits than PhysX's disabled internal clamp.
        indices = actuator.joint_indices
        if hasattr(indices, "detach"):
            indices = _numpy(indices)
        limits["joint_vel"][indices] = _numpy(actuator.velocity_limit[0])
        limits["applied_torque"][indices] = _numpy(actuator.effort_limit[0])
    return limits


def _violates_hard_limit(state: dict, limits: dict) -> bool:
    q = state["joint_pos"]
    tolerance = 1e-3  # PhysX solver tolerance at a hard stop, not a soft joint-limit margin.
    return bool(
        np.any(q < limits["joint_pos"][:, 0] - tolerance)
        or np.any(q > limits["joint_pos"][:, 1] + tolerance)
        or np.any(np.abs(state["joint_vel"]) > limits["joint_vel"] + tolerance)
        or np.any(np.abs(state["applied_torque"]) > limits["applied_torque"] + tolerance)
    )


def _initial_state(env, seed: int, limits: dict) -> dict:
    from rpo_wbc_artifact import actuator_delay_steps

    terrain = env.scene.terrain
    parameters = {
        "masses": env.robot.root_physx_view.get_masses(),
        "coms": env.robot.root_physx_view.get_coms(),
        "materials": env.robot.root_physx_view.get_material_properties(),
    }
    for name in ("joint_stiffness", "joint_damping", "joint_armature", "joint_friction_coeff"):
        parameters[name] = getattr(env.robot.data, name)
    return {
        "seed": seed,
        "terrain_seed": seed,
        "terrain_level": int(terrain.terrain_levels[0]),
        "terrain_column": int(terrain.terrain_types[0]),
        "env_origin": _trace_array(_numpy(env.scene.env_origins[0])),
        "startup_parameters": {name: _trace_array(_numpy(value[0])) for name, value in parameters.items()},
        "actuator_delay_steps": actuator_delay_steps(env.robot),
        "actuator_gains": {
            name: {
                "stiffness": _trace_array(_numpy(actuator.stiffness[0])),
                "damping": _trace_array(_numpy(actuator.damping[0])),
            }
            for name, actuator in env.robot.actuators.items()
        },
        "hard_limits": {name: _trace_array(value) for name, value in limits.items()},
        "state": {name: _trace_array(value) for name, value in _state(env).items()},
        "command": _trace_array(_numpy(env.command_generator.command[0])),
    }


def _add_camera(cfg, video_quality: str = "standard") -> None:
    from isaaclab import sim as sim_utils
    from isaaclab.sensors import CameraCfg

    width, height, fps = VIDEO_SETTINGS[video_quality]
    cfg.scene.eval_camera = CameraCfg(
        prim_path="/World/EvalCamera",
        update_period=1 / fps,
        height=height,
        width=width,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, clipping_range=(0.1, 100.0)),
        offset=CameraCfg.OffsetCfg(
            pos=(3.0, 3.0, 1.6), rot=(0.380595, 0.096385, 0.039924, -0.918838), convention="world"
        ),
    )
    cfg.num_rerenders_on_reset = 1


def _video_frame(env):
    import torch

    camera = env.scene["eval_camera"]
    position = env.robot.data.root_pos_w
    offset = torch.tensor([[3.0, 3.0, 1.5]], device=env.device)
    camera.set_world_poses_from_view(position + offset, position)
    env.sim.render()
    return _numpy(camera.data.output["rgb"][0, :, :, :3]).astype(np.uint8)


def _evaluate_case(session, env, seed: int, output: Path, record_video: bool,
                   video_quality: str = "standard") -> dict:
    import torch

    row = {
        "seed": seed,
        "case_id": f"rpo-flat-seed-{seed}",
        "task_id": "tsk_rpo_flat",
        "completed": False,
        "status": "infra_failure",
        "success": False,
        "actual_steps": 0,
        "episode_len": 0,
        "fall": False,
        "nan": False,
        "hard_limit": False,
        "reason": "infrastructure_failure",
    }
    commands, velocities = [], []
    writer = None
    trace_path = output / "traces" / f"seed_{seed}.jsonl.gz"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        obs, _ = env.reset(seed=seed)
        if tuple(obs["policy"].shape) != (1, OBS_DIM) or tuple(obs["critic"].shape) != (1, CRITIC_DIM):
            raise ValueError("official RPO-Flat environment observation contract mismatch")
        if abs(env.step_dt - CONTROL_PERIOD_S) > 1e-9:
            raise ValueError("official RPO-Flat environment policy period mismatch")
        from rpo_wbc import RPO_LAB_JOINT_ORDER

        if tuple(env.robot.joint_names) != RPO_LAB_JOINT_ORDER:
            raise ValueError("official RPO-Flat environment native joint order mismatch")
        limits = _hard_limits(env)
        initial_path = trace_path.with_name(f"seed_{seed}.initial.json")
        _write_json(initial_path, _initial_state(env, seed, limits))
        row["initial_state"] = {"path": str(initial_path.relative_to(output)), "sha256": sha256(initial_path)}
        if record_video:
            import imageio.v2 as imageio

            video_rel = f"seed_{seed}/0.mp4"
            video_path = output / "video/eval" / video_rel
            video_path.parent.mkdir(parents=True, exist_ok=True)
            _, _, fps = VIDEO_SETTINGS[video_quality]
            writer = imageio.get_writer(video_path, fps=fps, codec="libx264",
                macro_block_size=1,
                **({"quality":None, "ffmpeg_params":["-crf", "18", "-preset", "fast"]}
                   if video_quality == "hd" else {}))
            row["video_rel"] = video_rel
            writer.append_data(_video_frame(env))
        with gzip.open(trace_path, "wt", encoding="utf-8") as trace:
            for step in range(REQUIRED_STEPS):
                actor_obs = _numpy(obs["policy"]).astype(np.float32)
                command = _numpy(env.command_generator.command[0])
                heading = float(env.command_generator.heading_target[0])
                standing = bool(env.command_generator.is_standing_env[0])
                if (
                    not np.isfinite(actor_obs).all()
                    or not np.isfinite(_numpy(obs["critic"])).all()
                    or not np.isfinite(command).all()
                    or not math.isfinite(heading)
                ):
                    row.update(nan=True, reason="non_finite_policy_input")
                    trace.write(
                        json.dumps(
                            {
                                "step": step,
                                "advanced": False,
                                "nan": True,
                                "command_pre": _trace_array(command),
                            },
                            allow_nan=False,
                        )
                        + "\n"
                    )
                    break
                action = session.run(None, {session.get_inputs()[0].name: actor_obs})[0].astype(np.float32)
                if action.shape != (1, ACTION_DIM):
                    raise ValueError("actual ONNX action shape mismatch")
                if not np.isfinite(action).all():
                    row.update(nan=True, reason="non_finite_policy_output")
                    trace.write(
                        json.dumps(
                            {"step": step, "advanced": False, "nan": True, "action": _trace_array(action)},
                            allow_nan=False,
                        )
                        + "\n"
                    )
                    break
                if np.any(np.abs(action) > env.clip_actions):
                    row.update(hard_limit=True, reason="action_clip_limit")
                    trace.write(
                        json.dumps(
                            {
                                "step": step,
                                "advanced": False,
                                "hard_limit": True,
                                "command_pre": _trace_array(command),
                                "action": _trace_array(action),
                            },
                            allow_nan=False,
                        )
                        + "\n"
                    )
                    break
                obs, _, terminated, truncated, _ = env.step(torch.from_numpy(action).to(env.device))
                row["actual_steps"] += 1
                row["episode_len"] = row["actual_steps"]
                state = _state(env)
                finite = all(np.isfinite(value).all() for value in state.values())
                row["nan"] |= (
                    not finite
                    or not np.isfinite(_numpy(obs["policy"])).all()
                    or not np.isfinite(_numpy(obs["critic"])).all()
                )
                row["hard_limit"] |= _violates_hard_limit(state, limits)
                # Compare post-physics state with the command active before resampling in this step.
                metric_velocity = command_frame_velocity(state)
                velocities.append(metric_velocity)
                commands.append(command)
                relative_height = state["root_pos_w"][2] - float(env.scene.env_origins[0, 2])
                row["fall"] = bool(
                    row["fall"]
                    or bool(terminated.any())
                    or relative_height < 0.50
                    or -state["projected_gravity_b"][2] < math.cos(0.80)
                )
                done = bool((terminated | truncated).any())
                entry = {
                    "step": step + 1,
                    "time_s": (step + 1) * CONTROL_PERIOD_S,
                    "advanced": True,
                    "command_pre": _trace_array(command),
                    "command_post": _trace_array(_numpy(env.command_generator.command[0])),
                    "metric_velocity": _trace_array(metric_velocity),
                    "heading_target_pre": heading,
                    "standing_pre": standing,
                    "action": _trace_array(action[0]),
                    "terminated": bool(terminated.any()),
                    "truncated": bool(truncated.any()),
                    "nan": row["nan"],
                    "fall": row["fall"],
                    "hard_limit": row["hard_limit"],
                    **{name: _trace_array(value) for name, value in state.items()},
                }
                trace.write(json.dumps(entry, allow_nan=False) + "\n")
                if writer is not None and ((step + 1) % round(1 / (CONTROL_PERIOD_S * fps)) == 0
                        or done or row["fall"] or row["nan"] or row["hard_limit"]):
                    writer.append_data(_video_frame(env))
                if row["nan"] or row["fall"] or row["hard_limit"] or done:
                    row["reason"] = (
                        "non_finite_state"
                        if row["nan"]
                        else "fall"
                        if row["fall"]
                        else "physical_hard_limit"
                        if row["hard_limit"]
                        else "horizon"
                        if step + 1 == REQUIRED_STEPS
                        else "early_timeout"
                    )
                    break
            else:
                row["reason"] = "horizon"
        row["completed"] = True
    except Exception as exc:
        traceback.print_exc()
        row["error"] = f"{type(exc).__name__}: {exc}"[:2000]
    finally:
        if writer is not None:
            writer.close()
    row.update(tracking_metrics(commands, velocities))
    row["success"] = case_success(row)
    if row["completed"]:
        row["status"] = "success" if row["success"] else "task_failure"
        if row["reason"] == "horizon" and not row["success"]:
            row["reason"] = "velocity_tracking_error"
    row["termination_reason"] = row["reason"]
    if trace_path.is_file():
        row["trace"] = {
            "path": str(trace_path.relative_to(output)),
            "sha256": sha256(trace_path),
            "policy_steps": row["actual_steps"],
            "metric_steps": row["metric_steps"],
        }
    return row


def evaluate_isaac(
    policy: Path, output: Path, seeds: list[int], *, record_video: bool = False,
    video_quality: str = "standard", plan_hash: str | None = None
) -> dict:
    if video_quality not in VIDEO_SETTINGS or video_quality == "hd" and not record_video:
        raise ValueError("HD video requires recording with a supported quality")
    if not seeds or len(set(seeds)) != len(seeds) or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("evaluation requires distinct nonnegative integer seeds")
    import onnxruntime as ort
    from isaaclab.app import AppLauncher

    exported = json.loads((policy.parent / "export-manifest.json").read_text())
    if exported.get("status") != "verified" or exported["onnx"]["sha256"] != sha256(policy):
        raise ValueError("verified RPO-Flat export receipt missing or policy hash changed")
    session = ort.InferenceSession(str(policy), providers=["CPUExecutionProvider"])
    if session.get_inputs()[0].shape[-1] != OBS_DIM or session.get_outputs()[0].shape[-1] != ACTION_DIM:
        raise ValueError("evaluation requires the measured official 780x23 ONNX contract")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "eval-cases.jsonl").exists():
        raise ValueError("evaluation case output already exists; use a new output directory")
    app = AppLauncher(headless=True, enable_cameras=record_video).app
    rows = []
    environment_settings = None
    try:
        import gymnasium as gym
        import omni.usd
        from isaaclab.sim import SimulationContext
        from isaaclab.sensors import RayCaster
        from rpo_flat_extension import TASK_ID, RPOFlatEnvCfg, register_task

        register_task()
        for seed in seeds:
            env = None
            try:
                # ponytail: recreate per seed for independent startup events; batch needs per-env RNG.
                # Delayed sensor destruction leaves class-global Warp meshes in cfg.class_type.
                # Clear before validation and stage replacement to prevent cycles and stale terrain.
                RayCaster.meshes.clear()
                omni.usd.get_context().new_stage()
                cfg = RPOFlatEnvCfg()
                cfg.seed = seed
                cfg.scene.num_envs = 1
                cfg.scene.terrain.terrain_generator.seed = seed
                environment_settings = {
                    "commands": cfg.commands.to_dict(),
                    "noise": cfg.noise.to_dict(),
                    "events": cfg.events.to_dict(),
                    "terrain": cfg.scene.terrain.terrain_generator.to_dict(),
                    "episode_length_s": cfg.episode_length_s,
                    "control_period_s": cfg.decimation * cfg.sim.dt,
                    "startup_randomization": "fresh_environment_per_seed",
                    "hard_limit_tolerance": 1e-3,
                    "fall_base_height_relative_m": 0.50,
                    "fall_max_tilt_rad": 0.80,
                }
                # Preserve official settings; to_dict also includes class/function objects.
                environment_settings = json.loads(
                    json.dumps(
                        environment_settings,
                        default=lambda value: (
                            f"{value.__module__}.{value.__qualname__}" if callable(value) else str(value)
                        ),
                    )
                )
                if record_video:
                    _add_camera(cfg, video_quality)
                env = gym.make(TASK_ID, cfg=cfg, render_mode=None).unwrapped
                row = _evaluate_case(session, env, seed, output, record_video, video_quality)
            except Exception as exc:
                traceback.print_exc()
                row = {
                    "seed": seed,
                    "case_id": f"rpo-flat-seed-{seed}",
                    "task_id": "tsk_rpo_flat",
                    "completed": False,
                    "status": "infra_failure",
                    "success": False,
                    "actual_steps": 0,
                    "episode_len": 0,
                    "metric_steps": 0,
                    "metrics_available": False,
                    "velocity_rmse": None,
                    "yaw_rmse": None,
                    "fall": False,
                    "nan": False,
                    "hard_limit": False,
                    "reason": "environment_initialization_failure",
                    "error": f"{type(exc).__name__}: {exc}"[:2000],
                }
            finally:
                if env is not None:
                    env.close()
                elif SimulationContext.instance() is not None:
                    SimulationContext.instance().clear_all_callbacks()
                    SimulationContext.instance().clear_instance()
            rows.append(row)
            row["run_id"] = os.environ.get("EB_RUN_ID")
            row["plan_hash"] = plan_hash
            # Preserve completed cases if a later simulator case or the Job itself crashes.
            with (output / "eval-cases.jsonl").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            print(
                json.dumps(
                    {
                        "case": row["case_id"],
                        "status": row["status"],
                        "episode_len": row["episode_len"],
                        "success": row["success"],
                    }
                ),
                flush=True,
            )
    except BaseException:
        app.close()
        raise
    code_paths = [Path(__file__)]
    runner_root = Path(__file__).parent
    code_paths += [runner_root / name for name in ("rpo_flat_policy.py", "rpo_flat_extension.py")]
    asset_root = Path(os.environ.get("RPO_ASSET_ROOT", "/workspace/rpo-assets"))
    code_paths += [
        asset_root / "robolab/robolab/tasks/direct/base" / name
        for name in ("base_env.py", "base_config.py", "rpo_env_cfg.py", "terrain_generator_cfg.py")
    ]
    receipt = {
        "schema_version": "eb/rpo-flat-eval/v1",
        "backend": "isaac",
        "criterion": "velocity_commands",
        "required_steps": REQUIRED_STEPS,
        "plan_hash": plan_hash,
        "run_id": os.environ.get("EB_RUN_ID"),
        "record_video": record_video,
        "video_quality": video_quality,
        "video_settings": dict(zip(("width", "height", "fps"), VIDEO_SETTINGS[video_quality])) if record_video else None,
        "seeds": seeds,
        "official_revision": ROBOLAB_REVISION,
        "policy": "deterministic_onnx",
        "velocity_frame": "yaw_aligned",
        "yaw_rate_frame": "world_z",
        "onnx": exported["onnx"],
        "code_trace": [{"path": str(path), "sha256": sha256(path)} for path in code_paths if path.is_file()],
        "environment_settings": environment_settings,
        **summarize_cases(rows, len(seeds)),
    }
    _write_json(output / "eval-manifest.json", receipt)
    # Isaac shutdown can terminate the process. Persist the receipt first.
    app.close()
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    for name in ("export", "eval-checkpoint-isaac"):
        command = sub.add_parser(name)
        command.add_argument("--checkpoint", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        if name == "eval-checkpoint-isaac":
            command.add_argument("--seeds", required=True)
            command.add_argument("--record-video", action="store_true")
            command.add_argument("--video-quality", choices=VIDEO_SETTINGS, default="standard")
            command.add_argument("--plan-hash")
    args = parser.parse_args()
    if args.mode == "export":
        export(args.checkpoint, args.output)
    else:
        export(args.checkpoint, args.output / "exported")
        receipt = evaluate_isaac(
            args.output / "exported/policy.onnx",
            args.output,
            [int(seed) for seed in args.seeds.split(",")],
            record_video=args.record_video,
            video_quality=args.video_quality,
            plan_hash=args.plan_hash,
        )
        if receipt["cases"]["infra_failed"]:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
