"""Export the deterministic RPO actor and run the actual ONNX in MuJoCo."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


OBS_DIM = 890
ACTION_DIM = 23
FRAME_DIM = 89
EVAL_POLICY_STEPS = 1000
FALL_BASE_HEIGHT_M = 0.50
FALL_UPRIGHT_COS_MIN = float(np.cos(0.80))
CONTROL_DECIMATION = 4
CONTROL_PERIOD_S = 0.02
POLICY_TO_URDF = np.arange(ACTION_DIM)  # Managed RPO MJCF joint order matches canonical policy order.
DEFAULT_URDF = np.array([0, 0, -0.1, 0.3, -0.2, 0, 0, 0, -0.1, 0.3, -0.2, 0, 0, 0.18, 0.06, 0, 0.78, 0, 0.18, -0.06, 0, 0.78, 0], dtype=np.float32)
KP = np.array([100, 100, 100, 150, 40, 40, 100, 100, 100, 150, 40, 40, 150, 40, 40, 40, 30, 20, 40, 40, 40, 30, 20], dtype=np.float32)
KD = np.array([3.3, 3.3, 3.3, 5, 2, 2, 3.3, 3.3, 3.3, 5, 2, 2, 5, 2, 2, 2, 1.5, 1, 2, 2, 2, 1.5, 1], dtype=np.float32)
EFFORT_LIMIT = np.array([120, 120, 120, 120, 27, 27,
                         120, 120, 120, 120, 27, 27,
                         120, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27], dtype=np.float32)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _weights(path: Path) -> dict:
    import torch

    value = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("model", "state_dict", "module"):
        if isinstance(value, dict) and isinstance(value.get(key), dict):
            value = value[key]
    if not isinstance(value, dict):
        raise ValueError("checkpoint does not contain a state dict")
    cleaned = {}
    for key, tensor in value.items():
        key = key.removeprefix("module.").removeprefix("_fsdp_wrapped_module.")
        if key.startswith(("backbone.", "actor_mean.")):
            cleaned[key] = tensor
    return cleaned


def export(checkpoint: Path, output: Path, *, amp: bool = False) -> None:
    import torch
    from torch import nn
    obs_dim = 780 if amp else OBS_DIM
    frame_dim = 78 if amp else FRAME_DIM

    class Actor(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Sequential(nn.Linear(obs_dim, 256), nn.Tanh(), nn.Linear(256, 256), nn.Tanh(), nn.Linear(256, 256), nn.Tanh())
            self.actor_mean = nn.Linear(256, ACTION_DIM)
            self.register_buffer("obs_mean", torch.zeros(obs_dim))
            self.register_buffer("obs_var", torch.ones(obs_dim))

        def forward(self, observation):
            normalized = (observation - self.obs_mean) * torch.rsqrt(self.obs_var + 1e-5)
            return self.actor_mean(self.backbone(normalized))

    actor = Actor().eval()
    missing, unexpected = actor.load_state_dict(_weights(checkpoint), strict=False)
    required_missing = [key for key in missing if not key.startswith(("obs_mean", "obs_var"))]
    if required_missing or unexpected:
        raise ValueError(f"actor checkpoint mismatch: missing={required_missing}, unexpected={unexpected}")
    output.mkdir(parents=True, exist_ok=True)
    onnx_path = output / "policy.onnx"
    sample = torch.linspace(-1, 1, obs_dim).reshape(1, obs_dim)
    torch.onnx.export(actor, sample, onnx_path, input_names=["observation"], output_names=["action"],
                      dynamic_axes={"observation": {0: "batch"}, "action": {0: "batch"}}, opset_version=17)
    import onnxruntime as ort
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    expected = actor(sample).detach().numpy()
    actual = session.run(None, {"observation": sample.numpy()})[0]
    error = float(np.max(np.abs(expected - actual)))
    if (session.get_inputs()[0].shape[-1] != obs_dim
            or session.get_outputs()[0].shape[-1] != ACTION_DIM
            or not np.isfinite(expected).all() or not np.isfinite(actual).all()
            or not np.isfinite(error) or error > 1e-4):
        raise ValueError("exported ONNX failed measured I/O or golden numerical check")
    receipt = {"schema_version":"eb/rpo-amp-export/v1" if amp else "eb/rpo-wbc-export/v1", "status":"verified",
               "onnx":{"path":"policy.onnx", "sha256":sha256(onnx_path), "obs":obs_dim,
                       "act":ACTION_DIM, "opset":17, "max_abs_error":error},
               "codec":{"history_length":10, "frame_dim":frame_dim, "action_scale":0.25,
                        "policy_to_urdf":POLICY_TO_URDF.tolist(), "normalization":"embedded"}}
    (output / "export-manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")


def _gravity_from_wxyz(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat
    return np.array([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], dtype=np.float32) * -1


def case_success(criterion: str, steps: int, fell: bool, nan: bool, hard_limit: bool,
                 upper_rmse: float, velocity_rmse: float, *, scenario: str = "",
                 distance_m: float = 0.0, settings: dict | None = None) -> bool:
    settings = settings or {}
    if criterion == "stability":
        return steps >= EVAL_POLICY_STEPS and not (fell or nan)
    if criterion == "stand_walk":
        return (steps >= EVAL_POLICY_STEPS and not (fell or nan or hard_limit)
                and ((scenario == "stand" and distance_m <= settings.get("stand_max_drift_m", 0.5) and velocity_rmse <= 0.15)
                     or (scenario == "walk" and distance_m >= settings.get("walk_min_forward_m", 4.0) and velocity_rmse <= 0.20)))
    if criterion == "tracking":
        return not (fell or nan or hard_limit) and upper_rmse <= 0.1 and velocity_rmse <= 0.15
    raise ValueError("unknown RPO evaluation criterion")


def physical_fall(base_height_m: float, projected_gravity_z: float) -> str | None:
    if base_height_m < FALL_BASE_HEIGHT_M:
        return "low_base"
    if -projected_gravity_z < FALL_UPRIGHT_COS_MIN:
        return "excessive_tilt"
    return None


def pd_torque(q: np.ndarray, qd: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.clip((target - q) * KP - qd * KD, -EFFORT_LIMIT, EFFORT_LIMIT)


def control_decimation(physics_dt: float) -> int:
    value = round(CONTROL_PERIOD_S / physics_dt)
    if value < 1 or abs(value * physics_dt - CONTROL_PERIOD_S) > 1e-8:
        raise ValueError("MuJoCo physics timestep must divide the 20 ms policy period")
    return value


def probe_mujoco_stand(mjcf: Path, physics_dt: float) -> dict:
    """Check whether the source pose itself can stand under the managed MuJoCo PD."""
    import mujoco
    from rpo_wbc import AMP_STAND_HEIGHT_M, AMP_STAND_Q

    model = mujoco.MjModel.from_xml_path(str(mjcf))
    model.opt.timestep = physics_dt
    decimation = control_decimation(physics_dt)
    data = mujoco.MjData(model)
    data.qpos[2] = AMP_STAND_HEIGHT_M
    data.qpos[-ACTION_DIM:] = AMP_STAND_Q
    mujoco.mj_forward(model, data)
    initial_contacts = data.ncon
    initial_contact_dims = sorted({int(contact.dim) for contact in data.contact})
    origin = data.qpos[:2].copy()
    for physics_step in range(1, EVAL_POLICY_STEPS * decimation + 1):
        data.ctrl[:] = pd_torque(data.qpos[-ACTION_DIM:], data.qvel[-ACTION_DIM:], AMP_STAND_Q)
        mujoco.mj_step(model, data)
        if not np.isfinite(data.qpos).all():
            reason = "nan"
            break
        reason = physical_fall(float(data.qpos[2]),
                               float(_gravity_from_wxyz(np.asarray(data.sensor("orientation").data))[2]))
        if reason:
            break
    return {"physics_timestep_s":physics_dt,
            "initial_contact_count":initial_contacts,
            "initial_contact_dims":initial_contact_dims,
            "policy_steps":(physics_step + decimation - 1) // decimation,
            "fall_reason":reason, "base_height_m":float(data.qpos[2]),
            "drift_m":float(np.linalg.norm(data.qpos[:2] - origin))}


def actuator_delay_steps(robot) -> dict[str, int]:
    return {name:int(actuator.positions_delay_buffer.time_lags[0])
            for name, actuator in robot.actuators.items()
            if hasattr(actuator, "positions_delay_buffer")}


def evaluate_mujoco(policy: Path, mjcf: Path, output: Path, seeds: list[int], *,
                    record_video: bool = False, criterion: str = "tracking", settings: dict | None = None,
                    amp: bool = False) -> None:
    import mujoco
    import onnxruntime as ort
    from rpo_wbc import AMP_STAND_HEIGHT_M, AMP_STAND_POSE_ID, AMP_STAND_Q

    initial_pose = (settings or {}).get("initial_pose")
    if initial_pose not in (None, AMP_STAND_POSE_ID) or initial_pose and not amp:
        raise ValueError("unsupported RPO AMP evaluation initial pose")
    if (settings or {}).get("actuator_delay_steps") is not None:
        raise ValueError("MuJoCo evaluation does not use Isaac actuator delay")

    session = ort.InferenceSession(str(policy), providers=["CPUExecutionProvider"])
    obs_dim, frame_dim = (780, 78) if amp else (OBS_DIM, FRAME_DIM)
    if session.get_inputs()[0].shape[-1] != obs_dim or session.get_outputs()[0].shape[-1] != ACTION_DIM:
        raise ValueError(f"evaluation requires the measured {obs_dim}x23 ONNX contract")
    export_receipt = policy.parent / "export-manifest.json"
    if amp and not export_receipt.is_file():
        raise ValueError("AMP evaluation requires the actual verified ONNX export receipt")
    max_abs_error = json.loads(export_receipt.read_text())["onnx"]["max_abs_error"] if export_receipt.is_file() else 0.0
    output.mkdir(parents=True, exist_ok=True)
    stand_pose_hold = ([probe_mujoco_stand(mjcf, timestep) for timestep in (0.005, 0.001)]
                       if amp and initial_pose else None)
    rows = []
    commands = ((0.0, 0.0, 0.0), (0.35, 0.0, 0.0), (0.0, 0.0, 0.35))
    for seed in seeds:
        rng = np.random.default_rng(seed)
        model = mujoco.MjModel.from_xml_path(str(mjcf))
        model.opt.timestep = 0.005
        data = mujoco.MjData(model)
        renderer = writer = None
        video_rel = f"seed_{seed}/0.mp4"
        if record_video:
            import imageio.v2 as imageio
            video_path = output / "video/eval" / video_rel
            video_path.parent.mkdir(parents=True, exist_ok=True)
            renderer = mujoco.Renderer(model, height=240, width=320)
            writer = imageio.get_writer(video_path, fps=10, codec="libx264")
        data.qpos[-ACTION_DIM:] = AMP_STAND_Q if initial_pose else DEFAULT_URDF
        if initial_pose:
            data.qpos[2] = AMP_STAND_HEIGHT_M
        initial_yaw = 0.0
        if criterion == "stand_walk":
            data.qpos[:2] = rng.uniform(-0.5, 0.5, 2)
            initial_yaw = float(rng.uniform(-np.pi, np.pi))
            data.qpos[3:7] = (np.cos(initial_yaw / 2), 0.0, 0.0, np.sin(initial_yaw / 2))
            data.qpos[-ACTION_DIM:] += rng.uniform(-0.05, 0.05, ACTION_DIM)
            data.qvel[:2] = rng.uniform(-0.05, 0.05, 2)
            data.qvel[5] = rng.uniform(-0.05, 0.05)
        mujoco.mj_forward(model, data)
        start_xy = data.qpos[:2].copy()
        history = np.zeros((10, frame_dim), dtype=np.float32)
        action = np.zeros(ACTION_DIM, dtype=np.float32)
        scenario = ("stand" if seed % 2 == 0 else "walk") if criterion == "stand_walk" else ""
        reference = (np.zeros(11, dtype=np.float32) if scenario else
                     rng.uniform(-0.2, 0.2, 11).astype(np.float32))
        reference[0] *= 0.5
        command = np.array((0.0, 0.0, 0.0) if scenario == "stand" else
                           (0.3, 0.0, 0.0) if scenario == "walk" else
                           commands[seed % len(commands)], dtype=np.float32)
        upper_errors, speed_errors = [], []
        fell = nan = hard_limit = False
        fall_reason = None
        for step in range(EVAL_POLICY_STEPS * CONTROL_DECIMATION):
            q_urdf = data.qpos[-ACTION_DIM:].copy()
            qd_urdf = data.qvel[-ACTION_DIM:].copy()
            if step % CONTROL_DECIMATION == 0:
                q = (q_urdf - DEFAULT_URDF)[POLICY_TO_URDF]
                qd = qd_urdf[POLICY_TO_URDF]
                omega = np.asarray(data.sensor("angular-velocity").data, dtype=np.float32)
                gravity = _gravity_from_wxyz(np.asarray(data.sensor("orientation").data, dtype=np.float32))
                frame = np.concatenate((omega, gravity, command, q, qd, action)
                                       if amp else (omega, gravity, command, q, qd, action, reference))
                history[:-1], history[-1] = history[1:], frame
                action = session.run(None, {session.get_inputs()[0].name: history.reshape(1, -1)})[0][0].astype(np.float32)
                if not np.isfinite(action).all():
                    nan = True
                    break
                hard_limit |= bool(np.any(np.abs(action) > 100.0))
                action = np.clip(action, -100.0, 100.0)
                target = DEFAULT_URDF.copy()
                target[POLICY_TO_URDF] += 0.25 * action
                if not amp:
                    upper_errors.append(np.square((q_urdf - DEFAULT_URDF)[12:] - reference).mean())
                speed_errors.append(np.square(np.asarray(data.sensor("linear-velocity").data[:2]) - command[:2]).mean())
            raw_torque = (target - q_urdf) * KP - qd_urdf * KD
            hard_limit |= bool(np.any(np.abs(raw_torque) > 200.0))
            torque = pd_torque(q_urdf, qd_urdf, target)
            data.ctrl[:] = torque
            mujoco.mj_step(model, data)
            if writer is not None and step % 20 == 0:
                renderer.update_scene(data)
                writer.append_data(renderer.render())
            nan |= not np.isfinite(data.qpos).all()
            fall_reason = physical_fall(float(data.qpos[2]),
                                        float(_gravity_from_wxyz(np.asarray(data.sensor("orientation").data))[2]))
            fell |= fall_reason is not None
            if nan or fell:
                if writer is not None and step % 20:
                    renderer.update_scene(data)
                    writer.append_data(renderer.render())
                break
        if writer is not None:
            writer.close()
            renderer.close()
        upper_rmse = None if amp else float(np.sqrt(np.mean(upper_errors))) if upper_errors else 1e9
        velocity_rmse = float(np.sqrt(np.mean(speed_errors))) if speed_errors else 1e9
        episode_len = step // CONTROL_DECIMATION + 1
        delta_xy = data.qpos[:2] - start_xy
        distance_m = float(np.linalg.norm(delta_xy) if scenario == "stand" else
                           delta_xy[0] * np.cos(initial_yaw) + delta_xy[1] * np.sin(initial_yaw))
        success = case_success(criterion, episode_len, fell, nan, hard_limit,
                               upper_rmse, velocity_rmse, scenario=scenario, distance_m=distance_m, settings=settings)
        termination_reason = ("nan" if nan else "fall" if fell else "horizon" if success else
                              "hard_limit" if hard_limit and criterion == "tracking" else
                              "tracking_error" if criterion == "tracking"
                              else "goal_not_met" if criterion == "stand_walk" else "early_termination")
        rows.append({"seed":seed, "case_id":f"rpo-seed-{seed}",
                     "task_id":"tsk_rpo_amp_stand_walk" if amp else "tsk_rpo_stand_walk" if scenario else "tsk_rpo_walk_upperbody_tracking",
                     "init_state_id":f"seed_{seed}", "status":"success" if success else "task_failure",
                     "success":success, "episode_len":episode_len, "scenario":scenario,
                     "distance_m":distance_m, "fall_reason":fall_reason,
                     "sim_time_s":round((step + 1) * model.opt.timestep, 3),
                     "termination_reason":termination_reason, "completed":True,
                     "fall":fell, "nan":nan, "hard_limit":hard_limit,
                     "upper_rmse":upper_rmse, "velocity_rmse":velocity_rmse,
                     **({"video_rel":video_rel} if record_video else {})})
    (output / "eval-cases.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    receipt = {"schema_version":"eb/rpo-amp-eval/v1" if amp else "eb/rpo-wbc-eval/v1", "backend":"mujoco",
        "criterion":criterion, "required_steps":EVAL_POLICY_STEPS,
        "record_video":record_video, "environment_settings":settings or {},
        **({"source_pose_pd_probe":stand_pose_hold} if stand_pose_hold else {}),
        "onnx":{"sha256":sha256(policy), "obs":obs_dim, "act":ACTION_DIM,
                "max_abs_error":max_abs_error},
        "cases":{"planned":len(seeds), "completed":len(rows), "valid":len(rows), "invalid":0, "infra_failed":0},
        "metrics":{"n_trials":len(rows), "successes":sum(r["success"] for r in rows),
                   "success_rate":sum(r["success"] for r in rows) / len(rows),
                   "fall_count":sum(r["fall"] for r in rows), "hard_limit_count":sum(r["hard_limit"] for r in rows),
                   "nan_count":sum(r["nan"] for r in rows), "upper_rmse":None if amp else float(np.mean([r["upper_rmse"] for r in rows])),
                   "velocity_rmse":float(np.mean([r["velocity_rmse"] for r in rows]))}}
    (output / "eval-manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")


def evaluate_isaac(policy: Path, output: Path, seeds: list[int], *,
                   record_video: bool = False, criterion: str = "tracking", settings: dict | None = None,
                   amp: bool = False) -> None:
    import os
    from rpo_wbc import AMP_STAND_POSE_ID

    initial_pose = (settings or {}).get("initial_pose")
    if initial_pose not in (None, AMP_STAND_POSE_ID) or initial_pose and not amp:
        raise ValueError("unsupported RPO AMP evaluation initial pose")
    if initial_pose:
        os.environ["EB_RPO_AMP_INITIAL_POSE"] = initial_pose
    else:
        os.environ.pop("EB_RPO_AMP_INITIAL_POSE", None)
    fixed_delay = (settings or {}).get("actuator_delay_steps")
    if fixed_delay is not None:
        if type(fixed_delay) is not int or not 0 <= fixed_delay <= 2 or not amp:
            raise ValueError("unsupported Isaac AMP actuator delay")
        os.environ["EB_RPO_AMP_ACTUATOR_DELAY_STEPS"] = str(fixed_delay)
    else:
        os.environ.pop("EB_RPO_AMP_ACTUATOR_DELAY_STEPS", None)
    import onnxruntime as ort
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True, enable_cameras=record_video).app
    try:
        from isaaclab_tasks.utils import load_cfg_from_registry
        import gymnasium as gym
        import torch
        from rpo_wbc_extension import AMP_TASK_ID, BASIC_TASK_ID, TASK_ID, register_task

        register_task()
        session = ort.InferenceSession(str(policy), providers=["CPUExecutionProvider"])
        obs_dim = 780 if amp else OBS_DIM
        if session.get_inputs()[0].shape[-1] != obs_dim or session.get_outputs()[0].shape[-1] != ACTION_DIM:
            raise ValueError(f"evaluation requires the measured {obs_dim}x23 ONNX contract")
        export_receipt = policy.parent / "export-manifest.json"
        if amp and not export_receipt.is_file():
            raise ValueError("AMP evaluation requires the actual verified ONNX export receipt")
        max_abs_error = json.loads(export_receipt.read_text())["onnx"]["max_abs_error"] if export_receipt.is_file() else 0.0
        output.mkdir(parents=True, exist_ok=True)
        rows = []
        task_id = AMP_TASK_ID if amp else BASIC_TASK_ID if criterion == "stand_walk" else TASK_ID
        cfg = load_cfg_from_registry(task_id, "env_cfg_entry_point")
        cfg.seed = seeds[0]
        cfg.scene.num_envs = 1
        cfg.noise.add_noise = False
        if record_video:
            from isaaclab import sim as sim_utils
            from isaaclab.sensors import CameraCfg

            cfg.commands.debug_vis = False
            cfg.scene.left_feet_scanner.debug_vis = False
            cfg.scene.right_feet_scanner.debug_vis = False
            cfg.scene.terrain.visual_material = sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.25, 0.25, 0.25))
            cfg.scene.sky_light.spawn = sim_utils.DomeLightCfg(
                color=(0.75, 0.75, 0.75), intensity=750.0)
            cfg.scene.eval_camera = CameraCfg(
                prim_path="/World/EvalCamera", update_period=0.1, height=240, width=320,
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, clipping_range=(0.1, 20.0)),
                offset=CameraCfg.OffsetCfg(
                    pos=(3.0, 3.0, 1.6),
                    rot=(0.380595, 0.096385, 0.039924, -0.918838), convention="world"),
            )
            cfg.sim.render_interval = cfg.decimation * 5
            cfg.num_rerenders_on_reset = 1
            cfg.wait_for_textures = False
        env = gym.make(task_id, cfg=cfg, render_mode=None).unwrapped
        try:
            for seed in seeds:
                scenario = ("stand" if seed % 2 == 0 else "walk") if criterion == "stand_walk" else ""
                if scenario:
                    env.command_generator.cfg.rel_standing_envs = 1.0 if scenario == "stand" else 0.0
                writer = None
                video_rel = f"seed_{seed}/0.mp4"
                if record_video:
                    import imageio.v2 as imageio
                    video_path = output / "video/eval" / video_rel
                    video_path.parent.mkdir(parents=True, exist_ok=True)
                    writer = imageio.get_writer(video_path, fps=10, codec="libx264")
                try:
                    obs, _ = env.reset(seed=seed)
                    delays = actuator_delay_steps(env.robot)
                    start_xy = env.robot.data.root_pos_w[0, :2].detach().cpu().numpy().copy()
                    start_quat = env.robot.data.root_quat_w[0].detach().cpu().numpy().copy()
                    initial_yaw = float(np.arctan2(2 * (start_quat[0] * start_quat[3] + start_quat[1] * start_quat[2]),
                                                   1 - 2 * (start_quat[2] ** 2 + start_quat[3] ** 2)))
                    if writer is not None:
                        writer.append_data(env.scene["eval_camera"].data.output["rgb"][0, :, :, :3]
                                           .detach().cpu().numpy().astype(np.uint8))
                    upper_errors, speed_errors = [], []
                    fell = nan = hard_limit = False
                    fall_reason = None
                    for step in range(EVAL_POLICY_STEPS):
                        actor_obs = obs["policy"].detach().cpu().numpy().astype(np.float32)
                        action = session.run(None, {session.get_inputs()[0].name: actor_obs})[0].astype(np.float32)
                        if not np.isfinite(actor_obs).all() or not np.isfinite(action).all():
                            nan = True
                            break
                        hard_limit |= bool(np.any(np.abs(action) > 100.0))
                        obs, _, terminated, truncated, _ = env.step(torch.from_numpy(action).to(env.device))
                        if not amp:
                            ids = env.codec_cfg.joint_ids[12:]
                            upper = env.robot.data.joint_pos[:, ids] - env.robot.data.default_joint_pos[:, ids]
                            upper_errors.append(float(np.square(upper.detach().cpu().numpy() - env.upper_reference.detach().cpu().numpy()).mean()))
                        speed_errors.append(float(np.square(env.robot.data.root_lin_vel_b[:, :2].detach().cpu().numpy()
                                                            - env.command_generator.command[:, :2].detach().cpu().numpy()).mean()))
                        height = float(env.robot.data.root_pos_w[0, 2])
                        gravity_z = float(env.robot.data.projected_gravity_b[0, 2])
                        fall_reason = physical_fall(height, gravity_z)
                        fell |= fall_reason is not None
                        if writer is not None and (step + 1) % 5 == 0:
                            writer.append_data(env.scene["eval_camera"].data.output["rgb"][0, :, :, :3]
                                               .detach().cpu().numpy().astype(np.uint8))
                        nan |= (not np.isfinite(actor_obs).all() or not np.isfinite(action).all()
                                or not np.isfinite((height, gravity_z)).all())
                        if bool((terminated | truncated).any()) or fell or nan:
                            if bool(terminated.any()) and not fell:
                                fell, fall_reason = True, "contact"
                            if writer is not None and (step + 1) % 5:
                                writer.append_data(env.scene["eval_camera"].data.output["rgb"][0, :, :, :3]
                                                   .detach().cpu().numpy().astype(np.uint8))
                            break
                finally:
                    if writer is not None:
                        writer.close()
                upper_rmse = None if amp else float(np.sqrt(np.mean(upper_errors))) if upper_errors else 1e9
                velocity_rmse = float(np.sqrt(np.mean(speed_errors))) if speed_errors else 1e9
                episode_len = step + 1
                delta_xy = env.robot.data.root_pos_w[0, :2].detach().cpu().numpy() - start_xy
                distance_m = float(np.linalg.norm(delta_xy) if scenario == "stand" else
                                   delta_xy[0] * np.cos(initial_yaw) + delta_xy[1] * np.sin(initial_yaw))
                success = case_success(criterion, episode_len, fell, nan, hard_limit,
                                       upper_rmse, velocity_rmse, scenario=scenario, distance_m=distance_m, settings=settings)
                termination_reason = ("nan" if nan else "fall" if fell else "horizon" if success else
                                      "hard_limit" if hard_limit and criterion == "tracking" else
                                      "tracking_error" if criterion == "tracking"
                                      else "goal_not_met" if criterion == "stand_walk" else "early_termination")
                rows.append({"seed":seed, "case_id":f"rpo-seed-{seed}",
                             "task_id":"tsk_rpo_amp_stand_walk" if amp else "tsk_rpo_stand_walk" if scenario else "tsk_rpo_walk_upperbody_tracking",
                             "init_state_id":f"seed_{seed}",
                             "status":"success" if success else "task_failure", "success":success,
                             "episode_len":episode_len, "termination_reason":termination_reason,
                             "scenario":scenario, "distance_m":distance_m, "fall_reason":fall_reason,
                             "completed":True, "fall":fell, "nan":nan, "hard_limit":hard_limit,
                             "upper_rmse":upper_rmse, "velocity_rmse":velocity_rmse,
                             "actuator_delay_steps":delays,
                             **({"video_rel":video_rel} if record_video else {})})
        finally:
            env.close()
        (output / "eval-cases.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        receipt = {"schema_version":"eb/rpo-amp-eval/v1" if amp else "eb/rpo-wbc-eval/v1", "backend":"isaac",
            "criterion":criterion, "required_steps":EVAL_POLICY_STEPS,
            "record_video":record_video, "environment_settings":settings or {},
            "onnx":{"sha256":sha256(policy), "obs":obs_dim, "act":ACTION_DIM, "max_abs_error":max_abs_error},
            "cases":{"planned":len(seeds), "completed":len(rows), "valid":len(rows), "invalid":0, "infra_failed":0},
            "metrics":{"n_trials":len(rows), "successes":sum(r["success"] for r in rows),
                       "success_rate":sum(r["success"] for r in rows) / len(rows),
                       "fall_count":sum(r["fall"] for r in rows), "hard_limit_count":sum(r["hard_limit"] for r in rows),
                       "nan_count":sum(r["nan"] for r in rows), "upper_rmse":None if amp else float(np.mean([r["upper_rmse"] for r in rows])),
                       "velocity_rmse":float(np.mean([r["velocity_rmse"] for r in rows]))}}
        (output / "eval-manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
    finally:
        app.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    exp = sub.add_parser("export")
    exp.add_argument("--checkpoint", type=Path, required=True)
    exp.add_argument("--output", type=Path, required=True)
    exp.add_argument("--recipe", choices=("wbc", "amp"), default="wbc")
    eva = sub.add_parser("eval-mujoco")
    eva.add_argument("--policy", type=Path, required=True)
    eva.add_argument("--mjcf", type=Path, required=True)
    eva.add_argument("--output", type=Path, required=True)
    eva.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9,10,11")
    isaac = sub.add_parser("eval-isaac")
    isaac.add_argument("--policy", type=Path, required=True)
    isaac.add_argument("--output", type=Path, required=True)
    isaac.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9,10,11")
    isaac.add_argument("--record-video", action="store_true")
    direct_isaac = sub.add_parser("eval-checkpoint-isaac")
    direct_isaac.add_argument("--checkpoint", type=Path, required=True)
    direct_isaac.add_argument("--recipe", choices=("wbc", "amp"), default="wbc")
    direct_isaac.add_argument("--output", type=Path, required=True)
    direct_isaac.add_argument("--profile", choices=("smoke", "standard"), default="smoke")
    direct_isaac.add_argument("--criterion", choices=("stability", "tracking", "stand_walk"), default="tracking")
    direct_isaac.add_argument("--seeds")
    direct_isaac.add_argument("--record-video", action="store_true")
    direct_isaac.add_argument("--stand-max-drift-m", type=float, default=0.5)
    direct_isaac.add_argument("--walk-min-forward-m", type=float, default=4.0)
    direct_isaac.add_argument("--initial-pose", choices=("official_a1_stand_v1",))
    direct_isaac.add_argument("--actuator-delay-steps", type=int, choices=(0, 1, 2))
    direct = sub.add_parser("eval-checkpoint-mujoco")
    direct.add_argument("--checkpoint", type=Path, required=True)
    direct.add_argument("--recipe", choices=("wbc", "amp"), default="wbc")
    direct.add_argument("--mjcf", type=Path, required=True)
    direct.add_argument("--output", type=Path, required=True)
    direct.add_argument("--profile", choices=("smoke", "standard"), default="smoke")
    direct.add_argument("--criterion", choices=("stability", "tracking", "stand_walk"), default="tracking")
    direct.add_argument("--seeds")
    direct.add_argument("--record-video", action="store_true")
    direct.add_argument("--stand-max-drift-m", type=float, default=0.5)
    direct.add_argument("--walk-min-forward-m", type=float, default=4.0)
    direct.add_argument("--initial-pose", choices=("official_a1_stand_v1",))
    args = parser.parse_args()
    if args.mode == "export":
        export(args.checkpoint, args.output, **({"amp": True} if args.recipe == "amp" else {}))
    elif args.mode == "eval-mujoco":
        evaluate_mujoco(args.policy, args.mjcf, args.output, [int(seed) for seed in args.seeds.split(",")])
    elif args.mode == "eval-checkpoint-mujoco":
        export(args.checkpoint, args.output / "exported", **({"amp": True} if args.recipe == "amp" else {}))
        evaluate_mujoco(args.output / "exported/policy.onnx", args.mjcf, args.output,
                        [int(seed) for seed in args.seeds.split(",")] if args.seeds is not None else
                        list(range(3 if args.profile == "smoke" else 12)),
                        record_video=args.record_video, criterion=args.criterion,
                        settings={"stand_max_drift_m":args.stand_max_drift_m, "walk_min_forward_m":args.walk_min_forward_m,
                                  **({"initial_pose":args.initial_pose} if args.initial_pose else {})},
                        amp=args.recipe == "amp")
    elif args.mode == "eval-checkpoint-isaac":
        export(args.checkpoint, args.output / "exported", **({"amp": True} if args.recipe == "amp" else {}))
        evaluate_isaac(args.output / "exported/policy.onnx", args.output,
                       [int(seed) for seed in args.seeds.split(",")] if args.seeds is not None else
                       list(range(3 if args.profile == "smoke" else 12)),
                       record_video=args.record_video, criterion=args.criterion,
                       settings={"stand_max_drift_m":args.stand_max_drift_m, "walk_min_forward_m":args.walk_min_forward_m,
                                 **({"initial_pose":args.initial_pose} if args.initial_pose else {}),
                                 **({"actuator_delay_steps":args.actuator_delay_steps}
                                    if args.actuator_delay_steps is not None else {})},
                       amp=args.recipe == "amp")
    else:
        evaluate_isaac(args.policy, args.output, [int(seed) for seed in args.seeds.split(",")],
                       record_video=args.record_video)


if __name__ == "__main__":
    main()
