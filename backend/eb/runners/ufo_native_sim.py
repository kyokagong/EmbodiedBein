"""ONNX closed-loop rollouts on frozen MJLab and native MuJoCo backends."""

from __future__ import annotations

import importlib.util
import json
import math
import os
import subprocess
import time
from pathlib import Path

import numpy as np

from eb.runners.ufo_native_package import file_index, receipt, verified_manifest
from eb.ufo_native import read_json, sha256_file, validate_cases

SUBSTEP_COLUMNS = (
    "joint_position_excess_rad",
    "velocity_excess_rad_s",
    "torque_excess_nm",
    "abs_joint_position_rad",
    "abs_velocity_rad_s",
    "abs_torque_nm",
)
ROLLOUT_CHECK_STEPS = (0, 1, 2, 3, 49, 99, 249, 499, 749, 999)


def package_policy(payload):
    spec = importlib.util.spec_from_file_location("delivered_runtime", payload / "runtime.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Policy(payload)


def mass_weighted_com(centers, masses):
    return (centers * masses[..., None]).sum(-2) / masses.sum(-1)[..., None]


def physical_sample(q, dq, torque, limits, xp=np):
    lower, upper, velocity, effort = limits
    excess = xp.stack(
        (
            xp.maximum(lower - q, q - upper).max(),
            (xp.abs(dq) - velocity).max(),
            (xp.abs(torque) - effort).max(),
        )
    )
    peaks = xp.stack((xp.abs(q).max(), xp.abs(dq).max(), xp.abs(torque).max()))
    return xp.concatenate((xp.maximum(excess, xp.zeros_like(excess)), peaks))


def physical_metrics(samples, physics_dt):
    samples = np.asarray(samples).reshape(-1, 6)
    metrics = {"physics_substeps": len(samples)}
    for index, kind in enumerate(("joint", "velocity", "torque")):
        count = int(np.count_nonzero(samples[:, index] > 0.001))
        metrics.update(
            {
                f"{kind}_violation_count": count,
                f"{kind}_violation_fraction": count / len(samples),
                f"{kind}_violation_seconds": count * physics_dt,
            }
        )
    for index, name in enumerate(
        (
            "max_joint_excess_rad",
            "max_velocity_excess_rad_s",
            "max_torque_excess_nm",
            "max_abs_joint_position_rad",
            "max_abs_velocity_rad_s",
            "max_abs_torque_nm",
        )
    ):
        metrics[name] = float(samples[:, index].max())
    return metrics


def action_errors(expected, actual, gains):
    expected, actual = np.asarray(expected), np.asarray(actual)
    target_error = np.abs(np.clip(expected, -1, 1) - np.clip(actual, -1, 1)) * np.asarray(gains)
    if (
        expected.shape != actual.shape
        or not np.isfinite(expected).all()
        or not np.isfinite(actual).all()
        or not np.allclose(actual, expected, atol=1e-5, rtol=1e-4)
        or target_error.max() > 1e-4
    ):
        raise ValueError("Actual rollout Torch actor and ONNX are not equivalent")
    return float(np.abs(expected - actual).max()), float(target_error.max())


def rollout_action(model, policy, task):
    import torch

    observation = policy.last_observation
    obs = {
        "state": torch.from_numpy(observation[None, :52]),
        "last_action": torch.from_numpy(observation[None, 52:75]),
        "history_actor": torch.from_numpy(observation[None, 75:]),
    }
    context = torch.tensor([policy.prompts[task]], dtype=torch.float32)
    with torch.inference_mode():
        return model.act(obs, context, mean=True)[0].numpy()


def motor_torque(requested, velocity, effort, limit, keep):
    saturation = effort / (1.0 - keep / limit)
    clipped_velocity = np.clip(
        velocity, -limit * (1 + effort / saturation), limit * (1 + effort / saturation)
    )
    top = np.minimum(saturation * (1.0 - clipped_velocity / limit), effort)
    bottom = np.maximum(saturation * (-1.0 - clipped_velocity / limit), -effort)
    return np.clip(requested, bottom, top)


class CommandDelay:
    def __init__(self, lag):
        self.lag = lag
        self.history = None

    def push(self, target):
        if self.history is None:
            self.history = np.repeat(target[None], self.lag + 1, axis=0)
        else:
            self.history[1:] = self.history[:-1].copy()
            self.history[0] = target
        return self.history[self.lag].copy()


def case_lag(codec, seed):
    low, high = (codec["actuator"].get(k, 0) for k in ("min_delay", "max_delay"))
    return low + seed % (high - low + 1)


def orientation(quat):
    import mujoco

    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, np.asarray(quat, dtype=np.float64))
    rotation = rotation.reshape(3, 3)
    gravity = rotation.T @ np.array([0.0, 0.0, -1.0])
    yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    tilt = math.acos(np.clip(rotation[2, 2], -1.0, 1.0))
    return gravity, yaw, tilt


def case_metrics(trace, *, speed, yaw, initial_xy=None, physics_dt=0.0025):
    height, tilt = np.asarray(trace["height"]), np.asarray(trace["tilt"])
    falls = np.flatnonzero((height < 0.5) | (tilt > 0.8))
    velocity = np.asarray(trace["velocity"])
    error = velocity - np.array([speed, 0.0])
    rmse = float(np.sqrt(np.square(error).sum(1).mean()))
    start_xy = trace["xy"][0] if initial_xy is None else initial_xy
    delta = np.asarray(trace["xy"])[-1] - np.asarray(start_xy)
    forward = float(delta @ np.array([math.cos(yaw), math.sin(yaw)]))
    drift = float(np.linalg.norm(delta))
    metrics = {
        "speed_rmse_m_s": rmse,
        "forward_displacement_m": forward,
        "drift_m": drift,
        "min_height_m": float(height.min()),
        "max_tilt_rad": float(tilt.max()),
        "first_fall_step": int(falls[0] + 1) if len(falls) else None,
        "joint_violation_fraction": float(np.mean(trace["joint_violation"])),
        "velocity_violation_fraction": float(np.mean(trace["velocity_violation"])),
        "torque_violation_fraction": float(np.mean(trace["torque_violation"])),
        "inference_p50_ms": float(np.percentile(trace["latency_ms"], 50)),
        "inference_p95_ms": float(np.percentile(trace["latency_ms"], 95)),
        "deadline_miss_fraction": float(np.mean(np.asarray(trace["latency_ms"]) > 20.0)),
    }
    if "substep_limits" in trace:
        metrics.update(physical_metrics(trace["substep_limits"], physics_dt))
    if len(falls):
        return "task_failure", "fall", metrics
    if len(height) < 1000:
        return "timeout", "unexpected_early_end", metrics
    if (
        metrics["joint_violation_fraction"]
        or metrics["velocity_violation_fraction"]
        or metrics["torque_violation_fraction"]
    ):
        return "task_failure", "physical_limit", metrics
    if rmse > (0.15 if speed == 0 else 0.2) or (drift > 0.5 if speed == 0 else forward < 4.0):
        return "task_failure", "tracking", metrics
    return "success", "completed", metrics


class MujocoBackend:
    def __init__(self, payload, source):
        import mujoco

        self.mj = mujoco
        self.codec = read_json(payload / "codec.json")
        self.limits = tuple(np.asarray(self.codec[k]) for k in ("lower", "upper", "velocity", "effort"))
        spec = mujoco.MjSpec.from_file(str(payload / "assets/rpo.xml"))
        for actuator in list(spec.actuators):
            spec.delete(actuator)
        spec.worldbody.add_geom(
            name="deployment_floor",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[100.0, 100.0, 0.1],
            friction=[1.0, 0.005, 0.0001],
            rgba=[0.4, 0.42, 0.45, 1.0],
        )
        spec.worldbody.add_light(pos=[0.0, -3.0, 4.0], dir=[0.0, 0.5, -1.0])
        spec.option.timestep = self.codec["physics_dt"]
        for name in self.codec["joint_order"]:
            effort = self.codec["actuator"]["joints"][name]["effort_limit"]
            spec.add_actuator(
                name=name,
                target=name,
                trntype=mujoco.mjtTrn.mjTRN_JOINT,
                gear=[1, 0, 0, 0, 0, 0],
                ctrllimited=True,
                ctrlrange=[-effort, effort],
                forcelimited=True,
                forcerange=[-effort, effort],
            )
        self.model = spec.compile()
        from mjlab.sim.sim import MujocoCfg

        MujocoCfg(**read_json(payload / "physics.json")).apply(self.model)
        self.data = mujoco.MjData(self.model)
        ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in self.codec["joint_order"]]
        self.qadr = self.model.jnt_qposadr[ids]
        self.vadr = self.model.jnt_dofadr[ids]
        self.feet_names = read_json(source / "effective_config.json")["task"]["robot"]["contact_bodies"]
        self.feet_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n) for n in self.feet_names]
        if len(self.feet_ids) != 2 or min(self.feet_ids) < 0:
            raise ValueError("Native foot bodies are missing from MuJoCo scene")
        params = [self.codec["actuator"]["joints"][n] for n in self.codec["joint_order"]]
        self.effort = np.array([p["effort_limit"] for p in params])
        self.velocity_limit = np.array([p["velocity_limit"] for p in params])
        self.keep = np.array([p.get("keep_velocity", 0.0) for p in params])
        self.model.dof_armature[self.vadr] = [p["armature"] for p in params]
        self.model.dof_frictionloss[self.vadr] = [p["friction"] for p in params]
        for address, params_ in zip(self.vadr, params):
            if "viscous_friction" in params_:
                self.model.dof_damping[address] = params_["viscous_friction"]
        self.kp, self.kd = np.asarray(self.codec["kp"]), np.asarray(self.codec["kd"])
        self.torque = np.zeros(23)
        self.renderer = mujoco.Renderer(self.model, height=480, width=480)
        self.camera = mujoco.MjvCamera()
        self.camera.distance, self.camera.azimuth, self.camera.elevation = 3.0, 135.0, -18.0

    def reset(self, seed):
        self.delay = CommandDelay(case_lag(self.codec, seed))
        self.mj.mj_resetData(self.model, self.data)
        rng = np.random.default_rng(seed)
        self.yaw = float(rng.uniform(-math.pi, math.pi))
        self.data.qpos[:3] = [0.0, 0.0, 0.75]
        self.data.qpos[3:7] = [math.cos(self.yaw / 2), 0.0, 0.0, math.sin(self.yaw / 2)]
        self.data.qpos[self.qadr] = np.asarray(self.codec["default_pos"]) + rng.uniform(-0.05, 0.05, 23)
        self.data.qvel[:2] = rng.uniform(-0.05, 0.05, 2)
        self.data.qvel[5] = rng.uniform(-0.05, 0.05)
        self.mj.mj_forward(self.model, self.data)

    def state(self):
        gravity, yaw, tilt = orientation(self.data.qpos[3:7])
        world = self.data.qvel[:2]
        velocity = [
            math.cos(yaw) * world[0] + math.sin(yaw) * world[1],
            -math.sin(yaw) * world[0] + math.cos(yaw) * world[1],
        ]
        return {
            "q": self.data.qpos[self.qadr].copy(),
            "dq": self.data.qvel[self.vadr].copy(),
            "gravity": gravity,
            "omega": self.data.qvel[3:6].copy(),
            "height": self.data.qpos[2],
            "tilt": tilt,
            "velocity": velocity,
            "xy": self.data.qpos[:2].copy(),
            "qpos": self.data.qpos.copy(),
            "torque": self.torque.copy(),
            "com_position_w": mass_weighted_com(self.data.xipos, self.model.body_mass),
            **self._feet_state(),
        }

    def _feet_state(self):
        velocity = np.empty((len(self.feet_ids), 3))
        forces = np.zeros_like(velocity)
        spatial = np.empty(6)
        for index, body in enumerate(self.feet_ids):
            self.mj.mj_objectVelocity(self.model, self.data, self.mj.mjtObj.mjOBJ_XBODY, body, spatial, 0)
            velocity[index] = spatial[3:]
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            bodies = self.model.geom_bodyid[[contact.geom1, contact.geom2]]
            if not any(body in bodies for body in self.feet_ids):
                continue
            self.mj.mj_contactForce(self.model, self.data, index, spatial)
            world_force = np.asarray(contact.frame).reshape(3, 3).T @ spatial[:3]
            for foot, body in enumerate(self.feet_ids):
                if body == bodies[0]:
                    forces[foot] -= world_force
                if body == bodies[1]:
                    forces[foot] += world_force
        return {
            "foot_position_w": self.data.xpos[self.feet_ids].copy(),
            "foot_velocity_w": velocity,
            "foot_contact_force_w": forces,
        }

    def step(self, raw, target):
        samples = []
        for _ in range(8):
            dq = self.data.qvel[self.vadr]
            requested = self.kp * (self.delay.push(target) - self.data.qpos[self.qadr]) - self.kd * dq
            self.torque = motor_torque(requested, dq, self.effort, self.velocity_limit, self.keep)
            self.data.ctrl[:] = self.torque
            self.mj.mj_step(self.model, self.data)
            self.torque = self.data.qfrc_actuator[self.vadr].copy()
            samples.append(
                physical_sample(
                    self.data.qpos[self.qadr],
                    self.data.qvel[self.vadr],
                    self.torque,
                    self.limits,
                )
            )
        self.substeps = np.stack(samples)

    def render(self):
        self.camera.lookat[:] = [*self.data.qpos[:2], max(0.65, self.data.qpos[2])]
        self.renderer.update_scene(self.data, camera=self.camera)
        return self.renderer.render().copy()

    def close(self):
        self.renderer.close()


class MjlabBackend:
    def __init__(self, payload, source, *, seconds=21):
        import torch
        from ufo.inference.runtime import MujocoQposRenderer

        from eb.runners.ufo_native_package import native_env_config

        self.torch = torch
        cfg = native_env_config(source, seconds=seconds)
        self.env, self.info = cfg.build(num_envs=1)
        self.core = self.env._env
        self.core.is_evaluating = True
        self.codec = read_json(payload / "codec.json")
        self.limits = tuple(
            torch.tensor(self.codec[k], device=self.core.device)
            for k in ("lower", "upper", "velocity", "effort")
        )
        self.feet_names = [self.core.body_names[i] for i in self.core.feet_indices.tolist()]
        self._substeps = None
        self._scene_update = self.core.mjlab_env.scene.update
        self.core.mjlab_env.scene.update = self._track_substep
        self.renderer = MujocoQposRenderer(payload / "assets/rpo.xml", render_size=480)
        self.qpos = None
        self.native_obs = None
        self.codec_error = 0.0

    def reset(self, seed):
        from ufo.inference.reward_inference import _default_standing_target_states

        self.torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        self.yaw = float(rng.uniform(-math.pi, math.pi))
        states = _default_standing_target_states(self.env, device=self.core.device)
        states["root_states"][:, :3] = self.torch.tensor([0.0, 0.0, 0.75], device=self.core.device)
        states["root_states"][:, 3:7] = self.torch.tensor(
            [0.0, 0.0, math.sin(self.yaw / 2), math.cos(self.yaw / 2)], device=self.core.device
        )
        states["dof_states"][..., 0] += self.torch.tensor(
            rng.uniform(-0.05, 0.05, 23), device=self.core.device
        )
        states["root_states"][:, 7:9] = self.torch.tensor(
            rng.uniform(-0.05, 0.05, 2), device=self.core.device
        )
        states["root_states"][:, 12] = float(rng.uniform(-0.05, 0.05))
        self.native_obs, _ = self.env.reset(to_numpy=False, target_states=states)
        lag = case_lag(self.codec, seed)
        for actuator in self.core.robot.actuators:
            if actuator.cfg.delay_max_lag > 0:
                if actuator.cfg.delay_hold_prob != 1.0:
                    raise ValueError("Native actuator delay is not held per episode")
                actuator.set_lags(self.torch.tensor([lag], device=self.core.device, dtype=self.torch.long))
        self.qpos = self.env._get_qpos_qvel(to_numpy=True)[0][0]

    def state(self):
        def to_np(tensor):
            return tensor[0].detach().cpu().numpy().copy()

        data = self.core.robot.data
        com = mass_weighted_com(
            data.body_com_pose_w[:, self.core._body_ids, :3],
            data.model.body_mass[:, self.core.robot.indexing.body_ids[self.core._body_ids]],
        )
        root = to_np(self.core.robot_root_states)
        quat = root[[6, 3, 4, 5]]
        _, yaw, tilt = orientation(quat)
        velocity = [
            math.cos(yaw) * root[7] + math.sin(yaw) * root[8],
            -math.sin(yaw) * root[7] + math.cos(yaw) * root[8],
        ]
        return {
            "q": to_np(self.core.dof_pos),
            "dq": to_np(self.core.dof_vel),
            "gravity": to_np(self.core.projected_gravity),
            "omega": to_np(self.core.base_ang_vel),
            "height": root[2],
            "tilt": tilt,
            "velocity": velocity,
            "xy": root[:2],
            "qpos": np.asarray(self.qpos).copy(),
            "torque": to_np(self.core.torques),
            "com_position_w": to_np(com),
            "foot_position_w": to_np(self.core.body_pos[:, self.core.feet_indices]),
            "foot_velocity_w": to_np(self.core.body_vel[:, self.core.feet_indices]),
            "foot_contact_force_w": to_np(self.core.contact_forces[:, self.core.feet_indices]),
        }

    def _track_substep(self, dt):
        self._scene_update(dt=dt)
        if self._substeps is not None:
            data, ids = self.core.robot.data, self.core._joint_ids
            self._substeps.append(
                physical_sample(
                    data.joint_pos[0, ids],
                    data.joint_vel[0, ids],
                    data.qfrc_actuator[0, ids],
                    self.limits,
                    self.torch,
                )
            )

    def check_observation(self, codec_obs):
        native = np.concatenate(
            [self.native_obs[k][0].detach().cpu().numpy() for k in ("state", "last_action", "history_actor")]
        )
        self.codec_error = max(self.codec_error, float(np.abs(native - codec_obs).max()))
        if not np.allclose(native, codec_obs, atol=1e-5, rtol=1e-5):
            raise ValueError(f"Native observation/standalone codec disagree: {self.codec_error}")

    def step(self, raw, target):
        samples = []
        self._substeps = samples
        try:
            self.native_obs, _, done, truncated, info = self.env.step(
                self.torch.tensor(raw[None], device=self.core.device), to_numpy=False
            )
        finally:
            self._substeps = None
        if len(samples) != self.codec["decimation"]:
            raise ValueError("Native substep evidence count differs from frozen decimation")
        self.substeps = self.torch.stack(samples).detach().cpu().numpy()
        if bool((done | truncated).any()):
            raise RuntimeError("Unexpected native reset in fixed-horizon deployment simulation")
        self.qpos = info["qpos"][0].detach().cpu().numpy()

    def render(self):
        return self.renderer.render_qpos(self.qpos)

    def close(self):
        self.core.mjlab_env.scene.update = self._scene_update
        self.renderer.close()
        self.env.close()


def probe_video(path):
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-xerror",
            "-err_detect",
            "explode",
            "-i",
            str(path),
            "-f",
            "null",
            "-",
        ],
        check=True,
        capture_output=True,
        timeout=60,
    )
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_frames,avg_frame_rate,duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    stream = json.loads(result.stdout)["streams"][0]
    numerator, denominator = map(float, stream["avg_frame_rate"].split("/"))
    if not denominator or int(stream["nb_frames"]) < 1:
        raise ValueError("Video has no decodable frames")
    return {
        "decodable": True,
        "frames": int(stream["nb_frames"]),
        "fps": numerator / denominator,
        "seconds": float(stream["duration"]),
    }


def plots(output, traces, rows):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    labels = [r["task"].replace("move-ego-0-", "") for r in rows]
    axes[0].bar(
        range(len(rows)),
        [r["steps"] / 50 for r in rows],
        color=["#2a9d8f" if r["status"] == "success" else "#e76f51" for r in rows],
    )
    axes[0].set(ylabel="Completed seconds (max 20)", xticks=range(len(rows)), xticklabels=labels)
    axes[1].bar(range(len(rows)), [r["metrics"]["speed_rmse_m_s"] for r in rows])
    axes[1].set(ylabel="Horizontal speed RMSE (m/s)", xticks=range(len(rows)), xticklabels=labels)
    fig.tight_layout()
    fig.savefig(output / "summary.svg")
    plt.close(fig)
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    for row, trace in zip(rows, traces):
        t = np.asarray(trace["time_s"])
        for ax, key in zip(axes, ("height", "tilt", "latency_ms")):
            ax.plot(t, trace[key], label=row["case_id"])
    for ax, label in zip(axes, ("Base height (m)", "Tilt (rad)", "Inference latency (ms)")):
        ax.set(xlabel="Simulation time (s)", ylabel=label)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output / "timeseries.svg")
    plt.close(fig)


def save_case(output, row, arrays):
    np.savez_compressed(output / row["trace"], **arrays)
    row["evidence_files"] = [
        {"path": name, "bytes": (output / name).stat().st_size, "sha256": sha256_file(output / name)}
        for name in (row["video"], row["trace"])
    ]
    with (output / "cases.jsonl").open("a") as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n")
        stream.flush()


def evaluate(output, plan):
    import torch
    from ufo.agents.common.checkpoint import load_model_from_checkpoint_dir
    from ufo.inference.runtime import video_writer

    payload = Path("/input")
    verified_manifest(payload, plan["source_artifact_sha256"])
    source = Path("/native-source")
    verified_manifest(source, plan["native_source_sha256"])
    (output / "videos").mkdir()
    (output / "traces").mkdir()
    (output / "cases.jsonl").write_text("")
    policy = package_policy(payload)
    codec = policy.codec.config
    torch.set_num_threads(4)
    model = load_model_from_checkpoint_dir(source / "checkpoint", device="cpu")
    model.eval()
    backend = MjlabBackend(payload, source) if plan["backend"] == "mjlab" else MujocoBackend(payload, source)
    rows, traces = [], []
    start = time.monotonic()
    try:
        for case in plan["cases"]:
            backend.reset(case["seed"])
            policy.codec.reset()
            initial_xy = backend.state()["xy"].copy()
            differential = {
                "differential_step": [],
                "differential_observation": [],
                "differential_torch_action": [],
                "differential_onnx_action": [],
            }
            max_action_error, max_target_error = 0.0, 0.0
            trace = {
                key: []
                for key in (
                    "height",
                    "tilt",
                    "velocity",
                    "xy",
                    "joint_violation",
                    "velocity_violation",
                    "torque_violation",
                    "latency_ms",
                    "q",
                    "dq",
                    "qpos",
                    "torque",
                    "raw_action",
                    "target",
                    "time_s",
                    "substep_limits",
                    "foot_position_w",
                    "foot_velocity_w",
                    "foot_contact_force_w",
                    "com_position_w",
                )
            }
            video = f"videos/{case['case_id']}.mp4"
            with video_writer(output / video, shape=(480, 480), fps=25) as writer:
                for step in range(1000):
                    before = backend.state()
                    t = time.perf_counter()
                    raw, target = policy.act(
                        before["q"], before["dq"], before["gravity"], before["omega"], case["task"]
                    )
                    elapsed = (time.perf_counter() - t) * 1000
                    if isinstance(backend, MjlabBackend):
                        backend.check_observation(policy.last_observation)
                    if step in ROLLOUT_CHECK_STEPS:
                        expected = rollout_action(model, policy, case["task"])
                        action_error, target_error = action_errors(expected, raw, codec["action_gain"])
                        max_action_error = max(max_action_error, action_error)
                        max_target_error = max(max_target_error, target_error)
                        differential["differential_step"].append(step + 1)
                        differential["differential_observation"].append(policy.last_observation.copy())
                        differential["differential_torch_action"].append(expected.copy())
                        differential["differential_onnx_action"].append(raw.copy())
                    backend.step(raw, target)
                    state = backend.state()
                    values = {
                        **state,
                        "raw_action": raw,
                        "target": target,
                        "latency_ms": elapsed,
                        "time_s": (step + 1) * codec["decimation"] * codec["physics_dt"],
                        "substep_limits": backend.substeps.copy(),
                        "joint_violation": bool((backend.substeps[:, 0] > 0.001).any()),
                        "velocity_violation": bool((backend.substeps[:, 1] > 0.001).any()),
                        "torque_violation": bool((backend.substeps[:, 2] > 0.001).any()),
                    }
                    if not all(np.isfinite(np.asarray(v)).all() for v in values.values()):
                        raise ValueError("Non-finite deployment simulation state")
                    for key in trace:
                        trace[key].append(values[key])
                    if step % 2 == 0:
                        writer.add_image(backend.render())
                    if state["height"] < 0.5 or state["tilt"] > 0.8:
                        break
            status, reason, metrics = case_metrics(
                trace,
                speed=case["speed"],
                yaw=backend.yaw,
                initial_xy=initial_xy,
                physics_dt=codec["physics_dt"],
            )
            rows.append(
                {
                    **case,
                    "run_id": os.environ["EB_RUN_ID"],
                    "plan_hash": plan["hash"],
                    "model_id": plan["model_id"],
                    "payload_sha256": plan["source_artifact_sha256"],
                    "status": status,
                    "reason": reason,
                    "steps": len(trace["height"]),
                    "metrics": metrics,
                    "initial_xy_m": initial_xy.tolist(),
                    "actuator_lag_substeps": case_lag(codec, case["seed"]),
                    "torch_onnx": {
                        "verified": True,
                        "samples": len(differential["differential_step"]),
                        "max_action_error": max_action_error,
                        "max_target_error_rad": max_target_error,
                        "atol": 1e-5,
                        "rtol": 1e-4,
                        "target_atol_rad": 1e-4,
                    },
                    "video": video,
                    "trace": f"traces/{case['case_id']}.npz",
                    "media": probe_video(output / video),
                }
            )
            traces.append(trace)
            save_case(output, rows[-1], {k: np.asarray(v) for k, v in {**trace, **differential}.items()})
            print(json.dumps({"case": rows[-1]}), flush=True)
    finally:
        backend.close()
    plots(output, traces, rows)
    files = file_index(output / "videos")
    files = [{**f, "path": "videos/" + f["path"]} for f in files]
    files += [{**f, "path": "traces/" + f["path"]} for f in file_index(output / "traces")]
    for name in ("summary.svg", "timeseries.svg", "cases.jsonl"):
        p = output / name
        files.append({"path": name, "bytes": p.stat().st_size, "sha256": sha256_file(p)})
    metrics = validate_cases(plan, rows)
    receipt(
        output,
        plan,
        payload_sha256=plan["source_artifact_sha256"],
        cases=rows,
        files=files,
        metrics={k: v for k, v in metrics.items() if k != "cases"},
        wall_seconds=time.monotonic() - start,
        codec_max_error=getattr(backend, "codec_error", None),
        substep_limit_columns=SUBSTEP_COLUMNS,
        physical_limit_tolerance=0.001,
        physics_dt_s=codec["physics_dt"],
        foot_body_order=backend.feet_names,
        foot_contact_force_scope=(
            "net world-frame force on foot bodies from all contacts, including self-collisions"
        ),
        foot_kinematics_timing=(
            "native derived body fields at policy-step end; one physical substep behind integrated qpos"
        ),
        rollout_differential_policy_steps=[step + 1 for step in ROLLOUT_CHECK_STEPS],
        gpu_peak_bytes=torch.cuda.max_memory_allocated() if plan["backend"] == "mjlab" else 0,
        coverage="MuJoCo family backend transfer; hardware unverified",
    )
