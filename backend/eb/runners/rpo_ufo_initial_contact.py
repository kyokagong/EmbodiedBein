"""Bounded paired RSI probe; no policy updates or acceptance claims."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np

PROTOCOL = {
    "schema_version": "eb/ufo-rsi-contact/v1",
    "envs": 32,
    "rounds": 16,
    "substeps": 4,
    "physics_dt_s": 0.005,
    "seed": 64042,
    "modes": ["hold_reference_target", "default_target"],
    "gym_id": "RPO-UFO-Isaac-RSI-safe-v2",
    "qualification": "diagnostic_only",
    "force_semantics": "net foot vertical force on flat terrain; not isolated contact-pair impulse",
    "pairing": "same kinematic state and physical parameters; solver warm-start equality unverified",
}


def summarize(values):
    return {
        key: {
            "median": float(np.median(value)),
            "p95": float(np.quantile(value, 0.95)),
            "max": float(np.max(value)),
        }
        for key, value in values.items()
    }


def contact_metrics(arrays):
    metrics = []
    for index, name in enumerate(PROTOCOL["modes"]):
        measured = {
            "peak_abs_foot_vertical_N": np.abs(arrays["foot_vertical_N"][index]).max(axis=(1, 3)).reshape(-1),
            "peak_abs_dq_rad_s": np.abs(arrays["joint_vel_rad_s"][index]).max(axis=(1, 3)).reshape(-1),
            "peak_abs_ddq_rad_s2": np.abs(arrays["joint_acc_rad_s2"][index]).max(axis=(1, 3)).reshape(-1),
        }
        metrics.append(
            {
                "mode": name,
                "statistics": summarize(measured),
                "hard_first_control_count": int(arrays["hard_flags"][index].any(axis=(1, 3)).sum()),
            }
        )
    return metrics


def validate_receipt(receipt, validation, output):
    if (
        receipt.get("protocol") != PROTOCOL
        or receipt.get("asset_sha256") != validation.get("asset_sha256")
        or receipt.get("manifest_sha256") != validation.get("manifest_sha256")
        or receipt.get("status") != "measured"
        or receipt.get("samples_per_mode") != 512
    ):
        raise ValueError("RSI contact probe differs from frozen protocol/input")
    path = output / "initial-contact.npz"
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 16 * 1024**2:
        raise ValueError("RSI contact probe raw evidence unavailable or unbounded")
    with path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != receipt.get("raw_sha256"):
            raise ValueError("RSI contact raw evidence changed")
    with np.load(path, allow_pickle=False) as raw:
        for key, shape in {
            "q0": (16, 32, 23),
            "dq0": (16, 32, 23),
            "root0": (16, 32, 13),
            "foot_link_height0_m": (16, 32, 2),
            "decoder_projection_residual_rad": (16, 32, 23),
            "foot_vertical_N": (2, 16, 4, 32, 2),
            "foot_vertical_history_peak_N": (2, 16, 4, 32, 2),
            "joint_vel_rad_s": (2, 16, 4, 32, 23),
            "joint_acc_rad_s2": (2, 16, 4, 32, 23),
            "hard_flags": (2, 16, 4, 32, 3),
        }.items():
            if raw[key].shape != shape or not np.isfinite(raw[key]).all():
                raise ValueError("RSI contact raw shape/nonfinite state")
        observed, expected = receipt.get("metrics") or [], contact_metrics(raw)
        if len(observed) != len(expected):
            raise ValueError("RSI contact summary differs from raw evidence")
        for actual, measured in zip(observed, expected):
            if (
                actual.get("mode") != measured["mode"]
                or actual.get("hard_first_control_count") != measured["hard_first_control_count"]
                or set(actual.get("statistics") or {}) != set(measured["statistics"])
            ):
                raise ValueError("RSI contact summary differs from raw evidence")
            for key, stats in measured["statistics"].items():
                provided = actual["statistics"][key]
                # NumPy 1/2 quantile interpolation differs by float32 rounding.
                if set(provided) != set(stats) or any(
                    type(provided[name]) not in (int, float)
                    or not math.isclose(provided[name], value, rel_tol=1e-6, abs_tol=1e-6)
                    for name, value in stats.items()
                ):
                    raise ValueError("RSI contact summary differs from raw evidence")


def probe(output: Path):
    import os

    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True, enable_cameras=False).app
    env = None
    try:
        import gymnasium as gym
        import torch
        from rpo_bfm import ACTION_GAIN, DEFAULT_Q
        from rpo_bfm_extension import RPOBFMEnvCfg
        from rpo_ufo_isaac import SAFE_GYM_ID, physical_limit_flags, register_task, write_reference_state
        from rpo_ufo_motion import digest

        os.environ["EB_UFO_LEARNING_PROFILE"] = "physical_limits_v2"
        register_task()
        cfg = RPOBFMEnvCfg()
        cfg.seed, cfg.scene.num_envs = PROTOCOL["seed"], PROTOCOL["envs"]
        env = gym.make(SAFE_GYM_ID, cfg=cfg, render_mode=None).unwrapped
        assert env.sim.get_physics_dt() == PROTOCOL["physics_dt_s"]
        ids = torch.arange(env.num_envs, device=env.device)
        records = {
            key: [[], []]
            for key in (
                "foot_vertical_N",
                "foot_vertical_history_peak_N",
                "joint_vel_rad_s",
                "joint_acc_rad_s2",
                "hard_flags",
            )
        }
        q0, dq0, root0, foot0, residual0 = [], [], [], [], []
        for _ in range(PROTOCOL["rounds"]):
            env.reset()
            root = env.robot.data.root_link_state_w.clone()
            root[:, :3] -= env.scene.env_origins
            q, dq = env.robot.data.joint_pos.clone(), env.robot.data.joint_vel.clone()
            q0.append(q.cpu().numpy())
            dq0.append(dq.cpu().numpy())
            root0.append(root.cpu().numpy())
            foot0.append(
                (env.robot.data.body_pos_w[:, env.ufo_mapping["feet"], 2] - env.scene.env_origins[:, 2:3])
                .cpu()
                .numpy()
            )
            default, gain = q.new_tensor(DEFAULT_Q), 0.25 * q.new_tensor(ACTION_GAIN)
            projected = default + gain * ((q - default) / gain).clamp(-1, 1)
            residual0.append((projected - q).cpu().numpy())
            for mode in range(2):
                write_reference_state(env.robot, root, q, dq, ids, env.scene.env_origins)
                env.scene.reset()
                target = q if mode == 0 else env.robot.data.default_joint_pos
                env.robot.set_joint_position_target(target)
                previous = dq.clone()
                samples = {key: [] for key in records}
                for _ in range(PROTOCOL["substeps"]):
                    env.scene.write_data_to_sim()
                    env.sim.step(render=False)
                    env.scene.update(PROTOCOL["physics_dt_s"])
                    data = env.robot.data
                    velocity = data.joint_vel.clone()
                    samples["foot_vertical_N"].append(
                        env.contact_sensor.data.net_forces_w[:, env.ufo_mapping["contact_feet"], 2]
                        .cpu()
                        .numpy()
                    )
                    samples["foot_vertical_history_peak_N"].append(
                        env.contact_sensor.data.net_forces_w_history[:, :, env.ufo_mapping["contact_feet"], 2]
                        .abs()
                        .amax(1)
                        .cpu()
                        .numpy()
                    )
                    samples["joint_vel_rad_s"].append(velocity.cpu().numpy())
                    samples["joint_acc_rad_s2"].append(
                        ((velocity - previous) / PROTOCOL["physics_dt_s"]).cpu().numpy()
                    )
                    samples["hard_flags"].append(
                        physical_limit_flags(data.joint_pos, velocity, data.applied_torque, env.ufo_limits)
                        .cpu()
                        .numpy()
                    )
                    previous = velocity
                for key in records:
                    records[key][mode].append(np.stack(samples[key]))
        arrays = {key: np.stack([np.stack(mode) for mode in value]) for key, value in records.items()}
        arrays.update(
            q0=np.stack(q0),
            dq0=np.stack(dq0),
            root0=np.stack(root0),
            foot_link_height0_m=np.stack(foot0),
            decoder_projection_residual_rad=np.stack(residual0),
        )
        output.mkdir(parents=True, exist_ok=True)
        raw = output / "initial-contact.npz"
        np.savez_compressed(raw, **arrays)
        with raw.open("rb") as stream:
            raw_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        receipt = {
            "protocol": PROTOCOL,
            "status": "measured",
            "samples_per_mode": 512,
            "asset_sha256": os.environ["EB_UFO_ASSET_SHA256"],
            "manifest_sha256": digest(Path(os.environ["EB_UFO_MOTION_ROOT"]) / "manifest.json"),
            "raw_sha256": raw_hash,
            "metrics": contact_metrics(arrays),
        }
        (output / "initial-contact.json").write_text(json.dumps(receipt, indent=2))
    finally:
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    probe(Path("/outputs"))
