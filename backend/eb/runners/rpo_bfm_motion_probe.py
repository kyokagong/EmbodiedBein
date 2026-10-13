"""Real PD replay of every valid interval; never changes the input DatasetVersion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from eb.runners.rpo_bfm_motion import digest, load_motion_view
except ModuleNotFoundError:
    from rpo_bfm_motion import digest, load_motion_view


RECIPE = {
    "id": "rpo-bfm-pd-replay-v1",
    "seeds": [42, 43, 44],
    "minimum_frames": 9,
    "joint_rmse_rad_max": 0.15,
    "root_position_rmse_m_max": 0.15,
    "root_angle_rmse_rad_max": 0.25,
    "contact_foot_speed_rms_mps_max": 0.30,
    "hard_limit_tolerance_rad": 1e-4,
    "contact_force_threshold_n": 1.0,
}


def continuous_intervals(valid):
    edges = np.diff(np.r_[False, valid, False].astype(np.int8))
    return [
        (int(a), int(b))
        for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))
        if b - a >= RECIPE["minimum_frames"]
    ]


def tracking_pass(item):
    return (
        item["finite"]
        and not item["ended_early"]
        and item["steps"] == item["available_steps"]
        and item["hard_limit_frames"] == 0
        and item["contact_samples"] > 0
        and all(
            np.isfinite(item[key]) and 0 <= item[key] <= RECIPE[key + "_max"]
            for key in (
                "joint_rmse_rad",
                "root_position_rmse_m",
                "root_angle_rmse_rad",
                "contact_foot_speed_rms_mps",
            )
        )
    )


def finite_case(item):
    """Preserve a physics failure without invalid JSON numeric values."""
    result = dict(item)
    for key, value in result.items():
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            result[key] = None
            result["finite"] = False
    return result


def tracking_validation_commands(reference, asset, output, weights, weights_sha):
    import sys

    return [
        [
            sys.executable,
            str(Path(__file__)),
            "--retarget-dir",
            str(reference),
            "--asset-manifest",
            str(asset),
            "--output",
            str(output.parent / f"seed_{seed}/probe.json"),
            "--teacher-weights",
            str(weights),
            "--teacher-sha256",
            weights_sha,
            "--validation-seed",
            str(seed),
            "--video",
        ]
        for seed in [142, 143, 144]
    ]


def evaluate_teacher(reference, asset, output, weights, weights_sha):
    import subprocess

    rows, samples, videos, startups = [], [], [], []
    receipts = []
    for command in tracking_validation_commands(reference, asset, output, weights, weights_sha):
        subprocess.run(command, check=True)
        child_path = Path(command[command.index("--output") + 1])
        receipt = json.loads(child_path.read_text())
        seed = int(command[command.index("--validation-seed") + 1])
        if (
            receipt.get("startup_seed") != seed
            or len(receipt["cases"]) != 4
            or receipt["teacher_weights_sha256"] != weights_sha
            or any(case["seed"] != seed for case in receipt["cases"])
        ):
            raise ValueError("Teacher validation startup/case/weight binding mismatch")
        trajectory = child_path.parent / receipt["trajectory"]["file"]
        if digest(trajectory) != receipt["trajectory"]["sha256"]:
            raise ValueError("Teacher child trajectory changed")
        with np.load(trajectory, allow_pickle=False) as archive:
            values = archive["samples"].copy()
        values[:, 0] += len(rows)
        samples.append(values)
        rows.extend(receipt["cases"])
        videos.extend({**video, "file": f"seed_{seed}/{video['file']}"} for video in receipt["videos"])
        receipts.append(receipt)
        startup = receipt["startup"]
        startup_path = child_path.parent / startup["file"]
        if digest(startup_path) != startup["sha256"] or json.loads(startup_path.read_text())["seed"] != seed:
            raise ValueError("Teacher sampled physics witness changed")
        startups.append({**startup, "seed": seed, "file": f"seed_{seed}/{startup['file']}"})
    if any(
        r["manifest_sha256"] != receipts[0]["manifest_sha256"]
        or r["asset_sha256"] != receipts[0]["asset_sha256"]
        for r in receipts
    ):
        raise ValueError("Teacher validation reference/asset differ between seeds")
    trajectory = output.parent / "replay-trajectory.npz"
    np.savez_compressed(trajectory, samples=np.concatenate(samples))
    result = {
        **receipts[0],
        "recipe": {**RECIPE, "id": "rpo-tracking-teacher-validation-v1", "seeds": [142, 143, 144]},
        "startup_seed": None,
        "startup_seeds": [142, 143, 144],
        "startup_randomization": "fresh_process_per_seed",
        "cases": rows,
        "passed": sum(r["passed"] for r in rows),
        "total": len(rows),
        "intervals_checked": 4,
        "dynamic_replay_verified": all(r["passed"] for r in rows),
        "videos": videos,
        "startups": startups,
        "trajectory": {**receipts[0]["trajectory"], "sha256": digest(trajectory)},
    }
    result.pop("startup", None)
    output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "total": result["total"],
                "startup_seeds": result["startup_seeds"],
                "training_ready": False,
            }
        ),
        flush=True,
    )
    return result


def retarget_reference(root, asset_sha):
    from scipy.spatial.transform import Rotation

    path = root / "retarget-receipt.json"
    manifest = json.loads(path.read_text())
    if (
        manifest.get("schema_version") != "eb/100style-rpo-retarget-trial/v1"
        or manifest.get("asset_sha256") != asset_sha
        or manifest.get("training_ready") is not False
        or len(manifest.get("clips", [])) != 4
    ):
        raise ValueError("100STYLE retarget receipt does not bind the frozen RPO asset")
    arrays = {}
    for clip in manifest["clips"]:
        file = root / clip["file"]
        if file.parent != root or file.is_symlink() or digest(file) != clip["sha256"]:
            raise ValueError("100STYLE retarget NPZ changed after mapping")
        with np.load(file, allow_pickle=False) as archive:
            data = {key: archive[key].copy() for key in ("q", "root_pos", "root_quat", "mujoco_body_pos")}
            if archive["fps"] != 50:
                raise ValueError("100STYLE retarget frequency mismatch")
        count = clip["frames"]
        for key, shape in (
            ("q", (count, 23)),
            ("root_pos", (count, 3)),
            ("root_quat", (count, 4)),
            ("mujoco_body_pos", (count, 10, 3)),
        ):
            if data[key].shape != shape or not np.isfinite(data[key]).all() or count != 150:
                raise ValueError("100STYLE retarget shape/finite/budget mismatch")
        data["valid"] = np.ones(count, dtype=bool)
        data["dq"] = np.gradient(data["q"], 0.02, axis=0)
        data["body_lin_vel"] = np.zeros((count, 24, 3), dtype=np.float32)
        data["body_lin_vel"][:, 0] = np.gradient(data["root_pos"], 0.02, axis=0)
        data["body_ang_vel"] = np.zeros_like(data["body_lin_vel"])
        rotation = Rotation.from_quat(data["root_quat"][:, [1, 2, 3, 0]])
        omega = (rotation[1:] * rotation[:-1].inv()).as_rotvec() / 0.02
        data["body_ang_vel"][:, 0] = np.r_[omega, omega[-1:]]
        clip["id"] = Path(clip["file"]).stem
        arrays[clip["id"]] = data
    manifest["source_rights"] = "CC-BY-4.0; RPO quality unqualified"
    return manifest, arrays


def run(
    motion_dir,
    asset_manifest,
    output,
    *,
    retarget_dir=None,
    teacher_weights=None,
    teacher_sha=None,
    video=False,
    validation_seed=None,
):
    import os

    if teacher_weights:
        if retarget_dir is None or digest(teacher_weights) != teacher_sha:
            raise ValueError("Tracking teacher weights are not hash-bound")
        os.environ.update(
            {
                "EB_TRACKING_REFERENCE_ROOT": str(retarget_dir),
                "EB_TRACKING_REFERENCE_SHA256": digest(retarget_dir / "retarget-receipt.json"),
                "EB_TRACKING_ASSET_SHA256": digest(asset_manifest),
            }
        )
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True, enable_cameras=video).app
    env = None
    try:
        import gymnasium as gym
        import torch
        from rpo_bfm import ACTION_GAIN, CODEC_HASH, DEFAULT_Q
        from rpo_bfm_extension import GYM_ID, RPOBFMEnvCfg, register_task

        asset_sha = digest(asset_manifest)
        if retarget_dir is None:
            load_motion_view(motion_dir, asset_sha256=asset_sha)
            manifest_path = motion_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
        else:
            manifest_path = retarget_dir / "retarget-receipt.json"
            manifest, trial_arrays = retarget_reference(retarget_dir, asset_sha)
        if teacher_weights:
            from omegaconf import OmegaConf
            from rpo_tracking_extension import GYM_ID, register_task
            from rpo_tracking_extension import RPOTrackingEnvCfg as RPOBFMEnvCfg
            from rpo_tracking_policy import RPOTrackingPolicy

            teacher = RPOTrackingPolicy(
                OmegaConf.create(
                    dict(
                        obs_dim=547,
                        critic_obs_dim=905,
                        action_dim=23,
                        num_action_chunks=1,
                        add_value_head=True,
                    )
                )
            )
            teacher.load_state_dict(
                torch.load(teacher_weights, map_location="cpu", weights_only=True), strict=True
            )
            teacher = teacher.to("cuda").eval()
        register_task()
        cfg = RPOBFMEnvCfg()
        env_count = 1 if teacher_weights else 3
        cfg.scene.num_envs = env_count
        if teacher_weights and validation_seed not in [142, 143, 144]:
            raise ValueError("Teacher validation needs a held-out startup seed")
        cfg.seed = validation_seed if teacher_weights else 42
        cfg.episode_length_s = 120
        if video:
            from rpo_flat_artifact import _add_camera

            _add_camera(cfg)
        env = gym.make(GYM_ID, cfg=cfg, render_mode=None).unwrapped
        robot = env.robot
        if teacher_weights:
            parameters = {
                "masses": robot.root_physx_view.get_masses(),
                "coms": robot.root_physx_view.get_coms(),
                "materials": robot.root_physx_view.get_material_properties(),
            }
            parameters.update(
                {
                    key: getattr(robot.data, key)
                    for key in ("joint_stiffness", "joint_damping", "joint_armature", "joint_friction_coeff")
                }
            )
            startup_path = output.parent / "startup.json"
            startup_path.parent.mkdir(parents=True, exist_ok=True)
            startup_path.write_text(
                json.dumps(
                    {
                        "seed": cfg.seed,
                        "parameters": {key: value.cpu().tolist() for key, value in parameters.items()},
                    },
                    allow_nan=False,
                )
            )
        defaults = torch.tensor(DEFAULT_Q, device=env.device)
        gains = 0.25 * torch.tensor(ACTION_GAIN, device=env.device)
        limits = robot.data.joint_pos_limits[0]
        foot_names = [env.contact_sensor.body_names[i] for i in env.feet_cfg.body_ids]
        feet = [robot.body_names.index(name) for name in foot_names]
        rows, trajectories, excluded, videos = [], [], [], []
        seeds = [validation_seed] if teacher_weights else RECIPE["seeds"]
        for clip in manifest["clips"]:
            if retarget_dir is None:
                with np.load(motion_dir / clip["file"], allow_pickle=False) as archive:
                    data = {
                        name: archive[name].copy()
                        for name in (
                            "valid",
                            "q",
                            "dq",
                            "root_pos",
                            "root_quat",
                            "body_lin_vel",
                            "body_ang_vel",
                        )
                    }
            else:
                data = trial_arrays[clip["id"]]
            intervals = continuous_intervals(data["valid"])
            if not intervals:
                excluded.append({"clip": clip["id"], "reason": "no_continuous_9_frame_interval"})
            for begin, end in intervals:
                # Reset and initialize once; never teleport during replay.
                for i, seed in enumerate(seeds):
                    torch.manual_seed(seed)
                    env.reset(env_ids=torch.tensor([i], device=env.device))
                root = robot.data.default_root_state.clone()
                root[:, :3] = env.scene.env_origins + torch.from_numpy(data["root_pos"][begin]).to(env.device)
                root[:, 3:7] = torch.from_numpy(data["root_quat"][begin]).to(env.device)
                root[:, 7:10] = torch.from_numpy(data["body_lin_vel"][begin, 0]).to(env.device)
                root[:, 10:13] = torch.from_numpy(data["body_ang_vel"][begin, 0]).to(env.device)
                q = torch.from_numpy(data["q"][begin]).to(env.device).expand(env_count, -1).clone()
                dq = torch.from_numpy(data["dq"][begin]).to(env.device).expand(env_count, -1).clone()
                robot.write_root_state_to_sim(root)
                robot.write_joint_state_to_sim(q, dq)
                env.scene.write_data_to_sim()
                env.sim.forward()
                env.scene.update(env.step_dt)
                if teacher_weights:
                    index = env.reference_ids.index(clip["id"])
                    env.reference_clip[:] = index
                    env.reference_frame[:] = begin
                    env.bfm_history.reset(torch.arange(env_count, device=env.device))
                writer = None
                if video:
                    import imageio.v2 as imageio

                    video_path = output.parent / f"{clip['id']}-seed-{seeds[0]}.mp4"
                    video_path.parent.mkdir(parents=True, exist_ok=True)
                    writer = imageio.get_writer(video_path, fps=50, codec="libx264", macro_block_size=1)
                    from rpo_flat_artifact import _video_frame

                    writer.append_data(_video_frame(env))
                fk_error = None
                if retarget_dir is not None:
                    body_ids = [robot.body_names.index(name) for name in manifest["body_map"].values()]
                    actual = (robot.data.body_pos_w[0, body_ids] - env.scene.env_origins[0]).cpu().numpy()
                    fk_error = float(np.linalg.norm(actual - data["mujoco_body_pos"][begin], axis=-1).max())
                    if fk_error > 0.03:
                        raise ValueError(f"RPO retarget MJCF/Isaac FK mismatch: {fk_error} m")
                active = torch.ones(env_count, dtype=torch.bool, device=env.device)
                records = [[] for _ in range(env_count)]
                for frame in range(begin + 1, end):
                    target = torch.from_numpy(data["q"][frame]).to(env.device)
                    action = ((target - defaults) / gains).expand(env_count, -1).clone()
                    if not bool((action.abs() <= 1 + 1e-6).all()):
                        raise ValueError("BFM valid interval requires an unreachable PD target")
                    if teacher_weights:
                        obs = env._get_observations()
                        policy_obs = torch.where(
                            active[:, None], obs["policy"], torch.zeros_like(obs["policy"])
                        )
                        critic_obs = torch.where(
                            active[:, None], obs["critic"], torch.zeros_like(obs["critic"])
                        )
                        with torch.no_grad():
                            action = teacher.predict_action_batch(
                                {"states": policy_obs, "critic_states": critic_obs}, mode="eval"
                            )[0][:, 0]
                    _, _, terminated, truncated, _ = env.step(action)
                    if writer is not None and bool(torch.isfinite(robot.data.root_state_w[0]).all()):
                        camera = env.scene["eval_camera"]
                        position = robot.data.root_pos_w[:1]
                        camera.set_world_poses_from_view(
                            position + torch.tensor([[3.0, 3.0, 1.5]], device=env.device), position
                        )
                        env.sim.render()
                        writer.append_data(
                            camera.data.output["rgb"][0, :, :, :3].cpu().numpy().astype(np.uint8)
                        )
                    actual_pos = robot.data.root_pos_w - env.scene.env_origins
                    actual_quat = robot.data.root_quat_w
                    reference_pos = torch.from_numpy(data["root_pos"][frame]).to(env.device)
                    reference_quat = torch.from_numpy(data["root_quat"][frame]).to(env.device)
                    angle = 2 * torch.acos((actual_quat * reference_quat).sum(-1).abs().clamp(0, 1))
                    contacts = (
                        torch.linalg.vector_norm(
                            env.contact_sensor.data.net_forces_w[:, env.feet_cfg.body_ids], dim=-1
                        )
                        > 1
                    )
                    foot_speed = torch.linalg.vector_norm(robot.data.body_link_lin_vel_w[:, feet, :2], dim=-1)
                    finite = torch.isfinite(robot.data.joint_pos).all(-1) & torch.isfinite(
                        robot.data.root_state_w
                    ).all(-1)
                    violation = (
                        (robot.data.joint_pos < limits[:, 0] - 1e-4)
                        | (robot.data.joint_pos > limits[:, 1] + 1e-4)
                    ).any(-1)
                    terminate_force = torch.linalg.vector_norm(
                        env.contact_sensor.data.net_forces_w_history[
                            :, :, env.termination_contact_cfg.body_ids
                        ],
                        dim=-1,
                    ).amax(dim=(1, 2))
                    for i in active.nonzero(as_tuple=False).flatten().tolist():
                        records[i].append(
                            {
                                "frame": frame,
                                "joint_mse": float((robot.data.joint_pos[i] - target).square().mean()),
                                "root_position_sq": float((actual_pos[i] - reference_pos).square().sum()),
                                "root_angle_sq": float(angle[i].square()),
                                "contact_speed_sq": float(foot_speed[i][contacts[i]].square().sum()),
                                "contact_samples": int(contacts[i].sum()),
                                "finite": bool(finite[i]),
                                "hard_limit": bool(violation[i]),
                                "height": float(actual_pos[i, 2]),
                                "upright": float(-robot.data.projected_gravity_b[i, 2]),
                                "ended": bool(terminated[i] or truncated[i]),
                                "terminated": bool(terminated[i]),
                                "truncated": bool(truncated[i]),
                                "termination_contact_force_n": float(terminate_force[i]),
                            }
                        )
                        trajectories.append(
                            [
                                len(rows) + i,
                                frame,
                                *robot.data.joint_pos[i].cpu().tolist(),
                                *actual_pos[i].cpu().tolist(),
                                *actual_quat[i].cpu().tolist(),
                                float(foot_speed[i].max()),
                            ]
                        )
                    active &= ~(terminated | truncated) & finite
                    if not bool(active.any()):
                        break
                for i, records_i in enumerate(records):
                    contacts = sum(r["contact_samples"] for r in records_i)
                    item = {
                        "clip": clip["id"],
                        "begin": begin,
                        "end": end,
                        "seed": seeds[i],
                        "steps": len(records_i),
                        "available_steps": end - begin - 1,
                        "finite": all(r["finite"] for r in records_i),
                        "hard_limit_frames": sum(r["hard_limit"] for r in records_i),
                        "joint_rmse_rad": float(np.sqrt(np.mean([r["joint_mse"] for r in records_i]))),
                        "root_position_rmse_m": float(
                            np.sqrt(np.mean([r["root_position_sq"] for r in records_i]))
                        ),
                        "root_angle_rmse_rad": float(
                            np.sqrt(np.mean([r["root_angle_sq"] for r in records_i]))
                        ),
                        "contact_foot_speed_rms_mps": float(
                            np.sqrt(sum(r["contact_speed_sq"] for r in records_i) / max(1, contacts))
                        ),
                        "contact_samples": contacts,
                        "ended_early": len(records_i) < end - begin - 1
                        or records_i[-1]["terminated"]
                        or (records_i[-1]["truncated"] and not teacher_weights),
                        "minimum_height_m": min(r["height"] for r in records_i),
                        "minimum_upright": min(r["upright"] for r in records_i),
                    }
                    item = finite_case(item)
                    item["passed"] = tracking_pass(item)
                    item["initial_fk_max_error_m"] = fk_error
                    last = records_i[-1]
                    item["end_reason"] = (
                        "nonfinite"
                        if not item["finite"]
                        else "low_base"
                        if last["terminated"] and last["height"] < 0.5
                        else "excessive_tilt"
                        if last["terminated"] and last["upright"] < np.cos(0.8)
                        else "nonfoot_contact"
                        if last["terminated"] and last["termination_contact_force_n"] > 1
                        else "timeout"
                        if last["truncated"]
                        else "other_termination"
                        if last["terminated"]
                        else "complete"
                    )
                    item["maximum_termination_contact_force_n"] = max(
                        r["termination_contact_force_n"] for r in records_i
                    )
                    rows.append(item)
                if writer is not None:
                    writer.close()
                    videos.append({"file": video_path.name, "sha256": digest(video_path)})
                print(
                    json.dumps(
                        {
                            "clip": clip["id"],
                            "begin": begin,
                            "end": end,
                            "passed": sum(r["passed"] for r in rows[-env_count:]),
                        }
                    ),
                    flush=True,
                )
        output.parent.mkdir(parents=True, exist_ok=True)
        trajectory = output.with_name("replay-trajectory.npz")
        np.savez_compressed(trajectory, samples=np.asarray(trajectories, dtype=np.float64))
        receipt = {
            "schema_version": "eb/rpo-bfm-motion-probe/v1",
            "status": "succeeded",
            "recipe": {**RECIPE, "id": "rpo-tracking-teacher-validation-v1", "seeds": seeds}
            if teacher_weights
            else RECIPE,
            "manifest_sha256": digest(manifest_path),
            "asset_sha256": asset_sha,
            "codec_sha256": CODEC_HASH,
            "clips_checked": len(manifest["clips"]),
            "intervals_checked": len(rows) // env_count,
            "cases": rows,
            "excluded": excluded,
            "passed": sum(r["passed"] for r in rows),
            "total": len(rows),
            "dynamic_replay_verified": bool(rows) and all(r["passed"] for r in rows),
            "training_ready": False,
            "source_rights": manifest["source_rights"],
            **(
                {
                    "teacher_weights_sha256": teacher_sha,
                    "videos": videos,
                    "startup_seed": validation_seed,
                    "startup": {"file": "startup.json", "sha256": digest(startup_path)},
                    "startup_randomization": "fresh_process_per_seed",
                }
                if teacher_weights
                else {}
            ),
            "trajectory": {
                "file": trajectory.name,
                "sha256": digest(trajectory),
                "columns": [
                    "case_index",
                    "reference_frame",
                    *robot.joint_names,
                    "root_x",
                    "root_y",
                    "root_z",
                    "root_qw",
                    "root_qx",
                    "root_qy",
                    "root_qz",
                    "max_foot_speed_mps",
                ],
            },
        }
        output.write_text(json.dumps(receipt, indent=2, allow_nan=False))
        print(
            json.dumps(
                {key: value for key, value in receipt.items() if key not in {"cases", "recipe", "trajectory"}}
            ),
            flush=True,
        )
        return receipt
    finally:
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--motion-dir", type=Path)
    sources.add_argument("--retarget-dir", type=Path)
    parser.add_argument("--asset-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--teacher-weights", type=Path)
    parser.add_argument("--teacher-sha256")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--validation-seed", type=int)
    args = parser.parse_args()
    if args.teacher_weights and args.validation_seed is None:
        evaluate_teacher(
            args.retarget_dir, args.asset_manifest, args.output, args.teacher_weights, args.teacher_sha256
        )
    else:
        run(
            args.motion_dir,
            args.asset_manifest,
            args.output,
            retarget_dir=args.retarget_dir,
            teacher_weights=args.teacher_weights,
            teacher_sha=args.teacher_sha256,
            video=args.video,
            validation_seed=args.validation_seed,
        )
