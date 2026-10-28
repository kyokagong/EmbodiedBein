"""Independent ONNX closed-loop RPO terrain evaluation and real media capture."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from eb.pbfm_evaluation import route_outcome, validate_case_rows, validate_eval_plan
from eb.pbfm_io import file_index, read_json, safe_file, sha256, write_json
from eb.runners.pbfm_export import commanded_observation, onnx_action, parity_session, prepare_runtime


def terrain_height(renderer, position) -> float:
    """Measure actual static terrain, never infer stair direction from its name."""
    import mujoco
    import numpy as np

    if not getattr(renderer, "_eb_static_terrain_geoms", None):
        model = renderer.model
        # Training-mix primitives and the floating robot both use group 0.
        # Isolate static collision geometry in this render/ray copy by its
        # welded world identity; physics and camera inputs are unchanged.
        static = (model.body_weldid[model.geom_bodyid] == 0) & (
            (model.geom_contype != 0) | (model.geom_conaffinity != 0)
        )
        selected = np.flatnonzero(static)
        if not len(selected):
            raise ValueError("scene has no known static collision terrain for independent geometry rays")
        model.geom_group[(model.geom_group == 5) & ~static] = 0
        model.geom_group[selected] = 5
        renderer._eb_static_terrain_geoms = selected.tolist()

    origin = np.asarray(position, dtype=np.float64).copy()
    origin[2] += 5.0
    groups = np.zeros(6, dtype=np.uint8)
    groups[5] = 1
    hit = np.asarray([-1], dtype=np.int32)
    distance = mujoco.mj_ray(
        renderer.model, renderer.data, origin, np.asarray([0.0, 0.0, -1.0]), groups, 1, -1, hit
    )
    if distance < 0 or hit[0] < 0:
        raise ValueError("terrain geometry ray missed; route evidence is incomplete")
    return float(origin[2] - distance)


def foot_geometry(model, names: list[str]) -> dict:
    """Derive support points from frozen RPO collision geoms, including meshes."""
    import mujoco
    import numpy as np

    result = {}
    for name in names:
        matches = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, candidate)
            for candidate in (name, "robot/" + name)
        ]
        matches = [body for body in matches if body >= 0]
        if len(matches) != 1:
            raise ValueError("frozen RPO foot body is missing")
        body = matches[0]
        points = []
        for geom in np.flatnonzero(model.geom_bodyid == body):
            if not (model.geom_contype[geom] or model.geom_conaffinity[geom]):
                continue
            kind = model.geom_type[geom]
            if kind == mujoco.mjtGeom.mjGEOM_MESH:
                mesh = model.geom_dataid[geom]
                start = model.mesh_vertadr[mesh]
                size = model.mesh_vertnum[mesh]
                local = model.mesh_vert[start : start + size].astype(np.float64)
            elif kind == mujoco.mjtGeom.mjGEOM_BOX:
                local = (
                    np.asarray([[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)])
                    * model.geom_size[geom]
                )
            elif kind in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_SPHERE):
                radius, half_length = model.geom_size[geom, :2]
                local = np.asarray(
                    [
                        [radius, 0, -half_length],
                        [-radius, 0, -half_length],
                        [0, 0, -half_length - radius],
                        [radius, 0, half_length],
                        [-radius, 0, half_length],
                        [0, 0, half_length + radius],
                    ]
                )
            else:
                raise ValueError("unreviewed RPO foot collision geometry")
            matrix = np.zeros(9)
            mujoco.mju_quat2Mat(matrix, model.geom_quat[geom])
            points.extend(local @ matrix.reshape(3, 3).T + model.geom_pos[geom])
        points = np.asarray(points)
        if points.ndim != 2 or len(points) < 2:
            raise ValueError("RPO foot has no physical support geometry")
        bottom = points[points[:, 2] <= points[:, 2].min() + 0.01]
        result[name] = {
            "body_id": int(body),
            "toe_local": bottom[bottom[:, 0].argmax()].tolist(),
            "heel_local": bottom[bottom[:, 0].argmin()].tolist(),
            "geometry": "native_collision_support",
        }
    return result


def advance_with_successor(wrapped, action):
    """Capture physics before the native vector wrapper resets terminated rows."""
    import numpy as np

    core = wrapped._env
    original_step = core.step
    snapshot = {}

    def array(value):
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value).copy()

    def capture(actions):
        result = original_step(actions)
        qpos, qvel = wrapped._get_qpos_qvel(to_numpy=False)
        snapshot.update(
            qpos=array(qpos)[0],
            qvel=array(qvel)[0],
            root=array(core.robot_root_states)[0],
            torque=array(core.torques)[0],
            dof_pos=array(core.dof_pos)[0],
            gravity=array(core.projected_gravity)[0],
        )
        for name in ("body_pos", "body_vel", "contact_forces", "applied_actions", "default_dof_pos_offset"):
            if hasattr(core, name):
                snapshot[name] = array(getattr(core, name))[0]
        if "default_dof_pos_offset" in snapshot:
            snapshot["default_offset"] = snapshot["default_dof_pos_offset"]
        if hasattr(core, "mjlab_env"):
            term = core.mjlab_env.action_manager.get_term("actions")
            snapshot["pd_target"] = array(
                term._processed_actions - term._entity.data.encoder_bias[:, term.target_ids]
            )
            snapshot["pd_target_names"] = list(term.target_names)
            snapshot["encoder_bias_target"] = array(term._entity.data.encoder_bias[:, term.target_ids])
        return result

    core.step = capture
    try:
        result = wrapped.step(action, to_numpy=False)
    finally:
        core.step = original_step
    info = result[-1]
    if "final_qpos" in info:
        final = array(info["final_qpos"])
        if final.shape != (1, 30) or not np.allclose(final[0], snapshot["qpos"], atol=1e-7):
            raise ValueError("native terminal successor differs from captured pre-reset physics")
        snapshot["qpos"] = final[0]
        snapshot["qvel"] = array(info["final_qvel"])[0]
    return result, snapshot


def feet_world(renderer, geometry: dict) -> list:
    import numpy as np

    return [
        [
            (
                renderer.data.xpos[value["body_id"]]
                + renderer.data.xmat[value["body_id"]].reshape(3, 3) @ np.asarray(value[key])
            ).tolist()
            for key in ("toe_local", "heel_local")
        ]
        for value in geometry.values()
    ]


def diagnostic_metrics(
    joints,
    bodies,
    body_velocities,
    contact_forces,
    foot_indices,
    root_velocities,
    command,
    actions,
    clearances,
    reference_joints,
    reference_bodies,
):
    import numpy as np

    if not all(
        np.isfinite(value).all()
        for value in (
            joints,
            bodies,
            body_velocities,
            contact_forces,
            root_velocities,
            actions,
            clearances,
            reference_joints,
            reference_bodies,
        )
    ):
        raise ValueError("independent physical diagnostics contain nonfinite raw evidence")
    body_relative = bodies - bodies[:, :1]
    force = np.linalg.norm(contact_forces[:, foot_indices], axis=-1)
    stance = force >= 5.0
    horizontal_speed = np.linalg.norm(body_velocities[:, foot_indices, :2], axis=-1)
    slip = float(horizontal_speed[stance].mean()) if stance.any() else None
    return {
        "joint_tracking_rmse_rad": float(np.sqrt(np.mean((joints - reference_joints) ** 2))),
        "body_tracking_rmse_m": float(
            np.sqrt(np.mean(np.sum((body_relative - reference_bodies) ** 2, axis=-1)))
        ),
        "velocity_rmse_m_s": float(
            np.sqrt(np.mean(np.sum((root_velocities - np.asarray(command)) ** 2, axis=-1)))
        ),
        "foot_slip_m_s": slip,
        "stance_samples": int(stance.sum()),
        "foot_slip_status": "measured" if stance.any() else "not_observed",
        "max_contact_force_n": float(force.max()),
        "min_toe_clearance_m": float(clearances[..., 0].min()),
        "min_heel_clearance_m": float(clearances[..., 1].min()),
        "action_saturation_fraction": float(np.mean(np.abs(actions * 5.0) >= 5.0)),
    }


def forbidden_contact_summary(forces, foot_indices):
    import numpy as np

    forbidden = np.ones(forces.shape[1], dtype=bool)
    forbidden[foot_indices] = False
    norms = np.linalg.norm(forces[:, forbidden], axis=-1)
    return {
        "forbidden_contact_count": int((norms >= 5.0).sum()),
        "forbidden_contact_peak_n": float(norms.max()) if norms.size else 0.0,
    }


def reset_target(prompt: dict, core):
    import mujoco
    import numpy as np
    import torch

    root = np.asarray(prompt["reset_root_states_xyzw"]).copy()
    # Remove reference yaw while retaining source roll/pitch, then face the +X route.
    quat = root[0, 3:7][[3, 0, 1, 2]].astype(np.float64)
    heading = np.asarray(prompt["initial_heading_world_xy"])[0]
    yaw = math.atan2(float(heading[1]), float(heading[0]))
    inverse = np.asarray([math.cos(-yaw / 2), 0.0, 0.0, math.sin(-yaw / 2)])
    rotated = np.zeros(4)
    mujoco.mju_mulQuat(rotated, inverse, quat)
    root[0, 3:7] = rotated[[1, 2, 3, 0]]
    origin = core.env_origins[0].detach().cpu().numpy()
    root[0, :2] = origin[:2]
    root[0, 2] += origin[2]
    root[0, 7:] = 0.0
    dofs = np.asarray(prompt["reset_dof_states"]).copy()
    dofs[..., 1] = 0
    return {
        "root_states": torch.as_tensor(root, device=core.device),
        "dof_states": torch.as_tensor(dofs, device=core.device),
    }


def evaluate_case(plan, case, condition, config, robot, prompt, session, output: Path):
    import mediapy as media
    import numpy as np
    import torch
    from pbfm.inference.runtime import (
        MujocoQposRenderer,
        _assign_training_tile,
        _terrain_env_cfg,
        policy_qpos_from_env,
    )

    torch.manual_seed(case["seed"])
    np.random.seed(case["seed"])
    terrain_config = _terrain_env_cfg(config, case["family"], seed=case["seed"])
    wrapped = terrain_config.build(num_envs=1)[0]
    renderer = None
    try:
        core = wrapped._env
        _assign_training_tile(core, family=case["family"], difficulty_row=0)
        core._motion_lib.load_all_motions()
        core.is_evaluating = True
        renderer = MujocoQposRenderer(
            None,
            render_size=360,
            scene_spec=core.mjlab_env.scene.spec,
            source_xml_path=Path(robot.robot.xml_path),
            add_floor=False,
            expected_qpos_size=30,
        )
        geometry = foot_geometry(renderer.model, plan["robot_geometry"]["foot_bodies"])
        prompt = {
            key: value.copy() if isinstance(value, np.ndarray) else value for key, value in prompt.items()
        }
        reference_velocity = prompt["reference_velocity_world_xy"].copy()
        prompt["reference_velocity_world_xy"][:] = case["commands"]["velocity_world_xy"]
        prompt["heading_world_xy"][:] = case["commands"]["heading_world_xy"]
        prompt["velocity_valid"][:] = case["commands"]["valid"]
        prompt["heading_valid"][:] = case["commands"]["valid"]
        target = reset_target(prompt, core)
        target["root_states"][0, 7:9] = torch.as_tensor(
            case["initial_root_velocity_world_xy"], device=core.device
        )
        if case["regression"]:
            prompt["z"][:] = prompt["z"][0]
            prompt["reference_dof_pos"][:] = prompt["reference_dof_pos"][0]
            prompt["reference_body_pos_heading0"][:] = prompt["reference_body_pos_heading0"][0]
        observation, _ = wrapped.reset(to_numpy=False, target_states=target)
        qpos = policy_qpos_from_env(wrapped, expected_qpos_size=30)
        frames = [renderer.render_qpos(qpos)]
        rgb_times = [0.0]
        roots = [core.robot_root_states[0, :3].detach().cpu().numpy().copy()]
        heights = [terrain_height(renderer, roots[-1])]
        foot_trace = [feet_world(renderer, geometry)]
        foot_clearances = [
            [[point[2] - terrain_height(renderer, point) for point in foot] for foot in foot_trace[-1]]
        ]
        joint_positions, body_positions, body_velocities, contact_forces, root_velocities = [], [], [], [], []
        depths, presented, actions, tilts, torque_ratios, joint_excess = [], [], [], [], [], []
        terminal, goal_reached = False, False
        for step in range(1000):
            observation = commanded_observation(observation, core, prompt, step)
            depth = observation["depth_image"].detach().cpu().numpy().copy()
            if depth.shape != (1, 8, 18, 32) or not np.isfinite(depth).all():
                raise ValueError("native depth input has wrong shape or nonfinite values")
            depths.append(depth[0])
            if condition == "occluded":
                observation = dict(observation)
                observation["depth_image"] = torch.full_like(
                    observation["depth_image"],
                    255 if observation["depth_image"].dtype == torch.uint8 else 1.0,
                )
            presented.append(observation["depth_image"][0].detach().cpu().numpy().copy())
            z = torch.as_tensor(prompt["z"][step : step + 1], device=core.device)
            action = onnx_action(session, observation, z)
            result, successor = advance_with_successor(wrapped, torch.as_tensor(action, device=core.device))
            observation, _, terminated, truncated, _ = result
            actions.append(action[0])
            root = successor["root"][:3]
            roots.append(root)
            qpos = successor["qpos"]
            rgb = renderer.render_qpos(qpos)
            heights.append(terrain_height(renderer, root))
            foot_trace.append(feet_world(renderer, geometry))
            foot_clearances.append(
                [[point[2] - terrain_height(renderer, point) for point in foot] for foot in foot_trace[-1]]
            )
            joint_positions.append(successor["dof_pos"])
            body_positions.append(successor["body_pos"])
            body_velocities.append(successor["body_vel"])
            contact_forces.append(successor["contact_forces"])
            root_velocities.append(successor["root"][7:9])
            gravity = successor["gravity"]
            tilts.append(float(np.arccos(np.clip(-gravity[2], -1, 1))))
            limit = core.torque_limits
            torque_ratios.append(
                float(np.max(np.abs(successor["torque"]) / np.clip(limit.detach().cpu().numpy(), 1e-6, None)))
            )
            bounds = core.hard_dof_pos_limits
            joint_excess.append(
                float(
                    torch.maximum(
                        bounds[:, 0] - torch.as_tensor(successor["dof_pos"], device=core.device),
                        torch.as_tensor(successor["dof_pos"], device=core.device) - bounds[:, 1],
                    )
                    .clamp_min(0)
                    .max()
                    .item()
                )
            )
            if step % 2 == 1:
                frames.append(rgb)
                rgb_times.append((step + 1) * 0.02)
            terminal = bool(torch.as_tensor(terminated).any()) or bool(torch.as_tensor(truncated).any())
            goal_reached = bool(
                not case["regression"] and root[0] - roots[0][0] >= plan["suite_spec"]["route_distance_m"]
            )
            if terminal or goal_reached:
                if step % 2 == 0:
                    frames.append(rgb)
                    rgb_times.append((step + 1) * 0.02)
                break
        root_array = np.asarray(roots)
        forbidden_contacts = forbidden_contact_summary(
            np.asarray(contact_forces), core.feet_indices.detach().cpu().numpy()
        )
        constraints = bool(
            max(tilts) <= 0.8
            and min(root_array[:, 2] - np.asarray(heights)) >= 0.4
            and max(torque_ratios) <= 1.001
            and max(joint_excess) <= 0.001
            and np.linalg.norm(np.diff(root_array[:, :2], axis=0), axis=1).max() < 0.3
            and forbidden_contacts["forbidden_contact_count"] == 0
        )
        route = route_outcome(root_array.tolist(), heights, alive=not terminal, constraints_ok=constraints)
        drift = float(np.linalg.norm(root_array[:, :2] - root_array[0, :2], axis=1).max())
        final_speed = float(np.linalg.norm(successor["root"][7:9]))
        if case["regression"]:
            route["success"] = bool(
                not terminal
                and constraints
                and len(actions) == 1000
                and drift <= (0.25 if case["task"] == "stand" else 0.75)
                and final_speed <= 0.1
            )
        diagnostics = diagnostic_metrics(
            np.asarray(joint_positions),
            np.asarray(body_positions),
            np.asarray(body_velocities),
            np.asarray(contact_forces),
            core.feet_indices.detach().cpu().numpy(),
            np.asarray(root_velocities),
            case["commands"]["velocity_world_xy"],
            np.asarray(actions),
            np.asarray(foot_clearances),
            prompt["reference_dof_pos"][1 : len(actions) + 1],
            prompt["reference_body_pos_heading0"][1 : len(actions) + 1],
        )
        stem = case["id"] + "-" + condition
        for directory in ("videos", "depth", "depth-videos", "traces"):
            (output / directory).mkdir(exist_ok=True)
        video = "videos/" + stem + ".mp4"
        media.write_video(str(output / video), frames, fps=25)
        decoded = media.read_video(str(output / video))
        if len(decoded) != len(frames) or len(decoded) < 1 or decoded.shape[1:3] != frames[0].shape[:2]:
            raise ValueError("recorded native RGB video did not decode completely")
        depth_name = "depth/" + stem + ".npz"
        trace_name = "traces/" + stem + ".npz"
        np.savez_compressed(
            output / depth_name,
            raw=np.asarray(depths),
            policy=np.asarray(presented),
            control_hz=50,
            depth_input_time_s=np.arange(len(actions)) * 0.02,
        )
        with np.load(output / depth_name, allow_pickle=False) as saved:
            if not np.array_equal(saved["raw"], np.asarray(depths)) or saved["raw"].shape[0] != len(actions):
                raise ValueError("saved real depth is incomplete")
        depth_video = "depth-videos/" + stem + ".mp4"
        depth_images = np.asarray(presented)[:, -1]
        if depth_images.dtype != np.uint8:
            depth_images = np.rint(np.clip(depth_images, 0, 1) * 255).astype(np.uint8)
        depth_rgb = np.repeat(np.repeat(depth_images, 10, axis=1), 10, axis=2)
        depth_rgb = np.repeat(depth_rgb[..., None], 3, axis=-1)
        media.write_video(str(output / depth_video), depth_rgb, fps=50)
        decoded_depth = media.read_video(str(output / depth_video))
        if len(decoded_depth) != len(actions):
            raise ValueError("network-input depth preview video is incomplete")
        np.savez_compressed(
            output / trace_name,
            root_xyz=root_array,
            terrain_heights=np.asarray(heights),
            feet_toe_heel_xyz=np.asarray(foot_trace),
            actions=np.asarray(actions),
            tilt_rad=np.asarray(tilts),
            torque_ratios=np.asarray(torque_ratios),
            joint_excess_rad=np.asarray(joint_excess),
            dof_pos=np.asarray(joint_positions),
            body_pos=np.asarray(body_positions),
            body_vel=np.asarray(body_velocities),
            contact_forces=np.asarray(contact_forces),
            root_velocity_world_xy=np.asarray(root_velocities),
            reference_dof_pos=prompt["reference_dof_pos"][1 : len(actions) + 1],
            reference_body_pos_heading0=prompt["reference_body_pos_heading0"][1 : len(actions) + 1],
            reference_velocity_world_xy=reference_velocity[: len(actions)],
            commanded_velocity_world_xy=np.tile(case["commands"]["velocity_world_xy"], (len(actions), 1)),
            foot_clearances_m=np.asarray(foot_clearances),
            physics_time_s=np.arange(len(actions) + 1) * 0.02,
            diagnostic_time_s=np.arange(1, len(actions) + 1) * 0.02,
            rgb_time_s=np.asarray(rgb_times),
        )
        return {
            "case_id": case["id"],
            "seed": case["seed"],
            "family": case["family"],
            "condition": condition,
            "commands": case["commands"],
            "diagnostics": diagnostics,
            **forbidden_contacts,
            "complete": True,
            "success": route["success"],
            "route": route,
            "terminated": terminal,
            "goal_reached": goal_reached,
            "max_drift_m": drift,
            "final_speed_m_s": final_speed,
            "constraints_ok": constraints,
            "max_step_displacement_m": float(
                np.linalg.norm(np.diff(root_array[:, :2], axis=0), axis=1).max()
            ),
            "min_root_clearance_m": float(min(root_array[:, 2] - np.asarray(heights))),
            "steps": len(actions),
            "seconds": len(actions) / 50,
            "video": video,
            "depth": depth_name,
            "trace": trace_name,
            "video_verified": True,
            "video_frames": len(decoded),
            "video_fps": 25,
            "depth_verified": True,
            "depth_video": depth_video,
            "depth_video_verified": True,
            "depth_video_frames": len(decoded_depth),
            "depth_video_fps": 50,
            "depth_frames": len(depths),
            "foot_geometry": geometry,
            "max_tilt_rad": max(tilts),
            "max_torque_ratio": max(torque_ratios),
            "max_joint_excess_rad": max(joint_excess),
        }
    finally:
        if renderer is not None:
            renderer.close()
        wrapped.close()


def append_completed_case(output: Path, plan: dict, row: dict) -> None:
    """Durably retain every completed rollout before starting another one."""
    evidence = []
    for key in ("video", "depth_video", "depth", "trace"):
        path = safe_file(output, row[key])
        evidence.append({"path": row[key], "bytes": path.stat().st_size, "sha256": sha256(path)})
    record = {
        **row,
        "run_id": os.environ["EB_RUN_ID"],
        "plan_hash": plan["hash"],
        "model_version_id": plan["model_version_id"],
        "model_hash": plan["model_hash"],
        "evidence_files": evidence,
    }
    with (output / "cases.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def run(plan: dict, output: Path = Path("/outputs")):
    import numpy as np

    validate_eval_plan(plan)
    _, robot, config, _, source_receipt = prepare_runtime(plan, output)
    package = read_json(safe_file(Path("/export"), "manifest.json"))
    if (
        package.get("model_hash") != plan["model_hash"]
        or package.get("trace_verified") is not True
        or sha256(safe_file(Path("/export"), "policy.onnx")) != package.get("onnx_sha256")
        or sha256(safe_file(Path("/export"), "tracking-prompt.npz")) != package.get("prompt_sha256")
    ):
        raise ValueError("independent evaluation requires a complete current-model direct-depth export")
    session = parity_session(Path("/export/policy.onnx"))
    with np.load("/export/tracking-prompt.npz", allow_pickle=False) as archive:
        prompt = {key: archive[key] for key in archive.files}
    if prompt["z"].shape != (1000, 256):
        raise ValueError("export tracking context is incomplete")
    # The exported heading0-relative targets rotate together with the +X reset.
    prompt["heading_world_xy"] = prompt["heading_relative_xy"]
    prompt["reference_velocity_world_xy"] = prompt["reference_velocity_heading0_xy"]
    rows = []
    for case in plan["cases"]:
        for condition in ("normal", "occluded"):
            row = evaluate_case(plan, case, condition, config, robot, prompt, session, output)
            append_completed_case(output, plan, row)
            rows.append(row)
            print(
                json.dumps(
                    {
                        "case_id": row["case_id"],
                        "condition": condition,
                        "success": row["success"],
                        "seconds": row["seconds"],
                    }
                ),
                flush=True,
            )
    metrics = validate_case_rows(plan, rows)
    write_json(output / "cases.json", {"schema_version": "eb/pbfm-terrain-cases/v1", "cases": rows})
    write_json(output / "metrics.json", metrics)
    write_json(
        output / "media-index.json",
        {
            "schema_version": "eb/pbfm-media-index/v1",
            "cases": [
                {
                    key: row[key]
                    for key in ("case_id", "condition", "success", "video", "depth", "depth_video", "trace")
                }
                for row in rows
            ],
            "media_categories_absent": metrics["media_categories_absent"],
            "clock": {
                "depth_video_fps": 50,
                "depth_time_s": "input at step*0.02; depth NPZ depth_input_time_s",
                "rgb_video_fps": 25,
                "rgb_time_s": "initial plus selected physics successors; trace rgb_time_s",
                "physics_time_s": "initial 0 then (step+1)*0.02; terminal successor before reset",
                "alignment": "case/condition identity and explicit timestamps; actual sensor/physics frames",
            },
        },
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(12, 6))
    normal = [r for r in rows if r["condition"] == "normal"]
    axis.barh([r["case_id"] for r in normal], [r["route"]["progress_m"] for r in normal])
    axis.axvline(plan["suite_spec"]["route_distance_m"], color="red", label="route threshold")
    axis.set_xlabel("Measured +X route progress (m)")
    axis.legend()
    figure.tight_layout()
    (output / "analysis").mkdir(exist_ok=True)
    figure.savefig(output / "analysis/terrain.svg")
    plt.close(figure)
    write_json(
        output / "eval-receipt.json",
        {
            "schema_version": "eb/pbfm-eval-receipt/v1",
            "run_id": os.environ["EB_RUN_ID"],
            "operation": plan["operation"],
            "plan_hash": plan["hash"],
            "source_receipt": source_receipt,
            "metrics": metrics,
            "qualification": "diagnostic",
            "formal_acceptance": False,
            "files": file_index(output),
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-json", required=True)
    run(json.loads(parser.parse_args().plan_json))


if __name__ == "__main__":
    main()
