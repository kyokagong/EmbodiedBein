"""Bounded CPU A1 candidate worker (Python 3.10, no control-plane imports).

Uses the native RobotState/MotionData builder and the existing MuJoCo backend
for a reference PD diagnostic. No trainer, policy, root servo, or padding.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

from eb.runners.ufo_training_data import (
    digest,
    native_loader,
    read_json,
    regular,
    validate_groups,
    validate_records,
    write_json,
)
from eb.ufo_grounded_reference import (
    A1_PATH,
    A1_SHA,
    BASE_IMAGE,
    OPERATION,
    PLAN_SCHEMA,
    PROTOCOL,
    RECEIPT_SCHEMA,
    ROOT_Z_SHIFT,
    SOURCE_KEY,
    TRAIN_SHA,
    VALIDATION_SHA,
    quality_errors,
)


def grounded_positions(root, shift):
    root = np.asarray(root)
    if (
        root.ndim != 2
        or root.shape[1] != 3
        or not np.isfinite(root).all()
        or not np.isfinite(shift)
        or not 0 < shift <= 0.1
    ):
        raise ValueError("Grounding requires finite source positions and a bounded positive delta z")
    result = root.astype(np.float64, copy=True)
    result[:, 2] += shift
    return result


def playback_targets(q, source_fps, control_hz):
    q = np.asarray(q)
    duration = (len(q) - 1) / source_fps
    steps = round(duration * control_hz)
    if (
        q.ndim != 2
        or len(q) < 2
        or not np.isfinite(q).all()
        or not np.isclose(duration * control_hz, steps, atol=1e-9, rtol=0)
    ):
        raise ValueError(
            "Reference duration must cover whole control intervals without truncation or padding"
        )
    times = np.arange(steps) / control_hz
    source_time = np.arange(len(q)) / source_fps
    targets = np.column_stack([np.interp(times, source_time, q[:, j]) for j in range(q.shape[1])])
    return times, targets


def verify_snapshot(root, manifest_path, expected):
    """Bind every frozen file to its stored manifest; never scan the live repo."""
    from eb.ufo_native import safe_file

    manifest = read_json(regular(manifest_path, 16 * 1024**2))
    if "sha256:" + digest(manifest_path) != expected:
        raise ValueError("Frozen source manifest changed")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or not 0 < len(entries) <= 20000:
        raise ValueError("Frozen source manifest is unbounded")
    total = 0
    for entry in entries:
        relative = entry["path"]
        if (
            not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or any(p.is_symlink() for p in (root / relative, *(root / relative).parents))
        ):
            raise ValueError("Unsafe frozen source entry")
        if entry["sha256"] is None:
            if not (root / relative).is_dir():
                raise ValueError("Frozen source directory missing")
            continue
        path = safe_file(root, relative)
        total += path.stat().st_size
        if total > 4 * 1024**3 or digest(regular(path, 1024**3)) != entry["sha256"].removeprefix("sha256:"):
            raise ValueError("Frozen source file identity or byte budget differs")


def native_configuration(output, ufo_root, motion_path):
    """Resolve native task/control/physics without constructing a model/trainer."""
    import yaml
    from omegaconf import OmegaConf
    from ufo.envs.config import make_mjlab_ufo_env_cfg, resolve_task_config
    from ufo.tools.robot_config import load_robot_training_spec
    from ufo.train import build_mjlab_config

    original = ufo_root / "configs/robots/rpo_23dof.yaml"
    robot_dict = yaml.safe_load(original.read_text())
    robot_dict["xml_path"] = str(ufo_root / "ufo/data/robots/rpo/rpo_23dof.xml")
    robot_path = output / "robot.yaml"
    robot_path.write_text(yaml.safe_dump(robot_dict, sort_keys=False))
    native = load_robot_training_spec(robot_path)
    cfg = build_mjlab_config(
        device="cpu",
        work_dir=str(output / "unused-native-work"),
        num_envs=1,
        num_env_steps=1,
        seed=PROTOCOL["seed"],
        use_wandb=False,
        wandb_run_name=None,
        agent="fb",
        data_path=str(motion_path),
        robot_config=robot_path,
    )
    task = resolve_task_config(cfg.env, 1)
    simulation = make_mjlab_ufo_env_cfg(
        task,
        num_envs=1,
        seed=PROTOCOL["seed"],
        mjcf_path=cfg.env.mjcf_path,
        auto_reset=False,
        robot_training=cfg.env.robot_training,
    )
    robot = cfg.env.robot_training
    names = robot["robot"]["control_joint_names"]
    control = task["robot"]["control"]
    if (
        names != list(native.robot.control_joint_names)
        or len(names) != 23
        or task["simulator"]["config"]["sim"] != {"fps": 400, "control_decimation": 8}
        or control["stiffness"] != robot["stiffness"]
        or control["damping"] != robot["damping"]
    ):
        raise ValueError("Native RPO control/physics mapping differs")
    codec = {
        "schema_version": "eb/ufo-native-reference-pd-codec/v1",
        "joint_order": names,
        "physics_dt": 0.0025,
        "decimation": 8,
        "policy_hz": 50,
        "default_pos": [robot["default_joint_angles"][n] for n in names],
        "kp": [robot["stiffness"][n] for n in names],
        "kd": [robot["damping"][n] for n in names],
        "lower": [robot["robot"]["joint_ranges"][n][0] for n in names],
        "upper": [robot["robot"]["joint_ranges"][n][1] for n in names],
        "effort": robot["effort_limits"],
        "velocity": robot["velocity_limits"],
        "actuator": robot["actuator"],
        "root_quaternion": "xyzw",
        "qpos_quaternion": "wxyz",
    }
    if not all(
        np.isfinite(codec[k]).all()
        for k in ("kp", "kd", "lower", "upper", "effort", "velocity", "default_pos")
    ):
        raise ValueError("Native reference PD configuration is nonfinite")
    actuator_mapping = verify_actuator_mapping(simulation, codec)
    payload = output / "diagnostic-input"
    (payload / "assets").mkdir(parents=True)
    # The native callback also sets RPO collision priorities for plane friction.
    # Preserve those settings when using the existing explicit-motor backend.
    native_spec = simulation.scene.entities["robot"].spec_fn()
    (payload / "assets/rpo.xml").write_text(native_spec.to_xml())
    native_model = native_spec.compile()
    priorities, counts = np.unique(native_model.geom_priority, return_counts=True)
    write_json(payload / "codec.json", codec)
    write_json(payload / "physics.json", asdict(simulation.sim.mujoco))
    write_json(payload / "effective_config.json", {"task": OmegaConf.to_container(task, resolve=True)})
    write_json(
        output / "native-config-receipt.json",
        {
            "robot_source_sha256": digest(original),
            "robot_derived_sha256": digest(robot_path),
            "only_robot_yaml_change": {"xml_path": robot_dict["xml_path"]},
            "native_robot_xml_sha256": digest(Path(native.robot.xml_path)),
            "diagnostic_xml_sha256": digest(payload / "assets/rpo.xml"),
            "diagnostic_xml_builder": "native MJLab robot EntityCfg.spec_fn (including collision priorities)",
            "native_reward_set": str(task.rewards.set_reward),
            "native_geom_priorities": {str(int(p)): int(n) for p, n in zip(priorities, counts)},
            "config_builder": "ufo.train.build_mjlab_config (configuration only)",
            "task_resolver": "ufo.envs.config.resolve_task_config",
            "physics_builder": "ufo.envs.config.make_mjlab_ufo_env_cfg",
            "pd_control": codec,
            "actuator_mapping": actuator_mapping,
            "physics": asdict(simulation.sim.mujoco),
        },
    )
    return native, payload, codec


def verify_actuator_mapping(simulation, codec):
    """Compare PD/DC parameters and numerical clipping to the actual native cfg."""
    import torch
    from mjlab.actuator import DcMotorActuator

    from eb.runners.ufo_native_sim import motor_torque

    actuators = simulation.scene.entities["robot"].articulation.actuators
    if len(actuators) != len(codec["joint_order"]) or codec["actuator"]["model"] != "dc_motor":
        raise ValueError("A1 reference protocol requires the frozen native DC motor actuator model")
    rows = []
    for i, (name, cfg) in enumerate(zip(codec["joint_order"], actuators)):
        params = codec["actuator"]["joints"][name]
        expected = {
            "stiffness": codec["kp"][i],
            "damping": codec["kd"][i],
            "effort_limit": codec["effort"][i],
            "saturation_effort": codec["effort"][i],
            "velocity_limit": codec["velocity"][i],
            "armature": params["armature"],
            "frictionloss": params["friction"],
            "delay_min_lag": codec["actuator"]["min_delay"],
            "delay_max_lag": codec["actuator"]["max_delay"],
            "delay_hold_prob": 1.0,
        }
        if tuple(cfg.target_names_expr) != (name,) or any(getattr(cfg, k) != v for k, v in expected.items()):
            raise ValueError(f"Native PD/DC motor mapping differs for {name}")
        rows.append({"joint": name, **expected})
    effort, velocity = np.asarray(codec["effort"]), np.asarray(codec["velocity"])
    grid = np.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
    requested = np.broadcast_to(grid[:, None, None] * effort, (7, 7, len(effort))).copy()
    speed = np.broadcast_to(grid[None, :, None] * velocity, requested.shape).copy()
    native_clip = DcMotorActuator.control_law.__globals__["dc_motor_clip"]

    def tensor(a):
        return torch.as_tensor(a, dtype=torch.float64)

    expected = native_clip(
        tensor(requested), tensor(effort), tensor(velocity), tensor(effort), tensor(speed)
    ).numpy()
    actual = motor_torque(requested, speed, effort, velocity, np.zeros_like(effort))
    error = float(np.max(np.abs(expected - actual)))
    if not np.isfinite(expected).all() or error > 1e-10:
        raise ValueError("Existing MuJoCo backend DC motor clipping differs from native mjlab")
    return {
        "verified": True,
        "joints": rows,
        "dc_motor_grid_samples": int(expected.size),
        "max_clipping_error_nm": error,
        "native_clipping": "mjlab.actuator.DcMotorActuator.control_law/dc_motor_clip",
    }


def geometry_depths(backend, root, quat_wxyz, q):
    """Kinematic inspection on separate data; ground means the real plane pair."""
    mj, model = backend.mj, backend.model
    data = mj.MjData(model)
    floor = model.geom("deployment_floor").id
    depths = np.zeros((len(q), 2))
    for i in range(len(q)):
        data.qpos[:3], data.qpos[3:7] = root[i], quat_wxyz[i]
        data.qpos[backend.qadr] = q[i]
        mj.mj_forward(model, data)
        for contact in data.contact:
            kind = 0 if floor in (contact.geom1, contact.geom2) else 1
            depths[i, kind] = max(depths[i, kind], -contact.dist)
    return depths


def plane_contacts(backend):
    floor = backend.model.geom("deployment_floor").id
    feet = set(backend.feet_ids)
    result = np.zeros(len(feet), dtype=float)
    spatial = np.empty(6)
    for i in range(backend.data.ncon):
        contact = backend.data.contact[i]
        pair = (contact.geom1, contact.geom2)
        if floor not in pair:
            continue
        other = pair[1] if pair[0] == floor else pair[0]
        body = int(backend.model.geom_bodyid[other])
        if body in feet:
            backend.mj.mj_contactForce(backend.model, backend.data, i, spatial)
            result[backend.feet_ids.index(body)] += max(0.0, float(spatial[0]))
    return result


def reference_playback(backend, root, quat, q, fps, output):
    from ufo.inference.runtime import video_writer

    from eb.runners.ufo_native_sim import CommandDelay, case_lag, physical_metrics, probe_video

    mj = backend.mj
    times, targets = playback_targets(q, fps, 50)
    # Exactly one dynamic root initialization. Geometry uses a different MjData.
    backend.delay = CommandDelay(case_lag(backend.codec, PROTOCOL["seed"]))
    mj.mj_resetData(backend.model, backend.data)
    backend.data.qpos[:3], backend.data.qpos[3:7] = root[0], quat[0]
    backend.data.qpos[backend.qadr] = q[0]
    next_qpos = backend.data.qpos.copy()
    next_qpos[:3], next_qpos[3:7], next_qpos[backend.qadr] = root[1], quat[1], q[1]
    mj.mj_differentiatePos(backend.model, backend.data.qvel, 1 / fps, backend.data.qpos, next_qpos)
    mj.mj_forward(backend.model, backend.data)
    trace = {
        key: []
        for key in (
            "time_s",
            "simulation_time_s",
            "qpos",
            "qvel",
            "q",
            "dq",
            "target_q",
            "torque",
            "height",
            "tilt",
            "com_position_w",
            "foot_position_w",
            "foot_plane_normal_force_n",
        )
    }
    samples, complete, finite = [], 0, True

    def record(t, target):
        state = backend.state()
        values = {
            "time_s": t,
            "simulation_time_s": float(backend.data.time),
            "qpos": state["qpos"],
            "qvel": backend.data.qvel.copy(),
            "q": state["q"],
            "dq": state["dq"],
            "target_q": target.copy(),
            "torque": state["torque"],
            "height": state["height"],
            "tilt": state["tilt"],
            "com_position_w": state["com_position_w"],
            "foot_position_w": state["foot_position_w"],
            "foot_plane_normal_force_n": plane_contacts(backend),
        }
        if not all(np.isfinite(value).all() for value in values.values()):
            return False
        for key, value in values.items():
            trace[key].append(value)
        return True

    record(0.0, targets[0])
    video = output / "reference/replay.mp4"
    video.parent.mkdir()
    # Falls/physical-limit failures never end this loop or remove their frames.
    with video_writer(video, shape=(480, 480), fps=25) as writer:
        writer.add_image(backend.render())
        for index, (t, target) in enumerate(zip(times, targets)):
            backend.step(None, target)
            if not np.isfinite(backend.substeps).all() or not record(float(t + 0.02), target):
                finite = False
                break
            samples.extend(backend.substeps.copy())
            complete += 1
            if (index + 1) % 2 == 0:
                writer.add_image(backend.render())
    arrays = {key: np.asarray(value) for key, value in trace.items()}
    arrays["substep_limits"] = np.asarray(samples).reshape(-1, 6)
    np.savez_compressed(output / "reference/trace.npz", **arrays)
    physical = physical_metrics(samples, 0.0025) if samples else {}
    metrics = {
        **physical,
        "finite": finite,
        "completed_steps": complete,
        "expected_steps": len(times),
        "min_height_m": float(arrays["height"].min()),
        "max_tilt_rad": float(arrays["tilt"].max()),
        "drift_m": float(np.linalg.norm(arrays["qpos"][-1, :2] - root[0, :2])),
        "ground_contact_frames": int(np.count_nonzero(arrays["foot_plane_normal_force_n"].sum(1) > 0.0)),
        "physics_seconds": float(backend.data.time),
        "simulation_time_monotonic": bool(np.all(np.diff(arrays["simulation_time_s"]) > 0)),
        "mujoco_warning_count": int(np.asarray(backend.data.warning.number).sum()),
    }
    try:
        metrics["video"] = probe_video(video)
    except Exception as exc:
        metrics["video"] = {"decodable": False, "error": str(exc)}
    return metrics


def prepare(
    plan,
    dataset_id,
    storage_ref,
    output=Path("/outputs"),
    parent=Path("/parent"),
    ufo_root=Path("/sources/ufo"),
):
    import joblib
    import yaml
    from ufo.motion.manifest import prepare_motion_manifest
    from ufo.tools.motion_data import main as build_motion_data

    from eb.runners.ufo_native_sim import MujocoBackend
    from eb.ufo_native import plan_hash, safe_file

    unsigned = {k: v for k, v in plan.items() if k != "hash"}
    if (
        plan.get("schema_version") != PLAN_SCHEMA
        or plan.get("operation") != OPERATION
        or plan.get("protocol") != PROTOCOL
        or plan.get("base_training_image") != BASE_IMAGE
        or plan.get("hash") != plan_hash(unsigned)
    ):
        raise ValueError("A1 worker received a different protocol or input plan")
    # Host mount root is frozen in the plan so nondefault asset roots also work.
    asset_root = Path(plan["asset_mount"])
    for name, root in (("ufo", ufo_root), ("asset", asset_root)):
        verify_snapshot(root, Path(f"/sources/{name}-manifest.json"), plan[name + "_source_ref"]["commit"])
    for item in plan["parent_files"]:
        path = safe_file(parent, item["path"])
        if path.stat().st_size != item["bytes"] or digest(regular(path)) != item["sha256"]:
            raise ValueError("Frozen parent input changed")
    source = safe_file(asset_root, A1_PATH)
    if digest(regular(source)) != A1_SHA:
        raise ValueError("A1 original source changed")
    if any((output / name).exists() for name in ("data", "holdout", "reference", "robot-state")):
        raise ValueError("Refusing to overwrite candidate outputs")
    output.mkdir(parents=True, exist_ok=True)
    provenance_output = output / "provenance"
    provenance_output.mkdir()
    for filename in ("qualification.json", "import_report.json", "mask_feasibility.json"):
        shutil.copyfile(parent / filename, provenance_output / filename)
        (provenance_output / filename).chmod(0o444)
    raw = joblib.load(source)
    root = np.asarray(raw["root_pos"], dtype=float)
    quat = np.asarray(raw["root_rot"], dtype=float)
    original_q = np.asarray(raw["dof_pos"], dtype=float)
    fps = float(np.asarray(raw["fps"]).reshape(-1)[0])
    if (
        root.shape != (457, 3)
        or quat.shape != (457, 4)
        or original_q.shape != (457, 23)
        or fps != 120
        or not all(np.isfinite(a).all() for a in (root, quat, original_q))
        or not np.allclose(np.linalg.norm(quat, axis=1), 1.0, atol=1e-5)
    ):
        raise ValueError("Original A1 shape/fps/quaternion differs")
    mapping = yaml.safe_load((asset_root / "robolab/scripts/tools/retarget/config/rpo.yaml").read_text())
    names = mapping["lab_dof_names"]
    if len(names) != 23 or len(set(names)) != 23:
        raise ValueError("Original A1 joint mapping is incomplete")
    grounded = grounded_positions(root, ROOT_Z_SHIFT)
    native, payload, codec = native_configuration(output, ufo_root, parent / "train.pkl")
    order = [names.index(name) for name in native.robot.control_joint_names]
    q = original_q[:, order]
    # RobotState reader uses XYZW; original official data uses WXYZ.
    state = output / "robot-state/A1-grounded.npz"
    state.parent.mkdir()
    np.savez_compressed(
        state,
        root_pos=grounded,
        root_quat=quat[:, [1, 2, 3, 0]],
        dof_pos=q,
        fps=fps,
        joint_names=np.asarray(native.robot.control_joint_names),
        motion_key=SOURCE_KEY + "__grounded",
    )
    builder_args = [
        "build",
        "--robot",
        str(output / "robot.yaml"),
        "--source",
        str(state),
        "--format",
        "robot_state_npz",
        "--name",
        "rpo-stand-grounded-v1",
        "--out",
        str(output / "native-motion.yaml"),
        "--clip-seconds",
        "10",
        "--stride-seconds",
        "10",
        "--min-clip-seconds",
        "1",
    ]
    build_motion_data(builder_args)
    built = prepare_motion_manifest(output / "native-motion.yaml")
    if len(built.train_data_paths) != 1:
        raise ValueError("Native builder produced multiple candidate datasets")
    added = joblib.load(built.train_data_paths[0])
    if len(added) != 1 or len(next(iter(added.values()))["dof_pos"]) != 457:
        raise ValueError("Native builder cropped, duplicated or padded A1")
    new_record = next(iter(added.values()))
    preserved = (("dof_pos", q), ("root_trans_offset", grounded), ("root_quat", quat[:, [1, 2, 3, 0]]))
    if float(new_record["fps"]) != fps or any(
        not np.array_equal(new_record[field], values.astype(np.float32)) for field, values in preserved
    ):
        raise ValueError("Native conversion changed A1 joints, root trajectory, quaternion or fps")
    new_record["metadata"].update(
        motion_key=SOURCE_KEY,
        source_path=str(source),
        source_sha256=A1_SHA,
        source_group="official_rpo_unknown_ancestry",
        split="train",
        source_transform=PROTOCOL,
    )
    # Preserve the original audit path even though the execution mount is aliased.
    provenance = read_json(parent / "import_report.json")
    audit = next(row for row in provenance["audit"] if row["name"] == SOURCE_KEY)
    if audit["sha256"] != A1_SHA:
        raise ValueError("A1 ancestry is absent from the parent import audit")
    new_record["metadata"]["source_path"] = audit["source"]
    new_record["dof_vel"] = (np.gradient(q, axis=0) * fps).astype(np.float32)
    backend = MujocoBackend(payload, payload)
    try:
        before = geometry_depths(backend, root, quat, q)
        after = geometry_depths(backend, grounded, quat, q)
        np.savez_compressed(output / "geometry.npz", original_depths_m=before, grounded_depths_m=after)
        metrics = reference_playback(backend, grounded, quat, q, fps, output)
    finally:
        backend.close()
    metrics.update(
        floor_max_m=float(after[:, 0].max()),
        self_max_m=float(after[:, 1].max()),
        original_floor_max_m=float(before[:, 0].max()),
        original_floor_median_m=float(np.median(before[:, 0])),
        grounding_delta_error_m=abs(float(np.median(before[:, 0])) - ROOT_Z_SHIFT),
        source_joint_excess_rad=float(
            np.maximum(0, np.maximum(np.asarray(codec["lower"]) - q, q - np.asarray(codec["upper"]))).max()
        ),
        source_velocity_excess_rad_s=float(
            np.maximum(0, np.abs(np.gradient(q, axis=0) * fps) - np.asarray(codec["velocity"])).max()
        ),
    )
    errors = quality_errors(metrics)
    limitations = [
        "Reference PD diagnostic is not policy evaluation or proof of learned behavior.",
        "No root tracking force after t0; one 3.8-second take and one deterministic motor delay.",
        "Foot contact uses only actual deployment_floor plane pairs; self contacts are excluded.",
        "The original source ancestry and licensing remain independently unverified.",
        "No test motions were opened; the unchanged 15-motion validation split remains separate.",
        "Encoded 96 frames at 25 fps span 3.84 seconds; physics duration is exactly 3.8 seconds.",
    ]
    write_json(
        output / "reference/quality.json",
        {
            "metrics": metrics,
            "errors": errors,
            "protocol": PROTOCOL,
            "native_builder_argv": builder_args,
            "backend_step_sha256": __import__("hashlib")
            .sha256(inspect.getsource(MujocoBackend.step).encode())
            .hexdigest(),
            "geometry_method": "all original frames; MuJoCo collision geoms; plane pairs vs self contacts",
            "grounding": {
                "delta_z_m": ROOT_Z_SHIFT,
                "original_frames": 457,
                "retained_frames": 457,
                "source_arrays_unchanged": False,
                "joints_reordered_only": True,
            },
            "unchecked": limitations,
        },
    )
    report = {
        "dataset_id": dataset_id,
        "format": "ufo_native_motion",
        "storage_ref": storage_ref,
        "source_refs": [
            {
                "kind": OPERATION,
                "parent_dataset_id": plan["parent_dataset_id"],
                "parent_version_id": plan["parent_version_id"],
                "source_sha256": A1_SHA,
                "root_z_shift_m": ROOT_Z_SHIFT,
            }
        ],
        "validation": {
            "scope": "ufo_native_motion_v1",
            "status": "invalid" if errors else "validated",
            "training_ready": not errors,
            "storage_ref": storage_ref,
            "errors": errors,
            "unchecked": limitations,
        },
    }
    if not errors:
        report.update(
            freeze_candidate(output, parent, dataset_id, storage_ref, added, native, plan, limitations)
        )
    write_json(output / "dataset-index.json", report)
    files = []
    for path in sorted(output.rglob("*")):
        if path.is_symlink():
            raise ValueError("Linked candidate output")
        if path.is_file() and path.name != "candidate-receipt.json":
            if not 0 < path.stat().st_size <= 2 * 1024**3:
                raise ValueError("Candidate output exceeds file bounds")
            files.append(
                {
                    "path": path.relative_to(output).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": digest(path),
                }
            )
    if len(files) > 200 or sum(item["bytes"] for item in files) > 512 * 1024**2:
        raise ValueError("Candidate output index exceeds bounds")
    write_json(
        output / "candidate-receipt.json",
        {
            "schema_version": RECEIPT_SCHEMA,
            "operation": OPERATION,
            "run_id": os.environ["EB_RUN_ID"],
            "dataset_id": dataset_id,
            "plan_hash": plan["hash"],
            "image_digest": plan["image_digest"],
            "training_ready": not errors,
            "reference_pd_only": True,
            "protocol": PROTOCOL,
            "files": files,
        },
    )
    return report


def freeze_candidate(output, parent, dataset_id, storage_ref, added, native, plan, limitations):
    import joblib

    if digest(parent / "train.pkl") != TRAIN_SHA or digest(parent / "validation.pkl") != VALIDATION_SHA:
        raise ValueError("Parent split changed while preparing candidate")
    provenance, qualification = (
        read_json(parent / "import_report.json"),
        read_json(parent / "qualification.json"),
    )
    groups = validate_groups(qualification, provenance)
    train, validation = joblib.load(parent / "train.pkl"), joblib.load(parent / "validation.pkl")
    if len(train) != 58 or len(validation) != 15 or set(train) & set(added) or set(validation) & set(added):
        raise ValueError("Candidate would overwrite a motion identity or change its frozen holdout")
    train.update(added)
    train_summary = validate_records(train, "train", groups["train"], native.robot, provenance)
    validation_summary = validate_records(
        validation, "validation", groups["validation"], native.robot, provenance
    )
    for split, dirname, filename, summary in (
        ("train", "data", "train.pkl", train_summary),
        ("validation", "holdout", "validation.pkl", validation_summary),
    ):
        folder = output / dirname
        folder.mkdir()
        if split == "train":
            joblib.dump(train, folder / filename)
        else:
            shutil.copyfile(parent / filename, folder / filename)
        probe = native_loader(folder / filename, native.robot, native.motion_extensions)
        original_manifest = read_json(
            parent / ("manifest.json" if split == "train" else "validation-manifest.json")
        )
        manifest = {
            **original_manifest,
            **summary,
            "dataset_id": dataset_id,
            "file_sha256": digest(folder / filename),
            "native_loader": probe,
            "source_uri": "ufo-grounded-a1:" + plan["parent_dataset_id"],
            "source_filename": "derived-native-train.pkl" if split == "train" else "validation.pkl",
            "parent_dataset_id": plan["parent_dataset_id"],
            "parent_version_id": plan["parent_version_id"],
            "parent_manifest_sha256": plan["parent_manifest_sha256"],
            "source_transform": PROTOCOL,
            "validation_bytes_unchanged": split == "validation",
            "unchecked": limitations,
        }
        write_json(folder / "manifest.json", manifest)
        (folder / filename).chmod(0o444)
    if digest(output / "holdout/validation.pkl") != VALIDATION_SHA:
        raise ValueError("Candidate holdout bytes changed")
    return {
        **train_summary,
        "total_bytes": (output / "data/train.pkl").stat().st_size,
        "holdout": {
            **validation_summary,
            "storage_ref": str(Path(storage_ref).parent / "holdout"),
            "manifest_sha256": digest(output / "holdout/manifest.json"),
        },
        "validation": {
            "scope": "ufo_native_motion_v1",
            "status": "validated",
            "training_ready": True,
            "storage_ref": storage_ref,
            "manifest_sha256": digest(output / "data/manifest.json"),
            "errors": [],
            "unchecked": limitations,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    for name in ("plan-json", "dataset-id", "storage-ref"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    sys.path.insert(0, "/sources/ufo")
    os.chdir("/outputs")
    plan = json.loads(args.plan_json)
    try:
        report = prepare(plan, args.dataset_id, args.storage_ref)
        print(json.dumps({"event": "candidate_complete", "validation": report["validation"]}), flush=True)
    except Exception as exc:
        # Infrastructure/schema failures keep all produced files, and cannot publish a version.
        write_json(
            Path("/outputs/worker-error.json"),
            {"status": "failed", "error": str(exc), "training_ready": False, "reference_pd_only": True},
        )
        raise


if __name__ == "__main__":
    main()
