"""Independent frozen RPO terrain protocol; no UFO observation codecs."""

from __future__ import annotations

from eb.pbfm_io import fingerprint, finite, hash_value, relative

EVAL_OPERATION = "evaluate_pbfm_terrain"
EVAL_SCHEMA = "eb/pbfm-terrain-eval-plan/v1"
SUITE_ID = "es_rpo_pbfm_terrain_validation_v1"
FAMILIES = (
    "flat",
    "perlin_rough",
    "low_stairs_up",
    "low_stairs_down",
    "low_platforms",
    "hf_pyramid_slope_inv",
    "boxes",
)
SEEDS = (71000, 71001)


def suite_spec() -> dict:
    return {
        "schema_version": "eb/pbfm-terrain-suite/v1",
        "profile": "validation",
        "independent": True,
        "evaluation_mode": {
            "domain_randomization": False,
            "observation_noise": False,
            "camera_installation_randomization": False,
        },
        "qualification": "diagnostic",
        "formal_acceptance": False,
        "horizon_s": 20.0,
        "control_hz": 50,
        "difficulty_row": 0,
        "route_distance_m": 2.2,
        "route_frame": "center_reset_positive_x",
        "terrain_ray_filter": "static collision geoms: body_weldid[geom_bodyid]==0; floating robot excluded",
        "regression_stand_max_drift_m": 0.25,
        "regression_stop_max_drift_m": 0.75,
        "regression_stop_final_speed_m_s": 0.1,
        "diagnostic_definitions": {
            "joint_tracking_rmse_rad": "23 control-joint raw positions versus current-model prompt reference",
            "body_tracking_rmse_m": "24 root-relative body positions; reference fixed heading0 rotated to +X",
            "velocity_rmse_m_s": "root world XY velocity versus frozen case world XY command",
            "foot_slip_m_s": "mean foot-body world XY speed during measured contact force norm >= 5N",
            "stance_contact_force_min_n": 5.0,
            "toe_heel_clearance_m": "RPO mesh-derived toe/heel world Z "
            "minus measured static terrain at each point",
            "action_saturation_fraction": "fraction abs(raw ONNX action * 5) >= 5 before clip/motor delay",
        },
        "max_lateral_error_m": 0.6,
        "max_tilt_rad": 0.8,
        "forbidden_contact_policy": {
            "allowed_bodies": ["left_ankle_roll_link", "right_ankle_roll_link"],
            "force_threshold_n": 5.0,
            "maximum_forbidden_samples": 0,
        },
        "min_root_clearance_m": 0.4,
        "perception_conditions": ["normal", "occluded"],
        "quality_threshold": {"normal_success_rate_min": 0.75, "flat_regressions_success_min": 2},
        "cases": [
            {
                "id": f"{family}-{seed}",
                "family": family,
                "seed": seed,
                "regression": False,
                "commands": {"heading_world_xy": [1.0, 0.0], "velocity_world_xy": [0.3, 0.0], "valid": True},
                "initial_root_velocity_world_xy": [0.0, 0.0],
            }
            for family in FAMILIES
            for seed in SEEDS
        ]
        + [
            {
                "id": f"flat-regression-{seed}",
                "family": "flat",
                "seed": seed,
                "regression": True,
                "task": "stand" if seed == 71010 else "stop",
                "commands": {"heading_world_xy": [1.0, 0.0], "velocity_world_xy": [0.0, 0.0], "valid": True},
                "initial_root_velocity_world_xy": [0.5, 0.0] if seed == 71011 else [0.0, 0.0],
            }
            for seed in (71010, 71011)
        ],
    }


def build_eval_plan(
    *,
    model_version_id: str,
    model_hash: str,
    export_hash: str,
    dataset_manifest: str,
    image_digest: str,
    source_refs: dict,
    camera_contract: dict,
    robot_geometry: dict,
    profile: str = "validation",
    **bindings,
) -> dict:
    suite = suite_spec()
    if profile != "validation":
        raise ValueError(
            "initial 16-case PBFM suite is independent validation; "
            "formal acceptance requires a separate frozen Suite"
        )
    plan = {
        "schema_version": EVAL_SCHEMA,
        "operation": EVAL_OPERATION,
        "suite_id": SUITE_ID,
        "suite_spec": suite,
        "suite_hash": fingerprint(suite),
        "cases": suite["cases"],
        "model_version_id": model_version_id,
        "model_hash": model_hash,
        "export_hash": export_hash,
        "dataset_manifest": dataset_manifest,
        "image_digest": image_digest,
        "source_refs": source_refs,
        "camera_contract": camera_contract,
        "robot_geometry": robot_geometry,
        **bindings,
    }
    plan["hash"] = fingerprint(plan)
    validate_eval_plan(plan)
    return plan


def validate_eval_plan(plan: dict) -> dict:
    if (
        plan.get("schema_version") != EVAL_SCHEMA
        or plan.get("operation") != EVAL_OPERATION
        or plan.get("suite_id") != SUITE_ID
        or plan.get("suite_spec") != suite_spec()
        or plan.get("cases") != suite_spec()["cases"]
        or plan.get("suite_hash") != fingerprint(suite_spec())
        or not all(hash_value(plan.get(key)) for key in ("model_hash", "export_hash", "dataset_manifest"))
        or not hash_value(plan.get("image_digest"), prefixed=True)
        or plan.get("robot_geometry", {}).get("robot") != "rpo"
        or len(plan.get("robot_geometry", {}).get("foot_bodies", [])) != 2
        or not hash_value(plan.get("robot_geometry", {}).get("xml_sha256"))
        or not plan.get("camera_contract")
        or not plan.get("source_refs")
        or not finite(plan)
        or plan.get("hash") != fingerprint({k: v for k, v in plan.items() if k != "hash"})
    ):
        raise ValueError("PBFM independent eval plan inputs, cases or geometry differ")
    return plan


def route_outcome(
    root_xyz: list,
    ground_heights: list,
    *,
    alive: bool,
    constraints_ok: bool,
    distance_m: float = 2.2,
    max_lateral_m: float = 0.6,
) -> dict:
    if (
        len(root_xyz) < 2
        or len(root_xyz) != len(ground_heights)
        or not finite(root_xyz)
        or not finite(ground_heights)
        or any(len(p) != 3 for p in root_xyz)
    ):
        raise ValueError("route outcome requires complete finite root and measured terrain trace")
    progress = float(root_xyz[-1][0] - root_xyz[0][0])
    lateral = max(abs(float(p[1] - root_xyz[0][1])) for p in root_xyz)
    height_delta = float(ground_heights[-1] - ground_heights[0])
    direction = "ascent" if height_delta > 0.025 else "descent" if height_delta < -0.025 else "level"
    return {
        "progress_m": progress,
        "max_lateral_m": lateral,
        "ground_height_delta_m": height_delta,
        "vertical_direction": direction,
        "success": bool(alive and constraints_ok and progress >= distance_m and lateral <= max_lateral_m),
    }


def validate_case_rows(plan: dict, rows: list[dict], *, allow_partial: bool = False) -> dict:
    validate_eval_plan(plan)
    expected = {
        (c["id"], c["seed"], condition) for c in plan["cases"] for condition in ("normal", "occluded")
    }
    keys, media = set(), set()
    cases = {case["id"]: case for case in plan["cases"]}
    for row in rows:
        key = (row.get("case_id"), row.get("seed"), row.get("condition", "normal"))
        if key in keys or key not in expected:
            raise ValueError("case rows require unique frozen case/seed/condition")
        keys.add(key)
        if (
            row.get("complete") is not True
            or type(row.get("success")) is not bool
            or row.get("video_verified") is not True
            or row.get("depth_verified") is not True
            or not finite(row)
        ):
            raise ValueError("case evidence/media is incomplete")
        for name in ("video", "depth", "depth_video", "trace"):
            path = relative(row.get(name))
            if path in media:
                raise ValueError("case media paths must be unique")
            media.add(path)
        case = cases[row["case_id"]]
        diagnostic = row.get("diagnostics") or {}
        if (
            row.get("commands") != case["commands"]
            or any(
                type(diagnostic.get(key)) not in (int, float)
                for key in (
                    "joint_tracking_rmse_rad",
                    "body_tracking_rmse_m",
                    "velocity_rmse_m_s",
                    "max_contact_force_n",
                    "min_toe_clearance_m",
                    "min_heel_clearance_m",
                    "action_saturation_fraction",
                )
            )
            or type(diagnostic.get("stance_samples")) is not int
            or diagnostic["stance_samples"] < 0
            or (
                diagnostic["stance_samples"] == 0
                and (
                    diagnostic.get("foot_slip_m_s") is not None
                    or diagnostic.get("foot_slip_status") != "not_observed"
                )
            )
            or (
                diagnostic["stance_samples"] > 0
                and (
                    type(diagnostic.get("foot_slip_m_s")) not in (int, float)
                    or diagnostic.get("foot_slip_status") != "measured"
                )
            )
        ):
            raise ValueError("case lacks frozen commands or complete physical tracking/contact diagnostics")
        route = row.get("route") or {}
        required_values = ("progress_m", "max_lateral_m", "ground_height_delta_m")
        if (
            row.get("family") != case["family"]
            or any(type(route.get(k)) not in (int, float) for k in required_values)
            or type(row.get("steps")) is not int
            or not 1 <= row["steps"] <= 1000
            or row.get("seconds") != row["steps"] / 50
            or row.get("depth_frames") != row["steps"]
            or row.get("video_fps") != 25
            or row.get("depth_video_verified") is not True
            or row.get("depth_video_frames") != row["steps"]
            or row.get("depth_video_fps") != 50
            or row.get("video_frames") != 1 + (row["steps"] + 1) // 2
            or type(row.get("terminated")) is not bool
            or type(row.get("goal_reached")) is not bool
            or type(row.get("forbidden_contact_count")) is not int
            or row["forbidden_contact_count"] < 0
            or type(row.get("forbidden_contact_peak_n")) not in (int, float)
            or any(
                type(row.get(k)) not in (int, float)
                for k in (
                    "max_tilt_rad",
                    "min_root_clearance_m",
                    "max_torque_ratio",
                    "max_joint_excess_rad",
                    "max_step_displacement_m",
                    "max_drift_m",
                    "final_speed_m_s",
                )
            )
        ):
            raise ValueError("case metric/clock/media shape differs from frozen native protocol")
        constraints = (
            row["max_tilt_rad"] <= 0.8
            and row["min_root_clearance_m"] >= 0.4
            and row["max_torque_ratio"] <= 1.001
            and row["max_joint_excess_rad"] <= 0.001
            and row["max_step_displacement_m"] < 0.3
            and row["forbidden_contact_count"] == 0
        )
        if row.get("constraints_ok") is not constraints:
            raise ValueError("case constraints disagree with raw metric summary")
        if case["regression"]:
            expected_success = bool(
                not row["terminated"]
                and constraints
                and row["steps"] == 1000
                and row["max_drift_m"] <= (0.25 if case["task"] == "stand" else 0.75)
                and row["final_speed_m_s"] <= 0.1
            )
        else:
            expected_success = bool(
                not row["terminated"]
                and constraints
                and row["goal_reached"]
                and route["progress_m"] >= plan["suite_spec"]["route_distance_m"]
                and route["max_lateral_m"] <= 0.6
            )
        if (
            row["success"] is not expected_success
            or route.get("success") is not expected_success
            or not (row["terminated"] or row["goal_reached"] or row["steps"] == 1000)
        ):
            raise ValueError("standing, timeout or failed constraints cannot count as route success")
    complete = keys == expected
    if not complete and not allow_partial:
        raise ValueError("PBFM evaluation is incomplete: missing frozen case or perception condition")
    normal = [row for row in rows if row["condition"] == "normal"]
    success = sum(row["success"] for row in normal)
    occluded = [row for row in rows if row["condition"] == "occluded"]
    occluded_success = sum(row["success"] for row in occluded)
    regressions = sum(row["success"] for row in normal if row["case_id"].startswith("flat-regression"))
    return {
        "planned": len(expected),
        "completed": len(keys),
        "valid": len(keys),
        "invalid": 0,
        "infra_failed": 0,
        "cancelled": 0,
        "not_started": len(expected) - len(keys),
        "cases": len(normal),
        "perception_rollouts": len(rows),
        "normal_success_rate": success / len(plan["cases"]),
        "occluded_success_rate": occluded_success / len(plan["cases"]),
        "normal_minus_occluded_success_rate": (success - occluded_success) / len(plan["cases"]),
        "flat_regressions_success": regressions,
        "quality_passed": complete and success / len(plan["cases"]) >= 0.75 and regressions == 2,
        "media_categories_absent": [
            kind
            for kind, value in (
                ("success", sum(row["success"] for row in rows)),
                ("failure", sum(not row["success"] for row in rows)),
            )
            if value == 0
        ],
    }


def validate_action_plan(plan: dict) -> dict:
    from eb.pbfm_data import EXPORT_OPERATION

    if plan.get("operation") == EVAL_OPERATION:
        return validate_eval_plan(plan)
    if (
        plan.get("operation") != EXPORT_OPERATION
        or plan.get("schema_version") != "eb/pbfm-export-plan/v1"
        or not all(
            hash_value(plan.get(key))
            for key in ("model_hash", "dataset_manifest", "prompt_manifest_sha256", "prompt_file_sha256")
        )
        or not hash_value(plan.get("image_digest"), prefixed=True)
        or type(plan.get("motion_id")) is not int
        or plan["motion_id"] < 0
        or not plan.get("model_files")
        or not finite(plan)
        or plan.get("hash") != fingerprint({k: v for k, v in plan.items() if k != "hash"})
    ):
        raise ValueError("incomplete frozen PBFM export identity")
    return plan


def action_spec(plan: dict):
    from eb.domain.job import JobSpec
    from eb.pbfm_data import ENV_ID

    validate_action_plan(plan)
    evaluation = plan["operation"] == EVAL_OPERATION
    return JobSpec(
        kind="eval.sim" if evaluation else "data.convert",
        embodiment_env_id=ENV_ID,
        task_id="tsk_rpo_pbfm_terrain_locomotion_v1" if evaluation else None,
        params={"operation": plan["operation"], "pbfm_plan": plan},
        image=plan["image_digest"],
        inputs={"base_model": plan["model_path"]},
        resources={"gpus": 1, "cpus": 4, "memory_gb": 32, "shm_gb": 4},
        runtime={"network": "none"},
        policy={"timeout_h": 2, "max_retries": 0},
    )


def validate_snapshot_refs(refs: dict) -> None:
    from eb.settings import get_settings

    for name in ("pbfm", "asset"):
        ref = refs.get(name) or {}
        digest = ref.get("snapshot_hash")
        if not hash_value(digest, prefixed=True):
            raise ValueError("PBFM source snapshot is unfrozen")
        expected = get_settings().data_root / "snapshots" / digest.removeprefix("sha256:") / "code"
        if (
            ref.get("kind") != "content_snapshot"
            or ref.get("path") != str(expected)
            or ref.get("manifest_path") != str(expected.parent / "manifest.json")
            or not expected.is_dir()
            or any(p.is_symlink() for p in (expected, *expected.parents))
        ):
            raise ValueError("PBFM source must be an owned canonical content snapshot")


def render(spec, run_id):
    import json
    from pathlib import Path

    from eb.adapters.base import RenderedJob
    from eb.domain.job import CodeRefs
    from eb.pbfm_data import ENV_ID, OPERATION
    from eb.pbfm_data import render as data_render
    from eb.settings import get_settings

    plan = spec.params["pbfm_plan"]
    validate_snapshot_refs(plan["source_refs"])
    if plan["operation"] == OPERATION:
        return data_render(spec, run_id)
    validate_action_plan(plan)
    expected = action_spec(plan)
    # Submission persists CodeRefs() when the preview leaves code_refs unset.
    actual = spec.model_copy(update={"code_refs": None}) if spec.code_refs == CodeRefs() else spec
    if actual.model_dump(mode="json", exclude={"name"}) != expected.model_dump(mode="json", exclude={"name"}):
        raise ValueError("PBFM runtime differs from frozen action spec")
    if spec.embodiment_env_id != ENV_ID:
        raise ValueError("PBFM model interface differs")
    output = get_settings().data_root / "runs" / run_id / "outputs"
    model = Path(plan["model_path"])
    from eb.pbfm_io import safe_file

    for row in plan["model_files"]:
        safe_file(model, row["path"])
    mounts = [
        (str(model), "/model", "ro"),
        (plan["prompt_path"], "/prompt-data", "ro"),
        (str(output), "/outputs", "rw"),
    ]
    for name, target in (("pbfm", "/sources/pbfm"), ("asset", "/robot-assets")):
        ref = plan["source_refs"][name]
        mounts.extend(
            [(ref["path"], target, "ro"), (ref["manifest_path"], f"/sources/{name}-manifest.json", "ro")]
        )
    evaluation = plan["operation"] == EVAL_OPERATION
    if evaluation:
        mounts.append((plan["export_path"], "/export", "ro"))
    return RenderedJob(
        image=spec.image,
        command=[
            "python",
            "-m",
            "eb.runners.pbfm_eval" if evaluation else "eb.runners.pbfm_export",
            "--plan-json",
            json.dumps(plan),
        ],
        mounts=mounts,
        env={
            "PYTHONUNBUFFERED": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "EB_RUN_ID": run_id,
            "MUJOCO_GL": "egl",
            "PYOPENGL_PLATFORM": "egl",
            "OMP_NUM_THREADS": "1",
            "NVIDIA_DRIVER_CAPABILITIES": "compute,utility,graphics",
        },
        workdir="/outputs",
        gpus=1,
        shm_gb=4,
        writable_paths=[],
    )


async def validate_submission(session, spec) -> None:
    from eb.api.routes_pbfm import ActionIn, DataIn, build_action_plan, build_data_plan
    from eb.pbfm_data import OPERATION, spec_for
    from eb.store.models import Dataset

    proposed = spec.params["pbfm_plan"]
    validate_snapshot_refs(proposed["source_refs"])
    if proposed.get("operation") == OPERATION:
        dataset = await session.get(Dataset, spec.params.get("dataset_id"))
        if (
            not dataset
            or dataset.source_kind != "pbfm_rpo_motion"
            or dataset.meta.get("data_plan") != proposed
        ):
            raise ValueError("PBFM derivation requires a separately registered Dataset")
        if proposed.get("input_mode") == "registered_ufo_groups":
            from eb.api.routes_pbfm import UfoDataIn
            from eb.pbfm_ufo_bridge import build_registered_ufo_plan

            bridge = proposed["source_manifest"]["bridge"]
            plan = await build_registered_ufo_plan(
                UfoDataIn(
                    name=dataset.name,
                    parent_dataset_id=proposed["parent_dataset_id"],
                    split_groups=bridge["split_groups"],
                    excluded_groups=bridge["excluded_groups"],
                ),
                session,
            )
        else:
            plan = await build_data_plan(
                DataIn(
                    name=dataset.name,
                    parent_version_id=proposed["parent_version_id"],
                    source_manifest=proposed["source_manifest_path"],
                ),
                session,
            )
        expected = spec_for(plan, dataset.id)
    else:
        plan = await build_action_plan(
            proposed["model_version_id"],
            proposed["operation"],
            ActionIn(motion_id=proposed["motion_id"]),
            session,
        )
        expected = action_spec(plan)
    if spec.model_dump(mode="json", exclude={"name"}) != expected.model_dump(mode="json", exclude={"name"}):
        raise ValueError("PBFM submission differs from immutable current inputs")


async def verify_output_index(root, files: list[dict], required: set[str]) -> dict:
    from eb.pbfm_io import safe_file
    from eb.store.ebd import EbdClient

    if not files or len(files) > 10000:
        raise ValueError("invalid bounded PBFM evidence file index")
    indexed = {}
    for row in files:
        name = relative(row.get("path"))
        empty_package_marker = (
            name.startswith("policy/source/pbfm/")
            and name.endswith("/__init__.py")
            and row.get("bytes") == 0
            and row.get("sha256") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        )
        if (
            name in indexed
            or type(row.get("bytes")) is not int
            or (row["bytes"] < 1 and not empty_package_marker)
            or not hash_value(row.get("sha256"))
        ):
            raise ValueError("PBFM evidence requires unique complete files")
        path = safe_file(root, name)
        info = await EbdClient().scan(str(path))
        samples = info.get("samples") or []
        if (
            info.get("files") != 1
            or len(samples) != 1
            or samples[0].get("size") != row["bytes"]
            or samples[0].get("sha256") != row["sha256"]
        ):
            raise ValueError(f"PBFM evidence integrity differs: {name}")
        indexed[name] = row
    if not required <= indexed.keys():
        raise ValueError("PBFM evidence is incomplete")
    return indexed


async def collect_pbfm(session, job, run, outputs):
    from sqlalchemy import select

    from eb.ids import new_id
    from eb.pbfm_data import ENV_ID, EXPORT_OPERATION, OPERATION, SCHEMA, validate_data_plan
    from eb.pbfm_io import read_json, safe_file
    from eb.store.models import Artifact, Dataset, DatasetVersion, EvalReport, LineageEdge

    plan = job.spec["params"]["pbfm_plan"]
    operation = plan["operation"]
    receipt_name = (
        "data-receipt.json"
        if operation == OPERATION
        else "export-receipt.json"
        if operation == EXPORT_OPERATION
        else "eval-receipt.json"
    )
    receipt = read_json(safe_file(outputs, receipt_name))
    schema = (
        "eb/pbfm-data-receipt/v1"
        if operation == OPERATION
        else "eb/pbfm-export-receipt/v1"
        if operation == EXPORT_OPERATION
        else "eb/pbfm-eval-receipt/v1"
    )
    if (
        receipt.get("schema_version") != schema
        or receipt.get("run_id") != run.id
        or receipt.get("operation") != operation
        or receipt.get("plan_hash") != plan["hash"]
    ):
        raise ValueError("PBFM receipt identity differs")
    if operation == OPERATION:
        validate_data_plan(plan)
        required = {
            "dataset-index.json",
            "data/train.pkl",
            "data/manifest.json",
            "holdout/validation.pkl",
            "holdout/manifest.json",
            "acceptance/acceptance.pkl",
            "acceptance/manifest.json",
        }
    elif operation == EXPORT_OPERATION:
        validate_action_plan(plan)
        required = {
            "policy.zip",
            "policy/codec.json",
            "policy/normalizers.json",
            "policy/robot/rpo.xml",
            "policy/robot/robot.yaml",
            "policy/camera.json",
            "policy/terrain.json",
            "policy/task.json",
            "policy/sources.json",
            "policy/requirements.freeze.txt",
            "policy/pbfm_codec.py",
            "policy/paired/torch.npz",
            "policy/paired/onnx.npz",
            "policy/paired/summary.json",
            "policy/policy.onnx",
            "policy/policy.meta.json",
            "policy/tracking-prompt.npz",
            "policy/prompt.json",
            "policy/manifest.json",
        }
    else:
        validate_eval_plan(plan)
        required = {"cases.json", "cases.jsonl", "metrics.json", "media-index.json", "analysis/terrain.svg"}
    indexed = await verify_output_index(outputs, receipt.get("files") or [], required)
    kind = (
        "pbfm_motion"
        if operation == OPERATION
        else "pbfm_policy"
        if operation == EXPORT_OPERATION
        else "pbfm_terrain_evidence"
    )
    meta = {
        "receipt": receipt,
        "plan": plan,
        "model_version_id": plan.get("model_version_id"),
        "schema_version": schema,
    }
    metrics = {}
    if operation == OPERATION:
        report = read_json(outputs / "dataset-index.json")
        if (
            report.get("dataset_id") != job.spec["params"]["dataset_id"]
            or receipt.get("dataset_id") != report["dataset_id"]
            or report.get("storage_ref") != str(outputs / "data")
            or report.get("format") != "pbfm_rpo_motion"
        ):
            raise ValueError("PBFM Dataset report identity differs")
        ancestry = set()
        for split, root, filename in (
            ("train", "data", "train.pkl"),
            ("validation", "holdout", "validation.pkl"),
            ("acceptance", "acceptance", "acceptance.pkl"),
        ):
            manifest = read_json(outputs / root / "manifest.json")
            ids = set(manifest.get("ancestor_ids") or [])
            expected_ancestry = {
                ancestor
                for item in plan["source_manifest"]["splits"][split]
                for ancestor in item["ancestor_ids"]
            }
            if (
                manifest.get("schema_version") != SCHEMA
                or manifest.get("dataset_id") != report["dataset_id"]
                or manifest.get("split") != split
                or not ids
                or ids != expected_ancestry
                or ancestry & ids
                or manifest.get("parent_version_id") != plan["parent_version_id"]
                or manifest.get("source_manifest_sha256") != plan["source_manifest_sha256"]
                or manifest.get("file_sha256") != indexed[root + "/" + filename]["sha256"]
                or manifest.get("joint_order") != plan["source_manifest"]["joint_order"]
                or manifest.get("native_loader", {}).get("all_motions_loaded") is not True
                or manifest.get("fk", {}).get("verified") is not True
                or manifest.get("fk", {}).get("frames_checked") != manifest.get("frames")
            ):
                raise ValueError("PBFM split ancestry, hashes or native FK evidence differs")
            ancestry.update(ids)
        train_manifest = read_json(outputs / "data/manifest.json")
        if train_manifest.get("holdout_manifests") != {
            "validation": indexed["holdout/manifest.json"]["sha256"],
            "acceptance": indexed["acceptance/manifest.json"]["sha256"],
        }:
            raise ValueError("PBFM training manifest lacks immutable held-out split bindings")
        validation = report.get("validation") or {}
        if (
            validation.get("scope") != "pbfm_rpo_motion_v1"
            or validation.get("status") != "validated"
            or validation.get("training_ready") is not True
            or validation.get("manifest_sha256") != indexed["data/manifest.json"]["sha256"]
            or validation.get("file_sha256") != indexed["data/train.pkl"]["sha256"]
        ):
            raise ValueError("PBFM data readiness differs from evidence")
        dataset = await session.get(Dataset, report["dataset_id"])
        if not dataset or dataset.meta.get("data_plan") != plan:
            raise ValueError("PBFM derived Dataset registration differs")
        dataset.status = "validated"
        dataset.dest_path = str(outputs / "data")
        dataset.summary_stats = report
        dataset.meta = {
            **dataset.meta,
            "validation": validation,
            "validation_run_id": run.id,
            "usage": "train",
        }
        dataset.total_bytes = indexed["data/train.pkl"]["bytes"]
        version = await session.scalar(
            select(DatasetVersion).where(
                DatasetVersion.dataset_id == dataset.id, DatasetVersion.version == "pbfm-v1"
            )
        )
        if version and version.content_hash != "sha256:" + validation["manifest_sha256"]:
            raise ValueError("immutable PBFM DatasetVersion changed")
        if not version:
            version = DatasetVersion(
                id=new_id("dsv"),
                dataset_id=dataset.id,
                version="pbfm-v1",
                compatible_envs=[ENV_ID],
                content_hash="sha256:" + validation["manifest_sha256"],
                stats={**report, "producer_run": run.id},
            )
            session.add(version)
            await session.flush()
            session.add(
                LineageEdge(
                    src_type="dataset_version",
                    src_id=plan["parent_version_id"],
                    dst_type="dataset_version",
                    dst_id=version.id,
                    relation="derived",
                    run_id=run.id,
                )
            )
        metrics = {"dataset_id": dataset.id, "dataset_version_id": version.id, "training_ready": True}
    elif operation == EXPORT_OPERATION:
        policy = read_json(outputs / "policy/manifest.json")
        prompt = read_json(outputs / "policy/prompt.json")
        paired = read_json(outputs / "policy/paired/summary.json")
        if (
            policy.get("schema_version") != "eb/rpo-pbfm-policy/v1"
            or policy.get("model_hash") != plan["model_hash"]
            or policy.get("model_version_id") != plan["model_version_id"]
            or policy.get("trace_verified") is not True
            or policy.get("onnx_sha256") != indexed["policy/policy.onnx"]["sha256"]
            or prompt.get("model_hash") != plan["model_hash"]
            or prompt.get("motion_id") != plan["motion_id"]
            or prompt.get("source_manifest_sha256") != plan["prompt_manifest_sha256"]
            or paired.get("schema_version") != "eb/pbfm-paired-closed-loop/v1"
            or paired.get("complete") is not True
            or paired.get("model_hash") != plan["model_hash"]
            or paired.get("frozen_plan") != plan["paired_rollout"]
            or paired != policy.get("paired_closed_loop")
            or any(
                type(paired.get(key)) is not int or not 1 <= paired[key] <= 32
                for key in ("torch_steps", "onnx_steps", "shared_prefix_steps")
            )
        ):
            raise ValueError("PBFM export lacks current-model prompt or real-trace parity")
        package_files = policy.get("files") or []
        if not package_files:
            raise ValueError("PBFM inference package lacks a portable asset/codec/source index")
        for entry in package_files:
            name = "policy/" + relative(entry["path"])
            if indexed.get(name) != {**entry, "path": name}:
                raise ValueError("portable inference package index differs from verified output bytes")
        meta["schema_version"] = "eb/rpo-pbfm-policy/v1"
    else:
        case_file = read_json(outputs / "cases.json")
        rows = case_file.get("cases") or []
        metrics = validate_case_rows(plan, rows)
        required_media = {row[key] for row in rows for key in ("video", "depth", "depth_video", "trace")}
        if not required_media <= indexed.keys():
            raise ValueError("case result references missing real media")
        if read_json(outputs / "metrics.json") != metrics or receipt.get("metrics") != metrics:
            raise ValueError("PBFM derived metrics differ from complete case evidence")
        existing = await session.scalar(select(EvalReport).where(EvalReport.run_id == run.id))
        if not existing:
            existing = EvalReport(
                id=new_id("ev"),
                suite_id=SUITE_ID,
                model_version_id=plan["model_version_id"],
                run_id=run.id,
                metrics=metrics,
                gate_results={
                    "qualification": "diagnostic",
                    "formal_acceptance": False,
                    "validity": "complete",
                    "plan_hash": plan["hash"],
                    "receipt": receipt,
                },
            )
            session.add(existing)
            await session.flush()
            session.add(
                LineageEdge(
                    src_type="model_version",
                    src_id=plan["model_version_id"],
                    dst_type="eval_report",
                    dst_id=existing.id,
                    relation="evaluated",
                    run_id=run.id,
                )
            )
        metrics = {**metrics, "eval_report_id": existing.id, "qualification": "diagnostic"}
    digest = fingerprint(receipt)
    artifact = await session.scalar(select(Artifact).where(Artifact.run_id == run.id, Artifact.kind == kind))
    if artifact and artifact.sha256 != digest:
        raise ValueError("immutable PBFM output changed on repeated collection")
    if not artifact:
        artifact = Artifact(
            id=new_id("art"),
            kind=kind,
            sha256=digest,
            size_bytes=sum(r["bytes"] for r in indexed.values()),
            storage=str(outputs / "policy") if operation == EXPORT_OPERATION else str(outputs),
            run_id=run.id,
            pinned=True,
            meta=meta,
        )
        session.add(artifact)
        await session.flush()
        session.add(
            LineageEdge(
                src_type="run",
                src_id=run.id,
                dst_type="artifact",
                dst_id=artifact.id,
                relation="produced",
                run_id=run.id,
            )
        )
        if plan.get("model_version_id"):
            session.add(
                LineageEdge(
                    src_type="model_version",
                    src_id=plan["model_version_id"],
                    dst_type="artifact",
                    dst_id=artifact.id,
                    relation="derived",
                    run_id=run.id,
                )
            )
    run.metrics_summary = {**(run.metrics_summary or {}), **metrics, "artifact_id": artifact.id}
