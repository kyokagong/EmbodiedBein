"""Frozen identity and acceptance scope of the official RPO-Flat recipe."""

import hashlib
import json
from pathlib import Path

ROBOLAB_REVISION = "6b1c3d9988497c8961dcba77892de32edc1770e1"
FLAT_TEMPLATE = "rpo_v2_flat_ppo"
FLAT_ENV = "ee_rpo_v2_flat_23dof_v1"
FLAT_TASK = "tsk_rpo_flat"
FLAT_SUITE = "suite_rpo_v2_flat_isaac_velocity_commands_v1"


def flat_warm_start_receipt(model, artifact) -> dict:
    """Bind a published Flat package, retaining strict resume compatibility."""
    card, manifest = model.card or {}, (artifact.meta or {}).get("manifest") or {}
    files = {item["path"]: item for item in manifest.get("files", [])}
    weight = files.get("model/model_state_dict/full_weights.pt") or {}
    normalization = files.get("model/actor-normalization.json") or {}
    if (model.embodiment_env_id != FLAT_ENV or model.family_id != "mf_rpo_v2_flat"
            or card.get("schema_version") != "eb/rpo-flat-model/v1"
            or card.get("template_id") != FLAT_TEMPLATE
            or card.get("network") != {"hidden_dims": [512, 256, 128], "activation": "elu"}
            or card.get("normalization") != "empirical"
            or (card.get("official_source") or {}).get("robolab_revision") != ROBOLAB_REVISION
            or artifact.kind != "rlinf_checkpoint" or card.get("artifact_sha256") != artifact.sha256
            or manifest.get("status") != "complete" or manifest.get("run_id") != card.get("run_id")
            or manifest.get("step") != card.get("step") or manifest.get("template_id") != FLAT_TEMPLATE
            or not normalization.get("sha256") or not weight.get("sha256")
            or type(weight.get("bytes")) is not int or weight["bytes"] <= 0):
        raise ValueError("warm start 需要同接口、同官方来源和完整归一化的已发布 Flat 模型")
    return {"schema_version": "eb/warm-start/v1", "status": "validated", "mode": "warm_start",
            "model_id": model.id, "source_run_id": card["run_id"], "source_step": card["step"],
            "source_cumulative_updates": card.get("cumulative_updates", card["step"]),
            "weight_sha256": weight["sha256"], "weight_bytes": weight["bytes"],
            "normalization_sha256": normalization["sha256"],
            "state_coverage": {"model": "restored", "normalization": "restored",
                               "optimizer": "reset", "scheduler": "reset", "rng_per_rank": "reset",
                               "environment": "reset", "rollout": "reset"}}


def flat_suite_spec() -> dict:
    return {
        "benchmark": "rpo_flat", "backend": "isaac", "policy": "deterministic_onnx",
        "criterion": "velocity_commands", "robolab_revision": ROBOLAB_REVISION,
        "required_steps": 1000, "policy_hz": 50,
        "command_ranges": {"lin_vel_x": [-0.6, 1.0], "lin_vel_y": [-0.5, 0.5],
                           "ang_vel_z": [-1.57, 1.57], "heading": [-3.141592653589793, 3.141592653589793]},
        "command_resampling_s": [10.0, 10.0], "standing_fraction": 0.2,
        "heading_command": True, "terrain": {"flat": 0.7, "random_rough": 0.3},
        "velocity_frame": "yaw_aligned", "yaw_rate_frame": "world_z",
        "randomization": "official_RPOFlatEnvCfg", "velocity_vector_rmse_max": 0.2,
        "yaw_rmse_max": 0.3, "fall_height_m": 0.5, "fall_tilt_rad": 0.8,
        "planned_cases": 100, "minimum_successes": 80,
    }


def validate_flat_eval(output: Path, plan: dict, run_id: str) -> dict:
    """Validate small case receipts and hashes; leave large traces in the data plane."""
    from eb.onnx_io import sha256_file
    from eb.runners.rpo_flat_artifact import summarize_cases

    def checked_file(relative: str, limit: int) -> Path:
        path = output / relative
        if (output.is_symlink() or any(p.is_symlink() for p in path.parents if p != output.parent)
                or path.is_symlink() or not path.is_file() or path.stat().st_size > limit
                or not path.resolve().is_relative_to(output.resolve())):
            raise ValueError(f"RPO-Flat evidence unavailable or unsafe: {relative}")
        return path

    def load(relative: str, limit: int = 1024 * 1024) -> dict:
        return json.loads(checked_file(relative, limit).read_text(),
                          parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))

    expected_hash = "sha256:" + hashlib.sha256(json.dumps(
        {k:v for k,v in plan.items() if k != "hash"}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    seeds = plan.get("seeds") or []
    if (plan.get("hash") != expected_hash or plan.get("schema_version") != "eb/rpo-flat-eval-plan/v1"
            or plan.get("suite_id") != FLAT_SUITE or plan.get("suite_spec") != flat_suite_spec()
            or not seeds or len(set(seeds)) != len(seeds)):
        raise ValueError("RPO-Flat frozen evaluation plan changed")
    receipt = load("eval-manifest.json", max(
        1024 * 1024, min(len(seeds), flat_suite_spec()["planned_cases"]) * 64 * 1024))
    if (receipt.get("schema_version") != "eb/rpo-flat-eval/v1" or receipt.get("run_id") != run_id
            or receipt.get("plan_hash") != plan["hash"] or receipt.get("seeds") != seeds
            or receipt.get("backend") != "isaac" or receipt.get("criterion") != "velocity_commands"
            or receipt.get("required_steps") != 1000 or receipt.get("official_revision") != ROBOLAB_REVISION
            or receipt.get("record_video") is not plan.get("record_video")
            or receipt.get("video_quality", "standard") != plan.get("video_quality", "standard")
            or receipt.get("velocity_frame") != "yaw_aligned" or receipt.get("yaw_rate_frame") != "world_z"):
        raise ValueError("RPO-Flat receipt differs from frozen plan")
    exported = load("exported/export-manifest.json")
    onnx = exported.get("onnx") or {}
    version = exported.get("schema_version")
    numerical_valid = (
        isinstance(onnx.get("max_abs_error"), (int, float))
        and 0 <= onnx["max_abs_error"] < float("inf")
        and ((version == "eb/rpo-flat-export/v1" and onnx["max_abs_error"] <= 1e-4)
             or (version in {"eb/rpo-flat-export/v2", "eb/rpo-flat-export/v3"} and onnx.get("error_atol") == 1e-4
                 and onnx.get("error_rtol") == 1e-5
                 and isinstance(onnx.get("max_error_ratio"), (int, float))
                 and 0 <= onnx["max_error_ratio"] <= 1
                 and isinstance(onnx.get("max_clipped_action_error"), (int, float))
                 and 0 <= onnx["max_clipped_action_error"] <= (4e-4 if version.endswith("v3") else 1e-4)
                 and (version.endswith("v2") or (
                     onnx.get("max_joint_target_error_rad") == 0.25 * onnx["max_clipped_action_error"]
                     and (exported.get("golden") or {}).get("batch_sizes") == [1, 11])))))
    if (not numerical_valid or exported.get("status") != "verified"
            or onnx != receipt.get("onnx") or onnx.get("obs") != 780 or onnx.get("act") != 23
            or sha256_file(checked_file("exported/policy.onnx", 32 * 1024 * 1024)) != onnx.get("sha256")):
        raise ValueError("RPO-Flat measured policy export changed")
    for key, name in (("golden", "golden-traces.npz"), ("normalization", "actor-normalization.npz")):
        evidence = exported.get(key) or {}
        if (evidence.get("path") != name
                or sha256_file(checked_file("exported/" + name, 8 * 1024 * 1024)) != evidence.get("sha256")):
            raise ValueError("RPO-Flat export evidence changed")
    rows = [json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
            for line in checked_file("eval-cases.jsonl", 1024 * 1024).read_text().splitlines() if line]
    if [row.get("seed") for row in rows] != seeds:
        raise ValueError("RPO-Flat case seeds differ from plan")
    for row in rows:
        seed = row["seed"]
        if (row.get("run_id") != run_id or row.get("plan_hash") != plan["hash"]
                or row.get("task_id") != FLAT_TASK or row.get("case_id") != f"rpo-flat-seed-{seed}"
                or type(row.get("completed")) is not bool or type(row.get("success")) is not bool
                or row.get("status") != ("success" if row["success"] else "task_failure" if row["completed"] else "infra_failure")
                or type(row.get("actual_steps")) is not int or not 0 <= row["actual_steps"] <= 1000
                or row.get("episode_len") != row["actual_steps"]
                or type(row.get("metric_steps")) is not int or not 0 <= row["metric_steps"] <= row["actual_steps"]
                or any(type(row.get(key)) is not bool for key in ("fall", "nan", "hard_limit"))):
            raise ValueError("RPO-Flat case evidence malformed")
        if not row["completed"]:
            continue
        for key, name, limit in (("trace", f"seed_{seed}.jsonl.gz", 8 * 1024 * 1024),
                                 ("initial_state", f"seed_{seed}.initial.json", 64 * 1024)):
            evidence = row.get(key) or {}
            relative = "traces/" + name
            if (evidence.get("path") != relative
                    or sha256_file(checked_file(relative, limit)) != evidence.get("sha256")):
                raise ValueError("RPO-Flat actual command/state evidence changed")
        if (row["trace"].get("policy_steps") != row["actual_steps"]
                or row["trace"].get("metric_steps") != row["metric_steps"]):
            raise ValueError("RPO-Flat trace step counts changed")
        if plan.get("record_video"):
            if row.get("video_rel") != f"seed_{seed}/0.mp4":
                raise ValueError("RPO-Flat case video unavailable")
            checked_file("video/eval/" + row["video_rel"], 128 * 1024 * 1024)
    measured = summarize_cases(rows, len(seeds))
    if any(receipt.get(key) != measured[key] for key in ("cases", "metrics")):
        raise ValueError("RPO-Flat aggregate disagrees with individual cases")
    standard = plan.get("profile") == "standard" and not plan.get("seed_override")
    qualified = (standard and type(plan.get("checkpoint_step")) is int and plan["checkpoint_step"] >= 500
                 and seeds == list(range(50000, 50100))
                 and measured["cases"]["completed"] == 100 and measured["metrics"]["successes"] >= 80)
    return {"receipt": receipt, **measured,
            "qualification": ("not_requested" if not standard else "incomplete"
                              if measured["cases"]["infra_failed"] else "pass" if qualified else "fail"),
            "validity": "complete" if not measured["cases"]["infra_failed"] else "incomplete"}
