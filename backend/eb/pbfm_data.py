"""Frozen RPO motion derivation. Compute and pickle loading happen in the Job."""

from __future__ import annotations

from pathlib import Path

from eb.pbfm_io import fingerprint, hash_value, read_json, relative, safe_file, sha256

OPERATION = "prepare_pbfm_rpo_motion"
EXPORT_OPERATION = "export_pbfm_policy"
SCHEMA = "eb/pbfm-rpo-motion/v1"
SOURCE_SCHEMA = "eb/pbfm-motion-source/v1"
PLAN_SCHEMA = "eb/pbfm-data-plan/v1"
ENV_ID = "ee_rpo_v2_pbfm_depth_23dof_v1"
IMAGE = "eb/rlinf-pbfm:rpo-depth-v1"
SPLITS = ("train", "validation", "acceptance")


def validate_source_manifest(manifest: dict) -> dict:
    if (
        manifest.get("schema_version") != SOURCE_SCHEMA
        or manifest.get("robot_name") != "rpo"
        or manifest.get("coordinate_system") != "z_up"
        or manifest.get("root_quat_order") != "xyzw"
        or manifest.get("dof_unit") != "rad"
    ):
        raise ValueError("PBFM data requires declared RPO, z-up, xyzw radian semantics")
    joints = manifest.get("joint_order", [])
    if len(joints) != 23 or len(set(joints)) != 23 or any(not isinstance(j, str) or not j for j in joints):
        raise ValueError("PBFM motion requires 23 unique RPO control joints")
    if set(manifest.get("splits", {})) != set(SPLITS):
        raise ValueError("PBFM data requires actual train/validation/acceptance splits")
    if manifest.get("bridge"):
        from eb.pbfm_ufo_bridge import validate_grouped_source

        return validate_grouped_source(manifest)
    ancestry, paths, hashes = set(), set(), set()
    for split in SPLITS:
        files = manifest["splits"][split]
        if not isinstance(files, list) or not files:
            raise ValueError("every PBFM split requires actual source files")
        split_ancestors = set()
        for row in files:
            path = relative(row.get("path"))
            ancestors = row.get("ancestor_ids")
            if (
                row.get("format") not in {"ufo_pickle", "robot_state_npz"}
                or not hash_value(row.get("sha256"))
                or type(row.get("bytes")) is not int
                or row["bytes"] <= 0
                or not isinstance(ancestors, list)
                or not ancestors
                or len(set(ancestors)) != len(ancestors)
                or any(not isinstance(a, str) or not a for a in ancestors)
            ):
                raise ValueError("source files require frozen hash, format, size and ancestry")
            if path in paths or row["sha256"] in hashes:
                raise ValueError("source paths or bytes are reused across splits")
            paths.add(path)
            hashes.add(row["sha256"])
            split_ancestors.update(ancestors)
        if ancestry & split_ancestors:
            raise ValueError("train/holdout source ancestry overlaps")
        ancestry.update(split_ancestors)
    return manifest


def validate_data_plan(plan: dict) -> dict:
    validate_source_manifest(plan.get("source_manifest") or {})
    if plan["source_manifest"].get("bridge") and (
        plan.get("input_mode") != "registered_ufo_groups"
        or not plan.get("parent_run_id")
        or plan.get("source_manifest_sha256") != fingerprint(plan["source_manifest"])
    ):
        raise ValueError("grouped UFO preparation requires the controlled registered producer bridge")
    if (
        plan.get("schema_version") != PLAN_SCHEMA
        or plan.get("operation") != OPERATION
        or not hash_value(plan.get("image_digest"), prefixed=True)
        or not hash_value(plan.get("parent_manifest_sha256"))
        or plan.get("hash") != fingerprint({k: v for k, v in plan.items() if k != "hash"})
    ):
        raise ValueError("invalid frozen PBFM data plan")
    return plan


def require_pbfm_dataset(dataset) -> dict:
    if dataset is None:
        raise ValueError("PBFM Dataset is missing")
    validation = (dataset.summary_stats or {}).get("validation") or {}
    root = Path(dataset.dest_path or "")
    if (
        dataset.format != "pbfm_rpo_motion"
        or dataset.source_kind != "pbfm_rpo_motion"
        or (dataset.meta or {}).get("usage", "train") != "train"
        or dataset.embodiment_env_id != ENV_ID
        or dataset.status not in {"ready", "validated"}
        or validation.get("scope") != "pbfm_rpo_motion_v1"
        or validation.get("status") != "validated"
        or validation.get("training_ready") is not True
    ):
        raise ValueError("Dataset lacks successful PBFM RPO preparation evidence")
    manifest_path = safe_file(root, "manifest.json", maximum=16 * 1024**2)
    manifest = read_json(manifest_path)
    if (
        manifest.get("schema_version") != SCHEMA
        or manifest.get("split") != "train"
        or manifest.get("dataset_id") != dataset.id
        or manifest.get("train_file") != "train.pkl"
        or validation.get("manifest_sha256") != sha256(manifest_path)
        or manifest.get("file_sha256") != validation.get("file_sha256")
        or manifest.get("native_loader", {}).get("all_motions_loaded") is not True
        or manifest.get("fk", {}).get("verified") is not True
        or manifest.get("joint_order") is None
        or len(manifest["joint_order"]) != 23
        or {p.name for p in root.iterdir()} != {"train.pkl", "manifest.json"}
    ):
        raise ValueError("PBFM train split, native loader, FK or immutable manifest differs")
    safe_file(root, "train.pkl")
    return {**manifest, "manifest_sha256": validation["manifest_sha256"], "storage_ref": str(root)}


def spec_for(plan: dict, dataset_id: str):
    from eb.domain.job import JobSpec

    validate_data_plan(plan)
    return JobSpec(
        kind="data.convert",
        embodiment_env_id=ENV_ID,
        image=plan["image_digest"],
        params={"operation": OPERATION, "dataset_id": dataset_id, "pbfm_plan": plan},
        resources={"gpus": 0, "cpus": 4, "memory_gb": 32, "shm_gb": 2},
        runtime={"network": "none"},
        policy={"timeout_h": 2, "max_retries": 0},
    )


def render(spec, run_id):
    import json

    from eb.adapters.base import RenderedJob
    from eb.settings import get_settings

    plan = validate_data_plan(spec.params["pbfm_plan"])
    if (
        spec.runtime != {"network": "none"}
        or spec.image != plan["image_digest"]
        or spec.embodiment_env_id != ENV_ID
    ):
        raise ValueError("PBFM data Job differs from frozen runtime")
    output = get_settings().data_root / "runs" / run_id / "outputs"
    refs = plan["source_refs"]
    mounts = list(
        dict.fromkeys(
            (str(Path(plan["source_root"]) / row["path"]), "/motion-source/" + row["path"], "ro")
            for split in plan["source_manifest"]["splits"].values()
            for row in split
        )
    )
    mounts.extend(
        [
            (refs["pbfm"]["path"], "/sources/pbfm", "ro"),
            (refs["asset"]["path"], "/robot-assets", "ro"),
            (str(output), "/outputs", "rw"),
        ]
    )
    mounts.extend(
        (ref["manifest_path"], f"/sources/{name}-manifest.json", "ro") for name, ref in refs.items()
    )
    return RenderedJob(
        image=spec.image,
        command=[
            "python",
            "-m",
            "eb.runners.pbfm_data",
            "--plan-json",
            json.dumps(plan),
            "--dataset-id",
            spec.params["dataset_id"],
            "--storage-ref",
            str(output / "data"),
        ],
        env={"PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1", "EB_RUN_ID": run_id},
        mounts=mounts,
        writable_paths=[],
        workdir="/outputs",
        gpus=0,
        shm_gb=2,
    )
