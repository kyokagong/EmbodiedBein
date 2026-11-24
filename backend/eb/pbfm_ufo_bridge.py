"""Controlled group selection from a registered UFO producer, without API pickle IO."""

from __future__ import annotations

from eb.pbfm_io import hash_value, relative

BRIDGE_SCHEMA = "eb/pbfm-registered-ufo-groups/v1"


def grouped_source_manifest(joints, source_files, split_groups, excluded_groups):
    from eb.pbfm_data import SOURCE_SCHEMA, SPLITS

    if set(split_groups) != set(SPLITS) or len(source_files) != 2:
        raise ValueError("UFO bridge requires train/validation/acceptance and exactly two producer files")
    inventory, paths, hashes = {}, set(), set()
    for item in source_files:
        path = relative(item.get("path"))
        groups = item.get("available_groups")
        parent_split = item.get("parent_split")
        if (
            parent_split not in {"train", "validation"}
            or path in paths
            or item.get("sha256") in hashes
            or not hash_value(item.get("sha256"))
            or type(item.get("bytes")) is not int
            or item["bytes"] <= 0
            or not isinstance(groups, list)
            or not groups
            or len(set(groups)) != len(groups)
            or any(not isinstance(g, str) or not g for g in groups)
            or inventory.keys() & set(groups)
        ):
            raise ValueError("UFO producer inventory is incomplete, duplicated or overlapping")
        paths.add(path)
        hashes.add(item["sha256"])
        inventory.update({group: item for group in groups})
    if {item["parent_split"] for item in source_files} != {"train", "validation"}:
        raise ValueError("UFO bridge requires both original producer splits")
    assigned = set()
    splits = {}
    for split in SPLITS:
        groups = split_groups[split]
        if (
            not isinstance(groups, list)
            or not groups
            or len(set(groups)) != len(groups)
            or any(not isinstance(g, str) or not g or "unknown" in g.lower() for g in groups)
            or assigned & set(groups)
            or not set(groups) <= inventory.keys()
        ):
            raise ValueError("PBFM groups require nonempty disjoint known source ancestry")
        assigned.update(groups)
        rows = []
        for item in source_files:
            selected = sorted(set(groups) & set(item["available_groups"]))
            if selected:
                if item["parent_split"] != ("train" if split == "train" else "validation"):
                    raise ValueError("original UFO holdout cannot enter PBFM training or vice versa")
                rows.append(
                    {
                        **item,
                        "format": "ufo_grouped_pickle",
                        "record_groups": selected,
                        "ancestor_ids": selected,
                    }
                )
        splits[split] = rows
    if (
        not isinstance(excluded_groups, list)
        or len(set(excluded_groups)) != len(excluded_groups)
        or any(not isinstance(g, str) or not g for g in excluded_groups)
        or assigned & set(excluded_groups)
        or assigned | set(excluded_groups) != inventory.keys()
    ):
        raise ValueError("every original source group must be selected once or explicitly excluded")
    return {
        "schema_version": SOURCE_SCHEMA,
        "robot_name": "rpo",
        "coordinate_system": "z_up",
        "root_quat_order": "xyzw",
        "dof_unit": "rad",
        "joint_order": joints,
        "splits": splits,
        "bridge": {
            "schema_version": BRIDGE_SCHEMA,
            "source_files": source_files,
            "split_groups": split_groups,
            "excluded_groups": sorted(excluded_groups),
            "qualification": "diagnostic",
            "formal_acceptance": False,
        },
    }


def validate_grouped_source(manifest):
    bridge = manifest.get("bridge") or {}
    if bridge.get("schema_version") != BRIDGE_SCHEMA or manifest != grouped_source_manifest(
        manifest.get("joint_order"),
        bridge.get("source_files", []),
        bridge.get("split_groups", {}),
        bridge.get("excluded_groups", []),
    ):
        raise ValueError("grouped UFO source differs from frozen producer inventory or selection")
    return manifest


def select_grouped_records(records, row, split):
    if not isinstance(records, dict) or not records:
        raise ValueError("UFO producer pickle requires actual motion records")
    actual = {(record.get("metadata") or {}).get("source_group") for record in records.values()}
    if actual != set(row["available_groups"]):
        raise ValueError("actual UFO producer inventory differs from frozen source groups")
    selected = {}
    for key, record in records.items():
        metadata = record.get("metadata") or {}
        group = metadata.get("source_group")
        if group not in row["record_groups"]:
            continue
        if (
            not isinstance(key, str)
            or not key
            or not hash_value(metadata.get("source_sha256"))
            or not metadata.get("source_path")
            or not metadata.get("motion_key")
            or metadata.get("ancestor_id", group) != group
        ):
            raise ValueError("selected UFO record lacks original source ancestry")
        selected[key] = {
            **record,
            "metadata": {
                **metadata,
                "ancestor_id": group,
                "pbfm_split": split,
                "parent_ufo_split": row["parent_split"],
            },
        }
    if {record["metadata"]["source_group"] for record in selected.values()} != set(row["record_groups"]):
        raise ValueError("selected UFO source groups have no actual records")
    return selected


async def build_registered_ufo_plan(body, session):
    import asyncio
    from pathlib import Path

    from sqlalchemy import select

    from eb.api.routes_pbfm import image_digest
    from eb.pbfm_data import OPERATION, PLAN_SCHEMA, validate_source_manifest
    from eb.pbfm_io import fingerprint, read_json, safe_file, sha256
    from eb.settings import get_settings
    from eb.store.ebd import EbdClient
    from eb.store.models import Dataset, DatasetVersion, Job, Run
    from eb.ufo_training import require_native_dataset

    parent = await session.get(Dataset, body.parent_dataset_id)
    if parent is None or parent.source_kind != "ufo_native_motion":
        raise ValueError("bridge requires a registered validated UFO native Dataset")
    report = require_native_dataset(parent)
    version = await session.scalar(
        select(DatasetVersion).where(
            DatasetVersion.dataset_id == parent.id, DatasetVersion.version == "native-v1"
        )
    )
    meta = parent.meta or {}
    run = await session.get(Run, meta.get("validation_run_id"))
    job = await session.get(Job, meta.get("validation_job_id"))
    if (
        version is None
        or version.content_hash != "sha256:" + report["manifest_sha256"]
        or not run
        or run.status != "finished"
        or not job
        or job.status != "succeeded"
        or run.job_id != job.id
        or (version.stats or {}).get("producer_run") != run.id
        or (job.spec.get("params") or {}).get("dataset_id") != parent.id
        or (job.spec.get("params") or {}).get("operation")
        not in {"prepare_ufo_native_motion", "prepare_ufo_rpo_grounded_a1_v1"}
    ):
        raise ValueError("UFO bridge requires an immutable native-v1 version and successful producer lineage")
    settings = get_settings()
    root = settings.data_root / "runs" / run.id / "outputs"
    if parent.dest_path != str(root / "data"):
        raise ValueError("UFO Dataset is outside its registered producer output")
    manifests = [read_json(safe_file(root, name + "/manifest.json")) for name in ("data", "holdout")]
    expected_holdout = (version.stats or {}).get("holdout", {}).get("manifest_sha256")
    if (
        not hash_value(expected_holdout)
        or expected_holdout != sha256(root / "holdout/manifest.json")
        or manifests[0].get("joint_order") != manifests[1].get("joint_order")
    ):
        raise ValueError("UFO holdout is not frozen by the registered DatasetVersion")
    client = EbdClient()
    sources = []
    for manifest, path, split in zip(
        manifests, ("data/train.pkl", "holdout/validation.pkl"), ("train", "validation")
    ):
        groups = manifest.get("groups")
        if (
            manifest.get("schema_version") != "eb/ufo-native-motion/v1"
            or manifest.get("dataset_id") != parent.id
            or manifest.get("split") != split
            or not isinstance(groups, list)
            or not groups
            or {row.get("source_group") for row in manifest.get("episodes", [])} != set(groups)
        ):
            raise ValueError("UFO producer manifest lacks original complete source-group inventory")
        path = safe_file(root, path)
        info = await client.scan(str(path))
        samples = info.get("samples") or []
        if (
            info.get("files") != 1
            or len(samples) != 1
            or samples[0].get("sha256") != manifest.get("file_sha256")
            or not samples[0].get("size")
        ):
            raise ValueError("UFO producer motion bytes changed after validation")
        sources.append(
            {
                "path": str(path.relative_to(root)),
                "sha256": samples[0]["sha256"],
                "bytes": samples[0]["size"],
                "parent_split": split,
                "available_groups": groups,
            }
        )
    source = validate_source_manifest(
        grouped_source_manifest(manifests[0]["joint_order"], sources, body.split_groups, body.excluded_groups)
    )
    snapshots = await asyncio.gather(
        client.snapshot(str(settings.pbfm_root)),
        client.snapshot(str(settings.assets_root / "roboparty/rpo-v2")),
    )
    refs = {
        name: {
            "kind": "content_snapshot",
            "path": snap["path"],
            "snapshot_hash": snap["content_hash"],
            "manifest_path": str(Path(snap["path"]).parent / "manifest.json"),
        }
        for name, snap in zip(("pbfm", "asset"), snapshots)
    }
    plan = {
        "schema_version": PLAN_SCHEMA,
        "operation": OPERATION,
        "input_mode": "registered_ufo_groups",
        "parent_version_id": version.id,
        "parent_dataset_id": parent.id,
        "parent_run_id": run.id,
        "parent_manifest_sha256": report["manifest_sha256"],
        "source_root": str(root),
        "source_manifest": source,
        "source_manifest_sha256": fingerprint(source),
        "source_refs": refs,
        "image_digest": await image_digest(),
    }
    plan["hash"] = fingerprint(plan)
    return plan
