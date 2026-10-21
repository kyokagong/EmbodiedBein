"""Publish immutable DatasetVersion metadata for validated native motion jobs."""

from copy import deepcopy
from pathlib import Path

from sqlalchemy import select

from eb.ids import new_id
from eb.store.models import Dataset, DatasetVersion, LineageEdge
from eb.ufo_native import ENV_ID, read_json, safe_file
from eb.ufo_training import DATA_OPERATION, require_native_dataset


async def publish_native_dataset_version(session, row, job, run) -> bool:
    if (row.source_kind != "ufo_native_motion" or row.status != "validated"
            or job.status != "succeeded" or getattr(run, "status", None) != "finished"):
        return False
    params = (job.spec or {}).get("params") or {}
    meta = row.meta or {}
    from eb.ufo_grounded_reference import OPERATION as GROUNDED_OPERATION
    if (job.kind != "data.convert" or params.get("operation") not in {DATA_OPERATION, GROUNDED_OPERATION}
            or params.get("dataset_id") != row.id or meta.get("validation_job_id") != job.id
            or meta.get("validation_run_id") != run.id or run.job_id != job.id
            or (row.summary_stats or {}).get("validation", {}).get("status") != "validated"):
        raise ValueError("Native DatasetVersion producer differs from successful validation lineage")
    report = require_native_dataset(row)
    manifest = read_json(safe_file(Path(row.dest_path), "manifest.json"), limit=4 * 1024 * 1024)
    if (manifest.get("schema_version") != "eb/ufo-native-motion/v1"
            or manifest.get("dataset_id") != row.id or manifest.get("split") != "train"):
        raise ValueError("Native DatasetVersion manifest differs from its frozen dataset and training split")
    parent = None
    if params.get("operation") == GROUNDED_OPERATION:
        parent = await session.get(DatasetVersion, meta.get("parent_version_id"))
        plan = params.get("grounded_plan") or {}
        if (not parent or parent.dataset_id != meta.get("parent_dataset_id")
                or parent.id != plan.get("parent_version_id")
                or manifest.get("parent_version_id") != parent.id
                or manifest.get("source_transform") != plan.get("protocol")):
            raise ValueError("A1 candidate version has a different parent or transformation")
    content_hash = "sha256:" + report["manifest_sha256"]
    # Serialize GET-triggered backfill and validation completion on this Dataset.
    await session.execute(select(Dataset.id).where(Dataset.id == row.id).with_for_update())
    version = await session.scalar(select(DatasetVersion).where(
        DatasetVersion.dataset_id == row.id, DatasetVersion.version == "native-v1"))
    changed = version is None
    if version is None:
        version = DatasetVersion(id=new_id("dsv"), dataset_id=row.id, version="native-v1",
            compatible_envs=[ENV_ID], content_hash=content_hash, funnel={},
            stats={**deepcopy(row.summary_stats), "manifest": manifest,
                   "storage_ref": row.dest_path, "producer_run": run.id, "producer_job_id": job.id})
        session.add(version)
        await session.flush()
    elif (version.content_hash != content_hash or version.compatible_envs != [ENV_ID]
            or (version.stats or {}).get("storage_ref") != row.dest_path
            or (version.stats or {}).get("producer_run") != run.id):
        raise ValueError("Published native-v1 DatasetVersion identity changed; create a new Dataset")
    for source_type, source_id, relation in (
        ("run", run.id, "produced"), ("dataset", row.id, "has_version"),
    ):
        existing = await session.scalar(select(LineageEdge.id).where(
            LineageEdge.src_type == source_type, LineageEdge.src_id == source_id,
            LineageEdge.dst_type == "dataset_version", LineageEdge.dst_id == version.id,
            LineageEdge.relation == relation))
        if existing is None:
            session.add(LineageEdge(src_type=source_type, src_id=source_id,
                dst_type="dataset_version", dst_id=version.id, relation=relation, run_id=run.id))
            changed = True
    if parent is not None:
        edge = await session.scalar(select(LineageEdge.id).where(
            LineageEdge.src_type == "dataset_version", LineageEdge.src_id == parent.id,
            LineageEdge.dst_type == "dataset_version", LineageEdge.dst_id == version.id,
            LineageEdge.relation == "derived"))
        if edge is None:
            session.add(LineageEdge(src_type="dataset_version", src_id=parent.id,
                dst_type="dataset_version", dst_id=version.id, relation="derived", run_id=run.id))
            changed = True
    return changed
