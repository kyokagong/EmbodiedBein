"""Pinned MotionDecode views and operator-only, hash-bound kinematic previews."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PurePosixPath

import httpx
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import func, select

from eb.api.deps import PrincipalDep, SessionDep
from eb.api.file_stream import stream_storage_file
from eb.domain.data_view import compact_dataset_stats
from eb.domain.job import JobSpec
from eb.ids import new_id
from eb.motiondecode import ENV_ID, EXPECTED_FILES, HEADER, REPO, REVISION
from eb.motiondecode_view import EXCLUDED_PATH, EXCLUDED_SHA256, OPERATION
from eb.settings import get_settings
from eb.store.ebd import EbdClient, artifact_root, storage_root
from eb.store.models import Dataset, DatasetVersion, Job, LineageEdge, Run

router = APIRouter()
PREVIEW_OPERATION = "preview_motiondecode_v1"
PREVIEW_IMAGE = "eb/motiondecode-preview:latest"
DERIVE_OPERATION = "derive_motiondecode_view"
SHA256 = re.compile(r"[a-f0-9]{64}")


class ValidViewIn(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MotionPreviewIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str = Field(min_length=1, max_length=1024)
    start_frame: int = Field(default=0, ge=0, strict=True)


def operator(principal):
    if principal.actor_type == "agent":
        raise HTTPException(403, "MotionDecode preparation requires an operator")


def source_root() -> Path:
    return get_settings().datasets_root / "MotionDecode" / REVISION


def outputs(run) -> Path:
    if not re.fullmatch(r"run_[A-Za-z0-9_-]+", run.id):
        raise ValueError("Invalid producing Run identifier")
    return get_settings().data_root / "runs" / run.id / "outputs"


def require_view(row):
    if (row is None or row.source_kind != "motiondecode_view" or row.source_uri != REPO
            or row.embodiment_env_id != ENV_ID or row.dest_path != str(source_root())
            or (row.meta or {}).get("operation") != OPERATION
            or (row.meta or {}).get("source_revision") != REVISION
            or row.status != "validated"):
        raise HTTPException(400, "A validated pinned MotionDecode view is required")
    validation = (row.summary_stats or {}).get("validation", {})
    if (validation.get("scope") != OPERATION or validation.get("status") != "validated"
            or validation.get("preview_ready") is not True or validation.get("training_ready") is not False):
        raise HTTPException(400, "MotionDecode preview validation is incomplete")


def member(row, path: str) -> dict:
    if (not isinstance(path, str) or not path or path == EXCLUDED_PATH or "\\" in path
            or any(p in {"", ".", ".."} for p in path.split("/"))
            or PurePosixPath(path).is_absolute() or any(ord(c) < 32 for c in path)):
        raise HTTPException(400, "Invalid MotionDecode member path")
    episode = next((e for e in (row.summary_stats or {}).get("episodes", [])
                    if e.get("path") == path), None)
    if (not episode or not SHA256.fullmatch(str(episode.get("sha256", "")))
            or type(episode.get("length")) is not int or episode["length"] <= 0
            or type(episode.get("bytes")) is not int or episode["bytes"] <= 0):
        raise HTTPException(400, "Motion is not a validated view member")
    return episode


async def lock(session, key):
    lock_id = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big", signed=True)
    await session.execute(select(func.pg_advisory_xact_lock(lock_id)))


async def image_digest(client, name):
    image = await client.image_info(name)
    digest = image.get("digest", "")
    if image.get("ok") is not True or not re.fullmatch(r"sha256:[a-f0-9]{64}", digest):
        raise HTTPException(400, "MotionDecode task image is unavailable")
    return digest


def canonical_asset_ref(ref):
    if (not isinstance(ref, dict) or set(ref) != {"kind", "path", "commit"}
            or ref.get("kind") != "content_snapshot"
            or not re.fullmatch(r"sha256:[a-f0-9]{64}", str(ref.get("commit", "")))
            or ref.get("path") != str(get_settings().data_root / "snapshots" / ref["commit"][7:] / "code")):
        raise ValueError("Invalid frozen G1 preview snapshot")
    return ref


async def validate_motiondecode_job(session, spec):
    """Keep generic Job submission subject to the same frozen inputs as these routes."""
    try:
        row = await session.get(Dataset, spec.params.get("dataset_id"))
        if (row is None or spec.kind != "data.convert" or spec.embodiment_env_id != ENV_ID
                or spec.runtime != {"network": "none"} or spec.code_refs is not None
                or spec.inputs.model_dump() != {"datasets": [], "base_model": None, "resume_from": None}
                or spec.overrides or spec.outputs or spec.fanout is not None or spec.gates):
            raise ValueError("MotionDecode jobs require canonical isolated execution")
        client = EbdClient()
        operation = spec.params.get("operation")
        if operation == DERIVE_OPERATION:
            meta = row.meta or {}
            if (row.source_kind != "motiondecode_view" or meta.get("operation") != OPERATION
                    or row.source_uri != REPO or row.dest_path != str(source_root())
                    or row.embodiment_env_id != ENV_ID or meta.get("source_revision") != REVISION
                    or row.status not in {"queued", "failed", "invalid"}):
                raise ValueError("MotionDecode view registration differs")
            parent = await session.get(Dataset, meta.get("parent_dataset_id"))
            if parent is None:
                raise ValueError("MotionDecode raw parent is missing")
            run = await raw_source(session, parent)
            if meta.get("source_run_id") != run.id:
                raise ValueError("MotionDecode raw evidence Run changed")
            expected = {"operation": DERIVE_OPERATION, "dataset_id": row.id, "src": row.dest_path,
                "source_index": str(outputs(run) / "file-index.jsonl"),
                "index_sha": meta.get("source_evidence_sha256"),
                "manifest_sha": meta.get("source_manifest_sha256")}
            if any(not SHA256.fullmatch(str(expected[k])) for k in ("index_sha", "manifest_sha")):
                raise ValueError("MotionDecode source metadata is not hash-frozen")
            image = await image_digest(client, "eb/tools:latest")
            cpus, memory, timeout = 4, 8, 4
        elif operation == PREVIEW_OPERATION:
            require_view(row)
            ep = member(row, spec.params.get("path", ""))
            start = spec.params.get("start_frame")
            if (type(start) is not int or not 0 <= start < ep["length"]
                    or ep["bytes"] > 32 * 1024**2 or ep["length"] > 500000):
                raise ValueError("MotionDecode preview exceeds its source or resource bounds")
            ref = canonical_asset_ref((row.meta or {}).get("preview_asset_ref"))
            expected = {"operation": PREVIEW_OPERATION, "dataset_id": row.id,
                "source_file": str(source_root() / ep["path"]), "path": ep["path"],
                "source_sha256": ep["sha256"], "source_bytes": ep["bytes"],
                "source_frames": ep["length"], "start_frame": start, "asset_ref": ref}
            image = await image_digest(client, PREVIEW_IMAGE)
            cpus, memory, timeout = 2, 4, .5
        else:
            raise ValueError("Unsupported MotionDecode job operation")
        if (spec.params != expected or spec.image != image
                or spec.resources.model_dump() != JobSpec(kind="data.convert", embodiment_env_id=ENV_ID,
                    resources={"gpus": 0, "cpus": cpus, "memory_gb": memory}).resources.model_dump()
                or spec.policy.timeout_h != timeout or spec.policy.max_retries != 0 or spec.policy.retry_on):
            raise ValueError("MotionDecode job differs from its frozen source, image or resource policy")
    except HTTPException as exc:
        raise ValueError(str(exc.detail)) from exc


async def add_edge(session, src_type, src_id, dst_type, dst_id, relation, run_id=None):
    edge = await session.scalar(select(LineageEdge.id).where(
        LineageEdge.src_type == src_type, LineageEdge.src_id == src_id,
        LineageEdge.dst_type == dst_type, LineageEdge.dst_id == dst_id,
        LineageEdge.relation == relation))
    if edge is not None:
        return False
    session.add(LineageEdge(src_type=src_type, src_id=src_id, dst_type=dst_type,
                           dst_id=dst_id, relation=relation, run_id=run_id))
    return True


async def raw_source(session, parent):
    meta = parent.meta or {}
    if (parent.source_kind != "huggingface" or parent.source_uri != REPO
            or parent.dest_path != str(source_root()) or parent.embodiment_env_id != ENV_ID
            or meta.get("source_revision") != REVISION or parent.status != "invalid"):
        raise HTTPException(400, "Expected the pinned invalid raw MotionDecode source")
    job = await session.get(Job, meta.get("validation_job_id"))
    run = await session.scalar(select(Run).where(Run.job_id == meta.get("validation_job_id"))
                               .order_by(Run.attempt.desc()))
    params = ((job.spec or {}).get("params") or {}) if job else {}
    stats = parent.summary_stats or {}
    validation = stats.get("validation", {})
    if (not job or not run or job.kind != "data.convert" or job.status != "failed"
            or run.status != "aborted" or run.job_id != job.id
            or meta.get("validation_run_id") != run.id
            or params.get("operation") != "acquire_motiondecode" or params.get("dataset_id") != parent.id
            or params.get("src") != parent.dest_path or params.get("repo_id") != REPO
            or params.get("revision") != REVISION or stats.get("files") != EXPECTED_FILES
            or validation.get("scope") != "motiondecode_raw_csv_v1"
            or validation.get("invalid_files") != 1
            or validation.get("errors") != [f"{EXCLUDED_PATH}: ValueError: empty motion"]):
        raise HTTPException(400, "Raw MotionDecode producer or complete validation evidence differs")
    return run


@router.post("/datasets/{parent_id}/motiondecode/valid-view")
async def create_valid_view(parent_id: str, body: ValidViewIn, session: SessionDep, principal: PrincipalDep):
    from eb.api.routes_data import _ds_out, reconcile_dataset
    from eb.orchestrator.service import Orchestrator

    operator(principal)
    await lock(session, "motiondecode-view:" + parent_id)
    parent = await session.get(Dataset, parent_id)
    if parent is None:
        raise HTTPException(404)
    run = await raw_source(session, parent)
    children = (await session.scalars(select(Dataset).where(
        Dataset.source_kind == "motiondecode_view"))).all()
    child = next((r for r in children if (r.meta or {}).get("parent_dataset_id") == parent.id
                  and (r.meta or {}).get("operation") == OPERATION), None)
    if child is not None:
        await reconcile_dataset(child, session)
        # Reconciliation can commit, so reacquire the transaction lock before retrying.
        await lock(session, "motiondecode-view:" + parent_id)
        await session.refresh(child)
        if child.status not in {"failed", "invalid"}:
            return _ds_out(child)
    client = EbdClient()
    manifest = await client.hash_storage_file(storage_root(source_root()),
                                              ".embodiedbein/source-manifest.json")
    index = await client.hash_storage_file(storage_root(outputs(run)), "file-index.jsonl")
    image = await image_digest(client, "eb/tools:latest")
    if child is None:
        child = Dataset(id=new_id("ds"), name=parent.name + " · valid motions",
            source_kind="motiondecode_view",
            source_uri=REPO, dest_path=parent.dest_path, format="unknown", license=parent.license,
            embodiment_env_id=ENV_ID, total_bytes=0, status="queued", download_progress={},
            meta={"operation": OPERATION, "parent_dataset_id": parent.id, "source_revision": REVISION,
                  "usage": (parent.meta or {}).get("usage", "train")},
            summary_stats={"validation": {"status": "not_validated", "training_ready": False}})
        session.add(child)
        await session.flush()
    params = {"operation": DERIVE_OPERATION, "src": parent.dest_path, "dataset_id": child.id,
              "source_index": str(outputs(run) / "file-index.jsonl"), "index_sha": index["sha256"],
              "manifest_sha": manifest["sha256"]}
    child.meta = {**child.meta, "source_evidence_sha256": index["sha256"],
                  "source_manifest_sha256": manifest["sha256"], "source_run_id": run.id}
    await session.flush()
    spec = JobSpec(kind="data.convert", embodiment_env_id=ENV_ID, image=image, params=params,
        runtime={"network": "none"}, resources={"gpus": 0, "cpus": 4, "memory_gb": 8},
        policy={"timeout_h": 4, "max_retries": 0, "retry_on": []})
    key = f"motiondecode-view:{child.id}:{new_id('attempt')}"
    try:
        job = await Orchestrator().submit(session, spec, principal.id, key, commit=False)
    except (ValueError, PermissionError) as exc:
        raise HTTPException(400, str(exc)) from exc
    child.meta = {**child.meta, "validation_job_id": job.id}
    child.status, child.error_message = "queued", None
    parent.meta = {**parent.meta, "valid_view_id": child.id}
    await add_edge(session, "dataset", parent.id, "dataset", child.id, "derived")
    await session.commit()
    return _ds_out(child)


async def publish_view_version(session, row, job, run) -> bool:
    if (row.source_kind != "motiondecode_view" or row.status != "validated"
            or job.status != "succeeded" or run.status != "finished"):
        return False
    try:
        require_view(row)
    except HTTPException as exc:
        raise ValueError(str(exc.detail)) from exc
    stats, meta = row.summary_stats or {}, row.meta or {}
    params = (job.spec or {}).get("params") or {}
    index_hash = stats.get("index_hash", "")
    if (job.kind != "data.convert" or params.get("operation") != DERIVE_OPERATION
            or params.get("dataset_id") != row.id or params.get("src") != row.dest_path
            or meta.get("validation_job_id") != job.id or meta.get("validation_run_id") != run.id
            or run.job_id != job.id or stats.get("storage_ref") != row.dest_path
            or not SHA256.fullmatch(index_hash) or not SHA256.fullmatch(params.get("index_sha", ""))
            or stats.get("source_evidence_sha256") != params.get("index_sha")
            or meta.get("source_evidence_sha256") != params.get("index_sha")
            or meta.get("source_manifest_sha256") != params.get("manifest_sha")
            or params.get("source_index") != str(get_settings().data_root / "runs"
                / str(meta.get("source_run_id")) / "outputs/file-index.jsonl")
            or stats.get("exclusions") != [{"path": EXCLUDED_PATH, "sha256": EXCLUDED_SHA256,
                 "bytes": 970, "reason": "header_only_no_motion_frames"}]):
        raise ValueError("MotionDecode view publication differs from its frozen producer")
    await session.execute(select(Dataset.id).where(Dataset.id == row.id).with_for_update())
    version = await session.scalar(select(DatasetVersion).where(
        DatasetVersion.dataset_id == row.id, DatasetVersion.version == "valid-v1"))
    if version is None:
        actual = await EbdClient().hash_storage_file(storage_root(outputs(run)), "included-motions.jsonl")
        if actual["sha256"] != index_hash:
            raise ValueError("MotionDecode selection manifest hash differs from its report")
    frozen = {**compact_dataset_stats(stats), "storage_ref": row.dest_path,
        "selection_manifest": str(outputs(run) / "included-motions.jsonl"),
        "producer_job_id": job.id, "producer_run": run.id, "parent_dataset_id": meta.get("parent_dataset_id")}
    changed = version is None
    if version is None:
        version = DatasetVersion(id=new_id("dsv"), dataset_id=row.id, version="valid-v1",
            compatible_envs=[], content_hash="sha256:" + index_hash, stats=frozen, funnel={})
        session.add(version)
        await session.flush()
    elif (version.content_hash != "sha256:" + index_hash or version.compatible_envs != []
          or any((version.stats or {}).get(k) != frozen[k] for k in
                 ("storage_ref", "selection_manifest", "producer_job_id", "producer_run",
                  "parent_dataset_id"))):
        raise ValueError("Published MotionDecode valid-v1 identity changed; create a new dataset")
    for src_type, src_id, dst_type, dst_id, relation in (
        ("run", run.id, "dataset_version", version.id, "produced"),
        ("dataset", row.id, "dataset_version", version.id, "has_version"),
        ("dataset", meta["parent_dataset_id"], "dataset", row.id, "derived")):
        changed = await add_edge(session, src_type, src_id, dst_type, dst_id, relation, run.id) or changed
    return changed


@router.post("/datasets/{did}/motiondecode/previews")
async def create_preview(did: str, body: MotionPreviewIn, session: SessionDep, principal: PrincipalDep):
    from eb.orchestrator.service import Orchestrator

    operator(principal)
    row = await session.get(Dataset, did)
    if row is None:
        raise HTTPException(404)
    require_view(row)
    ep = member(row, body.path)
    if body.start_frame >= ep["length"]:
        raise HTTPException(400, "start_frame is outside the source motion")
    client = EbdClient()
    image = await image_digest(client, PREVIEW_IMAGE)
    asset_ref = (row.meta or {}).get("preview_asset_ref")
    if asset_ref is None:
        snapshot = await client.snapshot(str(get_settings().ufo_native_root / "ufo/data/robots/g1"))
        asset_ref = {"kind": "content_snapshot", "path": snapshot["path"], "commit": snapshot["content_hash"]}
        row.meta = {**row.meta, "preview_asset_ref": asset_ref}
    try:
        canonical_asset_ref(asset_ref)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    await session.flush()
    spec = JobSpec(kind="data.convert", embodiment_env_id=ENV_ID, image=image,
        params={"operation": PREVIEW_OPERATION, "dataset_id": row.id,
                "source_file": str(source_root() / body.path),
                "path": body.path, "source_sha256": ep["sha256"], "source_bytes": ep["bytes"],
                "source_frames": ep["length"], "start_frame": body.start_frame, "asset_ref": asset_ref},
        runtime={"network": "none"}, resources={"gpus": 0, "cpus": 2, "memory_gb": 4},
        policy={"timeout_h": .5, "max_retries": 0, "retry_on": []})
    frozen = json.dumps(spec.model_dump(mode="json"), sort_keys=True).encode()
    key = "motiondecode-preview:" + hashlib.sha256(frozen).hexdigest()
    await lock(session, key)
    previous = await session.scalar(select(Job).where(Job.idem_key.like(key + "%"))
                                    .order_by(Job.created_at.desc()))
    if previous and previous.status == "succeeded":
        try:
            _, previous, run, frozen_ep = await bound_preview(session, did, previous.id)
            receipt = await verified_preview(session, previous, run, frozen_ep, reverify_files=True)
        except HTTPException as exc:
            if exc.status_code not in {404, 409}:
                raise
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code not in {400, 404, 409, 413}:
                raise
        else:
            await session.commit()
            base = f"/api/v1/datasets/{did}/motiondecode/previews/{previous.id}/files"
            return {"id": previous.id, "status": previous.status, "error": previous.error,
                "preview": {**receipt, "video_url": base + "/replay.mp4", "table_base_url": base + "/table"}}
    elif previous and previous.status not in {"failed", "infra_failed", "cancelled", "timeout"}:
        await session.commit()
        return await preview_status(did, previous.id, session, principal)
    try:
        job = await Orchestrator().submit(session, spec, principal.id,
            key + (":" + new_id("attempt") if previous else ""), commit=False)
    except (ValueError, PermissionError) as exc:
        raise HTTPException(400, str(exc)) from exc
    await session.commit()
    return {"id": job.id, "status": job.status, "error": job.error}


async def bound_preview(session, did, jid):
    row = await session.get(Dataset, did)
    if row is None:
        raise HTTPException(404)
    require_view(row)
    job = await session.get(Job, jid)
    params = ((job.spec or {}).get("params") or {}) if job else {}
    if (not job or job.kind != "data.convert" or params.get("operation") != PREVIEW_OPERATION
            or params.get("dataset_id") != did or job.spec.get("embodiment_env_id") != ENV_ID):
        raise HTTPException(404, "Preview does not belong to this dataset")
    ep = member(row, params.get("path", ""))
    start = params.get("start_frame")
    if (params.get("source_file") != str(source_root() / ep["path"])
            or params.get("source_sha256") != ep["sha256"] or params.get("source_bytes") != ep["bytes"]
            or params.get("source_frames") != ep["length"] or type(start) is not int
            or not 0 <= start < ep["length"]):
        raise HTTPException(409, "Preview source differs from its frozen view")
    run = await session.scalar(select(Run).where(Run.job_id == jid).order_by(Run.attempt.desc()))
    return row, job, run, ep


def validate_receipt(receipt, params, ep):
    start = params["start_frame"]
    end = min(ep["length"], start + 7200)
    pages = (ep["length"] + 99) // 100
    if (not isinstance(receipt, dict) or receipt.get("schema_version") != "eb/motiondecode-preview/v1"
            or receipt.get("source_path") != ep["path"] or receipt.get("source_sha256") != ep["sha256"]
            or receipt.get("asset_commit") != params["asset_ref"]["commit"]
            or receipt.get("frames") != ep["length"] or receipt.get("start_frame") != start
            or receipt.get("end_frame") != end or receipt.get("source_fps") != 120
            or receipt.get("fps") != 30 or receipt.get("video_frames") != (end - start + 3) // 4
            or receipt.get("table_page_size") != 100 or receipt.get("table_pages") != pages
            or receipt.get("columns") != HEADER
            or receipt.get("source_frame_indices") != list(range(start, end, 4))
            or receipt.get("timebase") != "declared_120hz_no_source_timestamps"
            or receipt.get("kinematic_only") is not True or receipt.get("dynamics_validated") is not False
            or not 1 <= pages <= 9998):
        raise ValueError("Preview receipt differs from the frozen motion and frame range")
    files = receipt.get("files", [])
    expected = {"replay.mp4", *(f"table/{i:06d}.json" for i in range(pages))}
    if (not isinstance(files, list) or len(files) != len(expected)
            or any(not isinstance(f, dict) or type(f.get("bytes")) is not int
                   or not 0 < f["bytes"] <= 256 * 1024**2 or not SHA256.fullmatch(str(f.get("sha256", "")))
                   for f in files) or {f.get("path") for f in files} != expected):
        raise ValueError("Preview artifact manifest is incomplete or unsafe")
    return receipt


async def verified_preview(session, job, run, ep, *, reverify_files=False):
    if not run or run.status != "finished" or run.job_id != job.id or job.status != "succeeded":
        raise HTTPException(409, "Preview producer has not finished successfully")
    client = EbdClient()
    try:
        text = await client.read_storage_text(storage_root(outputs(run)), "preview-receipt.json",
                                              max_bytes=512 * 1024)
        receipt = validate_receipt(json.loads(text), job.spec["params"], ep)
        sha = hashlib.sha256(text.encode()).hexdigest()
        cached = (run.metrics_summary or {}).get("motiondecode_preview_receipt_sha256")
        if cached is not None and cached != sha:
            raise ValueError("Verified preview receipt changed")
        if cached is None or reverify_files:
            await client.verify_artifacts(artifact_root(outputs(run)), receipt["files"],
                                          max_files=10000, max_file_bytes=256 * 1024**2)
        if cached is None:
            run.metrics_summary = {**(run.metrics_summary or {}), "motiondecode_preview_receipt_sha256": sha}
            await session.commit()
        return receipt
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(409, f"Invalid MotionDecode preview evidence: {exc}") from exc


@router.get("/datasets/{did}/motiondecode/previews/{jid}")
async def preview_status(did: str, jid: str, session: SessionDep, principal: PrincipalDep):
    operator(principal)
    _, job, run, ep = await bound_preview(session, did, jid)
    result = {"id": job.id, "status": job.status, "error": job.error}
    if job.status == "succeeded":
        receipt = await verified_preview(session, job, run, ep)
        base = f"/api/v1/datasets/{did}/motiondecode/previews/{jid}/files"
        result["preview"] = {**receipt, "video_url": base + "/replay.mp4", "table_base_url": base + "/table"}
    return result


@router.api_route("/datasets/{did}/motiondecode/previews/{jid}/files/{path:path}", methods=["GET", "HEAD"])
async def preview_output(did: str, jid: str, path: str, request: Request,
                         session: SessionDep, principal: PrincipalDep):
    from eb.motiondecode_preview import preview_file

    operator(principal)
    _, job, run, ep = await bound_preview(session, did, jid)
    receipt = await verified_preview(session, job, run, ep)
    try:
        selected = preview_file(path, receipt)
    except ValueError as exc:
        raise HTTPException(404, "Unknown preview file") from exc
    root = storage_root(outputs(run))
    await session.rollback()
    return await stream_storage_file(request, root, selected)
