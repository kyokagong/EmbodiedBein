import hashlib
import json
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from fastapi import FastAPI
from fastapi.responses import Response

from eb.api import routes_motiondecode as api
from eb.api.deps import Principal, current_principal
from eb.domain.job import JobSpec
from eb.motiondecode import ENV_ID, HEADER, REPO, REVISION
from eb.motiondecode_view import EXCLUDED_PATH, EXCLUDED_SHA256, OPERATION
from eb.store.db import get_session
from eb.store.models import Dataset, DatasetVersion, Job, Run


@pytest.fixture
def view():
    return NS(id="ds_view", name="valid motions", license="CC-BY-NC", format="motiondecode_csv",
        total_bytes=12345, error_message=None, download_progress={},
        source_kind="motiondecode_view", source_uri=REPO,
        dest_path=str(api.source_root()), embodiment_env_id=ENV_ID, status="validated",
        meta={"operation": OPERATION, "source_revision": REVISION, "parent_dataset_id": "ds_raw",
              "validation_job_id": "job_derive", "validation_run_id": "run_derive",
              "source_run_id": "run_raw", "source_evidence_sha256": "a" * 64,
              "source_manifest_sha256": "b" * 64},
        summary_stats={"validation": {"scope": OPERATION, "status": "validated",
                "preview_ready": True, "training_ready": False},
            "storage_ref": str(api.source_root()), "source_evidence_sha256": "a" * 64,
            "index_hash": "c" * 64, "exclusions": [{"path": EXCLUDED_PATH,
                "sha256": EXCLUDED_SHA256, "bytes": 970, "reason": "header_only_no_motion_frames"}],
            "episodes": [{"path": "samples/Walk/walk.csv", "length": 101, "bytes": 12345,
                           "sha256": "d" * 64, "episode_id": "walk"}], "episodes_count": 1})


def asset_ref():
    return {"kind": "content_snapshot", "commit": "sha256:" + "e" * 64,
        "path": str(api.get_settings().data_root / "snapshots" / ("e" * 64) / "code")}


def preview_spec(view):
    ep = view.summary_stats["episodes"][0]
    view.meta["preview_asset_ref"] = asset_ref()
    return JobSpec(kind="data.convert", embodiment_env_id=ENV_ID, image="sha256:" + "f" * 64,
        params={"operation": api.PREVIEW_OPERATION, "dataset_id": view.id,
                "source_file": str(api.source_root() / ep["path"]), "path": ep["path"],
                "source_sha256": ep["sha256"], "source_bytes": ep["bytes"],
                "source_frames": ep["length"], "start_frame": 4, "asset_ref": asset_ref()},
        runtime={"network": "none"}, resources={"gpus": 0, "cpus": 2, "memory_gb": 4},
        policy={"timeout_h": .5, "max_retries": 0, "retry_on": []})


def preview_job(view):
    return NS(id="job_preview", kind="data.convert", status="succeeded", error=None,
              spec=preview_spec(view).model_dump(mode="json"))


def receipt_for(view):
    ep = view.summary_stats["episodes"][0]
    return {"schema_version": "eb/motiondecode-preview/v1", "source_path": ep["path"],
        "source_sha256": ep["sha256"], "asset_commit": asset_ref()["commit"],
        "frames": 101, "start_frame": 4, "end_frame": 101, "source_fps": 120,
        "fps": 30, "video_frames": 25, "columns": HEADER, "table_page_size": 100,
        "table_pages": 2, "source_frame_indices": list(range(4, 101, 4)),
        "timebase": "declared_120hz_no_source_timestamps", "kinematic_only": True,
        "dynamics_validated": False, "files": [{"path": path, "bytes": 100, "sha256": "a" * 64}
            for path in ["replay.mp4", "table/000000.json", "table/000001.json"]]}


def session_for(view, job=None, run=None):
    async def get(model, key):
        if model is Dataset:
            return view if key == view.id else None
        if model is Job:
            return job if job and key == job.id else None
        return None
    return NS(get=AsyncMock(side_effect=get), scalar=AsyncMock(return_value=run),
        execute=AsyncMock(), commit=AsyncMock(), rollback=AsyncMock(), flush=AsyncMock(),
        refresh=AsyncMock(), add=Mock())


def app_for(session, principal=None):
    app = FastAPI()
    app.include_router(api.router, prefix="/api/v1")
    app.dependency_overrides[get_session] = lambda: session
    app.dependency_overrides[current_principal] = lambda: principal or Principal("kk", "user")
    return app


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["../walk.csv", "/etc/passwd", "samples/../walk.csv", EXCLUDED_PATH,
                                  "samples\\Walk\\walk.csv", "samples/missing.csv"])
async def test_preview_request_rejects_untrusted_members_before_external_io(view, monkeypatch, path):
    image = AsyncMock()
    monkeypatch.setattr(api.EbdClient, "image_info", image)
    transport = httpx.ASGITransport(app=app_for(session_for(view)))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(f"/api/v1/datasets/{view.id}/motiondecode/previews", json={"path": path})
    assert response.status_code == 400
    image.assert_not_called()


@pytest.mark.asyncio
async def test_preview_job_cannot_be_read_through_another_dataset(view):
    job = preview_job(view)
    job.spec["params"]["dataset_id"] = "ds_other"
    session = session_for(view, job)
    transport = httpx.ASGITransport(app=app_for(session))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(f"/api/v1/datasets/{view.id}/motiondecode/previews/{job.id}")
    assert response.status_code == 404
    session.scalar.assert_not_called()


@pytest.mark.asyncio
async def test_preview_agent_is_denied_even_for_existing_job(view):
    session = session_for(view, preview_job(view))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(
        app=app_for(session, Principal("worker", "agent"))), base_url="http://test") as client:
        response = await client.get(f"/api/v1/datasets/{view.id}/motiondecode/previews/job_preview")
    assert response.status_code == 403
    session.get.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("start", [-1, 101, True])
async def test_preview_request_rejects_invalid_start(view, start):
    session = session_for(view)
    transport = httpx.ASGITransport(app=app_for(session))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(f"/api/v1/datasets/{view.id}/motiondecode/previews",
                                     json={"path": "samples/Walk/walk.csv", "start_frame": start})
    assert response.status_code in {400, 422}


@pytest.mark.asyncio
@pytest.mark.parametrize("tamper", ["source", "asset", "columns", "range", "indices", "files", "dynamics"])
async def test_preview_receipt_rejects_incomplete_or_rebound_evidence(view, tamper):
    receipt = receipt_for(view)
    if tamper == "source":
        receipt["source_sha256"] = "b" * 64
    elif tamper == "asset":
        receipt["asset_commit"] = "sha256:" + "b" * 64
    elif tamper == "columns":
        receipt["columns"] = list(reversed(HEADER))
    elif tamper == "range":
        receipt["end_frame"] += 1
    elif tamper == "indices":
        receipt["source_frame_indices"][-1] -= 1
    elif tamper == "files":
        receipt["files"][1]["path"] = "../secret.json"
    else:
        receipt["dynamics_validated"] = True
    with pytest.raises(ValueError):
        api.validate_receipt(receipt, preview_spec(view).params, view.summary_stats["episodes"][0])


@pytest.mark.asyncio
async def test_verified_receipt_cached_but_receipt_mutation_rejected(view, monkeypatch):
    receipt = json.dumps(receipt_for(view))
    read = AsyncMock(return_value=receipt)
    verify = AsyncMock(return_value={"verified": True})
    monkeypatch.setattr(api.EbdClient, "read_storage_text", read)
    monkeypatch.setattr(api.EbdClient, "verify_artifacts", verify)
    run = NS(id="run_preview", job_id="job_preview", status="finished", metrics_summary={})
    job = preview_job(view)
    session = session_for(view, job, run)
    ep = view.summary_stats["episodes"][0]
    await api.verified_preview(session, job, run, ep)
    await api.verified_preview(session, job, run, ep)
    assert verify.await_count == 1 and read.await_count == 2
    expected_sha = hashlib.sha256(receipt.encode()).hexdigest()
    assert run.metrics_summary["motiondecode_preview_receipt_sha256"] == expected_sha
    read.return_value = receipt + " "
    with pytest.raises(api.HTTPException, match="409"):
        await api.verified_preview(session, job, run, ep)
    assert verify.await_count == 1


@pytest.mark.asyncio
async def test_file_route_streams_only_whitelisted_finished_preview(view, monkeypatch):
    job = preview_job(view)
    run = NS(id="run_preview", job_id=job.id, status="finished", metrics_summary={})
    session = session_for(view, job, run)
    monkeypatch.setattr(api.EbdClient, "read_storage_text",
                        AsyncMock(return_value=json.dumps(receipt_for(view))))
    monkeypatch.setattr(api.EbdClient, "verify_artifacts", AsyncMock(return_value={"verified": True}))
    stream = AsyncMock(return_value=Response(content=b"video", media_type="video/mp4"))
    monkeypatch.setattr(api, "stream_storage_file", stream)
    transport = httpx.ASGITransport(app=app_for(session))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        base = f"/api/v1/datasets/{view.id}/motiondecode/previews/{job.id}/files/"
        for path in ["table/000002.json", "table/%2e%2e/secret.json", "secret.json"]:
            response = await client.get(base + path)
            assert response.status_code == 404
        stream.assert_not_called()
        response = await client.head(base + "replay.mp4")
        assert response.status_code == 200 and response.content == b""
    assert stream.await_args.args[2] == "replay.mp4"
    assert stream.await_args.args[0].method == "HEAD"


def derived_job(view):
    return NS(id="job_derive", kind="data.convert", status="succeeded", spec={"params": {
        "operation": api.DERIVE_OPERATION, "dataset_id": view.id, "src": view.dest_path,
        "index_sha": "a" * 64, "manifest_sha": "b" * 64,
        "source_index": str(api.get_settings().data_root / "runs/run_raw/outputs/file-index.jsonl")}})


@pytest.mark.asyncio
async def test_publication_requires_finished_producer_and_never_marks_training_ready(view, monkeypatch):
    job = derived_job(view)
    run = NS(id="run_derive", job_id=job.id, status="collecting_out")
    session = session_for(view)
    assert await api.publish_view_version(session, view, job, run) is False
    session.add.assert_not_called()
    run.status = "finished"
    monkeypatch.setattr(api.EbdClient, "hash_storage_file", AsyncMock(return_value={"sha256": "c" * 64}))
    assert await api.publish_view_version(session, view, job, run) is True
    version = next(call.args[0] for call in session.add.call_args_list
                   if isinstance(call.args[0], DatasetVersion))
    assert version.compatible_envs == []
    assert version.content_hash == "sha256:" + "c" * 64
    assert version.stats["validation"]["training_ready"] is False
    assert "episodes" not in version.stats
    assert version.stats["parent_dataset_id"] == "ds_raw"
    assert version.stats["producer_job_id"] == job.id and version.stats["producer_run"] == run.id


@pytest.mark.asyncio
async def test_publication_rejects_mismatched_hash_or_existing_version_identity(view, monkeypatch):
    job = derived_job(view)
    run = NS(id="run_derive", job_id=job.id, status="finished")
    session = session_for(view)
    monkeypatch.setattr(api.EbdClient, "hash_storage_file", AsyncMock(return_value={"sha256": "b" * 64}))
    with pytest.raises(ValueError, match="manifest hash"):
        await api.publish_view_version(session, view, job, run)
    session.scalar.return_value = NS(content_hash="sha256:" + "f" * 64, compatible_envs=[], stats={})
    with pytest.raises(ValueError, match="identity changed"):
        await api.publish_view_version(session, view, job, run)
    session.add.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("tamper", ["path", "hash", "assets", "network", "image", "resources", "inputs"])
async def test_generic_submission_cannot_bypass_preview_input_freeze(view, monkeypatch, tamper):
    spec = preview_spec(view)
    session = session_for(view)
    monkeypatch.setattr(api.EbdClient, "image_info",
                        AsyncMock(return_value={"ok": True, "digest": spec.image}))
    await api.validate_motiondecode_job(session, spec)
    bad = spec.model_copy(deep=True)
    if tamper == "path":
        bad.params["source_file"] = "/etc/passwd"
    elif tamper == "hash":
        bad.params["source_sha256"] = "a" * 64
    elif tamper == "assets":
        bad.params["asset_ref"]["path"] = "/etc"
    elif tamper == "network":
        bad.runtime = {"network": "host"}
    elif tamper == "image":
        bad.image = "sha256:" + "0" * 64
    elif tamper == "resources":
        bad.resources.gpus = 1
    else:
        bad.inputs.resume_from = "run_other"
    with pytest.raises(ValueError):
        await api.validate_motiondecode_job(session, bad)


def raw_parent():
    return NS(id="ds_raw", source_kind="huggingface", source_uri=REPO,
        dest_path=str(api.source_root()), embodiment_env_id=ENV_ID, status="invalid",
        meta={"source_revision": REVISION, "validation_job_id": "job_raw", "validation_run_id": "run_raw"},
        summary_stats={"files": api.EXPECTED_FILES, "validation": {"scope": "motiondecode_raw_csv_v1",
            "invalid_files": 1, "errors": [f"{EXCLUDED_PATH}: ValueError: empty motion"]}})


def raw_job(parent):
    return NS(id="job_raw", kind="data.convert", status="failed", spec={"params": {
        "operation": "acquire_motiondecode", "dataset_id": parent.id, "src": parent.dest_path,
        "repo_id": REPO, "revision": REVISION}})


@pytest.mark.asyncio
async def test_valid_view_request_reuses_existing_child_without_new_job_or_source_io(view, monkeypatch):
    parent = raw_parent()
    producing = raw_job(parent)
    run = NS(id="run_raw", job_id=producing.id, status="aborted")
    session = session_for(view)
    objects = {(Dataset, parent.id): parent, (Dataset, view.id): view, (Job, producing.id): producing}
    session.get.side_effect = lambda model, key: objects.get((model, key))
    session.scalar.return_value = run
    session.scalars = AsyncMock(return_value=NS(all=lambda: [view]))
    image = AsyncMock()
    monkeypatch.setattr(api.EbdClient, "image_info", image)
    transport = httpx.ASGITransport(app=app_for(session))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(f"/api/v1/datasets/{parent.id}/motiondecode/valid-view", json={})
    assert response.status_code == 200 and response.json()["id"] == view.id
    assert response.json()["status"] == "validated"
    image.assert_not_called()
    session.add.assert_not_called()
    assert parent.status == "invalid"


@pytest.mark.asyncio
async def test_generic_derived_job_requires_frozen_raw_producer_and_hashes(view, monkeypatch):
    parent = raw_parent()
    producing = raw_job(parent)
    run = NS(id="run_raw", job_id=producing.id, status="aborted")
    view.status = "queued"
    session = session_for(view)
    objects = {(Dataset, parent.id): parent, (Dataset, view.id): view, (Job, producing.id): producing}
    session.get.side_effect = lambda model, key: objects.get((model, key))
    session.scalar.return_value = run
    digest = "sha256:" + "f" * 64
    monkeypatch.setattr(api.EbdClient, "image_info", AsyncMock(return_value={"ok": True, "digest": digest}))
    spec = JobSpec(kind="data.convert", embodiment_env_id=ENV_ID, image=digest,
        params=derived_job(view).spec["params"], runtime={"network": "none"},
        resources={"gpus": 0, "cpus": 4, "memory_gb": 8},
        policy={"timeout_h": 4, "max_retries": 0, "retry_on": []})
    await api.validate_motiondecode_job(session, spec)
    spec.params["index_sha"] = "0" * 64
    with pytest.raises(ValueError, match="frozen source"):
        await api.validate_motiondecode_job(session, spec)
    spec.params["index_sha"] = "a" * 64
    producing.spec["params"]["revision"] = "untrusted"
    with pytest.raises(ValueError, match="producer"):
        await api.validate_motiondecode_job(session, spec)


def storage_error(status):
    request = httpx.Request("POST", "http://ebd/v1/artifacts/verify")
    response = httpx.Response(status, request=request)
    return httpx.HTTPStatusError("storage verification failed", request=request, response=response)


def completed_preview_request(view, monkeypatch):
    job = preview_job(view)
    receipt = json.dumps(receipt_for(view))
    run = NS(id="run_preview", job_id=job.id, status="finished", metrics_summary={
        "motiondecode_preview_receipt_sha256": hashlib.sha256(receipt.encode()).hexdigest()})
    session = session_for(view, job, run)
    session.scalar.side_effect = [job, run]
    monkeypatch.setattr(api.EbdClient, "image_info",
        AsyncMock(return_value={"ok": True, "digest": job.spec["image"]}))
    read = AsyncMock(return_value=receipt)
    verify = AsyncMock(return_value={"verified": True})
    monkeypatch.setattr(api.EbdClient, "read_storage_text", read)
    monkeypatch.setattr(api.EbdClient, "verify_artifacts", verify)
    submit = AsyncMock(return_value=NS(id="job_replacement", status="queued", error=None))
    monkeypatch.setattr("eb.orchestrator.service.Orchestrator.submit", submit)
    return session, read, verify, submit


@pytest.mark.asyncio
@pytest.mark.parametrize("damage", [
    "receipt_missing", "receipt_changed", "cached_video_missing", "table_corrupt"])
async def test_successful_preview_with_damaged_artifacts_submits_new_attempt(view, monkeypatch, damage):
    session, read, verify, submit = completed_preview_request(view, monkeypatch)
    if damage == "receipt_missing":
        read.side_effect = storage_error(404)
    elif damage == "receipt_changed":
        read.return_value += " "
    elif damage == "cached_video_missing":
        verify.side_effect = storage_error(404)
    else:
        verify.side_effect = storage_error(409)
    body = api.MotionPreviewIn(path="samples/Walk/walk.csv", start_frame=4)
    result = await api.create_preview(view.id, body, session, Principal("kk", "user"))
    assert result == {"id": "job_replacement", "status": "queued", "error": None}
    submit.assert_awaited_once()
    assert ":attempt_" in submit.await_args.args[3]
    assert submit.await_args.kwargs == {"commit": False}
    if damage in {"cached_video_missing", "table_corrupt"}:
        verify.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["receipt_503", "files_502", "connection"])
async def test_temporary_preview_storage_failure_does_not_submit_another_render(view, monkeypatch, failure):
    session, read, verify, submit = completed_preview_request(view, monkeypatch)
    error = httpx.ConnectError("temporarily disconnected") if failure == "connection" else storage_error(
        503 if failure == "receipt_503" else 502)
    if failure == "files_502":
        verify.side_effect = error
    else:
        read.side_effect = error
    with pytest.raises(type(error)):
        await api.create_preview(view.id, api.MotionPreviewIn(path="samples/Walk/walk.csv", start_frame=4),
                                 session, Principal("kk", "user"))
    submit.assert_not_called()


@pytest.mark.asyncio
async def test_reusing_successful_preview_rechecks_cached_artifacts_without_another_job(view, monkeypatch):
    session, read, verify, submit = completed_preview_request(view, monkeypatch)
    body = api.MotionPreviewIn(path="samples/Walk/walk.csv", start_frame=4)
    result = await api.create_preview(view.id, body, session, Principal("kk", "user"))
    assert result["id"] == "job_preview" and result["status"] == "succeeded"
    assert result["preview"]["video_url"].endswith("/files/replay.mp4")
    read.assert_awaited_once()
    verify.assert_awaited_once()
    submit.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["replay.mp4", "table/000000.json"])
async def test_preview_file_stream_does_not_load_expired_run_after_rollback(view, monkeypatch, path):
    job = preview_job(view)
    receipt = json.dumps(receipt_for(view))
    run = Run(id="run_preview", job_id=job.id, status="finished", metrics_summary={
        "motiondecode_preview_receipt_sha256": hashlib.sha256(receipt.encode()).hexdigest()})
    session = session_for(view, job, run)

    def expire_run():
        run._sa_instance_state._expire_attributes(run.__dict__, ["id"])

    session.rollback.side_effect = expire_run
    monkeypatch.setattr(api.EbdClient, "read_storage_text", AsyncMock(return_value=receipt))
    stream = AsyncMock(return_value=Response(content=b"preview"))
    monkeypatch.setattr(api, "stream_storage_file", stream)
    transport = httpx.ASGITransport(app=app_for(session))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(
            f"/api/v1/datasets/{view.id}/motiondecode/previews/{job.id}/files/{path}")
    assert response.status_code == 200
    assert "id" not in run.__dict__
    session.rollback.assert_awaited_once()
    assert stream.await_args.args[2] == path
    assert stream.await_args.args[1] == api.storage_root(
        api.get_settings().data_root / "runs/run_preview/outputs")
