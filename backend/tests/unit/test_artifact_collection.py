import hashlib
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import httpx
import pytest

from eb.store.ebd import EbdClient

RUN = "run_0123456789abcdef0123"


def settings(tmp_path, monkeypatch):
    value = NS(data_root=tmp_path, models_root=tmp_path / "models")
    monkeypatch.setattr("eb.store.ebd.get_settings", lambda: value)
    return value


def test_artifact_root_is_owned_and_preserves_nested_relative_path(tmp_path, monkeypatch):
    from eb.store.ebd import artifact_root
    settings(tmp_path, monkeypatch)
    assert artifact_root(tmp_path / "runs" / RUN / "outputs/native") == {
        "namespace": "run", "run_id": RUN, "path": "outputs/native",
    }
    assert artifact_root(tmp_path / "models/runs" / RUN) == {
        "namespace": "model_run", "run_id": RUN, "path": "",
    }
    for path in [tmp_path, tmp_path / "runs/not_run/output", Path("/etc"), tmp_path / "runs" / RUN / "../other"]:
        with pytest.raises(ValueError):
            artifact_root(path)


@pytest.mark.asyncio
async def test_native_verifier_sends_n_files_in_one_request(tmp_path, monkeypatch):
    from eb.orchestrator.ufo_native import verify_index
    settings(tmp_path, monkeypatch)
    root = tmp_path / "runs" / RUN / "outputs"
    root.mkdir(parents=True)
    records = []
    for i in range(3):
        path = root / f"file-{i}"
        path.write_bytes(b"state")
        records.append({"path": path.name, "bytes": 5, "sha256": hashlib.sha256(b"state").hexdigest()})
    verify = AsyncMock(return_value={"verified": True, "file_count": 3, "total_bytes": 15})
    monkeypatch.setattr(EbdClient, "verify_artifacts", verify)
    await verify_index(root, records)
    verify.assert_awaited_once_with({"namespace": "run", "run_id": RUN, "path": "outputs"}, records,
                                    max_files=200, max_file_bytes=2 * 1024**3)


@pytest.mark.asyncio
@pytest.mark.parametrize("source_change", ["rewrite", "remove"])
async def test_collector_reads_published_onnx_and_registers_its_identity(tmp_path, monkeypatch, source_change):
    from eb.orchestrator import collect
    from eb.store.models import Artifact, EmbodimentEnvRow, ModelFamily, ModelVersion

    config = settings(tmp_path, monkeypatch)
    monkeypatch.setattr(collect, "get_settings", lambda: config)
    source = tmp_path / "runs" / RUN / "outputs/policy.onnx"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"model")
    digest = hashlib.sha256(b"model").hexdigest()
    destination = tmp_path / "cas" / digest[:2] / digest

    async def publish(root, path, **expected):
        assert root == {"namespace": "run", "run_id": RUN, "path": "outputs"}
        assert path == "policy.onnx" and expected["expected_bytes"] == 5
        destination.parent.mkdir(parents=True)
        destination.write_bytes(b"model")
        if source_change == "remove":
            source.unlink()
        else:
            source.write_bytes(b"different source length")
        return {"sha256": digest, "size_bytes": 5, "storage": str(destination)}

    def signature(path):
        assert Path(path) == destination
        return {"obs": 45, "act": 12}

    def no_api_hash(_):
        pytest.fail("control plane still hashes the model payload")

    published = AsyncMock(side_effect=publish)
    monkeypatch.setattr(EbdClient, "publish_file", published)
    monkeypatch.setattr(collect, "read_signature", signature)
    monkeypatch.setattr(collect, "sha256_file", no_api_hash)
    added = []

    async def get(table, _):
        if table is ModelFamily:
            return NS(id="mf_run")
        assert table is EmbodimentEnvRow
        return None

    session = NS(get=get, scalar=AsyncMock(return_value=None), add=added.append, flush=AsyncMock())
    job = NS(kind="export.policy", spec={"params": {}}, embodiment_env_id="ee_test", status="running")
    run = NS(id=RUN, metrics_summary={})
    await collect.claim_outputs(session, job, run)
    assert published.await_count == 1
    artifact = next(item for item in added if isinstance(item, Artifact))
    assert (artifact.sha256, artifact.storage, artifact.size_bytes) == (digest, str(destination), 5)
    assert any(isinstance(item, ModelVersion) for item in added)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure,expected_status", [("transport", "infra_failed"), ("content", "failed")])
async def test_collection_io_failure_is_not_a_success_or_quality_receipt(tmp_path, monkeypatch, failure, expected_status):
    from eb.orchestrator.service import Orchestrator
    request = httpx.Request("POST", "http://ebd/v1/artifacts/verify")
    error = (httpx.ConnectError("unavailable", request=request) if failure == "transport" else
             httpx.HTTPStatusError("hash differs", request=request, response=httpx.Response(409, request=request)))
    monkeypatch.setattr("eb.orchestrator.collect.claim_outputs", AsyncMock(side_effect=error))
    monkeypatch.setattr("eb.orchestrator.collect.archive_partial_eval", AsyncMock())
    session = NS(flush=AsyncMock(), commit=AsyncMock(), refresh=AsyncMock(), begin_nested=AsyncMock(return_value=NS(commit=AsyncMock(), rollback=AsyncMock())))
    job, run = NS(status="running"), NS(id=RUN, status="running")
    await Orchestrator().collect_and_finish(session, job, run)
    assert job.status == expected_status
    assert run.status == "aborted"
    assert "产物" in job.error
    session.commit.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["rpc", "g3", "trace"])
async def test_failed_claim_rolls_back_earlier_registrations(tmp_path,monkeypatch,failure):
    from eb.orchestrator.service import Orchestrator
    pending,committed=[],[]
    monkeypatch.setattr("eb.orchestrator.service.get_settings",lambda:NS(data_root=tmp_path))
    if failure=="trace":
        log=tmp_path/"runs"/RUN/"console.log";log.parent.mkdir(parents=True)
        log.write_text("Traceback (most recent call last):\nruntime failed")
    async def claim(session,job,run):
        pending.append("first model")
        if failure=="rpc":
            raise httpx.ConnectError("second publication failed")
        if failure=="g3":
            job.status="failed"; job.error="G3 failed"; run.status="aborted"; run.abort_reason="g3_failed"
    monkeypatch.setattr("eb.orchestrator.collect.claim_outputs",claim)
    transaction=NS(commit=AsyncMock(),rollback=AsyncMock(side_effect=pending.clear))
    async def commit():
        committed.extend(pending)
    session=NS(flush=AsyncMock(),begin_nested=AsyncMock(return_value=transaction),refresh=AsyncMock(),commit=commit)
    job=NS(status="running",error=None)
    run=NS(id=RUN,status="running",abort_reason=None,metrics_summary={})
    await Orchestrator().collect_and_finish(session,job,run)
    assert committed==[]
    assert job.status==("infra_failed" if failure=="rpc" else "failed")
    assert run.status=="aborted"
    transaction.rollback.assert_awaited_once()
