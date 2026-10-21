import hashlib
import json
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException


def indexed_evidence(tmp_path, monkeypatch):
    output = tmp_path / "outputs"
    summary = {"schema_version": "eb/ufo-native-training-analysis/v1", "run_id": "run_native",
               "target_updates": 112, "decision": "wait_for_independent_validation",
               "resources": {"status": "snapshot"}}
    values = {"analysis/summary.json": json.dumps(summary).encode(),
              "analysis/train-log-snapshot.csv": b"distributed/optimizer_steps,fb_loss\n16,2\n112,1\n",
              "native/train_log.txt": b"distributed/optimizer_steps,fb_loss\n16,2\n112,1\n",
              "analysis/resource-snapshot.csv": b"wall_unix_s,memory_mib\n1792320934,100\n",
              "analysis/training-1.png": b"training chart bytes", "analysis/resources.png": b"resource chart bytes"}
    files = []
    for relative, value in values.items():
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
        files.append({"path": relative, "bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()})
    index = {"schema_version": "eb/ufo-native-metrics-index/v1", "run_id": "run_native",
             "decision": "wait_for_independent_validation", "files": files}
    (output / "metrics-index.json").write_text(json.dumps(index))

    async def scan(path):
        from pathlib import Path
        return {"files": 1, "samples": [{"sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}]}

    monkeypatch.setattr("eb.orchestrator.ufo_native.EbdClient", lambda: NS(scan=scan))
    return index


@pytest.mark.asyncio
async def test_verified_evidence_preserves_raw_paths_and_produces_accessible_links(tmp_path, monkeypatch):
    from eb.orchestrator.collect import read_native_training_evidence

    index = indexed_evidence(tmp_path, monkeypatch)
    result = await read_native_training_evidence(tmp_path, "run_native", target_updates=112)
    assert result["available"] is True
    assert result["decision"] == "wait_for_independent_validation"
    assert {entry["path"] for entry in result["files"]} == {"metrics-index.json", *[f["path"] for f in index["files"]]}
    assert all(entry["url"] == f"/api/v1/ufo-native/runs/run_native/files/{entry['path']}"
               for entry in result["files"])
    assert (tmp_path / "outputs/native/train_log.txt").read_bytes().endswith(b"112,1\n")


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["schema", "run", "missing_plot", "duplicate", "escape", "hash", "summary"])
async def test_evidence_refuses_incomplete_misattributed_or_changed_files(tmp_path, monkeypatch, change):
    from eb.orchestrator.collect import read_native_training_evidence

    index = indexed_evidence(tmp_path, monkeypatch)
    if change == "schema":
        index["schema_version"] = "other"
    elif change == "run":
        index["run_id"] = "run_other"
    elif change == "missing_plot":
        index["files"] = [f for f in index["files"] if f["path"] != "analysis/training-1.png"]
    elif change == "duplicate":
        index["files"].append(index["files"][0])
    elif change == "escape":
        index["files"][0]["path"] = "analysis/../../secret.json"
    elif change == "hash":
        # Same-length corruption must be caught by the ebd hash, not just stat().
        (tmp_path / "outputs/analysis/training-1.png").write_bytes(b"changed! chart bytes")
    else:
        path = tmp_path / "outputs/analysis/summary.json"
        value = json.loads(path.read_text())
        value["run_id"] = "run_other"
        path.write_text(json.dumps(value))
        index["files"][0].update(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (tmp_path / "outputs/metrics-index.json").write_text(json.dumps(index))
    with pytest.raises(ValueError):
        await read_native_training_evidence(tmp_path, "run_native", target_updates=112)


@pytest.mark.asyncio
async def test_invalid_evidence_is_checked_before_any_training_model_publication(tmp_path, monkeypatch):
    from eb.orchestrator.collect import collect_ufo_native_training
    from eb.ufo_training_checkpoint import canonical_hash

    plan = {"target_updates": 112}
    checkpoint = {"counts": {"optimizer_steps": 112}, "manifest": {
        "plan_hash": canonical_hash({**plan, "run_id": "run_native"})}}
    monkeypatch.setattr("eb.ufo_training_checkpoint.read_native_checkpoint", lambda *_args, **_kwargs: checkpoint)
    rows = []
    session = NS(add=rows.append)
    job = NS(spec={"params": {"native_train_plan": plan}})
    with pytest.raises(ValueError, match="Missing|unsafe"):
        await collect_ufo_native_training(session, job, NS(id="run_native"), tmp_path)
    assert not rows


@pytest.mark.asyncio
async def test_historical_evidence_endpoint_reads_disk_without_database_writes(tmp_path, monkeypatch):
    from eb.api.routes_train import native_training_evidence
    from eb.store.models import Job, TrainJob

    root = tmp_path / "runs/run_native"
    indexed_evidence(root, monkeypatch)
    train = NS(template_id="rpo_ufo_fb_native", job_id="job_native")
    job = NS(status="succeeded", spec={"params": {"native_train_plan": {"target_updates": 112}}})
    run = NS(id="run_native", status="finished")
    session = NS(get=AsyncMock(side_effect=lambda model, _key: train if model is TrainJob else job),
                 scalar=AsyncMock(return_value=run), commit=AsyncMock(), add=AsyncMock())
    monkeypatch.setattr("eb.api.routes_train.get_settings", lambda: NS(data_root=tmp_path))
    result = await native_training_evidence("trn_native", session, NS())
    assert result["run_id"] == run.id and result["available"] is True
    session.commit.assert_not_awaited()
    session.add.assert_not_called()
    run.status = "aborted"
    assert (await native_training_evidence("trn_native", session, NS()))["available"] is False
    run.status = "finished"
    (root / "outputs/metrics-index.json").unlink()
    with pytest.raises(HTTPException) as missing:
        await native_training_evidence("trn_native", session, NS())
    assert missing.value.status_code == 409
