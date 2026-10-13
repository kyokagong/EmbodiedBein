from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from eb.api.deps import Principal
from eb.orchestrator.deletion import delete_object, deletion_plan, guard_storage_inputs, references
from eb.store.models import (
    Artifact,
    AuditLog,
    EvalReport,
    Job,
    LineageEdge,
    ModelVersion,
    Run,
    TrainJob,
    WeightModel,
)


def session_with(*rows):
    async def scalars(statement):
        table = statement.column_descriptions[0]["entity"]
        return SimpleNamespace(all=lambda: [r for r in rows if isinstance(r, table)])

    return SimpleNamespace(scalars=scalars, execute=AsyncMock())


def train_records():
    return (
        Job(id="job_test", kind="train.rl", spec={}, status="failed"),
        TrainJob(id="trn_test", job_id="job_test", status="failed", run_dir=""),
        Run(id="run_first", job_id="job_test", status="aborted"),
        Run(id="run_second", job_id="job_test", status="finished"),
    )


def test_references_match_nested_ids_paths_and_overrides_without_prefix_collisions():
    paths = ["/state/runs/run_test"]
    assert references({"resume": {"run": "run_test"}}, {"run_test"}, [])
    assert references(["checkpoint=/state/runs/run_test/output/a.pt"], set(), paths)
    assert references("/state/runs", set(), paths)
    assert not references("/state/runs/run_test2/output", {"run_test"}, paths)
    assert not references("run_test2", {"run_test"}, [])


@pytest.mark.asyncio
async def test_training_deletes_all_attempts_and_keeps_models_as_blockers():
    records = train_records()
    model = ModelVersion(id="mv_test", card={"run_id": "run_first"})
    plan = await deletion_plan(session_with(*records, model), "train", "trn_test")
    assert plan["run_ids"] == ["run_first", "run_second"]
    assert len(plan["paths"]) == 6
    assert plan["blockers"][0]["id"] == "mv_test"
    assert "model_versions" not in plan["records"]


@pytest.mark.asyncio
async def test_eval_removes_own_reports_and_artifacts_without_deleting_input_model():
    job = Job(id="job_eval", kind="eval.sim", spec={"inputs": {"model": "mv_source"}}, status="cancelled")
    run = Run(id="run_eval", job_id=job.id, status="aborted")
    report = EvalReport(id="ev_test", run_id=run.id, model_version_id="mv_source")
    artifact = Artifact(id="art_test", run_id=run.id, pinned=False, refcount=0)
    source = ModelVersion(id="mv_source", card={})
    plan = await deletion_plan(session_with(job, run, report, artifact, source), "eval", job.id)
    assert plan["blockers"] == []
    assert plan["records"]["eval_reports"][0]["id"] == report.id
    assert "mv_source" not in plan["ids"]


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["queued", "running", "preempted", "preparing"])
async def test_active_jobs_cannot_be_deleted(status):
    records = train_records()
    records[0].status = status
    with pytest.raises(HTTPException) as exc:
        await deletion_plan(session_with(*records), "train", "trn_test")
    assert exc.value.status_code == 409


@pytest.mark.asyncio
async def test_shared_model_path_and_derived_lineage_are_protected():
    source = WeightModel(id="mdl_source", path="/data/kk/models/mdl_source", metrics_at_checkpoint={})
    shared = WeightModel(id="mdl_shared", path=source.path, metrics_at_checkpoint={})
    edge = LineageEdge(src_id=source.id, dst_id="mv_derived", dst_type="model_version")
    derived = ModelVersion(id="mv_derived", card={})
    plan = await deletion_plan(session_with(source, shared, edge, derived), "model", source.id)
    assert {b["id"] for b in plan["blockers"]} == {"mdl_shared", "mv_derived"}


@pytest.mark.asyncio
async def test_external_source_is_never_deleted():
    source = WeightModel(id="mdl_external", path="/data/kk/RLinf/weights", metrics_at_checkpoint={})
    plan = await deletion_plan(session_with(source), "model", source.id)
    assert plan["paths"] == []
    assert plan["preserved_paths"] == [source.path]


@pytest.mark.asyncio
async def test_legacy_ingest_does_not_restore_deleted_external_model(tmp_path, monkeypatch):
    from eb.ingest import ingest_checkpoints

    path = tmp_path / "model.pt"
    path.write_bytes(b"test")
    receipt = AuditLog(before={"kind": "model", "paths": [], "preserved_paths": [str(path)]})
    session = session_with(receipt)
    session.get = AsyncMock(return_value=SimpleNamespace(id="mf_legacy"))
    monkeypatch.setattr(
        "eb.ingest.sha256_file", lambda *_: pytest.fail("deleted external weight was re-ingested")
    )
    assert await ingest_checkpoints(session, root=tmp_path) == 0


@pytest.mark.asyncio
async def test_pending_and_completed_tombstones_block_new_consumers():
    receipt = AuditLog(before={"ids": ["run_test"], "paths": ["/state/runs/run_test"]})
    for state in ["pending", "completed"]:
        receipt.after = {"state": state}
        with pytest.raises(ValueError):
            await guard_storage_inputs(session_with(receipt), {"resume": "run_test"})


@pytest.mark.asyncio
async def test_agent_cannot_preview_or_delete():
    for preview in [False, True]:
        with pytest.raises(HTTPException) as exc:
            await delete_object(None, Principal("test", "agent"), "train", "trn_test", preview=preview)
        assert exc.value.status_code == 403
