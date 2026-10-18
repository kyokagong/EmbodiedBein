import json
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


@pytest.mark.parametrize("encode", [lambda value: value, json.dumps])
def test_source_build_history_is_not_a_checkpoint_lease(encode):
    receipt = {"guardfix_parent_run": "run_old", "extension_parent_run": "run_old"}
    assert not references(encode({"params": {"source_receipt": receipt}}), {"run_old"}, [])
    assert references(
        encode({"inputs": {"resume_from": "run_old"}, "source_receipt": receipt}), {"run_old"}, []
    )
    assert references(
        encode({"source_receipt": {**receipt, "checkpoint_run_id": "run_old"}}), {"run_old"}, []
    )
    assert references(encode({"inputs": {"guardfix_parent_run": "run_old"}}), {"run_old"}, [])


@pytest.mark.asyncio
async def test_ended_diagnostic_cleanup_keeps_its_model_input():
    job = Job(
        id="job_probe",
        kind="asset.validate",
        spec={"inputs": {"base_model": "mv_source"}},
        status="succeeded",
    )
    run = Run(id="run_probe", job_id=job.id, status="finished")
    source = ModelVersion(id="mv_source", card={})
    plan = await deletion_plan(session_with(job, run, source), "job", job.id)
    assert plan["run_ids"] == [run.id]
    assert not plan["blockers"] and "mv_source" not in plan["ids"]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["train.rl", "eval.sim", "data.convert"])
async def test_diagnostic_delete_rejects_other_job_types(kind):
    job = Job(id="job_other", kind=kind, spec={}, status="succeeded")
    with pytest.raises(HTTPException) as exc:
        await deletion_plan(session_with(job), "job", job.id)
    assert exc.value.status_code == 409


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["train.rl", "eval.sim", "data.convert", "export.policy"])
@pytest.mark.parametrize("status", ["failed", "infra_failed", "timeout"])
async def test_failed_job_cleanup_handles_all_kinds_and_training_projection(kind, status):
    job = Job(id="job_failed", kind=kind, spec={"inputs": {"model": "mv_input"}}, status=status)
    run = Run(id="run_failed", job_id=job.id, status="aborted")
    train = TrainJob(id="trn_failed", job_id=job.id, status="failed", run_dir="/state/trn_legacy")
    model = ModelVersion(id="mv_input", card={})
    records = [job, run, model] + ([train] if kind.startswith("train.") else [])
    plan = await deletion_plan(session_with(*records), "job", job.id)
    assert plan["run_ids"] == [run.id] and not plan["blockers"]
    assert "mv_input" not in plan["ids"]
    assert ("trn_failed" in plan["ids"]) == kind.startswith("train.")
    assert plan["preserved_paths"] == ([train.run_dir] if kind.startswith("train.") else [])


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["queued", "running", "cancelled", "succeeded"])
async def test_generic_job_cleanup_does_not_expand_to_other_states(status):
    job = Job(id="job_other", kind="data.convert", spec={}, status=status)
    with pytest.raises(HTTPException) as exc:
        await deletion_plan(session_with(job), "job", job.id)
    assert exc.value.status_code == 409


@pytest.mark.asyncio
async def test_failed_download_cleanup_owns_only_its_failed_model_and_checks_consumers():
    job = Job(id="job_download", kind="data.convert", status="failed",
              spec={"params": {"operation": "download_model", "model_id": "mdl_partial"}})
    partial = WeightModel(id="mdl_partial", kind="download_failed", path="", metrics_at_checkpoint={
        "download": {"job_id": job.id, "model_path": "/data/kk/models/mdl_partial"}})
    source = WeightModel(id="mdl_source", kind="downloaded", path="/data/kk/models/mdl_source",
                         metrics_at_checkpoint={})
    plan = await deletion_plan(session_with(job, partial, source), "job", job.id)
    assert plan["ids"] == [job.id, partial.id]
    assert plan["paths"] == ["/data/kk/models/mdl_partial"] and not plan["blockers"]
    consumer = Job(id="job_consumer", kind="train.rl", status="succeeded",
                   spec={"inputs": {"base_model": partial.id}})
    plan = await deletion_plan(session_with(job, partial, source, consumer), "job", job.id)
    assert {b["id"] for b in plan["blockers"]} == {consumer.id}
    partial.kind = "downloaded"
    plan = await deletion_plan(session_with(job, partial), "job", job.id)
    assert partial.id not in plan["ids"] and plan["blockers"]
    partial.kind = "download_failed"
    partial.metrics_at_checkpoint["download"]["model_path"] = "/data/kk/RLinf/weights"
    plan = await deletion_plan(session_with(job, partial), "job", job.id)
    assert not plan["paths"] and plan["preserved_paths"] == ["/data/kk/RLinf/weights"]


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
async def test_model_output_summary_does_not_form_a_deletion_cycle_with_its_producer():
    records = train_records()
    records[1].metrics_summary = {"model_version_id": "mv_test"}
    artifact = Artifact(
        id="art_test", run_id=records[2].id, storage="/data/kk/models/runs/run_first", refcount=0
    )
    records[2].metrics_summary = {"model_version_id": "mv_test", "checkpoint": {"path": artifact.storage}}
    model = ModelVersion(id="mv_test", card={"run_id": records[2].id}, artifact_id=artifact.id)
    plan = await deletion_plan(session_with(*records, model, artifact), "model", model.id)
    assert not plan["blockers"]
    assert records[2].metrics_summary["model_version_id"] == model.id
    assert records[2].metrics_summary["checkpoint"]["path"] == artifact.storage
    consumer = Run(id="run_consumer", job_id="job_consumer", metrics_summary={"model_version_id": model.id})
    plan = await deletion_plan(session_with(*records, model, artifact, consumer), "model", model.id)
    assert {row["id"] for row in plan["blockers"]} == {consumer.id}


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
