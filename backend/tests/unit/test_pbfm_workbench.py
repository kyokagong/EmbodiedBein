"""Read-only workbench views of real protected training Jobs."""
from copy import deepcopy
from datetime import UTC, datetime
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import HTTPException
from test_pbfm_training_validation import resumed_spec, validation_spec

from eb.api import routes_train
from eb.store.models import Job, ModelVersion


def fixture_record(*, status="succeeded"):
    spec = validation_spec()
    job = NS(id="job_depth", kind="asset.validate", name=spec.name, status=status,
             spec=spec.model_dump(mode="json"), error=None,
             created_at=datetime(2026, 10, 22, tzinfo=UTC))
    summary = {"optimizer_steps": 112, "local_time": 26624, "global_time": 106496,
               "diagnostic": True, "diagnostic_mode": "training_validation",
               "training_completed": True, "full_state_finite_verified": True,
               "model_version_id": "mv_depth", "qualification": "incomplete",
               "checkpoint": {"path": "/frozen/checkpoint", "bytes": 1000,
                              "step": 112, "mode": "pbfm_native_state"},
               "training_evidence": {"available": True, "run_id": "run_depth", "files": []}}
    run = NS(id="run_depth", job_id=job.id, attempt=1,
             status="finished" if status == "succeeded" else "running",
             metrics_summary=summary if status == "succeeded" else {},
             gpu_ids=[0, 1, 2, 3], started_at=job.created_at, finished_at=None)
    return job, run


def read_session(job, run, *, reports=()):
    async def get(cls, key):
        return job if cls is Job and key == job.id else None

    return NS(get=AsyncMock(side_effect=get), scalar=AsyncMock(return_value=run),
              scalars=AsyncMock(return_value=NS(all=lambda: list(reports))),
              add=Mock(), commit=AsyncMock(), flush=AsyncMock())


def test_original_job_identity_is_preserved_in_the_training_view():
    from eb.pbfm_workbench import record_for

    job, run = fixture_record()
    before = deepcopy(job.spec)
    view = record_for(job, run)
    assert view["id"] == view["job_id"] == job.id
    assert view["record_kind"] == "pbfm_validation" and view["job_kind"] == "asset.validate"
    assert view["kind"] == "native_rl" and view["run_id"] == run.id
    assert view["params"]["native_train_plan"]["target_updates"] == 112
    assert view["metrics_summary"]["model_version_id"] == "mv_depth"
    assert view["metrics_summary"]["qualification"] == "incomplete"
    assert job.spec == before and "template_id" not in job.spec["params"]


@pytest.mark.parametrize("change", [
    {"probe_mode": "runtime"}, {"probe_mode": "segment112"},
    {"operation": "other"}, {"probe_fail_rank": 0},
])
def test_other_probes_never_become_training_records(change):
    from eb.pbfm_workbench import is_workbench_job

    job, _ = fixture_record()
    job.spec["params"].update(change)
    assert is_workbench_job(job) is False


def test_queue_without_a_run_has_no_fabricated_progress():
    from eb.pbfm_workbench import record_for

    job, _ = fixture_record(status="queued")
    view = record_for(job, None)
    assert view["run_id"] is None and view["metrics_summary"] == {}
    assert view["status"] == "queued" and view["gpus_allocated"] == []


@pytest.mark.asyncio
async def test_train_list_includes_real_job_without_creating_trainjob(monkeypatch):
    job, run = fixture_record()
    session = read_session(job, run)
    session.scalars.side_effect = [NS(all=lambda: []), NS(all=lambda: [job]), NS(all=lambda: [run])]
    result = await routes_train.list_train(session, None)
    assert [row["id"] for row in result["items"]] == [job.id]
    session.add.assert_not_called()
    session.commit.assert_not_awaited()
    session.flush.assert_not_awaited()


@pytest.mark.asyncio
async def test_original_job_detail_checkpoint_and_evidence_are_read_only():
    job, run = fixture_record()
    session = read_session(job, run)
    # Detail also queries diagnostic continuations.
    session.scalars.side_effect = [NS(all=lambda: [job]), NS(all=lambda: [run])]
    view = await routes_train.get_train(job.id, session, None)
    assert view["id"] == job.id and view["continuations"] == []
    checkpoints = await routes_train.list_checkpoints(job.id, session, None)
    assert checkpoints["items"][0]["registered_model_id"] == "mv_depth"
    assert checkpoints["items"][0]["step"] == 112
    evidence = await routes_train.native_training_evidence(job.id, session, None)
    assert evidence == run.metrics_summary["training_evidence"]
    session.add.assert_not_called()
    session.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_unknown_or_non_training_job_remains_404():
    job, run = fixture_record()
    job.spec["params"]["probe_mode"] = "runtime"
    with pytest.raises(HTTPException) as error:
        await routes_train.get_train(job.id, read_session(job, run), None)
    assert error.value.status_code == 404


@pytest.mark.asyncio
async def test_resume_hint_requires_complete_own_validation_but_no_file_scan():
    from eb.pbfm_evaluation import SUITE_ID
    from eb.pbfm_workbench import record_for, resume_candidate

    job, run = fixture_record()
    session = read_session(job, run)
    blocked = await resume_candidate(session, record_for(job, run))
    assert blocked["available"] is False
    report = NS(id="ev_depth", suite_id=SUITE_ID, model_version_id="mv_depth",
                gate_results={"validity": "complete"},
                metrics={"planned": 32, "completed": 32, "valid": 32, "invalid": 0,
                         "infra_failed": 0, "cancelled": 0, "not_started": 0, "quality_passed": False})
    session.scalars.side_effect = None
    session.scalars.return_value = NS(all=lambda: [report])
    ready = await resume_candidate(session, record_for(job, run))
    assert ready["available"] is True and ready["suggested_target"] == 368
    assert ready["request"]["resume_job_id"] == job.id
    assert ready["request"]["validation_report_id"] == report.id
    assert ready["request"]["num_envs_per_rank"] == 2048
    report.model_version_id = "mv_other"
    assert (await resume_candidate(session, record_for(job, run)))["available"] is False


@pytest.mark.asyncio
async def test_only_explicit_pbfm_page_capability_is_available_while_template_stays_closed():
    response = await routes_train.train_templates(NS(actor_type="user"))
    template = next(row for row in response["items"] if row["id"] == "rpo_pbfm_fb_depth_native")
    assert template["ready"] is False
    assert template["controlled_training"]["gpus"] == 4
    assert template["controlled_training"]["target_updates"] == 112
    assert all(not row.get("controlled_training") for row in response["items"] if row is not template)


@pytest.mark.asyncio
async def test_model_list_links_only_the_model_owned_by_the_real_training_run():
    from eb.pbfm_workbench import model_links

    job, run = fixture_record()
    session = read_session(job, run)
    session.scalars.side_effect = [NS(all=lambda: [job]), NS(all=lambda: [run])]
    links = await model_links(session)
    assert links == {"mv_depth": {"train_page_id": job.id, "train_job_name": job.name}}
    session.add.assert_not_called()


@pytest.mark.asyncio
async def test_model_routes_link_existing_versions_to_their_actual_producer():
    from eb.api.routes import get_model, list_models

    job, run = fixture_record()
    model = ModelVersion(id="mv_depth", name="depth", artifact_id=None, embodiment_env_id=None,
                         card={"schema_version": "eb/rpo-pbfm-trained-model/v1", "train_job_id": job.id})
    session = read_session(job, run)
    session.scalars.side_effect = [NS(all=lambda rows=rows: rows) for rows in
                                  ([], [model], [], [], [job], [run])]
    result = await list_models(session, None)
    assert result["versions"][0]["train_page_id"] == job.id
    assert result["versions"][0]["train_job_name"] == job.name

    session.get = AsyncMock(return_value=model)
    session.scalar.return_value = None
    session.scalars.side_effect = [NS(all=lambda rows=rows: rows) for rows in ([job], [run], [])]
    result = await get_model(model.id, session, None)
    assert result["train_page_id"] == job.id and result["card"] == model.card
    session.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_final_512_record_keeps_resume_offset_and_does_not_offer_long_training():
    from eb.pbfm_workbench import checkpoints, record_for, resume_candidate

    job, run = fixture_record()
    spec = resumed_spec(validation_spec(), target=512)
    # The shared fixture derives from 112; freeze the actual previous 368 counts here.
    plan = spec.params["native_train_plan"]
    plan.update(start_updates=368, start_local_time=61440)
    from eb.pbfm_training import native_arguments
    plan["native_args"] = native_arguments(plan)
    spec.params["resume_step"] = 368
    job.spec = spec.model_dump(mode="json")
    run.metrics_summary.update(optimizer_steps=512, local_time=81920, global_time=327680)
    run.metrics_summary["checkpoint"]["step"] = 512
    record = record_for(job, run)
    assert record["params"]["native_train_plan"]["start_updates"] == 368
    assert checkpoints(record)[0]["step"] == 512
    candidate = await resume_candidate(read_session(job, run), record)
    assert candidate["available"] is False and "512" in candidate["reason"]
