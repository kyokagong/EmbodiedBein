"""Explicit single-Job 512→10000 exploration preserves the producer identity."""
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError
from test_pbfm_training_validation import resumed_spec, source_fixture, validation_spec

from eb.pbfm_runtime_probe import ProbeIn, _training_view, build_spec, render, validate_submission
from eb.pbfm_training import native_arguments


def request(**changes):
    return ProbeIn(**{
        "name": "direct 10000", "dataset_id": "ds_pbfm", "environment_id": "env_train_pbfm",
        "mode": "training_exploration", "gpus": 4, "num_envs_per_rank": 2048,
        "target_updates": 10000, "resume_job_id": "job_source", "validation_report_id": "ev_source",
        "timeout_seconds": 21600, **changes,
    })


def exploration_source(tmp_path, monkeypatch):
    session, _, job, run, checkpoint, common = source_fixture(tmp_path, monkeypatch, target=368)
    source = resumed_spec(validation_spec(tmp_path), target=512)
    plan = source.params["native_train_plan"]
    plan.update(start_updates=368, start_local_time=61440)
    source.params["resume_step"] = 368
    plan["native_args"] = native_arguments(plan)
    job.spec = source.model_dump(mode="json")
    run.metrics_summary["optimizer_steps"] = 512
    checkpoint.update(step=512, counts={"optimizer_steps": 512, "local_time": 81920})
    return session, source, job, run, checkpoint, common


def test_single_job_exploration_requires_explicit_mode_and_exact_target():
    assert request().target_updates == 10000
    for change in (
        {"mode": "training_validation"}, {"mode": "runtime"}, {"target_updates": 1008},
        {"target_updates": 10016}, {"target_updates": 112}, {"gpus": 1},
        {"num_envs_per_rank": 20248}, {"num_envs_per_rank": 1024},
        {"resume_job_id": None}, {"validation_report_id": None}, {"failure_rank": 0},
    ):
        with pytest.raises(ValidationError):
            request(**change)


@pytest.mark.asyncio
async def test_full_native_512_source_builds_one_10000_job_with_the_same_image_and_identity(
    tmp_path, monkeypatch,
):
    session, source, job, run, checkpoint, common = exploration_source(tmp_path, monkeypatch)
    spec = await build_spec(request(), session)
    assert spec.kind == "asset.validate" and spec.params["probe_mode"] == "training_exploration"
    assert spec.inputs.resume_from == run.id and spec.image == source.image
    assert spec.params["source_diagnostic_job_id"] == job.id
    assert spec.params["resume_step"] == 512
    assert "source_train_job_id" not in spec.params
    plan = _training_view(spec).params["native_train_plan"]
    assert plan["identity"] == source.params["native_train_plan"]["identity"]
    assert (plan["start_updates"], plan["start_local_time"], plan["target_updates"]) == (512, 81920, 10000)
    assert plan["native_args"][plan["native_args"].index("--num-env-steps") + 1] == "5193728"
    assert "--probe" not in render(spec, "run_preview").command
    assert spec.policy.timeout_h == 6 and spec.policy.max_retries == 0
    await validate_submission(session, spec)
    assert common.await_args.args[1].params["native_train_plan"]["identity"] == plan["identity"]


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["source_step", "checkpoint", "mode", "report"])
async def test_exploration_rejects_different_source_or_unbound_resume(tmp_path, monkeypatch, change):
    from eb import pbfm_diagnostic_training as diagnostic

    session, source, job, run, checkpoint, common = exploration_source(tmp_path, monkeypatch)
    if change == "source_step":
        job.spec = validation_spec(tmp_path).model_dump(mode="json")
    elif change == "checkpoint":
        checkpoint["full_state_finite_verified"] = False
    elif change == "mode":
        job.spec["params"]["probe_mode"] = "runtime"
    else:
        monkeypatch.setattr(
            diagnostic, "validation_binding", AsyncMock(side_effect=ValueError("wrong report")),
        )
    with pytest.raises(ValueError):
        await build_spec(request(), session)


@pytest.mark.asyncio
async def test_exploration_collector_and_training_view_use_actual_training(tmp_path, monkeypatch):
    from test_pbfm_workbench import fixture_record

    from eb import pbfm_runtime_probe
    from eb.orchestrator import pbfm_training
    from eb.pbfm_workbench import is_workbench_job, record_for, resume_candidate

    session, source, job, run, checkpoint, common = exploration_source(tmp_path, monkeypatch)
    spec = await build_spec(request(), session)
    collect = AsyncMock()
    monkeypatch.setattr(pbfm_training, "collect_pbfm_training", collect)
    row, final_run = fixture_record()
    row.spec = spec.model_dump(mode="json")
    final_run.metrics_summary["optimizer_steps"] = 10000
    final_run.metrics_summary["checkpoint"]["step"] = 10000
    await pbfm_runtime_probe.collect(session, row, final_run, tmp_path / "outputs")
    assert collect.await_args.kwargs["diagnostic_spec"] == _training_view(spec)
    assert is_workbench_job(row)
    record = record_for(row, final_run)
    assert record["training_mode"] == "training_exploration"
    assert (await resume_candidate(session, record))["available"] is False


@pytest.mark.asyncio
async def test_exploration_training_analysis_keeps_authenticated_integrity_checks(tmp_path, monkeypatch):
    import test_pbfm_training_validation as validation_tests

    session, source, job, run, checkpoint, common = exploration_source(tmp_path, monkeypatch)
    spec = await build_spec(request(), session)
    monkeypatch.setattr(validation_tests, "validation_spec", lambda *_: spec)
    await validation_tests.test_training_validation_media_uses_analysis_integrity_index(tmp_path, monkeypatch)


@pytest.mark.asyncio
async def test_exploration_reserves_training_capacity_not_two_environment_probe(tmp_path, monkeypatch):
    from eb.capacity import estimate_job

    session, *_ = exploration_source(tmp_path, monkeypatch)
    spec = await build_spec(request(), session)
    estimate = estimate_job(spec)
    assert estimate.per_gpu_peak_gb == 23.5 and estimate.verdict == "unknown"
    assert estimate.basis["inputs"]["num_envs_per_rank"] == 2048
    assert estimate_job(spec, host_vram_gb=8).verdict == "no_fit"


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [None, "source", "checkpoint", "report", "profile"])
async def test_queued_exploration_revalidates_before_start_and_blocks_changed_evidence(
    tmp_path, monkeypatch, failure,
):
    from test_pbfm_training_validation import admission_fixture

    from eb import pbfm_runtime_probe
    from eb.orchestrator.service import Orchestrator

    source_session, *_ = exploration_source(tmp_path, monkeypatch)
    spec = await build_spec(request(), source_session)
    session, job, client, *_ = await admission_fixture(tmp_path, monkeypatch, target=512)
    job.spec = spec.model_dump(mode="json")
    validate = AsyncMock(side_effect=ValueError(f"changed {failure}") if failure else None)
    monkeypatch.setattr(pbfm_runtime_probe, "validate_submission", validate)
    if failure:
        with pytest.raises(ValueError, match="changed"):
            await Orchestrator(client).admit_and_start(session, job, [0, 1, 2, 3])
        assert job.status == "blocked" and job.attempts == 0
        client.submit_run.assert_not_awaited()
    else:
        await Orchestrator(client).admit_and_start(session, job, [0, 1, 2, 3])
        assert job.status == "running" and job.attempts == 1
        client.submit_run.assert_awaited_once()
    validate.assert_awaited_once()
