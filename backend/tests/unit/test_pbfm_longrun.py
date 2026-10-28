"""Protect100000 continuations without opening historical probes or changing training semantics."""
from copy import deepcopy
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from eb.pbfm_runtime_probe import ProbeIn
from eb.pbfm_training import native_arguments, segment_endpoint, training_fingerprint, validate_frozen_plan

BASE = "sha256:8a1fea0322cee1af4ba8e98d76c1af20c88ff885057a49ab63ceaf2d6ec2c7de"
NEW = "sha256:" + "c" * 64


def request(**changes):
    return ProbeIn(**{
        "name": "continue100000", "dataset_id": "ds_pbfm", "environment_id": "env_train_pbfm",
        "mode": "training_long_exploration", "gpus": 4, "num_envs_per_rank": 2048,
        "target_updates": 100000, "resume_job_id": "job_source", "validation_report_id": "ev_source",
        "timeout_seconds": 108000, **changes,
    })


def source_plan(tmp_path):
    from test_pbfm_training_validation import validation_spec

    plan = deepcopy(validation_spec(tmp_path).params["native_train_plan"])
    plan["identity"]["image_digest"] = plan["image_digest"] = BASE
    plan["fingerprint"] = training_fingerprint(plan["identity"])
    return plan


def long_plan(tmp_path):
    from eb.pbfm_longrun import long_exploration_metadata

    plan = source_plan(tmp_path)
    metadata = long_exploration_metadata(plan["identity"], NEW)
    plan["identity"]["image_digest"] = plan["image_digest"] = NEW
    plan["fingerprint"] = training_fingerprint(plan["identity"])
    plan.update(target_updates=100000, start_updates=10000, start_local_time=1298432,
                segment_timeout_seconds=108000, communication_timeout_seconds=900,
                long_exploration=metadata,
                resume={"mode": "pbfm_native_state", "replay_policy": "restore"})
    plan["native_args"] = native_arguments(plan)
    return plan


def test_100000_request_requires_its_own_mode_and_bounded_resources():
    assert request().target_updates == 100000
    for changes in (
        {"mode": "training_exploration"}, {"mode": "training_validation"},
        {"target_updates": 10000}, {"target_updates": 100016}, {"timeout_seconds": 108001},
        {"gpus": 1}, {"num_envs_per_rank": 1024}, {"resume_job_id": None},
        {"validation_report_id": None}, {"failure_rank": 0},
    ):
        with pytest.raises(ValidationError):
            request(**changes)


def test_native_100000_resumed_endpoint_is_not_cold_start_or_rank_multiplied():
    assert segment_endpoint(100000, 4, start_updates=10000,
                            start_local_time=1298432, num_envs=2048) == 51281920
    with pytest.raises(ValueError):
        segment_endpoint(100016, 4, start_updates=10000, start_local_time=1298432, num_envs=2048)


def test_bridge_retains_old_checkpoint_identity_and_new_producer_identity(tmp_path):
    from eb.pbfm_longrun import checkpoint_source_plan

    source = source_plan(tmp_path)
    plan = long_plan(tmp_path)
    validate_frozen_plan(plan)
    projected = checkpoint_source_plan(plan)
    assert projected["image_digest"] == BASE and plan["image_digest"] == NEW
    assert projected["identity"] == source["identity"]
    assert projected["fingerprint"] == source["fingerprint"] != plan["fingerprint"]
    assert plan["native_args"][plan["native_args"].index("--checkpoint-every-steps") + 1] == "5120000"


@pytest.mark.parametrize("change", ["seed", "data", "world", "source_image", "receipt", "budget"])
def test_long_bridge_rejects_semantic_changes_even_with_recomputed_fingerprint(tmp_path, change):
    plan = long_plan(tmp_path)
    if change == "seed":
        plan["identity"]["seed"] += 1
    elif change == "data":
        plan["identity"]["dataset_manifest"] = "f" * 64
    elif change == "world":
        plan["world_size"] = plan["identity"]["world_size"] = 2
    elif change == "source_image":
        plan["long_exploration"]["source_identity"]["image_digest"] = NEW
    elif change == "receipt":
        plan["long_exploration"]["runtime_receipt_sha256"] = "sha256:" + "0" * 64
    else:
        plan["segment_timeout_seconds"] = 108001
    plan["fingerprint"] = training_fingerprint(plan["identity"])
    plan["native_args"] = native_arguments(plan)
    with pytest.raises(ValueError):
        validate_frozen_plan(plan)


def test_old_plan_cannot_gain_30_hour_budget_without_explicit_long_protocol(tmp_path):
    plan = source_plan(tmp_path)
    plan["segment_timeout_seconds"] = 108000
    with pytest.raises(ValueError):
        validate_frozen_plan(plan)


async def long_source(tmp_path, monkeypatch):
    from test_pbfm_exploration import exploration_source
    from test_pbfm_exploration import request as short_request

    from eb import pbfm_diagnostic_training as diagnostic
    from eb import pbfm_training_service as service
    from eb.pbfm_runtime_probe import build_spec

    session, _, job, run, checkpoint, common = exploration_source(tmp_path, monkeypatch)
    source = await build_spec(short_request(), session)
    plan = source.params["native_train_plan"]
    source.image = plan["identity"]["image_digest"] = plan["image_digest"] = BASE
    plan["fingerprint"] = training_fingerprint(plan["identity"])
    job.spec = source.model_dump(mode="json")
    run.metrics_summary.update(diagnostic_mode="training_exploration", optimizer_steps=10000)
    checkpoint.update(image_digest=BASE, compatibility_fingerprint=plan["fingerprint"], step=10000,
                      counts={"optimizer_steps": 10000, "local_time": 1298432, "global_time": 5193728})
    available = {"digest": NEW}
    from eb.pbfm_longrun import LONG_IMAGE

    monkeypatch.setattr(diagnostic, "EbdClient", lambda: NS(
        get_run=AsyncMock(return_value={"status": "exited"}),
        image_info=AsyncMock(side_effect=lambda ref: {
            "ok": True, "digest": available["digest"] if ref == LONG_IMAGE else BASE,
        }),
    ))
    monkeypatch.setattr(service, "_validate_frozen_inputs", common, raising=False)
    monkeypatch.setattr(service, "_validate_current_runtime", AsyncMock(), raising=False)
    return session, source, job, run, checkpoint, available


@pytest.mark.asyncio
async def test_real_long_builder_preserves_native_source_and_expands_only_its_scoped_budget(
    tmp_path, monkeypatch,
):
    from eb.pbfm_runtime_probe import _training_view, build_spec, render, validate_submission

    session, source, job, run, checkpoint, _ = await long_source(tmp_path, monkeypatch)
    spec = await build_spec(request(), session)
    plan = _training_view(spec).params["native_train_plan"]
    assert spec.image == NEW and source.image == checkpoint["image_digest"] == BASE
    assert plan["long_exploration"]["source_fingerprint"] == checkpoint["compatibility_fingerprint"]
    assert spec.params["source_diagnostic_job_id"] == job.id and spec.inputs.resume_from == run.id
    assert plan["target_updates"] == 100000 and plan["start_updates"] == 10000
    assert spec.policy.timeout_h == 30 and spec.policy.max_retries == 0
    assert "--probe" not in render(spec, "run_long_preview").command
    await validate_submission(session, spec)


@pytest.mark.asyncio
@pytest.mark.parametrize("changed", ["image", "source_mode", "counts", "report"])
async def test_long_admission_rejects_changed_frozen_source_or_image(tmp_path, monkeypatch, changed):
    from eb import pbfm_diagnostic_training as diagnostic
    from eb.pbfm_runtime_probe import build_spec, validate_submission

    session, source, job, run, checkpoint, available = await long_source(tmp_path, monkeypatch)
    spec = await build_spec(request(), session)
    if changed == "image":
        available["digest"] = "sha256:" + "d" * 64
    elif changed == "source_mode":
        job.spec["params"]["probe_mode"] = "runtime"
    elif changed == "counts":
        checkpoint["counts"]["local_time"] += 2048
    else:
        monkeypatch.setattr(diagnostic, "validation_binding",
                            AsyncMock(side_effect=ValueError("report changed")))
    with pytest.raises(ValueError):
        await validate_submission(session, spec)


@pytest.mark.asyncio
async def test_long_training_uses_existing_collection_capacity_and_training_view(tmp_path, monkeypatch):
    from test_pbfm_workbench import fixture_record

    from eb import pbfm_runtime_probe
    from eb.capacity import estimate_job
    from eb.orchestrator import pbfm_training
    from eb.pbfm_workbench import is_workbench_job, record_for, resume_candidate

    session, *_ = await long_source(tmp_path, monkeypatch)
    spec = await pbfm_runtime_probe.build_spec(request(), session)
    estimate = estimate_job(spec)
    assert estimate.per_gpu_peak_gb == 23.5 and estimate.verdict == "unknown"
    assert estimate_job(spec, host_vram_gb=8).verdict == "no_fit"
    row, run = fixture_record()
    row.spec = spec.model_dump(mode="json")
    run.metrics_summary.update(optimizer_steps=100000)
    run.metrics_summary["checkpoint"]["step"] = 100000
    collect = AsyncMock()
    monkeypatch.setattr(pbfm_training, "collect_pbfm_training", collect)
    await pbfm_runtime_probe.collect(session, row, run, tmp_path / "outputs")
    assert collect.await_args.kwargs["diagnostic_spec"] == pbfm_runtime_probe._training_view(spec)
    assert is_workbench_job(row)
    view = record_for(row, run)
    assert view["training_mode"] == "training_long_exploration"
    candidate = await resume_candidate(session, view)
    assert candidate["available"] is False and "100000" in candidate["reason"]


@pytest.mark.asyncio
async def test_long_training_analysis_uses_collected_integrity_index(tmp_path, monkeypatch):
    import test_pbfm_training_validation as validation_tests

    from eb.pbfm_runtime_probe import build_spec

    session, *_ = await long_source(tmp_path, monkeypatch)
    spec = await build_spec(request(), session)
    monkeypatch.setattr(validation_tests, "validation_spec", lambda *_: spec)
    await validation_tests.test_training_validation_media_uses_analysis_integrity_index(tmp_path, monkeypatch)
