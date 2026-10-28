from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock

import pytest
from pydantic import ValidationError
from test_pbfm_training_service import pbfm_spec

from eb.capacity import estimate_job
from eb.domain.job import JobKind, JobSpec
from eb.pbfm_runtime_probe import ProbeIn, _training_view, render
from eb.pbfm_training import TEMPLATE, native_arguments, training_fingerprint


def test_pbfm_native_motion_preparation_is_cpu_only():
    spec = JobSpec(
        kind=JobKind.DATA_CONVERT,
        embodiment_env_id="ee_rpo_v2_pbfm_depth_23dof_v1",
        resources={"gpus": 0},
        params={"operation": "prepare_pbfm_rpo_motion"},
    )
    estimate = estimate_job(spec)
    assert estimate.verdict == "fits"
    assert estimate.per_gpu_peak_gb == 0
    assert estimate.basis["inputs"]["optimizer_bytes"] == 0


def segment_spec(root=None):
    spec = pbfm_spec() if root is None else pbfm_spec(root)
    plan = spec.params["native_train_plan"]
    plan["identity"]["buffer_size"] = 32768
    plan["fingerprint"] = training_fingerprint(plan["identity"])
    plan["segment_timeout_seconds"] = 600
    plan["native_args"] = native_arguments(plan)
    spec.policy.timeout_h = 600 / 3600
    spec.kind = JobKind.ASSET_VALIDATE
    spec.params.pop("template_id")
    spec.params.update(
        operation="probe_pbfm_runtime", probe_recipe_id=TEMPLATE, probe_mode="segment112",
        environment_profile=plan["identity"]["environment_profile"],
    )
    return spec


def test_segment112_is_explicit_fixed_and_renders_native_training(tmp_path, monkeypatch):
    body = ProbeIn(name="diagnostic", dataset_id="ds_pbfm", environment_id="draft", mode="segment112")
    assert body.mode == "segment112"
    for change in ({"gpus": 2}, {"failure_rank": 0}):
        with pytest.raises(ValidationError, match="segment112"):
            ProbeIn(**{**body.model_dump(), **change})
    monkeypatch.setattr("eb.settings.get_settings", lambda: NS(data_root=tmp_path))
    spec = segment_spec(tmp_path)
    training = _training_view(spec)
    assert training.kind == "train.rl" and "probe_mode" not in training.params
    rendered = render(spec, "run_preview")
    assert "--probe" not in rendered.command
    assert rendered.metrics_glob.endswith("train_log.txt")
    assert spec.kind == "asset.validate" and "template_id" not in spec.params
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("field,value", [("target_updates", 128), ("world_size", 2), ("start_updates", 16)])
def test_segment112_rejects_tampered_plan(field, value):
    spec = segment_spec()
    spec.params["native_train_plan"][field] = value
    with pytest.raises(ValueError):
        _training_view(spec)


def test_segment112_capacity_uses_actual_training_reserve():
    spec = segment_spec()
    estimate = estimate_job(spec)
    assert estimate.basis["inputs"]["runtime_reserve_gib"] == 16
    assert estimate.basis["inputs"]["buffer_size_per_rank"] == 32768
    assert estimate.per_gpu_peak_gb > 17
    assert estimate.verdict == "unknown"
    assert estimate_job(spec, host_vram_gb=8).verdict == "no_fit"


@pytest.mark.parametrize("kind,operation", [
    (JobKind.DATA_CONVERT, "export_pbfm_policy"),
    (JobKind.EVAL_SIM, "evaluate_pbfm_terrain"),
])
def test_pbfm_single_environment_actions_use_inference_reserve(kind, operation):
    spec = JobSpec(kind=kind, embodiment_env_id="ee_rpo_v2_pbfm_depth_23dof_v1",
                   task_id="tsk_rpo_pbfm_terrain_locomotion_v1",
                   resources={"gpus": 1}, params={"operation": operation})
    estimate = estimate_job(spec)
    assert estimate.verdict == "unknown"
    assert estimate.per_gpu_peak_gb == 8
    assert estimate.basis["inputs"]["optimizer_bytes"] == 0
    assert estimate.basis["inputs"]["num_envs"] == 1
    assert estimate_job(spec, host_vram_gb=4).verdict == "no_fit"
    spec.embodiment_env_id = "ee_unrelated"
    assert estimate_job(spec).per_gpu_peak_gb != 8


@pytest.mark.parametrize("mutate", [
    lambda spec: spec.params.update(probe_mode="unknown"),
    lambda spec: spec.params.update(probe_fail_rank=0),
    lambda spec: setattr(spec.resources, "gpus", 2),
    lambda spec: spec.inputs.resume_from.update({"path": "/resume"}),
    lambda spec: spec.params["native_train_plan"]["identity"].update(buffer_size=65536),
    lambda spec: spec.overrides.append("--arbitrary"),
])
def test_segment112_rejects_boundary_mutation(mutate):
    spec = segment_spec()
    if spec.inputs.resume_from is None:
        spec.inputs.resume_from = {}
    mutate(spec)
    with pytest.raises(ValueError):
        _training_view(spec)


@pytest.mark.asyncio
async def test_segment_mode_is_frozen_in_preview_identity(monkeypatch):
    from eb import pbfm_runtime_probe, pbfm_training_service

    source = _training_view(segment_spec())
    build = AsyncMock(return_value=source)
    monkeypatch.setattr(pbfm_training_service, "_build_spec", build)
    args = {"name": "diagnostic", "dataset_id": "ds_pbfm", "environment_id": "draft"}
    runtime = await pbfm_runtime_probe.build_spec(ProbeIn(**args), NS())
    segment = await pbfm_runtime_probe.build_spec(ProbeIn(**args, mode="segment112"), NS())
    assert runtime.params["probe_mode"] == "runtime"
    assert segment.params["probe_mode"] == "segment112"
    assert runtime.canonical_hash() != segment.canonical_hash()
    assert build.await_args.kwargs == {"diagnostic": True}


@pytest.mark.asyncio
async def test_segment112_collect_keeps_original_job_and_explicitly_converts_spec(tmp_path, monkeypatch):
    from eb import pbfm_runtime_probe
    from eb.orchestrator import pbfm_training

    spec = segment_spec(tmp_path)
    collect = AsyncMock()
    monkeypatch.setattr(pbfm_training, "collect_pbfm_training", collect)
    job = NS(id="job_diagnostic", spec=spec.model_dump(mode="json"))
    run = NS(id="run_diagnostic")
    await pbfm_runtime_probe.collect(NS(), job, run, tmp_path / "outputs")
    assert job.spec["kind"] == "asset.validate"
    assert collect.await_args.args[1] is job
    assert collect.await_args.args[3] == tmp_path
    assert collect.await_args.kwargs["diagnostic_spec"] == _training_view(spec)


@pytest.mark.asyncio
async def test_segment112_publication_is_diagnostic_without_train_job(tmp_path, monkeypatch):
    import sys

    from test_pbfm_training_service import checkpoint_for

    from eb.orchestrator import pbfm_training as collector
    from eb.store.models import Artifact, ModelVersion

    spec = segment_spec(tmp_path)
    training = _training_view(spec)
    checkpoint = checkpoint_for(training, tmp_path)
    checkpoint.update(
        full_state_finite_verified=True, limitations=["environment is reset"], state_coverage={},
        manifest={
            "files": [{"path": "checkpoint/model/model.safetensors", "bytes": 10, "sha256": "b" * 64}],
            "image_digest": spec.image,
            "source_refs": training.params["native_train_plan"]["source_refs"],
        },
    )
    monkeypatch.setitem(sys.modules, "eb.pbfm_training_checkpoint", NS(
        read_pbfm_checkpoint=Mock(return_value=checkpoint), MAX_FILES=10000))
    monkeypatch.setattr(collector, "read_pbfm_training_evidence", AsyncMock(return_value={"available": True}))
    monkeypatch.setattr("eb.orchestrator.ufo_native.verify_index", AsyncMock())
    session = NS(scalar=AsyncMock(return_value=None), get=AsyncMock(return_value=None),
                 add=Mock(), flush=AsyncMock())
    job = NS(id="job_diagnostic", spec=spec.model_dump(mode="json"))
    run = NS(id="run_source", metrics_summary={})
    await collector.collect_pbfm_training(session, job, run, tmp_path, diagnostic_spec=training)
    added = [call.args[0] for call in session.add.call_args_list]
    model = next(item for item in added if isinstance(item, ModelVersion))
    assert model.card["diagnostic_mode"] == "segment112"
    assert model.card["diagnostic_job_id"] == job.id and "train_job_id" not in model.card
    assert model.card["qualification"] == "incomplete" and "diagnostic" in model.tags
    assert all(item.meta["diagnostic"] for item in added if isinstance(item, Artifact))
    assert run.metrics_summary["native_operation"] == "probe_pbfm_runtime"
    assert run.metrics_summary["diagnostic_mode"] == "segment112"
    assert run.metrics_summary["training_completed"] is True
    session.add.reset_mock()
    training.name = "forged conversion"
    with pytest.raises(ValueError, match="diagnostic"):
        await collector.collect_pbfm_training(session, job, run, tmp_path, diagnostic_spec=training)
    session.add.assert_not_called()
