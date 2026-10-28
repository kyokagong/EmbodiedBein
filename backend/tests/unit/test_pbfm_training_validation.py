"""Protected W4/E2048 segments keep a diagnostic identity and native state."""

from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock

import pytest
from pydantic import ValidationError
from test_pbfm_segment_diagnostic import segment_spec

from eb.capacity import estimate_job
from eb.domain.job import JobKind
from eb.pbfm_runtime_probe import ProbeIn, _training_view, render
from eb.pbfm_training import native_arguments, training_fingerprint


def validation_spec(root=None):
    spec = segment_spec(root)
    spec.params["probe_mode"] = "training_validation"
    plan = spec.params["native_train_plan"]
    plan["identity"].update(world_size=4, num_envs_per_rank=2048, buffer_size=131072)
    plan.update(world_size=4, log_every_steps=4096, segment_timeout_seconds=21600)
    plan["fingerprint"] = training_fingerprint(plan["identity"])
    plan["native_args"] = native_arguments(plan)
    spec.resources.gpus, spec.resources.cpus, spec.resources.memory_gb = 4, 32, 80
    spec.policy.timeout_h = 6
    return spec


def request(**changes):
    return ProbeIn(
        **{
            "name": "four card",
            "dataset_id": "ds_pbfm",
            "environment_id": "env_train_pbfm",
            "mode": "training_validation",
            "gpus": 4,
            "num_envs_per_rank": 2048,
            "timeout_seconds": 21600,
            **changes,
        }
    )


def test_training_validation_has_frozen_topology_and_stage_inputs():
    assert request().target_updates == 112
    assert (
        request(target_updates=368, resume_job_id="job_112", validation_report_id="ev_112").target_updates
        == 368
    )
    for changes in (
        {"gpus": 1},
        {"num_envs_per_rank": 1024},
        {"failure_rank": 0},
        {"target_updates": 512},
        {"resume_job_id": "job_112"},
        {"validation_report_id": "ev_112"},
        {"target_updates": 368, "resume_job_id": "job_112"},
        {"target_updates": 128},
        {"buffer_size": 65536},
    ):
        with pytest.raises(ValidationError):
            request(**changes)
    for mode in ("runtime", "segment112"):
        with pytest.raises(ValidationError):
            ProbeIn(name="old", dataset_id="ds", environment_id="env", mode=mode, target_updates=368)


def test_training_validation_renders_full_training_without_changing_job_identity(tmp_path, monkeypatch):
    monkeypatch.setattr("eb.settings.get_settings", lambda: NS(data_root=tmp_path))
    spec = validation_spec(tmp_path)
    view = _training_view(spec)
    assert view.kind == "train.rl" and "probe_mode" not in view.params
    assert spec.kind == "asset.validate" and "template_id" not in spec.params
    rendered = render(spec, "run_preview")
    assert "--probe" not in rendered.command
    assert rendered.metrics_glob.endswith("train_log.txt")
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "change",
    [
        lambda s: s.params.update(probe_fail_rank=0),
        lambda s: s.params["native_train_plan"]["identity"].update(buffer_size=65536),
        lambda s: s.params["native_train_plan"].update(target_updates=368),
        lambda s: s.params.update(validation_report_id="ev_without_source"),
    ],
)
def test_training_validation_rejects_unprotected_spec(change):
    spec = validation_spec()
    change(spec)
    with pytest.raises(ValueError):
        _training_view(spec)


def test_training_validation_capacity_reserves_whole_cards_without_claiming_peak():
    estimate = estimate_job(validation_spec())
    assert estimate.verdict == "unknown"
    assert estimate.per_gpu_peak_gb == 23.5
    basis = estimate.basis["inputs"]
    assert basis["reservation_gib_per_gpu"] == 23.5
    assert basis["known_lower_bound_gib"] == pytest.approx(14.9)
    assert basis["upper_bound_gib"] is None and basis["gpu_peak_measured"] is False
    assert estimate_job(validation_spec(), host_vram_gb=8).verdict == "no_fit"


@pytest.mark.asyncio
async def test_training_validation_collector_uses_explicit_training_view(tmp_path, monkeypatch):
    from eb import pbfm_runtime_probe
    from eb.orchestrator import pbfm_training

    spec = validation_spec(tmp_path)
    collect = AsyncMock()
    monkeypatch.setattr(pbfm_training, "collect_pbfm_training", collect)
    job = NS(id="job_diagnostic", spec=spec.model_dump(mode="json"))
    run = NS(id="run_diagnostic")
    await pbfm_runtime_probe.collect(NS(), job, run, tmp_path / "outputs")
    assert job.spec["kind"] == "asset.validate"
    assert collect.await_args.args[1] is job
    assert collect.await_args.kwargs["diagnostic_spec"] == _training_view(spec)


def source_fixture(tmp_path, monkeypatch, *, target=112):
    from eb import pbfm_diagnostic_training as diagnostic
    from eb import pbfm_training_checkpoint, pbfm_training_service
    from eb.store.models import EmbodimentEnvRow, Job, Run

    source = validation_spec(tmp_path)
    if target == 368:
        source = resumed_spec(source, target=368)
    job = NS(id="job_source", status="succeeded", spec=source.model_dump(mode="json"))
    run = NS(
        id="run_source",
        job_id=job.id,
        status="finished",
        metrics_summary={
            "diagnostic": True,
            "diagnostic_mode": "training_validation",
            "training_completed": True,
            "full_state_finite_verified": True,
            "model_version_id": "mv_source",
            "optimizer_steps": target,
        },
    )
    checkpoint = {
        "run_id": run.id,
        "native_path": str(tmp_path / f"runs/{run.id}/outputs/native"),
        "path": str(tmp_path / f"runs/{run.id}/outputs/native/checkpoint"),
        "step": target,
        "mode": "pbfm_native_state",
        "image_digest": source.image,
        "compatibility_fingerprint": source.params["native_train_plan"]["fingerprint"],
        "counts": {"optimizer_steps": target, "local_time": 26624 if target == 112 else 61440},
        "full_state_finite_verified": True,
    }
    source_binding = {"schema_version": "eb/pbfm-native-resume-source/v1", "run_id": run.id}
    values = {
        (Job, job.id): job,
        (Run, run.id): run,
        (EmbodimentEnvRow, source.embodiment_env_id): NS(status="active"),
    }
    session = NS(
        get=AsyncMock(side_effect=lambda cls, key: values.get((cls, key))), scalar=AsyncMock(return_value=run)
    )
    monkeypatch.setattr(pbfm_training_checkpoint, "read_pbfm_checkpoint", Mock(return_value=checkpoint))
    monkeypatch.setattr(
        pbfm_training_service, "build_resume_source_binding", AsyncMock(return_value=source_binding)
    )
    common = AsyncMock()
    monkeypatch.setattr(pbfm_training_service, "_validate_submission", common)
    monkeypatch.setattr(diagnostic, "validation_binding", AsyncMock(return_value={"report_id": "ev_source"}))
    monkeypatch.setattr(diagnostic, "get_settings", lambda: NS(data_root=tmp_path))
    monkeypatch.setattr(
        diagnostic,
        "EbdClient",
        lambda: NS(
            get_run=AsyncMock(return_value={"status": "exited"}),
            image_info=AsyncMock(return_value={"ok": True, "digest": source.image}),
        ),
    )
    return session, source, job, run, checkpoint, common


def resumed_spec(source, *, target=368):
    from eb.pbfm_training_service import build_resume_spec

    checkpoint = {
        "run_id": "run_previous",
        "native_path": "/frozen/runs/run_previous/outputs/native",
        "step": 112,
        "mode": "pbfm_native_state",
        "image_digest": source.image,
        "compatibility_fingerprint": source.params["native_train_plan"]["fingerprint"],
        "native_resume_source": {
            "schema_version": "eb/pbfm-native-resume-source/v1",
            "run_id": "run_previous",
        },
        "counts": {"optimizer_steps": 112, "local_time": 26624},
    }
    body = NS(
        name="continuation",
        model_name="continuation",
        target_updates=target,
        timeout_seconds=21600,
        max_steps=None,
        max_epochs=None,
        native_reconfigure=None,
        native_cuda_allocator=None,
        native_experiment_target_updates=None,
        native_new_budget=False,
    )
    training = build_resume_spec(_training_view(source), "job_previous", checkpoint, body)
    training.kind = JobKind.ASSET_VALIDATE
    training.params.pop("source_train_job_id")
    training.params.pop("template_id")
    training.params.update(
        operation="probe_pbfm_runtime",
        probe_recipe_id="rpo_pbfm_fb_depth_native",
        probe_mode="training_validation",
        source_diagnostic_job_id="job_previous",
        validation_report_id="ev_previous",
        validation_report_binding={"report_id": "ev_previous"},
    )
    return training


@pytest.mark.asyncio
@pytest.mark.parametrize("source_target,target,endpoint", [(112, 368, 245760), (368, 512, 327680)])
async def test_training_validation_resume_uses_complete_native_source_and_report(
    tmp_path,
    monkeypatch,
    source_target,
    target,
    endpoint,
):
    from eb.pbfm_runtime_probe import build_spec, validate_submission

    session, source, job, run, checkpoint, common = source_fixture(
        tmp_path, monkeypatch, target=source_target
    )
    spec = await build_spec(
        request(target_updates=target, resume_job_id=job.id, validation_report_id="ev_source"), session
    )
    assert spec.kind == "asset.validate"
    assert spec.params["source_diagnostic_job_id"] == job.id
    assert "source_train_job_id" not in spec.params
    assert spec.inputs.resume_from == run.id
    plan = _training_view(spec).params["native_train_plan"]
    assert plan["start_updates"] == source_target
    assert plan["identity"] == source.params["native_train_plan"]["identity"]
    assert plan["native_args"][plan["native_args"].index("--num-env-steps") + 1] == str(endpoint)
    await validate_submission(session, spec)
    cold_view = common.await_args.args[1]
    assert cold_view.inputs.resume_from is None
    assert cold_view.params["native_train_plan"]["identity"] == plan["identity"]
    assert cold_view.params["native_train_plan"]["target_updates"] == 112


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutate",
    [
        lambda s, j, r, c: setattr(j, "status", "failed"),
        lambda s, j, r, c: c.update(full_state_finite_verified=False),
        lambda s, j, r, c: c["counts"].update(optimizer_steps=368),
        lambda s, j, r, c: c.update(compatibility_fingerprint="sha256:" + "f" * 64),
        lambda s, j, r, c: c.update(image_digest="sha256:" + "f" * 64),
    ],
)
async def test_training_validation_rejects_failed_or_different_source(tmp_path, monkeypatch, mutate):
    from eb.pbfm_runtime_probe import build_spec

    session, source, job, run, checkpoint, common = source_fixture(tmp_path, monkeypatch)
    mutate(source, job, run, checkpoint)
    with pytest.raises(ValueError):
        await build_spec(
            request(target_updates=368, resume_job_id=job.id, validation_report_id="ev_source"), session
        )
    common.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutate",
    [
        lambda s: s.params["native_resume_source"].update(manifest_sha256="sha256:" + "f" * 64),
        lambda s: s.params["native_train_plan"]["resume"].update(host_path="/arbitrary/checkpoint"),
        lambda s: s.params.update(validation_report_binding={"report_id": "ev_wrong"}),
        lambda s: s.params.update(validation_report_id="ev_wrong"),
        lambda s: s.params["native_train_plan"]["identity"].update(dataset_manifest="f" * 64),
    ],
)
async def test_training_validation_submission_rechecks_all_resume_bindings(tmp_path, monkeypatch, mutate):
    from eb.pbfm_runtime_probe import build_spec, validate_submission

    session, source, job, run, checkpoint, common = source_fixture(tmp_path, monkeypatch)
    spec = await build_spec(
        request(target_updates=368, resume_job_id=job.id, validation_report_id="ev_source"), session
    )
    common.reset_mock()
    mutate(spec)
    with pytest.raises(ValueError):
        await validate_submission(session, spec)
    common.assert_not_awaited()


def report_fixture(tmp_path, monkeypatch):
    from copy import deepcopy

    from test_pbfm_data_evaluation import plan as eval_plan
    from test_pbfm_data_evaluation import valid_rows

    from eb import pbfm_diagnostic_training as diagnostic
    from eb.pbfm_evaluation import SUITE_ID, validate_case_rows
    from eb.pbfm_io import fingerprint, write_json
    from eb.store.models import Artifact, EvalReport, Job, ModelVersion, Run

    source = _training_view(validation_spec(tmp_path))
    identity = source.params["native_train_plan"]["identity"]
    job = NS(id="job_source")
    run = NS(id="run_source", metrics_summary={"model_version_id": "mv_source"})
    native = str(tmp_path / "runs/run_source/outputs/native")
    checkpoint = {"native_path": native}
    model = NS(
        id="mv_source",
        artifact_id="art_model",
        card={
            "diagnostic_job_id": job.id,
            "run_id": run.id,
            "diagnostic_mode": "training_validation",
            "train_plan": source.params["native_train_plan"],
            "full_state_finite_verified": True,
        },
    )
    network = NS(
        id="art_model",
        kind="pbfm_native_trained_model",
        run_id=run.id,
        storage=native + "/checkpoint/model",
        meta={"manifest": {"model": "source"}},
        sha256="sha256:" + fingerprint({"model": "source"}),
    )
    refs = {
        name: {
            "kind": "content_snapshot",
            "path": source.params[name + "_source_ref"]["path"],
            "snapshot_hash": identity[name + "_snapshot"],
            "manifest_path": str(tmp_path / "snapshots" / ("a" * 64) / "manifest.json"),
        }
        for name in ("pbfm", "asset")
    }
    plan = eval_plan()
    plan.update(
        model_version_id=model.id,
        model_hash=network.sha256.removeprefix("sha256:"),
        dataset_manifest=identity["dataset_manifest"],
        image_digest=source.image,
        source_refs=refs,
        camera_contract=identity["camera_contract"],
        export_artifact_id="art_export",
        export_path=str(tmp_path / "runs/run_export/outputs/policy"),
    )
    plan["hash"] = fingerprint({k: v for k, v in plan.items() if k != "hash"})
    export_plan = {
        "schema_version": "eb/pbfm-export-plan/v1",
        "operation": "export_pbfm_policy",
        "model_version_id": model.id,
        "model_hash": plan["model_hash"],
        "dataset_manifest": identity["dataset_manifest"],
        "image_digest": source.image,
        "source_refs": deepcopy(refs),
        "prompt_manifest_sha256": "a" * 64,
        "prompt_file_sha256": "b" * 64,
        "motion_id": 0,
        "model_files": [{"path": "model"}],
    }
    export_plan["hash"] = fingerprint(export_plan)
    exported = NS(
        id="art_export",
        kind="pbfm_policy",
        run_id="run_export",
        sha256=plan["export_hash"],
        storage=plan["export_path"],
        meta={"plan": export_plan},
    )
    export_run = NS(id="run_export", job_id="job_export", status="finished")
    export_job = NS(status="succeeded", spec={"image": source.image, "params": {"pbfm_plan": export_plan}})
    eval_run = NS(id="run_eval", job_id="job_eval", status="finished")
    eval_job = NS(
        status="succeeded", spec={"kind": "eval.sim", "image": source.image, "params": {"pbfm_plan": plan}}
    )
    rows = valid_rows()
    for row in rows:
        row.update(success=False, terminated=True, goal_reached=False)
        row["route"]["success"] = False
    metrics = validate_case_rows(plan, rows)
    assert metrics["quality_passed"] is False and metrics["completed"] == 32
    root = tmp_path / "runs/run_eval/outputs"
    receipt = {
        "schema_version": "eb/pbfm-eval-receipt/v1",
        "run_id": eval_run.id,
        "operation": "evaluate_pbfm_terrain",
        "plan_hash": plan["hash"],
        "metrics": metrics,
        "files": [{"path": "cases.json", "bytes": 1, "sha256": "a" * 64}],
    }
    report = NS(
        id="ev_source",
        suite_id=SUITE_ID,
        model_version_id=model.id,
        run_id=eval_run.id,
        metrics=metrics,
        gate_results={"validity": "complete", "plan_hash": plan["hash"], "receipt": receipt},
    )
    write_json(root / "cases.json", {"cases": rows})
    write_json(root / "metrics.json", metrics)
    write_json(root / "eval-receipt.json", receipt)
    values = {
        (ModelVersion, model.id): model,
        (Artifact, network.id): network,
        (EvalReport, report.id): report,
        (Run, eval_run.id): eval_run,
        (Job, eval_run.job_id): eval_job,
        (Artifact, exported.id): exported,
        (Run, export_run.id): export_run,
        (Job, export_run.job_id): export_job,
    }
    session = NS(get=AsyncMock(side_effect=lambda cls, key: values.get((cls, key))))
    monkeypatch.setattr(diagnostic, "get_settings", lambda: NS(data_root=tmp_path))
    verify_media = AsyncMock()
    monkeypatch.setattr("eb.pbfm_evaluation.verify_output_index", verify_media)
    return session, job, run, source, checkpoint, report, plan, exported, verify_media


@pytest.mark.asyncio
async def test_complete_zero_success_validation_can_continue_with_model_export_and_media_binding(
    tmp_path, monkeypatch
):
    from eb.pbfm_diagnostic_training import validation_binding

    session, job, run, source, checkpoint, report, plan, exported, verify_media = report_fixture(
        tmp_path, monkeypatch
    )
    binding = await validation_binding(session, job, run, source, checkpoint, report.id)
    assert binding["producer_run_id"] == run.id
    assert binding["model_version_id"] == report.model_version_id
    assert binding["validation_run_id"] == report.run_id
    assert binding["export_artifact_id"] == exported.id
    required = verify_media.await_args.args[2]
    assert len(required) == 5 + 32 * 4


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutate",
    [
        lambda r, p, e: setattr(r, "model_version_id", "mv_wrong"),
        lambda r, p, e: r.gate_results.update(validity="incomplete"),
        lambda r, p, e: p.update(export_artifact_id="art_wrong"),
        lambda r, p, e: setattr(e, "sha256", "f" * 64),
        lambda r, p, e: p.update(image_digest="sha256:" + "f" * 64),
        lambda r, p, e: r.metrics.update(infra_failed=1),
    ],
)
async def test_diagnostic_report_rejects_wrong_model_export_image_or_incomplete_evidence(
    tmp_path, monkeypatch, mutate
):
    from eb.pbfm_diagnostic_training import validation_binding

    session, job, run, source, checkpoint, report, plan, exported, verify_media = report_fixture(
        tmp_path, monkeypatch
    )
    mutate(report, plan, exported)
    with pytest.raises(ValueError):
        await validation_binding(session, job, run, source, checkpoint, report.id)


@pytest.mark.asyncio
async def test_common_orchestrator_submit_rechecks_the_original_diagnostic_resume(tmp_path, monkeypatch):
    from eb.orchestrator import service
    from eb.pbfm_runtime_probe import build_spec
    from eb.store.models import EmbodimentEnvRow

    session, source, producer, run, checkpoint, common = source_fixture(tmp_path, monkeypatch)
    spec = await build_spec(
        request(target_updates=368, resume_job_id=producer.id, validation_report_id="ev_source"), session
    )
    original_get = session.get.side_effect
    session.get.side_effect = lambda cls, key: (
        NS(spec={}, status="active") if cls is EmbodimentEnvRow else original_get(cls, key)
    )
    session.add, session.commit, session.refresh = Mock(), AsyncMock(), AsyncMock()
    monkeypatch.setattr("eb.orchestrator.deletion.guard_storage_inputs", AsyncMock())
    monkeypatch.setattr(service, "validate_flat_warm_start_submission", AsyncMock())
    monkeypatch.setattr(service, "EmbodimentEnv", NS(model_validate=lambda value: NS(embodiment_id="emb")))
    monkeypatch.setattr(service, "g1_submit", lambda *args: NS(passed=True))
    monkeypatch.setattr(service, "validate_mounts", Mock())
    monkeypatch.setattr(
        service, "get_settings", lambda: NS(bhl_repo=tmp_path / "missing", rlinf_repo=tmp_path / "missing")
    )
    monkeypatch.setattr("eb.settings.get_settings", lambda: NS(data_root=tmp_path))
    common.reset_mock()
    submitted = await service.Orchestrator(NS()).submit(session, spec, "operator", None)
    assert submitted.kind == "asset.validate" and submitted.status == "queued"
    assert submitted.spec["params"]["source_diagnostic_job_id"] == producer.id
    assert "source_train_job_id" not in submitted.spec["params"]
    common.assert_awaited_once()
    session.add.assert_called_once_with(submitted)
    spec.params["native_train_plan"]["resume"]["host_path"] = "/arbitrary"
    session.add.reset_mock()
    with pytest.raises(ValueError, match="binding changed"):
        await service.Orchestrator(NS()).submit(session, spec, "operator", None)
    session.add.assert_not_called()


@pytest.mark.asyncio
async def test_training_validation_publication_keeps_diagnostic_job_and_incomplete_artifacts(
    tmp_path, monkeypatch
):
    from test_pbfm_training_service import checkpoint_for

    from eb.orchestrator import pbfm_training as collector
    from eb.store.models import Artifact, ModelVersion

    spec = validation_spec(tmp_path)
    training = _training_view(spec)
    checkpoint = checkpoint_for(training, tmp_path)
    checkpoint.update(
        full_state_finite_verified=True,
        limitations=[],
        state_coverage={},
        manifest={
            "files": [{"path": "checkpoint/model/model.safetensors", "bytes": 10, "sha256": "b" * 64}],
            "image_digest": spec.image,
            "source_refs": training.params["native_train_plan"]["source_refs"],
        },
    )
    monkeypatch.setattr("eb.pbfm_training_checkpoint.read_pbfm_checkpoint", Mock(return_value=checkpoint))
    monkeypatch.setattr(collector, "read_pbfm_training_evidence", AsyncMock(return_value={"available": True}))
    monkeypatch.setattr("eb.orchestrator.ufo_native.verify_index", AsyncMock())
    session = NS(
        scalar=AsyncMock(return_value=None), get=AsyncMock(return_value=None), add=Mock(), flush=AsyncMock()
    )
    job = NS(id="job_diagnostic", spec=spec.model_dump(mode="json"))
    run = NS(id="run_source", metrics_summary={})
    await collector.collect_pbfm_training(session, job, run, tmp_path, diagnostic_spec=training)
    added = [call.args[0] for call in session.add.call_args_list]
    model = next(item for item in added if isinstance(item, ModelVersion))
    assert model.card["diagnostic_job_id"] == job.id and "train_job_id" not in model.card
    assert model.card["diagnostic_mode"] == "training_validation"
    assert model.card["qualification"] == "incomplete"
    for artifact in (item for item in added if isinstance(item, Artifact)):
        assert artifact.meta["diagnostic_job_id"] == job.id
        assert artifact.meta["diagnostic_mode"] == "training_validation"
        assert artifact.meta["qualification"] == "incomplete"
    assert run.metrics_summary["diagnostic_mode"] == "training_validation"
    assert run.metrics_summary["training_evidence"]["available"] is True


@pytest.mark.asyncio
async def test_training_validation_media_uses_analysis_integrity_index(tmp_path, monkeypatch):
    from fastapi import HTTPException

    from eb.api import routes_pbfm, routes_ufo_native
    from eb.pbfm_io import sha256, write_json
    from eb.store.models import Job, Run

    spec = validation_spec(tmp_path)
    root = tmp_path / "runs/run_source/outputs"
    (root / "analysis").mkdir(parents=True)
    path = root / "analysis/train-timeseries.svg"
    path.write_text("<svg/>")
    row = {"path": "analysis/train-timeseries.svg", "bytes": path.stat().st_size, "sha256": sha256(path)}
    write_json(
        root / "analysis/index.json",
        {"schema_version": "eb/pbfm-training-evidence/v1", "run_id": "run_source", "files": [row]},
    )
    run = NS(id="run_source", job_id="job_source", status="finished", metrics_summary={
        "training_evidence": {"available": True, "run_id": "run_source", "files": [row]},
    })
    job = NS(status="succeeded", spec=spec.model_dump(mode="json"))
    session = NS(
        get=AsyncMock(side_effect=lambda cls, key: run if cls is Run else job if cls is Job else None),
        scalar=AsyncMock(return_value=None),
    )
    monkeypatch.setattr(routes_pbfm, "get_settings", lambda: NS(data_root=tmp_path))
    monkeypatch.setattr(
        "eb.store.ebd.EbdClient.scan",
        AsyncMock(return_value={"files": 1, "samples": [{"size": row["bytes"], "sha256": row["sha256"]}]}),
    )
    stream = AsyncMock(return_value="indexed analysis")
    monkeypatch.setattr(routes_ufo_native, "stream_output", stream)
    assert await routes_pbfm.output_file(run.id, row["path"], None, session, None) == "indexed analysis"
    with pytest.raises(HTTPException) as error:
        await routes_pbfm.output_file(run.id, "runtime/rank-0.json", None, session, None)
    assert error.value.status_code == 404


async def admission_fixture(tmp_path, monkeypatch, *, target=368):
    from pathlib import Path

    from eb.adapters.base import RenderedJob
    from eb.orchestrator import service
    from eb.pbfm_runtime_probe import build_spec

    source_target = 112 if target == 368 else 368
    session, source, producer, source_run, checkpoint, common = source_fixture(
        tmp_path, monkeypatch, target=source_target
    )
    spec = await build_spec(
        request(target_updates=target, resume_job_id=producer.id, validation_report_id="ev_source"), session
    )
    job = NS(
        id="job_queued",
        status="queued",
        kind="asset.validate",
        attempts=0,
        version=0,
        embodiment_env_id=spec.embodiment_env_id,
        spec=spec.model_dump(mode="json"),
        error=None,
    )
    session.refresh, session.commit, session.add = AsyncMock(), AsyncMock(), Mock()
    client = NS(submit_run=AsyncMock(return_value={"status": "started", "container_id": "test_container"}))
    stack = Path(spec.code_refs.stack.path)
    stack.mkdir(parents=True)
    rendered = RenderedJob(spec.image, ["python"], {}, [], mounts=[])
    monkeypatch.setattr(service, "get_settings", lambda: NS(bhl_repo=tmp_path / "missing"))
    monkeypatch.setattr(service, "prepare_run_dirs", lambda rid: tmp_path / rid)
    monkeypatch.setattr(service, "prepare_worktree", AsyncMock(return_value=stack))
    monkeypatch.setattr(service, "validate_mounts", Mock())
    monkeypatch.setattr(service, "get_adapter", lambda spec: NS(render=lambda *_: rendered))
    common.reset_mock()
    return session, job, client, producer, checkpoint, common


@pytest.mark.asyncio
@pytest.mark.parametrize("target", [368, 512])
async def test_queued_diagnostic_resume_rechecks_before_admission_and_starts_once(
    tmp_path, monkeypatch, target
):
    from eb.orchestrator.service import Orchestrator

    session, job, client, producer, checkpoint, common = await admission_fixture(
        tmp_path, monkeypatch, target=target
    )
    run = await Orchestrator(client).admit_and_start(session, job, [0, 1, 2, 3])
    common.assert_awaited_once()
    assert job.status == "running" and run.status == "running" and job.attempts == 1
    client.submit_run.assert_awaited_once()
    assert client.submit_run.await_args.args[0]["kind"] == "asset.validate"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure", ["source", "checkpoint", "report", "media", "profile", "interface", "transport"]
)
async def test_queued_diagnostic_resume_blocks_changed_evidence_without_start(tmp_path, monkeypatch, failure):
    import httpx

    from eb import pbfm_diagnostic_training as diagnostic
    from eb.orchestrator.service import Orchestrator

    session, job, client, producer, checkpoint, common = await admission_fixture(tmp_path, monkeypatch)
    if failure == "source":
        producer.status = "failed"
    elif failure == "checkpoint":
        checkpoint["image_digest"] = "sha256:" + "f" * 64
    elif failure in {"report", "media"}:
        diagnostic.validation_binding.side_effect = ValueError(f"changed validation {failure}")
    elif failure == "interface":
        from eb.store.models import EmbodimentEnvRow

        interface = await session.get(EmbodimentEnvRow, job.embodiment_env_id)
        interface.status = "draft"
    elif failure == "transport":
        common.side_effect = httpx.ConnectError("ebd evidence integrity check unavailable")
    else:
        common.side_effect = ValueError("training Profile changed")
    with pytest.raises((ValueError, httpx.HTTPError)):
        await Orchestrator(client).admit_and_start(session, job, [0, 1, 2, 3])
    assert job.status == "blocked" and "verification failed" in job.error
    assert job.attempts == 0 and job.version == 1
    session.add.assert_not_called()
    client.submit_run.assert_not_awaited()
    session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_queued_resume_rehashes_actual_validation_media_before_start(tmp_path, monkeypatch):
    from pathlib import Path

    from eb import pbfm_diagnostic_training as diagnostic
    from eb.orchestrator.service import Orchestrator
    from eb.pbfm_diagnostic_training import validation_binding
    from eb.pbfm_evaluation import verify_output_index
    from eb.pbfm_io import file_index, read_json, sha256, write_json

    session, job, client, producer, checkpoint, common = await admission_fixture(tmp_path, monkeypatch)
    report_session, _, _, source, _, report, _, _, _ = report_fixture(tmp_path, monkeypatch)
    source_get, report_get = session.get.side_effect, report_session.get.side_effect
    session.get.side_effect = lambda cls, key: source_get(cls, key) or report_get(cls, key)
    monkeypatch.setattr(diagnostic, "validation_binding", validation_binding)
    monkeypatch.setattr("eb.pbfm_evaluation.verify_output_index", verify_output_index)
    root = tmp_path / "runs/run_eval/outputs"
    rows = read_json(root / "cases.json")["cases"]
    for row in rows:
        for key in ("video", "depth", "depth_video", "trace"):
            path = root / row[key]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"original-media")
    for name in ("cases.jsonl", "media-index.json", "analysis/terrain.svg"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"indexed evidence")
    receipt_path = root / "eval-receipt.json"
    receipt = read_json(receipt_path)
    receipt_path.unlink()
    receipt["files"] = file_index(root)
    write_json(receipt_path, receipt)
    report.gate_results["receipt"] = receipt

    async def scan(_, name):
        path = Path(name)
        return {"files": 1, "samples": [{"size": path.stat().st_size, "sha256": sha256(path)}]}

    monkeypatch.setattr("eb.store.ebd.EbdClient.scan", scan)
    from eb.store.models import Run

    source_run = await session.get(Run, "run_source")
    binding = await validation_binding(session, producer, source_run, source, checkpoint, report.id)
    job.spec["params"]["validation_report_binding"] = binding
    (root / rows[0]["video"]).write_bytes(b"modified-media")
    with pytest.raises(ValueError, match="integrity differs"):
        await Orchestrator(client).admit_and_start(session, job, [0, 1, 2, 3])
    assert job.status == "blocked" and job.attempts == 0
    session.add.assert_not_called()
    client.submit_run.assert_not_awaited()
