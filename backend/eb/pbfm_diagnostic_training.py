"""Bounded operator segments reuse native state without inventing a TrainJob."""

from pathlib import Path
from types import SimpleNamespace

from sqlalchemy import select

from eb.domain.job import JobKind, JobSpec
from eb.pbfm_training import TEMPLATE, native_arguments
from eb.settings import get_settings
from eb.store.ebd import EbdClient
from eb.store.models import Artifact, EmbodimentEnvRow, EvalReport, Job, ModelVersion, Run

STAGES = {112: (0, 0), 368: (112, 26624), 512: (368, 61440)}
ENDPOINTS = {112: 26624, 368: 61440, 512: 81920}
EXPLORATION_STAGES = {10000: (512, 81920)}
LONG_STAGES = {100000: (10000, 1298432)}
RESUME_PARAMS = {
    "native_resume_source",
    "resume_mode",
    "resume_step",
    "source_diagnostic_job_id",
    "validation_report_id",
    "validation_report_binding",
}


def stages_for_mode(mode):
    if mode == "training_long_exploration":
        return LONG_STAGES
    return EXPLORATION_STAGES if mode == "training_exploration" else STAGES


async def long_image():
    from eb.pbfm_longrun import LONG_IMAGE

    info = await EbdClient().image_info(LONG_IMAGE)
    if info.get("ok") is not True or not info.get("digest"):
        raise ValueError("P-BFM derived long-run image is unavailable")
    return info["digest"]


def validate_segment(spec, plan):
    if spec.params.get("probe_mode") == "training_long_recovery":
        from eb.pbfm_recovery import SOURCE_JOB, SOURCE_RUN, validate_recovery_plan

        validate_recovery_plan(plan)
        if (spec.inputs.resume_from != SOURCE_RUN
                or spec.params.get("source_diagnostic_job_id") != SOURCE_JOB
                or spec.params.get("resume_step") != 19984):
            raise ValueError("P-BFM recovery provenance differs")
        return
    target, identity = plan["target_updates"], plan["identity"]
    stages = stages_for_mode(spec.params.get("probe_mode"))
    if (
        target not in stages
        or plan["world_size"] != 4
        or identity["num_envs_per_rank"] != 2048
        or identity["buffer_size"] != 131072
        or (plan.get("start_updates", 0), plan.get("start_local_time", 0)) != stages[target]
        or bool(spec.inputs.resume_from) != (target != 112)
        or plan["segment_timeout_seconds"] < 60
    ):
        raise ValueError("P-BFM freezes W4/E2048/C131072: validation 112→368→512; exploration 512→10000")
    if target != 112 and (
        not RESUME_PARAMS.issubset(spec.params)
        or not isinstance(spec.params["source_diagnostic_job_id"], str)
        or not spec.params["source_diagnostic_job_id"]
        or not isinstance(spec.params["validation_report_id"], str)
        or not spec.params["validation_report_id"]
        or not isinstance(spec.params["validation_report_binding"], dict)
        or spec.params["validation_report_binding"].get("report_id") != spec.params["validation_report_id"]
        or spec.params.get("resume_mode") != "pbfm_native_state"
        or spec.params.get("resume_step") != stages[target][0]
    ):
        raise ValueError("P-BFM diagnostic continuation lacks its native source and independent report")


async def _source(session, job_id, target):
    from eb.pbfm_runtime_probe import _training_view
    from eb.pbfm_training_checkpoint import read_pbfm_checkpoint
    from eb.pbfm_training_service import build_resume_source_binding

    job = await session.get(Job, job_id)
    if not job or job.status != "succeeded":
        raise ValueError("P-BFM diagnostic resume requires a successful source Job")
    diagnostic = JobSpec.model_validate(job.spec)
    source_mode = "training_exploration" if target == 10000 else "training_validation"
    if target not in {112, 368, 512, 10000} or diagnostic.params.get("probe_mode") != source_mode:
        raise ValueError("P-BFM diagnostic resume source is not a W4/E2048 training segment")
    training = _training_view(diagnostic)
    plan = training.params["native_train_plan"]
    if plan["target_updates"] != target:
        raise ValueError("P-BFM diagnostic resume cannot skip or repeat a segment")
    run = await session.scalar(
        select(Run)
        .where(Run.job_id == job.id, Run.status == "finished")
        .order_by(Run.created_at.desc())
        .limit(1)
    )
    summary = (run.metrics_summary or {}) if run else {}
    if (
        not run
        or run.job_id != job.id
        or run.status != "finished"
        or summary.get("diagnostic") is not True
        or summary.get("diagnostic_mode") != source_mode
        or summary.get("training_completed") is not True
        or summary.get("full_state_finite_verified") is not True
        or summary.get("optimizer_steps") != target
        or not summary.get("model_version_id")
    ):
        raise ValueError("P-BFM diagnostic source lacks a complete published training Run")
    runtime = await EbdClient().get_run(run.id)
    image = await EbdClient().image_info(training.image)
    if runtime.get("status") in {"created", "running", "restarting", "paused"}:
        raise ValueError("P-BFM diagnostic source Run is still active")
    if image.get("ok") is not True or image.get("digest") != training.image:
        raise ValueError("P-BFM diagnostic producer image is unavailable or changed")
    checkpoint = read_pbfm_checkpoint(get_settings().data_root / "runs" / run.id, training, run.id)
    if (
        not checkpoint
        or checkpoint.get("mode") != "pbfm_native_state"
        or checkpoint.get("full_state_finite_verified") is not True
        or checkpoint.get("counts", {}).get("optimizer_steps") != target
        or checkpoint.get("counts", {}).get("local_time") != {**ENDPOINTS, 10000: 1298432}[target]
        or checkpoint.get("compatibility_fingerprint") != plan["fingerprint"]
        or checkpoint.get("image_digest") != training.image
    ):
        raise ValueError("P-BFM source checkpoint is incomplete or differs from its frozen segment")
    checkpoint = {
        **checkpoint,
        "native_resume_source": await build_resume_source_binding(session, run, checkpoint),
    }
    return job, run, training, checkpoint


def _resume_body(name, target, timeout):
    return SimpleNamespace(
        name=name,
        model_name=name,
        target_updates=target,
        timeout_seconds=timeout,
        max_steps=None,
        max_epochs=None,
        native_reconfigure=None,
        native_cuda_allocator=None,
        native_experiment_target_updates=None,
        native_new_budget=False,
    )


def _derived(source, job_id, checkpoint, body, report_id, binding, *,
             mode="training_validation", long_run_image=None):
    from eb.pbfm_training_service import build_resume_spec

    training = build_resume_spec(source, job_id, checkpoint, body, long_run_image=long_run_image)
    params = {**training.params}
    params.pop("template_id")
    params.pop("source_train_job_id")
    params.update(
        operation="probe_pbfm_runtime",
        probe_recipe_id=TEMPLATE,
        probe_mode=mode,
        source_diagnostic_job_id=job_id,
        validation_report_id=report_id,
        validation_report_binding=binding,
    )
    return training.model_copy(update={"kind": JobKind.ASSET_VALIDATE, "params": params})


async def build_resume_diagnostic(body, session):
    from eb.pbfm_runtime_probe import validate_submission

    stages = stages_for_mode(body.mode)
    job, run, source, checkpoint = await _source(session, body.resume_job_id, stages[body.target_updates][0])
    identity = source.params["native_train_plan"]["identity"]
    if (
        body.dataset_id != identity["dataset_id"]
        or body.environment_id != identity["environment_profile"]["id"]
    ):
        raise ValueError("P-BFM continuation cannot change its frozen Dataset or training Profile")
    binding = await validation_binding(session, job, run, source, checkpoint, body.validation_report_id)
    spec = _derived(
        source,
        job.id,
        checkpoint,
        _resume_body(body.name, body.target_updates, body.timeout_seconds),
        body.validation_report_id,
        binding,
        mode=body.mode,
        long_run_image=await long_image() if body.mode == "training_long_exploration" else None,
    )
    await validate_submission(session, spec)
    return spec


async def validate_diagnostic_submission(session, spec, training):
    from eb.pbfm_training_service import _validate_submission

    interface = await session.get(EmbodimentEnvRow, spec.embodiment_env_id)
    if interface is None or interface.status != "active":
        raise ValueError("P-BFM training_validation requires the published active embodiment interface")
    if spec.params.get("probe_mode") == "training_long_recovery":
        from eb.pbfm_recovery import IMAGE, SOURCE_JOB, build_recovery_spec
        from eb.pbfm_training_service import _validate_current_runtime, _validate_frozen_inputs

        identity = training.params["native_train_plan"]["identity"]
        body = SimpleNamespace(name=spec.name, resume_job_id=SOURCE_JOB, recovery_update=19984,
            dataset_id=identity["dataset_id"], environment_id=identity["environment_profile"]["id"],
            timeout_seconds=training.params["native_train_plan"]["segment_timeout_seconds"])
        expected = await build_recovery_spec(body, session)
        if expected != spec:
            raise ValueError("P-BFM recovery source, checkpoint, image or frozen specification changed")
        await _validate_frozen_inputs(session, training, diagnostic=True)
        await _validate_current_runtime(training, IMAGE)
        return
    plan = training.params["native_train_plan"]
    stages = stages_for_mode(spec.params.get("probe_mode"))
    if spec.inputs.resume_from:
        job, run, source, checkpoint = await _source(
            session, spec.params["source_diagnostic_job_id"], stages[plan["target_updates"]][0]
        )
        binding = await validation_binding(
            session, job, run, source, checkpoint, spec.params["validation_report_id"]
        )
        expected = _derived(
            source,
            job.id,
            checkpoint,
            _resume_body(spec.name, plan["target_updates"], plan["segment_timeout_seconds"]),
            spec.params["validation_report_id"],
            binding,
            mode=spec.params["probe_mode"],
            long_run_image=(await long_image()
                            if spec.params["probe_mode"] == "training_long_exploration" else None),
        )
        if spec != expected:
            raise ValueError("P-BFM diagnostic source/image/data/resources/checkpoint/report binding changed")
        if spec.params["probe_mode"] == "training_long_exploration":
            from eb.pbfm_longrun import LONG_IMAGE
            from eb.pbfm_training_service import _validate_current_runtime, _validate_frozen_inputs

            await _validate_frozen_inputs(session, training, diagnostic=True)
            await _validate_current_runtime(training, LONG_IMAGE)
            return
        # All original resume inputs were checked above; reuse only shared frozen-input checks.
        training = training.model_copy(deep=True)
        training.inputs.resume_from = None
        cold = training.params["native_train_plan"]
        cold.pop("resume")
        cold.update(target_updates=112, start_updates=0, start_local_time=0)
        cold["native_args"] = native_arguments(cold)
        for key in ("native_resume_source", "resume_mode", "resume_step"):
            training.params.pop(key)
    await _validate_submission(session, training, diagnostic=True)


async def validation_binding(session, job, run, source, checkpoint, report_id):
    """Recheck complete independent media, including zero-success early validation."""
    from eb.pbfm_data import EXPORT_OPERATION
    from eb.pbfm_evaluation import (
        EVAL_OPERATION,
        SUITE_ID,
        validate_action_plan,
        validate_case_rows,
        validate_eval_plan,
        verify_output_index,
    )
    from eb.pbfm_io import fingerprint, read_json, safe_file, sha256

    model = await session.get(ModelVersion, run.metrics_summary["model_version_id"])
    network = await session.get(Artifact, model.artifact_id) if model else None
    card = (model.card or {}) if model else {}
    if (
        not model
        or not network
        or network.kind != "pbfm_native_trained_model"
        or network.run_id != run.id
        or network.storage != checkpoint["native_path"] + "/checkpoint/model"
        or card.get("diagnostic_job_id") != job.id
        or card.get("run_id") != run.id
        or card.get("diagnostic_mode") != (
            "training_exploration"
            if source.params["native_train_plan"]["target_updates"] == 10000 else "training_validation"
        )
        or card.get("train_plan") != source.params["native_train_plan"]
        or card.get("full_state_finite_verified") is not True
        or network.sha256 != "sha256:" + fingerprint((network.meta or {}).get("manifest") or {})
    ):
        raise ValueError("P-BFM validation source model differs from its diagnostic producer")
    report = await session.get(EvalReport, report_id)
    eval_run = await session.get(Run, report.run_id) if report else None
    eval_job = await session.get(Job, eval_run.job_id) if eval_run else None
    gates = (report.gate_results or {}) if report else {}
    if (
        not report
        or report.suite_id != SUITE_ID
        or report.model_version_id != model.id
        or not eval_run
        or eval_run.status != "finished"
        or not eval_job
        or eval_job.status != "succeeded"
        or eval_job.spec.get("kind") != "eval.sim"
        or gates.get("validity") != "complete"
        or gates.get("partial_files")
    ):
        raise ValueError("P-BFM continuation requires its source model's complete independent validation")
    plan = validate_eval_plan(eval_job.spec.get("params", {}).get("pbfm_plan") or {})
    identity = source.params["native_train_plan"]["identity"]
    refs = {
        name: {
            "kind": "content_snapshot",
            "path": source.params[name + "_source_ref"]["path"],
            "snapshot_hash": identity[name + "_snapshot"],
            "manifest_path": str(Path(source.params[name + "_source_ref"]["path"]).parent / "manifest.json"),
        }
        for name in ("pbfm", "asset")
    }
    if (
        plan["model_version_id"] != model.id
        or plan["model_hash"] != network.sha256.removeprefix("sha256:")
        or plan["dataset_manifest"] != identity["dataset_manifest"]
        or plan["image_digest"] != source.image
        or eval_job.spec.get("image") != source.image
        or plan["source_refs"] != refs
        or plan["camera_contract"] != identity["camera_contract"]
        or gates.get("plan_hash") != plan["hash"]
    ):
        raise ValueError("P-BFM independent report is bound to another model/image/source/data")
    exported = await session.get(Artifact, plan.get("export_artifact_id"))
    export_run = await session.get(Run, exported.run_id) if exported else None
    export_job = await session.get(Job, export_run.job_id) if export_run else None
    export_plan = (exported.meta or {}).get("plan") if exported else None
    if (
        not exported
        or exported.kind != "pbfm_policy"
        or exported.sha256 != plan["export_hash"]
        or not export_run
        or export_run.status != "finished"
        or not export_job
        or export_job.status != "succeeded"
        or not export_plan
        or validate_action_plan(export_plan) != export_plan
        or export_plan.get("operation") != EXPORT_OPERATION
        or export_plan.get("model_version_id") != model.id
        or export_plan.get("model_hash") != plan["model_hash"]
        or export_plan.get("image_digest") != source.image
        or export_plan.get("source_refs") != refs
        or export_job.spec.get("params", {}).get("pbfm_plan") != export_plan
        or export_job.spec.get("image") != source.image
        or plan.get("export_path") != exported.storage
    ):
        raise ValueError("P-BFM independent validation lacks the source model's frozen actual export")
    root = get_settings().data_root / "runs" / eval_run.id / "outputs"
    receipt_path = safe_file(root, "eval-receipt.json")
    receipt = read_json(receipt_path)
    rows = read_json(safe_file(root, "cases.json")).get("cases") or []
    metrics = validate_case_rows(plan, rows)
    if (
        receipt != gates.get("receipt")
        or receipt.get("run_id") != eval_run.id
        or receipt.get("schema_version") != "eb/pbfm-eval-receipt/v1"
        or receipt.get("operation") != EVAL_OPERATION
        or receipt.get("plan_hash") != plan["hash"]
        or receipt.get("metrics") != metrics
        or report.metrics != metrics
        or read_json(safe_file(root, "metrics.json")) != metrics
    ):
        raise ValueError("P-BFM validation report differs from complete finite cases and receipt")
    required = {"cases.json", "cases.jsonl", "metrics.json", "media-index.json", "analysis/terrain.svg"}
    required.update(row[key] for row in rows for key in ("video", "depth", "depth_video", "trace"))
    await verify_output_index(root, receipt.get("files") or [], required)
    return {
        "schema_version": "eb/pbfm-diagnostic-validation-source/v1",
        "report_id": report.id,
        "producer_job_id": job.id,
        "producer_run_id": run.id,
        "model_version_id": model.id,
        "model_artifact_id": network.id,
        "model_sha256": network.sha256,
        "validation_run_id": eval_run.id,
        "plan_hash": plan["hash"],
        "receipt_sha256": "sha256:" + sha256(receipt_path),
        "report_fingerprint": "sha256:" + fingerprint({"metrics": report.metrics, "gates": gates}),
        "export_artifact_id": exported.id,
        "export_sha256": exported.sha256,
    }
