"""Read-only training views; original diagnostic Jobs and model versions stay immutable."""
from sqlalchemy import select

from eb.domain.job import JobSpec
from eb.pbfm_data import ENV_ID
from eb.pbfm_runtime_probe import OPERATION, TRAINING_MODES, _training_view
from eb.pbfm_training import TEMPLATE
from eb.settings import get_settings
from eb.store.models import EvalReport, Job, Run

CONTROLLED_TRAINING = {
    "mode": "training_validation", "gpus": 4, "num_envs_per_rank": 2048,
    "buffer_size": 131072, "seed": 42, "target_updates": 112,
    "stages": [112, 368, 512], "timeout_seconds": 21600,
}


def is_workbench_job(job) -> bool:
    if not job or getattr(job, "kind", None) != "asset.validate":
        return False
    raw = getattr(job, "spec", None)
    if not isinstance(raw, dict) or (raw.get("params") or {}).get("probe_mode") not in TRAINING_MODES:
        return False
    try:
        _training_view(JobSpec.model_validate(raw))
    except (ValueError, KeyError, TypeError):
        return False
    return True


def record_for(job, run) -> dict:
    spec = JobSpec.model_validate(job.spec)
    training = _training_view(spec)
    if run and run.job_id != job.id:
        raise ValueError("training Run belongs to another Job")
    def timestamp(value):
        return value.isoformat() if value else None
    return {
        "id": job.id, "job_id": job.id, "job_kind": job.kind,
        "record_kind": "pbfm_validation", "kind": "native_rl",
        "training_mode": spec.params["probe_mode"],
        "name": spec.name, "model_name": spec.params["model_name"],
        "status": job.status, "template_id": TEMPLATE, "image_digest": spec.image,
        "embodiment_id": "emb_roboparty_rpo", "dataset_ids": [x["id"] for x in spec.inputs.datasets],
        "base_model_id": None, "params": training.params,
        "inputs": spec.inputs.model_dump(mode="json"),
        "run_id": run.id if run else None, "gpus_requested": spec.resources.gpus,
        "gpus_allocated": (run.gpu_ids or []) if run else [], "ray_port": None,
        "run_dir": str(get_settings().data_root / "runs" / run.id) if run else None,
        "metrics_summary": (run.metrics_summary or {}) if run else {},
        "error_message": job.error, "auto_register_latest": False, "hydra_overrides": spec.overrides,
        "resumed_from": spec.params.get("source_diagnostic_job_id"),
        "resume_step": spec.params.get("resume_step"), "resume_mode": spec.params.get("resume_mode"),
        "created_at": timestamp(job.created_at),
        "started_at": timestamp(run.started_at) if run else None,
        "finished_at": timestamp(run.finished_at) if run else None,
    }


async def list_records(session) -> list[dict]:
    jobs = (await session.scalars(select(Job).where(
        Job.kind == "asset.validate", Job.embodiment_env_id == ENV_ID,
        Job.spec["params"]["operation"].astext == OPERATION,
        Job.spec["params"]["probe_mode"].astext.in_(TRAINING_MODES),
    ).order_by(Job.created_at.desc()))).all()
    jobs = [job for job in jobs if is_workbench_job(job)]
    if not jobs:
        return []
    runs = (await session.scalars(select(Run).where(
        Run.job_id.in_([job.id for job in jobs])
    ).order_by(Run.attempt.desc(), Run.created_at.desc()))).all()
    latest = {}
    for run in runs:
        latest.setdefault(run.job_id, run)
    return [record_for(job, latest.get(job.id)) for job in jobs]


async def get_record(session, job_id: str) -> dict | None:
    job = await session.get(Job, job_id)
    if not is_workbench_job(job):
        return None
    run = await session.scalar(select(Run).where(Run.job_id == job.id).order_by(
        Run.attempt.desc(), Run.created_at.desc()).limit(1))
    return record_for(job, run)


def checkpoints(record: dict) -> list[dict]:
    summary = record["metrics_summary"]
    if record["status"] == "cancelled":
        return [{**item, "run_id": record["run_id"], "name": f"周期原生更新 {item['step']}"}
                for item in summary.get("periodic_checkpoints", [])
                if item.get("complete") is True and item.get("registered_model_id")
                and item.get("full_state_finite_verified") is True]
    checkpoint = summary.get("checkpoint") or {}
    target = record["params"]["native_train_plan"]["target_updates"]
    if (record["status"] != "succeeded" or summary.get("training_completed") is not True
            or summary.get("full_state_finite_verified") is not True
            or not summary.get("model_version_id")
            or summary.get("optimizer_steps") != target or checkpoint.get("step") != target):
        return []
    return [{**checkpoint, "run_id": record["run_id"], "name": f"原生 agent 更新 {target}",
             "complete": True, "weights_only": False,
             "registered_model_id": summary["model_version_id"]}]


async def model_links(session) -> dict:
    return {item["registered_model_id"]: {
        "train_page_id": row["id"], "train_job_name": row["name"],
    } for row in await list_records(session) for item in checkpoints(row)}


async def resume_candidate(session, record: dict) -> dict:
    if record["status"] == "cancelled":
        return {"available": False, "reason": "训练已暂停；已登记周期模型可评估，续训需新的冻结恢复计划。"}
    if not checkpoints(record):
        return {"available": False, "reason": "等待训练成功并完整登记模型与恢复状态。"}
    plan = record["params"]["native_train_plan"]
    if record.get("training_mode") in {"training_exploration", "training_long_exploration",
                                       "training_long_recovery"}:
        return {"available": False,
                "reason": f"本轮单次{plan['target_updates']}更新探索已结束，不自动追加训练。"}
    target = {112: 368, 368: 512}.get(plan["target_updates"])
    if target is None:
        return {"available": False, "reason": "已达到本轮512更新终点；不会自动追加长训。"}
    from eb.pbfm_evaluation import SUITE_ID
    model_id = record["metrics_summary"]["model_version_id"]
    reports = (await session.scalars(select(EvalReport).join(
        Run, Run.id == EvalReport.run_id).join(Job, Job.id == Run.job_id).where(
        EvalReport.model_version_id == model_id, EvalReport.suite_id == SUITE_ID,
        Run.status == "finished", Job.status == "succeeded",
    ).order_by(EvalReport.created_at.desc()))).all()
    for report in reports:
        metrics, gates = report.metrics or {}, report.gate_results or {}
        if (report.model_version_id != model_id or report.suite_id != SUITE_ID
                or gates.get("validity") != "complete" or gates.get("partial_files")
                or any(metrics.get(key) != 32 for key in ("valid", "planned", "completed"))
                or any(metrics.get(key) != 0 for key in (
                    "invalid", "infra_failed", "cancelled", "not_started"))):
            continue
        return {
            "available": True, "source_run_id": record["run_id"],
            "step": plan["target_updates"], "mode": "pbfm_native_state",
            "termination": "target_updates", "suggested_target": target,
            "validation_report_id": report.id, "validation_metrics": metrics,
            "state_coverage": {"model_optimizer_replay_counts": "restored",
                               "environment_rng_context": "reinitialized"},
            "request": {"dataset_id": plan["identity"]["dataset_id"],
                        "environment_id": plan["identity"]["environment_profile"]["id"],
                        "mode": "training_validation", "gpus": 4, "num_envs_per_rank": 2048,
                        "target_updates": target, "resume_job_id": record["id"],
                        "validation_report_id": report.id,
                        "timeout_seconds": plan["segment_timeout_seconds"]},
        }
    return {"available": False, "reason": "请先在训练模型页完成策略导出与完整独立评估，再审阅结果决定续训。"}
