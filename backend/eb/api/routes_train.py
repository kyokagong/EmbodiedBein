from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select

from eb.api.deps import PrincipalDep, SessionDep
from eb.domain.job import JobKind, JobSpec
from eb.domain.environment_profile import compatible_profile, profile_snapshot
from eb.domain.rlinf_form import (TEMPLATES, build_overrides, get_template, ray_port_for,
                                  require_template_ready, template_entrypoint, template_image)
from eb.ids import new_id
from eb.native_training import is_native_template, service_for
from eb.orchestrator.service import Orchestrator
from eb.settings import get_settings
from eb.store.models import (Artifact, ModelVersion, EmbodimentRow, EnvironmentProfile, Job, LineageEdge, TrainJob, WeightModel,
                             Dataset, Run)
from eb.workbench import embodiment_root

router = APIRouter()
orch = Orchestrator()

_KIND = {"sft": JobKind.TRAIN_SFT, "ppo": JobKind.TRAIN_RLFT, "eval": JobKind.EVAL_SIM, "collect": JobKind.DATA_COLLECT}


def _memory_gb(template_id: str, gpus: int) -> int:
    if template_id == "rpo_v2_flat_ppo" and gpus > 1:
        return 96
    # Four-card PPO needs room for full-checkpoint gathering, beyond training RSS.
    if template_id == "robotwin_pi05_sft" and gpus == 4:
        return 72
    if template_id.startswith("libero_") and gpus == 4:
        return {"sft": 72, "ppo": 76}.get(get_template(template_id).kind, 64)
    return 64


class NativeTrainingIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target_updates: int = Field(default=112, ge=112, le=1000000, strict=True)
    buffer_size: int = Field(default=262144, ge=32768, le=1048576, strict=True)
    seed: int = Field(default=42, ge=0, le=1000000, strict=True)
    timeout_seconds: int = Field(default=21600, ge=60, le=21600, strict=True)


class PbfmTrainingIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target_updates: int = Field(default=112, ge=112, le=10000, strict=True)
    num_envs_per_rank: Literal[1024, 2048] = 1024
    buffer_size: int = Field(default=262144, ge=32768, le=5120000, strict=True)
    seed: int = Field(default=42, ge=0, le=1000000, strict=True)
    timeout_seconds: int = Field(default=21600, ge=60, le=21600, strict=True)


class TrainIn(BaseModel):
    name: str = Field(min_length=1, max_length=120, pattern=r"^[\w .-]+$")
    model_name: str | None = Field(default=None, min_length=1, max_length=120)
    template_id: str
    environment_id: str | None = None
    embodiment_id: str
    image: str | None = None
    gpus: int = Field(default=2, ge=1, le=4)
    dataset_ids: list[str] = []
    base_model_id: str | None = None
    warm_start_model_id: str | None = None
    max_epochs: int | None = None
    total_num_envs: int | None = None
    eval_num_envs: int | None = Field(default=None, ge=1)
    lr: str | None = None
    value_lr: str | None = None
    save_interval: int | None = Field(default=None, ge=-1)
    val_check_interval: int | None = Field(default=None, ge=-1)
    rollout_epoch: int | None = Field(default=None, ge=1)
    seed: int | None = Field(default=None, ge=0)
    extra_overrides: list[str] = []
    auto_register_latest: bool = False
    gym_id: str | None = None
    max_steps: int | None = Field(default=None, ge=1)
    micro_batch_size: int | None = Field(default=None, ge=1)
    global_batch_size: int | None = Field(default=None, ge=1)
    checkpoint_run_id: str | None = None
    checkpoint_step: int | None = Field(default=None, ge=0)
    native_training: NativeTrainingIn | None = None
    pbfm_training: PbfmTrainingIn | None = None
    preview_hash: str | None = Field(default=None, pattern=r"^sha256:[a-f0-9]{64}$")


class LiberoPipelineIn(TrainIn):
    ppo_max_epochs: int = Field(default=1, ge=1, le=10000, strict=True)


class NativeReconfigureIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    num_envs_per_rank: Literal[2048]
    buffer_size: int = Field(ge=65536, le=1048576, strict=True)
    replay_policy: Literal["fresh"]


class ResumeTrainIn(BaseModel):
    name: str = Field(min_length=1, max_length=120, pattern=r"^[\w .-]+$")
    model_name: str | None = Field(default=None, min_length=1, max_length=120)
    max_steps: int | None = Field(default=None, ge=1)
    max_epochs: int | None = Field(default=None, ge=1)
    checkpoint_step: int | None = Field(default=None, ge=1)
    target_updates: int | None = Field(default=None, ge=112, le=1000000, strict=True)
    timeout_seconds: int | None = Field(default=None, ge=60, le=21600, strict=True)
    preview_hash: str | None = Field(default=None, pattern=r"^sha256:[a-f0-9]{64}$", strict=True)
    native_reconfigure: NativeReconfigureIn | None = None
    native_cuda_allocator: Literal["expandable_segments"] | None = None
    native_experiment_target_updates: int | None = Field(default=None, ge=112, le=1000000, strict=True)
    native_new_budget: bool = Field(default=False, strict=True)


_RESUME_TERMINAL = {"succeeded", "failed", "timeout", "cancelled", "infra_failed"}


def require_resume_terminal(row) -> None:
    if row.status not in _RESUME_TERMINAL:
        raise ValueError("只有终态训练任务可以创建恢复任务")


def require_inactive_source_run(runtime: dict) -> None:
    if runtime.get("status") in {"created", "running", "restarting", "paused"}:
        raise ValueError("源 Run 容器仍在运行，不能创建恢复任务")


def _replace_override(overrides: list[str], key: str, value: int) -> list[str]:
    return [item for item in overrides if item.split("=", 1)[0] != key] + [f"{key}={value}"]


async def _lock_native_train_submission(session, key: str) -> None:
    import hashlib
    from sqlalchemy import func
    lock_id = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big", signed=True)
    await session.execute(select(func.pg_advisory_xact_lock(lock_id)))


def native_resume_preview_hash(spec: JobSpec) -> str:
    from eb.ufo_native import plan_hash
    return plan_hash(spec.model_dump(mode="json"))


def build_resume_spec(source: JobSpec, source_train_job_id: str, checkpoint: dict,
                      body: ResumeTrainIn) -> JobSpec:
    from eb.checkpoints import resume_fingerprint

    if checkpoint.get("compatibility_fingerprint") != resume_fingerprint(source):
        raise ValueError("checkpoint 与源训练配置指纹不一致")
    template = get_template(str(source.params.get("template_id")))
    image = checkpoint.get("image_digest") or source.image
    if template.id == "rpo_pbfm_fb_depth_native":
        return service_for(template.id).build_resume_spec(source, source_train_job_id, checkpoint, body)
    if template.id == "rpo_ufo_fb_native":
        from copy import deepcopy
        from eb.ufo_training import native_arguments, training_fingerprint
        if body.target_updates is None or body.max_epochs is not None or body.max_steps is not None:
            raise ValueError("UFO 恢复需要累计 target_updates，不支持 max_epochs/max_steps")
        if checkpoint.get("mode") != "ufo_native_state":
            raise ValueError("UFO checkpoint 缺少全 rank 原生状态")
        binding = checkpoint.get("native_resume_source")
        if (not isinstance(binding, dict) or binding.get("schema_version") != "eb/ufo-native-resume-source/v1"
                or binding.get("run_id") != checkpoint["run_id"]):
            raise ValueError("Native resume source binding is missing or incompatible")
        params = deepcopy(source.params)
        plan = params["native_train_plan"]
        source_plan = deepcopy(plan)
        if body.native_experiment_target_updates is not None:
            plan["experiment_target_updates"] = body.native_experiment_target_updates
        goal = plan.get("experiment_target_updates")
        if goal is not None and (goal < body.target_updates or goal % 16):
            raise ValueError("Native experiment target must cover the interval and align to 16 updates")
        if body.native_cuda_allocator is not None:
            plan["cuda_allocator"] = body.native_cuda_allocator
        policy = source.policy.model_copy(deep=True)
        if body.native_new_budget and body.timeout_seconds is None:
            raise ValueError("native_new_budget requires explicit timeout_seconds")
        if body.timeout_seconds is not None:
            source_timeout = plan.get("segment_timeout_seconds", 21600)
            if (type(source_timeout) is not int or not body.native_new_budget and (
                    body.timeout_seconds > source_timeout
                    or body.timeout_seconds / 3600 > source.policy.timeout_h)):
                raise ValueError("timeout_seconds 不能扩大源训练的本段墙钟预算")
            if body.native_new_budget:
                plan["budget_renewal"] = {"source_train_job_id": source_train_job_id,
                    "source_timeout_seconds": source_timeout,
                    "source_policy_timeout_h": source.policy.timeout_h,
                    "timeout_seconds": body.timeout_seconds}
            plan["segment_timeout_seconds"] = body.timeout_seconds
            policy.timeout_h = body.timeout_seconds / 3600
        counts = checkpoint["manifest"]["counts"]
        plan.update(target_updates=body.target_updates, start_updates=counts["optimizer_steps"],
                    start_local_time=counts["local_time"],
                    resume={"host_path": checkpoint["native_path"], "checkpoint_path": "/resume-native/checkpoint",
                            "step": checkpoint["step"]})
        mode = "ufo_native_state"
        if body.native_reconfigure is not None:
            from eb.ufo_training_checkpoint import validate_reconfigured_resume
            image = checkpoint.get("reconfigure_image_digest")
            if not isinstance(image, str) or not image.startswith("sha256:"):
                raise ValueError("Native reconfiguration requires a verified runtime image")
            options = body.native_reconfigure
            plan["identity"].update(image_digest=image, num_envs_per_rank=options.num_envs_per_rank,
                                    buffer_size=options.buffer_size)
            plan.update(image_digest=image, fingerprint=training_fingerprint(plan["identity"]))
            plan["resume"].update(replay_policy=options.replay_policy, source_plan=source_plan)
            validate_reconfigured_resume(plan, source_plan)
            mode = "ufo_native_state_fresh_replay"
        plan["native_args"] = native_arguments(plan)
        if body.native_reconfigure is None and training_fingerprint(plan["identity"]) != checkpoint["compatibility_fingerprint"]:
            raise ValueError("UFO 恢复配置改变了兼容性指纹")
        params.update(native_train_plan=plan, resume_mode=mode, resume_step=checkpoint["step"],
                      native_resume_source=deepcopy(binding),
                      source_train_job_id=source_train_job_id,
                      model_name=(body.model_name or body.name).strip())
        return source.model_copy(deep=True, update={"name": body.name, "image": image, "params": params,
            "policy": policy,
            "inputs": source.inputs.model_copy(update={"resume_from": checkpoint["run_id"]})})
    if body.native_reconfigure is not None:
        raise ValueError("native_reconfigure 只支持 UFO 原生恢复")
    if body.native_cuda_allocator is not None:
        raise ValueError("native_cuda_allocator 只支持 UFO 原生恢复")
    if body.native_experiment_target_updates is not None:
        raise ValueError("native_experiment_target_updates 只支持 UFO 原生恢复")
    if body.native_new_budget:
        raise ValueError("native_new_budget 只支持 UFO 原生恢复")
    if body.timeout_seconds is not None:
        raise ValueError("timeout_seconds 只支持 UFO 原生恢复")
    if body.target_updates is not None:
        raise ValueError("target_updates 只支持 UFO 原生恢复")
    if template.kind == "sft":
        if body.max_steps is None or body.max_epochs is not None:
            raise ValueError("SFT 恢复只允许设置 max_steps")
        if body.max_steps <= int(checkpoint["step"]):
            raise ValueError("max_steps 必须大于 checkpoint step")
        overrides = _replace_override(source.overrides, "runner.max_steps", body.max_steps)
        if checkpoint.get("mode") != "exact":
            raise ValueError("SFT checkpoint 缺少精确恢复所需的数据游标状态")
    elif template.kind == "ppo":
        if body.max_epochs is None or body.max_steps is not None:
            raise ValueError("PPO 恢复只允许设置 max_epochs")
        if body.max_epochs <= int(checkpoint["step"]):
            raise ValueError("max_epochs 必须大于 checkpoint step")
        overrides = _replace_override(source.overrides, "runner.max_epochs", body.max_epochs)
        if checkpoint.get("mode") != "trainer_state":
            raise ValueError("PPO checkpoint 恢复模式不兼容")
    else:
        raise ValueError("当前模板不支持 checkpoint 恢复")
    params = dict(source.params)
    params.update({"resume_mode": checkpoint["mode"], "resume_step": checkpoint["step"],
                   "source_train_job_id": source_train_job_id,
                   "model_name": body.model_name.strip() if body.model_name else body.name})
    if not params["model_name"]:
        raise ValueError("模型名称不能为空")
    if checkpoint.get("image_digest"):
        params["resume_source_image"] = source.params.get("resume_source_image") or source.image
    inputs = source.inputs.model_copy(update={"resume_from": checkpoint["run_id"]})
    result = source.model_copy(deep=True, update={"name": body.name, "image": image, "params": params,
                                                  "overrides": overrides, "inputs": inputs})
    if resume_fingerprint(result) != checkpoint["compatibility_fingerprint"]:
        raise ValueError("恢复配置改变了训练兼容性指纹")
    return result


def _job_out(row: TrainJob) -> dict[str, Any]:
    return {
        "id": row.id,
        "run_id": getattr(row, "_run_id", None),
        "kind": next((t.kind for t in TEMPLATES if t.id == row.template_id), "unknown"),
        "name": row.name,
        "model_name": (row.params or {}).get("model_name") or row.name,
        "status": row.status,
        "template_id": row.template_id,
        "image_digest": row.image_digest,
        "embodiment_id": row.embodiment_id,
        "dataset_ids": row.dataset_ids or [],
        "base_model_id": row.base_model_id,
        "gpus_allocated": row.gpus_allocated or [],
        "ray_port": row.ray_port,
        "params": row.params or {},
        "inputs": getattr(row, "_inputs", None),
        "resumed_from": (row.params or {}).get("source_train_job_id"),
        "resume_step": (row.params or {}).get("resume_step"),
        "resume_mode": (row.params or {}).get("resume_mode"),
        "hydra_overrides": row.hydra_overrides or [],
        "run_dir": row.run_dir,
        "auto_register_latest": row.auto_register_latest,
        "metrics_summary": row.metrics_summary or {},
        "error_message": row.error_message,
        "job_id": row.job_id,
        "started_at": row.started_at.isoformat() if row.started_at else None,
        "finished_at": row.finished_at.isoformat() if row.finished_at else None,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }


@router.get("/train/templates")
async def train_templates(principal: PrincipalDep) -> dict:
    from eb.pbfm_workbench import CONTROLLED_TRAINING
    return {
        "items": [
            {
                "id": t.id,
                "label": t.label,
                "kind": t.kind,
                "config_path": t.config_path,
                "config_name": t.config_name,
                "script": template_entrypoint(t),
                "embodiment_id": t.embodiment_id,
                "embodiment_env_id": t.embodiment_env_id,
                "task_id": t.task_id,
                "requires_dataset": t.requires_dataset,
                "requires_base_model": t.requires_base_model,
                "default_overrides": t.default_overrides,
                "default_image": template_image(t),
                "model_id": t.model_id,
                "model_framework": t.model_framework,
                "ready": t.ready,
                "dependency_ready": t.dependency_ready,
                "blocked_reason": t.blocked_reason,
                "native_embodiments": t.native_embodiments,
                **({"controlled_training": CONTROLLED_TRAINING}
                   if t.id == "rpo_pbfm_fb_depth_native" and getattr(principal, "actor_type", None) == "user" else {}),
            }
            for t in TEMPLATES if t.kind in {"sft", "ppo", "native_rl"}
        ]
    }


@router.get("/train")
async def list_train(session: SessionDep, principal: PrincipalDep, status: str | None = None) -> dict:
    q = select(TrainJob).order_by(TrainJob.created_at.desc())
    rows = (await session.scalars(q)).all()
    items = []
    for row in rows:
        job = await session.get(Job, row.job_id) if row.job_id else None
        if job and job.kind not in {"train.sft", "train.rl", "train.rlft"}:
            continue
        if not job:
            try:
                if get_template(row.template_id).kind not in {"sft", "ppo", "native_rl"}:
                    continue
            except KeyError:
                continue
        await _sync_run(session, row)
        items.append(_job_out(row))
    from eb.pbfm_workbench import list_records
    items.extend(await list_records(session))
    items.sort(key=lambda item: item.get("created_at") or "", reverse=True)
    if status:
        items = [i for i in items if i["status"] == status]
    return {"items": items}


async def _build_spec(body: TrainIn, session) -> JobSpec:
    from eb.paths import validate_allowed_path
    from eb.data_validation import require_amp_motion_dataset, require_validated_dataset

    try:
        tpl = get_template(body.template_id)
    except KeyError as e:
        raise ValueError(f"未知模板 {body.template_id}") from e
    if tpl.kind == "native_rl":
        if body.pbfm_training is not None and tpl.id != "rpo_pbfm_fb_depth_native":
            raise ValueError("pbfm_training 只支持 P-BFM 原生模板")
        return await service_for(tpl.id).build_spec(body, session)
    require_template_ready(tpl)
    if body.pbfm_training is not None:
        raise ValueError("pbfm_training 只支持 P-BFM 原生模板")
    if body.native_training is not None:
        raise ValueError("native_training 只支持 UFO 原生模板")
    if tpl.kind in {"sft", "ppo"} and not (body.model_name or "").strip():
        raise ValueError("产出模型名称不能为空")
    emb = await session.get(EmbodimentRow, body.embodiment_id)
    if not emb or body.embodiment_id != tpl.embodiment_id:
        raise ValueError("所选本体与模板接口不一致")
    environment = None
    if body.environment_id:
        environment = await session.get(EnvironmentProfile, body.environment_id)
        if not environment:
            raise ValueError("所选环境不存在")
        compatible_profile(environment, "train", tpl.id, tpl.embodiment_env_id, tpl.task_id)
    if body.auto_register_latest:
        raise ValueError("请在完整检查点发布后显式登记模型")
    if body.gym_id:
        raise ValueError("任务由模板绑定，不允许用 Gym ID 改写")
    if tpl.id in {"rpo_v2_wbc_ppo", "rpo_v2_wbc_stand_walk_ppo", "rpo_v2_flat_ppo"}:
        if body.dataset_ids or body.base_model_id:
            raise ValueError("RPO WBC 是从零开始的在线 PPO，不绑定数据集或基座大模型")
        if body.image and body.image != template_image(tpl):
            raise ValueError("RPO WBC 必须使用包含 Isaac、RPO 环境和 RLinf 补丁的专用镜像")
    if body.warm_start_model_id and tpl.id != "rpo_v2_flat_ppo":
        raise ValueError("已发布 checkpoint warm start 当前仅支持 RPO-Flat")
    if tpl.id == "rpo_v2_amp_stand_walk_ppo" and body.base_model_id:
        raise ValueError("RPO AMP 首期从零训练，不接受基座模型")
    if tpl.kind == "eval" and body.dataset_ids:
        raise ValueError("LIBERO 仿真评估使用 EvalSuite，不绑定示教 Dataset")
    extra = []
    if tpl.kind == "ppo" and body.max_steps is not None and body.max_epochs is None:
        extra.append(f"runner.max_epochs={body.max_steps}")
    for name, key in (("max_epochs", "runner.max_epochs"), ("max_steps", "runner.max_steps"),
                      ("total_num_envs", "env.train.total_num_envs"), ("eval_num_envs", "env.eval.total_num_envs"),
                      ("rollout_epoch", "env.train.rollout_epoch"),
                      ("save_interval", "runner.save_interval"), ("val_check_interval", "runner.val_check_interval"),
                      ("lr", "actor.optim.lr"), ("value_lr", "actor.optim.value_lr"), ("seed", "actor.seed"),
                      ("micro_batch_size", "actor.micro_batch_size"), ("global_batch_size", "actor.global_batch_size")):
        value = getattr(body, name)
        if value is not None:
            extra.append(f"{key}={value}")
    extra.extend(body.extra_overrides)
    build_overrides(tpl.id, body.gpus, body.name, extra)
    if tpl.kind == "sft" and body.total_num_envs is not None:
        raise ValueError("SFT 不使用仿真环境数量")
    datasets = []
    for dsid in body.dataset_ids:
        ds = await session.get(Dataset, dsid)
        if not ds:
            raise ValueError(f"数据集不存在: {dsid}")
        from eb.api.routes_data import reconcile_dataset
        await reconcile_dataset(ds, session)
        if ds.embodiment_env_id != tpl.embodiment_env_id:
            raise ValueError(f"数据集本体环境 {ds.embodiment_env_id or '未关联'} 与模板 {tpl.embodiment_env_id} 不一致")
        if (ds.meta or {}).get("usage", "train") != "train":
            raise ValueError("评估集不能用于训练")
        validation = (require_amp_motion_dataset(ds) if tpl.id == "rpo_v2_amp_stand_walk_ppo"
                      else require_validated_dataset(ds, tpl.embodiment_env_id if tpl.id.startswith(("libero_", "robotwin_")) else None))
        if tpl.id == "robotwin_dm05_sft" and validation.get("scope") != "opendm_robotwin2_view":
            raise ValueError("DM0.5 需要已验证的 OpenDM RoboTwin2 JSONL/三相机 view")
        path = validate_allowed_path(ds.dest_path)
        datasets.append({"id": ds.id, "path": str(path), "validation": ds.summary_stats.get("validation"),
                         "source_uri": ds.source_uri, "purpose": (ds.meta or {}).get("purpose")})
    if tpl.requires_dataset and len(datasets) != 1:
        raise ValueError("此训练模板必须选择一个已验证数据集")
    model_path = None
    if body.base_model_id:
        model = await session.get(WeightModel, body.base_model_id)
        if not model:
            raise ValueError("模型不存在")
        from eb.api.routes_model import reconcile_model
        await reconcile_model(model, session)
        if model.kind in {"downloading", "download_failed"}:
            raise ValueError("模型尚未下载完成")
        if model.embodiment_id and model.embodiment_id != body.embodiment_id:
            raise ValueError("模型本体接口不兼容")
        if tpl.model_framework and model.framework != tpl.model_framework:
            raise ValueError(f"模板需要 {tpl.model_framework} 基座模型")
        model_path = validate_allowed_path(model.path)
        if not model_path.is_dir():
            raise ValueError("选择含权重及归一化资产的模型目录，不能使用单个分片")
        if tpl.id == "robotwin_pi05_sft" and (
                not (model_path / "physical-intelligence/robotwin/norm_stats.json").is_file()
                or not (any(model_path.glob("*.safetensors"))
                        or (model_path / "model_state_dict/full_weights.pt").is_file())):
            raise ValueError("π0.5 RoboTwin 基座缺少模型权重或 physical-intelligence/robotwin/norm_stats.json")
        if tpl.id == "robotwin_dm05_sft" and any(
                not (model_path / name).is_file()
                for name in ("config.json", "model.safetensors", "processor_config.json", "norm_stats.json")):
            raise ValueError("DM0.5 基座缺少权重、processor、config 或 norm_stats.json")
        if tpl.id == "robotwin_lingbot_vla2_sft" and (
                not (model_path / "model.safetensors.index.json").is_file()
                or not any(model_path.glob("model-*.safetensors"))
                or not (model_path / "depth/model.pt").is_file()
                or not (model_path / "dino_video/teacher_step_10000.pth").is_file()):
            raise ValueError("LingBot-VLA-2 基座缺少分片权重、depth 或 DINO teacher 资产")
        if tpl.id == "robotwin_xr1_sft" and not (model_path / "model_states.pt").is_file():
            raise ValueError("Xiaomi-Robotics-1 基座缺少 model_states.pt")
        model_path = str(model_path)
    if tpl.requires_base_model and not model_path:
        raise ValueError("此模板必须选择基座模型")
    params = {"template_id": tpl.id, "config_name": tpl.config_name, "stack": "rlinf",
              "model_name": body.model_name.strip() if body.model_name else body.name}
    if body.warm_start_model_id:
        from eb.rpo_flat import flat_warm_start_receipt
        model = await session.get(ModelVersion, body.warm_start_model_id)
        artifact = await session.get(Artifact, model.artifact_id) if model else None
        if not model or not artifact:
            raise ValueError("warm start 模型或完整产物不存在")
        params["warm_start"] = flat_warm_start_receipt(model, artifact)
        model_path = str(validate_allowed_path(artifact.storage))
    if environment:
        params["environment_profile"] = profile_snapshot(environment)
        if tpl.id == "rpo_v2_wbc_stand_walk_ppo":
            params["rpo_train_settings"] = environment.definition.get("settings", {})
        if tpl.id == "rpo_v2_amp_stand_walk_ppo":
            params["rpo_amp_initial_pose"] = environment.definition.get("settings", {}).get("initial_pose")
            params["rpo_amp_initial_logstd"] = environment.definition.get("settings", {}).get("initial_logstd")
            params["rpo_amp_velocity_tracking_std"] = environment.definition.get("settings", {}).get("velocity_tracking_std")
            params["rpo_amp_velocity_reward_weight"] = environment.definition.get("settings", {}).get("velocity_reward_weight")
            params["rpo_amp_actuator_delay_steps"] = environment.definition.get("settings", {}).get("actuator_delay_steps")
    if tpl.id == "robotwin_gr00t_n17_sft":
        # Verified from the pinned N1.7 checkpoint: backbone is frozen and the
        # action stack exposes 1,620,515,968 trainable parameters.
        params["params_b"] = 1_620_515_968
    elif tpl.id == "robotwin_xr1_sft":
        params["params_b"] = 675_250_688
    if body.checkpoint_run_id is not None or body.checkpoint_step is not None:
        from eb.checkpoints import checkpoint_inference, checkpoint_weights
        if (tpl.kind != "eval" or not tpl.id.startswith("libero_") or not body.checkpoint_run_id
                or body.checkpoint_step is None):
            raise ValueError("checkpoint 只能用于 LIBERO 独立评估，并须指定源 Run 和 step")
        source_run = await session.get(Run, body.checkpoint_run_id)
        source = await session.scalar(select(TrainJob).where(TrainJob.job_id == source_run.job_id)) if source_run else None
        source_job = await session.get(Job, source_run.job_id) if source_run else None
        if (not source or not source_job or source_job.status not in _RESUME_TERMINAL
                or source_job.kind not in {"train.sft", "train.rlft"}
                or source.embodiment_id != body.embodiment_id or source.base_model_id != body.base_model_id
                or not source.template_id.startswith("libero_")):
            raise ValueError("源训练、基座模型或本体与评估 checkpoint 不兼容")
        checkpoint = checkpoint_weights(get_settings().data_root / "runs" / body.checkpoint_run_id,
                                        body.checkpoint_step)
        # Freeze the producer's actual inference settings, not today's eval template defaults.
        resolved = get_settings().data_root / "runs" / body.checkpoint_run_id / "output" / "resolved-config.yaml"
        inference = checkpoint_inference(get_settings().data_root / "runs" / body.checkpoint_run_id,
                                         source.template_id, (getattr(source_job, "spec", None) or {}).get("overrides", []))
        params.update({"eval_checkpoint_run_id": body.checkpoint_run_id,
                       "eval_checkpoint_step": body.checkpoint_step,
                       "eval_checkpoint_bytes": checkpoint["bytes"],
                       "eval_source_kind": get_template(source.template_id).kind,
                       "eval_source_inference": inference,
                       "eval_source_config_verified": bool(resolved.exists()),
                       "eval_source_train_job_id": source.id})
    code_refs = {"stack":{"kind":"content_snapshot", "path":str(get_settings().rlinf_repo)}}
    if tpl.id == "rpo_v2_flat_ppo":
        code_refs["embodiment"] = {"kind":"content_snapshot",
                                   "path":str(get_settings().assets_root / "roboparty/rpo-v2")}
    elif not tpl.id.startswith(("robotwin_", "rpo_v2_wbc_", "rpo_v2_amp_")):
        code_refs["embodiment"] = {"kind":"content_snapshot", "path":str(embodiment_root(emb))}
    kind = JobKind.TRAIN_RL if tpl.id in {"rpo_v2_wbc_ppo", "rpo_v2_wbc_stand_walk_ppo", "rpo_v2_amp_stand_walk_ppo", "rpo_v2_flat_ppo"} else _KIND[tpl.kind]
    return JobSpec(kind=kind, embodiment_env_id=tpl.embodiment_env_id,
        task_id=tpl.task_id, name=body.name, image=body.image or template_image(tpl),
        params=params,
        overrides=extra, inputs={"datasets":datasets, "base_model":model_path},
        resources={"gpus":body.gpus, "cpus":8 * body.gpus if tpl.id == "rpo_v2_flat_ppo" else 8,
                   "memory_gb":_memory_gb(tpl.id, body.gpus), "shm_gb":32},
        code_refs=code_refs)


@router.post("/train/preview")
async def preview_train(body: TrainIn, session: SessionDep, principal: PrincipalDep) -> dict:
    from eb.adapters import get_adapter
    try:
        if get_template(body.template_id).kind not in {"sft", "ppo", "native_rl"}:
            raise ValueError("请在评估或数据菜单创建此类任务")
        spec = await _build_spec(body, session)
        rendered = get_adapter(spec).render(spec, "run_preview")
    except (ValueError, KeyError) as e:
        raise HTTPException(400, str(e)) from e
    return {"template":body.template_id, "model_name": spec.params.get("model_name"),
            "preview_hash": "sha256:" + spec.canonical_hash(),
            "native_training": spec.params.get("native_train_plan"),
            "overrides":build_overrides(body.template_id, body.gpus, body.name, spec.overrides), "argv":rendered.command,
            "mounts":[f"{src} → {dst} {mode}" for src,dst,mode in rendered.mounts],
            "inputs":spec.inputs.model_dump(), "image":rendered.image, "resources":spec.resources.model_dump()}


@router.post("/train")
async def submit_train(body: TrainIn, session: SessionDep, principal: PrincipalDep,
                       idempotency_key: str | None = Header(default=None, min_length=1, max_length=120,
                                                           alias="Idempotency-Key")) -> dict:
    try:
        if get_template(body.template_id).kind not in {"sft", "ppo", "native_rl"}:
            raise ValueError("请在评估或数据菜单创建此类任务")
        spec = await _build_spec(body, session)
        if is_native_template(body.template_id) and body.preview_hash != "sha256:" + spec.canonical_hash():
            raise ValueError("原生训练预览已过期，请重新预览冻结输入")
        from eb.adapters import get_adapter
        rendered = get_adapter(spec).render(spec, "run_preview")
    except (ValueError, KeyError) as e:
        raise HTTPException(400, str(e)) from e
    tpl = get_template(body.template_id)
    tid = new_id("trn")
    port = ray_port_for(tid)
    run_dir = ""  # Set from the actual Run, never a second trn_* directory.
    image = rendered.image
    ov = build_overrides(body.template_id, body.gpus, body.name, spec.overrides)
    idem = None
    if tpl.kind == "native_rl":
        if not isinstance(idempotency_key, str) or not idempotency_key:
            raise HTTPException(400, "原生训练需要 Idempotency-Key")
        prefix = "ufo-train" if tpl.id == "rpo_ufo_fb_native" else f"native-train:{tpl.id}"
        idem = f"{prefix}:{principal.id}:{idempotency_key}"
        await _lock_native_train_submission(session, idem)
        existing = await session.scalar(select(Job).where(Job.idem_key == idem))
        if existing:
            if JobSpec.model_validate(existing.spec).canonical_hash() != spec.canonical_hash():
                raise HTTPException(409, "Idempotency-Key 已用于不同的训练配置")
            existing_train = await session.scalar(select(TrainJob).where(TrainJob.job_id == existing.id))
            if not existing_train:
                raise HTTPException(409, "训练请求正在创建，请稍后查询")
            await _sync_run(session, existing_train)
            return _job_out(existing_train)
    try:
        job = await orch.submit(session, spec, principal.id, idem, commit=False)
    except (ValueError, PermissionError) as e:
        raise HTTPException(400, str(e)) from e
    row = TrainJob(
        id=tid,
        name=body.name,
        status=job.status,
        template_id=tpl.id,
        image_digest=image,
        embodiment_id=body.embodiment_id,
        dataset_ids=body.dataset_ids,
        base_model_id=body.base_model_id,
        gpus_allocated=[],
        ray_port=port,
        params=spec.params,
        hydra_overrides=ov,
        run_dir=run_dir,
        auto_register_latest=body.auto_register_latest,
        job_id=job.id,
    )
    session.add(row)
    if body.warm_start_model_id:
        session.add(LineageEdge(src_type="model_version", src_id=body.warm_start_model_id,
                                dst_type="train_job", dst_id=tid, relation="warm_started_from"))
    session.add(
        LineageEdge(
            src_type="embodiment",
            src_id=body.embodiment_id,
            dst_type="train_job",
            dst_id=tid,
            relation="used_embodiment",
        )
    )
    for dsid in body.dataset_ids:
        session.add(
            LineageEdge(
                src_type="dataset",
                src_id=dsid,
                dst_type="train_job",
                dst_id=tid,
                relation="used_dataset",
            )
        )
    if spec.params.get("eval_source_train_job_id"):
        session.add(LineageEdge(src_type="train_job", src_id=spec.params["eval_source_train_job_id"],
                                dst_type="train_job", dst_id=tid, relation="evaluated_checkpoint"))
    await session.commit()
    await session.refresh(row)
    return _job_out(row)


async def _sync_run(session, row):
    if not row.job_id:
        return
    job = await session.get(Job, row.job_id)
    if job:
        row.status, row.error_message = job.status, job.error
        row._inputs = (job.spec or {}).get("inputs")
    run = await session.scalar(select(Run).where(Run.job_id == row.job_id).order_by(Run.attempt.desc()).limit(1))
    if run:
        row._run_id = run.id
        row.run_dir = str(get_settings().data_root / "runs" / run.id)
        row.gpus_allocated = run.gpu_ids
        row.started_at, row.finished_at = run.started_at, run.finished_at
        row.metrics_summary = run.metrics_summary


@router.get("/train/{tid}")
async def get_train(tid: str, session: SessionDep, principal: PrincipalDep) -> dict:
    row = await session.get(TrainJob, tid)
    if not row:
        from eb.pbfm_workbench import get_record, list_records
        record = await get_record(session, tid)
        if record:
            record["continuations"] = [item for item in await list_records(session) if item["resumed_from"] == tid]
            return record
        raise HTTPException(404)
    await _sync_run(session, row)
    result = _job_out(row)
    if is_native_template(row.template_id):
        children = (await session.scalars(select(TrainJob).where(
            TrainJob.params["source_train_job_id"].astext == tid,
            TrainJob.template_id == row.template_id,
        ).order_by(TrainJob.created_at.desc()).limit(20))).all()
        result["continuations"] = []
        for child in children:
            await _sync_run(session, child)
            result["continuations"].append(_job_out(child))
    return result


async def _resume_context(tid: str, session, *, checkpoint_step: int | None = None):
    from eb.checkpoints import latest_resume_checkpoint, resume_fingerprint, require_flat_rank_rng

    row = await session.get(TrainJob, tid)
    if not row:
        raise LookupError("训练任务不存在")
    if not row.job_id:
        raise ValueError("训练任务没有冻结 JobSpec")
    job = await session.get(Job, row.job_id)
    if not job:
        raise ValueError("训练任务关联的 Job 不存在")
    row.status = job.status
    require_resume_terminal(row)
    source = JobSpec.model_validate({**job.spec, "kind": job.kind,
                                    "embodiment_env_id": job.embodiment_env_id,
                                    "api_version": "eb/v1"})
    template = get_template(str(source.params.get("template_id")))
    if not is_native_template(template.id) and (not template.id.startswith(("libero_", "rpo_v2_wbc_", "rpo_v2_amp_", "rpo_v2_flat_")) or template.kind not in {"sft", "ppo"}):
        raise ValueError("该训练模板尚未验证 checkpoint 恢复")
    runs = (await session.scalars(select(Run).where(Run.job_id == job.id))).all()
    settings = get_settings()
    checkpoint = latest_resume_checkpoint(
        ((run.id, settings.data_root / "runs" / run.id) for run in runs), step=checkpoint_step,
        **({"spec": source} if template.id == "rpo_pbfm_fb_depth_native" else {}),
    )
    require_flat_rank_rng(source, checkpoint)
    source_run = next(run for run in runs if run.id == checkpoint["run_id"])
    require_inactive_source_run(await orch.ebd.get_run(source_run.id))
    try:
        provenance = json.loads(source_run.provenance)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("源 Run 缺少镜像 digest 证据") from exc
    image_digest = provenance.get("image_digest")
    if not isinstance(image_digest, str) or not image_digest.startswith("sha256:"):
        raise ValueError("源 Run 缺少镜像 digest 证据")
    checkpoint["image_digest"] = image_digest
    if checkpoint["compatibility_fingerprint"] != resume_fingerprint(source):
        raise ValueError("checkpoint 与源训练冻结配置不一致")
    if is_native_template(template.id):
        checkpoint["native_resume_source"] = await service_for(template.id).build_resume_source_binding(session, source_run, checkpoint)
    return row, job, source, checkpoint


def _source_limit(source: JobSpec, key: str) -> int:
    template = get_template(str(source.params.get("template_id")))
    values = dict(item.split("=", 1) for item in (*template.default_overrides, *source.overrides))
    return int(values.get(key, 0))


@router.get("/train/{tid}/resume-candidate")
async def get_resume_candidate(tid: str, session: SessionDep, principal: PrincipalDep) -> dict:
    if tid.startswith("job_"):
        from eb.pbfm_workbench import get_record, resume_candidate
        record = await get_record(session, tid)
        if record is None:
            raise HTTPException(404)
        return await resume_candidate(session, record)
    try:
        row, _, source, checkpoint = await _resume_context(tid, session)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except (ValueError, OSError) as exc:
        return {"available": False, "reason": str(exc)}
    template = get_template(str(source.params.get("template_id")))
    if template.kind == "native_rl":
        if ((source.params.get("native_train_plan") or {}).get("full_state_finite_required") is not True
                or checkpoint.get("full_state_finite_verified") is not True):
            return {"available": False, "reason": "历史运行未记录完整模型及优化器有限性验证；请用当前任务镜像新建训练。"}
        termination = "target_updates"
        suggested = next((step for step in (512, 1008, 5008, 10000) if step > checkpoint["step"]),
                         checkpoint["step"] + 16)
    elif template.kind == "sft":
        termination = "max_steps"
        suggested = max(_source_limit(source, "runner.max_steps"), checkpoint["step"] + 100)
    else:
        termination = "max_epochs"
        suggested = checkpoint["step"] + 1
    return {
        "available": True, "source_train_job_id": row.id,
        "source_run_id": checkpoint["run_id"], "step": checkpoint["step"],
        "bytes": checkpoint["bytes"], "mode": checkpoint["mode"],
        "state_coverage": checkpoint["state_coverage"], "termination": termination,
        "suggested_target": suggested,
    }


@router.post("/train/{tid}/resume/preview")
async def preview_resume(tid: str, body: ResumeTrainIn, session: SessionDep,
                         principal: PrincipalDep) -> dict:
    from eb.adapters import get_adapter
    try:
        _, _, source, checkpoint = await _resume_context(tid, session, checkpoint_step=body.checkpoint_step)
        await _resolve_native_reconfigure_image(body, checkpoint)
        spec = build_resume_spec(source, tid, checkpoint, body)
        rendered = get_adapter(spec).render(spec, "run_preview")
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except (ValueError, OSError, KeyError) as exc:
        raise HTTPException(422, str(exc)) from exc
    return {
        "source_run_id": checkpoint["run_id"], "step": checkpoint["step"],
        "mode": spec.params.get("resume_mode", checkpoint["mode"]),
        "state_coverage": {**checkpoint["state_coverage"], **(
            {"replay_per_rank": "reinitialized"} if body.native_reconfigure else {})},
        **({"preview_hash": native_resume_preview_hash(spec),
            "native_resume_source": spec.params["native_resume_source"],
            "timeout_seconds": spec.params["native_train_plan"].get("segment_timeout_seconds", 21600),
            "native_training_plan": spec.params["native_train_plan"]}
           if is_native_template(source.params.get("template_id")) else {}),
        "argv": rendered.command, "mounts": [f"{src} → {dst} {mode}" for src, dst, mode in rendered.mounts],
        "inputs": spec.inputs.model_dump(), "image": rendered.image,
        "resources": spec.resources.model_dump(),
        "policy": spec.policy.model_dump(mode="json"),
    }


async def _resolve_native_reconfigure_image(body: ResumeTrainIn, checkpoint: dict) -> None:
    if body.native_reconfigure is None:
        return
    from eb.store.ebd import EbdClient
    from eb.ufo_training import RECONFIGURE_IMAGE
    image = await EbdClient().image_info(RECONFIGURE_IMAGE)
    if image.get("ok") is not True or not image.get("digest", "").startswith("sha256:"):
        raise ValueError("2048-environment native runtime image is unavailable")
    checkpoint["reconfigure_image_digest"] = image["digest"]


@router.post("/train/{tid}/resume")
async def submit_resume(tid: str, body: ResumeTrainIn, session: SessionDep,
                        principal: PrincipalDep,
                        idempotency_key: str = Header(min_length=1, max_length=120,
                                                      alias="Idempotency-Key")) -> dict:
    from eb.adapters import get_adapter
    try:
        source_row, _, source, checkpoint = await _resume_context(tid, session, checkpoint_step=body.checkpoint_step)
        await _resolve_native_reconfigure_image(body, checkpoint)
        spec = build_resume_spec(source, tid, checkpoint, body)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except (ValueError, OSError, KeyError) as exc:
        raise HTTPException(422, str(exc)) from exc
    if (is_native_template(source.params.get("template_id"))
            and body.preview_hash != native_resume_preview_hash(spec)):
        raise HTTPException(409, "原生恢复预览已过期，请重新预览冻结输入")
    idem = f"train-resume:{principal.id}:{idempotency_key}"
    if is_native_template(source.params.get("template_id")):
        await _lock_native_train_submission(session, idem)
    existing = await session.scalar(select(Job).where(Job.idem_key == idem))
    if existing:
        if JobSpec.model_validate(existing.spec).model_dump(mode="json") != spec.model_dump(mode="json"):
            raise HTTPException(409, "Idempotency-Key 已用于其他恢复请求")
        existing_train = await session.scalar(select(TrainJob).where(TrainJob.job_id == existing.id))
        if not existing_train:
            raise HTTPException(409, "恢复请求正在创建，请稍后查询")
        await _sync_run(session, existing_train)
        return _job_out(existing_train)
    try:
        rendered = get_adapter(spec).render(spec, "run_preview")
    except (ValueError, OSError, KeyError) as exc:
        raise HTTPException(422, str(exc)) from exc
    try:
        job = await orch.submit(session, spec, principal.id, idem_key=idem, commit=False)
    except (ValueError, PermissionError) as exc:
        raise HTTPException(422, str(exc)) from exc
    template = get_template(str(spec.params.get("template_id")))
    train = TrainJob(
        id=new_id("trn"), name=spec.name, status=job.status, template_id=template.id,
        image_digest=rendered.image, embodiment_id=source_row.embodiment_id,
        dataset_ids=list(source_row.dataset_ids or []), base_model_id=source_row.base_model_id,
        gpus_allocated=[], ray_port=ray_port_for(job.id), params=spec.params,
        hydra_overrides=build_overrides(template.id, spec.resources.gpus, spec.name, spec.overrides)
            + ([] if template.kind == "native_rl" else [f"runner.resume_dir=/workspace/resume/global_step_{checkpoint['step']}"]),
        run_dir="", auto_register_latest=False, job_id=job.id,
    )
    train._inputs = spec.inputs.model_dump()
    session.add(train)
    session.add(LineageEdge(src_type="train_job", src_id=tid, dst_type="train_job",
                            dst_id=train.id, relation="derived_from"))
    await session.commit()
    await session.refresh(train)
    return _job_out(train)


@router.get("/train/{tid}/checkpoints")
async def list_checkpoints(tid: str, session: SessionDep, principal: PrincipalDep) -> dict:
    row = await session.get(TrainJob, tid)
    if not row:
        from eb.pbfm_workbench import checkpoints, get_record
        record = await get_record(session, tid)
        if record:
            return {"items": checkpoints(record)}
        raise HTTPException(404)
    await _sync_run(session, row)
    from eb.checkpoints import checkpoint_model, checkpoint_weights
    producer = None
    if getattr(row, "template_id", None) == "rpo_pbfm_fb_depth_native":
        job = await session.get(Job, row.job_id)
        producer = JobSpec.model_validate(job.spec) if job else None
    items = []
    runs = (await session.scalars(select(Run).where(Run.job_id == row.job_id))) if row.job_id else []
    for run in runs:
        root = get_settings().data_root / "runs" / run.id
        if is_native_template(getattr(row, "template_id", None)):
            from eb.checkpoints import resume_checkpoint
            try:
                published = resume_checkpoint(root, run.id, **({"spec": producer} if producer else {}))
                items.append({**published, "name": f"原生 agent 更新 {published['step']}",
                              "complete": True, "weights_only": False})
            except (ValueError, OSError):
                pass
            continue
        for directory in (root / "output").glob("*/checkpoints/global_step_*"):
            suffix = directory.name.removeprefix("global_step_")
            if not suffix.isdigit():
                continue
            try:
                checkpoint = checkpoint_weights(root, int(suffix))
            except (ValueError, OSError):
                continue
            items.append({"path": checkpoint["path"], "name": f"step-{suffix}",
                          "step": int(suffix), "run_id": run.id, "bytes": checkpoint["bytes"],
                          "complete": False, "weights_only": True})
        if row.status == "succeeded":
            try:
                published = checkpoint_model(root, run.id)
                items.append({"path": published["path"], "name": f"已发布 step-{published['step']}",
                              "step": published["step"], "run_id": run.id,
                              "bytes": published["bytes"], "complete": True})
            except (ValueError, OSError):
                pass
    return {"items": sorted(items, key=lambda item: (item["step"], item["complete"]), reverse=True)}


@router.get("/train/{tid}/evidence")
async def native_training_evidence(tid: str, session: SessionDep, principal: PrincipalDep) -> dict:
    """Read existing native analysis without mutating historical Run metadata."""
    row = await session.get(TrainJob, tid)
    if not row or not is_native_template(row.template_id):
        from eb.pbfm_workbench import get_record
        record = await get_record(session, tid) if row is None else None
        if record:
            return record["metrics_summary"].get("training_evidence") or {
                "available": False, "reason": "尚未收集完整训练证据。"}
        raise HTTPException(404, "没有原生训练分析")
    job = await session.get(Job, row.job_id) if row.job_id else None
    run = await session.scalar(select(Run).where(Run.job_id == row.job_id).order_by(Run.attempt.desc()))
    if not job or not run or job.status != "succeeded" or run.status != "finished":
        return {"available": False, "reason": "原生训练段尚未成功完成"}
    from eb.orchestrator.collect import read_native_training_evidence, read_pbfm_training_evidence
    try:
        reader = read_pbfm_training_evidence if row.template_id == "rpo_pbfm_fb_depth_native" else read_native_training_evidence
        return await reader(get_settings().data_root / "runs" / run.id, run.id,
            target_updates=job.spec["params"]["native_train_plan"]["target_updates"])
    except (ValueError, OSError, KeyError, TypeError) as exc:
        raise HTTPException(409, f"原生训练证据不完整：{exc}") from exc



class RegisterCkptIn(BaseModel):
    path: str
    name: str | None = Field(default=None, min_length=1, max_length=120)


@router.post("/train/{tid}/register-model")
async def register_ckpt(tid: str, body: RegisterCkptIn, session: SessionDep, principal: PrincipalDep) -> dict:
    from eb.paths import PathNotAllowed, validate_allowed_path

    if body.name is not None and not body.name.strip():
        raise HTTPException(400, "模型名称不能为空")
    row = await session.get(TrainJob, tid)
    if not row:
        raise HTTPException(404)
    try:
        path = validate_allowed_path(body.path)
    except PathNotAllowed as e:
        raise HTTPException(400, str(e)) from e
    await _sync_run(session, row)
    if row.status != "succeeded" or not row.run_dir:
        raise HTTPException(400, "只能登记该任务成功运行的完整检查点")
    if row.template_id == "rpo_pbfm_fb_depth_native":
        raise HTTPException(400, "P-BFM 模型由完整训练回执自动登记；推理包请使用 P-BFM 导出入口")
    from eb.checkpoints import checkpoint_model
    try:
        checkpoint = checkpoint_model(Path(row.run_dir), row._run_id)
        if path != Path(checkpoint["path"]).resolve():
            raise ValueError("选择的路径不是已发布模型目录")
    except (ValueError, OSError) as exc:
        raise HTTPException(400, str(exc)) from exc
    mid = new_id("mdl")
    from eb.orchestrator.deletion import guard_storage_inputs
    try:
        await guard_storage_inputs(session, {"train_job_id": tid, "path": str(path)})
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    model = WeightModel(
        id=mid,
        name=(body.name.strip() if body.name else None) or (row.params or {}).get("model_name") or path.name,
        kind="trained",
        path=str(path),
        size_bytes=checkpoint["bytes"],
        embodiment_id=row.embodiment_id,
        train_job_id=tid,
        parent_model_id=row.base_model_id,
    )
    session.add(model)
    session.add(
        LineageEdge(
            src_type="train_job",
            src_id=tid,
            dst_type="model",
            dst_id=mid,
            relation="produced_by",
        )
    )
    if row.base_model_id:
        session.add(
            LineageEdge(
                src_type="model",
                src_id=row.base_model_id,
                dst_type="model",
                dst_id=mid,
                relation="trained_from",
            )
        )
    await session.commit()
    return {"id": mid, "name": model.name, "path": model.path}


@router.post("/train/libero-pipeline")
async def create_libero_pipeline(body: LiberoPipelineIn, session: SessionDep, principal: PrincipalDep) -> dict:
    from eb.store.models import Pipeline
    if body.template_id != "libero_pi05_sft":
        raise HTTPException(400, "该管线从 LIBERO π0.5 SFT 开始")
    try:
        sft = await _build_spec(body, session)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    steps = [{"job":sft.model_dump(mode="json")}]
    for index, mode in enumerate(("ppo",), start=1):
        template = get_template(f"libero_spatial_pi05_{mode}")
        job = sft.model_copy(deep=True)
        job.kind = _KIND[mode]
        job.name = f"{body.name}-{mode}"
        job.task_id = template.task_id
        job.params = {"template_id":template.id, "config_name":template.config_name, "stack":"rlinf"}
        job.inputs.datasets = []
        job.overrides = [f"runner.max_epochs={body.ppo_max_epochs}"] if mode == "ppo" else []
        build_overrides(template.id, job.resources.gpus if mode == "ppo" else 1, job.name, job.overrides)
        job.resources.memory_gb = _memory_gb(template.id, job.resources.gpus)
        steps.append({"job":job.model_dump(mode="json"), "bindings":{
            "inputs.base_model":{"from_step":index-1, "output":"model_directory"}}})
    from eb.capacity import estimate_job
    for step in steps:
        candidate = JobSpec.model_validate(step["job"])
        if estimate_job(candidate).verdict == "no_fit":
            raise HTTPException(400, f"管线阶段 {candidate.params['template_id']} 超出所选 GPU 容量，请调整资源后创建")
    pipeline = Pipeline(id=new_id("pipe"), name=body.name, spec={"steps":steps,
        "description":"LIBERO spatial: SFT → online PPO; evaluate a checkpoint from the Evaluation menu"})
    session.add(pipeline)
    await session.commit()
    return {"id":pipeline.id, "name":pipeline.name, "spec":pipeline.spec}
