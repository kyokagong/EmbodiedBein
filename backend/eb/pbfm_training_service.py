"""Shared operator checks for native training preview and submission."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from pathlib import Path

from eb.domain.environment_profile import compatible_profile, profile_hash, profile_snapshot
from eb.domain.job import JobSpec
from eb.pbfm_data import require_pbfm_dataset
from eb.pbfm_training import (
    CAMERA_CONTRACT,
    ENV_ID,
    IMAGE,
    SCHEMA,
    TASK_ID,
    TEMPLATE,
    TERRAIN_CONTRACT,
    native_arguments,
    training_fingerprint,
    validate_train_job,
)
from eb.settings import get_settings
from eb.store.ebd import EbdClient
from eb.store.models import Dataset, EmbodimentEnvRow, EnvironmentProfile


async def build_resume_source_binding(session, source_run, checkpoint) -> dict:
    """Bind small immutable source receipts and their optional published Artifact."""
    from sqlalchemy import and_, or_, select

    from eb.store.models import Artifact
    from eb.ufo_native import safe_file

    root = get_settings().data_root / "runs" / source_run.id
    native = root / "outputs/native"
    if checkpoint.get("run_id") != source_run.id or checkpoint.get("native_path") != str(native):
        raise ValueError("Native resume source binding does not reference the owned source Run")
    values, hashes = {}, {}
    for name, relative in (
        ("manifest", "outputs/native/checkpoint-manifest.json"),
        ("receipt", "outputs/receipt.json"),
    ):
        path = safe_file(root, relative)
        with path.open("rb") as stream:
            raw = stream.read(4 * 1024**2 + 1)
        if not 0 < len(raw) <= 4 * 1024**2:
            raise ValueError("Native resume source binding JSON is missing or unbounded")
        value = json.loads(raw)
        if (
            not isinstance(value, dict)
            or value != checkpoint.get(name)
            or value.get("run_id") != source_run.id
        ):
            raise ValueError("Native resume source binding changed while reading the checkpoint")
        values[name] = value
        hashes[name] = hashlib.sha256(raw).hexdigest()
    if values["receipt"].get("native_manifest_sha256") != hashes["manifest"]:
        raise ValueError("Native resume source binding receipt no longer binds its manifest")
    digest = training_fingerprint(values["manifest"])
    summary = getattr(source_run, "metrics_summary", None) or {}
    if not isinstance(summary, dict):
        raise ValueError("Native resume Artifact association is invalid")
    published_id = summary.get("resume_artifact_id")
    if published_id is not None and not isinstance(published_id, str):
        raise ValueError("Native resume Artifact association is invalid")
    criterion = and_(
        Artifact.kind == "pbfm_native_resume",
        or_(Artifact.run_id == source_run.id, Artifact.sha256 == digest),
    )
    if published_id is not None:
        criterion = or_(criterion, Artifact.id == published_id)
    artifacts = (await session.scalars(select(Artifact).where(criterion).limit(2))).all()
    if len(artifacts) > 1 or published_id is not None and not artifacts:
        raise ValueError("Native resume Artifact association is missing or ambiguous")
    artifact = artifacts[0] if artifacts else None
    if artifact is not None and (
        artifact.kind != "pbfm_native_resume"
        or artifact.run_id != source_run.id
        or artifact.storage != str(native)
        or artifact.sha256 != digest
        or not isinstance(artifact.meta, dict)
        or artifact.meta.get("manifest") != values["manifest"]
        or published_id is not None
        and artifact.id != published_id
    ):
        raise ValueError("Published native resume Artifact differs from its source manifest, storage or Run")
    return {
        "schema_version": "eb/pbfm-native-resume-source/v1",
        "run_id": source_run.id,
        "manifest_sha256": "sha256:" + hashes["manifest"],
        "receipt_sha256": "sha256:" + hashes["receipt"],
        "artifact": {"id": artifact.id, "sha256": artifact.sha256} if artifact else None,
    }


def _require_profile(environment, *, diagnostic: bool = False) -> None:
    if not diagnostic:
        compatible_profile(environment, "train", TEMPLATE, ENV_ID, TASK_ID)
    elif (
        environment.status not in {"draft", "active"}
        or environment.usage != "train"
        or environment.template_id != TEMPLATE
        or environment.embodiment_env_id != ENV_ID
        or environment.task_id != TASK_ID
        or environment.content_hash != profile_hash(environment)
    ):
        raise ValueError("P-BFM probe requires a matching frozen draft or active training Profile")


async def build_spec(body, session) -> JobSpec:
    return await _build_spec(body, session)


async def _build_spec(body, session, *, diagnostic: bool = False) -> JobSpec:
    settings = get_settings()
    if body.embodiment_id != "emb_roboparty_rpo" or not (body.model_name or "").strip():
        raise ValueError("P-BFM 原生训练需要 RPO 本体和产出模型名称")
    if (
        any(
            getattr(body, field) is not None
            for field in (
                "base_model_id",
                "warm_start_model_id",
                "gym_id",
                "max_epochs",
                "max_steps",
                "total_num_envs",
                "eval_num_envs",
                "lr",
                "value_lr",
                "save_interval",
                "val_check_interval",
                "rollout_epoch",
                "seed",
                "micro_batch_size",
                "global_batch_size",
                "checkpoint_run_id",
                "checkpoint_step",
            )
        )
        or body.native_training is not None
        or body.extra_overrides
        or body.auto_register_latest
    ):
        raise ValueError("P-BFM 原生参数必须通过 pbfm_training 设置，不能混入 PPO/SFT 参数")
    if body.image and body.image != IMAGE:
        raise ValueError("P-BFM 原生训练必须使用已验证组合镜像")
    if len(body.dataset_ids) != 1 or not body.environment_id:
        raise ValueError("P-BFM 原生训练需要一个冻结训练数据版本和已发布训练 Profile")
    environment = await session.get(EnvironmentProfile, body.environment_id)
    if environment is None:
        raise ValueError("训练 Profile 不存在")
    _require_profile(environment, diagnostic=diagnostic)
    env = await session.get(EmbodimentEnvRow, ENV_ID)
    if env is None or env.status not in ({"draft", "active"} if diagnostic else {"active"}):
        raise ValueError("P-BFM 本体接口尚未发布")
    dataset = await session.get(Dataset, body.dataset_ids[0])
    if dataset is None:
        raise ValueError("P-BFM 训练数据不存在")
    from eb.api.routes_data import reconcile_dataset

    await reconcile_dataset(dataset, session)
    report = require_pbfm_dataset(dataset)
    options = body.pbfm_training
    if options is None:
        from eb.api.routes_train import PbfmTrainingIn

        options = PbfmTrainingIn()
    if body.gpus not in (1, 2, 4):
        raise ValueError("P-BFM 首期支持 1、2 或 4 个原生训练 rank")
    if options.target_updates % 16:
        raise ValueError("目标必须按原生 16 次更新对齐，例如 112、512、1008、5008")
    client = EbdClient()
    image = await client.image_info(IMAGE)
    if image.get("ok") is not True or not image.get("digest", "").startswith("sha256:"):
        raise ValueError("P-BFM/RLinf 组合镜像尚未构建或不可用")
    snapshots = await asyncio.gather(
        client.snapshot(str(settings.pbfm_root)),
        client.snapshot(str(settings.rlinf_repo)),
        client.snapshot(str(settings.assets_root / "roboparty/rpo-v2")),
    )
    pbfm, rlinf, assets = (
        {"kind": "content_snapshot", "path": s["path"], "commit": s["content_hash"]} for s in snapshots
    )
    identity = {
        "image_digest": image["digest"],
        "world_size": body.gpus,
        "num_envs_per_rank": options.num_envs_per_rank,
        "buffer_size": options.buffer_size,
        "seed": options.seed,
        "dataset_id": dataset.id,
        "dataset_manifest": report["manifest_sha256"],
        "contract_hash": env.spec_hash,
        "environment_profile": profile_snapshot(environment),
        "pbfm_snapshot": pbfm["commit"],
        "rlinf_snapshot": rlinf["commit"],
        "asset_snapshot": assets["commit"],
        "camera_contract": CAMERA_CONTRACT,
        "terrain_contract": TERRAIN_CONTRACT,
    }
    plan = {
        "schema_version": SCHEMA,
        "identity": identity,
        "fingerprint": training_fingerprint(identity),
        "image_digest": image["digest"],
        "world_size": body.gpus,
        "target_updates": options.target_updates,
        "start_updates": 0,
        "start_local_time": 0,
        "full_state_finite_required": True,
        "log_every_steps": 1024 * body.gpus,
        "source_refs": {
            name: {
                "path": f"/sources/{name}",
                "snapshot_hash": ref["commit"],
                "manifest_path": f"/sources/{name}-manifest.json",
            }
            for name, ref in (("pbfm", pbfm), ("rlinf", rlinf))
        },
        "communication_timeout_seconds": 300,
        "segment_timeout_seconds": options.timeout_seconds,
    }
    plan["source_refs"]["asset"] = {
        "path": "/robot-assets",
        "snapshot_hash": assets["commit"],
        "manifest_path": "/sources/asset-manifest.json",
    }
    plan["native_args"] = native_arguments(plan)
    spec = JobSpec(
        kind="train.rl",
        name=body.name,
        image=image["digest"],
        embodiment_env_id=ENV_ID,
        task_id=TASK_ID,
        params={
            "stack": "rlinf",
            "template_id": TEMPLATE,
            "model_name": body.model_name.strip(),
            "native_train_plan": plan,
            "pbfm_source_ref": pbfm,
            "asset_source_ref": assets,
            "environment_profile": profile_snapshot(environment),
        },
        inputs={"datasets": [{"id": dataset.id, "path": dataset.dest_path, "validation": report}]},
        resources={"gpus": body.gpus, "cpus": 8 * body.gpus, "memory_gb": 16 + 16 * body.gpus, "shm_gb": 8},
        policy={"timeout_h": options.timeout_seconds / 3600, "max_retries": 0, "retry_on": []},
        runtime={"network": "none", "hostname": "localhost"},
        code_refs={"stack": rlinf},
    )
    validate_train_job(spec)
    return spec


def build_resume_spec(source: JobSpec, source_train_job_id: str, checkpoint: dict, body, *,
                      long_run_image: str | None = None) -> JobSpec:
    """Continue only the complete native state with the frozen experiment identity."""
    from copy import deepcopy

    if (
        body.target_updates is None
        or body.max_steps is not None
        or body.max_epochs is not None
        or body.native_reconfigure is not None
        or body.native_cuda_allocator is not None
        or body.native_experiment_target_updates is not None
        or body.native_new_budget
    ):
        raise ValueError("P-BFM 恢复只允许累计 target_updates 和不扩大的 timeout_seconds")
    if checkpoint.get("mode") != "pbfm_native_state":
        raise ValueError("P-BFM checkpoint 缺少完整全 rank 原生状态")
    binding = checkpoint.get("native_resume_source")
    if (
        not isinstance(binding, dict)
        or binding.get("schema_version") != "eb/pbfm-native-resume-source/v1"
        or binding.get("run_id") != checkpoint.get("run_id")
    ):
        raise ValueError("P-BFM resume source binding is missing or incompatible")
    params = deepcopy(source.params)
    plan = params["native_train_plan"]
    if (
        training_fingerprint(plan["identity"]) != checkpoint.get("compatibility_fingerprint")
        or checkpoint.get("image_digest", source.image) != source.image
    ):
        raise ValueError("P-BFM 恢复配置与冻结来源指纹或镜像不一致")
    policy = source.policy.model_copy(deep=True)
    if long_run_image is not None:
        from eb.pbfm_longrun import LONG_TIMEOUT_SECONDS, long_exploration_metadata

        if (body.target_updates != 100000 or checkpoint["counts"]["optimizer_steps"] != 10000
                or checkpoint["counts"]["local_time"] != 1298432
                or plan["target_updates"] != 10000 or body.timeout_seconds is None
                or not 60 <= body.timeout_seconds <= LONG_TIMEOUT_SECONDS):
            raise ValueError("P-BFM long exploration requires complete10000 source and bounded budget")
        plan["long_exploration"] = long_exploration_metadata(plan["identity"], long_run_image)
        plan["identity"]["image_digest"] = plan["image_digest"] = long_run_image
        plan["fingerprint"] = training_fingerprint(plan["identity"])
        plan["segment_timeout_seconds"] = body.timeout_seconds
        plan["communication_timeout_seconds"] = 900
        policy.timeout_h = body.timeout_seconds / 3600
    elif body.timeout_seconds is not None:
        if (
            body.timeout_seconds > plan["segment_timeout_seconds"]
            or body.timeout_seconds / 3600 > source.policy.timeout_h
        ):
            raise ValueError("P-BFM 恢复不能扩大源训练的本段墙钟预算")
        plan["segment_timeout_seconds"] = body.timeout_seconds
        policy.timeout_h = body.timeout_seconds / 3600
    counts = checkpoint["counts"]
    plan.update(
        target_updates=body.target_updates,
        start_updates=counts["optimizer_steps"],
        start_local_time=counts["local_time"],
        resume={
            "host_path": checkpoint["native_path"],
            "checkpoint_path": "/resume-native/checkpoint",
            "step": checkpoint["step"],
            "source_run_id": checkpoint["run_id"],
            "source_binding": deepcopy(binding),
        },
    )
    plan["native_args"] = native_arguments(plan)
    params.update(
        native_train_plan=plan,
        native_resume_source=deepcopy(binding),
        source_train_job_id=source_train_job_id,
        resume_mode="pbfm_native_state",
        resume_step=checkpoint["step"],
        model_name=(body.model_name or body.name).strip(),
    )
    result = source.model_copy(
        deep=True,
        update={
            "name": body.name,
            "params": params,
            "policy": policy,
            "image": long_run_image or source.image,
            "inputs": source.inputs.model_copy(update={"resume_from": checkpoint["run_id"]}),
        },
    )
    validate_train_job(result)
    return result


async def validate_submission(session, spec):
    await _validate_submission(session, spec)


async def _validate_frozen_inputs(session, spec, *, diagnostic: bool = False):
    from eb.paths import validate_allowed_path

    plan = validate_train_job(spec)
    if plan.get("full_state_finite_required") is not True:
        raise ValueError(
            "Native training requires complete model and optimizer finite-state verification; "
            "create a new run with the current runtime"
        )
    identity = plan["identity"]
    row = await session.get(Dataset, identity["dataset_id"])
    if row is None:
        raise ValueError("P-BFM 训练数据不存在")
    report = require_pbfm_dataset(row)
    binding = spec.inputs.datasets[0]
    if (
        validate_allowed_path(binding["path"]) != validate_allowed_path(row.dest_path)
        or binding.get("validation") != report
    ):
        raise ValueError("P-BFM 训练数据绑定与登记版本不一致")
    env = await session.get(EmbodimentEnvRow, ENV_ID)
    if (
        env is None
        or env.status not in ({"draft", "active"} if diagnostic else {"active"})
        or env.spec_hash != identity["contract_hash"]
    ):
        raise ValueError("P-BFM 本体接口版本已变更")
    profile = await session.get(EnvironmentProfile, identity["environment_profile"]["id"])
    if profile is None:
        raise ValueError("P-BFM 训练 Profile 不存在")
    _require_profile(profile, diagnostic=diagnostic)
    if profile_snapshot(profile) != identity["environment_profile"]:
        raise ValueError("P-BFM 训练 Profile 已变更")
    # A snapshot may only resolve to its content-addressed owned directory.
    for ref in (
        spec.params["pbfm_source_ref"],
        spec.params["asset_source_ref"],
        spec.code_refs.stack.model_dump(),
    ):
        if ref.get("kind") != "content_snapshot" or not re.fullmatch(
            r"sha256:[a-f0-9]{64}", ref.get("commit", "")
        ):
            raise ValueError("P-BFM source reference is not a content digest")
        expected = get_settings().data_root / "snapshots" / ref["commit"].removeprefix("sha256:") / "code"
        if (
            Path(ref["path"]) != expected
            or not expected.is_dir()
            or any(p.is_symlink() for p in (expected, *expected.parents))
        ):
            raise ValueError("P-BFM source is not the owned frozen snapshot")
    return plan


async def _validate_submission(session, spec, *, diagnostic: bool = False):
    from eb.domain.rlinf_form import get_template, require_template_ready

    if not diagnostic:
        require_template_ready(get_template(TEMPLATE))
    plan = await _validate_frozen_inputs(session, spec, diagnostic=diagnostic)
    if spec.inputs.resume_from:
        from eb.checkpoints import resume_checkpoint
        from eb.store.models import Job, Run, TrainJob

        source_run = await session.get(Run, spec.inputs.resume_from)
        source_job = await session.get(Job, source_run.job_id) if source_run else None
        source_train = await session.get(TrainJob, spec.params.get("source_train_job_id"))
        if (
            not source_run
            or not source_job
            or not source_train
            or source_train.job_id != source_job.id
            or source_run.status not in {"finished", "aborted"}
            or source_job.status not in {"succeeded", "failed", "timeout", "cancelled", "infra_failed"}
        ):
            raise ValueError("P-BFM resume must reference an inactive registered source training Run")
        original = JobSpec.model_validate(source_job.spec)
        original_plan = validate_train_job(original)
        if (
            original_plan.get("full_state_finite_required") is not True
            or plan["identity"] != original_plan["identity"]
            or spec.image != original.image
            or spec.code_refs != original.code_refs
            or spec.inputs.datasets != original.inputs.datasets
            or spec.resources != original.resources
            or any(
                spec.params[key] != original.params[key] for key in ("pbfm_source_ref", "asset_source_ref")
            )
        ):
            raise ValueError("P-BFM resume source/image/data/resources differ from the producer Job")
        runtime = await EbdClient().get_run(source_run.id)
        if runtime.get("status") in {"created", "running", "restarting", "paused"}:
            raise ValueError("P-BFM source Run container is still active")

        resume = plan["resume"]
        checkpoint = resume_checkpoint(
            get_settings().data_root / "runs" / spec.inputs.resume_from,
            spec.inputs.resume_from,
            step=resume["step"],
            spec=original,
        )
        if (
            checkpoint["compatibility_fingerprint"] != original_plan["fingerprint"]
            or checkpoint["mode"] != "pbfm_native_state"
            or resume["host_path"] != checkpoint["native_path"]
            or plan.get("start_updates") != checkpoint["counts"]["optimizer_steps"]
            or plan.get("start_local_time") != checkpoint["counts"]["local_time"]
            or resume.get("source_run_id") != source_run.id
        ):
            raise ValueError("P-BFM 原生恢复来源不兼容")
        current_binding = await build_resume_source_binding(session, source_run, checkpoint)
        if (
            spec.params.get("native_resume_source") != current_binding
            or resume.get("source_binding") != current_binding
        ):
            raise ValueError("Native resume source binding changed or is missing; create a new preview")
    else:
        if "native_resume_source" in spec.params:
            raise ValueError("Fresh native training cannot include a resume source binding")
        await _validate_current_runtime(spec, IMAGE)


async def _validate_current_runtime(spec, image_reference):
    client = EbdClient()
    current_image = await client.image_info(image_reference)
    if not current_image.get("ok") or current_image.get("digest") != spec.image:
        raise ValueError("P-BFM native recipe image changed; create a fresh preview")
    settings = get_settings()
    current = await asyncio.gather(
        client.snapshot(str(settings.pbfm_root)), client.snapshot(str(settings.rlinf_repo)),
        client.snapshot(str(settings.assets_root / "roboparty/rpo-v2")),
    )
    frozen = (spec.params["pbfm_source_ref"], spec.code_refs.stack.model_dump(),
              spec.params["asset_source_ref"])
    if any(old["commit"] != new["content_hash"] or old["path"] != new["path"]
           for old, new in zip(frozen, current)):
        raise ValueError("P-BFM native recipe sources changed; create a fresh preview")
