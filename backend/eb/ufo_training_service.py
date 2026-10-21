"""Shared operator checks for native training preview and submission."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from pathlib import Path

from eb.domain.environment_profile import compatible_profile, profile_snapshot
from eb.domain.job import JobSpec
from eb.settings import get_settings
from eb.store.ebd import EbdClient
from eb.store.models import Dataset, EmbodimentEnvRow, EnvironmentProfile
from eb.ufo_native import ENV_ID, TASK_ID
from eb.ufo_training import (
    DATA_OPERATION,
    IMAGE,
    RECONFIGURE_IMAGE,
    SCHEMA,
    TEMPLATE,
    native_arguments,
    require_native_dataset,
    training_fingerprint,
    validate_train_job,
)


async def build_resume_source_binding(session, source_run, checkpoint) -> dict:
    """Bind small immutable source receipts and their optional published Artifact."""
    from sqlalchemy import and_, or_, select

    from eb.store.models import Artifact
    from eb.ufo_native import plan_hash, safe_file

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
    digest = plan_hash(values["manifest"])
    summary = getattr(source_run, "metrics_summary", None) or {}
    if not isinstance(summary, dict):
        raise ValueError("Native resume Artifact association is invalid")
    published_id = summary.get("resume_artifact_id")
    if published_id is not None and not isinstance(published_id, str):
        raise ValueError("Native resume Artifact association is invalid")
    criterion = and_(
        Artifact.kind == "ufo_native_resume", or_(Artifact.run_id == source_run.id, Artifact.sha256 == digest)
    )
    if published_id is not None:
        criterion = or_(criterion, Artifact.id == published_id)
    artifacts = (await session.scalars(select(Artifact).where(criterion).limit(2))).all()
    if len(artifacts) > 1 or published_id is not None and not artifacts:
        raise ValueError("Native resume Artifact association is missing or ambiguous")
    artifact = artifacts[0] if artifacts else None
    if artifact is not None and (
        artifact.kind != "ufo_native_resume"
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
        "schema_version": "eb/ufo-native-resume-source/v1",
        "run_id": source_run.id,
        "manifest_sha256": "sha256:" + hashes["manifest"],
        "receipt_sha256": "sha256:" + hashes["receipt"],
        "artifact": {"id": artifact.id, "sha256": artifact.sha256} if artifact else None,
    }


async def build_spec(body, session) -> JobSpec:
    settings = get_settings()
    if body.embodiment_id != "emb_roboparty_rpo" or not (body.model_name or "").strip():
        raise ValueError("UFO 原生训练需要 RPO 本体和产出模型名称")
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
        or body.extra_overrides
        or body.auto_register_latest
    ):
        raise ValueError("UFO 原生参数必须通过 native_training 设置，不能混入 PPO/SFT 参数")
    if body.image and body.image != IMAGE:
        raise ValueError("UFO 原生训练必须使用已验证组合镜像")
    if len(body.dataset_ids) != 1 or not body.environment_id:
        raise ValueError("UFO 原生训练需要一个冻结训练数据版本和已发布训练 Profile")
    environment = await session.get(EnvironmentProfile, body.environment_id)
    if environment is None:
        raise ValueError("训练 Profile 不存在")
    compatible_profile(environment, "train", TEMPLATE, ENV_ID, TASK_ID)
    env = await session.get(EmbodimentEnvRow, ENV_ID)
    if env is None or env.status != "active":
        raise ValueError("UFO 本体接口尚未发布")
    dataset = await session.get(Dataset, body.dataset_ids[0])
    if dataset is None:
        raise ValueError("UFO 训练数据不存在")
    from eb.api.routes_data import reconcile_dataset

    await reconcile_dataset(dataset, session)
    report = require_native_dataset(dataset)
    options = body.native_training
    if options is None:
        from eb.api.routes_train import NativeTrainingIn

        options = NativeTrainingIn()
    if options.target_updates % 16:
        raise ValueError("目标必须按原生 16 次更新对齐，例如 112、512、1008、5008")
    client = EbdClient()
    image = await client.image_info(IMAGE)
    if image.get("ok") is not True or not image.get("digest", "").startswith("sha256:"):
        raise ValueError("UFO/RLinf 组合镜像尚未构建或不可用")
    snapshots = await asyncio.gather(
        client.snapshot(str(settings.ufo_native_root)),
        client.snapshot(str(settings.rlinf_repo)),
        client.snapshot(str(settings.assets_root / "roboparty/rpo-v2")),
    )
    ufo, rlinf, assets = (
        {"kind": "content_snapshot", "path": s["path"], "commit": s["content_hash"]} for s in snapshots
    )
    identity = {
        "image_digest": image["digest"],
        "world_size": body.gpus,
        "num_envs_per_rank": 1024,
        "buffer_size": options.buffer_size,
        "seed": options.seed,
        "dataset_id": dataset.id,
        "dataset_manifest": report["manifest_sha256"],
        "contract_hash": env.spec_hash,
        "environment_profile": profile_snapshot(environment),
        "ufo_snapshot": ufo["commit"],
        "rlinf_snapshot": rlinf["commit"],
        "asset_snapshot": assets["commit"],
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
            for name, ref in (("ufo", ufo), ("rlinf", rlinf))
        },
        "communication_timeout_seconds": 300,
        "segment_timeout_seconds": options.timeout_seconds,
    }
    plan["native_args"] = native_arguments(plan)
    return JobSpec(
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
            "ufo_source_ref": ufo,
            "asset_source_ref": assets,
            "environment_profile": profile_snapshot(environment),
        },
        inputs={"datasets": [{"id": dataset.id, "path": dataset.dest_path, "validation": report}]},
        resources={"gpus": body.gpus, "cpus": 8 * body.gpus, "memory_gb": 16 + 16 * body.gpus, "shm_gb": 8},
        policy={"timeout_h": options.timeout_seconds / 3600, "max_retries": 0, "retry_on": []},
        runtime={"network": "none", "hostname": "localhost"},
        code_refs={"stack": rlinf},
    )


async def validate_submission(session, spec):
    from eb.paths import validate_allowed_path

    if spec.params.get("operation") == DATA_OPERATION:
        from eb.api.routes_data import ufo_data_spec

        row = await session.get(Dataset, spec.params.get("dataset_id"))
        if row is None:
            raise ValueError("UFO 数据登记不存在")
        expected = await ufo_data_spec(row)
        if spec.model_dump(exclude={"name"}) != expected.model_dump(exclude={"name"}):
            raise ValueError("UFO 数据任务与登记来源或冻结资源不一致")
        return
    plan = validate_train_job(spec)
    if plan.get("full_state_finite_required") is not True:
        raise ValueError(
            "Native training requires complete model and optimizer finite-state verification; "
            "create a new run with the current runtime"
        )
    identity = plan["identity"]
    row = await session.get(Dataset, identity["dataset_id"])
    if row is None:
        raise ValueError("UFO 训练数据不存在")
    report = require_native_dataset(row)
    binding = spec.inputs.datasets[0]
    if (
        validate_allowed_path(binding["path"]) != validate_allowed_path(row.dest_path)
        or binding.get("validation") != report
    ):
        raise ValueError("UFO 训练数据绑定与登记版本不一致")
    env = await session.get(EmbodimentEnvRow, ENV_ID)
    if env is None or env.status != "active" or env.spec_hash != identity["contract_hash"]:
        raise ValueError("UFO 本体接口版本已变更")
    profile = await session.get(EnvironmentProfile, identity["environment_profile"]["id"])
    if profile is None:
        raise ValueError("UFO 训练 Profile 不存在")
    compatible_profile(profile, "train", TEMPLATE, ENV_ID, TASK_ID)
    if profile_snapshot(profile) != identity["environment_profile"]:
        raise ValueError("UFO 训练 Profile 已变更")
    # A snapshot may only resolve to its content-addressed owned directory.
    for ref in (
        spec.params["ufo_source_ref"],
        spec.params["asset_source_ref"],
        spec.code_refs.stack.model_dump(),
    ):
        if ref.get("kind") != "content_snapshot" or not re.fullmatch(
            r"sha256:[a-f0-9]{64}", ref.get("commit", "")
        ):
            raise ValueError("UFO source reference is not a content digest")
        expected = get_settings().data_root / "snapshots" / ref["commit"].removeprefix("sha256:") / "code"
        if (
            Path(ref["path"]) != expected
            or not expected.is_dir()
            or any(p.is_symlink() for p in (expected, *expected.parents))
        ):
            raise ValueError("UFO source is not the owned frozen snapshot")
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
            raise ValueError("UFO resume must reference an inactive registered source training Run")
        original = JobSpec.model_validate(source_job.spec)
        original_plan = validate_train_job(original)
        reconfigured = plan["resume"].get("replay_policy") == "fresh"
        if reconfigured:
            from eb.ufo_training_checkpoint import validate_reconfigured_resume
            validate_reconfigured_resume(plan, original_plan)
            current_image = await EbdClient().image_info(RECONFIGURE_IMAGE)
            if not current_image.get("ok") or current_image.get("digest") != spec.image:
                raise ValueError("Native reconfiguration runtime image changed; create a new preview")
        if (
            original_plan.get("full_state_finite_required") is not True
            or not reconfigured and plan["identity"] != original_plan["identity"]
            or not reconfigured and spec.image != original.image
            or spec.code_refs != original.code_refs
            or spec.inputs.datasets != original.inputs.datasets
            or spec.resources != original.resources
            or any(spec.params[key] != original.params[key] for key in ("ufo_source_ref", "asset_source_ref"))
        ):
            raise ValueError("UFO resume source/image/data/resources differ from the producer Job")
        runtime = await EbdClient().get_run(source_run.id)
        if runtime.get("status") in {"created", "running", "restarting", "paused"}:
            raise ValueError("UFO source Run container is still active")

        resume = plan["resume"]
        checkpoint = resume_checkpoint(
            get_settings().data_root / "runs" / spec.inputs.resume_from,
            spec.inputs.resume_from,
            step=resume["step"],
        )
        if (
            checkpoint["compatibility_fingerprint"] != original_plan["fingerprint"]
            or checkpoint["mode"] != "ufo_native_state"
            or resume["host_path"] != checkpoint["native_path"]
            or plan.get("start_updates") != checkpoint["counts"]["optimizer_steps"]
            or plan.get("start_local_time") != checkpoint["counts"]["local_time"]
        ):
            raise ValueError("UFO 原生恢复来源不兼容")
        current_binding = await build_resume_source_binding(session, source_run, checkpoint)
        if spec.params.get("native_resume_source") != current_binding:
            raise ValueError("Native resume source binding changed or is missing; create a new preview")
    else:
        if "native_resume_source" in spec.params:
            raise ValueError("Fresh native training cannot include a resume source binding")
        client = EbdClient()
        current_image = await client.image_info(IMAGE)
        if not current_image.get("ok") or current_image.get("digest") != spec.image:
            raise ValueError("UFO native recipe image changed; create a fresh preview")
        settings = get_settings()
        current = await asyncio.gather(
            client.snapshot(str(settings.ufo_native_root)),
            client.snapshot(str(settings.rlinf_repo)),
            client.snapshot(str(settings.assets_root / "roboparty/rpo-v2")),
        )
        frozen = (
            spec.params["ufo_source_ref"],
            spec.code_refs.stack.model_dump(),
            spec.params["asset_source_ref"],
        )
        if any(
            old["commit"] != new["content_hash"] or old["path"] != new["path"]
            for old, new in zip(frozen, current)
        ):
            raise ValueError("UFO native recipe sources changed; create a fresh preview")
