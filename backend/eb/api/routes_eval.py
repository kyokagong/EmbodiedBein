"""Independent LIBERO evaluation entry; Job/Run remain the execution records."""
from __future__ import annotations

import hashlib
import json
from typing import Literal

import httpx
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from eb.api.deps import PrincipalDep, SessionDep
from eb.api.routes_data import LIBERO_REPO, LIBERO_REVISION, reconcile_dataset
from eb.api.routes_train import TrainIn, _build_spec
from eb.domain.environment_profile import compatible_profile, profile_snapshot, validate_definition
from eb.domain.job import JobKind, JobSpec
from eb.domain.robotwin_eval_runtime import (
    derive_robotwin_runtime,
    list_robotwin_runtime_profiles,
    resolve_robotwin_runtime,
)
from eb.domain.rlinf_form import get_template, require_template_ready, template_image
from eb.settings import get_settings
from eb.orchestrator.service import Orchestrator
from eb.store.models import Artifact, Dataset, EnvironmentProfile, EvalSuite, Job, LineageEdge, ModelVersion, Run, TrainJob

router = APIRouter(prefix="/evals")
orch = Orchestrator()
LIBERO_SUITES = {"spatial", "object", "goal", "10"}


class StalePreview(ValueError):
    pass


class EvalIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=120, pattern=r"^[\w .-]+$")
    suite_id: str
    environment_id: str | None = None
    profile: Literal["smoke", "validation", "standard", "acceptance", "official_reproduction"] = "smoke"
    model_version_id: str | None = None
    checkpoint_run_id: str | None = None
    checkpoint_step: int | None = Field(default=None, ge=0)
    gpus: Literal[1, 2, 4] = 1
    parallel_envs: int = Field(default=4, ge=1, le=16)
    runtime_profile_id: str | None = None
    resource_variant_id: str | None = None
    preview_hash: str | None = None


class RpoEvalIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_version_id: str
    backend: Literal["isaac", "mujoco"] = "mujoco"
    suite_id: str | None = None
    environment_id: str | None = None
    profile: Literal["smoke", "standard", "validation", "acceptance"] = "smoke"
    seed: int | None = Field(default=None, ge=0, le=1_000_000)
    record_video: bool = False
    video_quality: Literal["standard", "hd"] = "standard"
    eval_gpus: Literal[1, 4] = 1
    envs_per_gpu: Literal[1, 4] = 1
    preview_hash: str | None = None


async def _bind_environment(environment_id: str | None, suite: EvalSuite, task_id: str,
                            plan: dict, session) -> dict | None:
    if not environment_id:
        return None
    row = await session.get(EnvironmentProfile, environment_id)
    if not row:
        raise ValueError("所选评估环境不存在")
    compatible_profile(row, "eval", suite.id, suite.embodiment_env_id, task_id)
    definition = validate_definition("eval", suite.id, row.definition)
    snapshot = profile_snapshot(row)
    plan["environment_profile"] = snapshot
    if definition["settings"]:
        plan["environment_settings"] = definition["settings"]
        plan["diagnostic_environment"] = any(
            (suite.spec or {}).get(key) != value for key, value in definition["settings"].items())
    plan.pop("hash", None)
    plan["hash"] = "sha256:" + hashlib.sha256(json.dumps(
        plan, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return snapshot


@router.get("/rpo/candidates")
async def rpo_candidates(session: SessionDep, principal: PrincipalDep) -> dict:
    rows = (await session.scalars(select(ModelVersion).order_by(ModelVersion.created_at.desc()))).all()
    return {"items": [{"id": row.id, "name": row.name, "step": (row.card or {}).get("step"),
                       "embodiment_env_id": row.embodiment_env_id,
                       "readiness": (row.card or {}).get("readiness"),
                       "execution": (row.card or {}).get("execution"),
                       "progress_unit": (row.card or {}).get("progress_unit"),
                       "created_at": row.created_at.isoformat() if row.created_at else None}
                      for row in rows if (row.card or {}).get("schema_version") in {"eb/rpo-wbc-model/v1", "eb/rpo-amp-model/v1", "eb/rpo-flat-model/v1","eb/rpo-bfm-model/v1"}]}


async def _prepare_rpo(body: RpoEvalIn, session) -> tuple[JobSpec, dict]:
    from eb.checkpoints import checkpoint_model
    from eb.store.ebd import EbdClient

    model = await session.get(ModelVersion, body.model_version_id)
    if body.video_quality == "hd" and (not body.record_video or not model
            or (model.card or {}).get("schema_version") not in {"eb/rpo-flat-model/v1","eb/rpo-bfm-model/v1"}):
        raise ValueError("高清录像需要启用 RPO-Flat 录像评估")
    card = (model.card or {}) if model else {}
    if body.profile in {"validation", "acceptance"} and card.get("schema_version") != "eb/rpo-bfm-model/v1":
        raise ValueError("RPO validation/acceptance profiles currently require BFM")
    if body.profile in {"validation", "acceptance"} and not body.record_video:
        raise ValueError("BFM validation/acceptance requires actual case videos")
    if (body.eval_gpus!=1 or body.envs_per_gpu!=1) and card.get("schema_version")!="eb/rpo-bfm-model/v1":
        raise ValueError("批量四卡评估目前仅支持BFM")
    if card.get("schema_version") in {"eb/rpo-flat-model/v1","eb/rpo-bfm-model/v1"}:
        return await _prepare_rpo_flat(body, model, session)
    amp = card.get("schema_version") == "eb/rpo-amp-model/v1"
    expected_env = "ee_rpo_v2_amp_locomotion_23dof_v1" if amp else "ee_rpo_v2_wbc_23dof_v1"
    expected_card = "eb/rpo-amp-model/v1" if amp else "eb/rpo-wbc-model/v1"
    artifact = await session.get(Artifact, model.artifact_id) if model and model.artifact_id else None
    if (not model or model.embodiment_env_id != expected_env
            or card.get("schema_version") != expected_card
            or not artifact or artifact.kind != "rlinf_checkpoint"):
        raise ValueError("请选择已登记的 RPO V2 RL 模型")
    source_run = await session.get(Run, card.get("run_id"))
    if not source_run or source_run.status != "finished":
        raise ValueError("源训练 Run 尚未成功完成")
    checkpoint = checkpoint_model(get_settings().data_root / "runs" / source_run.id,
                                  source_run.id, verify_hashes=True)
    canonical = json.dumps(checkpoint["manifest"], sort_keys=True, separators=(",", ":"))
    if (checkpoint["path"] != artifact.storage or checkpoint["step"] != card.get("step")
            or artifact.sha256 != "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()):
        raise ValueError("模型登记信息与训练产物不一致")
    tracking_suite = ("suite_rpo_v2_wbc_isaac_v1" if body.backend == "isaac"
                      else "suite_rpo_v2_wbc_sim2sim_v1")
    stability_suite = f"suite_rpo_v2_wbc_{body.backend}_stability_v1"
    basic_suite = f"suite_rpo_v2_wbc_{body.backend}_stand_walk_v2"
    amp_suite = f"suite_rpo_v2_amp_{body.backend}_stand_walk_v1"
    suite_id = body.suite_id or (amp_suite if amp else basic_suite if card.get("template_id") == "rpo_v2_wbc_stand_walk_ppo"
                                 else stability_suite)
    if suite_id not in ((amp_suite,) if amp else (tracking_suite, stability_suite, basic_suite)):
        raise ValueError("评估套件与仿真环境不匹配")
    criterion = ("stability" if suite_id == stability_suite else
                 "stand_walk" if suite_id in {basic_suite, amp_suite} else "tracking")
    suite = await session.get(EvalSuite, suite_id)
    if not suite or not suite.frozen or suite.embodiment_env_id != model.embodiment_env_id:
        raise ValueError("RPO 评估套件未就绪")
    if criterion == "stability" and ((suite.spec or {}).get("criterion") != "stability"
                                     or (suite.spec or {}).get("required_steps") != 1000):
        raise ValueError("身体稳定套件的成功标准未冻结")
    if criterion == "stand_walk" and not amp:
        expected = {"criterion":"stand_walk", "required_steps":1000,
                    "cases":"even_seed_stand_odd_seed_walk",
                    "initial_joint_offset_rad":[-0.05, 0.05],
                    "initial_base_velocity_mps":[-0.05, 0.05],
                    "initial_yaw_rad":[-3.141592653589793, 3.141592653589793],
                    "stand_max_drift_m":0.5, "walk_min_forward_m":4.0,
                    "fall_height_m":0.5, "fall_tilt_rad":0.8}
        if any((suite.spec or {}).get(key) != value for key, value in expected.items()):
            raise ValueError("直立行走套件的任务和成功标准未冻结")
    amp_expected = {"benchmark":"rpo_amp", "backend":body.backend,
                    "policy":"deterministic_onnx", "criterion":"stand_walk",
                    "qualification":"diagnostic", "required_steps":1000,
                    "cases":"even_seed_stand_odd_seed_walk", "stand_max_drift_m":0.5,
                    "walk_min_forward_m":4.0, "fall_height_m":0.5, "fall_tilt_rad":0.8}
    if amp and any((suite.spec or {}).get(key) != value for key, value in amp_expected.items()):
        raise ValueError("AMP 诊断套件未冻结")
    image = await EbdClient().image_info("eb/rlinf:rpo-v2-wbc")
    if not image.get("ok") or not str(image.get("digest", "")).startswith("sha256:"):
        raise ValueError("RPO 评估镜像未就绪")
    seeds = [body.seed] if body.seed is not None else list(range(3 if body.profile == "smoke" else 12))
    plan = {"schema_version":"eb/rpo-amp-eval-plan/v1" if amp else "eb/rpo-wbc-eval-plan/v1", "model_version_id":model.id,
            "source_run_id":source_run.id, "checkpoint_step":checkpoint["step"],
            "checkpoint_sha256":artifact.sha256, "image_digest":image["digest"],
            "suite_id":suite.id, "profile":body.profile, "record_video":body.record_video,
            "backend":body.backend, "criterion":criterion, "required_steps":1000,
            "suite_spec":suite.spec or {},
            "seeds":seeds, "seed_override":body.seed is not None,
            "implicit_seed_defaults":True}
    task_id = "tsk_rpo_amp_stand_walk" if amp else "tsk_rpo_stand_walk" if criterion == "stand_walk" else "tsk_rpo_walk_upperbody_tracking"
    environment_snapshot = await _bind_environment(body.environment_id, suite, task_id, plan, session)
    plan["hash"] = "sha256:" + hashlib.sha256(json.dumps(
        {k:v for k,v in plan.items() if k != "hash"}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    spec = JobSpec(kind=JobKind.EVAL_SIM if body.backend == "isaac" else JobKind.EVAL_SIM2SIM,
        embodiment_env_id=model.embodiment_env_id,
        task_id=task_id,
        name=f"{model.name}-{body.backend}-{body.profile}{'-video' if body.record_video else ''}",
        image=image["digest"], inputs={"base_model":checkpoint["path"]},
        params={"rpo_checkpoint_eval":True, "model_version_id":model.id,
                "suite_id":suite.id, "seeds":seeds, "record_video":body.record_video,
                "rpo_eval_plan":plan,
                **({"environment_profile":environment_snapshot} if environment_snapshot else {}),
                **({"stack":"rlinf", "template_id":"rpo_v2_amp_eval" if amp else "rpo_v2_wbc_eval"}
                   if body.backend == "isaac" else {})},
        resources={"gpus":1, "cpus":4, "memory_gb":16, "shm_gb":8})
    return spec, plan


async def _prepare_rpo_flat(body: RpoEvalIn, model: ModelVersion, session) -> tuple[JobSpec, dict]:
    from eb.checkpoints import checkpoint_model
    from eb.rpo_flat import FLAT_ENV, FLAT_TASK, FLAT_SUITE, flat_suite_spec
    from eb.store.ebd import EbdClient
    bfm=(model.card or {}).get("schema_version")=="eb/rpo-bfm-model/v1"
    native = (model.card or {}).get("execution")
    if native:
        from eb.domain.training_recipe import ExecutionBinding, RECIPES
        binding = ExecutionBinding.model_validate(native)
        if binding.recipe_id not in RECIPES or binding != RECIPES[binding.recipe_id].binding or not bfm:
            raise ValueError("Unknown native model execution binding")
    native_four = bool(native and binding.topology_id == "one_learner_three_samplers")
    if native_four and body.envs_per_gpu != 1:
        raise ValueError("UFO evaluation currently admits one environment per GPU")
    if native and not native_four and (body.eval_gpus != 1 or body.envs_per_gpu != 1):
        raise ValueError("UFO 四卡评估尚未通过资格，当前仅支持单卡独立 case")
    parallel=body.eval_gpus!=1 or body.envs_per_gpu!=1
    if bfm and body.profile in {"validation", "acceptance"}:
        parallel = True
    if parallel and not bfm:raise ValueError("批量四卡评估目前仅支持BFM")
    eval_template="rpo_v2_bfm_zero_eval" if bfm else "rpo_v2_flat_eval"
    if native:
        eval_template = "rpo_bfm_ufo_agent_rlinf_isaac_eval_v1"
    if bfm:
        from eb.rpo_bfm import ENV_ID,TASK_ID,EVAL_SUITE,eval_suite_spec,eval_case_seeds
        FLAT_ENV,FLAT_TASK,FLAT_SUITE,flat_suite_spec=ENV_ID,TASK_ID,EVAL_SUITE,eval_suite_spec
        if parallel:
            from eb.rpo_bfm import EVAL_BATCH_SUITE,batch_suite_spec
            FLAT_SUITE,flat_suite_spec=EVAL_BATCH_SUITE,batch_suite_spec
        if body.profile in {"validation", "acceptance"}:
            from eb.rpo_bfm import VALIDATION_SUITE, ACCEPTANCE_SUITE, frozen_eval_suite_spec
            FLAT_SUITE = VALIDATION_SUITE if body.profile == "validation" else ACCEPTANCE_SUITE
            if native_four:
                from eb.rpo_bfm import UFO_VALIDATION_SUITE, UFO_ACCEPTANCE_SUITE, UFO_REPAIR_ACCEPTANCE_SUITE
                FLAT_SUITE = UFO_VALIDATION_SUITE if body.profile == "validation" else UFO_ACCEPTANCE_SUITE
                if body.profile == "acceptance" and body.suite_id == UFO_REPAIR_ACCEPTANCE_SUITE:
                    FLAT_SUITE = UFO_REPAIR_ACCEPTANCE_SUITE
            flat_suite_spec = lambda: frozen_eval_suite_spec(FLAT_SUITE, body.profile)

    if body.backend != "isaac" or body.suite_id not in {None, FLAT_SUITE} or model.embodiment_env_id != FLAT_ENV:
        raise ValueError("RPO-Flat 使用官方 Isaac 速度/转向套件；当前不提供 MuJoCo 验收")
    artifact = await session.get(Artifact, model.artifact_id)
    source_run = await session.get(Run, (model.card or {}).get("run_id"))
    source_job = await session.get(Job, source_run.job_id) if source_run else None
    suite = await session.get(EvalSuite, FLAT_SUITE)
    frozen_seeds = (eval_case_seeds(FLAT_SUITE, body.profile)
                    if bfm and body.profile in {"validation", "acceptance"} else list(range(50000, 50100)))
    if (not artifact or artifact.kind != "rlinf_checkpoint" or not source_run or source_run.status != "finished"
            or not source_job or not suite or not suite.frozen or suite.embodiment_env_id != FLAT_ENV
            or suite.spec != flat_suite_spec() or suite.seeds != frozen_seeds):
        raise ValueError("RPO-Flat 模型、源训练或冻结套件未就绪")
    checkpoint = checkpoint_model(get_settings().data_root / "runs" / source_run.id,
                                  source_run.id, verify_hashes=not native)
    digest = "sha256:" + hashlib.sha256(json.dumps(checkpoint["manifest"],
                                  sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if (checkpoint["path"] != artifact.storage or checkpoint["step"] != model.card.get("step")
            or digest != artifact.sha256 or checkpoint["manifest"].get("template_id") != (native["recipe_id"] if native else "rpo_v2_bfm_zero_pretrain" if bfm else "rpo_v2_flat_ppo")):
        raise ValueError("RPO-Flat 模型登记与完整 checkpoint 不一致")
    refs = source_job.spec.get("code_refs") or {}
    legacy_bfm_assets=bfm and not (refs.get("embodiment") or {}).get("commit", "").startswith("sha256:")
    if legacy_bfm_assets:
        if body.profile!="smoke":
            raise ValueError("历史BFM训练只有资产manifest绑定，完整验收需新的资产快照训练")
        refs={**refs,"embodiment":{"kind":"content_snapshot","path":str(get_settings().assets_root/"roboparty/rpo-v2")}}
    if not legacy_bfm_assets and not (refs.get("embodiment") or {}).get("commit", "").startswith("sha256:"):
        raise ValueError("RPO-Flat 源训练缺少冻结的官方环境和资产快照")
    # Preserve producer provenance; independently freeze the evaluator runtime at preview.
    provenance = source_run.provenance
    if isinstance(provenance, str):
        provenance = json.loads(provenance)
    source_image = (provenance or {}).get("image_digest")
    if not isinstance(source_image, str) or not source_image.startswith("sha256:"):
        raise ValueError("RPO-Flat 源训练缺少实际执行镜像凭证")
    producer = await EbdClient().image_info(source_image)
    if not producer.get("ok") or producer.get("digest") != source_image:
        raise ValueError("RPO-Flat 源训练镜像不可用")
    if native:
        from eb.domain.training_recipe import recipe_eval_image_digest
    image = await EbdClient().image_info(recipe_eval_image_digest(binding.recipe_id) if native else template_image(get_template(eval_template)))
    if not image.get("ok") or not str(image.get("digest", "")).startswith("sha256:"):
        raise ValueError("RPO-Flat 评估镜像不可用")
    if bfm and body.seed is not None:
        raise ValueError("BFM评估使用冻结独立seeds，不提供临时seed覆盖")
    seeds = [body.seed] if body.seed is not None else list(range(60000,60004 if bfm else 60003) if body.profile == "smoke" else suite.seeds)
    if bfm:
        from eb.rpo_bfm import eval_case_seeds
        seeds = eval_case_seeds(suite.id, body.profile)
    plan = {"schema_version":"eb/rpo-bfm-eval-plan/v1" if bfm else "eb/rpo-flat-eval-plan/v1", "model_version_id":model.id,
            "source_run_id":source_run.id, "checkpoint_step":checkpoint["step"],
            "checkpoint_sha256":artifact.sha256, "image_digest":image["digest"],
            "source_image_digest":source_image,
            "suite_id":suite.id, "profile":body.profile, "record_video":body.record_video,
            "video_quality":body.video_quality,
            "backend":"isaac", "criterion":"velocity_commands", "required_steps":1000,
            "suite_spec":suite.spec, "seeds":seeds, "seed_override":body.seed is not None,
            "asset_snapshot":refs["embodiment"]}
    if legacy_bfm_assets:plan["source_asset_snapshot"]="not_available_for_legacy_smoke"
    if parallel:
        plan["parallelism"]={"gpus":body.eval_gpus,"envs_per_gpu":body.envs_per_gpu,
                             "protocol":"independent_cases_fixed_terrain_v2" if native_four else "independent_cases_shared_terrain_v1"}
    datasets=[]
    if bfm:
        import pathlib
        from eb.data_validation import require_bfm_motion_dataset
        from eb.data_validation import require_ufo_robot_state_dataset
        from eb.rpo_bfm import CODEC_HASH
        frozen=(source_job.spec.get("inputs") or {}).get("datasets") or []
        if len(frozen)!=1:raise ValueError("BFM评估必须绑定源训练的单个冻结motion view")
        dataset=await session.get(Dataset,frozen[0]["id"])
        if not dataset:raise ValueError("BFM源数据版本缺失")
        validation=require_ufo_robot_state_dataset(dataset) if native else require_bfm_motion_dataset(dataset)
        if validation.get("manifest_sha256")!=(frozen[0].get("validation") or {}).get("manifest_sha256"):
            raise ValueError("BFM源训练motion hash改变")
        assets=pathlib.Path(refs["embodiment"]["path"])/"asset-manifest.json"
        asset_sha=hashlib.sha256(assets.read_bytes()).hexdigest()
        if asset_sha!=validation.get("asset_sha256"):raise ValueError("BFM评估资产与motion失配")
        datasets=frozen
        plan.update(codec_sha256=CODEC_HASH,asset_sha256=asset_sha,motion_manifest_sha256=validation["manifest_sha256"],
            source_weights_sha256=next(item["sha256"] for item in checkpoint["manifest"]["files"] if item["path"]==checkpoint["manifest"].get("weights_path", "model/model_state_dict/full_weights.pt")),
            source_config_sha256=hashlib.sha256((get_settings().data_root/"runs"/source_run.id/"output/resolved-config.yaml").read_bytes()).hexdigest(),
            diagnostic_commands=body.profile=="smoke")
        if native:
            plan.update(execution=native, progress_unit="learner_joint_update")
    environment = await _bind_environment(body.environment_id, suite, FLAT_TASK, plan, session)
    plan["hash"] = "sha256:" + hashlib.sha256(json.dumps(
        {k:v for k,v in plan.items() if k != "hash"}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    spec = JobSpec(kind=JobKind.EVAL_SIM, embodiment_env_id=FLAT_ENV, task_id=FLAT_TASK,
        name=f"{model.name}-flat-isaac-{body.profile}", image=image["digest"],
        inputs={"base_model":checkpoint["path"],"datasets":datasets}, code_refs=refs,
        params={"rpo_checkpoint_eval":True, "stack":"rlinf", "template_id":eval_template,
                "model_version_id":model.id, "suite_id":suite.id, "seeds":seeds,
                "record_video":body.record_video, "video_quality":body.video_quality, "rpo_eval_plan":plan,
                **({"model_execution": native} if native else {}),
                **({"environment_profile":environment} if environment else {})},
        resources={"gpus":body.eval_gpus, "cpus":64 if body.eval_gpus==4 else 32 if body.video_quality == "hd" else 4,
                   "memory_gb":96 if body.eval_gpus==4 else 24, "shm_gb":32 if body.eval_gpus==4 else 8},
        policy={"timeout_h":6})
    return spec, plan


@router.post("/rpo/preview")
async def preview_rpo(body: RpoEvalIn, session: SessionDep, principal: PrincipalDep) -> dict:
    from eb.adapters import get_adapter
    try:
        spec, plan = await _prepare_rpo(body, session)
        rendered = get_adapter(spec).render(spec, "run_preview")
    except (ValueError, OSError) as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"plan":plan, "image":rendered.image, "argv":rendered.command,
            "resources":spec.resources.model_dump()}


@router.post("/rpo/jobs")
async def create_rpo_job(body: RpoEvalIn, session: SessionDep, principal: PrincipalDep,
                         idempotency_key: str | None = Header(default=None, alias="Idempotency-Key")) -> dict:
    if not body.preview_hash:
        raise HTTPException(400, "请先预览 RPO 评估计划")
    if idempotency_key and len(idempotency_key) > 128:
        raise HTTPException(400, "幂等键过长")
    try:
        spec, plan = await _prepare_rpo(body, session)
        if body.preview_hash != plan["hash"]:
            raise HTTPException(409, "评估预览已过期，请重新检查")
        scope = f"eval:rpo:{principal.id}:{idempotency_key}" if idempotency_key else None
        if scope:
            old = await session.scalar(select(Job).where(Job.idem_key == scope))
            if old:
                if ((old.spec.get("params") or {}).get("rpo_eval_plan") != plan):
                    raise HTTPException(409, "幂等键已用于不同评估计划")
                return _job_out(old, None)
        job = await orch.submit(session, spec, principal.id, scope, commit=False)
        session.add(LineageEdge(src_type="model_version", src_id=body.model_version_id,
                                dst_type="job", dst_id=job.id, relation="evaluated"))
        session.add(LineageEdge(src_type="eval_suite", src_id=plan["suite_id"],
                                dst_type="job", dst_id=job.id, relation="evaluated_with"))
        await session.commit()
        return _job_out(job, None)
    except HTTPException:
        raise
    except (ValueError, OSError, PermissionError) as exc:
        raise HTTPException(400, str(exc)) from exc


def build_eval_plan(suite_id: str, suite_spec: dict, source: dict, profile: str,
                    benchmark: dict | None = None) -> dict:
    if suite_spec.get("benchmark") == "robotwin":
        suite_name = suite_spec.get("suite")
        if not isinstance(suite_name, str) or suite_id != f"suite_{suite_name}":
            raise ValueError("评估套件不是受支持的 RoboTwin 冻结版本")
        required = ("robotwin_revision", "rlinf_revision", "assets_manifest_sha256",
                    "controller_profile", "observation_profile", "domain", "max_episode_steps")
        missing = [key for key in required if not suite_spec.get(key)]
        if missing:
            raise ValueError("RoboTwin Suite 缺少冻结字段: " + ", ".join(missing))
        frozen_cases = suite_spec.get("cases")
        if not isinstance(frozen_cases, list) or not frozen_cases:
            raise ValueError("RoboTwin Suite 缺少冻结 case")
        case_ids, seeds = set(), set()
        for case in frozen_cases:
            if (not isinstance(case, dict) or not isinstance(case.get("case_id"), str)
                    or not isinstance(case.get("task_id"), str) or type(case.get("seed")) is not int
                    or not isinstance(case.get("scenario"), str)):
                raise ValueError("RoboTwin Suite case 字段不完整")
            if case["case_id"] in case_ids or (case["task_id"], case["seed"]) in seeds:
                raise ValueError("RoboTwin Suite 包含重复 case 或 seed")
            case_ids.add(case["case_id"])
            seeds.add((case["task_id"], case["seed"]))
        if profile == "smoke":
            cases = frozen_cases[:5]
        elif profile in {"validation", "standard", "acceptance", "official_reproduction"}:
            cases = frozen_cases
        else:
            raise ValueError("未知评估 profile")
        plan = {"schema_version": "eb/robotwin-eval-plan/v1", "suite_id": suite_id,
                "suite_spec": suite_spec, "profile": profile, "model": source,
                "planned": len(cases), "cases": cases, "qualification": "not_requested"}
        plan["hash"] = "sha256:" + hashlib.sha256(json.dumps(
            plan, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
        return plan
    suite_name = suite_spec.get("suite")
    if (suite_name not in {f"libero_{name}" for name in LIBERO_SUITES}
            or suite_id != f"suite_tsk_{suite_name}"):
        raise ValueError("评估套件不是受支持的 LIBERO 冻结版本")
    if profile == "smoke":
        tasks, trials = range(4), range(1)
    elif profile == "standard":
        tasks, trials = range(10), range(10)
    else:
        raise ValueError("未知评估 profile")
    cases = [{"case_id": f"{suite_name}:task{task}:init{trial}", "task_id": task,
              "init_state_id": trial} for trial in trials for task in tasks]
    plan = {"schema_version": "eb/libero-eval-plan/v1", "suite_id": suite_id,
            "suite_spec": suite_spec, "profile": profile, "checkpoint": source,
            "planned": len(cases), "cases": cases, "qualification": "not_requested"}
    if benchmark:
        plan["benchmark"] = benchmark
    plan["hash"] = "sha256:" + hashlib.sha256(json.dumps(
        plan, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    return plan


def robotwin_candidate(model: ModelVersion, artifact: Artifact | None, suite: EvalSuite) -> dict:
    card = model.card or {}
    blockers = []
    if model.embodiment_env_id != suite.embodiment_env_id:
        blockers.append({"code":"action_mapping_unverified", "field_path":"embodiment_env_id",
                         "message":"模型本体接口与 Suite 不一致"})
    if not artifact or artifact.kind != "model_package" or not artifact.pinned:
        blockers.append({"code":"model_package_incomplete", "field_path":"artifact_id",
                         "message":"缺少固定的完整模型包"})
    readiness = card.get("readiness")
    if readiness not in {"adapter_ready", "smoke_passed", "validated"}:
        blockers.append({"code":card.get("blocker_code") or "eval_recipe_blocked",
                         "field_path":"card.readiness", "message":card.get("blocker") or "闭环评估 recipe 未就绪"})
    return {"id":model.id, "name":model.name, "family_id":model.family_id,
            "artifact_sha256":artifact.sha256 if artifact else None,
            "template_id":card.get("eval_template_id"), "readiness":"blocked" if blockers else readiness,
            "blockers":blockers, "io_signature":model.io_signature or {}, "card":card}


async def _prepare_robotwin(body: EvalIn, suite: EvalSuite, session):
    if not body.model_version_id or body.checkpoint_run_id is not None or body.checkpoint_step is not None:
        raise ValueError("RoboTwin 评估必须且只能选择 ModelVersion")
    model = await session.get(ModelVersion, body.model_version_id)
    artifact = await session.get(Artifact, model.artifact_id) if model and model.artifact_id else None
    if not model:
        raise ValueError("模型版本不存在")
    candidate = robotwin_candidate(model, artifact, suite)
    if candidate["blockers"]:
        raise ValueError(candidate["blockers"][0]["message"])
    template = get_template(str(candidate["template_id"]))
    require_template_ready(template)
    from eb.checkpoints import checkpoint_model
    checkpoint = checkpoint_model(get_settings().data_root / "runs" / artifact.run_id, artifact.run_id,
                                  verify_hashes=(candidate.get("execution") or {}).get("checkpoint_format") != "ufo_agent_state_v1")
    if checkpoint["path"] != artifact.storage:
        raise ValueError("模型包路径与不可变 Artifact 不一致")
    if body.runtime_profile_id or body.resource_variant_id:
        if not body.runtime_profile_id or not body.resource_variant_id:
            raise ValueError("RoboTwin 运行档位与资源变体必须同时选择")
        runtime, resources = resolve_robotwin_runtime(
            body.runtime_profile_id, body.resource_variant_id)
    else:
        runtime, resources = derive_robotwin_runtime(body.gpus, body.parallel_envs)
    from eb.store.ebd import EbdClient
    from eb.adapters.rlinf import _git_head
    from eb.runners.robotwin_source import PATCH_SHA256, PATCHED_BASE_TASK_SHA256
    settings = get_settings()
    if _git_head(settings.rlinf_repo) != (suite.spec or {}).get("rlinf_revision"):
        raise ValueError("RLInf 源码 revision 与冻结 Suite 不一致")
    if _git_head(settings.robotwin_repo) != (suite.spec or {}).get("robotwin_revision"):
        raise ValueError("RoboTwin 源码 revision 与冻结 Suite 不一致")
    ebd = EbdClient()
    try:
        snapshot = await ebd.snapshot(str(settings.rlinf_repo))
        robotwin_snapshot = await ebd.snapshot(str(settings.robotwin_repo))
    except httpx.HTTPError as exc:
        raise HTTPException(503, "评估数据面暂不可用，无法冻结 RLInf 源码") from exc
    content_hash = snapshot.get("content_hash")
    robotwin_content_hash = robotwin_snapshot.get("content_hash")
    if (not isinstance(snapshot.get("path"), str) or not isinstance(content_hash, str)
            or not content_hash.startswith("sha256:")
            or not isinstance(robotwin_snapshot.get("path"), str)
            or not isinstance(robotwin_content_hash, str)
            or not robotwin_content_hash.startswith("sha256:")):
        raise ValueError("RoboTwin/RLInf 内容快照未就绪")
    image = await ebd.image_info(template_image(template))
    if image.get("status") == "unavailable":
        raise HTTPException(503, "评估数据面暂不可用，无法检查任务镜像")
    if not image.get("ok") or not str(image.get("digest", "")).startswith("sha256:"):
        raise ValueError("评估任务镜像未就绪，无法冻结执行版本")
    source = {"model_version_id":model.id, "artifact_id":artifact.id,
              "artifact_sha256":artifact.sha256, "package_path":artifact.storage,
              "template_id":template.id, "inference":{
                  "num_action_chunks":50, "precision":(suite.spec or {}).get("precision", "bf16")},
              "runtime":{"image_digest":image["digest"], "policy_bridge":"rlinf_openpi_pi05",
                         "resources":resources, **runtime.model_dump(),
                         "rlinf_content_hash":content_hash}}
    source["runtime"].update(robotwin_content_hash=robotwin_content_hash,
                             robotwin_evidence_patch_sha256=PATCH_SHA256,
                             patched_base_task_sha256=PATCHED_BASE_TASK_SHA256)
    plan = build_eval_plan(suite.id, suite.spec or {}, source, body.profile)
    environment_snapshot = await _bind_environment(body.environment_id, suite, template.task_id, plan, session)
    if body.preview_hash and body.preview_hash != plan["hash"]:
        raise StalePreview("预览已过期，请重新检查评估配置")
    spec = JobSpec(kind=JobKind.EVAL_SIM, embodiment_env_id=suite.embodiment_env_id,
        task_id=template.task_id, name=body.name, image=image["digest"],
        params={"template_id":template.id, "config_name":template.config_name,
                "stack":"rlinf", "suite_id":suite.id, "model_version_id":model.id,
                "eval_plan":plan}, inputs={"base_model":artifact.storage},
        resources=resources,
        code_refs={"stack":{"kind":"content_snapshot", "path":snapshot["path"],
                            "commit":content_hash},
                   "embodiment":{"kind":"content_snapshot", "path":robotwin_snapshot["path"],
                                  "commit":robotwin_content_hash}})
    if environment_snapshot:
        spec.params["environment_profile"] = environment_snapshot
    return spec, plan


async def _prepare(body: EvalIn, session):
    suite = await session.get(EvalSuite, body.suite_id)
    if not suite or not suite.frozen:
        raise ValueError("评估套件不存在或未冻结")
    if (suite.spec or {}).get("benchmark") == "robotwin":
        return await _prepare_robotwin(body, suite, session)
    if not body.checkpoint_run_id or body.checkpoint_step is None or body.model_version_id is not None:
        raise ValueError("LIBERO 评估必须且只能选择 checkpoint")
    suite_name = (suite.spec or {}).get("suite")
    if suite_name not in {f"libero_{name}" for name in LIBERO_SUITES}:
        raise ValueError("当前仅支持 LIBERO 仿真评估")
    dataset = await session.scalar(select(Dataset).where(
        Dataset.source_kind == "libero_benchmark", Dataset.source_uri == LIBERO_REPO))
    if dataset:
        await reconcile_dataset(dataset, session)
    benchmark_stats = (dataset.summary_stats or {}).get("benchmark") if dataset else None
    if (not dataset or dataset.status != "validated"
            or (dataset.meta or {}).get("source_revision") != LIBERO_REVISION
            or (dataset.meta or {}).get("usage") != "eval" or not benchmark_stats
            or not isinstance(benchmark_stats.get("sha256"), str)):
        raise ValueError("LIBERO 测试资产尚未在数据菜单完成下载与验证")
    benchmark = {"dataset_id": dataset.id, "repo": LIBERO_REPO,
                 "revision": LIBERO_REVISION, "sha256": benchmark_stats["sha256"]}
    source_run = await session.get(Run, body.checkpoint_run_id)
    source_job = await session.get(Job, source_run.job_id) if source_run else None
    source_train = await session.scalar(select(TrainJob).where(TrainJob.job_id == source_job.id)) if source_job else None
    if not source_train or source_job.kind not in {"train.sft", "train.rlft"}:
        raise ValueError("checkpoint 必须来自训练任务")
    template_id = f"{suite_name}_pi05_eval"
    template = get_template(template_id)
    if suite.embodiment_env_id != template.embodiment_env_id:
        raise ValueError("套件与评估本体环境不一致")
    train_body = TrainIn(name=body.name, template_id=template_id,
        embodiment_id=source_train.embodiment_id, image=source_train.image_digest,
        gpus=1, base_model_id=source_train.base_model_id,
        checkpoint_run_id=body.checkpoint_run_id, checkpoint_step=body.checkpoint_step)
    spec = await _build_spec(train_body, session)
    from eb.store.ebd import EbdClient
    provenance = {}
    if isinstance(getattr(source_run, "provenance", None), str):
        try:
            provenance = json.loads(source_run.provenance)
        except json.JSONDecodeError:
            pass
    if not isinstance(provenance, dict):
        provenance = {}
    source_image = provenance.get("image_digest") or spec.image
    image = await EbdClient().image_info(source_image)
    if image.get("status") == "unavailable":
        raise HTTPException(503, "评估数据面暂不可用，无法检查任务镜像")
    if not image.get("ok") or not str(image.get("digest", "")).startswith("sha256:"):
        raise ValueError("评估任务镜像未就绪，无法冻结执行版本")
    spec = spec.model_copy(update={"image": image["digest"]})
    try:
        frozen = await EbdClient().freeze_checkpoint(body.checkpoint_run_id, body.checkpoint_step)
    except httpx.HTTPStatusError as exc:
        raise ValueError(f"checkpoint 冻结失败：{exc.response.text[:300]}") from exc
    except httpx.HTTPError as exc:
        raise HTTPException(503, "评估数据面暂不可用，无法冻结 checkpoint") from exc
    if frozen.get("bytes") != spec.params["eval_checkpoint_bytes"] or not isinstance(frozen.get("sha256"), str):
        raise ValueError("checkpoint 冻结前后内容已改变")
    checkpoint = {"run_id": body.checkpoint_run_id, "step": body.checkpoint_step,
                  "bytes": spec.params["eval_checkpoint_bytes"],
                  "sha256": frozen["sha256"],
                  "base_model_id": source_train.base_model_id,
                  "source_job_id": source_job.id,
                  "source_template_id": source_train.template_id,
                  "source_inference": spec.params["eval_source_inference"],
                  "source_image": spec.image,
                  "source_image_ref": source_train.image_digest}
    plan = build_eval_plan(body.suite_id, suite.spec or {}, checkpoint, body.profile, benchmark)
    environment_snapshot = await _bind_environment(body.environment_id, suite, template.task_id, plan, session)
    if body.preview_hash and body.preview_hash != plan["hash"]:
        raise StalePreview("预览已过期，请重新检查评估配置")
    # Smoke is four total cases; standard requests ten cases per task.
    overrides = ["env.eval.total_num_envs=4", "env.eval.rollout_epoch=1"] if body.profile == "smoke" else [
        "env.eval.total_num_envs=10", "env.eval.rollout_epoch=10"]
    spec = spec.model_copy(update={"params": {**spec.params, "eval_plan": plan,
                                                  **({"environment_profile":environment_snapshot} if environment_snapshot else {}),
                                                  "eval_checkpoint_sha256": frozen["sha256"],
                                                  "suite_id": body.suite_id,
                                                  "benchmark_dataset_id": dataset.id,
                                                  "benchmark_sha256": benchmark["sha256"]},
                                   "overrides": overrides})
    if spec.kind != JobKind.EVAL_SIM:
        raise ValueError("评估请求未生成 eval.sim Job")
    return spec, plan


@router.get("/candidates")
async def list_candidates(suite_id: str, session: SessionDep, principal: PrincipalDep) -> dict:
    suite = await session.get(EvalSuite, suite_id)
    if not suite or not suite.frozen:
        raise HTTPException(404, "评估套件不存在或未冻结")
    if (suite.spec or {}).get("benchmark") != "robotwin":
        return {"items": []}
    rows = (await session.scalars(select(ModelVersion).order_by(ModelVersion.created_at.desc()))).all()
    items = []
    for model in rows:
        if "robotwin" not in (model.tags or []):
            continue
        artifact = await session.get(Artifact, model.artifact_id) if model.artifact_id else None
        items.append(robotwin_candidate(model, artifact, suite))
    return {"items":items}


@router.get("/runtime-profiles")
async def runtime_profiles(principal: PrincipalDep) -> dict:
    return {"items": list_robotwin_runtime_profiles()}


@router.get("/jobs")
async def list_jobs(session: SessionDep, principal: PrincipalDep) -> dict:
    rows = (await session.scalars(select(Job).where(Job.kind.in_(("eval.sim", "eval.sim2sim")))
                                  .order_by(Job.created_at.desc()))).all()
    if not rows:
        return {"items": []}
    runs = (await session.scalars(select(Run).where(Run.job_id.in_([j.id for j in rows]))
                                  .order_by(Run.attempt.desc()))).all()
    latest = {}
    for run in runs:
        latest.setdefault(run.job_id, run)
    return {"items": [_job_out(job, latest.get(job.id)) for job in rows]}


@router.get("/checkpoints")
async def list_checkpoints(session: SessionDep, principal: PrincipalDep) -> dict:
    """Expose managed legacy RLInf weights without accepting arbitrary host paths."""
    from eb.checkpoints import checkpoint_weights
    from eb.settings import get_settings
    rows = (await session.scalars(select(TrainJob).order_by(TrainJob.created_at.desc()))).all()
    items = []
    for train in rows:
        if not train.job_id or not train.template_id.startswith("libero_"):
            continue
        job = await session.get(Job, train.job_id)
        if not job or job.kind not in {"train.sft", "train.rlft"} or job.status not in {
                "succeeded", "failed", "timeout", "cancelled", "infra_failed"}:
            continue
        runs = (await session.scalars(select(Run).where(Run.job_id == job.id))).all()
        for run in runs:
            root = get_settings().data_root / "runs" / run.id
            for path in (root / "output").glob("*/checkpoints/global_step_*"):
                suffix = path.name.removeprefix("global_step_")
                if not suffix.isdigit():
                    continue
                try:
                    ckpt = checkpoint_weights(root, int(suffix))
                except (ValueError, OSError):
                    continue
                items.append({"key": f"{run.id}:{suffix}", "run_id": run.id,
                              "step": ckpt["step"], "bytes": ckpt["bytes"],
                              "name": train.name, "source_status": job.status,
                              "base_model_id": train.base_model_id})
    return {"items": items}


def _job_out(job: Job, run: Run | None) -> dict:
    spec = job.spec or {}
    params = spec.get("params") or {}
    plan = params.get("eval_plan") or {}
    return {"id": job.id, "name": spec.get("name") or job.id, "kind": job.kind,
            "status": job.status, "error": job.error, "template_id": params.get("template_id"),
            "suite_id": params.get("suite_id") or (plan.get("suite_id")),
            "plan": plan or params.get("rpo_eval_plan") or None,
            "model_version_id":params.get("model_version_id"),
            "record_video":params.get("record_video", False),
            "checkpoint": plan.get("checkpoint") or {"run_id": params.get("eval_checkpoint_run_id"),
                                                       "step": params.get("eval_checkpoint_step")},
            "run_id": run.id if run else None,
            "metrics_summary": run.metrics_summary or {} if run else {},
            "created_at": job.created_at.isoformat() if job.created_at else None,
            "started_at": run.started_at.isoformat() if run and run.started_at else None,
            "finished_at": run.finished_at.isoformat() if run and run.finished_at else None}


@router.get("/jobs/{job_id}")
async def get_job(job_id: str, session: SessionDep, principal: PrincipalDep) -> dict:
    job = await session.get(Job, job_id)
    if not job or job.kind not in {"eval.sim", "eval.sim2sim"}:
        raise HTTPException(404)
    run = await session.scalar(select(Run).where(Run.job_id == job.id).order_by(Run.attempt.desc()))
    return _job_out(job, run)


@router.get("/jobs/{job_id}/progress")
async def get_job_progress(job_id: str, session: SessionDep, principal: PrincipalDep) -> dict:
    """Count only persisted case records; no wall-clock estimate."""
    from eb.settings import get_settings
    job = await session.get(Job, job_id)
    if not job or job.kind != "eval.sim":
        raise HTTPException(404)
    params = (job.spec or {}).get("params") or {}
    plan = params.get("eval_plan") or {}
    flat_plan = params.get("rpo_eval_plan") or {}
    flat = flat_plan.get("schema_version") == "eb/rpo-flat-eval-plan/v1"
    if flat:
        plan = {**flat_plan, "planned":len(flat_plan["seeds"]),
                "cases":[{"case_id":f"rpo-flat-seed-{seed}", "task_id":"tsk_rpo_flat"}
                         for seed in flat_plan["seeds"]]}
    run = await session.scalar(select(Run).where(Run.job_id == job_id).order_by(Run.attempt.desc()))
    if not plan or not run:
        return {"planned": plan.get("planned"), "recorded": 0, "run_id": run.id if run else None}
    runs_root = get_settings().data_root / "runs"
    run_root = runs_root / run.id
    root = run_root / "output"
    if any(part.is_symlink() for part in (runs_root, run_root, root)) or not root.is_dir():
        return {"planned": plan["planned"], "recorded": 0, "run_id": run.id}
    expected = {case["case_id"] for case in plan["cases"]}
    expected_by_task: dict[str, int] = {}
    for case in plan["cases"]:
        expected_by_task[case["task_id"]] = expected_by_task.get(case["task_id"], 0) + 1
    seen = set()
    raw_records = duplicate_records = invalid_records = 0
    recorded_by_task: dict[str, int] = {}
    for path in root.glob("eval-cases.jsonl" if flat else "eval-episodes-rank*.jsonl"):
        if path.is_symlink() or path.stat().st_size > max(1024 * 1024, int(plan["planned"]) * 4096):
            raise HTTPException(409, "评估案例进度证据不可用")
        data = path.read_text()
        lines = data.splitlines()
        if len(lines) > plan["planned"]:
            raise HTTPException(409, "评估案例进度证据超出上限")
        for line in lines:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                if not data.endswith("\n") and line == lines[-1]:
                    continue
                raise HTTPException(409, "评估案例进度证据损坏")
            raw_records += 1
            if not isinstance(row, dict) or row.get("case_id") not in expected:
                invalid_records += 1
                continue
            if row["case_id"] in seen:
                duplicate_records += 1
                continue
            seen.add(row["case_id"])
            task_id = row.get("task_id")
            recorded_by_task[task_id] = recorded_by_task.get(task_id, 0) + 1
    planned = int(plan["planned"])
    return {"planned": planned, "recorded": len(seen), "percent": round(len(seen) * 100 / planned, 2),
            "raw_records": raw_records, "duplicate_records": duplicate_records,
            "invalid_records": invalid_records, "run_id": run.id,
            "by_task":[{"task_id":task_id, "recorded":recorded_by_task.get(task_id, 0), "planned":count}
                       for task_id, count in expected_by_task.items()]}


@router.post("/previews")
async def preview(body: EvalIn, session: SessionDep, principal: PrincipalDep) -> dict:
    try:
        spec, plan = await _prepare(body, session)
        from eb.adapters import get_adapter
        rendered = get_adapter(spec).render(spec, "run_preview")
    except (ValueError, KeyError) as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"plan": plan, "image": rendered.image, "resources": spec.resources.model_dump(),
            "argv": rendered.command,
            "mounts":[{"source":source, "destination":destination, "mode":mode}
                      for source, destination, mode in rendered.mounts]}


@router.post("/jobs")
async def create_job(body: EvalIn, session: SessionDep, principal: PrincipalDep,
                     idempotency_key: str | None = Header(default=None, alias="Idempotency-Key")) -> dict:
    if not body.preview_hash:
        raise HTTPException(400, "请先预览并提交冻结的评估计划")
    if idempotency_key and len(idempotency_key) > 128:
        raise HTTPException(400, "幂等键过长")
    try:
        spec, plan = await _prepare(body, session)
        scope = f"eval:{principal.id}:{idempotency_key}" if idempotency_key else None
        if scope:
            old = await session.scalar(select(Job).where(Job.idem_key == scope))
            if old:
                if ((old.spec.get("params") or {}).get("eval_plan", {}).get("hash") != plan["hash"]
                        or old.spec.get("name") != spec.name):
                    raise HTTPException(409, "幂等键已用于不同评估计划")
                return _job_out(old, None)
        from eb.adapters import get_adapter
        get_adapter(spec).render(spec, "run_preview")
        job = await orch.submit(session, spec, principal.id, scope, commit=False)
        if body.model_version_id:
            session.add(LineageEdge(src_type="model_version", src_id=body.model_version_id,
                                    dst_type="job", dst_id=job.id, relation="evaluated"))
        else:
            session.add(LineageEdge(src_type="run", src_id=body.checkpoint_run_id,
                                    dst_type="job", dst_id=job.id, relation="evaluated_checkpoint"))
        session.add(LineageEdge(src_type="eval_suite", src_id=body.suite_id,
                                dst_type="job", dst_id=job.id, relation="evaluated_with"))
        await session.commit()
        await session.refresh(job)
        return _job_out(job, None)
    except HTTPException:
        raise
    except IntegrityError as exc:
        await session.rollback()
        raise HTTPException(409, "评估提交发生并发冲突，请刷新") from exc
    except StalePreview as exc:
        raise HTTPException(409, str(exc)) from exc
    except (ValueError, KeyError, PermissionError) as exc:
        raise HTTPException(400, str(exc)) from exc
