from __future__ import annotations

import base64
import hashlib
import json
import math
import re
import zlib
from pathlib import Path

from eb.adapters.base import Adapter, RenderedJob, _image
from eb.checkpoints import checkpoint_weights, resume_checkpoint, resume_fingerprint, require_flat_rank_rng
from eb.domain.job import JobKind, JobSpec
from eb.domain.environment_profile import rpo_eval_arguments
from eb.domain.rlinf_form import (build_overrides, get_template, require_template_ready,
                                  template_entrypoint, template_image)
from eb.settings import get_settings
from eb.runners.rlinf_entry import RLINF_PATCH_SHA256


def _eval_plan_env(plan: dict) -> dict[str, str]:
    raw = json.dumps(plan, ensure_ascii=False, separators=(",", ":")).encode()
    if len(raw) < 64 * 1024:
        return {"EB_EVAL_PLAN":raw.decode()}
    return {"EB_EVAL_PLAN_ZLIB":base64.b64encode(zlib.compress(raw, 9)).decode()}


def _git_head(root) -> str | None:
    try:
        head = (root / ".git/HEAD").read_text().strip()
        return (root / ".git" / head.removeprefix("ref: ")).read_text().strip() if head.startswith("ref: ") else head
    except OSError:
        return None


class RlinfAdapter(Adapter):
    kind = JobKind.TRAIN_SFT
    name = "rlinf"

    def render(self, spec: JobSpec, run_id: str) -> RenderedJob:
        from eb.domain.training_recipe import validate_training_execution
        binding = validate_training_execution(spec)
        native_ufo = binding is not None
        settings = get_settings()
        tpl = get_template(str(spec.params.get("template_id") or "berkeley_wbc_ppo_mlp"))
        require_template_ready(tpl)
        if native_ufo:
            from types import SimpleNamespace
            from eb.domain.environment_profile import profile_hash
            from eb.domain.training_recipe import recipe_profile_id
            from eb.environment_catalog import training_definition
            definition = training_definition(tpl)
            profile = SimpleNamespace(usage="train", embodiment_env_id=tpl.embodiment_env_id,
                task_id=tpl.task_id, template_id=tpl.id, suite_id=None, definition=definition)
            if spec.params.get("environment_profile") != {"id": recipe_profile_id(tpl.id),
                    "hash": profile_hash(profile), "definition": definition}:
                raise ValueError("UFO requires the frozen reference-reset Profile; old/custom Profiles cannot substitute")
        expected_kind = (JobKind.TRAIN_RL if native_ufo or tpl.id in {"rpo_v2_wbc_ppo", "rpo_v2_wbc_stand_walk_ppo", "rpo_v2_amp_stand_walk_ppo", "rpo_v2_flat_ppo", "rpo_v2_bfm_zero_pretrain", "rpo_v2_bfm_zero_tracking_teacher"} else
                         {"sft": JobKind.TRAIN_SFT, "ppo": JobKind.TRAIN_RLFT,
                          "eval": JobKind.EVAL_SIM, "collect": JobKind.DATA_COLLECT}[tpl.kind])
        expected_task = ("tsk_rpo_stand_walk" if tpl.id == "rpo_v2_wbc_eval"
                         and (spec.params.get("rpo_eval_plan") or {}).get("criterion") == "stand_walk"
                         else tpl.task_id)
        if (spec.kind != expected_kind or spec.embodiment_env_id != tpl.embodiment_env_id
                or spec.task_id != expected_task):
            raise ValueError("Job 类型、本体接口、任务必须与 RLinf 模板一致")
        if tpl.requires_base_model and not spec.inputs.base_model:
            raise ValueError("此模板必须选择基座模型")
        if tpl.requires_dataset and len(spec.inputs.datasets) != 1:
            raise ValueError("此模板必须绑定一个已验证的数据版本")
        flat_warm_start_binding = tpl.id == "rpo_v2_flat_ppo" and spec.params.get("warm_start")
        tracking = tpl.id == "rpo_v2_bfm_zero_tracking_teacher"
        if tracking and (spec.image not in (None, "eb/rlinf:rpo-v2-bfm-zero")
                         or spec.inputs.datasets or spec.inputs.base_model or spec.inputs.resume_from
                         or spec.policy.timeout_h > 1 or spec.policy.max_retries != 0):
            raise ValueError("Tracking teacher requires fresh single-GPU exploration bounded to 1h without retries")
        flat_warm_start = flat_warm_start_binding if not spec.inputs.resume_from else None
        if tpl.id in {"rpo_v2_wbc_ppo", "rpo_v2_wbc_stand_walk_ppo", "rpo_v2_flat_ppo"} and (spec.inputs.datasets or (spec.inputs.base_model and not flat_warm_start_binding)):
            raise ValueError("RPO WBC 在线 PPO 不接受数据集或基座大模型")
        if flat_warm_start_binding and not spec.inputs.base_model:
            raise ValueError("Flat warm start 必须绑定已发布权重")
        if tpl.id == "rpo_v2_amp_stand_walk_ppo" and spec.inputs.base_model:
            raise ValueError("RPO AMP 首期不接受基座模型")
        if spec.runtime.get("env"):
            raise ValueError("训练环境变量由运行配置管理，不允许覆盖")
        if tpl.id in {"rpo_v2_bfm_zero_eval", "rpo_bfm_ufo_agent_rlinf_isaac_eval_v1"}:
            from eb.rpo_bfm import frozen_eval_suite_spec,eval_case_seeds,CODEC_HASH,EVAL_BATCH_SUITE,VALIDATION_SUITE,ACCEPTANCE_SUITE,UFO_VALIDATION_SUITE,UFO_ACCEPTANCE_SUITE,UFO_REPAIR_ACCEPTANCE_SUITE
            from eb.adapters.isaaclab_rslrl import IsaacLabRslRlAdapter
            plan=spec.params.get("rpo_eval_plan") or {}
            if plan.get("profile") in {"validation", "acceptance"} and plan.get("record_video") is not True:
                raise ValueError("BFM validation/acceptance requires actual case videos")
            invariant = plan.get("suite_id") in {UFO_VALIDATION_SUITE,UFO_ACCEPTANCE_SUITE,UFO_REPAIR_ACCEPTANCE_SUITE}
            parallel=plan.get("suite_id") in {EVAL_BATCH_SUITE, VALIDATION_SUITE, ACCEPTANCE_SUITE,UFO_VALIDATION_SUITE,UFO_ACCEPTANCE_SUITE,UFO_REPAIR_ACCEPTANCE_SUITE}
            parallelism=plan.get("parallelism") or {}
            if tpl.id == "rpo_bfm_ufo_agent_rlinf_isaac_eval_v1" and parallelism.get("envs_per_gpu", 1) != 1:
                raise ValueError("UFO evaluation currently admits one environment per GPU")
            if parallel and (parallelism.get("gpus") not in (1,4) or parallelism.get("envs_per_gpu") not in (1,4)
                    or parallelism.get("protocol")!=("independent_cases_fixed_terrain_v2" if invariant else "independent_cases_shared_terrain_v1")):
                raise ValueError("BFM parallel evaluation profile differs")
            expected=eval_case_seeds(plan.get("suite_id"), plan.get("profile"))
            canonical=json.dumps({k:v for k,v in plan.items() if k!="hash"},sort_keys=True,separators=(",", ":"))
            if (plan.get("schema_version")!="eb/rpo-bfm-eval-plan/v1"
                    or plan.get("suite_spec")!=frozen_eval_suite_spec(plan.get("suite_id"),plan.get("profile")) or plan.get("codec_sha256")!=CODEC_HASH
                    or plan.get("hash")!="sha256:"+hashlib.sha256(canonical.encode()).hexdigest()
                    or spec.image!=plan.get("image_digest") or spec.resources.gpus!=(parallelism["gpus"] if parallel else 1)
                    or plan.get("seeds")!=expected or spec.params.get("seeds")!=expected
                    or plan.get("record_video") is not spec.params.get("record_video")
                    or plan.get("diagnostic_commands") is not (plan.get("profile")=="smoke")):
                raise ValueError("BFM independent evaluation plan, inputs or seeds differ")
            dataset=spec.inputs.datasets[0] if len(spec.inputs.datasets)==1 else {}
            if (dataset.get("validation") or {}).get("manifest_sha256")!=plan.get("motion_manifest_sha256"):
                raise ValueError("BFM evaluation needs the source training motion view")
            export_spec=spec.model_copy(update={"kind":JobKind.EXPORT_POLICY,"image":None,
                "params":{**spec.params,"source_run_id":plan["source_run_id"],"asset_sha256":plan["asset_sha256"]}})
            rendered=IsaacLabRslRlAdapter().render(export_spec,run_id)
            rendered.image=spec.image
            runner="rpo_bfm_parallel_eval.py" if parallel else "rpo_bfm_eval.py"
            rendered.command=[rendered.command[0],"/workspace/eb_runner/"+runner,*rendered.command[3:]]
            rendered.command[-1]="/workspace/output"
            rendered.command.extend(["--seeds",",".join(map(str,expected)),"--plan-hash",plan["hash"],
                "--video-quality",plan.get("video_quality","standard"),
                "--weights-sha256",plan["source_weights_sha256"],"--motion-sha256",plan["motion_manifest_sha256"],
                "--config-sha256",plan["source_config_sha256"]])
            if plan.get("record_video"):rendered.command.append("--record-video")
            if plan.get("diagnostic_commands"):rendered.command.append("--diagnostic-commands")
            if parallel:rendered.command.extend(["--eval-gpus",str(parallelism["gpus"]),"--envs-per-gpu",str(parallelism["envs_per_gpu"])])
            if invariant:rendered.command.append("--partition-invariant")
            assets=Path(spec.code_refs.embodiment.path) if spec.code_refs and spec.code_refs.embodiment else settings.assets_root/"roboparty/rpo-v2"
            asset_file=assets/"asset-manifest.json"
            if asset_file.is_symlink() or hashlib.sha256(asset_file.read_bytes()).hexdigest()!=plan["asset_sha256"]:
                raise ValueError("BFM evaluation asset hash changed")
            rendered.mounts.extend([(str(assets),"/workspace/rpo-assets","ro"),
                (str(assets/"rpo_description"),"/workspace/rpo-assets/robolab/data/robots/roboparty/rpo","ro")])
            rendered.env.update({"EB_RUN_ID":run_id,"RPO_ASSET_ROOT":"/workspace/rpo-assets"})
            return rendered
        if tpl.id in {"rpo_v2_wbc_eval", "rpo_v2_amp_eval", "rpo_v2_flat_eval"}:
            amp_eval = tpl.id == "rpo_v2_amp_eval"
            flat_eval = tpl.id == "rpo_v2_flat_eval"
            checkpoint = Path(spec.inputs.base_model or "")
            if checkpoint.is_dir():
                checkpoint /= "model_state_dict/full_weights.pt"
            assets = settings.assets_root / "roboparty/rpo-v2"
            if flat_eval and spec.code_refs and spec.code_refs.embodiment:
                assets = Path(spec.code_refs.embodiment.path)
            if not checkpoint.is_file() or not (assets / "asset-manifest.json").is_file():
                raise ValueError("RPO Isaac 评估必须绑定受管 checkpoint 和资产")
            seeds = spec.params.get("seeds", list(range(12)))
            if (not isinstance(seeds, list) or not 1 <= len(seeds) <= (100 if flat_eval else 12)
                    or any(type(seed) is not int or not 0 <= seed <= 1_000_000 for seed in seeds)
                    or len(seeds) != len(set(seeds))):
                raise ValueError("RPO Isaac seeds 必须为 1–12 个不重复的非负整数")
            plan = spec.params.get("rpo_eval_plan") or {}
            if flat_eval:
                from eb.rpo_flat import FLAT_SUITE, flat_suite_spec
                expected_seeds = list(range(60000, 60003) if plan.get("profile") == "smoke" else range(50000, 50100))
                if (plan.get("schema_version") != "eb/rpo-flat-eval-plan/v1"
                        or plan.get("suite_spec") != flat_suite_spec() or plan.get("suite_id") != FLAT_SUITE
                        or plan.get("backend") != "isaac" or seeds != plan.get("seeds")
                        or plan.get("video_quality", "standard") not in {"standard", "hd"}
                        or spec.params.get("video_quality", "standard") != plan.get("video_quality", "standard")
                        or plan.get("video_quality") == "hd" and not spec.params.get("record_video")
                        or not plan.get("seed_override") and seeds != expected_seeds):
                    raise ValueError("RPO-Flat 评估必须使用冻结的官方指令分布与案例计划")
            if not flat_eval and plan.get("implicit_seed_defaults") is True and plan.get("seed_override") is False:
                expected = list(range(3 if plan.get("profile") == "smoke" else 12))
                if seeds != expected:
                    raise ValueError("RPO Isaac 评估计划的默认 seed 与规模不一致")
            seed_args = (["--profile", plan["profile"]]
                         if (plan.get("implicit_seed_defaults") is True
                             and plan.get("seed_override") is False
                             and plan.get("profile") in ("smoke", "standard"))
                         else ["--seeds", ",".join(map(str, seeds))])
            run_root = settings.data_root / "runs" / run_id
            return RenderedJob(
                image=_image(spec, template_image(tpl)),
                command=["/workspace/isaaclab_extension_template/.venv/bin/python",
                         "/workspace/eb_runner/rpo_flat_artifact.py" if flat_eval else "/workspace/eb_runner/rpo_wbc_artifact.py", "eval-checkpoint-isaac",
                         *(["--recipe", "amp"] if amp_eval else []),
                         "--checkpoint", "/workspace/model/full_weights.pt",
                         "--output", "/workspace/output", *(["--seeds", ",".join(map(str, seeds)), "--plan-hash", plan["hash"]] if flat_eval else seed_args),
                         *(["--criterion", plan["criterion"]] if plan.get("criterion") in ("stability", "stand_walk") else []),
                         *rpo_eval_arguments(plan),
                         *(["--video-quality", "hd"] if flat_eval and plan.get("video_quality") == "hd" else []),
                         *(["--record-video"] if spec.params.get("record_video") else [])],
                env={"RPO_ASSET_ROOT":"/workspace/rpo-assets", "PYTHONUNBUFFERED":"1",
                     "EB_RUN_ID":run_id,
                     "PYTHONPATH":"/workspace/rpo-assets/robolab:/workspace/eb_runner:/workspace/rlinf"},
                writable_paths=["output"],
                mounts=[(str(checkpoint), "/workspace/model/full_weights.pt", "ro"),
                        (str(assets), "/workspace/rpo-assets", "ro"),
                        (str(assets / "rpo_description"),
                         "/workspace/rpo-assets/robolab/data/robots/roboparty/rpo", "ro"),
                        (str(run_root / "output"), "/workspace/output", "rw")],
                gpus=spec.resources.gpus or 1, shm_gb=spec.resources.shm_gb,
                workdir="/workspace", outputs=["output"], metrics_glob="",
            )
        overrides = build_overrides(tpl.id, spec.resources.gpus, spec.name or run_id, spec.overrides)
        eval_plan = spec.params.get("eval_plan")
        if eval_plan:
            if tpl.kind != "eval" or eval_plan.get("hash") is None:
                raise ValueError("评估计划与模板不兼容")
            overrides.extend(("env.eval.auto_reset=False", "env.eval.ignore_terminations=False"))
        run_root = settings.data_root / "runs" / run_id
        model_root = settings.models_root / "runs" / run_id
        mounts = [(str(run_root / "output"), "/workspace/output", "rw"),
                  (str(model_root), "/workspace/output/model", "rw"),
                  (str(run_root / "tb"), "/workspace/tb", "rw")]
        refs = spec.code_refs
        robotwin_eval = bool(eval_plan and eval_plan.get("schema_version") == "eb/robotwin-eval-plan/v1")
        if refs and refs.embodiment and not robotwin_eval:
            mounts.append((refs.embodiment.path, "/workspace/berkeley", "ro"))
        stack_source = refs.stack.path if refs and refs.stack else str(settings.rlinf_repo)
        image_owned_rlinf = robotwin_eval or tpl.id.startswith(("rpo_v2_wbc_", "rpo_v2_amp_", "rpo_v2_flat_", "rpo_v2_bfm_zero_", "rpo_bfm_ufo_"))
        mounts.append((stack_source,
                       "/workspace/source-receipts/rlinf" if image_owned_rlinf else "/workspace/rlinf",
                       "ro"))
        if tpl.id.startswith(("rpo_v2_wbc_", "rpo_v2_amp_", "rpo_v2_flat_", "rpo_v2_bfm_zero_", "rpo_bfm_ufo_")):
            asset_root = settings.assets_root / "roboparty/rpo-v2"
            if (native_ufo or tpl.id in {"rpo_v2_flat_ppo","rpo_v2_bfm_zero_pretrain"}) and refs and refs.embodiment:
                asset_root = Path(refs.embodiment.path)
            required = (
                "robolab/robolab/tasks/direct/base/base_env.py",
                "rpo_description/urdf/rpo.urdf",
                "rpo_description/mjcf/rpo.xml",
            )
            if asset_root.is_symlink() or any(not (asset_root / path).is_file() for path in required):
                raise ValueError("RPO V2 受管资产尚未落地或不完整")
            try:
                asset_manifest = json.loads((asset_root / "asset-manifest.json").read_text())
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError("RPO V2 资产缺少校验 manifest") from exc
            if asset_manifest.get("status") != "verified":
                raise ValueError("RPO V2 资产校验未通过")
            if tpl.id == "rpo_v2_flat_ppo":
                from eb.rpo_flat import ROBOLAB_REVISION
                if asset_manifest.get("revisions", {}).get("robolab") != ROBOLAB_REVISION:
                    raise ValueError("RPO-Flat 必须使用固定的 RoboLab 6b1c3d9 资产版本")
            mounts.append((str(asset_root), "/workspace/rpo-assets", "ro"))
            mounts.append((str(asset_root / "rpo_description"),
                           "/workspace/rpo-assets/robolab/data/robots/roboparty/rpo", "ro"))
            if native_ufo:
                from eb.runners.rpo_ufo_motion import validate_robot_state_manifest
                from eb.rpo_bfm import CODEC_HASH, codec_hash_for_execution
                codec_sha256 = codec_hash_for_execution(binding.model_dump())
                state_scope = "rpo_ufo_robot_state_v1" if codec_sha256 == CODEC_HASH else "rpo_ufo_robot_state_v2"
                dataset = spec.inputs.datasets[0]
                motion_root = Path(dataset["path"])
                manifest = motion_root / "manifest.json"
                validation = dataset.get("validation") or {}
                asset_sha256 = hashlib.sha256((asset_root / "asset-manifest.json").read_bytes()).hexdigest()
                if (motion_root.is_symlink() or manifest.is_symlink() or not manifest.is_file()
                        or validation.get("scope") != state_scope
                        or validation.get("status") != "validated"
                        or hashlib.sha256(manifest.read_bytes()).hexdigest() != validation.get("manifest_sha256")
                        or asset_sha256 != validation.get("asset_sha256")):
                    raise ValueError("UFO raw state DatasetVersion/asset changed after validation")
                validate_robot_state_manifest(json.loads(manifest.read_text()), asset_sha256=asset_sha256,
                                              expected_codec_sha256=codec_sha256)
                overrides.extend((f"actor.model.motion_root=/workspace/data/{dataset['id']}",
                    f"actor.model.asset_sha256={asset_sha256}"))
                if codec_sha256 != CODEC_HASH:
                    overrides.append(f"+actor.model.codec_sha256={codec_sha256}")
            if tpl.id == "rpo_v2_amp_stand_walk_ppo":
                dataset = spec.inputs.datasets[0]
                validation = dataset.get("validation") or {}
                manifest = Path(str(dataset.get("path", ""))) / "manifest.json"
                asset_sha256 = hashlib.sha256((asset_root / "asset-manifest.json").read_bytes()).hexdigest()
                if (dataset.get("purpose") != "motion_prior"
                        or validation.get("scope") != "rpo_amp_motion_v1"
                        or validation.get("status") != "validated"
                        or validation.get("training_ready") is not True
                        or validation.get("asset_sha256") != asset_sha256
                        or manifest.is_symlink() or not manifest.is_file()
                        or hashlib.sha256(manifest.read_bytes()).hexdigest() != validation.get("manifest_sha256")):
                    raise ValueError("AMP motion DatasetVersion or RPO asset changed after validation")
                overrides.extend((f"motion_prior.manifest_path=/workspace/data/{dataset['id']}/manifest.json",
                                  f"motion_prior.asset_sha256={asset_sha256}"))
            if tpl.id == "rpo_v2_bfm_zero_pretrain":
                from eb.rpo_bfm import validate_motion_manifest, REFERENCE_POOL_CONVERTER_REVISION, UNCHECKED_REFERENCE_REVISION,G1_CONVERTER_REVISION,G1_IMAGE
                dataset = spec.inputs.datasets[0]
                validation = dataset.get("validation") or {}
                motion_root = Path(str(dataset.get("path", "")))
                manifest = motion_root / "manifest.json"
                asset_sha256 = hashlib.sha256((asset_root / "asset-manifest.json").read_bytes()).hexdigest()
                if (dataset.get("purpose") != "behavior_prior"
                        or validation.get("scope") != "rpo_bfm_motion_v1"
                        or validation.get("status") != "validated"
                        or validation.get("smoke_ready") is not True
                        or validation.get("storage_ref") != str(motion_root)
                        or validation.get("asset_sha256") != asset_sha256
                        or manifest.is_symlink() or not manifest.is_file()
                        or hashlib.sha256(manifest.read_bytes()).hexdigest() != validation.get("manifest_sha256")):
                    raise ValueError("BFM motion DatasetVersion or RPO asset changed after validation")
                current = validate_motion_manifest(json.loads(manifest.read_text()), asset_sha256=asset_sha256)
                if (json.loads(manifest.read_text()).get("converter_revision")==G1_CONVERTER_REVISION
                        and spec.image not in (None,G1_IMAGE)):
                    raise ValueError("G1 BFM view needs the calibrated G1 codec image")
                if (json.loads(manifest.read_text()).get("converter_revision")
                        in {REFERENCE_POOL_CONVERTER_REVISION, UNCHECKED_REFERENCE_REVISION}
                        and spec.image == "eb/rlinf:rpo-v2-bfm-zero"):
                    raise ValueError("BFM reference motion needs the versioned reference image")
                if not current["smoke_ready"]:
                    raise ValueError("BFM legacy source quaternion conversion is incorrect; derive a new DatasetVersion")
                from eb.rpo_bfm import learner_updates, validate_training_budget
                values = dict(item.split("=", 1) for item in overrides if "=" in item)
                per_round = int(values.get("actor.model.updates_per_round", 1))
                updates = learner_updates(int(values["runner.max_epochs"]), per_round)
                purpose=validate_training_budget(updates,current,
                    start_update=int(spec.params.get("resume_step", 0)) * per_round)
                if spec.params.get("bfm_update_target", updates) != updates:
                    raise ValueError("BFM frozen learner target differs from runner rounds")
                if validation.get("training_ready")!=current["training_ready"]:
                    raise ValueError("BFM frozen training qualification differs from current manifest")
                if spec.params.get("bfm_experiment_purpose",purpose)!=purpose:
                    raise ValueError("BFM frozen experiment purpose differs from its data/budget")
                overrides.extend((f"actor.model.motion_root=/workspace/data/{dataset['id']}",
                                  f"actor.model.asset_sha256={asset_sha256}"))
            if tracking:
                from eb.runners.rpo_tracking_reference import reference_receipt
                binding = spec.params.get("tracking_reference") or {}
                source_id = str(binding.get("run_id", ""))
                if not re.fullmatch(r"run_[0-9a-f]+", source_id):
                    raise ValueError("Tracking teacher requires a frozen retarget source Run")
                reference_root = settings.data_root / "runs" / source_id / "outputs/retargeted"
                asset_sha256 = hashlib.sha256((asset_root / "asset-manifest.json").read_bytes()).hexdigest()
                reference_receipt(reference_root, binding.get("receipt_sha256"), asset_sha256)
                mounts.append((str(reference_root), "/workspace/tracking-reference", "ro"))
        if eval_plan and eval_plan.get("schema_version") == "eb/libero-eval-plan/v1":
            dataset_id = str(spec.params.get("benchmark_dataset_id", ""))
            digest = spec.params.get("benchmark_sha256")
            if (not re.fullmatch(r"ds_[0-9a-f]+", dataset_id)
                    or not re.fullmatch(r"[0-9a-f]{64}", str(digest))
                    or eval_plan.get("benchmark", {}).get("dataset_id") != dataset_id
                    or eval_plan["benchmark"].get("sha256") != digest):
                raise ValueError("LIBERO 测试数据绑定无效")
            benchmark_root = settings.datasets_root / dataset_id / "bundle"
            package = "/workspace/isaaclab_extension_template/.venv/lib/python3.11/site-packages/libero/libero"
            if benchmark_root.is_symlink() or not (benchmark_root / "benchmark-manifest.json").is_file():
                raise ValueError("LIBERO 测试数据尚未落地")
            mounts.append((str(benchmark_root), "/workspace/libero-benchmark", "ro"))
            for folder in ("assets", "bddl_files", "init_files"):
                source = benchmark_root / folder
                if source.is_symlink() or not source.is_dir():
                    raise ValueError(f"LIBERO 测试数据缺少 {folder}")
                mounts.append((str(source), f"{package}/{folder}", "ro"))
        elif eval_plan and eval_plan.get("schema_version") == "eb/robotwin-eval-plan/v1":
            robotwin_ref = refs.embodiment if refs and refs.embodiment else None
            robotwin = Path(robotwin_ref.path) if robotwin_ref else settings.robotwin_repo
            assets = settings.robotwin_assets_root
            suite_spec = eval_plan.get("suite_spec") or {}
            runtime = (eval_plan.get("model") or {}).get("runtime") or {}
            from eb.runners.robotwin_source import PATCH_SHA256, PATCHED_BASE_TASK_SHA256
            if (runtime.get("robotwin_evidence_patch_sha256") != PATCH_SHA256
                    or runtime.get("patched_base_task_sha256") != PATCHED_BASE_TASK_SHA256):
                raise ValueError("RoboTwin 精确动作证据 patch 未冻结")
            if (robotwin.is_symlink() or not (robotwin / "assets").is_dir()
                    or not (robotwin / "robotwin/envs/vector_env.py").is_file()):
                raise ValueError("RoboTwin RLInf_support 源码或资产尚未就绪")
            expected_robotwin = runtime.get("robotwin_content_hash")
            if robotwin_ref and robotwin_ref.kind == "content_snapshot":
                if not expected_robotwin or robotwin_ref.commit != expected_robotwin:
                    raise ValueError("RoboTwin 内容快照与冻结评估计划不一致")
            elif _git_head(robotwin) != suite_spec.get("robotwin_revision"):
                raise ValueError("RoboTwin 源码 revision 与冻结 Suite 不一致")
            stack_ref = refs.stack if refs and refs.stack else None
            stack = stack_ref.path if stack_ref else settings.rlinf_repo
            if stack_ref and stack_ref.kind == "content_snapshot":
                expected = (eval_plan.get("model", {}).get("runtime", {})
                            .get("rlinf_content_hash"))
                if not expected or stack_ref.commit != expected:
                    raise ValueError("RLInf 内容快照与冻结评估计划不一致")
            elif _git_head(Path(stack)) != suite_spec.get("rlinf_revision"):
                raise ValueError("RLInf 源码 revision 与冻结 Suite 不一致")
            try:
                asset_receipt = json.loads((assets / "asset-manifest.json").read_text())
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError("RoboTwin 官方资产尚未完成验证") from exc
            from eb.robotwin_suite import asset_manifest_sha256
            if (assets.is_symlink() or not (assets / "embodiments/aloha-agilex").is_dir()
                    or not (assets / "objects").is_dir()
                    or asset_receipt.get("status") != "verified"
                    or asset_receipt.get("sha256") != asset_manifest_sha256(asset_receipt)
                    or asset_receipt.get("sha256") != suite_spec.get("assets_manifest_sha256")):
                raise ValueError("RoboTwin 资产与冻结 Suite 不一致")
            if (eval_plan.get("planned") != len(eval_plan.get("cases") or [])
                    or not re.fullmatch(r"[0-9A-Za-z._-]+", str(suite_spec.get("robotwin_revision", "")))):
                raise ValueError("RoboTwin 评估计划未冻结")
            mounts.append((str(robotwin), "/workspace/robotwin", "ro"))
            mounts.append((str(assets), "/workspace/robotwin/assets", "ro"))
            overrides.extend((f"env.eval.total_num_envs={int(runtime['parallel_envs'])}",
                f"env.eval.max_episode_steps={int(suite_spec['max_episode_steps'])}",
                f"env.eval.max_steps_per_rollout_epoch={int(suite_spec['max_episode_steps'])}",
                f"rollout.inference_micro_batch_size={int(runtime['inference_micro_batch_size'])}",
                f"rollout.max_inflight_batches={int(runtime['max_inflight_batches'])}",
                "env.eval.rollout_epoch=1", "env.eval.use_fixed_reset_state_ids=True",
                "env.eval.seeds_path=/workspace/output/robotwin-eval-seeds.json",
                "env.eval.assets_path=/workspace/robotwin"))
        elif eval_plan:
            raise ValueError("未知评估计划协议")
        for ds in spec.inputs.datasets:
            dsid = str(ds["id"])
            if not re.fullmatch(r"[A-Za-z0-9_-]+", dsid):
                raise ValueError("非法数据版本 ID")
            mounts.append((str(ds["path"]), f"/workspace/data/{dsid}", "ro"))
        if spec.inputs.datasets and tpl.kind == "sft":
            overrides.append(f"data.train_data_paths=/workspace/data/{spec.inputs.datasets[0]['id']}")
        if spec.inputs.base_model:
            mounts.append((spec.inputs.base_model, "/workspace/models/base", "ro"))
            overrides.append(("rollout" if tpl.kind == "eval" else "actor")
                             + ".model.model_path=/workspace/models/base")
            if flat_warm_start:
                weight = Path(spec.inputs.base_model) / "model_state_dict/full_weights.pt"
                if weight.is_symlink() or not weight.is_file():
                    raise ValueError("Flat warm start 缺少完整已发布权重")
                overrides.append("runner.ckpt_path=/workspace/models/base/model_state_dict/full_weights.pt")
            if tpl.kind == "eval" and tpl.id.startswith("robotwin_"):
                overrides.append("runner.ckpt_path=/workspace/models/base/model_state_dict/full_weights.pt")
            if tpl.id == "robotwin_lingbot_vla2_sft":
                tokenizer = settings.models_root / "Qwen3-VL-4B-Instruct"
                if not (tokenizer / "config.json").is_file() or not (tokenizer / "tokenizer.json").is_file():
                    raise ValueError("LingBot-VLA-2 缺少 Qwen3-VL-4B-Instruct tokenizer/config 依赖")
                mounts.append((str(tokenizer), "/workspace/models/qwen3-vl", "ro"))
            elif tpl.id == "robotwin_gr00t_n17_sft":
                cosmos = settings.models_root / "nvidia/Cosmos-Reason2-2B"
                if not (cosmos / "config.json").is_file() or not list(cosmos.glob("*.safetensors")):
                    raise ValueError("GR00T-N1.7 缺少本地 Cosmos-Reason2-2B 依赖")
                mounts.append((str(cosmos), "/workspace/models/cosmos", "ro"))
            elif tpl.id == "robotwin_xr1_sft":
                processor = settings.models_root / "Qwen/Qwen3-VL-4B-Instruct"
                if not (processor / "config.json").is_file() or not (processor / "tokenizer.json").is_file():
                    raise ValueError("Xiaomi-Robotics-1 缺少本地 Qwen3-VL-4B-Instruct processor")
                mounts.append((str(processor), "/workspace/models/qwen3-vl", "ro"))
            if tpl.kind in {"ppo", "eval"}:
                value = "rollout.model.model_path=/workspace/models/base"
                if value not in overrides:
                    overrides.append(value)
        checkpoint_run = spec.params.get("eval_checkpoint_run_id")
        if checkpoint_run:
            if tpl.kind != "eval" or not tpl.id.startswith("libero_") or not re.fullmatch(r"run_[A-Za-z0-9_-]+", str(checkpoint_run)):
                raise ValueError("评估 checkpoint 的模板或源 Run ID 无效")
            checkpoint = checkpoint_weights(settings.data_root / "runs" / checkpoint_run,
                                            spec.params.get("eval_checkpoint_step"))
            if checkpoint["bytes"] != spec.params.get("eval_checkpoint_bytes"):
                raise ValueError("评估 checkpoint 大小已变更")
            digest = spec.params.get("eval_checkpoint_sha256")
            if digest:
                if not re.fullmatch(r"[0-9a-f]{64}", str(digest)):
                    raise ValueError("冻结 checkpoint 摘要无效")
                frozen = settings.data_root / "cas" / digest[:2] / digest
                if frozen.is_symlink() or not frozen.is_file() or frozen.stat().st_size != checkpoint["bytes"]:
                    raise ValueError("冻结 checkpoint 缺失或大小变化")
                checkpoint["path"] = str(frozen.resolve())
            elif eval_plan:
                raise ValueError("新评估必须使用冻结 checkpoint")
            mounts.append((checkpoint["path"], "/workspace/checkpoint/full_weights.pt", "ro"))
            overrides.append("runner.ckpt_path=/workspace/checkpoint/full_weights.pt")
            if spec.params.get("eval_source_kind") == "sft":
                overrides.extend(("actor.model.add_value_head=False", "actor.model.openpi.add_value_head=False"))
                source_defaults = spec.params.get("eval_source_inference") or dict(
                    item.split("=", 1) for item in get_template("libero_pi05_sft").default_overrides)
                for key in ("actor.model.num_action_chunks", "actor.model.num_steps"):
                    overrides.append(f"{key}={source_defaults[key]}")
            elif spec.params.get("eval_source_inference"):
                for key, value in spec.params["eval_source_inference"].items():
                    overrides.append(f"{key}={value}")
        fingerprint = resume_fingerprint(spec)
        resume_receipt = None
        if spec.inputs.resume_from:
            if not re.fullmatch(r"run_[A-Za-z0-9_-]+", spec.inputs.resume_from):
                raise ValueError("非法的源 Run ID")
            if not tpl.id.startswith(("libero_", "rpo_v2_wbc_", "rpo_v2_amp_", "rpo_v2_flat_", "rpo_v2_bfm_zero_", "rpo_bfm_ufo_")) or tpl.kind not in {"sft", "ppo"}:
                raise ValueError("该 RLInf 模板尚未验证 checkpoint 恢复")
            checkpoint = resume_checkpoint(
                settings.data_root / "runs" / spec.inputs.resume_from, spec.inputs.resume_from,
                step=spec.params.get("resume_step"),
            )
            require_flat_rank_rng(spec, checkpoint)
            if tpl.id == "rpo_v2_amp_stand_walk_ppo" and "amp/state.pt" not in {
                    item["path"] for item in checkpoint["files"]}:
                raise ValueError("AMP checkpoint lacks discriminator, optimizer or sampler state")
            if tpl.id == "rpo_v2_bfm_zero_pretrain" and "bfm/state.pt" not in {
                    item["path"] for item in checkpoint["files"]}:
                raise ValueError("BFM checkpoint lacks replay, prompt RNG or expert sampler state")
            if checkpoint["compatibility_fingerprint"] != fingerprint:
                raise ValueError("checkpoint 与恢复任务的训练配置不兼容")
            if (spec.params.get("resume_mode") != checkpoint["mode"]
                    or spec.params.get("resume_step") != checkpoint["step"]):
                raise ValueError("checkpoint 与恢复任务回执不一致")
            if binding and binding.recipe_id=="rpo_bfm_ufo_agent_rlinf_isaac_4gpu_compliance_v11" and checkpoint['step']==524:
                from eb.runners.rpo_ufo_compliance import require_long_training_gate
                require_long_training_gate(Path(checkpoint['path']),step=checkpoint['step'])
            destination = f"/workspace/resume/global_step_{checkpoint['step']}"
            mounts.append((checkpoint["path"], destination, "ro"))
            overrides.append(f"runner.resume_dir={destination}")
            resume_receipt = {
                "schema_version": "eb/resume-request/v1", "status": "validated",
                "source_run_id": spec.inputs.resume_from, "step": checkpoint["step"],
                "mode": checkpoint["mode"], "checkpoint": destination,
                "compatibility_fingerprint": fingerprint,
            }
        entrypoint = template_entrypoint(tpl)
        command = ["/workspace/isaaclab_extension_template/.venv/bin/python", "/workspace/eb_runner/rlinf_entry.py",
                   "--script", entrypoint, "--config-path", f"/workspace/rlinf/{tpl.config_path}",
                   "--config-name", tpl.config_name, *overrides]
        if not tpl.id.startswith(("libero_", "robotwin_", "rpo_v2_wbc_", "rpo_v2_amp_", "rpo_v2_flat_", "rpo_v2_bfm_zero_", "rpo_bfm_ufo_")):
            command = [command[0], entrypoint, "--config-path", f"/workspace/rlinf/{tpl.config_path}",
                       "--config-name", tpl.config_name, *overrides]
        pythonpath = "/workspace/berkeley:/workspace/rlinf:/workspace/eb_runner:/opt/eb-openpi"
        if tpl.id.startswith(("rpo_v2_wbc_", "rpo_v2_amp_", "rpo_v2_flat_", "rpo_v2_bfm_zero_", "rpo_bfm_ufo_")):
            pythonpath = ("/workspace/isaaclab/source/isaaclab:"
                          "/workspace/isaaclab/source/isaaclab_tasks:"
                          "/workspace/rpo-assets/robolab:" + pythonpath)
        if eval_plan and eval_plan.get("schema_version") == "eb/robotwin-eval-plan/v1":
            pythonpath = "/workspace/robotwin:" + pythonpath
        if tpl.id == "robotwin_dm05_sft":
            pythonpath += ":/opt/opendm"
        elif tpl.id == "robotwin_lingbot_vla2_sft":
            pythonpath += ":/opt/lingbot-vla2"
        elif tpl.id == "robotwin_gr00t_n17_sft":
            pythonpath += ":/opt/isaac-gr00t"
        elif tpl.id == "robotwin_xr1_sft":
            pythonpath += ":/opt/xr1"
        env = {"NVIDIA_DRIVER_CAPABILITIES":"compute,utility,graphics", "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl", "PYTHONUNBUFFERED": "1",
               "PYTHONDONTWRITEBYTECODE": "1", "EB_RUN_ID": run_id, "EB_OUTPUT_DIR": "/workspace/output", "EMBODIED_PATH": "/workspace/rlinf/examples/embodiment",
               "PYTHONPATH": pythonpath,
               "HF_HOME": "/workspace/output/cache/huggingface", "HF_HUB_OFFLINE": "1",
               "EB_INPUT_RECEIPT": json.dumps({"template_id": tpl.id, "inputs": spec.inputs.model_dump()}, ensure_ascii=False),
               "EB_RESUME_FINGERPRINT": fingerprint}
        if native_ufo:
            from eb.runners.bfm_ufo_agent import validate_source_receipt
            source = validate_source_receipt(spec.params.get("source_receipt") or {})
            receipt = json.loads(env["EB_INPUT_RECEIPT"])
            receipt.update(execution=binding.model_dump(), source_receipt=source,
                target_learner_updates=spec.params["target_learner_updates"], purpose=spec.params["purpose"],
                environment_profile=spec.params.get("environment_profile"), asset_sha256=asset_sha256)
            env.update(EB_INPUT_RECEIPT=json.dumps(receipt),
                EB_UFO_TARGET_UPDATES=str(spec.params["target_learner_updates"]),
                EB_UFO_MOTION_ROOT=f"/workspace/data/{dataset['id']}", EB_UFO_ASSET_SHA256=asset_sha256)
            env["PYTHONPATH"] = "/opt/ufo:" + env["PYTHONPATH"]
            if binding.recipe_id in {"rpo_bfm_ufo_agent_rlinf_isaac_4gpu_safe_v2", "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_stable_v3"}:
                env["EB_UFO_LEARNING_PROFILE"] = "physical_limits_v2"
            if binding.recipe_id == "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_slew_v4":
                env["EB_UFO_LEARNING_PROFILE"] = "physical_limits_slew_v4"
            if binding.recipe_id in {"rpo_bfm_ufo_agent_rlinf_isaac_4gpu_slew_velocity_v5", "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_scale6144_v6"}:
                env.update(EB_UFO_LEARNING_PROFILE="physical_limits_slew_v4", EB_UFO_AUXILIARY_PROFILE="soft_velocity_v5")
            if binding.recipe_id == "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_scale6144_v6":
                env["EB_UFO_WALL_SECONDS"] = str(int(spec.policy.timeout_h * 3600) - 300)
            if binding.recipe_id in {"rpo_bfm_ufo_agent_rlinf_isaac_4gpu_reachable_v7", "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_contact_v8", "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_ground_v9"}:
                env.update(EB_UFO_LEARNING_PROFILE="physical_limits_slew_v4", EB_UFO_AUXILIARY_PROFILE="soft_velocity_v7",
                           EB_UFO_CONTROL_PROFILE="reachable_v2", EB_UFO_WALL_SECONDS=str(int(spec.policy.timeout_h * 3600) - 300))
            if binding.recipe_id in {"rpo_bfm_ufo_agent_rlinf_isaac_4gpu_contact_v8", "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_ground_v9"}:
                env.update(EB_UFO_CONTACT_PROFILE="contact_stable_v8", EB_UFO_DIAGNOSTICS_DIR="/workspace/output/contact-pools")
            if binding.recipe_id == "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_ground_v9":
                env["EB_UFO_GROUND_PROFILE"] = "wide_plane_v9"
            if binding.recipe_id in {"rpo_bfm_ufo_agent_rlinf_isaac_4gpu_compliance_v11","rpo_bfm_ufo_agent_rlinf_isaac_4gpu_tracking_contact_v10","rpo_bfm_ufo_agent_rlinf_isaac_4gpu_tracking_contact_kd_v10"}:
                env.update(EB_UFO_LEARNING_PROFILE="physical_limits_slew_v4", EB_UFO_AUXILIARY_PROFILE="soft_velocity_v7",
                    EB_UFO_CONTROL_PROFILE="reachable_v2",EB_UFO_CONTACT_PROFILE="contact_stable_v8",
                    EB_UFO_GROUND_PROFILE="wide_plane_v9",EB_UFO_CONTEXT_PROFILE="environment_tracking_v10",
                    EB_UFO_ANKLE_CONTROL="lpf15",EB_UFO_DIAGNOSTICS_DIR="/workspace/output/contact-pools",
                    EB_UFO_WALL_SECONDS=str(int(spec.policy.timeout_h*3600)-300))
                if binding.recipe_id == "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_tracking_contact_kd_v10":
                    env["EB_UFO_ANKLE_CONTROL"]="kd25"
            if binding.recipe_id=="rpo_bfm_ufo_agent_rlinf_isaac_4gpu_compliance_v11":
                env.update(EB_UFO_DYNAMICS_PROFILE="contact_compliance_v11",EB_UFO_ANKLE_CONTROL="kd25")
            if binding.simulator_id == "mjlab":
                env.update(EB_UFO_LEARNING_PROFILE="physical_limits_slew_v4", EB_UFO_AUXILIARY_PROFILE="soft_velocity_v7",
                    EB_UFO_CONTROL_PROFILE="reachable_v2", EB_UFO_CONTEXT_PROFILE="environment_tracking_v10",
                    EB_UFO_PHYSICS_PROFILE="native_mjcf_viability_v13" if binding.recipe_id.endswith("viability_v13") else "native_mjcf_v12", EB_UFO_DIAGNOSTICS_DIR="/workspace/output/contact-pools",
                    EB_UFO_WALL_SECONDS=str(int(spec.policy.timeout_h * 3600) - 300),
                    WARP_CACHE_PATH="/workspace/output/cache/warp")
        if flat_warm_start:
            env["EB_WARM_START_RECEIPT"] = json.dumps(flat_warm_start, ensure_ascii=False)
        if tracking:
            env.update({"EB_TRACKING_REFERENCE_ROOT":"/workspace/tracking-reference",
                        "EB_TRACKING_REFERENCE_SHA256":binding["receipt_sha256"],
                        "EB_TRACKING_ASSET_SHA256":asset_sha256})
            receipt = json.loads(env["EB_INPUT_RECEIPT"])
            receipt["tracking_reference"] = binding
            env["EB_INPUT_RECEIPT"] = json.dumps(receipt)
        if tpl.id.startswith(("rpo_v2_wbc_", "rpo_v2_amp_", "rpo_v2_flat_", "rpo_v2_bfm_zero_", "rpo_bfm_ufo_")):
            env["RPO_ASSET_ROOT"] = "/workspace/rpo-assets"
        if tpl.id == "rpo_v2_flat_ppo":
            from eb.rpo_flat import ROBOLAB_REVISION
            receipt = json.loads(env["EB_INPUT_RECEIPT"])
            receipt["official_source"] = {"robolab_revision":ROBOLAB_REVISION,
                "asset_manifest_sha256":hashlib.sha256((asset_root / "asset-manifest.json").read_bytes()).hexdigest(),
                "asset_snapshot":refs.embodiment.model_dump() if refs and refs.embodiment else None,
                "training_environment":"RPOFlatEnvCfg", "upstream_task":"RPO-Flat"}
            env["EB_INPUT_RECEIPT"] = json.dumps(receipt, ensure_ascii=False)
        if tpl.id == "rpo_v2_wbc_stand_walk_ppo":
            from eb.domain.environment_profile import TRAIN_SETTINGS
            for key, value in (spec.params.get("rpo_train_settings") or {}).items():
                if key not in TRAIN_SETTINGS or not isinstance(value, (int, float)) or isinstance(value, bool) or not TRAIN_SETTINGS[key][0] <= value <= TRAIN_SETTINGS[key][1]:
                    raise ValueError("RPO 环境参数无效")
                env["EB_RPO_" + key.upper()] = str(value)
        if tpl.id == "rpo_v2_amp_stand_walk_ppo" and spec.params.get("rpo_amp_initial_pose"):
            if spec.params["rpo_amp_initial_pose"] != "official_a1_stand_v1":
                raise ValueError("不支持的 RPO AMP 初始姿态")
            env["EB_RPO_AMP_INITIAL_POSE"] = spec.params["rpo_amp_initial_pose"]
        if tpl.id == "rpo_v2_amp_stand_walk_ppo" and spec.params.get("rpo_amp_initial_logstd") is not None:
            value = spec.params["rpo_amp_initial_logstd"]
            if type(value) not in (int, float) or not -4.0 <= value <= 0.0:
                raise ValueError("RPO AMP 初始探索方差无效")
            env["EB_RPO_AMP_INITIAL_LOGSTD"] = str(value)
        if tpl.id == "rpo_v2_amp_stand_walk_ppo" and spec.params.get("rpo_amp_velocity_tracking_std") is not None:
            value = spec.params["rpo_amp_velocity_tracking_std"]
            if type(value) not in (int, float) or not math.isfinite(value) or not 0.05 <= value <= 0.5:
                raise ValueError("RPO AMP 速度奖励容差无效")
            env["EB_RPO_VELOCITY_TRACKING_STD"] = str(value)
        if tpl.id == "rpo_v2_amp_stand_walk_ppo" and spec.params.get("rpo_amp_velocity_reward_weight") is not None:
            value = spec.params["rpo_amp_velocity_reward_weight"]
            if type(value) not in (int, float) or not math.isfinite(value) or not 0.0 <= value <= 10.0:
                raise ValueError("RPO AMP 速度奖励权重无效")
            env["EB_RPO_VELOCITY_REWARD_WEIGHT"] = str(value)
        if tpl.id == "rpo_v2_amp_stand_walk_ppo" and spec.params.get("rpo_amp_actuator_delay_steps") is not None:
            value = spec.params["rpo_amp_actuator_delay_steps"]
            if type(value) is not int or not 0 <= value <= 2:
                raise ValueError("RPO AMP 执行器延迟无效")
            env["EB_RPO_AMP_ACTUATOR_DELAY_STEPS"] = str(value)
        if resume_receipt:
            env["EB_RESUME_RECEIPT"] = json.dumps(resume_receipt, ensure_ascii=False)
        if eval_plan:
            env.update(_eval_plan_env(eval_plan))
            if eval_plan.get("schema_version") == "eb/libero-eval-plan/v1":
                env["EB_LIBERO_BENCHMARK_SHA256"] = str(spec.params["benchmark_sha256"])
            else:
                env.update({"ROBOTWIN_PATH":"/workspace/robotwin", "ROBOT_PLATFORM":"ALOHA",
                            "EB_ROBOTWIN_REVISION":str(eval_plan["suite_spec"]["robotwin_revision"]),
                            "EB_RLINF_CONTENT_HASH":str(eval_plan["model"]["runtime"]["rlinf_content_hash"]),
                            "EB_RLINF_PATCH_SHA256":RLINF_PATCH_SHA256})
        from eb.domain.training_recipe import recipe_image_digest
        return RenderedJob(image=recipe_image_digest(tpl.id) if native_ufo else _image(spec, template_image(tpl)), command=command, env=env,
            writable_paths=["output", "tb"], mounts=mounts, gpus=spec.resources.gpus,
            shm_gb=spec.resources.shm_gb, workdir="/workspace/rlinf", outputs=["output"])
