from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from eb.adapters import equivalent_command, get_adapter
from eb.capacity import estimate_job
from eb.domain.contract import EmbodimentEnv
from eb.domain.fsm import JobStatus, RunStatus, transition
from eb.domain.job import CONTAINERLESS_KINDS, CodeRefs, GitCodeRef, JobKind, JobSpec
from eb.domain.task import TaskSpec
from eb.gates.engine import GateContext, g1_submit
from eb.ids import new_id
from eb.settings import get_settings
from eb.store.ebd import EbdClient
from eb.store.models import EmbodimentEnvRow, EmbodimentRow, Job, LineageEdge, Run, TaskRow
from eb.worktree import overlay_mounts, prepare_run_dirs, prepare_worktree, seed_rsl_resume

log = logging.getLogger(__name__)


async def validate_flat_warm_start_submission(session, spec):
    from eb.rpo_flat import flat_warm_start_receipt
    from eb.store.models import ModelVersion, Artifact

    receipt = spec.params.get("warm_start")
    if spec.params.get("template_id") != "rpo_v2_flat_ppo" or not receipt:
        return
    if not isinstance(receipt, dict):
        raise ValueError("Flat warm start 回执无效")
    model = await session.get(ModelVersion, receipt.get("model_id"))
    artifact = await session.get(Artifact, model.artifact_id) if model else None
    if (not model or not artifact or receipt != flat_warm_start_receipt(model, artifact)
            or not spec.inputs.base_model
            or Path(spec.inputs.base_model).resolve() != Path(artifact.storage).resolve()):
        raise ValueError("Flat warm start 输入与已发布模型不一致")


def _stack_mount_destination(mounts: list[tuple[str, str, str]]) -> str:
    if any(destination == "/sources/rlinf" for _, destination, _ in mounts):
        return "/sources/rlinf"
    receipt = "/workspace/source-receipts/rlinf"
    return receipt if any(destination == receipt for _, destination, _ in mounts) else "/workspace/rlinf"


def _is_snapshot_source(source: dict | None) -> bool:
    return (source or {}).get("kind") in {"git_external", "git_snapshot"}


async def _snapshot_ref(ebd: EbdClient, path: Path, commit: str | None) -> GitCodeRef:
    snapshot = await ebd.snapshot(str(path), commit)
    return GitCodeRef(kind="content_snapshot", path=snapshot["path"], commit=snapshot["content_hash"])


def validate_mounts(mounts):
    from eb.paths import default_prefixes, validate_allowed_path
    settings = get_settings()
    writable_roots = tuple(root.resolve() for root in (settings.data_root, settings.models_root, settings.datasets_root))
    for source, destination, mode in mounts:
        path = validate_allowed_path(source, default_prefixes())
        if mode not in {"ro", "rw"} or not destination.startswith("/") or ":" in source or ":" in destination:
            raise ValueError("非法容器挂载")
        if mode == "rw" and not any(path.is_relative_to(root) for root in writable_roots):
            raise ValueError("写挂载必须属于平台受管存储目录")


def apply_ebd_result(job, run, result: dict) -> None:
    """Map ebd submit_run JSON onto Job/Run. Never invent success or metrics."""
    status = result.get("status")
    if status == "started":
        run.container_id = result["container_id"]
        run.status = RunStatus.RUNNING.value
        job.status = JobStatus.RUNNING.value
        return
    run.status = RunStatus.ABORTED.value
    run.finished_at = datetime.now(UTC)
    run.abort_reason = status or "docker_error"
    job.status = JobStatus.INFRA_FAILED.value
    job.error = result.get("reason") or status
    run.metrics_summary = {}


async def validate_libero_eval_submission(session: AsyncSession, spec: JobSpec) -> None:
    """Apply the same immutable-plan boundary to /jobs, pipelines and the eval UI."""
    import hashlib
    import json
    import re
    from eb.checkpoints import checkpoint_inference, checkpoint_weights
    from eb.domain.rlinf_form import get_template
    from eb.store.models import Dataset, EvalSuite, TrainJob, Run, WeightModel
    from eb.api.routes_data import LIBERO_REPO, LIBERO_REVISION

    params = spec.params
    plan = params.get("eval_plan")
    if not isinstance(plan, dict):
        raise ValueError("LIBERO 评估必须从冻结评估计划创建")
    source = plan.get("checkpoint") or {}
    if not isinstance(source, dict):
        raise ValueError("LIBERO checkpoint 计划无效")
    suite = await session.get(EvalSuite, plan.get("suite_id"))
    run = await session.get(Run, source.get("run_id"))
    train = await session.scalar(select(TrainJob).where(TrainJob.job_id == run.job_id)) if run else None
    producer = await session.get(Job, run.job_id) if run else None
    try:
        template = get_template(params["template_id"])
    except KeyError as exc:
        raise ValueError("未知 LIBERO 评估模板") from exc
    raw = {key: value for key, value in plan.items() if key != "hash"}
    expected_hash = "sha256:" + hashlib.sha256(json.dumps(
        raw, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    if suite:
        from eb.api.routes_eval import build_eval_plan
        if plan != build_eval_plan(suite.id, suite.spec or {}, source, plan.get("profile"), plan.get("benchmark")):
            raise ValueError("LIBERO case 清单不是冻结套件生成的结果")
    benchmark = plan.get("benchmark") or {}
    if not isinstance(benchmark, dict):
        raise ValueError("LIBERO benchmark 绑定无效")
    dataset = await session.get(Dataset, benchmark.get("dataset_id"))
    summary = (dataset.summary_stats or {}).get("benchmark") if dataset else None
    digest = params.get("eval_checkpoint_sha256")
    if (plan.get("hash") != expected_hash or not suite or not suite.frozen
            or suite.spec != plan.get("suite_spec") or params.get("suite_id") != suite.id
            or not train or not producer or producer.kind not in {"train.sft", "train.rlft"}
            or producer.status not in {"succeeded", "failed", "timeout", "cancelled", "infra_failed"}
            or source.get("source_job_id") != producer.id
            or source.get("base_model_id") != train.base_model_id
            or source.get("source_template_id") != train.template_id
            or source.get("source_inference") != params.get("eval_source_inference")
            or source.get("source_image") != spec.image
            or source.get("source_image_ref") != train.image_digest
            or not dataset or dataset.status != "validated" or dataset.source_kind != "libero_benchmark"
            or dataset.source_uri != LIBERO_REPO or (dataset.meta or {}).get("source_revision") != LIBERO_REVISION
            or (dataset.meta or {}).get("usage") != "eval"
            or not summary or summary.get("sha256") != benchmark.get("sha256")
            or params.get("benchmark_dataset_id") != dataset.id
            or params.get("benchmark_sha256") != benchmark.get("sha256")
            or dataset.dest_path != str(get_settings().datasets_root / dataset.id / "bundle")
            or source.get("step") != params.get("eval_checkpoint_step")
            or source.get("run_id") != params.get("eval_checkpoint_run_id")
            or source.get("sha256") != digest
            or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)
            or template.kind != "eval" or template.task_id != spec.task_id
            or template.embodiment_env_id != spec.embodiment_env_id
            or suite.embodiment_env_id != spec.embodiment_env_id
            or not isinstance(spec.image, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", spec.image)):
        raise ValueError("LIBERO 评估计划、源 checkpoint 或本体接口不一致")
    model = await session.get(WeightModel, train.base_model_id) if train.base_model_id else None
    if not model or Path(spec.inputs.base_model or "").resolve() != Path(model.path).resolve():
        raise ValueError("LIBERO 评估基座模型与源训练不一致")
    frozen = get_settings().data_root / "cas" / digest[:2] / digest
    checkpoint = checkpoint_weights(get_settings().data_root / "runs" / run.id, source["step"])
    if isinstance(run.provenance, str):
        try:
            provenance = json.loads(run.provenance)
            producer_image = provenance.get("image_digest") if isinstance(provenance, dict) else None
        except json.JSONDecodeError:
            producer_image = None
        if producer_image and producer_image != spec.image:
            raise ValueError("评估镜像不是源训练的执行镜像")
    actual_inference = checkpoint_inference(get_settings().data_root / "runs" / run.id,
                                            train.template_id, (producer.spec or {}).get("overrides", []))
    if (frozen.is_symlink() or not frozen.is_file() or frozen.stat().st_size != checkpoint["bytes"]
            or source.get("bytes") != checkpoint["bytes"]
            or actual_inference != source.get("source_inference")):
        raise ValueError("LIBERO 评估冻结 checkpoint 丢失或与源权重不一致")
    if plan.get("profile") == "smoke":
        expected_overrides, count = ["env.eval.total_num_envs=4", "env.eval.rollout_epoch=1"], 4
    elif plan.get("profile") == "standard":
        expected_overrides, count = ["env.eval.total_num_envs=10", "env.eval.rollout_epoch=10"], 100
    else:
        raise ValueError("未知 LIBERO 评估 profile")
    if spec.overrides != expected_overrides or plan.get("planned") != count or len(plan.get("cases", [])) != count:
        raise ValueError("LIBERO 评估 case 计划或执行参数被改写")


class Orchestrator:
    def __init__(self, ebd: EbdClient | None = None) -> None:
        self.ebd = ebd or EbdClient()

    async def submit(
        self,
        session: AsyncSession,
        spec: JobSpec,
        principal: str,
        idem_key: str | None,
        *,
        commit: bool = True,
    ) -> Job:
        from eb.orchestrator.deletion import guard_storage_inputs
        await guard_storage_inputs(session, spec.model_dump(mode="json"))
        if idem_key:
            existing = await session.scalar(select(Job).where(Job.idem_key == idem_key))
            if existing:
                return existing
        await validate_flat_warm_start_submission(session, spec)
        if spec.params.get("operation") == "probe_pbfm_runtime":
            if principal.startswith('agent:'):
                raise PermissionError('P-BFM runtime diagnostics require an operator')
            from eb.pbfm_runtime_probe import validate_submission as validate_pbfm_probe
            await validate_pbfm_probe(session, spec)
        from eb.native_training import is_native_template, service_for
        from eb.ufo_training import DATA_OPERATION as UFO_DATA_OPERATION
        native_training = is_native_template(spec.params.get("template_id"))
        native_data = spec.params.get("operation") == UFO_DATA_OPERATION
        from eb.ufo_failure_probe import OPERATION as FAILURE_PROBE_OPERATION
        failure_probe = spec.params.get("operation") == FAILURE_PROBE_OPERATION
        if failure_probe:
            if principal.startswith('agent:'):
                raise PermissionError('GPU failure diagnostics require an operator')
            from eb.ufo_failure_probe import validate_submission as validate_failure_probe
            await validate_failure_probe(session, spec)
        from eb.ufo_native_baseline import OPERATION as BASELINE_OPERATION
        native_baseline = spec.params.get("operation") == BASELINE_OPERATION
        if native_baseline:
            if principal.startswith('agent:'):
                raise PermissionError('Native training baseline requires an operator')
            from eb.ufo_native_baseline_service import validate_submission as validate_baseline_submission
            await validate_baseline_submission(session, spec)
        from eb.ufo_grounded_reference import OPERATION as GROUNDED_OPERATION
        grounded_data = spec.params.get("operation") == GROUNDED_OPERATION
        if grounded_data:
            if principal.startswith('agent:'):
                raise PermissionError('A1 reference candidate preparation requires an operator')
            from eb.ufo_grounded_reference import validate_submission as validate_grounded_submission
            await validate_grounded_submission(session, spec)
        if native_training or native_data:
            if principal.startswith('agent:'):
                raise PermissionError('Native training and data preparation require an operator')
            if native_training:
                await service_for(spec.params["template_id"]).validate_submission(session, spec)
            else:
                from eb.ufo_training_service import validate_submission as validate_train_submission
                await validate_train_submission(session, spec)
        if spec.params.get("operation") in {"prepare_pbfm_rpo_motion", "export_pbfm_policy", "evaluate_pbfm_terrain"}:
            if principal.startswith('agent:'):
                raise PermissionError('P-BFM preparation, export and evaluation require an operator')
            from eb.pbfm_evaluation import validate_submission as validate_pbfm_submission
            await validate_pbfm_submission(session, spec)
        from eb.ufo_native import ENV_ID as UFO_NATIVE_ENV, OPERATIONS as UFO_NATIVE_OPERATIONS
        if (spec.embodiment_env_id == UFO_NATIVE_ENV or spec.params.get('operation') in UFO_NATIVE_OPERATIONS) and not (native_training or native_data or grounded_data or native_baseline or failure_probe):
            if principal.startswith('agent:'):
                raise PermissionError('Native deployment jobs require an operator; use ToolBus for platform Agent actions')
            from eb.orchestrator.ufo_native import validate_submission
            await validate_submission(session, spec)
        if spec.kind == JobKind.EVAL_SIM and spec.params.get("template_id", "").startswith("libero_"):
            await validate_libero_eval_submission(session, spec)
        row = await session.get(EmbodimentEnvRow, spec.embodiment_env_id)
        if not row:
            raise ValueError(f"unknown embodiment env {spec.embodiment_env_id}")
        env = EmbodimentEnv.model_validate(row.spec)
        task = None
        if spec.task_id:
            trow = await session.get(TaskRow, spec.task_id)
            if trow:
                task = TaskSpec.model_validate(trow.spec)
        params = dict(spec.params)
        if task and not params.get("task"):
            gym = task.gym_id()
            if gym:
                params["task"] = gym
        if task and spec.kind in {JobKind.TRAIN_SFT, JobKind.TRAIN_RLFT} and not params.get("config_name"):
            cfg = task.rlinf_config_name()
            if cfg:
                params["config_name"] = cfg
        if params != spec.params:
            spec = spec.model_copy(update={"params": params})
        if spec.params.get("template_id", "").startswith("libero_"):
            from eb.store.models import Dataset
            from eb.data_validation import require_validated_dataset
            from eb.paths import validate_allowed_path
            for binding in spec.inputs.datasets:
                dataset = await session.get(Dataset, binding.get("id"))
                if dataset is None:
                    raise ValueError("训练输入必须引用已纳管的数据集")
                require_validated_dataset(dataset, spec.embodiment_env_id)
                if validate_allowed_path(str(binding.get("path", ""))) != validate_allowed_path(dataset.dest_path):
                    raise ValueError("训练数据路径与登记版本不一致")
            if spec.inputs.base_model:
                if not validate_allowed_path(spec.inputs.base_model).is_dir():
                    raise ValueError("基座模型目录不存在")
        gate = g1_submit(spec, env, GateContext(dataset_versions=[], task=task))
        if not gate.passed:
            raise PermissionError({"violations": [v.model_dump() for v in gate.violations]})
        est = estimate_job(spec)
        if est.verdict == "no_fit":
            raise PermissionError({"capacity": est.__dict__})
        if spec.kind not in CONTAINERLESS_KINDS:
            validate_mounts(get_adapter(spec).render(spec, "run_validate").mounts)
        emb = await session.get(EmbodimentRow, env.embodiment_id)
        settings = get_settings()
        refs = spec.code_refs or CodeRefs()
        emb_path = Path(
            (refs.embodiment.path if refs.embodiment else "")
            or ((emb.source or {}).get("path") if emb else "")
            or settings.bhl_repo
        )
        if (
            emb_path.exists()
            and (refs.embodiment or (emb and _is_snapshot_source(emb.source)))
            and spec.kind not in CONTAINERLESS_KINDS
            and spec.kind.value.startswith(("train.", "eval.", "export."))
        ):
            refs = refs.model_copy(
                update={
                    "embodiment": await _snapshot_ref(
                        self.ebd, emb_path, refs.embodiment.commit if refs.embodiment else None
                    )
                }
            )
        stack_kind = (spec.runtime.get("stack") if spec.runtime else None) or spec.params.get("stack")
        need_rlinf = spec.kind in {JobKind.TRAIN_SFT, JobKind.TRAIN_RLFT} or stack_kind == "rlinf"
        if need_rlinf:
            stack_path = Path((refs.stack.path if refs.stack else "") or settings.rlinf_repo)
            if stack_path.exists():
                refs = refs.model_copy(
                    update={
                        "stack": await _snapshot_ref(
                            self.ebd, stack_path, refs.stack.commit if refs.stack else None
                        )
                    }
                )
        spec = spec.model_copy(update={"code_refs": refs})
        job = Job(
            id=new_id("job"),
            embodiment_env_id=spec.embodiment_env_id,
            kind=spec.kind.value,
            spec=spec.model_dump(mode="json"),
            spec_hash=spec.canonical_hash(),
            idem_key=idem_key,
            status=JobStatus.QUEUED.value,
            priority=spec.policy.priority,
            created_by=principal,
        )
        session.add(job)
        if commit:
            await session.commit()
        else:
            await session.flush()
        await session.refresh(job)
        return job

    async def admit_and_start(self, session: AsyncSession, job: Job, gpu_ids: list[int]) -> Run:
        await session.refresh(job, with_for_update=True)
        if job.status != JobStatus.QUEUED.value:
            await session.commit()
            raise ValueError("job is no longer queued for admission")
        spec = JobSpec.model_validate(
            {**job.spec, "kind": job.kind, "embodiment_env_id": job.embodiment_env_id, "api_version": "eb/v1"}
        )
        from eb.native_training import is_native_template, service_for
        from eb.pbfm_runtime_probe import TRAINING_MODES
        diagnostic_training = (
            spec.params.get("operation") == "probe_pbfm_runtime"
            and spec.params.get("probe_mode") in TRAINING_MODES
        )
        if diagnostic_training or (
            is_native_template(spec.params.get("template_id")) and spec.inputs.resume_from
        ):
            try:
                if diagnostic_training:
                    from eb.pbfm_runtime_probe import validate_submission as validate_diagnostic_training

                    await validate_diagnostic_training(session, spec)
                else:
                    await service_for(spec.params["template_id"]).validate_submission(session, spec)
            except (ValueError, OSError, KeyError, TypeError, httpx.HTTPError) as exc:
                job.status = transition(JobStatus(job.status), JobStatus.BLOCKED).value
                context = "P-BFM diagnostic training" if diagnostic_training else "Native resume source"
                job.error = f"{context} verification failed: {exc}"
                job.version += 1
                await session.commit()
                raise
        job.status = transition(JobStatus(job.status), JobStatus.ADMITTED).value
        job.status = transition(JobStatus(job.status), JobStatus.PREPARING).value
        job.attempts += 1
        job.version += 1
        run = Run(
            id=new_id("run"),
            job_id=job.id,
            attempt=job.attempts,
            status=RunStatus.CREATED.value,
            gpu_ids=gpu_ids,
            started_at=datetime.now(UTC),
        )
        session.add(run)
        rlinf_resume = bool(spec.inputs.resume_from and spec.params.get("stack") == "rlinf")
        if rlinf_resume:
            session.add(LineageEdge(src_type="run", src_id=spec.inputs.resume_from,
                                    dst_type="run", dst_id=run.id, relation="resumed_from"))
        spec_kind = spec.kind
        if spec_kind in CONTAINERLESS_KINDS:
            try:
                await self._run_containerless(session, job, run, spec)
            except Exception as e:
                apply_ebd_result(
                    job,
                    run,
                    {"status": "docker_error", "reason": str(e), "container_id": None},
                )
                await session.commit()
            await session.refresh(job)
            await session.refresh(run)
            return run
        run_root = prepare_run_dirs(run.id)
        if spec.inputs.resume_from and not rlinf_resume:
            try:
                load_run, ckpt = seed_rsl_resume(run_root, spec.inputs.resume_from)
            except FileNotFoundError as e:
                apply_ebd_result(
                    job,
                    run,
                    {"status": "docker_error", "reason": str(e), "container_id": None},
                )
                await session.commit()
                return run
            spec = spec.model_copy(
                update={"overrides": ["--load_run", load_run, "--checkpoint", ckpt, *spec.overrides]}
            )
        rendered = get_adapter(spec).render(spec, run.id)
        mounts = list(rendered.mounts)
        emb_ref = spec.code_refs.embodiment if spec.code_refs else None
        repo = Path(emb_ref.path) if emb_ref and emb_ref.path else get_settings().bhl_repo
        isaac_like = rendered.workdir == "/workspace/isaaclab_extension_template"
        if emb_ref or isaac_like:
            try:
                wt = await prepare_worktree(repo, emb_ref.commit if emb_ref else None, run.id, ebd=self.ebd)
                if isaac_like:
                    writable = rendered.writable_paths or ["logs", "outputs", "configs"]
                    mounts = overlay_mounts(wt, run_root, writable)
                else:
                    mounts = [(src, dst, mode) for src, dst, mode in mounts if dst != "/workspace/berkeley"]
                    mounts.append((str(wt), "/workspace/berkeley", "ro"))
            except Exception as e:
                log.exception("prepare_worktree failed for %s", run.id)
                apply_ebd_result(
                    job,
                    run,
                    {
                        "status": "docker_error",
                        "reason": f"worktree snapshot failed: {e}",
                        "container_id": None,
                    },
                )
                await session.commit()
                return run
        stack_ref = spec.code_refs.stack if spec.code_refs else None
        if stack_ref and stack_ref.path:
            sp = Path(stack_ref.path)
            dst = _stack_mount_destination(mounts)
            mounts = [(src, d, mode) for src, d, mode in mounts if d != dst]
            if sp.exists():
                try:
                    wt_s = await prepare_worktree(sp, stack_ref.commit, run.id, ebd=self.ebd)
                    mounts.append((str(wt_s), dst, "ro"))
                except Exception as e:
                    apply_ebd_result(
                        job,
                        run,
                        {
                            "status": "docker_error",
                            "reason": f"stack worktree snapshot failed: {e}",
                            "container_id": None,
                        },
                    )
                    await session.commit()
                    return run
            else:
                apply_ebd_result(
                    job, run, {"status": "docker_error", "reason": "stack snapshot source is missing"}
                )
                await session.commit()
                return run
        try:
            validate_mounts(mounts)
            result = await self.ebd.submit_run(
                {
                    "run_id": run.id,
                    "job_id": job.id,
                    "image": rendered.image,
                    "command": rendered.command,
                    "env": rendered.env,
                    "gpus": gpu_ids,
                    "shm_gb": rendered.shm_gb,
                    "cpus": spec.resources.cpus,
                    "memory_gb": spec.resources.memory_gb,
                    "network": spec.runtime.get("network", "bridge"),
                    "hostname": spec.runtime.get("hostname"),
                    "devices": rendered.devices,
                    "workdir": rendered.workdir,
                    "kind": spec.kind.value,
                    "mounts": [{"src": src, "dst": dst, "mode": mode} for src, dst, mode in mounts],
                }
            )
            apply_ebd_result(job, run, result)
            digest = result.get("image_digest")
            if digest:
                from eb.domain.fingerprint import fingerprint

                commit = emb_ref.commit if emb_ref else None
                run.provenance = json.dumps(fingerprint(spec, digest, commit))
        except Exception as e:
            apply_ebd_result(
                job,
                run,
                {"status": "docker_error", "reason": str(e), "container_id": None},
            )
        await session.commit()
        await session.refresh(job)
        await session.refresh(run)
        return run

    async def _run_containerless(self, session: AsyncSession, job: Job, run: Run, spec: JobSpec) -> None:
        if spec.kind == JobKind.DATA_SCAN:
            path = spec.params.get("path") or "/data/kk/berkeley_ego_data"
            result = await self.ebd.scan(path)
            run.status = RunStatus.FINISHED.value
            run.finished_at = datetime.now(UTC)
            job.status = JobStatus.SUCCEEDED.value
            run.metrics_summary = {
                "files": result.get("files"),
                "bytes": result.get("bytes"),
                "content_hash": result.get("content_hash"),
            }
            from eb.store.models import DatasetVersion

            ver = await session.scalar(
                select(DatasetVersion).where(DatasetVersion.dataset_id == "ds_berkeley_ego")
            )
            if ver:
                ver.content_hash = result.get("content_hash")
                ver.stats = {
                    "files": result.get("files"),
                    "bytes": result.get("bytes"),
                    "kinds": result.get("kinds"),
                    "samples": result.get("samples") or [],
                }
                funnel = result.get("funnel") or {}
                ver.funnel = {
                    "source": funnel.get("source") or {"n": result.get("files") or 0, "rejected": []},
                    "raw": funnel.get("raw") or {"n": result.get("files") or 0, "rejected": []},
                    "valid_episode": {
                        "n": int((result.get("kinds") or {}).get("ego_jsonl") or 0),
                        "rejected": [],
                    },
                    "retargeted": {"n": 0, "rejected": []},
                    "canonical": {"n": 0, "rejected": []},
                    "slice": {"n": 0, "rejected": []},
                }
            await session.commit()
            return
        if spec.kind == JobKind.DATA_SLICE:
            query = spec.params.get("query") or {}
            src_hash = spec.params.get("content_hash") or "unknown"
            from eb.ids import new_id
            from eb.store.models import DatasetVersion

            vid = new_id("dsv")
            session.add(
                DatasetVersion(
                    id=vid,
                    dataset_id=spec.params.get("dataset_id") or "ds_berkeley_ego",
                    version=str(spec.params.get("version") or vid),
                    compatible_envs=spec.params.get("compatible_envs") or [],
                    stats={"query": query, "parent_hash": src_hash},
                    funnel={"slice": {"n": 0, "rejected": [{"reason": "empty_query", "count": 0}]}},
                    content_hash=src_hash,
                )
            )
            run.status = RunStatus.FINISHED.value
            run.finished_at = datetime.now(UTC)
            job.status = JobStatus.SUCCEEDED.value
            run.metrics_summary = {"dataset_version_id": vid, "content_hash": src_hash}
            await session.commit()
            return
        apply_ebd_result(
            job, run, {"status": "docker_error", "reason": f"unhandled containerless {spec.kind}"}
        )
        await session.commit()

    async def collect_and_finish(self, session: AsyncSession, job: Job, run: Run) -> None:
        """COLLECTING_OUT then SUCCEEDED. Artifact claim is filled in later tasks."""
        run.status = RunStatus.COLLECTING_OUT.value
        await session.flush()
        from eb.orchestrator.collect import claim_outputs

        try:
            await claim_outputs(session, job, run)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            job.status = JobStatus.FAILED.value
            job.error = f"产物验证失败: {exc}"
            run.status = RunStatus.ABORTED.value
            run.abort_reason = "invalid_output"
            run.finished_at = datetime.now(UTC)
            from eb.orchestrator.collect import archive_partial_eval
            await archive_partial_eval(session, job, run, invalid=True)
            await session.commit()
            return
        if job.status in {JobStatus.FAILED.value, JobStatus.INFRA_FAILED.value}:
            await session.commit()
            return
        if run.status == RunStatus.ABORTED.value:
            if job.status == JobStatus.RUNNING.value:
                job.status = JobStatus.FAILED.value
            await session.commit()
            return
        log = get_settings().data_root / "runs" / run.id / "console.log"
        onnx_ok = any(get_settings().data_root.joinpath("runs", run.id).rglob("exported/policy.onnx"))
        claimed_output = any(key in (run.metrics_summary or {}) for key in ("eval_report_id", "checkpoint"))
        if log.exists() and not onnx_ok and not claimed_output:
            text = log.read_text(errors="replace")
            if "Traceback (most recent call last):" in text:
                job.status = JobStatus.FAILED.value
                job.error = "\n".join(text.strip().splitlines()[-12:])[:2000]
                run.status = RunStatus.ABORTED.value
                run.abort_reason = "python_exception"
                await session.commit()
                return
        run.status = RunStatus.FINISHED.value
        run.finished_at = datetime.now(UTC)
        job.status = JobStatus.SUCCEEDED.value
        await session.commit()

    def preview_command(self, spec: JobSpec, run_id: str = "run_preview") -> str:
        return equivalent_command(spec, run_id)
