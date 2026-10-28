"""Protected diagnostics retain frozen inputs without enabling the training recipe."""

import re
from dataclasses import replace
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from eb.domain.job import JobKind, JobSpec
from eb.pbfm_training import TEMPLATE, training_fingerprint, validate_train_job

OPERATION = "probe_pbfm_runtime"
TRAINING_MODES = {"training_validation", "training_exploration", "training_long_exploration",
                  "training_long_recovery"}


class ProbeIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=120, pattern=r"^[\w .-]+$")
    dataset_id: str
    environment_id: str
    mode: Literal[
        "runtime", "segment112", "training_validation", "training_exploration", "training_long_exploration",
        "training_long_recovery"
    ] = "runtime"
    gpus: Literal[1, 2, 4] = 1
    num_envs_per_rank: Literal[1024, 2048] = 1024
    target_updates: Literal[112, 368, 512, 10000, 100000] = 112
    resume_job_id: str | None = Field(default=None, min_length=1)
    recovery_update: int | None = Field(default=None, ge=112, le=100000, strict=True)
    validation_report_id: str | None = Field(default=None, min_length=1)
    timeout_seconds: int = Field(default=600, ge=60, le=108000, strict=True)
    failure_rank: int | None = Field(default=None, ge=0, le=3, strict=True)
    preview_hash: str | None = Field(default=None, pattern=r"^sha256:[a-f0-9]{64}$")

    @model_validator(mode="after")
    def rank_within_group(self):
        if self.mode in TRAINING_MODES:
            targets = ({100000} if self.mode in {"training_long_exploration", "training_long_recovery"} else
                       {10000} if self.mode == "training_exploration" else {112, 368, 512})
            if self.target_updates not in targets:
                raise ValueError(
                    "P-BFM exploration requires explicit target10000; validation retains 112/368/512"
                )
            if (self.mode not in {"training_long_exploration", "training_long_recovery"}
                    and self.timeout_seconds > 21600):
                raise ValueError("Historical P-BFM modes retain their six-hour budget")
            if self.gpus != 4 or self.num_envs_per_rank != 2048 or self.failure_rank is not None:
                raise ValueError("training_validation requires W4/E2048 and forbids failure injection")
            if self.mode == "training_long_recovery":
                if (self.recovery_update != 19984 or not self.resume_job_id
                        or self.validation_report_id is not None):
                    raise ValueError("Recovery requires complete19984 and no fabricated independent report")
            elif self.target_updates == 112:
                if self.resume_job_id is not None or self.validation_report_id is not None:
                    raise ValueError("training_validation target112 must start cold without a report")
            elif self.resume_job_id is None or self.validation_report_id is None:
                raise ValueError("training_validation continuation requires source Job and validation report")
        elif (
            self.target_updates != 112 or self.resume_job_id is not None
            or self.validation_report_id is not None or self.timeout_seconds > 3600
        ):
            raise ValueError("runtime/segment112 require target112 without resume and a bounded probe budget")
        if self.mode != "training_long_recovery" and self.recovery_update is not None:
            raise ValueError("Recovery generation is forbidden outside explicit recovery")
        if self.mode == "segment112" and (self.gpus != 1 or self.failure_rank is not None):
            raise ValueError("segment112 requires one GPU and forbids failure injection")
        if self.mode == "segment112" and self.num_envs_per_rank != 1024:
            raise ValueError("segment112 preserves E1024")
        if self.failure_rank is not None and self.failure_rank >= self.gpus:
            raise ValueError("failure_rank must identify a leased probe rank")
        return self


async def build_spec(body: ProbeIn, session) -> JobSpec:
    from eb.api.routes_train import TrainIn
    from eb.pbfm_training_service import _build_spec

    if body.mode == "training_long_recovery":
        from eb.pbfm_recovery import build_recovery_spec

        spec = await build_recovery_spec(body, session)
        await validate_submission(session, spec)
        return spec
    if body.mode in TRAINING_MODES and body.resume_job_id:
        from eb.pbfm_diagnostic_training import build_resume_diagnostic

        return await build_resume_diagnostic(body, session)
    request = TrainIn(
        name=body.name,
        model_name=body.name,
        template_id=TEMPLATE,
        embodiment_id="emb_roboparty_rpo",
        gpus=body.gpus,
        dataset_ids=[body.dataset_id],
        environment_id=body.environment_id,
        pbfm_training={
            "target_updates": 112, "num_envs_per_rank": body.num_envs_per_rank,
            "buffer_size": 131072 if body.num_envs_per_rank == 2048 else 32768,
            "timeout_seconds": body.timeout_seconds,
        },
    )
    training = await _build_spec(request, session, diagnostic=True)
    params = {
        **training.params, "operation": OPERATION, "probe_recipe_id": TEMPLATE,
        "probe_mode": body.mode,
    }
    params.pop("template_id")
    if body.failure_rank is not None:
        params["probe_fail_rank"] = body.failure_rank
    spec = training.model_copy(update={"kind": JobKind.ASSET_VALIDATE, "params": params})
    if body.mode in TRAINING_MODES:
        await validate_submission(session, spec)
    return spec


def _training_view(spec: JobSpec) -> JobSpec:
    mode = spec.params.get("probe_mode", "runtime")
    resumed = mode in TRAINING_MODES and bool(spec.inputs.resume_from)
    extra = {
        "native_resume_source", "resume_mode", "resume_step", "source_diagnostic_job_id",
        "validation_report_id", "validation_report_binding",
    } if resumed else set()
    if (
        spec.kind != "asset.validate"
        or spec.params.get("operation") != OPERATION
        or spec.params.get("probe_recipe_id") != TEMPLATE
        or "template_id" in spec.params
        or (spec.inputs.resume_from and not resumed)
        or mode not in TRAINING_MODES | {"runtime", "segment112"}
        or set(spec.params) - {"probe_fail_rank", "probe_mode"} - extra
        != {
            "stack",
            "operation",
            "probe_recipe_id",
            "model_name",
            "native_train_plan",
            "pbfm_source_ref",
            "asset_source_ref",
            "environment_profile",
        }
    ):
        raise ValueError("P-BFM runtime probe is not a protected diagnostic Job")
    failure_rank = spec.params.get("probe_fail_rank")
    if mode in TRAINING_MODES and (
        spec.resources.gpus != 4 or "probe_fail_rank" in spec.params
        or resumed and not extra.issubset(spec.params)
    ):
        raise ValueError("P-BFM training_validation requires W4 and protected native resume inputs")
    if spec.params.get("probe_mode") == "segment112" and (
        spec.resources.gpus != 1 or "probe_fail_rank" in spec.params
    ):
        raise ValueError("P-BFM segment112 requires W=1 and forbids failure injection")
    if failure_rank is not None and (
        type(failure_rank) is not int or not 0 <= failure_rank < spec.resources.gpus
    ):
        raise ValueError("P-BFM failure injection must select one leased diagnostic rank")
    params = {**spec.params, "template_id": TEMPLATE}
    params.pop("operation")
    params.pop("probe_recipe_id")
    params.pop("probe_fail_rank", None)
    params.pop("probe_mode", None)
    for key in ("source_diagnostic_job_id", "validation_report_id", "validation_report_binding"):
        params.pop(key, None)
    training = spec.model_copy(update={"kind": JobKind.TRAIN_RL, "params": params})
    plan = validate_train_job(training)
    if mode in TRAINING_MODES:
        from eb.pbfm_diagnostic_training import validate_segment

        validate_segment(spec, plan)
    elif (
        plan["target_updates"] != 112
        or plan["identity"]["buffer_size"] != (
            131072 if mode == "runtime" and plan["identity"]["num_envs_per_rank"] == 2048 else 32768
        )
        or plan.get("start_updates", 0)
        or plan.get("start_local_time", 0)
        or plan["segment_timeout_seconds"] > 3600
    ):
        raise ValueError("P-BFM diagnostic requires target112/buffer32768 and no resume or enlarged budget")
    return training


async def validate_submission(session, spec) -> None:
    from eb.pbfm_training_service import _validate_submission

    training = _training_view(spec)
    if spec.params.get("probe_mode") in TRAINING_MODES:
        from eb.pbfm_diagnostic_training import validate_diagnostic_submission

        await validate_diagnostic_submission(session, spec, training)
    else:
        await _validate_submission(session, training, diagnostic=True)


def render(spec, run_id):
    from eb.pbfm_training import render_training

    rendered = render_training(_training_view(spec), run_id)
    if spec.params.get("probe_mode") in TRAINING_MODES | {"segment112"}:
        return rendered
    failure = (
        ["--probe-fail-rank", str(spec.params["probe_fail_rank"])]
        if spec.params.get("probe_fail_rank") is not None
        else []
    )
    return replace(rendered, command=[*rendered.command, "--probe", *failure], metrics_glob="")


async def collect(session, job, run, outputs) -> None:
    from eb.orchestrator.ufo_native import verify_index
    from eb.pbfm_io import read_json, safe_file
    from eb.pbfm_training import CAMERA_CONTRACT
    from eb.runners.pbfm_depth_probe import (
        PHASES,
        RECEIPT_SCHEMA,
        required_probe_names,
        validate_depth_report,
    )

    spec = JobSpec.model_validate(job.spec)
    training = _training_view(spec)
    if spec.params.get("probe_mode") in TRAINING_MODES | {"segment112"}:
        from eb.orchestrator.pbfm_training import collect_pbfm_training

        await collect_pbfm_training(session, job, run, outputs.parent, diagnostic_spec=training)
        return
    plan = training.params["native_train_plan"]
    if spec.params.get("probe_fail_rank") is not None:
        raise ValueError(
            "Injected P-BFM failure probe must fail the Job and cannot publish a success receipt"
        )
    receipt = read_json(safe_file(outputs, "probe-receipt.json"))
    if (
        receipt.get("schema_version") not in {"eb/pbfm-native-runtime-probe/v1", RECEIPT_SCHEMA}
        or receipt.get("run_id") != run.id
        or receipt.get("status") != "complete"
        or receipt.get("plan_hash") != training_fingerprint({**plan, "run_id": run.id})
        or receipt.get("world_size") != plan["world_size"]
        or receipt.get("training_completed") is not False
    ):
        raise ValueError("P-BFM runtime probe receipt differs from its frozen diagnostic Job")
    ranks, files = receipt.get("ranks"), receipt.get("files")
    world = plan["world_size"]
    if (
        not isinstance(ranks, list)
        or len(ranks) != world
        or any(not isinstance(row, dict) for row in ranks)
        or {row.get("rank") for row in ranks} != set(range(world))
        or len({row.get("gpu_uuid") for row in ranks}) != world
        or len({row.get("pid") for row in ranks}) != world
    ):
        raise ValueError("P-BFM runtime probe lacks unique complete rank/GPU evidence")
    for row in ranks:
        contract = row.get("native_contract") or {}
        if (
            row.get("status") != "probe_complete"
            or row.get("world_size") != world
            or type(row.get("pid")) is not int
            or row["pid"] <= 0
            or not isinstance(row.get("gpu_uuid"), str)
            or not row["gpu_uuid"]
            or row.get("worker_class") != "PbfmTrainWorker"
            or row.get("training_completed") is not False
            or row.get("environment_reset_step") is not True
            or row.get("egl_render_verified") is not True
            or row.get("depth_verified") is not True
            or row.get("process_group") is not (world > 1)
            or row.get("source_refs") != plan["source_refs"]
            or contract.get("actor_scalar_dim") != 380
            or contract.get("action_dim") != 23
            or contract.get("physics_hz") != 400
            or contract.get("policy_hz") != 50
            or contract.get("camera_contract") != CAMERA_CONTRACT
        ):
            raise ValueError("P-BFM runtime probe rank failed native RPO/depth/renderer verification")
    required = {
        f"runtime/probe-rank-{rank}/{name}"
        for rank in range(world)
        for name in ("network-depth.png", "network-depth.npy", "metric-depth.npy", "robot-terrain.png")
    }
    if receipt["schema_version"] == RECEIPT_SCHEMA:
        required.update(
            f"runtime/probe-rank-{rank}/{name}"
            for rank in range(world) for name in required_probe_names()
        )
    if (
        not isinstance(files, list)
        or not files
        or any(
            not isinstance(row, dict)
            or not isinstance(row.get("path"), str)
            or not re.fullmatch(
                r"runtime/(?:rank-[0-3]\.json|probe-rank-[0-3]/[\w.-]+\.(?:png|npy|json))", row["path"]
            )
            for row in files
        )
        or not required.issubset({row["path"] for row in files})
    ):
        raise ValueError("P-BFM runtime probe lacks indexed real depth and robot/terrain evidence")
    if receipt["schema_version"] == RECEIPT_SCHEMA:
        await verify_index(outputs, files, max_files=128 * world, max_file_bytes=4 * 1024**2)
    else:
        await verify_index(outputs, files)
    if receipt["schema_version"] == RECEIPT_SCHEMA:
        indexed = {item["path"]: item for item in files}
        for row in ranks:
            folder = f"runtime/probe-rank-{row['rank']}/"
            report = validate_depth_report(read_json(safe_file(outputs, folder + "depth-validation.json")))
            if report.get("rank") != row["rank"] or row.get("depth_validation") != report:
                raise ValueError("P-BFM depth comparison producer differs from its rank result")
            for prefix in PHASES.values():
                actual = indexed[folder + prefix + "network-depth.npy"]
                reference = indexed[folder + prefix + "reference-network-depth.npy"]
                if any(actual[key] != reference[key] for key in ("bytes", "sha256")):
                    raise ValueError("P-BFM native depth history differs from its indexed reference")
    run.metrics_summary = {
        "native_operation": OPERATION,
        "runtime_probe": receipt,
        "qualification": "incomplete",
        "training_completed": False,
        "files": [{**item, "url": f"/api/v1/pbfm/runs/{run.id}/files/{item['path']}"} for item in files],
    }
