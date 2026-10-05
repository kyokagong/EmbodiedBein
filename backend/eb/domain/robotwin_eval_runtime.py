"""Validated single-policy runtime profiles for RoboTwin evaluation."""
from __future__ import annotations

from copy import deepcopy

from pydantic import BaseModel, ConfigDict, model_validator


PROFILES: dict[tuple[int, int], dict[str, int]] = {
    (1, 4): {"env_worker_count":1, "envs_per_worker":4,
             "inference_micro_batch_size":4, "cpus":8, "memory_gb":48, "shm_gb":24},
    (2, 8): {"env_worker_count":2, "envs_per_worker":4,
             "inference_micro_batch_size":4, "cpus":16, "memory_gb":56, "shm_gb":32},
    (4, 16): {"env_worker_count":4, "envs_per_worker":4,
              "inference_micro_batch_size":4, "cpus":32, "memory_gb":80, "shm_gb":40},
}

_SYNC_VARIANTS = (
    {"id":"sync_1gpu_4env_48g", "gpus":1, "cpus":8, "memory_gb":48, "shm_gb":24,
     "policy_worker_count":1, "env_worker_count":1, "envs_per_worker":4,
     "parallel_envs":4, "inference_micro_batch_size":4, "max_inflight_batches":1,
     "probe_receipt_sha256":"sha256:d082b2161a76864144135f033457707827757114e6ea62fe4a447836feeed131"},
    {"id":"sync_2gpu_8env_56g", "gpus":2, "cpus":16, "memory_gb":56, "shm_gb":32,
     "policy_worker_count":1, "env_worker_count":2, "envs_per_worker":4,
     "parallel_envs":8, "inference_micro_batch_size":4, "max_inflight_batches":1,
     "probe_receipt_sha256":"sha256:50a6c32bc5e539ea43af90f0679d7d342592f890c0e1d7f39efc3048c5eb70a2"},
    {"id":"sync_4gpu_16env_80g", "gpus":4, "cpus":32, "memory_gb":80, "shm_gb":40,
     "policy_worker_count":1, "env_worker_count":4, "envs_per_worker":4,
     "parallel_envs":16, "inference_micro_batch_size":4, "max_inflight_batches":1,
     "probe_receipt_sha256":"sha256:d81e268d1bce909365ef40819ebdfe6542889c1f525ee73805ff3ea7901e7f6a"},
)

_RUNTIME_PROFILES = (
    {"id":"memory_saver_sync", "name":"节省显存", "status":"ready",
     "description":"单 Policy、同步有界批处理", "scheduling_mode":"sync",
     "recommended_variant_id":"sync_4gpu_16env_80g", "blockers":[],
     "variants":_SYNC_VARIANTS},
    {"id":"balanced_async", "name":"均衡异步", "status":"blocked",
     "description":"单 Policy、动态批处理", "scheduling_mode":"dynamic_batching",
     "recommended_variant_id":None, "variants":(),
     "blockers":[{"code":"RLINF_DYNAMIC_BATCHING_NOT_PROBED",
                  "message":"RLInf 动态 batching 执行路径和持续吞吐 probe 尚未完成"}]},
    {"id":"throughput_async", "name":"吞吐优先", "status":"blocked",
     "description":"多 Policy 分片异步推理", "scheduling_mode":"sharded_async",
     "recommended_variant_id":None, "variants":(),
     "blockers":[{"code":"RLINF_MULTI_POLICY_NOT_PROBED",
                  "message":"RLInf 多 Policy 路由、显存隔离和吞吐 probe 尚未完成"}]},
)


class RobotwinEvalRuntime(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    gpu_count: int
    policy_worker_count: int = 1
    env_worker_count: int
    envs_per_worker: int
    parallel_envs: int
    inference_micro_batch_size: int
    max_inflight_batches: int = 1
    headless: bool = True
    offscreen_rendering: bool = True
    save_video: bool = False
    reuse_policy_across_tasks: bool = True
    profile_id: str | None = None
    resource_variant_id: str | None = None
    scheduling_mode: str = "sync"
    probe_receipt_sha256: str | None = None

    @model_validator(mode="after")
    def validate_topology(self):
        if self.policy_worker_count != 1:
            raise ValueError("RoboTwin 评估必须只有一个 Policy Worker")
        if self.env_worker_count > self.gpu_count or self.env_worker_count <= 0:
            raise ValueError("RoboTwin Env Worker 数与 GPU 数不兼容")
        if self.parallel_envs != self.env_worker_count * self.envs_per_worker:
            raise ValueError("RoboTwin 总并发环境数与 Worker 拓扑不一致")
        if not 1 <= self.inference_micro_batch_size <= self.parallel_envs:
            raise ValueError("RoboTwin 推理 micro-batch 无效")
        if self.parallel_envs % self.inference_micro_batch_size:
            raise ValueError("RoboTwin 完整 wave 必须能按 micro-batch 切分")
        if self.max_inflight_batches != 1:
            raise ValueError("RoboTwin 首期只允许一个在途 batch")
        if (not self.headless or not self.offscreen_rendering or self.save_video
                or not self.reuse_policy_across_tasks):
            raise ValueError("RoboTwin 全量评估运行模式无效")
        return self


def list_robotwin_runtime_profiles() -> list[dict]:
    return deepcopy(list(_RUNTIME_PROFILES))


def resolve_robotwin_runtime(profile_id: str, variant_id: str) -> tuple[RobotwinEvalRuntime, dict[str, int]]:
    profile = next((row for row in _RUNTIME_PROFILES if row["id"] == profile_id), None)
    if profile is None:
        raise ValueError("RoboTwin 运行档位不存在")
    if profile["status"] != "ready":
        raise ValueError("RoboTwin 运行档位未就绪: " + profile["blockers"][0]["message"])
    variant = next((row for row in profile["variants"] if row["id"] == variant_id), None)
    if variant is None:
        raise ValueError("RoboTwin 资源变体未验证")
    runtime = RobotwinEvalRuntime(
        gpu_count=variant["gpus"], policy_worker_count=variant["policy_worker_count"],
        env_worker_count=variant["env_worker_count"], envs_per_worker=variant["envs_per_worker"],
        parallel_envs=variant["parallel_envs"],
        inference_micro_batch_size=variant["inference_micro_batch_size"],
        max_inflight_batches=variant["max_inflight_batches"],
        profile_id=profile_id, resource_variant_id=variant_id,
        scheduling_mode=profile["scheduling_mode"],
        probe_receipt_sha256=variant["probe_receipt_sha256"],
    )
    resources = {key:variant[key] for key in ("gpus", "cpus", "memory_gb", "shm_gb")}
    return runtime, resources


def derive_robotwin_runtime(gpus: int, parallel_envs: int) -> tuple[RobotwinEvalRuntime, dict[str, int]]:
    profile = PROFILES.get((gpus, parallel_envs))
    if profile is None:
        raise ValueError("RoboTwin GPU/并发组合尚未验证")
    runtime = RobotwinEvalRuntime(
        gpu_count=gpus,
        env_worker_count=profile["env_worker_count"],
        envs_per_worker=profile["envs_per_worker"],
        parallel_envs=parallel_envs,
        inference_micro_batch_size=profile["inference_micro_batch_size"],
    )
    resources = {"gpus":gpus, "cpus":profile["cpus"],
                 "memory_gb":profile["memory_gb"], "shm_gb":profile["shm_gb"]}
    return runtime, resources


def read_robotwin_runtime(plan: dict, *, submission: bool) -> RobotwinEvalRuntime:
    raw = ((plan.get("model") or {}).get("runtime") or {})
    if "policy_worker_count" not in raw:
        if submission:
            raise ValueError("历史 RoboTwin 评估计划必须重新预览后提交")
        gpus = raw.get("gpu_count", 1)
        per_gpu = raw.get("per_gpu_concurrency")
        parallel = raw.get("eval_batch_size")
        if type(gpus) is not int or type(parallel) is not int or parallel <= 0:
            raise ValueError("历史 RoboTwin 评估并发无效")
        if per_gpu is not None and parallel != gpus * per_gpu:
            raise ValueError("历史 RoboTwin 评估并发无效")
        workers = min(gpus, parallel)
        if parallel % workers:
            raise ValueError("历史 RoboTwin 评估并发无效")
        micro = min(4, parallel)
        while parallel % micro:
            micro -= 1
        return RobotwinEvalRuntime(
            gpu_count=gpus, env_worker_count=workers,
            envs_per_worker=parallel // workers, parallel_envs=parallel,
            inference_micro_batch_size=micro,
        )
    if raw.get("profile_id") or raw.get("resource_variant_id"):
        expected, _ = resolve_robotwin_runtime(
            str(raw.get("profile_id")), str(raw.get("resource_variant_id")))
        fields = RobotwinEvalRuntime.model_fields
        runtime = RobotwinEvalRuntime.model_validate({key:raw[key] for key in fields if key in raw})
        if runtime != expected:
            raise ValueError("RoboTwin 评估运行档位与资源变体不一致")
        return runtime
    fields = RobotwinEvalRuntime.model_fields
    runtime = RobotwinEvalRuntime.model_validate({key:raw[key] for key in fields if key in raw})
    verified, _ = derive_robotwin_runtime(runtime.gpu_count, runtime.parallel_envs)
    if runtime != verified:
        raise ValueError("RoboTwin 评估运行拓扑不是已验证档位")
    return runtime
