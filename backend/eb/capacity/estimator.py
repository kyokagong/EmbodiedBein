from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from eb.domain.job import JobKind, JobSpec

Confidence = Literal["measured", "analytic", "estimated", "interpolated"]
Verdict = Literal["fits", "fits_tight", "no_fit", "unknown"]


@dataclass
class Estimate:
    verdict: Verdict
    per_gpu_peak_gb: float
    headroom_gb: float
    confidence: Confidence
    basis: dict[str, Any]
    verify: str
    alternatives: list[dict[str, Any]] = field(default_factory=list)


def _optimizer_bytes(name: str) -> int:
    n = name.lower()
    if "8bit" in n or "8-bit" in n or "adamw8bit" in n:
        return 8
    if "sgd" in n:
        return 10
    return 16


def estimate_job(spec: JobSpec, host_vram_gb: float = 24.0, free_now: list[float] | None = None) -> Estimate:
    """L2 analytic estimator. L1 measured profiles can override later."""
    kind = spec.kind
    gpus = max(spec.resources.gpus, 1) if spec.resources.gpus else 1
    params = spec.params
    if params.get('operation') in {'ufo_native_import', 'ufo_native_export', 'ufo_native_bundle'}:
        return _finish(0., host_vram_gb, gpus, 'offline CPU native UFO inference package; no optimizer',
            {'num_envs': 0, 'optimizer_bytes': 0, 'gpu_peak_measured': False})
    if params.get('operation') in {'ufo_native_eval', 'ufo_native_live'}:
        return _finish(8. if spec.resources.gpus else 0., host_vram_gb, gpus,
            'single native UFO inference/simulation case; no optimizer; unmeasured reserve',
            {'num_envs': 1, 'optimizer_bytes': 0, 'gpu_peak_measured': False})
    template_id = params.get("template_id", "")
    if kind == JobKind.EVAL_SIM and template_id in {"rpo_v2_wbc_eval", "rpo_v2_amp_eval", "rpo_v2_flat_eval"}:
        # One Isaac environment uses GPU rendering; ONNX inference runs on CPU.
        # A video probe used 4.1 GiB. Reserve extra for longer episodes and driver variation.
        reserve = 8.0 if params.get("record_video") else 6.0
        return _finish(reserve * 1.08, host_vram_gb, gpus,
            "single Isaac environment + camera reserve; CPU ONNX inference",
            {"num_envs":1, "optimizer_bytes":0, "record_video":bool(params.get("record_video")),
             "reserve_gb":reserve})
    if (kind == JobKind.ASSET_VALIDATE and params.get("operation") == "probe_rpo_amp_env"
            and spec.embodiment_env_id == "ee_rpo_v2_amp_locomotion_23dof_v1"):
        return _finish(6.0 * 1.08, host_vram_gb, gpus,
            "two Isaac environments + RPO asset validation", {"num_envs":2, "optimizer_bytes":0})
    if kind == JobKind.TRAIN_RL and template_id in {"rpo_v2_amp_stand_walk_ppo", "rpo_v2_flat_ppo"}:
        overrides = dict(item.split("=", 1) for item in spec.overrides if "=" in item)
        num_envs = float(overrides.get("env.train.total_num_envs", 512))
        local_envs = num_envs / gpus if template_id == "rpo_v2_flat_ppo" else num_envs
        peak = (5.0 + local_envs * 0.0038 + 3.0) * 1.08
        return _finish(peak, host_vram_gb, gpus,
            "Isaac environments + RLinf PPO/AMP learner reserve",
            {"num_envs":num_envs, "envs_per_gpu":local_envs, "optimizer_bytes":16, "learner_reserve_gb":3.0})
    if kind == JobKind.TRAIN_RLFT and template_id.startswith("libero_"):
        from eb.domain.rlinf_form import build_overrides
        tuning = dict(item.split("=", 1) for item in build_overrides(template_id, gpus, spec.name or "estimate", spec.overrides) if "=" in item)
        micro_batch = max(1, float(tuning["actor.micro_batch_size"]))
        env_scale = max(1, float(tuning["env.train.total_num_envs"]) / 4,
                        float(tuning["env.eval.total_num_envs"]) / 4)
        reserve = 4.8 * micro_batch + 1.2 * env_scale * max(1, float(tuning["env.train.rollout_epoch"]))
        # Synchronous expert-only recipe offloads rollout before actor training.
        # Keep full model weights as a conservative bound; shard only optimizer/grad/master state.
        weight_bytes, trainable = 7_483_974_500, 432_853_025
        peak = (weight_bytes / 1e9 + trainable * 16 / gpus / 1e9 + reserve) * 1.08
        return _finish(peak, host_vram_gb, gpus, "full model + expert state/N + activation/environment reserve",
            {"weight_bytes":weight_bytes, "P_train":trainable, "N_gpus":gpus,
             "micro_batch":micro_batch, "reserve_gb":reserve, "offload":True})
    if kind == JobKind.EVAL_SIM and template_id == "robotwin_pi05_eval":
        runtime = (((params.get("eval_plan") or {}).get("model") or {}).get("runtime") or {})
        parallel = int(runtime.get("parallel_envs", gpus * 4))
        micro_batch = int(runtime.get("inference_micro_batch_size", 4))
        envs_per_worker = max(1, parallel // gpus)
        peak = (3_616_757_520 * 2 / 1e9 + 1.1 * micro_batch + 0.8 * envs_per_worker) * 1.08
        return _finish(peak, host_vram_gb, gpus,
            "provisional single policy weights + micro-batch activation + renderer reserve",
            {"P_total":3_616_757_520, "weight_bytes_per_parameter":2,
             "optimizer_bytes":0, "policy_replicas":1,
             "parallel_envs":parallel, "micro_batch":micro_batch,
             "envs_per_worker":envs_per_worker})
    if kind == JobKind.EVAL_SIM and template_id.startswith("libero_"):
        # Rollout replicas each own the model; adding GPUs does not shard these weights.
        peak = (3_616_757_520 * 4 / 1e9 + 6.0) * 1.08
        return _finish(peak, host_vram_gb, gpus, "replicated P_total*4 + inference/context",
            {"P_total":3_616_757_520, "optimizer_bytes":0, "V_inference_ctx":6.0})
    if kind == JobKind.TRAIN_SFT and params.get("template_id") == "libero_pi05_sft":
        # Counts verified against the pi05_libero meta model and public weight header.
        # ponytail: activation budget is analytic; replace with measured run peaks when available.
        trainable, frozen = 430_098_464, 3_186_659_056
        from eb.domain.rlinf_form import build_overrides
        tuning = dict(item.split("=", 1) for item in build_overrides(template_id, gpus, spec.name or "estimate", spec.overrides) if "=" in item)
        micro_batch = max(1, float(tuning["actor.micro_batch_size"]))
        activation_context = 4.8 * micro_batch + 1.2
        peak = ((trainable * 16 + frozen * 2) / gpus / 1e9 + activation_context) * 1.08
        return _finish(peak, host_vram_gb, gpus, "(P_train*16 + P_frozen*2)/N + activations/context",
            {"P_train":trainable, "P_frozen":frozen, "N_gpus":gpus, "micro_batch":micro_batch, "V_act_ctx":activation_context})
    if (kind == JobKind.DATA_CONVERT and params.get("operation") in {"download_model", "validate_lerobot_v3", "acquire_libero_benchmark", "acquire_robotwin", "convert_rpo_amp_official", "audit_rpo_amp_official", "validate_rpo_amp_motion"}) or kind in {JobKind.EVAL_SIM2SIM, JobKind.DATA_SCAN, JobKind.DATA_SLICE, JobKind.REPORT_RENDER, JobKind.ASSET_VALIDATE}:
        return Estimate(
            verdict="fits",
            per_gpu_peak_gb=0,
            headroom_gb=host_vram_gb,
            confidence="analytic",
            basis={"formula": "no_gpu", "inputs": {}},
            verify="eb job submit --dry-run --probe-memory",
        )
    if kind in {JobKind.TRAIN_RL, JobKind.DATA_COLLECT, JobKind.EVAL_SIM, JobKind.EXPORT_POLICY} and params.get("stack") != "rlinf":
        n_envs = float(params.get("num_envs", 4096))
        c_env = 0.0032 if "biped" in spec.embodiment_env_id else 0.0038
        peak = 5.0 + n_envs * c_env
        peak *= 1.08
        return _finish(peak, host_vram_gb, gpus, "n_envs*c_env + isaac_base", {"n_envs": n_envs, "c_env": c_env})
    p_train = float(params.get("params_b", 3.3e9 if kind != JobKind.TRAIN_SFT or "openvla" not in str(params).lower() else 7.5e9))
    if "lora" in str(params).lower():
        p_train = float(params.get("lora_params", 4e7))
        frozen = float(params.get("params_b", 7.5e9))
        static = frozen * 2 / gpus / 1e9 + p_train * 16 / gpus / 1e9
    else:
        b = _optimizer_bytes(str(params.get("optimizer", "adamw")))
        static = p_train * b / gpus / 1e9
    v_act = float(params.get("v_act_gb", 4.8))
    v_ctx = 1.2
    peak = (static + v_act + v_ctx) * 1.08
    return _finish(
        peak,
        host_vram_gb,
        gpus,
        "P_train*B/N + V_act + V_ctx + frag",
        {"P_train": p_train, "N_gpus": gpus, "V_act": v_act, "V_ctx": v_ctx},
    )


def _finish(peak: float, host: float, gpus: int, formula: str, inputs: dict) -> Estimate:
    head = host - peak
    if peak > host:
        verdict: Verdict = "no_fit"
    elif head < 1.5:
        verdict = "fits_tight"
    else:
        verdict = "fits"
    alts = []
    if verdict != "fits":
        alts.append({"change": "gpus +", "note": "增加卡数或改 LoRA / 8-bit 优化器 / 降 batch"})
        if "n_envs" in inputs:
            alts.append({"change": "num_envs: 降一档", "note": "Isaac 主旋钮"})
    return Estimate(
        verdict=verdict,
        per_gpu_peak_gb=round(peak, 1),
        headroom_gb=round(head, 1),
        confidence="analytic",
        basis={"formula": formula, "inputs": inputs, "safety": 1.08},
        verify="eb job submit --dry-run --probe-memory",
        alternatives=alts,
    )
