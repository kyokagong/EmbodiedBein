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
    if kind in {JobKind.EVAL_SIM2SIM, JobKind.DATA_SCAN, JobKind.DATA_SLICE, JobKind.REPORT_RENDER, JobKind.ASSET_VALIDATE}:
        return Estimate(
            verdict="fits",
            per_gpu_peak_gb=0,
            headroom_gb=host_vram_gb,
            confidence="analytic",
            basis={"formula": "no_gpu", "inputs": {}},
            verify="eb job submit --dry-run --probe-memory",
        )
    if kind in {JobKind.TRAIN_RL, JobKind.DATA_COLLECT, JobKind.EVAL_SIM, JobKind.EXPORT_POLICY}:
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
        basis={"formula": formula, "inputs": inputs, "safety": 1.25},
        verify="eb job submit --dry-run --probe-memory",
        alternatives=alts,
    )
