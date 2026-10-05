"""L3 Cross-Domain Consistency Diagnosis & Observation Drift Heatmap.

Design reference: docs/design/05-evaluation-delivery.md (sim2sim and sim2real)
Computes standardized distribution drift |Δμ|/σ per observation term across:
  - Isaac vs MuJoCo (sim2sim)
  - MuJoCo vs Real (sim2real)
  - Isaac vs Real (direct sim2real)

Identifies root cause dimension (IMU calibration, joint damping, contact dynamics)
and estimates divergence step t*.
"""

from __future__ import annotations
import math
from typing import Any, Dict, List, Optional, TypedDict


class DriftTermResult(TypedDict):
    term: str
    dim: int
    offset: int
    isaac_mean: float
    mujoco_mean: float
    real_mean: Optional[float]
    pooled_std: float
    sim2sim_drift: float      # |μ_isaac - μ_mujoco| / σ
    sim2real_drift: Optional[float] # |μ_mujoco - μ_real| / σ
    isaac2real_drift: Optional[float]
    severity: str             # "ok" (<0.3), "warn" (0.3~0.7), "critical" (>0.7)
    diagnostic_hint: str


class CrossDomainDriftReport(TypedDict):
    embodiment_env_id: str
    divergence_step: Optional[int]  # requires measured paired trajectories
    max_drift_term: str
    max_drift_value: Optional[float]
    passed_sim2sim_gate: Optional[bool]  # statistics alone cannot qualify a gate
    terms: List[DriftTermResult]
    summary: str


# Diagnostic knowledge base matching design 06.3
DIAGNOSTIC_RULES = {
    "projected_gravity": "重力投影项漂移显著 → 通常由 IMU 物理安装角或加速度计静态零偏未校准引起",
    "base_ang_vel": "角速度项漂移显著 → 通常由 陀螺仪滤波延迟或躯干惯量矩阵 (Inertia) 误差引起",
    "base_lin_vel": "线速度项漂移显著 → 通常由 足底打滑或线速度估计器漂移引起",
    "joint_pos": "关节角度项漂移显著 → 通常由 编码器零位偏移或初始站姿偏差引起",
    "joint_pos_rel": "相对关节位置漂移 → 通常由 零位偏移或目标命令缩放系数 (action_scale) 不匹配引起",
    "joint_vel": "关节速度项漂移显著 → 通常由 电机阻尼系数 (Damping) 或摩擦力模型不准确引起",
    "joint_vel_rel": "关节速度项漂移显著 → 通常由 电机阻尼系数 (Damping) 或摩擦力模型不准确引起",
    "last_action": "上一帧动作偏差显著 → 通常由 策略推理延迟 (Latency) 或控制环下发丢帧引起",
    "command": "速度指令偏差 → 请检查指令生成器分布",
}


def compute_observation_drift(
    embodiment_env_id: str,
    layout: List[Dict[str, Any]],
    sim_stats: Optional[Dict[str, Any]] = None,
    mujoco_stats: Optional[Dict[str, Any]] = None,
    real_stats: Optional[Dict[str, Any]] = None,
) -> CrossDomainDriftReport:
    """Compute standardized observation drift heatmap across simulation and real domains.

    Missing runtime statistics never produce synthetic measurements or gate approval.
    """
    terms: List[DriftTermResult] = []
    current_offset = 0
    max_drift = 0.0
    max_term_name = ""

    for item in layout:
        name = item.get("name", "unknown")
        dim = int(item.get("dim", 1))

        if not sim_stats or not mujoco_stats or name not in sim_stats or name not in mujoco_stats:
            current_offset += dim
            continue
        source, target = sim_stats[name], mujoco_stats[name]
        s_mean, m_mean, p_std = source.get("mean"), target.get("mean"), source.get("std")
        r_mean = (real_stats or {}).get(name, {}).get("mean")
        values = [s_mean, m_mean, p_std] + ([] if r_mean is None else [r_mean])
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in values) or p_std <= 0:
            current_offset += dim
            continue

        sim2sim = round(abs(s_mean - m_mean) / p_std, 3)
        sim2real = round(abs(m_mean - r_mean) / p_std, 3) if r_mean is not None else None
        i2r = round(abs(s_mean - r_mean) / p_std, 3) if r_mean is not None else None

        if sim2sim > max_drift:
            max_drift = sim2sim
            max_term_name = name

        if sim2sim < 0.3:
            sev = "ok"
        elif sim2sim < 0.7:
            sev = "warn"
        else:
            sev = "critical"

        hint = DIAGNOSTIC_RULES.get(name, f"{name} 观测分布存在跨域偏差")

        terms.append({
            "term": name,
            "dim": dim,
            "offset": current_offset,
            "isaac_mean": round(s_mean, 4),
            "mujoco_mean": round(m_mean, 4),
            "real_mean": round(r_mean, 4) if r_mean is not None else None,
            "pooled_std": round(p_std, 4),
            "sim2sim_drift": sim2sim,
            "sim2real_drift": sim2real,
            "isaac2real_drift": i2r,
            "severity": sev,
            "diagnostic_hint": hint,
        })
        current_offset += dim

    summary = (f"sim2sim 已有 {len(terms)}/{len(layout)} 项实测分布统计；最大漂移 {max_drift}σ。"
               "尚无配对轨迹发散步数与完整门禁证据，未执行放行验收。" if terms else
               "sim2sim 缺少实测跨域统计，尚未评估。")

    return {
        "embodiment_env_id": embodiment_env_id,
        "divergence_step": None,
        "max_drift_term": max_term_name,
        "max_drift_value": max_drift if terms else None,
        "passed_sim2sim_gate": None,
        "terms": terms,
        "summary": summary,
    }
