from __future__ import annotations

from typing import Any

from eb.domain.contract import EmbodimentEnv
from eb.gates.engine import g2_dataset, g3_model, g4_export, g5_robot

TERM_COLORS = {
    "velocity_commands": "#53d4ff",
    "arm_reference": "#ff7ac0",
    "base_ang_vel": "#55d6be",
    "projected_gravity": "#5b8cff",
    "joint_pos_rel": "#b28dff",
    "joint_vel_rel": "#ffc857",
    "last_action": "#ff884d",
    "state": "#b28dff",
}
SOURCE_COLORS = {
    "command": "#53d4ff",
    "imu": "#55d6be",
    "encoder": "#b28dff",
    "policy": "#ff884d",
}


def _color(term: str, source: str) -> str:
    return TERM_COLORS.get(term) or SOURCE_COLORS.get(source) or "#53d4ff"


def observation_layout_view(ctr: EmbodimentEnv) -> list[dict[str, Any]]:
    start = 0
    out: list[dict[str, Any]] = []
    for t in ctr.observation.layout:
        out.append(
            {
                "term": t.term,
                "dim": t.dim,
                "source": t.source,
                "index_start": start,
                "index_end": start + t.dim - 1 if t.dim else start,
                "color": _color(t.term, t.source),
            }
        )
        start += t.dim
    return out


def action_terms_view(ctr: EmbodimentEnv) -> list[dict[str, Any]]:
    if ctr.action.terms:
        return [
            {
                "name": t.name,
                "type": t.type,
                "dim": t.dim,
                "scale": t.scale,
                "indices": t.indices,
            }
            for t in ctr.action.terms
        ]
    return [
        {
            "name": "all",
            "type": ctr.action.type or "joint_position",
            "dim": ctr.action.dim,
            "scale": ctr.action.scale,
            "indices": list(range(ctr.action.dim)),
            "absolute": ctr.action.absolute,
        }
    ]


def joint_groups_view(ctr: EmbodimentEnv) -> list[dict[str, Any]]:
    i = 0
    groups: list[dict[str, Any]] = []
    items = list(ctr.joint_groups.items())
    if not items:
        return [
            {
                "name": "all",
                "count": len(ctr.joint_order),
                "index_start": 0,
                "joints": list(ctr.joint_order),
            }
        ]
    for name, spec in items:
        count = spec.get("count") if isinstance(spec, dict) else int(spec)
        joints = ctr.joint_order[i : i + count]
        groups.append({"name": name, "count": count, "index_start": i, "joints": joints})
        i += count
    if i < len(ctr.joint_order):
        rest = ctr.joint_order[i:]
        groups.append({"name": "rest", "count": len(rest), "index_start": i, "joints": rest})
    return groups


def frequency_view(ctr: EmbodimentEnv) -> dict[str, Any]:
    derived = round(1 / (ctr.control.sim_dt * ctr.control.decimation))
    return {
        "sim_dt": ctr.control.sim_dt,
        "decimation": ctr.control.decimation,
        "policy_hz": ctr.control.policy_hz,
        "control_hz": ctr.control.control_hz,
        "derived_hz": derived,
        "ok": derived == ctr.control.policy_hz,
        "max_policy_latency_ms": ctr.control.max_policy_latency_ms,
        "command_dim": ctr.control.command_dim,
    }


def assertion_table(ctr: EmbodimentEnv) -> list[dict[str, Any]]:
    layout_sum = sum(t.dim for t in ctr.observation.layout) * ctr.observation.history_length
    if ctr.action.terms:
        action_expected = sum(t.dim for t in ctr.action.terms)
        action_actual = ctr.action.dim
        action_name = "action.dim == sum(terms)"
    else:
        action_expected = ctr.dof
        action_actual = ctr.action.dim
        action_name = "action.dim == dof"
    derived = round(1 / (ctr.control.sim_dt * ctr.control.decimation))
    rows = [
        {
            "name": "observation.dim == sum(layout.dim) × history_length",
            "expected": layout_sum,
            "actual": ctr.observation.dim,
            "passed": (not ctr.observation.concat) or layout_sum == ctr.observation.dim,
        },
        {
            "name": action_name,
            "expected": action_expected,
            "actual": action_actual,
            "passed": action_expected == action_actual,
        },
        {
            "name": "1/(sim_dt × decimation) == policy_hz",
            "expected": derived,
            "actual": ctr.control.policy_hz,
            "passed": derived == ctr.control.policy_hz,
        },
        {
            "name": "len(joint_order) == dof",
            "expected": ctr.dof,
            "actual": len(ctr.joint_order),
            "passed": len(ctr.joint_order) == ctr.dof,
        },
    ]
    return rows


def contract_workspace(ctr: EmbodimentEnv) -> dict[str, Any]:
    return {
        "layout": observation_layout_view(ctr),
        "action_terms": action_terms_view(ctr),
        "joint_groups": joint_groups_view(ctr),
        "assertions": assertion_table(ctr),
        "frequency": frequency_view(ctr),
        "concat": ctr.observation.concat,
        "modalities": [m.model_dump() for m in ctr.observation.modalities],
        "morphology": ctr.morphology,
        "robot_family": ctr.robot_family,
        "sim_bindings": ctr.sim_bindings.model_dump(),
        "real_bindings": ctr.real_bindings.model_dump(),
        "history_length": ctr.observation.history_length,
        "action_scale": ctr.action.scale,
        "action_type": ctr.action.type,
        "action_absolute": ctr.action.absolute,
    }


def g3_cell(io: dict[str, Any] | None, ctr: EmbodimentEnv) -> dict[str, Any]:
    io = io or {}
    obs, act = io.get("obs"), io.get("act")
    if obs is None and act is None:
        return {"status": "none", "label": "未注册"}
    r = g3_model(obs, act, ctr)
    label = f"{obs} → {act}"
    if r.passed:
        return {"status": "ok", "label": f"{label} ✓"}
    return {"status": "fail", "label": label, "violations": [v.model_dump() for v in r.violations]}


def g2_cell(dsv: dict[str, Any], ctr: EmbodimentEnv) -> dict[str, Any]:
    compatible = dsv.get("compatible_envs") or []
    stats = dsv.get("stats") or {}
    action_dim = stats.get("action_dim")
    state_dim = stats.get("state_dim")
    fps = stats.get("fps")
    if action_dim is not None and state_dim is not None and fps is not None:
        r = g2_dataset(int(action_dim), int(state_dim), float(fps), ctr)
        if r.passed:
            return {"status": "ok", "label": f"state {state_dim} @{fps} ✓"}
        return {"status": "fail", "label": "不兼容", "violations": [v.model_dump() for v in r.violations]}
    if ctr.id in compatible:
        return {"status": "ok", "label": f"{ctr.id} ✓"}
    if not compatible and action_dim is None:
        return {"status": "none", "label": "—"}
    return {"status": "warn", "label": "需重定向"}


def g4_cell(bundle: dict[str, Any] | None, ctr: EmbodimentEnv) -> dict[str, Any]:
    if not bundle:
        return {"status": "none", "label": "—"}
    manifest = bundle.get("manifest") or {}
    deploy = manifest.get("deploy_yaml") or manifest.get("deploy") or {}
    if not deploy:
        return {"status": "none", "label": "无 .ebp"}
    r = g4_export(deploy, ctr)
    if r.passed:
        return {"status": "ok", "label": "YAML ✓"}
    return {"status": "fail", "label": "YAML 不符"}


def g5_cell(robot: dict[str, Any] | None, ctr: EmbodimentEnv) -> dict[str, Any]:
    if ctr.sim_bindings.isaac_task and "WholeBody" in (ctr.sim_bindings.isaac_task or ""):
        return {"status": "none", "label": "仅仿真"}
    if ctr.morphology == "fixed_hand":
        return {"status": "none", "label": "仅仿真"}
    if not robot:
        return {"status": "warn", "label": "未扫描总线"}
    cal = robot.get("calibration") or {}
    expired = bool(cal.get("expired") or cal.get("expires_at") == "expired")
    r = g5_robot(ctr, robot.get("firmware_n_joints"), expired, robot.get("latency_ms"))
    if r.passed:
        return {"status": "ok", "label": f"n={robot.get('firmware_n_joints') or ctr.dof}"}
    return {"status": "fail", "label": "G5 阻断", "violations": [v.model_dump() for v in r.violations]}


def gate_matrix(
    envs: list[EmbodimentEnv],
    *,
    versions: list[dict[str, Any]],
    models: list[dict[str, Any]],
    jobs: list[dict[str, Any]],
    bundles: list[dict[str, Any]],
    robots: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    def cells(fn) -> dict[str, Any]:
        return {c.id: fn(c) for c in envs}

    def g2_for(c: EmbodimentEnv) -> dict[str, Any]:
        chosen = {"status": "none", "label": "—"}
        for v in versions:
            cand = g2_cell(v, c)
            if cand["status"] == "ok":
                return cand
            if cand["status"] != "none":
                chosen = cand
        return chosen

    def g3_for(c: EmbodimentEnv) -> dict[str, Any]:
        chosen = {"status": "none", "label": "未注册"}
        for m in models:
            io = m.get("io_signature") or {}
            if not io.get("obs") and not io.get("act"):
                continue
            cell = g3_cell(io, c)
            if m.get("embodiment_env_id") == c.id and cell["status"] == "ok":
                return cell
            if cell["status"] == "ok":
                chosen = cell
            elif cell["status"] == "fail" and chosen["status"] == "none":
                chosen = cell
        return chosen

    def g4_for(c: EmbodimentEnv) -> dict[str, Any]:
        mine = [b for b in bundles if b.get("embodiment_env_id") == c.id]
        return g4_cell(mine[0] if mine else None, c)

    def g5_for(c: EmbodimentEnv) -> dict[str, Any]:
        mine = [r for r in robots if r.get("embodiment_env_id") == c.id]
        return g5_cell(mine[0] if mine else None, c)

    return [
        {"gate": "G2", "title": "数据集 dsv（G2）", "cells": cells(g2_for)},
        {"gate": "G3", "title": "ONNX 导出（G3）", "cells": cells(g3_for)},
        {"gate": "G1", "title": "Job 提交（G1）", "cells": cells(lambda c: g1_cell(jobs, c))},
        {"gate": "G4", "title": "策略包 YAML（G4）", "cells": cells(g4_for)},
        {"gate": "G5", "title": "真机固件（G5）", "cells": cells(g5_for)},
    ]


def g1_cell(jobs: list[dict[str, Any]], ctr: EmbodimentEnv) -> dict[str, Any]:
    mine = [j for j in jobs if j.get("embodiment_env_id") == ctr.id]
    if not mine:
        bound = ctr.sim_bindings.isaac_task
        return {"status": "ok" if bound else "none", "label": "task 匹配" if bound else "—"}
    bad = 0
    for j in mine:
        spec = j.get("spec") or {}
        params = spec.get("params") or {}
        task = params.get("task") or spec.get("task")
        bound = ctr.sim_bindings.isaac_task
        if task and bound and task != bound:
            bad += 1
    if bad:
        return {"status": "fail", "label": "task 写错会 400"}
    return {"status": "ok", "label": "task 匹配"}
