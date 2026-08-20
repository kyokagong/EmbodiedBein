from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from eb.domain.contract import Contract
from eb.domain.job import JobSpec
from eb.domain.result import CheckResult, Violation


@dataclass
class GateContext:
    dataset_versions: list[Any]
    base_model: Any | None = None
    onnx_io: dict[str, Any] | None = None
    deploy_yaml: dict[str, Any] | None = None
    firmware_n_joints: int | None = None
    calibration_expired: bool = False
    latency_ms: float | None = None


def g1_submit(spec: JobSpec, ctr: Contract, ctx: GateContext) -> CheckResult:
    v: list[Violation] = []
    bound = ctr.sim_bindings.isaac_task
    if spec.task and bound and spec.task != bound:
        v.append(
            Violation(
                path="task",
                expected=bound,
                actual=spec.task,
                hint="契约与任务不匹配，可能选错了 contract_id",
            )
        )
    for dsv in ctx.dataset_versions:
        compatible = getattr(dsv, "compatible_contracts", None) or dsv.get("compatible_contracts", [])
        if ctr.id not in compatible:
            v.append(
                Violation(
                    path=f"datasets[{getattr(dsv, 'id', dsv.get('id', '?'))}]",
                    expected=ctr.id,
                    actual=compatible,
                    hint="该数据集动作维度与本契约不符，需先做重定向",
                )
            )
    if ctx.base_model is not None:
        mid = getattr(ctx.base_model, "contract_id", None) or ctx.base_model.get("contract_id")
        if mid and mid != ctr.id:
            v.append(
                Violation(
                    path="base_model.contract_id",
                    expected=ctr.id,
                    actual=mid,
                    hint="基座模型契约与 Job 契约不一致",
                )
            )
    v.extend(ctr.self_assertions())
    return CheckResult(passed=not v, violations=v)


def g2_dataset(action_dim: int, state_dim: int, fps: float, ctr: Contract) -> CheckResult:
    v: list[Violation] = []
    if action_dim != ctr.action.dim:
        v.append(Violation("action_dim", ctr.action.dim, action_dim, "样本动作维与契约不符"))
    if ctr.observation.concat and state_dim != ctr.observation.dim:
        v.append(Violation("state_dim", ctr.observation.dim, state_dim, "样本状态维与契约不符"))
    if abs(fps - ctr.control.policy_hz) > 1e-6:
        v.append(Violation("fps", ctr.control.policy_hz, fps, "数据帧率与 policy_hz 不一致"))
    return CheckResult(passed=not v, violations=v)


def g3_model(io_in: int | None, io_out: int | None, ctr: Contract) -> CheckResult:
    v: list[Violation] = []
    if io_in is not None and io_in != ctr.observation.dim:
        v.append(Violation("onnx.inputs[0].shape[-1]", ctr.observation.dim, io_in, "策略观测维与契约不符"))
    if io_out is not None and io_out != ctr.action.dim:
        v.append(Violation("onnx.outputs[0].shape[-1]", ctr.action.dim, io_out, "策略动作维与契约不符"))
    return CheckResult(passed=not v, violations=v)


def g4_export(deploy: dict[str, Any], ctr: Contract) -> CheckResult:
    v: list[Violation] = []
    mapping = {
        "num_observations": ctr.observation.dim,
        "num_actions": ctr.action.dim,
    }
    if ctr.action.scale is not None:
        mapping["action_scale"] = ctr.action.scale
    for key, expected in mapping.items():
        actual = deploy.get(key)
        if actual != expected:
            v.append(Violation(f"deploy_yaml.{key}", expected, actual, "导出 YAML 与契约不一致"))
    return CheckResult(passed=not v, violations=v)


def g5_robot(ctr: Contract, firmware_n_joints: int | None, calibration_expired: bool, latency_ms: float | None) -> CheckResult:
    v: list[Violation] = []
    expected = ctr.real_bindings.firmware_n_joints or ctr.dof
    if firmware_n_joints is not None and firmware_n_joints != expected:
        v.append(Violation("firmware.n_joints", expected, firmware_n_joints, "把错误 DOF 策略推给实体机"))
    if calibration_expired:
        v.append(Violation("calibration", "valid", "expired", "标定过期，真机放行必须被拒"))
    if latency_ms is not None and latency_ms > ctr.control.max_policy_latency_ms:
        v.append(
            Violation(
                "latency_ms",
                f"<{ctr.control.max_policy_latency_ms}",
                latency_ms,
                "推理延迟超过契约上限",
            )
        )
    return CheckResult(passed=not v, violations=v)


def evaluate_gate_dsl(assertions: list[dict[str, Any]], actuals: dict[str, Any]) -> CheckResult:
    """Minimal gate DSL: {name, path, op, expected} against a flat actuals map."""
    v: list[Violation] = []
    for a in assertions:
        path = a["path"]
        op = a.get("op", ">=")
        expected = a["expected"]
        actual = actuals.get(path)
        ok = False
        if actual is None:
            ok = False
        elif op == ">=":
            ok = actual >= expected
        elif op == ">":
            ok = actual > expected
        elif op == "<=":
            ok = actual <= expected
        elif op == "<":
            ok = actual < expected
        elif op == "==":
            ok = actual == expected
        elif op == "!=":
            ok = actual != expected
        if not ok:
            v.append(Violation(path, f"{op} {expected}", actual, a.get("hint", a.get("name", ""))))
    return CheckResult(passed=not v, violations=v)
