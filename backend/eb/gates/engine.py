from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from eb.domain.contract import EmbodimentEnv
from eb.domain.job import TASK_REQUIRED_KINDS, JobKind, JobSpec
from eb.domain.result import CheckResult, Violation
from eb.domain.task import TaskSpec


@dataclass
class GateContext:
    dataset_versions: list[Any]
    base_model: Any | None = None
    onnx_io: dict[str, Any] | None = None
    deploy_yaml: dict[str, Any] | None = None
    firmware_n_joints: int | None = None
    calibration_expired: bool = False
    latency_ms: float | None = None
    task: TaskSpec | None = None


_OVERRIDE_BLOCKLIST = ("observation.dim", "action.dim", "joint_order", "policy_hz")


def g1_submit(spec: JobSpec, env: EmbodimentEnv, ctx: GateContext) -> CheckResult:
    v: list[Violation] = []
    amp_probe = (spec.kind == JobKind.ASSET_VALIDATE
                 and spec.params.get("operation") == "probe_rpo_amp_env"
                 and env.id == "ee_rpo_v2_amp_locomotion_23dof_v1")
    if env.status != "active" and spec.kind != JobKind.DATA_CONVERT and not amp_probe:
        v.append(
            Violation(
                path="embodiment_env.status",
                expected="active",
                actual=env.status,
                hint="草稿或已废弃的本体环境不能提交 Job",
            )
        )
    if spec.kind in TASK_REQUIRED_KINDS:
        if not spec.task_id:
            v.append(Violation("task_id", "tsk_*", None, "训练/评估必须引用任务"))
        elif ctx.task is None:
            v.append(Violation("task_id", spec.task_id, None, "未知任务"))
        else:
            if ctx.task.status != "active":
                v.append(Violation("task.status", "active", ctx.task.status, "任务未发布"))
            if env.id not in ctx.task.compatible_env_ids:
                v.append(
                    Violation(
                        path="task.compatible_env_ids",
                        expected=env.id,
                        actual=ctx.task.compatible_env_ids,
                        hint="任务与本体环境不兼容",
                    )
                )
            gym = ctx.task.gym_id()
            if spec.task and gym and spec.task != gym:
                v.append(
                    Violation(
                        path="task",
                        expected=gym,
                        actual=spec.task,
                        hint="params.task 必须等于 Task 的 gym_id",
                    )
                )
    for raw in spec.overrides:
        if any(tok in raw for tok in _OVERRIDE_BLOCKLIST):
            v.append(
                Violation(
                    path="overrides",
                    expected="不改接口数字",
                    actual=raw,
                    hint="overrides 不得改 observation.dim / action.dim / joint_order / policy_hz",
                )
            )
    for dsv in ctx.dataset_versions:
        compatible = getattr(dsv, "compatible_envs", None) or dsv.get("compatible_envs", [])
        if env.id not in compatible:
            v.append(
                Violation(
                    path=f"datasets[{getattr(dsv, 'id', dsv.get('id', '?'))}]",
                    expected=env.id,
                    actual=compatible,
                    hint="该数据集动作维度与本本体环境不符，需先做重定向",
                )
            )
    if ctx.base_model is not None:
        mid = getattr(ctx.base_model, "embodiment_env_id", None) or ctx.base_model.get("embodiment_env_id")
        if mid and mid != env.id:
            v.append(
                Violation(
                    path="base_model.embodiment_env_id",
                    expected=env.id,
                    actual=mid,
                    hint="基座模型与 Job 的本体环境不一致",
                )
            )
    v.extend(env.self_assertions())
    return CheckResult(passed=not v, violations=v)


def g2_dataset(action_dim: int, state_dim: int, fps: float, env: EmbodimentEnv) -> CheckResult:
    v: list[Violation] = []
    if action_dim != env.action.dim:
        v.append(Violation("action_dim", env.action.dim, action_dim, "样本动作维与本体环境不符"))
    if env.observation.concat and state_dim != env.observation.dim:
        v.append(Violation("state_dim", env.observation.dim, state_dim, "样本状态维与本体环境不符"))
    if abs(fps - env.control.policy_hz) > 1e-6:
        v.append(Violation("fps", env.control.policy_hz, fps, "数据帧率与 policy_hz 不一致"))
    return CheckResult(passed=not v, violations=v)


def g3_model(io_in: int | None, io_out: int | None, env: EmbodimentEnv) -> CheckResult:
    v: list[Violation] = []
    if io_in is not None and io_in != env.observation.dim:
        v.append(Violation("onnx.inputs[0].shape[-1]", env.observation.dim, io_in, "策略观测维与本体环境不符"))
    if io_out is not None and io_out != env.action.dim:
        v.append(Violation("onnx.outputs[0].shape[-1]", env.action.dim, io_out, "策略动作维与本体环境不符"))
    return CheckResult(passed=not v, violations=v)


def g4_export(deploy: dict[str, Any], env: EmbodimentEnv) -> CheckResult:
    v: list[Violation] = []
    mapping = {
        "num_observations": env.observation.dim,
        "num_actions": env.action.dim,
    }
    if env.action.scale is not None:
        mapping["action_scale"] = env.action.scale
    for key, expected in mapping.items():
        actual = deploy.get(key)
        if actual != expected:
            v.append(Violation(f"deploy_yaml.{key}", expected, actual, "导出 YAML 与本体环境不一致"))
    return CheckResult(passed=not v, violations=v)


def g5_robot(env: EmbodimentEnv, firmware_n_joints: int | None, calibration_expired: bool, latency_ms: float | None) -> CheckResult:
    v: list[Violation] = []
    expected = env.real_bindings.firmware_n_joints or env.dof
    if firmware_n_joints is not None and firmware_n_joints != expected:
        v.append(Violation("firmware.n_joints", expected, firmware_n_joints, "把错误 DOF 策略推给实体机"))
    if calibration_expired:
        v.append(Violation("calibration", "valid", "expired", "标定过期，真机放行必须被拒"))
    if latency_ms is not None and latency_ms > env.control.max_policy_latency_ms:
        v.append(
            Violation(
                "latency_ms",
                f"<{env.control.max_policy_latency_ms}",
                latency_ms,
                "推理延迟超过本体环境上限",
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
        ok = True
        if op == ">=":
            ok = actual is not None and actual >= expected
        elif op == "==":
            ok = actual == expected
        elif op == "<=":
            ok = actual is not None and actual <= expected
        if not ok:
            v.append(Violation(path, expected, actual, a.get("hint", "gate dsl failed")))
    return CheckResult(passed=not v, violations=v)
