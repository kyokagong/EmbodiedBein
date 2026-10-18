"""Seed existing executable recipes into the operator environment catalog."""

from __future__ import annotations

from sqlalchemy import select

from eb.domain.environment_profile import profile_hash, validate_definition
from eb.domain.rlinf_form import TEMPLATES
from eb.store.models import EmbodimentEnvRow, EnvironmentProfile, EvalSuite


def _training_definition(tpl) -> dict:
    if tpl.id == "rpo_v2_flat_ppo":
        return {
            "task": "官方 RPO-Flat：按前后、横向速度及 heading/转向指令运动；20% 站立指令，每 10 秒重采样。",
            "reward": "RoboLab 6b1c3d9 的完整 RPORewardCfg；速度/yaw 跟踪、直立、足部、能耗与关节约束，按 0.02 秒计奖。",
            "failure": "沿用官方关键躯干接触终止和 20 秒超时；独立评估另检查高度、倾角、非有限状态和硬限位。",
            "success": "训练完成不代表通过；100 个独立官方指令分布案例中至少 80 个达到冻结的速度/转向跟踪判据。",
            "settings": {},
        }
    if tpl.id == "rpo_v2_amp_stand_walk_ppo":
        return {
            "task": "平地原地直立和 0.3 m/s 前向直行；手臂自由自然摆动。",
            "reward": "任务奖励与冻结训练动作先验的风格奖励合成；风格奖励在 PPO GAE 前写入。",
            "failure": "跌倒、硬限位或非有限状态终止；具体门限由训练 recipe 固定。",
            "success": "训练完成不代表通过；须在独立 Isaac 和 MuJoCo 套件验收实际 ONNX。",
            "settings": {},
        }
    if tpl.id == "rpo_v2_wbc_stand_walk_ppo":
        return {
            "task": "平地原地直立或以 0.3 m/s 前向直行，每种任务约占一半。",
            "reward": (
                "每步乘 0.02：速度跟踪 +3、yaw 跟踪 +1、直立 +0.5、上肢中立 +1；"
                "姿态误差 -1、跌倒终止 -200、站立偏差 -0.2、躯干偏差 -0.3；"
                "并保留能耗、关节限位、足部接触/滑动等固定项。"
            ),
            "failure": "机体高度低于 0.50 m、倾角超过 0.80 rad 或关键躯干接触则终止。",
            "success": "训练完成不等于任务成功；需使用独立直立直行评估套件验证。",
            "settings": {
                "target_speed_mps": 0.3,
                "stand_fraction": 0.5,
                "velocity_reward_weight": 3.0,
                "upright_reward_weight": 0.5,
            },
        }
    return {
        "task": tpl.label,
        "reward": "无环境奖励；使用监督学习损失。"
        if tpl.kind == "sft"
        else "由绑定的固定训练 recipe 定义，尚未登记逐项奖励。",
        "failure": "任务失败判据尚未在环境目录登记；请检查绑定的任务与 recipe。",
        "success": "训练作业完成不等于行为成功；需通过独立评估验证。",
        "settings": {},
    }


def training_definition(tpl, settings: dict | None = None) -> dict:
    definition = _training_definition(tpl)
    values = {**definition["settings"], **(settings or {})}
    values = validate_definition("train", tpl.id, {**definition, "settings": values})["settings"]
    definition["settings"] = values
    if tpl.id == "rpo_v2_amp_stand_walk_ppo" and values.get("initial_pose"):
        definition["task"] += "初态采用官方 A1 站姿；策略初始均值对应相同 PD 目标。"
    if tpl.id == "rpo_v2_wbc_stand_walk_ppo":
        speed = values["target_speed_mps"]
        stand = values["stand_fraction"]
        velocity = values["velocity_reward_weight"]
        upright = values["upright_reward_weight"]
        definition["task"] = (
            f"平地原地直立（比例 {stand:g}）或以前向 {speed:g} m/s 直行；"
            "上肢参考固定中立位，无推搡和横移/转向命令。"
        )
        definition["reward"] = (
            f"每步乘 0.02：速度跟踪 +{velocity:g}、yaw 跟踪 +1、直立 +{upright:g}、上肢中立 +1；"
            "姿态误差 -1、跌倒终止 -200、站立偏差 -0.2、躯干偏差 -0.3；"
            "并保留能耗、关节限位、足部接触/滑动等固定项。"
        )
    return definition


def _eval_definition(suite) -> dict:
    spec = suite.spec or {}
    if spec.get("benchmark") == "rpo_flat":
        return {
            "task": "100 个独立种子，每例 20 秒；保留官方速度/转向指令、20% 站立、70/30 地形及随机化。",
            "reward": "独立加载实际 ONNX；不使用训练回报判定成功。",
            "failure": "倒地、NaN、硬越限、提前终止、平移速度向量或转向跟踪误差超标。",
            "success": "每例平移向量 RMSE ≤0.20 m/s、yaw RMSE ≤0.30 rad/s 且完整保持直立；标准 100 例至少 80 例通过。",
            "settings": {},
        }
    if spec.get("benchmark") == "rpo_amp":
        return {"task":"偶数 seed 原地直立、奇数 seed 0.3 m/s 前向直行，各 20 秒。",
                "reward":"独立评估不计算训练奖励或判别器分数。",
                "failure":"倒地、非有限数值、硬越限、提前终止或未达到位移阈值。",
                "success":"直立漂移 ≤0.5 m；直行前进 ≥4 m。此 v1 套件只作诊断，不构成自然度正式验收。",
                "settings":{}}
    if spec.get("criterion") == "stand_walk" and suite.id.endswith("_v2"):
        return {
            "task": "偶数 seed 原地站立 20 秒；奇数 seed 以 0.3 m/s 前向直行 20 秒。",
            "reward": "独立评估不计算训练奖励。",
            "failure": "跌倒、非有限数值、提前终止或未达到任务阈值。",
            "success": "站立漂移不超过 0.5 m；直行前进至少 4 m，且全程直立。",
            "settings": {"stand_max_drift_m": 0.5, "walk_min_forward_m": 4.0},
        }
    if spec.get("benchmark") == "rpo_wbc":
        if spec.get("criterion") == "stability":
            return {
                "task": "在 Isaac Lab 或 MuJoCo 中保持身体稳定 1000 个策略步。",
                "reward": "独立评估不计算训练奖励。",
                "failure": "提前跌倒或出现非有限状态。",
                "success": "完整存活 1000 步，且无跌倒、无非有限状态。",
                "settings": {},
            }
        return {
            "task": "跟踪给定速度命令与上肢参考。",
            "reward": "独立评估不计算训练奖励。",
            "failure": "跌倒、非有限状态、硬限位或跟踪误差超标。",
            "success": "速度 RMSE ≤ 0.15，上肢 RMSE ≤ 0.1，且无跌倒或硬限位。",
            "settings": {},
        }
    if spec.get("benchmark") == "robotwin":
        tasks = [item.get("task_id") for item in spec.get("tasks", []) if isinstance(item, dict)]
        return {
            "task": "RoboTwin 固定案例：" + ("、".join(filter(None, tasks)) or suite.name),
            "reward": "独立评估不计算训练奖励。",
            "failure": "在冻结案例内未完成目标、超时或发生无效/基础设施失败；分别计数。",
            "success": "以 RoboTwin 环境逐案例 success 信号和完整 case 证据为准。",
            "settings": {},
        }
    if isinstance(spec.get("suite"), str) and spec["suite"].startswith("libero_"):
        return {
            "task": f"完成 LIBERO {spec['suite'].removeprefix('libero_')} 套件中的语言指令。",
            "reward": "独立评估不计算训练奖励。",
            "failure": "案例未完成目标、超时或无效；无效案例不计作成功。",
            "success": "以 LIBERO 环境逐案例成功信号和冻结案例完整性为准。",
            "settings": {},
        }
    return {
        "task": suite.name,
        "reward": "独立评估不计算训练奖励。",
        "failure": "以冻结套件逐案例判据为准；目录尚未登记更细的失败条件。",
        "success": "以冻结套件成功函数和完整案例证据为准。",
        "settings": {},
    }


def evaluation_definition(suite, settings: dict | None = None) -> dict:
    definition = _eval_definition(suite)
    values = {**definition["settings"], **(settings or {})}
    values = validate_definition("eval", suite.id, {**definition, "settings": values})["settings"]
    definition["settings"] = values
    if values and (suite.spec or {}).get("criterion") == "stand_walk":
        definition["success"] = (
            f"站立漂移不超过 {values.get('stand_max_drift_m', 0.5):g} m；"
            f"直行前进至少 {values.get('walk_min_forward_m', 4.0):g} m，且全程直立。"
        )
    if values.get("initial_pose"):
        definition["task"] += "初态采用官方 A1 站姿。"
    return definition


def suite_task_id(suite) -> str:
    spec = suite.spec or {}
    if spec.get("benchmark") == "rpo_flat":
        return "tsk_rpo_flat"
    if spec.get("benchmark") == "rpo_amp":
        return "tsk_rpo_amp_stand_walk"
    if spec.get("benchmark") == "rpo_wbc":
        return (
            "tsk_rpo_stand_walk"
            if spec.get("criterion") == "stand_walk"
            else "tsk_rpo_walk_upperbody_tracking"
        )
    if spec.get("benchmark") == "robotwin":
        return "tsk_robotwin_multitask"
    if isinstance(spec.get("suite"), str) and spec["suite"].startswith("libero_"):
        return f"tsk_libero_{spec['suite'].removeprefix('libero_')}"
    return "tsk_bhl_velocity_biped" if suite.embodiment_env_id == "ee_bhl_biped_12dof_v1" else ""


def suite_executable(suite) -> bool:
    spec = suite.spec or {}
    suite_name = spec.get("suite")
    return (
        spec.get("benchmark") in {"rpo_wbc", "rpo_amp", "rpo_flat", "robotwin"}
        or isinstance(suite_name, str)
        and suite_name
        in {
            "libero_spatial",
            "libero_object",
            "libero_goal",
            "libero_10",
        }
    )


async def seed_environment_profiles(session) -> None:
    env_ids = set((await session.scalars(select(EmbodimentEnvRow.id))).all())
    for tpl in TEMPLATES:
        if tpl.kind not in {"sft", "ppo"} or tpl.embodiment_env_id not in env_ids:
            continue
        id_ = f"env_train_{tpl.id}"
        if await session.get(EnvironmentProfile, id_):
            continue
        definition = validate_definition("train", tpl.id, training_definition(tpl))
        row = EnvironmentProfile(
            id=id_,
            name=tpl.label,
            usage="train",
            embodiment_env_id=tpl.embodiment_env_id,
            task_id=tpl.task_id,
            template_id=tpl.id,
            suite_id=None,
            definition=definition,
            content_hash="",
            status="draft" if tpl.id == "rpo_v2_amp_stand_walk_ppo" else "active" if tpl.ready else "deprecated",
        )
        row.content_hash = profile_hash(row)
        session.add(row)
    suites = (await session.scalars(select(EvalSuite))).all()
    for suite in suites:
        if (suite.embodiment_env_id not in env_ids or not suite.frozen
                or suite.id.startswith("suite_rpo_v2_wbc_") and suite.id.endswith("_stand_walk_v1")):
            continue
        id_ = f"env_eval_{suite.id}"
        if await session.get(EnvironmentProfile, id_):
            continue
        task_id = suite_task_id(suite)
        definition = validate_definition("eval", suite.id, evaluation_definition(suite))
        row = EnvironmentProfile(
            id=id_,
            name=suite.name,
            usage="eval",
            embodiment_env_id=suite.embodiment_env_id,
            task_id=task_id,
            template_id=None,
            suite_id=suite.id,
            definition=definition,
            content_hash="",
            status="draft" if (suite.spec or {}).get("benchmark") == "rpo_amp" else "active" if suite_executable(suite) else "deprecated",
        )
        row.content_hash = profile_hash(row)
        session.add(row)
