"""Seed existing executable recipes into the operator environment catalog."""

from __future__ import annotations

from sqlalchemy import select

from eb.domain.environment_profile import profile_hash, validate_definition
from eb.domain.rlinf_form import TEMPLATES
from eb.store.models import EmbodimentEnvRow, EnvironmentProfile, EvalSuite


def _training_definition(tpl) -> dict:
    if tpl.id == 'rpo_bfm_ufo_agent_rlinf_mjlab_4gpu_viability_v13':
        definition = _training_definition(next(t for t in TEMPLATES if t.id == 'rpo_bfm_ufo_agent_rlinf_mjlab_4gpu_v12'))
        definition['task'] += ' RSI检查当前与20ms外推q均距原位置边界至少1mrad；仅筛选，参考q/dq不改。'
        definition['failure'] += ' 训练位置终止容差±0.02rad；严格影子±0.001rad与MJCF物理限位不变。'
        definition['success'] = '从头100更新验证流程，完整状态恢复新增500更新至600；预算完成不自动扩训，训练未达标不独立评估。'
        return definition
    if tpl.id == "rpo_bfm_ufo_agent_rlinf_mjlab_4gpu_v12":
        return {"task": "RPO23DoF原生MJCF/MJLab/MuJoCo-Warp；GPU0 learner，GPU1–3各2048环境；435M UFO六优化器，Batch2048、Replay4M；独立环境250步Tracking。",
            "reward": "九项raw aux；八个物理子步速度罚均值，权重-25；Actor裁剪1，其余5，w<=50，D lr3e-6/reward[-2,2]。",
            "failure": "400Hz原生物理/50Hz控制；MJCF质量惯量碰撞掩码，零延迟DC motor PD，踝Kd2.5、armature0.01；RSI与运行时几何一致，足法向速度<=0.15m/s；1.3倍训练速度终止，原额定限位严格影子。",
            "success": "新后端从头16→24完整恢复后新增500更新，最多2小时/无重试；训练未达标不独立评估。原physics_hz=200为历史codec元数据，实际物理频率由本Profile明确。", "settings": {}}
    if tpl.id=='rpo_bfm_ufo_agent_rlinf_isaac_4gpu_compliance_v11':
        definition=_training_definition(next(t for t in TEMPLATES if t.id=='rpo_bfm_ufo_agent_rlinf_isaac_4gpu_tracking_contact_kd_v10'))
        definition['task']+=' 柔顺地面接触60000N/m、600Ns/m；RSI支撑腿膝>=0.15rad、髋pitch<=0.10rad，不限制摆动腿，只筛选不改q/dq。'
        definition['reward']+=' 子步速度raw罚relu(r-0.7)^2+8relu(r-1)^2，保留8帧均值与-25一次加权。'
        definition['success']='新16→24完整恢复后新增500更新；末两100更新窗口严格影子>35、全段物理速度比<2.5等门禁通过，才允许同配方524→3524新增3000更新/4h。训练未达标不独立评估。'
        return definition
    if tpl.id == "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_tracking_contact_kd_v10":
        definition=_training_definition(next(t for t in TEMPLATES if t.id=="rpo_bfm_ufo_agent_rlinf_isaac_4gpu_tracking_contact_v10"))
        definition["task"]+=" 配对接触诊断选定踝关节Kd2.5，原Kp40/27Nm限幅/目标slew保持，无新增低通。"
        return definition
    if tpl.id == "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_tracking_contact_v10":
        return {"task":"RPO23DoF完整435M网络；GPU0 learner、GPU1–3各2048环境；batch2048、CPU replay400万。固定50%独立环境跟随250步train专家片段，与RSI起点绑定；512m地面覆盖。",
            "reward":"九项UFO aux；速度罚为8个已完成物理子步的归一化均值，软阈值0.70、系数-25。原生六优化器；Actor梯度裁剪1，其余5；w<=50，D lr3e-6/reward[-2,2]。",
            "failure":"train接触点|v_normal|<=0.15m/s；物理400Hz、控制50Hz。探索速度终止1.3倍，原位置/力矩/宏观跌倒约束保持；原严格标准影子存活与所有子步违规独立记录。",
            "success":"四卡16→24完整状态恢复后新增500联合更新；最多2小时/无自动重试。训练不足不独立评估，不改变正常行走验收门槛。","settings":{}}
    if tpl.id == "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_ground_v9":
        definition = _training_definition(next(t for t in TEMPLATES if t.id == "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_contact_v8"))
        definition["task"] += " 512m离线地面；启动核对实际网格+所有RSI足碰撞网格包络与20m余量。"
        definition["failure"] += " 每步记录世界XY与足部保守覆盖余量，越界拒绝继续采样。"
        return definition
    if tpl.id == "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_contact_v8":
        definition = _training_definition(next(t for t in TEMPLATES if t.id == "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_reachable_v7"))
        definition["task"] += " RSI仅train接触候选、足底2mm间隙、有限root上移；写入后刷新FK，专家数据不改。"
        definition["failure"] += " 显式PD增益保持，physics dt=2.5ms/decimation8/50Hz；物理子步硬限位锁存，接触/高度/倾角原因分列。"
        return definition
    if tpl.id == "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_reachable_v7":
        return {"task": "RPO v2有限URDF动作目标，原始q/dq与观察零点保持；6144环境，每个sampler2048；435M原生网络、batch2048、replay100万，每24576新转移触发16次联合更新。",
            "reward": "九项raw aux，limits_dof_vel=sum relu(abs(dq)/Vmax-0.70)^2，权重-25仅加权一次；判别奖励截[-2,2]，D lr=3e-6；Actor正则尺度上限50。",
            "failure": "真实硬位置/速度/力矩终止保持；target slew每步0.5×Vmax；Actor裁剪1，其余五Adam裁剪5，非有限梯度失败。",
            "success": "训练/完整恢复资格16→24后新增500联合更新，最多2小时且不自动重试；训练指标不足不启动独立评估。正常行走尚未验证。", "settings": {}}
    if tpl.id == "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_scale6144_v6":
        definition = _training_definition(next(t for t in TEMPLATES if t.id == "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_slew_velocity_v5"))
        definition["task"] = definition["task"].replace("batch1024", "batch2048") + " 6144环境，每个sampler2048；每24576新转移触发16次联合更新。"
        definition["success"] = "先16→24完整恢复与独立资格，再增加5000联合更新；Job最多6小时，段末独立评估。"
        return definition
    if tpl.id == "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_slew_velocity_v5":
        definition = _training_definition(next(t for t in TEMPLATES if t.id == "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_slew_v4"))
        definition["reward"] += " 第九项limits_dof_vel=sum relu(abs(dq)/Vmax-0.8)^2，原生agent固定权重-5，仅加权一次。"
        return definition
    if tpl.id == "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_slew_v4":
        return {"task": "RPO v1应用动作codec、train RSI；原生完整2048×6网络、batch1024、replay100万，关闭tracking覆盖。",
            "reward": "原生八项raw aux保持，action-rate以实际执行动作计算；replay保留请求动作。",
            "failure": "safe-v2硬终止保持；每步PD target限速为0.5×硬Vmax，reset投影anchor并保留实际q/dq。超盒首步残差和实际dq仍需测量。",
            "success": "先16→24资格、完整回读、独立速度/held-out评估和媒体；target限速不代表正常行走验收。",
            "settings": {}}
    if tpl.id == "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_stable_v3":
        return {"task": "RPO v1 codec、仅train RobotState RSI；原生F/actor/critic/aux critic 2048×6，D 1024×3，B 256×1；batch1024、replay100万。",
                "reward": "八项原生raw aux与权重保持；关闭rollout_expert_trajectories，保留expert学习、mixed latent和每100步原生提示更新。",
                "failure": "平地/无噪声；继承safe-v2位置/速度/力矩硬限位及真实terminated，拒绝初始已越限参考帧。",
                "success": "先16→24完整恢复、独立ONNX验证和媒体资格；正常保存或loss下降不代表正常行走。",
                "settings": {}}
    if tpl.id == "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_safe_v2":
        return {"task": "RPO v1 codec、仅train RobotState RSI；原生UFO agent及四卡拓扑不变。",
                "reward": "八项原生raw aux与权重不变；flat/无噪声保持单因素对照。",
                "failure": "继承高度/倾斜/接触终止；追加与验收相同的位置/速度/力矩硬限位(容差0.001)，RSI拒绝已越限参考帧。",
                "success": "先16→24完整恢复资格；训练正常结束不代表行走合格。每步记录requested/applied/target/速度/力矩及越限。",
                "settings": {}}
    if tpl.id in {"rpo_bfm_ufo_agent_rlinf_isaac_v1", "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_v1"}:
        return {"task": "RPO v1 codec；仅 train 原始 RobotState 参考初态；RLinf 管理采样，固定 UFO FBcprAuxAgent 更新。无教师。",
            "reward": "八项原始 aux 不乘 dt、不预加权；UFO 原生权重只应用一次。平地、无噪声/延迟，native norm 每次 replay 更新。",
            "failure": "高度 <0.50m、倾角 >0.80rad 或关键躯干接触终止；20s 超时保留 terminal bootstrap。缺少 raw 状态/映射直接拒绝。",
            "success": "单段至绝对 learner_joint_updates 目标后正常保存释放；资格运行最多24更新、1h。独立录像/恢复/GPU资格尚未验证，不代表正常行走。",
            "settings": {}}
    if tpl.id == "rpo_v2_bfm_zero_tracking_teacher":
        return {"task":"100STYLE 冻结参考条件下的 RLinf PPO 残差动作跟踪，平地、无物理辅助。",
                "reward":"q/root position/orientation/velocity 指数跟踪奖励与 alive，乘 0.02；signed limit/torque/action-rate penalties，跌倒 -2。详见设计17。",
                "failure":"高度低于 .50m、倾角超过 .80rad 或关键躯干接触；参考尾端 truncation，保留真实终态。",
                "success":"作业完成不代表教师合格；独立 seeds 完整片段须达到冻结跟踪门限，才能生产正式 BFM prior。", "settings":{}}
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
    if spec.get("benchmark") == "rpo_bfm":
        return {
            "task": f"冻结的{spec.get('planned_cases', 100)}个独立case，每例20秒；保持官方指令、地形和随机化。",
            "reward": "独立加载实际checkpoint导出的ONNX，不以训练回报判断成功。",
            "failure": "倒地、非有限状态、硬限位、提前终止或速度/转向误差超标。",
            "success": "每例速度向量RMSE≤0.20 m/s、yaw RMSE≤0.30 rad/s，无跌倒/非有限数/硬限位且完成1000步；验收至少80/100。",
            "settings": {},
        }
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
    if spec.get("benchmark") == "rpo_bfm":
        return "tsk_rpo_bfm_pretrain"
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
        spec.get("benchmark") in {"rpo_wbc", "rpo_amp", "rpo_flat", "rpo_bfm", "robotwin"}
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
        from eb.domain.training_recipe import RECIPES, recipe_profile_id
        id_ = recipe_profile_id(tpl.id) if tpl.id in RECIPES else f"env_train_{tpl.id}"
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
        if (suite.spec or {}).get("benchmark") == "rpo_bfm":
            # Earlier BFM profiles have an empty task binding; retain them unchanged.
            id_ += "_binding_v2"
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
