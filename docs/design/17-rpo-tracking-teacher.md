# RPO 闭环跟踪教师（RLinf PPO）

> 2026-10-09用户最新约束：**本阶段禁止训练教师，也不使用教师生成数据替代动捕参考。** 本文保留历史实现设计；此前条件通过后继续训练的授权不再适用于教师分支。当前路线按[设计18](18-bfm-reference-prior-and-g1-rpo-retarget.md)执行。以下单卡预算与教师资格是历史记录，不是待执行计划。

本方案执行设计 16 的数据资格补齐分支。用户已授权单卡教师探索最多一小时；通过后自动继续四卡 BFM 1000 次 joint learner 更新及独立录像评估。资格失败不能启动正式 BFM。

教师是数据生产工具，不能冒充 BFM 模型。新接口 `ee_rpo_v2_tracking_23dof_v1`、任务 `tsk_rpo_tracking_teacher`、模板 `rpo_v2_bfm_zero_tracking_teacher`。沿用 BFM 的 native23、24 bodies、50/200Hz、无延迟、默认姿态和 gain，不扩大驱动范围。

演员 547 维：BFM 当前状态 52 + 实际上次动作 23 + 四帧历史 300 + 未来 1/5/10 控制步状态 156 + 世界根位置误差 3 + 相对姿态矩阵前两列 6 + 根速度误差 3 + 实际根线速度 3 + 参考相位 1。Critic 905 维 = actor547 + privileged358。网络沿用现有 PPO 的 ELU MLP 512/256/128，独立可训练 Gaussian std，初始化 residual mean=0/std=.10；演员/critic 分别 empirical normalization。RLinf 的 PPO worker、GAE、KL 调节、checkpoint、指标收集不另建训练循环。

相对姿态6固定为 `R_current^T R_reference` 的 `[r00,r01,r10,r11,r20,r21]`（前两列，按行展开）；未来状态52与实际状态52使用同一native23/wxyz codec，不混用xyzw。

RLinf 输出归一化残差 r。物理动作 `a=clamp((q_ref-next-default_q)/(.25*gain)+r,-1,1)`；记录 r 的 Normal log probability，history 记录实际 a，PPO likelihood 不混入确定性的参考动作。默认运行平面，不施加外力辅助、不在控制步中重置根位置。

参考来自已冻结的 100STYLE Neutral ID/FW/BW/SW 三秒映射试验：Job/Run/receipt/hash/CC BY 4.0 溯源保留。输入是运动目标，尚不是合格 BFM prior。参考只读挂载，提交及运行时复核 manifest、每个 NPZ、资产 hash、recipe、150 帧/50Hz/有限数值。复位一次写入参考根状态/q/dq；训练随机 clip 与起始帧，独立验证使用完整片段、固定新 seeds，不能用于训练或再筛选教师。

奖励每控制步积分 dt：`4 exp(-joint_MSE/.09)+4 exp(-root_position_sq/.04)+2 exp(-root_angle_sq/.25)+2 exp(-root_velocity_sq/.25)+alive1` 加现有 signed joint-limit/torque/action-rate penalties；跌倒额外 -2。终止维持原 BFM height .50/tilt .80 与碰撞；参考尾端是 truncation，保留真实 terminal observation。未来帧只在片段边界 clamp，不跨 clip。

单卡短训先验证 reset/history/动作/概率/normalization/checkpoint 和真实采样。随后有限训练；Job timeout ≤1h、max_retries=0，记录 GPU 时间及停止原因。独立验证复用设计16阈值（q .15rad、root .15m、orientation .25rad、接触足速 .30m/s、无提前终止/硬限位/非有限数）。输出逐 case 结果与视频。通过才能导出真实物理轨迹、派生新 DatasetVersion；未通过保留诊断产物，禁止 training_ready 升级。

验证的三个 held-out seeds 142/143/144 各在独立任务子进程中重新建立单环境，使 startup 摩擦、质量、COM、PD/armature 随机化确实绑定该 seed；每 seed 完整跑四片段，共12份录像。不能仅在 reset 前设置 seed 后宣称启动随机化独立。非有限 case 保留 finite=false/passed=false，无法计算的数字以 null 输出，不伪造零误差；停用坏 observation 的策略输入，完整保存失败证据。

反思一：参考 pose FK 合格不证明动态稳定；教师必须在真实物理中补偿。反思二：残差与实际动作必须分开记录，否则 PPO ratio 错误。反思三：三秒四片段通过也不足以证明多样性/来源分组独立；正式 prior 仍须补齐原始祖先分组、权利声明和独立验证证明，不能直接复制原始参考或评估轨迹。
