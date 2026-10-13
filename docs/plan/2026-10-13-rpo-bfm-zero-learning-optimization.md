# RPO BFM-zero 学习效果优化执行计划

> **For agentic workers:** 后续实施使用 `superpowers:executing-plans`，按任务顺序执行并逐项复核；本计划不要求并行 Agent。所有实施步骤使用复选框跟踪。用户要求不提交 Git，因此不执行暂存、commit 或 push。

**Goal:** 在 UFO BFM-zero + RLinf + MJLab 路线上，提高 RPO 的持续合格控制能力，以可归因的实验解决短回合、位置越界和足踝控制问题，再进入站立、行走及跟踪验证。

**Architecture:** 保留原生 FB/判别器/辅助 critic、单 learner 三 sampler、现有 Job/Run/ebd 生命周期。先补物理和奖励证据，按证据选择 RPO 专属辅助奖励或控制修复；数据课程、终止语义、动作接口、采样配比分开实验。复用现有 runner、checkpoint、评估和媒体通道，不另建训练平台。

**Tech Stack:** Python / PyTorch、固定 UFO revision `0b89378de02f77712072d983bc43ec2a46455dc0`、RLinf、MJLab / MuJoCo-Warp、Docker、现有 FastAPI 编排和 Rust ebd 数据面。

**Spec:** [设计22：失效分析与优化](../design/22-ufo-rpo-bfm-learning-diagnosis-and-optimization.md)，配合[设计18：专家参考与重映射](../design/18-bfm-reference-prior-and-g1-rpo-retarget.md)、[v13 执行记录](../progress/2026-10-13-ufo-mjlab-rsi-viability.md)、[v13 历史计划](2026-10-13-ufo-mjlab-rsi-viability.md)。设计真源仍为 `docs/design/`。

日期：2026-10-13，Asia/Shanghai。**状态：已按用户后续授权开始实施；实际证据及未完成项见[执行记录](../progress/2026-10-13-rpo-bfm-learning-optimization.md)。** 编写本计划时仅授权总结，后续用户已授权冻结与诊断、按证据推进。下文拟新增项是否实现以执行记录及测试/Job证据为准。实施时将新奖励定义同步到设计22，再落代码；不改写历史设计/实验结果来制造通过。

## Global Constraints

- 目标是模型效果；500 steps 指 **500 次 learner_joint_update**。一次联合更新包含六个 Adam 各执行一次，不等于六步，也不等于500个物理控制步。
- 用户确认暂按现有接口值讨论：踝关节 **27 Nm、8 rad/s**；速度成本基线从 **5.6 rad/s** 开始。其连续/峰值/额定/空载含义未知，不据此计算保证安全的制动距离。
- 保持现有资产、物理关节范围和严格验收约束。没有新证据不提高力矩、惯量或速度上限，不硬裁仿真 q/dq，不把 G1 的50 Nm/37 rad/s复制到RPO。
- 外部 `/data/kk/RLinf` 等仓只读；补丁只进入本仓受管镜像。Python 测试容器化；训练、物理诊断和评估都经过 API → Job → Run → ebd。
- 已发布接口、数据、模型、Suite、recipe 的行为不能原地改写。v12/v13及v13累计600更新终点保持冻结，禁止以新计划解锁旧模型续训。
- 不训练教师，不将固定PD跟随失败作为撤销 observation-only 专家数据资格的充分条件；参考动画和策略物理录像分别标记。
- 保留祖先级 train/validation/acceptance 隔离；重映射、镜像、重叠窗口不是独立样本。奖励标定和数据课程只使用 train。
- 主线先保留 GPU0 learner、GPU1–3各2048环境。缩环境的S1/S2仅为后续分支，不默认替代此前固定拓扑；多卡采样不称DDP。
- 当前阶段只设计有限实验。首次趋势最多一个对照加两个候选，不自动遍历全部候选、不自动启动5000–10000更新。
- 先准备周期检查、恢复和评估证据链。遵循历史专属协议“训练未达到触发条件时不做独立策略评估”；工程检查和训练诊断不能标为独立validation通过。
- 不提交Git，不覆盖工作区中已有改动。文档中所有成本权重为负，成本原值为非负；改变符号、单位或聚合方式必须派生奖励身份。

## Review Focus

1. **子步失败在末帧消失：** 第3子步越限/接触、第8子步恢复，必须保留首次事件及相应成本。任务2、4用合成轨迹锁定。
2. **终止自动重置污染奖励：** replay应保存终止前状态和成本；重复读观测不重复累计，partial reset不影响其他env。任务2、6验证。
3. **同形状、不同含义的恢复：** 两个配方都返回9列aux也不能跨语义恢复；奖励系数、数据课程、物理和采样身份均应参与恢复指纹。任务6验证。
4. **参考、RSI和latent不同步：** source/clip/frame、q/dq、active tracking window及B编码版本保持对应；不能用新latent解释旧参考跟踪奖励。任务3、5验证。
5. **固定更新预算掩盖时序不足：** K=100检查不得反复重建env导致轨迹一直不满250步；未完成episode按右删失统计，环境未恢复不能标exact。任务7、9验证。

---

## 1. 已确认事实与本轮取舍

### 1.1 为什么需要这份计划

v13新增500联合更新产生786,432条转移，每env名义推进128控制步，即2.56秒。末100更新窗口训练episode为12.71步，严格影子为7.30步；约86.76%的已完成episode终止转移包含训练位置违规。踝速度峰值42.58 rad/s时，电机已经反向制动饱和。它们证明持续控制失败，但尚不能将接触、惯量、动作请求和学习目标的贡献唯一分离。

UFO原生G1训练使用RSI：采样motion/time，写入完整root状态和关节q/dq；默认还以30%概率构造躺倒初态。原生MJLab关闭跌倒/接触/限位提前终止，保留恢复轨迹。RPO已有RSI和50% tracking，但有额外筛选与提前终止，二者训练MDP不同。**不把“已有RSI”列为待新增功能，也不将取消终止当成已确认修复。**

RPO专家输入为官方RPO参考与G1→RPO重映射参考，source权重0.5/0.5。专家8步序列、RSI候选和250步tracking窗口分别有资格检查；v13的3043个RSI候选和557个tracking起点不等于这些数量的独立动作。

### 1.2 当前奖励清单

实现落点为 `backend/eb/runners/rpo_ufo_isaac.py::raw_aux_rewards`，MJLab复用此函数；`rpo_ufo_mjlab.py::get_observations`把速度项替换为8子步平均。默认8项权重来自UFO FB preset，第9项由`bfm_ufo_agent.py::agent_config`添加。

| 辅助项 | 当前权重 | 当前定义/采样口径 | 本计划处理 |
|---|---:|---|---|
| `limits_dof_pos` | -10 | 超出90%软位置范围的距离和，控制步末帧 | 优先检查子步覆盖，再对照余量归一化 |
| `limits_dof_vel` | -25 | 8子步平均，Σ relu(abs(dq)/V−0.7)² | 首轮保持；踝V=8、起罚5.6 |
| `penalty_action_rate` | -0.1 | applied归一化目标相邻差平方和 | 先加实际目标变化诊断，暂不改权重 |
| `penalty_slippage` | -2 | 接触时足link原点三维速度模长，末帧 | 核对足底点、切向速度和接触时序 |
| `penalty_feet_ori` | -0.4 | 接触时局部重力xy分量模长 | 核对足底坐标及支撑/离地阶段；首轮保持 |
| `penalty_ankle_roll` | -4 | 左右踝roll角平方和，持续偏好零角度 | 可做权重置0的单因素消融 |
| `penalty_undesired_contact` | -1 | 保护body任一力分量绝对值>1N，末帧 | 与子步终止的力向量模长判据对账 |
| `penalty_torques` | 0 | 实际力矩平方和 | 记录但不参与优化；先不启用 |
| `limits_torque` | 0 | 实际力矩超过接口上限的量 | 执行器已裁剪，可能恒为0；不靠增权激活 |

v13的101–200→501–600窗口，原始加权位置成本−0.656→−1.084，速度−0.534→−0.345，滑动−0.577→−0.509，踝roll−0.132→−0.272。末窗口绝对成本占比约47%/15%/22%/12%。这些是replay批次均值且在RewardNormalizer之前，不能解释为Actor梯度占比或独立成功率。

Actor目标保持 `Q_FB + 0.05*w*Q_D + 0.02*w*Q_aux`，本地`w=min(mean(abs(Q_FB)),50)`。raw aux先加权求和，再由一个RewardNormalizer缩放。统一放大所有权重不保证同倍增强Actor约束；任何调权应同时检查归一化后成本和梯度。

### 1.3 主线与条件分支

主线：冻结证据 → 奖励/物理诊断 → 专家及RSI一致性 → 最多两个单因素候选 → 配方/恢复/周期检查 → 100→500趋势 → 满足条件才到1000或正式训练。

条件分支只有四类：固定锚点也失败时的动力学校准；动作大量落入slew等效区时的增量接口；有证据表明恢复轨迹缺失时的终止MDP对照；采样重用/时序不足时的S1/S2。每个分支有独立触发条件，不与奖励候选捆绑。

## 2. 文件职责与最小修改面

以下是后续实施落点，本次只创建本文。标为“新增候选”的文件仅在对应任务进入实施时创建。

| 职责 | 现有文件 / 拟新增文件 | 连带约束 |
|---|---|---|
| 当前奖励/子步统计 | `backend/eb/runners/rpo_ufo_isaac.py`、`rpo_ufo_mjlab.py` | 保留旧profile数值和223列trace；新诊断另存有schema的sidecar |
| 物理诊断 | `backend/eb/runners/rpo_ufo_mjlab_probe.py` | 复用现有受管operation；新模式同步adapter、capacity、collector |
| 专家/RSI/tracking | `backend/eb/runners/rpo_ufo_motion.py`、`rpo_ufo_contact.py`、`bfm_ufo_replay.py`、`bfm_ufo_rollout_context.py` | 祖先隔离、空池失败、窗口身份、partial reset |
| Actor/学习计数 | `backend/eb/runners/bfm_ufo_agent.py`、`bfm_ufo_worker.py` | 原生六优化器；同一transition仅加权一次 |
| 周期决策 | **新增候选** `backend/eb/runners/bfm_ufo_interval_gate.py`、`deploy/patches/rlinf/patch_rpo_ufo_learning.py` | 仅recipe局部的纯决策函数及owned镜像补丁；不建设通用调度器 |
| 配方/校验 | `backend/eb/rpo_bfm.py`、`domain/training_recipe.py`、`domain/rlinf_form.py`、`environment_catalog.py`、`adapters/rlinf.py` | 新recipe与Profile；preview/submit一致，不开放任意Hydra覆盖 |
| 恢复/产物 | `backend/eb/checkpoints.py`、`runners/bfm_ufo_checkpoint.py`、`bfm_ufo_artifact.py`、`api/routes_train.py`、`orchestrator/collect.py` | 完整fingerprint、旧模型可读、跨语义恢复拒绝 |
| 独立评估 | `backend/eb/runners/rpo_bfm_eval.py`、`api/routes_eval.py`、`adapters/mujoco.py`、`bfm_ufo_tracking.py` | 明确backend；**新增候选**`rpo_bfm_mjlab_eval.py`只实现MJLab执行分支 |
| 任务镜像 | **新增候选**`deploy/images/rlinf-ufo-learning.Dockerfile` | 固定基镜像digest、source receipt；训练/评估所需依赖明确 |
| 行为检查 | 现有`test_ufo_mjlab*`、`test_bfm_ufo_*`；新增`test_ufo_reward_diagnostics.py`、`test_ufo_reward_profile.py`、`test_ufo_learning_gate.py`、`test_ufo_learning_platform.py` | 每个新增文件仅服务对应非平凡逻辑 |

不预先修改Rust/UI。只有现有collector/展示无法传递必要证据时，记录具体缺口后修改对应现有分支。

## 3. Task 0：冻结实验输入与判据

**Files:** 读取设计22、v12/v13进度、`backend/eb/rpo_bfm.py`、`domain/training_recipe.py`、`checkpoints.py`；产物写入未来诊断Run输出目录，进度记录使用 `docs/progress/2026-10-13-rpo-bfm-learning-optimization.md`（执行开始后才创建）。

**Interfaces:** 消费历史manifest、checkpoint receipt和运行配置；产出`experiment-contract.json`。该JSON是本轮实验记录，不是新增API请求协议。必需字段为baseline refs/hashes、only_change、reward/physics/data/codec/sampling身份、stage_target、decision_interval、limits、budget、validation/acceptance绑定。

- [x] 记录现有工作区差异和受影响文件hash；保留v12 step524、v13 step600及原始日志。确认v13镜像digest仍为`sha256:e23281518d388565f4b1162dc9d2f21947543b2fd0a014b0288365be2788cee2`；不一致则等待解释。
- [x] 将27Nm、8rad/s记录为`interface_assumed`，将真实连续/峰值/空载曲线标为unknown；不填造硬件资料。
- [x] 从已有记录核对原门槛：末两窗口严格影子≥35控制步、全段物理峰值速度比≤2.5、首步训练终止≤0.1%；独立评估还要求末两窗口训练episode≥100步。这些不是100更新流程测试的立即淘汰条件，但不满足时不得放量或运行受限制的独评。
- [x] 冻结主线S0：3×2048 env、L=4、U=16、batch2048、replay4M、现有LR/网络/discount、50% tracking、source0.5/0.5；首轮only_change只能是一个已选择的奖励定义或权重。
- [ ] 将训练、严格影子、独立validation、最终acceptance四种结果分开；最终保留20秒/1000步、速度RMSE≤0.20m/s、yaw≤0.30rad/s、无fall/NaN/硬越限、至少80/100和完整媒体等既有标准。

**检查/出口：** 冻结记录能解析到真实文件/对象，所有来源hash可追溯；候选和基线差异可逐字段列出。没有选定only_change或预算时只完成诊断，不启动训练。

## 4. Task 1：完成物理事故与锚点对照

**Files:** 修改`rpo_ufo_mjlab_probe.py`、`rpo_ufo_mjlab.py`；必要时同步现有诊断operation对应的adapter、`capacity/estimator.py`、`orchestrator/collect.py`。测试扩展`backend/tests/unit/test_ufo_mjlab.py`和`test_ufo_mjlab_platform.py`。

**Interfaces:** 消费Task 0身份和冻结train初态；输出`physics-events.jsonl`及数组sidecar。每条事件绑定env/rank/episode/control_step/substep、source/clip/frame、当前和上一target、requested/applied、q/dq、实际/请求力矩、接触及原严格标志；可取得时附完整qpos/qvel、质量矩阵、接触雅可比和广义力。

- [x] 增加`test_event_keeps_pre_reset_state_and_integrated_force`：reset后仍能读取失败前事件；接触力与被积分的子步对应，不能取下一次forward后的力替代。
- [x] 在当前probe中添加有限事件捕获；每类最多32例，覆盖首步位置、后续位置、踝速度峰和正常落足对照，每例保留前后各最多8个控制步。另留全体轻量汇总，避免只采最大值。（v4全局每类最多30；首步位置类未出现，如实缺失。）
- [x] 通过受管Job对同一冻结初态分别运行固定目标锚点与原策略动作。新捕获状态缺solver warmstart等内部状态时标为“配对诊断”，不称精确重放。
- [ ] 只有接触贡献仍不清楚时追加无接触受控诊断；只有离散化敏感性仍不清楚时比较2.5ms与1.25ms，均保持50Hz并核对refsafe后的有效接触参数。现有8子步实现必须先支持16，不能只改dt。
- [ ] 输出事件级首次原因排名，分别报告逐关节位置余量、速度p95/p99/max、制动饱和方向和接触冲量；未取得的物理字段显式缺失。

**检查/出口：** 若固定锚点也普遍越界，进入动力学分支并停止奖励训练候选；若仅策略请求明显失控，继续Task 2并检查slew等效区。不得仅凭力峰降低宣布控制改善。

## 5. Task 2：建立奖励与价值的可追溯账目

**Files:** 修改`rpo_ufo_mjlab.py::get_observations`、`rpo_ufo_isaac.py::PhysicsInterval`、`bfm_ufo_worker.py`及必要的owned UFO诊断补丁；新增`backend/tests/unit/test_ufo_reward_diagnostics.py`。复用现有trace/metrics产物，不增加平台服务。

**Interfaces:** 保持既有9列`ufo_aux_rewards`和223列trace兼容；输出独立`reward-audit.json`及有字段名的sidecar。新增诊断不进入Actor观测。新字段需要经过RLinf传输时，同步owned schema patch、EnvOutput和replay边界测试。

- [x] 增加`test_aux_once_before_reset_and_observation_read_is_idempotent`：一个terminal transition只有一份原始aux；读观测两次无变化；重复子步不会双计数；partial reset只清相应env。（复用已有边界测试、补事件复制检查，v4运行时逐步核对读观测和partial reset；没有重复造同名测试。）
- [ ] 对全部9项记录raw值、权重×raw、加权和、归一化后标量；恒零项、非有限项、接触激活频率单独报告。验证raw加权和等于learner的normalizer输入。
- [x] 对位置、速度和接触记录末帧与子步口径差异。测试“第3子步位置越界、第8子步恢复”仍可在事件诊断中检出，不先改训练奖励。
- [x] 记录实际目标角度变化、slew触发率、requested/applied差、目标误差和饱和方向。保留replay=requested、last_action=applied的正确MDP语义。
- [ ] 在冻结train replay小批次上计算三项Actor目标的梯度范数与夹角、Q_aux预测与相同策略/latent的多步成本对照。诊断不执行optimizer.step、不更新norm、不消耗训练RNG；重复诊断后训练状态hash不变。
- [x] 以v13历史表复算成本比例作为账目回归；明确其为replay分布统计，不能与在线env窗口样本混算。（本轮另核验983040条累积replay；不把累计分布与旧末100窗口比例混算。）

**检查/出口：** 得到“哪项被触发、在哪个关节/相位触发、是否进入Q_aux、是否影响Actor”的连续证据。若记录缺口导致判断不可靠，先补证据，不直接增加奖励权重。

## 6. Task 3：审查专家、RSI和tracking的物理语义

**Files:** 修改或复用`rpo_ufo_motion.py`、`rpo_ufo_contact.py`、`bfm_ufo_replay.py::UFOExpertSlicer`、`bfm_ufo_rollout_context.py`；测试复用`test_bfm_ufo_replay.py`、`test_bfm_ufo_stable_context.py`、`test_ufo_mjlab_viability.py`。

**Interfaces:** 输入固定train参考；输出`reference-audit.json`，分别列expert8、RSI、tracking250三个集合的来源、祖先、动作标签、窗口数和拒绝原因。保持原q/dq和原valid mask不变。

- [ ] 增加`test_expert_rsi_tracking_sets_are_distinct_and_train_only`：长窗口不能跨valid洞或祖先split；RSI新增筛选不能静默改变expert8采样资格；缺某来源候选时明确失败，不改为全量池。
- [x] 同一RobotState分别走expert codec和MJLab reset，逐字段比较q/dq、root姿态/速度、重力、body局部坐标；root lift与history清零等允许差异单列，不能用零容差误判预期差异。（第二轮完成52/358对账、CPU原生FK与有限差分来源复算；区别已解释，未改参考，详见第二轮进度。）
- [ ] 按经核实的行为标签整理稳定支撑、重心转移、慢走、正常行走；无可靠标签标unknown，不从文件名自动推定。报告独立take数量，不把重叠窗口当新增数据。
- [ ] 核对RSI reference与本env tracking起点、相位、B编码版本对应；一个env reset不得改变另一env的cursor。50% tracking分区不被误当成RSI采样比例。
- [ ] 如证据支持简单初态课程，单独派生sampling manifest/hash，优先既有低法向速度、支撑有效候选；分别报告source与行为权重。课程变更与奖励变更不在同一首轮实验中发生。

**检查/出口：** 明确缺的是数据覆盖、状态编码、支撑可行性还是控制持续性。没有动态可执行证据时如实标未知；不补教师，不零掉参考速度，不把PD失败当参考失格。

## 7. Task 4：实现被选中的RPO物理奖励候选

**Files:** 修改`rpo_ufo_isaac.py`中的纯成本计算、`PhysicsInterval`和`rpo_ufo_mjlab.py`；新增`backend/tests/unit/test_ufo_reward_profile.py`。Task 2/3完成后只选择下列最多两个候选，其余保持未执行。

**Interfaces:** 拟新增纯函数`normalized_position_cost(q, joint_limits)`和`stance_slip_cost(point_velocity, normals, normal_forces, contact_mask)`，均输入Torch张量、输出每env非负成本；仅在被选中时实现。reward profile为recipe拥有的冻结元数据，默认路径仍返回原9项数值。

### 7.1 R-pos-time：先验证位置成本的时间覆盖

- [ ] 添加测试：8子步只有一个子步成本c时，平均值=c/8；全8步相同则均值=c；不能乘8或再乘dt。
- [ ] 将原线性位置成本从末帧改为8子步平均，保留90%软位置范围和权重−10。所有其他奖励、限位、终止、RSI与采样设置不变。
- [ ] 保留末帧、均值和峰值三个诊断口径；只有均值进入该候选训练。旧v13仍用末帧，不能重解释其历史指标。

### 7.2 R-pos-scale：时间口径固定后的余量归一化候选

设关节边界为`lo_j, hi_j`，`m_j=0.05*(hi_j-lo_j)`，`d_j=min(q_j-lo_j, hi_j-q_j)`：

```text
c_pos = mean_substeps Σ_j relu(1 - d_j / m_j)^2
```

- [ ] 测试单关节范围[-1,1]：q=0.9成本0，q=0.95成本0.25，q=1成本1，q=1.01成本1.21；同时测试非对称范围、负向边界、无效/零宽范围拒绝。
- [ ] 使用冻结train诊断样本估计旧、新成本均值；新权重取`-10 * mean(c_old) / mean(c_new)`，保存样本hash及系数，使初始平均成本尺度可比。任一均值为0/非有限时停止该标定，补有激活事件的train样本；不拿validation调系数。
- [ ] 系数在训练开始前冻结；它是成本定义改变后的尺度换算，不另做权重搜索。与R-pos-time比较，不同时增加向外速度项。

向外速度和制动余量先仅作为诊断：接近上限且dq>0、接近下限且dq<0分别记录。没有完整耦合制动能力证据时，不把`dq²/(2a)`写成硬安全边界。

### 7.3 R-slip：先对齐足底物理语义

对经核实的足底接触点，计算`v_point = v_origin + omega × r`，再计算相对地面速度的切向分量`v_t = v_rel - dot(v_rel,n)*n`。每足用有效接触点的压缩法向力加权平均`norm(v_t)`，再对双足求和，避免接触点数量改变成本尺度；当前静止平面地面速度为0。函数输入点速度/法向形状`[env, foot, contact, 3]`，法向力和有效mask为`[env, foot, contact]`；padding不参与计算。

- [ ] 测试接触点静止但link原点因转动而移动时成本为0；纯法向落足不产生切向滑动成本；摆动足不受罚；复制同一接触点并平分力后成本不变。有效接触法向非单位/非有限时拒绝；未接触足为0，已触发接触却无有效点/正法向力时报告证据错误，不能填0。
- [ ] 从冻结MJCF确定足底几何和局部坐标，实际接触点/法向读取完成子步的运行时数据，核对坐标和接触力符号。若Warp不能提供可靠接触点，不用link原点冒充；先输出“足link切向速度代理”诊断，真正接触点候选等待数据接口补齐。
- [ ] 第一对照只替换滑动成本定义，保留末帧时刻、原足部net-force模长>1N的触发口径和权重−2；需要子步平均或接触滞回时，另作后续单因素候选。
- [ ] 核对脚跟落地、脚尖离地和双支撑样本；接触法向/点可用性不达标时不启动此候选。

### 7.4 R-roll-off：最小踝roll消融

- [ ] 保持原`penalty_ankle_roll`原值及日志，只把该候选权重−4改为0；其余8项和normalizer机制不变。
- [ ] 测试权重变化只影响该项加权和，原始九列逐值保持，旧profile仍为−4。
- [ ] 评估严格越限、侧向姿态、足滑和持续控制，不因踝角变大就判退化。仅在消融支持原归零目标有害时，后续再设计基于RPO中性姿态的允许区间；不预先造新的姿态目标。

**任务检查/出口：** 每个候选先出现能捕获错误的失败测试，再实现并通过。首轮建议根据事故选择R-pos-time；第二候选从R-slip/R-roll-off中按证据选择。R-pos-scale需要独立对照预算，不自动成为第三个实验。

## 8. Task 5：约束辅助目标与任务目标的边界

**Files:** 读取/必要时扩展`bfm_ufo_agent.py`、`bfm_ufo_rollout_context.py`、`bfm_ufo_artifact.py`及owned UFO patch；测试放入`test_ufo_reward_profile.py`和`test_bfm_ufo_stable_context.py`。

**Interfaces:** 首轮所有新aux只依赖物理状态、控制动作及固定本体参数；不引入未编码的reference/command上下文。`Q_aux`继续由RPO在线转移学习。

- [ ] 增加`test_physical_aux_is_invariant_under_latent_relabel`：同一物理转移更换z不改变raw物理成本；确认当前0.8 relabel比例不使奖励条件失配。
- [ ] 保持FB、D、aux三项结构、discount=0.98、现有D reward裁剪和w上限；同时记录三项对Actor的实际贡献，不能用0.02解释为2%影响。
- [ ] 将固定站姿、速度命令和指定参考误差归入任务提示/独立评估。首轮不加入恒定存活奖励、所有z共用的固定站姿奖励或未声明参考相位的tracking奖励。
- [ ] 若后续确需任务条件aux，另行定义command/reference进入critic与replay的协议，并在z relabel时重算对应奖励或使用明确隔离的学习分支；该改动不混入本轮最小物理奖励修复。

**检查/出口：** 所有候选保持BFM多行为目标，没有通过压制走动来只学会站着，也没有用躺倒存活时间伪造稳定性。

## 9. Task 6：派生配方并贯通恢复与产物

**Files:** 修改第2节列出的配方、adapter、checkpoint、artifact、capacity落点；新增`deploy/images/rlinf-ufo-learning.Dockerfile`和`backend/tests/unit/test_ufo_learning_platform.py`。复用当前config文件，不复制整套生命周期。

**Interfaces:** 每个已选择候选拥有独立recipe/Profile、镜像digest、reward profile及系数hash。命名采用`rpo_bfm_ufo_agent_rlinf_mjlab_<candidate>_<revision>`，具体revision在实施时查重冻结；此处不占用假定未使用的v14。旧23DoF动作未变时复用原接口，只有动作含义改变才派生合同/codec。

- [ ] 增加`test_learning_preview_and_submit_freeze_same_recipe`：错误backend、镜像、数据、未批准override或任意候选名均被拒绝；不能仅改YAML绕过`validate_ufo_overrides`。
- [ ] 更新运行配置严格检查、agent配置、aux layout和采样worker消费，保证实际执行参数与预览一致。奖励名称相同但公式不同，也必须绑定不同reward身份。
- [ ] 增加`test_resume_rejects_reward_semantics_and_sampling_mismatch`：同形状9列、不同系数/聚合/数据课程均拒绝完整恢复；同配置可恢复六Adam、targets、norm、replay、RNG、z池及sampler状态。
- [ ] 奖励/MDP改变的首轮候选统一从相同seed的全新初始化开始，计数归零；不把v13完整状态带进新成本，避免旧replay/Q_aux污染比较。
- [ ] 为Task 7派生`ufo_agent_state_v3`，在计数中加入`pending_updates: int`与`pending_trigger: int|null`。pending为0时trigger为null；pending>0时小于U、trigger等于当前`next_trigger`且不大于`replay.received`。该块全部完成后才将`next_trigger`增加C，且只增加一次。同步后端发现、完整性校验、export和collector；旧v1/v2仍可按旧配方读取，不能缺字段却标为新配方完整恢复。`environment=not_available`仍如实保留。
- [ ] 构建owned镜像并生成source receipt；确认新profile实际进入镜像。恢复后每个Adam计数等于真实joint update，半写checkpoint拒绝。

**检查/出口：** API预览、实际镜像、checkpoint与模型metadata身份一致；v13的600→1100仍被拒绝。候选只能通过其新recipe运行。

## 10. Task 7：实现不破坏连续轨迹的周期决策

**Files:** 新增`bfm_ufo_interval_gate.py`、`test_ufo_learning_gate.py`；修改`bfm_ufo_worker.py`、checkpoint计数，以及owned `patch_rpo_ufo_learning.py`对RLinf runner的有限补丁。

**Interfaces:** 拟新增`decide_interval(stats: dict, protocol: dict) -> dict`，仅返回`continue/stop_adjust/wait_evidence/stage_complete`及理由、真实update、checkpoint和下一边界。函数不做IO/改配置。`protocol`来自Task 0，`stats`含窗口范围、证据完整性、物理指标和预算。

- [ ] 测试`test_boundary_splits_update_block_without_loss`：U=16的块经过96→100时执行4次，剩余12次明确保留，continue后继续；不能跳到112再把图标为100，也不能丢弃12次改变实际U/C。
- [ ] 将未执行更新块及所对应采样批次状态纳入checkpoint；同一transition不重复append，恢复后pending只消费一次。明确checkpoint时采样/权重同步屏障，避免在途数据丢失。
- [ ] 每100真实联合更新，在采样屏障完成原始指标、时序摘要、完整checkpoint及决策JSON；continue时保留当前env、history和tracking cursor，随后完成剩余更新块。
- [ ] 测试`test_continue_preserves_episode_and_tracking_cursor`、`test_missing_evidence_cannot_continue`、`test_old_interval_report_cannot_release_new_interval`及`test_cancel_preserves_last_complete_checkpoint`。
- [ ] stop/wait正常结束Run并释放GPU，续训重新走Job；不以杀容器冒充暂停。阶段100→500若使用新Run，明确环境重建RSI，不能把两段环境时间拼成连续episode。

**关键约束：** 当前源码没有验证过上述周期闭环，必须先补再承诺。若实现暂不可用，先停在流程验证；不能用每100更新重启env的做法声称已经验证长时序tracking，也不无条件预排后续区间。

## 11. Task 8：补齐MJLab独立评估与媒体出口

**Files:** 修改`rpo_bfm_eval.py`、`routes_eval.py`、`adapters/mujoco.py`、模型导出/collector；需要时新增`rpo_bfm_mjlab_eval.py`作为既有入口的backend分支。测试扩展`test_bfm_eval_bridge.py`、`test_bfm_eval_platform.py`及新profile测试。

**Interfaces:** 冻结模型、推理codec、Suite、case seed、backend/physics/reward身份；输出现有逐case报告、trace、视频与可访问索引。评估使用导出的实际推理包，不能偷偷换训练Actor或Isaac环境。

- [ ] 添加`test_mjlab_model_cannot_silently_evaluate_in_isaac`，对不支持的地形、噪声、reset或模型版本返回明确错误；单元测试不以假成功填充case。
- [ ] 工程上核对同状态Torch/导出推理与动作解码一致；验证录像可解码、FPS/时长、case/hash对应及失败视频保留。需要物理渲染的检查通过受管诊断Job完成，标为工程媒体检查。
- [ ] 在训练尚未满足Task 0独评触发条件时，只完成上述工程检查，不额外运行独立策略评估，也不标validation通过。
- [ ] 达触发条件后先用冻结validation case分别检查静态保持、held-out tracking、command平地；正常评估episode必须真实连续推进到其结束/失败，不能受训练K=100边界截断。
- [ ] 正式acceptance必须覆盖既有噪声/地形/时长/数量要求。MJLab尚不支持某项时报告能力缺口，不能用纯平地替代70/30地形分布或宣称sim2sim通过。

**检查/出口：** 能保留真实成功/失败媒体；未出现的类别如实注明。媒体损坏/缺失为incomplete，独立结果不能由训练影子值代填。

## 12. Task 9：按有界矩阵执行与选优

执行补充：首轮4个诊断Job已结束；用户后续授权的第二轮独立冻结为3个Job，各≤30分钟、总≤90分钟、无自动重试，见[第二轮进度](../progress/2026-10-13-rpo-bfm-learning-round2.md)。该额度已用满，物理出口仍失败；采样修复镜像只完成准备，训练预算未消费，不因诊断成功自动放行。

只有Task 0–8相应前置项完成后执行。用户后续已授权实施与按证据实验；以下是预算上限，实际运行及GPU占用见执行记录。

| 执行阶段 | 上限/配置 | 决策出口 |
|---|---|---|
| 证据整理、CPU行为测试 | 无训练更新 | 身份、数据隔离、成本定义一致 |
| 物理诊断 | 最多4个Job，每个≤30分钟、无自动重试；总≤2小时 | 确定首要控制/物理原因；等待证据时不转训练 |
| B0对照 | 原物理/奖励语义，补诊断的新recipe；seed42，100→累计500 | 为同条件候选提供对照，v13历史仅辅助解释 |
| R1、R2候选 | 最多2个，每个只有一个only_change，seed42，100→累计500 | 每100更新决策；失败即停止该候选 |
| 时序扩展 | 仅已有严格改善且预算不足以覆盖时序者，到累计1000 | 不以“还没走完250步”自动放行 |
| 独立seed复核 | 首轮筛出最多1个候选后，再冻结seed43、44及对照的独立预算 | 验证可重复性；不把单seed小差异称显著 |
| 正式训练 | 通过趋势、物理门槛及独立validation后另定累计5000/10000 | 逐周期决策，最后独立acceptance |

首轮B0+R1+R2合计≤3000联合更新、每配置≤2,000,000新转移及≤2小时，总训练≤6小时。所有运行产物预留上限合计200GiB；准入前按实际checkpoint/replay/录像大小核算，超过则减少候选或停止等待调整预算，不删除必需证据。独评/跨seed/正式训练不隐含包含在这个首轮上限中。历史v13约23分钟/新增500更新仅作估时参考，不承诺新诊断具有相同速度。

- [ ] 对新配置完成100真实更新工程检查：六Adam、checkpoint恢复、有限值、诊断和媒体链路。已完成且配置完全一致的证据可复用，不反复插入16/24更新短训。
- [ ] 100→500增加400更新；同一配置继续到1000再增加500。恢复计数用receipt回读，不能按采样轮数或优化器个数换算。
- [ ] 将100模型作为本配置趋势基线；每100更新记录训练/严格影子长度、首次违规原因、p95/p99/max、固定相位跟踪误差和reward/Q分项。新数据课程必须用共同冻结诊断case比较，避免只因起点更容易而“改善”。
- [ ] 初期零成功或短回合不单独取消100流程测试。NaN/Inf、错误身份/计数、空池、OOM、证据损坏立即停止；warmup例外仅限已冻结流程预算。
- [ ] 新配方在Task 0预先冻结100→300为最小趋势观察范围：200边界只有一个相对100基线的窗口时，若工程证据完整且未触发明确停止项，可按`minimum_observation`继续到300，不误称已证明改善。300开始检查101–200和201–300，之后滚动检查相邻两窗口。
- [ ] 最小观察范围后，趋势继续要求连续两个不重叠窗口的严格质量改善。采用设计22的候选研究阈值“严格长度相对100基线提高≥20%、位置/速度违规率不增加、固定相位误差不恶化”，在Task 0冻结；基线严格长度为0时相对比例不可用，返回wait_evidence，不能把0→0算改善。研究阈值不替代35步、峰值比2.5等原物理放量门槛。
- [ ] 最小观察范围结束仍因样本少/右删失而无法判断时返回wait_evidence；指标明确未改善或恶化则stop_adjust。不插值伪造测量点，不把同take相邻窗口当独立样本。跨seed复核前只称单seed迹象。
- [ ] 相同更新数、相同online转移数和相同墙钟三种比较分别报告。保留峰值并匹配样本量，不能只选择对候选有利的横轴或均值。

**训练输出：** 每个边界保留`interval-decision.json`、原始metrics、时序图、checkpoint receipt、失败事件和媒体索引。Task失败不自动重试；取消走平台API并检查GPU释放。

## 13. 条件分支：何时才调整主线以外的因素

| 分支 | 启动证据 | 最小动作 | 必须保持/验证 |
|---|---|---|---|
| 动力学/执行器 | 固定锚点也持续越限，广义力/接触证据支持 | 核对质量、惯量、约束和电机曲线；一次只改被证据指向的参数/实现 | 未知硬件参数不猜；不提高额定能力、不裁qvel |
| 动作接口 | 锚点可保持，策略大量请求被slew压成等效动作，critic梯度失真证据成立 | 按设计22比较围绕上一target的有界增量 | 新合同/codec/replay/Actor/target critic/导出/评估一起改；不恢复旧动作模型冒充同语义 |
| 终止与恢复 | 事件显示存在可恢复偏离，当前提前终止截断其后继经验 | 派生独立MDP对照，先冻结严重失稳/数值异常停止条件 | 原严格首次违规指标和acceptance不变；不照抄G1的30%躺倒；不能把躺着活满算成功 |
| RSI课程 | 专家/在线一致但初态相位集中在难支撑状态 | 仅改变train采样manifest，固定动作、奖励和终止 | 保留原参考及拒绝原因，单独对照，不把站立课程当多行为完成 |
| 采样配比 | 物理改善后仍显示重用不足/时序覆盖不足 | 研究S1/S2，下表参数在独立recipe中固定 | 不默认改变每卡2048约束；先补validator/准入；不同时改batch/LR |

采样候选沿用设计22，`ρ=U*B/C`，B=2048，L=4：

| 配比 | 每sampler / 总env | U / C | 稳态ρ | 新增500更新的名义每env跨度 |
|---|---:|---|---:|---:|
| S0当前 | 2048 / 6144 | 16 / 24576 | 1.333 | 约2.56秒 |
| S1候选 | 128 / 384 | 16 / 1536 | 21.333 | 约2.56秒 |
| S2候选 | 128 / 384 | 8 / 1536 | 10.667 | 约5.12秒 |

表中均为稳态近似，实际需计入warmup、尾块、reset和恢复。S1只改变规模及其必然派生采样间隔，S2在S1上只改变U；不能把它们与奖励同时修改后归因于某一个因素。当前`UFOLearner`只接受1024/24576采样间隔，必须同步worker、config、领域校验、capacity、完整恢复和计数测试，不能直接提交1536。

若保留S0，即使不失败，500更新的2.56秒也不足以完成5秒tracking窗口；每100更新重启更短。训练阶段时序覆盖与独立评估持续时长分别记录，不以总并行样本量替代。

## 14. 验证命令与交付检查

从仓库根目录执行。下列是后续实现后的CPU测试命令；本次编写计划没有运行这些测试。现有v13镜像已只读确认包含pytest、torch、mujoco、mjlab。新profile依赖owned UFO补丁时必须改用Task 6生成的冻结镜像digest，不能用旧镜像的通过结果证明新镜像已正确打包。

```bash
# 物理成本与边界行为；挂载只读，pytest临时文件在/tmp。
docker run --rm --network none --read-only --tmpfs /tmp:rw,size=1g \
  --cap-drop ALL --security-opt no-new-privileges \
  -v "$PWD:/workspace/repo:ro" -v /data/kk/assets:/data/kk/assets:ro \
  -e PYTHONPATH=/workspace/repo/backend -e PYTHONDONTWRITEBYTECODE=1 \
  -w /workspace/repo/backend \
  --entrypoint /workspace/isaaclab_extension_template/.venv/bin/python \
  eb/rlinf:rpo-bfm-ufo-mjlab-viability-v13 -m pytest -q -p no:cacheprovider \
  tests/unit/test_ufo_mjlab.py tests/unit/test_ufo_mjlab_viability.py \
  tests/unit/test_ufo_reward_diagnostics.py tests/unit/test_ufo_reward_profile.py \
  tests/unit/test_ufo_learning_gate.py

# 平台协议与恢复校验，使用既有backend测试镜像。
docker run --rm --network none -v "$PWD:/workspace:ro" -v /data/kk:/data/kk:ro \
  -e PYTHONDONTWRITEBYTECODE=1 eb/test:local \
  python -m pytest -q -p no:cacheprovider \
  tests/unit/test_ufo_learning_platform.py tests/unit/test_ufo_mjlab_platform.py \
  tests/unit/test_bfm_ufo_checkpoint.py tests/unit/test_bfm_eval_platform.py
```

预期相关测试PASS；因torch/mujoco缺失而SKIP不能算数值行为验证通过。首次红灯必须源于预期行为断言，不以缺包/import失败当有效失败测试。GPU真实probe、训练和评估不使用这些docker命令直启，走平台Job。

- [ ] 新增候选的修改清单覆盖调用者、config、tests、镜像、receipt、checkpoint、export及评估；旧v12/v13行为回归通过。
- [ ] 数据/代码/镜像/协议hash、真实更新数、GPU角色、预算与产物保留信息齐全。
- [ ] 图表有原始点、单位、窗口和缺失说明；训练影子与独立评估明确标记，恢复不连成虚假连续episode。
- [ ] 视频逐case可解码、hash与模型绑定；失败素材完整，无成功时明确写无成功案例。
- [ ] 更新实际执行进度；列出未实现分支、未通过门槛和剩余未知，不把计划复选框勾选当运行证据。

## 15. 编写时的多轮复核

**第一轮：能力与物理语义。** 核对RPO已有专家状态、RSI、tracking、Q_aux，避免重复建设；区分接口8rad/s与DC空载速度含义，保留硬件未知；不宣称RSI能保证支撑稳定。

**第二轮：奖励与因果。** 对账9项权重、子步/末帧差异和normalizer；将滑动三维速度、踝roll归零列为待证伪假设。位置时间聚合、归一化、滑动定义和权重消融分别实验，不把多个变更合成一个“奖励优化成功”。

**第三轮：学习与数据隔离。** 区分expert8、RSI、tracking250资格；补入latent relabel与任务奖励条件一致性。候选从头初始化，跨奖励语义拒绝完整恢复，train-only标定不接触acceptance。

**第四轮：执行闭环。** 明确K=100与U=16的尾块、checkpoint pending更新及持续env时序；保留未达标不独评的专属约束。新配方/缩环境参数当前未获validator支持，不把计划表写成可直接运行命令。

**第五轮：结论与成本。** 首轮最多对照+两候选，诊断/更新/转移/墙钟/磁盘均有边界；正式训练、跨seed和终止MDP分支另行冻结。最终质量依独立Suite，不能用loss、Job成功、放宽训练容差或更长躺倒时间代替。

本次完成范围仅为计划文件及文档检查；奖励优化、物理修复、周期决策、MJLab独评和模型质量提升均仍待实施验证。

## 执行证据纠正（2026-10-13）

详细阶段结果以[第五轮](../progress/2026-10-13-rpo-bfm-learning-round5.md)、[第六轮](../progress/2026-10-13-rpo-bfm-learning-round6.md)、[第七轮](../progress/2026-10-13-rpo-bfm-learning-round7.md)及各冻结契约为准。用户批准物理最后三轮总≤9Job/4.5小时/60GiB、每轮≤3Job/90分钟，允许证据驱动PD/接触/solver单因素，接口限位/奖励/RSI保持；失败及修正Job都占名额，无自动重试或Git提交。

第五轮实际配置审计证实：历史诊断过滤context_profile后使用legacy_global随机重复tracking IDs，而真实v13 worker使用environment_tracking_v10；旧诊断policy数值不能代表真实v13训练context。固定动作回放用于物理比较仍成立。实际step600 actor配置没有ankle_control_profile，kd25 mode覆盖分支未激活；不能仅凭该分支修复就认为正常训练采样有变化。进入训练前须证明候选在实际v13运行中产生有效且唯一的采样/更新比或时序覆盖差异，复用本计划已有配比候选与冻结流程，不运行无效名义对照。

第六/七轮物理诊断候选`S1`（摩擦锥）/`S2`（摩擦约束阻抗）属于`asset.validate`命名空间，**不是第13节的训练采样配比S1/S2**。物理必要条件、同配置确认、100更新工程门槛及独立模型验收仍分开；计划原始采样公式、原任务语义和质量门槛不因诊断Job成功而放宽。

获批三轮物理预算已关闭（末Run于2026-10-14 00:24:28 Asia/Shanghai结束）：实际8/9Job、83.836分钟Run时间/28.004GiB完整Run，失败包含在内。I1/S1/D1/S2/C2均未全部满足冻结物理条件；最后第9名额限定为C2全过后的确认，前提未满足而未使用。详见[第七轮关闭及后续入口](../progress/2026-10-13-rpo-bfm-learning-round7.md#c2独立关闭与本轮结论)。最大穿透与速度峰为不同case，后续先按现有保存状态分别核验几何/求解与制动，再另行冻结候选和有限预算；本轮没有启动第13节采样训练或奖励候选，也没有模型质量改善验收。

第八轮按用户新授权先复用六case，再独立冻结最多2Job/90分钟/12GiB：相对C2仅ground margin 0→3mm（A1）。首跑anchor满足全部必要条件，确认仍同镜像/初态/动作；六phase峰差≤0.31%却在旧env453出现真实5.750mm穿透，原5mm门槛失败。首跑该环境在0.38s因位置违规提前终止，旧/确认0.48s尾部被删失，不能把首跑通过视为稳定物理修复。实际2/2Job、23.409分钟Run/8.452GiB，停止晋级；全部状态/CPU几何/制动/图表/索引与Job/Run见[第八轮关闭记录](../progress/2026-10-14-rpo-bfm-learning-round8.md)。后续先冻结完整求解隐状态及接触/终止case证据，再选单因素；未启动第13节采样训练，接口限位/奖励/RSI保持，不提交Git。
