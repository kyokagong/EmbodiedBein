# UFO/RPO 长程 Tracking 与接触动力学对齐 Implementation Plan

> **For agentic workers:** 使用 `superpowers:executing-plans` 逐任务执行；步骤用复选框记录。创建计划时仅获规划授权；用户随后明确授权实施、验证与500次新增四卡联合更新，现按该授权执行。禁止 Git 暂存和提交；训练未达标不独立评估。

**Goal:** 在 RPO 23DoF、IsaacLab、RLinf 和 UFO 原生 FB-CPR-aux 链路中恢复可靠的环境级专家时序提示，补齐接触与奖励反馈，扩大有效探索时间，并以可比较的训练指标决定是否继续。

**Architecture:** 复用现有一个 learner、三个 sampler、六个原生优化器及 Job/Run/ebd 生命周期。环境级 tracking 状态、子步奖励、RSI 候选和探索终止条件使用新的冻结版本；严格验收标准保持不变。先补诊断和恢复，再做单因素物理对照与 500 次新增联合更新，最后按实测资源准入扩大 replay。

**Tech Stack:** Python、PyTorch、UFO `FBcprAuxAgent`、RLinf、IsaacLab/PhysX；现有 MuJoCo 几何检查、容器测试、原子 checkpoint 和平台指标通道。

**Spec:** 用户本轮建议及本文件的冻结决策；设计依据为 `/data/kk/EmbodiedBein/docs/design/20-ufo-bfm-rlinf-integration.md`、`/data/kk/EmbodiedBein/docs/design/21-training-runtime-contracts.md`、`/data/kk/EmbodiedBein/docs/design/19-bfm-staged-walking-training.md`。关键基线、参数、判据均在本文展开，执行者不需要从历史报告补猜实验配置。

日期：2026-10-12。状态：**实现、容器验证、配对诊断、四卡16→24恢复及24→524新增500更新完成；训练未达标，不独立评估**。第1节保留既有ground-v9基线；新增结果在文末执行记录中分列，不能当作正常行走验收。

## Global Constraints

- 每次四卡训练固定 GPU0 learner、GPU1–3 sampler；**每张采样卡 2048 环境，总计 6144**。这不是四个 learner 或 DDP。
- 首轮 batch 保持 **2048**，保留已经对齐的完整网络；不同时增加网络、batch、学习率和采样频率。
- Python/测试/物理诊断在容器中执行；真实采样、资格更新和训练必须经过 API → Job/Run → ebd。不得宿主直接起训练脚本或由控制面 subprocess 跑训练。
- 不修改外部只读仓 `/data/kk/RLinf`、`/data/kk/Berkeley-Humanoid-Lite`、`/data/kk/berkeley_ego_data`；补丁只在本仓与任务镜像中生效。
- 不覆盖已经发布的本体接口、数据、模型、EnvironmentProfile、recipe、评估 Suite 和产物。派生版本记录原版本、代码/镜像/数据/配置 hash。
- 专家窗口、RSI 候选及提示构造仅使用 train split，保留祖先分组隔离和两个来源各 0.5 的权重；不得用 validation/acceptance 筛起点、调优或填 replay。
- 不引入教师控制器，不换成 PPO，不重写原生六个 loss。辅助项只通过现有 UFO aux 链路进入优化器。
- **训练未达标不启动独立 validation、held-out 或 acceptance。** 物理排障和恢复资格属于诊断/训练检查，不登记成独立评估成绩；报告明确评估为 `not_run`。
- 用户已授权按本计划实施与训练；不得 Git 暂存或提交。所有本地文件路径使用绝对路径。

## Review Focus

1. 某一个环境 reset、另一个环境到第 250 步、权重同步或重复读观测，均不能重置其他活动环境的专家片段或 cursor。由任务 2 测试覆盖。
2. 超速只发生在物理子步而步末已恢复时，当步奖励仍必须包含它；不能计入 reset 前的旧接触力，不能重复累计最后一帧。由任务 1、3 测试覆盖。
3. 足 link 原点/质心速度不等于足底碰撞点速度；相位过滤不能把正常摆动足也当作支撑足清除，不能导致正权重来源为空后静默降级。由任务 4 测试覆盖。
4. 放宽探索速度终止阈值不能被记为严格存活提升，也不能影响评估阈值；历史训练状态不兼容新语义时必须拒绝精确恢复。由任务 5、7 测试覆盖。
5. 大 replay 在保存、恢复和三 sampler tracking 缓存同时存在时不能超出准入 RAM；不完整 checkpoint、混合代际分片或部分 rank 状态不能发布。由任务 6、7 测试覆盖。

## 1. 已知事实、合理建议与需修正的判断

### 1.1 冻结起点：ground-v9 的正式 500 更新

| 项目 | 已有结果/配置 |
| --- | --- |
| TrainJob / Job / Run | `trn_8bdabc160cd04d2bb100` / `job_5157c51b35464239b2b6` / `run_13cb5781b3944be88906` |
| ModelVersion / checkpoint | `mv_f1c2fd66a7f6402fa7a1`；原生累计 524 更新，资格 16→24 后新增 500 |
| recipe / profile | `rpo_bfm_ufo_agent_rlinf_isaac_4gpu_ground_v9` / `rpo_bfm_ufo_isaac_4gpu_rsi_ground_v9` |
| 接口与维度 | `ee_rpo_v2_bfm_23dof_v2`；actor 375、FB 733、action 23、latent 256 |
| 网络 | 435,116,484 参数；F/Actor/Q_D/Q_aux 宽 2048、6 个 residual 块；D 1024×3；B 256×1 |
| 采样和更新 | 6144 环境，每 round 4 控制步产生 24,576 transitions；每触发 16 次联合更新；batch 2048 |
| replay | CPU FIFO 容量 1,000,000；最终实际 860,160 条，正式新增 786,432 条，**本段尚未发生容量淘汰** |
| 控制/物理 | 50Hz；control dt 0.02s；physics dt 0.0025s、decimation 8；horizon 1000 控制步 |
| 地面 | 512m×512m，三 sampler 实际网格均覆盖；正式最小足部保守余量 193.260m、越界 0 |
| episode_len 五个 100 更新窗口均值 | 7.130 → 10.087 → 11.034 → 12.026 → 13.271 步；末点 13.498 步，即 0.270s |
| 速度极值 | 控制步末最大 45.985rad/s；全局物理子步标量最大 67.525rad/s |
| 已定位右踝案例 | `right_ankle_pitch_joint`，round13、episode step0；dq −1.139→+45.985rad/s；目标变化 −0.08rad，符合现有 slew |
| 接触侧证 | 上述案例右足 link 初态竖直速度 −0.524m/s；碰撞网格最低点 +7.151mm→−0.508mm；当步右足竖直净力峰 10,049.84N |
| 首步终止 | 正式段 366/73,200 = 0.50%；首步足竖直净力 p95 573.69N、p99 1319.12N、最大 10,049.84N |
| 子步/边界错配 | 21,733 条转移在子步触发速度锁存终止、但步末速度已低于硬阈值 |
| reset/数值 | 完整 retained replay 中 87,980 次 reset 均未超物理目标盒；138 项标量均有限；六个优化器裁剪后 Actor≤1，其余≤5 |
| 独立评估 | 未运行；尚无正常行走通过证据 |

正式新增样本折合每环境 128 个控制步、2.56s 物理交互。这是探索量不足的事实，但不能据此断言所有损失下降都是“虚假收敛”，也不能保证追加更新就会行走。

45.985rad/s 有具体关节和轨迹定位；67.525rad/s 只有全局标量，**不能将后者也标为右踝极值**。竖直净力不是逐碰撞点法向力或接触冲量；当前数据无法唯一分离 PD、碰撞求解和全身耦合的因果贡献。

### 1.2 官方对齐核对

固定对比 UFO revision `0b89378de02f77712072d983bc43ec2a46455dc0`，不随着官网 main 变化偷偷换版本。

- 官方 **FB preset** 启用 mixed rollout、250 步专家 tracking、比例 0.5，网络宽度/深度与当前完整网络一致。基础 `FBAgentTrainConfig` 的默认值与 FB preset 不同，比较应以实际 preset 为准。[固定 FB preset](https://github.com/Roboparty/UFO/blob/0b89378de02f77712072d983bc43ec2a46455dc0/humanoidverse/agents/presets/fb.py)
- 原生 tracking 对连续帧做 B 编码和最多 8 帧前向均值，得到逐相位的 `z`。原实现遇到任一环境周期归零会重新选择整批 tracking 环境和轨迹，并使用可重复的随机索引；**恢复正确 tracking 后，合法的相邻 z 不要求余弦等于 1**。[固定 rollout 源码](https://github.com/Roboparty/UFO/blob/0b89378de02f77712072d983bc43ec2a46455dc0/humanoidverse/agents/fb/agent.py)
- 官方入口使用 MJLab/MuJoCo-Warp，默认 1024 环境、5,120,000 replay 容量按 rank 配置，总样本预算为全局值；官方多个 rank 同步学习与本项目单 learner/三个 sampler 的资源和更新语义不同，不能只复制 env/replay 数字。[固定训练入口](https://github.com/Roboparty/UFO/blob/0b89378de02f77712072d983bc43ec2a46455dc0/humanoidverse/train.py)
- 官方环境桥包含 `limits_dof_vel` 的计算，但默认 FB preset 只选择八项 aux；“完全没有速度项”不准确。桥还显式断言关闭多项 HumanoidVerse 终止开关；“官方一律只有高度/倾角/保护接触终止”也不能当作已验证事实。后续按最终 MJLab 配置核对，不推断 MuJoCo 保证无冲击。[固定 MJLab 环境桥](https://github.com/Roboparty/UFO/blob/0b89378de02f77712072d983bc43ec2a46455dc0/humanoidverse/agents/envs/humanoidverse_mjlab.py)

当前剩余主差异为：tracking 关闭；RPO/PhysX 的接触与显式 PD；自定义 slew/reset 锚点；CPU transition FIFO 而非官方 trajectory replay；RPO 自定义 history；九项 aux 和稳定化保护。网络容量已对齐，不再用“模型太小”解释 ground-v9 的失败。

### 1.3 本轮采纳与调整

| 建议 | 决策 |
| --- | --- |
| 环境粒度 tracking、恢复专家 rollout | 采纳，并增加 RSI 片段/相位绑定和完整状态保存；不复制全 sampler 重抽行为 |
| 子步速度罚积分或峰值 | 采纳 **时间归一化积分/均值** 为优化输入，峰值作为独立诊断，避免无意放大八倍奖励 |
| RSI 碰撞点法向速度≤0.15m/s | 采纳为候选筛选门槛；针对近地接触候选点，保留正常摆动足和原始速度 |
| 踝 LPF 与增加 Kd | 分开进行单因素对照，不预先保证消除接触飞车 |
| 放宽探索速度硬终止 | 先采用训练专属 1.3×课程，保留原严格阈值的影子统计；完全取消速度终止仅作为有界后备对照 |
| replay 扩至 400–512 万 | 采纳容量目标，先修保存/恢复内存复制和容量准入，不直接在旧配额下扩容 |
| 撤掉裁剪、奖励保护以恢复原生灵敏度 | 本阶段不做；原生核心已复用，保护效果的收益/损害需要后续单因素证据 |

现有显式 DelayedPD **已经有力矩限幅**，脚部名义 Kp=40、Kd=2、armature=0.01、effort=27Nm、velocity=8rad/s；腿部额定速度 25rad/s、effort=120Nm。启动随机化继续影响实际 Kp/Kd、惯量、摩擦和恢复系数，不能把名义增益或地面 μ=1 当作所有环境的实值。

已修复的上肢目标盒问题不再自动归因为肘关节零点/符号错误；先保留 v2 关节语义，不改 URDF 原点。当前 replay 没有满，“早期优质样本已被洗出”不是本段已证实根因，扩容主要针对后续更长训练。

## 2. 执行顺序与边界

按 **诊断与子步反馈 → RSI 接触筛选 → 环境级 tracking → PD 单因素对照/探索课程 → replay 恢复与容量 → 四卡资格和 500 更新观察** 执行。代码任务可以各自验证，但正式训练不能同时更改全部参数后笼统归因。

最终集成候选用新的 recipe/profile、源回执和恢复 fingerprint。已发布 ground-v9 只作为冻结对照。不同 MDP/奖励/控制/容量版本间默认不允许精确 trainer-state 恢复；首轮新 recipe 从相同 seed 新初始化，避免混用旧 aux reward/replay。若后续使用权重 warm start，单列 lineage，重置优化器、replay、计数和不兼容 normalizer，不冒充从 524 连续恢复。

### 任务 1：补齐物理子步诊断与严格标准影子统计

**Files（修改/测试）:**

- `/data/kk/EmbodiedBein/backend/eb/runners/rpo_ufo_isaac.py`
- `/data/kk/EmbodiedBein/backend/eb/runners/bfm_ufo_worker.py`
- `/data/kk/EmbodiedBein/backend/tests/unit/test_rpo_ufo_isaac.py`
- `/data/kk/EmbodiedBein/backend/tests/unit/test_bfm_ufo_worker.py`

**Interfaces:** 保留 `_sample_substep()` 的唯一物理采样入口；控制步只累计本次八个已完成的物理帧。向 trace/指标提供逐关节速度峰、峰值子步序号、越限持续时间、逐足竖直净力积分、原严格标志及首次严格违规 step。

- [x] 增加测试 `test_substeps_count_completed_frames_once`：恰好八帧，reset 前旧力不进入本段，重复 observation/get_dones 不重复累计，subset reset 仅清理对应行。
- [x] 增加测试 `test_strict_shadow_stops_at_first_violation`：探索轨迹继续时，严格存活计数锁定在第一次原阈值/原姿态终止事件；未结束的轨迹标明右删失，不能随 reset 延长。
- [x] 记录原始 q/dq、requested/applied action、目标、同一子步的实际 applied torque、足 link 原点线角速度及时间戳；极值保存最小事件片段，不能只保留无法定位的最大值。
- [x] 同时记录每关节 `abs(dq)/Vmax`，避免“最大速度<25rad/s”掩盖踝部 8rad/s 越限。力积分单位 N·s，力峰单位 N，二者分别命名；启动质量与对应体重 N 由实际仿真资产和 DR 实值计算。
- [x] trace 版本、列定义、worker 解析和统计分母同步更新；保持 actor 输入维度。全量保留精简标量，子步详细事件有界采集，预算内不写每个环境全部原始接触点。

**通过条件:** 新事件可定位关节、env、episode、子步；原严格事件计数不遗漏；训练终止标志与严格影子标志分列。硬限位是子步锁存、在控制步边界返回 terminal，并非一违规就中断该控制步的余下积分。

### 任务 2：环境级 Tracking Context，恢复 250 步专家提示

**Files（修改/必要时新增/测试）:**

- `/data/kk/EmbodiedBein/backend/eb/runners/bfm_ufo_policy.py`
- 必要时新增 `/data/kk/EmbodiedBein/backend/eb/runners/bfm_ufo_rollout_context.py`，仅承载 rollout 状态；不修改已有 held-out 推理模块的用途。
- `/data/kk/EmbodiedBein/backend/eb/runners/bfm_ufo_replay.py`
- `/data/kk/EmbodiedBein/backend/eb/runners/bfm_ufo_checkpoint.py`
- `/data/kk/EmbodiedBein/backend/tests/unit/test_bfm_ufo_rollout.py`
- `/data/kk/EmbodiedBein/backend/tests/unit/test_bfm_ufo_replay.py`
- `/data/kk/EmbodiedBein/backend/tests/unit/test_bfm_ufo_checkpoint.py`

**Interfaces:** 在 policy 适配边界替代原生全批次 context 更新，保留 B 编码、8 帧前向均值及 latent 投影算法。复用 `UFOExpertSlicer.sample(..., seq_length=250)` 的 `sample_ids`，不另建专家采样器；新增 `sample_at(clip_index: int, start_frame: int, seq_length: int = 250) -> dict` 支持 RSI 对齐的合法窗口读取，不修改现有 source 抽样规则。

- [x] 增加 `test_tracking_reset_is_environment_local`、`test_tracking_period_is_environment_local`：一个 env reset/250 边界不能更换其他 env 的 mode、clip/start、cursor、tracking_z 或编码版本。
- [x] 每 rank 使用固定 seed 的无放回分区，**恰好 1024 个 tracking env、1024 个 mixed env**，避免原 `randint` 重复导致实际比例缩水。环境与 rollout 共用冻结分区及 hash，不分别随机决定 mode；初始化时核对一致。各 rank RNG 独立并可恢复。
- [x] tracking 环境 reset 使用任务 4 的合法 RSI 起点，其 clip/time 同时作为专家窗口起点；非 tracking 环境继续 mixed-z 和现有每 100 步独立刷新。传递 `ufo_reset_reference` 元数据（source/clip/frame），不拼入 actor/FB 特征。
- [x] 250 个 transition 需要 251 个有效连续状态；不跨片段、split 或无效帧。先审计两个来源与接触池的交集；交集为空时明确失败，不能缩短成 8 帧或用 held-out 补齐。250 步到期后按该 env 的合法窗口规则续段，续段事件单独记录。
- [x] 缓存一个活动窗口的逐相位 z，不因模型同步在 episode 中重新编码；下一次该 env reset/续段才使用新 B/norm 版本。编码按有界小批执行，避免一次构建全部 250×1024 帧的临时激活峰值。
- [x] 增加 `test_tracking_phase_matches_expert_sequence` 和 `test_tracking_sync_and_repeated_read_do_not_advance`：cursor 每真实控制步加一，重复读不动、不消耗 RNG；期望 z 与参考编码对应帧逐值匹配。
- [x] rollout schema 派生新版本，保存分区、clip/start/source、cursor、窗口 z、编码版本、每 rank RNG、expert sampler 与 mixed pool；缺状态拒绝恢复。进程重建时物理状态仍未恢复，必须 reset 环境并建立新 context，不能套用旧 cursor 到新身体状态。

**通过条件:** 非 tracking 且非周期边界的被动 z 改变为 0；tracking 的跨环境非预期重抽为 0，phase mismatch 为 0，实际唯一 tracking 环境占比 50%；源权重、分区/RNG、上下文读回一致。自然的专家相位变化单独统计，不使用“所有相邻 z 余弦=1”作为通过标准。

### 任务 3：第九项速度软罚与物理子步对齐

**Files（修改/测试）:**

- `/data/kk/EmbodiedBein/backend/eb/runners/rpo_ufo_isaac.py`
- `/data/kk/EmbodiedBein/backend/eb/runners/bfm_ufo_agent.py`
- `/data/kk/EmbodiedBein/backend/tests/unit/test_bfm_ufo_soft_velocity.py`
- `/data/kk/EmbodiedBein/backend/tests/unit/test_rpo_ufo_isaac.py`

**Interfaces:** `_ufo_aux()` 的 `limits_dof_vel` 改读本控制步子步累计量；其他八项计算和 native update 保持现状。新 auxiliary profile 冻结计算语义，保留九项名称和 replay 数值维度。

对第 i 个完成的物理子步定义：

`p_i = sum_j(max(abs(dq_ij)/Vmax_j - 0.70, 0)^2)`

`limits_dof_vel_raw = sum_i(p_i * 0.0025) / 0.020 = mean(p_1 … p_8)`

- [x] 增加 `test_velocity_penalty_covers_interior_substep`：中间子步 1.2×、其余含末帧 0.6× 时，raw 为 `(1.2-0.7)^2/8 > 0`；全程不超过 0.7×时 raw=0。
- [x] 增加 `test_substep_penalty_preserves_constant_velocity_scale`：八帧相同速度时新旧 raw 一致；积分没有重复乘 policy dt，也没有权重重复应用。
- [x] coefficient 初始保留 **−25.0**，只加权一次，再进入原生 RewardNormalizer；记录 raw、weighted、normalized，以及边界版、峰值版诊断，不同时扩大系数。
- [x] 新增 `velocity_reward_coverage_miss_count`：物理子步超过原硬速度阈值，但当步软罚未覆盖相应事件的计数；验证为 0。
- [x] 保留 `latched_velocity_terms_without_boundary_violation` 历史计数：它反映真实子步/边界差异，不因改 reward 就应该归零。另报违规持续时间/暴露率，区分计算漏记消除与动力学改善。

**通过条件:** 每个已完成子步都进入当前 transition 的速度罚；覆盖漏记为 0，有限性/加权/normalizer 分项可对账。该信号供 RL 学习，不声称对 PhysX 接触求解器直接反向传播梯度。

### 任务 4：RSI 候选增加接触点法向速度约束

**Files（修改/测试）:**

- `/data/kk/EmbodiedBein/backend/eb/runners/rpo_ufo_contact.py`
- `/data/kk/EmbodiedBein/backend/eb/runners/rpo_ufo_isaac.py`
- `/data/kk/EmbodiedBein/backend/tests/unit/test_rpo_ufo_contact.py`
- `/data/kk/EmbodiedBein/backend/tests/unit/test_rpo_ufo_initial_contact.py`

**Interfaces:** 扩展 `ContactResetPool` 的候选校验/审计，不原地改原始 RobotState。新增 `contact_point_normal_velocity(link_velocity: ndarray, link_angular_velocity: ndarray, point_offset_world: ndarray, ground_normal_world: ndarray) -> ndarray`，按 `dot(v_link + omega_link × r, n_ground)` 计算。

- [x] 用实际碰撞 hull/FK 找到近地接触候选点；离线填完整 qvel，核对浮动基姿态、世界/局部 frame 变换及角速度，不把 qvel 清零后的 FK 当成真实速度。
- [x] 增加 `test_contact_point_speed_includes_angular_velocity`：link 原点 vz=0、足底旋转仍可产生非零法向速度；与有限差分 FK/仿真 link-origin 速度交叉核对。
- [x] 现有几何条件继续生效：最低足点最终 2–20mm、向上 root lift≤30mm、自穿透≤2mm、初始 dq≤0.70×、root height≥0.53m、tilt≤0.70rad、保护部位净空>25mm。
- [x] 对**该姿态中 20mm 内的近地碰撞候选点**要求 `abs(v_normal)≤0.15m/s`；检查左右足的近地点，高于接触范围的正常摆动足不套支撑门槛。增加 `test_swing_foot_is_not_filtered_as_support`。
- [x] 输出每来源/片段/相位的原候选、几何保留、速度保留、250 步可绑定窗口和拒绝原因；记录足点高度/法向速度分位数。两个来源的普通 RSI 池和 tracking 可绑定池分别保持非空，否则停止配置资格，禁止 fallback；至少保留真实行走的左/右支撑相位，不能只留下静态站姿。
- [x] `reference_reset(env_ids)` 按任务2的冻结 mode 选择普通池或250步交集池，写入与实际 root/q/dq 同一次采样的 `ufo_reset_reference`：source索引、train clip索引、start_frame，共 `[num_envs,3]` int64。未reset行不变；trace保留对应clip身份表及frame/source，元数据经RLinf传给policy且不进入actor特征。
- [x] 除已有有限 root lift 外不抹掉 q/dq/root_vel，不以降低 velocity 数据来伪造低冲击起点。

**通过条件:** 所有新起点满足定义的接触点门槛；无数据泄漏/空源降级，tracking 起点与参考一致。首步终止率目标 **≤0.1%**；首步足竖直净力峰 **<2500N** 为排障目标，同时报告 p95/p99、冲量和实际体重归一化值。这些结果须实测，不能把 2500N直接称为 RPO 的“3–5倍体重”或保证必然达到。

### 任务 5：训练专属探索课程与踝控制单因素对照

**Files（修改/测试）:**

- `/data/kk/EmbodiedBein/backend/eb/runners/rpo_bfm_codec.py`
- `/data/kk/EmbodiedBein/backend/eb/runners/rpo_bfm_extension.py`
- `/data/kk/EmbodiedBein/backend/eb/runners/rpo_ufo_isaac.py`
- `/data/kk/EmbodiedBein/backend/eb/environment_catalog.py`
- `/data/kk/EmbodiedBein/backend/tests/unit/test_rpo_ufo_slew.py`
- `/data/kk/EmbodiedBein/backend/tests/unit/test_bfm_ufo_contact_profile.py`

**Interfaces:** 新冻结训练 profile 的 `training_velocity_termination_multiplier` 初始 1.3；原严格 `physical_limit_flags`、验收 Suite 和影子阈值仍为 1.0。LPF 复用上一 applied target 状态，避免增加不可观测的控制隐藏状态。

- [x] 增加 `test_exploration_limit_does_not_change_strict_flags`：1.2×速度有严格标志/软罚但不因速度终止训练；1.31×训练终止。原位置/力矩硬限位、root height<0.50m、tilt>0.80rad、保护接触>1N、非有限状态仍终止或失败。
- [x] LPF 只作用左右 ankle pitch/roll 目标，初值为 reset 实测锚点；候选 cutoff **15Hz**，`alpha=1-exp(-2*pi*15*0.02)≈0.848`。输出经过现有 slew 后记录为 applied action/target，history 与部署推理协议一致。Replay的`action`仍为actor实际采样的requested behavior action，环境变换属于MDP；不把applied action替换为Actor未输出过的训练动作，两者在trace中成对保存。
- [x] 增加 `test_ankle_filter_resets_only_requested_envs`、`test_ankle_filter_step_response_and_slew`：非踝目标不变，subset reset 不扰动其他 env，无首步旧滤波残留，目标仍在盒内/变化率内。
- [x] 固定初态和动作序列比较三个物理诊断 profile：基准；只加 15Hz LPF；只把踝名义 Kd **2.0→2.5**。Kp/力矩/惯量/solver/physics dt/DR 其余值不变；Kd 实值及延迟/扭矩饱和同时记录。合并两项之前必须有各自收益证据。
- [x] 不照搬连续二阶系统“阻尼越大越稳定”的结论；在 2.5ms 显式积分、有效惯量和 DR 条件下确认没有新振荡或持续力矩饱和。50Hz 控制的 Nyquist 为 25Hz，15–20Hz 滤波较弱，不能承诺其能滤除所有子步碰撞峰。
- [x] 第一轮选定一个控制候选并冻结。课程仅在段间从 1.3→1.15→1.0 派生 profile；不在同一冻结 Run 内悄悄改阈值。原额定速度暴露、严格存活与接触冲击同步改善才进入下一档。
- [ ] 若 1.3×仍被轻微速度毛刺困住，才安排后备的“只保留宏观姿态终止”有界物理对照；保留原位置/力矩、非有限、地面覆盖检查与 **2.0×速度失控终止上界**。不默认执行无限制速度 rollout，不将 40–67rad/s 算成可接受训练动力学。 未执行：本段仍有多倍超速，未触发仅轻微毛刺的后备对照条件。

**通过条件:** 所有硬件额定值和严格验收未变；训练/严格存活分别显示；PD/LPF 实际输入、目标/状态相位和接触响应可比较。任务的目标是降低逐关节归一化峰值及暴露率，不能用放宽 threshold 声称已消除超速。

### 任务 6：Replay 扩容、原子保存与资源准入

**Files（修改/测试）:**

- `/data/kk/EmbodiedBein/backend/eb/runners/bfm_ufo_replay.py`
- `/data/kk/EmbodiedBein/backend/eb/runners/bfm_ufo_checkpoint.py`
- `/data/kk/EmbodiedBein/backend/eb/runners/bfm_ufo_artifact.py`
- `/data/kk/EmbodiedBein/backend/eb/checkpoints.py`
- `/data/kk/EmbodiedBein/backend/eb/capacity/estimator.py`
- `/data/kk/EmbodiedBein/backend/tests/unit/test_bfm_ufo_replay.py`
- `/data/kk/EmbodiedBein/backend/tests/unit/test_bfm_ufo_checkpoint.py`
- `/data/kk/EmbodiedBein/backend/tests/unit/test_bfm_ufo_artifact.py`
- `/data/kk/EmbodiedBein/backend/tests/unit/test_capacity.py`

**Interfaces:** 容量从现有白名单上限 1M 扩至批准的 **4M/5.12M**；先复用现有 FIFO/随机 sampling，明确仍不是原生 trajectory buffer。大容量保存以停止追加/更新的一致性边界为前提，逐字段流式写入 replay 数组文件，manifest 原子发布；不再同时构造两份完整 replay 副本。

| 容量 | 当前每 transition 7042B 的基础数组估算 | 6144 环境下可保留历史 |
| --- | ---: | ---: |
| 1,000,000 | 7.042GB，约 6.56GiB | 每环境约 163 控制步，3.26s |
| 4,000,000 | 28.168GB，约 26.23GiB | 每环境约 651 控制步，13.02s |
| 5,120,000 | 36.055GB，约 33.58GiB | 每环境约 833 控制步，16.67s |

以上是满池全部基础数组，**不是整个任务 RSS**。当前 `state_dict()` copy 后再 numpy→tensor copy，满 5.12M 池保存时仅 replay 就可能约 108GB 十进制；另有 Ray、模型、专家、通信和三 sampler。旧 96GiB 配额不可未经验证沿用。tracking z 缓存按 1024×250×256×float32 约每 sampler 250MiB，三卡合计 750MiB，编码临时激活另计。

- [x] 增加 `test_replay_ring_wrap_and_restore_at_large_capacity`、`test_replay_sampler_rng_roundtrip`：容量边界、回卷、cursor/count/received、恢复后抽样序列一致；小数组逻辑测试配合满容量 CPU RSS 实测，不把只分配稀疏虚页当作满池峰值。
- [x] 新 checkpoint 格式记录每数组 dtype/shape/count/cursor/hash、单一代际 ID、六 Adam/targets/norm、所有 RNG、专家采样和三 rank context。保留旧格式读取；新格式不冒充 `ufo_agent_state_v1`。
- [x] 增加 `test_checkpoint_rejects_partial_or_mixed_generation`：缺片、hash错误、维度变化、错误容量、rank缺失均拒绝；保存中断不发布 complete，原完整 checkpoint 保全。
- [x] 保存不复制整个满池两次；逐字段读写块上限 **64MiB**，恢复到预分配数组，不同时持有完整输入池与完整新池。暂停/排空通信后固定replay与rank状态代际，完成发布后再继续采样。容量/格式改变影响 fingerprint，禁止忽略不兼容字段强行 resume。
- [ ] 容器内先测满 4M 的保存/读回 RSS、时长和文件 hash，再测 5.12M；资源估算纳入实际 Ray object store、sampler/learner RSS、shm、GPU allocated/reserved 峰值、I/O 和 staging。通过后首轮批准 4M，5.12M 为下一档。 CPU满池两档已通过；GPU组合资源为观测及容量估算，完整allocated/reserved峰和精确I/O峰未采集。
- [ ] 统计 replay 年龄的 transition/控制步/更新/墙钟分位数、不同 policy_version/来源上下文、终止样本比及淘汰量；`retained_fraction=capacity/received` 只是累计留存比例，不称半衰期。专家数据仍由独立 train-only slicer 提供，不能声称在线 FIFO 保存了全部专家。 transition/控制步/更新/版本/终止比例/淘汰量已报告；逐转移墙钟时间未保存，不补造墙钟分位数。

**通过条件:** 完整保存和恢复通过、RSS 峰值含裕量低于准入值；资源不足时暂停扩容并记录，不自动减少每卡 2048 环境、不借用其他任务 GPU，也不靠删历史模型腾空间。

### 任务 7：冻结平台配置和容器资格，不伪造兼容恢复

**Files（按现有注册链修改/新增/测试）:**

- `/data/kk/EmbodiedBein/backend/eb/domain/training_recipe.py`
- `/data/kk/EmbodiedBein/backend/eb/domain/rlinf_form.py`
- `/data/kk/EmbodiedBein/backend/eb/environment_catalog.py`
- `/data/kk/EmbodiedBein/backend/eb/adapters/rlinf.py`
- `/data/kk/EmbodiedBein/backend/eb/api/routes_train.py`
- `/data/kk/EmbodiedBein/backend/eb/ufo_source.py`
- `/data/kk/EmbodiedBein/backend/eb/runners/bfm_ufo_worker.py`
- 新增 `/data/kk/EmbodiedBein/deploy/images/rlinf-ufo-tracking-contact.Dockerfile`、`/data/kk/EmbodiedBein/deploy/patches/rlinf/patch_rpo_ufo_tracking_contact.py`；修改 `/data/kk/EmbodiedBein/Makefile`
- `/data/kk/EmbodiedBein/backend/tests/unit/test_training_recipe.py`、`/data/kk/EmbodiedBein/backend/tests/unit/test_rlinf_adapter.py`、`/data/kk/EmbodiedBein/backend/tests/unit/test_bfm_ufo_checkpoint.py`、`/data/kk/EmbodiedBein/backend/tests/unit/test_ufo_source_contract.py`

- [x] 为诊断、集成候选和每个课程档派生明确新版本，展示实际 physics_hz=400/control_hz=50。保留接口 v2；仅当 actor 特征语义发生变化才派生新接口，不因增加旁路诊断元数据滥增本体版本。
- [x] 保持 F/Actor/Q_D/Q_aux LR=3e-4、B LR=1e-5、D LR=3e-6；discount=0.98、FB tau=0.01、Q tau=0.005、ortho_coef=100、z_pool=8192、relabel_ratio=0.8、expert_asm_ratio=0.6、actor std=0.05、AMP/compile/cudagraphs=False。
- [x] 保留 Actor grad clip=1、其他五 Adam=5、正则 w≤50、D reward[-2,2]和原八项 aux 权重；速度 raw 改为任务 3，系数−25。native `clip_grad_norm=0` 必须说明由 owned hook 实际裁剪，记录前后值。
- [x] 检查调用链中的容量上限/recipe白名单、checkpoint format/fingerprint、实际 image/source receipt、trace解析和导出控制元数据；拒绝 UI 覆盖保护字段。复用既有 adapter 和生命周期，不新增通用脚本入口或数据库状态绕行。
- [x] CPU 容器先跑任务 1–6 的针对性测试及 adapter/config 快照；含 native 模型测试用固定 UFO 任务镜像，基础平台测试用现有 `eb/test:local`。使用下文只读挂载/禁缓存命令，不执行宿主 pytest；原生链测试不得因为缺UFO/torch而静默skip后宣称通过。
- [x] GPU 资格仍使用四卡真实 6144 环境：新初始化到 16 更新 → 完整读回 → 同 recipe 恢复到 24。核对六 optimizer step、sampler 权重/池、RNG、replay、tracking schema与真实环境 reset 行为；资格不开展独立评估。

**通过条件:** 错误 profile/镜像/格式/容量/源 revision 直接拒绝；16→24 恢复证据完整。物理环境状态 `not_available` 如实保留，不能宣称逐物理状态的精确续跑。

**任务1–7的测试节奏及预定容器命令（仅记录，本轮不执行）:** 每个任务先加入上述失败用例，单独运行并确认失败原因对应未实现行为，再实施最小变更并运行所属测试；不能把ImportError当作有效失败用例。下面两组通过后，仅新修改/新失败才扩大验证。

新镜像计划tag为 `eb/rlinf:rpo-bfm-ufo-tracking-contact-v10`，构建后实际digest/hash必须进入receipt；名称不代表本轮已注册或构建。所有native专项用新镜像在CPU执行：

```bash
docker run --rm --runtime=runc --network=none --cpus=4 \
  -e NVIDIA_VISIBLE_DEVICES=void -e CUDA_VISIBLE_DEVICES= \
  -e PYTHONDONTWRITEBYTECODE=1 -e OMP_NUM_THREADS=1 \
  -e PYTHONPATH=/repo/backend:/repo/backend/eb:/workspace/eb_runner:/opt/ufo:/workspace/rlinf \
  -v /data/kk/EmbodiedBein:/repo:ro -v /data/kk:/data/kk:ro \
  -w /repo/backend --entrypoint python \
  eb/rlinf:rpo-bfm-ufo-tracking-contact-v10 -m pytest -q -p no:cacheprovider \
  /repo/backend/tests/unit/test_bfm_ufo_rollout.py \
  /repo/backend/tests/unit/test_bfm_ufo_soft_velocity.py \
  /repo/backend/tests/unit/test_rpo_ufo_isaac.py \
  /repo/backend/tests/unit/test_rpo_ufo_initial_contact.py \
  /repo/backend/tests/unit/test_rpo_ufo_contact.py \
  /repo/backend/tests/unit/test_rpo_ufo_slew.py \
  /repo/backend/tests/unit/test_bfm_ufo_replay.py \
  /repo/backend/tests/unit/test_bfm_ufo_checkpoint.py \
  /repo/backend/tests/unit/test_bfm_ufo_artifact.py \
  /repo/backend/tests/unit/test_bfm_ufo_worker.py

docker run --rm --runtime=runc --network=none \
  -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=/repo/backend \
  -v /data/kk/EmbodiedBein:/repo:ro -v /data/kk:/data/kk:ro \
  -w /repo/backend eb/test:local python -m pytest -q -p no:cacheprovider \
  /repo/backend/tests/unit/test_training_recipe.py \
  /repo/backend/tests/unit/test_rlinf_form.py \
  /repo/backend/tests/unit/test_rlinf_adapter.py \
  /repo/backend/tests/unit/test_capacity.py \
  /repo/backend/tests/unit/test_bfm_ufo_contact_profile.py \
  /repo/backend/tests/unit/test_ufo_source_contract.py
```

预期：所有新增行为用例通过、无意外skip、adapter/recipe快照与实际镜像receipt一致。满池RSS/保存恢复是单独有界的CPU运行检查，不由这些小数组单测代替；PhysX诊断与GPU资格另走受管Job。

### 任务 8：500 次新增联合更新与段间决策

**Files（实现/报告）:**

- 复用 `/data/kk/EmbodiedBein/backend/eb/runners/bfm_ufo_worker.py` 的真实进度与指标通道。
- 运行后新增 `/data/kk/EmbodiedBein/docs/progress/2026-10-12-ufo-tracking-contact-500-observation.md`；实际运行跨日时使用真实日期，不覆盖已有报告。
- 运行后更新 `/data/kk/EmbodiedBein/docs/progress/README.md`。

- [x] 从合格 24 checkpoint 追加 **500** 原生联合更新到 524；不是 sampler step、epoch 或容器日志行数。新增/累计计数分别报告，五个 100 更新窗口固定，不把资格统计混入正式段。
- [x] 每窗口采集 learner loss/六优化器梯度前后与裁剪比例、Q_fb/D/aux 尺度、raw/normalized aux、行为与数据质量。现有 env/reward/return/success 的占位 0 不作收敛指标。
- [x] 同时展示训练 episode_len 与 `strict_shadow_episode_len`（仅按原阈值截断的诊断，**不等同独立评估**）、每来源/模式/重置相位的分位数、tracking 250 帧完成比例、首步终止率、逐关节速度峰/暴露率/持续时间、位置/力矩/接触/高度/倾角原因、动作饱和、参考跟踪误差。
- [x] 源采样比例、RSI/250 窗口数量、tracking 非预期重抽及 phase mismatch、速度 reward 覆盖漏记、replay 年龄/容量/淘汰量均入报告；避免总体均值掩盖某来源或非 tracking 环境退化。
- [x] 固定 draw/new 口径：每正常触发 `16×2048/24576=1.3333`；updates/new 为 `16/24576=0.0006510`。完整 500 段按真实 partial trigger 与 transitions 重算，不用术语相同但单位不同的“UTD”比较。
- [ ] 新/累计样本量、各卡峰值、RSS、sampling/update/sync/save 阶段耗时、budget余量和 artifact 完整性形成段末回执；所有配置更换/warm start 曲线分开。 新/累计计数、阶段耗时、GPU/容器采样观测及产物预算已报告；非完整GPU/RSS峰。

**继续/停止规则（训练筛选，不能宣称正常行走通过）:**

| 情况 | 决策 |
| --- | --- |
| 非有限值/梯度、ground 越界、数据 split 泄漏、跨 env tracking 重抽、奖励覆盖漏记、格式/恢复/资源失败 | 停止并保全证据；修配置或实现，不追加训练 |
| 后两个 100 更新窗口训练与严格影子存活都较前两窗口提升≥20%，末两严格影子均值≥50步；速度原额定暴露和首步终止均未恶化 | 有正向信号，允许提出下一段 500 更新或收紧到 1.15×的具体计划；不自动跳到 5000/长训 |
| 训练存活增长，但严格影子存活仍≤ground-v9量级、速度/位置违规增加或仅 tracking 子组受益 | 不达标；停止调整，对照终止阈值/相位/接触/控制，不独立评估 |
| 后两个窗口严格影子均值较本段前两个窗口下降≥20%，或原额定速度暴露反弹≥25% | 策略退化；段内取消仅使用平台已支持入口，保全最近完整 checkpoint，下一段不继续 |
| 末两窗口训练与严格影子存活均≥100步且无退化；首步终止≤0.1%，无 ground/位置/力矩违规，逐关节子步速度未超过原额定上限（沿用既有数值容差） | 才进入独立验证准备；仍不是1000步正常行走验收或交付通过 |
| 趋势不足、指标缺失或事件稀少导致分母不可靠 | 等待/补训练诊断证据，不把 lack of evidence 判成成功，不调用独立评估 |

阈值是下一轮实验预先声明的观察判据，不保证一次 500 更新达到。硬越限“单调下降”不作为必须条件；按窗口、暴露率和固定分母判断，不以随机单点波动停训。软课程阈值收紧与 recipe 变化需新 fingerprint，不假装兼容完整恢复。

## 3. 未来执行预算与资源检查

本节是冻结的执行预算上限。用户后续明确授权实施、资格和500新增更新，现通过平台Job/Run/ebd执行；不授权无限自动实验或额外长训。

- 物理诊断：最多三种冻结控制 profile，采用相同 RSI clip/frame 与动作序列；每卡仍 2048 环境，保存其中固定 case 的详细子步事件。每 profile≤32 控制步/episode、≤256 个被详细记录的起点；总诊断墙钟≤1小时，属于 contact/PD 排障而非成功率评估。
- 资格：新 recipe 16→24 两段各≤1小时，保留完整 checkpoint/readback。失败不自动换 seed 重试，不为通过扩预算。
- 正式训练：第一候选 500 次新增联合更新，墙钟≤2小时；同拓扑新增样本上限 1,000,000 条。满足继续判据才制定下一段；本文件不直接授权无限追加或 5000 次长段。
- 初始 RAM/shm/CPU 参照旧 96GiB/32GiB/24核，仅作对照；最终按满池保存/恢复与实际组件峰值加≥20%裕量准入。整机空闲和其他作业有冲突时等待，不降每采样卡环境数。
- 新产物预算规划上限 **192GiB**，含两次资格、首个正式完整 checkpoint、满池CPU保存/恢复检查的临时产物、流式保存 staging、指标/图表和有界事件片段；启动前按实测 package 大小复算。旧 H/I 产物不计作可删空间；磁盘不足不自动删除旧数据或不保存必要恢复状态。
- 当前网络/batch2048 保持；4M replay 经任务6通过后启用，5.12M须另测满池峰值。500段预计仍小于旧1M容量，因此本轮可验证容量、资源与恢复，但**不能证明扩容改善FIFO留存质量**；收益需后续累计数据实际超过1M后的年龄/行为对照。单纯扩大容量不会替代填满、时间覆盖或更长 episode 的证据。

## 4. 尚未对齐内容的后续边界

1. **MJLab 对照仍缺。** 若 PhysX 下经上述修复和有限物理对照仍无法解决接触脉冲，按设计20另制定 RPO/MJLab 环境桥计划，对齐资产/惯量/碰撞网格、动作解码、reset、aux、终态观测和控制频率，再用相同起点/动作比较。当前不切换主线，不承诺换模拟器就解决问题。
2. **trajectory replay 与 source 内优先采样仍不同。** 当前先解决 FIFO 容量/年龄/恢复；不在本次把全 replay 重写为官方轨迹组织。未运行独立评估时，不用 held-out 成绩更新专家 priorities。
3. **history、控制协议与优化保护仍是 RPO 派生版本。** LPF/slew 若启用须进入推理 metadata，并能从上一 applied target 恢复；若确需增加独立隐藏状态/改变观测语义，应先派生接口并重训，不加载旧 normalizer 硬套同 shape。
4. **保护项的原生对齐后置。** 动力学和 tracking 稳定后分别比较 D LR 3e-6→1e-5、reward clamp、w cap、grad clip；每次只改一项，按训练行为/数值共同判断，不一次撤掉全部保护。
5. **评估只在训练门槛通过后进行。** 到时冻结独立 validation/acceptance，补齐新 physics/control profile 的 evaluator（实际400Hz物理、50Hz控制），不能误用接口历史200Hz字段；保留逐 case 图表和真实成功/失败录像。正常行走仍按原冻结完整 horizon、指令跟踪、跌倒和物理限位标准验收，训练指标不能代替验收。

## 5. 三轮规划复核与执行交接

| 复核轮次 | 本次规划核对结果 | 实施后必须补齐的证据 |
| --- | --- | --- |
| 1：设计、本体、数据和控制语义 | 已区分原生算法/派生物理；保持23DoF完整网络和train隔离；修正“无力矩限幅”“残余肘零点错误”“所有tracking z必须恒定”等判断 | RSI接触点速度/FK、两源250窗口交集、per-env context测试、原始数据hash |
| 2：GPU、真实step、预算和恢复 | 固定1 learner+3×2048、batch2048、先16→24再500；扩replay先验证复制峰值；新语义不伪造旧精确恢复 | 满池RSS/存盘/读回、六Adam/RNG/三rank状态、资源准入/释放、真实新增计数 |
| 3：指标、因果解释和证据完整性 | 区分速度奖励漏记与真实子步毛刺；训练长度与严格影子长度并列；独立评估保持not_run，不因放宽阈值宣称成功 | 全部指标原始值/分窗统计/图表、可定位极值与首步事件、段末继续/停止结论、自包含报告 |

### 本轮交付确认

- [x] 将合理建议转换成分阶段、可验证的规划，并纠正未经证实的因果/对齐说法。
- [x] 明确每采样卡2048环境、完整网络、batch、恢复语义、容量成本及训练/评估门槛。
- [x] 所有本地文件引用使用绝对路径，基线参数与关键数据在本文自包含。
- [x] 实施代码/配置、构建任务镜像、部署服务——已按后续实施授权完成。
- [x] 四卡16→24资格与完整checkpoint回读；独立评估门槛保持不变。
- [x] GPU正式500新增更新和总结——已完成；独立评估门槛未达，本段not_run。

### 执行记录（2026-10-12）

- 配对接触诊断Job `job_896cb307da8749ef9464` / Run `run_53f27bf937f842b3b248`完成：相同6144初态比较baseline、LPF15和Kd2.5。Kd速度暴露约下降10.8%、归一化极值下降3.8%，全局力峰反升3.3%；选Kd2.5，不叠加LPF，不声称接触超速消除。
- 初次16资格在更新前因继承HF初始化二次seed覆写1051而非43失败；真实镜像RED复现后，新context专属修复，旧seed行为保持，失败Run完整保留，未换seed自动重试。
- 修复后16 `trn_f05ad644bca5442099e8` / Run `run_85ae9116da5549e98143`，完整恢复24 `trn_4db44db408e14ba8805a` / Run `run_9b47b8e1134b4fb39c0a`均成功；六Adam和SHA均通过，fingerprint相同。物理内部状态不可恢复，各段重新RSI。
- 98项原生镜像测试与544项backend测试通过（后者27项可选依赖跳过）；4M与5.12M满池保存/恢复及下一次抽样RNG实测通过。所有3442候选足点法向速度复核通过，左右支撑和两源250交集均保留。
- 正式500新增：Train `trn_c4ab80368f5b40d1a203` / Job `job_6594eb07b76a48129ebd` / Run `run_1299df81ef5740f99f23`；镜像 `sha256:1970d0dc894b0a80c5dc27bd6a5b94c8d8463c52f3ec5c3e8108dda0b1b073f9`，每采样卡2048、batch2048、replay4M。段末按既定训练/严格影子窗口和动力学门槛决策。

正式500结束：五窗训练存活8.2545→11.5582→15.1945→19.2217→22.3134步，严格影子7.3680→9.2128→11.7758→14.7347→16.0141步。首步48/54524=0.0880%；子步峰55.7632rad/s，原额定归一化6.9704倍。六Adam524/哈希/860160累计转移通过，GPU释放；最终模型mv_e95f1d9da4574b05b300，无评估报告。停止本段，未达到下一段50步及独立评估100步门槛，未提交代码。完整自包含总结位于/data/kk/EmbodiedBein/docs/progress/2026-10-12-ufo-tracking-contact-500-observation.md。

勾选表示实现和对应行为检查完成，不表示所有动力学目标或正常行走验收通过；未勾选项保留实际未测/未执行边界。
