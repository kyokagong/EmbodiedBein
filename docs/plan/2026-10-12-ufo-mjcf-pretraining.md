# RPO 23DoF 原生 MJCF / MJLab 预训练接入

日期：2026-10-12。用户授权：改造成 MJCF 并进行一轮预训练测试；不提交代码。设计依据：现有 UFO / RLinf 集成设计的 MJLab 阶段。

## 范围与实验边界

接入 MuJoCo-Warp / MJLab 原生 MJCF 物理后端，继续使用 RLinf 调度与 UFO 原生六优化器。保留 IsaacLab 旧配方，派生新的不可变训练配方、环境 Profile、镜像和来源凭证。不能仅将 MJCF 导入 PhysX 后称为原生 MuJoCo 对齐。

使用受管 RPO MJCF 的质量、惯量、碰撞网格；显式记录求解器、步长、摩擦、接触、armature 和执行器参数。RSI 几何过滤必须与运行时碰撞模型一致。禁止将 URDF 简化膝部碰撞体与 MJCF STL 碰撞体混作同一实验。初始化仅取冻结数据集 train split；跟踪上下文按环境独立维护。

继续使用现有 375 / 733 维、wxyz、四帧 oldest-first 观测编码与 RobotState v2 数据，不把 UFO 原生 G1 历史布局当成兼容。原已发布接口不原地修改；运行 Profile 明确实际物理频率，旧 codec 的 physics_hz=200 字段属于已知历史元数据问题，不代表本实验实际步长。若观测或动作语义改变，必须派生新接口及视图。

新后端是新的 MDP：从头构建 agent 与 replay，不恢复 PhysX 旧训练状态。先完成真实更新及完整 checkpoint 的同后端回读/恢复，再进行 500 次新增 learner_joint_update 的有界预训练。GPU0 learner，GPU1–3 各 2048 环境，总计 6144；Batch 2048，Replay 4M，网络 2048×6，潜空间 256。沿用梯度裁剪和尺度保护，避免同时取消已证实必要的数值保护。测试最多两小时、无自动重试、无自动长训。

训练未达标不启动独立评估。500 更新是观察实验，不是正常行走验收。训练趋势记录 episode_len、严格额定限位影子存活、首步终止、子步各关节速度与触地冲量、动作饱和、tracking 完成数、六优化器与各损失、replay 和逐卡资源。

## Task 1：原生物理与控制契约

先新增最小行为检查，确认旧接口不变、MJCF 参数和碰撞过滤一致、关节顺序与 root 速度/四元数语义正确。新增 owned MJLab 环境桥，复用 codec、历史、tracking、raw aux 和子步诊断。运动学/初始接触/PD 静态与脉冲检查在容器内完成。

接口：新桥输出 RLinf 已支持的 states / privileged_state / last_action / history_actor / episode_id / step_id / UFO traces；terminal observation 必须在自动 reset 前捕获；episode 内反复读取观测不能推进历史或计数。物理参数必须写入 startup receipt。

## Task 2：镜像与平台闭环

固定 MJLab 1.4.0、兼容 MuJoCo / Warp 和 Torch 2.7。通过受管镜像补丁添加 RLinf MJLab 环境分派，不修改外部只读仓库。新增 recipe、template、environment catalog、adapter 环境变量、容量准入、完整 checkpoint 身份和收集入口，复用既有 Job 生命周期。

接口：execution.simulator_id=mjlab；来源凭证包含依赖版本、代码/补丁和实际镜像摘要。旧 Isaac 配方不能误用新镜像，跨后端恢复必须拒绝。先通过容器单元检查与 resolved config 核对，才部署控制面及提交 GPU Job。

## Task 3：资格更新与恢复

先通过平台 Job/Run/ebd 运行必要的小规模物理 probe 和实际四卡资格更新。记录新后端实际峰值，确认每张 sampler 2048 环境可准入。验证 16→24 更新恢复，六 Adam step、replay、normalizer、采样状态与 episode reset 协议一致。失败保存原始证据，不将运行成功当成质量通过。

## Task 4：500 更新与观察总结

从同后端资格 checkpoint 恢复，新增 500 更新（24→524）。运行结束登记完整模型，分析五个等更新窗口与 episode 加权窗口，保存全量指标、配置、来源和曲线到 docs/progress 的独立文档。保留失败/未验证项，未达训练门槛不运行独立评估，不继续追加长训。

准入决策沿用原严格影子口径：影子存活末两窗口≥35 步、物理额定速度峰值比≤2.5、首步终止≤0.1%、数值有限，作为后续实验参考；独立正常行走评估仍要求训练末两窗口≥100 步及其余冻结门槛。本次不自动启动下一段。

## 复核与执行记录

至少三轮复核：①本体、布局、惯量、碰撞与 split；②真实 step、四卡角色、预算和同后端恢复；③全部指标、曲线、checkpoint 和结论。实施完成后进行一次独立代码复核。

- Task 1–4完成；本轮500更新结束，训练门槛未达标，不独立评估或追加长训。
- Task 1：CPU 原生两环境重置/观测幂等/部分 reset/完整 8 子步通过；原质量、惯量、codec 和实际足地接触已核对。发现并修正传感器 primary→secondary 力方向与 aux 支撑 +Z 约定相反；新增测试 RED→GREEN。
- Task 2：新模板/执行绑定/来源镜像/adapter/环境目录/容量/恢复指纹已实现；控制面 62 项回归通过，原生契约 5 项通过，复用 tracking/子步检查 21 项通过。独立评估入口仍拒绝未资格模型。
- Task 3：平台物理验证 Job `job_64a0f77b123d4439a052` / Run `run_69bac540803a4a13bee1` 已启动，三 sampler 各2048环境。此诊断镜像早于力符号修正，物理求解、终止与峰值绝对量不受符号影响；后续训练使用修正后的独立镜像和来源凭证。
- 决定：复用平台已冻结的最多24次资格预算做16→24同后端恢复，再新增500次趋势观察；这两次短段仅验证完整状态恢复，不能代替质量趋势测试。
- 独立代码复核：发现实际积分接触力被额外 forward 覆盖、以及 MJLab 可重复追加500更新的预算缺口；两项均以 RED→GREEN 修正。接触力在 step 后先复制，再刷新 post-integration FK；524后恢复候选关闭，正式训练必须完整恢复24→524。增加实际contact/constraint容量耗尽检查。
- 复核决定：部分 reset 的全局 forward 可能重算 alive 行的力矩诊断，但当前 RLinf 在任意 done 时使用整批 reset 前 final_obs，未证实 replay 污染；保留原采集协议，不额外改写该路径。solver warmstart 和其他调用者的诊断影响尚未验证。
- 待补：修正后的四卡实际积分力验证、同后端16→24恢复、500更新与全量报告。此前的 probe 不作为修正后冲量的最终证据。
- Task 3更新：最终镜像物理复验 Job `job_b141c3a526f440bb87ed` / Run `run_4a0d372b41d7438c8e64` 成功；三卡各2048、32步，容量未耗尽。原生同步检查68项通过；控制面555项通过/30跳过。已提交16更新 Train `trn_b707f0de1d3b43a4b55b`，使用最终镜像 `sha256:4897bb2f2aed1bdeac39dad458e186b8c9649255b82d16de7f3840654fd68399`。
- 决定：使用 MJCF 原生求解器而非格式转换；原生动作和接触参数差异写入报告，收益不能单独归因为“文件格式”。
- 决定：旧工作区已有大量未提交改动，原地保留并精确修改本次相关文件；不使用 git add / commit / push，不删除既有资产或任务。

- 首次16资格更新前TypeError：RLinf NumPy动作与Torch环境边界不兼容，已在step入口转换float32/device，新增行为检查RED→GREEN，6项原生契约通过。失败Job/Run保留，无完整模型；修复镜像d7bc…重新资格，不覆盖旧Run。

- 来源摘要遗漏和131/128 trace接入缺口均保留失败记录；修正严格/放宽限位诊断口径，2项行为检查RED→GREEN，44项相关原生检查通过。最终资格前镜像b559…，预览核验一致后Train trn_8d5de58d8a334c3e82f2重新资格。取消请求被自动审批拒绝，未绕过，旧任务自然失败释放四卡。

- 后续复核发现原始MJCF15个关节范围与冻结RPO接口不同，以及位置罚误用硬范围。派生MjSpec统一冻结接口范围，位置罚用90%soft limits，RSI同模型；原始资产/codec不改。两检查RED→GREEN、相关44项/最终8契约/平台40项通过。旧资格平台取消时已记录14更新，无完整模型；最终071d…镜像Train trn_7e1e42b773cf49d5934a重新从头资格。
- 统计复核：本段500更新只给每环境128控制步，无法完成250步Tracking窗口；不将窗口完成数0单独判为算法错误，另报右删失episode长度。
- Task 3完成：071d…镜像fresh16与完整恢复24成功。16/24六Adam和全部文件哈希核验通过，指纹cedee…一致；计数2/3采样轮、49152/73728转移，恢复三sampler同一version16/state hash。实际RPO33.7568956kg、400Hz Newton/implicitfast、三rank各2048及1024tracking核验；153参考帧位姿差最大0.996mm/0.00117rad。recipe提升为smoke_passed仅表示链路资格；训练存活未达标，不独立评估。
- Task 4完成：Train trn_68a5fc34919446c5adf4 / Run run_4627d50438b84a07bc24，完整24→524新增500更新成功，六Adam524/全部SHA/恢复指纹通过；149项有限标量、32采样轮/786432新增转移与replay逐行对账。训练五窗3.24→4.14→5.21→6.54→8.49，严格2.98→3.84→4.79→5.84→7.47；最高右踝pitch35.78rad/s=4.47倍，首步5478/155862=3.51%，全部涉及位置违规。停止本段，API恢复候选实际关闭，无独立评估、无Git提交。
- 最终三轮复核：本体/数据/框架（含153帧FK与root速度点坐标系）；四卡/真实更新/六Adam与完整恢复/不可变来源；全部标量/32轮控制与replay/右删失/资源释放/完整总结。后续建议优先处理贴位置边界且朝外的RSI初态、MuJoCo软约束与子步硬终止容差，以及50Hz差分速度离群帧；本轮没有继续实施或延长训练。
