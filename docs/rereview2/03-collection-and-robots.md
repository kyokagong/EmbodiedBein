# 03 · 自动化仿真采集与三机器人接入

## 1. 当前资产边界

| 机器人 | 用户确认 | 本地已有基础 | 当前缺口 |
|---|---|---|---|
| SO101 | 单臂 | LeIsaac SO101 场景/状态机/Mimic/遥操；LeRobot 设备与数据 | 平台接入、标准 episode、模型数据映射、目标端验证 |
| G1 | EDU，只有仿真 | IsaacLab G1 locomotion 与若干 manipulation/Mimic 环境 | 确定具体仿真资产/手部；RLinf wrapper/注册；不宣称真机已具备 |
| Berkeley Humanoid | 用户已有机器人；具体在用形态待接入核对 | 四接口 YAML、历史短训、RLinf WBC/Reach、采集和评估代码 | 新工作台绑定断点、接口映射、完整训练评估与现场验证 |

三个机型均是本次设计与验收的一等输入。实施顺序按依赖推进，不把其他机型降为“将来支持”。未具备的 G1 真机阶段标为 `not_available`，不是 failed，也不是通过。

## 2. 通用采集配方

CollectionRecipe 固定以下七组输入：

1. **机器人与任务**：interface revision、scene revision、控制器 revision、task binding。
2. **来源策略**：teleop、scripted_expert、mimic、policy_rollout、human_correction；对应工具入口和依赖。
3. **覆盖计划**：场景类别、物体、地形、相机、光照、质量/摩擦/延迟区间及 seed 计划；保存采样实值。
4. **记录计划**：必需 streams、实际时间轴、attempt 账、成功与失败保留规则、原始格式、preview 策略。
5. **停止条件**：目标合格 episode 数、max_attempts、max_wall_time、max_gpu_hours、max_output_bytes；必须有有限上界。
6. **输出条件**：用途、质量规则、分组划分、导出 profile、所需 model loader smoke。
7. **执行画像**：镜像、CPU/GPU/内存/shm、并行 shard 数、每 shard 并行 env 数、可恢复粒度。

失败分为资源/基础设施失败与任务失败。前者重试同一逻辑 shard；后者是一次有价值的 attempt，进入账目并按下一 seed 继续。达到预算但合格数不足时，Session 为 `budget_exhausted`，已完成原始数据可登记，目标训练 Slice 不自动成功。

UI 支持“按配方生成一次”和“按不足覆盖单元补采”，不引入任意 shell 节点。不因在线成功率低就无限重试；不把重试产生的同 seed 重复样本计作覆盖增加。

## 3. SO101 单臂：三条共享出口的采集支线

### 3.1 状态机专家（无人自动采集）

本地 `scripts/datagen/state_machine/generate.py` 可运行已注册的 SO101 PickOrange 等任务。平台接入顺序：

`probe → 场景/相机/动作检查 → 生成 shard 计划 → 状态机 rollout → 原始 HDF5 全尝试 → 索引/质量/回放 → 固定选择与导出`。

首个 probe 用单 env 并启用相机，验证可记录一次成功与一次失败。然后依据 4090 实测渲染显存、GPU 利用率、CPU 编码、磁盘吞吐调整并行；不能照搬无相机 PPO 的 1024 env 默认值。

源码表明 HDF5 路径使用 EXPORT_ALL，直接 LeRobot recorder 使用成功集导出。采用前者存原始尝试，导出成功训练集另做 Job。状态机自己的 success 判定和环境任务判定必须交叉验证；脚本说成功不代表物体真的在目标内保持达标时间。

原工具只有成功数停止条件时，平台 wrapper 补上 max_attempts/时长/空间账，输出 structured summary；wrapper 放平台管理的工具镜像，不改只读 leisaac 仓。当前 HDF5 能否保留所有必需事件必须 probe 验证，不能仅由 EXPORT_ALL 常量推断完整性。

### 3.2 人类示教 → Mimic 扩增

本地链路为：遥操示教→`eef_action_process --to_ik`→`annotate_demos`→`generate_dataset`→按最终 action profile 转回 joint→仿真回放。

每一步产物必须记录 parent episode、输入 action 表达、工具/IK/机器人资产版本、对象位姿和子任务边界。Mimic 依赖适合的任务定义与物体/末端信息，不是任意视频一键扩增；SO101 的可达范围、碰撞与夹爪开度必须验证。自动标注不确定处在 UI 请求人工补标后才能继续，未完成节点是 blocked。

先把示教祖先分配到 train/validation 组，再在各组内部扩增；验收场景独立采样。Mimic 输出不能与父示教算作独立泛化证据。扩增后既检查任务成功，也对比动作重放与状态重放；只有状态重放好看不说明策略动作有效。

### 3.3 策略 rollout 与纠正

输入为已注册模型+相同的处理器/归一化+场景计划。记录所用 policy version、实际 action、人工接管区间与 expert target。纠正数据形成新候选 Slice；纯失败 rollout 不直接当 SFT 专家标签。支持 RL/DAgger 的 recipe 才能消费对应标签。

### 3.4 遥操作边界

仿真可以通过已有 SO101 leader 或键盘等工具连接采集进程，平台负责启动受控 session、观察、停止与归档。真实机器人遥操作继续由现场设备工具完成，浏览器不转发实时关节命令；网络/页面断开不得破坏端侧控制。采集结束后导入文件及 session manifest。

SO101 的单臂关节+夹爪由 scanner 和设备配置共同确认；不可仅写“6DoF”就推断全部 action 含义。leader/follower 标定与相机外参绑定真实实例，仿真资产和真机标定不是同一个对象。

## 4. G1 EDU：仿真控制与数据并行推进

### 4.1 机型配置

选择具体 G1 EDU 仿真资产，登记手部/夹爪、驱动关节、锁定关节、浮动基座、控制模式和传感器。不同任务允许引用不同接口版本，如 locomotion、固定基座上肢、全身移动操作。不能把不同 G1 模型的关节顺序视作兼容。

### 4.2 Locomotion / WBC

本地 IsaacLab 有 `Isaac-Velocity-Flat-G1-v0`、`Isaac-Velocity-Rough-G1-v0`，配有 RSL-RL 配置；本地 RLinf IsaacLab 注册表目前只有 BHL Reach/WBC 与 Franka Stack。

因此工作项是**平台管理的 G1 环境桥接包**：在容器内组合/注册 RLinf 所需 env wrapper，核对 reset/step、done mask、reward、观测/action、并行环境、设备及 seed。不能只把 Gym id 填入现有表单。若注册必须经过现有 registry，使用受控 launcher 初始化桥接包，再启动上游入口；依赖按镜像固定，升级后重验，不改 `/data/kk/RLinf`。

既有 Isaac/RSL-RL 作为对照基线保留，RLinf 作为主训练接入方向，两者共享数据与评估证据。纯在线 RL 不强制要求离线训练集：输入场景/奖励/接口，输出 rollout 摘要与选定轨迹、checkpoint；记录计划避免落盘全部高频训练状态。

### 4.3 操作与仿真数据

若所选 G1 manipulation 环境具备专家/遥操/Mimic 条件，走与 SO101 相同的 Session→Episode→版本管线，但数据附带基座、左右手、全身状态和各控制器信息。缺少专家、成功函数或相机观测时，该采集模式明确 unavailable，不用“支持 G1”总开关掩盖。

高层 VLA 只输出上层目标时，底层 locomotion/WBC 控制器也必须作为版本化依赖记录、评估和交付；不能将高层模型 action 直接发电机。

G1 当前验收包含仿真采集、训练、独立评估和 Orin 离线推理基准；不包含真机执行、安全放行或现场回滚通过。

## 5. BHL：复用严格采集，消除工作台特例

`RLinf/examples/embodiment/collect_berkeley_reach.py` 已包含有限 attempts、训练/validation seed 划分、所有原始尝试保留、成功视图 manifest、关节与 planner hash、20Hz 和图像/state/action 回读校验。优先复用这一条，而不是退回假漏斗。

不过该脚本对 `pi05_berkeley_reach` 和 10 维接口有明确假设；通用化应抽出**协议字段与输出规则**，保留 BHL recipe 特例。WBC/双足数据不强行套用 Reach 的 RGB 和语言格式。

三条路线：

- BHL fixedhand：专家采集→成功 Slice+失败池→OpenPI SFT→相同接口 closed-loop eval→Orin 目标实测。
- BHL WBC/biped：纯 RL 或指定参考轨迹→训练→导出→固定任务 eval/sim2sim→端侧离线验证。
- ego：事件→场景/可选重定向→仿真可行性→真正的动作 episode→训练版本；失败停在可解释的 L2/L3，不把 35 个场景称为 35 条动作示教。

## 6. 采集资源、自动化与吞吐

一个 Collection Session 由多个有限 shard Job 构成。每个 shard 独占输出；随机化计划提前固定，索引为 `(scenario_case, seed, attempt_index)`。单机先整卡分配，采集渲染和大模型训练按实际显存画像排队，不共享未经容量验证的 GPU。

每轮先小样本测 `attempts/hour, accepted/hour, GPU-hour/accepted-episode, bytes/min, encode backlog, disk write bandwidth`。总资源预算随实测更新，但冻结的采集范围和 eval 集合不自动变化。编码背压不能改变未记录的仿真时间语义；允许降低预览质量，不能静默丢必需训练帧。

自动补采输入是覆盖缺口（如橙子远端位置、低照度、G1 特定坡度、BHL 左手类别）而非单一总数；输出新 DatasetVersion。补采依据来自 train/validation 诊断，不能窥看隔离 acceptance 后反复调整并继续报告同一最终分数。

## 7. 三机器人共同的完成标准

每个机器人至少有一个经验证的任务接口、一个真实来源数据版本、一个训练消费记录和一个独立评估报告。SO101 与 BHL 额外按现场硬件条件推进真实端交付；G1 明确停在仿真和 Orin 离线验证。三者共享对象/API/失败恢复，实现中不得增加 `if robot == ...` 分散业务校验；差异集中在已有 adapter/recipe 的受控配置与任务实现。
