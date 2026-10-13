# BFM-Zero / RPO / RLinf 冒烟实施计划

> **For agentic workers:** 使用 `superpowers:executing-plans` 按任务顺序实施。复选框记录进度；遵守用户要求，不执行 Git commit/push。保留执行记录，不以 Git 历史代替证据。

**Goal:** 将 BFM-Zero 预训练方法和提示/连续命令前端接入现有 EmbodiedBein → RLinf → RPO 容器管线，完成真实采样、六组件更新、保存/恢复与导出 smoke。P4/P5 行为资格另行实施。

**Architecture:** 保留 Job/Run、RLinf adapter、ebd 和 RLinf EmbodiedRunner。增加独立 RPO BFM codec、观测动作数据视图、条件模型和单 learner worker；RLinf 持有优化器、replay、target 更新、权重同步和 checkpoint，不调用上游训练循环。受管补丁只修改镜像中的 RLinf 副本。

**Tech Stack:** Python 3.11、PyTorch 2.7、Isaac Lab、RLinf、NumPy、现有 PostgreSQL/ebd；复用 RPO 任务镜像作为专用 BFM 镜像基础。

**Spec:** [14 · 接入设计](../../design/14-bfm-zero-rpo-rlinf-integration.md)。本轮以 P0–P3 和推理接口冒烟为验收；P4 正式 Flat 替换、P5 通用行为质量、少样本 latent 适应及真机交付不因冒烟通过而获得资格。

## 全局约束

- 最终训练由 RLinf 的 runner/worker 执行；不使用 BFM 原生 trainer，也不替换为 PPO/SAC loss。
- 六组件、双 F/双 Q、六组 optimizer 和 target 完整保留；smoke 可减小宽度、batch 和环境数，z 保持 256。
- `/data/kk/RLinf`、BHL、原始动作与已发布数据/模型/环境只读；新数据派生新版本。
- Python 测试、数据转换、训练和 GPU probe 全部容器化。正式冒烟通过平台 API → Job/Run → ebd 创建容器。
- 23 维动作使用当前 Flat 原生关节序；50 Hz 策略、200 Hz 物理；冻结增益、history、坐标和 body 顺序。
- observation-only 专家不提供伪造 action；连续 8 帧且不跨 clip/split；不使用独立评估轨迹。
- 现有动作仅为诊断来源：分别记录运动学有效、动力学未验证及权利链限制；仅允许预算受限的 smoke，不升级成正式训练合格数据。
- 恢复模式为 `trainer_state`，仿真从新 episode 开始；缺 replay/target/optimizer/normalizer/RNG 时拒绝恢复。
- 不修改旧 Flat 默认模板、已发布接口、100 case 判据或真机门禁；不提交 Git。

## 重点复核的五类失败

1. 同为 23 个关节但顺序/动作 gain 不同：命名映射、默认姿态及物理 target 必须逐通道检查。
2. 部分 reset 或 truncation：replay 使用 reset 前真实末态；其他环境历史不变；截断 bootstrap、终止不 bootstrap。
3. 数据与模型来源变化：路径、文件 hash、资产/codec/split 指纹失配必须失败，不加载不可信 pickle。
4. BFM 算法被简化：检查全部组件梯度、target、更新顺序、连续专家采样和 pairwise FB loss；拒绝 microbatch 切分配对。
5. 半写 checkpoint/提示失配：清单、原子完成标记与严格恢复；旧 z 不可与新 B/normalizer 混搭。

## 执行方法与证据

工作区保留当前 `feat/make-features-real` 分支及已有设计修改；不迁移/覆盖未跟踪的 `scripts/`。执行台账放 [进度记录](../../progress/2026-10-08-bfm-zero-rpo-rlinf.md)，逐任务登记 RED→GREEN、实际命令、偏差裁定和 Job/Run 证据。技能中“频繁提交”和基于提交的台账脚本与用户禁止提交冲突，本轮用文件差异及运行证据替代。

每项非平凡行为先写测试并观察预期失败，再实现、容器验证。计划不正确时先记录最小裁定再继续；算法/数据/执行边界不能静默放宽。

### Task 0：固定输入与目标运行时

**Files:** 本计划、进度台账；新建 `backend/tests/runtime/rpo_bfm_smoke.py` 的环境 probe；读取现有资产和固定上游源码。

**Interfaces:** 产出实际 `body_names`、P=15×body_count−2、默认姿态/动作范围、镜像与来源 hash；后续 codec/数据/模型均消费同一规格。

- [x] 记录已有工作区差异；核对设计、现有 Flat/AMP/SAC 链路。
- [x] 目标镜像探测依赖、GPU、RPO articulation 的关节/body 名和 root；记录真实维数，不用设计算例发布合同。
- [x] 固定可用于冒烟的动作来源、资产/许可限制及小预算；确认不引用独立测试轨迹。
- [x] 复核本计划接口依赖，记录第一轮遗漏检查。

**Expected:** 真实 GPU/资产可加载，输入已固定；若资源/来源不可用，保留具体失败，不用 mock 代替最终冒烟。

### Task 1：RPO codec、motion view 与环境时序

**Files:** 新建 `backend/eb/rpo_bfm.py`、`contracts/ee_rpo_v2_bfm_23dof_v1.yaml`、`backend/eb/runners/rpo_bfm_codec.py`、`rpo_bfm_motion.py`、`rpo_bfm_extension.py`；测试 `backend/tests/unit/test_rpo_bfm.py` 和 runtime probe。

**Interfaces:** `encode_state`/`encode_privileged`、四帧 history、`load_motion_view`/连续序列采样；RLinf 顶层张量 `states,last_action,history_actor,privileged_state`，reset 前末态及辅助奖励。

- [x] 写关节映射、heading/坐标、body 维数、连续窗与部分 reset 的失败测试并容器运行。
- [x] 实现共用 codec 和独立合同；按 Task 0 probe 冻结维数、字段次序、动作 gain 及历史布局。
- [x] 从可信冻结来源派生受控数组/manifest，记录诊断限制，验证 hash、split、有限值与窗口；禁止从 74 维 AMP 特征猜造完整 body 状态。
- [x] 复用 Flat 环境资产/PD/延后 reset 模式，添加 BFM 状态和无命令依赖的辅助奖励。
- [x] 运行专家/在线姿态数值对照及两环境 reset/终止/截断探针。

**Expected:** state=52、last_action=23、history=300；P 与 probe 一致；错误数据/顺序被拒绝，专家与环境编码对齐。

### Task 2：完整模型、学习目标、提示和 replay

**Files:** 新建 `backend/eb/runners/bfm_zero_policy.py`、`bfm_zero_learning.py`、`bfm_zero_upstream/` 的四个固定网络/normalizer 依赖；必要许可证/来源回执；测试 `backend/tests/runtime/bfm_zero_learning_smoke.py`。

**Interfaces:** `BFMZeroPolicy`、六组件/四类 target、可求导 loss、混合 latent、goal/tracking/reward inference、按 transition 限容的 CPU replay 和严格采样状态。

- [x] 写网络 shape/分支、FB loss 手算、全部梯度、terminated discount、专家连续窗、提示退化和 replay 淘汰/恢复测试，观察失败。
- [x] 按固定公开实现迁移 residual actor/F/Q、B/D、normalizer 与采样动作；保留 Actor 与 F/Q embedding 的不同 builder。
- [x] 实现 D → F/B → Q_D → Q_aux → actor → target 的更新；六组 optimizer 独立，保留 pairwise batch、detach、pessimism、正交项及 Q 正则。
- [x] 生成三类提示及连续 command→z 前端；命令前端绑定 B/normalizer/训练参考、asset 与 codec hash；退化向量失败。
- [x] 实现 CPU replay 容量、索引、RNG 与混合 latent 采样状态；真实 backward/step、target 更新与恢复 sample ID 对照通过。

**偏差：** 未与固定上游实现比较完整 update 的逐值 loss/gradient/parameter delta；本轮用手算 FB loss、六组件梯度和平台 RLinf 更新验证。恢复后成功执行下一次更新，但未宣称 bitwise exact。

**Expected:** 六组件有有限且正确的梯度/参数变化；target 不参与 optimizer；实际 RLinf 更新和 replay 恢复采样通过。

### Task 3：接入 RLinf worker 与镜像补丁

**Files:** 新建 `backend/eb/runners/bfm_zero_worker.py`、`backend/eb/runners/config/rpo_bfm_zero_pretrain.yaml`、`deploy/patches/rlinf/patch_rpo_bfm.py`、`deploy/images/rlinf-bfm.Dockerfile`；修改 `Makefile` 必要构建入口；补丁/配置测试。

**Interfaces:** `RPOBFMActor` 复用 `EmbodiedFSDPActor` 生命周期，`BFMZeroUpdater` 负责离策略更新；`loss_type=bfm_zero` 显式分派、模型注册、完整顶层张量透传；保存 `actor/model_state_dict/full_weights.pt` 及完整 trainer 状态。

- [x] 写入口分派、EnvOutput/Trajectory/replay 字段往返和无 PPO logprob 版本张量测试，观察预期失败并验证修复。
- [x] worker 负责采样、normalization、backward/optimizer、target、权重版本和保存/加载；不调用上游 update/train 主循环。
- [x] 补丁覆盖实际 schema 白名单、rollout 版本/评估路径、env 辅助奖励与配置验证；一个 learner、一个 action chunk、关闭 compile/graph/pipeline。
- [x] 配置提供 bringup 与显式 smoke 缩小参数；不允许拆分 FB pairwise batch。
- [x] 构建专用镜像，写入源码/补丁/算法/codec 版本回执；容器验证 import、resolved config 和补丁重复/源码漂移失败。

**Expected:** RLinf 真正选择 BFM worker；实际镜像代码与回执一致，外部 RLinf 源没有修改。

### Task 4：平台模板、数据准入、发布与恢复

**Files:** 修改 `domain/rlinf_form.py`、`api/routes_train.py`、`adapters/rlinf.py`、`bootstrap.py`、`environment_catalog.py`、`runners/rlinf_entry.py`、`checkpoints.py`、`orchestrator/collect.py`；复用现有数据转换入口，仅补所需格式分支；测试 `test_rpo_bfm_platform.py`。

**Interfaces:** `rpo_v2_bfm_zero_pretrain` → `train.rl`、独立接口/Task/Profile、必需 motion DatasetVersion；发布 `mf_rpo_v2_bfm_zero`，恢复指纹含数据/codec/架构/算法版本。

- [x] 写预览/提交共用校验、数据/基座失配、保护 override、渲染挂载、半写产物、完整恢复与错误旧模型测试，并验证失败路径。
- [x] 注册本体接口、任务、数据视图、模板、模型族和专用镜像；保留诊断限定，不冒充正式 qualification。
- [x] 通过现有 Job/Run/ebd 与受管只读挂载运行；不新增服务、JobKind 或控制面训练 subprocess。
- [x] 补完整 checkpoint 清单/atomic marker、BFM 发布分支和 collector 内容校验；resume 明确 trainer_state。
- [x] 容器运行聚焦平台测试和既有 Flat/AMP/训练恢复回归。

**Expected:** 用户入口可预览/提交 BFM；安全/失配输入被拒绝；成功训练登记可追溯新模型，半写/失败不登记。

### Task 5：真实平台冒烟、恢复与推理导出

**Files:** runtime smoke 脚本；新建 `backend/eb/runners/rpo_bfm_artifact.py`，复用现有 `export.policy` 能力；台账及运行回执。

**Interfaces:** 冻结模型 → actor ONNX（proprio/history,z）+ prompt/reference/codec manifest；源模型/B/normalizer 指纹必须一致。

- [x] 先写导出形状、batch=1/批量、不同 z、退化提示/资产失配和 PyTorch/ONNX 物理目标对照检查。
- [x] 实现 actor/normalization 导出，冻结 goal/tracking/reward 提示与 command→z ONNX 前端；normalizer 固定，不在推理时更新。
- [x] 平台提交受限单 GPU 小规模真实 Isaac/RLinf 冒烟，至少两次更新，检查六组件指标和完整 checkpoint。
- [x] 通过平台从完整 checkpoint 恢复并执行下一次更新；核对模型登记、血缘、replay/optimizer/target/RNG 状态。sample ID 已有精确对照，未做恢复 update 的 bitwise 对照。
- [x] 运行真实导出产物推理，记录误差、不同 z、历史行为、ONNX 与 prompt/command Artifact 大小。
- [ ] 未测 GPU/RAM 峰值、端侧全周期时延、长期采样/更新吞吐；这些不是本轮 smoke 通过条件，进入正式容量准入再测。

**Expected:** 原始 Run 和恢复 Run 成功；真实梯度更新、完整状态回读及 ONNX 对照通过。随机初始短训策略不要求学会走路，也不登记正式效果 pass。

### Task 6：多轮反思、回归与交付

**Files:** 进度记录、计划复选框及必要设计索引说明。

- [x] 完成五轮执行复核：算法/数据/codec；reset 与恢复；模板/镜像/哈希；导出 command 前端遗漏；smoke 与正式资格边界。
- [x] 对本次改动做最终代码复核；修复 command→z 产物遗漏，并留有 RED→GREEN 行为检查。
- [x] 跑聚焦容器回归、Ruff、`git diff --check` 和外部 RLinf 内容 hash 对照；记录实际结果及未验证项。完整 `make test` / Rust 全量测试不在本轮 smoke 范围。
- [x] 更新计划状态及 Job/Run/Model/Artifact 引用；不执行 Git 提交。

**Expected:** 本轮接入 smoke 已通过；GPU/RAM 峰值等容量指标仍未测。正式 Flat/通用行为/少样本适应/真机效果仍未验收，数据 `training_ready=false`。
