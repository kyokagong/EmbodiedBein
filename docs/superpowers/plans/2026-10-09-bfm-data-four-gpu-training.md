# BFM 数据补充、四卡、1000 次更新与录像评估执行计划

> **For agentic workers:** 使用 superpowers:executing-plans 在当前会话逐任务执行，不派生 agent。用户禁止 Git commit，所有阶段以文件和 Run 回执登记进度。

**Goal:** 获得合格 RPO behavior prior 后，通过 RLinf 完成四卡采样下新增 1000 次 BFM learner 联合更新，导出并执行带录像的独立评估。

**Architecture:** 复用 DatasetVersion、API/Job/Run/ebd、RLinf BFM worker、checkpoint 与评估生命周期；保持一个 learner，扩大 env/rollout rank。数据资格、四卡组合容量和独立评估桥分别验证后才能进入后续阶段。

**Tech Stack:** Python/FastAPI、RLinf/PyTorch、Isaac Lab、ONNXRuntime、Rust ebd、Docker Compose。

**Spec:** `docs/design/16-bfm-zero-data-and-4gpu-training.md`，以及 `docs/design/14-bfm-zero-rpo-rlinf-integration.md`。

## Global Constraints

- 不修改 `/data/kk/RLinf`、BHL 与外部数据源；不修改已发布本体接口、模型或数据版本。
- 计算任务经平台 Job/Run/ebd；开发检查容器化；不从 API subprocess 启动训练。
- 四卡首先用于采样，不启用四 learner；保持 FB pairwise batch 的单 learner 语义。
- 预算为新增 1000 次 BFM 联合更新，单独报告 env 控制步、transition 与 UTD。
- 未验证数据不升级 `training_ready`；未通过模型不改 qualification；不提交代码。

## Review Focus

- 原始祖先的重采样、切窗和镜像泄漏：必须同组划分 train/validation/test。
- root/body 状态坐标与实际速度：FK/静态检查不能替代真实物理跟踪。
- GPU0 共用 learner 时的峰值：必须包含优化器、全 rank 接收、replay、H2D 和保存。
- latent、reset 与跨 rank 终态：实际 action/z/next_obs 同步，不能把 reset 初态作为终态。
- 导出/评估版本混用：actor、normalization、reference、prompt、codec、数据和 suite hash 一致。

## Task 0：输入与设备证据

Files: `docs/progress/2026-10-09-bfm-zero-next-training-preflight.md`、`docs/progress/2026-10-09-bfm-motion-inventory.json`。

- [x] 读取当前差异、设计、进度与训练/数据/评估调用者；保留原 smoke 和其他工作区改动。
- [x] API 核查 BFM 数据、Flat 教师报告、环境/模板和 Doctor；识别四张卡及实际余量。
- [x] 用户确认 1000 次 learner 更新；记录当前没有新增数据。
- [x] 在离线限资源容器盘点 30 个动作，识别 pickle/joblib，固定每文件 hash；静态检查 50 Hz 连续窗口、实测关节限位和当前 PD 目标范围。
- [ ] 确定数据研究/商用用途及适用来源。此项不以等待时长或文件公开可下载视为确认。

验收：来源 inventory 可回读，未修改原文件；静态筛查结果明确不等于训练资格。当前已完成这部分只读证据。

## Task 1：候选动作、真实回放与数据资格

Modify: `backend/eb/api/routes_data.py`、`backend/eb/adapters/data_convert.py`、`backend/eb/adapters/mujoco.py`、`backend/eb/data_worker.py`、`backend/eb/data_validation.py`、`backend/eb/rpo_bfm.py`、`backend/eb/runners/rpo_bfm_motion.py`。Create: `backend/eb/runners/rpo_bfm_motion_probe.py`。Test: `backend/tests/unit/test_rpo_bfm.py`、`backend/tests/unit/test_rpo_bfm_platform.py`；runtime probe 回执。

- [x] 先为多格式固定 hash、旧 converter 兼容、坏 hash/shape、无连续窗口与祖先泄漏补最小行为检查，容器运行确认 RED。
- [x] 用新的受控转换 operation 接受固定选择清单；处理可信 pickle/joblib、保留完整时间轴、50 Hz 原生 23 关节/24 body codec；输出新 staging/manifest，拒绝覆盖旧版本。
- [x] 固定真实物理回放验证 recipe，记录完整回放/提前终止、误差、接触/足滑和硬限制。补测试：静态 FK 成功但回放失败不得提升 training_ready；来源资格缺失也不得提升。
- [x] 经平台转换/验证 Job 执行，发布新 DatasetVersion，记录有效连续时长和行为/祖先组覆盖。来源不足时调查已固定的外部 retarget 路线；未合格 Flat 教师不作替代。
- [x] 只有实测来源/质量条件满足，才通过原设计的数据准入；否则记录未满足项，不进入正式训练。实际：本地27/291、外部两轮均0/12；均保持diagnostic。

输出：新数据 ID、manifest/asset/codec/split hash、来源与真实动态回放报告。若本地数据不足，需外部合法来源或单独预算的合格教师；不能将 1000 次 BFM 更新预算挪作未经明确约定的教师训练。

## Task 2：正式准入与单 learner 多卡采样

Modify: `backend/eb/domain/rlinf_form.py`、`backend/eb/api/routes_train.py`、`backend/eb/orchestrator/service.py`、`backend/eb/adapters/rlinf.py`、`backend/eb/capacity/estimator.py`、BFM config/worker/policy、`deploy/patches/rlinf/patch_rpo_bfm.py`。Test: `test_rlinf_form.py`、`test_train_bindings.py`、`test_capacity.py`、`test_rpo_bfm_platform.py`，目标镜像 runtime。

- [ ] 先测试正式提交拒绝 training_ready=false/伪造数据绑定；API 与通用 Job 入口一致。受限 smoke 模式另行明确预算，不以诊断数据冒称正式训练。
- [ ] 定义 actor:GPU0、env/rollout:GPU1–3 的 placement，并保留 GPU0–3 采样候选；修改当前 gpus=1/batch=8 限制，按单 actor rank 校验 batch 和专家八帧整除，不套 PPO 四 learner 规则。
- [ ] 固定当前六组件网络 profile 和真实专家/replay batch；四 rank 的采样 seed 与 latent 上下文独立，检查接收拼接、实际 action/z、终止/截断末态和只读观测。
- [ ] 复核 rollout latent 批量性能；只有实测逐环境 Python/CPU-GPU 传输是瓶颈时才改，保持 resample/reset/bootstrap 语义的行为检查。
- [ ] 明确多采样 rank 的 checkpoint 覆盖/新 Run 初始化语义，不放宽旧 trainer_state 指纹。容器回归通过后构建任务镜像、登记 source/patch/image receipt，再更新 API 服务。

验收：真实四 GPU 短 probe 确认只有一个 actor rank、全部采样 rank 的 transition 进入 learner，发生真实 BFM 更新，完整 checkpoint/恢复覆盖与未覆盖项明确。

## Task 3：逐卡容量与组合吞吐

Create: `backend/eb/runners/rpo_bfm_capacity_probe.py`。Modify:现有 ASSET_VALIDATE tools adapter/准入估算。Evidence: 新 probe Job/Run 与每卡资源回执。

- [ ] 用固定 profile/batch 做模型+optimizer warmup+完整更新峰值测试，再测各卡 env/rollout。
- [ ] 从 512/1024/2048 env/rank 逐级测试，按实测内存趋势继续扩大；每卡至少留 2 GiB 和 10%，不通过反复 OOM 搜索极限。
- [ ] 组合测试 learner 接收全 rank 数据、normalizer、CPU replay/H2D、权重同步与 checkpoint；核对 RSS/RAM/shm，记录 env transitions/s、updates/s、UTD。
- [ ] 比较三采样卡与四采样卡实际总吞吐，在组合峰值通过的规模内尽量多启动 env。replay 至少覆盖一个联合 rollout，防止 rank 拼接后整段被淘汰。

验收：每卡角色/env/实测峰值表、最终规模和预计 1000 更新时长。独立 Isaac 环境数量或 Flat 历史 2048 env/卡不作为验收。

## Task 4：平台提交与完成 1000 次更新

Evidence: `/train/preview`、正式 TrainJob/Job/Run、resolved config、input receipt、指标、checkpoint/ModelVersion。

- [ ] 固定 Task 1 的训练数据、Task 2 的网络/batch/seed/拓扑与 Task 3 的环境数/资源。选择新的初始化语义，不误称原 smoke 的同状态续训。
- [ ] 将新增 BFM 联合更新预算固定为 1000；preview 核查实际 actor rank=1，GPU=4，env 数及 CPU/RAM/shm/time budget。每 50 次联合更新保存完整 checkpoint，日志每轮输出真实计数与六组件训练状态。
- [ ] 经现有训练 API 提交。监控非有限数、资源峰值、吞吐、transition/更新/UTD 与保存；失败保留原 Run，通过平台取消/恢复，不手改状态。
- [ ] 完成后回读模型/优化器/target/replay 状态，核对新增联合更新=1000、全部张量有限、最后完整 checkpoint 与模型血缘。

## Task 5：BFM 推理桥、录像与独立评估

Modify: `backend/eb/runners/rpo_bfm_artifact.py`、`backend/eb/api/routes_eval.py`、现有 eval adapter/collector、容量估算与 Suite/Profile 注册。Create: `backend/eb/rpo_bfm_eval.py`、`backend/eb/runners/rpo_bfm_eval.py`、对应 unit/runtime 检查。

- [ ] 导出本次 actor、normalization、command→z reference/prompts；验证 Torch/ONNX parity、hash/shape、命令范围及退化输入拒绝。
- [ ] 先写输入桥/reset/历史/实际 command-z-action 对齐、版本失配、坏 receipt/录像缺失的行为检查，再实现冻结 BFM eval plan 与实际物理 rollout，不套 Flat 780 维输入。
- [ ] 通过平台先运行少量独立诊断种子的站立/直行/侧向/转向录像，回看物理数据和可播放性；不以这些 case 代替正式套件。
- [ ] 使用独立冻结 BFM 完整 Flat 比较 revision，运行 100 case，每例 20 秒，记录每例录像、轨迹、失败原因和原机械/速度/yaw 判据；验证接收与报告分母完整。
- [ ] 输出通过数、RMSE、跌倒/硬越限/非有限、qualification、模型/Run/报告及视频链接。Job succeeded 与模型 pass 分开；结果差也如实交付。

## Task 6：逐轮复核与交付

- [ ] 数据复核：来源/祖先分组/有效窗口/动力学/数据资格与训练消费一致。
- [ ] 算法资源复核：单 learner、pairwise batch、真实 UTD、四卡峰值、所有 rank 数据与 checkpoint 覆盖。
- [ ] 用户入口复核：训练曲线/计数、模型血缘、完整评估分母、录像实际播放与失配/失败可见。
- [ ] 记录实际执行证据和未验证能力，更新进度；不提交 Git。

当前状态（2026-10-09）：Task0摸底和Task1历史数据转换/回放已执行；修正wxyz/xyzw并派生新数据，2更新RLinf smoke通过。实际Run见[本轮证据](../../progress/2026-10-09-bfm-data-replay-and-100style-trial.md)。用户最新指令禁止教师训练；本计划Task1原来的“PD/教师动态通过才准入”假设由[设计18](../../design/18-bfm-reference-prior-and-g1-rpo-retarget.md)纠正，先执行[参考重审计划](2026-10-09-bfm-reference-reassessment.md)。Task2–5仍待新的参考资格/准入实现与四卡证据，不修改旧数据状态或挪用1000次BFM预算。
