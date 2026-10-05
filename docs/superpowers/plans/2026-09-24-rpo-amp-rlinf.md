# RoboParty RPO AMP / RLinf 执行计划

> **For agentic workers:** 按 `superpowers:executing-plans` 逐任务执行，以[实施记录](../../progress/2026-09-24-rpo-amp-rlinf.md)核对已完成证据和剩余阻断。不得仅凭源文件存在勾选运行时任务；不执行 Git 提交。

**Goal:** 在既有 EmbodiedBein 管线内实现 RPO V2 的 RLinf PPO＋AMP 训练，交付可复核的直立站走模型、完整恢复状态和独立评估证据。

**Architecture:** 保留 Python 控制面、Rust ebd 执行面及 Job/Run/Artifact/Lineage。复用 RPO 物理 stepping 和 RLinf MLP/PPO，在受管 motion loader、环境 sidecar 和 AMP learner 中补齐数据与判别器；外部源码只读，补丁仅作用于镜像副本。

**Tech Stack:** Python、现有 NumPy/PyTorch、IsaacLab/PhysX、RLinf/FSDP、ONNX Runtime、MuJoCo、Docker；复用现有 Vue 页面及 pytest，不增加通用算法框架或服务。

**Spec:** [15 · AMP 接入设计](../../design/15-amp-rpo-rlinf-integration.md)，以及其引用的 [11 · 基础站走](../../design/11-rpo-stand-walk-phase-a.md)、[12 · 环境目录](../../design/12-environment-catalog-and-selection.md)。设计说明目标和协议，本文说明实现顺序、文件、检查和停止条件。

**状态：** 2026-09-24，T0–T6 的部分接入代码与纯 CPU 检查已实施，未完成目标镜像、官方数据实转、RLinf/Isaac 联调、训练及双仿真验证；逐项状态见[实施记录](../../progress/2026-09-24-rpo-amp-rlinf.md)。当前工作区有大量既有改动；保留用户改动，不 reset/clean，不暂存、不提交。下文未核验的 runtime 参数仍是实施目标，不是已证实能力。

## 全局约束

- 正式训练必须由 RLinf PPO/AMP 完成；禁止用 RSL-RL trainer 包装成 RLinf。
- 执行一律容器化；计算 Job 由 ebd 动态创建，只有 ebd 持有 `docker.sock`。平台部署 `make up`，任务镜像 `make images`。
- `/data/kk/RLinf`、RoboParty、BHL、原始数据源只读；输出只进受管 Dataset/Run/Artifact 路径。
- 不修改已发布旧 WBC 接口；新 locomotion 接口 `ee_rpo_v2_amp_locomotion_23dof_v1`，actor/critic/action 为 `780/1390/23`。
- 策略 50 Hz，物理 200 Hz，decimation=4，最长 episode 1000 策略步/20 s；跌倒 height<0.50 m 或 tilt>0.80 rad，超时与跌倒分开。
- 单 learner、同步 PPO，`use_training_pipeline=false`、`overlap_env_bootstrap=false`、`num_action_chunks=1`、`rollout_epoch=1`；首期多 GPU 用于独立 seed。
- Actor/Critic 三个 256/Tanh 隐藏层；D 两个 1024/512/ELU 隐藏层，输入 222，末层线性。
- motion 的 `purpose=motion_prior` 与现有 `usage=train|eval` 分开；不伪造 action 标签，不允许验证/验收数据参与训练或归一化。
- Actor 原始采样动作用于联合 log-prob；历史存实际执行动作。真实末态必须在 reset 前保存，timeout bootstrap 只加一次。
- 先基础自然站走，后启停/转向/鲁棒性和独立上身控制；这些扩展不自动获得首期模型能力声明。
- 没有运行证据的状态保持 pending/incomplete；容器成功退出、模型文件存在和训练 reward 上升不等于学会。
- 不新增平台远程关节控制、真机放行或 Gate override；仿真验收不等于硬件通过。
- 不使用 Git 提交代码；本计划没有提交步骤。

## 重点复核的五类失败

| 输入/条件 | 预期行为 | 对应任务和检查 |
|---|---|---|
| 一环境跌倒，另一环境继续或超时；含同时 terminated/truncated | 保存各自真实末态；只 reset 已结束环境；GAE 不跨回合；真终止优先 | T2/T3：两环境异步 reset、边界 continuation、手算 GAE |
| q 顺序错误、clip 接缝、yaw 旋转、并行 env 原点偏移 | 同姿态专家/Isaac/MuJoCo 特征一致；非法窗拒绝；原点不污染高度 | T1/T2/T6：黄金特征和坏样本 |
| 模板覆盖 YAML、未知 AMP 字段丢失、AMP mask 被当 PPO mask | resolved config 与预览一致；sidecar 完整往返；PPO 样本数不缩水 | T0/T3/T4：最终参数、序列化、真实梯度 |
| checkpoint 缺 D/buffer/RNG，数据/hash 变动，增加轮数上限 | 缺状态/不兼容拒绝；仅预算扩展可恢复且 scheduler 不重置 | T5：完整状态固定 batch 等价、预算扩展和拒绝用例 |
| 零动作滑行、硬越限但存活、缺 case/视频、NaN/空支撑统计 | 对应 task_failure 或 incomplete，不能以生存步数、空均值或退出码通过 | T6/T8：评估反例、完整性及人工 rubric |

## 顺序、工作量边界与交付物

```text
T0 冻结接口与配置
 → T1 motion 数据/loader
 → T2 环境/终态
 → T3 RLinf AMP 更新
 → T4 平台提交/收集
 → T5 完整恢复
 → T6 最小独立导出/评估
 → T7 smoke/资源探测/1000 轮诊断
 → T8 三 seed 对照/正式验收
 → T9 包络与鲁棒性
 → T10 上身 WBC
 → T11 目标板与交付
```

T0–T8 对应设计 P0–P6，是第一阶段完整仿真接入；T9–T11 对应扩展课程和 P7。T6 前置于长训，T8 负责冻结后的最终验收。数据不合格可继续 T0、合成数据的协议检查和评估器开发，但不得发布 training_ready 或开始正式 AMP 长训。

不估计“几天必收敛”。T7 输出吞吐和资源实测后才能计算 T8 GPU 小时。每一任务完成后将命令、代码/镜像/资产 hash、Run/Artifact ID、结果和剩余问题写入 `docs/progress/2026-09-24-rpo-amp-rlinf.md`；实施时创建该进度文件，本文不把计划标成进度。

### 最小文件分工

下表路径均相对仓库根；“新增”表示本计划需要的文件，其中部分已在工作区实现，完成度以实施记录为准。已有模块只增显式 AMP recipe 分支，不按关键词宽泛接受任意模板，也不借此重构 domain。

| 文件 | 操作/职责 |
|---|---|
| `contracts/ee_rpo_v2_amp_locomotion_23dof_v1.yaml` | 新增：固定 780/1390/23、历史、关节、PD/限制和 hash |
| `backend/eb/runners/config/rpo_amp_stand_walk_ppo.yaml` | 新增：PPO/D/motion 和固定同步设置 |
| `backend/eb/runners/rpo_amp_motion.py` | 新增：数组/manifest loader、命名映射、统一 74 维编码、连续窗采样及转换入口；训练路径不读 pickle |
| `backend/eb/runners/rpo_amp_extension.py` | 新增：独立 locomotion 观测、任务奖励、AMP 窗口；调用共享 stepping |
| `backend/eb/runners/rpo_wbc_extension.py` | 修改：最小共享点、实际执行动作历史及不推进历史的末态查询；旧 WBC 回归 |
| `backend/eb/runners/rpo_amp_worker.py` | 新增：D、固定 normalizer、历史 buffer、GAE 前 reward、PPO 后 D 更新及完整状态保存 |
| `deploy/patches/rlinf/patch_rpo_amp.py` | 新增：环境注册、AMP worker 选择、post-step sidecar、传输和保存钩子，按固定 patch 位点失败即停 |
| `deploy/images/rlinf-rpo.Dockerfile` | 修改：复制扩展、应用两组受管补丁，记录补丁/content hash、构建期 import 检查 |
| `backend/eb/adapters/data_convert.py`、`backend/eb/data_validation.py`、`backend/eb/api/routes_data.py`、`backend/eb/domain/data_view.py` | 修改：受管 motion 转换/发布、purpose 专用校验、Slice 和数据摘要 |
| `backend/eb/domain/rlinf_form.py`、`backend/eb/adapters/rlinf.py`、`backend/eb/api/routes_train.py` | 修改：模板、参数约束、训练用途、真实 Slice 挂载和恢复 |
| `backend/eb/bootstrap.py`、`backend/eb/environment_catalog.py`、`backend/eb/domain/environment_profile.py` | 修改：新 Task/接口/Profile/Suite，参数白名单及有界评估参数 |
| `backend/eb/runners/rlinf_entry.py`、`backend/eb/checkpoints.py`、`backend/eb/orchestrator/collect.py` | 修改：模型/恢复 manifest、完成性、AMP 模型身份、数据血缘和评估收集 |
| `backend/eb/runners/rpo_wbc_artifact.py`、`backend/eb/rpo_wbc.py`、`backend/eb/api/routes_eval.py`、`backend/eb/adapters/mujoco.py` | 修改：显式 codec、实际 ONNX、AMP 套件、按冻结 case 集验证报告 |
| `frontend/src/views/TrainNew.vue`、`DatasetDetail.vue`、`EvalNew.vue` | 修改：已有页面呈现 motion 数据用途、AMP 参数与正确评估入口，沿用现有 API/client |
| `backend/tests/unit/test_rpo_amp.py`、`test_rpo_amp_platform.py`、`test_rpo_amp_eval.py` | 新增：纯数据/契约、平台回路和评估反例的集中回归 |
| `backend/tests/runtime/rpo_amp_runtime_smoke.py` | 新增：目标训练镜像中按 `--case features|transitions|update|resume|export` 执行真实模块检查，落 JSON 回执 |

复用 `test_rpo_wbc.py`、`test_train_resume.py`、`test_dataset_usage.py` 和 `test_rlinf_entry.py` 覆盖旧分支；新测试不要只查源码字符串。Rust 如现有挂载/产物协议可表达 AMP，则不改 Rust；只有实测发现协议缺口才修改对应 crate，并保留其安全限制。

### 检查的运行位置

纯控制面最终仍需沿用仓库容器测试方式；当前只有本地 Python/前端检查通过，目标镜像容器检查尚未执行：

```bash
docker build -f deploy/test.Dockerfile -t eb/test:local .
docker run --rm -v /data/kk/EmbodiedBein:/workspace:ro -v /data/kk:/data/kk:ro eb/test:local python -m pytest -q -p no:cacheprovider tests/unit/test_rpo_amp.py tests/unit/test_rpo_amp_platform.py tests/unit/test_rpo_amp_eval.py tests/unit/test_rpo_wbc.py tests/unit/test_train_resume.py tests/unit/test_dataset_usage.py tests/unit/test_rlinf_entry.py
```

它们是现有 `make test` 的开发检查方式，不是在 API 内取得 docker 权限。T1–T6 的 Isaac/PyTorch 模块检查沿用容器化开发检查，使用与训练同一 digest 的 RPO 镜像、只读测试脚本和独占临时输出；runtime 脚本挂载到 `/workspace/tests/runtime/`。T7/T8 的完整训练和评估必须走 ebd Job/Run。下列 runtime 命令均指**检查容器内部**，不能直接用宿主 Python 运行：

```bash
/workspace/isaaclab_extension_template/.venv/bin/python /workspace/tests/runtime/rpo_amp_runtime_smoke.py --case update --output /workspace/output/amp-update.json
```

脚本只能选择列出的 case，不向平台用户开放任意 shell；平台训练的入口/路径仍由 adapter 冻结。最后执行相关回归和 `make test`；前端检查放在既有容器构建流程。每次代码变更只先跑相关检查，通过后不重复无关长测。

## T0：冻结接口、网络和可核算的配置

**文件：** 新 contract、AMP YAML、`test_rpo_amp.py`；读取旧 WBC contract、`MLPPolicy` 和 `patch_rpo_wbc.py`，不修改外部仓。

**输入：** 设计 §4/6/8；当前资产关节名称和物理 profile。

**输出：** 新 contract 草稿、一个候选 recipe、dim/codec/feature hash；资源 profile 在 T7 前仍未验证。

- [ ] 在 `test_rpo_amp.py` 写 contract 加载检查，断言 780/1390/23、10 帧、50/200 Hz、canonical 23 个唯一 joint names；对调关节列表而不更新 mapping 必须拒绝。先运行并确认因新 contract 缺失失败。
- [ ] 复制必要 WBC contract 内容到新身份，去掉 11 维上身命令；保留逐关节物理参数来源。依据实际 probe 通过后再发布 active，文件存在不发布。
- [ ] 从现有 PPO YAML 派生 AMP YAML，明确三网结构；继承既有 critic 补丁，禁止模型 builder 静默落回 actor 输入维度。
- [ ] 用以下数值断言固定单 learner 的布局检查；它们在 T4 必须作用于最终解析配置，而不是只读 YAML：

```python
profiles = [(8, 64, 32), (256, 2048, 512), (512, 4096, 1024)]
for envs, global_batch, micro_batch in profiles:
    transitions = envs * 24
    assert transitions % global_batch == 0
    assert global_batch % micro_batch == 0
    assert transitions // global_batch * 5 == 15
assert 512 * 24 * 5000 == 61_440_000
```

- [ ] 将 PPO clip=0.2、gamma=0.99、lambda=0.95、5 epochs、lr=3e-4 和 D 参数写入 resolved recipe；首期沿用当前 PPO entropy 设置并记录实际 log-std，不把探索噪声关闭，也不未经对照改上游 symmetry/网络。
- [ ] 跑 contract/配置检查；以 smoke/probe/formal 三组 profile 保存预期解析快照。

**验收/停止：** 形状、关节和 timing 全部一致；如果资产不能确定 soft/hard limit 或坐标定义，保持接口草稿并记录缺项，不从其他机器人猜参数。

## T1：发布可用的 motion Slice，并证明 loader 真正读取

**文件：** 新 `rpo_amp_motion.py`，数据 adapter/validation/API/data_view，`test_rpo_amp.py` 和 runtime `features` case。

**输入：** 外部候选动作只读清单、批准资产、来源/许可；T0 编码协议。

**输出：** 不含 pickle 的受管数组文件、manifest、不可变 Slice、训练集 normalizer、质量报告、input receipt 和回放视频。

**实现接口：** 以下名称在本任务定义，后续任务复用；数组数值采用 FP32，时间戳保留足够精度：

```python
import numpy as np

def encode_amp_frame(*, root_ang_vel_b, q_abs, dq, key_body_pos_b,
                     projected_gravity_b, root_height, root_lin_vel_b):
    # 输入末维分别为 3,23,23,(6,3),3,1,3，输出 [...,74]。
    return np.concatenate((root_ang_vel_b, q_abs, dq,
        key_body_pos_b.reshape(*q_abs.shape[:-1], 18),
        projected_gravity_b, root_height, root_lin_vel_b), axis=-1)

def valid_window_ends(clip_id, valid):
    # 边界、不连续时间和非法数值在发布验证中先标记 valid=False。
    return [i for i in range(2, len(valid))
            if all(valid[i-2:i+1]) and clip_id[i-2] == clip_id[i-1] == clip_id[i]]
```

GPU 环境按同顺序使用 torch；以黄金输入对照上面的 CPU 编码。不要把 GPU 帧搬到 CPU 再算特征。loader 定义 `MotionWindowSampler.sample(count)`，返回 `windows[count,3,74]` 和可复核的 `(clip_id,end_frame)`；实现 `state_dict()/load_state_dict()` 保存随机数和采样配置。函数入口检查 shape、有限值和 hash；上述代码是核心排列/边界算法，不能代替完整文件验证。

- [ ] 先写反例并运行：

```python
from eb.runners.rpo_amp_motion import valid_window_ends

def test_windows_never_cross_clip_or_invalid_frame():
    assert valid_window_ends([0, 0, 0, 1, 1, 1], [True] * 6) == [2, 5]
    assert valid_window_ends([0, 0, 0, 1, 1, 1],
                            [True, False, True, True, True, True]) == [5]
```

- [ ] 清点少量站立、低速直行、启停片段，记录原始 group/来源/许可/资产与文件 hash。来源未清的片段不进入正式 Slice。
- [ ] 在无凭据、无出网、最小只读挂载的转换容器导入旧 PKL；优先发布 `.npy`＋JSON、`allow_pickle=False`，逐文件 hash，拒绝越界路径/符号链接。外部源码不产生缓存。
- [ ] 按设计 §5 做时间重采样、四元数连续性、速度/FK、接触和物理可行性检查。补 yaw=90°、单关节、env origin 偏移和错序黄金用例；回放与动力学跟踪分别出报告。
- [ ] 按原始动作/session/近重复分组划分 train、validation、held-out reference；派生/镜像片段沿用原 split。训练采样先平衡 stand/walk，再在合法窗口内采样。
- [ ] 只用 train 计算固定 mean/std 和按物理单位确定的 std 下限，写入 feature/normalizer hash；站立或行走缺失、所有有效窗为空、文件/hash 错误立即拒绝。
- [ ] 将 `purpose=motion_prior` 接到专用验证分支，保留 `usage=train|eval`；API 只处理小 manifest/索引，数组解析与 hash 在任务容器/数据面完成。
- [ ] 在目标训练镜像实际采样，receipt 记录 train Slice hash、有效秒数/group/窗口数和实际样本 IDs；断言 IDs 全在 Slice 内，跨 split 的 normalizer 或 sampler 引用拒绝。

**验收/停止：** loader/特征/来源/回放全部有证据才发布 training_ready。合成 fixture 只证明协议；缺数据时停止正式长训，不以“文件存在”补齐证据。

## T2：环境末态、历史与动作协议

**文件：** `rpo_amp_extension.py`、共享 WBC bridge、runtime `features/transitions`；`test_rpo_wbc.py` 回归旧分支。

**输入：** T0 contract；T1 编码与黄金姿态。

**输出：** `RPO-AMP-StandWalk-v1`，每步 `states[env,780]`、`critic_states[env,1390]` 以及 reset 前 `amp_window[env,3,74]`、`amp_valid[env]` 和终止原因。

- [ ] 先构造两环境时序 fixture：A 第一步跌倒，B 继续；后续 B 超时，并加入 terminated/truncated 同为真的样本。记录 reset 次数、episode ID、每窗口末帧和 final critic obs；错误地先 reset 时检查必须失败。
- [ ] 提取最少的共享 hook，沿用现有 4 次 physics stepping；新任务单独构建 78/139 单帧输入，移除上身命令/中立位跟踪奖励，保留批准的任务/物理约束。
- [ ] 执行历史在裁剪/限位/延迟后写入；从 `q_target` 还原等效归一化执行值，不用原始采样值替代。原始动作另留给 log-prob。修正共享行为后重新生成 codec hash，旧模型不继承修正后的验证状态。
- [ ] 在 post-step、reset 前更新 AMP 三帧环；初始不足三真实帧时 valid=false，跨 rollout 不清空，partial reset 仅清本 env。读取 terminal 或录像不得再次推进 actor/critic 历史。
- [ ] 包装器保留所有环境的 AMP info 和真实 final obs，然后执行唯一 partial reset；不能只在 done 分支传 sidecar。
- [ ] 容器运行 `--case transitions`，再运行相关旧 WBC 检查；至少跨两个 24 步采集边界，证明未结束环境连续推进，而非每 24 步重置。

**关键判定序列：** reset 时只计初始状态 s0，执行第一个动作到 s1 时无完整三帧窗，第二个动作到 s2 时窗口为 `[s0,s1,s2]`；若 s2 跌倒仍是合法策略负样本。重置后的新 s0 不得接到旧窗口。

**验收/停止：** actor/critic/AMP 形状正确，时间轴与终态正确，旧 WBC 回归通过；不能用 padding 窗当真实运动来提高有效样本数。

## T3：在 RLinf learner 中完成真实 PPO＋D 更新

**文件：** `rpo_amp_worker.py`、`patch_rpo_amp.py`、RPO Dockerfile、runtime `update/transitions`。

**只读参考/镜像补丁位点：** RLinf `examples/embodiment/train_embodied_agent.py`、`rlinf/workers/env/env_worker.py`、`rlinf/data/schema/embodied_types.py`、trajectory builder/serialization、`EmbodiedFSDPActor.compute_advantages_and_returns/run_training` 及保存钩子。先追踪现有调用者，不复制完整 runner。

**输入：** T2 post-step 字段与既有 trajectory；T1 train sampler/normalizer。

**输出：** `RPOAMPActor(EmbodiedFSDPActor)`，在 GAE 前合成风格奖励，调用原 PPO 更新后训练 D；rollout worker 仍只推理 actor/value。

- [ ] 先写手算 GAE/奖励和 sidecar 往返反例；EnvWorker 必须在动作执行后合入当前 transition 的 `env_id/episode_id/step_id/policy_version/task_reward/bootstrap_reward/amp_window/amp_valid/terminated/truncated`，拆分合并/设备转移后与动作严格对齐。
- [ ] `forward_inputs` 若承载 sidecar，调用 MLP 之前必须取出；禁止训练字段流入 actor，禁止把 `amp_valid` 用作 PPO loss mask。
- [ ] 对 patch 每个位点校验唯一匹配及上游内容 hash；在镜像中先应用已有 critic patch，再应用 AMP patch。入口只对显式 AMP recipe 选择 worker，其余分支保持原样。
- [ ] 实现独立 D、Adam 和 FP32 gradient penalty；用以下核心奖励函数和数值检查，不改写任务奖励尺度：

```python
import torch

def style_reward(scores, valid, weight, dt=0.02):
    return dt * weight * (1.0 - 0.25 * (scores - 1.0).square()).clamp_min(0) * valid

scores = torch.tensor([-1.0, 0.0, 1.0, 3.0])
valid = torch.tensor([True, True, True, True])
assert torch.allclose(style_reward(scores, valid, 1.5),
                      torch.tensor([0.0, 0.0225, 0.03, 0.0]))
```

- [ ] 固定 D/normalizer 用 no-grad 生成整轮一次性 reward，再执行原 GAE；保留 bootstrap 独立项。两例必须匹配：`r_task=1,r_style=.03,gamma=.99,V_final=10,V_t=2`，真终止 delta=-.97，timeout delta=8.93；二者 advantage 均不得跨 reset 传播。
- [ ] 调用原 `run_training` 更新 PPO；old/new log-prob 对同一 23 维高斯样本先按关节求和，更新前 ratio≈1；style reward 不随 PPO epoch 重算。
- [ ] PPO 后按设计 §7.3 更新 D：5 epochs×4 minibatches，expert/policy 平衡，policy 当前/历史各半，空历史用当前；零有效窗跳过并计数，连续两轮为空停止。更新后才向容量 100000 的 FIFO 加本轮有效窗口。
- [ ] 课程以已冻结 transition 数推进：512 env 时前 122880 条 style=0，至 2457600 条升至 1.5；记录全局计数，probe 另用其冻结配置，不混用轮数和 transition。
- [ ] 容器 `--case update` 至少两轮，记录 actor、critic、D 各自参数差、optimizer step、reward version、梯度/有限值；固定 reward 后改 D 参数，已生成 reward/returns 必须不变。

**验收/停止：** 三个网络真实更新，真实 motion 样本被消费，GAE/末态/联合 log-prob 全部通过；只有 D loss 或日志变化不算训练接通。reward scale/gradient/sidecar 异常先修复，禁止用长训诊断接口错误。

## T4：接入平台预览、提交、目录、血缘和界面

**文件：** 模板/form、RLinf/data adapter、routes_train/data、bootstrap/environment_catalog/profile、rlinf_entry/collect、既有训练/数据页面、`test_rpo_amp_platform.py`。

**输入：** 新接口/Task、T1 冻结 Slice、T3 可执行镜像。

**输出：** `rpo_v2_amp_stand_walk_ppo` 在现有训练流程可选择；一个 JobSpec 在预览和执行中绑定相同数据、配置、接口和镜像。

- [ ] 先写单个端到端 adapter 测试：AMP 要求一个 train/motion_prior Slice，旧 WBC 仍无 motion 输入；错误 purpose/split、无效 hash、错资产、任意挂载路径和未知 AMP 参数均拒绝。
- [ ] 完整追踪 RPO 分支：kind 必须 `train.rl`，template/image、RPO 只读资产挂载、镜像内 RLinf、恢复允许列表、启动环境变量、entry 发布及 collect 都识别新 recipe；不能只改 `startswith` 后漏收集。
- [ ] 将 `motion_prior` 的受管路径和 hash 送入 learner，禁走 `data.train_data_paths` SFT 分支；实际 loader 生成 receipt，与提交 Slice 对账。
- [ ] 预览显示最终 env/horizon/rollout/batch 和预计 transitions；合并 YAML、模板、Profile 和用户白名单参数后再检查整除。验证正式候选确实为 512/24/1/4096/1024，而非旧模板默认 1024/128。
- [ ] 禁止多 learner、pipeline、overlap bootstrap、多 action chunks、reward filtering 和任意源码/entrypoint 覆盖。保护字段在预览与提交都校验。
- [ ] 新身份在 bootstrap 先以未验证状态注册；Profile 发布后不可原地修改。界面沿用已有表单，根据 requires_dataset/purpose 展示动作库，不展示“AMP 不需要数据”或“选择 VLA 基座”。
- [ ] `rlinf_entry.py`、collect 的 model card、io_signature 和 normalization 必须按新 codec 发布；移除这些路径对 890 的隐式依赖，保留旧 recipe 精确值。Lineage 包含 motion Slice、资产、代码、镜像和 checkpoint。
- [ ] 日志沿用已有标量链：task/style/total reward、D loss/梯度、有效窗数、stand/walk 倒地率/速度误差、探索 std；不在 FastAPI 解析大轨迹。
- [ ] 跑纯控制面检查和一个受管短 Job，确认从提交、Run、input receipt 到模型元数据/产物索引都能追溯；不以此提升模型效果状态。

**验收/停止：** 预览、resolved config、挂载与 receipt 相等，完整 Job/Run 路径成立；数据 ID 只显示在界面但未实际加载时本任务不通过。

## T5：完整 checkpoint 和可解释的恢复

**文件：** AMP worker、RLinf 镜像保存钩子、rlinf_entry/checkpoints/collect、`test_train_resume.py`、runtime `resume`。

**输入：** 已完成一轮更新的所有训练状态；既有恢复 fingerprint 和原子 manifest 流程。

**输出：** 原 actor/DCP 文件外增加 `amp/state.pt` 和 `amp/manifest.json`，纳入 checkpoint 文件/hash 清单；mode=`trainer_state`，环境状态明确不可 exact 恢复。

- [ ] 先写坏状态测试：逐个缺 D、D optimizer、normalizer、sampler RNG、buffer 或游标、critic/log-std、DCP shard；任何必需项缺失都拒绝 AMP resume，不静默回随机初始化。
- [ ] 保存 actor/critic/log-std、两组 optimizer、scheduler/scaler、D/normalizer、buffer、采样器/各 worker RNG、累计轮/transition、课程与 policy/reward version、数据/资产/代码/codec/hash。
- [ ] 对既有 resume schema 保留兼容：通用 `state_coverage` 继续原定义，AMP 专用完整性在算法 manifest 中明确列出并作为强制验证项；producer 与 consumer 同时校验，不只检查文件存在。大字节 hash 由任务/数据面完成。
- [ ] 仅轮边界临时目录写全分片并 barrier，最后原子发布 manifest；中断写入不得覆盖最近完整点。评估/best 引用的 checkpoint 保留，不参与未引用文件清理。
- [ ] runtime `--case resume` 保存状态，取固定 batch 做下一次更新；重建全部训练组件后恢复，再用同 batch 更新，对比 loss/参数/optimizer（容差按固定 FP32 平台记录），并对比下一批专家/历史 IDs。
- [ ] 恢复时环境从新 episode 起步、历史清空；保留全局课程计数。不要声称物理 rollout 与不中断训练逐位相同。
- [ ] 验证从累计 1000 增至总计 3000/5000：复用现有 fingerprint 对预算 override 的排除，但实际 scheduler 必须保留原语义/状态；若原调度依赖总轮数变化，拒绝当作普通 resume，先冻结兼容策略。env/batch/数据/奖励变动走新实验 warm_start。
- [ ] 故意终止一次保存过程并从最近完整点恢复；启动成功但只恢复 actor 不算通过。

**验收/停止：** 完整恢复有固定 batch 证据；缺状态、错 hash、损坏分片拒绝；AMP 推理包和可恢复 checkpoint 分别登记。

## T6：长训前先完成最小独立导出与评估

**文件：** rpo_wbc_artifact/rpo_wbc、routes_eval、MuJoCo adapter、Profile/Suite、collect、EvalNew、`test_rpo_amp_eval.py`、runtime `export`。

**输入：** T4 模型身份、T5 checkpoint 和新 codec；早期随机/短训权重也可以用于验证评估器。

**输出：** 实际 780→23 ONNX、数值 golden trace、独立 Isaac/MuJoCo report、逐 case 完整性；此阶段允许模型任务失败，但评估器必须如实报告。

- [ ] 先写反例：存活 1000 步但静止的 walk、越限但不倒、滑行 4 m 无交替迈步、提前跌倒、无支撑样本/NaN；均不得通过相应基础/自然度规则。
- [ ] 按显式 recipe/codec 构建 actor/normalization/history；不能以输入长度猜配置，也不能通过给 780 输入填零调用旧 890 导出器。
- [ ] 校验初始、普通运动、裁剪边界、终态和 partial reset 的整条 PyTorch→ONNX golden trace；不仅比较 action，还比较执行 `q_target`。FP32 最大动作绝对误差候选 1e-4，其他量按冻结单位阈值记录。
- [ ] 对旧 `case_success` 调用者做完整检索：当前 `stand_walk` 分支未检查 `hard_limit`，按既有设计补齐硬越限失败；旧结果不据新规则自动标通过。
- [ ] 实现 20 s 站/走基础判据；新 Suite 独立身份，validation 的 case seed 与最终 acceptance 隔离。原有 RPO seeds≤12 的 planner/adapter 限制不能承担 200-case 正式套件，改为校验冻结的 case 清单及有界规模，不开放任意无上限用户列表。
- [ ] 同一个交付 ONNX 在独立 Isaac 和 MuJoCo 任务运行；asset/PD/限位/接触/时间/关节名各自对账，禁止训练进程里回放模型代替交付文件检查。
- [ ] 每条 report 校验 case ID/seed、所有 hash、实际时长/步数、termination、randomization、metric availability 和媒体。duplicate/missing/unknown case 拒绝完整性；策略 NaN 生成 task_failure，未定义指标用 null，基础设施异常记 incomplete。
- [ ] 按设计 §10.2 实现姿态、膝屈曲、脚朝向、支撑滑动、着地交替和物理代价。接触迟滞 5%/2.5% 体重及两 physics step 做录像标定；稳态只统计 2–20 s，失败检查覆盖全程。
- [ ] 两仿真器跑少量 smoke case，确认坏模型得到 fail；12 个冻结基础 validation case 能独立执行、保存、汇总。Job succeeded 与 qualification 分开。

**可直接复用的最小反例断言：**

```python
from eb.runners.rpo_wbc_artifact import case_success
from eb.eval.wilson import wilson_ci

def test_survival_does_not_hide_hard_limit():
    assert not case_success("stand_walk", 1000, False, False, True,
                            0.0, 0.0, scenario="walk", distance_m=6.0)

def test_twelve_cases_do_not_meet_formal_confidence():
    assert wilson_ci(12, 12)["lower"] < 0.95
    assert wilson_ci(100, 100)["lower"] >= 0.95
```

**验收/停止：** 随机策略不要求站稳；要求评估流程正确完成、坏模型不能骗过判据。没有独立 validation 时不得开始 T8。

## T7：真实算法 smoke、资源 probe 和第一轮学习诊断

**文件：** 已有 AMP recipe/Profile、runtime 脚本、Run 配置/产物；不为阶段性试验新增训练框架。

**输入：** T1–T6 通过的协议、数据、镜像、恢复和评估。

**输出：** 真更新/恢复回执、吞吐资源报告、一条最多 1000 轮的诊断曲线和独立 validation。

- [ ] 8 env、24 horizon、global/micro=64/32，至少两轮真实 PPO＋D；本轮有效窗不足或全无有效窗按规则停止，不用 fake reward 跑通。
- [ ] 256 env、global/micro=2048/512 做资源 probe，测冷启动与稳态；统计完整轮耗时、峰值 VRAM/RAM/shm、D 开销、checkpoint 空间/I/O、NaN、超时和数据吞吐。
- [ ] 有资源余量再测 512 env、4096/1024；若 OOM 优先减 microbatch 并记录配置变化，再评估 env 数。不能在同一已冻结正式 Run 内悄悄更改。
- [ ] 冻结一个正式候选 Profile（默认 512），站/走 50:50，平地、无负载、0/0.3 m/s、默认站姿＋小初态偏差、无推搡/广域随机化；专家含合格 stand/walk。
- [ ] 按累计 10–50、200–500、1000 轮检查；每 25 轮做冻结 validation，每 50 轮存完整 checkpoint。站立/行走分开报告，不能让站立样本盖住走路失败。
- [ ] 1000 轮时按基础稳定/速度→自然度→风格距离/能耗顺序选 candidate；若所有回合仍立即跌倒或趋势不改善，停止扩预算，依次查关节/PD/终止、奖励尺度、特征归一化、数据范围和探索，而非先放大网络。
- [ ] 根据完整轮耗时估算 T8 剩余 GPU 小时，外加 validation/checkpoint、冷启动、视频和失败重跑；冻结计算/磁盘预算与停止规则。

**预算：** 512×24 每轮 12288 transitions、每轮 PPO 15 次 optimizer step。1000 轮=12288000；PPO 重用五遍不计为五倍环境数据。24 步=.48 s 采样段，完整 episode 跨采集继续到跌倒/20 s。

**验收/停止：** 本任务证明学习趋势和资源可行性，不能保证已经学会。资源 probe 的 epoch/hash 与正式比较分开；数据、梯度、reset 错误返回相应任务修复。

## T8：三 seed 对照、预算扩展与冻结验收

**文件：** 受管实验配置、Suite/Profile revision、Run/ModelVersion/评估 Artifact；进度文档记录实际结论。

**输入：** T7 固定 recipe 和选模规则、T6 validation/acceptance 实现、独立 motion reference。

**输出：** AMP-off/on 配对实验报告、每 seed 模型和视频、实际 ONNX 两仿真器结果、pass/fail/incomplete 资格结论。

- [ ] 固定三个配对训练 seed，例如 0/1/2，记录 actor/env/sampler 全部实际 seed；每组只变化 motion prior 的启用，网络/任务/初始化/transition 预算相同。AMP-off 不训练 D，也不给 style reward，报告因此节省的时间。
- [ ] 每个 seed 从 1000 轮检查点判断是否继续；最多扩至累计 3000/5000。比较采用共同 transition 检查点，不把提前失败 run 从汇总中删除；首轮诊断 run 只有配置在其开始前已冻结时才可计入正式对照。
- [ ] 3000/5000 分别是 36864000/61440000 transitions；三 seed×两算法×5000 上限=368640000，不含试验/评估。以多卡跑独立 Job 时使用 GPU UUID 和独立 Ray 端口。
- [ ] 只用 validation 选择 checkpoint，冻结 ModelVersion/ONNX/codec/asset/Suite hash 后运行 acceptance。12 个基础 case 全通过；另至少 100 stand＋100 walk 的独立 case，每关键层自然度/基础联合成功率 Wilson 95% 下界≥0.95。
- [ ] 保留启动阶段全部失败，不能仅统计存活到 2 s 的样本；姿态/机械指标逐项通过，不以综合平均抵消失败。
- [ ] 自然摆臂按设计三项 rubric 执行：预选至少 10 个直行 case 正/侧面全程视频、两名隐藏算法标签的评阅者、分歧复核。缺必需视频/评阅时该项 incomplete，不能用自动 D 分数替代。
- [ ] 两仿真器分别给出基础/自然度结果；报告每 seed、成功率区间、失败类别、力矩/能耗、训练/数据成本。声明 AMP 优于 PPO 时给配对差异与不确定性；差异不明确就只报告是否达标。
- [ ] 固定报告引用和最佳模型租约，发布通过范围（平地/无负载/站立/0.3 m/s）及未验证范围。5000 轮仍失败就保留失败结论，修改数据/奖励另开实验，不降低 Suite 门槛。

**验收/停止：** 达标才声明自然站走；AMP 不优于基线也是有效结果。验收失败不得回看 acceptance 调参并反复用同一套件冒充新的独立验收。

## T9：启停、速度包络与鲁棒性

**依赖：** T8 基础行为达标；对应命令/扰动的动作数据和参数有来源。

**文件：** AMP recipe/Profile、环境 reward/命令、Suite 和既有评估模块；不改变输入接口时仍需新的任务/实验修订。

- [ ] 先添加独立启停和 0–0.5 m/s 速度段的 case；训练前冻结加减速规则、分层门槛和数据覆盖。
- [ ] 单独加入横移/转向，缺合格数据则保持未支持；检查 root/body frame 与 keypoint 在转向时一致。
- [ ] 按实测逐项加入质量/惯量、摩擦、PD、噪声、时延/抖动和扰动，保存每 case 真实随机化值；名义性能和每关键扰动层分别验收。
- [ ] 确保新增软/硬限制和 delayed executed action 历史一致；资源/状态变化更新 checkpoint 指纹，必要时 warm_start，不伪装 resume。

**验收：** 只扩大通过独立套件的命令/扰动包络，不把最坏压力 case 混入未声明的基础能力比较。

## T10：独立上半身控制的 AMP WBC

**依赖：** T8/T9 下身能力；C5 启动前明确需要的上身通道、动作范围/速率和评价任务。当前计划默认 11 通道，不把未答复当额外能力确认。

**文件：** 新增 `backend/eb/runners/config/rpo_amp_walk_upper_ppo.yaml`；修改既有 AMP 环境/worker/codec 的显式 recipe、模板和 Suite；必要时派生新 contract，禁止改旧已发布接口。

**输出：** `rpo_v2_amp_walk_upper_ppo`，actor/critic=890/1500、23 actions、下身 D 输入138；独立模型/normalizer/数据特征 hash。

- [ ] 先写冲突反例：同一下身轨迹、不同合法上身参考，不得因专家自由摆臂先验惩罚被明确要求的抬臂；运动/碰撞/限位仍必须约束上身。
- [ ] 下身 D 每帧只取 root 角速度/重力/高度/线速度、12 腿关节 q/dq、双脚双膝局部位置：`3+3+1+3+24+12=46`，三帧138，删腰/臂 q 与肘 keypoint。
- [ ] Actor 接入 11 维上身目标，由单一策略输出全部 23 关节；部署不能事后覆盖手臂动作。重新训练 D/normalizer，维度变更使用新训练或显式共享层 warm_start。
- [ ] 从单臂到双臂、腰、组合、转向叠加和近限位变化；逐通道检查站/走稳定、速度及上身 RMSE 候选≤0.1 rad，同时冻结峰值和响应延迟指标。
- [ ] 肘 yaw/腰须有独立证据，不能继承上游转换器将肘 yaw 置零的“支持”。如实际 codec 与旧 890 接口不一致就发布新接口。

**验收：** 所声明的 11 通道均通过联合下身稳定检查；自然 locomotion 与上身 WBC 独立包，不在运动中热切换。自由摆臂/部分遥操动态切换仍需另一个明确 mode/mask 设计，不在本任务暗加语义。

## T11：目标板离线与交付证据

**依赖：** 对应仿真能力通过；目标 profile、V2 实物配置和标定已确认。硬件未知不阻断 T0–T8。

**文件：** 复用平台已有 TargetProfile/Bundle/交付与证据路径；无目标端真实实现时按设计单独列差距，不能凭文档生成伪证据。

- [ ] 先核对训练/硬件 URDF 的 effort/velocity/角度差异、踝部闭链、零位/符号、IMU 外参、固件与热限制；缺项保持 hardware_unverified。
- [ ] 目标板无电机权限运行完整 sensor-to-command-ready 离线 benchmark，测 p50/p95/p99、deadline miss、内存/温度/功耗；50 Hz 周期20 ms不等于推理可占20 ms，现有8 ms策略预算须按实测范围冻结。
- [ ] Bundle 只装 actor/codec/scale/默认位/PD/限制/runtime/签名与证据；校验错接口/资产/签名、过期传感器、NaN、超时、通信失联和 watchdog。
- [ ] 现场急停、支撑、停车与回滚由现场人员和本地 runtime 验证；平台只接收报告，不取得远程关节控制或真机放行权限。

**验收：** 目标板和现场证据独立通过才提升对应状态；只有仿真模型时交付仿真资格包，硬件仍 awaiting_target_evidence。

## 多轮反思与计划覆盖核对

| 轮次 | 核对问题 | 本计划处理 |
|---|---|---|
| 一：目标/网络 | AMP 是方法，实际训练与部署结构是否明确？ | T0/T3 的三网结构、780/1390/222 与部署 actor；T10 独立 WBC |
| 二：数据/时序 | 缺动作、错误坐标、边界窗口和 reset 能否被发现？ | T1 来源/黄金特征/split；T2 真终态；T3 sidecar/GAE 数值反例 |
| 三：样本/计算 | 256/512、24步、PPO五遍、batch 覆盖是否混淆？ | T0/T4 resolved config；T7/T8 精确预算，不承诺收敛 |
| 四：平台/恢复 | 只改 trainer 是否漏掉提交、收集、续训和导出？ | T4 全链与元数据；T5 D/buffer/RNG；T6 显式 codec 和 planner 容量 |
| 五：证据/效果 | 存活、滑行、硬越限、缺视频是否能骗过验收？ | T6 反例；T8 分层独立 case、人工 rubric 和不确定性 |
| 六：资源/范围 | 能否在验证器缺失时长训，或把仿真当硬件通过？ | T6 先于 T7/T8；T9–T11 分阶段证据与硬件边界 |

| 设计章节 | 执行覆盖 |
|---|---|
| §1–4 可行性/平台边界 | 全局约束、T0/T4 |
| §5 数据 | T1 |
| §6 环境/网络/接口 | T0/T2/T6 |
| §7 RLinf AMP | T3/T4 |
| §8 课程/样本/资源 | T7/T8/T9 |
| §9 恢复/发布 | T4/T5 |
| §10 独立评估 | T6/T8/T9 |
| §11 WBC/目标/现场 | T10/T11 |
| §12–14 验收/反思/不确定项 | 各任务验收、上述六轮复核、分阶段依赖 |

完成标准是 **T0–T8 的实现与运行证据齐全**，并按实际结果发布模型资格；若算法未收敛，必须给出失败结果和定位，不把“代码能运行”写成“基本走路＋直立已学会”。T9–T11 单独计进度。本文件交付仅代表总结与计划完成。
