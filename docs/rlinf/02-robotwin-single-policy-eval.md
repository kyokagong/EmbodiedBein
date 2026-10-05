# RoboTwin 单 Policy、多环境评估的 RLInf 改动设计

日期：2026-09-20。本文定义 EmbodiedBein 通过 RLInf 执行 RoboTwin 全量评估时的目标运行架构。本文描述的是已选定的 B 方案：一个评估容器、一个 Ray 集群、一个常驻 Policy Worker、多个 Env Worker。它替代当前“每张 GPU 启动一个 rollout rank，并为每个 case 波次重启整套 RLInf”的实现。

平台总体评估语义、冻结 Suite、模型兼容、证据和 qualification 仍以 [`../design/09-robotwin-evaluation.md`](../design/09-robotwin-evaluation.md) 为准。本文只展开 RLInf 运行拓扑、批量推理、环境生命周期、资源画像、补丁路径和验收方法。

`/data/kk/RLinf` 和 RoboTwin 外部源码保持只读。本文列出的外部路径均为补丁目标；实现时由平台固定内容快照，在任务镜像构建副本中应用最小补丁，并记录 upstream revision、内容 hash、补丁 hash 和镜像 digest。禁止运行时回写外部仓。

## 1. 背景与根因

当前 RoboTwin 多卡评估使用以下 placement：

```text
cluster.component_placement={env, rollout:0-(N-1)}
```

4 张 GPU 会产生 4 个 rollout rank，每个 rank 各自加载一份完整 π0.5。实测 4 个 rollout 进程的 CPU RSS 合计约 46.7 GiB；4 GPU、每卡 1 个环境时容器总内存已经达到约 83.7 GiB。把环境并发提高到每卡 4 个后，渲染、图像和 Ray 缓冲进一步增长，最终超过 92 GiB 容器限制。

平台运行器还把完整 case 集按并发数切成小分片，并为每个分片执行一次 `subprocess.run()`。5000 case、并发 16 时会启动三百余次 RLInf 子进程，反复创建 Ray、加载模型和初始化 SAPIEN。第二个分片可能在前一个分片的内存尚未完全回收时进入启动峰值。

问题的根因不是 RoboTwin 的 16 个环境天然需要 92 GiB，而是：

1. GPU 数被错误地等同于 Policy 模型副本数；
2. case 波次被错误地等同于进程生命周期；
3. GPU 数、环境并发数、Policy Worker 数和推理 micro-batch 没有独立建模；
4. 全量评估默认配置仍允许视频与不必要的图像复制；
5. 资源画像只按 GPU 数选择，未包含并发环境、相机和模型加载峰值。

## 2. 目标与非目标

### 2.1 目标

1. 每个评估容器在任意时刻只存在一个 π0.5 Policy Worker 和一份模型参数。
2. 一个 Ray 集群覆盖同一 Eval Job 的完整生命周期。
3. 多个 RoboTwin Env Worker 将 observation 汇总给同一个 Policy Worker 批量推理。
4. Policy Worker 在 case 波次之间保持常驻；切换 RoboTwin task 时也不重新加载模型。
5. 评估保持 headless，但保留 VLA 必需的 EGL/Vulkan/SAPIEN 离屏三相机渲染。
6. 全量评估默认不录制视频，不保存逐帧媒体。
7. observation 和 action 通道必须有界，不允许图像在 Ray object store 中无界堆积。
8. 5000 个冻结 case 恰好执行一次；失败恢复只补未完成 case。
9. 继续由 RLInf 管理 Cluster、Worker、Channel、Policy 加载和评估调度。
10. 保持现有逐 case 证据、模型 receipt、聚合指标和报告接口兼容。

### 2.2 非目标

- 不建设跨容器 Policy Server。
- 不让多个 Eval Job 共享同一模型进程。
- 不为 π0.5 引入 tensor parallel 或 pipeline parallel。
- 不关闭三相机离屏渲染。
- 不修改训练、PPO、SFT 或 LIBERO 评估路径。
- 不新增数据库表。
- 不允许平台绕开 RLInf 直接运行厂商推理主循环。

## 3. 目标拓扑

以 4 GPU、16 个并发环境为例：

```text
单个 ebd 动态任务容器
└── 一个 Ray Cluster
    ├── Suite Coordinator（CPU）
    ├── RolloutGroup
    │   └── Policy Worker rank 0（GPU 0，唯一 π0.5）
    └── EnvGroup
        ├── Env Worker rank 0（GPU 0，4 env）
        ├── Env Worker rank 1（GPU 1，4 env）
        ├── Env Worker rank 2（GPU 2，4 env）
        └── Env Worker rank 3（GPU 3，4 env）
```

逻辑 placement 为：

```yaml
cluster:
  component_placement:
    rollout: 0
    env: 0-3
```

Policy Worker 数固定为 1。GPU 数只决定容器可见 GPU 和 Env Worker 的渲染分布，不再决定模型副本数。GPU 0 首期允许 Policy 与一个 Env Worker 共置；真实 probe 证明显存不足时，才增加“Policy 独占 GPU 0、Env 使用 GPU 1..N-1”的资源档位。

单卡使用相同结构：

```yaml
cluster:
  component_placement:
    rollout: 0
    env: 0
```

## 4. 数据流与背压

每个控制周期的数据流为：

```text
Env rank 0: B0 observations ┐
Env rank 1: B1 observations ├─ Channel merge ──> Policy Worker
Env rank 2: B2 observations ┤                       │
Env rank 3: B3 observations ┘                       │
                                                    ▼
                                     inference micro-batches
                                                    │
                                                    ▼
                                      merged actions [B, H, 14]
                                                    │
                                      split by recorded route
                     ┌──────────────────────────────┼───────────┐
                     ▼                              ▼           ▼
                  Env rank 0                    Env rank 1   Env rank N
```

RLInf `MultiStepRolloutWorker` 已有 `_merge_obs_batches()`、`_infer_env_batch_size()`、route 记录和 policy output 切分能力。首期复用这些能力，不新增 RPC 服务。需要补齐的行为是：

1. 一个 rollout rank 必须接收所有 Env rank 的 observation；
2. 合并顺序必须记录来源 rank 和 env slot；
3. action 必须按原来源精确拆回；
4. 一个 observation batch 在 action 返回前，Env Worker 不得继续产生下一批图像；
5. `max_inflight_batches` 首期固定为 1；
6. 任一 wave 结束前必须清空对应 Channel 消息。

有界同步模式优先于动态异步 batching。它牺牲少量潜在吞吐，但内存边界明确，也能保持 case 与 action 的确定映射。只有 profile 数据证明 Policy 长期空闲时，才引入最多 2 个 in-flight batch。

## 5. 推理 batch 与 micro-batch

`parallel_envs` 是同时推进的环境数量，`inference_micro_batch_size` 是单次 π0.5 前向处理的样本数，两者必须分离。

例如：

```text
parallel_envs = 16
inference_micro_batch_size = 4
```

Policy Worker 执行：

```text
16 observations
→ 4 × micro-batch(4)
→ 拼接 16 份 action chunk
→ 按 route 拆回 Env Worker
```

这样仍然只有一份模型，同时控制 GPU 0 激活和临时图像张量峰值。micro-batch 必须满足：

- 每个切片保持主相机、左右腕相机、状态和 instruction 对齐；
- 输出按原 batch 顺序拼接；
- action shape 固定为 `[B, num_action_chunks, 14]`；
- 单样本与 micro-batch 推理在固定随机输入下满足模型定义的数值容差；
- 发生异常时整个 observation batch 失败，禁止只返回一部分 action。

## 6. 全量 case 生命周期

### 6.1 从子进程分片改为进程内 wave

当前 `backend/eb/runners/robotwin_eval.py::evaluate_shards()` 为每个小分片启动一个新进程。目标实现保留确定性分波，但 wave 只表示 case 调度单位：

```text
创建 Cluster 一次
加载 Policy 一次
for task in frozen_suite_order:
    创建或重配置该 task 的 Env Worker
    for wave in task.case_waves:
        注入 wave 的 case_id 和 seed
        reset 对应 env slot
        执行到每个 case 终态
        原子落盘逐 case 证据
    关闭该 task 的 VectorEnv 和 renderer
关闭 Policy
关闭 Ray
聚合并封存报告
```

case wave 大小不超过 `parallel_envs`。尾波只启用实际需要的 env slot，禁止补虚假 case。

### 6.2 task 切换

RoboTwin task 的场景、资产和 step limit 不同。切换 task 时：

1. 停止当前 task 接收新 observation；
2. 等待所有在途 action 返回；
3. 检查当前 task 每个计划 case 恰有一个终态记录；
4. 调用 Env Worker 的显式 close；
5. 等待 VectorEnv 子进程和 renderer 退出；
6. 清空 task 级 Channel 状态；
7. 创建下一个 task 的环境；
8. 保持 Policy Worker、processor、norm stats 和模型权重不变。

Policy receipt 在第一次模型加载完成后生成；后续 task 必须引用同一个 receipt hash。

### 6.3 断点恢复

恢复集合严格按 case ID 计算：

```text
pending = frozen_plan_cases - valid_completed_case_ids
```

只有同时满足 plan hash、model receipt hash、task config hash 和 case 字段校验的记录才算已完成。恢复任务不得重跑合法完成的 case，也不能把旧模型或旧计划的证据混入新任务。

## 7. Headless 与离屏渲染

Headless 在本设计中的含义是：

- 不创建 GUI、viewer 或桌面窗口；
- 不显示调试画面；
- 全量评估默认不录制视频；
- 不保留帧历史和非必要媒体；
- 仍通过 EGL/Vulkan/SAPIEN 生成模型需要的 RGB observation。

π0.5 的 RoboTwin observation profile 使用主相机、左腕相机和右腕相机，因此不能关闭 Vulkan 或 renderer。容器仍保留：

```text
NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
MUJOCO_GL=egl
PYOPENGL_PLATFORM=egl
```

强制评估 override：

```text
env.eval.video_cfg.save_video=False
runner.only_eval=True
```

如果后续增加失败样本媒体，必须采用“终态后按需重放”或严格限量的失败录制，不能在 5000 case 主评估中为所有环境保存视频。

## 8. observation 内存优化

### 8.0 运行档位和容器内存门禁

控制面以不可变运行档位调用 RLInf，不接收任意 placement、Policy 数、环境数、memory 或 shm。
`memory_saver_sync` 对应本文已经实现的单 Policy 同步路径；`balanced_async` 和
`throughput_async` 分别要求动态 batching 与多 Policy 分片执行路径。在代码和持续 probe 完成前，
后二者只能作为带 blocker 的目录项展示。

每个可提交资源变体必须把 `memory_gb` 与 `shm_gb` 和拓扑共同冻结。Ray 95% 内存保护保持开启；
probe 发生 Ray worker eviction、内核 OOM、证据缺失或吞吐未达门槛时，该变体不能进入白名单。
高级配置只是已验证资源变体选择器，不是任意数字输入框。

`rlinf/envs/sim/robotwin/robotwin_env.py` 当前在 `center_crop=False` 时仍可能执行 NumPy → PIL → RGB → NumPy 转换。补丁应：

1. 原图已经是连续 `uint8 RGB` 时直接使用；
2. 只有确实需要 crop 或颜色转换时才创建 PIL 对象；
3. 三相机张量只组装一次；
4. Channel send 完成后释放原始 Python list 和临时数组；
5. `save_video=False` 时不建立帧缓存；
6. 禁止把完整 observation 历史写入 evidence；
7. 仅在测量证明有收益时启用 RLInf 已有 observation compression，避免无依据增加 CPU 编解码开销。

## 9. 模型加载与加载峰值

稳态只有一个 Policy Worker，但 checkpoint 加载期间仍可能同时存在模型参数和完整 state dict。实现应：

- 只允许 rollout rank 0 调用模型 builder；
- 优先使用 `torch.load(..., weights_only=True, mmap=True)`，前提是当前 PyTorch 和 checkpoint 格式支持；
- 加载完成后立即删除 state dict 并执行 `gc.collect()`；
- 不在 Env Worker 导入或构造 π0.5 模型；
- 不在 task/wave 切换时调用 `reload_model()`；
- `enable_offload` 在完整 suite 内关闭，避免每个 wave CPU/GPU 往返搬运整模型；
- receipt 记录 `policy_worker_count=1`、`model_load_count=1` 和实际模型 artifact hash。

若 mmap 不兼容，则保留普通加载，但资源画像必须覆盖加载峰值；不得通过关闭 cgroup/Ray 内存保护掩盖问题。

## 10. EvalPlan 运行协议

新计划使用以下运行字段：

```json
{
  "model": {
    "runtime": {
      "gpu_count": 4,
      "policy_worker_count": 1,
      "env_worker_count": 4,
      "envs_per_worker": 4,
      "parallel_envs": 16,
      "inference_micro_batch_size": 4,
      "max_inflight_batches": 1,
      "headless": true,
      "offscreen_rendering": true,
      "save_video": false,
      "reuse_policy_across_tasks": true
    }
  }
}
```

约束：

```text
policy_worker_count == 1
parallel_envs == env_worker_count × envs_per_worker
env_worker_count <= gpu_count
1 <= inference_micro_batch_size <= parallel_envs
parallel_envs % inference_micro_batch_size == 0，尾 wave 除外
max_inflight_batches == 1
headless == true
offscreen_rendering == true
save_video == false
reuse_policy_across_tasks == true
```

客户端只提交 GPU 数和期望总环境并发；服务端从已验证资源档位派生其余字段。客户端不能提交 placement、Policy Worker 数或 Ray 队列深度。

历史计划缺少新字段时可以读取和展示，但不得自动以旧的多副本语义提交新 Job。继续执行历史任务时必须重新 preview，生成新的 plan hash。

## 11. 资源模型与准入

删除仅按 GPU 数选择内存的固定表。CPU 内存估算改为：

```text
M_container =
    M_single_policy_steady
  + M_checkpoint_load_peak
  + env_worker_count × M_env_worker_base
  + parallel_envs × M_env_and_camera
  + M_ray_object_store_bounded
  + M_runtime
  + safety_margin
```

GPU 0 显存估算：

```text
V_gpu0 = V_policy + V_micro_batch + V_local_renderer + fragmentation
```

其他环境 GPU：

```text
V_env_gpu = V_renderer_base + envs_per_worker × V_camera_env
```

资源画像的身份至少包含：

- 模型 family 和 artifact hash；
- GPU 型号；
- GPU 数；
- Env Worker 数；
- 总并发环境数；
- 三相机分辨率和数据类型；
- inference micro-batch；
- video 是否开启。

首轮只开放真实 probe 通过的档位：

| GPU | Policy Worker | Env Worker | 每 Worker 环境 | 总并发 |
|---:|---:|---:|---:|---:|
| 1 | 1 | 1 | 4 | 4 |
| 2 | 1 | 2 | 4 | 8 |
| 4 | 1 | 4 | 4 | 16 |

某档位没有测量结果时，控制面将其标记为未验证并禁止提交。禁止提高 `RAY_memory_usage_threshold` 或关闭内存监控来通过验收。

## 12. 代码改动范围

### 12.1 EmbodiedBein

| 路径 | 必要改动 |
|---|---|
| `backend/eb/api/routes_eval.py` | 用已验证拓扑档位替代 `ROBOTWIN_EVAL_RESOURCES` 固定表；生成新 runtime 字段；校验 preview hash 和准入状态。 |
| `backend/eb/domain/rlinf_form.py` | RoboTwin eval override 改为单 rollout placement；冻结 headless、video、micro-batch 默认值。 |
| `backend/eb/adapters/rlinf.py` | 渲染分离后的 env/rollout placement；强制关闭视频；保持 graphics/EGL 能力和只读挂载。 |
| `backend/eb/runners/robotwin_eval.py` | 删除每 wave `subprocess.run()`；实现单 Cluster、单 Policy、task/wave 循环、断点恢复、清理屏障和最终对账。 |
| `backend/eb/runners/robotwin_env.py` | case/seed 注入、env slot 身份、逐 case 终态证据、wave reset、显式 close。 |
| `backend/eb/runners/robotwin_source.py` | 如果外部 RoboTwin close/headless 行为需要补丁，更新固定 patch、来源 hash 和应用校验。 |
| `backend/eb/capacity/estimator.py` | 按单模型、micro-batch 和环境数量计算显存；不再随 GPU 数复制模型权重。 |
| `backend/eb/orchestrator/collect.py` | 接受同一 receipt 下的 task/wave 证据；恢复时严格排除已完成 case；报告结构保持不变。 |
| `frontend/src/views/EvalNew.vue` | 展示一个模型副本、总环境并发、micro-batch、headless/offscreen/video 和测量资源；不暴露 Ray placement。 |
| `frontend/src/views/EvalJobDetail.vue` | 展示 task/wave 进度、Policy 加载次数、实际并发和峰值内存。 |

`ebd`、Docker 动态容器机制、数据库表和模型注册机制不需要架构改动。

### 12.2 RLInf 补丁目标

| 外部路径 | 改动 |
|---|---|
| `rlinf/runners/embodied_eval_runner.py` | 支持 Policy Worker 初始化一次、多个 task/wave evaluate、task 边界屏障以及最终统一关闭。 |
| `rlinf/workers/rollout/hf/huggingface_worker.py` | 单 rollout rank 聚合多个 Env rank；评估 micro-batch；按 route 精确拆 action；Policy 生命周期跨 wave 保持。 |
| `rlinf/workers/env/env_worker.py` | 配置 wave seeds、启用/禁用 env slot、task 级 close、Channel 清空和终态屏障。 |
| `rlinf/envs/sim/robotwin/robotwin_env.py` | 去除无意义 PIL 往返复制；关闭视频帧缓存；实现幂等 close。 |
| `rlinf/utils/placement.py` | 优先不改；仅当现有解析器不能表达 `rollout:0` 与 `env:0-(N-1)` 时增加最小修复。 |
| `evaluations/robotwin/robotwin_adjust_bottle_openpi_pi05_eval.yaml` | 将默认 placement 改为单 Policy 语义；video 默认关闭；配置 micro-batch 和有界队列。 |

不同任务不应复制配置文件。平台通过冻结 task config 和 Hydra override 选择 task，公共 topology 保持一份。

## 13. 前端交互

页面保留两个用户旋钮：

1. GPU 数；
2. 总并发环境数，仅提供已验证选项。

预览明确显示：

```text
GPU：4
Policy Worker：1
π0.5 模型副本：1
Env Worker：4
并发环境：16
推理 micro-batch：4
Headless：是
离屏三相机：是
视频保存：否
预计 CPU 内存：来自实测资源画像
```

原“每卡并发”字段不再直接展示或提交，避免将环境并发误解为模型副本数。后端仍可从总并发派生 `envs_per_worker`。

## 14. 证据扩展

保持现有逐 case JSONL 字段，并增加以下运行诊断字段：

```json
{
  "task_wave": 3,
  "env_worker_rank": 2,
  "env_slot": 1,
  "policy_worker_id": "rollout:0",
  "model_load_id": "sha256:...",
  "inference_batch_size": 16,
  "inference_micro_batch_size": 4
}
```

Policy receipt 增加：

```json
{
  "policy_worker_count": 1,
  "model_load_count": 1,
  "headless": true,
  "offscreen_rendering": true,
  "save_video": false
}
```

这些字段用于证明运行拓扑，不改变 success rate、timeout 或 qualification 的计算口径。

## 15. 错误处理

| 失败 | 行为 |
|---|---|
| 单个 case 达到 step limit | 写 timeout 终态，继续其他 case。 |
| 单个 Env slot 崩溃 | 当前 case 记 infra failure；停止该 wave；保留此前合法证据。 |
| Env Worker OOM | 停止新 wave；关闭全部环境；Job 失败并保存 partial report。 |
| Policy Worker OOM | Job 失败；不得自动增加 micro-batch 或换模型重试。 |
| Policy 推理返回错维/非有限 action | 当前 wave 失败；禁止把 action 发给环境。 |
| task close 超时 | 不创建下一 task，防止 renderer 残留叠加。 |
| Channel 中存在旧 task 消息 | task 切换失败，禁止继续。 |
| 模型加载次数大于 1 | 拓扑验收失败，即使 case 指标成功也不能标记完整评估通过。 |
| case 缺失、重复或 receipt 不一致 | 禁止生成 complete report。 |

平台只对明确的基础设施失败提供人工触发恢复；恢复仍使用同一冻结计划和模型，不自动改变并发、GPU、seed 或成功定义。

## 16. 测试

### 16.1 EmbodiedBein 单元测试

更新：

- `backend/tests/unit/test_eval_plan.py`
- `backend/tests/unit/test_robotwin_eval.py`
- `backend/tests/unit/test_rlinf_adapter.py`
- `backend/tests/unit/test_capacity.py`
- `backend/tests/unit/test_eval_receipt.py`
- `frontend/tests/robotwin-workbench.test.mjs`

必须覆盖：

1. 1/2/4 GPU 都只生成一个 rollout rank；
2. Env Worker 数和并发环境数派生正确；
3. preview 和 submit 的拓扑及 plan hash 一致；
4. 5000 case 分 wave 后无丢失、重复或补位；
5. 不再调用每 wave `subprocess.run()`；
6. task 切换时 Policy 对象和 receipt 不变；
7. 断点恢复只包含未完成 case；
8. 默认视频关闭且 offscreen rendering 保持开启；
9. 资源估算不随 GPU 数复制 Policy 权重；
10. 未验证的资源档位无法提交。

### 16.2 RLInf 补丁测试

新增或扩展 `/data/kk/RLinf/tests/e2e_tests/evaluations/` 对应快照中的测试：

1. 1 Policy + 1 Env Worker + 4 env；
2. 1 Policy + 2 Env Worker + 8 env；
3. 1 Policy + 4 Env Worker + 16 env；
4. 多 Env rank observation 合并顺序；
5. action 精确回到原 Env rank/env slot；
6. micro-batch 输出顺序和 shape；
7. task 切换后模型对象 identity 不变；
8. close 后 VectorEnv 子进程和 renderer 全部退出；
9. in-flight batch 永远不超过 1；
10. 单 case 失败不会污染其他 case 证据。

## 17. 真实验证与门槛

按以下顺序执行，不跳级：

1. 5 case、1 GPU、4 env smoke；
2. 16 case、4 GPU、16 env topology smoke；
3. 单 task 100 case；
4. 多 task 150 case；
5. 全量 5000 case。

每次记录：

- 容器 RSS 与峰值；
- 各 Ray Worker RSS；
- GPU 0 Policy/renderer 显存峰值；
- 其他 GPU renderer 显存峰值；
- Ray object store 和 `/dev/shm` 峰值；
- Policy 模型加载次数；
- observation batch 与 micro-batch 分布；
- 每 task 关闭前后的 RSS；
- 完成 case、重复 case、缺失 case 数；
- 环境步吞吐和端到端 case 吞吐。

验收标准：

```text
policy_worker_count == 1
model_load_count == 1
max_inflight_batches <= 1
5000 case 完整率 == 100%
重复 case == 0
未知 case == 0
每次 task 切换后 RSS 不呈单调增长
4 GPU / 16 env 峰值低于经 probe 冻结的容器限制并保留至少 15% 余量
同一冻结计划的单并发与多并发聚合结果在统计口径上可比
```

内存目标先以实测结果冻结，不在设计中承诺未经测量的绝对 GiB 数。此前约 40–55 GiB 仅是分析范围，不能作为验收值。

## 18. 实施顺序

### 阶段 1：单 Policy 拓扑

- 分离 rollout/env placement；
- 4 GPU 只加载一个模型；
- 完成 16 case topology smoke；
- 暂不改全量 case 生命周期。

### 阶段 2：进程内 wave

- 删除每 wave 子进程；
- Policy 跨 wave 常驻；
- 单 task 100 case；
- 验证内存不会随 wave 增长。

### 阶段 3：跨 task 常驻

- task 级 Env close/recreate；
- Policy 跨 task 保持；
- 多 task 150 case；
- 验证 renderer 和 Channel 清理。

### 阶段 4：全量与恢复

- 运行 5000 case；
- 验证断点恢复、完整对账和 partial report；
- 将通过测量的资源档位开放到页面。

每阶段必须保留现有证据协议。不得以 mock 成功代替真实 π0.5、真实 RoboTwin 相机和真实 action 执行。

## 19. 最终不变量

实现完成后必须始终成立：

1. 一个 Eval Job 对应一个动态任务容器和一个 Ray 集群；
2. 一个容器只有一个 π0.5 Policy Worker；
3. GPU 数增加不会增加 Policy 模型副本；
4. 多个 Env Worker 通过 RLInf Channel 共享同一个 Policy；
5. Policy 在完整 suite 内只加载一次；
6. headless 不等于关闭离屏渲染；
7. 全量评估默认不保存视频；
8. 队列和图像缓冲有明确上限；
9. case 波次不创建新的 RLInf 子进程；
10. task 切换只重建环境，不重建 Policy；
11. 所有结果仍绑定冻结 plan、模型 receipt、RoboTwin revision 和 task config hash；
12. 外部 RLInf/RoboTwin 仓保持只读，补丁和镜像可追溯。
