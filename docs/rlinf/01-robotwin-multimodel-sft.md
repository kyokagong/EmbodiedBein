# RoboTwin 多模型 SFT 的 RLinf 改动设计

日期：2026-09-17。本文描述计划改动，不代表文件已经创建、模型已经下载或 smoke 已通过。

## 1. 目标与不变量

在 RLinf 现有 VLA SFT 链路中接入 RoboTwin ALOHA 数据，使 π0.5、LingBot-VLA-2、GR00T-N1.7、星海图 G0、原力灵机 DM0.5 和 Xiaomi-Robotics-1 能分别完成受控 smoke。训练循环、FSDP worker、梯度更新、优化器、checkpoint 和恢复继续由 RLinf 管理。

以下约束不能通过适配器绕开：

1. 厂商 Trainer、Lightning 或 VeOmni 主循环不能藏在模型 wrapper 内运行。
2. 旧 `lingbotvla` 不能改名冒充 LingBot-VLA-2；`dexbotic_dm0` 不能冒充 OpenDM DM0.5。
3. action 维度相同不代表本体兼容。每个模型必须绑定自己的字段映射、单位、frame、夹爪语义、normalization 和 action decode。
4. RLinf 运行时只读模型和数据，不下载依赖，不向模型源码写统计量。模型由平台下载到 `/data/kk/models/<model-id>/<revision>/`，注册 hash 后以相同绝对路径只读挂载。
5. `/data/kk/RLinf` 保持只读。补丁应用于固定快照的镜像构建副本，镜像清单记录 upstream revision、工作树内容 hash、补丁 hash 和依赖 lock。

## 2. 复用的现有链路

入口继续使用：

```text
examples/sft/train_vla_sft.py
  → rlinf.runners.sft_runner.SFTRunner
  → rlinf.workers.sft.fsdp_vla_sft_worker.FSDPVlaSftWorker
  → rlinf.models.get_model
  → model(forward_type=ForwardType.SFT, data=batch)
  → RLinf optimizer/checkpoint
```

不计划修改以下文件：

| 路径 | 复用原因 |
|---|---|
| `examples/sft/train_vla_sft.py` | 当前入口已经通用，不含 OpenPI 专用路径检查 |
| `rlinf/runners/sft_runner.py` | 已负责 SFT 生命周期和 step 调度 |
| `rlinf/workers/sft/fsdp_sft_worker.py` | 已负责优化器、反向传播和通用 checkpoint |
| `rlinf/models/embodiment/base_policy.py` | 已定义 `ForwardType.SFT` |
| `examples/sft/config/hybrid_engines/fsdp.yaml` | 首轮 smoke 复用现有 FSDP 配置 |

只有真实加载结果证明这些通用实现无法满足契约时才能扩大改动面。

## 3. 公共必改文件

补丁目标根目录均为 `/data/kk/RLinf`。

| 路径 | 类型 | 改动 |
|---|---|---|
| `rlinf/config.py` | 修改 | 注册 `lingbotvla_v2`、`galaxea_g0`、`opendm_dm05`、`xiaomi_xr1`，加入 `EMBODIED_MODEL`；GR00T 复用 `gr00t_n1d7`。校验每类模型必需的本地模型路径、processor/norm 和动作 profile 配置。 |
| `rlinf/models/__init__.py` | 修改 | 为四个新类型注册 `get_model` builder；GR00T 继续走已有 N1.7 builder。未知类型必须硬失败，不能返回一个近似模型。 |
| `rlinf/workers/sft/fsdp_vla_sft_worker.py` | 修改 | 在现有 `build_dataloader()` 分支加入五个待适配模型的数据加载器。保持 `get_train_model_output()`、checkpoint 和 RNG/data-loader state 逻辑不变。 |

首期直接延续当前显式分支，不新增通用 plugin registry。只有上游 RLinf 后续提供稳定注册接口时才迁移，避免为五个固定模型再造框架。

以下是条件改动：

| 路径 | 触发条件 | 改动上限 |
|---|---|---|
| `rlinf/hybrid_engines/fsdp/utils.py` | 模型自身 `_no_split_modules` 无法产生合法 FSDP wrap | 只增加对应模块名或 policy，不改变其他模型策略 |
| `rlinf/hybrid_engines/fsdp/fsdp_model_manager.py` | 官方 loss 必须使用现有 optimizer builder 无法表达的参数组 | 增加窄模型分支；首轮优先使用现有 AdamW |

条件未触发时不修改这两个文件。

## 4. 各模型补丁面

### 4.1 π0.5

π0.5 已有 RoboTwin ALOHA DataConfig 和示例配置，首期不增加模型实现。复用：

```text
rlinf/models/embodiment/openpi/dataconfig/robotwin_aloha_dataconfig.py
examples/sft/config/robotwin_sft_openpi_pi05.yaml
examples/sft/config/model/pi0_5.yaml
```

只新增 smoke 配置 `tests/e2e_tests/sft/robotwin_pi05_sft.yaml`。配置必须指向 RoboTwin ALOHA 模型版本和 norm，不能选择 LIBERO π0.5 基座冒充兼容模型。

### 4.2 LingBot-VLA-2

现有 `rlinf/models/embodiment/lingbotvla/` 使用 Qwen2.5-VL 和旧 `lingbot-vla-4b` 配置，保留为旧模型。V2 使用独立类型和目录：

```text
rlinf/models/embodiment/lingbotvla_v2/
├── __init__.py
├── lingbotvla_v2_action_model.py
└── sft_builder.py

rlinf/data/datasets/lingbotvla_v2/
├── __init__.py
└── robotwin_sft_data_loader.py

examples/sft/config/model/lingbotvla_v2.yaml
examples/sft/config/robotwin_sft_lingbotvla_v2.yaml
```

wrapper 加载官方 V2 processor、Qwen3-VL/MoE 模型和原生 loss，向 RLinf 暴露 `ForwardType.SFT`。RoboTwin profile 使用三相机和经过验证的 14D 双臂映射；55D 是模型内部表示，不能改变环境接口。future/depth teacher 输入只能出现在训练标签侧，推理时不得泄漏。

### 4.3 GR00T-N1.7

保留现有 N1.7 builder，补齐 SFT 和 RoboTwin modality：

```text
rlinf/models/embodiment/gr00t/gr00t_n1d7/
├── gr00t_action_model.py              # 修改：增加 ForwardType.SFT
└── robotwin_modality_config.py        # 新增

rlinf/data/datasets/gr00t/
├── __init__.py
└── robotwin_sft_data_loader.py

examples/sft/config/model/gr00t_n1d7.yaml
examples/sft/config/robotwin_sft_gr00t_n1d7.yaml
```

现有 `gr00t_action_model.py` 只处理默认 RL forward，收到 `ForwardType.SFT` 会抛出 `NotImplementedError`，因此这是必改项。模型必须使用独立 RoboTwin ALOHA modality/processor identity。若官方 N1.7 processor 的 `embodiment_id.json` 已能表达该 identity，就不修改共享 tag；否则在 `rlinf/models/embodiment/gr00t/embodiment_tags.py` 增加 `ROBOTWIN_ALOHA`，不能复用 `NEW_EMBODIMENT` 或 LIBERO tag。

只有 rollout 仍需手工环境转换时才修改 `rlinf/models/embodiment/gr00t/simulation_io.py`，增加 RoboTwin observation/action 映射；纯 SFT 能由 processor 完成时不改。

### 4.4 星海图 G0

```text
rlinf/models/embodiment/galaxea_g0/
├── __init__.py
├── g0_action_model.py
└── robotwin_profile.py

rlinf/data/datasets/galaxea_g0/
├── __init__.py
└── robotwin_sft_data_loader.py

examples/sft/config/model/galaxea_g0.yaml
examples/sft/config/robotwin_sft_g0.yaml
```

builder 必须锁定与 G0-VLA 权重匹配的历史源码和 processor，不能借用 G0.5。`robotwin_profile.py` 只有在关节顺序、状态/action 语义、相机和 normalization 均有可回放证据后才能发布。缺少映射时模板保持 `action_mapping_unverified`，不能靠截断或补零通过 smoke。

### 4.5 原力灵机 DM0.5

现有 `rlinf/models/embodiment/dexbotic_dm0/` 继续代表 DM0。DM0.5 新增独立实现：

```text
rlinf/models/embodiment/opendm_dm05/
├── __init__.py
├── dm05_action_model.py
└── robotwin_profile.py

rlinf/data/datasets/opendm_dm05/
├── __init__.py
└── robotwin_sft_data_loader.py

examples/sft/config/model/opendm_dm05.yaml
examples/sft/config/robotwin_sft_dm05.yaml
```

wrapper 复用 OpenDM 官方模型和数学 loss，但由 RLinf 执行 forward/backward/optimizer/checkpoint。数据加载器读取冻结的 `Dexmal/robotwin2-full` 派生 view 和 train-only norm；禁止运行时向 OpenDM 源码目录生成统计量。

### 4.6 Xiaomi-Robotics-1

```text
rlinf/models/embodiment/xiaomi_xr1/
├── __init__.py
├── xr1_action_model.py
├── robotwin_profile.py
└── action_codec.py

rlinf/data/datasets/xiaomi_xr1/
├── __init__.py
└── robotwin_sft_data_loader.py

examples/sft/config/model/xiaomi_xr1.yaml
examples/sft/config/robotwin_sft_xr1.yaml
```

`action_codec.py` 是必要边界：它从 RoboTwin joint14、相机和经固定 FK/TCP 得到的 EEF 状态构造官方 60D 相对 action/state，并实现预测动作到环境命令的逆变换。必须固定旋转表示、相对参考时刻、夹爪、保留位和 chunk mask；简单补零不构成适配。

## 5. 配置与模型路径

Hydra 配置只接收平台解析后的容器内绝对路径，不自行拼接仓库路径。每个 recipe 至少冻结：

```yaml
actor:
  model:
    model_type: <exact-model-type>
    model_path: /data/kk/models/<model-id>/<revision>
    processor_path: /data/kk/models/<model-id>/<revision>
    action_profile: robotwin_aloha_joint14_v1
data:
  train_data_paths: /data/kk/data/<managed-view>
```

真实字段可随模型不同，但必须由平台 renderer 从已注册 ModelVersion、TrainingView 和 recipe 生成。用户不能在高级 overrides 中替换 `model_type`、`model_path`、`processor_path`、数据路径、action profile 或 norm identity。模型目录必须包含完整 manifest、文件 hash、许可、上游 revision 和 processor/norm；仅有权重文件的目录不算可选基座模型。

## 6. 最小测试集

只新增一个参数化契约测试，避免为相同规则复制五套单测：

```text
tests/unit_tests/test_robotwin_multimodel_sft.py
```

它逐模型验证 builder 注册、loader 返回、batch schema、有限 loss、非零梯度和预期可训练参数变化。模型专属 codec 的数值往返放在同一文件的对应参数/用例中；只有 XR1 等复杂 codec 的测试明显失控时才拆文件。

每个模型新增一个真实 e2e smoke 配置：

```text
tests/e2e_tests/sft/robotwin_pi05_sft.yaml
tests/e2e_tests/sft/robotwin_lingbotvla_v2_sft.yaml
tests/e2e_tests/sft/robotwin_gr00t_n1d7_sft.yaml
tests/e2e_tests/sft/robotwin_g0_sft.yaml
tests/e2e_tests/sft/robotwin_dm05_sft.yaml
tests/e2e_tests/sft/robotwin_xr1_sft.yaml
```

复用 `tests/e2e_tests/sft/run_vla_sft.sh`，不新增六个 shell 脚本。每个 smoke 使用冻结的小 Slice，至少完成：

1. 实际加载首尾窗口、短 episode 和 chunk 尾部。
2. 一次模型前向，loss 有限且各 mask 生效。
3. 一次 backward 和 optimizer step，预期参数发生变化。
4. 保存 RLinf checkpoint，重新加载后用同一输入前向。
5. 导出 processor/norm 与模型 manifest，receipt 记录 sample IDs、配置、模型/数据/代码 hash 和峰值显存。

平台工程 smoke 可运行 20 step，但不能把它作为模型效果结论。标准训练和闭环评测另由冻结 Suite 验收。

## 7. 实施和准入顺序

| 顺序 | 模型 | 进入条件 | 退出条件 |
|---|---|---|---|
| R0 | π0.5 | 已有 RoboTwin 配置、正确 ALOHA 基座和数据 view | 建立公共 smoke 基线，证明平台渲染和 RLinf checkpoint 链路 |
| R1 | LingBot-VLA-2 | 官方 V2 revision、模型/processor、本体 mapping 已冻结 | V2 独立 builder/loader/loss smoke 通过 |
| R2 | GR00T-N1.7 | N1.7 processor identity 和 RoboTwin modality 已冻结 | SFT 分支、保存重载及 codec probe 通过 |
| R3 | DM0.5 | OpenDM DM05 权重、reader、loss 和 train-only norm 已冻结 | RLinf 管理 optimizer/checkpoint 的 smoke 通过 |
| R4 | G0 | G0 历史源码与权重严格配对，RoboTwin action profile 可回放 | smoke 通过；否则保持 blocked |
| R5 | XR1 | FK/TCP 与 60D 双向 codec 有数值和 replay 证据 | smoke 通过；否则保持 blocked |

顺序表示降低排障耦合，不降低六个目标模型的最终要求。每个模型独立升级状态，一个失败不阻止其他模型继续。

## 8. Ready 判定与失败原因

模型已下载、builder 已注册或配置文件存在，都不能单独让模板变为“就绪”。RLinf recipe 的 `smoke_passed` 必须同时满足：

- 精确模型类型已注册，模型 manifest 和所有文件 hash 完整；
- 数据 view 与 action profile 匹配，loader probe 读取真实样本；
- `ForwardType.SFT` 完成有限 loss、backward 和 optimizer step；
- checkpoint 保存、重载和同输入前向通过；
- 运行镜像 digest、RLinf 快照和补丁 hash 与模板声明一致。

失败必须返回稳定 reason code，例如 `rlinf_model_unregistered`、`sft_loader_missing`、`sft_forward_missing`、`action_mapping_unverified`、`model_files_incomplete`、`loader_probe_failed`、`checkpoint_reload_failed`、`runtime_digest_mismatch`。平台据此显示具体原因，不以一个无上下文的“未就绪”代替诊断。

## 9. 改动规模边界

确定的共享修改只有 3 个文件；新增内容集中在模型 wrapper、数据 loader、Hydra 配置和测试。FSDP 两个核心文件保持条件改动。首期不建设新的 RLinf 插件系统、通用 VLA schema、第二套 checkpoint 管理器或厂商训练子进程。

