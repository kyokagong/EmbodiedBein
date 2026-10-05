# 04 · 模型与训练设计

## 1. 模型目录与能力

模型目录收录本地训练栈真实注册的 recipe，但“被发现”不等于“可训练或可部署”。能力按以下维度独立记录：

`load / SFT / RL / RLFT / offline_eval / closed_loop_eval / export / quantize / target_infer`

每条能力由 `(recipe revision, code snapshot, image digest, interface revision, task binding, device profile)` 定位，状态为 `discovered / adapter_ready / dependency_ready / smoke_passed / validated / unsupported / blocked`，并带证据、时间和失败原因。

一个 `ModelRecipe` 至少描述：

- builder、用途、训练算法/入口、配置路径和输入格式；
- checkpoint 文件集与 hash、processor/tokenizer、图像预处理、norm、state/action 映射、chunk/mask；
- full/LoRA/frozen/expert-only 等实际支持的微调方式；
- 适用本体接口、任务、rollout/value/log-prob 能力；
- 镜像、资源画像、导出后端、目标设备和许可证。

π0/π0.5、OpenVLA、GR00T、LingBot、DreamZero 等各走自己的已验证 recipe。控制策略、价值模型、生成模型不能因为同在 RLinf registry 就都出现在“机器人策略”选择框。

## 2. 训练输入与提交

训练路径：

```text
选择已发布 DataSlice 或纯 RL 场景
→ 选择本体接口、任务与 ModelRecipe
→ 解析数据/基座模型 ID 到不可变版本
→ 生成并校验唯一 JobSpec
→ 在目标镜像内解析配置和 loader probe
→ 资源准入、排队、Run
→ checkpoint 完整性校验
→ 注册 ModelVersion
→ 独立评估
```

预览与提交必须复用同一个 renderer。最终 JobSpec 的 `inputs` 包含真实数据版本、模型版本、接口、代码内容快照和镜像 digest，不能只在 TrainJob UI 行里保存 ID。

目标镜像内必须解析 Hydra defaults/search path/interpolation，再验证保护字段；仅检查字符串含 `=` 不够。输出路径、挂载、镜像、设备、网络、Ray、secret 和系统注入参数不可由高级 override 覆盖。launcher 使用固定程序和 argv；用户字符串不得拼入 `bash -lc`。

训练开始前写 `input_receipt.json`，至少记录数据/模型/processor/norm/interface hash、真实路径映射、features、有效窗口、样本摘要与 sampler。训练过程保存各来源实际采样计数。SFT 缺数据直接失败；纯 RL 可无数据，但必须冻结 Task、Scenario、reward/success、seed 与 normalization。

## 3. Adapter 与镜像

领域层只表达意图，adapter 负责把 JobSpec 翻译为 IsaacLab/RSL-RL、RLinf、MuJoCo、LeRobot 或部署工具的参数、挂载和输出观察规则。新增训练栈只增加 adapter/recipe 及快照测试，不修改通用领域对象。

Isaac 任务镜像从与目标 Isaac Lab 版本匹配的 `isaac-lab-template`/官方基础派生，固定 CUDA、驱动最低要求、Python 包、资产缓存和 entrypoint。大镜像由 `make images` 独立构建，不绑进 `make up`。

镜像能力通过容器内 probe 证明：GPU、EGL/Vulkan、任务注册、相机、模型加载、最小 forward/backward。API 容器看见文件或宿主 GPU 不代表训练镜像健康。

## 4. 4×4090 准入与调度

四张 24GB GPU 不是一张 96GB GPU。准入同时计算 actor、rollout、env、optimizer、activation、相机渲染、加载峰值、CPU offload，并检查 CPU、RAM、shm、磁盘、I/O 和非平台进程。

- 默认整卡独占；1/2/4 卡有独立资源画像，不静默改卡数或 batch。
- 先运行模型加载、一个小 batch 或最小相机 env probe，再进入长训。
- 容器内逻辑 GPU 依据已租用物理 UUID/索引生成，worker placement 不得越界。
- `--shm-size` 是上限，不是已预留 RAM；主机内存与容器限制一并核算。
- OOM 后改变 batch、环境数或 offload 产生新 spec，保留原失败，不冒充同一实验重试。

容量估算采用“相同配置实测 > 框架解析 > 目录经验值”的优先级，结果携带依据和误差范围。估算器负责准入和提示，不将预测值展示为实测峰值。

## 5. Checkpoint 与模型登记

Run 输出路径只由 run_id 决定。checkpoint 清单包含 logical name、step、完整文件集合、大小/hash、processor/norm 和完成状态。发现单个 `.pt` 或 `.safetensors` 不代表分片 checkpoint 完整。

- `latest` 是最大完整 step；`best` 由冻结 validation 指标和选择规则确定，两者独立。
- 每个 Run 默认只保留最新完整 checkpoint。新 checkpoint 原子提交后才能清理旧版本；半写目录不能淘汰最后完整版本。
- 模型登记幂等；只有 Run 已认领、清单完整的输出才能挂入该血缘。
- 训练成功、产物完整、评估通过、目标可部署是四个独立状态。

## 6. 恢复训练

恢复训练创建新的派生 TrainJob、Job 和 Run；源 Run、日志、checkpoint 和冻结配置保持只读。

| 模式 | 恢复内容 | 语义 |
|---|---|---|
| `exact` | 模型、optimizer、scheduler、各 rank RNG、step/epoch、sampler/dataloader 游标 | 全部状态有证据后才可使用 |
| `trainer_state` | 模型、optimizer、scheduler、RNG、step；环境/rollout 从 episode 边界重建 | PPO 等当前模式 |
| `warm_start` | 已发布权重和 processor/norm | 数据、拓扑或训练语义变化后的新训练 |

每次分布式保存完成并 barrier 后，由 rank 0 原子写 `resume-manifest.json`。候选校验 run_id、step、路径边界、无符号链接、DCP metadata、全部分片、大小稳定和兼容性指纹。没有完整清单的目录忽略。

恢复兼容性固定 recipe、数据/Slice、基座、接口/任务、镜像、代码快照、模型结构、processor/norm、optimizer/scheduler、seed、GPU/world size、FSDP、batch 和 sampler。表单只允许新名称和更大的终止目标。加载失败时新 Run 失败，绝不退回从头训练。

源 checkpoint 只读挂到 `/workspace/resume/...`，新输出仍在自己的 `/workspace/output`。排队和运行期间以租约阻止 GC。SFT 只有下一批 sample IDs 与未中断基线一致时才标 exact；PPO 环境未持久化时明确显示 trainer_state。

## 7. 训练工作台验收

训练详情必须显示冻结输入、resolved config、资源 probe、active Run、input receipt、日志/指标、GPU/RAM/IO、checkpoint 写入状态和血缘。latest/best/人工选择有不同标记。

最小故障验收包括：重复提交、服务重启重认领、两个两卡任务隔离、错误/恶意 override、半写/缺分片 checkpoint、源被 GC、恢复幂等、恢复后首个 step 与下一批样本，以及加载失败不污染源 Run。
