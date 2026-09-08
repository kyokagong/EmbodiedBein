# 04 · RLinf 模型目录与训练管线

## 1. 默认纳管，分能力验证

本地 `/data/kk/RLinf/rlinf/models/__init__.py` 静态扫描有 **26 个 builder 注册调用**，其中 24 个 category=embodied、2 个 diffusion。这不是 24 个均能操作三类机器人的已验证模型：其中包括控制策略、价值模型、变体包装和大模型。

默认目录覆盖本机真实注册项；不自动下载全部权重，不默认加载自定义远程代码。按用途过滤而不是把世界模型/价值模型全摆进“机器人策略”选择框。

| 类别 / 配方族 | 本地 builder/配置证据 | 当前 FSDPVlaSftWorker 有分支？ | 平台接入策略 |
|---|---|---|---|
| π0 / π0.5 | openpi、openpi_rlinf；pi0/pi0_5 配置 | 有，两条实现分别验证 | 默认候选；优先补齐 SO101/BHL 数据处理与 action 配置 |
| OpenVLA / OpenVLA-OFT | openvla、openvla_oft | 本 worker 无 | 展示上游 RL/推理候选；SFT 显示 missing_adapter |
| GR00T 系列 | gr00t、gr00t_n1d6、gr00t_n1d7 | 本 worker 无 | 有注册不承诺 SFT；LeIsaac 独立训练能力也不等于 RLinf SFT |
| LingBot-VLA | lingbotvla | 有 | 检查 repack/processor 与数据适配 |
| EVO1 | evo1 | 有 | 同上，独立 recipe |
| DreamZero | dreamzero；5B/14B 配置 | 有 | 强容量预检；4×24GB 不代表任意规模均可运行 |
| Cosmos3 | cosmos3；SFT 配置 | 有 | 先验证模型语义、资源、数据和入口 |
| MolmoAct2 / ABot-M0 / StarVLA | molmoact2、abot_m0、starvla | 本 worker 无 | 纳管目录；逐任务验证 RL/推理，不强开 SFT |
| Dexbotic | dexbotic_pi、dexbotic_dm0 | 本 worker 无 | 依赖与环境配对验证 |
| 常规控制策略 | mlp_policy、rlt_mlp_policy、rlt_td3_mlp_policy、cnn_policy、flow_policy | 不走此 VLA SFT 入口 | 用于 locomotion/WBC 或相应 RL recipe |
| 包装/价值辅助 | cfg_model、recap_value_model、steam_value_model | 本 worker 无 | 标为 guidance/value，不当电机策略 |
| 生成辅助 | sd3、wan22_ti2v_5b，category=diffusion | 不走此入口 | 可作为数据生成研究候选，不混入动作模型目录 |

证据仅适用于本轮读取的本地文件，完整名称/行号/指纹见 [证据索引](evidence/local-sources.json)。上游会演进；更新源码/镜像后使旧能力验证过期，不用在线官网覆盖本地事实。

## 2. 模型能力是一组独立证据

每条 capability 以 `(model recipe revision, code snapshot, image digest, interface revision, task binding, device profile)` 定位，状态取 discovered / adapter_ready / dependency_ready / smoke_passed / validated / unsupported / blocked，并带 evidence_ref、时间与失败原因。

能力维度：load、SFT、RL/RLFT、offline_eval、closed_loop_eval、export、quantize、target_infer。一个阶段通过不会自动提升其他阶段。例如 SFT loss 可计算，不代表 closed-loop eval 已能还原动作。

本地额外断点：`examples/sft/train_vla_sft.py:validate_berkeley_sft_paths` 直接读取 `cfg.actor.model.openpi.config_name`；对非 OpenPI 配置可能在创建 worker 前就失败。`FSDPVlaSftWorker.get_eval_model_output` 明确未实现 embodied SFT eval。通用接入要用受控的兼容入口或等待上游版本修复，不能仅有 worker 分支就标 smoke_passed；评估安排为独立 Job。

## 3. ModelRecipe 最小内容

- builder key、模型用途、明确支持的训练入口/算法与配置路径、支持的输入格式。
- checkpoint manifest、文件 hash、processor/tokenizer、resize/crop/color、norm、state/action 映射、image key、embodiment token、action chunk/mask。
- finetune 范围：full / LoRA / frozen backbone / expert-only，仅显示该模型实现支持的方式；不能因通用 `is_lora` 字段存在就承诺所有模块正确接入。
- 环境需求：可用 observation/action interface、rollout worker、value head/log-prob 算法能力；不支持策略梯度的 model wrapper 不自动支持 PPO。
- 已验证镜像、资源画像、导出后端、目标设备画像和模型许可证引用。

数据挂载只改变容器可见文件；还必须将路径映射到真实 loader 配置。例如通用 SFT 的 `data.train_data_paths`、`actor.model.model_path` 与模型特有 norm/assets 字段，按 recipe 生成。BHL successful-view 不是普通 LeRobot 根目录，不能直接互换。

## 4. 训练管线与输入回执

`选择数据 Slice/纯 RL 场景 → 选机器人接口与 ModelRecipe → 冻结配置 → CPU/小 GPU probe → 排队 → Run → checkpoint 完成校验 → 注册 ModelVersion → 独立 Eval Job`。

提交前解决以下输入：

1. 从 dataset/base_model **ID 解析到具体发布版本**，验证存在、权限、格式和用途；不能只把 ID 保存到 TrainJob。
2. 生成唯一 JobSpec，填完整 `inputs`、任务/接口、代码快照、镜像 digest、输出规则；预览和提交使用同一 renderer。
3. 在目标工具/训练镜像内 compose Hydra，resolve defaults/searchpath/配置插值，并用上游 schema/校验函数检查。拒绝未许可 resolver/import；容器没有 secrets/docker.sock/真机设备。
4. 校验 image/camera/state/action/norm 与 loader 实读结果；输出 `input_receipt.json`：实际路径对应的版本 hash、features、episodes、有效窗口、sample IDs 摘要、processor/norm hash、model hash、接口 hash、实际 sampler 配置。
5. 对比回执与冻结输入，错配立即结束；训练过程记录各来源实际采样计数，防止某个机器人的数据从未被采到。

纯 RL 的 data input 可以为空，但必须指定 Task、Scenario、reward/success 版本、seed plan 与 observation normalization 策略。SFT 缺数据不得自动变成空跑；RLFT 必须使用兼容基座、环境和算法，不把所有非 SFT 模板都映射为 train.rlft。采集从训练列表移到数据工作区，但复用 Job 调度。

## 5. 配置与启动安全

最小可靠方案是固定 launcher + 参数数组，显式启动 Ray 和 Python 训练入口；不要把用户字符串拼进 `bash -lc`。若旧镜像需 shell 包装，脚本内容固定，用户参数仅作为经过解析的 argv 传入，不执行用户模板脚本。

保护字段至少包括输出/缓存路径、挂载、代码/数据/模型绑定、镜像、设备、Ray/placement、secret/env 和网络模式。普通训练超参允许覆盖；最终解析后验证保护字段，不能只把强制项插在列表前面。Hydra 的带逗号 key/value、列表/区间和字符串必须按其语法正确传递并有 compose 验证，不照抄 UI 字符串。

当前上游 BHL SFT YAML 使用 runner.logger.log_path + experiment_name 组织 checkpoint；不能假定所有模板接受 `runner.checkpoint.save_dir`。每个 recipe 声明真实输出配置与观察路径，平台最终以 manifest 识别产物。

## 6. 4×4090 的准入与多角色放置

4 张卡各约 24GB，不是透明的 96GB 大卡。显存画像由 actor、rollout、env、optimizer、激活、相机渲染、加载瞬态组成；FSDP/offload 还消耗主机 RAM 和通信带宽。现场 doctor 报告主机约 128644 MB RAM，不能无上限 CPU offload。

默认整卡租约，按照 UUID/物理索引分配，容器内 worker placement 使用实际可见的逻辑设备。1/2/4 卡都有独立 profile；SFT 当前入口请求 actor strategy，RL 依 recipe 决定 actor/env/rollout 共享或拆分，不能把所有 worker 默认各复制一遍模型到所有 GPU。

准入同时检查 GPU、RAM、CPU、shm、磁盘、I/O 预算和其他非平台进程占用。`--shm-size` 是容量上限而非内存预留；容器 memory limit、宿主可用内存和共享内存使用一起计量。先执行短的模型加载/一小批前后向/最小相机 env probe，记录峰值与配置指纹再进入长训。

默认每任务独立网络命名空间，Ray 不发布端口，不跨 Job 自动发现。Redis 仍是 6380。只有需要 host 网络且通过明确兼容验证的 profile 才租用完整 Ray/通信端口集合；CRC32 % 50 不能作排他锁。缓存按工具版本/输入 hash 分区，可变缓存禁止当可复现输入。

## 7. Checkpoint 与复现

Run 输出路径唯一从 run_id 推导，TrainJob 保存 active_run_id/history，不再各建 trn 路径。输出清单包括 kind、logical_name、step、完整文件集合、hash、processor/norm、写入完成标记；分片权重不能只因发现一个 safetensors 文件就登记完成。

latest 是最大**完整** step；best 必须有固定 validation 指标与选择规则，二者独立。手动登记的路径必须属于该 Run 已认领输出，不能把白名单内任意文件挂在这个训练血缘下。自动登记开关由执行完成后消费者落实，重复回调不得生成重复模型。

resume 同时恢复模型、optimizer/scheduler、RNG、采样器/dataloader、epoch/step 和环境状态能力。缺某项时标为 warm_start/new_run，不能声称精确续训。OOM 后减少 batch/环境数是新 spec 派生，记录改变与有效 global batch，不静默重试成另一场实验。

模型登记的完整性通过不代表评估通过。训练成功、产物完整、评估有效、目标可部署四种结果独立显示；不存在评估输出时不得写入占位 reward/success_rate。
