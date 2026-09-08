# 06 · Jetson AGX Orin 64GB：量化、实测与交付

## 1. 目标与非目标

用户的目标设备是 **Jetson AGX Orin 64GB**。量化的目标是满足具体机器人的内存、sensor-to-action 时延、任务质量和功耗/热稳定要求，不只是让文件变小。

首期将优化作为现有模型管线中的 `export → optimize → target_benchmark → eval → bundle` 阶段，复用 Job/Run/Artifact。无需另建模型压缩平台，也不默认所有 RLinf 模型都能导出成一个 TensorRT engine。

G1 当前无真机，仍可在 Orin 上做离线推理与仿真输入回放；这些结果不等于真实 G1 部署通过。

## 2. TargetProfile 必须来自实机

记录 Orin 型号/SKU、可用统一内存、aarch64、JetPack/L4T、CUDA、TensorRT、cuDNN、PyTorch/ORT（若使用）、驱动/容器依赖、插件 hash、存储、功耗模式、时钟/散热策略、温度、摄像头解码负载和其他进程预算。

软件版本不能从“Orin 64G”猜出，也不建议本轮为了量化升级设备。NVIDIA 的 [TensorRT 支持矩阵](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html)区分版本、平台和精度支持；未来实现需选择设备实际版本对应条目，不把网页 latest 当已安装环境。

4×4090 主机是 x86_64，Orin 是 aarch64；依赖、编译插件和 runtime 镜像分别构建。NVIDIA 文档明确 engine 默认绑定构建环境/设备，JetPack 的硬件兼容模式也有限制，因此 **不直接复制 4090 上的 TensorRT engine 到 Orin**。[Engine Compatibility](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/engine-compatibility.html)

## 3. 按模型选择最小可行路径

| 模型类型 | 先建立的基线 | 后续优化候选 | 停止条件 |
|---|---|---|---|
| BHL/G1 小 MLP、CNN 控制策略 | 原 FP32 导出/CPU 或 GPU runtime | FP16；必要时 INT8 | 原模型已满足延迟/内存时不强制量化 |
| π0/π0.5、LingBot 等 VLA | 能在 Orin 正确运行的原生或 FP16 路径 | 视觉编码器/语言主干/动作头分模块混合精度；PTQ→必要时 QAT/蒸馏 | 算子不支持、模型行为退化或 deadline 不满足时不交付 |
| 多阶段/大规模视觉动作或世界模型 | 先验证整体可装载性和资源 | 模块裁剪/专家冻结后专门优化，或学生模型 | 超出预算时标 target_no_fit，不承诺“64GB一定装得下” |

FP16 是降低精度基线，不等同 INT8 校准量化。INT8、INT4 的可用性分别取决于算子、工具链和模型，不能由“模型已压成4bit”推断 Orin 会加速。FP8/NVFP4 不列为 Orin 默认方案；NVIDIA 当前 [TensorRT Edge-LLM 支持矩阵](https://nvidia.github.io/TensorRT-Edge-LLM/user_guide/getting_started/support-matrix.html)对 Orin 和精度有专门限制。Edge-LLM 支持部分模型也不能证明这里所有 VLA 都有相同导出路径。

模型越小不必然越快，反量化、数据搬运、动态 shape 与不支持算子 fallback 都可能增加延迟。只对实现了相关 recipe 的模型显示可运行“优化”动作。

## 4. 内存与时延预算

单以 7B 参数示例，FP16 权重约 14GB，INT8 约 7GB，INT4 约 3.5GB，均为十进制粗估，未计 scale/zero-point、metadata、其他精度模块。实际预算：

`peak_memory = weights + activations/workspace + caches + image/decode_buffers + runtime + OS + controller + safety_margin`。

Orin 的 CPU/GPU 使用统一内存，不能给模型 64GB 又给 CPU 64GB。记录峰值 RSS/设备统计的口径，避免把共享内存重复加总。加载峰值与稳态峰值分开，量化构建内存与推理内存分开。

支持多类机器人不意味着一块 Orin 必须同时驱动全部机器人。部署绑定 target_instance、model variant 与 robot_instance；首期每目标只启用经测量的单套策略组合。若同设备并发服务多实例或高/低层模型，必须对完整组合测试峰值内存、抢占、调度和 deadline，不能相加单模型平均 FPS 作为验收。

时延测 `capture → decode → preprocess → transfer → inference → postprocess → command-ready`，同时测 sensor age、动作 age、排队、chunk overlap 和 deadline miss。GPU event 只计 inference，不能作全链路时延。

BHL fixedhand 现有接口为20Hz且 max_policy_latency_ms=15；这两个数都必须保留在校验里，不能因为一轮 VLA 推理耗时大就静默将门限改成50ms。若需要异步 action chunk/不同推理周期，必须新增明确的接口/控制策略版本，经闭环和端侧验证后采用；chunk 输出10步不代表每一步都用了新观测。

## 5. 量化管线契约

1. **Freeze**：固定模型、processor、normalization、interface、baseline eval、目标 profile。
2. **Export probe**：小规模实际观测走原模型与导出模型，检查 shape/dtype、数值、动作语义；不支持的算子明确列出。
3. **Calibration**：从指定 calibration Slice 取代表性图像、状态、指令、chunk 时刻与边界情况；样本选择/hash/预处理、量化算法/参数/模块精度记录到 manifest。不能拿 acceptance 拟合 scale。
4. **Build**：生成 Q/DQ 或后端认可的量化表示；目标 Orin 环境构建 engine/插件，记录所有构建参数、动态 shape profiles 和 engine hash。
5. **Numerical validation**：逐模块/逐动作维度比较反归一化后的实际单位误差、夹爪离散决策、边界越限；量化不能只检查平均 L2。
6. **Closed-loop validation**：相同 eval cases 上比较原版/量化版；通过扰动与长序列测试，避免单步小误差在闭环积累。若 engine 只能在 Orin 运行，必须让仿真评估调用那个目标产物，具体方式见第7节。
7. **Target benchmark**：运行下一节实验；同时报告质量变化和性能变化。
8. **Register**：生成新的 deployment variant，保留 parent_model、calibration、recipe、engine、完整报告；不覆盖原模型。

原设计 FP32 导出对照的 ≤1e-4 检查继续用于适用导出路径。量化需要独立版本化误差预算与任务非劣性判据，不通过修改原阈值把量化失败“放过去”。物理单位误差预算由接口/任务制定，存在缺项则该优化 variant 不可交付。

## 6. Orin 实测协议

每个实验固定 batch=1（除非明确多机器人服务场景）、相机数/分辨率、指令长度、历史长度、chunk 长度、采样/denoise steps、warm-up、power mode、时钟、散热和后台负载。比较配置需相同输入及任务条件。

建议工程冒烟先预热100次、计时1000次；正式报告另做持续负载至少30分钟并报告有效样本数/分桶变化。这些是起始实验设计，不是实时性保证，p99 需要足够样本和长尾观察。实验无法达到规定样本时明确 incomplete。

| 报告字段 | 用途 |
|---|---|
| load/startup time，build time/peak | 启动与构建是否可行 |
| inference 与全链路 p50/p95/p99/max | 区分 GPU 推理和用户真正关心的时延 |
| deadline miss 次数/比例、stale observation/chunk | 判断控制有效性 |
| 峰值内存、steady memory、OOM | 64GB 预算验证 |
| power mode、功率来源、温度、降频计数 | 持续运行是否变慢；传感器估计与外部电源测量区分 |
| 原版/优化版成功率与区间、动作误差、越限事件 | 判断量化是否损害任务 |
| 完整 software/hardware/input/engine hashes | 复现与失效条件 |

选择质量、延迟、内存、功耗均可接受的 variant；如果 FP16 满足需求且 INT8 无收益，就保留 FP16。若不满足，可按 recipe 降相机分辨率、减少采样步、调整可验证的异步推理或蒸馏；每次改变都产生新 variant 并重评，不能偷改已发布配置。

## 7. 在单机平台边界内执行目标测试

平台训练/转换仍由本机 ebd 管理。**首期 Orin 不加入通用远程调度，不让 robot-agent 持有 docker.sock。** 操作者在 Orin 用预置、版本固定的 Compose 测试容器执行离线 benchmark/build，平台生成测试输入与 manifest，接收并校验结果包。这属于明确标注的 external_run，UI 状态 `awaiting_target_evidence`；没有目标回执就不能生成已验证包。

Orin 容器只读挂测试输入，输出写本地隔离目录，无真实电机设备权限；真实端侧控制器独立部署。平台可导出作业说明并导入结果，而不通过 SSH/Agent shell 远程执行。将来若确需自动设备作业，另行设计受限端点与唯一执行权限，不把本轮通用单机编排扩大为集群。

**实际目标产物的闭环评估**：操作者在 Orin 预先启动受限推理测试容器，本机仿真 eval Job 向已登记、鉴权的测试端点发送 observation 并接收 action，闭环对象只有仿真环境。端点加载前自验 engine/processor/hash，双方按 episode/step/request_id 记录结果；无需远程 Docker 权限，也不能访问电机。网络时延单独记账，既不冒充板上 sensor-to-action 时延，也不从结果中悄悄删去。如果没有这条测试链或目标端等效仿真器，则只做离线回放并标 `closed_loop_target_unverified`，不判定量化包可交付。

目标回执至少绑定 external_run_id、一次性 challenge、设备身份、运行时/engine/输入 hash、原始 timing 样本和日志；签名或已鉴权的人工导入记录用于防止错配与重复提交，不声称能证明硬件/计时完全可信。缺原始证据或环境不匹配保持 incomplete；不能只上传一个 `passed=true` JSON 就提升 capability。

## 8. Bundle 与降级

Bundle 包含 manifest、模型/engine hash 清单、processor/tokenizer/norm、动作映射、interface、目标 profile、插件/runtime 依赖、输入样例、离线自检、评估报告、来源/许可、完整性校验与签名，以及适用的端侧安全配置。VLA 多模块包允许多个文件；不能把 `policy.onnx` 当所有模型的唯一交付形态。

包签名验证、目标版本匹配、缺依赖/错接口/错标定拒绝加载。量化变体的 failure 不影响原模型登记；Orin 上 no_fit 或未验证时保持候选态。active/previous 的切换与现场安全流程见 [05](05-evaluation-and-sim2real.md)，平台不执行真机放行。
