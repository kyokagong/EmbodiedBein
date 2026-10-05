# 05 · 评估与交付设计

## 1. 独立评估

评估是独立生产管线，不是训练 reward 的展示。输入冻结模型内容、processor/norm、本体接口、Task/Scenario、EvalSuite、case/seed、runtime、硬件和指标定义；输出逐 case 结果、失败媒体、汇总、门禁和完整性证据。

case 状态分 `success / task_failure / timeout / invalid / infra_failed / cancelled`。模型按时未完成任务是 task failure；仿真起不来是 infra failure。报告同时给出 planned、completed、valid、invalid 和 infra_failed；必需 case 未完成时 qualification 为 `incomplete`，不能用剩余成功样本排名。

Job `succeeded` 只表示执行和报告完整；模型的 `qualification=pass|fail|incomplete` 独立。表现差不触发基础设施无限重试。

## 2. EvalSuite 与指标

Suite revision 固定 Task/接口/Scenario hash、case IDs、seed、重复次数、超时、成功/失败函数、指标单位、判定阈值、置信要求和媒体策略。模型参评前冻结，不能看结果后换 seed 或降门限。

任务指标分榜：

| 任务 | 主指标 | 关键诊断 |
|---|---|---|
| SO101 抓放/到达 | 固定判据成功率、完成时间 | 物体位姿、保持时间、碰撞、夹爪、越限 |
| G1 locomotion | 完成/摔倒、速度跟踪误差 | 地形、足滑、接触、功率、姿态恢复 |
| G1 操作 | 任务成功且基座稳定 | 末端误差、支撑、控制器切换与延迟 |
| BHL biped/WBC | 跟踪、稳定、终止原因 | 躯干/关节误差、越限、能耗、接触 |
| BHL fixedhand | 分手别/目标的成功率 | 距离、关节余量、语言响应 |
| VLA 离线 | 有效 mask 内 action/pose 误差 | chunk 位置、语言/视觉消融 |
| 目标运行 | sensor-to-action p50/p95/p99、deadline miss | observation/action age、内存、功耗、温度 |

成功函数固定代码 hash。成功率显示分母和 Wilson 区间；相关样本按 group 聚合或 cluster bootstrap。模型比较使用相同 case/seed，条件不一致时禁止混榜。validation 用于选择，最终 acceptance 只在候选冻结后运行，且不得回流训练。

## 3. sim2sim 与 sim2real

检查顺序固定为：

1. 资产、关节序、零位、单位、限制、控制增益、相机和 frame 一致；
2. resize/color/tokenizer/norm、state 构造、action 解码/clip/chunk 一致；
3. 用准备交付的实际 ONNX/engine/包评估，不只评 PyTorch checkpoint；
4. 扰动延迟、抖动、丢帧、噪声、标定、摩擦、质量、光照和相机；
5. 在目标设备测完整处理链；
6. 仅对具备真实实例的机器人进行现场端验证。

没有第二仿真器资产时显示“未覆盖”，不能用 Isaac 扰动测试冒充 sim2sim。领域随机化也不是 sim2real 已解决的证据。

## 4. LIBERO 首个评估实例

LIBERO 是独立评估协议的首个具体 recipe，不改变通用对象：

- `spatial / object / goal / 10` 四套件分别冻结，跨套件总分只能在四份完整、条件一致的标准报告上计算。
- 测试场景资产作为 `usage=eval` 的 DatasetVersion 在数据模块下载、固定 revision、校验 hash；示教数据不是仿真成功率真值。
- 评估从受管训练 Run 和明确 step 选择 checkpoint。权重、源推理配置、镜像 digest、suite、case 列表和 benchmark hash 形成 eval plan hash；预览过期后禁止提交。
- `smoke` 只检查协议和证据链；`standard` 才能进入对应套件榜单。部分报告保留诊断价值，但 qualification 为 incomplete。
- 逐 case JSONL 必须与计划对账：重复、缺失、未知 case 或非法数值均使报告无效。视频缺失显示 unavailable，不补造媒体。
- 评估直接创建 `eval.sim` Job，不伪装为 TrainJob，也不出现在训练列表；checkpoint 与 suite 到 Job 的血缘必须保留。

未来增加 LeRobot 原生 runner 时，必须用相同冻结输入和 case 对两条 runner 做等价性验证；新增入口不能自动继承 RLinf 路线的通过状态。

## 5. Jetson AGX Orin 64GB

优化是 `export → optimize → target_benchmark → eval → bundle` 的派生管线，复用 Job/Run/Artifact。TargetProfile 必须记录实机 SKU、JetPack/L4T、CUDA/TensorRT、aarch64 依赖、插件、功耗/时钟/散热、存储和后台预算。4090 上构建的 engine 不能直接当作 Orin 产物。

先建立能正确运行的 FP32/FP16 基线，再按 recipe 尝试 INT8、混合精度、QAT 或蒸馏。原模型已满足预算就停止；算子不支持、质量退化或 no-fit 时保留失败证据，不承诺所有 VLA 都能在 64GB 上实时。

内存预算包含权重、activation/workspace、cache、图像/解码 buffer、runtime、OS/控制器和安全余量。Orin 是统一内存，CPU/GPU 预算不能各算一遍 64GB。

量化 calibration 从 train 或专用 Slice 派生，禁止使用最终 acceptance。每个 variant 记录模型、processor、norm、calibration、算法、模块精度、engine、输入 profile 与 hash；数值误差按反归一化物理单位和离散夹爪决策检查，并重新运行闭环套件。

目标报告区分 GPU inference 与 `capture → decode → preprocess → transfer → inference → postprocess → command-ready` 全链路时延，报告 p50/p95/p99/max、deadline miss、峰值/稳态内存、功耗、温度、降频和任务质量。

首期 Orin 不加入通用远程调度。操作者在 Orin 运行版本固定、无电机权限的离线容器，平台导出输入/manifest 并导入带设备、runtime、engine、input hash 和原始 timing 的 external report。没有目标证据时状态保持 `awaiting_target_evidence` 或 `closed_loop_target_unverified`。

## 6. PolicyBundle

Bundle 是平台与端侧之间的唯一交付物，可包含多个权重/engine，而非固定单个 `policy.onnx`。manifest 至少包含：

- 能不能装：本体接口、目标 profile、I/O、processor/norm、runtime/plugin、固件与标定要求；
- 从哪来：模型、Run、数据、代码快照、镜像、优化 variant 和许可证；
- 为什么可信：导出、评估、目标 benchmark 和 Gate 结果；
- 如何安全：适用的动作限制、watchdog 与端侧安全配置；
- 完整性：所有文件 hash 和签名。

安全字段由端侧 runtime 执行，平台只负责交付和核验证据。缺依赖、错接口、错标定或签名失败必须拒绝加载。优化失败不覆盖原模型。

## 7. 真机红线

平台永远不进入实时控制回路。它只交付包、接收异步遥测、记录现场证据。加载、标定、急停检查、低风险试验、切包和回滚由现场端侧工具与人员完成。

- 真机放行、门禁豁免和远程运动接口结构性禁止。
- 异常时先由端侧进入经验证的安全状态；运动中不得热切策略或清空 recurrent/chunk 状态。
- active/previous 包在端侧本地保留并校验，网络或平台故障不影响安全控制。
- G1 当前真实阶段固定为 `hardware_unavailable`，仿真/Orin 离线通过不能升级真机状态。
- SO101/BHL 的具体电机限值、E-stop 和回滚动作来自设备接口与现场验证，不写一个通用参数冒充安全方案。

遥测异步回流，记录设备/包/标定/软件、控制延迟、age、关节摘要、温度、电流和事件。上传支持 seq、chunk hash、断点与去重；失败样本经质量和标注审核后才进入下一数据版本。
