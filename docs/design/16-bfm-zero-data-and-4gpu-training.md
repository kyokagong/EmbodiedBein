# 16 · BFM-Zero 数据补充、四卡训练与可视化评估

本设计补充 [14 · BFM 接入](14-bfm-zero-rpo-rlinf-integration.md)。2026-10-09按用户最新要求由[18 · 原始参考先验与G1→RPO](18-bfm-reference-prior-and-g1-rpo-retarget.md)纠正PD动态跟踪与参考数据资格混用；本阶段不训练教师。数据内容、来源、恢复和最终策略独立评估要求保留。用户要求的下一阶段预算为 **1000 次 BFM learner 联合更新**，使用四张 GPU，在实测资源范围内增加并行环境，训练后输出录像与独立评估结果。不是每环境仅运行 1000 个物理控制步。

## 1. 区分两类数据

BFM 的在线 replay 来自当前策略与仿真交互，保存真实 transition、实际动作、行为 latent、终止与截断。增大 env 数可以扩大这部分采样量。

离线 behavior prior 提供 observation-only 专家状态和连续下一状态，不要求人工标注动作。但轨迹必须与 RPO 的关节/body/坐标/控制频率兼容，并具备真实质量与来源资格。不能把随机策略的 replay 或未验证模型的 rollout 自动称为专家；反复切窗、重复文件或增加采样次数也不产生新的行为覆盖。

正式训练仍须消费冻结、`training_ready=true` 的行为数据。现有 smoke view 不原地修改。新数据发布新 DatasetVersion，归一化、专家采样和 command→latent 参考集都绑定同一份训练版本；不使用评估轨迹、测试 seed 或测试 reference 校准。

## 2. 数据来源顺序

| 路线 | 处理方式 | 条件与边界 |
|---|---|---|
| 本地 RPO 动作库 | 先盘点与筛查，再按真实时长重采样、Isaac FK转换和参考几何/时序检查 | 首选低成本候选来源；已有仓库许可证不自动证明每个原始动作的使用资格 |
| 人体动作重新映射到 RPO | 固定人体来源与retargeter，映射到原生23关节，再检查FK/时序/接触几何 | 不要求先有PD或教师能跟踪；须核对来源、人体模型依赖和RPO映射 |
| UFO G1 LAFAN重新映射 | 源模型和状态导出→hhtools R2R→RPO原生FK/状态参考；细节见设计18 | UFO不执行重定向，不能删列；实际数值映射尚须输入/校准与运行证据 |

外部路线的事实依据：

- [BFM-Zero 官方实现](https://github.com/LeCAR-Lab/BFM-Zero)训练使用 `lafan_29dof_10s-clipped.pkl`，完整参考为 `lafan_29dof.pkl`。切片与完整运动可以有共同祖先，不能仅凭文件名宣称独立train/test；本平台仍冻结独立分组。官方源码为CC BY-NC 4.0，数据使用条件另核对。
- [RoboParty GMR](https://github.com/Roboparty/GMR)当前支持表将 Origin 23 DoF 的 SMPL-X 列为支持、LAFAN1 BVH 列为 TBD。不得把“GMR 支持 G1 LAFAN”当成 RPO 已可直接使用。
- [RoboParty human-humanoid-tools](https://github.com/Roboparty/human-humanoid-tools)提供 BVH/SMPL 等导入、URDF 映射、机器人到机器人映射；这只是可调查的工具，RPO 校准和闭环质量仍需实际验证。
- [AMASS 的来源条款](https://amass.is.tue.mpg.de/license.html)限制用途；[LAFAN1 官方说明](https://github.com/ubisoft/ubisoft-laforge-animation-dataset)标注 CC BY-NC-ND 4.0。代码许可证与数据使用范围分别记录；研究或商用用途明确后再选择外部输入，不自动将公开下载等同于适用。

具体盘点、教师资格和资源事实见[本轮进度](../progress/2026-10-09-bfm-zero-next-training-preflight.md)，不把候选来源数量视为正式可用数据量。

## 3. 本地候选视图和验证

原始目录只读。选择清单固定文件 hash、上游 revision、pickle/joblib 格式、原始 FPS、时长和 source_group。可信固定字节的 pickle/joblib 只在离线受限任务容器处理；API 不反序列化。

转换不得沿用 smoke 的“每文件最多 500 帧”来报告完整数据量：保留实际时间轴，输出 50 Hz；关节映射复用冻结的 RPO Lab 序；root 四元数、link-origin 速度和完整 body 状态沿用 codec。转换行为变化派生 converter revision，旧 view 保持兼容。

静态筛查包含有限值、四元数、硬限位、原始时间轴、速度、连续窗口，以及当前 PD 目标范围。PD 目标范围筛查是保守条件，不能当作真实物理可达性证明。不得通过裁剪关节、补帧或简单时间缩放制造合格样本。有效 mask 不能让连续专家窗口跨越缺口；镜像、切段和同一原始动作的变体属于同一祖先组。

控制器动态验证必须真实推进物理，不能每步写入参考姿态来替代跟踪。固定PD的倒地/跟踪误差保留为诊断，不能单独否定参考先验；参考质量检查与实际BFM控制器资格分别记录，具体执行设计18。当前已发布数据保持原状态；新参考准入实现必须派生converter/validation recipe，并验证内容、来源和祖先分组后发布新版本。本阶段禁止教师训练及教师生成数据，不新增或沿用教师预算。

train/validation/test 按原始动作祖先组划分并冻结；重采样窗口及镜像不得跨组泄漏。报告完整帧数、有效连续帧数、行为覆盖和拒绝原因。只有站立或短片段时，不宣称已覆盖后退、侧移、转向或完整 BFM 能力。

扩大数据后复核当前 action gain。参考关节目标超出现有命令范围时报告失配；不能原地改变已发布接口。确需新 action gain 时派生 codec/本体接口与训练环境 revision，重新做对齐、reset、终态、导出和推理检查。

## 4. 四卡保持一个 learner

复用 RLinf 的 actor/env/rollout 分派和现有 Job/Run/ebd 生命周期。首选候选拓扑为 GPU0 learner、GPU1–3 env+rollout；另测 GPU0 同时采样、GPU0–3 均 env+rollout 的拓扑，依据实测安全容量与吞吐选用。

四卡不等于四个 BFM learner。FB pairwise batch 不做未经数值对照的跨 learner 切分；微批/全局 batch 按单 learner 语义校验，不能套用四卡 PPO 的 batch/world-size 关系。网络 profile、专家/replay batch、更新/采样比在容量探测前冻结，不因为环境数大而默默改变算法。

当前等分 env 的 RLinf 路径使用统一每 rank 环境数。共享 learner 的 GPU0 较紧时，选择实测总吞吐更好的三采样卡拓扑，不为了满足“每卡同数量”而浪费资源；报告每张卡的角色和 env 数。

各 rollout rank 需要独立 seed/RNG 与 episode/latent 上下文；跨 rank transition 必须保留对齐末态、实际动作和行为 z。训练 checkpoint 保存单 learner 的模型、优化器、target、replay、采样/RNG，及实际覆盖的 rollout 状态；没有覆盖的进程状态明确记录，不称 exact resume。拓扑或数据版本变更不能套用原 trainer-state 恢复凭据。

## 5. 容量探测和预算

先测模型、optimizer warmup 和完整更新峰值，再测 env+rollout，最后测真实组合与权重同步/接收/replay/保存峰值。单独 Isaac 能创建多少环境不是训练可承载数；原 Flat 每卡 2048 的结果不能当作 BFM 容量。

候选采样规模从 512、1024、2048 env/rank 递增；还可扩大时按已测内存趋势继续探测，不以反复 OOM 搜索上界。每卡至少留 2 GiB 且至少 10% 显存余量；同时核对 CPU/RAM/shm 与保存峰值。超出余量即停止扩大，不静默减少网络/batch。

每卡报告 env 数、角色、设备实际显存峰值、Torch allocated/reserved、RSS、环境 transitions/s 和 learner updates/s。最终采用通过完整组合探测的最大安全规模，并记录实际更新/采样比。replay 容量至少能覆盖一个完整联合 rollout，避免固定 rank 拼接顺序使部分 rank 整批被淘汰。大量采样降低 UTD 或拖慢 H2D 时，明确报告；“更多 env”不能直接解释成更高学习效率。

训练目标是新增 1000 次 BFM 联合更新，以 updater 计数和 optimizer 回读验收；分别记录 runner 轮数、联合更新次数、环境控制步和总 transitions。通过平台 preview 后提交有限时长的新 Run，定期 checkpoint。若模型/数据/拓扑变化，从新的初始化语义起跑；不冒称继承 smoke 的 exact trainer state。

## 6. 导出、录像与独立评估

导出新的 actor、normalization、command→latent reference 和 prompts，验证 hash、shape、Torch/ONNX parity 与命令范围；历史 smoke 前端不可与新 actor 混用。

新增 BFM 专属评估绑定和推理桥：真实 articulation 状态进入同一 codec/history，当前 command→z→确定性 actor；reset 时清空正确的历史与上下文；记录 c/z/动作/真实物理结果与视频。禁止用 Flat 的 780 维观测直接喂入 BFM 375+256 模型。

先运行站立/直行/侧向/转向的有限录像诊断，验证桥和录像可播放。随后执行 [14 §9](14-bfm-zero-rpo-rlinf-integration.md) 的独立 100 case Flat 比较，使用冻结的完整指令与随机化、每例 20 秒/1000 控制步、原机械限位和速度/yaw 判据；数据转换、参考集与正式 case 的祖先/seed 分离。

输出逐例 pass/失败原因、RMSE、跌倒/NaN/硬越限、完整案例分母、轨迹与录像链接，并报告模型资格；不能用 Job succeeded 代替至少 80/100 的判定。阶段 A 诊断与完整 Flat 比较分别标注，不能改场景后声称通过旧 suite。MuJoCo 和真机能力仍需另外的实际证据。

## 7. 反思边界

1. 在线量与专家质量分开：四卡采样不会自动修复离线 prior，重复窗口不算新行为。
2. 来源、参考与控制器分开：格式可读/FK对齐不等于策略可跟踪；PD失败也不等于BFM不能使用参考。公开仓库许可证不自动覆盖原始动作。
3. 规模与算法分开：一个 learner 的 pairwise batch、replay 以及 UTD 都需保持显式；四卡显存不相加。
4. 完成与能力分开：1000 次更新和可播放录像可以全部完成，模型仍可能评估失败；如实报告。

## 8. 当前数据来源目录与选择建议

本地逐文件名称、原始时长和静态候选结果见[30个RPO动作目录](../progress/2026-10-09-rpo-motion-catalog.md)。文件名可辨认的覆盖为：站立1、后退3、横移2、步行转向8、跑步与变速/启停10，另有6个数字编号片段。数字编号语义不因名称与其他数据集相似就自动确认；按源片段回放确认动作和转换谱系。

外部候选按本轮行走任务选择，不把数据源和 retarget 工具混为一谈：

| 数据源 | 内容/格式 | RPO 路径与使用边界 |
|---|---|---|
| [100STYLE](https://www.ianxmason.com/100style/) / [固定发布](https://zenodo.org/records/8127870) | 100种风格、超过400万帧；前进/后退/侧步行走和跑步、站立/动作转换；60 Hz BVH | 作者标注CC BY 4.0。hhtools提供Xsens/BVH导入，但RPO校准仍未实测。优先小批低幅度行走，剔除T-pose前后冗余；夸张风格不保证适合RPO。 |
| [CMU原始动捕](https://mocap.cs.cmu.edu/) | 多类人体运动，原始ASF/AMC及来源提供的其他格式 | 官网允许用于商售产品、禁止直接转售数据（含转换形式）。直接取得原始来源并自行转换，不把AMASS中的CMU派生版视为拥有相同使用条款；骨架转换与RPO校准需补齐。 |
| [AMASS](https://amass.is.tue.mpg.de/) | 多来源人体动作的统一参数数据；可选择行走/转身等 | RoboParty GMR的Origin 23DoF支持SMPL-X路径；需匹配实际SMPL-X数据/模型文件，不混用SMPL-H。AMASS与人体模型的来源条件分别核对，非商业用途限制见其官方条款。 |
| [LAFAN1](https://github.com/ubisoft/ubisoft-laforge-animation-dataset) | 人体BVH动作库 | 官方标注CC BY-NC-ND 4.0；明确变换/使用/分发条件。RoboParty GMR当前RPO BVH路径仍为TBD，hhtools提供另一候选适配路径，均不代表已在平台验证。 |
| [BFM-Zero官方运动文件](https://github.com/LeCAR-Lab/BFM-Zero) / [UFO下载入口](https://github.com/Roboparty/UFO/blob/main/scripts/download_data.sh) | 训练切片lafan_29dof_10s-clipped.pkl、完整参考lafan_29dof.pkl | 29DoF输入不直接适配RPO23；需要R2R或从人体源重新映射，追溯来源条件及原始clip。完整参考文件不是天然独立测试集。 |
| [MotionDecode来源条款](https://huggingface.co/datasets/CMRobot/MotionDecode/blob/main/LICENSE.md) | 大型动捕计划和可申请/公开样本；hhtools描述的样本为G1 CSV | 来源条款列出研究/个人学习/非商业原型用途，商用需联系。G1样本按真实120Hz做机器人→机器人映射；不能把宣传1000小时当作已下载可用规模，也不能直接喂给RPO。 |

[hhtools官方说明](https://github.com/Roboparty/human-humanoid-tools)提供100STYLE、AMASS、LAFAN1导入与MotionDecode R2R路径，是候选转换工具而非数据授权或动力学质量证明。物体交互类OMOMO/OmniContact会增加物体状态和接触场景，本轮行走优先级低于上述来源。

工程建议为先验证本地连续候选，然后以100STYLE低幅度行走/后退/侧步的小样本做RPO校准和真实物理回放；希望扩大通用动作覆盖时再考虑CMU或AMASS。该优先级基于内容、公开条款与工具入口，不是已经完成的RPO接入结果。没有确定使用范围前，不自动下载带用途限制的数据或获取需要登记的人体模型。
