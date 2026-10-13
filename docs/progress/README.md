# EmbodiedBein 进度

设计真源仍是 [`docs/design/`](../design/README.md)。本目录只记录**实现对照设计的进度**，不改设计。

| 日期 | 文档 | 说明 |
|---|---|---|
| 2026-10-14 | [RPO BFM第八轮接触几何与时序验证](2026-10-14-rpo-bfm-learning-round8.md) | 复用C2六case，同接触CPU单步差缩小>99.95%；A1地面margin 3mm首跑通过但确认env453穿透5.750mm失败，揭示提前终止删失。2/2Job、23.409分钟Run/8.452GiB，完整索引/图表/平台链接齐全，停止晋级、未训练 |
| 2026-10-13 | [RPO 参考动作目录与三维回放](2026-10-13-rpo-motion-browser.md) | 官方30片段与当前UFO70片段离线CPU预览、全部API回读；三维/时间轴/关节曲线/来源与有效帧，保留原数据与训练资格 |
| 2026-10-13 | [RPO BFM第七轮物理预算关闭](2026-10-13-rpo-bfm-learning-round7.md) | S2/C2独立完整核验，物理均未全过；三轮实用8/9Job、83.84分钟/28.00GiB，确认名额未用，未训练；补齐不同速度峰/穿透case、图表、完整索引与平台链接（跨至10-14） |
| 2026-10-13 | [RPO BFM第六轮固定动作物理验证](2026-10-13-rpo-bfm-learning-round6.md) | 3Job已关闭，I1/S1/D1均未全过；加大Kd改善存活却恶化峰和最大穿透；6phase/同Run尾部/图表/完整Run预算及平台链接齐全，未启动训练 |
| 2026-10-13 | [RPO BFM第五轮动作回放与物理验证](2026-10-13-rpo-bfm-learning-round5.md) | 3Job/23.88分钟/6.93GiB；同Run全部事件前史保留，固定动作R0峰差≤0.23%但波形仍不同；纠正legacy诊断/实际v13采样语义混淆，I1进入第六轮；平台链接/图表/预算见报告 |
| 2026-10-13 | [RPO BFM第四轮积分步长与制动时序](2026-10-13-rpo-bfm-learning-round4.md) | 3Job/14.95分钟/252.30MiB含日志；I0连续制动证据已保留但policy42极值跨重复偏移19.7%，anchor峰2.82/穿透最大5.92mm未过门槛；停止I1与采样训练，1.25ms仅CPU验证，事故前史缺口/平台链接/图表/下一轮回放方案见报告 |
| 2026-10-13 | [RPO BFM第三轮接触单因素验证](2026-10-13-rpo-bfm-learning-round3.md) | 3Job/14.2分钟；B0六组逐行复现，C1地面time constant单因素令anchor峰值2.82→2.09但穿透最大5.47→15.97mm，门槛失败；停止候选、采样训练未启动，完整平台链接/时序/穿透图/预算见报告 |
| 2026-10-13 | [RPO BFM第二轮动力学与RSI诊断](2026-10-13-rpo-bfm-learning-round2.md) | 有限3Job；积分力/M/J/接触快照与52/358状态对账，参考有限差分与原生瞬时速度口径确认；采样修复镜像准备，物理门槛未通过，训练未启动；完整平台链接见报告 |
| 2026-10-13 | [RPO BFM学习优化执行](2026-10-13-rpo-bfm-learning-optimization.md) | 冻结v12/v13与诊断预算；全量replay/祖先及RSI审计；训练mode覆盖缺陷复现修复；受管配对物理诊断，质量出口以执行记录为准 |
| 2026-10-13 | [MJLab v13动态RSI与位置容差优化实验](/data/kk/EmbodiedBein/docs/progress/2026-10-13-ufo-mjlab-rsi-viability.md) | 初态20ms前瞻/1mrad余量，训练位置容差0.02rad、严格影子/物理限位不变。每采样卡2048、Batch2048、Replay4M；100→600新增500更新成功，六Adam/哈希/983040累计转移闭环。五窗训练4.43→12.71、严格3.52→7.30步；首步训练0.00950%但严格1.49%，右踝峰值42.58rad/s（5.32×）仍不达标。冻结模型、关闭续训候选、无独立评估/提交；完整自包含配置、149指标、分组/事故/资源与恢复报告 |
| 2026-10-12 | [原生MJCF/MJLab与四卡500更新预训练总结](/data/kk/EmbodiedBein/docs/progress/2026-10-12-ufo-mjcf-pretraining.md) | 新MJCF/MuJoCo-Warp后端，RPO质量/惯量/网格、冻结接口位置范围与90%软罚对齐；每采样卡2048、Batch2048、Replay4M、435M网络。16→24→524六Adam/哈希恢复通过，149有限指标、786432新增转移逐行核验；末窗训练8.49/严格7.47步、右踝35.78rad/s、首步3.51%未达标，位置早停为主。停止500段、不独立评估、不提交；报告自包含配置、全指标、RSI与速度一致性诊断 |
| 2026-10-12 | [接触顺应v11与四卡500更新完整实验总结](/data/kk/EmbodiedBein/docs/progress/2026-10-12-ufo-contact-compliance-500-observation.md) | 每采样卡2048、batch2048、4M replay、完整435M；实际柔顺地面60000/600、支撑腿屈膝RSI、增强mean8速度罚。16→24→524完整恢复/SHA/六Adam通过，154有限指标与786432转移复核；训练末窗22.64步但严格13.07低于旧16.01、右踝pitch50.31rad/s。35/2.5长段门禁失败，API候选/preview实际拒绝3000；不独立评估、不提交，自包含配置/指标/事故/分组/资源证据 |
| 2026-10-12 | [环境级tracking、子步反馈与四卡500更新完整总结](/data/kk/EmbodiedBein/docs/progress/2026-10-12-ufo-tracking-contact-500-observation.md) | 每采样卡2048、batch2048、完整435M；RSI足点法向速度过滤、mean8速度罚、1.3×探索/原阈值影子、配对选Kd2.5、4M流式replay。16→24→524六Adam/哈希通过，151指标与786432转移逐行核对；五窗训练8.25→22.31步、严格7.37→16.01步，首步0.088%、全局左踝子步55.76rad/s仍未达标。停止本段、不独立评估、不提交；自包含完整参数/分组/删失/预算与失败记录 |
| 2026-10-12 | [完整地面v9、RSI接触/PD与6144环境500更新总结](/data/kk/EmbodiedBein/docs/progress/2026-10-12-ufo-ground-v9-500-observation.md) | 新512m地面、实际三卡2048起点包络与20m余量检查、131列trace；16→24→524完整六Adam/哈希回读，138有限指标与五窗episode7.13→13.27步。786432正式转移越界0、首步366/73200=0.50%；右踝pitch首步45.99rad/s、全局子步67.52仍未解决，停止本段，不独立评估、不提交代码；报告自包含H受影响对照 |
| 2026-10-12 | [RSI接触/PD v8、6144环境与500更新完整报告](/data/kk/EmbodiedBein/docs/progress/2026-10-12-ufo-contact-v8-500-observation.md) | train-only碰撞网格RSI池、400Hz显式PD积分、完整子步终止锁存与首帧旧力修复；136指标及六Adam/哈希回读。第三轮确认100m地面无法覆盖2048网格约20%中心，保留为受影响对照并派生512m ground-v9；本段未独立评估，未通过行走验收 |
| 2026-10-12 | [可达控制v7、6144环境与500更新观察报告](/data/kk/EmbodiedBein/docs/progress/2026-10-12-ufo-reachable-v7-500-observation.md) | v2目标映射、六Adam梯度裁剪、Actor尺度/D奖励保护与速度软罚；435M、batch2048、每采样卡2048环境，16→24→524完整状态与哈希核验。128指标五窗episode均值8.52→14.15步，速度越限275→185次/轮；仍有右踝68.68rad/s极值与1.45%首步终止，结束观察段，不启动独立评估或追加长训，正常行走未验证 |
| 2026-10-11 | [每采样卡2048环境、batch2048、5000更新训练总结](/data/kk/EmbodiedBein/docs/progress/2026-10-11-ufo-6144-env-5000-experiment.md) | F原生435M、九项aux、6144环境、batch2048完成5000次新增更新(24→5024)、769万新增转移与完整checkpoint；112指标/五窗与控制记录显示episode_len后程退化、真实速度越限增多；按用户要求不做本段独立评估，先总结分析 |
| 2026-10-11 | [完整优化摘要](/data/kk/EmbodiedBein/docs/progress/2026-10-11-ufo-optimization-summary.md) / [原生对齐与全部逐点证据](/data/kk/EmbodiedBein/docs/progress/2026-10-11-ufo-native-alignment.md) / [历史交接](/data/kk/EmbodiedBein/docs/progress/2026-10-11-ufo-walking-handoff.md) / [分段流水](/data/kk/EmbodiedBein/docs/progress/2026-10-11-ufo-walking-repair.md) | safe5024三轮无改善停止；C/D原生435M网络、batch1024、replay100万和稳定latent均完成资格，D限速四卡恢复至1024更新；独立0/16、中位35步，held-out17/87步均hard_limit；E速度软惩罚已容器恢复/ONNX/平台验证并部署dependency_ready，未GPU资格；下一段含checkpoint与既定余量超80GiB，budget_end；未达到正常行走标准，新100例未运行 |
| 2026-10-10 | [UFO/RLinf四卡训练](2026-10-10-ufo-four-gpu-training.md) | B四卡新增1000更新/累计1024完成；latest验证0/16，冻结best24最终验收0/100/fail，100录像可访问，收集故障修复并保留原失败记录；磁盘临时层曾超预算已清理，C未实施 |
| 2026-10-10 | [UFO/RLinf实现与容器验证](2026-10-10-ufo-rlinf-implementation.md) | B分支原生agent/RSI/worker/完整恢复/ONNX/API页面及CPU验证；adapter_ready，GPU资格/长训未执行，C与新四卡依赖B资格 |
| 2026-10-10 | [UFO框架与RLinf设计分析](2026-10-10-ufo-framework-analysis.md) | 固定官方源码，核实完整FB agent/RSI/history/依赖差异；设计20/21输出；未实现新桥、未运行训练 |
| 2026-10-10 | [BFM行走闭环优化](2026-10-10-bfm-staged-walking-training.md) | latent闭环修复；实际16→24完整恢复及四卡16更新通过；两次16例快评均0成功、32录像解码；上游配方差异待修正、长训预算待确认 |
| 2026-10-10 | [BFM训练39项指标与问题分析](2026-10-10-bfm-training-metric-analysis.md) | API逐点一致/实际v6代码核对；全部指标释义、前后50点趋势、裁剪比例、低采样使用率及动作饱和；仅分析未追加训练 |
| 2026-10-10 | [BFM四卡1000更新与可视化评估](2026-10-09-bfm-four-gpu-1000-evaluation.md) | 四卡RLinf1000更新/9830万transition完成；四卡×4env标准100例与全部HD录像核验完成；0/100、66fall/36硬限位，模型验收失败 |
| 2026-10-09 | [G1标定与BFM codec](2026-10-09-g1-calibration-bfm-codec.md) | 14原生关键点标定实际进入CPU solver，100有效帧/92窗口；实际Isaac FK/52+358 codec、loader/RNG及RLinf两次更新/完整checkpoint通过 |
| 2026-10-09 | [官方参考接入与G1→RPO试验](2026-10-09-official-reference-and-g1-rpo.md) | 官方v5候选6135有效帧/5206窗口、真实RLinf两次更新通过；G1→RPO原生100帧/87有效/79窗口，诊断候选非正式视图，无教师 |
| 2026-10-09 | [BFM 原始参考重审与UFO重映射](2026-10-09-bfm-reference-reassessment.md) | 禁止教师；30片段训练URDF静态候选9276帧/连续9205帧，BM速度需重建，UFO需hhtools桥接；尚未晋升数据或执行G1映射 |
| 2026-10-09 | [单卡教师录像诊断结果](2026-10-09-rpo-tracking-teacher-evaluation.md) | RLinf416轮/511万transition；独立验证4/12case完成，1/4通过，预算59分24秒；正式四卡条件未满足 |
| 2026-10-09 | [RLinf 闭环跟踪教师](2026-10-09-rpo-tracking-teacher.md) | 历史独立547/905参考残差接口与PPO路径；用户最新指令禁止继续教师，当前转为原始参考重审 |
| 2026-10-09 | [BFM 动态回放与100STYLE试验](2026-10-09-bfm-data-replay-and-100style-trial.md) | 本地30动作真实回放、源wxyz根因修正、新数据版本、外部小样本获取/映射证据；正式训练资格未通过 |
| 2026-10-09 | [BFM 下一阶段摸底](2026-10-09-bfm-zero-next-training-preflight.md) · [动作目录](2026-10-09-rpo-motion-catalog.md) | 30 个本地动作名称/时长/静态筛查、四卡资源与正式训练前缺口 |
| 2026-10-09 | [BFM-Zero 指标展示](2026-10-09-bfm-zero-metric-display.md) | 单轮恢复 smoke 的真实指标、显示顺序和关键数值修复 |
| 2026-10-08 | [BFM-Zero / RPO / RLinf](2026-10-08-bfm-zero-rpo-rlinf.md) | 完整网络与离策略更新接入、训练/续训/导出 smoke；正式质量验收未做 |
| 2026-10-01 | [RPO-Flat 四卡扩容](2026-10-01-rpo-flat-four-gpu.md) | step-3000 权重 warm start，4×2048 环境，新增 2000 轮配置与执行证据 |
| 2026-09-30 | [官方 RPO-Flat / RLinf](2026-09-30-rpo-flat-rlinf.md) | 完整指令范围接入、多轮核查、GPU 冒烟与正式验收证据 |
| 2026-09-24 | [RPO V2 AMP / RLinf 接入](2026-09-24-rpo-amp-rlinf.md) | 代码接入、CPU 检查与待 GPU/平台验证的阻断 |
| 2026-09-21 | [RPO V2 WBC P0–P4](2026-09-21-rpo-wbc-p4.md) | RPO 受管资产、RLinf WBC、导出与双仿真评估实现状态 |
| 2026-09-12 | [LIBERO 独立评估](2026-09-12-libero-evaluation.md) | 独立评估实现与剩余真实运行验收 |
| 2026-09-10 | [可配置存储](2026-09-10-configurable-storage.md) · [数据集本体关联](2026-09-10-dataset-embodiment-link.md) | 存储迁移与数据筛选增量 |
| 2026-09-09 | [平台与 LIBERO](2026-09-09-platform-implementation.md) · [运行时验证](2026-09-09-libero-runtime-validation.md) | 通用执行基础与 SFT/PPO 实测 |
| 2026-09-09 | [数据](data-implementation-report.md) · [流水线](pipeline-implementation-report.md) · [运行时](runtime-implementation-report.md) · [工作台](workbench-implementation-report.md) | 分模块实现记录 |
| 2026-09-09 | [LIBERO 操作说明](libero-pi05-runbook.md) | 页面操作和边界 |
| 2026-08-21 | [当前状态](2026-08-21-status.md) · [验收证据](2026-08-21-make-features-real.md) · [第一版](2026-08-21-first-version.md) | 历史基线；不代表当前代码 |

判定约定：

- **完成**：对照该里程碑验收项，主路径已可运行（允许大镜像/真机仍用占位）。
- **可用骨架**：领域逻辑或 API 已有最小实现，端到端或缺关键环节，或 UI 仅为 JSON 回显。
- **占位**：有路由/模块/空镜像，但不能完成设计描述的工作。
- **未做**：设计要求的能力尚不存在。
