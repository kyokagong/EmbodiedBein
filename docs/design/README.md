# EmbodiedBein 设计真源

本目录是 EmbodiedBein 当前唯一的设计真源。实现与设计冲突时应修正实现；实现进度与运行证据记录在 [`../progress/`](../progress/README.md)，不得据此降低设计约束。

## 阅读顺序

| 文档 | 内容 |
|---|---|
| [01 · 产品与领域](01-product-domain.md) | 平台边界、核心对象、本体接口与门禁 |
| [02 · 系统架构](02-system-architecture.md) | 进程、Python/Rust 分工、执行、存储与故障边界 |
| [03 · 数据与采集](03-data-collection.md) | 来源、episode、版本、质量、采集与可视化 |
| [04 · 模型与训练](04-model-training.md) | 模型能力、Job/Run、调度、checkpoint 与恢复 |
| [05 · 评估与交付](05-evaluation-delivery.md) | 独立评估、sim2real、Orin 优化、策略包与真机边界 |
| [06 · Agent 与安全](06-agent-security.md) | ToolBus、权限、审计与结构性禁止项 |
| [07 · 工作台与 API](07-workbench-api.md) | 信息架构、关键页面、接口与交互约定 |
| [08 · RoboTwin 接入](08-robotwin-integration.md) | 双来源数据生成/导入、六模型 RLinf 训练、独立评测、实施验收与多轮反思 |
| [09 · RoboTwin 评估](09-robotwin-evaluation.md) | RLInf 驱动的闭环评估架构、冻结套件、模型桥、证据协议与分阶段验收 |
| [10 · RoboParty RPO WBC 接入](10-roboparty-rpo-wbc-integration.md) | RPO V2.0 本体、58 仓库分析、RLinf WBC 训练、真机交付与多轮反思 |
| [11 · RPO 直立直行第一阶段](11-rpo-stand-walk-phase-a.md) | 训练目标、奖励、终止、独立评估判据与冻结协议 |
| [12 · 环境目录与选择](12-environment-catalog-and-selection.md) | 训练/评估环境实体、修订、编辑、绑定、执行规划与验收 |
| [13 · SONIC / RPO / RLinf](13-sonic-rpo-rlinf-integration.md) | SONIC 组件迁移、RPO 动作库与新接口、RLinf PPO/辅助损失、平台适配及多轮反思 |
| [15 · AMP / RPO / RLinf](15-amp-rpo-rlinf-integration.md) | 可行性、Actor–Critic/判别器结构、环境/step/rollout 预算、完整接入与独立评估、多轮反思 |
| [23 · 原生 UFO/RPO 部署](23-ufo-native-rpo-deployment.md) | 原生模型导入、推理契约、部署前双后端仿真与交付设计 |
| [24 · 实时仿真验证](24-ufo-live-simulation-validation.md) | 冻结模型/环境的部署会话、连续切换任务 z、真实姿态展示与诊断证据 |
| [26 · P-BFM / RPO / RLinf](26-pbfm-rpo-rlinf-integration.md) | RLinf 承载 P-BFM 原生训练、RPO 深度/地形契约、数据与恢复、仿真评估及多轮复核 |
| [26补充 · P-BFM四卡每卡2048环境](26-pbfm-rpo-4x2048-profile.md) | W4/E2048的512更新档、冻结Replay/batch/LR、分段恢复、逐卡容量及独立快评出口 |
| [26补充 · P-BFM单次512→10000探索](26-pbfm-rpo-direct-10000.md) | 用户明确选择的单Job有界恢复，沿用4×2048与原镜像，结束后独立评估；不泛化替代默认周期决策 |
| [26 · P-BFM 10000→100000扩展](26-pbfm-rpo-100000.md) | 已确认单Job直训、30小时上限、结束评估；严格镜像兼容、真实更新、周期保存与平台证据出口 |
| [26补充 · P-BFM故障恢复](26-pbfm-rpo-recovery.md) | 从完整19984周期状态恢复，保留原生终止、定位数值异常、冻结历史来源并继续100000目标 |
| [26补充 · P-BFM周期checkpoint评估](26-pbfm-rpo-periodic-evaluation.md) | 用户要求6万附近保存后暂停；已停止Run的完整周期状态登记、原镜像导出与32例/64视频评估 |

RoboTwin 所需的 RLinf 逐文件补丁设计单独维护在 [`../rlinf/`](../rlinf/README.md)。该目录展开组件实现边界，不改变本目录定义的平台不变量。

AMP 的逐任务落点、依赖、检查与验收见 [RPO AMP / RLinf 执行计划](../superpowers/plans/2026-09-24-rpo-amp-rlinf.md)。设计与计划已编写不代表接入或训练已经完成。

## 设计状态口径

- 本目录写“系统必须是什么”，不是功能完成清单。
- `必须`、`禁止` 是硬约束；`首期` 是当前实现目标；`以后` 不构成预留代码的理由。
- 数据、模型、评估和部署状态只能由可定位的运行证据提升，文件存在或容器退出码为 0 都不够。
- 示例数值只用于解释协议，除非明确标为实测，否则不是性能承诺。

## 核心结论

EmbodiedBein 是单机具身学习工作台。它以一套版本化资产和 Job/Run 生命周期连接机器人、本体接口、数据、训练、评估与交付；机器人和训练栈差异集中在受控 adapter/recipe，不复制业务系统。

系统保持以下不变量：

1. Docker Compose 是平台唯一部署入口，计算任务由 `ebd` 动态创建，不进入 Compose。
2. Python 负责控制面，Rust 负责执行与大字节数据面；只有 `ebd` 持有 `docker.sock`。
3. 外部源码和数据源只读，平台只写受管目录；正式训练只消费冻结版本。
4. 平台不进入机器人实时控制回路，不提供真机放行、门禁豁免或远程关节控制。
5. Agent 没有后门，所有动作经过 ToolBus、领域校验、预算与审计。
6. 已发布本体接口、数据版本、模型版本、评估套件与交付包不可原地修改，只能派生新版本。

## 当前范围

- Berkeley Humanoid Lite（BHL）：四个现有本体接口及其训练、评估和交付路径。
- SO101 单臂：数据接入、仿真采集、训练和具备现场条件时的端侧验证。
- Unitree G1 EDU：当前只设计仿真采集、训练、评估及 Orin 离线验证，不创建虚假真机能力。
- 单机 4×RTX 4090 训练；Jetson AGX Orin 64GB 是目标设备之一，但其软件栈和性能必须来自实机证据。

不建设多机训练集群、通用工作流平台、数据湖、任意脚本插件市场或浏览器实时遥操作。出现经过测量的单机瓶颈后再提出新设计。
