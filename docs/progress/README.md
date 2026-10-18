# EmbodiedBein 进度

设计真源仍是 [`docs/design/`](../design/README.md)。本目录只记录**实现对照设计的进度**，不改设计。

| 日期 | 文档 | 说明 |
|---|---|---|
| 2026-10-18 | [UFO交互环境关闭与资源回收](2026-10-18-ufo-session-close.md) | 独立关闭、切页等待确认、刷新keepalive、创建中按幂等键取消和活动环境管理；六项实际HTTP浏览器/PG竞态通过，5个Job取消、3个容器删除、四GPU空闲。88后端/67前端检查，无Git提交；异常断网的初始化阶段仍受总超时兜底 |
| 2026-10-18 | [UFO原生GPU画面WebSocket推流](2026-10-18-ufo-websocket-stream.md) | 替代被拒绝的HTTP轮询/Three方案；GPU原生640×480、同连接任务/相机、8帧有界窗口；修复EGL驱动选择和ebd扫描停顿。两环境实际HTTP浏览器27–27.5fps，帧龄p95约89/96ms；120后端/53前端/28ebd/4collector检查，媒体摘要与Range已核验；MJLab物理约0.59×，仅交互诊断，无训练/无Git提交 |
| 2026-10-18 | [UFO实时仿真验证](2026-10-18-ufo-live-simulation.md) | 已部署模型/环境选择、同Run连续z切换、独立重置和真实姿态/诊断媒体；修复IP+HTTP的randomUUID错误，入口改为先启动环境再切换。两后端/实际HTTP浏览器/租约/部分下载已验，88后端/53前端/21ebd检查；仅diagnostic，仍有越限，无训练、无Git提交 |
| 2026-10-17 | [原生 UFO RPO 部署与仿真接入](2026-10-17-ufo-native-deployment.md) | 指定模型导入/ONNX 数值一致性、双后端 6 个完整案例/6 视频、登录 GET/SHA/Range 与 41.5MB 仿真包已验证；0/6 满足任务判据，diagnostic/simulation_only。三轮复核修复语义/调度/失败证据/来源链；无训练、无 Git 提交；正式验收/真机未验证 |
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
