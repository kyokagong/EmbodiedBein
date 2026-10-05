# EmbodiedBein 进度

设计真源仍是 [`docs/design/`](../design/README.md)。本目录只记录**实现对照设计的进度**，不改设计。

| 日期 | 文档 | 说明 |
|---|---|---|
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
