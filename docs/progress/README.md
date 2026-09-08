# EmbodiedBein 进度

设计真源仍是 [`docs/design/`](../design/index.html)。本目录只记录**实现对照设计的进度**，不改设计。

| 文档 | 说明 |
|---|---|
| [2026-08-21 当前状态](./2026-08-21-status.md) | **最新**：`feat/make-features-real` 对照 M0–M5；短训已通；剩余缺口 |
| [2026-08-21 make-features-real 验收证据](./2026-08-21-make-features-real.md) | 本机命令输出（GPU、短训、TSDB、CAS ONNX、infra_failed） |
| [2026-08-21 第一版状态与后续计划](./2026-08-21-first-version.md) | 骨架期基线（stub / busybox / Page.vue）；**不再代表当前代码** |

判定约定：

- **完成**：对照该里程碑验收项，主路径已可运行（允许大镜像/真机仍用占位）。
- **可用骨架**：领域逻辑或 API 已有最小实现，端到端或缺关键环节，或 UI 仅为 JSON 回显。
- **占位**：有路由/模块/空镜像，但不能完成设计描述的工作。
- **未做**：设计要求的能力尚不存在。
