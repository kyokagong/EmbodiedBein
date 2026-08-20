# EmbodiedBein 去 Mock 执行计划

索引：本目录只放**可执行**计划，不改设计。对照分析见 [`docs/progress/2026-08-21-first-version.md`](../progress/2026-08-21-first-version.md)。

| 计划 | 范围 | 完成定义 |
|---|---|---|
| [2026-08-21 全部功能真实可用](./2026-08-21-make-features-real.md) | M0–M5 去掉 stub / JSON 模板 / busybox 镜像 | 用户在 UI 上走完训练、评估、数据、打包、Agent；每条路径都有真实容器或真实解析，禁止假指标 |

执行时用 `AGENTS.md` 硬约束。推荐按该计划的 Task 顺序，一任务一提交。
