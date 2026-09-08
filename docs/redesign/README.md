# EmbodiedBein 再设计（RLinf 训练工作台）

本目录是对照你给出的产品方向，对 `docs/design/`、`docs/review/` 与线上 http://1.94.0.106:31800/ 重新审视后的**新一版设计**。

它**不替换** `docs/design/` 作为历史真源；落地时以本目录为准。旧设计里评估、部署、Agent、Showcase 等仍可查阅，但**本版不做、不进导航**。

## 怎么读

| 文件 | 内容 |
|---|---|
| [index.html](./index.html) | 可视化总览：定位、导航对照、主路径 |
| [00-delta.md](./00-delta.md) | 与现行设计/线上的保留、改名、砍掉 |
| [01-overview.md](./01-overview.md) | 产品定位、信息架构、用户主路径 |
| [02-metadata-and-storage.md](./02-metadata-and-storage.md) | 元数据目录 + 磁盘挂载 |
| [03-modules.md](./03-modules.md) | 本体 / 数据 / 环境 / 训练 / 模型 详规 |
| [04-architecture.md](./04-architecture.md) | 控制面、数据面、容器、调度 |
| [05-self-review.md](./05-self-review.md) | 多轮反思：遗漏、矛盾、待确认 |
| [prototype/workbench.html](./prototype/workbench.html) | **可点击工作台原型**（本版主交付） |

建议顺序：先点开 [原型](./prototype/workbench.html)，再读 [01](./01-overview.md) 与 [00](./00-delta.md)。

## 一句话

把平台从「具身研发全流水线（数据漏斗 · 双训练栈 · 评估门禁 · 真机部署 · Agent）」收成：**在单机上，用磁盘上的代码/数据/模型，填 RLinf 表单，跑 Isaac Lab 仿真训练，并看日志、指标与产物。**
