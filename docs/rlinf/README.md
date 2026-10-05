# RLinf 变更设计

本目录记录 EmbodiedBein 为 RoboTwin 多模型训练与评估所需的 RLinf 组件级改动。平台总体约束、数据语义、本体兼容和验收口径仍以 [`../design/08-robotwin-integration.md`](../design/08-robotwin-integration.md) 和 [`../design/09-robotwin-evaluation.md`](../design/09-robotwin-evaluation.md) 为准；本目录只展开 RLinf 内部的目标路径、接口和测试，不记录实现进度。

`/data/kk/RLinf` 是外部只读仓。本文列出的 RLinf 路径表示补丁目标，不授权直接修改该目录。实现应固定 RLinf 内容快照，在任务镜像构建时复制源码并应用平台维护的最小补丁，产出新的镜像 digest；运行时不得 monkeypatch 或回写外部仓。

| 文档 | 内容 |
|---|---|
| [01 · RoboTwin 多模型 SFT](01-robotwin-multimodel-sft.md) | 公共改动、五个待适配模型、配置、模型目录、smoke 测试和验收顺序 |
| [02 · RoboTwin 单 Policy、多环境评估](02-robotwin-single-policy-eval.md) | 一个容器一个常驻模型、多 Env Worker 批量推理、进程内 wave、headless 离屏渲染、资源画像和全量评估验收 |
