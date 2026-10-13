# EmbodiedBein 工具与技能

本目录服务于维护仓库的开发 Agent 和开发者；平台内 Agent 仍只能经 ToolBus 使用已注册工具，不能使用本文的 shell、文件系统或 Docker 命令。

设计依据是 [docs/design](../docs/design/README.md)，实现与运行证据见 [docs/progress](../docs/progress/README.md)。这里记录操作方法，不替代设计，也不因文档存在而宣称能力已验证。

| 当前任务 | 使用入口 |
|---|---|
| 接入本体、数据、模型、算法/训练栈或评估能力；新接入的分阶段训练 | [eb-integrate](skills/eb-integrate/SKILL.md) |
| 定位 API、排队、容器、训练、数据、模型、评估或页面故障 | [eb-debug](skills/eb-debug/SKILL.md) |
| 修改实现、迁移、测试、构建、ToolBus 工具与文档 | [eb-develop](skills/eb-develop/SKILL.md) |
| 查找现有工具与可复制命令 | [工具手册](tools.md) |
| 技能整理的多轮检查与验证边界 | [复核记录](review.md) |

技能采用 `skills/<name>/SKILL.md` 结构，包含 `name` / `description` 元数据。使用时只读相关技能及它指向的专题；工具不支持自动发现时，直接按上述路径读取。所有 shell 示例从仓库根目录执行，先确认所需镜像、服务及实际配置。

涉及训练或评估的接入，必须读取 `eb-integrate` 的[训练与评估闭环](skills/eb-integrate/references/training-evaluation.md)：先跑通，再按实测利用更多可用 GPU 扩容；每个固定 step 区间完成快评、训练指标时序分析和继续/停止决策，保留成功/失败录像与证据。本文描述执行要求，不表示所有 recipe 已支持周期评估或多卡训练。
