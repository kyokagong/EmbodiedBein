---
name: eb-integrate
description: Use when adding or extending EmbodiedBein embodiments, datasets, model sources, algorithms or training stacks, including 本体接入、数据接入、模型接入、算法接入 and their runtime bindings.
---

# 接入本体、数据、模型与算法

先读根目录 [AGENTS.md](../../../AGENTS.md) 和对应设计；能力以实际证据分级，登记、代码实现、容器 smoke、正式验收分别报告。适用于仓库开发，平台内 Agent 仍受 ToolBus 禁令约束。

## 按接入对象选择

| 任务 | 必读操作参考 | 设计 |
|---|---|---|
| 新机器人、本体接口、任务/环境 | [本体](references/embodiments.md) | [领域](../../../docs/design/01-product-domain.md)、[环境目录](../../../docs/design/12-environment-catalog-and-selection.md) |
| 外部数据、格式转换、采集数据 | [数据](references/data.md) | [数据与采集](../../../docs/design/03-data-collection.md) |
| 本地/HF 模型、训练产物、模型桥 | [模型](references/models.md) | [模型与训练](../../../docs/design/04-model-training.md) |
| 新算法、RLinf 模板、新训练栈 | [算法](references/algorithms.md) | [架构](../../../docs/design/02-system-architecture.md)、[模型与训练](../../../docs/design/04-model-training.md) |

多个对象一起接入时，按本体接口 → 数据/模型兼容 → 模板/运行时 → 独立评估推进，只读相关专题。

## 接入前固定最少信息

- 来源、许可证、固定 revision/hash、只读源路径、受管输出目录。
- 本体接口 ID/版本、Task、训练/评估用途；控制频率、动作单位/关节顺序/传感器等不能猜。
- 目标是“纳管可见”“可训练”“可评估”还是交付；所需依赖、GPU/主存、时间和输出预算。
- 现有入口是否已覆盖。先复用 adapter、template、worker 和 JobKind，不为每个算法复制业务生命周期。

缺少动作语义、来源权利、验收标准或高成本运行预算等关键输入时询问用户；期间可以完成只读梳理及无需该输入的改动，不捏造运行证据。

## 完成检查

1. 预览与提交用一致校验；冻结接口、数据、模型、代码快照、镜像和环境版本。
2. 计算经过 Job/Run/ebd；外部源只读，输出只写受管位置；失败、取消和重复提交有明确语义。
3. 验证实际 loader/runner 可消费输入，checkpoint/receipt 可被 collector 接收，并能追溯源版本。
4. 用独立冻结 Suite 验证模型；Job 成功不等于 `qualification=pass`，smoke 不等于正式验收。交付/目标设备见 [评估与交付](../../../docs/design/05-evaluation-delivery.md)。
5. 留下所执行检查、Job/Run/Artifact/报告引用、未验证项；更新 `docs/progress/`，不降低设计要求。

例如“新增一个 23 关节机器人 PPO”：先核实资产和动作契约，再建立 Task/Profile/模板与镜像，最后完成短训、checkpoint 回读及独立评估；仅增加 YAML 或下拉选项不能宣称接入完成。

常见遗漏：`EmbodimentEnv` 与 `EnvironmentProfile` 混用；只加 adapter 未注册分发；下载完成当作兼容；恢复只加载权重却标为完整训练状态恢复。命令入口见 [工具手册](../../tools.md)。
