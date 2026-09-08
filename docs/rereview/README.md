# EmbodiedBein 再审方案（RLinf 训练工作台终版设计）

本目录是对用户设计需求、`docs/design/`（01~13章）、`docs/review/`（全系统审查）、线上运行态（http://1.94.0.106:31800/）以及第一版重设计 `docs/redesign/` 进行**全面再审查（Re-Review）与深化**后的完整方案。

---

## 核心设计定位

将平台从「具身研发全流水线（数据漏斗 · 双训练栈 · 评估门禁 · 真机部署 · Agent）」收敛为：
**在单台 4×RTX 4090 机器上，以磁盘本地目录为真源，管理本体代码与 3D 资产、下载管理数据集、管理 Isaac Lab 镜像、填写表单提交 RLinf 仿真训练、实时查看日志/指标/结果，并对基座与训练产出模型建立 DAG 血缘。**

---

## 目录导航与阅读指引

| 文件 | 格式 | 核心内容 |
|---|---|---|
| [**index.html**](./index.html) | HTML 可视化看板 | **交互式总览大屏**：五模块架构图、状态机流转、雷达图、版本对比矩阵 |
| [**prototype/workbench.html**](./prototype/workbench.html) | HTML 原型 | **全功能可交互工作台原型**：包含五模块所有交互、3D查看、多URDF切换、RLinf表单与DAG血缘 |
| [**00 · 综合再审结论**](./00-executive-summary.md) | Markdown | 现行设计、初版重设计与终版再审的三方对比，核心遗漏与补强清单 |
| [**01 · 本体模块深度详规**](./01-embodiment-deep-review.md) | Markdown | 本地路径登记、多 URDF 切换、3D 静态资产代理服务、关节树与 DOF 联动、Gym 自动探测 |
| [**02 · 数据模块深度详规**](./02-data-deep-review.md) | Markdown | HF/HTTP/本地三来源异步下载状态机、磁盘配额预检、大目录惰性树、Episode 媒体流播放 |
| [**03 · 环境与镜像深度详规**](./03-env-and-images-deep-review.md) | Markdown | Isaac Lab 仿真栈绑定、Docker 镜像实探与 Pull、GPU/Vulkan/EGL 诊断自检、Nucleus 缓存 |
| [**04 · 训练与 RLinf 深度详规**](./04-training-and-rlinf-deep-review.md) | Markdown | RLinf 模板体系、Hydra 校验引擎、Ray 多卡拓扑与端口隔离、日志/tfevents 采集、产物解析 |
| [**05 · 模型与血缘深度详规**](./05-models-and-lineage-deep-review.md) | Markdown | 基座与训练产物双轨管理、Checkpoint 手动/自动登记、多输入多输出 DAG 血缘图谱 |
| [**06 · 元数据与磁盘存储详规**](./06-metadata-and-storage-deep-review.md) | Markdown | 完整 PostgreSQL 表结构、索引设计、磁盘目录布局、挂载安全矩阵与 UID/GID 权限 |
| [**07 · 七轮深度反思与边界审查**](./07-multiround-reflection.md) | Markdown | 真实运行态反思：并发冲突、挂载越权、Ray 端口竞争、大文件 OOM、数据一致性等 7 轮剖析 |
| [**08 · 工程迁移与落地实施计划**](./08-migration-and-implementation-plan.md) | Markdown | 渐进式重构路线图：API 路由拆分、Adapter 重构、前端 Vue 3 极简重塑与排期 |

---

## 终版再审关键确认（对齐用户意图）

1. **本体 3D 资产展示**：支持多 URDF 切换（如 BHL 的 Humanoid 22DoF, Biped 12DoF, WBC 22DoF, FixedHand 10DoF），提供关节树与基础位姿联动。
2. **训练产物模型登记**：训练详情页支持手动选择 checkpoint 文件登记为模型，同时支持在提交时勾选「训练成功后自动登记 latest」两种模式结合。
3. **数据接入与下载**：以本地路径直挂为主（默认），同时完整支持 Hugging Face（支持配置 `HF_TOKEN`）与 HTTP URL 异步后台下载。
4. **存储与挂载**：Postgres 管理元数据，磁盘管理真实文件，训练时只读挂载代码/数据/基座模型，可写挂载输出工作区。
