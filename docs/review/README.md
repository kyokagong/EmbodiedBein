# EmbodiedBein 全系统实现度审查与改进建议总览

- **审查基准**：`docs/design/`（全套 14 篇设计规范真源，涵盖 01~13 章及本体环境规范）
- **当前服务**：http://1.94.0.106:31800/ （实测 4× RTX 4090 宿主 `eai4090x4m1`）
- **审查日期**：2026-08-27
- **审查方法**：多轮深度反思 + 真实运行态 API 探测 + 源代码（Backend Python / Data-Plane Rust / Frontend Vue）逐行对照

---

## 1. 核心结论与全景评分卡

EmbodiedBein 当前已成功走过 **M0（骨架与只读纳管）** 并打通了 **M1（Isaac/RSL-RL 短训主路径）**，具备「真实 4090 容器化短训、Rust TSDB 采集、ONNX 维度实测 G3、无镜像诚实失败、本体环境自洽校验」的核心能力。

但对照设计全景，系统目前处于 **「领域内核可用，但上下游子系统大多为薄骨架，离设计验收的完整闭环和工业级交付存在明显差距」** 的阶段。

### 1.1 全系统九大子系统成熟度评分

| 子系统 / 模块 | 设计章节 | 实现成熟度 | 核心现状 | 最主要缺口 |
|---|---|---|---|---|
| **1. 架构与数据面** | 01 · Architecture | **65%** | 单机 Compose、多 crate 结构、UDS 通信、4 GPU 真实探测已就绪 | Redis Streams 未作为事件总线激活；ebd 与 FastAPI 仍靠直接轮询；CAS 缺少 Range/HLS |
| **2. 领域模型与本体环境** | 02 · Domain / 12 / 13 | **75%** | 4 份 YAML 落库自洽、CRUD 原型已落地、Three.js 3D 查看器初版可用 | 历史数据表仍混杂 `contract_id` / `ec_*`；提交仅支持 `train.rl`；草稿表单未做可视化双向同步 |
| **3. 数据管理与漏斗** | 03 · Data | **40%** | ebd scanner 可用、manifests 扫描 17/9 真统计、Episode 单视频回显 | 缺少六层模型实体 (L0-L5)；漏斗无下钻与敏感性滑块；重定向缺 pinocchio；缺脚本专家采集流 |
| **4. 模型注册与容量估算** | 04 · Model | **60%** | 6 遗留 ONNX 实测维已入库、L2 显存公式与 no_fit 拒绝可用 | 缺少 L1 实测峰值回填与配置指纹索引；ModelCard 必填项未严格阻断；缺少资产一致性自动化比对 |
| **5. 任务编排与训练可视化** | 05 · Training | **60%** | IsaacLab 适配器已跑通、worktree 隔离有效、Run 基础曲线/日志可见 | RLinf 尚未长周期压测；调度器缺优先级/抢占/超时自动杀；Run 页面缺少设计网格与终止原因堆叠 |
| **6. 评估体系与门禁** | 06 · Evaluation | **35%** | G1~G5 DSL 引擎已通过单测、离线 eval JSON 能解析 | BHL HEAD `play.py` 缺 CLI 参数限制现场评测；缺 L3 跨域漂移热力图 (|Δμ|/σ)；缺失败视频墙与聚类 |
| **7. 部署与真机遥测** | 07 · Deploy | **35%** | Bundle 基础打包、G4/sim2sim 校验拦截生效、S0/S1 放行 403 阻断 | 缺少 ONNX vs PyTorch 数值一致性自检；robot-agent 仅为存根；真机遥测无聚合时序分析 |
| **8. Agent 与 ToolBus** | 08 · Agent | **50%** | 禁止动作硬拦截生效、7 剧本目录注册、无 LLM 规则路径通过 | ToolBus 缺速率桶/周预算硬顶/磁带回放；剧本多为静态文本而非多步编排；UI 控制台过于简陋 |
| **9. 前端 UI 与 Showcase** | 09 · UI / 11 · Views | **45%** | 31 路由已无 Page.vue 占位、主流程可点通、支持 Bearer/Cookie | 大多数页面视觉与交互停留在极简表单，缺少 09/11 章暗黑高密设计系统；Showcase 仅为基础 JSON 回显 |

---

## 2. 审查文档索引导航

本审查结果按模块与设计章节拆分为 10 份专项深度报告，并在 `docs/review/` 提供交互式 HTML 仪表盘：

1. [**00 · 交互式综合审查仪表盘 (HTML)**](./index.html)：包含动态雷达图、子系统打分卡、GAP 矩阵可视化与阶段演进图。
2. [**01 · 系统架构与数据面审查**](./01-architecture-and-dataplane.md)：对照 01 章，深入分析 Python/Rust 边界、UDS 通信、Redis Streams、worktree 隔离、Docker 容器生命周期。
3. [**02 · 领域模型与本体/本体环境/任务审查**](./02-domain-embodiment-and-tasks.md)：对照 02/12/13 章，评估本体环境演进、四元组提交、3D 查看器、CRUD 原型及 G1~G5 规则。
4. [**03 · 数据管理子系统审查**](./03-data-subsystem.md)：对照 03 章，评估六层数据模型、格式嗅探、155k→35 漏斗审计、Episode 三联预览、专家采集与重定向。
5. [**04 · 模型注册与容量估算审查**](./04-models-and-capacity.md)：对照 04 章，评估 A/B 两类模型管理、三层容量估算闭环、ModelCard 规范与资产一致性。
6. [**05 · 任务编排与训练可视化审查**](./05-training-and-orchestration.md)：对照 05 章，评估 5 大适配器、调度器队列策略、TSDB 实时多图表网格与 Run 对比显著性。
7. [**06 · 评估体系、门禁与排行榜审查**](./06-evaluation-and-gates.md)：对照 06 章，评估五层指标体系、EvalSuite seed 注册表、L3 跨域漂移热力图与失败样本库。
8. [**07 · 部署流水线与真机遥测审查**](./07-deploy-and-telemetry.md)：对照 07 章，评估 PolicyBundle 规范、导出数值自检、分级放行流程与 robot-agent 遥测架构。
9. [**08 · Agent 体系与 ToolBus 审查**](./08-agent-and-toolbus.md)：对照 08 章，评估 ToolBus 七道处理、预算账本、7 个剧本多步编排、磁带回放与控制台。
10. [**09 · 前端 UI 系统与 Showcase 审查**](./09-ui-and-showcase.md)：对照 09/11 章，评估 31 个页面的高保真还原度、设计系统组件库、交互一致性与 Showcase 策展分享。
11. [**10 · 综合差距矩阵与演进路线图**](./10-comprehensive-gap-and-roadmap.md)：全景 GAP 清单、风险登记册、按 P0~P5 优先级的实施改进计划。

---

## 3. 总体改进实施建议路线 (P0 ~ P5)

```
阶段 P0 (基础闭环与真实硬件，2周)
  ├── P0.1 调度器补齐 (优先级/超时kill/OOM重试)
  ├── P0.2 Run 详情页重构 (ECharts多网格 + 终止原因堆叠 + WS重连)
  └── P0.3 Showcase 页面高保真还原 (里程碑时间线 + 月度演进图 + 演示视频)

阶段 P1 (评估体系与门禁闭环，2周)
  ├── P1.1 评估套件 (EvalSuite) 与 Wilson 95% 置信区间
  ├── P1.2 sim2sim 跨域一致性诊断 (L3 观测漂移热力图 |Δμ|/σ)
  └── P1.3 排行榜与对比页 (显著性差异判定 + 配对对比)

阶段 P2 (数据主线与多栈支持，3周)
  ├── P2.1 六层数据模型落库 (DataSource / CanonicalEvent / Scenario / DataSlice)
  ├── P2.2 漏斗审计视图升级 (拒绝原因下钻 + 规则敏感性模拟)
  └── P2.3 RLinf SFT/RLFT 多卡实测与环境固化

阶段 P3 (策略包与部署遥测，2~3周)
  ├── P3.1 PolicyBundle 导出数值自检 (ONNX vs PyTorch ≤ 1e-4)
  ├── P3.2 分级放行流程 (S0/S1/S2) 界面与审计
  └── P3.3 robot-agent 遥测时序聚合 (Jitter/温度/CAN状态时间线)

阶段 P4 (Agent ToolBus 完善，2周)
  ├── P4.1 ToolBus 完善 (周预算硬顶 + 速率令牌桶 + 补偿机制)
  ├── P4.2 7 个剧本升级为真实多步编排
  └── P4.3 磁带回放测试 (Tape Fixture, llm_calls == 0)

阶段 P5 (工程架构收尾，全程穿插)
  ├── P5.1 backend/eb/api/routes.py 按资源拆分文件
  ├── P5.2 Redis Streams 事件流完全打通
  └── P5.3 补充 CI 四项门禁 (Ruff, 快照测试, Domain 单测, Alembic 双向迁移)
```
