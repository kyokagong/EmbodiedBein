# 07 · 工作台与 API 设计

## 1. 信息架构

主流程入口收敛为：**总览 / 机器人 / 数据 / 训练 / 模型 / 评估 / 交付 / 环境**。设置位于侧栏底部；Agent 是辅助入口，不是主流程依赖。

不设置独立的任务、实验、队列、排行榜一级导航：任务配置放机器人/场景，队列是各作业列表状态，排行榜属于评估。现有兼容路由可以保留，但不能形成第二套概念。

| 入口 | 必须回答的问题 |
|---|---|
| 总览 | GPU/RAM/磁盘是否可用，哪些管线失败或等待证据 |
| 机器人 | 哪些形态、资产、接口、任务和标定可用，证据到哪一级 |
| 数据 | 来源、采集、版本、质量、Slice 和训练兼容性 |
| 训练 | 冻结了什么输入，实际用了什么，资源/指标/checkpoint 状态 |
| 模型 | 权重、处理链、能力矩阵、血缘、评估和目标 variant |
| 评估 | Suite、case 完成度、质量、失败、对比和回归 |
| 交付 | Bundle、目标设备、待补证据和现场记录；没有运动按钮 |
| 环境 | 采集/训练/转换/评估镜像和容器内 probe |

## 2. 关键页面

### 机器人

详情展示资产版本/3D、关节树、观测条带、组合动作、频率、frame、限制、仿真/真机绑定和 Gate 引用。发布版本只读；修改通过 clone 草稿。G1 同时明确显示“仅仿真”。

### 数据

列表区分文件数、attempt、episode、有效时长和 published 状态；unknown 不显示成 0。版本概览的覆盖、质量、split 和许可统计都能下钻到固定成员。

Episode 视图共享时间游标，显示多视频 PTS/age、3D、状态、requested/applied action、事件、指令和规则。Slice 向导执行版本选择→筛选→group/split→采样→export profile→loader probe→发布。

### 训练与模型

新建页展示 resolved config、真实版本、挂载、资源 probe 和保护字段。训练详情以 Job/Run 状态为真源，展示 input receipt、指标、日志、资源、checkpoint 完整性和恢复链。

模型详情按能力维度展示证据，不用一个“已支持”徽章；血缘可追到数据、Run、processor/norm、评估、优化 variant 与 Bundle。

### 评估与交付

评估页同时展示 case 完成率、valid 分母、区间和 qualification。条件不一致时禁止并排排名并列出差异。失败墙链接对应 episode/time range。

交付页展示包与目标设备匹配、目标性能和缺失证据。现场报告上传只记录结果，不触发设备动作。G1 真机阶段固定不可用。

## 3. API 约定

沿用 `/api/v1`、现有鉴权和 JobSpec。资源 API 与执行 API 分离：创建/修改元数据不应同步处理大目录；耗时动作返回 `202 + stable id`。

核心资源语义：

| 资源 | 关键动作 |
|---|---|
| embodiments/envs/tasks | 草稿 CRUD、扫描、验证、clone、发布、deprecate |
| datasets/versions | 登记来源、索引、校验、episode/tree/media、Slice/导出 |
| collection/pipeline runs | 创建、查节点/shard、幂等取消与失败重试 |
| train | template、preview、submit、详情、checkpoint、resume candidate/preview/submit |
| models | 权重登记/下载、版本、能力、血缘、优化 |
| eval | suites、preview、jobs、progress、reports、compare/failures |
| bundles/targets | 构建、下载、目标 external report；无 release-to-robot 动作 |
| agent | session、ToolBus 结果、审批和审计 |

现有端点可以合并重叠功能，不要求机械新增每个路径。不得继续使用 `train.rlft` 表示数据采集等错误语义。

错误统一含 `code, message, field_path, retryable, evidence_ref`。结构错误 400/422、权限 403、版本/幂等冲突 409、缺资源 404；资源暂时不足排队，永远无法满足则返回 `no_fit`。

`Idempotency-Key` 绑定主体、操作和请求 hash；同键同输入返回原结果，同键不同输入 409。列表使用 cursor 和稳定排序；过滤是结构化受限表达式，禁止传原始 SQL。

## 4. 事件、媒体与大数据

事件包含 event_id、run_id、seq、timestamp 和 payload schema。WebSocket/SSE 重连从 cursor 补持久状态；慢客户端可丢预览帧，不能丢 Job 终态和资产发布事件。

媒体 API 只接受受控 stream ref，不接受宿主路径。`ebd` 支持 Range、ETag 和鉴权后的流式响应；非法范围正确报错。缩略图/代理视频标为预览，像素级验证读取原始发布帧。

时序查询接受 stream、time range 和 max points，返回单位、缺失 mask 与 min/max 分桶。热路径不在 FastAPI 中解析。

## 5. 安全、分享与可访问性

列表、详情、媒体、导出和分享使用同一资产权限。公开 Showcase 只读取冻结快照，不能成为原数据下载旁路；许可不允许时不生成公开媒体。

凭据只用 secret ref，前端不获取 HF token 或容器环境。路径选择器只浏览登记根，诊断路径默认脱敏。用户提供的 HTML/文本以安全文本预览。

颜色不是唯一状态信号；表格、表单、时间游标和图表可键盘操作，图表有单位、样本量和文本摘要。缺媒体、权限不足、索引未完成、版本已回收和目标未连接使用不同空态。

## 6. 性能验收

在 10 万 episode 的代表性索引上，首期目标是列表和已索引摘要 p95 < 1s、代理视频首帧 p95 < 2s。报告必须注明硬件、缓存、媒体和网络；这些是待验证 SLO，不是未测量的承诺。达不到时先修索引、查询、分片和缩略图，不用前端假进度遮掩。
