# 07 · 页面、可视化与接口设计

## 1. 从六模块扩展为八个稳定入口

主导航：**总览 / 机器人 / 数据 / 训练 / 模型 / 评估 / 交付 / 环境**；设置保持侧栏底部。机器人沿用 `/embodiments` 兼容路由；交付用词说明平台负责包和证据，不提供远程运动按钮。

不恢复独立“任务/实验/提交/队列/排行榜”一级入口。任务配置嵌入机器人与场景；队列是各作业列表状态；排行榜在评估里；量化在模型里。Agent/Showcase 保留既有路由与禁止边界，本轮不扩展为主流程依赖。

| 模块 | 列表关注点 | 详情页与主操作 |
|---|---|---|
| 总览 | 4卡、RAM、磁盘、各管线运行和阻塞 | 直接跳到需处理的缺资产/失败步骤/设备证据 |
| 机器人 | SO101 单臂、G1 EDU 仿真、BHL 形态；仿真/真机状态分别显示 | 资产3D、接口、任务、标定、能力证据；登记/扫描/验证 |
| 数据 | Sources、Collection Sessions、DatasetVersions、Slices、Pipelines 标签 | 接入、采集、审核、切片、导出、训练 |
| 训练 | 当前/历史训练作业与资源；按机器人/模型/状态筛选 | 新建、日志指标、输入回执、产物、重试/停止 |
| 模型 | 基座、训练版本、部署变体；每项能力状态 | 数据/运行血缘、评估、优化、目标基准 |
| 评估 | 套件版本、报告、失败、对比 | 冻结套件、运行、对比、回归候选 |
| 交付 | 候选包、目标设备、待补证据、现场记录 | 下载包/自检说明、导入目标报告；无真机放行接口 |
| 环境 | 采集/训练/转换/评估镜像，设备能力、资源画像 | pull/probe、版本依赖、缺失库与相机诊断 |

## 2. 数据工作台是产品核心

### 2.1 数据列表与概览

固定列：名称+版本、机器人/接口、任务族、来源 sim/real/ego、物理格式版本、有效 episode/小时、attempt 成功率、质量状态、训练兼容模型、大小、最近 Job。

`indexed=false` 时数量为“未索引”，不显示0；没有 episode reader 时显示“文件可浏览，训练不可用”。来源是 G1 仿真必须显式显示“仿真”；不显示机器人在线点。

概览展示覆盖矩阵（机器人×任务×来源×场景分组）、成功/失败/invalid 占比、动作/状态分布、时间对齐异常、split 分布和许可用途状态。所有图表有版本、单位、样本量、筛选条件，可点击打开相同成员集。

### 2.2 Collection Session

页面左侧为冻结配方与覆盖计划，中间为 DAG/步骤表，右侧为实测资源和尝试统计。顶部显示“合格目标 / 已合格 / 全部尝试 / 预算剩余 / 预计空间”。ETA 是估计区间，基于有效吞吐；刚开始没有数据时显示未知。

每个 shard 可查 seed/case 范围、Run、已提交 episode、失败、日志与输出 manifest。取消动作说明保留哪些已完成数据；“重试失败 shard”只作用于未完成项。对于人工标注或 Orin 回执等待，展示需要的输入、完成条件和责任人，不能给无限旋转进度条。

### 2.3 Episode 诊断视图

```text
版本 / episode / 机器人接口 / 来源 / split    [质量] [原始-转换对照]
┌───────────────────────┬────────────────────────┐
│ front / wrist 相机     │ 机器人3D / 目标位姿     │
│ 每路 PTS、age、缺帧    │ 真实状态重建、资产版本   │
├───────────────────────┴────────────────────────┤
│ 共享时间游标  ───── 事件/接触/接管/失败区间 ─── │
├───────────────────────┬────────────────────────┤
│ requested/applied/    │ 指令、任务目标、success │
│ measured 曲线及单位   │ 规则、转换、来源与标注   │
└───────────────────────┴────────────────────────┘
```

用户能切换原始时钟/控制时钟，固定关节组，点规则跳到问题时刻；多个视频按 stream ref 与 episode offset 加载。原始样本检查和浏览器同步预览分开：显示 seek 精度与同步误差，不宣称浏览器毫秒级完全同步。

“加入候选 Slice”保留原 episode_id 与来源；“标注错误”生成 annotation revision；“派生裁剪”生成新数据版本。操作不修改已发布 payload。不支持的 MJCF/USD Web 预览显示关节树/静态资产信息，不用占位机器人。

### 2.4 Slice 与训练准备

向导：选固定版本→按任务/质量/来源筛选→检查 group/split→配比/采样→选 export profile/model recipe→loader 预检→发布。

提交前显示原始数、有效数、训练/验证/测试成员、祖先交集、每机型实际动作 schema、norm 来源。冲突例：“SO101 joint target(rad) 与选定 EEF delta(m) recipe 不兼容”，附定位与可用转换；不只显示 G2 红点。

## 3. 训练、评估与优化的页面串联

训练新建页保留模板+常用项+高级项，但模板按机器人接口/模型能力筛选；右侧显示**已解析**配置、真实输入版本、挂载、资源 probe 和禁止覆盖的系统字段。无数据的纯 RL 有明确标签；SFT 必填数据。

训练详情显示 active Run、输入回执、loss/reward/throughput、GPU/RAM/IO、checkpoint 状态。latest/best/人工挑选有不同标记；输出 hash 尚未完整时显示 writing。状态根据 Job/Run 真源刷新，不能只在打开详情时写回 TrainJob。

评估页同时显示完成率与质量指标。对比条件不一致时禁止混榜并列出差异；失败墙连接 Episode 诊断。量化页并排展示“原模型/候选精度/实际 Orin 延迟与内存/任务质量/未完成检查”，没有数据的图表保留空态。

交付页列出包与设备的匹配证据。G1 仿真有离线目标基准入口，真实部署阶段固定 hardware_unavailable。SO101/BHL 的现场结果上传不会触发设备行为；UI 不出现“点击后机器人开始运动”的按钮。

## 4. API 草案（增量设计，尚未实现）

沿用 `/api/v1`、现有身份/权限和 JobSpec，不另建下载、采集、训练三套执行服务。

| API / 变化 | 输入与返回要点 |
|---|---|
| `POST /datasets`（演进现有） | source kind/revision + source_ref；返回 dataset_id 与 import job_id，不能同步遍历整个目录 |
| `GET /datasets/{id}/versions` | published/discovered/invalid 状态、schema、接口、内容 hash、summary |
| `POST /datasets/{id}/versions/{vid}/index` | 幂等键；返回 data.index Job |
| `GET /datasets/{id}/versions/{vid}/episodes` | cursor、稳定排序、受限 filters；返回 episode 摘要 |
| `GET .../episodes/{ep}` | stream refs、timestamps/index refs、action schema、事件、质量 |
| `GET .../episodes/{ep}/series` | stream/time_range/max_points；返回单位、min/max 分桶与缺失 mask |
| `GET .../episodes/{ep}/media` | logical stream + version；鉴权定位 Range，禁任意绝对路径 |
| `POST /collection-sessions` | recipe revision、robot interface、scene、seed plan、budget；返回 pipeline/run 引用 |
| `GET /collection-sessions/{id}` | 冻结输入、节点/shard 状态、attempt/accepted 账、预算 |
| `POST /pipelines/{id}/runs` | 已发布模板+绑定输入；创建运行实例，不能接受任意命令 |
| `POST /pipelines/{id}/runs/{rid}/cancel` | 幂等取消；返回各节点最后状态和保留产物 |
| `POST /pipelines/{id}/runs/{rid}/retry` | 明确失败节点/shard，沿用输入版本，创建新 attempt |
| `POST /datasets/{id}/slices/preview` | 结构化筛选 AST，受限算子；返回 preview job/结果引用，禁止原始 SQL |
| `POST /datasets/{id}/slices` | 冻结成员、split/norm/export；返回版本化 Slice 与准备 Job |
| `GET /model-recipes` | model/operation/device/interface 过滤，各能力独立 evidence |
| `POST /train/preview`、`POST /train`（修复） | 共用 renderer，inputs 解析、probe、版本冻结与 Job 引用 |
| `POST /eval/suites/{id}/runs` | 固定模型/套件/variant；返回 eval Job |
| `POST /models/{id}/optimizations` | recipe+calibration+target profile；返回派生管线 |
| `POST /target-reports` | 已登记 external_run 的结果 manifest/附件引用，hash/身份/重放检查 |

真实 implementation 可以合并功能重叠的既有端点；这些是资源语义，不要求照表新增每个 route。JobKind 可为 data.index/validate/export/model.optimize 等增加所需枚举，但应先复用现有 scan/convert/slice/export/bundle；不能继续用 train.rlft 表示采集。

异步动作返回202与稳定 ID；结构错误400/422、权限403、版本冲突409、缺资产404；资源暂时不足可排队，永远无法满足的 profile 明确 no_fit。错误统一带 `code,message,field_path,retryable,evidence_ref`；logs 不包含 token。Idempotency-Key 绑定 principal/operation/request hash，相同键不同输入返回409。

## 5. 权限与大数据传输

列表、详情、媒体、预览、导出都验证同一资产访问权限，不因为分享缩略图而绕过数据限制。凭据以引用传递；前端不能拿到 HF token。平台 Agent 使用同一 ToolBus 的受限资源 ID，不可提交绝对路径、任意 URL 或自定义执行器。

媒体使用受控 stream ref。Range 请求保留 Content-Range/Accept-Ranges/ETag；非法/越界 Range 正确返回错误，字幕/文本/HTML 等不可信内容用安全文本预览。API 返回的外部路径用于诊断时脱敏，路径选择器只浏览已登记根。

事件包含 event_id、run_id、seq、timestamp 和 payload schema；WS 重连以 cursor 补查持久事件/状态。慢客户端可丢预览帧并显示 skipped，不丢 Job 终态和资产发布结果。

## 6. UI 性能与无障碍验收目标

在10万 episode 的索引验收数据上，目标是默认列表 p95<1s、已索引摘要 p95<1s、可播放首个代理帧 p95<2s；结果须注明机器、冷热缓存、媒体格式和网络，属于待验证 SLO。若达不到，先修查询/index/分片和缩略图，不用虚假前端进度遮掩。

颜色不是唯一状态提示；键盘可操作表格、时间游标和表单；图表有文本摘要、单位和屏幕阅读标签。缺媒体、权限不足、版本已删除、目标未连接、索引未完成均有不同空态与下一步动作。
