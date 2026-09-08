# 08 · 实施顺序、迁移与验收

## 1. 交付不是恢复一排导航

先修输入/输出与状态真源，再验证三机器人共同协议，随后做有意义的数据生产、训练、评估与端侧交付。每阶段按**证据条件**结束，不按某页是否存在或“第几周”算完成。

本轮是设计交付，不实施以下代码任务。执行阶段仍须容器化、外部仓只读、不使用 git，不调用会隐式执行 git 的现有训练快照路径。

## 2. 分阶段实施

| 阶段 | 共同工作 | SO101 单臂 | G1 EDU 仿真 | BHL | 完成证据 |
|---|---|---|---|---|---|
| A 基础可信 | JobSpec.inputs、Run 输出、幂等、状态同步、受控启动、路径保护、内容快照 | 注册资产/接口 | 注册具体仿真形态 | 保留四接口与历史数据 | 相同输入生成相同执行规格；错输入被拒；重启不重建已存在容器 |
| B 通用数据骨架 | source/version/episode/slice、索引、质量、版本发布、媒体 | 导入真实结构的少量 LeIsaac HDF5/LeRobot | 导入仿真状态/动作 episode | 导入现有 Reach 原始与成功视图 | 三种数据能索引、查时序、导出回读；无文件名猜测 |
| C 自动采集 | CollectionRecipe、有限 shard、attempt 账、失败恢复、预算 | 状态机+示教/Mimic | 接选定场景、controller 与 expert/rollout | 复用严格 Reach collector；WBC 轨迹摘要 | 成功/失败均可查；中断恢复不丢已提交记录；覆盖不足可补采 |
| D 多模型训练 | 注册目录、六种 SFT worker 分支的能力状态、loader smoke、资源 profile | 至少一个 RLinf VLA recipe 消费 SO101 Slice | RLinf env bridge + locomotion/control recipe | Reach SFT/WBC 回归 | 每机型真实训练有输入回执、指标、完整模型；未支持模型显式原因 |
| E 独立评估 | suite freeze、逐 case、CI、失败墙、扰动、sim2sim 条件 | 固定抓放/到达套件 | 固定地形或操作套件 | 对应接口 eval/sim2sim | 完整报告可复现；同病例配对；不完整评估禁止通过 |
| F Orin 与交付 | target profile、导出/优化/基准、包规范、external_run 回执 | 可行策略的64GB目标验证与现场方案 | Orin 离线验证；真实阶段不可用 | 策略目标验证与现场方案 | 原版/优化版质量+时延+内存证据；未满足时保持候选 |
| G 回流与持续改进 | 失败→候选数据、版本演进、备份恢复、GC | 实例数据按标定回流 | 仅仿真失败回流 | 实例/仿真失败分别回流 | acceptance 不泄漏；版本引用、恢复与空间账可核对 |

阶段 A/B 的共同协议必须同时用三个机型检查，不能 A–F 全跑完一个机型后才发现无法表达第二个。阶段 C–F 的机型适配可以独立推进，但依赖失败时状态真实保留；G1 不具备真机不阻止其他仿真工作。

## 3. 文件改造边界

| 现有位置 | 建议改动 | 不需要重建的部分 |
|---|---|---|
| `backend/eb/api/routes_train.py` | 同一解析器服务预览/提交；填 inputs；绑定 active_run；从 manifest 认领模型 | 训练 API 与列表框架 |
| `domain/rlinf_form.py`、`adapters/rlinf.py` | 版本化 recipe、能力验证、实际 Hydra resolve、参数数组 launcher | 现有 Adapter→RenderedJob 边界 |
| `orchestrator/service.py`、`collect.py` | 无 git 内容快照、唯一 Run 路径、产物完成提交、幂等恢复 | Job/Run 状态机与既有短训路径 |
| `api/routes_data.py`、`domain/data_view.py` | 下载转 Job、统一统计和索引引用、语义化 episode 查询 | 数据集资源与既有列表/详情入口 |
| `pipelines.py`、`scheduler/loop.py` | 节点依赖/output_ref、有限 shard、状态租约、资源准入 | 单机队列；不引入外部工作流引擎 |
| `store/models.py`、migrations | 版本/索引/执行关系增量字段与最少新表 | DatasetVersion、Task、ModelVersion、EvalSuite、Bundle |
| `rust/crates/eb-runner` | 生命周期、资源/网络限制、重认领、写根与挂载验证 | 只有 ebd 持 docker.sock |
| `eb-scanner/eb-artifact/eb-collector/eb-tsdb` | indexed manifests、Range、质量摘要、流式认领、降采样 | 现有 crate 和单个 ebd 服务 |
| `frontend/src/views` | 版本视图、Episode 时间轴、Collection、能力状态、评估和交付证据 | Vue/router/ECharts/3D 已有技术栈 |
| 平台管理的工具镜像与 launcher | LeIsaac/RLinf/G1 适配、验证/导出/目标测试配方 | 外部源码只读，不在原仓打补丁 |

领域只放纯语义校验与类型；不要把 `Path.iterdir/stat` 等 I/O 放进 domain。格式库的 Python 解析可留在工具容器，不必把 HDF5/LeRobot 全部重写为 Rust。

## 4. 数据模型最小增量

首期必要新增：`collection_sessions`（配方/预算/状态/管线引用）、`episode_index`（可筛选摘要）、`pipeline_runs`（冻结输入的一次管线执行）、`pipeline_steps`（运行节点→Job、输出、幂等键）。复用 Pipeline 作为模板目录；本次已明确需要重复补采和稳定运行 URL，因此运行实例单独落表，不覆盖模板 spec。

沿用 DatasetVersion 存 published 状态、manifest/quality/index refs、source version、schema、storage kind；DataSlice/AnnotationRevision/Scenario 先作为不可变 manifest 与 Artifact 记录，达到高频跨版本查询需求才各建表。逐帧 rows 和大视频不进 PG。

关键约束：episode_id 在 version 内唯一；attempt_key 在 session 内唯一；pipeline_run/node/shard→Job 唯一；产物 `(run_id,logical_name,content_hash)` 幂等；published payload/manifest 不可修改；列出资产引用的 GC 必须与读租约一致。

模型 capability 目录首先由平台自带受校验配置+证据对象实现，不搭建任意代码插件市场。表可通过 JSONB 容纳版本化 profile，外键必须引用已登记对象，不能只保存显示名称。

## 5. 兼容迁移顺序

1. **盘点和备份**：PG、现有 CAS/数据目录/manifest；记录无归属输出和缺文件引用。建立不可变源备份，验证可恢复。
2. **增量 schema**：加字段/关联，新旧读路径都能处理 null。历史没有实测 hash 的资产标 legacy_unverified，不伪造完整 provenance。
3. **别名映射**：`ec_*`→`ee_*` 仅在接口语义与 hash 等价时映射，原 ID 可继续查询；不直接字符串替换已关联的历史数据。差异需要独立版本和人工确认。
4. **执行真源**：TrainJob 只投影 Job/Run，旧详情路由重定向到可解析执行记录；新 Job 不双写两套可分叉状态。旧 Run 路径按 storage_ref 查，不能强搬目录改成新布局。
5. **模型收敛**：WeightModel 的 path 导入后生成候选 ModelVersion；旧 mdl/mv ID 建映射。相同权重但不同 processor/norm/interface 不能去重为一个部署模型。旧 ONNX/CAS 模型保留可用路径。
6. **数据索引回填**：异步逐版本、限速重建。9 个 ego_jsonl 文件只显示文件统计；没有 reader 验证的 episode_count 回填为 unknown。
7. **新写入口切换**：通过契约验收后将新作业统一到新的 inputs/outputs/发布规则；历史作业只读兼容。先单模块切流，不删旧表或历史真源。
8. **回退方案**：新写切换可关闭；新列/表保留，旧读接口继续服务；不做导致新版本元数据丢失的破坏性 down migration。回退只恢复服务，不撤销已产生的资产事实。

## 6. 验收矩阵

| 编号 | 验收场景 | 必须得到的结果 |
|---|---|---|
| A01 | 三机器人不同 action schema | 明确关节/单位/频率/相机；错配阻断，不隐式补零 |
| A02 | 表单选择数据/基座 | JobSpec、实际挂载、loader 回执、模型血缘一致 |
| A03 | 重复提交/完成回调/服务重启 | 只有一个逻辑 Job、一个已提交产物；可重认领 Run |
| D01 | 下载中断、ETag 变化、错误 HF 类型 | 恢复或明确失败，无污染的发布版本 |
| D02 | 多频传感器、未来帧、时钟回退 | 分段报错、age 可查，不产生未来观测训练样本 |
| D03 | LeRobot v3 共享视频与短 episode | 依据 metadata 定位；裁剪和错位均可检测 |
| D04 | Mimic 祖先跨 split、acceptance 入 calibration | 阻断发布；报告交集 ID |
| D05 | 100次尝试多个拒绝标签 | 互斥主结果之和=100；标签不重复扣减 |
| C01 | kill 采集 writer、漏 shard、取消后重试 | 已完成 episode 保留；partial 不发布；重试不重计覆盖 |
| C02 | 最大尝试/时长/磁盘预算到达 | 有限停止并保留原始结果，目标不足不假成功 |
| T01 | 非 OpenPI recipe 进入当前 SFT launcher | 提前检测缺配置/不兼容；不能直接宣称支持 |
| T02 | 两个2卡任务并发 | GPU/网络/CPU/RAM/shm/输出隔离，Ray 不串集群 |
| T03 | checkpoint 半写、分片缺失、latest/best不同 | 只登记完整版本；选择语义准确 |
| E01 | 测试20/100只跑完部分 | incomplete，不能用剩余成功样本通过 |
| E02 | 不同 suite/接口的两个模型 | 禁止直接混榜，显示条件差异 |
| Q01 | 4090 engine、错 JetPack、缺插件导入 Orin | 目标验证拒绝，保留原因 |
| Q02 | INT8更快但任务退化/超deadline | variant 不通过，不覆盖原版 |
| R01 | G1 仿真点击交付 | 真实阶段 unavailable；无伪造设备/现场结果 |
| R02 | 平台/网络不可用、端侧异常 | 端侧控制与安全机制独立；平台没有放行后门 |
| S01 | source 写入、路径逃逸、恶意 override/URL | 拒绝；外部目录无写入，凭据不泄漏 |
| S02 | 清理活跃版本、恢复备份 | 活跃引用保护；恢复后媒体/权重/血缘可定位 |

执行验收使用现有容器测试环境与 ebd 托管的最小 smoke Job；不在宿主安装并运行训练栈。单元检查只覆盖纯逻辑，不能代替 GPU/数据/目标运行证据。

## 7. 仍需接入时确认的参数

G1 EDU 仿真资产及手部、BHL 实际部署形态/固件/标定、SO101 相机和 leader 可用性、Orin JetPack/L4T/TensorRT 与功耗模式、数据盘容量/挂载点、各任务成功判据与现场端控制器。它们影响具体 profile 的可运行状态，不影响本轮通用对象和管线设计完整性；必要信息缺失时只阻断相关设备/任务节点。
