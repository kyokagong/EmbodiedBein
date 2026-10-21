# UFO 原生训练接入执行记录

设计真源：[设计 25](../design/25-ufo-native-training-integration.md)。本记录描述实际进度，不更改设计。用户要求不提交 Git；外部 UFO/RLinf 源码只读。

当前状态（14:10 UTC）：接入已实现并部署，主v2完成112→512→1008→5008→10000及五段双后端validation（30例全部fall）；独立全量Replay推断对照也完成（12例全部fall）。任务质量仍不合格，10000原协议未达继续门槛，停止自动追加20000，保留5008诊断候选包和10000最新完整状态，等待下一段策略确认；尚未跑满用户授权的6小时上限（截止16:55:31 UTC）。v3正向全链、真实OOM/NCCL/更新后取消及回收通过。正式acceptance未冻结也未执行。

## 冻结的执行约束

- RLinf 基础 Worker 每卡持有完整 UFO Trainer，FB-CPR-Aux/mjlab/RPO 23DOF；不重写算法或嵌套 launcher。
- 用户授权首轮训练累计最多 6 小时，先看效果；112 → 512（必要时 1008）→ 5008 累计 agent 更新，阶段推进必须有独立快评和决策。
- 站立、0.3 m/s validation；0.7 m/s 诊断；正式 acceptance 阈值尚未确认，不能宣布模型正式合格。
- 分段结束、全 rank Replay 保存、冻结、独立评估、新 Run 恢复；恢复模式 ufo_native_state，环境/RNG 等重新初始化。

## 实施与验证清单

- [x] A0：锁定最小 RLinf scheduler + UFO 组合依赖、双源码/适配身份、Ray 隔离、设备/EGL/collective/失败退出。
- [x] A：冻结原生机器人、资产、数据来源/质量/split；更新数与环境步映射、有界资源和恢复指纹。
- [x] B：专属 native_rl 模板、预览/提交、Job/Run 分发、Profile/能力、adapter、CSV 指标收集、checkpoint/receipt/血缘、工作台入口。
- [x] C：112 次真实更新，完整 Replay/native loader 回读、新 Run 恢复、独立快评与真实媒体。
- [x] D：512/1008 趋势和区间决策；任务指标支持进入5008探索，正式质量仍未合格。
- [x] E：6小时上限内已分段训练到10000、保全候选/最新并产出效果报告；按未达继续门槛停止自动追加，未跑满6小时。正式acceptance仍待确认。
- [x] 复核一：设计、本体动作语义、数据隔离与来源；official祖先/许可未知已单列。
- [x] 复核二：GPU/计数/恢复、真实OOM/NCCL/更新后取消与预算；自动Job总超时仍以行为测试为主，未宣称该独立实测已覆盖。
- [x] 复核三：五段原始曲线、逐case、30个完整失败视频及实际HTTP/浏览器入口；后续阶段仍逐次核对。

## 初始检查

- 工作区只有用户提供的设计 25 未跟踪；未改动或暂存它。
- 2026-10-18 检查四张 RTX 4090，均约 1 MiB/24564 MiB、利用率 0%；平台无动态任务容器。
- 平台当前仅 UFO import/export/eval/bundle/live；训练需完整接线。
- 当前 API 数据集中没有 UFO 原生数据登记，正在定位本地导入产物；不能因此断言用户的数据不存在。
- 默认每 rank Replay 5,120,000 transitions 的观测下界约 28 GiB，不能用于 24 GiB 卡；容量需要冻结并实测。

## 实施中的复核记录（持续补充）

1. 数据/本体：定位 v4 数据及原始 v2 导入审计。原生 CPU loader 实测 58 train clips、10,255 frames、165.988333 s；15 validation clips、3,018 frames、60.06 s。`walk2` 已进训练，使用剩余 holdout。official 祖先与许可未独立核实，不能宣称严格泛化验收。冻结 Job 单独输出 train 和 holdout，训练只挂 train.pkl 与 manifest。
2. 计数/恢复：原生 E=1024/G=16 的冷启动 112 次更新终点是 18,432 local transitions；从此 checkpoint 恢复至 512 次更新，checker 重新起计，终点为 45,056 local transitions。恢复标记 ufo_native_state，环境/RNG/latent 缓存等非精确恢复。全 rank Replay/半写/hash/source/world-size 检查已实现并测试。
3. 运行时/依赖：最小 scheduler profile 解析并导入成功，保留基线 159 项依赖、增加 35 项。Torch 声明 NCCL 2.26.2 与 UFO 锁定的 2.26.5 有冲突；以受管且留 hash 的打包元数据补丁保留 2.26.5，未修改 Torch 算法或 UFO/RLinf 源码。完整解析锁保存于 deploy/images/ufo-native-train-resolved.lock。
4. 实际 A0 失败发现：Ray 2.47 reporter 在 network=none 下解析容器随机主机名失败。已验证 network=none + hostname=localhost；显式向 ebd 传递有界主机名，保留网络隔离。单卡及四卡 reset/step/64×64 真实渲染通过；四卡唯一 GPU UUID/EGL 映射、NCCL all-reduce 通过。两 rank 注入 rank 0 失败后整组及独占 Ray 在 25 秒内退出，无完整成功 receipt。
5. 实际数据 Job `job_9c14a2a3672f49ae854d` / `run_b42c945834154e1fbeb4` 正确失败且未发布数据：校验器以容器别名 motion-source 拼 qualification 的原始成员路径。改为使用冻结 source_uri 的原始版本目录名，保留失败记录，下一次在新 Run 验证。
6. 旧能力复核：更新快照排除策略后，旧受管 snapshot 必须继续按已冻结 manifest 验证；已补旧策略兼容测试，不强制旧模型/训练重建来源。

7. 提交复核：真实 API 预览的 hash 缺少请求 schema 所需 sha256: 前缀；已修正并补预览到提交的 schema 往返检查。原生首次提交及恢复使用数据库 advisory lock 防止相同幂等键并发创建多个 TrainJob。训练只挂载 train.pkl/manifest 与机器人 mesh 子目录，不挂 validation/test 或整个资产根。

最近聚焦后端 139 项通过、1 项 Torch 检查在 backend 镜像跳过；目标任务镜像 40 项通过。Rust runner/collector/ebd 共 53 项通过，前端 Node 测试、vue-tsc 和 Vite 构建通过。make up 已发布服务。此结果不代表真实训练或模型质量通过。

## 冻结身份与实际 Job

- 任务镜像 `eb/ufo-native:rpo-train-v1`：`sha256:afd9e17227d89d8f02bc95e869b60509c2bcf7d8756cbd3ffd8f914727566969`；依赖、元数据补丁及构建回执在 `deploy/images/ufo-native-train-*`。
- UFO snapshot `sha256:1d3457979dd7ade69cd2e82a0173144a9b60bb60ba91a13fcc4eb5822c86e99e`；RLinf snapshot `sha256:3cf50ddf948ab993361316647b338ba991e404c5c0c72e1c591782730d7ef676`；assets snapshot `sha256:741e9df140d0a662afc099a03e21defc20980707ffb7db2ac897cf35a491f119`。
- 数据集 `ds_6541f726d7804007a93a`，成功冻结 Job `job_bb639ec0c5754d0eaa7b` / Run `run_62698e4956174db1b697`；train manifest SHA256 `242fc13c3dd2f24b8d6aafcb688c35e010745eaba27efbe68f195c2185843f1b`。独立 holdout 位于该 Run 的 `outputs/holdout`，未挂进训练。
- 首段 TrainJob `trn_9e9420c6765f4ee0a0d9` / Job `job_2efc8d4646544df686ad` / Run `run_98db9ba7512d4b62ae30`，2026-10-18 10:55:31 UTC 开始；本轮预算最晚到 16:55:31 UTC（包含训练、保存、派生与评估）。
- 首段目标 112 agent updates；4×1024 environments、每 rank Replay 容量 262144、seed 42；预计段末 local/global transitions 为 18432/73728，实际计数以 receipt 为准。准入 4 GPU、32 CPU、80 GiB RAM、8 GiB shm。
- A0 原始证据：`/tmp/eb-ufo-probe-1-none-localhost`、`/tmp/eb-ufo-probe-4-none-localhost`、`/tmp/eb-ufo-probe-2-none-localhost-fail0`、`/tmp/eb-ufo-frozen-input-probe`；均不构成 agent 更新或模型质量证据。

## 112 更新段结果与决策

- 首段成功，实际容器运行约 291 秒，退出码 0、无 OOM；四 rank 均为累计 112 updates、18432 local / 73728 global transitions。
- 四 rank trainable actor/forward/backward 初始 hash `sha256:ad7d47febc9e231f2f28e66e9787c473ff13bd70e3ec0d7cd8053c4fc9f1d2a2`，最终 hash `sha256:ccbe6d420f9cebb7d0bad53daeb4b9b36c1b654100592d2df649c175b5c65edb`；各 rank 最终一致，固定样本 delta L2=0.4157128、maxAbs=0.03037867。该样本 norm 不是全部参数的全量 norm。
- all-rank Replay 同 snapshot、world=4、global=73728、transition_format=2；每 shard 18×1024 已采集 transitions，capacity 为 256×1024。原生 agent/优化器与各 rank Replay loader 回读通过；`.saving` 不存在。
- 原生 checkpoint manifest SHA256 `d13db62d4bfccfa57fc1071c1cdb5d95572d9edad3a2fd16de47611aa38e54a2`；模型 `mv_b8c497ffcda54447bed2`、推理权重 Artifact `art_063658517a6445e1b33f`、恢复 Artifact `art_5523cdf5cabb43bb838e`。模型 qualification=incomplete，尚无任务能力结论。
- 原始 CSV、各 loss 曲线、资源图与 analysis/summary.json 已生成，路径位于 Run 的 `outputs/analysis`；每卡显存峰值 13222–13226 MiB，容器内存峰值 35297 MiB。资源 CSV 与 TSDB 逐 tag/Unix 秒核对一致。
- DatasetVersion `dsv_304fffb96b30417db5d8/native-v1` 已核对内容 hash 与 producer lineage。
- **区间决策：等待独立 validation，暂不进入 512。** 真实 prompt 预览成功，提交暴露容量估算误入通用 GPU 优化器分支，已补 CPU operation 回归。第三轮复核发现原生 RewardInference staging 缺 checkpoint/config.json，以及训练分析/prompt 下载路由索引缺口；先修控制面后继续派生，不改已冻结训练镜像或源 checkpoint。
- 单卡训练协议验证 TrainJob `trn_0f9aa5b94e43483190f2` / Job `job_967c3c83e74d4147a939` 已提交，同镜像/数据/seed、112 次实际更新目标；该独立拓扑验证不替代四卡主实验或改变主实验 fingerprint。
- 原生入口/薄 Worker 配置一致性：冻结镜像内捕获真实 run_train 的 builder 调用，4 rank 各 27 个参数、完整 cfg 各 523 个叶值，与薄 Worker 及实际训练保存配置均零值差异。robot_config 的 Path→str 是序列化差异；报告 `/tmp/eb-ufo-native-config-equivalence.json`。该检查不构造 Trainer，不等于独立原生入口更新的数值对照，后者尚未执行。

### 首次独立评估前冻结的区间判据

- 周期 validation 使用现有双后端 smoke-v2 Suite 的物理案例与原生 provisional 成功函数，所有输出保留 diagnostic 资格；这不是最终 acceptance。站立 seed71000、0.3 m/s seed71001 为主任务；0.7 m/s seed71002 单列诊断，不混入主任务趋势。
- latent 派生固定 rank0、seed71000、16384 samples、n_inferences=1；各 checkpoint 分别推导并绑定新模型，不能复用前段 z。
- 112 档权重/完整保存/loader、独立派生/ONNX 数值验证及所有逐 case 媒体完整后，可进入 512；不要求 112 档站稳。
- 512 与 112 在相同后端/case/seed 下对比主任务平均存活步数、速度误差、姿态/终止原因与限位异常。混合或微弱趋势时到 1008 再决策；若出现非有限、codec/导出不一致或新基础设施错误，先停止调整。
- 进入 5008 前要求两后端主任务平均存活步数较 112 至少提升 10%，单个主任务不退化超过 20%，并且速度误差/姿态/限位没有抵消该改善的明显退化；不足则停止调整，不用 loss 下降代替任务改善。阈值仅是本轮继续训练的工程判据，小样本不提供正式置信结论。
- 每次保存/派生/媒体与时序分析消耗均计入 6 小时；后续段按实际吞吐预留保存和评估时间，最晚 16:55:31 UTC 停止本轮新计算并保全已有产物。

## 第三轮复核与派生闭环

- 单卡协议 Run `run_f07cabbba4494a7f8db5` 成功，112 updates、18432 local/global transitions；模型 `mv_5224e43abd1b4cca8e1f`。不以单卡和四卡不同有效 batch 的结果宣称数值等价。
- 修复完成的输出路由已经实际 HTTP 验证：主四卡 Run 的 receipt、metrics-index、training-1.png、两个原始 native CSV 均返回200，内容 hash 与索引一致；索引原始 native CSV 的 655 tags / 1105 TSDB points 已逐坐标和值全量对账。UI新增 evidence 入口；终态计数优先完整 checkpoint 的 summary，避免把最后一个采样日志的 pre-transition 步数当最终计数。
- 原生 prompt Job `job_0491798c1e1f4246a002` / Run `run_7881cc4401b5460e8977` 成功：原生 rank0 Replay 实际18432 transitions，分别推导三任务新 latent；派生模型 `mv_cce6b1588c344672ae16`、prompt Artifact `art_ab5f8a4b7e1845398340`，seed71000、每任务16384 samples。
- ONNX 导出 Job `job_ecda698f4d7e48388967` / Run `run_b8834e1bcc7f40948f47` 成功，模型 `mv_4db0b4e129454199aa00`、包65703359 bytes；96组数值比较最大动作误差4.934845492243767e-6、target误差1.5690922737121582e-5 rad，满足冻结 atol1e-5/rtol1e-4比较规则。
- 旧镜像112档独立快评已完成：mjlab `job_e854e79b99ea4f8682f7`，mujoco `job_3d79255eadae44dea4e8`；所有案例及录像结果见下表。
- 复核发现原生恢复缺少预览 hash 绑定。已补完整 JobSpec（含名称）的 sha256 预览/提交校验，以及前端输入修改与异步旧响应失效；旧栈保持兼容。相关后端55项、前端17项检查通过。
- 复核用坏 critic/normalizer 复现了完整状态有限性缺口：旧 runtime 的 actor/FB 参数变化和 loader 成功不能证明全部模型/优化器有限。正在补原生 full-state 检查。将保留 `rpo-train-v1` 和旧产物，派生 `rpo-train-v2` 并从头进行新版主实验112档；不跨镜像改变恢复指纹。旧档结果用于接线验证，不能补造其历史回执验证字段。
- prompt 控制面改为按来源 checkpoint 的冻结 image digest 派生，避免新镜像发布后破坏旧的不可变版本；source/image/Replay绑定校验仍保留。

### 旧镜像112档物理结果（完整失败证据）

| 后端 / Run | 站立存活步数 | 0.3 m/s存活步数 | 0.7 m/s诊断存活步数 | 结果 |
|---|---:|---:|---:|---|
| mjlab / run_42928c1c29af4eabb8ee | 11 | 41 | 5 | 三例均fall |
| mujoco / run_b1a894bda2d1448e87fb | 10 | 38 | 5 | 三例均fall |

每步0.02秒；主任务存活不足1秒且有大量关节/速度越限，不能宣称站立或行走能力。两个后端属于同一MuJoCo物理家族。6个案例均planned/completed/valid=3/3/3（每后端），没有成功案例，不造成功录像。6段失败视频全部可解码，实际HTTP GET后大小/SHA256与receipt逐项一致；图表/逐case索引HEAD200，视频Range206通过。

A0与配置对照回执已保存到仓库 `docs/progress/evidence/2026-10-18-ufo-native/`，避免只依赖临时目录；原始探测目录保留。

## v2 主实验（当前）

- 冻结镜像 `eb/ufo-native:rpo-train-v2` / `sha256:e122bb0c96f28c3d2006dd0c466bf392f2f4dc8f0e09fba0e5f2e36f6cefdfe6`；194项依赖、算法默认、源码和训练数据保持前述身份。镜像构建回执已更新，v1回执单独存档在 evidence 目录。
- 新运行训练返回后及原生CPU agent loader回读后均调用原生全模型与optimizer_dict有限性检查；所有rank验证成功才生成完整receipt。新plan明确 `full_state_finite_required=true`，旧manifest缺字段只读为未验证，恢复入口禁止把旧验证标记补成新验证。
- 新镜像69项目标运行时/有限性/checkpoint检查通过；147项聚焦后端检查通过，另新增恢复候选行为10项通过。两个仅平台Python3.11的checkpoint/collector检查在目标Python3.10镜像排除，已在平台镜像执行通过；不修改领域代码使其迁就目标镜像。
- 旧四卡112 checkpoint另经只读完整回读诊断：378个named parameters、15个buffers、5套optimizers全部有限；该诊断没有修改旧receipt或模型，也不作为跨镜像恢复豁免。
- 新主训练 `trn_32f580698ddc42a3a509` / `job_9d046c6692df4b96862a` / `run_648c7193afd64a188100`，11:22:14 UTC开始；fingerprint `sha256:d2cd2c2750a8af7fe3118bb97b885cf0529da78c881012102c0d608bfc203f48`，目标仍112次更新。
- 原生批次对齐的预算输入能力支持至1000000累计updates，便于更大明确预算沿用同镜像恢复；这不改变当前112→512→趋势/5008的执行计划，扩大实际区间仍须先写出任务指标依据、目标与剩余墙钟预算。
- 实际HTTP复核：旧v1恢复候选返回available=false并说明缺少已记录的完整状态验证；旧训练证据GET正常返回9个可发现链接。正式acceptance仍未执行。

### v2 112段与512区间决策

- v2 112段成功，四rank全状态有限性/原生agent及Replay loader均通过，25个文件完整hash通过；checkpoint1823425411 bytes。模型 `mv_816413f83bc344fe9beb`，恢复Artifact `art_e699725c821b49e18375`。
- Prompt `job_f988fca8598a41f18670` / `run_ca9206a7ecdd42fdba9e` → `mv_c1c1924bf52c4986b0fc`；导出 `job_c34a3d357b9d4f848826` / `run_18f34748a3c7417c924a` → `mv_972bce8b8f4742838021`。96组ONNX比较通过，最大action error7.599592208862305e-6。
- mjlab `job_ef3a302fd24e4aaaa380` / `run_13ba7bb8b1364b43b04b` 三案例存活13/10/6步；mujoco `job_b6996b73a0bd454c8e1d` / `run_68006a803c2e465fb5a0` 存活12/9/5步；均fall。两后端6段完整失败视频已HTTP GET/SHA/大小校验。没有成功案例。
- **11:33 UTC区间决策：继续到累计512作趋势验证。** 112档满足真实更新、完整保存/loader与独立快评/媒体门槛，质量仍差。本轮保留原58clips输入、四卡拓扑、image e122...及fingerprint d2cd2c...；恢复本段新增400updates，累计local/global预期45056/180224，目标/日志/保存边界冻结。新Run只读来源112 checkpoint，不能恢复RNG/环境，不能称exact_resume。
- 已实际预览512恢复，hash `sha256:f2c26bc955724782301aedab05ec412cd87ba43bb9f7194cd8ad0891df7efeaf`，镜像/源码/数据/资源与来源一致、resume-native挂载只读；提交必须携此hash。完成后固定同一独立Suite/seed重新派生新z/导出/评估，再按前述冻结判据进入1008或5008；不以loss下降决定长训。
- 512恢复已实际提交：`trn_e4356e3475b141a2b2d6` / `job_9db0b9f0e3c746638dde` / `run_fdb3e7bc6e444148848b`，11:34:16 UTC开始。来源25文件hash重新核对通过，新Run模型文件是独立inode、resume-native为只读；实际原生loader/训练终点仍待运行结果。
- 512恢复成功，11:40:03 UTC容器正常退出；新增400updates，累计local45056/global180224，四rank最终hash一致且从源终值真实变化。五套Adam的全部state.step实际112→512（条目44/6/76/80/10不变），不是只延续外部计数；四Replay cursor18→44，完整finite/agent/Replay loader与新旧各25文件hash通过，源manifest/receipt前后未变。模型 `mv_906d215bc2f14a9da8b8`，恢复Artifact `art_59aec597693c4882b7bd`，checkpoint2621625915 bytes。
- 恢复段CSV/TSDB有25个真实更新点128,144,...512；fb_loss从48439.8125到8230.9765625，但训练内反馈/环境重置会改变指标分布，不能把此loss下降当独立任务趋势。每卡显存峰14496–14548 MiB，容器内存峰36403.2 MiB，资源采样status全ok。
- 512 prompt `job_4246abeb0ea04fb3af47` / `run_6858079053e4423bb920` → `mv_c5f1d8714edd4673abef`；export `job_3552430cefac4d4b80a5` 已提交。区间结论仍为等待独立validation，不提前进入5008。

### v2 512快评与1008区间决策

- 导出 `job_3552430cefac4d4b80a5` / `run_592c266c354c481dbb01` → `mv_b733e37c54dc45b8a9a9`，96组数值验证通过。
- mjlab `job_52461d018cad4a439816` / `run_2e4167bbe8d743d48d68`：站立/0.3/0.7存活31/24/27步；mujoco `job_c21d43fcc22246708519` / `run_d569a7aae0524528bdff`：32/24/29步。六例全部fall，六段可解码失败录像HTTP GET后hash/大小逐项通过。
- 主任务存活较112改善，但站立速度RMSE在mjlab从0.462→0.832 m/s、mujoco从0.352→0.853，drift从约0.03→0.33m，joint violation fraction仍约0.97；0.3速度RMSE约0.81仍差。0.7的明显改善仅属诊断，不能掩盖主任务缺口。
- **11:46 UTC区间决策：混合趋势，继续到1008作复测，不进5008。** 参数/镜像/数据/拓扑不变，来源512，新增496updates，预期累计local77824/global311296。沿用同Suite/seed及独立新latent，结束后复核速度/姿态/越限是否抵消存活改善；完整状态/媒体或预算不满足则停止调整。当前主候选仅用于诊断，未达到站立/前行质量。
- 1008段 `trn_a35ba5f4ed704f778f29` / `job_29da8efc180a48478712` / `run_e22e731b0acf4904b9d1` 成功，11:47:09→11:53:19 UTC，exit0/无OOM；累计local77824/global311296/updates1008，新增32768 local/496updates。四rank五套Adam的全部state.step实际512→1008，Replay cursor44→76；新旧各25文件hash通过，源manifest/receipt不变。四rank权重809459da...→d4dd03c0...且最终同步maxdiff0，完整finite与原生agent/Replay loader均true，无.saving。模型 `mv_0d061e47bcc24b34aa45`，恢复Artifact `art_e75119fd7b624df8a10a`，checkpoint3604015001 bytes。
- 1008段CSV/TSDB同为31点528..1008；资源56个采样点全ok，各GPU显存峰13344–13346MiB、容器RAM峰35440.64MiB。独立prompt `job_a11450e09e8f41f199f2` / `run_a09a47c0550840428fa5` 正在进行，仍等待任务快评后决定5008。

### v2 1008快评与5008进入决策

- Prompt派生 `mv_1e45e4a65f6d47a1a6dd`；导出 `job_4b1511197afc45dabbfd` / `run_e3c8ec7a59d44a90b77e` → `mv_5a5b329aa90847c7abd8`，96样本数值比较通过，最大action误差7.659196853637695e-6。
- mjlab `job_b790a3a4a07b4d8b9be4` / `run_11c9971413554deb863c`：站立/0.3/0.7存活29/38/32步；mujoco `job_a71b6f16821b402794d0` / `run_daa5901024c84e3aa3e7`：32/38/34步。全部fall，六个完整失败视频HTTP GET大小/hash通过；无成功录像可提供。
- 主任务平均存活为33.5/35步，分别为112档11.5/10.5步的2.91/3.33倍，且每个主任务均改善。两任务速度RMSE均值mjlab约0.615→0.619m/s（+0.7%）、mujoco约0.572→0.658（+15%）；站立单项0.462→0.548、0.352→0.598仍退化，但较512的0.832/0.853回落，0.3前行则较112改善。站立drift仍约0.216/0.270m，不能称已稳定站立。
- 主任务max joint excess均值较112分别下降约44%/34%，velocity violation fraction均值下降约24%/26%；max tilt均值约-5%/+2%。少数速度峰值及前行越限占比仍退化，全部均未达到任务合格条件。原始逐case结果保留，未只选择有利指标。
- **12:01 UTC区间决策：可进入5008受控探索。** 按已冻结工程判据，存活改善明显、无单个主任务存活退化，且平均速度/姿态和越限幅度的总体变化没有抵消该改善；站立误差和大量越限仍是主要风险，不作质量验收通过。先补独立原生入口实际112更新对照，再以同e122/58clips/四卡配置从1008恢复，新增4000updates，预计累计local334848/global1339392。按当前实测吞吐预计约50分钟训练，另预留保存/派生/快评时间；最晚预算仍16:55:31 UTC。
- 实际HTTP Chromium检查通过6项：1008终态三计数、9个证据链接/所有图表解码、原生恢复预览、修改目标立即失效、四卡新建预览；23个API响应无错误、0个JS异常，没有提交训练。报告与截图在 [browser/browser.json](evidence/2026-10-18-ufo-native/browser/browser.json)。浏览器容器缺CJK字体，截图中文未完整绘制；DOM文字与功能断言通过。
- 三段双后端原始receipt及18个实际HTTP视频hash校验保存在 [validation-trend.json](evidence/2026-10-18-ufo-native/validation-trend.json)，主任务趋势图 [validation-trend.png](evidence/2026-10-18-ufo-native/validation-trend.png) / [SVG](evidence/2026-10-18-ufo-native/validation-trend.svg)。0.7仅在原始结果保留，不混入主任务图；全部18例均fall。

### 实际取消与资源回收负面验证

独立单卡112目标探针 `trn_8063234417834ea49fc8` / `job_b6ca36c56faf4f9d9494` / `run_61e02e0dd37a4848958d` 在WorkerGroup就绪、尚未发生更新时，经平台API取消。11:44:56.289 UTC请求至返回cancelled约1.008秒；TrainJob/Job为cancelled，Run为aborted/cancelled。该Run容器被删除，记录的32个宿主PID和独占Ray均消失，GPU1从1388MiB回到1MiB；没有checkpoint、成功receipt、模型或恢复候选。主512段仍succeeded，未取消其他作业。报告与关键断言已保存在 [cancellation/REPORT.md](evidence/2026-10-18-ufo-native/cancellation/REPORT.md)。这验证取消清理路径，不构成更新或模型质量证据；实际OOM和collective超时尚未单独注入。

### 生命周期复核修复与版本边界

- 取消回执统一要求HTTP成功且JSON object的`ok is True`，否则API/scheduler保留未确认取消状态与证据。实际Rust ebd失败已返回503，但此前控制面忽略HTTP200失败body的边界已由测试复现并修复。
- 发现v2 entry的collector.stop抛异常会跳过Ray/shutdown及信号恢复，并可能在磁盘留下先前写出的success receipt。新entry保证各清理步骤均尝试、保留原异常，清理失败写独立诊断并删除success/probe receipt；成功路径清理失败同样令Run失败。实际signals、CPU c10d/Gloo缺peer超时、各阶段/清理异常和发布边界已执行行为测试，不将它们称作真实NCCL或GPU OOM。
- 修复构建为新fresh镜像 `eb/ufo-native:rpo-train-v3` / `sha256:7a0babe5e5b8cff1b63bedee6c1746cb4710710a4305ec93667d8a52df58cb3e`。依赖/原生算法未改，镜像内源码hash与仓库逐项核对；目标Py3.10环境85项通过、两个仅平台Py3.11的collector/public-checkpoint检查排除后在平台测试覆盖。v2/e122、已有产物与恢复指纹未变；不跨镜像恢复，不补造历史验证。
- 当前v2若出现cleanup异常则停止该链，不能自动把磁盘receipt当已发布模型；外层容器退出回收进程namespace，阻塞仍依赖Job硬超时。主实验目前三个v2段均exit0且未出现此异常。v3 fresh真实训练尚未执行。
- 原生恢复新增操作者可选`timeout_seconds`，严格60..21600整数，只能缩短源plan与policy上限，两处同步；identity/算法参数不变。预览显式列出墙钟并用全spec hash绑定，预算改变必须重新预览。74项恢复/service检查通过。后续恢复将按16:55:31 UTC全局截止冻结剩余秒数，避免每次恢复重新获得原6小时。
- 新接口通过`make up`部署；独立原生入口对照镜像 `eb/ufo-native:rpo-baseline-v1` / `sha256:3cedc8a3e770310f85e2257fc4e7a3f54eeaeaaa46efd0081381298f513ace7c`，固定继承e122，源spec原样保留。实际diagnostic Job `job_ac44547d784248d1a048` / `run_5a72bfdb7566420cbf72` 进行中，4GPU112更新，不发布TrainJob/Model；比较规则在 [baseline-protocol.md](evidence/2026-10-18-ufo-native/baseline-protocol.md) 及预览hash冻结。

### 原生入口对照完成与5008提交

- 原生入口Job成功完成112实际更新，local18432/global73728，Artifact `art_901b655637f94360b4cb`。四rank实际cfg叶值与Worker相同（仅声明的env.device映射差异），实际初始actor/FB完整hash均为ad7d47fe...；共享终值bf684110...，与Worker的1c74bae2...不同，逐列差异保留，不宣称GPU数值等价。更新/文件布局、共享checkpoint原生agent/optimizer loader与五套Adam内部step112、四rank Replay loader及模型同步均通过。
- **对照范围纠正：** 原baseline producer误将共享checkpoint检查填写为每rank full-state verified。原始输出与hash保持不变，这些逐rank优化器声明不被认可：只观察了共享rank0模型/optimizer检查和四rank模型同步，未观察rank1–3实时optimizer有限性。正在补不可变scope附录与后续专用diagnostic manifest；该baseline不获得主训练checkpoint恢复资格。主实验Worker逐rank真实检查不受此诊断元数据问题影响。
- 按已冻结1008快评结论提交5008段：`trn_fcc5dbcbc67843778bf0` / `job_9840fddc7c644cffb447`，来源1008，原e122镜像/58clips/四卡身份；预览hash `sha256:7ce902205313e809913dae41b1686ca3c466f106ac79244b9dcff527ddf29ccc`，本段hard timeout5400秒（90分钟），全局截止仍16:55:31 UTC。预计local334848/global1339392，结束后重新派生新z、导出并跑同Suite双后端快评后再决策。

### 独立数据候选准备（未加入当前实验）

数据复核确认A1原457帧/3.8s纯站立全因floor穿透被历史导入过滤。实际RPO FK脚踝原点约贴地，但脚底碰撞mesh再向下约5cm；抽样源key bodies与目标FK最大差0.995mm，暂无关节重排错误证据。内存反事实统一root_z+49.176mm后最大穿透约2.281mm，而q/xy/quat/fps/帧数不变；这只是几何可行性，不是动态合格。

用户已授权补参考动作，因此准备独立受管data.convert候选：显式记录高度变换、直接复用原生MotionData build、逐帧质量检查、自由基座PD真实mj_step回放与全程媒体；只在t0设root，后续不得强制root跟轨。若PD仍跌倒/越限，保留失败、标not_training_ready，不为凑59片段发布。候选通过后才会形成新DatasetVersion；现58训练/15holdout、测试隔离及当前恢复链不变。

候选实际Job `job_0536b59329db4d5ca2df` / `run_451320744e234c25b82d` 已完整完成，但候选 `ds_fd49187d32234e299a44` 被拒绝：首次fall1.54秒，最低高度0.055675m、最大tilt1.616345rad、drift0.888080m、一次子步velocity excess1.381897rad/s。几何穿透2.281mm通过没有替代动态判据；完整190控制步/1520子步/191状态、96帧录像与457帧原生loader均核对。Dataset invalid/training_ready=false/versions=[]，没有生成59训练数据，父58/15文件hash未变。报告：[a1-candidate-run/REPORT.md](evidence/2026-10-18-ufo-native/a1-candidate-run/REPORT.md)。本地尚未证实原始AMASS/SMPL与完整mocap→RPO IK链路就绪；不凭参考失败推导当前58片段绝对不足。

独立候选 operation/API/CPU容量/adapter/collector/版本血缘与只读媒体接线已实现，聚焦容器测试86项通过。冻结e122中实际原生配置、物理/renderer和逐关节PD/DC参数映射通过，1127组原生DC扭矩曲线对比最大误差0Nm。准备阶段的历史报告当时尚未执行真实回放，后续实际回放及拒绝结论以上述Run为准。实现协议与证据范围见 [A1候选准备验证](evidence/2026-10-18-ufo-native/a1-candidate-preparation/README.md)。

### v2 5008完成与v3生命周期验证

- 5008段 Run `run_7197303fb1594f3cb573` 成功，12:19:56→12:38 UTC；累计updates5008/local334848/global1339392，新增4000次更新。四rank完整状态有限性与原生loader通过；各Replay capacity256、cursor71、full=true，已发生环形覆盖。checkpoint9130083786 bytes；模型 `mv_3351cc68d9bc45699c82`、恢复Artifact `art_3f60c9c220fb4367b918`。
- 5008独立prompt Job `job_b7706561719c4ae19396` 已按相同seed71000/16384 samples启动，等待新z、导出与双后端固定Suite结果后再决定下一段；不因loss下降或训练成功提前继续。
- 新fresh v3清理修复需获得真实正向生命周期证据，单独从头运行112 updates：Train `trn_0f1a177f941f476ea9d7` / Job `job_8153dd9137ec48c8bfec`，目标1800秒、4GPU、同58clips；预览hash `sha256:7a7c168f9d7de16620599d35e9801191d42d016f9b35aa3f92fc66110827af79`，fingerprint `sha256:724bce8847963affc7e114153b5d6dac49575d7536263abc0b3f53c49a471b46`。此段验证新版运行时，不替换v2主实验、不跨镜像恢复，不作为重复质量短训。
- 新固定故障诊断操作 `ufo_failure_probe_v1` 的190项聚焦/共享路径回归通过（其中新probe42项），目标Py3.10实际源码hash与Worker序列化通过；专用镜像 `eb/ufo-native:failure-probe-v1` / `sha256:1c6a284304e6f526f212e82910a73c14e33530eaffdc96745a6a80f613f6bb05` 冻结。两case各2GPU，真实CUDA超物理显存单次分配OOM或缺peer的NCCL allreduce10秒超时；driver90秒、外层100秒TERM+5秒KILL、Job120秒。仅允许diagnostic Artifact，不训练、不发布模型；实际执行仍待空卡窗口。
- 训练模板展示镜像由统一IMAGE读取，修正了默认已v3而元数据仍报v1的遗漏，32项相关检查通过。新界面只列出validated/training_ready的原生数据，A1被拒候选不能误选；11项相关界面检查与类型检查通过。上述控制面及baseline不可变scope附录已通过`make up`部署，启动就绪后复核实际HTTP。

### 5008独立validation与10000区间冻结

- 原生prompt `job_b7706561719c4ae19396` / `run_36092148e87e427ca105` → `mv_811edd0cc14644df8353`；导出 `job_c05e6559f7334692af06` / `run_9d0cbd03e60447df8fd5` → `mv_e92d7e5943af4009b8a6`。96样本数值比较通过，最大action误差3.9637088775634766e-6、target误差9.910762310028075e-6 rad。
- mjlab `job_64fc4822af7d417cb439` / `run_bfde5b435cbd47cc8ac4`：站立/0.3/0.7存活65/45/86步，速度RMSE0.479318/0.364536/0.546963 m/s；MuJoCo `job_b7b24e1f28e94fe29a04` / `run_1bcbb41819d64dd68218`：85/41/42步，RMSE0.576048/0.240292/0.481784。六例仍全部fall，六视频与六trace实际HTTP GET后大小/SHA通过，图表与case入口200。
- 两后端主任务平均存活55/63步，较1008档33.5/35提升64%/80%，各主任务均改善；平均速度RMSE较1008约下降32%/38%，velocity violation fraction约下降36%/46%。反向指标同时保留：max joint excess均值约增加42%/39%，MuJoCo平均max tilt增加约21%；站立drift mjlab0.237m、MuJoCo0.733m，joint violation fraction约0.92–0.96。能力仍不合格，不能以平均存活推导真实行走。
- **12:50 UTC区间决策：继续到累计10000更新，同配置受控探索。** 新增4992次更新，source为本5008完整checkpoint，保持e122镜像、fingerprint d2cd...、58clips、4GPU/E1024/每rank262144 Replay/seed42+rank；不改变算法、采样更新比例、学习率或验收阈值。按5008稳定段约4.9 updates/s，加初始化/完整保存预计约22分钟，段hard timeout5400秒，仍服从全局16:55:31 UTC截止。新恢复提交前先补来源manifest/receipt具体内容hash与已发布Artifact绑定。
- **10000结束后的预先判据：** 先独立推导新z、导出并运行相同双后端Suite/seed、全部媒体与训练时序；非有限、导出/codec不一致、新基础设施故障先停止调整。仅当两后端主任务平均存活较5008均提升至少10%、单个主任务不退化超过20%、平均速度RMSE不恶化超过20%，且姿态/限位/漂移没有抵消改善的明显退化，才讨论下一段20000；混合或退化则停止调整或等待证据，不凭loss追加训练。该门槛用于下一段工程决策，仍非acceptance标准。
- 真实OOM探针v1 `job_750224c6fe4049c0b6ed` / `run_2a45bad419d6499bba15` 在故障注入前因NVML `GPU-`前缀与Torch UUID格式不同而预检失败，exit1且无diagnostic/model。保留现场；UUID标准化同时保留原NVML值，复现测试先失败、修复后43项probe行为检查通过，另建v2诊断镜像，不覆盖v1或训练镜像。
- baseline验证范围纠正已实际发布不可变Artifact `art_6131e8ed9f92438e841a`，hash `sha256:567366a29996b17b7518420afa4c96385ac28242e469af239e7c456ed6cdce0a`，重复POST幂等；旧报告/receipt原字节HTTP hash通过，下载响应附范围链接。详情：[scoped-verification.json](evidence/2026-10-18-ufo-native/baseline-run/scoped-verification.json)。
- 5008独立只读审计：[v2-5008/REPORT.md](evidence/2026-10-18-ufo-native/v2-5008/REPORT.md)。新旧各25文件hash、独立inode/只读源挂载、五套Adam实际1008→5008、四Replay原生loader与环形覆盖通过；667 tags/21298指标坐标严格f32一致，10个HTTP证据hash通过。GPU0/1/2/3峰13914/13914/13916/13914MiB，RAM40130.56MiB，656资源行全ok，容器1093.941秒exit0/noOOM。提交时未记录source manifest/receipt原始字节hash，报告明确不宣称其历史字节before/after相同；当前manifest内容与1008既有不可变resume Artifact一致。
- 该缺口已在新控制面修正：`params.native_resume_source`独立冻结源Run、manifest/receipt原字节hash及可选发布Artifact身份；预览/提交/准入均复核，队列期间变化转blocked且不创建Run，fresh拒绝夹带绑定。201项相关回归通过，覆盖真实文件替换后的提交409与Artifact错配。仅改控制面，冻结checkpoint helper f8b7bec...、v3 entry1ab41f...及v2/v3镜像均保持不变。10000段将实际使用此新绑定。
- v3 fresh112已实际成功：Run `run_3db50a3942f04fd287c9`、模型 `mv_d6ae7c4f98c64babbf33`，112/local18432/global73728，四rank完整finite/loader/Replay true，9个证据文件实际HTTP hash通过。新prompt `job_22c735960021407199f9` / `run_1ac6f4077eb842608c2b` → `mv_e7938374f4f142d2a5a9`；export `job_0868f5c049384d7c98b5` / `run_f6cc0a7b9fb44491b713` → `mv_b0cac1ddf2a049288eb4`，96样本数值比较通过，maxaction5.230307579040527e-6。双后端快评 `job_e63f462d3dbe47a39728` / `job_e5899ccad3c649d580d1` 已提交；此链只验证v3新运行时，不纳入v2训练趋势。
- v3双后端快评均已完成（mjlab `run_023c6d5aa2ee41eaba73`，MuJoCo `run_f27f178f87064bd7b2fa`），6例全部fall、6视频及6trace的实际HTTP大小/hash通过，所有媒体decodable，图表/case入口200。证据：[v3-lifecycle/evaluation-evidence.json](evidence/2026-10-18-ufo-native/v3-lifecycle/evaluation-evidence.json)。正向协议完成，非能力验收。
- 10000段实际已使用新来源绑定：源manifest字节hash `b528e60953e11c0cb5bfb0071722108ae9218dba1d0130cbf39f765aa7b773a1`、receipt字节hash `b04b665383ca2dca31222d80e7639c2987c4d106829c616507d14d38a3de317c`，绑定恢复Artifact `art_3f60c9c220fb4367b918` / `sha256:03a5ccad992b6f97f2f1531cd7e6a01c99d25ee2ffa8adc9dbb918cb8b132307`。原始预览：[v2-10000/resume-preview.json](evidence/2026-10-18-ufo-native/v2-10000/resume-preview.json)。目标10000、新增4992更新，预计local655360/global2621440，hard timeout5400秒。
- 故障探针v2 `job_70815f5fd01a4fd8a3b1` / `run_31185cb4c95b49ba9216` 再次在注入前预检失败，未发生CUDA OOM。实测四卡无计算进程、memory.used均1MiB，但total−free为484MiB；原检查错误包含驱动保留。改读实际used字段，原256MiB门槛不变，保留total/free/used及原NVML UUID；实测格式复现先2项失败，修复后45项通过（包含真实used超限拒绝），lint通过。最终v3专用诊断镜像 `sha256:dc3c82c0f9a6baa4c5b92acce0109936c2a81b7bd83f979c0d9cd5f44be63e26` 已冻结，源码hash逐项相符；`job_9ad6dfcfd9a145dcb3b9`（CUDA OOM）和 `job_4cf8189b9e7c42569705`（NCCL超时）已进入平台队列，待10000段释放GPU，不抢占主训练。
- 依赖复核区分NCCL口径：Torch `torch.cuda.nccl.version()`报告tuple2.26.2；实际已加载库路径明确，直接`ncclGetVersion`返回22605，与包元数据2.26.5一致。两种原始读数均保存于 [checks/nccl-runtime-version.json](evidence/2026-10-18-ufo-native/checks/nccl-runtime-version.json)，未改变库或镜像，不把Torch报告tuple误当实际替换了运行库。

### 10000完成与实际GPU故障验证

- 10000段已成功，累计local655360/global2621440，新增4992更新；checkpoint9130129669 bytes，模型 `mv_83ddd78dabd3480a8172`。独立CPU只读审计已开始，实际prompt `job_977b1971c1244369902d` / `run_79b001d4be584f72bb73` 运行中，仍等待物理validation后决定20000。
- 额外只读实测确认：仅向容器暴露宿主GPU2时，NVML报告index0但UUID仍指向宿主GPU2。当前诊断回收证明依赖平台既有doctor索引（无UUID），因此没有扩展未验证映射或放宽校验；已取消尚无Run的并行NCCL Job `job_4cf8189b9e7c42569705`，改为经已验证宿主GPU0/1顺序执行。该探针不能宣称任意GPU子集的回收映射均验证。
- v3真实CUDA OOM `job_9ad6dfcfd9a145dcb3b9` / `run_8854243c232e4d558ef0` 成功识别原生 `torch.OutOfMemoryError`（请求24.52GiB、CUDA可用总容量23.52GiB）；两独立GPU rank已退出、Ray清理完成、容器exit0/no host or cgroup OOM，GPU0/1 lease清空且free均恢复24080MiB。外部回收检查距Run开始27.210742秒，小于120秒；diagnostic Artifact `art_30838539e4144f1e811b`，actual_optimizer_updates=0/model_published=false，不能当训练baseline。
- OOM七个索引成员均实际HTTP GET并验证大小/hash，原始文本与回执、资源回收证据保存在 [gpu-faults/cuda-oom.json](evidence/2026-10-18-ufo-native/gpu-faults/cuda-oom.json)。两次未注入的历史预检失败现场另存 [fault-preflight](evidence/2026-10-18-ufo-native/fault-preflight/)，不将它们计为OOM证据。
- OOM回收确认后提交独立NCCL Job `job_ceca8ea8b46f45a09236` / `run_8f00bb4f262942eab6b1`，固定双rank先真实warmup allreduce，再rank0独自allreduce/10秒PG timeout；仍待真实结果。

### 10000独立validation退步与诊断候选保全

- 10000独立完整审计见 [v2-10000/REPORT.md](evidence/2026-10-18-ufo-native/v2-10000/REPORT.md)：新旧各25文件hash、独立inode、提交前来源原始字节绑定、五套Adam实际5008→10000、四Replay原生loader及保留浮点全量finite均通过。655 native tags/23980坐标与12资源tags/2400坐标，合计26380坐标在CSV/TSDB/API严格f32一致。GPU四卡峰14488MiB，RAM39987.2MiB，800行资源均ok；容器1332.238秒exit0/noOOM。
- 新prompt `run_79b001d4be584f72bb73` → `mv_c0683852eb0e4b789d9e`，新export `run_0b531549458c427fa815` → `mv_5c07067b95ba4a56afbc`；96样本数值检查通过，max action7.32392e-6、target2.63661e-5rad，使用原atol/rtol组合规则。
- mjlab `run_2582562be95f4c9c8d43`：站立/0.3/0.7存活12/53/46步，速度RMSE0.309812/0.640527/0.573641m/s；MuJoCo `run_6b1d9a4faa78447d9c23`：11/41/43步，RMSE0.553410/0.624647/0.555589。六例均fall，六视频/六trace实际HTTP大小/hash通过，图表与case入口200。五段共30个失败案例已更新 [趋势图](evidence/2026-10-18-ufo-native/validation-trend.png)，主任务趋势不混入0.7诊断。
- **13:28 UTC区间结论：停止自动推进20000。** 相比5008，两后端主任务平均存活55/63→32.5/26步，下降41%/59%；站立65/85→12/11，下降82%/87%；0.3m/s速度误差增加约76%/160%。这违反预先冻结继续门槛，不能用训练loss或已授权完整训练默默豁免。已询问剩余预算用于诊断还是明确越过门槛继续探索；等待期间只做不依赖回答的核查。
- 按两后端主任务平均存活选5008作为本轮较好的诊断候选，10000最新完整状态独立保留；两者各自M/MP/ME及Replay/prompt Artifact的实际pinned与hash见 [candidate-selection.json](evidence/2026-10-18-ufo-native/candidate-selection.json)。这不是正式最佳模型或acceptance结论。
- 已经平台Job `job_b004fe03a48f4e6fa9ab` / `run_7549cc2e8c6c4d328519` 生成候选仿真包 `bnd_65f11e32b63f4ce48e04`：41479071 bytes，raw SHA256 `e61498c6c57597fc6cd707e1c91dcbea2eb4820faa1bbd95937d60ac957f5032`。实际HEAD200、Range206及完整GET大小/hash通过；仅simulation_only/unsigned/diagnostic，不代表真机或正式验收通过。

### NCCL故障现场与清理修复复核

- v3 `run_8f00bb4f262942eab6b1` 两rank均完成实际NCCL warmup；rank0进入第二次allreduce，rank1明确跳过；原生日志记录WorkNCCL ALLREDUCE超时10000ms、实耗10001ms及DistBackendError，故障注入确实发生。
- 随后的driver清理37.5089秒报告remaining_actor_pids=[1503]、complete=false，容器exit1，未发布diagnostic Artifact或模型。历史文件没记录该PID的/proc状态，不能事后认定其一定是zombie。Docker外层13:19:37.492658407→13:20:16.511732617，39.019秒内退出、Pid0/no host OOM；外部GPU空闲直接观察时间13:22:01.669944，不能把这次观察宣称为120秒内GPU回收证据。现场见 [nccl-v3-failed](evidence/2026-10-18-ufo-native/gpu-faults/nccl-v3-failed/) 与 [外层恢复](evidence/2026-10-18-ufo-native/gpu-faults/nccl-outer-recovery.json)。
- v4改为记录真实actor PID/start_ticks/pid namespace，区分gone、Z与live；Z表示已退出但尚待父进程回收，不写成PID消失。仅对身份核实的本容器已记录actor通过pidfd定向SIGKILL，固定总等待仍5秒；身份未知、pidfd不支持或清理异常均拒绝发布。没有扩大到宿主或其他Job进程。
- 151项回归通过（含57项probe），冻结训练v3目标Python3.10真实子进程/pidfd7项通过；原v3 OOM Artifact经新GET handler真实文件回放，响应/scope/hash完全相同。日志见 [v4-checks](evidence/2026-10-18-ufo-native/gpu-faults/v4-checks/)。训练runtime、checkpoint helper和训练镜像均未改。专用v4镜像 `sha256:02731146f8399299324c910a25437de9c940c8ad87a3dbf4cbda185bc448a734` 已构建，实际新Job验证待部署就绪。

### NCCL v4真实验证完成与更新后取消准备

- v4 `job_fe7c3fbd5a544bfab13f` / `run_75c9ed32a37e421c8baa` 已成功：两rank warmup通过，rank0独自allreduce实际触发10000ms超时/DistBackendError；driver清理32.966秒、两actor实际gone、forced/zombie/live/unverified均空。外部collector在Run开始后36.079899秒确认容器exit0/no host OOM、GPU0/1 lease空且free均24080MiB。
- 发布不可变diagnostic Artifact `art_0b5c590d0d6742fbb7b8` / `sha256:ac6ed419e13aa9d616135053e225debf5d45f432c67a198ad9730017a4c6e70f`；七个索引文件均实际HTTP GET并逐项验证大小/hash，完整小型证据见 [gpu-faults/nccl-v4.json](evidence/2026-10-18-ufo-native/gpu-faults/nccl-v4.json)。本次没走强制pidfd分支；该分支已由目标Python3.10真实进程检查覆盖，不冒充GPU实测。v3失败Artifact未补造。
- 最终复核发现原取消只覆盖trainer.train前0更新，不能宣称实际更新中也已验证。单独提交 `trn_07a50443179940328005` / `job_ff4914075977459b8090` / `run_e3baea56495942cdb031`，从已成功v3四卡112 checkpoint恢复，目标512、hard timeout600秒，镜像/源/数据/拓扑保持原身份。冻结触发条件：新Run自己的原生CSV optimizer_steps≥128且TSDB一致、Job仍running且无完成回执，再经API cancel；回收期限为请求后120秒，不发布新模型或恢复候选。该实验不为追加质量短训，也不对它运行prompt/export/快评。

### 实际更新后取消与全量Replay诊断

- 更新后取消实验 `run_e3baea56495942cdb031` 新CSV/TSDB记录128→304共12点，源只有112，证明至少新增192次agent更新；四rank实际UfoTrainWorker身份及GPU UUID保留。13:48:17.608733 UTC在Job仍running且无完成receipt时调用API cancel，13:48:20.261269返回cancelled（2.652532秒）。最后CSV304不是退出精确更新数，不补造完整checkpoint。
- 13:49:00.169844实际外部doctor确认四GPU lease=null、free均24080MiB（距请求42.56秒，小于120秒）；任务容器已删除，ebd保存exit137/no host OOM；记录的191个宿主PID全部不在/proc，实际GPU compute-process列表为空。TrainJob/Job cancelled、Run aborted/cancelled，Artifact/WeightModel/命名ModelVersion均无发布，checkpoints=[]/resume unavailable/完整receipt不存在。源25个payload及manifest/receipt hash均未变、源resume依然available。详细证据见 [cancellation-after-updates](evidence/2026-10-18-ufo-native/cancellation-after-updates/)。本次未触发Job总时限，自动timeout仍主要由行为测试覆盖，不冒充真实超时训练实测。
- 只读质量诊断完成：[quality-diagnosis/REPORT.md](evidence/2026-10-18-ufo-native/quality-diagnosis/REPORT.md)。未见source/旧z/task/过滤数据接线错误。5008保留1008旧5行的六字段逐元素相同，占storage1.953%，原生4096draw中实际50.293%来自旧尾；10000已覆盖。直立目标覆盖减少，但原生reward均值并非都下降；质量因果未证，不改算法或凭此重写阈值。
- 在原6小时上限内冻结独立全量Replay推断实验：[full-buffer-diagnostic/PROTOCOL.md](evidence/2026-10-18-ufo-native/full-buffer-diagnostic/PROTOCOL.md)。5008/10000各自冻结模型+自身Replay，以num_samples262144/seed71000走原生get_full_buffer，再新export与相同双后端Suite；保留原16384协议与继续门槛。全量推断同时改变样本量，仍非纯因果证明。
- 两prompt Job分别 `job_c8dfb3643a63432ebca2`（5008，Run `run_6e256f5abd7849d392ac`）及 `job_4b0076a9ccb643b880a7`（10000）。各4CPU/64GiBRAM/0GPU、1小时hard timeout；按实际内存余量顺序准入。主训练没有进入20000。

### 全量Replay推断对照进度（独立协议）

- 5008原生262144/seed71000推断完成，Run `run_6e256f5abd7849d392ac` → `mv_4775af0d47c045278c2a`；export `run_8cedd699ca5e45ce9307` → `mv_452959fdc3f645b9929f`，96样本通过、max action5.44265e-6、target1.82390e-5rad。
- 新mjlab `run_5db8cc3bd08b44018e81` 站立/0.3/0.7存活11/42/33步，RMSE0.257311/0.638901/0.412948；新MuJoCo `run_01844a97687c47988383` 11/38/36步，RMSE0.226414/0.534022/0.419612。六例均fall，全部视频可解码，两个receipt的完整索引成员均经实际HTTP GET大小/hash核对。原5008的站立65/85步优势在全量推断中消失，支持推断采样/窗口对观测质量有贡献，不能单独证明训练参数好坏或纯权重因果。
- 10000原生262144推断 `run_e2e19ceeff5f421bbcef` → `mv_b08b8cddb1bc442b8b92`；export `run_06685f7ea3f64b978db7` → `mv_140b3f987e4a47a0a410`，96样本通过、max action7.48038e-6、target2.69294e-5rad。双后端同Suite快评已提交，待收齐结果；未推进主训练20000。

### 全量Replay诊断完成与本轮停止决定

- 10000全量推断双后端完成：mjlab `run_c4c0a3b6a7db4f1ea5ed` 站立/0.3/0.7存活12/66/55步，RMSE0.339179/0.528002/0.576188；MuJoCo `run_b9f57bf24d314576acec` 13/82/35步，RMSE0.391793/0.667344/0.576628。新增12例全部fall、12失败视频/12trace完整，四receipt索引的36成员实际HTTP GET大小/SHA256全部通过，无成功录像。报告：[full-buffer-diagnostic/REPORT.md](evidence/2026-10-18-ufo-native/full-buffer-diagnostic/REPORT.md)。
- 全量协议10000较5008主任务平均存活26.5/24.5→39/47.5步；平均速度RMSE mjlab略改善、MuJoCo约恶化39%，后者0.3m/s漂移1.214m。原5008站立优势消失但任务仍失败，不能把旧16384组回退单独归因于训练权重，也不能追溯替换旧继续判据。原主30例与本独立12例分开记录。
- 独立prompt审计通过：[PROMPT-AUDIT.md](evidence/2026-10-18-ufo-native/full-buffer-diagnostic/PROMPT-AUDIT.md)。同模型新旧policy.onnx字节完全相同，权重/各自Replay/seed/参数与PKL→JSON→export latent绑定一致；5008站立新旧z cos0.49707、L2 16.0468，10000 cos0.924135、L2 6.2324。独立原生loader/get_full_buffer实际返回262144且next qpos多重集合逐字一致；原任务无branch marker，只按冻结源码分支条件及参数确认，不冒充直接trace。没有改算法/采样器/训练默认。
- **本轮决定：停止自动追加主训练，等待下一段策略确认。** 已执行10000累计真实agent更新、五段主validation及全量Replay对照，仍没有可可靠站立/行走的模型；原继续门槛未达到。保全原5008诊断候选包和10000最新完整checkpoint，未提交20000。用户授权首轮6小时上限不等于必须耗尽：当前约3小时15分钟墙钟，尚未跑满6小时；若用户仍要求原配置持续探索，需明确本次未达门槛的继续决定。新增原始动作路径尚未收到，A1候选已因真实动态失败拒绝，未凑数据或改变当前58/15版本。
- 接入、实际训练/恢复、独立原生入口对照、导出、双后端媒体、OOM/NCCL/更新后取消的证据已收齐；自动Job总超时主要为行为测试、官方数据祖先/许可及严格泛化、正式acceptance与真机均未获验证。没有Git暂存/提交/push，设计25与外部源码保持原样。

### 工作台复现入口

在“训练→新建”选择RLinf的 `rpo_ufo_fb_native`，接口 `ee_rpo_v2_ufo_native_23dof_v2`、Profile `env_train_rpo_ufo_fb_native`，使用已验证数据集 `ds_6541f726d7804007a93a`。当前fresh默认镜像为v3；目标使用16的倍数（100档实际112），参数和预览确认后提交。详情页提供实际更新、local/global transitions、原始指标/图表及来源绑定恢复预览。主v2恢复必须仍用原e122镜像及既有指纹，不跨到fresh v3。

训练模型详情先按自身Replay推导任务prompt，再打开新派生模型进行导出、smoke评估和仿真包构建。原周期协议seed71000/16384；全量262144只对应本次独立诊断。Job succeeded代表执行成功，正式验收仍需另冻结Suite/阈值/置信要求。用户可以直接查看候选bundle `bnd_65f11e32b63f4ce48e04` 的受控下载；它始终为simulation_only/unsigned/diagnostic。
