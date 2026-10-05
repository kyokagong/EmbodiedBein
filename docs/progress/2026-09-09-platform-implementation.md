# 2026-09-09 通用平台实施与 LIBERO 验收记录

当时的实施依据已合并进现行 `docs/design/`；新增必要场景为 LeRobot LIBERO v3 → π0.5 SFT → 在线 PPO → 独立仿真评估。全过程不使用 Git，不修改外部仓。

## 当前验收状态（持续更新）

下面各轮记录是历史证据；状态以本表及后续实测更新为准。

| 环节 | 实测状态 |
|---|---|
| 平台下载模型、数据索引与实际视频 loader | 通过；数据仍是结构验证状态，未冒充不可变数据版本 |
| π0.5 SFT | 三次四卡、2步训练成功并发布完整检查点；最新72GiB预算也已通过 |
| PPO | 76GiB下完成两轮真实rollout、优化及两次保存，最终完整模型发布成功 |
| 独立评估 | 自动消费成功PPO输出，4条完成轨迹及报告凭证通过；仍明确为未完成发布资格验收 |
| 平台回归 | 后端144项、Rust工作区、前端类型/5项行为检查与生产构建已通过；最终部署后API/ebd健康，线上预览资源与保存策略通过 |
| 多机器人采集、不可变数据版本、Orin交付 | 仍有设计范围待实施/设备验收，不能以LIBERO运行检查替代 |

## 最终端到端运行检查

平台在用户说明“无需代替训练”前已经启动的验收管线已自然完成；此后不再提交训练。最终成功执行为`pr_daf7506388ed4e57b4ce`：

- SFT复用本次72GiB成功阶段`job_044775d3b0054d7a9844` / `run_be000fb3359744a5a112`，发布step2完整包8,526,579,376字节。
- PPO `job_c772e55345c2456a99ab` / `run_25bdb3d2dde942cb9409`在76GiB完成两轮rollout、优化和保存，exit0且OOM false；step2完整包8,537,600,296字节，核心`full_weights.pt` SHA256为`62cda34b5b42bd239b7b782a999c2a723e9531e5943d7205e5ec49d75f36cf69`。
- 192个5秒样本中cgroup总量最高触及76.000GiB，扣inactive_file后的最高工作量70.324GiB，主机可用内存最低11.525GiB；`oom/oom_kill`及Ray杀worker计数均为0。该配置有回收压力，不能描述为余量充裕。
- 独立评估`job_7692e78854da44b98216` / `run_9cc87be2f4e749cab872`绑定上述PPO模型目录，完成4条Spatial轨迹并生成报告`ev_194156f0ce904c0ca492`。小样本success_once为1.0，Wilson 95%区间为[0.5101, 1.0]；报告保持`not_qualified`，不作为论文基准、四套件结论或真机发布门禁。

新建四卡PPO默认内存据此设为76GiB，四卡SFT保持72GiB，评估保持一卡64GiB；长训练每40轮及最终轮保存，避免每轮全量模型和优化器分片写入。用户通过页面自行填写正式步数、batch、学习率和PPO轮数后再启动。

最终平台镜像已用`make up`更新。部署后的三个只读训练预览均无错误，实际返回SFT/PPO/评估内存`72/76/64 GiB`，PPO覆盖包含`runner.save_interval=40`，API与ebd健康。检查期间未创建或启动作业，活动作业列表为空。

### 第七轮：页面是否能证明任务镜像已经就绪

最终复核发现训练模板虽已绑定`eb/rlinf:libero-pi05`，环境页目录却没有该条目，旧目录状态也只是静态登记。现由ebd提供只读镜像检查，API仍不持有`docker.sock`；环境页实时显示任务镜像为`ready`、摘要`0b13e47916c6…`、体积44,019,073,948字节。Rust定向8项、重新构建部署和线上目录读取均通过。

当前失败运行 `pr_21e9d27614414c17a5e6` / `run_e90c050a50ed45bea9ba` 使用72GiB预算，Ray在约68.6–69.93GiB达到95%阈值时杀掉环境和通信worker；内核OOM计数为0也不能视为健康。平台保留失败记录、阻止评估阶段启动；后续修复不关闭Ray内存保护或终止外部进程。

准入、取消和运行回收现在共用Job数据库行锁并重读状态，避免“取消已提交后又启动容器”及回收覆盖取消结果；空操作及时释放锁。相关容器回归12项通过，包含异步事务模拟；真实PostgreSQL并发验证单独记录。

真实PostgreSQL双事务检查已通过：`pg_blocking_pids`确认取消事务阻塞于准入行锁，准入提交后刷新同一缓存Job并取消新Run，最终数据库状态为cancelled/aborted。脚本保留在`backend/tests/integration/postgres_cancel_admission.py`，仅显式隔离DSN启用，临时数据库已清理。

长运行指标采集已移除整文件读取及16MiB文件前缀限制，改为逐条TFRecord读取；16MiB现在仅是单记录内存上限，较大的媒体记录跳过，不阻断后续scalar。超过17MiB后继续采集及不完整尾部写完后重试均有回归。仍沿用旧解析器不校验CRC的边界，以及每轮全扫描/按tag和step去重；尚未引入持久化字节游标。

新完整验收管线 `pipe_27be6b1fc6de479aba61` / `pr_5e28e6e0d13f4bd38496` 已启动：SFT四卡72GiB、2步 → PPO四卡72GiB、2轮 → 独立评估一卡64GiB、4环境。SFT Run为`run_be000fb3359744a5a112`。任务镜像`sha256:51d26a688325acea6c3a06c2327ff8ac2b5defa4f26ca39ef753432032cf85fc`增加模型加载及CPU/GPU切换后的glibc空闲堆回收；真实CPU模型实验RSS从11,011,556降至8,357,724KiB，各参数的抽样值保持一致，不能据此代替多轮PPO验收。新表单独立设置`ppo_max_epochs`，旧运行不变；最新后端144项通过。

该新SFT已正式`succeeded`：`job_044775d3b0054d7a9844` / `run_be000fb3359744a5a112`，step2完整包8,526,579,376字节。两步loss为0.0387138、0.1603783，grad_norm为0.5967993、1.6586518；仅作执行证据。管线自动创建PPO `job_23d3455fb56a4e7fa499` / `run_a8de9323222b4e4cbc3f`，绑定该SFT新输出并设置2轮，正在执行。

该PPO随后在第一轮保存阶段失败：20:16:37记录`Saving checkpoint at step 1`，20:17:15通信服务退出；ebd发现`OOMKilled`后，调度器清理剩余容器并将Run记为`aborted/oom`，没有启动评估。保存前cgroup采样为56,666,263,552字节（52.78GiB），当时Ray杀worker计数与内核OOM计数均为0；这不能覆盖后续保存瞬时峰值。主机没有`memory.peak`文件，因此不报告虚构峰值。继续定位检查点阶段的CPU副本与写入压力。

磁盘证据显示四份DCP分片和`.metadata`均已生成，但没有`full_weights.pt`。自有PPO保存入口现将原生DCP保存与原生全量模型汇聚分为两个阶段，中间回收临时CPU堆；完整训练状态并未省略。双进程Gloo原生检查点回归验证模型参数、Adam动量、学习率进度和torch随机状态均可恢复。镜像`sha256:0b13e47916c6867d91aab776b580c2043c77b21f71147ed376047e0038c326a2`已用于续跑`pr_256630e0b0314bee8e49` / PPO `job_03757569f1ee4f66ab55` / `run_ac55cc7dd32741ddbc24`；仍是72GiB、2轮，复用新SFT成功阶段。

该续跑已完成第一轮训练及完整保存（`full_weights.pt`为8,537,595,564字节），同一组环境/rollout进程成功完成第二轮交互。第一轮实际记录4条完成轨迹，policy_loss0.0367851、value_loss0.4718533、grad_norm56.4045。保存期间采样触及72GiB上限，`memory.events.max`增至176118，表示发生大量限制回收；`oom/oom_kill`与Ray杀worker计数仍为0，保存后内存回落。不能将此配置描述成内存余量充裕，也不能用4条训练轨迹证明基准性能提升。

两轮训练和两份完整权重最终均写出，但第2次保存期间Ray杀掉rollout，随后`pop_execution_times`传播失败，Run以exit255退出，未发布健康最终模型。杀进程前5秒采样总量约72.00GiB，扣inactive_file约71.45GiB，anon+shmem约68.45GiB；脏页约165MB，并非主要写盘堆积。Ray错误报告68.62/72GiB超过95%阈值。主机当时可用83,267MiB（81.31GiB），保留4GiB后允许尝试76GiB；新续跑`pr_daf7506388ed4e57b4ce` / `run_25bdb3d2dde942cb9409`保持2轮、同镜像与输入，仅新PPO预算72→76GiB。保护阈值和外部进程不变，仍待真实结果，不能把预算推算当成验收通过。

## 已落地的共同执行基础

- ebd 内容快照（清单 SHA256、复制稳定性验证、只读发布、路径边界），替换原来的 Git 工作树操作。
- 运行 CPU/RAM、共享内存与 bridge 网络传递；修复调度器错误读取 Job.started_at，使用 Run 时间。
- 训练预览与提交共用输入构建器；实际挂载选中的数据/模型；模板 kind/env/task 共同校验，数字调参不能覆盖输入、接口和运行环境。
- 保留现有 Python/Rust/Compose 架构，所有新增下载、索引、训练和评估任务由 ebd 容器执行。
- Makefile显式关闭Buildx的Git信息、标签及dirty检测配置；版本证据使用内容快照和镜像摘要。配置依据：[Docker构建变量](https://docs.docker.com/build/building/variables/#buildx_git_info)。

## LIBERO 操作链路

1. 数据页登记 `lerobot/libero`，固定完整 revision，选择 LIBERO 本体接口，执行下载/结构验证。不能把登记成功或文件存在当成可训练。
2. 模型页下载 `RLinf/RLinf-Pi05-LIBERO-SFT`。已核对公开版本 `45ccfcc4e28634f1576ebf78cab0fbe2fd82432d`，权重和 `physical-intelligence/libero/norm_stats.json` 共约 7.47 GB。这是已完成上游 LIBERO SFT 的起点，后续 SFT 是继续微调，不冒充未经 LIBERO 微调的通用基座。
3. 训练页选择 `LIBERO π0.5 SFT`、数据及完成下载的模型。可以单独提交，或创建 SFT → spatial PPO → 独立评估管线。
4. 每次管线启动保存独立 PipelineRun；下一步只消费成功生产者的完整模型目录凭证，不猜测 latest 文件。
5. 评估单独使用 EmbodiedEvalRunner；结果进入 EvalReport/Run 指标。作业成功不表示发布资格通过。

公共参考：[LeRobot LIBERO](https://huggingface.co/docs/lerobot/main/en/libero)、[RLinf 模型卡](https://huggingface.co/RLinf/RLinf-Pi05-LIBERO-SFT)。本机原有 LeRobot 格式 pi05_base 不直接作为 OpenPI 权重使用。

## 数据与界面

v3 元数据索引检查 episode/frame 范围、总数、分片列、共享 MP4 的起止区间；页面可按区间播放。结构验证、解码/loader 验收、不可变发布是不同层次，当前记录明确保留 unchecked 项。下载任务固定 revision、已知文件大小与空间预算，拒绝 API 内后台下载和明文 token。

训练详情从实际 Run 投影状态、路径、资源和指标；去掉错误 reward/loss 回退和未实现的自动登记开关。导航露出已有数据、训练、模型、评估、交付与环境能力。

## 三轮反思及修正

### 第一轮：输入是否真正到达训练

发现原 TrainJob 保存 dataset_ids/base_model_id，却没有填入 JobSpec.inputs；修复共享 builder 和只读挂载。发现 shell 拼接可改变命令，改为 argv 与有限数值覆盖。发现 GPU comma 覆盖并非合法 Hydra key，使用实际镜像合成后改为转义字典。SFT 模型配置显式选择 pi05_libero，10 步动作与 raw delta 语义一致。

### 第二轮：完成、失败与恢复是否可信

发现 trn_* 输出目录与 run_* 实际目录脱节；现在只使用 Run。发现单个权重文件被任意登记；改为完整发布凭证。发现旧管线在模板上累加执行状态；改为独立运行和步骤幂等键。发现超时检查访问不存在字段；补回归后修正。发现损坏评估凭证可反复阻断整个调度循环；现在持久化失败状态。

### 第三轮：部署和真实依赖是否一致

发现 API 镜像内 runner 路径不是宿主机路径，移入任务镜像。发现原训练镜像虚拟环境解释器和依赖不完整；逐项修复并使用真实 OpenPI/LeRobot loader 检查。发现 only_eval 配置不会让训练入口自动切换，改为专用评估入口。发现容器无法直连 HF，验证镜像地址并做配置；LIBERO 环境资产和 tokenizer 纳入镜像验收。

## 验证与边界

- 后端、Rust、前端、迁移和实际 loader 结果以本文件后续验收记录为准；不把单元测试或配置合成当作 SFT/RL 成功。
- 4×4090 是四张独立显存卡，不能按 96 GB 统一显存估计。外部进程可能占用 GPU；不终止其他运行中的任务来制造可用资源。
- SO101 单臂、G1 EDU 仅仿真、Berkeley 实物配置未完全确认、Orin 64 GB 的边界保持不变。本轮运行基础可复用，但并未据此宣称所有机器人采集适配器、全部 RLinf 模型配方、Orin 量化/真机交付都已验收。
- 大规模数据不可变版本/分组切分、完整物理采集、全部模型配方、目标设备量化仍须按设计逐项实施与硬件验收，不能隐藏为已完成。

子任务证据：runtime-implementation-report.md、data-implementation-report.md、pipeline-implementation-report.md、workbench-implementation-report.md；训练镜像与运行检查见后续补充。

## 实机环境验收记录（持续更新）

- `make up` 已完成一次实际升级；API/ebd 健康，12 个模板可读，PipelineRun 迁移已执行。
- 容器内后端全量 100 项检查通过；随后新增下载凭证异常检查，3 项定向检查通过。Rust workspace、Vue 类型检查、3 项前端回归和 Vite 构建通过。隔离 PostgreSQL 的 0001→0004 迁移及种子验证通过。
- 平台实际下载并校验 `lerobot/libero@a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4`：数据 ID `ds_398a3043320d4ba69566`，1693 episodes、273465 frames、40 tasks。数据标注 10 Hz，与仿真接口 20 Hz 有差异；不自动重采样动作，不宣称 G2 已通过。
- 实际数据在任务镜像中只读加载，AV1 视频解码和模型输入形状检查通过（actions 10×32、state 32、224×224 图像、prompt 200）。该形状检查使用测试归一化数据，不等同于真实权重训练。
- 用户指定的公开 π0.5 模型下载由平台作业执行：模型 `mdl_c01022c5c05e463d91e0`，Run `run_56ede7bc25df49e1a6a4`。模型仍传输时保持 downloading，不允许用未完成目录训练。
- 9 个 LIBERO 配方经真实 Hydra 合成通过；离线 tokenizer 和无 GPU 的 EGL Panda 图像渲染通过。LIBERO 资产镜像构建、SFT/RL 实际训练尚未记为通过。

### 第四轮：真实输入和硬件反馈

真实数据暴露旧overview显示0集和缺失动作维的问题。已改为v3摘要与episode分页，恢复40个task的真实标签；线上详情8977字节，50项分页连续，未知任务0。重新索引通过持久化作业执行（`run_318b78b67c7447a4bd80`），不在API读Parquet。

真实SFT配置合成发现上游SFT默认全参，不能照搬PPO的专家模式假设。平台现显式选择冻结VLM、仅微调动作专家，并在表单说明。实际meta模型及公开权重头核对总参数3,616,757,520，可训练430,098,464，冻结3,186,659,056；按此修正解析显存估算（不是实测峰值）。FSDP分组检查115个单元无混合requires_grad/dtype问题。调度按实时剩余显存筛选GPU，计入平台外进程占用，不终止它们。

公开模型下载已完成：7,473,095,971字节，平台状态`base`；真实模型与数据绑定的2步SFT预览返回200。最新后端全量106项检查通过；Vue类型/3项前端回归/生产构建通过。更新镜像`eb-api`为`c3db684c9110f79e2dce2069ba6fbbcf0c169844cdbdaecec52ef90ddff22799`，包含只读重索引与显存准入修复。

### 第五轮：实际提交与独立审查

首次实际SFT提交`trn_99edb6f51fa5455fbfa6` / `run_34aaf7b685cd4e809514`未进入训练：先暴露代码目录混有15GB旧权重导致快照缓慢，随后Docker多GPU参数被CSV拆分导致infra_failed。均保留真实失败记录。代码快照v2排除常见权重文件并记录策略，普通mesh.bin仍保留；Docker GPU列表改为正确的CSV字段引用。

独立复审补充修复：管线创建预检各阶段容量；评估不再分配虚构优化器显存；微批数影响激活预算；下载凭证逐文件检查大小、必要资产和路径边界。PPO依据实际432,853,025个可训练参数及同步offload路径估算，仍标记analytic，不能替代真实峰值验收。

实际四卡任务第二次运行`run_a98d0e5bf11744aba1bb`因旧LeRobot import失败；第三次`run_74b338fced024840923e`已完成模型和数据初始化，但首次前向因OpenPI绕过GemmaDecoderLayer.forward导致FSDP空分片访问失败。平台正确显示failed及结束时间，不将Global Step 0当作训练成功。官方rlinf-openpi fork也具有该直接访问路径，不能只用版本替换解释修复；正在验证实际调用边界的FSDP配置。

CPU真实全权重加载峰值约15.48GiB；四卡容器内存预算设为96GiB。观察到四卡初始化阶段54.59GiB，尚不是全程峰值。后端最新全量121项通过。

验收管线定义`pipe_5038365fcad142e9a08f`已创建，资源为SFT4卡/96GiB、PPO4卡/96GiB、评估1卡/64GiB；尚未启动时根据实测修正了内存配置，不更改已启动PipelineRun。

## SFT 实际通过

- `trn_36f8b09445584b158bb2` / `run_273f2363c6d14d79af45`：真实4×4090、2步SFT已`succeeded`，结束时间2026-09-09 18:00:47 UTC。
- 第1步loss约0.0079、grad_norm约0.129；第2步loss约0.0818、grad_norm约1.21。仅证明优化步骤执行，不是性能提升结论。
- 完整模型包8,526,579,376字节，step2，检查点凭证和归一化资产通过共同校验。已显式登记`mdl_e58083e4927b4cd4858a`，名称`LIBERO-pi05-SFT-2step-validated`。
- 完整任务镜像`sha256:542dd9bd38404dfdd117d01c8436b10f3ae15aa8b8223fec7f3849b751546471`：固定rlinf-openpi0.1.1、rlinf-libero0.1.3、LeRobot0.4.3，包含586个固定版本仿真资产和离线tokenizer；真实LIBERO spatial task0 reset+初态恢复+7维动作一步+双64×64摄像头通过。
- 真实训练暴露TensorBoard位于output而旧collector仅扫描logs；已扩展logs/output/tb并去重，保持Rust数据面处理。原16MiB事件解析前缀上限仍存在，超长运行日志需后续增量读取实现。

### 第六轮：真实 RL 初始化与评估真实性

三阶段运行 `pr_83760935575045eeacdc` 的 SFT 再次成功（`run_6f4f373ac798480cb8e6`），PPO 已正确绑定该运行自己的模型；首次 RL 因快照递归排除了 `rlinf/envs/venv` 真实包而失败。排除规则现仅忽略源码根的虚拟环境目录，包目录保留，实际不可变快照的离线 LIBERO reset/step/双摄像头检查通过。

续跑 `pr_c8f5d05ceec147e6bddf` 复用成功 SFT，创建新的 PPO `run_61ae205ef16747e29428`。实际初始化触发主机级内存压力（可用内存约0.81GiB、PSI full约98%、容器oom_kill=1但主进程仍在运行），已通过平台取消整个作业；未终止其他服务，不将其记录为成功训练。正在减少rollout副本并串行CPU加载，新增实际可用内存准入与4GiB主机预留。

独立审查修复：评估拒绝空指标、非有限数和零完成轨迹；成功率定义为已完成轨迹中至少成功一次（success_once），分母来自num_trajectories。旧页面缺证据时补64次试验和合成sim2sim门禁通过的逻辑已删除。checkpoint根目录符号链接绕过已封堵。流水线失败续跑保持旧记录不变，仅在新运行中复用完整成功前缀。

低内存续跑 `pr_a70a5389f5f44540b627` 已于18:59 UTC启动，PPO `job_3ba136a1d6f747cd8864` / `run_26ca5f6650ea4f9e90fe`，新快照记录内存96→72GiB，旧运行不变。任务镜像`sha256:57d390a00dfba950573496d72e2223b3cbfd75409f7afc2f2766bdca28b87ea9`；四actor/单rollout/单env布局通过真实Hydra合成，四rank串行CPU加载通过gloo测试。初期内存约28.41GiB、主机可用约53.38GiB、oom_kill=0，尚不能当作全程峰值。

资源探测由全进程扫描改为仅刷新内存，实际ebd CPU采样从高峰回落约1%。运行详情修复状态不刷新和GPU字段名错误，新增真实评估报告、完整检查点入口。最新完整后端130项通过，Rust工作区与前端类型/行为检查通过；OOM子进程仍存活时的清理回归另3项通过，待随下一次部署生效。

低内存PPO最终完成全部actor加载与FSDP初始化（19:06:22 UTC），进入首次仿真交互后因外部RLinf `env_worker.py` 缺少已有`add_time_limit_bootstrap_rewards`导入而失败，非OOM或barrier死锁。平台保持外部源码只读，在自有EnvWorker适配边界修复函数绑定。对实际env/actor/rollout/runners/OpenPI路径的F821只读检查仅发现这一处未定义名称。

后续续跑 `pr_7ed370a7e4ae4e909628` / PPO `job_e24d1419aee14f35912b` / `run_dd61a1646aec451ebd66` 于19:16 UTC启动，镜像`sha256:640d2e53ea2820eb69da4de9766f0876d2572993e77aea1df74035f54e893cd4`。镜像包含bootstrap兼容、独立评估完整模型配置以及SFT/PPO共享串行加载。回归先复现原bootstrap NameError，再核验首个有效timeout奖励与输入不被修改；四套实际eval配置验证通过。完整后端最新132项通过。

取消接口不再吞掉Docker失败；ebd删除失败返回503，控制面保留活动状态和GPU占用。已终态Job禁止取消改写。子worker OOM而主进程仍运行时会被识别并清理。独立`make images-rlinf`实际构建通过，使用部署配置里的HF端点；基础镜像及只读OpenPI源码依赖已写入README。

`run_dd61a1646aec451ebd66` 已实际完成1轮rollout（62.02秒），随后训练批次整理在`process_nested_dict_for_train`索引越界，未执行optimizer更新。定位到LIBERO不提供可选valid_step_mask，而bootstrap单独附加一行，形成[1,B,10]与T步批次不一致；自有actor边界正在针对真实轨迹构建器做兼容回归，不改奖励或损失公式。

有效步mask兼容已在镜像`sha256:3083a1987b3749b022c23f4fcd9dfd131e9084f03f3d882a330b6d2436862e50`通过真实trajectory builder/shuffle回归，仅清理bootstrap-only全true元数据，保留full-length/false masks与奖励排序。续跑`pr_21e9d27614414c17a5e6`于19:36 UTC启动，仍复用原成功SFT。

独立CPU加载对比采用同一完整3,619,512,081参数模型、8CPU/24GiB、无GPU：2线程80.151秒，48线程88.152秒，峰值RSS均16.307GiB，均正常退出。仅为单次有序对比，未据此宣称通用提速或更改线程默认值，临时容器已清理。
