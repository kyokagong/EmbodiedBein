# P-BFM / RPO 四卡每卡2048环境的派生配置

本文件补充[设计26](26-pbfm-rpo-rlinf-integration.md)第7–9节，落实用户2026-10-21的新目标：真实RLinf四卡、每卡2048环境，不限总时间，跑通500-step档。沿用已发布23DOF深度接口、P-BFM算法及现有Job/Run/ebd，不改已发布接口、数据或旧Run。500档按原生16次联合更新向上对齐为512；不自动进入1008或正式长训。

## 冻结训练语义

| 项目 | 派生配置 |
|---|---|
| Workers / GPU | W=4，每卡一个真实RLinf Worker，共置完整原生Workspace，原生DDP同步 |
| 环境 | E=2048/rank，合计8192；保持七类training_mix与400/50Hz |
| 主Replay | C=131072/rank，64个采样时间槽、1.28s；最新29/64槽可容纳35龄历史，容量边界padding仍沿用原生 |
| batch | 1024/rank、global4096；不把8192环境当作learner batch |
| 学习率 | 原生`lr_scale=1.0`，不随W/E自动缩放；实际各优化器LR写入有效配置 |
| 更新 | seed=10240 local transitions；checker间隔1024，E2048下每个有效采样轮16次联合更新 |
| 更新/采样 | 每rank每2048新transitions做16次batch1024更新，约8个样本读取/新transition；相比E1024基线减半，明确保留此差异 |
| 目标与决策点 | 累计112 → 368 → 512；每段完整保存/回读、独立快评/媒体、记录结论后才继续 |
| 数据 | 复用已验证`ds_5bf835c98ccd4da4967c`，只挂载train；validation沿用walk1，保留集不用于调参 |
| 起点 | W4/E2048全新冷启，不能把原W1/E1024改变身份后冒充恢复 |
| 仿真执行布局 | E2048派生producer使用MJLab原生eager：构建前`wp.set_mempool_enabled("cuda:0", False)`并保持worker生命周期；每次main-env采样前归还闲置Torch缓存；不改nconmax512/njmax4096、物理函数或更新相位 |

原生联合更新只计`agent.update`成功次数，不把8个优化器或4个rank相加。恢复保持W/E/C/batch/LR/镜像/源码/数据/接口/相机完全一致，恢复模式仍为`pbfm_native_state`，不宣称exact。

## 从原生循环推导端点

Workspace在采样轮起点L完成采样和更新，段末保存的是后继采样边界。首次更新必须满足`L > 10240`；恢复checker从源L重新起算，因此恢复第一轮先采样而不更新。

令E为每rank环境数、S为恢复源local_time（冷启为0）、N为本段新增联合更新数，首个更新轮为`max(S + E, (10240 // E + 1) * E)`，保存local端点为`first + (N // 16) * E`，global端点乘W。必须用源码checker模拟验证，不能只测试同一公式的副本。

| 累计更新 | 源L | 首更新轮L | 保存L | 保存global |
|---:|---:|---:|---:|---:|
| 112 | 0 | 12288 | 26624 | 106496 |
| 368 | 26624 | 28672 | 61440 | 245760 |
| 512 | 61440 | 63488 | 81920 | 327680 |

完整512冷启单段端点不同于上述分段恢复端点，不能混用。课程仍从0按global transitions运行、间隔19200000；本轮不代表覆盖全部地形难度。

## 有界运行与能力状态

先验证四卡真实rank/UUID/EGL/NCCL与整组故障退出，再以完整W4/E2048的112段验证加载、首个update、DDP、Replay、checkpoint和保存回读。两环境A0不能替代2048环境容量。未取得证据时普通模板保持关闭，operator使用精确限制的实验训练入口，仍沿用Job/Run/ebd；产物保持diagnostic/incomplete，不生成假TrainJob或质量合格状态。

A0不得把合法全远距离输入当作传感器故障：maxcrop的无命中/超距本就映射为2.5m和uint8 255。保全随机reset/step的真实输入、原始射线和姿态后，在同一环境追加固定直立姿态正对照，不修改训练配置/数据/种子。使用实际编译场景、物理qpos和包含安装扰动的实际world rays进行CPU/GPU原始Euclidean range比较，保持geom groups(0,2,5)、static开启与bodyexclude=-1。每环境固定16条下半幅射线，正对照必须至少4条确定静态可碰撞命中、距离跨度>0.1m；冻结atol0.01m、rtol0.001，距量程边界0.02m内不作确定命中证据，不因失败放宽。随机全远图只有在几何对照一致且该rank正对照通过时才可接受；两边均无命中本身不代表传感器健康。同步核对原生预处理与37帧/8输出/0或1帧延迟，保留逐rank原始数据、比较报告和RGB的完整索引。这是修复A0随机画面unique检查的语义错误，不改变已发布深度接口。

实验入口仅接受W4/E2048/C131072和112/368/512端点。112必须冷启；368只能恢复成功112段，512只能恢复成功368段。后续段要求对应源模型的完整独立validation报告和已验证checkpoint，不接受任意路径、跳段、改镜像/配置、空Replay或权重warm start。保留现有runtime及W1 segment112诊断行为。

每段Job冻结独立超时与资源预算，无自动重试；NaN/Inf、OOM、collective失败和不完整保存均失败并保全现场。512以内允许早期task成功率为0，不能据此宣称质量合格；基础设施/媒体缺失必须先修复，训练数值失稳或冻结硬停止条件触发时停止调整。

## 容量、证据与出口

逐卡准入，不能把四张24GiB相加。CPU元计数得到在线参数320300123、targets280226884，FP32权重+targets约2.237GiB、梯度1.193GiB、Adam两状态2.386GiB；合计5.817GiB尚不含activation/DDP/Replay/专家/仿真/渲染。旧E1024构建峰9.621GiB加首次Adam/grad、DDP bucket和foreach临时持有约14.9GiB，仍未覆盖E扩容和激活峰值。首次受控W4/E2048请求每卡23.5GiB整卡reservation，verdict为unknown，明确它是准入保留量而非已测峰值或确定上界；低于该reservation的显存仍no_fit，实际更新/保存测量通过后才能固定容量画像。

有限compact FIFO最旧35槽缺少更早帧，原生采样会clamp；无reset长轨迹满64槽时仅最新29槽可完整重建35龄历史。增加C不消除所有容量边界padding，真实reset的历史填充另算。本配置保留原生存储/采样语义，不暗加halo或排除边界样本；报告容量与reset边界影响，不能宣称所有Replay样本均全历史保真。

沿用每rank8CPU、Job RAM80GiB/shm8GiB作为首次预算，结合Doctor和运行峰值复核；不占用其他Job。记录DDP stage覆盖、参数同步、逐卡物理映射、GPU/RAM/shm、采样/更新/同步/保存耗时及吞吐。禁止反复OOM搜索上界；若24GiB无法承载，不缩环境或网络冒充达成目标，先据首错和实际峰值调整获准的存储/执行布局并派生新配置。

初始化与仿真间允许显式归还闲置Torch缓存：在原生Workspace完整构建、模型与RNG状态取证后，先设备级同步，再调用`gc.collect()`和`torch.cuda.empty_cache()`，保持所有live模型/target/DDP/Replay/仿真引用及CUDA Graph不变。保留逐rank构建前后、归还前后及失败时的allocated/reserved/max/free/total；不重置峰值，检查归还前后参数与RNG不变，取证失败不得掩盖训练首错。此执行布局变更须派生producer并重新冷启，不能把失败Run当恢复源；实际更新、Adam展开与保存容量仍须完整112段验证，释放量不预设。

若首次图上传失败且归还初始化缓存不足以解决，E2048采用原生eager派生布局。使用Warp公共分配器开关使MJLab自身的`_should_use_cuda_graph`为false，step/forward/reset/sense沿用与捕获体相同的原生函数；不改接触/约束容量、求解器、相机或模型。该开关只控制Warp Python数组分配器及MJLab图选择，不宣称禁用了进程全部CUDA内存池。记录实际分配器状态及四个graph均为空；后建priority-eval环境也须继承该模式。E1024保持原行为。

该布局允许每次main-env采样前设备同步并归还闲置Torch缓存，保留所有live模型/梯度/优化器/DDP张量，逐次记录allocated/reserved/free与不重置的峰值。包装原生step须逐次原样转发动作、返回与异常，不增加reset/采样/更新，不在每步重复完整模型hash。理由是碰撞临时数组由Warp分配，不能依赖Torch在其分配失败时自动回收闲置缓存；此为采样与学习间的显存交接，不改变训练算法。仍须用实际112段验证全部联合峰值，若不适配则停止取证，不能以再次OOM搜索代替分析。

E2048 eager的凸碰撞临时数组按本次实际候选数分配：读取已完成broadphase的单值C，要求`0 <= C <= min(d.naconmax,d.naccdmax)`，将17个临时数组的首维改为`max(1,C)`；其余维度、Data容量、候选IDs、原kernel输入/launch维度及顺序全部保持。负值、非单值或容量溢出直接失败，不截断候选。CUDA graph、capture及CPU路径保持原始分配且不进行D2H读取；只在明确CUDA eager非capture路径生效。以精确依赖文件SHA与替换数锁定受管patch，新producer保存patch收据和有界的实际候选/行数证据，不修改外部仓。

此行数由原生索引界推导：每个候选至多一次atomic取得ccdid，因此`ccdid < C`；高度场的多个prism复用同一ccdid行。保留零候选时原launch序列，不能用capacity裁剪或碰撞重排替代。CPU验证0/1/普通/接近容量/非法计数、真实factory形状、未改kernel及graph路径不读回；GPU仍经新producer A0和完整112，随后按原112→368→512。不能把逻辑分配字节当成图池物理测量或借此预先宣称容量通过。

导出必须在首次训练前修复已发现的真实successor动作batch形状问题，冻结同一个producer镜像。每段模型独立导出实际完整depth ONNX、32step配对验证，随后固定32rollout validation，保留真实逐case、RGB/depth/轨迹/图表及鉴权/Range证据。通过运行流程不等于正式acceptance或真机可用。完成500档的出口为：四rank各累计512真实更新、完整保存与派生恢复证据、对应模型/原始曲线/资源数据、独立快评和可访问媒体；质量结论按实测单独报告。
