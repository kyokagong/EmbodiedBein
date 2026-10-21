# UFO 每卡 2048 环境：六小时续训实验

2026-10-19，时区 Asia/Shanghai。状态：**4×2048环境、每卡Replay262144的长区间正在运行**。用户已确认保留上一轮模型和全部优化器、初次重新采集Replay；后续段恢复完整Replay。预算截止今日23:07:21 CST，禁止Git提交。

[实时状态与录像索引](evidence/2026-10-19-ufo-native-2048/STATUS.md) · [全局环境步验证曲线](evidence/2026-10-19-ufo-native-2048/validation-trend.png)。当前长区间TrainJob `trn_44b8b94a98034ce7aa99` / Run `run_3709a8fc84e645f38c3b`，目标14992累计更新 / 5242880 global环境步，已产生真实更新。六小时尚未结束，此文不把计划写成最终训练结果。

## 初始冻结方案与容量探测

此方案遵循设计25的原生 Trainer、显式 fresh Replay、独立 Job/Run 和分段保存/评估路径。目标是检验较长环境步热身的效果，不把运行成功当作正式验收。2000万按 UFO `num_env_steps` 的累计全局 transition 数计，区别于 agent 更新。源模型有10000次更新、655360 local / 2621440 global 环境步；新数据量从该计数扣除，配置变更在曲线上明确标注。

| 项 | 冻结值 |
|---|---|
| 源 TrainJob / Run | `trn_76bf6191e61e465bb1cf` / `run_f9b935bdd2ad4db2a053` |
| 源模型 | `mv_83ddd78dabd3480a8172` |
| GPU / 环境 | 4张4090，整卡独占；2048/rank，共8192环境 |
| Replay | 524288/rank，共2097152 transitions；256时间片，保持原1024环境实验的时间覆盖 |
| 恢复 | 模型、target、normalizer与5个Adam原样加载；旧Replay不复制；仿真/RNG/latent重新初始化 |
| batch / LR | 每卡1024，global4096；原生各优化器LR不变 |
| 更新 | 原生 `update_agent_every=1024 local`、每次16更新；每轮采样2048，因此每条新transition的更新密度为原实验一半；不自动放大LR/batch |
| 重新采集 | 每卡至少10240 local新环境步；严格按原生checker/readiness判断后恢复更新 |
| 源码/数据/接口 | 沿用源实验的UFO、RLinf、资产、train split、RPO23DoF/50Hz及Profile快照；holdout不挂到训练 |
| 任务镜像 | 独立 `eb/ufo-native:rpo-train-v4`，digest `sha256:0d2713348b82ae1a8fae4dc627963c0f76e9d6a9fa53935b673d88d00a76a118`；基于冻结v3，只有平台恢复适配更新 |
| CPU / RAM / shm | 32核 / 80GiB / 8GiB；组合实测后决定是否放行长区间 |
| 总预算 | 首个GPU训练Job提交起6小时墙钟，含新配置探测、排队、保存、prompt/export、评估与录像；实际训练秒数另报；不复用上一轮过期预算 |
| 存储 | 本轮新增上限200GB；源checkpoint、已评模型和全部媒体保留 |

源原生目录：`/data/kk/embodiedbein/runs/run_f9b935bdd2ad4db2a053/outputs/native/`。源manifest原字节SHA256 `4f15da8de637a3e674b5684249f303acd97e439c5f2bea1722eec1ef3d12d2f6`；receipt `43cc55ae85e2f1f355bbff34df401644fc1de0c88b4e7ad7ba6afb0951396894`。预览/提交/排队准入仍绑定同一不可变source Artifact及这两个hash；不以放松旧完整恢复校验实现扩容。

## 阶段与决策

1. 首段新增112更新，累计10112；源码checker推导终点681984 local / 2727936 global。真实4×2048训练、全部rank Replay保存、原生CPU回读、5个Adam实际step、前后参数hash、source不变与资源释放均须核对，再完成双后端录像快评。此段验证新配置完整链路，不要求短热身已学会站立。
2. 后续固定累计global终点：5242880 → 10485760 → 15728640 → 20004864；每段依据最新checkpoint实际计数推导累计更新目标。若首段计数完全符合，预期对应15008、25232、35456、43792更新；实测不符则先定位，不能伪造或沿用错误目标。
3. 此后以5242880 global为快评间隔，在六小时剩余预算内推进；最后不足完整间隔时按实测吞吐保守缩短，预留保存和最终评估时间。每段独立Job正常SAVE_AND_STOP，全部GPU释放后评估，不预提交未经本段证据审查的下一段。
4. 本轮是用户指定的长热身实验，2000万之前的低成功率/跌倒只记录，不套用上一轮10000更新的提前终止门槛。2000万处必须明确质量趋势；剩余预算用于同配置观察，不自动调参或追加时长。每个决策记录实际质量、训练窗口及是否仍在授权观察范围。
5. 非有限模型/优化器/指标、OOM、collective失败、checkpoint不完整、数据/来源失配、媒体缺失或资源不安全时停在当前可保全边界，保留上一份完整产物；不将缺证据当作继续依据。不承诺取消会保存运行中Replay。

每段的墙钟硬上限不扩大源Job的5400秒；六小时通过多个有界分段实现。末段需覆盖剩余时间内可完成的保存/回读/评估，不能等总预算到点才强杀丢失未保存状态。

## 评估与证据

每个checkpoint重新执行原生RewardInference：rank0训练Replay、每任务16384样本、seed71000、n_inferences=1；使用相同站立/0.3m/s主任务与0.7m/s诊断、mjlab和MuJoCo smoke-v2，全部6case录像。重新计算latent造成的任务差异与权重变化分开解释；不以推理prompt样本数当训练量。沿用前轮冻结阈值作validation，正式acceptance标准仍待用户后续确认。

保存原生CSV、训练窗口分析和曲线、每卡显存/利用率、CPU/RAM/shm、吞吐、完整checkpoint及5个Adam实际计数、逐case报告/视频/索引/hash。新实验与旧实验不拼成未标注的同配置曲线；没有成功例时如实标注无成功录像。

## 已执行检查

- 08:50–08:55 UTC平台API/ebd健康，四GPU各24080MiB free、lease为空；系统可用内存约106GiB、磁盘约1TB。Doctor的旧Vulkan ICD提示不替代已验证的EGL/NCCL映射。
- 隔离CPU容器调用同一冻结UFO builder，构造2048环境/524288 Replay，CUDA目标agent配置与源`config.json`逐项完全相同；原生EveryNStepsChecker核对fresh采集起点667648和首段2727936 global。未分配GPU或训练。
- 扩容/完整恢复/来源绑定/收集证据/ToolBus后端回归：161通过、1跳过；相关静态检查通过。实际任务镜像Python3.10恢复/失败检查40通过，fresh复制独立行为检查通过；前端13检查、类型检查与实际部署构建通过。
- 第一轮复核：原生算法与数据/契约保持冻结；配置变化和fresh Replay已单独声明，旧恢复路径继续严格拒绝失配。

## 首段执行

首段TrainJob `trn_ec0504fe3df34bb29527` / Job `job_92ba28a9f3084167aef3` / Run `run_59969e7f999a46eb8b4b`。17:07:21 CST提交，六小时预算截止23:07:21 CST（15:07:21 UTC）。[预算起止](evidence/2026-10-19-ufo-native-2048/session.json)、[首段预览](evidence/2026-10-19-ufo-native-2048/u10112/train-preview.json)、[提交结果](evidence/2026-10-19-ufo-native-2048/u10112/train-submission.json)已保留；源checkpoint、模型和全部Adam保持冻结。

GPU实测、后续阶段、媒体复核与最终训练结果将在实际执行后补充；当前不宣称质量改善。

## 第二轮复核：容量修正

初始Run成功完成10112累计更新、681984 local / 2727936 global；五个Adam实际step均10112，四个初始模型参数hash均与源10000更新模型一致，源manifest/receipt原字节hash未变。Replay每rank为256槽、cursor13、full=false；所有rank原生CPU模型/Replay回读及全状态有限性通过。[独立核对](evidence/2026-10-19-ufo-native-2048/u10112/train/verification.json)。

实测各卡进程峰值23692/23694/23692/23694 MiB，物理空闲最低约388MiB；该峰值在内评后训练窗口仍接近，不能只看Torch的6.33GB allocated/6.93GB reserved就忽略物理仿真等非Torch分配。此配置不进入长区间，保持为已完成容量探测，原镜像/产物不覆盖。初始双后端6case均fall，模型`mv_aafc3d17f6d84dc7ab97`；录像和图表原件已由平台登记，工作区媒体核对索引待完成。

仅做一次有界修正：**仍4×2048环境，Replay恢复原实验262144 transitions/rank**，128个时间片、每环境2.56秒；batch/LR/原生16更新不变。预计每卡减少约1.8GiB，仍须真实112更新/保存/回读/独立评估验证；最低实测物理余量要求1536MiB。不会调物理增益、网络、精度或原生loss，不以反复OOM搜索上界。

修正使用新v5镜像`sha256:7217526a893b2293817b0210117f0b9427be055a63103ecd6d1682be23648af1`，v4保持不变。恢复来源仍是用户指定的**上一轮10000更新**完整checkpoint；524288容量的112更新只用于探测，不与修正实验拼接。首次仍目标10112，后续环境步节点不变；六小时原截止23:07:21 CST不延期。[修正协议](evidence/2026-10-19-ufo-native-2048/protocol-v2.json)保留与初始协议的区别。

## 第三轮复核与长区间放行

修正配置的首次112更新：TrainJob `trn_e9180da15f38417c873e` / Run `run_ee92f86fd04e4da088a0` / 模型 `mv_dc192cc4539244939fbb`。每卡峰值21516–21518MiB，最低物理余量2562MiB；五个Adam实际step10112、原始source模型/receipt/manifest未变。[checkpoint核对](evidence/2026-10-19-ufo-native-2048/b262144-u10112/train/verification.json)、[六例录像与HTTP/Range/逐帧校验](evidence/2026-10-19-ufo-native-2048/b262144-u10112/README.md)已保存，主任务0/4、全部6例fall。

必要的完整Replay恢复验证新增512更新：TrainJob `trn_044d4348319649a9ba5b` / Run `run_8ca3c165fbf3444f8da9` / 模型 `mv_9ccdd4b2887b47348112`。累计10624更新、749568 local / 2998272 global，5个Adam实际step均10624；全rank Replay恢复、原生CPU回读、全状态有限性和指标完整，恢复后`replay_update_after=665600`未重置，未暗中回退fresh采集。[独立回读与资源](evidence/2026-10-19-ufo-native-2048/b262144-u10624/train/verification.json)、[六例录像与校验](evidence/2026-10-19-ufo-native-2048/b262144-u10624/README.md)已保存；主任务仍0/4，mjlab站立/0.3m/s存活35/51步，MuJoCo39/39步，未据短热身宣布质量合格。

完整恢复峰值22550MiB/rank，最低物理余量1530MiB。原探测1536MiB门槛**未达到**，首次运维检查因此停在已保存边界；原始失败标记保留。该1.5GiB是本次保守探测设置，不是设计25或用户指定的硬约束。基于四卡独占、真实完整恢复及512更新稳定、所有checkpoint/媒体完整，长区间采用显式`fits_tight`准入，物理余量下限设1024MiB；**不修改模型/数据/算法或validation/acceptance阈值**。[运行准入调整记录](evidence/2026-10-19-ufo-native-2048/admission-profile-v3.json)记录了6MiB差值、原门槛失败事实和继续依据。

因此原global节点仍为5242880 → 10485760 → 15728640 → 20004864，但加入必要的恢复验证后，正常按节点推进的更新目标变为14992 → 25216 → 35440 → 43776；实际以每段source计数重新计算。之后约每5242880 global快评，预算末段保守收缩。

后续运维客户端以独立CPU维护容器`eb-ufo2048-operator-20261019`持久运行，未分配GPU、未挂docker.sock；只调用既有预览/训练/评估API，全部真正训练/物理评估仍由Job/Run/ebd执行。每段先保存、核对5个Adam和source、图表/原始时序及全部录像，再记录决策并提交下一段；失败不自动重训。七项计数/末段预算/失败/到期仅取消本Job检查通过，现有两份checkpoint的完整闭环实测已通过。[客户端冻结hash](evidence/2026-10-19-ufo-native-2048/operator/manifest.json)及[源码](evidence/2026-10-19-ufo-native-2048/operator/ufo2048_supervise.py)保留可复核实现。

本轮已保留18段真实原片（容量探测6段、修正首次6段、恢复验证6段）；每组18个索引文件HTTP大小/hash通过、6视频Range206及逐帧解码通过；原生训练分析文件的HTTP入口亦已核对。所有原件留在受管Run，工作区副本索引可访问。正式acceptance、真机与最终六小时质量结论尚未执行/产生；没有Git提交。
