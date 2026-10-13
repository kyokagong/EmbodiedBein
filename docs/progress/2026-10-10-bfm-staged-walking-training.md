# BFM 行走闭环进度

设计：[19](../design/19-bfm-staged-walking-training.md)；计划：[执行计划](../superpowers/plans/2026-10-10-bfm-staged-walking-training.md)。本台账只记录实际证据；长训总预算等待用户答复。无教师，无Git提交。

## 已核实根因

源Run `run_d8c1001d92f4450a85eb`的1000更新checkpoint实际回读：`z_pool.shape=[8192,256]`、nonzero=0、count=0、position=0。真实worker新增断言先失败于“learner mixed-z never reaches the rollout pool”，修复后worker/loss/replay/inference检查通过。固定上游FBcprAuxAgent.update写入z_buffer，原接入遗漏该调用。

## 实施台账

Task 1：mixed-z写入已有model buffer，输出已有loss细项和D logits、三项actor价值/贡献、B批次有效秩、动作饱和、grad裁剪、aux各项、replay实际年龄/抽样比例；1/8更新worker真实检查通过。补rollout池计数随trajectory返回，用于真实多rank同步证据。

Task 2：平台支持1/8更新，checkpoint和runner横轴保留采样轮数，真实learner目标单独冻结；每Run新增最多1000，累计平台上限100000不是用户预算。恢复counter和轮数强校验；新progress回执在发布/模型登记检查，终止目标独立于compatibility指纹。focused预算/恢复/数据38项通过，计数发布器先失败后修复。实际GPU完整恢复16→24已完成，见下文。

Task 3：派生validation61000–61015和acceptance62000–62099，API/adapter/collector共用冻结case绑定；新套件共用v3物理协议。validation不请求资格，acceptance仍80/100标准，旧Suite不修改。平台聚焦回归70通过；最终任务镜像10项并行/桥检查通过，覆盖真实视频解码和逐case隔离。

Ruling：本次直接授权继续优化，沿用用户要求的顺序执行；不新增重复设计审批。用户尚未确认的长训预算保持挂起。源目录/历史镜像只读；不新增隔离worktree去搬移已有接入工作，局部增量修改并保留本次计划与证据。

Ruling：`max_epochs`保留RLinf采样轮数以复用已验证恢复生命周期；更新乘数只取1或8，真实update目标/optimizer回读负责验收。非constant scheduler未纳入该配方。

## 当前状态与未执行项

v10单卡16更新/完整恢复到24、v11独立16 case快评、四卡16更新组合探测及其16 case快评均完成。学习语义对齐的新配方尚未实现；当前没有合格行走模型。长训总预算尚未答复，未启动新的1000更新或多配置长训，acceptance保持未执行。

## 三轮复核状态

①数据/codec/单位与上游latent闭环已检查，追加发现权重初始化/normalizer/专家连续提示与学习配方差异；原数据资格不晋升控制器资格。②预算/恢复/并行API边界测试及真实16→24恢复、三个采样rank同步通过；四卡训练峰值监测缺失，下一版扩容前须重测。③真实短训曲线、两次完整16 case快评及所有32个录像解码完成，逐case首尾图已检查；两次均0成功，停止调整，不能宣布行走/验收成功。

## 独立审查与修复

只读review指出两项important，无critical：新acceptance可在不录像时通过、保存/恢复只核对updater自报计数。已增加新profile必须录像、任务逐帧解码并与真实物理步数核对、collector校验解码回执及视频hash；保存和恢复核对六组Adam所有参数真实step。worker错误optimizer-step和真实MP4解码/损坏拒绝均先失败再修复通过。保留v9预检查镜像，新任务使用v10 `sha256:7f4a8eab63778ab2114b204db80890d893ee3861521320a821188ad55eb26367`。

完整后端回归460通过/4跳过（Torch与imageio专项在任务镜像补跑）；v9任务镜像9项并行/桥检查通过，7类runtime网络/codec/loss/replay/worker/inference/command检查通过；修复后的optimizer与视频检查通过。v10及部署完成后才启动API短训，不覆盖历史模型。

Ruling：实际查询发现v9部署已发布validation/acceptance v1，未含required_video；不原地修改。派生v2套件（同seed，媒体必需，validation planned_cases改为实际16且不设80门限），评估派生v11镜像，已运行的v10训练/恢复固定原digest。

## 单卡运行证据

TrainJob `trn_9921b8148dcb4ce9bbee` / Run `run_98be93a30d2f4076aeb6` succeeded：64env、4控制步/轮、batch128、8更新/轮、2轮=16联合更新。保存回执六组Adam计数均16，replay_total_seen512，z_pool_count2048；完整恢复候选available=true、1.335GB、trainer_state。不是行走质量验收。恢复至3轮/24更新按相同配方完成，详见下一段。

恢复 `trn_95b44e7fab7941d3b12f` / `run_39d5fcd14f074acea7ed` succeeded，发布模型 `mv_263d4617c4d14989a8eb`。独立DCP回读168个Adam step全部24、312权重全有限、4个targets与online不同；replay768/池3072。原512行replay逐tensor保持一致；新增256行行为z全部匹配源mixed-z池，源权重仍匹配初次发布hash。该证明超过“只收到非空pool”，核实实际使用。恢复回执loaded，环境/rollout未持久化，不称exact。

84项真实指标每采样轮记录；实际learner横轴8/16/24，只有3点，不生成50点平滑。训练动作饱和20.65%→39.18%→48.93%；FB总loss主要由100倍orth项贡献，裁剪前FB norm仍大。B的批次有效秩16.08→20.64→22.12，不代表全局表征或覆盖已合格。没有完成episode的训练段缺reward/episode_len属真实缺失。见[曲线](artifacts/2026-10-10-bfm-staged-walking/short-training-curves.png)、[全部统计](artifacts/2026-10-10-bfm-staged-walking/short-training-statistics.json)、[恢复核验](artifacts/2026-10-10-bfm-staged-walking/checkpoint24.json)、[行为z证明](artifacts/2026-10-10-bfm-staged-walking/latent-proof.json)。

## 首次独立快评与决策

Job `job_26d5ae2bcce94e918929` / Run `run_50e0897659f1431d89e2` / Report `ev_66e4fe9e1e02426fa45d` succeeded，v11 digest `sha256:b48680865d7c4a2297000005e0cabd074cb591eb98695660767c3f7b9d2cfb92`。4独立GPU进程各4env，共16冻结validation seed；耗时277.66s。全部16有效、0infra/NaN、0成功、15硬限位（14位置/1速度/0力矩）和1跌倒；326真实控制步，median存活0.39s；线速度RMSE0.746908m/s、yaw2.535121rad/s，评估动作69.63%饱和。不是正常行走，qualification=not_requested。

全部物理trace逐步重算fall/硬限位/速度/yaw/z范数与hash；16个1920×1080/25fps视频实际解码184帧，与控制步数对账；查看全部16case首尾图，未出现成功录像，不补造。[评估证据](artifacts/2026-10-10-bfm-staged-walking/validation-24.json)、[逐case首尾](artifacts/2026-10-10-bfm-staged-walking/validation-24-all-cases.png)、[结果图](artifacts/2026-10-10-bfm-staged-walking/validation-24-results.png)。

Decision：**stop_adjust / wait_budget**。诊断/恢复/媒体链路完整，允许有限四卡组合探测以核验3采样rank同步与资源；不据此扩大当前bringup长训。设计19记录与固定上游的batch/LR/FB pessimism/relabel/clip/专家连续rollout差异，长训须先派生学习语义对齐配方并验证；用户新预算尚未答复。最终62000–62099 acceptance保持未执行，未用于调参。

## 四卡有限组合探测

TrainJob `trn_c3aa963b85f042ad8a68` / Job `job_7c93aa7977e74799b760` / Run `run_3b7e8b63e16f4d519aaf` succeeded，ModelVersion `mv_656c69b96a8242919b8d`。v10同digest、同batch128/8更新配方新初始化；物理GPU0 learner，1/2/3各1024独立env及rollout，合计3072env。00:39:39.995384–00:41:57.214651 UTC，137.219s，2采样轮、16联合更新、24,576新transition；采样使用比每轮0.083333，是旧1000实验64倍，但仍远低于上游16。

独立DCP回读168个Adam step全为16、312权重有限、4组targets已更新、池2048。第一轮rollout池0是初始化，第二轮全部trajectory行的池min=max=1024，证明三rank同步；learner轮末1024→2048。88项指标各2点，没有50点平滑；训练actor饱和15.82%→40.98%，行为action饱和0.09%→10.26%；训练完成episode计数2→14、平均长度3→7.5控制步，success_once均0，不作独立质量验收。两个round的actor_training合计2.239s、采样5.193s、同步2.019s，分别是嵌套时间，不能相加代表整个阶段；含启动/保存/发布墙钟约0.1166更新/s，不能外推长训稳定吞吐。见[训练曲线](artifacts/2026-10-10-bfm-staged-walking/four-probe-training-curves.png)、[全部数值/资源限制](artifacts/2026-10-10-bfm-staged-walking/four-probe-statistics.json)、[checkpoint核验](artifacts/2026-10-10-bfm-staged-walking/checkpoint-four16.json)。

GPU监控timeout提前结束于00:40:43，仅覆盖该Run加载，未覆盖00:41:31后的learner更新；训练GPU/RAM/shm/I/O峰值**未取得**。不能把加载样本或评估的显存当作四卡训练上限证据。组合可执行已验证，下一配方扩容前必须补足峰值监控；本阶段不再追加同配方容量试错。

## 四卡probe模型独立快评

Job `job_aa458c9043da4267ab4a` / Run `run_94b6e0a54e304d17a539` / Report `ev_5ed0dffa3fce4ee2a7f1` succeeded；plan hash `sha256:15ec3825fd17f68f130c8adb0a37e128daec934428cebee42f6b2c2df500ffb8`，v11、相同validation_v2，4GPU×4env。模型checkpoint hash `sha256:b5e1e974964fac0497d8061be9e054ff97525630ce9fbb775a013f49941d7020`；评估来自本段16更新模型，不拿前一段24更新评估替代。

| 指标 | 单卡16→24模型快评 | 四卡16模型快评 |
|---|---:|---:|
| planned / completed / valid | 16 / 16 / 16 | 16 / 16 / 16 |
| invalid / infra_failed / NaN | 0 / 0 / 0 | 0 / 0 / 0 |
| 成功 | 0 / 16 | 0 / 16 |
| fall / hard_limit | 1 / 15 | 13 / 3 |
| 实际控制步总计 | 326 | 134 |
| median存活（s） | 0.39 | 0.19 |
| 线速度RMSE（m/s） | 0.746908 | 0.785854 |
| yaw RMSE（rad/s） | 2.535121 | 2.651056 |
| 实际action饱和 | 69.63% | 78.78% |
| 实际解码视频 / 帧 | 16 / 184 | 16 / 88 |
| 全阶段墙钟（s） | 277.661 | 266.211 |

两模型的初始化后在线数据/topology/更新数不同，表格不作为单因素优劣结论。当前没有成功录像；四卡16模型3次硬限位均速度，0位置/力矩越限。全部trace逐步重算物理指标，所有视频实际解码，逐case首尾图已查看；最多存活13控制步，录像中出现身体倾斜/失稳，没有20秒正常行走。见[证据JSON](artifacts/2026-10-10-bfm-staged-walking/validation-four16.json)、[全部case首尾](artifacts/2026-10-10-bfm-staged-walking/validation-four16-all-cases.png)、[失败图](artifacts/2026-10-10-bfm-staged-walking/validation-four16-results.png)。

5秒采样覆盖两次完整评估Job区间：四卡16模型每卡显存最大观察值8479/8260/8259/8208MiB，利用率观察最大28/34/18/36%；容器RAM一次快照50.68GiB/96GiB。前一评估最大8501/8260/8268/8272MiB。均不是Torch allocated/reserved或瞬时峰值，也不是连续CPU/RAM峰值；16例大量早停和启动成本不能推导满20秒或增加env的吞吐。见[资源证据](artifacts/2026-10-10-bfm-staged-walking/evaluation-resources.json)、[四卡评估资源曲线](artifacts/2026-10-10-bfm-staged-walking/validation-four16-gpu.png)。Doctor终态核实四卡run_id均null、每卡空闲24080MiB，任务自身观察进程已停止。

访问：[24更新评估报告](http://1.94.0.106:31800/eval/reports/ev_66e4fe9e1e02426fa45d)、[四卡16更新评估报告](http://1.94.0.106:31800/eval/reports/ev_5ed0dffa3fce4ee2a7f1)、[四卡训练](http://1.94.0.106:31800/train/trn_c3aa963b85f042ad8a68)。全部媒体绑定源Run/model/plan；完整片位于对应Run `output/video/eval/seed_61000..61015/0.mp4`，例如[四卡seed61000](</data/kk/embodiedbein/runs/run_94b6e0a54e304d17a539/output/video/eval/seed_61000/0.mp4>)。当前新validation/acceptance profile通过API执行；前端新建评估表单尚未扩展这两个选项。

## 下一版必须先解决的学习差异与等待项

固定上游审计新增发现：显式`weight_init`、replay每更新norm、FB专用pessimism=0、0.8 relabel、模块LR/batch/clip、连续专家z提示均未对齐；原本的mixed-z池修复解决真实遗漏，但不能据此称完整原论文复现。Task 5需按设计19派生核心配方并完成真实小样本/恢复/快评，保持RLinf、无教师、不改已发布资产/codec/门限。

train有效窗口实测：8/50/100/250步分别11,922/7,931/5,748/2,210，250步覆盖15片段（9 G1、6官方RPO）；52个train列表条目中2个无有效窗口。原始数据不额外复制/分发。连续提示可从现有数据接入，但不能把全部52片段称都支持250步连续分支；见[窗口审计](artifacts/2026-10-10-bfm-staged-walking/expert-window-audit.json)。

本阶段新增训练总计**40**联合更新（单卡16 + 同配置恢复新增8 + 四卡16），两次快评32case，不含历史1000更新。Decision再次为 **stop_adjust / wait_budget**。不存在达到正常行走的证据；新的长训与最终acceptance未运行。用户此前预算问题仍待回答：候选最多3配置、累计20,000真实learner更新、训练+评估≤24h，逐1000快评、连续3窗口无改善停止。平台100000上限不是用户预算；预算未确认不继续高成本长训。已完成后端460通过/4跳过、任务镜像7类真实Torch行为检查及10项并行/桥测试；最后文档/证据改动经diff检查。
