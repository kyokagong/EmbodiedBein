# UFO/RPO四卡行走排查与后续分段训练

用户明确授权分析0/100、修复并继续分段训练，目标达到正常行走标准；不提交代码。开始时间北京时间2026-10-10 23:31（UTC15:31）。本轮继承既有设计20/21和冻结物理判据，不把旧acceptance参与调参后仍当未见验收。

按推荐有界协议冻结：最多3个配置、累计新增20,000次联合更新（含资格），训练/排障/排队/评估/录像总24h，截止北京时间2026-10-11 23:31；每段最多新增1000次，正常保存/结束/释放四卡→validation16/全部录像→三轮复核→决策后再提交下一段。前2000更新为质量观察热身，之后3个无改善窗口停止调整；不得凭loss下降继续。NaN/Inf、完整恢复/计数错误、必需证据缺失即停止等待证据。每种infra错误仅一次有因修复，task失败不换seed。磁盘本轮新增80GiB硬预算，包含受管数据/模型/输出、任务可写层和本轮新增镜像层；每段后保留输出与退出证明并清理已结束任务临时层，达到预算停止，不再漏算临时层。

进入第一个长段前补齐总采样/媒体上限：最多2,000,000条新增online transition（包含warmup与取消资格，独立评估步数单列），最多600条本轮新录像、每条最长20s。原生1024采样阈值/16更新块、每轮768条真实样本保持固定；expert重采样次数单列、不混作online物理采样。本轮界限包含旧v1的1000更新对照与safe16→24，不自动追加预算。

旧best24验收0/100已用于失败机制分析，原报告不改；仅用validation选择后续候选。最终需要冻结新的未见acceptance版本和100个case，保持20s/1000步、速度RMSE≤.20、yaw≤.30、无fall/NaN/硬限位、至少80/100且证据完整；最终case不得在训练/调参期间运行。

## 已查证事实

- 旧0/100来自best step24，latest step1024独立validation也是0/16；不能描述为1024模型已评100例。
- 全76800条真实replay：next.last_action与行为action最大差0；next.history最后75维与当前state/last_action逐条一致；无相同current/next state的假物理transition，终止比例4.609%。尚未发现动作/history/自动reset对齐错误。
- 当前是bringup配方：batch128、1024宽/3层、10万replay；固定UFO上游默认batch1024、F/actor/critic/aux2048宽6层、replay512万。小配方不等于算法错误，不能未经对照声称唯一根因；旧1024更新仅76800 transition/每env400名义控制步。
- 实现存在明确训练/验收条件差异：训练仅flat/noise=false/RSI、按高度和倾斜终止；官方独立command评估70%flat/30%rough、有噪声/随机化，并硬检查关节位置/速度/力矩。真实训练dq最大60.69rad/s，评估部分关节限速8rad/s；seed61000最新模型19步首先发生关节速度越限，尚未fall。
- 原始train参考物理审计：G1有效7510帧中60帧速度越限，官方有效6134帧中164帧速度越限；位置均未越限。valid只是数据/编码有效，不保证每帧物理限速安全。不会原地改数据valid/split。

接下来先复现并修正训练物理/诊断缺口，派生配方与Profile并重新16→24资格；新镜像/学习语义不得直接恢复旧完整状态冒充连续训练。架构/批量扩容按实测显存单因素资格后决定。C/MJLab、教师和真机不属于本轮解决0/100的必要路径。

## 首轮修正与对照

派生 `rpo_bfm_ufo_agent_rlinf_isaac_4gpu_safe_v2`，镜像 `sha256:adbca4ec9cf899e18da848b7e2a47d2df50d1f8e74e9015655b6deac193b0f99`，Profile `rpo_bfm_ufo_isaac_4gpu_rsi_safe_v2`。保持网络1024/3、batch128、192环境/一learner三samplers、平地无噪声；增加与验收同容差的关节位置/速度/力矩终止、限位不合格train RSI帧有界重采样，以及118列真实动作/目标/速度/力矩/限位诊断。参考过滤只核对关节位置和速度，不声称初态完整接触稳定性已证明；不修改数据有效标志和split。新增trace按时间/环境flatten，与行为episode_id/step_id逐条对应。

独立代码复核发现并修正容量估算和checkpoint收集只认识旧四卡recipe的问题：新recipe也必须校验三个sampler的actor/B/norm/pool哈希与版本，不得无证据晋升。平台测试60项通过/2项因无torch跳过；扩展真实API配方参数化后48项通过/2项跳过；任务镜像聚焦原生行为24项通过，包括真实IsaaclabBaseEnv自动reset→EnvOutput→原生trajectory builder的terminal trace保留和全不合格RSI的33次总draw失败。曾误将平台API测试放进缺少pytest-asyncio/pydantic-settings的任务镜像，47通过/6环境依赖失败；已将这些检查放回平台测试镜像通过，不当作算法故障或隐藏失败。

2026-10-11 00:00前旧配方对照完成 `trn_92c5027f96a84a72b539` / `job_8112549c1fbf4d5b9250` / `run_b11446aba2274ef0a87c`，1024→2024，模型 `mv_180ddc8d0211446880ce`；仅本轮新增1000更新计入预算。三采样器最后广播版本2016完全一致，最终checkpoint2024。独立validation Job `job_b0fd8d31184340cf90b5` 已提交，尚未以结果作继续决策。该结束任务容器退出0的State/挂载/可写层证明保留后清理，所有原始训练输出和模型保留。

最终验收新冻结套件 `suite_rpo_ufo_isaac_velocity_acceptance_v2`：66000–66099，与旧62000–62099及validation61000–61015不相交；spec与旧UFO acceptance逐字段相同，不更改硬限位、RMSE、1000控制步、80/100要求。新case仅最终候选选择结束后运行，不在调参期间运行。

### 旧配方2024决策：stop_adjust

validation Run `run_bab1dcbb28134d39b097` / Report `ev_60594c9ac9f74968b78f`：0/16，13硬越限、3fall、无NaN，平均速度RMSE0.820701、yaw4.820021，总实际步183。相比1024与best24无质量改善，不再继续旧配置A。16/16视频真实解码并通过HTTP范围访问，逐case初态/trace/video和四个export文件哈希全通过。原始数据、时序图及三个复核结论见本轮artifact目录。

三轮复核：(1) RPO 23DoF接口、冻结资产和仅train参考未变，未使用新acceptance数据；(2) 六Adam2024、真实joint_update以及3sampler同步/完整checkpoint回读均通过，environment重建与trainer_state恢复如实区分，未出现NaN/OOM；(3) 16例指标、初态、trace、录像与访问证据完整，但任务质量失败。下一步配置B从头完成safe-v2的16→24资格，不跨旧image/Profile恢复。

第二次独立代码复核补齐BFM评估目录缺口：新派生 `env_eval_<suite_id>_binding_v2` 绑定 `tsk_rpo_bfm_pretrain`，active且物理成功定义明确；保留此前空task/deprecated Profile原样。真实seed/compatible_profile回归从失败到通过；相关环境/API测试21通过，涉及路径Ruff F检查通过。

### 配置B：safe-v2资格

16更新 `trn_7175ce80bcf64dbf8bae` / `job_f19ed54029394dd49778` / `run_bb5367704bc246bc8c77` 成功，模型 `mv_02374e95badc46d686bf`，镜像adbca4…，fingerprint `sha256:f5b2a521243bda9ad31e8ed78510d84bd0953254b7ebaed95152b7148270c0e1`。完整文件hash/六Adam计数16回读通过；原生Isaac写入API会失效body缓存，经只读源码核对，无依据添加额外sim.forward。

真实GPU生成的15轮/11520条trace与replay.train逐条episode_id/step_id/行为动作全一致。692条硬限位（position119/velocity604/重叠31），全部next.terminated=True；总终止870条，无torque越限。实际速度峰值34.0768；修复是识别并终止违规样本，不谎称物理动作已被速度投影或从未越限。诊断脚本最初误读replay外层，已按实际replay.train结构修正后通过，未改训练数据。16更新属于可运行性证据，尚未证明行走。

完整恢复到24已提交 `trn_c1dd5131a9c14d9a83bf` / `job_adce521ef07b49b5b2ef` / `run_7afe305a163544f59eb7`；仅新增8更新。新旧Profile/fingerprint不同，不接受跨旧v1的完整状态恢复。API实际新acceptance Profile为active且task正确，原空task/deprecated行仍保持原hash。

24更新已成功，模型 `mv_5d9f639f9eb64051bc3d`，完整hash/六Adam24回读通过、恢复fingerprint一致；累计12288 online transition。恢复后新增768条trace与replay环形位置逐条对应，21条硬越限全部terminated。资格快评 `job_fb49d4cbe2d34050b5f3` / `run_dc4fac32d4884921b30e` / Report `ev_e81aa9bada7248d5ab3a`：0/16，13fall/3hard、无NaN，136实际控制步，汇总速度RMSE0.970536、yaw2.606977；16录像/trace/初态hash和实际导出golden全部通过。无成功录像，不称正常行走。完整指标、更新/墙钟、逐例图保留在artifact目录。

按计划补新checkpoint的两clip held-out tracking：`job_30d1a60b1bf3429bb05d` / `run_6afcb67064f14bd0ad6b`，只读validation参考，不更新norm或训练；待结束。为确认修改后任务生命周期还可靠，另提交 `trn_5e5ab11e17974f7c8cc0` / `job_a1c6c385028f4a6bb560` 有界24目标取消资格，计划在warmup取消、允许无新checkpoint，原16/24完整产物保全。API合法1000更新普通提交实际返回400 `UFO training requires smoke_passed evidence`；未提前放行长训。

上述held-out诊断已正常退出0/OOM=false：dance7步hard_limit/q RMSE0.456017，walk10步fall/q RMSE0.533309。两例原始trace与视频hash/真实帧解码和CAS媒体Range206通过；共用scene_seed63000，不能称两个独立随机场景；official RPO原始split没有held-out，如实保留unavailable标记。全为失败录像，仅证明原生B提示→actor→RPO物理/录像可执行，不证明跟踪质量。

取消资格实际启动 `run_4c756781324f4ab9bdb9` 后，通过API取消为Job cancelled/Run aborted(cancelled)，无模型/Artifact/半写checkpoint发布；ebd已移除任务容器，未保留可读Docker State，不能捏造其数字退出码。Doctor证明四卡均释放、每卡空闲24080MiB；原16/24完整模型保留。未进入真实update区间，新增joint更新0；取消前采样量若未落原始点则仅按冻结24资格≤12288的上界计入采样预算，不伪造精确值。

资格三轮复核：(1) 本体仍RPO23DoF/v1 codec和冻结原始train数据，仅新learning Profile改变物理终止/RSI筛选，UFO原生算法/八aux/无教师不变；(2) 六Adam16→24、相同fingerprint、三sampler身份、真实terminal trace和API取消/释放通过，native环境只能重建不能exact；(3) 原始曲线、16command与2held-out失败录像/hash/解码/可访问证据齐全，质量0成功且无成功录像。决策continue进行前2000更新观察热身，配方仅提升smoke_passed，不提升validated/acceptance pass。首个研究区间冻结24→1024，新增1000，source `run_7afe305a163544f59eb7`/`mv_5d9f639f9eb64051bc3d`；本轮已用新增更新1024，剩余18976。

## 配置B分段记录

首段已提交 `trn_72c8746f2f3344ebb1e1` / `job_0a98d7422964483aa2de`，从24→1024仅新增1000、trainer_state；同safe-v2 image/Profile/data/seed/拓扑。每段完整结束后验证六Adam/terminal trace、validation16全部媒体、原始曲线与三轮复核，再写continue/stop_adjust/wait_evidence/budget_end。明确质量热身到2024（两完整1000区间）；之后3024/4024/5024至少三个完整观察窗口，连续三次相对best无改善则stop_adjust。计数不能把actor的10240样本warmup算成2000联合更新。

首段已成功，Run `run_e69e94b6f06c4a50a688`，模型 `mv_d206cdd31f974cd78408`：六Adam1024、完整hash、fingerprint与24一致；累计sampling_rounds100/online76800，本段84轮/64512新增transition。3sampler最后广播1016哈希一致，最终checkpoint1024。新增trace64512条与replay逐条动作/episode/step一致，4001硬限位（1490position/2912velocity/重叠401）全部terminated，总5794终止，无torque越限。真实训练曲线包含1→16→24→1024边界；FB loss前50点均值117206→最后50点23030，动作饱和约40.1%→43.3%，训练episode均值10.43→11.08控制步（新硬限位缩短了episode，不能与旧无硬限位训练平均步数直接当改善）。原生actor_std实际0.05，不能把假定的大探索噪声当根因；原生B正交项使用B@B.T的样本Gram，不声称batch128<z256数学上不可能。batch128相对上游1024仍是待对照的配置差异。

首段独立快评 `job_7bd15a665d6f4fe7bbde` / `run_88abecc0b9bd4f9d8466` 正在运行；质量决策尚未输出，不提交下一段。本轮实际新增更新2024，剩余17976，取消资格额外采样按上界计，未触及2M/80GiB/600录像预算。

### 1024窗口：continue（质量热身）

上述快评已succeeded/finished，Report `ev_e749391eec184773bc84`：0/16、6fall/11hard（重叠1）、无NaN，实际196控制步，汇总速度0.732007/yaw3.940960。median10.5，范围1–30；qual24 median9，本次+1.5低于冻结+10的改善门槛，best仍24，不以速度均值或loss下降改best。16/16 initial/trace/video hash、真实解码、HTTP媒体、4export文件均通过。没有成功录像。

三轮复核：①RPO/v1接口、raw train-only及原生8aux/math仍固定，新acceptance未参与；②六Adam/hash/fingerprint/3rank与64512条真实trace对齐通过，每硬越限terminated，退出0/OOM=false且四卡释放；③全16case/录像/图和访问完整，仍质量失败、平均训练episode约11步、饱和约43%。固定前2000更新热身允许continue至2024；此窗口不记后热身无改善strike。决策文件绑定Run/Model/Suite/plan hash/剩余17976更新，下一段只从safe1024完整状态恢复，最多新增1000。

### 2024窗口：continue（热身结束，随后三个完整观察窗口）

`trn_139df833ef9344eabe3e` / `job_2874ff35791e42398d63` / `run_298bb473985642c29436` 成功，模型 `mv_fe96c1b9a2454741a9a5`；六Adam2024/全部文件hash/相同fingerprint回读通过，累计141312 online、184轮，本段64512新增。4226硬越限全terminated（1337position/3259velocity/重叠370），总终止5326，真实dq峰64.40065，无非有限值或torque越限。

快评 `job_dfb410c7361548c690e5` / `run_6fada0a0daaf4e01b85e` / Report `ev_779f3e256275450182dd`：0/16、5fall/12hard（重叠1）、无NaN；184步，median9、范围1–32，速度0.829192/yaw4.839281；16/16媒体真实解码/HTTP访问及所有initial/trace/video/export文件hash通过。best仍qual24，不以loss或误差均值选择新best。热身结束，此窗口不冒充三个后热身窗口之一。

追加排障：真实PD目标与 `DEFAULT_Q+.25*ACTION_GAIN*applied_action` 最大误差1.669e−7，未发现parent裁剪/scale/默认姿态错位。replay保留100000条中的8597个step0初态，无height<.5或tilt>.8，首步终止5.71%；不能把大量已倒置RSI当根因。初态仍未全面证明足底接触稳定性，未修改物理/数据以绕过验收。

三轮复核：①同RPO/codec/train数据/native数学，PD解码和真实reset高度/倾角审计通过；②六Adam/恢复/三采样器/真实ring replay和terminal证据完整，训练退出0/OOM=false，1000新增严格边界；③完整16case媒体与曲线齐全、质量仍失败，best不变。该次评估漏启动宿主资源采样，不能伪造本窗口RSS/nvidia显存峰值；保留退出/OOM、worker原生allocator记录，沿用相同c2b镜像4×1已实测准入，后续提前记录。决策continue到3024为第一个完整后热身窗口；与best同success且median不足+10即无改善strike1。本轮新增3024更新、剩余16976；新acceptance仍未执行。

3024窗口当前活动：`trn_8f995f148c7347398519` / `job_c01e23f4f8434396966f` / `run_f96169ae70bc4012be13`，同镜像从2024完整恢复。操作状态另存 `artifacts/2026-10-11-ufo-walking-repair/repair-session.json`；未达到正常行走，目标仍进行中，没有提交代码。

### 3024窗口：无改善strike1；用户要求先交接，后段未提交

上述训练已succeeded/finished，模型`mv_38467550b71e49c28ac6`：六Adam3024/完整hash/同fingerprint通过，累计205824 online、268轮，新增64512。4129硬越限全terminated（position1183/velocity3283/重叠337），总6190终止，无torque越限；dq峰48.5758。PD解码最大误差1.669e−7，所有trace身份与ring replay逐条对应。

独立快评`job_55005a7fa2e84a6599f7` / `run_50e7662d0a4e456e9cb8` / Report`ev_f2ec53ce578e4467a747`：0/16、4fall/12hard、无NaN，296实际步；median13.5、范围4–48，汇总速度.809093/yaw5.032918。全部16媒体解码/HTTP访问和initial/trace/video/4export hash通过，退出0/OOM=false，训练及评估资源观察日志保留。+4.5median小于冻结+10门槛，best仍24，后热身无改善strike1。本轮新增4024joint updates、剩15976，online含取消资格上界282624。

用户截图所示episode_len下降经原始指标确认：最新段mean10.112/末8.945，前段mean11.693/末11.986；最新末10点mean9.104、num_trajectories末91，FB loss末50点10612仍下降。不能将loss下降当行走改善，也不能将启动几步的短episode随后上升当作学习进展。

三轮复核：①RPO/codec/数据隔离及原生数学保持，PD/terminal真实证据无偏离；②六Adam/hash/计数/三rank和环境重建覆盖正确，正常退出/无OOM，GPU已释放，新增1000严格边界；③16case、图表、全部失败视频及访问完整，但质量仍失败，best未变。按既定协议原可continue下一观察窗口，当前用户明确要求先输出交接给其它模型分析，故不提交4024，第三配置也未实施。

独立只读审查新增机制证据：继承的native tracking在任一reset时重选全sampler tracking序列；safe2024继续episode的相邻步排除step%100后39.64–39.83%实际z改变，变化方向median cosine.055–.062。z范数16是原生投影，不证明方向稳定。候选只关闭原生rollout_expert_trajectories，仍保留expert学习/mixed pool；其质量因果与GPU结果尚未验证。审查报告、CPU复现与source/input hash保留`review-native-context.*`。

给其它模型的完整当前总结见[独立分析文档](/data/kk/EmbodiedBein/docs/progress/2026-10-11-ufo-walking-handoff.md)，内嵌九段完整设置、全部指标统计、228条case记录和关键源码；[完整逐点指标](/data/kk/EmbodiedBein/docs/progress/2026-10-11-ufo-walking-raw-metrics.md)保留330620个原始点。全部文档引用为绝对路径。新100例acceptance未运行，未达到行走标准，不提交代码。

### 4024窗口：无改善strike2；继续既定最后观察窗口5024

交接完成后按仍活动的目标继续原有界协议，冻结的3024完整交接快照不覆盖。新Train `trn_57ba63fb6b994979be47` / Job `job_9cbda63459814cd7b467` / Run `run_a4b6ef0d96584b29b20c` / Model `mv_758c24169168471a80ed` 正常结束4024；六Adam/hash/同fingerprint回读通过。累计270336 online/352轮，本段64512新增；4064hard全terminated（position1242、velocity3166、重叠344），7254总终止，无torque越限，dq峰42.958。三sampler版本4016/SHA一致，PD/trace/ring replay逐条通过。

独立快评 `job_3e78e61168934b6fb2f3` / `run_e4df98d9c2d64a408f85` / Report `ev_c11f14b2b4914e199922`：0/16、2fall/14hard、无NaN，171实际步，median8、范围1–35，速度.597459/yaw3.853483。全部16 initial/trace/video hash、真实解码/HTTP和4export核验通过；训练/评估退出0/OOM=false，GPU释放/资源日志保存，自己的结束临时容器清理而输出保留。训练episode mean8.678，比上一段10.112继续下降；误差或loss下降不改变best24/median9。

三轮复核：①RPO/codec/native八aux/数据隔离保持，未使用新acceptance；②六Adam4024/hash/计数/三rank/环境重建/正常退出与释放完整，新增1000边界；③16case与全部失败媒体/图/访问完整，但median8不满足best+10，故strike2。决策continue至5024为第三个完整后热身窗口；若仍无改善必须stop_adjust，不再追加B热身。本轮新增5024更新、剩14976、online含取消上界347136。

追加CPU控制敏感性证据：固定2024模型和相同观测，仅换相邻z，在1024个changed pair样本中988个有任一PD目标差>1rad，平均abs动作差.8442/目标差.5290rad；unchanged对照全部0。真实连续行为排除policy-version边界后，changed19135对的平均目标差.6277rad，unchanged28965对为.1718rad。真实比较包含观测变化，不当作随机因果；目标差也不能冒报实测dq。它加强只关闭原生tracking覆盖候选的依据，但新配置尚未实施/验证。全部配置与新case、CPU原始结果内嵌[续段报告](/data/kk/EmbodiedBein/docs/progress/2026-10-11-ufo-post-handoff-analysis.md)，`z-action-sensitivity.*`保留代码/权重hash；原始完整交接文档保持其快照身份。

最后B观察窗口已实际运行：`trn_602c852d5db243afa6ce` / `job_fc4be20d66b74ac4a8d1` / `run_2375cb7b95504d99bf72`，同完整trainer_state从4024→5024，新增最多1000。按实际Run持续观测，不因等待重启；结束后须独立16case完整媒体与三轮复核才能写第三strike/stop_adjust。第三配置尚未实施，新100例未运行，没有提交代码。

### 5024窗口：无改善strike3，stop_adjust B

Train trn_602c852d5db243afa6ce / Run run_2375cb7b95504d99bf72 成功至5024，Model mv_8bfdd837d60e40a7ba5f；六Adam/完整hash/相同fingerprint及trace回读通过。新增1000，累计334848 online，本段64512。

Report ev_486def3c7a5c4b7e8aef（Run run_790659ab097740889fa2）：0/16，8fall/8hard，无NaN，140实际步；median8.5，范围1–18，velocity RMSE0.797671/yaw3.879159。全部16 initial/trace/video hash、解码/HTTP及4export通过。第三个完整后热身窗口未达到best median9+10，停止B；不能因loss下降继续追加。

本轮新增6024真实更新，online含取消上界411648，剩13976更新；当前受管输出/模型按实际唯一inode占23.457GiB，CAS/新镜像另计。Run结束时任务容器已不在docker清单，不能补造Docker State退出码或OOM字段；保留Job/Run finished、资源日志和完整产物。

三轮复核：①本体/原生八aux/数据隔离及新acceptance保持；②真实六Adam、计数/恢复/3rank/终态trace完整，段边界严格；③全部独立case和失败媒体可访问，但质量失败。决策stop_adjust；按用户新要求第三配置将同时对齐原生网络和稳定提示，单因素归因限制如实说明。

[完整决策与指标](/data/kk/EmbodiedBein/docs/progress/artifacts/2026-10-11-ufo-walking-repair/safe5024-decision.json)、[原始逐点指标](/data/kk/EmbodiedBein/docs/progress/artifacts/2026-10-11-ufo-walking-repair/safe5024-metrics.json)。

### 用户要求原生模型对齐：新C资格16已启动

[完整对齐与优化记录](/data/kk/EmbodiedBein/docs/progress/2026-10-11-ufo-native-alignment.md)内嵌原生模型完整配置、参数/CPU/GPU估计、修复及测试、数据mask审计和所有5024结果。新C Train trn_020c736301c443109bb0 / Job job_0cda805cff0c40829926 / Run run_184b584d297e403ca4c0 正在执行；batch1024/replay1M/native2048×6/trackingFalse，一learner三sampler，无B权重或优化器继承。保持dependency_ready，待16→24/独立评估/媒体完整才提升smoke。尚未达到正常行走，没有提交代码。


C实际6048累计新更新，原生full2048×6/batch1024/replay100万、trackingFalse；16→24恢复/验证/取消已执行。C24速度0/16、held-out两例5步fall；尚无正常行走证据。首步512×2接触诊断支持默认target跳变加重dq/接触力，但hold-reference14/512仍越限。完整输入、24更新全部原始指标、16case、2held-out、512×2诊断全部统计及失败历史已嵌入[原生对齐报告](/data/kk/EmbodiedBein/docs/progress/2026-10-11-ufo-native-alignment.md)。D下一步按冻结限速公式实施，保留原始请求replay和执行历史、显式记录RSI投影残差；无需直接删除G1专家。


## D原生模型与限速资格闭合

C原生对齐/关闭tracking之后，D加入共享PD target限速。四卡16→24完整恢复、独立速度/held-out/实际控制trace/所有媒体/取消均已通过，smoke_passed；速度仍0/16、median7步，held-out23/9步最终fall，尚未行走合格。下一段24→1024固定1000更新，之后独立评估再决定。全部参数、原始指标和逐case/逐步信息见 [完整原生对齐记录](/data/kk/EmbodiedBein/docs/progress/2026-10-11-ufo-native-alignment.md)。


D24→1024已结束：新增1000联合更新/64512在线转移，独立0/16、中位35步，heldout17/87步hard。E9速度软惩罚已原生/容器实现且dependency_ready，未GPU资格；下一段含checkpoint/预留超80GiB，决定budget_end。全部配置/指标/逐case见[隔离摘要](/data/kk/EmbodiedBein/docs/progress/2026-10-11-ufo-optimization-summary.md)与[逐点完整记录](/data/kk/EmbodiedBein/docs/progress/2026-10-11-ufo-native-alignment.md)。未提交。
