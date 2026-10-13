# RPO BFM-zero 优化执行记录

计划：[优化执行计划](../plan/2026-10-13-rpo-bfm-zero-learning-optimization.md)。本次用户授权：冻结实验条件，执行物理、奖励、RSI 诊断，按证据推进；不提交 Git。

## 执行账本

- Task 0：诊断条件已冻结，训练新候选的Suite/case及身份尚未绑定。准入前平台与 ebd 健康，四张 4090 各空闲约 24 GB，宿主可用内存约 104 GB，磁盘约 813 GB。诊断上限 4 Job × 30 分钟，无自动重试，合计 2 小时；训练预算另按计划，当前未启动。
- Task 1–3：首轮诊断完成；v4受管Job成功，证据已核验。仍有明确未验证项，不能把任务的所有复选框勾选为完成。诊断只读消费v13 step600，保持旧 recipe、终止、奖励和模型冻结。
- Task 4–9：等待诊断出口；不能因本记录存在解锁旧配方或跳过周期决策/独评前提。

Ruling：在当前用户指定且含大量未提交/未跟踪 RPO 实现的工作区继续，保留内容快照和文件 hash，不新建丢失这些输入的空 worktree；不暂存或提交。执行账本保留在本文件，供后续恢复工作。

Ruling：计划编写时的“尚未授权实施”已由本次用户指令替代；实验仍按条件出口与预算执行，不逐任务重复询问授权。

## Task 0：已冻结诊断条件

[experiment-contract.json](artifacts/2026-10-13-rpo-learning-optimization/experiment-contract.json)，SHA256 `5238a0fe36b46791cdb4cc7d68f14dde1099cec80464dfd3238e53d72461deeb`。v12 step524、v13 step600 各31个文件校验通过，六个Adam计数分别全部524/600。保留原严格门槛、S0拓扑、27Nm/8rad/s接口假设与未知硬件曲线。尚未绑定新候选的validation case/Suite，`training_enabled=false`，本合同只授权有界诊断。

## Task 2/3：全量只读审计

[reward-replay-audit.json](artifacts/2026-10-13-rpo-learning-optimization/reward-replay-audit.json)：983040条累积行为转移，位置成本占47.950%、滑动21.086%、速度13.978%、踝roll12.315%、足姿态2.523%、动作变化2.147%。这是归一化前绝对加权均值占比，不能代替Actor梯度占比。保护body接触成本全零，须对账子步接触和末帧aux激活，尚不能据此说没有接触。

[reference-audit.json](artifacts/2026-10-13-rpo-learning-optimization/reference-audit.json)：train-only专家覆盖9个祖先分组，独立take数量仍待核实；official RPO保守合并为未核实祖先组，不能把分组数解释成已验证独立take。official_rpo/g1专家8步窗口5206/6716，原250步窗口1279/931，RSI1633/1410，RSI+tracking203/354。祖先split隔离通过。`states[:23]=q-DEFAULT_Q`、`states[23:46]=dq`及完整state/privileged codec复算通过，不把预期零位偏移误判为数据损坏。行为标签缺可靠manifest，明确unknown；未改valid mask或课程，未以PD失败撤销参考资格。

## 已证实的采样实现缺陷及最小修复

`UFOBFMPolicy.predict_action_batch` 的 tracking 分区检查把字符串`mode`替换为bool张量。实际行为检查证实：热身0转移及热身结束10240转移时，train/eval动作都完全相同，训练均值分支被误选，随机热身及策略动作噪声被绕过。它是实现缺陷，尚不能断言是所有物理失败的唯一原因。

修复仅将分区变量改为`tracking_mode`，并使动作缓存包含调用mode，防止train/eval混用缓存。保留replay=requested、last_action=applied及动作映射。修复已在仓库源码通过行为检查；本轮物理配对仍消费旧冻结镜像里的原策略，新训练配方尚未构建/启动，v13没有被改写或恢复续训。

Ruling：该缺陷影响实际行为策略及参考采样，未来首个对照应先单独比较修复，而非与奖励同时变化。奖励候选顺序待物理出口；新recipe身份及恢复指纹必须显式绑定采样修复。

## 受管物理诊断执行

v1：Job `job_2fa282f071934f278b8d`，Run `run_aa5ea574726841759335`，exit_1。新入口错误假设ebd一定设置`CUDA_VISIBLE_DEVICES`；实际四卡由Docker设备准入提供，变量可缺省。三个worker都在导入仿真前退出，没有物理或训练更新。这是本次新增诊断入口错误，不是模型质量失败。记录和镜像保留；无自动重试。

新增映射行为检查复现后，派生v2，显式保持四卡约束，复用旧probe的0/1/2/3缺省映射。v2新增Job将计入4个诊断Job总预算，v1已计一个；不得无界重试。

v2：Job `job_8dd93d57dc454304ace0`，Run `run_aa6d6303e97d42eb8a52`。六个物理阶段都完成并保留原始数组、事件及初态reference，随后价值诊断对UFO冻结Pydantic配置写入`device`失败，整个Job保持aborted。`partial-physics-evidence.json`只作为已完成物理输出的分析索引，没有补造成功回执或改平台状态。该版每worker每类限32，未满足计划“全Job每类32”的更紧口径；v4改为六个阶段各5例，每类总计最多30，保留全体轻量统计。

v2固定锚点48步原始episode存活3254/6144，峰值速度比2.819175，位置软范围在子步发生但末帧消失1391次。原始回合存活是右删失，不是20秒合格站立。锚点保留了原参考dq/root动量，不能将其失败直接解释成静态站姿无法保持。

复核发现v2策略分支只加载step600权重，mixed半区没有恢复保存的z池。因此v2策略结果仅为“冻结权重+新随机latent”初步诊断，不用它代表历史v13行为分布。最后一个v4诊断将恢复step600 z池，并明确fresh episode上下文、tracking B600；仍不称精确重放历史sampler596或旧物理状态。

v3：Job `job_49155a071e1941cd859e`，Run `run_65757b8be2a742059680`。只做价值诊断，修改冻结配置的错误已消除，但推理model构造默认`requires_grad_(False)`导致`autograd.grad`失败；没有优化器更新。这个错误属于本次诊断代码，不能推导原训练Actor没有梯度。三个失败Job及输出均保留，无自动重试。

继续源码复核还确认UFO `EMA.forward`即使eval也更新mean/mean_square/counter。诊断改为读取保存的EMA统计并按原公式缩放，避免任何forward更新；Actor仅在隔离诊断副本里暂时启用参数梯度，结束后恢复flags，保留`.grad=None`。真实CPU小模型测试验证FB/D/aux三项梯度非零、所有参数与buffer逐值不变、Torch RNG及requires_grad flags保持。最后一个v4 Job将验证实际435M模型，不因前三次入口错误再无界追加作业。

v4：Job `job_cc05573774c4419bac64`，镜像 `sha256:b4d976ea9163a6f57b033572cb3fb78225ec9f480e6b40e2afd3cae0ddf0e455`。它是第四个且最后一个诊断Job；只补足诊断，不启动训练/策略独评。`image-code-audit.json`核对七个原模型/物理/RSI/replay/context模块与冻结前源码hash一致，仅新增诊断入口；训练mode修复不会混入这个原行为对照。

## 最终v4结果与决定

Run `run_1a86aa399ea64ca79387`，平台Job状态 **succeeded**，原回执`/data/kk/embodiedbein/runs/run_1a86aa399ea64ca79387/outputs/learning-diagnostics.json`，资格仅`diagnostics_only`。六个阶段、数组/事件hash、原始RSI配对、全Job每类事件≤32均核验通过；共保留83个有限事件例子。成功回执与前三个失败Job分别保留，未修改任何Job状态。

| 指标 | 固定RSI目标 | step600策略＋保存的z池 |
|---|---:|---:|
| 原始配对初态数 | 6144 | 6144 |
| 控制时长 | 48步／0.96秒 | 48步／0.96秒 |
| 原始episode到48步未终止 | 3254／52.96% | 0／0% |
| 全子步速度峰值比 | 2.819233 | 3.927786 |
| 首步训练终止／所有RSI首步 | 0／9405 | 4／26406，0.01515% |
| 子步软位置越界但末帧为0成本 | 1364次／294912转移 | 13204次／294912转移，4.477% |
| 请求动作被slew改变的元素比例 | 0% | 84.525% |
| 位置成本末帧／子步平均 | 0.023404／0.022955 | 0.063075／0.057542 |
| 保护body子步接触／末帧aux激活 | 0／0 | 0／0 |

只有原始episode的初态是配对的；分支终止时间不同后，reset消耗参考RNG的次序也不同，后续重置回合不是逐例配对。幸存episode在48步处右删失，不能称为20秒站立成功。所谓`normal_landing`事件仅指该区间无违规且足力>20N的对照，不等于行走验收。

事件例子：rank2/env798，episode第8控制步、第3子步，右踝pitch为 **31.4223 rad/s**，实际积分力矩已经 **−27 Nm反向制动**，功率约−848.4W，右足垂直力798.7N；此时q=0.3024rad，距物理位置边界仍有0.2976rad余量。该区间结束时速度已回到训练速度容差内，但子步速度违规仍被锁存并终止。这说明不是通过“加大速度罚”就能证明解决，也不能将反向制动误判成电机继续正向加速。83个所选例子中有4个速度训练违规在末帧恢复、没有位置训练违规在末帧恢复；它们是例子计数，不是全体恢复率。

冻结replay的64条固定转移，使用保存的行为z：Actor三项梯度L2范数 **FB=40.6709、D=5.38182、aux=1.38149**，aux约为FB范数的3.40%；FB/aux梯度余弦−0.15835、FB/D−0.25450、D/aux+0.35507。w达到50上限，FB/D/aux对应loss项−115.624/+11.420/+4.602。此结果支持“成本占比不能当梯度占比”：位置成本约占48%，而整个aux项对该批次Actor的梯度较小且部分对抗FB。它只是单个64条批次，不包含训练0.8 latent relabel分布，不能定为全训练的固定百分比或直接据此放大权重。

实际435M模型诊断前后网络及buffer hash均为`b27197ced522284d85819748741a667b3a551295c9165308b750662c1260cd89`，优化器更新0、EMA统计/RNG未更新。归一化成本按step600的冻结EMA统计计算，而非再调用会更新统计的EMA.forward。

![最终配对诊断时序](artifacts/2026-10-13-rpo-learning-optimization/diagnostic-curves.png)

![冻结Actor梯度审计](artifacts/2026-10-13-rpo-learning-optimization/actor-gradient-audit.png)

完整汇总见[diagnostic-analysis.json](artifacts/2026-10-13-rpo-learning-optimization/diagnostic-analysis.json)，事件索引见[event-analysis.json](artifacts/2026-10-13-rpo-learning-optimization/event-analysis.json)，文件hash及Run根路径见[complete-diagnostic-evidence.json](artifacts/2026-10-13-rpo-learning-optimization/complete-diagnostic-evidence.json)。原始large数组保留在受管Run输出，未搬进Git。v2初步图/汇总另存`diagnostic-v2-*`，不能混入v4结论。

**决定：停止奖励训练候选，等待控制/动力学与状态语义证据。** 固定目标分支也未满足峰值比≤2.5，约47%的原始动态RSI回合在0.96秒内终止；模型分支更弱。遵守Task 1出口，不启动R-pos-time、R-slip或R-roll-off训练，不解锁v13续训，也不做受限制的独立策略评估。首轮4个诊断Job预算已用满，无第5个Job。

按证据推进的优化是已证实的`mode`覆盖及动作缓存修复；它已通过源码行为检查，还没有训练质量结论。未来第一个新配方应只改变该实现因素，再按100工程检查及K=100周期决策推进；奖励候选与它分开。R-pos-time保留为后续候选：它能覆盖末帧遗漏，但平均成本反而略小，不能宣称必然“罚得更重”。本轮保护body接触在子步也为0，不能靠提高该项权重取得信号。

## 下一阶段所需证据与未完成项

> 后续执行纠正（2026-10-13）：第五轮发现旧诊断phase过滤actor配置时丢弃context_profile/ankle_control_profile，实际使用legacy_global+train_stochastic，并非真实v13 environment_tracking_v10。该事实及完整runtime审计见[第五轮](2026-10-13-rpo-bfm-learning-round5.md)。保留本报告历史raw数值，但不把旧诊断policy质量推导为真实v13训练语义。实际step600 actor配置没有ankle_control_profile，已发现的kd25 mode覆盖分支未激活；mode/cache修复仍是代码正确性修复，尚无证据改变当前正常训练采样，不能自动启动名义单因素对照。固定anchor与同包replay的物理缺口继续成立。物理候选、三轮有限总预算、最新结果与平台链接见[第六轮](2026-10-13-rpo-bfm-learning-round6.md)和[第七轮](2026-10-13-rpo-bfm-learning-round7.md)。

1. 对事故补齐完整qpos/qvel、实际积分广义接触/执行器/被动力、M(q)与接触Jacobian，区分参考动量、接触耦合和求解离散化贡献。固定目标保留了非零参考dq/root速度，失败不证明静态站姿或某个PD增益本身错误。先冻结新的有界诊断预算；不猜电机惯量、不抬接口上限、不裁仿真qvel。
2. 对相同RobotState补MJLab reset后的358列privileged与参考逐字段对照，单列root lift与history清零。已完成的是源codec全量复算、q/dq及实际reset的52列state对应；这还不能排除privileged的跨引擎差异。
3. 对相同策略/latent完整轨迹做Q_aux多步成本校准，并覆盖多个固定批次及0.8 relabel口径。当前梯度证据只是64条stored-z诊断，不能把弱梯度唯一归因于系数。
4. 运行时接触点/法向与脚跟/脚尖接触未取得，R-slip不实施；行为标签/独立take未核实，RSI课程不实施。保持原参考速度、valid mask与专家资格，动态PD失败不撤销observation-only参考。
5. 新训练recipe/镜像/恢复指纹、v3 checkpoint pending更新块、K=100连续环境决策、MJLab独评及媒体出口尚未实现。训练mode修复未打进旧训练镜像，不把旧模型当新行为配方完整恢复。未绑定新Suite/case前不做新模型validation/acceptance。

最终相关检查：冻结栈51项数值/状态行为检查通过；平台36项检查通过。无skip；仅有pytest asyncio配置及passlib crypt弃用警告。`git diff --check`通过。`postflight-doctor.json`确认四卡均释放、各空闲24080MB，ebd健康；GPU0训练未启动，所有训练更新为0。没有运行全仓测试、训练工程100更新或独立录像评估，没有提交Git。

## 三轮交付复核

- 第一轮（设计/物理/隔离）：保持原限位与v13终点；七个基线模块hash不变，训练与acceptance祖先隔离，未将分组数冒充独立take；明确q−DEFAULT_Q、原始参考动量及privileged未闭环。
- 第二轮（执行/资源/失败）：四Job含三个入口失败均保留，无自动重试；第四Job完整成功，零优化器更新、冻结EMA/RNG/模型状态通过；预算用满即不再追加。未实现周期恢复，未启动新训练。
- 第三轮（图表/事件/证据）：保留初步与最终图、原始点和逐事件索引，核验83个例子及全局上限；明确右删失、reset后非配对、stored-z梯度样本局限与无独评/录像。任何上述未验证项均不标为验收通过。

## 已执行检查

后续：用户已授权独立第二轮有限诊断，详情见[第二轮动力学与RSI记录](2026-10-13-rpo-bfm-learning-round2.md)。该轮补齐M/J/接触及52/358来源对账，准备采样修复候选镜像；固定目标物理门槛仍失败，训练继续关闭。本文件保留首轮历史结论，不把第二轮额度并回首轮。

- 新诊断4项行为检查先失败，再实现通过；GPU缺省映射检查先失败，再修正。
- tracking模式缺陷以真实小网络及真实专家/上下文复现，非mock动作；修复后相关栈回归49项通过，无skip。pytest缺少asyncio插件配置警告不影响这些同步检查。
- 平台相关检查14通过、3skip，skip来自基础测试镜像缺数值依赖，不充当数值验证；对应数值场景在冻结训练镜像中已跑通。
- 初次测试漏挂只读资产、漏补UFO/RLinf/EMBODIED_PATH导致的失败已重跑解决；不是产品失败。
- 镜像首次误用本地image ID作registry manifest digest，构建失败后改为先验证本地ID再从本地tag构建，构建步骤禁网络。未更改基线镜像。

## 最新执行状态（2026-10-14 Asia/Shanghai）

后续获批三轮物理优化已关闭，完整结果见[第七轮](2026-10-13-rpo-bfm-learning-round7.md)。实际8/9Job、83.836分钟Run时间、28.004GiB完整Run，含一次失败；最后确认名额因C2未全过保留未用。I1积分、S1锥、D1阻尼、S2摩擦阻抗、C2地面法向刚度均保留单因素实测证据，无可放行训练的配置。C2剩余失败为seed43峰2.59798、seed43/44最大穿透7.786/5.406mm及三个相邻严格均值；制动已经饱和仍继续加速，最大穿透又来自不同case。后续应分别核验这些case的接触几何/求解与控制，新的GPU候选另行冻结有限预算。

动态RSI动量、接口限位、奖励和数据版本保持；第二轮已解释358参考/在线差异来源，第五轮纠正旧诊断context与实际v13混淆。当前没有启动mode/cache名义训练对照，也没有100/500更新、模型独评或录像；物理必要条件、有效采样差异与周期/恢复/媒体前置项仍须全部完成。各阶段Job/Run链接、图表、196个第七轮receipt成员及完整预算索引已记录，无Git提交。
