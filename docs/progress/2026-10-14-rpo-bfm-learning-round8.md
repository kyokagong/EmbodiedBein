# RPO BFM 第八轮：复用状态核验接触、求解与制动

日期：2026-10-14 Asia/Shanghai。用户授权先复用已有状态补齐速度峰与最大穿透案例证据，再冻结新一轮有限预算选择单因素候选。前轮见[第七轮](2026-10-13-rpo-bfm-learning-round7.md)。原接口27Nm/8rad/s、奖励、RSI及数据版本保持，不使用Git提交。

**本轮已关闭，结论为停止晋级。** A1（相对C2仅地面margin 0→3mm）首跑通过必要anchor条件，但同配置确认在旧env453出现真实5.750418mm穿透，超过5mm上限。首跑该env提前位置终止掩盖了晚到尾部，不能据首次通过放行。实际2/2Job，0训练更新；完整平台链接、逐阶段结果、三轮复核、21张诊断图及哈希索引见下文。

## 阶段0：已有状态证据补齐

先冻结离线容器分析合计≤30分钟、派生输出≤2GiB、新Job=0、optimizer=0。只读原Run/资产和冻结C2镜像，不原地修改发布环境或旧收据。新GPU候选尚未选定，先完成接触几何、同状态CPU求解与历史制动对账；若证据不足，记录缺口，不凭猜测冻结参数。

源：[Job job_9d9d692030be4dcd8462](http://1.94.0.106:31800/api/v1/jobs/job_9d9d692030be4dcd8462) · [Run run_661fbd961a1d4a10a491](http://1.94.0.106:31800/train/runs/run_661fbd961a1d4a10a491)。receipt SHA`24192b54f7615f758279795ab855bf60fbae34d874de4e8eb2df9089227e9d62`，镜像`sha256:ffe631ac28292746b1e883f6e9b7d9e66a16e874cd623df47665fcf49d04432d`。三个anchor速度峰及三个最大穿透case均来自同Run原episode，完整初态/动作包沿用A0；旧policy context不代表真实v13训练。

执行步骤：同Scene/actuator/terrain编译入口在CPU建模并逐字段核验startup；用事件qpos_before/qvel_before及warmstart复算速度峰M/J/接触/制动；用全population保存的穿透case积分前后qpos/qvel复算几何、核对最大深度及接触对象。历史GPU未保存的M/J/warmstart明确标缺，不把新CPU结果冒称历史GPU采集。

MuJoCo `mj_forward`重算动力学而不积分，`mj_step`会推进状态；求解输出可能因有限收敛改变，必须保持状态时刻及实际驱动力对应。参考[MuJoCo 3.8 API](https://mujoco.readthedocs.io/en/3.8.0/APIreference/APIfunctions.html#mj-forward)。本轮CPU几何/求解证据用于候选选择，不能替代完整GPU物理门槛和同配置确认。

## 执行记录

工作区差异已保存，只读镜像ID核验与C2一致。当前进行离线编译身份和状态提取；新Job尚未提交。后续逐阶段补结果、图表、预算、完整索引及新Job/Run链接。

### 已有状态核验结果与独立复核

CPU同Scene编译与C2 startup逐项/编译XML SHA完全相同，nq30/nv29/nu23。六case源数组按receipt哈希核验并提取；三个速度峰积分前后q/v与历史事件精确相同。最大穿透都在足部mesh与地面，脚尖侧最低顶点处，CPU与GPU最大深度差≤2.582µm；不是误用了速度峰约3mm的几何来解释7.786mm尾部。[状态审计](artifacts/2026-10-14-rpo-learning-round8/state-audit.json)。

CPU原生碰撞与历史GPU接触集合不同：三个速度峰CPU2contact（每脚1），GPU8contact（每脚4）。保持模型、q/v、实际电机力矩和保存warmstart，注入历史接触后，单步关节速度误差由0.400838/0.642723/0.416523降至0.000193471/0.000168140/0.000077291rad/s，缩小>99.95%。native CPU Newton3–5次，注入后4次；native零warmstart单步差<5e−16。说明这些同状态单步差主要由接触集合解释，不支持优先盲增求解器迭代，也不能认定GPU collider错误：[官方3.8说明](https://mujoco.readthedocs.io/en/3.8.0/mjwarp/index.html#memory)已明确plane–mesh最多Warp4点、CPU3点。[接触集合重放](artifacts/2026-10-14-rpo-learning-round8/contact-layout-replay.json)。

同集合比较J最大差2.58–3.35e−6；已单列smooth qfrc/qacc、bias/passive差异。历史GPU未保存efc_aref/D/R，剩余差异不能严格拆成构造、浮点或停止误差；三个穿透case也缺历史接触/M/J/warmstart，不能把速度峰单步结论外推到它们或48controls整体。新CPU力均标为重算值，没有伪造GPU快照。

分析过程修正：MuJoCo3.8 solver统计是flat list，首次二维索引错误已修；CPU/GPU约束行顺序不同，先按contact_efc_address排序，再对23个type1摩擦DOF建立唯一行置换。保留原GPU数组和identity guard，不用宽松阈值掩盖差异。独立只读review确认状态时刻、行映射/几何及主要归因成立，并指出派生文件哈希须补入最终索引。

六case制动窗口按历史实际力矩/速度独立绘图，并核对case末状态和电机力矩与提取快照精确相同。[有界窗口摘要](artifacts/2026-10-14-rpo-learning-round8/bounded-case-braking.json)明确选取case末时最快归一化关节，不将它冒称一定对应最深足点。最大穿透时三例该关节速度仅−1.852/−6.378/−3.647rad/s，电机力矩+7.293/+19.076/−0.612Nm、接触力矩−15.762/−20.339/−19.992Nm，直接关节限位力矩均0。尤其seed43的7.786mm案例中电机尚未饱和，不能套用速度峰案例的“−27Nm制动饱和仍不够”；须分别看脚尖几何压入和载荷突变。这里的零直接限位力矩也不能排除其它关节约束的耦合影响。

![同状态接触集合求解与独立穿透几何](artifacts/2026-10-14-rpo-learning-round8/offline-contact-geometry-and-solve.png)

![seed43最大穿透案例真实制动窗口](artifacts/2026-10-14-rpo-learning-round8/offline-depth-43-braking.png)

完整六张窗口图及求解/几何图保存在artifact目录。穿透case历史GPU接触/M/J/warmstart缺失仍未补造；三穿透case的CPU native重算Newton分别4/5/4次，只能证明CPU该状态能求解，不能据此宣布历史GPU求解器合格。额外A1编译审计首次字符串比较受XML属性顺序影响失败；改用标准XML canonicalize，严格对比仍仅允许terrain.margin一个属性改变，通过后保留预期XML SHA用于核验每个实际phase。

### 基于证据选择A1并冻结新预算

降低滑动摩擦1→.9的同状态CPU屏查仅产生很小变化，保留[F1局部结果](artifacts/2026-10-14-rpo-learning-round8/contact-layout-F1.json)，不启动该候选GPU Job。选择 **A1：相对C2只将ground.geom.margin从0改为0.003m**，gap仍0；控制/PD/原有接触solref、solimp、friction、condim及4点collider均保持，不减少接触点或改变足部形状。动机是把接触力建立移到穿透前的窄带，检验冲击制动窗口和真实几何压入是否改善，而不是再次把法向接触整体变软。

沿已保存轨迹，三最坏速度案例主冲击足进入3mm带比几何触地早3.75–5ms；其它足早5–6.25ms。[接触可用时序](artifacts/2026-10-14-rpo-learning-round8/anticipation-screen.json)是旧轨迹上的窗口证据，不预测新轨迹峰值或真实力 onset。CPU核验margin改变不改变同pose的真实signed distance；冻结MJWarp3.8.0.3源码也将dist与includemargin分别存储、constraint使用dist−includemargin。新收据必须记录实际geom.margin/gap和contact.includemargin；穿透门槛仍用**原几何−contact.dist**，不能减margin或放宽2/5mm条件。正margin是仿真接触正则化，不冒称硬件存在3mm垫层。

本次新预算独立于前轮：最多2Job，单Job≤30分钟，Run合计≤60分钟，完整Run≤12GiB；从A1契约冻结起部署/排队/运行/收集/独立分析总墙钟≤90分钟，retry=0、optimizer=0。第1名额A1；只有全部物理条件及完整identity通过才用第2名额同配置确认；失败停止，不挪用前轮第9名额、不盲扫第二个参数。继续使用3rank×2048、seed42–44、48controls、同A0动作包、冻结step600、50Hz及16×1.25ms。每anchor gate同时对原R0及相邻C2：峰≤2.5，存活各−2pp以内，严格均值不退化，首步终止≤.001，原episode几何depth p99≤2mm/max≤5mm。旧policy完整报告但不作真实v13训练结论。

A1只在真实CPU行为负例/正例、平台入口/收据/单因素边界、不可变镜像和独立review通过后提交。没有启动训练/模型独评。

### 阶段1：实现、验证和平台启动

真实Terrain hook负例失败于margin仍为0；正例及冻结诊断链合计39项数值检查通过，49项平台/准入检查通过。独立只读review另跑2项数值、2项平台检查，验证实际子进程hook顺序S1→S2→install_trace→C2→A1、CPU Scene只改变地面margin及13个镜像源码哈希；保留旧physics/policy哈希，没有把主机上的采样修复policy混入物理对照。A1 validator要求实际geom.margin/gap及事件contact.includemargin，旧C2默认入口拒绝A1 header。

冻结预算SHA`02ef10ff65b0455dd3ce19df5aa5e331a65b411c32d2a1db80b4f8c08b3d93d2`；阶段契约SHA`fd625fb411aa9d56f16271045516c40541b8273116cf09b9a7ae5a33b6d57659`；最终不可变镜像`sha256:91659e6d06263baf49619c4aa0ae737e2450776459b769b0cdb5f068a1aec397`。总墙钟从2026-10-14 05:05:50起，到06:35:50止（Asia/Shanghai），包含构建/部署/排队/运行/收集/分析，不能只按GPU Run耗时计费。完整请求、preview、health/doctor与负例日志保存在本轮artifact目录。

`make up`部署完成，health中ebd.ok=true，doctor四卡均空闲，preview固定上述镜像与replay入口后提交：[Job job_1fe749f03fc141bdbaac](http://1.94.0.106:31800/api/v1/jobs/job_1fe749f03fc141bdbaac) · [Run run_222b8644892c4a04b529](http://1.94.0.106:31800/train/runs/run_222b8644892c4a04b529)。当前运行中，kind=asset.validate，optimizer=0；不会出现在训练模型列表中。只有完整producer/collector及独立分析都通过，才可能用第二名额确认；启动成功不代表物理门槛通过。

阶段0计时说明：离线部分没有统一的起止计时收据；首份编译日志末修改至最后screen产物的时间跨度为32分06秒，不能据此宣称最初30分钟总墙钟达标，也不能以该跨度代替容器累计运行时长。阶段0新Job为0、原始数据只读；派生产物容量将纳入最终索引。A1及其确认采用上述明确统一墙钟契约，不将构建/复核时间排除在90分钟之外。

平台已完成A1收集（Job succeeded、Run finished且无abort），完整收据6phase/98成员。独立review再次确认原C2与A1 XML哈希链、单因素及确认规则，4项聚合规则检查通过：对R0/C2的门槛保持，对首A1只作六phase身份和≤2%峰复现，不误加相对首A1严格均值必须上涨的条件。晋级显式要求`physical_screening_passed AND all_phase_identity_passed`，后者不能被anchor-only物理字段替代。完整数组独立分析完成前不使用第二名额。

### 阶段2：A1完整独立核验通过，使用最后确认名额

[独立分析](artifacts/2026-10-14-rpo-learning-round8/analysis.json)完成：六phase初态reference/qpos/qvel/state和全部requested相同，共同原episode target/applied差≤1e−6；每phase编译XML都等于独立CPU预期SHA`f488a1720e2efae255c39e97985be407785642648feaaeeebf7cf6dd78c2fa6c`，差异仅地面margin及其实际记录。solver、PD、质量/惯量、接触solref/solimp/friction、gap、dt、控制频率和接口限制保持。真实contact.includemargin验证通过；producer、collector和完整CPU分析均通过。

| anchor seed | 速度峰比 C2→A1 | 原episode存活 C2→A1 /2048 | 严格均值 C2→A1（控制步） | 原contact p99上界 A1（mm） | 真实最大穿透 C2→A1（mm） | 所有冻结门槛 |
|---|---|---|---|---|---|---|
| 42 | 1.83731→1.58557 | 1102→1094 | 32.2266→32.6631 | 0.36 | 4.87783→4.33428 | 通过 |
| 43 | 2.59798→2.34944 | 1044→1038 | 31.3179→31.5913 | 0.36 | 7.78589→4.25090 | 通过 |
| 44 | 2.35974→2.31805 | 1074→1054 | 31.9175→32.3291 | 0.36 | 5.40555→3.81235 | 通过 |

三anchor首步终止均0；对相邻C2存活最多下降0.977pp，对原R0最多下降1.270pp，均≤2pp；严格均值同时高于两冻结基线。全局峰下降9.57%，三seed分别下降13.70%/9.57%/1.77%。原预设“至少两seed在完整/共同时间网格均下降≥10%”的清晰敏感性标签未满足，结果标`uncertain_or_opposing`；这里三seed方向同降但幅度不均匀，不把它描述为所有seed都有强峰改善。

有效contact计数8.36–8.66M→4.00–4.11M，0–10µm首bin比例4.26–4.38%→59.57–60.28%，不能只用contact p99证明因果。固定原episode的env×completed-substep最大深度分布另计：p99由2.139/2.170/2.169mm降至0.640/0.630/0.630mm；零深度比例由5.60–5.71%升至52.42–53.37%；样本总数变化很小（各约1.26–1.29M）。最大真实深度也同时降到5mm内，排除了“只靠正margin把contact分母稀释”的结论，但仍保留正margin接触正则化与真实硬件不同的限制。

同一环境的[匹配案例复核](artifacts/2026-10-14-rpo-learning-round8/matched-C2-A1-cases.json)进一步限制结论：旧三速度峰env1882/1808/177在共同原episode前缀峰比分别1.83731→1.55584、2.59798→2.34944、2.35974→2.31805；seed42全局最坏速度env换成888。该峰比取同env全部23关节，制动/超额定时刻才锁定旧case关节。旧深度env1783/108在共同前缀真实最大深度4.87783→1.60977、5.40555→3.48865mm；旧seed43 env453在A1中原episode仅到0.38s（C2到0.52s），旧最深时刻0.48s已不属于原episode，因此该时刻填null，**不能将旧7.786mm→新全局4.251mm解释成同case完整修复**。其共同前缀仅能比较1.09675→0.70420mm；seed42旧深度case共同前缀速度峰还从1.00557升至1.08633，需要保留该权衡。冻结门槛允许≤2pp存活变化，但不能因此消除删失偏差或证明每个旧case都改善。

A1新最大穿透分别发生于env1389/1049/1772、0.420/0.430/0.44375s，参考clip27的1034/1034/1033帧；仍为左足mesh–地面。保存三份实际积分前后q/v、电机/接触/直接限位力矩、target、reference快照；CPU同pose几何与历史A1深度差≤1.480µm，真实dist独立于includemargin .003。新深case的历史GPU contact/M/J/warmstart仍缺失并明确标记，CPU几何不冒称GPU采集。

实际制动时序也不能解释成峰时接触力矩必然减少：同env1808（seed43）总足Fz首次>20N由0.04375提前到0.03750s，首次饱和制动由0.07375到0.07250s，速度峰由0.08000到0.07875s；峰时接触力矩却32.331→33.670Nm，电机仍−27Nm。速度峰降低与力建立时序/轨迹变化一致，不能只凭不同峰时刻的力矩证明接触冲量减少。

旧legacy policy完整记录：峰比3.34181/3.33220/3.83964、最大穿透21.728/17.880/17.981mm、0.96s存活均0，峰与深度门槛仍失败，严格均值7.9629/7.9678/8.0483。这些失败不能隐藏，也不能把anchor门槛通过扩大成旧policy或真实v13站立/行走已通过。原始`rsi_unexpected_mismatch_rows=2048`仍是既有358离线/在线表达差异字段；本轮初态q/v与动作包精确相同，不重新解释成RSI丢失速度。

![C2到A1的真实接触深度对照](artifacts/2026-10-14-rpo-learning-round8/contact-depth.png)

![C2到A1的速度、严格长度和存活对照](artifacts/2026-10-14-rpo-learning-round8/contact_margin_comparison.png)

绘图首次glob误读`bounded-case-braking.json`而缺mode字段，已改为只读`replay-*-braking.json`，保留错误日志；最终六张Run对照/制动图生成通过，未改变数值或门槛。

确认契约冻结于05:45:50 Asia/Shanghai，SHA`3479a44973f299a53f126517d025ce4cacb2c3980e72ef644ad5be94f23c5241`；首A1 receipt SHA`5a2bda6d44a83186caef79c043a160c7d73fa5476ce979373de4d37ef9a697aa`。四GPU释放/doctor及同镜像preview确认后，使用第二且最后名额：[Job job_bfde676116574f4597d4](http://1.94.0.106:31800/api/v1/jobs/job_bfde676116574f4597d4) · [Run run_e58771a9cfa04abe8b9c](http://1.94.0.106:31800/train/runs/run_e58771a9cfa04abe8b9c)。仍是同A0动作、同seed/初态、同镜像A1、optimizer=0，不扫描新参数；最终要求对原R0/C2全部anchor门槛、六phase身份，以及对首A1六phase峰差≤2%。当前确认运行中。

### 阶段3：确认出现真实尾部反例，停止晋级

确认Run producer/collector均完成，Job succeeded、Run finished且无abort；这是执行成功，**不是物理确认通过**。一次状态读取在collector期间90秒无响应而超时，随后仅重读状态成功，没有重复提交或重试Run；独立分析仍对全部三份完整Run核验。

确认seed43原episode真实最大穿透5.750418mm（冻结上限5mm），发生在env453、0.47750s、reference `[0,27,218]`，就是旧C2深度尾部环境。[反例审计](artifacts/2026-10-14-rpo-learning-round8/confirmation-depth-counterexample.json)和保存状态按新receipt成员SHA核验。CPU同pose真几何5.750966mm，差0.548µm，左足mesh–地面；右足正距离1.279–1.831mm仍因margin进入候选接触，未减margin计算真实穿透。

峰时最快归一化关节为left_ankle_pitch，真实dq−6.94861rad/s、电机+20.0668Nm、接触−22.3308Nm、直接限位0，电机尚未到27Nm。CPU native同积分前q/v、真实电机输入、零warmstart重算Newton3次、力平衡残差8.82e−11N、该关节接触−21.8908Nm；历史GPU M/J/contact集合/warmstart/solver残差缺失仍明确，不能拿CPU收敛结果替代GPU求解质量证据，也不能据此盲增iterations。

首A1同env在0.38s已终止，确认则到0.52s。保存的223列interval中两次首次终止的training flags均`[position=1, velocity=0, torque=0]`，容差仍0.02rad。首A1最后控制段right_ankle_pitch最大位置越界0.0211101rad；确认最终段left_ankle_pitch为0.0471821rad。宏观height/tilt/protected-contact是否同时触发尚未独立拆分；至少位置违规及时间删失有直接记录，不能把首A1缺少0.4775s数据当作反例被修复。

同env的共同原episode前缀最大穿透两次都0.704197mm；23个受控关节速度最大差首次>0.01rad/s在0.07375s，>0.1在0.08250s，已排除根部线/角速度分量。没有保持相同后续存活范围，越过首次终止后的状态不能作同episode对照；这说明只检查全局速度峰复现或均值改善会遗漏接触尾部及终止边界的不稳定。所见差异可能涉及接触/浮点/约束路径，现有历史GPU字段不足以严格归因到某个求解选项。

![同环境A1确认穿透反例与真实制动](artifacts/2026-10-14-rpo-learning-round8/confirmation-depth-counterexample.png)

两名额已用完，无第三Job，无训练、独立模型评估或奖励/RSI候选；A1不准晋级。不因首Run门槛通过而放量。

### 阶段4：完整确认及预算关闭

[确认独立分析](artifacts/2026-10-14-rpo-learning-round8/confirmation/analysis.json)最终`all_phase_identity_passed=true`、`reproduction_passed=true`、`physical_screening_passed=false`、`confirmation_passed=false`。确认receipt SHA`8432ace93cc8c48f17d57cebf3b3e8b095490d3eba29d37d054e2436287ed8bb`。六phase初态/动作/目标和真实编译身份全部核验；峰复现最大相对差0.309861%（旧policy seed43），三anchor仅0.000684%/0.000974%/0.004320%。这只通过已冻结的峰值入口检查，**不能称精确轨迹复现**：共同原episode网格23受控关节最大速度差anchor5.14449/4.77473/4.65997rad/s，policy12.97293/11.86085/13.23733rad/s。[独立单位口径审计](artifacts/2026-10-14-rpo-learning-round8/joint-repro-audit.json)验证这些极值确属受控关节；原通用qvel比较含6个根DOF，不能不加区分统一解释为rad/s。

| 确认anchor seed | 峰比 | 存活/2048 | 严格均值（控制步） | 真contact p99上界（mm） | 真实最大深度（mm） | 结果 |
|---|---|---|---|---|---|---|
| 42 | 1.585581 | 1091 | 32.671875 | 0.36 | 4.077195 | 所有冻结条件通过 |
| 43 | 2.349415 | 1039 | 31.568359 | 0.36 | **5.750418** | **最大穿透失败** |
| 44 | 2.317951 | 1053 | 32.300781 | 0.36 | 3.698349 | 所有冻结条件通过 |

首步终止均0，对R0/C2的存活及严格均值均符合原冻结条件；唯一anchor失败项为seed43最大深度。旧policy峰3.34162/3.34253/3.83964、深度p99 3.59/3.32/3.42mm、max21.752/17.880/17.981mm、末存活均0，仍不通过且未被anchor汇总隐藏。六phase汇总图/逐phaseJSON、复现对照、最坏anchor/policy制动及六旧case/反例图齐全，代表图已实际查看；[媒体校验](artifacts/2026-10-14-rpo-learning-round8/media-check.json)确认21张PNG均可解码。这轮没有独立模型评估，也没有生成或宣称模型评估录像。

![确认Run真实最大穿透及旧policy失败](artifacts/2026-10-14-rpo-learning-round8/confirmation/contact-depth.png)

[预算结果](artifacts/2026-10-14-rpo-learning-round8/budget-result.json)和[完整证据索引](artifacts/2026-10-14-rpo-learning-round8/evidence-index.json)已核验。实际2/2Job、每Job11.730/11.679分钟，Run合计23.409分钟；两个完整Run所有800文件（含console、缓存、配置、原始数组）合计9,074,782,747bytes≈8.452GiB，163份派生产物128,580,551bytes≈122.62MiB。最终索引完成于06:15:47 Asia/Shanghai，从05:05:50契约冻结起69.964分钟，部署/排队/Run/收集/CPU独立分析及最终单位/媒体检查计入其中，均在本轮90分钟/12GiB限制内；阶段0旧离线计时缺口另见前文，不冒称它有完整计时证明。索引绑定两新Run全部文件、六旧case源成员与所有派生state/history/CPU模型/脚本/JSON/PNG及镜像source receipt；大数组仍保留原Run，不复制到文档目录。

最终health中ebd.ok=true，doctor healthy=true，四GPU run_id均null且空闲约24080MiB，没有遗留GPU Job。未用旧三轮第9名额、没有第三新Job、没有训练、没有Git提交。[关闭决策](artifacts/2026-10-14-rpo-learning-round8/decision.json)为`stop_adjust`：A1典型穿透和速度峰有所改善，但最大深度不能跨确认稳定满足5mm，因此不晋级采样训练。

### 三轮复核与后续执行入口

1. **语义与因果复核**：速度峰/深度最大是不同case，CPU接触集合差不能外推成GPU错误；首A1的env453提前终止导致尾部删失，明确保留null与旧/新极值。初始q/v相同也不能证明所有求解隐状态已冻结；现有字段不能严格区分warmstart、接触构造、float或停止误差，不把假设写成根因。
2. **预算与晋级复核**：新预算独立冻结、两Job均经Job/Run/ebd，GPU0保留但无learner，GPU1–3三rank×2048，0更新/无DDP。确认只在首A1六phase身份及全部anchor门槛通过后启动；确认失败停止，不能按峰复现小就放行，不能用其它候选预算补第三Job。采样mode覆盖修复在实际v13未激活，仍不能运行名义无变化对照。
3. **证据与出口复核**：39数值/49平台检查与独立小型复核已通过；两完整receipt/源文件hash、实际margin/gap/includemargin、真实depth及分母、相同env删失/制动、图表和完整预算索引齐全。唯一anchor失败明确，旧policy失败未隐藏；GPU深case M/J/warmstart及完整solver停止证据仍缺，硬件27Nm/8rad/s类别未知，不宣称模型/真机合格。

下一步先复用本轮保存的env453首A1/确认状态及A1深度快照，核对终止前控制段、右踝位置越界1.11mrad余量与左足接触压入；不提高27Nm/8rad/s或0.02rad训练容差，不把深度超限减margin。需要新GPU证据时另行冻结有限预算，先做同A1物理配置的固定case诊断：记录初始化和每次reset的q/v、warmstart/相关求解状态；env453速度分歧、首次位置终止及最大深度前后保存contact点/geom/normal/dist/includemargin、M/J、efc_aref/D/R、实际电机/约束分解、GPU迭代/残差与同一时刻对应。对这些字段建立实际运行身份后，再据证据选**一个**控制/接触/solver候选，保持本轮接口/奖励/RSI不变。

不预排4mm等margin扫描、再增Kd或盲加iterations。只有必要物理门槛在同配置确认也通过，才派生实际v13训练运行条件，证明采样候选有真实唯一差异，按原计划冻结100更新→逐100决策到500的完整恢复/原指标/媒体预算；本轮没有消费训练额度，32步anchor严格均值也不能替代训练末两窗35步及独立验收。
