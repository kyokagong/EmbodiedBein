# RPO BFM 第四轮：积分步长敏感性与制动时序

日期：2026-10-13。用户授权继续第三轮所提出的积分步长验证；遵循学习优化计划的动力学分支，物理通过后才考虑采样修复训练。第三轮 C1 软地面已停止，不带入本轮。

## 1. 执行前冻结

本轮最多 **3个Job，每个≤30分钟，总运行时间≤90分钟，所有Run输出合计≤20GiB，无自动重试，learner/optimizer更新0**。顺序：I0新诊断入口按原积分设置复现第三轮B0，同时采集旧失败case连续制动时序；I1仅积分dt减半；第3个名额只用于I1产生的新失败case时序补证/相同配置物理复核，或明确根因的执行修正。已覆盖所有必要case时不消耗第3个名额，不扫描PD/接触参数。训练预算另行冻结。

| 项目 | I0 | I1 |
|---|---|---|
| physics dt / 子步 | 2.5ms / 8 | 1.25ms / 16 |
| control dt / Hz | 20ms / 50Hz | 同左 |
| 接触参数 | 原地面solref=[.02,1]，实际混合[.0105,1.5] | 同左 |
| 模型/RSI/接口/奖励/PD/几何/惯量/摩擦/solver | 原v13冻结条件 | 同左 |

模型 Run `run_bc7df2383d8f498bb08d` step600、原动态RSI数据 `ds_fb0ecda9e70644f2afdd`、接口 `ee_rpo_v2_bfm_23dof_v2`、seed42/43/44、GPU1–3各2048环境、4卡准入、anchor/旧policy各48控制步保持。奖励定义/权重和归一化不变：速度罚按每20ms区间的完成子步时间平均、暴露率按区间占比、冲量为ΣF·dt；控制步末位置等奖励和slew保持。16子步的新增采样分辨率不当作奖励增权。训练/严格终止在全部完成子步锁存，223列trace保持字段含义，仅实际dt与子步数准确更新。20ms RSI前瞻不随积分dt改变，q/dq/root运动量不重编码、不置零。

必要物理门槛沿用第三轮：anchor各rank峰值比≤2.5、48步存活不低于I0−2个百分点、严格影子均值不低于I0、首步训练终止≤0.1%；完成子步原episode接触穿透p99上界≤2mm、最大≤5mm。I0须与第三轮B0逐rank/mode相同reference/52初态，峰值相对偏差≤2%，再进入I1。原计划≥35步训练末两窗、恢复/周期评估/媒体条件仍保留；48步诊断不能替代模型验收。

敏感性判读预冻结：同时报告全部物理子步峰值及共同2.5ms时间网格的峰值，避免把16子步观测更密误判成物理变坏；48控制步终止/存活按相同真实时长比较。anchor至少2/3 seed峰值改善≥10%为明确变化信号；各seed及整体峰值变化≤5%为本次峰值指标对dt弱敏感；其余为不确定/异向变化，不推导dt问题已解决。该分类不是训练放行，仍需全部物理条件。

制动时序优先选第三轮B0的velocity_peak、braking_peak、contact_peak、normal_landing、position_excursion原始episode case，按rank/mode去重（最多5个）。记录完成子步时间、q/dq前后、target/requested/applied、实际电机/接触/限位广义力、足接触力、reference和original_alive；用于判断饱和制动与首次接触/首次额定速度越界的先后，而非从单个峰值猜Kd。若I1最坏case不在已有集合，使用第3个Job对I1保留的事件重新冻结case补证，物理配置仍I1。

## 2. 实现和阶段记录

执行前工作区差异保存于 `/tmp/rpo-round4-start-status.txt`，不覆盖已有修改、不创建Git提交。复用原interval、MJLab、动力学phase、contact统计和Job/Run/ebd生命周期；只增加两档有限积分设置和时序sidecar，发布v13镜像/接口/数据不变。

Ruling：上一轮批准的下一步和本轮明确指令已提供积分设置/控制周期/保留项的授权，继续实施，不重复请求批准。I0按默认8子步复现是新诊断语义的必要检查，不是额外短训练；原policy采样bug保留用于物理单因素对照，准备的修复镜像不提前训练。

新增模块和平台分支的最终容器检查：数值/配置/receipt **56 passed**，平台/容量/调度 **47 passed**；仅数值镜像缺pytest-asyncio产生已有配置警告。独立只读复核 **52 passed**，未发现静态阻断；随后4个轨迹负例也通过。拒绝错误dt组合、控制频率额外覆盖、来源hash错误、时序错位、alive复活、连续q/v断裂及约束力分解不一致。CPU编译检查确认两档除积分dt外的物理字段相同。

预算契约SHA256 `bf8c64b97e0f85b4f7e763d6ddc7fa697f3f64a116bf89188baade21d78015c2`；诊断镜像 `eb/rlinf:rpo-ufo-integration-factor-v1` / `sha256:1b375d7d6de390479329edafc9515aa932a9987d90d8b801585e7a37f89205e7`。内置源码hash见[image-source.json](artifacts/2026-10-13-rpo-learning-round4/image-source.json)；原policy hash仍为`f884aeae9768d40c447e6b2aae16b2a8aa62cf727b70b285c931f48971ed09e2`。I0/I1请求已落盘，均绑定第三轮B0 receipt和本轮预算，不改变已发布接口/模型。

## 3. 阶段执行记录

### I0提交

I0平台预览ok、健康检查ebd.ok，4卡均空闲后提交。[Job job_7eb33d22d7864f9293d0](http://1.94.0.106:31800/api/v1/jobs/job_7eb33d22d7864f9293d0) / [Run run_6f12c4d9da234eda8ad7](http://1.94.0.106:31800/train/runs/run_6f12c4d9da234eda8ad7)。这是asset.validate诊断，优化器更新0，不会产生训练模型记录；Run页查看日志/原始证据。

时序操作定义：first_foot_load_gt20N为双足|Fz|合计首次超过20N，与既有事件选择阈值一致；另报>1e-6N的首次可测载荷。均不是零力几何接触点的精确起始。制动为τ·dq_before<−1e−6，饱和为|τ|≥99%接口力矩限位，额定越界为|dq_after|>速度接口+0.001rad/s。时刻是完成子步末端；若第一个样本已制动，不能推断未记录的更早起点。初态仍有原始动量。源码复核纠正了离线事件1基控制步/0基子步的时间索引，并明确第3Run应映射回I1来源事件而不是只报告自身事件覆盖。

本节保留提交和异常后的阶段决策；最终Job/Run、物理/制动证据、图表与未验证项见第4–8节。

### 执行修正（消耗预留第3名额，预算不扩张）

首次I0失败：Job failed、Run aborted/exit_1。记录路径的原生worldid int32直接用于torch.scatter_reduce_，触发`Expected dtype int64 for index`；3个anchor子进程均在第一批时序记录退出。未产生完整receipt、不能用于物理结论；失败日志保留于[I0-failure-logs.json](artifacts/2026-10-13-rpo-learning-round4/I0-failure-logs.json)。

仅将诊断穿透统计索引转为long，派生不可混淆的v2镜像`sha256:564e13b45162d06561f9ddf105c65730a18735a2db8f47f7ba5ce0f786251b9d`，不覆盖v1镜像。新增使用实际打包模块、原生int32 worldid的CPU回归：v1准确复现同一RuntimeError，v2须通过后再提交。此前测试桩nacon误用numpy数组，已修正成提供.numpy()的tensor；该测试桩失败也不计作物理证据。

后续顺序收紧为：第2Job修正I0 → 复现通过后第3Job I1。原本预留的新case/相同配置复核名额已用于执行修正，不能再开第4Job；若I1新极值case连续轨迹未覆盖，如实列出缺口并关闭本轮。积分参数、控制、接触、限位、奖励、RSI和冻结预算均不变。修正请求与镜像来源hash见`correction-sha256.txt`。

v2最终检查 **57数值 +47平台 passed**，细节及容器命令见[verification.md](artifacts/2026-10-13-rpo-learning-round4/verification.md)。健康检查和修正请求预览均ok后，第2Job启动：[Job job_297c26b417bd46d4bb08](http://1.94.0.106:31800/api/v1/jobs/job_297c26b417bd46d4bb08) / [Run run_743dba585660411ca866](http://1.94.0.106:31800/train/runs/run_743dba585660411ca866)。待完整receipt及独立CPU复算后决定是否进入I1。

### I0阶段结果与条件预算决策

第2Job succeeded/Run finished，完整receipt通过平台和独立CPU校验。与第三轮B0的6组reference和52在线初态完全相同，privileged最大差≤4.77e−7；compiled唯一变化为新增描述字段integration_actual。3组anchor存活仍1115/1060/1079，峰值相对偏差0.00255%/0.00266%/0.00534%，基线anchor复现。

但policy seed42全局峰3.437893→4.283543（+24.598%），违反已冻结的全部phase≤2%复现规则，**停止I1，不放宽容差**。policy seed43/44最终极值虽接近，逐env分布也分叉：3个seed分别463/479/474个env峰差>0.05，严格首次违规长度有254/273/261个env不同；最早原episode存活计数差出现在0.0425s。不能把5/6最终max接近称为闭环逐轨迹复现。

独立复核：旧seed42最坏env1522的新同时间dq全向量最大差0.0004233rad/s、target差2.22e−5rad、motor差0.002628Nm，旧极值本身复现；新最大换成mixed非tracking env1756/reference[0,32,21]，其旧峰仅1.225502，新峰4.283543，旧新严格首次违规均第3步。新峰−33.2028→−34.2683rad/s，motor+27Nm已反向制动，contact−32.5203Nm、限位力0。说明极值case切换和闭环分叉，**尚不能唯一归因GPU非确定性、新trace或其他因素**。旧policy仍有mode覆盖bug，不能把随机动作噪声作为已证实解释。

根据冻结预算允许的条件“同配置复核／新case补证”，**最后1个名额改用于I0复核并记录env1756，而非在基线未通过时提交I1**；该选择在提交前再次冻结于[I0-confirmation-contract.json](artifacts/2026-10-13-rpo-learning-round4/I0-confirmation-contract.json)，SHA`7c5ccfc152fb49f2830ea595a488beed0fae62513ca005705c4894862cf1041a`。预算总量仍3Job，门槛保持，不自动重试，无第4Job。只改变诊断case来源为本轮完整I0 receipt（SHA`8645106c193ea0564e1d4035011df73e29321c23bf80de85bdd4e676e318f7f5`），全部物理/控制/接口/奖励/RSI不变。I1的1.25ms敏感性结果仍**未验证**。

## 4. 最终Job/Run与预算关闭

| 阶段 | Job | Run | 平台结果 | 实际运行秒 |
|---|---|---|---|---:|
| 首次I0，v1记录错误 | [job_7eb33d22d7864f9293d0](http://1.94.0.106:31800/api/v1/jobs/job_7eb33d22d7864f9293d0) | [run_6f12c4d9da234eda8ad7](http://1.94.0.106:31800/train/runs/run_6f12c4d9da234eda8ad7) | failed / aborted，exit1 | 160.692574 |
| 修正I0，v2完整制动轨迹 | [job_297c26b417bd46d4bb08](http://1.94.0.106:31800/api/v1/jobs/job_297c26b417bd46d4bb08) | [run_743dba585660411ca866](http://1.94.0.106:31800/train/runs/run_743dba585660411ca866) | succeeded / finished，复现门槛失败 | 376.866090 |
| I0同配置复核／新case补证 | [job_467d198c673e427a9cbe](http://1.94.0.106:31800/api/v1/jobs/job_467d198c673e427a9cbe) | [run_c35abe2b5fce4120a2ec](http://1.94.0.106:31800/train/runs/run_c35abe2b5fce4120a2ec) | succeeded / finished，物理门槛失败 | 359.596836 |

共 **3Job /897.1555秒（14.95分钟）**，outputs目录含Warp缓存和失败证据共 **264,466,550 bytes（约252.21MiB）**，console日志91,733 bytes，全部Run保留产物合 **264,558,283 bytes（约252.30MiB）**，低于90分钟/20GiB。无自动重试、无优化器更新；预算已关闭，GPU0–3均释放。每Run≤30分钟；失败Job也计入数量与时长。[预算审计](artifacts/2026-10-13-rpo-learning-round4/budget-audit.json)来自平台Run起止时间只读SQL和所有输出递归索引，不以Job创建时间估算。

两个成功Run分别有6个完整phase、86个root receipt成员、396个全目录文件；首次失败Run325个文件保留。完整文件路径/大小/SHA见三份`run_*-output-index.json`；root receipt及其hash另存。Job是asset.validate，Run页面有诊断记录，训练任务列表不会新增训练模型。**平台执行成功与物理通过分开。**

## 5. I0复核结果与物理门槛

两个成功I0逐phase的reference/52初态精确相同，privileged差≤4.77e−7；compiled模型字典完全相同。所有实际积分仍是**8×2.5ms**，控制20ms；地面参数与混合solref仍[.0105,1.5]，接口/PD/奖励/RSI保持。1.25ms/16子步仅完成CPU配置及数值语义检查，**没有实际I1 Run，因此本轮不能判定积分步长弱敏感、有效或无效**。

| mode/seed | 修正I0峰值比 | I0复核峰值比 | 严格影子均值：I0→复核 | 48控制步存活：I0→复核 | 穿透p99上界mm：I0→复核 | 穿透最大mm：I0→复核 |
|---|---:|---:|---:|---:|---:|---:|
| anchor42 | 1.904368 | 1.904406 | 32.0215→32.0269 | 1115→1115 /2048 | 1.57→1.57 | 5.9200→5.9202 |
| anchor43 | 2.819167 | 2.819203 | 30.9775→30.9746 | 1060→1060 /2048 | 1.58→1.58 | 5.8158→5.8197 |
| anchor44 | 2.651168 | 2.653345 | 31.4761→31.4697 | 1079→1079 /2048 | 1.59→1.58 | 5.4689→5.4678 |
| policy42 | 4.283543 | 3.437912 | 6.9575→6.9614 | 1→0 /2048 | 4.91→4.92 | 18.8588→20.3598 |
| policy43 | 3.348412 | 3.348363 | 7.0088→7.0107 | 0→1 /2048 | 4.77→4.76 | 24.8102→24.7755 |
| policy44 | 3.927786 | 3.927779 | 7.1899→7.2114 | 0→0 /2048 | 4.72→4.73 | 19.3968→23.9317 |

anchor三个seed首控制步训练终止均0；0.96秒总存活3254/6144=52.9622%，两次相同。两次policy存活都仅1/6144，存活者换seed，不能当改善。复核anchor43/44峰值仍>2.5，三组最大穿透均>5mm，物理门槛失败。严格均值43/44小降也不满足预冻结“不得低于I0”的条件；降幅很小，不把它解释成显著算法退化。上述48步诊断不是训练末两窗≥35步或独立质量验收。

policy42在同配置复核中从4.283543降到3.437912（−19.7414%），违反相对当前I0≤2%的复现条件；其结果又接近历史B0。因此“复核相对历史B0通过”不能取消“当前I0之间不复现”，更不能把这次下降归因积分优化，积分没有改变。anchor峰值和末存活接近，也不意味着所有内部状态/接触时序逐行相同。

原始指标及判据见[I0-confirmation-analysis.json](artifacts/2026-10-13-rpo-learning-round4/I0-confirmation-analysis.json)。图表均是相同I0配置重复运行，不是I0/I1对照：

![相同I0设置复核](artifacts/2026-10-13-rpo-learning-round4/I0-confirmation.png)

![原episode完整时序](artifacts/2026-10-13-rpo-learning-round4/I0-timeseries.png)

![原episode完成子步穿透](artifacts/2026-10-13-rpo-learning-round4/I0-contact-depth.png)

## 6. 连续制动证据及覆盖限制

全部时刻为本Run起点后的完成子步末时刻，额定速度按当前接口8rad/s、力矩27Nm。下表只针对所列关节；“首次制动”不是持续制动开始，不排除之后转向/驱动阶段。

| case | 首次足载荷>20N | 首次该关节制动 | 首次额定越界 | 首次饱和制动 | 峰值时刻/速度 | 峰值motor/contact/limit Nm |
|---|---:|---:|---:|---:|---|---|
| anchor43 env1808，右踝pitch，两个I0 | 45ms | 2.5ms | 72.5ms | 75ms | 80ms，21.86→22.55rad/s | −27 /+30.41 /0 |
| policy44 env798，右踝pitch，两个I0 | 145ms | 37.5ms | 145ms | 147.5ms | 150ms，26.90→31.42rad/s | −27 /+47.83 /0 |
| policy42 env1756，左踝pitch，复核I0 | 50ms | 2.5ms | 395ms | 397.5ms | 397.5ms，−10.086→−10.406rad/s | +27 /−27.62 /0 |

anchor1808的target仍初始−0.153034rad，峰值前q约−0.01813rad，初始dq仅−0.08343rad/s；policy798初始dq0.13368rad/s，峰值target0.133776rad、q约0.223847rad。两次运行中的前两条事故相近地复现。存在“速度超过额定后才饱和、饱和反向制动仍继续增速”的时段，支持继续检查接触冲量与可用制动余量，不能仅靠增大Kd假定解决；已饱和时提高Kd不会增加当前接口允许的27Nm。

首次越界子步还提供了积分候选的直接依据：[first-speed-violation.json](artifacts/2026-10-13-rpo-learning-round4/first-speed-violation.json)中，anchor1808在72.5ms从−0.01089跳到13.67483rad/s，motor仅+0.12193Nm、contact+58.30799Nm；policy798在145ms从0.94065跳到18.33986rad/s，motor−1.43038Nm、contact+80.54214Nm。两者下一2.5ms子步才达到−27Nm反向饱和。这个“冲击发生的子步制动尚未满幅，下一子步才饱和”的事实使dt减半仍值得验证；它**不证明**1.25ms会通过门槛，也不等于50Hz控制器额外延迟一个控制周期。完整耦合动力学和PD离散更新影响应由冻结动作回放的I0/I1区分。

![anchor43原episode制动时序](artifacts/2026-10-13-rpo-learning-round4/braking-anchor.png)

![复核policy最坏已记录case制动时序](artifacts/2026-10-13-rpo-learning-round4/braking-policy.png)

**env1756的旧极值事故未复现。** 修正I0只有它在360ms的事件快照，没有该case此前连续动作/z/RNG轨迹；复核补齐了它的原episode连续记录，但峰值仅−10.406rad/s，而非前次−34.268rad/s。同reference、同360ms的qvel全向量最大差37.053rad/s，qpos差0.4144、target差0.5050rad、motor差48.9607Nm。不能把同ID/同reference/同时间有合法轨迹称为复现原事故、补齐原事故前史。

![env1756复核专属图；前次未记录该case连续轨迹](artifacts/2026-10-13-rpo-learning-round4/braking-policy-seed42-env1756.png)

源事件共30条，3条policy contact_peak在复核对应时刻已不是原episode：seed42 env571、seed43 env219、seed44 env1554。其reset后的轨迹虽被保存，**不得拿来替代原episode事件**。case-time覆盖与source snapshot等价分列；state/target/motor点比较复用1e−6 atol/rtol，仅是审计标记，不是新增验收门槛，也不证明整段轨迹相同。详细时刻、before/after q/dq、动作/target、actual motor/contact/limit/other constraint和来源覆盖见[braking-analysis.json](artifacts/2026-10-13-rpo-learning-round4/braking-analysis.json)。

力使用积分前solver解、速度使用积分后状态、穿透使用forward刷新后的几何。policy798峰值后的穿透可以是0，不能反推之前没有接触冲量。接触广义力大于反向电机力矩只是同关节的证据，完整加速度还由质量矩阵耦合、bias/passive和implicitfast积分决定；不把M⁻¹Fdt当成精确dv。动态RSI保留了root/关节动量，anchor不是零速静态站立实验。27Nm/8rad/s仍是现有接口值，没有新增硬件额定/峰值资料。

## 7. 三轮复核与本轮决策

1. **设计/物理/数据语义。** 两档配置均由CPU编译验证，50Hz、原接触、PD/质量/惯量/摩擦、奖励权重、20ms RSI前瞻保持；默认8子步回归和16子步区间均值/冲量检查通过。实际只有I0。52初态精确、358部分速度有限差分/瞬时差异沿用第二轮解释；phase中rsi_unexpected_mismatch_rows不能被误读为参考损坏或置零动量的理由。
2. **GPU/预算/恢复/失败。** 所有GPU运行均走API→Job→Run→ebd；4卡租约，GPU0空闲、1–3各2048，无learner。失败Run保留，独立v2派生，不覆盖v1；同配置复核占最后名额。无新增训练/恢复/checkpoint，不假称100/500流程或周期评估完成。57+47检查通过，代码/镜像/请求/契约hash保留，无Git提交。
3. **图表/完整证据/独立审查。** 两个完整receipt均通过主线程和独立只读CPU验证，12个phase全部连续q/v、alive锁存和constraint分解校验通过；6张图已生成并实际打开检查。明确复现失败、3条来源时刻缺失、env1756事故前史仍缺，不补造媒体。本轮是动力学诊断，无成功/失败评估录像，也不构成MJLab评估出口或模型质量验收。

**决策：关闭第四轮预算；停止I1入口；不启动采样修复训练。** 已完成有限预算冻结、两档积分语义实现/CPU验证、I0原接触复核、主要旧case制动时序、新case记录和证据完整性审计；积分1.25ms敏感性对照未执行，分叉根因未证实，不能把本轮结果写成积分修复无效。

## 8. 下一轮建议（本轮未执行，不追加预算）

优先把闭环极值分叉隔离出来，再做积分对照；不先放宽2%门槛、不改变接口27Nm/8rad/s，不改奖励/RSI/PD/接触，也不靠更多500更新掩盖物理不确定性。

- 先增加诊断证据：每50Hz控制步记录选定case原始输入state/history、实际requested/applied/target、z、mixed/tracking/reference/cursor和original episode身份；全局记录CPU/CUDA/NumPy/Python RNG状态hash及reset ID集合，冻结policy mean/mode行为和norm/model身份。在**发生事故的同一个Run**保留前史，避免再次等峰值后换Run补证而事故路径消失。记录不修正旧mode bug，以免混入训练采样因素。
- 可同时保留所有6144env的48步requested动作包，仅float32动作约25.875MiB，绑定初态、接口/codec、动作slew、模型和完整hash；不需要保存所有env的巨量history。用途是诊断回放，不能作为新专家数据、不能混入acceptance。
- 下一有限预算可预案为最多3Job、每≤30分钟、总≤90分钟/20GiB：①I0带上述日志并冻结动作包；②同初态/相同动作包回放I0，检查前史/峰值/穿透/终止复现；③前项通过才用相同包回放I1（16×1.25ms）。控制仍50Hz，目标转换与PD保持，奖励时间平均保持；在original_alive共同前缀比较，reset后不拼接。该回放隔离闭环动作分叉，使积分物理因素可识别；它不能代替闭环策略质量评估。各阶段均须新Job/Run和执行前独立冻结，当前没有提交这些作业。
- 若首批动作/z已经不同，先沿输入、ctx/RNG和批量reset路径找来源；若动作相同但物理轨迹分叉，沿接触排序/求解、CUDA同步和数值敏感性补证。GPU非确定性目前只是待验证解释；不要预设唯一根因，也不要用浮点限幅隐藏分叉。
- 通过回放与物理门槛后，再补原计划的真实100更新边界/pending checkpoint、同语义完整恢复、周期决策、MJLab独立快评及媒体链路，派生sampling-fix独立recipe并冻结100→500单因素训练预算。准备好的sampling-fix镜像不能直接代替上述条件，v13旧完整状态也不能冒充新采样语义恢复。

## 9. 第五轮追溯纠正（保留第四轮原始记录）

第五轮A0运行失败后核对实际phase配置，发现AGENT_CONFIG_KEYS过滤了`context_profile`/`ankle_control_profile`；旧policy诊断实际走legacy全局context，`mode='train'`保留随机动作，并未进入导致mode覆盖的逐环境分支。因此上文关于“旧policy mode覆盖排除随机动作解释”的推论应撤回，旧env tracking/mixed标志也不等同实际legacy policy tracking ID成员身份。原receipt/峰值/失败/物理未通过及2%基线门槛均不改写。新的动作/RNG/context证据和复现结果见[第五轮](2026-10-13-rpo-bfm-learning-round5.md)；该问题不证明GPU非确定性唯一来源，也不把旧诊断policy结果冒充v13实际训练context行为。
