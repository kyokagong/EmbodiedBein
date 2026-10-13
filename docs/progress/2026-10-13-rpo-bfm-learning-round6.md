# RPO BFM 第六轮：固定动作积分敏感性与证据驱动候选

日期：2026-10-13（Asia/Shanghai）。属于用户批准的最多3轮9Job总预算第2轮，前轮见[第五轮](2026-10-13-rpo-bfm-learning-round5.md)。接口限位、奖励和动态RSI保持，不提交Git。

## 执行前冻结

本轮最多3Job、单Job≤30分钟、合计Run运行时间≤90分钟、全部Run输出≤20GiB、retry=0、optimizer=0。第五轮已耗3Job/1432.573520秒/7435979692字节（完整Run含失败/缓存/console.log），全局剩余最多6Job；实际物理通过即停止，不消耗未用额度。

首Job仅在R0独立复现通过后执行R1：原动作包来自A0 `run_4f8ea733687e4f389082`，receipt `d4638074a861a3967a8e45f3d6561d98ce7dd86f6dbf448e59a9dd154f1683ea`；比较基线R0 `run_1005015a4cfa418bb8c0` / receipt `028388bf3725b022cd085f00854261fe8e0b30b7adb480dbd92343b04dca6135`。唯一因素I0(8×2.5ms)→I1(16×1.25ms)，控制仍50Hz，原接触/PD/solver/惯量/摩擦/几何/限位/奖励/RSI/模型保持。48步/3rank×2048/seed42–44/anchor及旧诊断policy动作全部同包，actor不调用；不称旧诊断policy为v13真实training context。

初态/reference/动作/原episode共同前缀target须一致；编译模型只有积分dt相关字段变化。比较全部子步和共同2.5ms网格；anchor至少2/3seed峰值改善≥10%为明确变化，所有seed及整体两种网格变化≤5%为弱敏感，其余异向/不确定。均非放行判据。

物理必要条件沿用：anchor每seed峰值比≤2.5、48步存活≥R0−2pp、严格影子均值≥R0、首控制步训练终止≤0.1%、原episode完成子步接触深度p99上界≤2mm及最大≤5mm。policy全部报告。I1若全通过，第二名额用于同动作包/相同配置复核；若未通过，根据实际首冲击与制动/穿透证据选择一个允许的PD/接触/solver单因素，并在提交前追加独立候选契约，不预扫参数。第三名额也须前阶段明确证据与记录，失败占名额不自动重试。

本轮只验证物理必要条件；≥35训练末两窗、100更新边界/完整恢复/独立快评与媒体出口仍未通过，训练预算须独立冻结。已有仅mode/cache修复候选对v13真实训练是否产生采样差异也须先核验，不能启动无效名义对照。

## 阶段记录

R0完整独立比较已确认：全部6phase初态/reference/52/requested/共同前缀applied-target和compiled一致，峰差最大0.22990%<2%，冻结入口通过。逐轨迹仍有明显qvel差，见第五轮；本轮不是exact replay，物理通过须复核。R1即将提交，实际Job/Run与结果逐阶段补入。

### R1启动

合同SHA `40e47397639e2b35ba6ef3997443b4f01037fb1ea19d93d0697942ef274dbdf8`；镜像仍为第五轮修正v2 `sha256:67085ed631fa00c912c0f36841d21a5d143c05438834f3bbf10928ca507e598a`，不添加其它物理因素。健康ebd.ok=true、预览ok后启动：[Job job_078844a0789d4084bbe8](http://1.94.0.106:31800/api/v1/jobs/job_078844a0789d4084bbe8) / [Run run_8a8b89522bdf4487a1af](http://1.94.0.106:31800/train/runs/run_8a8b89522bdf4487a1af)。总批准预算第4Job，本轮第1Job；更新0，源动作包只读挂载。

### R1完整证据与下一候选执行前冻结

R1六phase及完整receipt生成，SHA `97654be64d3a3de65c15daf6678e13484084735fbe1160705583321a41ae510d`；独立CPU全面复算运行中，不用部分phase替代最终对账。anchor峰1.84833/2.66710/2.37765，严格均值32.39551/31.26563/31.97852，首步终止0；穿透p99上界1.75/1.78/1.79mm、最大5.386/5.396/6.547mm，必要门槛未全满足。只有一个seed峰改善≥10%，不能先称明确dt修复。完整统计和两网格结果待独立分析。

已关闭事件快照的condim=4、pyramidal约束每contact6行。按±边成对平均提取法向Jacobian、总边force求法向广义力，再以total−normal求切向/扭转剩余；类型/行数均已assert校验。最坏anchor踝峰接触总力矩28.1375/31.3227/27.5861Nm，其中normal13.0086/9.7184/11.8111、切向/扭转15.1289/21.6043/15.7750（约53.8%/69.0%/57.2%）。这是峰值事件分解，不是首次冲击整段加速度归因，且不直接等同完整耦合dv。

据此冻结下一候选 **S1：相对I1仅cone从pyramidal(0)改为elliptic(1)**，dt仍16×1.25ms、50Hz、同A0动作包，原geom.solref/solimp/friction/condim/priority/solmix、Newton100/容差、PD、质量/惯量、接口限位、奖励和RSI保持。不提高电机能力，不再盲目软化地面。选择依据为摩擦分量占主要剩余驱动力，而非猜Kd或硬件参数。[MuJoCo 3.8说明](https://mujoco.readthedocs.io/en/3.8.0/modeling.html#solver-settings)指出锥类型改变接触优化模型，并非仅换算法；本轮将其明确作为接触摩擦锥单因素。冻结MJLab1.4.0支持cone配置，MuJoCo/MJWarp3.8；[MJWarp支持范围](https://mujoco.readthedocs.io/en/3.8.0/mjwarp/index.html#feature-parity)与实际CPU配置核对。当前尚无GPU候选效果证据。

S1只有R1完整独立对账、配置/实际源码与平台回归检查通过后才能提交，使用本轮第2名额。必要物理条件对原R0及相邻I1均报告；不能以旧policy坏动作掩盖anchor门槛。全通过后第3名额复核相同S1；未通过则按证据关闭/调整，剩余第7轮最多3Job，总9不扩。

### R1独立复核关闭；S1提交前检查

平台Job succeeded / Run finished，运行697.261586秒，完整Run409文件/4517017755字节（含console.log及缓存）。全局已用4/9Job、2129.835106秒、11952997447字节；本轮剩2名额，全局剩5。依据[完整预算](artifacts/2026-10-13-rpo-learning-round6/R1-budget-audit.json)，并非只统计receipt文件。

独立验证所有98项receipt文件、6phase动作/RSI初态/实际编译及30个原episode事件。全部初态52/reference/full qpos/qvel、48 requested及共同前缀applied/target一致。最初分析脚本错误要求scene XML随dt变化，导致identity误报false；MJLab将积分配置施加于compiled model，scene XML相同。保留[初次误判记录](artifacts/2026-10-13-rpo-learning-round6/I1-initial-analysis-check-error.json)，修正后[独立结果](artifacts/2026-10-13-rpo-learning-round6/analysis.json)确认编译差异仅physics_dt_s、decimation、integration_actual、compiled_solver.timestep，未改变其它物理字段。此修正没有改变任何数值门槛或失败结论。

峰值相对R0改善2.735%/5.398%/10.388%，共同2.5ms网格anchor峰与全部子步相同，仅一个seed改善≥10%；判定积分敏感性不确定，不能称已修复。存活1118/1058/1080，严格均值与首步终止通过；seed43峰、三个seed最大穿透未通过。policy峰3.01899/3.17685/3.80905，48步存活0/0/1；这是旧诊断固定坏动作的结果，未冒充真实v13训练。

最坏anchor同Run制动时序：seed42加载50ms→首次超额定71.25ms→首次饱和反向制动73.75ms→峰80ms；seed43为43.75→72.5→73.75→80ms。seed43峰仍dq21.3368rad/s、电机−27Nm、接触+31.3227Nm；由同Run轨迹可见冲击后制动已经饱和，不能仅把它归咎于缺少Kd。所有6phase连续制动与30事件一致，见[制动明细](artifacts/2026-10-13-rpo-learning-round6/replay-I1-braking.json)。

已渲染并实际检查[指标](artifacts/2026-10-13-rpo-learning-round6/integration_comparison.png)、[穿透](artifacts/2026-10-13-rpo-learning-round6/contact-depth.png)、[同Run最坏anchor时序](artifacts/2026-10-13-rpo-learning-round6/braking-replay-I1-anchor.png)，曲线只保留原episode完成子步。没有新评估/视频，不宣布模型验收。

S1新增两个最小数值行为检查先RED（helper不存在）再GREEN；实际配置仅cone变化。完整数值63 passed，平台50 passed（最初误写不存在的scheduler测试路径，未运行；已改为实际scheduler_gpus并通过）。新镜像`sha256:b3b5846081daad2ff5060d3229279665a4294af3988e1af03f0afd7c9b807657`从原action-v2派生，只拷贝action hook及cone helper，保留旧policy哈希和全部其它物理实现。旧I0/I1入口仍使用原v2镜像。S1契约SHA`e9af8be6c8b433ec268fd7b8bb2f65f3808f54d70cefc71c8b2bedc5e52cbfa7`，平台严格拒绝I0/capture及任意friction等额外参数；producer与collector都检查实际六phase cone=1。部署/健康/预览及独立review完成后才能启动。

### S1启动

`make up`退出0；首次健康请求发生在API启动中收到connection reset，等待启动后`/healthz`及ebd.ok均true；preview-command ok。独立review已完成20项数值/3项平台入口检查，无静态阻断；额外CPU微型运行分支检查继续，未另开GPU。

[Job job_6836d943939443108d12](http://1.94.0.106:31800/api/v1/jobs/job_6836d943939443108d12) / [Run run_b1bc95a7d26f4e17a766](http://1.94.0.106:31800/train/runs/run_b1bc95a7d26f4e17a766)，running，总预算第5Job/本轮第2Job。严格replay同A0包，无actor/optimizer，不修改已发布环境。保留[提交spec](artifacts/2026-10-13-rpo-learning-round6/S1-job-request.json)、[预览](artifacts/2026-10-13-rpo-learning-round6/S1-preview.json)和实际状态，完成后复核6phase、98文件、实际compiled cone及同Run全population轨迹。

### S1完成校验中；条件性冻结D1

六phase已完成，producer完整大数组校验仍在运行，未当作platform succeeded。anchor峰1.84799/2.68079/2.38574，严格均值32.4668/31.3374/32.0415、存活1136/1068/1088。p99上界1.93/1.94/1.96mm、最大6.397/5.513/6.556mm。与I1相比不能解决峰或穿透，完整独立对账前不提交下一Job。

依据[I1接触分解](artifacts/2026-10-13-rpo-learning-round6/I1-contact-components.json)：最坏anchor切向力矩15.8685/21.5550/15.8507Nm，扭转−0.73954/+0.04927/−0.07569Nm；重构总接触力误差≤8.29e−5Nm。排除“扭转是这些最坏踝峰的主要驱动力”的猜测，不据此降低扭转系数。M逆映射是瞬时广义力作用，不冒充implicitfast完整速度增量。[MuJoCo接触说明](https://mujoco.readthedocs.io/en/3.8.0/computation/index.html#contact)区分condim=4的法向/切向/扭转分量。

为避免盲扫PD，在I1同Run三个最坏anchor前史上重构`clip(40*(target-q_before)-2.5*dq_before,±27)`，至各自首次峰与实际motor误差均0。保持原轨迹代入Kd10仅用于选择候选：首次饱和反向制动由73.75→71.25、73.75→72.5、85→82.5ms，增加峰前反向力矩积分0.04212/0.03153/0.04973Nms。见[反事实核算](artifacts/2026-10-13-rpo-learning-round6/I1-braking-damping-counterfactual.json)，不预测新轨迹峰值，也不承诺穿透改善。

条件性冻结D1：**相对I1原pyramidal基线，只将四个踝PD damping从2.5改为10 Nm/(rad/s)**。所有Kp/其它关节Kd、原接触、dt/控制频率、质量/armature、接口27Nm/8rad/s、奖励/RSI保持；不叠加失败S1锥。Kd10使零位置误差时饱和制动阈值由10.8降为2.7rad/s，早于额定8；保留原力矩上限，实际稳定性仍由Job验证。此处只验证仿真控制候选，无电机硬件带宽/接触标定资料，不宣称可部署到真机。S1完整证据关闭且配置/平台回归/review通过后才使用本轮最后名额；全通过也需第七轮同配置复核。未通过时依据实际尾部选择下一单因素，总9Job封顶。

### S1独立关闭；D1提交前检查

S1 Job succeeded / Run finished，receipt SHA `e9b80de83372538ad3d641c7443a992a330ddb964138720793e47b6504c20a5f`，Run804.144198秒。完整400文件/4531901498字节，非训练成功或模型合格。独立[完整比较](artifacts/2026-10-13-rpo-learning-round6/S1/analysis.json)确认6phase仅compiled_solver.cone 0→1，实际初态/52/reference/full q-v/全部requested及共同前缀applied-target一致，30个同Run事件前史都覆盖。anchor最坏峰相对I1整体+0.5131%，各seed峰变化均<1%，锥因素未改善峰；三个最大穿透均失败，停止S1，不叠加到D1。旧policy峰3.11608/3.11387/4.17614，存活均0；不用于宣布v13采样结果。

本轮已用2Job/1501.405784秒/9048919253字节；全局5Job/2933.979304秒/16484898945字节，剩4Job，下一D1占本轮最后名额。完整Run预算见[对账](artifacts/2026-10-13-rpo-learning-round6/budget-audit-2jobs.json)，没有隐性重试。

D1已完成最小行为RED→GREEN与实际配置检查；最终数值65 passed、平台52 passed，覆盖纯API环境无Torch的receipt校验。子进程只hook actuator_parameters，四踝Kd变化；parent/collector检查23关节键、四踝40/10/27/8及cone=0；独立compiled对账进一步要求非踝完全相同，不能仅凭receipt宣布单因素成立。parent现在只选择一次完整校验器，减少重复读取大数组，不更改物理。D1镜像`sha256:9d326d40074efaa5b77cb77aaa2e32017205944297b85c38d20b0e5ef45ca4b3`，从原action-v2派生，未拷贝当前host policy；契约SHA`f71a826709b8726e08236b2d56a02f71c52cdaed239ccd370456f6296a0f70e8`。仅在独立review、部署、健康和preview-command通过后提交。

### D1启动

独立review完成2项数值/5项平台检查及打包子进程CPU hook自检，确认四踝阻尼变化、原cone、九个源码哈希和S1/D1互斥，无阻断。`make up`退出0、健康ebd.ok=true、preview-command ok后启动：[Job job_7549a8bab1474427a856](http://1.94.0.106:31800/api/v1/jobs/job_7549a8bab1474427a856) / [Run run_d5e8619d5fc9496aa7f2](http://1.94.0.106:31800/train/runs/run_d5e8619d5fc9496aa7f2)，总第6Job、本轮第3Job。没有第4个本轮Job额度；若未全通过，下一候选必须在第七轮最多3Job/90分钟/20GiB内冻结，不能沿用本轮名义加跑。

S1[指标图](artifacts/2026-10-13-rpo-learning-round6/S1/solver_comparison.png)、[穿透图](artifacts/2026-10-13-rpo-learning-round6/S1/contact-depth.png)、[同Run最坏anchor时序](artifacts/2026-10-13-rpo-learning-round6/S1/braking-replay-S1-anchor.png)已实际查看；elliptic接触type7的独立分解重构通过，见[S1 contact components](artifacts/2026-10-13-rpo-learning-round6/S1-contact-components.json)。没有缺失类型补造、没有评估或录像。

### D1独立关闭与本轮结论

D1 Job succeeded / Run finished；receipt SHA `eeb521ec9bd10363c5ef99ee01bed9832f007dab8d2953427fb192048e59e42b`。独立6phase全identity_pass：compiled差异仅actuator_parameters，且只有4踝Kd2.5→10；其余PD/solver/接触/质量/几何/初态/reference/全部requested及共同前缀applied-target完全相同。30个事件均有同Run连续前史。

|模式/seed|峰值比 I1→D1|严格均值 I1→D1|48步存活 I1→D1|p99上界mm|最大穿透mm|
|---|---|---|---|---|---|
|anchor/42|1.84833→3.17070|32.39551→33.66357|1118→1248|1.620|12.799|
|anchor/43|2.66710→2.67386|31.26562→32.45605|1058→1195|1.630|11.815|
|anchor/44|2.37765→3.25689|31.97852→33.11377|1080→1245|1.630|14.336|
|policy/42|3.01899→2.98728|7.64062→8.77881|0→1|5.380|20.716|
|policy/43|3.17685→3.41653|7.68506→8.98389|0→1|5.220|20.518|
|policy/44|3.80905→3.75502|7.81299→8.90869|1→0|5.110|22.752|

**停止D1，不进入训练。** 三个anchor峰及最大穿透均未通过，虽然严格长度/存活改善，首步终止均0、p99仍≤2mm。新最坏case转移到更晚的原episode阶段，不能把原I1案例上的反事实提前制动当候选效果或直接比较不同case的力百分比。
[完整对账](artifacts/2026-10-13-rpo-learning-round6/D1/analysis.json) · [同Run制动](artifacts/2026-10-13-rpo-learning-round6/D1/replay-D1-braking.json) · [指标图](artifacts/2026-10-13-rpo-learning-round6/D1/control_comparison.png) · [穿透图](artifacts/2026-10-13-rpo-learning-round6/D1/contact-depth.png) · [最坏anchor](artifacts/2026-10-13-rpo-learning-round6/D1/braking-replay-D1-anchor.png)。以上三类图已实际检查，无评估/视频或站立验收。

本轮3/3 Job关闭：Run合计2198.595309秒（36.643分钟），完整Run 13567693548字节（12.636GiB），均在冻结限额内。全局6/9 Job已用，剩3，进入[第七轮最后预算](2026-10-13-rpo-bfm-learning-round7.md)，不增加第六轮第4Job。

三轮复核：①配置语义/同模型及数据/RSI/动作一致，旧诊断context与真实v13训练区分；②资源/真实0更新/失败Job计数及下一阶段决策，未启动无效采样修复对照；③全部receipt/逐事件与同Run连续轨迹/实际查看图表/完整Run预算。没有改门槛、Git提交或硬件放行。

[完整证据索引](artifacts/2026-10-13-rpo-learning-round6/complete-evidence-index.json)包含3个成功Run的294个receipt成员、18个startup文件、18幅已解码实际图表及各Run工作台链接；视频/独立模型评估均明确缺省，不当作已完成验收。
