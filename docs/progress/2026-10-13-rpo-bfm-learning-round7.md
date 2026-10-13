# RPO BFM 第七轮：最后一轮物理门槛预算

日期：2026-10-13（Asia/Shanghai，若运行跨日实际UTC/本地时间以Run时间戳为准）。用户批准的最多3轮9Job预算第3轮；前轮见[第六轮](2026-10-13-rpo-bfm-learning-round6.md)。不使用Git提交。

## 执行前冻结

本轮≤3Job、单Job≤30分钟、合计Run时间≤90分钟、完整Run≤20GiB，retry=0、optimizer=0。已用6Job/3631.168829秒/21003673240字节（第5轮+第6轮完整Run，含失败、缓存、日志），剩3Job；本轮结束或9Job用尽仍未通过就如实关闭，不偷偷追加。全局≤4.5小时/60GiB。[总预算](artifacts/2026-10-13-rpo-learning-round7/experiment-contract.json)提交前冻结。

## 候选与理由

本报告S1/S2是接触solver诊断代号，不是优化计划第13节的训练采样配比S1/S2；operation和Job均为asset.validate，optimizer=0。

积分I1的必要物理门槛未全通过；S1改锥未改善峰/最大穿透；D1四踝Kd10虽改善存活与严格长度，却产生更大的尾部峰及12–14mm穿透，不能把原轨迹上“提前制动”核算当实际效果。D1独立完整对账关闭后才进入本轮。

选S2：**相对已保留的S1 elliptic基线，仅impratio从1改为0.1**，恢复并保持原四踝40/2.5/27/8，不叠加D1。最坏I1踝峰的接触力矩主要由法向及切向产生，扭转很小；加大电机阻尼未消除力矩饱和下的接触耦合。需要区分摩擦维度柔顺与法向支撑，避免再次用软法向接触降低峰同时扩大穿透。

[MuJoCo 3.8参数说明](https://mujoco.readthedocs.io/en/3.8.0/XMLreference.html#option-impratio)说明impratio控制elliptic锥的摩擦/法向约束阻抗比，不改变实际摩擦系数；pyramidal会混合这些维度，因此使用已有S1作为相邻单因素基线。S1自身未合格，复用它是为隔离此参数，未发布为训练候选。MJLab1.4.0实际MujocoCfg含impratio并施加于compiled model；新的动作metadata记录实际cone/impratio/tolerance/linesearch设置，不能只报告配置字符串。S2效果尚未验证，也不等同硬件摩擦/接触校准。

保持16×1.25ms/50Hz、模型/同A0动作包、动态RSI、全部PD、接口限位、奖励、摩擦系数、geom.solref/solimp/condim/priority/solmix、Newton100及其它solver选项、质量/惯量/几何。6phase、3rank×2048、seed42–44、48controls，无actor和learner更新。与S1相邻比较只允许impratio变化，原R0也作为存活/严格长度基线完整报告。

物理门槛不变：每anchor seed峰≤2.5、存活≥原R0−2pp且≥相邻S1−2pp、严格均值≥原R0且≥S1、首步终止≤0.1%、原episode完成子步接触深度p99上界≤2mm且最大≤5mm。policy全部报告；同Run全population覆盖所有新尾部，不以全局平均或部分seed替代。

首名额S2若全部过，则第二名额同配置确认；否则依据新证据冻结一个允许的单因素，第三名额仍须独立证据与阶段决策。无确认不得称物理通过。训练的100更新/≥35末两窗、周期恢复、独立评估/媒体和有效采样对照另行冻结，不含本轮预算。

## 阶段记录

以下按冻结、运行、独立核验顺序记录；Job成功与物理门槛合格分别判定。

### S2执行前检查与启动

数值新行为先RED（helper不存在），实现后最终67 passed；平台53 passed。检查真实CPU MjModel.apply与ControlRecorder.begin输出实际impratio/solver options，拒绝额外参数、D1叠加或错误stage；旧I0/I1/S1/D1镜像入口常量保持。独立review4项数值/4项无Torch平台检查通过，未发现静态阻断，额外打包子进程和旧镜像CPU核查继续。

[S1 baseline审计](artifacts/2026-10-13-rpo-learning-round7/S1-baseline-impratio-audit.json)在原S1不可变镜像内调用实际native config+MjModel.apply，证明cone1/impratio1/tol1e−8/LS50/.01；**旧S1 Run没有实测记录impratio，该项是冻结源码/CPU证明**。S2收据要求六phase新实际值cone1/impratio.1及其它选项保持，避免用配置声明替代本轮实测。

镜像`sha256:e8fae50c4c2d5cd7b35d36cb826b2c009344144e25f75bcc245892e204fc13fa`从原S1镜像派生，仅action观测/分支及impratio helper，不复制host新policy或D1物理模块。本轮预算SHA`78d73adcbee2d5f44dccb7827585843de571589eb97a3b8919dda306e8fea101`，S2契约SHA`00be3befb8383ca16bd977797d7aa1d372857480c1823cd842f8f9318665b235`。`make up`退出0、health/ebd.ok=true、preview-command ok后提交[Job job_2871dea56d2e4c0da1e6](http://1.94.0.106:31800/api/v1/jobs/job_2871dea56d2e4c0da1e6)，总第7Job、本轮第1Job，retry=0、optimizer=0。

S2 [Run run_4e04e4eb4b6f49a3a014](http://1.94.0.106:31800/train/runs/run_4e04e4eb4b6f49a3a014)已启动。额外独立打包子进程自检及旧S1 CPU apply核查完成：原PD2.5、先cone后impratio、十个源码哈希和API无Torch均确认，无阻断。实际物理仍待完整Run/独立对账；不从CPU配置通过推导门槛通过。

### S2完整收据与条件性冻结C2

S2六phase完成，receipt SHA`1d43355528bb8bc4c180cc95fd41fd75520bcc41c9976402318547c05e972c97`。独立完整CPU对账运行中。anchor峰1.77811/2.53647/2.29273、存活1114/1050/1080，严格均值32.46484/31.35205/32.04346，首步终止0。p99上界1.93/1.95/1.96mm，最大5.663/8.480/5.895mm。seed43峰仍超2.5，三个最大穿透失败；seed42严格均值还比相邻S1少0.00195步，不放宽冻结条件。

条件性冻结最后优化候选C2：**相对S2只改ground.geom.solref[1]从1到0.1**，timeconstant仍.02，不改robot solref、solimp或其它任何接触值。实际混合solref从[.0105,1.5]到[.0105,1.05]。[MuJoCo 3.8公式](https://mujoco.readthedocs.io/en/3.8.0/modeling.html#reference)给出b不含dampratio、k与dampratio平方成反比；同位置d(r)时法向参考加速度刚度增至2.040816倍，阻尼b不变。elliptic摩擦维度的k恒0、dampratio被忽略，故此参数隔离法向刚度，不再同时软化法向/切向。混合后的1.05仍高于临界1，tc .0105也高于2dt .0025。静态压入缩放只用于候选理由，不能把瞬态最大8.480mm直接乘.49当实际结果，也不把参考加速度参数当硬件N/m标定。

C2只有S2完整独立证据关闭、实际geom/有效solref/默认旧入口边界检查及review通过后才提交本轮第二名额（总第8Job）。仍用同A0动作包、原PD、S2 cone/impratio、原限位/奖励/RSI及其它solver。C2全部必要门槛通过才用最后第9Job同配置确认；若失败则关闭本候选，不为凑通过再盲扫、改门槛或启动训练。

### S2独立关闭与C2提交前检查

S2 Job succeeded / Run finished，独立6phase及98文件全部核验。startup compiled字段完全同S1；新metadata实测impratio.1，其它solver options保持，旧值1的来源为冻结S1镜像+CPU apply，未冒称旧Run实测。初态52/reference/fullq-v/全部requested及共同前缀applied-target一致，30个同Run事件前史一致。

|模式/seed|峰 S1→S2|严格均值 S1→S2|存活 S1→S2|p99上界mm|最大mm|
|---|---|---|---|---|---|
|anchor/42|1.84799→1.77811|32.46680→32.46484|1136→1114|1.930|5.663|
|anchor/43|2.68079→2.53647|31.33740→31.35205|1068→1050|1.950|8.480|
|anchor/44|2.38574→2.29273|32.04150→32.04346|1088→1080|1.960|5.895|
|policy/42|3.11608→3.13420|7.77539→7.83545|0→0|5.680|23.772|
|policy/43|3.11387→3.63604|7.80273→7.88818|0→1|5.510|23.597|
|policy/44|4.17614→3.76974|7.85791→7.95508|0→1|5.510|27.119|

决定停止S2，不启动训练。速度整体改善5.383%，但seed43峰/三seed最大穿透和seed42相邻严格均值仍失败。相对原R0的严格均值/存活也报告，不以S1本身未合格回避必要条件。
[独立结果](artifacts/2026-10-13-rpo-learning-round7/analysis.json) · [同Run制动](artifacts/2026-10-13-rpo-learning-round7/replay-S2-braking.json) · [指标](artifacts/2026-10-13-rpo-learning-round7/friction_comparison.png) · [穿透](artifacts/2026-10-13-rpo-learning-round7/contact-depth.png) · [最坏anchor](artifacts/2026-10-13-rpo-learning-round7/braking-replay-S2-anchor.png)。三类图已实际查看。

本轮已用1Job/700.628002秒/4535061735字节，含400个完整Run文件。全局7Job/4331.796831秒/25538734975字节；剩2名额，C2用一个，最后一个只用于全过后的同配置确认。

C2新行为先RED再GREEN，最终70数值/54平台通过，含真实TerrainEntity/MjSpec编译、C2实际单geom参数与旧S2默认拒绝C2header检查。默认原ground ratio1的旧入口保持；C2显式逐层传.1并核验实际混合solref.0105/1.05，不放宽所有入口或更改统计门槛。新镜像`sha256:ffe631ac28292746b1e883f6e9b7d9e66a16e874cd623df47665fcf49d04432d`，来自原S2，只改诊断分支/校验和ground hook；core rpo_ufo_mjlab/PD/奖励/RSI/旧policy保持。C2契约SHA`ecc8292dcdf28027f7617cf37dddc3e42b8572869de0fabbbf5f656ea9dcfe50`。部署与预览健康通过，独立review完成后启动总第8Job。

### C2启动与独立复核

2026-10-14 00:12:47 Asia/Shanghai（2026-10-13 16:12:47 UTC）提交总第8Job、本轮第2Job：[Job job_9d9d692030be4dcd8462](http://1.94.0.106:31800/api/v1/jobs/job_9d9d692030be4dcd8462) · [Run run_661fbd961a1d4a10a491](http://1.94.0.106:31800/train/runs/run_661fbd961a1d4a10a491)。该任务是asset.validate，训练列表不会新建模型训练行，可从Run链接查看；retry=0、optimizer=0。

独立review最终17项数值/5项平台检查通过，无静态阻断：打包子进程实际顺序为install_trace→B0原接触guard→C2，CPU编译确认只改变地面solref[1]；十二个源码哈希一致，API链无Torch/MuJoCo/Warp导入，默认S2仍拒绝C2。实际物理门槛待本Run完整收据和独立核验。第9名额仅在C2全部必要物理门槛通过后用于同配置确认。

### C2完整收据；独立对账中

平台Job succeeded，receipt SHA`24192b54f7615f758279795ab855bf60fbae34d874de4e8eb2df9089227e9d62`。六phase全部产生，实际effective solref min/max均为[.0105,1.05]（float32误差）；独立完整CPU分析运行中，额外加入原R0收据/统计SHA核验及原R0存活/严格均值条件，不只与相邻S2比较。收集大数组期间一次API状态查询30秒超时，保留事实；后续正常取得succeeded，没有重提Job或更改平台状态。

anchor峰1.83731/2.59798/2.35974、存活1102/1044/1074，严格均值32.22656/31.31787/31.91748，首步终止均0；p99上界1.47/1.50/1.50mm，最大4.878/7.786/5.406mm。seed43峰及seed43/44最大穿透未过，三个严格均值都低于相邻S2。地面刚度候选改善深度但未达到全部物理条件；最终identity/同Run时序、图表及预算关闭后结束该候选，不启用第9确认名额。

### C2独立关闭与本轮结论

独立完整CPU分析退出0，6phase全部identity_pass，98个收据成员和30事件同Run连续前史核验通过。startup差异仅geom_solref及其XML SHA；实际数组只有[ground,1]从1→.1，六phase的PD、solver options及其它编译字段保持。52/reference/full qpos-qvel初态、全部requested、共同原episode前缀applied/target完全一致。控制后state/history/qvel有差异，报告保留，不称逐轨迹exact replay。

|模式/seed|峰 S2→C2|严格均值 S2→C2|存活 S2→C2|p99上界mm|最大mm|
|---|---|---|---|---|---|
|anchor/42|1.77811→1.83731|32.46484→32.22656|1114→1102|1.470|4.878|
|anchor/43|2.53647→2.59798|31.35205→31.31787|1050→1044|1.500|7.786|
|anchor/44|2.29273→2.35974|32.04346→31.91748|1080→1074|1.500|5.406|
|policy/42|3.13420→3.23284|7.83545→7.57959|0→0|4.700|26.248|
|policy/43|3.63604→3.28559|7.88818→7.60205|1→1|4.670|23.874|
|policy/44|3.76974→3.95948|7.95508→7.72607|1→1|4.610|24.906|

逐anchor决定：seed42只有相邻严格均值失败；seed43峰/相邻严格均值/最大穿透失败；seed44相邻严格均值/最大穿透失败。各seed相对原R0的存活与严格均值、相邻存活−2pp、首步终止和p99均通过，不能以这些通过项代替缺失项。整体anchor峰相对S2增加2.42477%；此处是地面参数比较，不把脚本通用sensitivity标签当再次积分验证。旧policy是legacy诊断的冻结动作，未冒称真实v13环境tracking训练结果。

**决定：关闭C2与本次获批三轮物理优化，物理门槛未满足；不启动第9确认Job，不启动采样修复训练。** 本轮2/3名额实际使用，最后一个是提交前冻结的“全过后确认”额度，保留未使用，不称9Job已耗尽，也不挪作新候选。阶段状态为`stop_training_and_resolve_dynamics`，没有选出可进入训练的物理配置。

[独立比较](artifacts/2026-10-13-rpo-learning-round7/C2/analysis.json) · [同Run制动](artifacts/2026-10-13-rpo-learning-round7/C2/replay-C2-braking.json) · [指标](artifacts/2026-10-13-rpo-learning-round7/C2/ground_comparison.png) · [穿透](artifacts/2026-10-13-rpo-learning-round7/C2/contact-depth.png) · [最坏anchor时序](artifacts/2026-10-13-rpo-learning-round7/C2/braking-replay-C2-anchor.png)。以上三类C2图已实际查看，S2同三类亦已查看；共12幅图全部解码验证。

### 补齐制动与穿透两种失败的定位

C2最坏速度仍是seed43/env1808、右踝pitch、初始reference编码[0,31,192]：足加载43.75ms→首次超额定72.5ms→首次饱和反向制动73.75ms→峰80ms。峰前后dq20.19978→20.78381rad/s，实际motor−27Nm、直接该踝限位项0、contact+32.33133Nm。由完整约束行独立分解：法向9.98564、切向22.30159、扭转0.04410Nm；三个速度峰分解最大重构误差4.674e−5Nm。见[接触分解](artifacts/2026-10-13-rpo-learning-round7/C2-contact-components.json)及[实际事件事实](artifacts/2026-10-13-rpo-learning-round7/C2-event-force-facts.json)。接触/惯量耦合下制动已饱和仍继续加速的事实保留；不把M逆瞬时贡献当完整implicitfast积分，也不宣称已唯一证明硬件能力不足。

复核发现只分析速度峰会遗漏最大穿透的对象，因此追加**无新Job**的全population原episode定位；最大深度与独立收据统计逐seed差≤1e−9m。三者与速度峰均非同env：

|seed|最大穿透env / reference编码|时刻|最大mm|该时刻最高关节速度比|双足绝对Fz和N|
|---|---|---|---|---|---|
|42|1783 / [0,23,360]|97.5ms|4.878|0.23148|369.02|
|43|453 / [0,27,218]|480ms|7.786|0.79731|567.18|
|44|108 / [0,23,658]|85ms|5.406|0.45582|1141.38|

[穿透case与连续轨迹索引](artifacts/2026-10-13-rpo-learning-round7/C2/penetration-cases.json)记录初始/当时reference、速度/力矩/足载荷及3份小型NPZ。力为step后geometry refresh前，深度为完成积分后刷新，未混用时刻。**这些新定位的最大深度时刻没有M/J/geom ID快照**；不能用速度峰约3mm的接触几何推导480ms的7.786mm原因，也不能补造那些快照。现有群体qpos/qvel可供后续原生几何复算；新的GPU精确事件捕获须另行冻结预算。

### 预算与三轮复核

本轮S2运行700.628002秒，C2运行698.334997秒（2026-10-14 00:12:50.296124→00:24:28.631121 Asia/Shanghai）；合计1398.962999秒/23.316分钟、9065871186字节/8.443GiB。完整Run含失败、日志、配置、缓存，不只统计receipt成员。获批三轮实际共8Job/5030.131828秒/83.836分钟、30069544426字节/28.004GiB；最初failed Job的Run实际为aborted，保留并计入。Run时间合计与开发/构建/测试/分析时间分别列示；首Run开始至末Run结束含间隔162.731分钟。[完整预算](artifacts/2026-10-13-rpo-learning-round7/complete-budget-audit.json)全部在冻结限额内，没有隐藏retry。

第一轮复核（语义/隔离）：同模型、同数据版本与参考动作/RSI动量、原限位/奖励；确认C2实际单geom参数、其余PD/solver unchanged；区分legacy诊断和真实v13 context。358列原参考FD与在线瞬时速度差异沿用[第二轮结论](2026-10-13-rpo-bfm-learning-round2.md#rsi一致性结论)，本轮收据中的mismatch字段仍是原参考/在线比较，不能据此重报reset丢速度，也不称参考近似对学习无影响。

第二轮复核（执行/预算/决策）：全部GPU走API→Job→Run→ebd，GPU0无learner、三采样rank，每Job真实optimizer=0；失败计入8/9，总3轮关闭，第9确认前提不满足。不启动会对真实v13采样无作用的名义mode/cachefix对照，不修改recipe/恢复指纹。最终相关数值70 passed/平台54 passed，独立review17数值/5平台及真实CPU编译检查通过；实际GPU必要条件失败与这些检查通过分别报告。

第三轮复核（证据/出口）：完整收据/初态/动作/同Run30事件、速度和新定位穿透case均可追溯。完整索引含2Run的196个receipt成员、12个startup文件、12幅解码图及Job/Run链接。没有独立模型评估、训练checkpoint或评估录像；不宣布站立/行走验收。[Doctor](artifacts/2026-10-13-rpo-learning-round7/doctor-final.json)确认四卡run_id=null、各空闲24080MB；ebd.ok=true。Vulkan检查仍false，本轮未验证录像链路，不将doctor总体healthy推导成图形链路合格。

[完整证据索引](artifacts/2026-10-13-rpo-learning-round7/complete-evidence-index.json) · [Job最终状态](artifacts/2026-10-13-rpo-learning-round7/C2-status-final.json) · [原始Run时间](artifacts/2026-10-13-rpo-learning-round7/all-run-timing.json)。没有Git提交；未跑全仓测试、真实采样修复训练或模型独评。

最终独立只读交付review再次核验原R0收据/六份统计哈希、C2六phase门槛、完整Run时间戳/文件字节及索引：与上述失败、8Job预算和停止结论一致，无新增阻断。其余未知包括新定位穿透时刻的M/J/geom快照，继续保留，不用已通过的CPU/平台检查代替实际物理结果。

测试口径补充：C2首次RED实际是模块尚不存在的import失败，[原日志](artifacts/2026-10-13-rpo-learning-round7/C2-initial-import-red.log)保留；它不满足计划第14节要求的有效行为断言，不能仅凭“先RED再GREEN”宣称该要求完成。最后增加无新Job的CPU行为负例：仅在测试进程禁用C2 hook，实际原生地面保持ratio1，已有真实MjSpec断言在第15行按预期失败；恢复实际hook，同一检查1 passed。见[行为负例](artifacts/2026-10-13-rpo-learning-round7/C2-behavior-negative-check.log)和[正例](artifacts/2026-10-13-rpo-learning-round7/C2-behavior-positive-check.log)。没有修改workspace/镜像或伪造首次执行顺序。4份文档/103个本地链接及6phase失败结论/3份穿透trace索引检查通过；首次链接检查漏挂绝对路径，补只读挂载后通过，不把检查环境问题改成文档错误。

### 后续入口（本轮不自动启动）

1. 分开核验80ms的速度/制动case与85–480ms的穿透case；复用保存的qpos/qvel做原生MJCF接触几何核验，检查具体geom/接触点/足底形状、M/J与实际求解残差。当前没有证据把全部尾部归为单一solver不收敛，不能先任意加迭代数。
2. 依据上述结果选一个控制/接触/solver候选并冻结新的有限预算；原27Nm/8rad/s、奖励、RSI与数据身份继续保持。较小dt、锥、Kd10、摩擦阻抗、法向刚度这些已测因素的收益/代价全部保留，不从局部穿透改善推导全门槛合格。若需要改变资产/接口/参考数据，须单独立项派生版本，不能混入当前对照。
3. 每seed全部物理条件及同配置确认通过后，再冻结真正有采样差异的训练单因素；先核验v13实际context/采样曝光与update比例，mode/cachefix在该实际配置中未触发行为差异不能作为名义对照。周期checkpoint/恢复、独立快评/媒体链路完整后执行100→500实际更新，并按K=100独立区间决策。当前这些训练前置项仍未完成。
