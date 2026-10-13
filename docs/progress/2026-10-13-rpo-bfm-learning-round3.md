# RPO BFM 第三轮：接触时间常数单因素验证

日期：2026-10-13。依据学习优化计划 Task 1 的动力学分支；第二轮已关闭，第三轮单独计费。用户授权保持接口限位和奖励不变，先通过物理门槛，再考虑采样修复训练。

## 1. 执行前冻结

预算：最多 **3 个 Job，每个 30 分钟，总运行时上限 90 分钟，输出合计 ≤20 GiB，无自动重试**。B0 为带新增诊断的原配置复现；C1 为一个候选；第3个名额仅用于 C1 通过后的复核，或有明确根因的收集/执行故障修正。禁止在余量里扫描其他参数。物理未通过则关闭本轮，不训练。训练对照预算另行冻结。

固定：v13 step600（Run `run_bc7df2383d8f498bb08d`）、数据 `ds_fb0ecda9e70644f2afdd`、原始动态 RSI、种子42/43/44、每卡2048环境、GPU1–3采样/4卡准入、anchor 和原冻结 policy 各48控制步。优化器更新0。控制50Hz、积分400Hz、8子步、原始几何/惯量/摩擦/锥/求解器、PD及27Nm/8rad/s踝接口限值、全部奖励权重/计算口径、终止/严格影子均固定。旧采样 bug 保留于物理对照；准备好的修复镜像暂不训练。

候选 C1：**仅地面 geom.solref[0] 从0.020秒改为0.040秒**；地面阻尼比1、机器人 solref=[0.001,2]、solimp、friction、priority/solmix 不变。B0 已保留事件的实际混合 solref=[0.0105,1.5]，C1 预期=[0.0205,1.5]，须由运行接触数组确认。它是仿真敏感性候选，尚无硬件接触标定资料；不能直接宣称更真实。

选择依据：第二轮右踝速度26.901→31.423rad/s时电机已−27Nm制动，接触广义力矩+47.829Nm；anchor 也有接触力超过电机制动。优先测试降低接触瞬时激励；不在没有电机资料时猜惯量/提高限位。v11在PhysX上60000N/m、600Ns/m同时伴随其它变化且失败，不能推导本次MuJoCo单因素必然有效。[MuJoCo参数说明](https://mujoco.readthedocs.io/en/stable/modeling.html#solver-parameters)说明正solref为时间常数/阻尼比，参数经接触对混合；本轮以冻结运行镜像与实际数组为准。

冻结判据：B0 输入哈希须与第二轮配对；物理峰值应复现至相对2%以内，否则先解释差异。C1 物理必要条件为 anchor 所有rank原始episode峰值比 **≤2.5**；同rank原始episode存活率不能下降超过2个百分点；首次控制步训练终止比例 **≤0.1%**；原始episode严格影子长度均值不能低于B0。新增接触穿透保护：原始episode所有完成子步接触深度p99≤2mm、最大≤5mm；必须同时核对支撑/存活，不能仅靠软接触压低峰值。穿透阈值为本轮仿真敏感性筛查预算的保守保护，非硬件标定结论。policy 数据用于归因，不把旧坏策略的失败等同候选不可学习；其峰值/终止/严格影子仍完整报告。

通过上述必要条件只进入第3个名额的物理复核；**不等于训练门禁全部通过**。训练的≥35严格影子步、两个窗、恢复/周期评估/媒体、100→500分段条件仍按原计划执行，不用本次48步诊断替代。未通过任何必要条件则停止本候选。

## 2. 阶段记录

- 合同见 [experiment-contract.json](artifacts/2026-10-13-rpo-learning-round3/experiment-contract.json)，SHA256 `e223af8ef674e0af2eb07f41aae5de91dd2d61e2515df8e98fcb683cde57ce36`。
- 最终镜像 `eb/rlinf:rpo-ufo-contact-factor-v1`，ID `sha256:418859f7e561d44a70108790d7ea9ae5908e880d5c7a04df06c15dfdfcf865ee`；诊断包装器 SHA `f989b6bb7f2e3b5b903286d19c36718771d029cc0384ebd0dcdc4b4c363c53da`。父镜像ID核对，7个核心物理/奖励/策略/RSI/Replay模块与冻结v13相同，见 [image-audit.json](artifacts/2026-10-13-rpo-learning-round3/image-audit.json)。
- 验证：最终相关数值测试34 passed；控制面候选/容量/准入回归45 passed。两个依赖镜像分别运行；先失败后实现的单因素、直方图矛盾摘要、receipt伪造首步统计拒绝检查已通过。未运行全仓测试。此前测试环境缺控制面依赖/误用评估adapter的问题已定位，以上为纠正后实际验证。
- 独立只读review提出并修正：积分后接触深度、提前固定GPU映射、深度直方图与最大值一致、首步/末步人口对账。当前无静态阻断；B0预览 `ok=true`、ebd健康、4卡各空闲24080MiB。运行结果待填。
- Ruling：复用原动力学 phase 和 collector 生命周期，只派生诊断镜像与受限入口；不改已发布v13或参考数据。新增统计只观测原始episode，reset后数据不计入物理筛查。

### B0：原配置复现（第1个Job）

[Job job_340772568f364380b361](http://1.94.0.106:31800/api/v1/jobs/job_340772568f364380b361) · [Run run_1f610dacc21e4ef5a9bf](http://1.94.0.106:31800/train/runs/run_1f610dacc21e4ef5a9bf)。请求/预览/初始状态已保存在本轮artifact目录；Job链接为需要平台登录凭据的API，Run链接为工作台页面。此类 `asset.validate` 诊断不会生成训练列表中的 `TrainJob`。

首个运行在原RSI reset的 `_refresh()` 调用处失败：诊断包装器漏保留默认dt，实际 `TypeError` 与新增回归检查的RED一致。它不是物理候选失败，没有完整phase/合格receipt，不作为模型或接触效果证据。错误保存于 `B0-first-error-log.json`；试图取消时已终态返回409，未改写状态。

只修正包装器默认 `dt=.0025`，与冻结基类相同，保留该失败Job并计入预算。派生镜像 `eb/rlinf:rpo-ufo-contact-factor-v2` / `sha256:2abedffa36778c6cf4d8ef4b07293d9db5ebd35210f0105e501008d8e3809e22`，包装器 SHA `f5f9b3e6d25496a1789bb8213ce8fa4b44b73638291bfa35ff8903ac18bdd119`。

修正后最终数值回归 **35 passed**；v2核心模块/hash见 [image-audit-v2.json](artifacts/2026-10-13-rpo-learning-round3/image-audit-v2.json)，独立复读确认默认参数与基类完全相同。首个Run耗时157.022467秒，Job failed / Run aborted(exit_1)。未把部分初始化输出当作诊断结果。

Ruling：使用预留的明确根因修正名额，不自动重试、不增加总Job数；剩余仅一个B0修正运行和一个C1运行。若C1必要条件全通过，因复核名额已消耗，仍须单独记录待复核，不在本轮绕过确认启动训练。

### B0修正运行（第2个Job）

[Job job_fb430591b788426e9444](http://1.94.0.106:31800/api/v1/jobs/job_fb430591b788426e9444) · [Run run_6ed60271580c4d6c98f2](http://1.94.0.106:31800/train/runs/run_6ed60271580c4d6c98f2)。Job succeeded / Run finished，六phase完整receipt被collector接受。修正版请求/预览 `ok=true` 和运行身份已保留；[correction-scope-audit.json](artifacts/2026-10-13-rpo-learning-round3/correction-scope-audit.json)证实首个镜像与修正镜像的8个核心/原动力学模块哈希完全相同。

[B0-analysis.json](artifacts/2026-10-13-rpo-learning-round3/B0-analysis.json)：六phase与第二轮 `reference` 数组及引用哈希逐行一致、在线52维状态exact equal，358维浮点最大差4.77e−7。峰值最大相对差0.087735% <2%，复现通过，决定 `continue_C1`。anchor三rank峰值1.90442 / 2.81924 / 2.65103；原始episode48步存活1115 / 1060 / 1079（共3254/6144）。严格影子均值32.0205 / 30.9731 / 31.4790，均含首次违规步和48步右删失；不能当作训练末窗或长期稳定站立表现。首步终止0。接触穿透p99上界1.57–1.58mm，最大5.47mm，基线仍存在物理缺口。

### C1：地面接触时间常数（第3个Job）

[Job job_0489fca3f06247038d40](http://1.94.0.106:31800/api/v1/jobs/job_0489fca3f06247038d40) · [Run run_e3084eb5cd86491794d4](http://1.94.0.106:31800/train/runs/run_e3084eb5cd86491794d4)。Job succeeded / Run finished，72个根文件完整receipt被collector接受，优化器更新0。预览 `ok=true`；[job-pair-audit.json](artifacts/2026-10-13-rpo-learning-round3/job-pair-audit.json)确认B0/C1同数据/模型/镜像/资源/运行策略，params只有candidate值不同。

六组compiled启动参数逐键对比仅 `geom_solref` 和对应 `compiled_xml_sha256` 不同；geom_solref仅[0,0]由.02变为.04，其余项exact equal。运行接触solref从[.0105,1.5]变为[.0205,1.5]；质量/惯量/关节/碰撞掩码/摩擦/solver/接口/PD不变。两Job实际reference数组/在线52状态exact equal，358状态最大差4.77e−7。没有把换RSI或换策略引入物理比较。

| 模式/种子 | 峰值比 B0→C1 | 严格影子均值 B0→C1 | 48步原episode存活 B0→C1 | 接触p99上界mm B0→C1 | 接触最大mm B0→C1 |
|---|---|---|---|---|---|
| anchor/42 | 1.9044→1.4594 | 32.0205→32.6514 | 1115→1121 | 1.57→5.45 | 5.378→14.483 |
| anchor/43 | 2.8192→2.0847 | 30.9731→31.4053 | 1060→1062 | 1.58→5.46 | 5.472→14.484 |
| anchor/44 | 2.6510→2.0852 | 31.4790→32.0410 | 1079→1090 | 1.58→5.54 | 5.364→15.966 |
| policy/42 | 3.4379→2.7414 | 6.9795→8.3462 | 0→0 | 4.78→13.79 | 24.473→56.798 |
| policy/43 | 3.3484→2.6235 | 6.9111→8.4224 | 0→0 | 4.75→13.43 | 24.811→42.899 |
| policy/44 | 3.9278→3.4910 | 7.2261→8.3901 | 0→2 | 4.64→13.60 | 20.694→50.903 |

首步训练终止各组0；depth仅统计原episode完成子步，包含当前终止步，reset后不计。p99为10µm分桶保守上界，最大值直接观测并与最高非空bin核验；原policy尾部超过20mm的样本没有被分桶丢弃。存活每组分母2048；严格影子含首次违规步、未违规者截断48步，不能替代未删失长episode或训练末两窗。

**结果：停止C1，不进入采样修复训练。** anchor最大速度比2.819→2.085（约下降26%）满足峰值必要条件；存活52.962%→53.271%，平均严格影子31.491→32.033步，改善幅度有限，仍不能证明原计划≥35步训练末窗条件。三个rank均穿透p99>2mm、最大>5mm，冻结穿透保护失败。原policy仍最高3.491倍、仅2/6144原episode存活至.96秒，无站立/行走/跟踪合格结论。平台Job成功代表诊断执行/收集成功。

[event-comparison.json](artifacts/2026-10-13-rpo-learning-round3/event-comparison.json)保留60个真实选中事件。B0最坏policy事件右踝26.901→31.422rad/s、motor−27Nm、contact+47.828Nm；C1最坏policy事件右踝−26.680→−27.928rad/s、motor+27Nm、contact−32.901Nm，仍在饱和制动下被接触推动。两个最坏事件可属于不同case，不能把它们当作同一状态下的力矩百分比对照；瞬时M逆力分解也不能当作implicitfast实际速度增量的精确分摊。此结果支持“接触响应显著影响速度尖峰”，没有证明单纯软化地面解决控制能力。接触距离是仿真几何量，不是硬件足底压缩实测。

权威筛查来自 [comparison.json](artifacts/2026-10-13-rpo-learning-round3/comparison.json)，决定 `stop_candidate_no_sampling_fix_training`。复用旧CPU事件绘图器的RSI启发式只保存在 `historical_heuristic_decision`，不用于准入；第二轮已解释原参考有限差分与瞬时雅可比速度的口径差异，没有新的RSI数据失配证据。

## 3. 证据、预算和三轮复核

[指标对照图](artifacts/2026-10-13-rpo-learning-round3/comparison.png) · [完成子步时序](artifacts/2026-10-13-rpo-learning-round3/timeseries.png) · [接触穿透图](artifacts/2026-10-13-rpo-learning-round3/contact-depth.png) · [完整证据索引](artifacts/2026-10-13-rpo-learning-round3/complete-evidence-index.json)。图表由保留数组生成，已检查图像；事件NPZ/完整M/J/实际积分力/原始RSI保留在两个成功Run，不改写receipt。诊断没有独立评估或评估视频，索引如实注明。

[run-budget-audit.json](artifacts/2026-10-13-rpo-learning-round3/run-budget-audit.json)：3/3 Job，首个失败157.022467秒、B0修正348.975101秒、C1 345.668166秒；合计851.665734秒（14.194分钟）<90分钟，单Job<30分钟。全部Run输出234295861 bytes（约223.44MiB）<20GiB，无自动重试、新learner更新0；本轮预算已关闭，不开第4个Job。

1. 设计/语义复核：接口、奖励、动态RSI、几何/惯量与旧policy均冻结；编译模型逐键和实际初态逐行比较确认只改地面time constant。RSI的原运动量保留。
2. 运行/失败复核：三Job均走API/Job/Run/ebd；GPU0无learner、GPU1–3实际采样；默认dt缺陷留失败记录，按相同默认值最小修复并RED→GREEN。没有取消改成功、门禁豁免、超预算重试、降限位或换奖励。
3. 完整证据复核：两成功Run各六phase/72文件与collector校验闭合；人口/逐子步深度/事件摘要重新计算、图表检查、指标与穿透同时判定。独立review及35数值+45平台测试通过；没有把Job succeeded、48步存活或仿真敏感性当模型验收。

收尾检查：所有本轮JSON/索引文件哈希和报告本地链接有效，`git diff --check`通过，ebd健康，GPU0–3各已释放至1MiB使用/24080MiB空闲。CPU复算和绘图在容器中完成，没有额外GPU积分、policy推理或Job。

## 4. 下一阶段入口

C1不注册训练recipe、不替换v13、不与准备好的采样修复捆绑。保留原接触配置B0作为后续基线；原基线自身仍有速度/穿透尾部，不应称已校准通过。下一轮应先冻结新有限预算，优先做**积分时间步敏感性**：仅将physics dt从2.5ms减至1.25ms，同时16子步保持20ms控制周期、原接触参数/PD/接口/奖励时间平均口径不变。须先核对8子步硬编码的interval、终止锁存和trace，使诊断的新16子步不会意外改变奖励尺度；不能直接把控制频率改成100Hz作为同因素比较。

该试验用于分清数值积分敏感性与真实控制/接触耦合；尚无结果，不预判会通过。若对dt不敏感，再针对保存的失败reference case记录着地前后q/dq/target/实际tau/contact impulse的连续时序，核实什么时候开始饱和制动，再选择一个制动提前量/控制阻尼候选；峰值时已±27Nm本身不足以支持加大Kd，不重复盲目扫PD/猜电机惯量。任何接触标定候选仍需限制穿透并保持支撑，而不只压低速度峰值。

物理全部条件通过且周期checkpoint/恢复/独立快评与媒体链路满足原计划后，另行冻结训练预算，才进入同初始化的采样修复单因素100→500对照。C1本轮未通过，训练阶段没有启动。

## 5. 尚未完成

物理候选是否有效、硬件接触标定、采样修复训练对照、独立评估和录像、正式站立/行走/跟踪验收均未完成。未使用git提交。
