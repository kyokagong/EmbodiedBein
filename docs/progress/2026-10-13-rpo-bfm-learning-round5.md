# RPO BFM 第五轮：冻结动作回放与积分物理验证

日期：2026-10-13（Asia/Shanghai）。继续第四轮第8节已批准的方案；用户补充授权最多3轮、每轮≤3个Job/90分钟，合计≤9个Job/4.5小时/60GiB，允许根据证据单因素调整PD、接触或求解参数。接口限位、奖励、RSI不变，不使用Git提交。

## 执行前冻结

本轮3个名额：A0原I0捕获 → R0相同动作包I0回放 → R1相同动作包I1回放。每Job≤30分钟、retry=0、optimizer_updates=0、每轮所有Run输出≤20GiB。失败作业计入额度；每阶段分析后才提交下一阶段；未用额度不消耗。总预算含后续两轮，后续候选必须有本阶段证据并在执行前独立冻结。预算耗尽而未通过则如实停止，不放宽物理门槛。

模型step600、数据/接口/codec/资产、种子42/43/44、GPU1–3各2048环境、anchor/旧policy各48控制步沿用第四轮。原policy采样行为保留；回放只替换动作来源，不修复训练采样。A0保存全部环境每控制步requested/applied/target、state/history、latent/context/reset及RNG摘要；保存所有环境的子步q/dq、motor/contact/limit/足力/接触深度，保证同Run最坏case前史可查。既有事件仍保留质量矩阵和约束力完整分解。数据仅用于诊断，不登记为专家数据或模型验收产物。

I0=8×2.5ms，I1=16×1.25ms，均50Hz。R0先核对原始reference/52及完整q/dq初态、动作、目标转换、同原episode前缀、逐rank/mode峰值≤2%偏差；不要求reset后的不同episode路径相同。R1仅在R0通过后提交，比较全部子步及共同2.5ms网格。

物理门槛保留：anchor每seed峰值比≤2.5，48步存活≥I0−2pp、严格影子均值≥I0、首控制步训练终止≤0.1%；完成子步原episode接触深度p99上界≤2mm、最大≤5mm。policy同时报告，不以动作回放通过宣称闭环训练质量通过。原计划训练末两窗≥35步、真实100更新/完整恢复/周期评估/媒体证据条件仍有效；满足诊断物理条件后先补训练边界链路，不提前运行500更新。

## 实现范围与裁定

工作区起点保存在`/tmp/rpo-round5-start-status.txt`，保留已有差异。复用动力学phase、积分trace、ToolsAdapter、collector和容量分支；只加有界capture/replay入口与证据校验、派生诊断镜像，不改发布版本、不改外部只读仓。

Ruling：既有计划、第四轮下一步和本轮用户明确回复已授权此执行设计及预算，不重复要求方案批准；技能的提交建议不适用用户“不要git提交”。诊断走API→Job→Run→ebd，CPU测试/分析也容器化。实现完成后独立只读复核，最终按语义/预算与失败/完整证据三轮核对。

## 阶段记录

实现检查：新实际打包镜像数值60 passed、平台49 passed；新增3个检查先因缺模块失败，随后通过。独立只读复核平台/动作7 passed、数值21 passed，内置8个源码hash与镜像实际文件匹配。host的采样修复policy与镜像旧policy有意不同，不称源码全部与workspace一致。校验包括动作包shape/finite/alive锁存、完整动量初态、同Run全population与选择trace的一致性、冻结capture/replay准入与容量。

预算SHA `ceb859e5a2cad176fc806af2a257063b12d8d262906f4f0684771ade5a0badba`，镜像`sha256:14d68ee04ec07c4c0535e7bbbbfa27a769e26c06aab8ff1962d761f05dcef796`。`make up`构建完成但遭遇Compose迁移临时容器名称冲突；只读检查发现并发部署已完成，迁移exit0、API/ebd均healthy，实际API常量与本轮镜像一致，未删除容器或绕过ebd。A0健康含ebd.ok=true、平台预览ok、4卡空闲，请求/预览/镜像来源均保留于本轮artifacts。

Ruling：receipt验证本Run证据，R0跨Run复现由独立CPU分析负责，必须reference/52/full q-v初态一致、全部requested相同、original_alive共同前缀applied/target差≤1e-6、全部6phase峰差≤2%；并报告state/history/共同网格qvel波形差。receipt成功本身不放行R1，不称exact replay（未恢复全部仿真内部warmstart状态）。

### A0捕获

[Job job_4e7f400c765e4fc99365](http://1.94.0.106:31800/api/v1/jobs/job_4e7f400c765e4fc99365) / [Run run_1926cb13224f4e699605](http://1.94.0.106:31800/train/runs/run_1926cb13224f4e699605)，已准入running。独立review最终判定无A0阻断，21数值+7平台以及阶段分析小型正反例检查通过。作业属于asset.validate，不新增训练模型任务；通过Run链接查看日志。

### A0记录错误与修正预算

首Job最终failed/Run aborted(exit_1)：3个anchor完整，policy第一控制步记录时`UFOBFMPolicy has no attribute rollout_context`。新增记录错误，不能认定模型或物理异常；原始失败日志保留，部分输出不作完整诊断receipt。失败计入本轮3/总9Job预算，不自动重试。

核验实际冻结源码和resolved-config发现：旧`phase`仅传入AGENT_CONFIG_KEYS，过滤了`context_profile`和`ankle_control_profile`，因此**历史policy诊断实际为legacy_global、mode=train随机动作**，并非v13训练的environment_tracking_v10。上一轮“诊断中mode被tracking覆盖、故不能归因随机动作”的推论不适用于实际诊断入口，在此纠正；v13训练配置是否被桥层覆盖须单独核对，不能将旧诊断policy代表训练语义。legacy使用torch.randint选择tracking ID（可重复），任一reset可重采全局tracking，latent写入重复索引具有待核验的数值风险。该代码机制有证据，但不能据此断言已唯一解释全部物理分叉。

只修记录：保留实际z、legacy tracking ID原顺序/重复数、真实episode步与RNG；legacy丢弃专家source ID的地方明确记unknown=-1，禁止用RSI source替代。补legacy负例先失败再检查，通过新镜像后才再提交。

执行前调整本轮剩余名额：第2Job修正A0捕获，第3Job R0回放；若R0通过，第6轮第1Job执行原定R1，不追加总预算、不改变物理配置。第6轮仍最多3Job/90分钟，候选只依据R1证据决定。

修正v2镜像`sha256:67085ed631fa00c912c0f36841d21a5d143c05438834f3bbf10928ca507e598a`，数值61+平台49 passed，第二次独立只读review实际after路径自检及4项回归通过，明确此前review未覆盖真实legacy分支。修正契约SHA `59fdaab9b991a3c754606b8bd90926e3e793ee895ad4ec59c28c5c15db4da49c`，make up exit0、健康和预览ok后启动：[Job job_5f8c1c8389fb481a8601](http://1.94.0.106:31800/api/v1/jobs/job_5f8c1c8389fb481a8601) / [Run run_4f8ea733687e4f389082](http://1.94.0.106:31800/train/runs/run_4f8ea733687e4f389082)。第1失败Run实际344.683827秒，未重试/删除。

### 训练采样假设追溯（不混入物理因素）

[runtime-context-audit.json](artifacts/2026-10-13-rpo-learning-round5/runtime-context-audit.json)核对step600 resolved-config、冻结worker和模型工厂：训练context为environment_tracking_v10，ankle_control_profile未设置；worker只加precision/path/seed/rank后直接UFOBFMPolicy(cfg)，无该字段注入。因此旧mode覆盖的kd25条件在当前v13配置不激活。准备的sampling-fix仍修缓存区分train/eval的正确性，但不能再把“v13训练动作被覆盖成mean”或该候选带来采样质量改善当作已证实事实。正常训练循环是否混用同观测train/eval缓存尚未运行核验。

Ruling：物理通过后的训练前置检查须先证明真实行为对照有差异；不能消耗100→500预算运行仅名义上修复、实际不改变当前训练行为的对照。采样跨度/更新比例候选须按原设计及实测另行冻结，当前不改训练recipe或采样语义。

### 修正A0结果 / R0执行前决定

A0 succeeded/Run finished，六phase完整receipt SHA `d4638074a861a3967a8e45f3d6561d98ce7dd86f6dbf448e59a9dd154f1683ea`；平台及独立CPU验证通过。全部30个事件的qvel与同Run全population在同原episode/同子步exact equal，事故前史已在发生Run保存，不用另一次轨迹替代。

anchor峰1.90436/2.81923/2.65326、存活1115/1060/1079、严格均值32.01855/30.97607/31.47852、首步终止0；穿透p99上界1.57/1.58/1.59mm，最大5.921/5.639/5.469mm，基线仍未过物理门槛。policy峰3.46067/3.35327/3.92778，存活均0，严格均值7.000/7.03076/7.26660，仅作旧诊断动作激励下的动力学归因。

实际legacy首步1024个tracking赋值只覆盖816/810/783个不同env，重复208/214/241次；47个控制步边界有46次tracking ID集合变化。证据支持该诊断context全局重采频繁、存在重复写入机制，仍不能唯一归因物理分叉；不当作v13训练context事实。原生逐环境tracking应单独做恢复/采样核验。

决定continue_R0：完整动作包和初态可验证，平台预览ok；R0只读消费A0全部requested，完全不调用actor。当前输出约2.90GiB，原失败输出保留，未扩预算。

R0已启动：[Job job_7e28d11ab0d242d292c4](http://1.94.0.106:31800/api/v1/jobs/job_7e28d11ab0d242d292c4) / [Run run_1005015a4cfa418bb8c0](http://1.94.0.106:31800/train/runs/run_1005015a4cfa418bb8c0)。本轮3个Job名额已全部使用，任何下一GPU Job须进入第6轮预算；R0结果决定I1是否可执行。

![同Run anchor最坏事件制动前史](artifacts/2026-10-13-rpo-learning-round5/braking-capture-I0-anchor.png)

图已实际打开核对。anchor43/env1808：72.5ms首次越额定，75ms首次反向饱和，80ms峰22.5539rad/s，motor−27Nm/contact+30.4129Nm；足载荷早于踝冲击，不能将首次>20N载荷直接等同踝速度峰产生接触。记录force在solver后/forward前、depth在forward后，M⁻¹Fdt仍不当作精确implicit积分速度增量。

## 本轮闭合与下一阶段入口

R0 succeeded/Run finished，receipt SHA `028388bf3725b022cd085f00854261fe8e0b30b7adb480dbd92343b04dca6135`，平台和独立CPU验证通过。全部6phase：完整q/v、reference/52初态exact equal；48×2048的requested相同，原episode共同前缀applied/target最大差0，compiled字段无变化，actor调用0。峰值偏差−0.21344%/+0.00226%/−0.000081%（anchor）、−0.22990%/−0.14686%/+0.000091%（policy），冻结2%入口通过。

**统计入口通过不等于逐轨迹相同。** 共同原episode前缀qvel最大差anchor4.95/5.23/4.80rad/s、policy18.55/18.40/16.88rad/s；每seed分别363/331/340和660/687/739个env出现>1rad/s差异。存活差仅0/0/+1（anchor）、0/0/+1（policy），严格均值差0/−0.02197/+0.02051和+0.01758/−0.01270/+0.00146。anchor43穿透最大值5.639→5.070mm，尾部也有变动；不能称exact replay或把某个不同事故快照当相同前史。动作/z路径已隔离，剩余物理数值/接触排序/批量reset敏感性仍须定位，不唯一归因为GPU非确定性。

Ruling：本轮执行前入口明确冻结为初态/动作/目标/compiled身份+峰差≤2%，波形差明确报告，未临时放宽门槛。该入口通过，允许第六轮I1统计敏感性诊断；后续候选须达到全部物理条件并复核，考虑尾部变动，不用一次更好的max穿透值宣称解决。

预算：3/3Job，运行合计1432.573520秒（23.876分钟），完整Run保留量7435979692字节（6.925GiB，含失败/日志/缓存），0更新、0重试。独立复核发现初版仅遍历outputs，漏计Run根console.log共107477字节，已按完整Run目录纠正，仍低20GiB。见[budget-audit.json](artifacts/2026-10-13-rpo-learning-round5/budget-audit.json)。第六轮属于总3轮预算第2轮，剩余最多6Job；第一项I1只改变积分dt。

三轮复核：①物理与数据语义、完整初态/动作/编译身份，原RSI动量/奖励/限位不变；纠正legacy诊断与真实训练context混淆。②全部GPU走Job/Run/ebd，失败保留并计数，独立v2、Run耗时和全输出预算闭合，未训练/恢复。③全部事件同Run前史验证、图表由实际数组生成、捕获两图已打开；R0图表生成后再实际查看，逐轨迹残余和无评估录像如实保留，不称训练/模型验收。

补充复核：[source-peak-reproduction.json](artifacts/2026-10-13-rpo-learning-round5/source-peak-reproduction.json)中A0来源velocity/braking共12条在R0同ID/同timestamp均仍original_alive，target差0，qvel全向量最大差2.86e−5至0.294526rad/s（anchor最大0.042866），motor差1.72e−5至1.04252Nm。比大部分整段最坏波形差更小，但仍不称精确事故重放。env1756本次A0的3.46067倍事故前史已同Run保留、R0相近；第四轮旧4.28354倍事故的完整前史仍缺，不能倒推补齐它。

最终5张图（capture两张、replay两张、reproduction）均已实际打开。图的最大速度case不等同最大穿透case，穿透保护以全population接触直方图与max为准；没有新增独立评估/视频。

下一阶段：[第六轮](2026-10-13-rpo-bfm-learning-round6.md)。
