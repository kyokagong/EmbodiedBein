# RPO BFM-zero 第二轮动力学与 RSI 诊断

计划：[优化执行计划](../plan/2026-10-13-rpo-bfm-zero-learning-optimization.md)；前序：[首轮记录](2026-10-13-rpo-bfm-learning-optimization.md)。用户已授权冻结第二轮有限预算，补齐动力学和 RSI 证据，再按结果推进采样修复单因素对照；不提交 Git。

**最终结论：第二轮3个Job已结束，动力学与RSI来源证据已补齐。接触/惯量耦合在制动饱和时继续加速踝关节；RSI reset没有丢失q/dq，358维差异来自已冻结的离线有限差分及跨后端FK口径。固定目标峰值比2.819仍高于2.5，停止训练放行。采样修复候选镜像已准备，未登记训练配方、未训练或独评。**

## 阶段 0：条件与预算冻结

- 合同：[experiment-contract.json](artifacts/2026-10-13-rpo-learning-round2/experiment-contract.json)，SHA256 `1099f54f50a508ef35b1343ab856605bf6e39b662578cf172e8a71ceed3192ae`。
- 上限：最多 3 个新诊断 Job，各 ≤30 分钟，无自动重试，总运行 ≤90 分钟，新增产物 ≤20 GiB。失败 Job 也计入预算；未用额度不自动转训练。
- 冻结 v13 step600 模型、训练参考、原始 RSI q/dq/root 速度、控制/限位/奖励/终止语义及 50Hz 控制。沿用已验证的四卡独占拓扑，GPU1–3 每卡2048环境，GPU0不运行 learner；诊断更新数为0。
- 初始 seed42/43/44，固定目标与旧冻结策略各48控制步，仅原始 episode 用于配对和事故取样；重置后不拼成共同轨迹。每阶段最多5个最终保留子步事件，全Job最多30个；明确这是动力学快照而非完整物理状态精确重放。
- 服务实际入口 `http://127.0.0.1:31800`，API/ebd健康；四卡空闲各24080 MB，可用主存约104 GiB。Doctor 的 Vulkan 配置检查失败，不把它解释成GPU缺失；本轮先采动力学数值，录像能力另验证。
- 工作区已有差异已保存在 artifact 目录，只读外部仓。沿用当前工作区，保留 hash 与失败证据，不暂存、不提交。

Ruling：首轮预算已结束；本轮由用户新指令授权，独立计数，不能继续使用首轮额度。动力学出口未满足前 `training_enabled=false`。只在接触/离散化仍不明时使用剩余 Job 做单因素诊断。

## 阶段 1：动力学与 RSI 证据

状态：已执行并取得真实数据。在 `sim.step()` 后、任何 `forward()` 刷新前保存被积分广义力、质量矩阵和约束Jacobian；同时保存积分前后 qpos/qvel、warmstart，区分约束接触与关节限位。358列按高度、局部位置、朝向、线速度和角速度分别对账，root lift与history清零单列。

实现及检查：新增诊断runner/镜像，复用Mujoco adapter、现有准入及collector，不改原训练配方。数值检查先复现缺实现，再通过；平台新operation准入先失败，再通过。独立只读复核发现回执证据不完整仍可通过、制动标签可能引用另一关节，两项均以失败行为检查复现并修复。最终镜像 `sha256:399a9a9ef0f03a36725c6bb35b7c2df8fc8a0346f1360e3d35dab239b21f7a98`；准备镜像未提交Job，保留原tag。

Ruling：MuJoCo-Warp implicitfast 的实际积分使用 `M-dt*dF/dv`；保存的M/J/求解器qacc和力来自该步积分前求解。`M^-1*force*dt`只解释瞬时贡献，不冒充完整积分归因；其与实际delta-v的差不能直接定为求解器错误。复核次要项：缓存/启动目录不在根回执文件hash列表，最终单独统计全输出大小与启动文件；52列分项在结果分析中补齐，保留原数组。

## 阶段 2：结果决策与采样对照

状态：RSI差异来源已解释，动力学出口未通过；采样修复镜像准备完成。周期决策、完整checkpoint恢复及媒体出口尚未验证，未启动训练。

## 平台账本

诊断使用 `asset.validate`，训练使用 `train.rl`，不将诊断伪装成训练列表记录。

| 次序 | Job / Run | 状态与预算 |
|---|---|---|
| 1 | [Job job_c368eb2517dc4d43add2](http://1.94.0.106:31800/api/v1/jobs/job_c368eb2517dc4d43add2) · [Run run_f985809e24ce44e3bfba](http://1.94.0.106:31800/train/runs/run_f985809e24ce44e3bfba) | 六阶段完成、原回执已写，collector标failed/invalid_output；计1个Job，剩余最多2个 |
| 2 | [Job job_81a01321f6754068881f](http://1.94.0.106:31800/api/v1/jobs/job_81a01321f6754068881f) · [Run run_132a776a1c33412e9dae](http://1.94.0.106:31800/train/runs/run_132a776a1c33412e9dae) | succeeded/finished，复制并校验真实物理证据，CPU FK/来源对账完成 |
| 3 | [Job job_35c5dbea6ddc493d90c5](http://1.94.0.106:31800/api/v1/jobs/job_35c5dbea6ddc493d90c5) · [Run run_7f8fe95ef20a41878a31](http://1.94.0.106:31800/train/runs/run_7f8fe95ef20a41878a31) | succeeded/finished，Run.gpu_ids=[]；最后一次CPU准入与证据复核完成，预算3/3 |

Job链接是需鉴权的原始API记录；Run链接是登录后可查看状态与日志的页面。提交请求/预览/回执留在本轮artifact目录。相关检查：27项数值行为、37项平台回归通过，无skip；pytest asyncio配置及passlib crypt弃用警告保持原状。

## 阶段 1 的收集失败与阶段 2 冻结

v1六个阶段实际完成，保留30个子步快照及6144个配对原始初态、全量RSI字段数组；原始回执SHA256 `65f594b9efe1795e85b2d772f7b752e58a273de8629fdb425a25bc4c72648f7d`。平台收集失败来自NumPy2.4.6与任务镜像版本的浮点归约：`linear_velocity.mean_abs_after_lift`分别为0.00028639487572945654与0.0002863948466256261，差约2.91e-11。原代码要求dict浮点完全相等，误判为证据不同；不是物理Job未执行或原数据损坏。保留原failed/aborted状态，不补造成功回执或改数据库。

Ruling：采用相对1e-6、绝对1e-10的浮点统计对账，保持数组hash、人数、字段边界及更新数严格校验。最小检查复现归约差可接受而明显改动被拒；真实MuJoCo小模型验证root世界角速度转换与link原点Jacobian速度。最终相关检查29项数值、38项平台通过。

第二个Job冻结在[stage2-contract.json](artifacts/2026-10-13-rpo-learning-round2/stage2-contract.json)：CPU8核/16GiB、GPU0张、≤30分钟、无自动重试，计入同一3Job/90分钟预算。只校验并复制v1保留的真实物理证据，再通过CPU FK/Jacobian对账RSI来源；新环境积分0步、训练更新0。阶段镜像 `sha256:66b8b7ea6665ee9d5713c020ffc1c5185829cbfc5929ef4bf98271d1588ab894`。不再占四卡复跑原物理。

源码证据：原参考converter从Isaac FK读取body pose，再以20ms有限差分得到body线速度/角速度；MJLab在线采用同一q/dq下link原点的瞬时速度。参考几何筛选原容差为2mm。当前1e-4诊断容差下的差异计数不等于参考失格，需经CPU原生FK与有限差分复算区分已知近似与实现错误，不直接撤销参考资格或修改采样池。

资源复核发现：v2请求GPU0张，但旧scheduler把所有asset.validate默认提升到至少一张，Run.gpu_ids实际为[0]。计算入口确为CPU FK且未调用CUDA训练；不能宣称调度未占GPU。针对该冻结CPU operation增加例外，保留原GPU诊断默认行为；行为检查复现实际[0]与预期[]的差异后修复。前两次测试fixture遗漏必需实体字段的错误已纠正，不把fixture失败当根因证据。预算内第三个且最后一个Job只验证修复后的真实CPU准入及保留证据，不再重跑GPU物理。

## 最终动力学结果

最终索引：[diagnostic-analysis.json](artifacts/2026-10-13-rpo-learning-round2/diagnostic-analysis.json)。完整原数组与来源审计在最终Run的`outputs/`；物理数据的真实执行源仍为v1 Run，v2/v3显式携带source_run_id及原回执hash，只重新验证证据和CPU运动学，没有重跑GPU物理或重写旧失败状态。

| 指标 | 固定RSI目标 | 冻结step600策略＋保存z池 |
|---|---:|---:|
| 原始配对初态 | 6144 | 6144 |
| 到48控制步/0.96秒未终止 | 3254/52.96% | 2/0.0326% |
| 原始episode子步峰值速度比 | 2.819233 | 3.927814 |
| 最终保留子步快照 | 15 | 15 |

只统计原始episode，后续reset回合不配对。存活者在0.96秒处右删失，不是20秒站立合格。与首轮旧策略0存活的差异只有2个episode，本轮不声称精确物理重放或学习改善；策略未更新。

代表事件：policy/rank2/env798，第8控制步、第3子步，右踝pitch速度 **26.901→31.423rad/s**，实际delta-v **+4.5215rad/s**；电机力矩 **−27Nm**，功率约 **−848.4W**，右足Z力约 **798.72N**。实际约束Jacobian×约束力得到该关节接触广义力矩 **+47.829Nm**，接触广义冲量 **0.11957Nm·s**；该关节直接限位力矩为0。全矩阵瞬时分解中接触项delta-v为 **+10.5398**，全部执行器项 **−6.0017**，被动项 **−0.06324**，bias项 **+0.04748**，限位耦合项 **+0.01199rad/s**。这是接触/惯量耦合压过制动的事件证据；不能将全部执行器项误称为该踝单电机贡献。

该事件的`J.T@force`与实际qfrc_constraint最大差 **1.53e-5**；M×求解器qacc与力和的最大残差 **0.01723**。瞬时delta-v之和与实际积分的差约 **0.01044rad/s**，符合需要单列的implicitfast积分口径，不直接判定求解器错误。瞬时自由度等效惯量约 **0.010737kg·m²**，与配置armature=0.01相近；这不是实机反射惯量测量，也不据此猜硬件惯量或提高27Nm/8rad/s上限。

固定目标rank1/env1808同样出现右踝22.554rad/s、接触广义力矩+30.413Nm、电机−27Nm的组合。由此保留动力学分支，奖励调权不能直接证明解决。这些快照是有界选例，不是全体原因频率；尚未完成更小dt的独立敏感性实验。

![配对初态结果](artifacts/2026-10-13-rpo-learning-round2/diagnostic-curves.png)
![原始episode子步时序](artifacts/2026-10-13-rpo-learning-round2/diagnostic-timeseries.png)
![峰值事件的瞬时力贡献](artifacts/2026-10-13-rpo-learning-round2/force-contributions.png)

## RSI一致性结论

所有6144个原始配对初态的q/dq差为0；52列重力最大差1.34e-7、root局部角速度最大差3.58e-7，history清零符合预期。对358列扣除root lift后，与原参考的局部位置最大差约0.996mm、线速度0.13717m/s、角速度0.29345rad/s；本轮逐source/rank分项和原数组均保留。

CPU原生MJCF/Jacobian复算与MJLab在线358列比较：局部位置最大差1.23e-5m、线速度3.41e-5m/s、角速度3.70e-6rad/s；在冻结1e-4诊断容差下，6144行均无非预期差异。原参考线/角速度与从保存pose重新进行20ms有限差分的结果，最大差均为0；原参考body位置与原生FK最大差0.99622mm，处于原geometry_screen的2mm范围内。

Ruling：差异来源是参考离线有限差分与在线瞬时速度，以及参考Isaac FK与MJCF FK的已有近似，未发现reset丢速度或错误世界/局部转换。解释来源不等于证明该近似对学习无影响；若后续选择原生MJCF重编码参考，应派生新数据版本，保持原q/dq/祖先split，单独验证，不能混入采样修复对照。此次不改valid mask、不撤销专家资格、不零掉参考速度。

## 采样修复对照的准备与停止条件

准备镜像 `eb/rlinf:rpo-ufo-sampling-fix-prepared-v1`，ID `sha256:10b86670d8c77e4f483381207fb25835a98710430b83dd4628d6b272fbb9a0f2`。[sampling-only-change.diff](artifacts/2026-10-13-rpo-learning-round2/sampling-only-change.diff)证明相对冻结v13仅改变mode变量与mode-aware缓存；[sampling-image-audit.json](artifacts/2026-10-13-rpo-learning-round2/sampling-image-audit.json)核验其余六个核心模块hash完全相同。新policy SHA256 `d9276166d69c6d0a7875c0e275ae7760f0287a985b032a3213d909a877f7edee`；原控制、奖励、RSI和UFO网络不变。

这是未训练、未注册的候选镜像，不能选旧v13配方套入它或把旧完整训练状态冒充新行为配方恢复。新的recipe身份/恢复指纹、pending更新块、K=100周期决策、MJLab独评和媒体链路仍待实现；准备镜像不等于这些能力已通过。

**阶段决定：stop_training_and_resolve_dynamics。** 按第二轮合同，固定目标峰值比2.819>2.5，训练放行未满足。没有启动100/500新训练、没有模型质量改善或独评/acceptance结论；奖励仍冻结。下一阶段应优先将接触点/法向、完整M/J与制动提前量用于选择一个控制/接触候选，在新有限预算内单因素验证；没有证据时不先猜PD/惯量参数。仅当物理出口和周期/恢复/媒体前置检查满足后，才执行准备候选与同初始化对照的100→500真实更新。

## 预算、检查与三轮复核

- [run-budget-audit.json](artifacts/2026-10-13-rpo-learning-round2/run-budget-audit.json)：三次Run实际365.355、16.093、16.053秒，合计397.501秒/6.625分钟；每个低于30分钟，累计低于90分钟。这是Job运行墙钟，不包括代码实现、构建和测试。3/3额度已用，不追加第4个Job。
- 最终相关检查：29项数值/状态行为、55项平台/调度回归通过，无skip；警告只有既有pytest asyncio配置及passlib crypt弃用。实际最终Run `finished`、`gpu_ids=[]`、optimizer_updates=0，诊断回执已进入metrics_summary。
- 最终输出含启动文件约49.8MB；原始v1及两次派生输出均保留。[complete-evidence-index.json](artifacts/2026-10-13-rpo-learning-round2/complete-evidence-index.json)核验三个Run含缓存共189374028字节，约181MiB，低于20GiB总上限。各启动JSON独立hash索引，缓存不冒充模型产物。Doctor确认四卡释放、各空闲24080MB，ebd健康。Vulkan配置问题尚未修复，未声明录像链路可用。
- 第一轮复核（语义/数据）：完整qpos/qvel和积分前后时刻明确，约束接触与限位分开；52/358和来源FD/CPU原生FK闭环，参考资格与祖先隔离保持，不把近似差异当损坏。
- 第二轮复核（执行/资源）：保留v1 collector失败与v2误分GPU0的事实；浮点统计与CPU准入缺陷均以行为检查复现修复，v3实际[]验证；3Job预算闭合，更新数0。
- 第三轮复核（图表/证据）：图表来自真实原始点，30个快照与6144原始配对初态可追溯；右删失、选例偏差、非精确重放、implicit瞬时分解、无独评/录像均单列。未通过物理出口，未宣布训练验收。

没有运行全仓测试、GPU采样修复训练或正式独立评估，没有提交Git。
