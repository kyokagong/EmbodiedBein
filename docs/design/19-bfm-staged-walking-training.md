# 19 · BFM 行走训练的诊断与分段闭环

2026-10-10（Asia/Shanghai）。延续设计14/16/18；训练仍为 API → Job/Run → ebd → RLinf，禁止教师和 Git 提交。上一轮1000更新及全部历史数据、模型、Suite、镜像保持只读。

## 目标和依据

目标是在真实 Isaac 物理中学会正常行走。训练完成不等于模型合格。既有模型100 case通过0、66跌倒、动作97.2%饱和；每轮98304新transition却仅抽128学习，replay每轮替换98.3%。另已实际回读checkpoint：`z_pool_count=0`且整个latent池为0；learner从未调用已有`add_mixed_z`。固定上游`FBcprAuxAgent.update`会将mixed-z加入缓冲区，这个闭环必须修正后再扩大训练。

不以增加网络/环境或PPO任务奖励替代BFM。先保持bringup网络、已有codec/PD增益、冻结混合motion版本`ds_27fbaa34696e4f26b2c7`及其祖先分组。公式迁移未逐位复现上游，修正latent池也不能单独保证收敛。

### 运行中追加的上游配方复核

固定[上游实际训练配置](https://github.com/LeCAR-Lab/BFM-Zero/blob/318cf44a3262e5bdec5944f82f1a5f509b95d09b/humanoidverse/train.py)的数值与现有bringup有实质差异：

| 项 | 上游 | 当前bringup |
|---|---|---|
| joint batch | 1024 | 128 |
| F / B / D学习率 | 3e-4 / 1e-5 / 1e-5 | 1e-4 / 1e-4 / 1e-4 |
| Q_D / Q_aux学习率 | 3e-4 / 3e-4 | 1e-4 / 1e-4 |
| FB target pessimism | 0 | 与actor/Q共用0.5 |
| relabel比例 | 0.8 | 全部重标记 |
| gradient clip | 配置0（禁用） | 所有模块阈值10 |
| expert连续rollout | 50%环境、250步专家片段 | 仅mixed-z池、每100步刷新 |
| 显式网络初始化 | `weight_init`后复制target | 构造器默认初始化 |
| norm更新输入/频率 | replay obs/next、每个联合更新 | 全量新rollout、每采样轮 |
| online replay容量 | 5,120,000 | 100,000 |
| draw/new transition | 每1024新样本抽16×1024，即16 | 初始0.001302；新probe0.083333 |

因此首次16/24更新只验证闭环与恢复，不能把bringup称原论文训练复现。长训前须把FB pessimism与actor/Q分开、显式relabel、模块学习率及batch/clip语义对齐到新的冻结配方，并核对专家连续提示与reference初始化路径。参考状态初始化/提示调度不训练教师，不把随机latent池当成专家连续rollout的等价实现。该阶段改学习语义须新初始化、新镜像/配置与小样本验证；不能从旧bringup完整恢复后声称同配方比较。RPO特有辅助约束及fall/reset仍需明确报告与G1差异。

进一步核对[上游FBAgent](https://github.com/LeCAR-Lab/BFM-Zero/blob/318cf44a3262e5bdec5944f82f1a5f509b95d09b/humanoidverse/agents/fb/agent.py)：专家分支对连续250步reference用B编码，再按8帧向前均值生成逐步z；先随机选环境，然后按episode时间执行连续z。配置的0.5是抽取索引数比例，源码索引有放回，不能称恰好50%独立环境。该机制本身不设置机器人参考初态；reference-state initialization须另核对环境实现，不能猜测原文要求或擅自增加教师。

现有冻结train组实际有2,210个250步窗口、15个具备该长度的clip；8步有11,922窗口。连续分支必须只消费train有效连续区间，不能跨invalid帧或拼接短动作伪造连续性；短片段仍用于8帧expert训练。不同长度的片段覆盖须单独记录，避免只看总片段数。上游总预算约600万联合更新，1000次属于早期诊断；拟议20,000预算也只是有界尝试，不承诺足够收敛。

下一版实施必须覆盖现有worker/policy/motion sampler、模板白名单与配置、恢复指纹/回执、单卡和四卡检查。先新增冻结核心配方：上游模块LR、FB pessimism=0、relabel=0.8、norm按replay每更新、显式初始化、clip禁用语义、joint batch1024；RPO bringup网络和物理接口差异明确保留。随后接入连续专家提示，bootstrap重复读不推进片段、episode reset只重采样对应环境、expert RNG/轨迹cursor回读；新增actual expert比例/窗口来源指标。每项先最小行为测试，再新镜像真实短训/恢复/同seed快评。env数重新按抽样重用、replay留存、实际updates/s及显存余量联合选择；本次3072env不是下一版已准入上限。recipe改变不接旧optimizer，旧图不拼成新recipe连续曲线。

## 实现边界

1. mixed-z进入模型buffer，随RLinf完整state权重同步抵达rollout；原随机初始化只用于池尚未填充时。checkpoint必须覆盖池/游标以及optimizer、targets、replay、learner RNG、expert sampler。
2. 复用loss函数已经产生的诊断量，记录D分类/GP/logits、FB分项/B范数、actor三项价值及贡献、动作饱和、梯度裁剪、aux三个原始加权分量、replay年龄与实际采样量、latent池填充量。不将归一化z范数当作多样性证明；小batch的B有效秩只作为批次诊断。
3. 显式支持BFM `actor.model.updates_per_round`为1或8。RLinf `runner.max_epochs`、checkpoint step和日志横轴继续表示**采样轮数**；真实learner更新为轮数乘该值，由updater/optimizer回读核验。1保持历史语义。8时125轮=1000更新，不能启动1000轮误称1000更新。scheduler保持冻结constant、每采样轮推进；不声称非constant schedule已支持。
4. 单Run最多新增1000联合更新；累计目标最多100000更新是平台上限，不是用户授权预算。恢复从同配方完整trainer_state开始，终止目标可增大，模型/数据/拓扑/LR/UTD保持不变。末环境和3个rollout进程不持久化，重建初态，明确不是exact。恢复检查learner计数与checkpoint轮数一致，错误直接失败，禁止从头训练兜底。
5. 初始候选总env3072（3×1024），每轮4步，batch128，8更新，draw/new=.08333，是原配置64倍。先单卡小样本/恢复/快评，再四卡同配方组合探测；选择依据更新吞吐和任务进展，不追求最多env。256/512 batch或更多更新不在首个配置内。

## validation 与 acceptance

派生独立冻结Suite：validation使用61000–61015共16 seed，acceptance使用62000–62099共100 seed；均复用v3官方指令/地形/噪声/随机化及共享场景批处理协议，不覆盖v3旧Suite。validation可4GPU×4独立env/卡；每例20s/1000控制步、平移RMSE≤.20m/s、yaw≤.30rad/s、无跌倒/NaN/硬限位。validation只诊断/选模型，qualification不请求；acceptance至少80/100且证据完整才通过。已观察的50000–50099仅回归对照，不作为新泛化验收。

每段正常结束发布ModelVersion、释放GPU后运行独立快评，真实录像及逐case物理trace必须保留。先真实单卡采样→更新→完整checkpoint→恢复后首个更新→独立快评→实际解码及查看录像；通过后才四卡。运行中暂停/评估没有证据，不依赖它。若全部失败，则注明没有成功录像。

## 周期决策与预算

新一轮总预算须由用户确认；候选为最多3配置、累计20000 learner更新、训练/评估24h，不能用旧1000实验预算推导无限调参授权。基础接入修复及有限smoke不依赖长训预算。

第一段1000更新；每1000更新快评、绘制原始曲线和50点均值，列出实际计数/恢复偏移、采样量、墙钟和资源峰值。前2000更新为warmup，之后至少3个评估窗口；连续3次无改善停止当前配置。改善按成功数优先，其次median存活控制步（至少+10步）；均无提升时不以训练loss下降继续。任何非有限更新/损坏checkpoint/指标与optimizer计数不符/必需媒体缺失停止等待证据；基础设施失败最多一次有原因的修复重跑。动作持续高度饱和且存活退化时停止分析actor价值尺度，不追加无界训练。

调整按证据一次一个维度，最多3配置：先latent闭环+UTD；若饱和持续，独立新初始化配置降低actor学习率；若诊断显示辅助约束价值不足，另派生任务无关aux系数配置。每次变更必须新spec/镜像，不把不同配方权重warm-start称完整恢复。具体数值由首段诊断决定，不提前把候选称优化结果。

每段台账记录输入hash、TrainJob/Run/ModelVersion/快评ID、起止真实learner计数、case汇总、图/媒体索引、预算消耗以及continue/stop/wait决定；没有决定不得进入下一段。冻结候选后才运行新的acceptance，失败不将该acceptance继续作为调参反馈后复测同一组声称泛化。

## 三轮复核

①设计/本体单位、数据隔离、上游算法与latent同步；②组合GPU、实际update预算、恢复失败/取消、不占其他作业；③曲线/逐case/视频解码与实际观看、源hash及完整报告。进度只报告实际证据，不修改资格门槛或伪造缺失媒体。
