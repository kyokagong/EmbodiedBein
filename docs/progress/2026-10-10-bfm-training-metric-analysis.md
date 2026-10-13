# BFM/RPO 四卡训练：39项指标释义与问题分析

对象：`trn_3593c876d82e489a9475`，Job `job_3732ab6715164a319c69`，Run `run_d8c1001d92f4450a85eb`，最终模型 `mv_41f640b997cc499f84ea`。日期：2026-10-10 Asia/Shanghai。仅分析，不修改训练代码/模型/已发布数据，不启动新训练，不训练教师、不提交Git。

结论：1000次联合更新确实完成且日志/权重有限，但模型未学会稳定控制。最明确的配置不足是采样与更新严重失衡；同时存在FB/Q梯度频繁裁剪、在线行为与专家可轻易区分、评估动作饱和，以及训练页面的指标语义和诊断拆分不足。不能凭总loss断言唯一根因。

## 证据与读图规则

本次从训练API再次回读冻结参数，Run metrics API的39条series与上轮归档逐点完全一致。通过实际训练镜像`eb/rlinf:rpo-v2-bfm-four-gpu-v6`读取指标实现，四个BFM runner源文件hash与工作区对应文件一致；没有使用新评估镜像推测旧训练算法。

- [39项完整数值统计](artifacts/2026-10-09-bfm-four-gpu-1000-eval/metric-statistics.json)：点数、首末值、均值、范围、前50/后50点均值、梯度超过阈值的比例。
- [全部原始曲线](artifacts/2026-10-09-bfm-four-gpu-1000-eval/metrics.json)、[训练/评估完整报告](2026-10-09-bfm-four-gpu-1000-evaluation.md)。
- 横轴是runner step 0–999，共1000轮；`train/bfm/update_count`的值是1–1000。它不是每个env的控制步，也不是累计transition数。
- 四个episode指标只有999点，是第一轮4步尚无episode完成而未记录，不能当成缺失一次训练更新。
- 环境指标来自本轮**完成episode**的均值，不是独立评估成功率；`num_trajectories`是本轮完成episode数，不是累计数。
- `grad_norm`是`clip_grad_norm_`返回的**裁剪前L2范数**，阈值10。不是裁剪失败或非有限梯度；本次所有记录标量有限。
- 下文loss/环境表用最后一个实际记录值；时间表用1000轮的平均值。多个loss尺度和符号不同，不能横向比大小，也不是全部都以接近0为成功。

## 一、环境与采样（6项）

| 完整指标名 | 本次最终值 | 实际含义与判断 |
|---|---:|---|
| `env/episode_len` | 9.0542控制步 | 本轮完成episode的平均长度；50Hz下约0.1811秒。应结合结束原因看，直立任务的持续时长越长通常越好。前50点13.3294，后50点9.0547，整体变短。 |
| `env/num_trajectories` | 10808 | 本轮所有采样rank完成的episode数。第一轮0、全轮均值9213.37，1000轮合计9213369。高值也可能表示频繁失败/重置，不能当成成功次数。 |
| `env/return` | −0.630351 | 完成episode中累计原生辅助奖励的均值。短episode会少累加负惩罚，数值变大并不自动表示控制变好。 |
| `env/reward` | −0.0683701 | 先对每个episode算return/episode_len，再求均值。不是判别器奖励或速度跟踪奖励；也不必等于均值return除以均值episode_len。 |
| `env/success_once` | 0，全程0 | RLinf Isaac桥按episode是否出现过`step_reward>0`计数。本环境奖励恒非正，所以该值结构上恒0；它不适合作为本任务成功率。真实成功率来自独立100-case物理Suite。 |
| `rollout/bfm/collected_transitions` | 98304，每轮恒定 | 一轮实际交给learner的`(s,a,s')`数，`24576 env × 4控制步`。累计98304000，不能把最后98304当总量。名字在rollout分组，但函数实际只统计收到的批次。 |

环境奖励由[rpo_bfm_extension.py](../../backend/eb/runners/rpo_bfm_extension.py)计算：

`r_aux = 0.02 × [−10 Σ软关节位置越界 − 1e−5 Σ施加力矩² − 0.1 Σ动作变化²]`。

三项均非正。`env/reward`与`env/return`不包含D风格奖励或FB价值；在线学习的D reward是在learner内另外计算的logit。RPO机械约束中的关节速度、跌倒和指令跟踪没有单独出现在这三项辅助奖励里；行为先验承担其他行为学习信号，不能将它误解成原RPO-Flat的速度任务奖励。

`env/success_once=0`的语义问题由冻结镜像`rlinf/envs/sim/isaaclab/isaaclab_env.py:96`和上述非正公式共同确认。已发布历史指标保留，不修改它来制造成功率。

## 二、学习与资源（16项）

下表字段均以`train/bfm/`为前缀。

| 字段 | 最终值 | 含义与本次判断 |
|---|---:|---|
| `disc_loss` | 0.0705143 | 专家/在线二分类损失，加10倍输入/latent梯度惩罚。首值9.74048，前50均值2.27285→后50均值0.0597246。D在训练样本上容易分辨两者，不能解释成策略像专家；没有分类holdout，不能确认过拟合。 |
| `disc_grad_norm` | 2.29372 | D裁剪前梯度范数。全程1.6%更新超过10，最终未超阈值；不是当前最大的梯度尺度异常。 |
| `fb_loss` | 25417.3 | F/B的后继状态表示目标：非对角Bellman误差、负对角目标、100倍B正交约束之和。前50均值316013→后50均值18490.7，下降明显，但总loss可能为负且缺少拆分，不能仅凭绝对值判定爆炸/收敛。 |
| `fb_grad_norm` | 267351 | F+B合并的裁剪前梯度范数。首值1066610，最高5945508，最低188757；100%更新超过10。严重依赖裁剪，需拆解哪项产生大梯度。 |
| `critic_loss` | 2.39772 | 两成员Q_D ensemble对D-logit奖励Bellman目标的均方误差。前50均值8.62339→后50均值1.74969；在拟合移动价值目标，不是成功率。 |
| `critic_grad_norm` | 25.0913 | Q_D裁剪前梯度。99.8%更新超过10，最高13348.9，后50均值29.0093。价值分支长期依赖裁剪。 |
| `aux_critic_loss` | 0.751031 | Q_aux对经过EMA尺度归一化的辅助奖励Bellman目标的均方误差。前50均值7.86376→后50均值0.515038；说明拟合损失下降，不保证实际硬限位安全。 |
| `aux_critic_grad_norm` | 6.58908 | Q_aux裁剪前梯度。72.4%更新超过10，最高8959.72，后50均值6.53546，晚期较早期平稳。 |
| `actor_loss` | 7.52703 | 最小化负FB价值、负D正则价值和负辅助价值的组合；不是BC动作MSE，也不是PPO loss。前50均值−0.311564→后50均值9.39303，价值函数本身同时变化，不能单独从正负/上升判断真实策略性能。 |
| `actor_grad_norm` | 0.165001 | actor裁剪前梯度。1.2%超过10，范围0.0265–31.9109，后50均值0.165861。没有记录到零梯度；结合评估饱和应检查tanh前logit，但此指标不能独自证明梯度消失。 |
| `expert_z_norm` | 16，全程16 | 连续8帧专家B编码平均后投影到256维球面，`√256=16`。只证明归一化正确，不能证明latent有多样性或语义质量；全部方向相同也可有这个范数。 |
| `replay_size` | 100000 | 当前CPU replay内的transition行数。首轮98304，第二轮起满容量，不是累计采样量；本轮几乎整个池都被新数据替换。 |
| `update_count` | 1000 | 已实际执行的learner联合更新累计次数，每次含D、F/B、Q_D、Q_aux、actor更新及target软更新。不是单个optimizer累计次数之和。实际168个optimizer step张量均1000。 |
| `target_update_count` | 1，每点恒1 | 一次联合更新中完成一轮target软更新的回执；包括F/B/Q_D/Q_aux四组target。本次每轮一次，因此累计1000，并非target只更新了一次。 |
| `learner_peak_allocated_mb` | 3101.65 MiB | learner进程从启动至今Torch实际tensor分配峰值，名称mb但计算为`/1024²`。不是整卡显存或本轮当前值；累积峰值不上降正常。 |
| `learner_peak_reserved_mb` | 3286 MiB | learner Torch缓存分配器累计保留峰值，含未使用缓存。大于allocated正常，不能据此判断泄漏；不含采样卡/PhysX/驱动的全部显存。 |

主要实现：[bfm_zero_worker.py](../../backend/eb/runners/bfm_zero_worker.py)、[bfm_zero_learning.py](../../backend/eb/runners/bfm_zero_learning.py)。actor目标为：

`L_actor = −mean(Q_FB) − mean(|Q_FB|).detach() × [0.05 mean(Q_D) + 0.02 mean(Q_aux)]`，价值先经过双成员悲观聚合。不能将三类Q的尺度变化全部归因于actor变坏；目前各分项未写入日志。

各梯度超过10的比例依据全部1000点统计：FB100%、Q_D99.8%、Q_aux72.4%、D1.6%、actor1.2%。最终FB梯度被整体缩放约`10/267350.65625=3.74e−5`；这是当次梯度的缩放，**不等于Adam参数更新幅度必然同比例缩小**。

## 三、时间与吞吐（17项）

全部单位秒。以下值是1000轮的平均；Worker内同名多次调用时间相加，跨rank默认取max，而非三个rank相加。主流程同步等待、GPU异步执行、嵌套计时均存在，所以这些不是纯GPU内核时间。只能对互不重叠的主阶段分析墙钟占比，不能将整个表相加。

| 完整指标名 | 均值s | 含义 |
|---|---:|---|
| `time/step` | 11.2843 | 整轮runner墙钟时间，含同步、采样、联合更新及该轮checkpoint等开销；最后保存轮较慢正常。 |
| `time/generate_rollouts` | 8.90716 | 主进程等待env/rollout产生并向learner交付整轮经验的关键路径。 |
| `time/actor_training` | 1.19943 | 一次learner联合更新的主进程等待时间，含replay抽样、专家编码、全部网络更新与target更新；不只是actor网络。 |
| `time/sync_weights` | 1.06261 | 主进程执行learner→rollout权重同步的时间。 |
| `time/cal_adv_and_returns` | 0.002416 | 沿用RLinf阶段名，但BFM实现只是检查收到的transition并记录数量，没有执行PPO GAE。很小是正常现象。 |
| `time/actor/actor/recv_traj` | 8.85141 | learner等待并接收采样轨迹，包含等待轨迹产生，不能把8.85秒都归因于通信慢。与generate_rollouts重叠。 |
| `time/actor/actor/sync_model_to_rollout` | 0.540480 | learner侧发送模型同步数据的时间；与sync_weights是不同层的同一阶段。 |
| `time/rollout/sync_model_from_actor` | 1.06046 | rollout侧接收/应用权重的最长rank耗时，与主进程sync_weights接近。 |
| `time/rollout/rollout/generate` | 7.76997 | rollout worker生成整轮动作、收发观测的总时间，包含等待环境。 |
| `time/rollout/generate_one_epoch` | 7.76943 | 一个rollout epoch的内部时间；本次epoch=1，所以与上一项几乎相同。 |
| `time/rollout/predict` | 3.79247 | 本轮各动作批次predict调用累计的最长rank耗时，含观测归一化、latent维护和actor推理；不是单个env推理3.79秒。 |
| `time/env/interact` | 8.11289 | env worker整轮交互入口的最长rank时间，含等待动作。 |
| `time/env/run_interact_once` | 8.11286 | 上一项包裹的单轮交互内部时间；基本同义，不能重复累计。 |
| `time/env/env_interact_step` | 0.926068 | 本轮实际环境chunk-step调用时间之和，包含动作准备、仿真和终止/重置/指标整理；不是单纯PhysX内核时间。 |
| `time/env/env/send_rollout_trajectories` | 0.338311 | env worker将构建好的完整trajectory批次发送给learner的时间。 |
| `time/env/env/bootstrap_step` | 0.000302 | 交互开头准备初始/缓存观测状态的开销；不代表learner中的Bellman target计算。 |
| `time/env/compute_bootstrap_rewards` | 0.000296 | 通用桥奖励整理/可选bootstrap处理的累计开销；BFM没有PPO value head，实际Q/FB target在learner计算。 |

实际吞吐：按轮计时约8712 transition/s；包括容器启动与最终收集的整体约8544 transition/s。它衡量采样吞吐，不能等同于学习/控制质量。

## 四、根据指标判断本轮问题

### 1. 采样与更新严重失衡，是最明确的配置不足

每轮`3×8192×4=98304`条新经验，而只抽replay128行做一次联合更新：

- 用于梯度优化的抽样使用量/新增量=`128/98304=0.00130208`，即0.1302%；1000轮抽取128000行，对应新增98304000行。完整在线批次还参与观测归一化统计，不能说其余数据完全没有被处理。
- replay容量100000，每轮新增98304，约98.304%缓存被替换，历史留存仅约1.02轮。单条经验在被覆盖前有约0.13%的抽中概率量级；随机重复抽样意味着128000行不是128000条唯一经验。
- 一个learner承受三张卡的大规模采样，每轮优化预算却维持一次。更多env提高了容量/吞吐，降低了给定1000更新预算下的经验利用率。

这是原方案选择8192env/rank时的工程不足：容量探测证明能运行，不能证明学习预算匹配。我此前只把低UTD记录为风险，正式配置应该同时约束更新/采样比。不能把9830万transition当成充分优化的证据，也不能承诺只调整这一项就一定通过评估。

### 2. 物理持续时间没有改善，reward改善不能代替它

前50个有效episode记录平均13.3294步（0.2666s）→最后50个9.0547步（0.1811s）。完成episode数则前50轮平均7627.92→后50轮平均10853.3。组合信号表明更频繁结束/重置，未学到长时间维持。

辅助奖励前50平均−0.093205→后50−0.071198，数值有所改善；累计return也受更短episode影响。这是典型的“部分代价/拟合损失改善，但行为未改善”。训练是stage-A平地、无传感噪声和训练latent，完整评估恢复terrain/noise并使用command latent，因此两者时长不宜逐轨迹直接比较；即使训练本身，持续时间也不足。

### 3. FB和Q_D长期依赖裁剪，需要先查尺度与loss分解

FB每次裁剪，Q_D99.8%裁剪；FB目标的前50/后50均值虽下降约94%，仍有显著波动。这里可确认大梯度和剪裁依赖，不能确认总loss持续发散：它整体在下降、权重和梯度都有限。

F/B目标带100倍正交项，B输出范数16，总loss里又有负对角项。因此需要记录`fb_offdiag/fb_diag/orth_loss/B_norm/target_M`和梯度分项，定位到底是表征相似度、归一化、损失权重、target尺度还是其他因素。不要仅因为值大就增大裁剪阈值，或把统一降低学习率当成已证明的修复。

### 4. 判别器容易区分策略与专家，策略未追上先验

D分类项和GP项均非负，最后50点总loss约0.0597，说明本轮训练样本上的专家/在线区分已容易。这与“D训练成功”相容，却不能证明“actor模仿成功”。没有D概率/logit、专家与在线分项、holdout分类指标，不能区分行为差距、容易的观测特征捷径或过拟合。

Q_aux loss下降只是惩罚价值拟合更好；Q_aux是学习偏好，不是机械安全保证，尤其当前辅助项没有显式关节速度项。

### 5. 评估显示真实输出饱和，actor_loss本身无法定位原因

最终四卡100例：0/100成功，66触发fall、36触发硬限位、0NaN/infra，速度RMSE0.830548、yaw3.924208，平均时长0.094s、最大0.22s。fall/硬限位可重叠；硬限位中15例关节位置、25例关节速度，4例重叠、0力矩越限。

10810个实际动作值中97.206%满足abs(action)≥0.99，平均绝对动作0.991535。确定性actor使用tanh输出，所以这些评估输入处于输出饱和区；归一化动作未越过±1并不保证物理关节目标/速度安全，因为执行目标是`q_default+0.25×gain×action`。

actor晚期梯度范数很小但非零，可能与饱和、估值/目标尺度或分布有关，不能凭0.165证明梯度消失。必须检查tanh前logit、训练态饱和比例、actor各Q项及实际参数更新范数。现有指标不能将问题唯一归因于官方RPO或G1动作库，静态FK资格也不证明动态接触/控制匹配。

### 6. 大部分时间用于采样等待，而不是学习

主阶段平均占比：generate_rollouts78.93%、learner联合更新10.63%、sync_weights9.42%，其余约1.02%。本次未启用训练pipeline重叠，一个learner通常在等新经验；这个比例不是GPU利用率测量，但直接体现墙钟预算分配。

`predict`约3.79s含8192env的逐例latent上下文循环及CPU↔GPU复制，是值得profile的候选热点。现有时间指标不足以证明某个Python循环或通信组件是唯一瓶颈。allocated/reserved峰值后期平稳，整卡GPU0峰4400MiB、采样卡最高8539MiB，本次没有显存OOM证据；继续增加env或网络规模不是首要改进。

## 五、建议的下一步顺序（本次未执行）

1. **先补诊断，保留旧指标**：真实终止原因/训练态动作饱和、三个辅助惩罚分项；D的专家/在线CE与logit/GP；FB分项、B协方差/effective rank；actor的Q_FB/Q_D/Q_aux及缩放权重；各分支裁剪后norm与参数更新norm。用实际任务判据派生成功指标，不能原地重解释历史success_once。
2. **用相同learner更新预算做小规模比例实验**，不要以最大env数为默认。例如候选3×1024env、4步rollout、batch128、每轮8次更新，共125轮=1000联合更新，样本使用/新增比变为0.08333（当前64倍）。这是可测试的实验点，不是已验证或论文最优超参；新增多次更新时必须按实际learner计数停止，不能仍跑1000轮导致8000更新。
3. **定位梯度尺度与饱和后做单因素实验**。先检查loss归一化、FB正交分项、Q目标/归一化、actor更新幅度，再按证据调整权重/LR/裁剪。固定来源/分组/codec/动作gain作对照，不通过改gain或成功阈值制造通过。
4. **用固定小评估集周期观察真实控制**，每个checkpoint比较实际持续时间、硬限位、指令RMSE和动作饱和；不等到1000更新后才第一次看最终能力。没有确认改善时，不直接追加大规模训练或换更大的网络。

![实际39项中的关键趋势及时间分配](artifacts/2026-10-09-bfm-four-gpu-1000-eval/metric-analysis-curves.png)

本轮未做参数消融、未新增训练/诊断指标、未验证论文research网络配置及真机；未确认唯一根因。现有证据足以判断模型不合格、配置比例和观测指标不足，但不能证明某一项改动必然恢复行走。
