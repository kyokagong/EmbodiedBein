# RPO UFO MJLab v13：动态RSI与训练位置容差优化实验

本文为独立完整记录；配置、原始指标统计、诊断和结论全部在本文展开，不要求读取其它文件。时间为UTC，上海时间加8小时。

## 当前结论与分析校正

最终判读：首步训练终止率已通过0.1%门槛，但严格首步位置违规1539次、速度违规27次，唯一违规1566次（1.4873%），不能将训练终止减少称为严格边界彻底解决。末窗训练12.7107步比v12的8.4938高约49.7%，严格末窗7.3027比v12的7.4692低约2.23%；训练位置容差扩大是明确的混杂因素，当前证据不足以宣布步态质量优于v12。

本段99547个已完成episode中，86364个终止转移包含训练位置违规（86.76%，与其它原因可重叠）；严格位置标志有243700条转移，因为严格违规后训练仍可继续，不能把它们当作243700个独立终止episode。已完成episode中位6步、严格中位4步，严格最长33步。位置边界与足踝动力学仍是主要瓶颈。

数值没有发散，但Critic窗口均值0.475→0.369→0.342→0.392→0.504存在后程回升，判别奖励末点−0.893、Q_fb末点63.62；不能仅凭有限标量和梯度裁剪就宣称算法完全稳定。requested/applied全段饱和37.73%/11.31%，后者高于v12的9.21%。足部垂直净力峰值4701N略低于v12，但速度42.58rad/s更高，说明单看力峰值不足以判定冲击可控。

全局超速发生于right_ankle_pitch_joint，Replay第16步（零起算，即第17个控制步）、子步6；dq=−42.578rad/s、q=−0.30856rad、target=+0.25530rad，电机实际+27Nm处于反向制动饱和。同帧右足垂直净力1219.91N。该证据支持“控制器已制动仍不能约束瞬态”的判断，不足以单独分离接触力矩、关节耦合与惯量原因；垂直净力不是完整碰撞扭矩。未来先用冻结事故状态/动作做配对复现、核验完整质量矩阵与接触雅可比，再选择控制/软限位改动；没有硬件惯量数据前保留armature=0.01。

执行完成与质量通过分开：Train/Job succeeded代表执行与产物已收集，ModelVersion mv_5db6825607364c9fb1cc不是正常行走合格证明。平台模板kind沿用ppo标签，真实执行binding是bfm_fb_cpr_aux/UFO，未运行PPO损失。恢复最后一轮只执行剩余4更新，六Adam600为实际回读值；不要用采样轮×16反推更新计数或将trainer_state恢复称为连续物理轨迹。

v13已完成优化与100→600四卡实验，新增500联合更新成功，四卡已释放；工程流程通过，正常行走未达标。训练五窗口4.43→6.11→7.70→9.87→12.71步；严格影子3.52→4.57→5.45→6.36→7.30步。首步训练终止10/105291=0.00950%，严格首步违规1566/105291=1.4873%。右踝子步峰值42.58rad/s（5.32×），高于v12的35.78rad/s；严格末窗7.30仍略低于v12的7.47。已停止放量、不独立评估、不提交代码，v12和v13模型分别冻结。

已有RSI逻辑检查静态q范围及|dq|≤0.70×额定速度，并检查足底几何与法向速度。新增的是q与dq组合的20ms动态位置相容性检查，并非补上此前完全不存在的q/dq检查。线性外推未包含PD、重力、关节耦合和碰撞，不保证随机策略首步零终止。

原始MJCF和官方机器人配置均有armature=0.01kg·m²。没有找到该机器人各电机转子惯量与减速比，不能称该值已证实是占位符，不能任意提高该值掩盖飞车。反射惯量关系为I_reflected=N²I_rotor，需硬件数据才能校准；本轮保持原值。MuJoCo使用软约束，训练终止容差与实际物理joint range是不同机制；本轮只改前者。[MuJoCo官方3.8模型说明](https://mujoco.readthedocs.io/en/3.8.0/modeling.html)、[官方XML惯量定义](https://mujoco.readthedocs.io/en/3.3.6/XMLreference.html#body-joint-armature)支持这些机制说明，不提供RPO硬件参数。

v11与v12更换了后端、几何碰撞和初态有效池等多项条件；力峰值降低与吞吐变化不能单独归因于求解器。本轮同时改筛选与训练位置容差，不能将所有存活提升都归给过滤器；严格影子结果保持原判据，用于识别放宽训练容差带来的差异。

## 冻结基线v12

| 项目 | v12实测 |
|---|---:|
| 训练 | trn_68a5fc34919446c5adf4 |
| Run | run_4627d50438b84a07bc24 |
| 新增联合更新 | 24→524，500次 |
| 新增状态转移 | 786432 |
| 训练episode_len五个100更新窗口 | 3.2383、4.13525、5.21094、6.53911、8.49378 |
| 末点训练episode_len | 9.538614步 |
| 严格影子五窗口 | 2.98339、3.83911、4.79267、5.83511、7.469236 |
| 末点严格影子 | 8.355494步 |
| 首步终止 | 5478/155862=3.5146476% |
| 首步位置原因 | 5478例全部包含位置违规 |
| 全段物理峰值 | 35.775730rad/s，4.471966倍额定值 |
| 足部垂直净力峰值 | 5212.729N |
| requested/applied动作饱和 | 39.7377% / 9.2125% |
| 贴边外冲初态占比 | 12.3789% |
| 外冲组首步终止 / 其它组 | 18.7934% / 1.3561% |
| 20ms线性预测越界比例 | 3.3927449% |
| 独立评估/扩训 | 未运行，门槛未通过 |

旧镜像digest：sha256:071d5aacb6660f0bb48144fa2c6c0af34a8cfff9ef90819909dd4d2b4b63c986。

## 任务与完整关键配置

| 项目 | 冻结值 |
|---|---|
| recipe | rpo_bfm_ufo_agent_rlinf_mjlab_4gpu_viability_v13 |
| EnvironmentProfile | rpo_bfm_ufo_mjlab_4gpu_rsi_viability_v13 |
| Gym ID | RPO-UFO-MJLab-RSI-viability-v13 |
| 本体接口 | ee_rpo_v2_bfm_23dof_v2，RPO23DoF |
| 平台栈 | RLinf调度/采样/通信 + UFO原生FBcprAuxAgent + MJLab/MuJoCo-Warp |
| 镜像 | eb/rlinf:rpo-bfm-ufo-mjlab-viability-v13 |
| digest | sha256:e23281518d388565f4b1162dc9d2f21947543b2fd0a014b0288365be2788cee2 |
| GPU拓扑 | GPU0单learner；GPU1、2、3各sampler2048，共6144；不是DDP |
| 资源 | 4张RTX4090/每卡24GB；24CPU、96GB主存、32GBshm；exclusive |
| seeds | Actor42，采样42/43/44 |
| 网络 | 435116484可训练参数；Forward/Actor/Critic/AuxCritic2048宽×6层；Backward256×1；Discriminator1024×3；latent256、范数16 |
| Batch与Replay | batch2048；4M容量CPU流式FIFO；本轮从空Replay开始 |
| 采样/更新比 | 4控制步×6144=24576新转移/采样轮；warmup10240；触发16联合更新；每次六个Adam各更新一次 |
| 运行预算 | 从头100更新≤1小时，无自动重试；验证成功后trainer_state恢复100→600，新增500更新≤2小时 |
| Tracking | 每卡1024 tracking+1024 mixed；tracking绑定train参考起点、250步环境独立上下文 |
| 动作 | URDF有限范围中点/半宽解码；目标slew≤0.5×Vmax；策略请求动作进Replay，实际动作进last_action观察 |
| 物理 | dt0.0025s，decimation8，控制50Hz/物理400Hz；Newton100 iterations、implicitfast；原始MJCF质量/惯量/碰撞网格，self-collision关闭，无限平面 |
| 接触 | 原MJCF solref/solimp/friction；未增加恢复系数或虚构软接触参数 |
| 驱动 | MJLab DcMotorActuator显式零延迟；armature0.01、frictionloss0.01；无qvel硬裁剪 |
| 增益与额定值 | thigh Kp100/Kd3.3/120Nm/25rad/s；knee/torso150/5/120/25；ankle40/2.5/27/8；arm40/2/27/8；elbow_pitch30/1.5/27/8；elbow_yaw20/1/27/8 |
| RSI原约束 | 仅train；足底间隙2mm，初始高度≤20mm，root lift≤30mm；近地碰撞点法向速度≤0.15m/s，初态dq≤0.70额定值；保护部位离地检查 |
| RSI新增约束 | 当前q与q+dq×0.02均在原接口位置范围内并留1mrad内侧余量；只筛选，不改参考q/dq |
| 训练终止 | 原位置边界±0.02rad绝对容差，速度1.3×额定+1mrad/s；力矩原标准；保护部位接触>1N，高度<0.53m、倾角>0.70rad；8个子步锁存 |
| 严格影子 | 位置原边界±0.001rad，速度/力矩1.0×额定；8子步和宏观跌倒判据保持 |
| 辅助奖励 | 9项raw aux，位置90% soft limits；速度为8完成子步均值sum_j relu(abs(dq)/Vmax−0.7)^2、一次权重−25 |
| 优化保护 | Actor梯度裁剪1.0，其余5.0；scale_reg权重≤50；D lr3e−6、reward[-2,2]；discount0.98 |
| 精度 | FP32，compile/cudagraphs关闭 |
| 独立评估门槛 | 末两个100更新窗口训练episode_len≥100步；本轮未满足则不评估 |
| 扩训门槛 | 末两窗口strict shadow≥35步，全段物理速度峰值比≤2.5，首步训练终止≤0.1%；达到也不自动追加预算 |

数据集ds_fb0ecda9e70644f2afdd；manifestSHA dc0c6585887507df59f675963b2733d25c9836d805e9bb89f2402398f07bd288；assetSHA b58868a782d8863ac38059eb4eab89d27ee1d70565bf58f5a055f44510c8e842；codecSHA dce25d554ceb7fa69287515946d2f5cd9d41d14f1b1ef38715b819d7e962b6ef。专家source权重official_rpo0.5/g10.5；数据版本和原始MJCF未改写。v13改变MDP，不从v12恢复模型/优化器/Replay。

## 已执行验证与物理诊断

容器行为测试覆盖前瞻外冲/内收、负向不对称边界、q/dq不改写、非有限输入拒绝、训练位置容差与严格影子分离、原力矩终止及旧v12预算保持。新增3测试先失败后通过；原生/共享RSI/Tracking组合27passed；配方/资源/环境目录等59passed；修复资源漏接后相关组合29passed；原生MJCF组合11passed。组合有重叠，不相加作为独立总数。

完整backend单元回归558passed/31skipped；最初两个失败源于测试容器挂载/工作目录（缺少/workspace/contracts及相对deploy路径），修正运行方式后通过，未为此改产品代码。随后新增两个恢复候选检查，v13只允许确认过的100→600、600时关闭候选，相关平台组合8passed。原生解析配置覆盖v12与v13，2passed。可选依赖/GPU和真实仿真专项不由31个跳过测试替代。

独立复核发现v13诊断operation未被容量估算分支识别，落入0GB通用分支；已修复并新增行为测试（真实失败KeyError:num_envs→通过）。实际执行时四卡均有足够空闲显存；后续部署补齐准入证据，不将估算称为实测峰值。

物理诊断Job job_1f0b68bbcd5448ffb6be，Run run_e0c1916118bc42e1932b，已成功收集。每卡2048环境、32控制步，固定RSI PD锚点，不调用策略；观测重复读取不推进历史，部分reset仅清理选中环境历史，trace维数131/223、每区间8子步、所有状态有限检查通过。

| rank/seed | RSI首控制步样本 | 首步训练终止 | 首步严格限位违规 | 初始episode存活32步 | 全段终止 | 子步最大额定速度比 | 足部垂直净力峰值N |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0/42 | 2578 | 0 | 0 | 1543 | 562 | 1.90436 | 2465.15 |
| 1/43 | 2622 | 0 | 0 | 1486 | 618 | 2.81925 | 2685.35 |
| 2/44 | 2599 | 0 | 0 | 1518 | 579 | 2.65110 | 2685.36 |
| 合计 | 7799 | 0 | 0 | 4547/6144 | 1759 | 2.81925 | 2685.36 |

每个rank保存32×256个环境的原始trace和文件SHA，聚合计数来自全部2048环境，原始trace仅覆盖固定前256，不将其称为全量原始物理轨迹。Torch峰值reserved约108MiB不含Warp和驱动显存；显存以平台doctor记录补充。固定锚点首步改善证明该初态池在该控制协议下更相容，不能外推为随机策略首步终止必为零。2.81925×峰值仍未达到扩训门槛。

## 三轮复核与未验证项

计数解释：100阶段8采样轮，每环境32控制步（0.64秒）；新增500阶段预计32轮，每环境128控制步（2.56秒）。恢复时物理环境重新RSI，因此不能把两段拼成同一连续episode。本段即使零终止也不足250控制步，tracking_windows_completed=0不能单独证明Tracking实现失效；真实短episode和严格违规仍独立构成质量不达标证据。

梯度解释：下面原生算法配置保留train.clip_grad_norm=0.0这一UFO默认字段；实际运行由解析配置中的stabilized_v7包装，对Actor裁剪1.0、其它五优化器裁剪5.0，并记录grad_clip_limit/grad_norm_after_clip_l2。不能仅从原生默认字段断言本轮未裁剪；需使用本文实际梯度指标。

追加只读train参考可达性核验：当前557个允许的Tracking起点（official_rpo203、g1354）全部251帧均满足冻结位置范围±1mrad，没有发现“合法起点之后必然跟随越界目标”。完整train原始帧中official_rpo有149/9431位置越界帧，g1为0/8800；该总体统计包含原始帧，不等同于专家采样实际选择率。当前允许Tracking窗口均未覆盖这些越界帧，不能将其认定为本轮Tracking死亡主因。本轮未改写或清洗数据版本。

实际三采样rank启动候选池一致：原接触/速度筛选3514帧，新增动态筛选拒绝471帧（13.404%），保留3043；其中official_rpo1633、g11410，250步Tracking起点557（原651）。两种来源和Tracking模式均非空，sampling seeds与模式划分独立。100更新Train为trn_ca72106a1b7744cebf64，Run为run_72deb358859f4098bb80。

第一轮：派生v13身份、不改发布接口/数据/旧checkpoint；严格影子与物理joint range保持。第二轮：实际联合更新计数、六Adam、checkpoint哈希、GPU资源与100→600恢复；物理环境状态不支持精确恢复，恢复后重新RSI，不冒充连续物理轨迹。第三轮：真实时序/所有指标、首步训练与严格区别、原始trace与资源证据、预算结束关闭继续训练入口。

尚未校准实际电机转子/减速器惯量，尚未完成正常行走或sim2sim/真机验证；本轮未进行独立策略评估，因此没有评估录像。两段完整实验结果在下文展开。

## 100更新决策（在提交500观察前冻结）

工程流程已验证，不代表步态合格。末点训练4.44815步、严格影子3.35446步；区间出现先升后降。按预设100流程→500趋势预算仅继续有界观察，不因初始化改善放行长训或独立评估。

```json
{
  "stage": "fresh 100 updates",
  "decision": "continue only bounded 100\u2192600 observation",
  "workflow_verified": true,
  "walking_qualified": false,
  "independent_evaluation": false,
  "reason": "all finite, six Adam100 and SHA complete; dynamic RSI first-step training termination <=0.1%; early policy survival still low and nonmonotonic; no long-training release",
  "first_step_terms": 7,
  "first_step_samples": 39781,
  "first_step_fraction": 0.00017596339961288053,
  "strict_first_position_velocity_torque": [
    719,
    12,
    0
  ],
  "episode_last": 4.448153018951416,
  "strict_shadow_last": 3.3544561862945557,
  "physical_peak_rated_ratio": 3.9739580154418945
}
```

## 运行结果

以下统计来自原始TensorBoard事件（不限采样点），以实际联合更新为横轴；环境指标按采样轮记录，不能按每更新都有一个环境点理解。所有终止原因可能重叠。

### 阶段0→100：新增100真实联合更新

Train trn_ca72106a1b7744cebf64；Job job_b11f3efed76c433aa0a7；Run run_72deb358859f4098bb80；平台状态succeeded。
完整checkpoint与计数核验：
```json
{
  "step": 100,
  "counters": {
    "learner_joint_updates": 100,
    "sampling_rounds": 8,
    "online_transitions": 196608,
    "next_trigger": 206848
  },
  "six_adam_steps": {
    "forward_optimizer": 100,
    "backward_optimizer": 100,
    "discriminator_optimizer": 100,
    "actor_optimizer": 100,
    "critic_optimizer": 100,
    "aux_critic_optimizer": 100
  },
  "hashes_verified": true,
  "state_coverage": {
    "model": "restored",
    "optimizer": "restored",
    "scheduler": "not_applicable",
    "scaler": "not_applicable",
    "rng_per_rank": "restored",
    "replay": "restored",
    "expert_sampler": "restored",
    "z_pool": "restored",
    "rollout": "restored",
    "environment": "not_available"
  },
  "fingerprint": "sha256:29724c802c07204bdea91f0cc890492a202925ad5bd079547b3d163d6836d558",
  "execution": {
    "schema_version": "eb/training-binding/v1",
    "recipe_id": "rpo_bfm_ufo_agent_rlinf_mjlab_4gpu_viability_v13",
    "algorithm_id": "bfm_fb_cpr_aux",
    "provider_id": "ufo",
    "runtime_id": "rlinf",
    "simulator_id": "mjlab",
    "progress_unit": "learner_joint_update",
    "checkpoint_format": "ufo_agent_state_v2",
    "topology_id": "one_learner_three_samplers"
  }
}
```
每30秒doctor采样的整卡已用显存最大值MiB（包含驱动/Warp与常驻占用，并非连续测量的真正峰值）：
```json
{
  "0": 16726,
  "1": 4539,
  "2": 4539,
  "3": 4539
}
```
原始指标标签数149；联合更新记录100，范围1–100；逐行trace 196608条。
| 窗口 | 环境点数 | 训练episode_len步 | 严格影子步 | 严格影子秒 |
|---|---:|---:|---:|---:|
| 1–20 | 1 | 6.437744 | 5.223897 | 0.1044779 |
| 21–40 | 1 | 7.57978 | 5.570771 | 0.1114154 |
| 41–60 | 1 | 5.957417 | 4.177729 | 0.08355458 |
| 61–80 | 2 | 4.545186 | 3.233508 | 0.06467015 |
| 81–100 | 2 | 4.584533 | 3.358872 | 0.06717744 |

末点训练4.448153步、严格影子3.354456步。首步训练终止7/39781=0.017596%；严格首步唯一违规731例（1.837561%），位置/速度/力矩计数[719, 12, 0]。子步全局峰值31.79166rad/s、3.973958倍额定值；垂直净力峰值5705.269N。

完整Replay/初态/奖励/上下文与事故审计（报告passed仅表示核验一致，不表示行走达标）：
```json
{
  "status": "passed",
  "rows": 196608,
  "rounds": 8,
  "round_first": 1,
  "round_last": 8,
  "replay_count": 196608,
  "replay_capacity": 4000000,
  "fifo_evictions": 0,
  "requested_saturation_fraction": 0.34423695439877716,
  "applied_saturation_fraction": 0.05342168393342391,
  "strict_position_velocity_torque_counts": [
    46331,
    31044,
    0
  ],
  "training_position_velocity_torque_counts": [
    25605,
    8632,
    0
  ],
  "macro_contact_height_tilt_counts": [
    0,
    529,
    3408
  ],
  "boundary_max_radps": 31.496828079223633,
  "physics_max_radps": 31.791664123535156,
  "physics_max_rated_ratio": 3.9739580154418945,
  "boundary_max_event": {
    "round": 3,
    "row": 16370,
    "episode_id_run_local": 76124857346,
    "episode_step": 10,
    "joint": "right_ankle_roll_joint",
    "rated_radps": 8.0,
    "boundary_dq_radps": 31.496828079223633,
    "physics_signed_dq_radps": 31.791664123535156,
    "physics_abs_dq_radps": 31.791664123535156,
    "peak_substep_0_to_7": 6,
    "q_at_joint_peak_rad": 0.3759723901748657,
    "torque_at_joint_peak_Nm": -27.0,
    "target_at_joint_peak_rad": 0.22585168480873108,
    "requested_action": 0.9999989867210388,
    "applied_action": 0.45170336961746216,
    "foot_peak_vertical_net_force_N": [
      0.0,
      2286.631591796875
    ],
    "foot_vertical_net_force_integral_Ns": [
      0.0,
      18.666522979736328
    ],
    "global_peak_joint": "right_ankle_roll_joint",
    "feet_at_global_abs_dq_peak_origin_pose_wxyz_linear_angular": [
      [
        -56.76808166503906,
        -32.79681396484375,
        0.20839138329029083,
        0.32071006298065186,
        0.0788329467177391,
        0.6851244568824768,
        0.6492571830749512,
        -3.952538013458252,
        0.6056923866271973,
        -0.4743962287902832,
        6.040720462799072,
        9.727640151977539,
        -10.538413047790527
      ],
      [
        -55.84302520751953,
        -32.4177131652832,
        0.046254515647888184,
        0.8102214932441711,
        -0.07950437068939209,
        -0.09483896940946579,
        0.5729098320007324,
        1.6143274307250977,
        -0.29560089111328125,
        -0.08689689636230469,
        8.761136054992676,
        27.100027084350586,
        -1.1114184856414795
      ]
    ],
    "feet_vertical_net_force_at_global_peak_N": [
      0.0,
      785.4297485351562
    ],
    "strict_flags_pos_vel_torque": [
      0,
      1,
      0
    ],
    "training_flags_pos_vel_torque": [
      0,
      1,
      0
    ],
    "macro_contact_height_tilt": [
      0,
      1,
      0
    ]
  },
  "physics_max_event": {
    "round": 3,
    "row": 16370,
    "episode_id_run_local": 76124857346,
    "episode_step": 10,
    "joint": "right_ankle_roll_joint",
    "rated_radps": 8.0,
    "boundary_dq_radps": 31.496828079223633,
    "physics_signed_dq_radps": 31.791664123535156,
    "physics_abs_dq_radps": 31.791664123535156,
    "peak_substep_0_to_7": 6,
    "q_at_joint_peak_rad": 0.3759723901748657,
    "torque_at_joint_peak_Nm": -27.0,
    "target_at_joint_peak_rad": 0.22585168480873108,
    "requested_action": 0.9999989867210388,
    "applied_action": 0.45170336961746216,
    "foot_peak_vertical_net_force_N": [
      0.0,
      2286.631591796875
    ],
    "foot_vertical_net_force_integral_Ns": [
      0.0,
      18.666522979736328
    ],
    "global_peak_joint": "right_ankle_roll_joint",
    "feet_at_global_abs_dq_peak_origin_pose_wxyz_linear_angular": [
      [
        -56.76808166503906,
        -32.79681396484375,
        0.20839138329029083,
        0.32071006298065186,
        0.0788329467177391,
        0.6851244568824768,
        0.6492571830749512,
        -3.952538013458252,
        0.6056923866271973,
        -0.4743962287902832,
        6.040720462799072,
        9.727640151977539,
        -10.538413047790527
      ],
      [
        -55.84302520751953,
        -32.4177131652832,
        0.046254515647888184,
        0.8102214932441711,
        -0.07950437068939209,
        -0.09483896940946579,
        0.5729098320007324,
        1.6143274307250977,
        -0.29560089111328125,
        -0.08689689636230469,
        8.761136054992676,
        27.100027084350586,
        -1.1114184856414795
      ]
    ],
    "feet_vertical_net_force_at_global_peak_N": [
      0.0,
      785.4297485351562
    ],
    "strict_flags_pos_vel_torque": [
      0,
      1,
      0
    ],
    "training_flags_pos_vel_torque": [
      0,
      1,
      0
    ],
    "macro_contact_height_tilt": [
      0,
      1,
      0
    ]
  },
  "normalized_max_event": {
    "round": 3,
    "row": 16370,
    "episode_id_run_local": 76124857346,
    "episode_step": 10,
    "joint": "right_ankle_roll_joint",
    "rated_radps": 8.0,
    "boundary_dq_radps": 31.496828079223633,
    "physics_signed_dq_radps": 31.791664123535156,
    "physics_abs_dq_radps": 31.791664123535156,
    "peak_substep_0_to_7": 6,
    "q_at_joint_peak_rad": 0.3759723901748657,
    "torque_at_joint_peak_Nm": -27.0,
    "target_at_joint_peak_rad": 0.22585168480873108,
    "requested_action": 0.9999989867210388,
    "applied_action": 0.45170336961746216,
    "foot_peak_vertical_net_force_N": [
      0.0,
      2286.631591796875
    ],
    "foot_vertical_net_force_integral_Ns": [
      0.0,
      18.666522979736328
    ],
    "global_peak_joint": "right_ankle_roll_joint",
    "feet_at_global_abs_dq_peak_origin_pose_wxyz_linear_angular": [
      [
        -56.76808166503906,
        -32.79681396484375,
        0.20839138329029083,
        0.32071006298065186,
        0.0788329467177391,
        0.6851244568824768,
        0.6492571830749512,
        -3.952538013458252,
        0.6056923866271973,
        -0.4743962287902832,
        6.040720462799072,
        9.727640151977539,
        -10.538413047790527
      ],
      [
        -55.84302520751953,
        -32.4177131652832,
        0.046254515647888184,
        0.8102214932441711,
        -0.07950437068939209,
        -0.09483896940946579,
        0.5729098320007324,
        1.6143274307250977,
        -0.29560089111328125,
        -0.08689689636230469,
        8.761136054992676,
        27.100027084350586,
        -1.1114184856414795
      ]
    ],
    "feet_vertical_net_force_at_global_peak_N": [
      0.0,
      785.4297485351562
    ],
    "strict_flags_pos_vel_torque": [
      0,
      1,
      0
    ],
    "training_flags_pos_vel_torque": [
      0,
      1,
      0
    ],
    "macro_contact_height_tilt": [
      0,
      1,
      0
    ]
  },
  "foot_vertical_net_force_peak_N": 5705.2685546875,
  "first_step_foot_vertical_net_force_N": {
    "n": 39781,
    "mean": 41.987083435058594,
    "median": 0.0,
    "p95": 318.3429260253906,
    "p99": 554.4788696289053,
    "max": 1527.121826171875
  },
  "first_step_foot_absolute_vertical_net_force_integral_Ns": {
    "n": 39781,
    "mean": 0.27409324049949646,
    "median": 0.0,
    "p95": 2.2927560806274414,
    "p99": 4.125787544250478,
    "max": 10.489026069641113
  },
  "reset_lift_m": {
    "n": 39781,
    "mean": 0.0013989851577207446,
    "median": 0.0,
    "p95": 0.013831143267452717,
    "p99": 0.019589802250266075,
    "max": 0.021992022171616554
  },
  "resets": 39781,
  "first_step_terminated": 7,
  "first_step_termination_fraction": 0.00017596339961288053,
  "first_step_strict_any": 731,
  "first_step_strict_any_fraction": 0.01837560644528795,
  "first_step_strict_counts": [
    719,
    12,
    0
  ],
  "first_step_macro_counts": [
    0,
    0,
    0
  ],
  "raw_velocity_penalty_mean": 0.02498613779122631,
  "weighted_velocity_penalty_mean": -0.6246534447806577,
  "strict_velocity_substep_joint_exposure_fraction": 0.0030568440755208335,
  "velocity_reward_coverage_miss_count": 0,
  "latched_velocity_terms_without_boundary_violation": 16690,
  "reset_q_preservation_max_error_rad": 8.821487429422348e-08,
  "reset_dq_preservation_max_error_radps": 0.0,
  "decoder_max_error_rad": 2.1219253509840996e-07,
  "infinite_plane_trace_margin_sentinel_m": 1000000.0,
  "ground_outside_count": 0,
  "mixed_nonperiodic_z_equal_pairs": 78253,
  "mixed_nonperiodic_cosine": {
    "n": 78253,
    "mean": 1.0,
    "median": 1.0,
    "p95": 1.0000001192092896,
    "p99": 1.0000001192092896,
    "max": 1.0000001192092896
  },
  "tracking_unexpected_refreshes": 0,
  "tracking_phase_mismatches": 0,
  "tracking_window_stable_pairs": 78574,
  "tracking_adjacent_phase_cosine": {
    "n": 78574,
    "mean": 0.9975624084472656,
    "median": 0.9983745813369751,
    "p95": 0.9999945163726807,
    "p99": 0.9999973177909851,
    "max": 0.9999991059303284
  },
  "tracking_window_renewals": 0,
  "terminal_following_new_episode_pairs": 33637,
  "tracking_envs_per_rank": [
    1024,
    1024,
    1024
  ],
  "mixed_envs_per_rank": [
    1024,
    1024,
    1024
  ],
  "completed_episode_len": {
    "n": 35052,
    "mean": 5.2789284491612465,
    "median": 4.0,
    "p95": 12.0,
    "p99": 16.0,
    "max": 31.0
  },
  "completed_strict_shadow_len": {
    "n": 35052,
    "mean": 3.8482540225950017,
    "median": 3.0,
    "p95": 9.0,
    "p99": 12.0,
    "max": 22.0
  },
  "completed_tracking_shadow_len": {
    "n": 17311,
    "mean": 3.8374443995147596,
    "median": 3.0,
    "p95": 9.0,
    "p99": 12.0,
    "max": 18.0
  },
  "completed_mixed_shadow_len": {
    "n": 17741,
    "mean": 3.858801645904966,
    "median": 3.0,
    "p95": 9.0,
    "p99": 12.0,
    "max": 22.0
  },
  "unfinished_train_episodes_right_censored": 4729,
  "unfinished_strict_shadow_right_censored": 3366,
  "unfinished_train_observed_lengths": {
    "n": 4729,
    "mean": 2.4468175089871007,
    "median": 2.0,
    "p95": 5.0,
    "p99": 8.0,
    "max": 17.0
  },
  "RSI_source_by_rollout_mode": {
    "official_rpo/mixed": {
      "rows": 50861,
      "resets": 10069,
      "first_terms": 5,
      "strict_velocity_rows": 7187,
      "completed_episode_len": {
        "n": 8876,
        "mean": 5.379112212708427,
        "median": 4.0,
        "p95": 12.0,
        "p99": 16.0,
        "max": 27.0
      },
      "completed_strict_shadow_len": {
        "n": 8876,
        "mean": 4.108269490761605,
        "median": 3.0,
        "p95": 9.0,
        "p99": 12.0,
        "max": 20.0
      },
      "first_term_fraction": 0.0004965736418710895,
      "strict_velocity_substep_joint_exposure_fraction": 0.0028132942042378077,
      "raw_velocity_penalty_mean": 0.02427301935439793
    },
    "official_rpo/tracking": {
      "rows": 51507,
      "resets": 9816,
      "first_terms": 0,
      "strict_velocity_rows": 8485,
      "completed_episode_len": {
        "n": 8802,
        "mean": 5.620768007271074,
        "median": 4.0,
        "p95": 12.0,
        "p99": 16.0,
        "max": 24.0
      },
      "completed_strict_shadow_len": {
        "n": 8802,
        "mean": 4.104294478527607,
        "median": 3.0,
        "p95": 9.0,
        "p99": 12.0,
        "max": 18.0
      },
      "first_term_fraction": 0.0,
      "strict_velocity_substep_joint_exposure_fraction": 0.0035134523716067296,
      "raw_velocity_penalty_mean": 0.02752786589286313
    },
    "g1/mixed": {
      "rows": 47443,
      "resets": 9982,
      "first_terms": 1,
      "strict_velocity_rows": 7015,
      "completed_episode_len": {
        "n": 8865,
        "mean": 5.035758601240834,
        "median": 4.0,
        "p95": 12.0,
        "p99": 16.0,
        "max": 31.0
      },
      "completed_strict_shadow_len": {
        "n": 8865,
        "mean": 3.6090242526790752,
        "median": 3.0,
        "p95": 9.0,
        "p99": 13.0,
        "max": 22.0
      },
      "first_term_fraction": 0.00010018032458425165,
      "strict_velocity_substep_joint_exposure_fraction": 0.002566581041414457,
      "raw_velocity_penalty_mean": 0.022334205012014625
    },
    "g1/tracking": {
      "rows": 46797,
      "resets": 9914,
      "first_terms": 1,
      "strict_velocity_rows": 8357,
      "completed_episode_len": {
        "n": 8509,
        "mean": 5.074156775179222,
        "median": 4.0,
        "p95": 12.0,
        "p99": 15.0,
        "max": 19.0
      },
      "completed_strict_shadow_len": {
        "n": 8509,
        "mean": 3.5614055705723353,
        "median": 3.0,
        "p95": 9.0,
        "p99": 12.0,
        "max": 18.0
      },
      "first_term_fraction": 0.00010086746015735324,
      "strict_velocity_substep_joint_exposure_fraction": 0.0033160105952536904,
      "raw_velocity_penalty_mean": 0.025652182057114913
    }
  },
  "per_joint_physical_diagnostics": {
    "left_thigh_yaw_joint": {
      "rated_radps": 25.0,
      "max_physics_abs_radps": 23.534093856811523,
      "max_boundary_abs_radps": 21.248010635375977,
      "max_rated_ratio": 0.9413637542724609,
      "substep_exposure_fraction": 0.0
    },
    "right_thigh_yaw_joint": {
      "rated_radps": 25.0,
      "max_physics_abs_radps": 23.899187088012695,
      "max_boundary_abs_radps": 21.452674865722656,
      "max_rated_ratio": 0.9559674835205079,
      "substep_exposure_fraction": 0.0
    },
    "torso_joint": {
      "rated_radps": 25.0,
      "max_physics_abs_radps": 23.25999641418457,
      "max_boundary_abs_radps": 22.255338668823242,
      "max_rated_ratio": 0.9303998565673828,
      "substep_exposure_fraction": 0.0
    },
    "left_thigh_roll_joint": {
      "rated_radps": 25.0,
      "max_physics_abs_radps": 26.690567016601562,
      "max_boundary_abs_radps": 21.685338973999023,
      "max_rated_ratio": 1.0676226806640625,
      "substep_exposure_fraction": 3.1789143880208335e-06
    },
    "right_thigh_roll_joint": {
      "rated_radps": 25.0,
      "max_physics_abs_radps": 25.5331974029541,
      "max_boundary_abs_radps": 24.66370964050293,
      "max_rated_ratio": 1.021327896118164,
      "substep_exposure_fraction": 1.9073486328125e-06
    },
    "left_arm_pitch_joint": {
      "rated_radps": 8.0,
      "max_physics_abs_radps": 13.103690147399902,
      "max_boundary_abs_radps": 12.94641399383545,
      "max_rated_ratio": 1.6379612684249878,
      "substep_exposure_fraction": 0.00846099853515625
    },
    "right_arm_pitch_joint": {
      "rated_radps": 8.0,
      "max_physics_abs_radps": 13.725658416748047,
      "max_boundary_abs_radps": 13.725658416748047,
      "max_rated_ratio": 1.7157073020935059,
      "substep_exposure_fraction": 0.0041904449462890625
    },
    "left_thigh_pitch_joint": {
      "rated_radps": 25.0,
      "max_physics_abs_radps": 25.04235076904297,
      "max_boundary_abs_radps": 24.994203567504883,
      "max_rated_ratio": 1.0016940307617188,
      "substep_exposure_fraction": 6.357828776041666e-07
    },
    "right_thigh_pitch_joint": {
      "rated_radps": 25.0,
      "max_physics_abs_radps": 23.814321517944336,
      "max_boundary_abs_radps": 23.0782470703125,
      "max_rated_ratio": 0.9525728607177735,
      "substep_exposure_fraction": 0.0
    },
    "left_arm_roll_joint": {
      "rated_radps": 8.0,
      "max_physics_abs_radps": 13.58974552154541,
      "max_boundary_abs_radps": 13.58974552154541,
      "max_rated_ratio": 1.6987181901931763,
      "substep_exposure_fraction": 0.0016422271728515625
    },
    "right_arm_roll_joint": {
      "rated_radps": 8.0,
      "max_physics_abs_radps": 14.6273832321167,
      "max_boundary_abs_radps": 14.6273832321167,
      "max_rated_ratio": 1.8284229040145874,
      "substep_exposure_fraction": 0.008412679036458334
    },
    "left_knee_joint": {
      "rated_radps": 25.0,
      "max_physics_abs_radps": 25.186786651611328,
      "max_boundary_abs_radps": 24.01140785217285,
      "max_rated_ratio": 1.007471466064453,
      "substep_exposure_fraction": 1.9073486328125e-06
    },
    "right_knee_joint": {
      "rated_radps": 25.0,
      "max_physics_abs_radps": 22.955734252929688,
      "max_boundary_abs_radps": 21.72886848449707,
      "max_rated_ratio": 0.9182293701171875,
      "substep_exposure_fraction": 0.0
    },
    "left_arm_yaw_joint": {
      "rated_radps": 8.0,
      "max_physics_abs_radps": 6.711154460906982,
      "max_boundary_abs_radps": 5.745726585388184,
      "max_rated_ratio": 0.8388943076133728,
      "substep_exposure_fraction": 0.0
    },
    "right_arm_yaw_joint": {
      "rated_radps": 8.0,
      "max_physics_abs_radps": 6.4934797286987305,
      "max_boundary_abs_radps": 5.492526531219482,
      "max_rated_ratio": 0.8116849660873413,
      "substep_exposure_fraction": 0.0
    },
    "left_ankle_pitch_joint": {
      "rated_radps": 8.0,
      "max_physics_abs_radps": 25.800037384033203,
      "max_boundary_abs_radps": 25.800037384033203,
      "max_rated_ratio": 3.2250046730041504,
      "substep_exposure_fraction": 0.017417271931966145
    },
    "right_ankle_pitch_joint": {
      "rated_radps": 8.0,
      "max_physics_abs_radps": 26.748966217041016,
      "max_boundary_abs_radps": 25.407196044921875,
      "max_rated_ratio": 3.343620777130127,
      "substep_exposure_fraction": 0.02315839131673177
    },
    "left_elbow_pitch_joint": {
      "rated_radps": 8.0,
      "max_physics_abs_radps": 7.370490550994873,
      "max_boundary_abs_radps": 6.8982343673706055,
      "max_rated_ratio": 0.9213113188743591,
      "substep_exposure_fraction": 0.0
    },
    "right_elbow_pitch_joint": {
      "rated_radps": 8.0,
      "max_physics_abs_radps": 7.136786460876465,
      "max_boundary_abs_radps": 6.966986179351807,
      "max_rated_ratio": 0.8920983076095581,
      "substep_exposure_fraction": 0.0
    },
    "left_ankle_roll_joint": {
      "rated_radps": 8.0,
      "max_physics_abs_radps": 28.296539306640625,
      "max_boundary_abs_radps": 26.03598976135254,
      "max_rated_ratio": 3.537067413330078,
      "substep_exposure_fraction": 0.003983815511067708
    },
    "right_ankle_roll_joint": {
      "rated_radps": 8.0,
      "max_physics_abs_radps": 31.791664123535156,
      "max_boundary_abs_radps": 31.496828079223633,
      "max_rated_ratio": 3.9739580154418945,
      "substep_exposure_fraction": 0.0030339558919270835
    },
    "left_elbow_yaw_joint": {
      "rated_radps": 8.0,
      "max_physics_abs_radps": 4.226924896240234,
      "max_boundary_abs_radps": 3.7985894680023193,
      "max_rated_ratio": 0.5283656120300293,
      "substep_exposure_fraction": 0.0
    },
    "right_elbow_yaw_joint": {
      "rated_radps": 8.0,
      "max_physics_abs_radps": 4.222880840301514,
      "max_boundary_abs_radps": 3.7863526344299316,
      "max_rated_ratio": 0.5278601050376892,
      "substep_exposure_fraction": 0.0
    }
  },
  "right_ankle_pitch_physics_peak_event": {
    "round": 6,
    "row": 19739,
    "episode_id_run_local": 2264951076867,
    "episode_step": 3,
    "joint": "right_ankle_pitch_joint",
    "rated_radps": 8.0,
    "boundary_dq_radps": 23.141111373901367,
    "physics_signed_dq_radps": 26.748966217041016,
    "physics_abs_dq_radps": 26.748966217041016,
    "peak_substep_0_to_7": 3,
    "q_at_joint_peak_rad": 0.05683158338069916,
    "torque_at_joint_peak_Nm": -27.0,
    "target_at_joint_peak_rad": -0.15082736313343048,
    "requested_action": -0.1833881288766861,
    "applied_action": -0.2513789236545563,
    "foot_peak_vertical_net_force_N": [
      0.0,
      2506.811767578125
    ],
    "foot_vertical_net_force_integral_Ns": [
      0.0,
      22.125606536865234
    ],
    "global_peak_joint": "right_ankle_pitch_joint",
    "feet_at_global_abs_dq_peak_origin_pose_wxyz_linear_angular": [
      [
        -14.768160820007324,
        -47.06074523925781,
        0.06488564610481262,
        0.908978283405304,
        0.21620896458625793,
        0.14885374903678894,
        0.3238131105899811,
        -0.2356863021850586,
        1.5154796838760376,
        0.3873763084411621,
        10.401628494262695,
        7.963207244873047,
        12.993744850158691
      ],
      [
        -14.091975212097168,
        -47.34562683105469,
        0.06273321807384491,
        0.9560373425483704,
        0.019294921308755875,
        -0.2697369158267975,
        0.1134122908115387,
        2.074376106262207,
        -0.8701839447021484,
        -0.3208894729614258,
        12.078531265258789,
        28.749982833862305,
        -1.2409604787826538
      ]
    ],
    "feet_vertical_net_force_at_global_peak_N": [
      0.0,
      947.8239135742188
    ],
    "strict_flags_pos_vel_torque": [
      1,
      1,
      0
    ],
    "training_flags_pos_vel_torque": [
      1,
      1,
      0
    ],
    "macro_contact_height_tilt": [
      0,
      0,
      0
    ]
  },
  "right_ankle_roll_physics_peak_event": {
    "round": 3,
    "row": 16370,
    "episode_id_run_local": 76124857346,
    "episode_step": 10,
    "joint": "right_ankle_roll_joint",
    "rated_radps": 8.0,
    "boundary_dq_radps": 31.496828079223633,
    "physics_signed_dq_radps": 31.791664123535156,
    "physics_abs_dq_radps": 31.791664123535156,
    "peak_substep_0_to_7": 6,
    "q_at_joint_peak_rad": 0.3759723901748657,
    "torque_at_joint_peak_Nm": -27.0,
    "target_at_joint_peak_rad": 0.22585168480873108,
    "requested_action": 0.9999989867210388,
    "applied_action": 0.45170336961746216,
    "foot_peak_vertical_net_force_N": [
      0.0,
      2286.631591796875
    ],
    "foot_vertical_net_force_integral_Ns": [
      0.0,
      18.666522979736328
    ],
    "global_peak_joint": "right_ankle_roll_joint",
    "feet_at_global_abs_dq_peak_origin_pose_wxyz_linear_angular": [
      [
        -56.76808166503906,
        -32.79681396484375,
        0.20839138329029083,
        0.32071006298065186,
        0.0788329467177391,
        0.6851244568824768,
        0.6492571830749512,
        -3.952538013458252,
        0.6056923866271973,
        -0.4743962287902832,
        6.040720462799072,
        9.727640151977539,
        -10.538413047790527
      ],
      [
        -55.84302520751953,
        -32.4177131652832,
        0.046254515647888184,
        0.8102214932441711,
        -0.07950437068939209,
        -0.09483896940946579,
        0.5729098320007324,
        1.6143274307250977,
        -0.29560089111328125,
        -0.08689689636230469,
        8.761136054992676,
        27.100027084350586,
        -1.1114184856414795
      ]
    ],
    "feet_vertical_net_force_at_global_peak_N": [
      0.0,
      785.4297485351562
    ],
    "strict_flags_pos_vel_torque": [
      0,
      1,
      0
    ],
    "training_flags_pos_vel_torque": [
      0,
      1,
      0
    ],
    "macro_contact_height_tilt": [
      0,
      1,
      0
    ]
  },
  "semantics": "Run-local episode IDs; original physical thresholds shadow; interval mean of exactly 8 completed substeps; vertical NET force not contact point normal impulse; unfinished trajectories are right-censored, excluded from completed means"
}
```

所有实际指标的全段统计与窗口均值：

| tag | n | first | last | mean | min | max | median | p95 | 按更新窗口均值(n) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| env/episode_len | 7 | 6.437744 | 4.448153 | 5.462054 | 4.406586 | 7.57978 | 4.720913 | 7.237169 | 1–20: 6.437744(1); 21–40: 7.57978(1); 41–60: 5.957417(1); 61–80: 4.545186(2); 81–100: 4.584533(2) |
| env/num_trajectories | 7 | 1791 | 5902 | 4927.143 | 1791 | 6679 | 5002 | 6445.9 | 1–20: 1791(1); 21–40: 4995(1); 41–60: 5002(1); 61–80: 5662.5(2); 81–100: 5688.5(2) |
| env/return | 7 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1–20: 0(1); 21–40: 0(1); 41–60: 0(1); 61–80: 0(2); 81–100: 0(2) |
| env/reward | 7 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1–20: 0(1); 21–40: 0(1); 41–60: 0(1); 61–80: 0(2); 81–100: 0(2) |
| env/success_once | 7 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1–20: 0(1); 21–40: 0(1); 41–60: 0(1); 61–80: 0(2); 81–100: 0(2) |
| rollout/ufo/received_transitions | 7 | 24576 | 24576 | 24576 | 24576 | 24576 | 24576 | 24576 | 1–20: 24576(1); 21–40: 24576(1); 41–60: 24576(1); 61–80: 24576(2); 81–100: 24576(2) |
| time/actor_training | 7 | 14.33612 | 4.089915 | 11.3435 | 4.089915 | 14.33612 | 12.20354 | 13.70988 | 1–20: 14.33612(1); 21–40: 12.22483(1); 41–60: 12.15133(1); 61–80: 12.17684(2); 81–100: 8.169278(2) |
| time/cal_adv_and_returns | 7 | 0.007490396 | 0.011415 | 0.01415614 | 0.007490396 | 0.02260208 | 0.01158166 | 0.02194591 | 1–20: 0.007490396(1); 21–40: 0.02041483(1); 41–60: 0.02260208(1); 61–80: 0.01289177(2); 81–100: 0.01140106(2) |
| time/env/compute_bootstrap_rewards | 7 | 0.0002651631 | 0.0002763539 | 0.0002636532 | 0.0002271209 | 0.0002870951 | 0.000268142 | 0.0002851208 | 1–20: 0.0002651631(1); 21–40: 0.0002271209(1); 41–60: 0.0002805141(1); 61–80: 0.0002546625(2); 81–100: 0.0002817245(2) |
| time/env/env/bootstrap_step | 7 | 0.0002589531 | 0.000310546 | 0.000289916 | 0.000252893 | 0.000343537 | 0.0002814441 | 0.0003378937 | 1–20: 0.0002589531(1); 21–40: 0.000257313(1); 41–60: 0.000252893(1); 61–80: 0.0003030851(2); 81–100: 0.0003270415(2) |
| time/env/env/send_rollout_trajectories | 7 | 0.4069368 | 0.4757149 | 0.4266658 | 0.4053223 | 0.4757149 | 0.4112972 | 0.4665755 | 1–20: 0.4069368(1); 21–40: 0.4112972(1); 41–60: 0.4053223(1); 61–80: 0.4402246(2); 81–100: 0.4413275(2) |
| time/env/env_interact_step | 7 | 1.064192 | 1.055543 | 1.071072 | 1.023087 | 1.110249 | 1.064192 | 1.10933 | 1–20: 1.064192(1); 21–40: 1.023087(1); 41–60: 1.107184(1); 61–80: 1.086822(2); 81–100: 1.064698(2) |
| time/env/interact | 7 | 2.386123 | 2.810971 | 2.806255 | 2.386123 | 3.031006 | 2.810971 | 3.023882 | 1–20: 2.386123(1); 21–40: 2.738041(1); 41–60: 3.031006(1); 61–80: 2.835191(2); 81–100: 2.909116(2) |
| time/env/run_interact_once | 7 | 2.386096 | 2.810945 | 2.806226 | 2.386096 | 3.030978 | 2.810945 | 3.023852 | 1–20: 2.386096(1); 21–40: 2.738013(1); 41–60: 3.030978(1); 61–80: 2.835164(2); 81–100: 2.909085(2) |
| time/generate_rollouts | 7 | 2.742332 | 3.137794 | 3.118522 | 2.742332 | 3.380432 | 3.137794 | 3.373301 | 1–20: 2.742332(1); 21–40: 3.060817(1); 41–60: 3.380432(1); 61–80: 3.122278(2); 81–100: 3.200759(2) |
| time/rollout/generate_one_epoch | 7 | 1.977208 | 2.420584 | 2.409867 | 1.977208 | 2.653846 | 2.420584 | 2.642848 | 1–20: 1.977208(1); 21–40: 2.34095(1); 41–60: 2.653846(1); 61–80: 2.429649(2); 81–100: 2.518885(2) |
| time/rollout/predict | 7 | 0.2417237 | 0.4762539 | 0.4188593 | 0.2417237 | 0.5824991 | 0.4331661 | 0.5506256 | 1–20: 0.2417237(1); 21–40: 0.4214391(1); 41–60: 0.4331661(1); 61–80: 0.4578639(2); 81–100: 0.459979(2) |
| time/rollout/rollout/generate | 7 | 1.977334 | 2.420718 | 2.410148 | 1.977334 | 2.654498 | 2.420718 | 2.643342 | 1–20: 1.977334(1); 21–40: 2.341071(1); 41–60: 2.654498(1); 61–80: 2.430051(2); 81–100: 2.519014(2) |
| time/rollout/sync_model_from_actor | 7 | 13.49516 | 13.25337 | 13.15153 | 12.87032 | 13.49516 | 13.1225 | 13.42262 | 1–20: 13.49516(1); 21–40: 13.1225(1); 41–60: 12.87032(1); 61–80: 13.12142(2); 81–100: 13.16495(2) |
| time/step | 7 | 34.32718 | 23.97355 | 31.17838 | 23.97355 | 34.32718 | 32.1064 | 33.68333 | 1–20: 34.32718(1); 21–40: 32.18101(1); 41–60: 31.8634(1); 61–80: 31.89857(2); 81–100: 28.03998(2) |
| time/sync_weights | 7 | 17.24062 | 16.73381 | 16.70163 | 16.30853 | 17.24062 | 16.59907 | 17.13075 | 1–20: 17.24062(1); 21–40: 16.87439(1); 41–60: 16.30853(1); 61–80: 16.586(2); 81–100: 16.65795(2) |
| train/ufo/B | 100 | 0.01805907 | -0.004449195 | -0.007598862 | -0.01820472 | 0.01805907 | -0.008992855 | 0.009664461 | 1–20: 0.004213316(20); 21–40: -0.01527432(20); 41–60: -0.01165015(20); 61–80: -0.008496268(20); 81–100: -0.006786891(20) |
| train/ufo/B_norm | 100 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 1–20: 16(20); 21–40: 16(20); 41–60: 16(20); 61–80: 16(20); 81–100: 16(20) |
| train/ufo/F1 | 100 | 0.01306153 | 0.00956913 | 0.01149616 | -0.02159296 | 0.08691171 | 0.0150953 | 0.02663117 | 1–20: 0.009798634(20); 21–40: -0.006383062(20); 41–60: 0.01486944(20); 61–80: 0.02231708(20); 81–100: 0.01687872(20) |
| train/ufo/M1 | 100 | -5.502787 | 3.527194 | 3.332168 | -43.28454 | 117.8768 | 2.844152 | 5.534752 | 1–20: 5.568952(20); 21–40: 2.329749(20); 41–60: 2.936345(20); 61–80: 2.940418(20); 81–100: 2.885376(20) |
| train/ufo/Q1 | 100 | -0.245254 | -2.419637 | -3.057437 | -9.628309 | 9.593767 | -3.288385 | -1.37716 | 1–20: -2.045551(20); 21–40: -3.765883(20); 41–60: -3.67152(20); 61–80: -3.172535(20); 81–100: -2.631696(20) |
| train/ufo/Q_aux | 100 | -24.80069 | -3.322207 | -4.069649 | -24.80069 | 14.17292 | -4.100171 | -1.892527 | 1–20: -4.072126(20); 21–40: -4.78412(20); 41–60: -4.296395(20); 61–80: -3.712199(20); 81–100: -3.483404(20) |
| train/ufo/Q_discriminator | 100 | -23.2566 | -2.374106 | -3.635707 | -23.2566 | 6.459823 | -3.468828 | -2.469438 | 1–20: -4.295794(20); 21–40: -4.163694(20); 41–60: -3.853301(20); 61–80: -3.195628(20); 81–100: -2.67012(20) |
| train/ufo/Q_fb | 100 | -40.65714 | 12.36071 | 3.264441 | -40.65714 | 14.20016 | 5.798681 | 12.40169 | 1–20: -10.18732(20); 21–40: 0.8711442(20); 41–60: 5.753917(20); 61–80: 8.321706(20); 81–100: 11.56276(20) |
| train/ufo/action_abs_mean | 7 | 0.5505452 | 0.9235233 | 0.8601631 | 0.5505452 | 0.9235233 | 0.9073043 | 0.9215509 | 1–20: 0.5505452(1); 21–40: 0.9016314(1); 41–60: 0.9068106(1); 61–80: 0.9121265(2); 81–100: 0.9189509(2) |
| train/ufo/action_saturation_fraction | 7 | 0.03108547 | 0.4712639 | 0.3920001 | 0.03108547 | 0.4712639 | 0.4460732 | 0.468839 | 1–20: 0.03108547(1); 21–40: 0.4321501(1); 41–60: 0.4421298(1); 61–80: 0.454627(2); 81–100: 0.4646907(2) |
| train/ufo/actor_loss | 100 | 122.0631 | -9.24786 | 0.5652116 | -10.61599 | 122.0631 | -2.447144 | 12.98079 | 1–20: 16.56253(20); 21–40: 2.104853(20); 41–60: -2.448724(20); 61–80: -5.118794(20); 81–100: -8.273802(20) |
| train/ufo/actor_optimizer/grad_clip_limit | 100 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1–20: 1(20); 21–40: 1(20); 41–60: 1(20); 61–80: 1(20); 81–100: 1(20) |
| train/ufo/actor_optimizer/grad_norm_after_clip_l2 | 100 | 0.9999999 | 0.9999995 | 0.9989896 | 0.9508629 | 0.9999999 | 0.9999994 | 0.9999997 | 1–20: 0.9991118(20); 21–40: 0.9958377(20); 41–60: 0.9999994(20); 61–80: 0.9999995(20); 81–100: 0.9999995(20) |
| train/ufo/actor_optimizer/grad_norm_l2 | 100 | 22.62985 | 2.032677 | 1.942973 | 0.9508629 | 22.62985 | 1.694807 | 2.551963 | 1–20: 3.175492(20); 21–40: 1.078082(20); 41–60: 1.582091(20); 61–80: 1.817261(20); 81–100: 2.061941(20) |
| train/ufo/actor_optimizer/lr | 100 | 0.0003 | 0.0003 | 0.0003 | 0.0003 | 0.0003 | 0.0003 | 0.0003 | 1–20: 0.0003(20); 21–40: 0.0003(20); 41–60: 0.0003(20); 61–80: 0.0003(20); 81–100: 0.0003(20) |
| train/ufo/actor_optimizer/parameter_delta_l2 | 100 | 1.682914 | 0.3617505 | 0.44582 | 0.3278023 | 1.682914 | 0.3736398 | 0.7165658 | 1–20: 0.6916366(20); 21–40: 0.4454433(20); 41–60: 0.3611932(20); 61–80: 0.3637072(20); 81–100: 0.3671198(20) |
| train/ufo/actor_reg_weight_applied | 100 | 49.07394 | 16.81265 | 13.79381 | 8.801826 | 49.07394 | 12.66098 | 17.86701 | 1–20: 17.37863(20); 21–40: 9.794816(20); 41–60: 11.89043(20); 61–80: 13.71368(20); 81–100: 16.19148(20) |
| train/ufo/actor_reg_weight_raw | 100 | 49.07394 | 16.81265 | 13.79381 | 8.801826 | 49.07394 | 12.66098 | 17.86701 | 1–20: 17.37863(20); 21–40: 9.794816(20); 41–60: 11.89043(20); 61–80: 13.71368(20); 81–100: 16.19148(20) |
| train/ufo/applied_action_abs_mean | 7 | 0.3085989 | 0.5261884 | 0.4457512 | 0.3085989 | 0.5261884 | 0.4540272 | 0.5234526 | 1–20: 0.3085989(1); 21–40: 0.4072951(1); 41–60: 0.4311453(1); 61–80: 0.4649811(2); 81–100: 0.5216286(2) |
| train/ufo/applied_action_saturation_fraction | 7 | 0.001119862 | 0.102509 | 0.06101115 | 0.001119862 | 0.102509 | 0.06829038 | 0.1011354 | 1–20: 0.001119862(1); 21–40: 0.0316569(1); 41–60: 0.0539197(1); 61–80: 0.06997105(2); 81–100: 0.1002197(2) |
| train/ufo/auxQ1 | 100 | 0.07173939 | -3.356109 | -3.705961 | -23.7133 | 14.56758 | -3.836617 | -0.2413023 | 1–20: -3.181737(20); 21–40: -4.446181(20); 41–60: -3.978925(20); 61–80: -3.605212(20); 81–100: -3.317752(20) |
| train/ufo/aux_critic_loss | 100 | 5.761865 | 0.9573032 | 13.04377 | 0.9573032 | 508.4125 | 1.824704 | 20.86257 | 1–20: 58.07503(20); 21–40: 2.439067(20); 41–60: 1.855985(20); 61–80: 1.442018(20); 81–100: 1.406739(20) |
| train/ufo/aux_critic_optimizer/grad_clip_limit | 100 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 1–20: 5(20); 21–40: 5(20); 41–60: 5(20); 61–80: 5(20); 81–100: 5(20) |
| train/ufo/aux_critic_optimizer/grad_norm_after_clip_l2 | 100 | 5 | 5 | 4.982048 | 3.204867 | 5.000001 | 5 | 5 | 1–20: 5(20); 21–40: 5(20); 41–60: 4.910243(20); 61–80: 5(20); 81–100: 5(20) |
| train/ufo/aux_critic_optimizer/grad_norm_l2 | 100 | 68.12362 | 14.33177 | 65.22885 | 3.204867 | 1250.883 | 19.16827 | 228.172 | 1–20: 257.067(20); 21–40: 17.75226(20); 41–60: 19.34321(20); 61–80: 16.75227(20); 81–100: 15.2295(20) |
| train/ufo/aux_critic_optimizer/lr | 100 | 0.0003 | 0.0003 | 0.0003 | 0.0003 | 0.0003 | 0.0003 | 0.0003 | 1–20: 0.0003(20); 21–40: 0.0003(20); 41–60: 0.0003(20); 61–80: 0.0003(20); 81–100: 0.0003(20) |
| train/ufo/aux_critic_optimizer/parameter_delta_l2 | 100 | 3.46055 | 0.2889568 | 0.5065722 | 0.2310749 | 3.46055 | 0.358887 | 1.549721 | 1–20: 1.179576(20); 21–40: 0.4233245(20); 41–60: 0.3455576(20); 61–80: 0.3094843(20); 81–100: 0.2749185(20) |
| train/ufo/aux_rew/limits_dof_pos | 100 | 0.008656619 | 0.05243167 | 0.03046518 | 0.007654512 | 0.05398546 | 0.03562194 | 0.04780433 | 1–20: 0.01040726(20); 21–40: 0.02126933(20); 41–60: 0.03374114(20); 61–80: 0.03975853(20); 81–100: 0.04714965(20) |
| train/ufo/aux_rew/limits_dof_vel | 100 | 0.01458504 | 0.0242957 | 0.02422635 | 0.0127616 | 0.03188714 | 0.02498952 | 0.03038868 | 1–20: 0.01750425(20); 21–40: 0.02620172(20); 41–60: 0.02560935(20); 61–80: 0.02536211(20); 81–100: 0.02645431(20) |
| train/ufo/aux_rew/limits_torque | 100 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1–20: 0(20); 21–40: 0(20); 41–60: 0(20); 61–80: 0(20); 81–100: 0(20) |
| train/ufo/aux_rew/penalty_action_rate | 100 | 0.7219021 | 0.6393009 | 0.7042337 | 0.634298 | 0.7402419 | 0.7113858 | 0.7365268 | 1–20: 0.7245177(20); 21–40: 0.7334785(20); 41–60: 0.7147834(20); 61–80: 0.6905442(20); 81–100: 0.6578448(20) |
| train/ufo/aux_rew/penalty_ankle_roll | 100 | 0.01661116 | 0.03193535 | 0.02575699 | 0.01569881 | 0.03193535 | 0.02729383 | 0.03007176 | 1–20: 0.0182735(20); 21–40: 0.02568014(20); 41–60: 0.02749896(20); 61–80: 0.02789698(20); 81–100: 0.02943538(20) |
| train/ufo/aux_rew/penalty_feet_ori | 100 | 0.09336361 | 0.08363227 | 0.09149917 | 0.07748439 | 0.1177817 | 0.08769107 | 0.1120169 | 1–20: 0.09426522(20); 21–40: 0.1057664(20); 41–60: 0.09070298(20); 61–80: 0.08414223(20); 81–100: 0.08261897(20) |
| train/ufo/aux_rew/penalty_slippage | 100 | 0.341164 | 0.2816201 | 0.3129969 | 0.2691687 | 0.3767489 | 0.3108474 | 0.3481894 | 1–20: 0.3285099(20); 21–40: 0.3350303(20); 41–60: 0.3092447(20); 61–80: 0.2980146(20); 81–100: 0.294185(20) |
| train/ufo/aux_rew/penalty_torques | 100 | 4230.153 | 5117.382 | 5184.874 | 4143.425 | 5865.047 | 5305.939 | 5643.798 | 1–20: 4518.606(20); 21–40: 5571.801(20); 41–60: 5439.411(20); 61–80: 5275.971(20); 81–100: 5118.582(20) |
| train/ufo/aux_rew/penalty_undesired_contact | 100 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1–20: 0(20); 21–40: 0(20); 41–60: 0(20); 61–80: 0(20); 81–100: 0(20) |
| train/ufo/backward_optimizer/grad_clip_limit | 100 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 1–20: 5(20); 21–40: 5(20); 41–60: 5(20); 61–80: 5(20); 81–100: 5(20) |
| train/ufo/backward_optimizer/grad_norm_after_clip_l2 | 100 | 5 | 5 | 5 | 4.999999 | 5 | 5 | 5 | 1–20: 5(20); 21–40: 5(20); 41–60: 5(20); 61–80: 5(20); 81–100: 5(20) |
| train/ufo/backward_optimizer/grad_norm_l2 | 100 | 551316.2 | 161190.5 | 255542.4 | 143429.7 | 598089.7 | 199554.4 | 551943.4 | 1–20: 488172.5(20); 21–40: 242534.5(20); 41–60: 205046.9(20); 61–80: 179357.6(20); 81–100: 162600.5(20) |
| train/ufo/backward_optimizer/lr | 100 | 1e-05 | 1e-05 | 1e-05 | 1e-05 | 1e-05 | 1e-05 | 1e-05 | 1–20: 1e-05(20); 21–40: 1e-05(20); 41–60: 1e-05(20); 61–80: 1e-05(20); 81–100: 1e-05(20) |
| train/ufo/backward_optimizer/parameter_delta_l2 | 100 | 0.004137998 | 0.004202081 | 0.003895048 | 0.003650863 | 0.004202081 | 0.003856346 | 0.004154024 | 1–20: 0.003828755(20); 21–40: 0.003761432(20); 41–60: 0.003835763(20); 61–80: 0.003942828(20); 81–100: 0.004106462(20) |
| train/ufo/collected_transitions | 7 | 24576 | 24576 | 24576 | 24576 | 24576 | 24576 | 24576 | 1–20: 24576(1); 21–40: 24576(1); 41–60: 24576(1); 61–80: 24576(2); 81–100: 24576(2) |
| train/ufo/contact_termination_count | 7 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1–20: 0(1); 21–40: 0(1); 41–60: 0(1); 61–80: 0(2); 81–100: 0(2) |
| train/ufo/critic_loss | 100 | 3.583375 | 0.7837901 | 7.912635 | 0.7035854 | 318.447 | 1.910631 | 20.91395 | 1–20: 33.18451(20); 21–40: 2.322172(20); 41–60: 1.885854(20); 61–80: 1.305575(20); 81–100: 0.865067(20) |
| train/ufo/critic_optimizer/grad_clip_limit | 100 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 1–20: 5(20); 21–40: 5(20); 41–60: 5(20); 61–80: 5(20); 81–100: 5(20) |
| train/ufo/critic_optimizer/grad_norm_after_clip_l2 | 100 | 5 | 5 | 4.984942 | 3.494209 | 5 | 5 | 5 | 1–20: 5(20); 21–40: 5(20); 41–60: 5(20); 61–80: 4.999999(20); 81–100: 4.92471(20) |
| train/ufo/critic_optimizer/grad_norm_l2 | 100 | 48.3155 | 13.91325 | 45.60927 | 3.494209 | 924.0448 | 15.78989 | 197.0961 | 1–20: 168.1489(20); 21–40: 18.60328(20); 41–60: 16.09481(20); 61–80: 12.57341(20); 81–100: 12.62596(20) |
| train/ufo/critic_optimizer/lr | 100 | 0.0003 | 0.0003 | 0.0003 | 0.0003 | 0.0003 | 0.0003 | 0.0003 | 1–20: 0.0003(20); 21–40: 0.0003(20); 41–60: 0.0003(20); 61–80: 0.0003(20); 81–100: 0.0003(20) |
| train/ufo/critic_optimizer/parameter_delta_l2 | 100 | 3.45968 | 0.3240681 | 0.5115109 | 0.2587039 | 3.45968 | 0.3597042 | 1.35896 | 1–20: 1.146876(20); 21–40: 0.4324433(20); 41–60: 0.33772(20); 61–80: 0.3364431(20); 81–100: 0.3040725(20) |
| train/ufo/disc_expert_loss | 100 | 0.6469326 | 0.4694254 | 0.5368347 | 0.4694254 | 0.6469326 | 0.522796 | 0.6266223 | 1–20: 0.6131232(20); 21–40: 0.5602568(20); 41–60: 0.5249006(20); 61–80: 0.5014781(20); 81–100: 0.4844147(20) |
| train/ufo/disc_loss | 100 | 4.88035 | 2.163018 | 3.069084 | 2.163018 | 4.88035 | 2.946398 | 4.289474 | 1–20: 4.09732(20); 21–40: 3.375887(20); 41–60: 2.948738(20); 61–80: 2.60028(20); 81–100: 2.323195(20) |
| train/ufo/disc_reward_clip_fraction | 100 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1–20: 0(20); 21–40: 0(20); 41–60: 0(20); 61–80: 0(20); 81–100: 0(20) |
| train/ufo/disc_train_loss | 100 | 0.7325081 | 0.6783022 | 0.7171259 | 0.6767797 | 0.7388562 | 0.7243712 | 0.7333006 | 1–20: 0.7274484(20); 21–40: 0.7311224(20); 41–60: 0.7259786(20); 61–80: 0.7095025(20); 81–100: 0.6915779(20) |
| train/ufo/disc_wgan_gp_loss | 100 | 0.3500909 | 0.101529 | 0.1815123 | 0.101529 | 0.3500909 | 0.1700086 | 0.2942175 | 1–20: 0.2756748(20); 21–40: 0.2084508(20); 41–60: 0.1697859(20); 61–80: 0.13893(20); 81–100: 0.1147202(20) |
| train/ufo/discriminator_optimizer/grad_clip_limit | 100 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 1–20: 5(20); 21–40: 5(20); 41–60: 5(20); 61–80: 5(20); 81–100: 5(20) |
| train/ufo/discriminator_optimizer/grad_norm_after_clip_l2 | 100 | 5 | 4.999999 | 4.999999 | 4.999999 | 5 | 5 | 5 | 1–20: 5(20); 21–40: 4.999999(20); 41–60: 5(20); 61–80: 4.999999(20); 81–100: 4.999999(20) |
| train/ufo/discriminator_optimizer/grad_norm_l2 | 100 | 7.941738 | 7.128434 | 8.060264 | 7.12654 | 8.713987 | 8.191833 | 8.544593 | 1–20: 8.462132(20); 21–40: 8.427548(20); 41–60: 8.19141(20); 61–80: 7.845643(20); 81–100: 7.374589(20) |
| train/ufo/discriminator_optimizer/lr | 100 | 3e-06 | 3e-06 | 3e-06 | 3e-06 | 3e-06 | 3e-06 | 3e-06 | 1–20: 3e-06(20); 21–40: 3e-06(20); 41–60: 3e-06(20); 61–80: 3e-06(20); 81–100: 3e-06(20) |
| train/ufo/discriminator_optimizer/parameter_delta_l2 | 100 | 0.005005511 | 0.0051301 | 0.004776969 | 0.004520011 | 0.0051301 | 0.004749989 | 0.005097238 | 1–20: 0.004586101(20); 21–40: 0.004584352(20); 41–60: 0.004742889(20); 61–80: 0.004910139(20); 81–100: 0.005061364(20) |
| train/ufo/environment_control_steps | 7 | 8 | 32 | 20 | 8 | 32 | 20 | 30.8 | 1–20: 8(1); 21–40: 12(1); 41–60: 16(1); 61–80: 22(2); 81–100: 30(2) |
| train/ufo/expert_draws/g1 | 7 | 16088 | 102896 | 63920 | 16088 | 102896 | 65776 | 101624 | 1–20: 16088(1); 21–40: 32664(1); 41–60: 49216(1); 61–80: 73960(2); 81–100: 100776(2) |
| train/ufo/expert_draws/official_rpo | 7 | 16680 | 101904 | 63641.14 | 16680 | 101904 | 65296 | 100718.4 | 1–20: 16680(1); 21–40: 32872(1); 41–60: 49088(1); 61–80: 73496(2); 81–100: 99928(2) |
| train/ufo/fb_diag | 100 | 2.5798 | -65.49105 | -32.67876 | -97.33417 | 28.24045 | -31.3764 | -6.59436 | 1–20: -16.88666(20); 21–40: -14.99704(20); 41–60: -29.90608(20); 61–80: -43.14586(20); 81–100: -58.45817(20) |
| train/ufo/fb_loss | 100 | 287851.2 | 109611.1 | 155403.6 | 107856.2 | 300518.6 | 141136.6 | 261205.2 | 1–20: 238961.2(20); 21–40: 159971.9(20); 41–60: 140349.1(20); 61–80: 124588.3(20); 81–100: 113147.5(20) |
| train/ufo/fb_offdiag | 100 | 554.0863 | 99.91914 | 436.55 | 90.33247 | 18987.11 | 115.5223 | 756.1563 | 1–20: 1713.676(20); 21–40: 155.9683(20); 41–60: 114.7999(20); 61–80: 98.93881(20); 81–100: 99.36751(20) |
| train/ufo/first_step_samples | 7 | 1548 | 6026 | 4772 | 1548 | 6564 | 5001 | 6402.6 | 1–20: 1548(1); 21–40: 4001(1); 41–60: 5001(1); 61–80: 5743.5(2); 81–100: 5683.5(2) |
| train/ufo/first_step_terminated | 7 | 0 | 1 | 1 | 0 | 3 | 1 | 2.7 | 1–20: 0(1); 21–40: 2(1); 41–60: 0(1); 61–80: 1.5(2); 81–100: 1(2) |
| train/ufo/foot_contact_peak_N | 7 | 2974.891 | 4109.075 | 3828.205 | 2928.79 | 5705.269 | 3573.312 | 5226.41 | 1–20: 2974.891(1); 21–40: 2928.79(1); 41–60: 3523.618(1); 61–80: 4639.291(2); 81–100: 4045.778(2) |
| train/ufo/forward_optimizer/grad_clip_limit | 100 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 1–20: 5(20); 21–40: 5(20); 41–60: 5(20); 61–80: 5(20); 81–100: 5(20) |
| train/ufo/forward_optimizer/grad_norm_after_clip_l2 | 100 | 5 | 5 | 5 | 4.999999 | 5.000001 | 5 | 5 | 1–20: 5(20); 21–40: 5(20); 41–60: 5(20); 61–80: 5(20); 81–100: 5(20) |
| train/ufo/forward_optimizer/grad_norm_l2 | 100 | 1747.581 | 675.6267 | 2050.543 | 456.6245 | 47713.45 | 1171.752 | 4146.194 | 1–20: 5997.201(20); 21–40: 1188.189(20); 41–60: 1252.335(20); 61–80: 990.1873(20); 81–100: 824.8006(20) |
| train/ufo/forward_optimizer/lr | 100 | 0.0003 | 0.0003 | 0.0003 | 0.0003 | 0.0003 | 0.0003 | 0.0003 | 1–20: 0.0003(20); 21–40: 0.0003(20); 41–60: 0.0003(20); 61–80: 0.0003(20); 81–100: 0.0003(20) |
| train/ufo/forward_optimizer/parameter_delta_l2 | 100 | 3.474647 | 0.3809022 | 0.4395581 | 0.2029156 | 3.474647 | 0.3026348 | 1.138549 | 1–20: 1.026541(20); 21–40: 0.3068204(20); 41–60: 0.2582833(20); 61–80: 0.2682928(20); 81–100: 0.3378529(20) |
| train/ufo/hard_position_count | 7 | 1746 | 8001 | 6535.429 | 1746 | 9916 | 7082 | 9341.5 | 1–20: 1746(1); 21–40: 5227(1); 41–60: 7169(1); 61–80: 8261.5(2); 81–100: 7541.5(2) |
| train/ufo/hard_torque_count | 7 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1–20: 0(1); 21–40: 0(1); 41–60: 0(1); 61–80: 0(2); 81–100: 0(2) |
| train/ufo/hard_velocity_count | 7 | 3521 | 2639 | 4108.286 | 2639 | 5086 | 4274 | 4950.7 | 1–20: 3521(1); 21–40: 5086(1); 41–60: 4274(1); 61–80: 4590(2); 81–100: 3348.5(2) |
| train/ufo/height_termination_count | 7 | 1 | 6 | 75.57143 | 1 | 293 | 40 | 246.8 | 1–20: 1(1); 21–40: 293(1); 41–60: 139(1); 61–80: 40.5(2); 81–100: 7.5(2) |
| train/ufo/latched_velocity_terms_without_boundary_violation | 7 | 1989 | 1536 | 2156.143 | 1536 | 2479 | 2219 | 2475.1 | 1–20: 1989(1); 21–40: 2466(1); 41–60: 2479(1); 61–80: 2202(2); 81–100: 1877.5(2) |
| train/ufo/learner_joint_updates | 100 | 1 | 100 | 50.5 | 1 | 100 | 50.5 | 95.05 | 1–20: 10.5(20); 21–40: 30.5(20); 41–60: 50.5(20); 61–80: 70.5(20); 81–100: 90.5(20) |
| train/ufo/mean_aux_reward | 100 | -0.616884 | -0.6117909 | -0.5814225 | -0.6676612 | -0.4853851 | -0.5871526 | -0.4984464 | 1–20: -0.5375297(20); 21–40: -0.5741132(20); 41–60: -0.5758536(20); 61–80: -0.5960445(20); 81–100: -0.6235714(20) |
| train/ufo/mean_disc_reward | 100 | 0.08482538 | 0.01929672 | 0.05576086 | 0.002083196 | 0.09284317 | 0.06262246 | 0.08486732 | 1–20: 0.07605113(20); 21–40: 0.07260699(20); 41–60: 0.06594425(20); 61–80: 0.04105519(20); 81–100: 0.02314672(20) |
| train/ufo/mean_disc_reward_raw | 100 | 0.08482538 | 0.01929672 | 0.05576086 | 0.002083196 | 0.09284317 | 0.06262246 | 0.08486732 | 1–20: 0.07605113(20); 21–40: 0.07260699(20); 41–60: 0.06594425(20); 61–80: 0.04105519(20); 81–100: 0.02314672(20) |
| train/ufo/mean_next_Q | 100 | -0.1215248 | -1.687263 | -1.972973 | -2.391725 | -0.1215248 | -2.082675 | -0.8332253 | 1–20: -1.315599(20); 21–40: -2.310328(20); 41–60: -2.317172(20); 61–80: -2.083977(20); 81–100: -1.837789(20) |
| train/ufo/mean_next_auxQ | 100 | 0.01465873 | -2.749624 | -3.16065 | -3.990033 | 0.01465873 | -3.342071 | -1.51671 | 1–20: -2.382932(20); 21–40: -3.789199(20); 41–60: -3.52014(20); 61–80: -3.20691(20); 81–100: -2.904068(20) |
| train/ufo/online_transitions | 100 | 49152 | 196608 | 114032.6 | 49152 | 196608 | 122880 | 172032 | 1–20: 54067.2(20); 21–40: 83558.4(20); 41–60: 113049.6(20); 61–80: 142540.8(20); 81–100: 176947.2(20) |
| train/ufo/orientation_termination_count | 7 | 4 | 905 | 486.8571 | 4 | 1247 | 327 | 1144.4 | 1–20: 4(1); 21–40: 196(1); 41–60: 327(1); 61–80: 364.5(2); 81–100: 1076(2) |
| train/ufo/orth_loss | 100 | 2872.946 | 1095.766 | 1549.997 | 1078.15 | 2872.946 | 1410.571 | 2600.364 | 1–20: 2372.644(20); 21–40: 1598.309(20); 41–60: 1402.642(20); 61–80: 1245.325(20); 81–100: 1131.066(20) |
| train/ufo/orth_loss_diag | 100 | -256 | -256 | -256 | -256 | -256 | -256 | -256 | 1–20: -256(20); 21–40: -256(20); 41–60: -256(20); 61–80: -256(20); 81–100: -256(20) |
| train/ufo/orth_loss_offdiag | 100 | 3128.946 | 1351.766 | 1805.997 | 1334.15 | 3128.946 | 1666.571 | 2856.364 | 1–20: 2628.644(20); 21–40: 1854.309(20); 41–60: 1658.642(20); 61–80: 1501.325(20); 81–100: 1387.066(20) |
| train/ufo/physics_peak_joint_velocity | 7 | 24.52727 | 25.4072 | 26.60315 | 23.89919 | 31.79166 | 25.80004 | 30.66848 | 1–20: 24.52727(1); 21–40: 31.79166(1); 41–60: 28.0477(1); 61–80: 25.32408(2); 81–100: 25.60362(2) |
| train/ufo/physics_peak_normalized_velocity | 7 | 3.065908 | 3.1759 | 3.296288 | 2.78366 | 3.973958 | 3.225005 | 3.833559 | 1–20: 3.065908(1); 21–40: 3.973958(1); 41–60: 3.505963(1); 61–80: 3.063641(2); 81–100: 3.200452(2) |
| train/ufo/q_loss | 100 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1–20: 0(20); 21–40: 0(20); 41–60: 0(20); 61–80: 0(20); 81–100: 0(20) |
| train/ufo/replay_draw_new_ratio | 7 | 0.6666667 | 1.041667 | 0.9882653 | 0.6666667 | 1.142857 | 1.041667 | 1.133333 | 1–20: 0.6666667(1); 21–40: 0.8888889(1); 41–60: 1(1); 61–80: 1.088889(2); 81–100: 1.092262(2) |
| train/ufo/replay_draws | 7 | 32768 | 204800 | 127561.1 | 32768 | 204800 | 131072 | 202342.4 | 1–20: 32768(1); 21–40: 65536(1); 41–60: 98304(1); 61–80: 147456(2); 81–100: 200704(2) |
| train/ufo/replay_policy_age_max_updates | 7 | 16 | 100 | 62.28571 | 16 | 100 | 64 | 98.8 | 1–20: 16(1); 21–40: 32(1); 41–60: 48(1); 61–80: 72(2); 81–100: 98(2) |
| train/ufo/replay_policy_age_min_updates | 7 | 16 | 4 | 14.28571 | 4 | 16 | 16 | 16 | 1–20: 16(1); 21–40: 16(1); 41–60: 16(1); 61–80: 16(2); 81–100: 10(2) |
| train/ufo/replay_retained_fraction | 7 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1–20: 1(1); 21–40: 1(1); 41–60: 1(1); 61–80: 1(2); 81–100: 1(2) |
| train/ufo/replay_size | 7 | 49152 | 196608 | 122880 | 49152 | 196608 | 122880 | 189235.2 | 1–20: 49152(1); 21–40: 73728(1); 41–60: 98304(1); 61–80: 135168(2); 81–100: 184320(2) |
| train/ufo/requested_action_abs_mean | 7 | 0.5505452 | 0.9235234 | 0.8601631 | 0.5505452 | 0.9235234 | 0.9073041 | 0.9215509 | 1–20: 0.5505452(1); 21–40: 0.9016314(1); 41–60: 0.9068104(1); 61–80: 0.9121263(2); 81–100: 0.918951(2) |
| train/ufo/requested_action_saturation_fraction | 7 | 0.03108547 | 0.4712639 | 0.3920001 | 0.03108547 | 0.4712639 | 0.4460732 | 0.468839 | 1–20: 0.03108547(1); 21–40: 0.4321501(1); 41–60: 0.4421298(1); 61–80: 0.454627(2); 81–100: 0.4646907(2) |
| train/ufo/requested_applied_max_diff | 7 | 1.909816 | 1.9528 | 1.945588 | 1.909816 | 1.9528 | 1.9528 | 1.9528 | 1–20: 1.909816(1); 21–40: 1.945296(1); 41–60: 1.9528(1); 61–80: 1.9528(2); 81–100: 1.9528(2) |
| train/ufo/reset_control_samples | 7 | 1548 | 6026 | 4772 | 1548 | 6564 | 5001 | 6402.6 | 1–20: 1548(1); 21–40: 4001(1); 41–60: 5001(1); 61–80: 5743.5(2); 81–100: 5683.5(2) |
| train/ufo/reset_projection_outside_fraction | 7 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1–20: 0(1); 21–40: 0(1); 41–60: 0(1); 61–80: 0(2); 81–100: 0(2) |
| train/ufo/reset_projection_residual_max_rad | 7 | 1.788139e-07 | 1.788139e-07 | 1.788139e-07 | 1.788139e-07 | 1.788139e-07 | 1.788139e-07 | 1.788139e-07 | 1–20: 1.788139e-07(1); 21–40: 1.788139e-07(1); 41–60: 1.788139e-07(1); 61–80: 1.788139e-07(2); 81–100: 1.788139e-07(2) |
| train/ufo/reset_projection_residual_mean_rad | 7 | 6.376912e-08 | 6.375344e-08 | 6.391457e-08 | 6.369449e-08 | 6.409785e-08 | 6.399523e-08 | 6.409624e-08 | 1–20: 6.376912e-08(1); 21–40: 6.409785e-08(1); 41–60: 6.399523e-08(1); 61–80: 6.389348e-08(2); 81–100: 6.38764e-08(2) |
| train/ufo/reset_projection_residual_p95_rad | 7 | 1.192093e-07 | 1.192093e-07 | 1.192093e-07 | 1.192093e-07 | 1.192093e-07 | 1.192093e-07 | 1.192093e-07 | 1–20: 1.192093e-07(1); 21–40: 1.192093e-07(1); 41–60: 1.192093e-07(1); 61–80: 1.192093e-07(2); 81–100: 1.192093e-07(2) |
| train/ufo/reset_root_lift_max_m | 7 | 0.02199202 | 0.02199202 | 0.0219854 | 0.02194565 | 0.02199202 | 0.02199202 | 0.02199202 | 1–20: 0.02199202(1); 21–40: 0.02199202(1); 41–60: 0.02194565(1); 61–80: 0.02199202(2); 81–100: 0.02199202(2) |
| train/ufo/sampling_rounds | 100 | 2 | 8 | 4.64 | 2 | 8 | 5 | 7 | 1–20: 2.2(20); 21–40: 3.4(20); 41–60: 4.6(20); 61–80: 5.8(20); 81–100: 7.2(20) |
| train/ufo/strict_shadow_episode_len | 7 | 5.223897 | 3.354456 | 4.022451 | 3.182092 | 5.570771 | 3.363288 | 5.466709 | 1–20: 5.223897(1); 21–40: 5.570771(1); 41–60: 4.177729(1); 61–80: 3.233508(2); 81–100: 3.358872(2) |
| train/ufo/strict_shadow_mixed_episode_len | 7 | 5.24017 | 3.26948 | 4.044627 | 3.168401 | 5.630075 | 3.409524 | 5.513103 | 1–20: 5.24017(1); 21–40: 5.630075(1); 41–60: 4.25177(1); 61–80: 3.288962(2); 81–100: 3.306223(2) |
| train/ufo/strict_shadow_tracking_episode_len | 7 | 5.205883 | 3.438745 | 4.002854 | 3.173704 | 5.515304 | 3.438745 | 5.422478 | 1–20: 5.205883(1); 21–40: 5.515304(1); 41–60: 4.10122(1); 61–80: 3.186559(2); 81–100: 3.412227(2) |
| train/ufo/strict_velocity_exposure_fraction | 7 | 0.00243632 | 0.001798768 | 0.003356333 | 0.001798768 | 0.005197498 | 0.003254545 | 0.004787169 | 1–20: 0.00243632(1); 21–40: 0.005197498(1); 41–60: 0.003254545(1); 61–80: 0.003814144(2); 81–100: 0.002488841(2) |
| train/ufo/strict_velocity_transition_fraction | 7 | 0.1432699 | 0.1073812 | 0.1671666 | 0.1073812 | 0.2069499 | 0.1739095 | 0.2014445 | 1–20: 0.1432699(1); 21–40: 0.2069499(1); 41–60: 0.1739095(1); 61–80: 0.1867676(2); 81–100: 0.1362508(2) |
| train/ufo/target_M | 100 | -3.042383 | 2.104261 | 1.733012 | -3.042383 | 2.959927 | 2.0435 | 2.708663 | 1–20: 0.4887505(20); 21–40: 1.408771(20); 41–60: 2.190502(20); 61–80: 2.313955(20); 81–100: 2.263081(20) |
| train/ufo/target_Q | 100 | -0.9409766 | -2.322197 | -3.083145 | -4.037799 | -0.9409766 | -3.1826 | -1.334977 | 1–20: -2.182918(20); 21–40: -3.783219(20); 41–60: -3.663587(20); 61–80: -3.144976(20); 81–100: -2.641023(20) |
| train/ufo/target_auxQ | 100 | -1.393499 | -3.095283 | -3.742666 | -4.664165 | -1.393499 | -3.784648 | -2.633514 | 1–20: -3.373777(20); 21–40: -4.438231(20); 41–60: -3.98222(20); 61–80: -3.604213(20); 81–100: -3.314887(20) |
| train/ufo/terminated_transitions | 7 | 1791 | 5902 | 4927.143 | 1791 | 6679 | 5002 | 6445.9 | 1–20: 1791(1); 21–40: 4995(1); 41–60: 5002(1); 61–80: 5662.5(2); 81–100: 5688.5(2) |
| train/ufo/tracking_episode_starts | 7 | 718 | 3080 | 2363.714 | 718 | 3434 | 2523 | 3327.8 | 1–20: 718(1); 21–40: 2040(1); 41–60: 2540(1); 61–80: 2822.5(2); 81–100: 2801.5(2) |
| train/ufo/tracking_q_rmse_rad | 7 | 0.2355341 | 0.114614 | 0.1605808 | 0.114614 | 0.2471359 | 0.1310449 | 0.2436554 | 1–20: 0.2355341(1); 21–40: 0.2471359(1); 41–60: 0.1504239(1); 61–80: 0.1226563(2); 81–100: 0.1228295(2) |
| train/ufo/tracking_unique_env_fraction | 7 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 1–20: 0.5(1); 21–40: 0.5(1); 41–60: 0.5(1); 61–80: 0.5(2); 81–100: 0.5(2) |
| train/ufo/tracking_windows_completed | 7 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1–20: 0(1); 21–40: 0(1); 41–60: 0(1); 61–80: 0(2); 81–100: 0(2) |
| train/ufo/training_velocity_count | 7 | 1130 | 704 | 1172.571 | 704 | 1739 | 1130 | 1597.4 | 1–20: 1130(1); 21–40: 1739(1); 41–60: 1031(1); 61–80: 1252(2); 81–100: 902(2) |
| train/ufo/truncated_transitions | 7 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1–20: 0(1); 21–40: 0(1); 41–60: 0(1); 61–80: 0(2); 81–100: 0(2) |
| train/ufo/unc_Q | 100 | 1.929751 | 2.277272 | 3.262051 | 1.360271 | 4.33872 | 3.356784 | 4.255332 | 1–20: 2.23038(20); 21–40: 4.032545(20); 41–60: 4.052737(20); 61–80: 3.377266(20); 81–100: 2.617327(20) |
| train/ufo/unc_auxQ | 100 | 1.730957 | 0.59027 | 0.9420976 | 0.5693983 | 1.730957 | 0.8909872 | 1.526627 | 1–20: 1.360648(20); 21–40: 1.104482(20); 41–60: 0.8941119(20); 61–80: 0.7312207(20); 81–100: 0.620026(20) |
| train/ufo/velocity_reward_coverage_miss_count | 7 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1–20: 0(1); 21–40: 0(1); 41–60: 0(1); 61–80: 0(2); 81–100: 0(2) |
| train/ufo/wall_time_s | 100 | 48.57257 | 242.7759 | 137.9921 | 48.57257 | 242.7759 | 145.7984 | 218.0904 | 1–20: 59.23417(20); 21–40: 98.22225(20); 41–60: 137.2415(20); 61–80: 176.014(20); 81–100: 219.2487(20) |
| train/ufo/z_norm | 100 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 1–20: 16(20); 21–40: 16(20); 41–60: 16(20); 61–80: 16(20); 81–100: 16(20) |

按实际采样轮记录的关键原始数值（—表示该step没有记录该指标，未插值）：

| step | env/episode_len | train/ufo/strict_shadow_episode_len | train/ufo/physics_peak_normalized_velocity | train/ufo/first_step_samples | train/ufo/first_step_terminated | train/ufo/hard_position_count | train/ufo/training_position_count | train/ufo/critic_loss | train/ufo/Q_fb | train/ufo/mean_disc_reward | train/ufo/applied_action_saturation_fraction |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 3.542705 | 3.33452 | 2.265168 | 6377 | 0 | 583 | — | — | — | — | 0.0002954455 |
| 16 | 6.437744 | 5.223897 | 3.065908 | 1548 | 0 | 1746 | — | 3.120994 | -3.270495 | 0.05522592 | 0.001119862 |
| 32 | 7.57978 | 5.570771 | 3.973958 | 4001 | 2 | 5227 | — | 1.995522 | -0.2772701 | 0.06442413 | 0.0316569 |
| 48 | 5.957417 | 4.177729 | 3.505963 | 5001 | 0 | 7169 | — | 1.85187 | 6.676256 | 0.06855037 | 0.0539197 |
| 64 | 4.683785 | 3.284923 | 2.78366 | 6564 | 3 | 9916 | — | 1.367008 | 9.822695 | 0.05286748 | 0.06829038 |
| 80 | 4.406586 | 3.182092 | 3.343621 | 4923 | 0 | 6607 | — | 0.9695925 | 8.715391 | 0.03926889 | 0.07165173 |
| 96 | 4.720913 | 3.363288 | 3.225005 | 5341 | 1 | 7082 | — | 0.7035854 | 9.89638 | 0.007085762 | 0.09793047 |
| 100 | 4.448153 | 3.354456 | 3.1759 | 6026 | 1 | 8001 | — | 0.7837901 | 12.36071 | 0.01929672 | 0.102509 |

本阶段冻结的实际解析配置：
```yaml
actor:
  model:
    model_type: mlp_policy
    model_path: ''
    policy_setup: rpo-v2-bfm-ufo
    obs_dim: 375
    action_dim: 23
    num_action_chunks: 1
    hidden_dim: 2048
    precision: '32'
    add_value_head: false
    is_lora: false
    lora_rank: 32
    critic_obs_dim: 733
    bfm_profile: ufo_native_bringup
    motion_root: /workspace/data/ds_fb0ecda9e70644f2afdd
    asset_sha256: b58868a782d8863ac38059eb4eab89d27ee1d70565bf58f5a055f44510c8e842
    batch_size: 2048
    replay_capacity: 4000000
    hidden_layers: 6
    topology_id: one_learner_three_samplers
    learning_profile: physical_limits_stable_slew_v4
    architecture_profile: native_fb_v1
    rollout_expert_trajectories: true
    auxiliary_profile: soft_velocity_v7
    scaling_profile: env6144_batch2048_v6
    control_profile: reachable_v2
    optimizer_profile: stabilized_v7
    context_profile: environment_tracking_v10
    contact_profile: native_mjcf_viability_v13
    physics_profile: native_mjcf_viability_v13
    codec_sha256: dce25d554ceb7fa69287515946d2f5cd9d41d14f1b1ef38715b819d7e962b6ef
  group_name: ActorGroup
  training_backend: ufo_agent
  micro_batch_size: 2048
  global_batch_size: 2048
  seed: 42
  enable_offload: false
  optim: null
  fsdp_config: null
weight_syncer:
  type: patch
  patch:
    snapshot_device: cpu
    delta_encoding: true
    compression: none
    init_sync:
      enabled: true
      prefixes: null
      bucket_size: 134217728
cluster:
  num_nodes: 1
  component_placement:
    actor: 0
    env, rollout: 1-3
runner:
  task_type: embodied
  logger:
    log_path: /workspace/output
    project_name: rlinf
    experiment_name: RPO UFO MJLab v13 RSI viability 100-update workflow
    logger_backends:
    - tensorboard
  max_epochs: 4096
  max_steps: -1
  only_eval: false
  use_training_pipeline: false
  overlap_env_bootstrap: false
  val_check_interval: -1
  save_interval: -1
  resume_dir: null
  ckpt_path: null
algorithm:
  loss_type: ufo_fb_cpr_aux
  update_epoch: 1
  normalize_advantages: false
  kl_penalty: kl
  group_size: 1
  reward_type: action_level
  logprob_type: action_level
  entropy_type: action_level
  adv_type: gae
  loss_agg_func: token-mean
  bootstrap_type: standard
  gamma: 0.98
  gae_lambda: 1.0
env:
  group_name: EnvGroup
  train:
    env_type: mjlab
    total_num_envs: 6144
    auto_reset: true
    ignore_terminations: false
    use_rel_reward: false
    seed: 42
    group_size: 1
    reward_coef: 1.0
    use_fixed_reset_state_ids: false
    rollout_epoch: 1
    max_episode_steps: 1000
    max_steps_per_rollout_epoch: 4
    video_cfg:
      save_video: false
      info_on_video: false
      fps: 50
      video_base_dir: /workspace/output/video/train
    init_params:
      id: RPO-UFO-MJLab-RSI-viability-v13
      num_envs: null
      max_episode_steps: 1000
      task_description: RPO V2 BFM-Zero online interaction with pinned observation-only
        expert motion.
  eval:
    env_type: mjlab
    total_num_envs: 3
    auto_reset: true
    ignore_terminations: false
    use_rel_reward: false
    seed: 42
    group_size: 1
    reward_coef: 1.0
    use_fixed_reset_state_ids: true
    rollout_epoch: 1
    max_episode_steps: 1000
    max_steps_per_rollout_epoch: 1000
    video_cfg:
      save_video: false
      info_on_video: false
      fps: 50
      video_base_dir: /workspace/output/video/eval
    init_params:
      id: RPO-UFO-MJLab-RSI-viability-v13
      num_envs: null
      max_episode_steps: 1000
      task_description: RPO V2 BFM-Zero online interaction with pinned observation-only
        expert motion.
rollout:
  group_name: RolloutGroup
  backend: huggingface
  enable_offload: false
  pipeline_stage_num: 1
  enable_torch_compile: false
  enable_cuda_graph: false
  collect_transitions: true
  collect_prev_infos: true
  model:
    model_path: ''
    precision: '32'
reward:
  use_reward_model: false
critic:
  use_critic_model: false

```

算法原生完整配置（本轮未改变算法超参）：
```json
{
  "name": "FBcprAuxAgent",
  "model": {
    "name": "FBcprAuxModel",
    "device": "cuda",
    "archi": {
      "name": "FBcprAuxModelArchiConfig",
      "z_dim": 256,
      "norm_z": true,
      "f": {
        "name": "ForwardArchi",
        "hidden_dim": 2048,
        "model": "residual",
        "hidden_layers": 6,
        "embedding_layers": 2,
        "num_parallel": 2,
        "ensemble_mode": "batch",
        "input_filter": {
          "name": "DictInputFilterConfig",
          "key": [
            "state",
            "privileged_state",
            "last_action",
            "history_actor"
          ]
        }
      },
      "b": {
        "name": "BackwardArchi",
        "hidden_dim": 256,
        "hidden_layers": 1,
        "norm": true,
        "input_filter": {
          "name": "DictInputFilterConfig",
          "key": [
            "state",
            "privileged_state"
          ]
        }
      },
      "actor": {
        "name": "actor",
        "model": "residual",
        "hidden_dim": 2048,
        "hidden_layers": 6,
        "embedding_layers": 2,
        "input_filter": {
          "name": "DictInputFilterConfig",
          "key": [
            "state",
            "last_action",
            "history_actor"
          ]
        }
      },
      "critic": {
        "name": "ForwardArchi",
        "hidden_dim": 2048,
        "model": "residual",
        "hidden_layers": 6,
        "embedding_layers": 2,
        "num_parallel": 2,
        "ensemble_mode": "batch",
        "input_filter": {
          "name": "DictInputFilterConfig",
          "key": [
            "state",
            "privileged_state",
            "last_action",
            "history_actor"
          ]
        }
      },
      "discriminator": {
        "name": "DiscriminatorArchi",
        "hidden_dim": 1024,
        "hidden_layers": 3,
        "input_filter": {
          "name": "DictInputFilterConfig",
          "key": [
            "state",
            "privileged_state"
          ]
        }
      },
      "aux_critic": {
        "name": "ForwardArchi",
        "hidden_dim": 2048,
        "model": "residual",
        "hidden_layers": 6,
        "embedding_layers": 2,
        "num_parallel": 2,
        "ensemble_mode": "batch",
        "input_filter": {
          "name": "DictInputFilterConfig",
          "key": [
            "state",
            "privileged_state",
            "last_action",
            "history_actor"
          ]
        }
      }
    },
    "obs_normalizer": {
      "name": "ObsNormalizerConfig",
      "normalizers": {
        "state": {
          "name": "BatchNormNormalizerConfig",
          "momentum": 0.01
        },
        "privileged_state": {
          "name": "BatchNormNormalizerConfig",
          "momentum": 0.01
        },
        "last_action": {
          "name": "BatchNormNormalizerConfig",
          "momentum": 0.01
        },
        "history_actor": {
          "name": "BatchNormNormalizerConfig",
          "momentum": 0.01
        }
      },
      "allow_mismatching_keys": true
    },
    "inference_batch_size": 500000,
    "seq_length": 8,
    "actor_std": 0.05,
    "amp": false,
    "norm_aux_reward": {
      "name": "RewardNormalizer",
      "translate": false,
      "scale": true
    }
  },
  "train": {
    "name": "FBcprAuxAgentTrainConfig",
    "lr_f": 0.0003,
    "lr_b": 1e-05,
    "lr_actor": 0.0003,
    "weight_decay": 0.0,
    "clip_grad_norm": 0.0,
    "fb_target_tau": 0.01,
    "ortho_coef": 100.0,
    "train_goal_ratio": 0.2,
    "fb_pessimism_penalty": 0.0,
    "actor_pessimism_penalty": 0.5,
    "stddev_clip": 0.3,
    "q_loss_coef": 0.0,
    "batch_size": 2048,
    "discount": 0.98,
    "use_mix_rollout": true,
    "update_z_every_step": 100,
    "z_buffer_size": 8192,
    "rollout_expert_trajectories": true,
    "rollout_expert_trajectories_length": 250,
    "rollout_expert_trajectories_percentage": 0.5,
    "lr_discriminator": 3e-06,
    "lr_critic": 0.0003,
    "critic_target_tau": 0.005,
    "critic_pessimism_penalty": 0.5,
    "reg_coeff": 0.05,
    "scale_reg": true,
    "expert_asm_ratio": 0.6,
    "relabel_ratio": 0.8,
    "grad_penalty_discriminator": 10.0,
    "weight_decay_discriminator": 0.0,
    "lr_aux_critic": 0.0003,
    "reg_coeff_aux": 0.02,
    "aux_critic_pessimism_penalty": 0.5
  },
  "aux_rewards": [
    "penalty_torques",
    "penalty_action_rate",
    "limits_dof_pos",
    "limits_torque",
    "penalty_undesired_contact",
    "penalty_feet_ori",
    "penalty_ankle_roll",
    "penalty_slippage",
    "limits_dof_vel"
  ],
  "aux_rewards_scaling": {
    "penalty_action_rate": -0.1,
    "penalty_feet_ori": -0.4,
    "penalty_ankle_roll": -4.0,
    "limits_dof_pos": -10.0,
    "penalty_slippage": -2.0,
    "penalty_undesired_contact": -1.0,
    "penalty_torques": 0.0,
    "limits_torque": 0.0,
    "limits_dof_vel": -25.0
  },
  "cudagraphs": false,
  "compile": false
}
```

镜像实测源码/依赖版本回执：
```json
{
  "schema_version": "eb/ufo-image-source/v1",
  "ufo": {
    "commit": "0b89378de02f77712072d983bc43ec2a46455dc0",
    "tree": "3a6b8b0d6a04287f37c1d7834e51561a8e014fc9",
    "content_hash": "sha256:2ec3a9015dda0fd4cfe6861e6ebb35fbf55c78bf43c22544a82ab2368a222e84",
    "license": "CC-BY-NC-4.0",
    "scope": "pinned native agent plus measured v7 regularization/reward bounds",
    "upstream_content_hash": "sha256:2191cf26c77262f7ef2e3525ca23878a8f08c46fd02a20b7776a791f8f5e7582"
  },
  "rlinf": {
    "commit": "018e6a9e0da60f96eeda8b11008543911027d414",
    "content_hash": "sha256:dbaa8bed56be412f6d012ee34030822941945786cede6bf9bc3de0fec5445d48",
    "license": "Apache-2.0",
    "base_patch_hash": "sha256:df5e404b711161d7068894890d1a90bdcf61f064c8c9bfea60210899ecf787a5"
  },
  "patch_hash": "sha256:f99aa821d9ccdf436f91010a7b175209a2bda0f32f40afe8e0a6b49ddaf7d5b5",
  "packages": {
    "python": "3.11.16",
    "torch": "2.7.0+cu128",
    "gymnasium": "1.2.0",
    "pydantic": "2.11.5",
    "safetensors": "0.6.2",
    "tensordict": "0.11.0",
    "mjlab": "1.4.0",
    "mujoco": "3.8.0",
    "mujoco-warp": "3.8.0.3",
    "warp-lang": "1.12.0"
  },
  "scope": "native UFO agent + RPO MJLab dynamic RSI and training-only position tolerance",
  "topology_patch_hash": "sha256:3078506628924464cb42d0b2f108b3fab502a43fd75e091f358d504565294d23",
  "learning_profile_patch_hash": "sha256:d47623e48b4cc757d5e63336a1fa7d49234c3833a2f1c6bbdbee4fe7ffd9a272",
  "rollout_profile_patch_hash": "sha256:777456ce171734326cf8c55eaa434fc5a69a27d0b2d0b725608dd54c190f4a77",
  "action_control_source_sha256": "sha256:0cad3bac9e3efb3f1f37d6cac873f789548143b9cd597ed90a93a3045339cd24",
  "action_control_profile_patch_hash": "sha256:548260ca5e3eedd53ae739c00672c7e273e7835faed734778039cf5977b56e0d",
  "auxiliary_profile": "soft_velocity_v7",
  "auxiliary_source_sha256": "sha256:6cbb53ec59105284b00f72f3b752fdae8e098361ef5a5959936aefb34ba75a37",
  "auxiliary_profile_patch_hash": "sha256:30e7b4ea7a3ec0e45cdbe0a9dbf718eb219c2c6fd36d07f9b18bd5222183a03a",
  "scaling_profile": "env6144_batch2048_v6",
  "scaling_profile_patch_hash": "sha256:58a2dca8377ed2ede804c7101b39bc6992ec669ef95e610ddec4b2a59bfc6b20",
  "control_profile": "reachable_v2",
  "optimizer_profile": "stabilized_v7",
  "stabilization_patch_hash": "sha256:564f5452e12e3179112e6aa3020efba4cbcedc0ad90dd26b98aed78a50dd85b1",
  "contact_patch_hash": "sha256:e20a0ff930b3b9430ece70c21ab316a06cef92b679b638c0a35d7fed62d6d814",
  "ground_patch_hash": "sha256:554a18609c9b8a420811fa6b1ded908a4bdf181e52e8d513c8772de6084eff59",
  "context_profile": "environment_tracking_v10",
  "tracking_contact_patch_hash": "sha256:d9b075574bf38e19cd74b39db52edf92573a019ba24d06e08fb82861f07dbf15",
  "simulator": "mjlab",
  "physics_profile": "native_mjcf_viability_v13",
  "contact_profile": "native_mjcf_viability_v13",
  "mjlab_patch_hash": "sha256:3b06c0e0883c415c910fee686422f7c25827b047ec37528083ddf87b0a6a8eb4",
  "viability_patch_hash": "sha256:46cd010c4c1c1353bdd943b4dbbb4b0aed44794ba1e4a29d400d9416ee9dec0d"
}
```

实际编译物理与完整RSI候选池回执（其余rank使用同一物理配置与候选池，seed43/44；各1024tracking）：
```json
{
  "schema_version": "eb/ufo-mjlab-physics/v2",
  "backend": "mjlab",
  "physics_dt_s": 0.0025,
  "decimation": 8,
  "control_dt_s": 0.02,
  "solver": "newton",
  "iterations": 100,
  "integrator": "implicitfast",
  "geometry": "original RPO MJCF collision meshes",
  "self_collision": false,
  "joint_position_limits": "frozen RPO interface limits; 90% soft position reward margin",
  "contact": "original MJCF solref/solimp/friction; native infinite plane",
  "actuator": "MJLab DcMotorActuator, zero delay, original RPO gains, ankle Kd2.5",
  "armature_kg_m2": 0.01,
  "velocity_limit": "DC motor torque-speed curve; no qvel clipping",
  "training_velocity_termination_multiplier": 1.3,
  "strict_shadow": "original rated position/velocity/torque and macro fall, all substeps",
  "codec_metadata_note": "published physics_hz=200 is historical; runtime=400Hz",
  "rsi_joint_lookahead_s": 0.02,
  "rsi_joint_margin_rad": 0.001,
  "training_position_tolerance_rad": 0.02,
  "strict_position_tolerance_rad": 0.001,
  "armature_evidence": "original asset value; rotor inertia and reduction ratio unavailable",
  "seed": 42,
  "num_envs": 2048,
  "joint_order": [
    "left_thigh_yaw_joint",
    "right_thigh_yaw_joint",
    "torso_joint",
    "left_thigh_roll_joint",
    "right_thigh_roll_joint",
    "left_arm_pitch_joint",
    "right_arm_pitch_joint",
    "left_thigh_pitch_joint",
    "right_thigh_pitch_joint",
    "left_arm_roll_joint",
    "right_arm_roll_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_arm_yaw_joint",
    "right_arm_yaw_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_elbow_pitch_joint",
    "right_elbow_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_elbow_yaw_joint",
    "right_elbow_yaw_joint"
  ],
  "body_order": [
    "base_link",
    "left_thigh_yaw_link",
    "right_thigh_yaw_link",
    "torso_link",
    "left_thigh_roll_link",
    "right_thigh_roll_link",
    "left_arm_pitch_link",
    "right_arm_pitch_link",
    "left_thigh_pitch_link",
    "right_thigh_pitch_link",
    "left_arm_roll_link",
    "right_arm_roll_link",
    "left_knee_link",
    "right_knee_link",
    "left_arm_yaw_link",
    "right_arm_yaw_link",
    "left_ankle_pitch_link",
    "right_ankle_pitch_link",
    "left_elbow_pitch_link",
    "right_elbow_pitch_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_elbow_yaw_link",
    "right_elbow_yaw_link"
  ],
  "joint_ids": [
    0,
    6,
    12,
    1,
    7,
    13,
    18,
    2,
    8,
    14,
    19,
    3,
    9,
    15,
    20,
    4,
    10,
    16,
    21,
    5,
    11,
    17,
    22
  ],
  "body_ids": [
    0,
    1,
    7,
    13,
    2,
    8,
    14,
    19,
    3,
    9,
    15,
    20,
    4,
    10,
    16,
    21,
    5,
    11,
    17,
    22,
    6,
    12,
    18,
    23
  ],
  "mass_kg": 33.7568956,
  "body_mass_kg": [
    0.0,
    0.0,
    5.51295532,
    1.71784575,
    1.66236465,
    2.34600715,
    1.70223945,
    0.084,
    0.33517651,
    1.71632614,
    1.66310652,
    2.34600715,
    1.70223945,
    0.084,
    0.33517651,
    7.58189737,
    0.47284363,
    0.08401883,
    1.15324354,
    0.09168684,
    0.68145595,
    0.47284363,
    0.08401883,
    1.15429959,
    0.09168684,
    0.68145595
  ],
  "body_inertia_kg_m2": [
    [
      0.0,
      0.0,
      0.0
    ],
    [
      0.0,
      0.0,
      0.0
    ],
    [
      0.03111829301183687,
      0.029470257020187213,
      0.015362329967975923
    ],
    [
      0.0035918527388379604,
      0.002441203876322949,
      0.0021225433848390913
    ],
    [
      0.0035121058769795755,
      0.0023677123419928643,
      0.002043781781027557
    ],
    [
      0.02183421380335951,
      0.020189627567974985,
      0.0027833386286655075
    ],
    [
      0.010136152158590775,
      0.009824259071964984,
      0.0010950887694442506
    ],
    [
      1.448e-05,
      8.63e-06,
      8.63e-06
    ],
    [
      0.0005967131935749102,
      0.0005375536839552841,
      0.00013191312246980537
    ],
    [
      0.0035881634177304545,
      0.00243750000187366,
      0.002121716580395884
    ],
    [
      0.003512799404136993,
      0.0023620987748848806,
      0.0020508718209781276
    ],
    [
      0.021834213753062014,
      0.02018962763590513,
      0.0027833386110328816
    ],
    [
      0.010136152158590775,
      0.009824259071964984,
      0.0010950887694442506
    ],
    [
      1.448e-05,
      8.63e-06,
      8.63e-06
    ],
    [
      0.0005967135123689472,
      0.000537552634069252,
      0.00013191385356180218
    ],
    [
      0.06920592568566547,
      0.056795183596489246,
      0.03872180071784518
    ],
    [
      0.00018907,
      0.0001624892942070216,
      7.592070579297844e-05
    ],
    [
      5.649e-05,
      5.6389553588921e-05,
      1.6740446411079014e-05
    ],
    [
      0.006195530101548059,
      0.006163788921573123,
      0.0003251709768788171
    ],
    [
      6.277e-05,
      6.272667537963485e-05,
      1.8093324620365156e-05
    ],
    [
      0.0013267802791525215,
      0.0013253097449007628,
      0.0002448099759467154
    ],
    [
      0.00018907,
      0.0001624892942070216,
      7.592070579297844e-05
    ],
    [
      5.649e-05,
      5.6389553588921e-05,
      1.6740446411079014e-05
    ],
    [
      0.006204300135751578,
      0.006170425580249767,
      0.0003265442839986478
    ],
    [
      6.277e-05,
      6.272667537963485e-05,
      1.8093324620365156e-05
    ],
    [
      0.0013267802791525215,
      0.0013253097449007628,
      0.0002448099759467154
    ]
  ],
  "dof_armature": [
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.01,
    0.01,
    0.01,
    0.01,
    0.01,
    0.01,
    0.01,
    0.01,
    0.01,
    0.01,
    0.01,
    0.01,
    0.01,
    0.01,
    0.01,
    0.01,
    0.01,
    0.01,
    0.01,
    0.01,
    0.01,
    0.01,
    0.01
  ],
  "actuator_parameters": {
    "left_thigh_yaw_joint": [
      100.0,
      3.3,
      120.0,
      25.0
    ],
    "right_thigh_yaw_joint": [
      100.0,
      3.3,
      120.0,
      25.0
    ],
    "torso_joint": [
      150.0,
      5.0,
      120.0,
      25.0
    ],
    "left_thigh_roll_joint": [
      100.0,
      3.3,
      120.0,
      25.0
    ],
    "right_thigh_roll_joint": [
      100.0,
      3.3,
      120.0,
      25.0
    ],
    "left_arm_pitch_joint": [
      40.0,
      2.0,
      27.0,
      8.0
    ],
    "right_arm_pitch_joint": [
      40.0,
      2.0,
      27.0,
      8.0
    ],
    "left_thigh_pitch_joint": [
      100.0,
      3.3,
      120.0,
      25.0
    ],
    "right_thigh_pitch_joint": [
      100.0,
      3.3,
      120.0,
      25.0
    ],
    "left_arm_roll_joint": [
      40.0,
      2.0,
      27.0,
      8.0
    ],
    "right_arm_roll_joint": [
      40.0,
      2.0,
      27.0,
      8.0
    ],
    "left_knee_joint": [
      150.0,
      5.0,
      120.0,
      25.0
    ],
    "right_knee_joint": [
      150.0,
      5.0,
      120.0,
      25.0
    ],
    "left_arm_yaw_joint": [
      40.0,
      2.0,
      27.0,
      8.0
    ],
    "right_arm_yaw_joint": [
      40.0,
      2.0,
      27.0,
      8.0
    ],
    "left_ankle_pitch_joint": [
      40.0,
      2.5,
      27.0,
      8.0
    ],
    "right_ankle_pitch_joint": [
      40.0,
      2.5,
      27.0,
      8.0
    ],
    "left_elbow_pitch_joint": [
      30.0,
      1.5,
      27.0,
      8.0
    ],
    "right_elbow_pitch_joint": [
      30.0,
      1.5,
      27.0,
      8.0
    ],
    "left_ankle_roll_joint": [
      40.0,
      2.5,
      27.0,
      8.0
    ],
    "right_ankle_roll_joint": [
      40.0,
      2.5,
      27.0,
      8.0
    ],
    "left_elbow_yaw_joint": [
      20.0,
      1.0,
      27.0,
      8.0
    ],
    "right_elbow_yaw_joint": [
      20.0,
      1.0,
      27.0,
      8.0
    ]
  },
  "joint_position_limits_rad": [
    [
      -1.0,
      0.20000000298023224
    ],
    [
      -0.20000000298023224,
      1.0
    ],
    [
      -3.140000104904175,
      3.140000104904175
    ],
    [
      -0.20000000298023224,
      1.0
    ],
    [
      -1.0,
      0.20000000298023224
    ],
    [
      -3.140000104904175,
      1.5700000524520874
    ],
    [
      -3.140000104904175,
      1.5700000524520874
    ],
    [
      -2.0940001010894775,
      0.7853999733924866
    ],
    [
      -2.0940001010894775,
      0.7853999733924866
    ],
    [
      -0.25,
      3.140000104904175
    ],
    [
      -3.140000104904175,
      0.25
    ],
    [
      -0.20000000298023224,
      2.5
    ],
    [
      -0.20000000298023224,
      2.5
    ],
    [
      -1.5700000524520874,
      1.5700000524520874
    ],
    [
      -1.5700000524520874,
      1.5700000524520874
    ],
    [
      -0.6000000238418579,
      0.6000000238418579
    ],
    [
      -0.6000000238418579,
      0.6000000238418579
    ],
    [
      -0.6000000238418579,
      1.5700000524520874
    ],
    [
      -0.6000000238418579,
      1.5700000524520874
    ],
    [
      -0.5,
      0.5
    ],
    [
      -0.5,
      0.5
    ],
    [
      -1.5700000524520874,
      1.5700000524520874
    ],
    [
      -1.5700000524520874,
      1.5700000524520874
    ]
  ],
  "soft_joint_position_limits_rad": [
    [
      -0.940000057220459,
      0.14000001549720764
    ],
    [
      -0.14000001549720764,
      0.940000057220459
    ],
    [
      -2.8259999752044678,
      2.8259999752044678
    ],
    [
      -0.14000001549720764,
      0.940000057220459
    ],
    [
      -0.940000057220459,
      0.14000001549720764
    ],
    [
      -2.9045000076293945,
      1.3344998359680176
    ],
    [
      -2.9045000076293945,
      1.3344998359680176
    ],
    [
      -1.9500300884246826,
      0.6414299011230469
    ],
    [
      -1.9500300884246826,
      0.6414299011230469
    ],
    [
      -0.08050000667572021,
      2.9704999923706055
    ],
    [
      -2.9704999923706055,
      0.08050000667572021
    ],
    [
      -0.06500005722045898,
      2.365000009536743
    ],
    [
      -0.06500005722045898,
      2.365000009536743
    ],
    [
      -1.4129999876022339,
      1.4129999876022339
    ],
    [
      -1.4129999876022339,
      1.4129999876022339
    ],
    [
      -0.5400000214576721,
      0.5400000214576721
    ],
    [
      -0.5400000214576721,
      0.5400000214576721
    ],
    [
      -0.49150002002716064,
      1.4615000486373901
    ],
    [
      -0.49150002002716064,
      1.4615000486373901
    ],
    [
      -0.44999998807907104,
      0.44999998807907104
    ],
    [
      -0.44999998807907104,
      0.44999998807907104
    ],
    [
      -1.4129999876022339,
      1.4129999876022339
    ],
    [
      -1.4129999876022339,
      1.4129999876022339
    ]
  ],
  "geom_contype": [
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1
  ],
  "geom_conaffinity": [
    1,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0
  ],
  "geom_solref": [
    [
      0.02,
      1.0
    ],
    [
      0.001,
      2.0
    ],
    [
      0.001,
      2.0
    ],
    [
      0.001,
      2.0
    ],
    [
      0.001,
      2.0
    ],
    [
      0.001,
      2.0
    ],
    [
      0.001,
      2.0
    ],
    [
      0.001,
      2.0
    ],
    [
      0.001,
      2.0
    ],
    [
      0.001,
      2.0
    ],
    [
      0.001,
      2.0
    ],
    [
      0.001,
      2.0
    ],
    [
      0.001,
      2.0
    ],
    [
      0.001,
      2.0
    ],
    [
      0.001,
      2.0
    ],
    [
      0.001,
      2.0
    ],
    [
      0.001,
      2.0
    ],
    [
      0.001,
      2.0
    ],
    [
      0.001,
      2.0
    ],
    [
      0.001,
      2.0
    ],
    [
      0.001,
      2.0
    ],
    [
      0.001,
      2.0
    ],
    [
      0.001,
      2.0
    ],
    [
      0.001,
      2.0
    ],
    [
      0.001,
      2.0
    ]
  ],
  "geom_friction": [
    [
      1.0,
      0.005,
      0.0001
    ],
    [
      0.9,
      0.2,
      0.2
    ],
    [
      0.9,
      0.2,
      0.2
    ],
    [
      0.9,
      0.2,
      0.2
    ],
    [
      0.9,
      0.2,
      0.2
    ],
    [
      0.9,
      0.2,
      0.2
    ],
    [
      0.9,
      0.2,
      0.2
    ],
    [
      0.9,
      0.2,
      0.2
    ],
    [
      0.9,
      0.2,
      0.2
    ],
    [
      0.9,
      0.2,
      0.2
    ],
    [
      0.9,
      0.2,
      0.2
    ],
    [
      0.9,
      0.2,
      0.2
    ],
    [
      0.9,
      0.2,
      0.2
    ],
    [
      0.9,
      0.2,
      0.2
    ],
    [
      0.9,
      0.2,
      0.2
    ],
    [
      0.9,
      0.2,
      0.2
    ],
    [
      0.9,
      0.2,
      0.2
    ],
    [
      0.9,
      0.2,
      0.2
    ],
    [
      0.9,
      0.2,
      0.2
    ],
    [
      0.9,
      0.2,
      0.2
    ],
    [
      0.9,
      0.2,
      0.2
    ],
    [
      0.9,
      0.2,
      0.2
    ],
    [
      0.9,
      0.2,
      0.2
    ],
    [
      0.9,
      0.2,
      0.2
    ],
    [
      0.9,
      0.2,
      0.2
    ]
  ],
  "geom_solimp": [
    [
      0.9,
      0.95,
      0.001,
      0.5,
      2.0
    ],
    [
      0.9,
      0.95,
      0.001,
      0.5,
      2.0
    ],
    [
      0.9,
      0.95,
      0.001,
      0.5,
      2.0
    ],
    [
      0.9,
      0.95,
      0.001,
      0.5,
      2.0
    ],
    [
      0.9,
      0.95,
      0.001,
      0.5,
      2.0
    ],
    [
      0.9,
      0.95,
      0.001,
      0.5,
      2.0
    ],
    [
      0.9,
      0.95,
      0.001,
      0.5,
      2.0
    ],
    [
      0.9,
      0.95,
      0.001,
      0.5,
      2.0
    ],
    [
      0.9,
      0.95,
      0.001,
      0.5,
      2.0
    ],
    [
      0.9,
      0.95,
      0.001,
      0.5,
      2.0
    ],
    [
      0.9,
      0.95,
      0.001,
      0.5,
      2.0
    ],
    [
      0.9,
      0.95,
      0.001,
      0.5,
      2.0
    ],
    [
      0.9,
      0.95,
      0.001,
      0.5,
      2.0
    ],
    [
      0.9,
      0.95,
      0.001,
      0.5,
      2.0
    ],
    [
      0.9,
      0.95,
      0.001,
      0.5,
      2.0
    ],
    [
      0.9,
      0.95,
      0.001,
      0.5,
      2.0
    ],
    [
      0.9,
      0.95,
      0.001,
      0.5,
      2.0
    ],
    [
      0.9,
      0.95,
      0.001,
      0.5,
      2.0
    ],
    [
      0.9,
      0.95,
      0.001,
      0.5,
      2.0
    ],
    [
      0.9,
      0.95,
      0.001,
      0.5,
      2.0
    ],
    [
      0.9,
      0.95,
      0.001,
      0.5,
      2.0
    ],
    [
      0.9,
      0.95,
      0.001,
      0.5,
      2.0
    ],
    [
      0.9,
      0.95,
      0.001,
      0.5,
      2.0
    ],
    [
      0.9,
      0.95,
      0.001,
      0.5,
      2.0
    ],
    [
      0.9,
      0.95,
      0.001,
      0.5,
      2.0
    ]
  ],
  "compiled_solver": {
    "timestep_s": 0.0025,
    "solver": 2,
    "integrator": 3,
    "iterations": 100,
    "cone": 0,
    "gravity_mps2": [
      0.0,
      0.0,
      -9.81
    ]
  },
  "xml_sha256": "36a5c5eccfba758ef76173d92817b04320b5308bcec37403640101bdc3e3e6f3",
  "compiled_xml_sha256": "0a58ac102cb5ab8734ff844945987c1c26ffd5b25cdb7b52873f521177c8b0d0",
  "tracking_count": 1024,
  "tracking_partition_sha256": "5b653aad83c9abf3cbcb14748009093b85939c3186041e13b527d8370463637d",
  "rsi_pool": [
    {
      "clip": "g1_dance1_subject1_clip3",
      "source": "g1",
      "eligible": 114,
      "original_starts": 320,
      "max_root_lift_m": 0.0019999786745756865,
      "joint_lookahead_s": 0.02,
      "joint_margin_rad": 0.001,
      "joint_lookahead_rejected": 41,
      "geometric_eligible": 183,
      "velocity_rejected": 28,
      "tracking_eligible": 0,
      "normal_speed_max_mps": 0.14939954052475093,
      "eligible_frames": [
        83,
        84,
        85,
        86,
        108,
        109,
        110,
        111,
        115,
        116,
        117,
        118,
        119,
        120,
        122,
        123,
        124,
        125,
        126,
        127,
        128,
        129,
        130,
        131,
        132,
        133,
        134,
        135,
        136,
        137,
        138,
        139,
        140,
        141,
        142,
        143,
        144,
        145,
        146,
        147,
        148,
        149,
        150,
        151,
        152,
        153,
        154,
        155,
        156,
        157,
        158,
        159,
        160,
        161,
        162,
        163,
        164,
        165,
        166,
        167,
        168,
        169,
        170,
        171,
        172,
        173,
        174,
        175,
        176,
        177,
        178,
        179,
        180,
        181,
        182,
        183,
        184,
        185,
        186,
        187,
        188,
        189,
        190,
        191,
        192,
        196,
        197,
        198,
        199,
        200,
        201,
        331,
        332,
        333,
        334,
        335,
        336,
        337,
        343,
        344,
        345,
        346,
        347,
        362,
        363,
        364,
        365,
        366,
        367,
        368,
        369,
        370,
        375,
        376
      ]
    },
    {
      "clip": "g1_dance1_subject2_clip3",
      "source": "g1",
      "eligible": 0,
      "original_starts": 207,
      "max_root_lift_m": null,
      "joint_lookahead_s": 0.02,
      "joint_margin_rad": 0.001,
      "joint_lookahead_rejected": 1,
      "geometric_eligible": 8,
      "velocity_rejected": 7,
      "tracking_eligible": 0,
      "normal_speed_max_mps": null,
      "eligible_frames": []
    },
    {
      "clip": "g1_dance1_subject3_clip3",
      "source": "g1",
      "eligible": 10,
      "original_starts": 157,
      "max_root_lift_m": 0.0,
      "joint_lookahead_s": 0.02,
      "joint_margin_rad": 0.001,
      "joint_lookahead_rejected": 1,
      "geometric_eligible": 11,
      "velocity_rejected": 0,
      "tracking_eligible": 0,
      "normal_speed_max_mps": 0.08903391599250743,
      "eligible_frames": [
        237,
        238,
        239,
        240,
        241,
        242,
        243,
        244,
        245,
        246
      ]
    },
    {
      "clip": "g1_fallAndGetUp1_subject1_clip3",
      "source": "g1",
      "eligible": 70,
      "original_starts": 188,
      "max_root_lift_m": 0.0019290782511234283,
      "joint_lookahead_s": 0.02,
      "joint_margin_rad": 0.001,
      "joint_lookahead_rejected": 28,
      "geometric_eligible": 137,
      "velocity_rejected": 39,
      "tracking_eligible": 0,
      "normal_speed_max_mps": 0.14749255044349804,
      "eligible_frames": [
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        14,
        15,
        16,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        60,
        88,
        89,
        90,
        91,
        100,
        101,
        102,
        319,
        320,
        321,
        322,
        326,
        328,
        329,
        330,
        331,
        332,
        333,
        334,
        335,
        336,
        337,
        338,
        339,
        340,
        341,
        342,
        367,
        368,
        369,
        370,
        373,
        374,
        375,
        376,
        377,
        378,
        379,
        380,
        381,
        387,
        388,
        389
      ]
    },
    {
      "clip": "g1_fallAndGetUp1_subject4_clip3",
      "source": "g1",
      "eligible": 0,
      "original_starts": 153,
      "max_root_lift_m": null,
      "joint_lookahead_s": 0.02,
      "joint_margin_rad": 0.001,
      "joint_lookahead_rejected": 0,
      "geometric_eligible": 0,
      "velocity_rejected": 0,
      "tracking_eligible": 0,
      "normal_speed_max_mps": null,
      "eligible_frames": []
    },
    {
      "clip": "g1_fallAndGetUp1_subject5_clip3",
      "source": "g1",
      "eligible": 2,
      "original_starts": 130,
      "max_root_lift_m": 0.0,
      "joint_lookahead_s": 0.02,
      "joint_margin_rad": 0.001,
      "joint_lookahead_rejected": 3,
      "geometric_eligible": 8,
      "velocity_rejected": 3,
      "tracking_eligible": 0,
      "normal_speed_max_mps": 0.13500343275066987,
      "eligible_frames": [
        291,
        292
      ]
    },
    {
      "clip": "g1_fight1_subject2_clip3",
      "source": "g1",
      "eligible": 1,
      "original_starts": 340,
      "max_root_lift_m": 0.0020000170916318893,
      "joint_lookahead_s": 0.02,
      "joint_margin_rad": 0.001,
      "joint_lookahead_rejected": 1,
      "geometric_eligible": 9,
      "velocity_rejected": 7,
      "tracking_eligible": 0,
      "normal_speed_max_mps": 0.10905021868275344,
      "eligible_frames": [
        1
      ]
    },
    {
      "clip": "g1_fight1_subject3_clip3",
      "source": "g1",
      "eligible": 0,
      "original_starts": 337,
      "max_root_lift_m": null,
      "joint_lookahead_s": 0.02,
      "joint_margin_rad": 0.001,
      "joint_lookahead_rejected": 0,
      "geometric_eligible": 0,
      "velocity_rejected": 0,
      "tracking_eligible": 0,
      "normal_speed_max_mps": null,
      "eligible_frames": []
    },
    {
      "clip": "g1_fight1_subject5_clip3",
      "source": "g1",
      "eligible": 2,
      "original_starts": 376,
      "max_root_lift_m": 0.0,
      "joint_lookahead_s": 0.02,
      "joint_margin_rad": 0.001,
      "joint_lookahead_rejected": 6,
      "geometric_eligible": 16,
      "velocity_rejected": 8,
      "tracking_eligible": 0,
      "normal_speed_max_mps": 0.09863059232474918,
      "eligible_frames": [
        14,
        63
      ]
    },
    {
      "clip": "g1_jumps1_subject1_clip3",
      "source": "g1",
      "eligible": 30,
      "original_starts": 313,
      "max_root_lift_m": 0.0,
      "joint_lookahead_s": 0.02,
      "joint_margin_rad": 0.001,
      "joint_lookahead_rejected": 2,
      "geometric_eligible": 42,
      "velocity_rejected": 10,
      "tracking_eligible": 0,
      "normal_speed_max_mps": 0.11807671882510415,
      "eligible_frames": [
        26,
        174,
        258,
        274,
        275,
        276,
        277,
        278,
        279,
        280,
        281,
        282,
        284,
        286,
        288,
        290,
        292,
        294,
        296,
        298,
        300,
        311,
        312,
        313,
        314,
        315,
        316,
        317,
        318,
        319
      ]
    },
    {
      "clip": "g1_jumps1_subject2_clip3",
      "source": "g1",
      "eligible": 0,
      "original_starts": 358,
      "max_root_lift_m": null,
      "joint_lookahead_s": 0.02,
      "joint_margin_rad": 0.001,
      "joint_lookahead_rejected": 3,
      "geometric_eligible": 8,
      "velocity_rejected": 5,
      "tracking_eligible": 0,
      "normal_speed_max_mps": null,
      "eligible_frames": []
    },
    {
      "clip": "g1_jumps1_subject5_clip3",
      "source": "g1",
      "eligible": 0,
      "original_starts": 344,
      "max_root_lift_m": null,
      "joint_lookahead_s": 0.02,
      "joint_margin_rad": 0.001,
      "joint_lookahead_rejected": 5,
      "geometric_eligible": 6,
      "velocity_rejected": 1,
      "tracking_eligible": 0,
      "normal_speed_max_mps": null,
      "eligible_frames": []
    },
    {
      "clip": "g1_run1_subject2_clip3",
      "source": "g1",
      "eligible": 5,
      "original_starts": 344,
      "max_root_lift_m": 0.0,
      "joint_lookahead_s": 0.02,
      "joint_margin_rad": 0.001,
      "joint_lookahead_rejected": 10,
      "geometric_eligible": 29,
      "velocity_rejected": 14,
      "tracking_eligible": 0,
      "normal_speed_max_mps": 0.13675031932974357,
      "eligible_frames": [
        98,
        99,
        100,
        259,
        260
      ]
    },
    {
      "clip": "g1_run1_subject5_clip3",
      "source": "g1",
      "eligible": 0,
      "original_starts": 127,
      "max_root_lift_m": null,
      "joint_lookahead_s": 0.02,
      "joint_margin_rad": 0.001,
      "joint_lookahead_rejected": 4,
      "geometric_eligible": 5,
      "velocity_rejected": 1,
      "tracking_eligible": 0,
      "normal_speed_max_mps": null,
      "eligible_frames": []
    },
    {
      "clip": "g1_sprint1_subject2_clip3",
      "source": "g1",
      "eligible": 147,
      "original_starts": 306,
      "max_root_lift_m": 0.001999995205551386,
      "joint_lookahead_s": 0.02,
      "joint_margin_rad": 0.001,
      "joint_lookahead_rejected": 18,
      "geometric_eligible": 178,
      "velocity_rejected": 13,
      "tracking_eligible": 0,
      "normal_speed_max_mps": 0.14655554343129124,
      "eligible_frames": [
        151,
        152,
        193,
        194,
        195,
        196,
        244,
        245,
        246,
        247,
        248,
        249,
        250,
        251,
        252,
        253,
        254,
        255,
        256,
        257,
        258,
        262,
        263,
        264,
        265,
        266,
        267,
        268,
        269,
        270,
        271,
        272,
        273,
        274,
        275,
        276,
        277,
        278,
        279,
        280,
        281,
        282,
        283,
        284,
        285,
        286,
        287,
        288,
        289,
        290,
        291,
        292,
        293,
        294,
        295,
        296,
        297,
        298,
        299,
        300,
        301,
        302,
        303,
        304,
        305,
        306,
        307,
        308,
        309,
        310,
        311,
        312,
        313,
        314,
        315,
        316,
        317,
        318,
        319,
        320,
        321,
        322,
        323,
        324,
        325,
        326,
        327,
        328,
        329,
        330,
        331,
        332,
        333,
        334,
        335,
        336,
        337,
        338,
        339,
        340,
        341,
        342,
        343,
        344,
        345,
        346,
        347,
        348,
        349,
        350,
        351,
        353,
        354,
        355,
        356,
        357,
        358,
        359,
        360,
        361,
        362,
        363,
        364,
        365,
        366,
        367,
        368,
        369,
        370,
        371,
        372,
        373,
        374,
        375,
        376,
        377,
        378,
        379,
        380,
        381,
        382,
        383,
        384,
        385,
        389,
        390,
        391
      ]
    },
    {
      "clip": "g1_sprint1_subject4_clip3",
      "source": "g1",
      "eligible": 149,
      "original_starts": 379,
      "max_root_lift_m": 0.0019999679643660784,
      "joint_lookahead_s": 0.02,
      "joint_margin_rad": 0.001,
      "joint_lookahead_rejected": 11,
      "geometric_eligible": 178,
      "velocity_rejected": 18,
      "tracking_eligible": 79,
      "normal_speed_max_mps": 0.14864236534598466,
      "eligible_frames": [
        0,
        6,
        7,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        58,
        59,
        60,
        61,
        62,
        63,
        64,
        74,
        75,
        76,
        77,
        78,
        79,
        80,
        81,
        82,
        83,
        88,
        89,
        90,
        91,
        92,
        93,
        94,
        95,
        96,
        97,
        98,
        99,
        100,
        101,
        102,
        103,
        104,
        105,
        106,
        107,
        108,
        109,
        110,
        111,
        112,
        113,
        114,
        115,
        116,
        117,
        118,
        119,
        120,
        121,
        122,
        123,
        124,
        125,
        126,
        127,
        128,
        129,
        130,
        131,
        132,
        133,
        134,
        135,
        136,
        137,
        138,
        139,
        140,
        141,
        142,
        143,
        144,
        145,
        146,
        147,
        148,
        149,
        150,
        151,
        152,
        153,
        154,
        155,
        156,
        157,
        158,
        159,
        160,
        161,
        162,
        163,
        164,
        165,
        166,
        167,
        168,
        176,
        177,
        178,
        179,
        180,
        181,
        182,
        183,
        184,
        185,
        186,
        188,
        189,
        190,
        191,
        192
      ]
    },
    {
      "clip": "g1_walk1_subject1_clip3",
      "source": "g1",
      "eligible": 104,
      "original_starts": 392,
      "max_root_lift_m": 0.0,
      "joint_lookahead_s": 0.02,
      "joint_margin_rad": 0.001,
      "joint_lookahead_rejected": 56,
      "geometric_eligible": 173,
      "velocity_rejected": 13,
      "tracking_eligible": 7,
      "normal_speed_max_mps": 0.1430332513635609,
      "eligible_frames": [
        3,
        4,
        7,
        36,
        39,
        95,
        96,
        170,
        171,
        172,
        173,
        174,
        175,
        176,
        177,
        178,
        179,
        180,
        181,
        182,
        183,
        184,
        185,
        186,
        187,
        188,
        189,
        190,
        191,
        192,
        193,
        194,
        195,
        196,
        197,
        198,
        199,
        200,
        201,
        202,
        203,
        204,
        205,
        206,
        207,
        208,
        209,
        210,
        211,
        212,
        213,
        214,
        215,
        216,
        217,
        218,
        219,
        220,
        221,
        222,
        223,
        224,
        225,
        226,
        227,
        228,
        229,
        230,
        231,
        232,
        233,
        234,
        235,
        236,
        237,
        238,
        239,
        240,
        241,
        242,
        243,
        244,
        245,
        246,
        247,
        248,
        249,
        250,
        251,
        252,
        253,
        254,
        260,
        279,
        280,
        281,
        282,
        355,
        356,
        357,
        358,
        359,
        360,
        361
      ]
    },
    {
      "clip": "g1_walk1_subject2_clip3",
      "source": "g1",
      "eligible": 351,
      "original_starts": 392,
      "max_root_lift_m": 0.0,
      "joint_lookahead_s": 0.02,
      "joint_margin_rad": 0.001,
      "joint_lookahead_rejected": 7,
      "geometric_eligible": 380,
      "velocity_rejected": 22,
      "tracking_eligible": 127,
      "normal_speed_max_mps": 0.14876935331450417,
      "eligible_frames": [
        3,
        7,
        8,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        45,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        60,
        61,
        62,
        63,
        64,
        65,
        66,
        67,
        68,
        69,
        70,
        71,
        72,
        73,
        74,
        75,
        76,
        77,
        78,
        79,
        80,
        81,
        82,
        83,
        84,
        85,
        86,
        87,
        88,
        89,
        90,
        91,
        92,
        93,
        94,
        95,
        96,
        97,
        98,
        99,
        100,
        101,
        102,
        103,
        104,
        105,
        106,
        107,
        108,
        109,
        110,
        111,
        112,
        113,
        114,
        115,
        116,
        117,
        118,
        119,
        120,
        125,
        126,
        127,
        128,
        129,
        130,
        131,
        132,
        133,
        134,
        138,
        139,
        140,
        144,
        145,
        146,
        154,
        155,
        156,
        157,
        158,
        159,
        160,
        161,
        162,
        163,
        164,
        165,
        166,
        167,
        168,
        169,
        170,
        171,
        172,
        173,
        174,
        175,
        176,
        177,
        178,
        179,
        180,
        181,
        182,
        183,
        184,
        185,
        186,
        187,
        188,
        189,
        190,
        191,
        192,
        193,
        194,
        195,
        196,
        197,
        198,
        199,
        200,
        201,
        202,
        203,
        204,
        205,
        206,
        207,
        208,
        209,
        210,
        211,
        212,
        213,
        214,
        215,
        216,
        217,
        218,
        219,
        220,
        221,
        222,
        223,
        224,
        225,
        226,
        227,
        228,
        229,
        230,
        231,
        232,
        235,
        236,
        237,
        238,
        239,
        240,
        241,
        242,
        243,
        244,
        245,
        246,
        247,
        248,
        249,
        252,
        253,
        254,
        255,
        256,
        257,
        258,
        259,
        260,
        261,
        262,
        263,
        264,
        265,
        266,
        267,
        268,
        269,
        270,
        271,
        272,
        273,
        274,
        275,
        276,
        277,
        278,
        279,
        281,
        282,
        283,
        284,
        285,
        286,
        287,
        288,
        289,
        290,
        291,
        292,
        293,
        294,
        295,
        296,
        297,
        298,
        299,
        300,
        301,
        302,
        303,
        304,
        305,
        306,
        308,
        309,
        310,
        311,
        312,
        313,
        314,
        315,
        316,
        317,
        318,
        319,
        320,
        321,
        322,
        323,
        324,
        325,
        326,
        327,
        328,
        329,
        330,
        331,
        332,
        333,
        334,
        335,
        336,
        337,
        338,
        339,
        340,
        341,
        342,
        343,
        344,
        345,
        346,
        347,
        348,
        349,
        350,
        351,
        352,
        353,
        354,
        355,
        356,
        357,
        358,
        359,
        360,
        361,
        362,
        363,
        364,
        365,
        366,
        367,
        368,
        369,
        370,
        371,
        372,
        373,
        374,
        375,
        376,
        377,
        378,
        379,
        380,
        389,
        390,
        391
      ]
    },
    {
      "clip": "g1_walk1_subject5_clip3",
      "source": "g1",
      "eligible": 36,
      "original_starts": 392,
      "max_root_lift_m": 0.0018110821256414056,
      "joint_lookahead_s": 0.02,
      "joint_margin_rad": 0.001,
      "joint_lookahead_rejected": 54,
      "geometric_eligible": 130,
      "velocity_rejected": 40,
      "tracking_eligible": 13,
      "normal_speed_max_mps": 0.14463494078018915,
      "eligible_frames": [
        32,
        33,
        34,
        57,
        58,
        59,
        98,
        100,
        101,
        102,
        103,
        144,
        145,
        198,
        199,
        201,
        223,
        224,
        225,
        226,
        303,
        304,
        305,
        306,
        307,
        330,
        331,
        332,
        353,
        354,
        374,
        375,
        376,
        377,
        378,
        379
      ]
    },
    {
      "clip": "g1_walk2_subject1_clip3",
      "source": "g1",
      "eligible": 120,
      "original_starts": 392,
      "max_root_lift_m": 0.0,
      "joint_lookahead_s": 0.02,
      "joint_margin_rad": 0.001,
      "joint_lookahead_rejected": 56,
      "geometric_eligible": 206,
      "velocity_rejected": 30,
      "tracking_eligible": 77,
      "normal_speed_max_mps": 0.1481767628218937,
      "eligible_frames": [
        3,
        4,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        45,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        64,
        80,
        81,
        82,
        83,
        84,
        86,
        87,
        88,
        93,
        133,
        134,
        140,
        141,
        142,
        143,
        144,
        145,
        146,
        147,
        148,
        157,
        158,
        159,
        160,
        188,
        189,
        190,
        191,
        214,
        215,
        216,
        217,
        218,
        219,
        220,
        221,
        222,
        223,
        224,
        248,
        249,
        250,
        251,
        252,
        253,
        254,
        262,
        263,
        264,
        265,
        266,
        268,
        271,
        292,
        293,
        294,
        295,
        296,
        297,
        338,
        339,
        371,
        372
      ]
    },
    {
      "clip": "g1_walk2_subject3_clip3",
      "source": "g1",
      "eligible": 86,
      "original_starts": 392,
      "max_root_lift_m": 0.0,
      "joint_lookahead_s": 0.02,
      "joint_margin_rad": 0.001,
      "joint_lookahead_rejected": 54,
      "geometric_eligible": 153,
      "velocity_rejected": 13,
      "tracking_eligible": 51,
      "normal_speed_max_mps": 0.1272731352055922,
      "eligible_frames": [
        3,
        4,
        5,
        6,
        7,
        8,
        80,
        81,
        82,
        83,
        84,
        85,
        86,
        87,
        88,
        89,
        90,
        91,
        92,
        93,
        94,
        95,
        96,
        97,
        98,
        99,
        100,
        101,
        127,
        128,
        129,
        130,
        131,
        132,
        133,
        134,
        135,
        136,
        137,
        138,
        139,
        140,
        141,
        142,
        143,
        144,
        145,
        146,
        147,
        148,
        149,
        150,
        151,
        152,
        153,
        154,
        155,
        156,
        157,
        158,
        159,
        160,
        161,
        162,
        163,
        164,
        165,
        166,
        167,
        168,
        170,
        171,
        172,
        173,
        174,
        175,
        176,
        206,
        207,
        208,
        209,
        210,
        211,
        212,
        213,
        319
      ]
    },
    {
      "clip": "g1_walk2_subject4_clip3",
      "source": "g1",
      "eligible": 183,
      "original_starts": 377,
      "max_root_lift_m": 0.0019999807700514793,
      "joint_lookahead_s": 0.02,
      "joint_margin_rad": 0.001,
      "joint_lookahead_rejected": 68,
      "geometric_eligible": 271,
      "velocity_rejected": 20,
      "tracking_eligible": 0,
      "normal_speed_max_mps": 0.1487947890569247,
      "eligible_frames": [
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        58,
        59,
        60,
        61,
        62,
        63,
        64,
        65,
        66,
        67,
        68,
        69,
        70,
        73,
        74,
        75,
        76,
        78,
        79,
        80,
        81,
        82,
        83,
        84,
        85,
        86,
        87,
        88,
        89,
        90,
        91,
        92,
        93,
        94,
        95,
        110,
        111,
        112,
        113,
        114,
        115,
        116,
        122,
        123,
        124,
        125,
        126,
        127,
        128,
        129,
        130,
        131,
        132,
        133,
        134,
        135,
        136,
        137,
        138,
        139,
        140,
        141,
        142,
        143,
        144,
        145,
        146,
        147,
        148,
        149,
        150,
        151,
        152,
        153,
        154,
        155,
        156,
        193,
        194,
        195,
        196,
        197,
        198,
        199,
        200,
        201,
        202,
        218,
        219,
        220,
        221,
        222,
        223,
        224,
        225,
        226,
        227,
        228,
        229,
        234,
        235,
        236,
        237,
        238,
        239,
        240,
        260,
        281,
        282,
        283,
        284,
        285,
        286,
        287,
        288,
        304,
        305,
        306,
        307,
        308,
        309,
        320,
        322,
        323,
        324,
        325,
        326,
        343,
        344,
        345,
        346,
        347,
        348,
        349,
        350,
        351,
        352,
        353,
        358,
        359,
        360,
        361,
        362,
        363,
        379,
        380,
        381,
        382,
        383,
        384,
        385,
        386,
        387
      ]
    },
    {
      "clip": "rpo_114_08",
      "source": "official_rpo",
      "eligible": 12,
      "original_starts": 435,
      "max_root_lift_m": 0.02023807168006897,
      "joint_lookahead_s": 0.02,
      "joint_margin_rad": 0.001,
      "joint_lookahead_rejected": 1,
      "geometric_eligible": 23,
      "velocity_rejected": 10,
      "tracking_eligible": 10,
      "normal_speed_max_mps": 0.12476524495724414,
      "eligible_frames": [
        70,
        71,
        72,
        73,
        74,
        75,
        76,
        77,
        78,
        79,
        498,
        526
      ]
    },
    {
      "clip": "rpo_114_09",
      "source": "official_rpo",
      "eligible": 69,
      "original_starts": 774,
      "max_root_lift_m": 0.0,
      "joint_lookahead_s": 0.02,
      "joint_margin_rad": 0.001,
      "joint_lookahead_rejected": 15,
      "geometric_eligible": 94,
      "velocity_rejected": 10,
      "tracking_eligible": 17,
      "normal_speed_max_mps": 0.14718967526243504,
      "eligible_frames": [
        3,
        4,
        5,
        6,
        18,
        34,
        360,
        361,
        381,
        382,
        384,
        385,
        386,
        387,
        388,
        389,
        390,
        623,
        624,
        625,
        626,
        627,
        628,
        629,
        647,
        648,
        649,
        650,
        651,
        652,
        653,
        655,
        656,
        657,
        658,
        659,
        660,
        661,
        662,
        663,
        672,
        673,
        674,
        675,
        676,
        695,
        696,
        697,
        698,
        699,
        700,
        701,
        702,
        703,
        704,
        705,
        706,
        707,
        708,
        709,
        734,
        735,
        736,
        737,
        738,
        739,
        740,
        741,
        773
      ]
    },
    {
      "clip": "rpo_127_04",
      "source": "official_rpo",
      "eligible": 0,
      "original_starts": 1,
      "max_root_lift_m": null,
      "joint_lookahead_s": 0.02,
      "joint_margin_rad": 0.001,
      "joint_lookahead_rejected": 0,
      "geometric_eligible": 0,
      "velocity_rejected": 0,
      "tracking_eligible": 0,
      "normal_speed_max_mps": null,
      "eligible_frames": []
    },
    {
      "clip": "rpo_36_01",
      "source": "official_rpo",
      "eligible": 46,
      "original_starts": 113,
      "max_root_lift_m": 0.021992022171616554,
      "joint_lookahead_s": 0.02,
      "joint_margin_rad": 0.001,
      "joint_lookahead_rejected": 0,
      "geometric_eligible": 63,
      "velocity_rejected": 17,
      "tracking_eligible": 0,
      "normal_speed_max_mps": 0.13336111476340784,
      "eligible_frames": [
        71,
        72,
        73,
        74,
        75,
        76,
        77,
        78,
        79,
        80,
        81,
        140,
        150,
        151,
        152,
        153,
        154,
        155,
        156,
        157,
        158,
        178,
        179,
        180,
        181,
        182,
        183,
        184,
        185,
        186,
        187,
        188,
        201,
        204,
        205,
        206,
        207,
        208,
        209,
        210,
        211,
        212,
        213,
        214,
        215,
        216
      ]
    },
    {
      "clip": "rpo_36_11",
      "source": "official_rpo",
      "eligible": 96,
      "original_starts": 1066,
      "max_root_lift_m": 0.0219456497579813,
      "joint_lookahead_s": 0.02,
      "joint_margin_rad": 0.001,
      "joint_lookahead_rejected": 7,
      "geometric_eligible": 185,
      "velocity_rejected": 82,
      "tracking_eligible": 14,
      "normal_speed_max_mps": 0.1487074913323966,
      "eligible_frames": [
        213,
        214,
        215,
        216,
        217,
        218,
        219,
        232,
        233,
        234,
        235,
        236,
        237,
        238,
        239,
        240,
        241,
        304,
        313,
        314,
        315,
        316,
        317,
        321,
        322,
        323,
        324,
        325,
        326,
        327,
        328,
        330,
        334,
        335,
        336,
        337,
        338,
        339,
        340,
        341,
        342,
        343,
        344,
        345,
        346,
        347,
        384,
        385,
        386,
        387,
        388,
        396,
        397,
        398,
        399,
        400,
        461,
        462,
        463,
        464,
        465,
        466,
        467,
        468,
        469,
        470,
        471,
        472,
        473,
        474,
        597,
        667,
        668,
        669,
        680,
        1027,
        1028,
        1029,
        1030,
        1031,
        1032,
        1033,
        1034,
        1051,
        1052,
        1053,
        1054,
        1055,
        1056,
        1314,
        1315,
        1316,
        1317,
        1655,
        1656,
        1657
      ]
    },
    {
      "clip": "rpo_B10_-__Walk_turn_left_45_stageii",
      "source": "official_rpo",
      "eligible": 6,
      "original_starts": 13,
      "max_root_lift_m": 0.0036081799771636724,
      "joint_lookahead_s": 0.02,
      "joint_margin_rad": 0.001,
      "joint_lookahead_rejected": 0,
      "geometric_eligible": 13,
      "velocity_rejected": 7,
      "tracking_eligible": 0,
      "normal_speed_max_mps": 0.10923854966130817,
      "eligible_frames": [
        285,
        286,
        287,
        288,
        289,
        290
      ]
    },
    {
      "clip": "rpo_B11_-__Walk_turn_left_135_stageii",
      "source": "official_rpo",
      "eligible": 10,
      "original_starts": 14,
      "max_root_lift_m": 0.02008538879454136,
      "joint_lookahead_s": 0.02,
      "joint_margin_rad": 0.001,
      "joint_lookahead_rejected": 0,
      "geometric_eligible": 14,
      "velocity_rejected": 4,
      "tracking_eligible": 0,
      "normal_speed_max_mps": 0.13797726363764667,
      "eligible_frames": [
        243,
        244,
        245,
        246,
        268,
        271,
        294,
        295,
        296,
        297
      ]
    },
    {
      "clip": "rpo_B13_-__Walk_turn_right_90_stageii",
      "source": "official_rpo",
      "eligible": 156,
      "original_starts": 229,
      "max_root_lift_m": 0.021529648452997208,
      "joint_lookahead_s": 0.02,
      "joint_margin_rad": 0.001,
      "joint_lookahead_rejected": 3,
      "geometric_eligible": 209,
      "velocity_rejected": 50,
      "tracking_eligible": 0,
      "normal_speed_max_mps": 0.1446504917512562,
      "eligible_frames": [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        45,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        60,
        64,
        65,
        66,
        67,
        68,
        69,
        70,
        71,
        72,
        73,
        74,
        75,
        76,
        77,
        78,
        79,
        80,
        81,
        82,
        83,
        84,
        93,
        94,
        95,
        96,
        97,
        98,
        99,
        100,
        101,
        102,
        103,
        104,
        105,
        106,
        107,
        108,
        109,
        110,
        111,
        112,
        113,
        121,
        122,
        123,
        124,
        125,
        126,
        127,
        128,
        129,
        130,
        131,
        132,
        133,
        148,
        149,
        150,
        151,
        152,
        153,
        154,
        155,
        156,
        160,
        161,
        162,
        163,
        165,
        176,
        177,
        178,
        179,
        184,
        185,
        190,
        191,
        192,
        193,
        199,
        201,
        202,
        203,
        204,
        205,
        206,
        210,
        211,
        212,
        227,
        228,
        229,
        230,
        235,
        236,
        237,
        238,
        239,
        260
      ]
    },
    {
      "clip": "rpo_B14_-__Walk_turn_right_45_t2_stageii",
      "source": "official_rpo",
      "eligible": 104,
      "original_starts": 143,
      "max_root_lift_m": 0.020397944375872612,
      "joint_lookahead_s": 0.02,
      "joint_margin_rad": 0.001,
      "joint_lookahead_rejected": 0,
      "geometric_eligible": 142,
      "velocity_rejected": 38,
      "tracking_eligible": 0,
      "normal_speed_max_mps": 0.14825249313373162,
      "eligible_frames": [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        41,
        42,
        43,
        44,
        45,
        46,
        47,
        48,
        55,
        56,
        57,
        58,
        59,
        60,
        61,
        62,
        63,
        64,
        65,
        66,
        67,
        68,
        69,
        70,
        71,
        72,
        74,
        76,
        82,
        83,
        84,
        86,
        87,
        88,
        89,
        90,
        91,
        92,
        93,
        94,
        95,
        96,
        97,
        109,
        112,
        113,
        114,
        115,
        116,
        117,
        118,
        119,
        120,
        121,
        122,
        123,
        144,
        145,
        146,
        147,
        196,
        197,
        198,
        199,
        221,
        222,
        223
      ]
    },
    {
      "clip": "rpo_B15_-__Walk_turn_around_stageii",
      "source": "official_rpo",
      "eligible": 5,
      "original_starts": 7,
      "max_root_lift_m": 0.018899673596024513,
      "joint_lookahead_s": 0.02,
      "joint_margin_rad": 0.001,
      "joint_lookahead_rejected": 0,
      "geometric_eligible": 7,
      "velocity_rejected": 2,
      "tracking_eligible": 0,
      "normal_speed_max_mps": 0.1271776023096718,
      "eligible_frames": [
        291,
        292,
        293,
        317,
        318
      ]
    },
    {
      "clip": "rpo_B4_-_Stand_to_Walk_backwards_stageii",
      "source": "official_rpo",
      "eligible": 0,
      "original_starts": 1,
      "max_root_lift_m": null,
      "joint_lookahead_s": 0.02,
      "joint_margin_rad": 0.001,
      "joint_lookahead_rejected": 0,
      "geometric_eligible": 1,
      "velocity_rejected": 1,
      "tracking_eligible": 0,
      "normal_speed_max_mps": null,
      "eligible_frames": []
    },
    {
      "clip": "rpo_B5_-__Walk_backwards_stageii",
      "source": "official_rpo",
      "eligible": 33,
      "original_starts": 108,
      "max_root_lift_m": 0.021808993071317673,
      "joint_lookahead_s": 0.02,
      "joint_margin_rad": 0.001,
      "joint_lookahead_rejected": 0,
      "geometric_eligible": 101,
      "velocity_rejected": 68,
      "tracking_eligible": 0,
      "normal_speed_max_mps": 0.14196069787896384,
      "eligible_frames": [
        66,
        67,
        68,
        69,
        70,
        81,
        87,
        88,
        89,
        91,
        92,
        93,
        114,
        119,
        120,
        121,
        122,
        140,
        142,
        147,
        148,
        154,
        155,
        156,
        157,
        158,
        166,
        169,
        224,
        225,
        245,
        246,
        271
      ]
    },
    {
      "clip": "rpo_B9_-__Walk_turn_left_90_stageii",
      "source": "official_rpo",
      "eligible": 5,
      "original_starts": 7,
      "max_root_lift_m": 0.018748613074421883,
      "joint_lookahead_s": 0.02,
      "joint_margin_rad": 0.001,
      "joint_lookahead_rejected": 0,
      "geometric_eligible": 7,
      "velocity_rejected": 2,
      "tracking_eligible": 0,
      "normal_speed_max_mps": 0.1279541372804039,
      "eligible_frames": [
        223,
        224,
        225,
        226,
        227
      ]
    },
    {
      "clip": "rpo_C12_-_run_turn_left_45_stageii",
      "source": "official_rpo",
      "eligible": 0,
      "original_starts": 11,
      "max_root_lift_m": null,
      "joint_lookahead_s": 0.02,
      "joint_margin_rad": 0.001,
      "joint_lookahead_rejected": 0,
      "geometric_eligible": 2,
      "velocity_rejected": 2,
      "tracking_eligible": 0,
      "normal_speed_max_mps": null,
      "eligible_frames": []
    },
    {
      "clip": "rpo_C17_-_run_change_direction_stageii",
      "source": "official_rpo",
      "eligible": 1,
      "original_starts": 67,
      "max_root_lift_m": 0.02094338834285736,
      "joint_lookahead_s": 0.02,
      "joint_margin_rad": 0.001,
      "joint_lookahead_rejected": 0,
      "geometric_eligible": 13,
      "velocity_rejected": 12,
      "tracking_eligible": 0,
      "normal_speed_max_mps": 0.08836470190685453,
      "eligible_frames": [
        94
      ]
    },
    {
      "clip": "rpo_C1_-_stand_to_run_stageii",
      "source": "official_rpo",
      "eligible": 100,
      "original_starts": 136,
      "max_root_lift_m": 0.021625429391860962,
      "joint_lookahead_s": 0.02,
      "joint_margin_rad": 0.001,
      "joint_lookahead_rejected": 0,
      "geometric_eligible": 107,
      "velocity_rejected": 7,
      "tracking_eligible": 0,
      "normal_speed_max_mps": 0.14340187104558202,
      "eligible_frames": [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        45,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        60,
        61,
        62,
        63,
        64,
        65,
        66,
        67,
        68,
        69,
        70,
        71,
        72,
        73,
        74,
        75,
        76,
        77,
        78,
        79,
        80,
        81,
        82,
        83,
        84,
        85,
        86,
        87,
        88,
        89,
        90,
        91,
        92,
        93,
        94,
        95,
        96,
        97,
        107,
        110
      ]
    },
    {
      "clip": "rpo_C3_-_run_stageii",
      "source": "official_rpo",
      "eligible": 0,
      "original_starts": 36,
      "max_root_lift_m": null,
      "joint_lookahead_s": 0.02,
      "joint_margin_rad": 0.001,
      "joint_lookahead_rejected": 1,
      "geometric_eligible": 6,
      "velocity_rejected": 5,
      "tracking_eligible": 0,
      "normal_speed_max_mps": null,
      "eligible_frames": []
    },
    {
      "clip": "rpo_C4_-_run_to_walk_a_stageii",
      "source": "official_rpo",
      "eligible": 14,
      "original_starts": 72,
      "max_root_lift_m": 0.01279268879443407,
      "joint_lookahead_s": 0.02,
      "joint_margin_rad": 0.001,
      "joint_lookahead_rejected": 0,
      "geometric_eligible": 48,
      "velocity_rejected": 34,
      "tracking_eligible": 0,
      "normal_speed_max_mps": 0.1406593510426886,
      "eligible_frames": [
        107,
        108,
        109,
        133,
        134,
        135,
        136,
        137,
        138,
        149,
        150,
        152,
        175,
        176
      ]
    },
    {
      "clip": "rpo_C5_-_walk_to_run_stageii",
      "source": "official_rpo",
      "eligible": 9,
      "original_starts": 47,
      "max_root_lift_m": 0.01850222982466221,
      "joint_lookahead_s": 0.02,
      "joint_margin_rad": 0.001,
      "joint_lookahead_rejected": 0,
      "geometric_eligible": 25,
      "velocity_rejected": 16,
      "tracking_eligible": 0,
      "normal_speed_max_mps": 0.14396789904815765,
      "eligible_frames": [
        10,
        11,
        12,
        36,
        37,
        60,
        61,
        152,
        153
      ]
    },
    {
      "clip": "rpo_move_back",
      "source": "official_rpo",
      "eligible": 271,
      "original_starts": 304,
      "max_root_lift_m": 0.0038758968003094196,
      "joint_lookahead_s": 0.02,
      "joint_margin_rad": 0.001,
      "joint_lookahead_rejected": 0,
      "geometric_eligible": 298,
      "velocity_rejected": 27,
      "tracking_eligible": 62,
      "normal_speed_max_mps": 0.14790810872584087,
      "eligible_frames": [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        45,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        60,
        61,
        62,
        63,
        64,
        65,
        66,
        67,
        68,
        69,
        70,
        77,
        78,
        79,
        80,
        81,
        82,
        83,
        84,
        85,
        86,
        87,
        88,
        89,
        90,
        91,
        92,
        93,
        94,
        95,
        96,
        97,
        98,
        99,
        100,
        101,
        102,
        103,
        104,
        105,
        106,
        107,
        108,
        109,
        110,
        111,
        112,
        113,
        114,
        115,
        116,
        117,
        118,
        119,
        120,
        121,
        122,
        123,
        124,
        125,
        126,
        127,
        128,
        131,
        132,
        133,
        134,
        135,
        136,
        137,
        138,
        139,
        140,
        141,
        142,
        143,
        144,
        145,
        146,
        147,
        148,
        149,
        150,
        151,
        152,
        153,
        154,
        155,
        156,
        157,
        158,
        159,
        160,
        161,
        162,
        163,
        164,
        165,
        166,
        167,
        170,
        171,
        172,
        173,
        174,
        175,
        176,
        177,
        178,
        179,
        180,
        181,
        182,
        183,
        184,
        185,
        186,
        187,
        188,
        189,
        190,
        191,
        192,
        193,
        194,
        195,
        196,
        197,
        198,
        199,
        200,
        201,
        202,
        203,
        204,
        207,
        208,
        209,
        210,
        211,
        212,
        213,
        214,
        215,
        216,
        217,
        218,
        219,
        220,
        221,
        222,
        223,
        224,
        225,
        226,
        227,
        228,
        229,
        230,
        239,
        240,
        241,
        242,
        243,
        244,
        245,
        246,
        247,
        248,
        249,
        250,
        251,
        252,
        253,
        254,
        255,
        256,
        257,
        258,
        259,
        260,
        261,
        262,
        263,
        264,
        272,
        273,
        274,
        275,
        276,
        277,
        278,
        279,
        280,
        281,
        282,
        283,
        284,
        285,
        286,
        287,
        288,
        289,
        290,
        291,
        292,
        293,
        294,
        295,
        302,
        303
      ]
    },
    {
      "clip": "rpo_move_l",
      "source": "official_rpo",
      "eligible": 97,
      "original_starts": 272,
      "max_root_lift_m": 0.00035552948247641325,
      "joint_lookahead_s": 0.02,
      "joint_margin_rad": 0.001,
      "joint_lookahead_rejected": 0,
      "geometric_eligible": 111,
      "velocity_rejected": 14,
      "tracking_eligible": 30,
      "normal_speed_max_mps": 0.13400432594949468,
      "eligible_frames": [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        40,
        41,
        42,
        43,
        44,
        45,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        60,
        61,
        62,
        63,
        71,
        72,
        73,
        74,
        75,
        86,
        87,
        88,
        89,
        90,
        91,
        92,
        93,
        103,
        104,
        105,
        106,
        107,
        108,
        109,
        110,
        111,
        112,
        116,
        117,
        118,
        133,
        134,
        135,
        136,
        156,
        157,
        158,
        159,
        160
      ]
    },
    {
      "clip": "rpo_move_r",
      "source": "official_rpo",
      "eligible": 113,
      "original_starts": 242,
      "max_root_lift_m": 0.004947061184793711,
      "joint_lookahead_s": 0.02,
      "joint_margin_rad": 0.001,
      "joint_lookahead_rejected": 0,
      "geometric_eligible": 128,
      "velocity_rejected": 15,
      "tracking_eligible": 0,
      "normal_speed_max_mps": 0.1487628561516832,
      "eligible_frames": [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        45,
        46,
        47,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        93,
        96,
        97,
        98,
        99,
        100,
        101,
        102,
        103,
        104,
        134,
        135,
        138,
        139,
        140,
        141,
        142,
        143,
        144,
        145,
        176,
        177,
        181,
        182,
        183,
        184,
        185,
        186,
        187,
        188,
        189,
        190,
        191,
        192,
        193,
        194,
        195,
        196,
        197,
        198,
        199,
        200,
        201,
        202,
        211,
        212,
        213,
        214,
        215,
        216,
        217,
        218,
        225,
        226,
        227,
        228,
        229,
        230,
        231,
        232,
        233,
        234,
        235,
        236,
        237,
        238,
        239,
        240,
        241
      ]
    },
    {
      "clip": "rpo_run_start_180_R_001__A345",
      "source": "official_rpo",
      "eligible": 3,
      "original_starts": 114,
      "max_root_lift_m": 0.0,
      "joint_lookahead_s": 0.02,
      "joint_margin_rad": 0.001,
      "joint_lookahead_rejected": 0,
      "geometric_eligible": 5,
      "velocity_rejected": 2,
      "tracking_eligible": 0,
      "normal_speed_max_mps": 0.13470170948044125,
      "eligible_frames": [
        26,
        27,
        125
      ]
    },
    {
      "clip": "rpo_run_start_180_R_001__A345_M",
      "source": "official_rpo",
      "eligible": 4,
      "original_starts": 115,
      "max_root_lift_m": 0.0,
      "joint_lookahead_s": 0.02,
      "joint_margin_rad": 0.001,
      "joint_lookahead_rejected": 0,
      "geometric_eligible": 6,
      "velocity_rejected": 2,
      "tracking_eligible": 0,
      "normal_speed_max_mps": 0.13790660873340807,
      "eligible_frames": [
        9,
        10,
        27,
        125
      ]
    },
    {
      "clip": "rpo_run_stop_180_R_001__A345",
      "source": "official_rpo",
      "eligible": 110,
      "original_starts": 158,
      "max_root_lift_m": 0.0,
      "joint_lookahead_s": 0.02,
      "joint_margin_rad": 0.001,
      "joint_lookahead_rejected": 0,
      "geometric_eligible": 112,
      "velocity_rejected": 2,
      "tracking_eligible": 0,
      "normal_speed_max_mps": 0.10800908574350976,
      "eligible_frames": [
        82,
        83,
        84,
        87,
        88,
        89,
        90,
        91,
        92,
        93,
        94,
        95,
        96,
        97,
        98,
        99,
        100,
        101,
        102,
        103,
        104,
        105,
        106,
        107,
        108,
        109,
        110,
        111,
        112,
        113,
        114,
        115,
        116,
        117,
        118,
        119,
        120,
        121,
        122,
        123,
        124,
        125,
        126,
        127,
        128,
        129,
        130,
        131,
        132,
        133,
        134,
        135,
        136,
        137,
        138,
        139,
        140,
        141,
        142,
        143,
        144,
        145,
        146,
        147,
        148,
        149,
        150,
        151,
        152,
        153,
        154,
        155,
        156,
        157,
        158,
        159,
        160,
        161,
        162,
        163,
        164,
        165,
        166,
        167,
        168,
        169,
        170,
        171,
        172,
        173,
        174,
        175,
        176,
        177,
        178,
        179,
        180,
        181,
        182,
        183,
        184,
        185,
        186,
        187,
        188,
        189,
        190,
        191,
        192,
        193
      ]
    },
    {
      "clip": "rpo_run_stop_180_R_001__A345_M",
      "source": "official_rpo",
      "eligible": 110,
      "original_starts": 158,
      "max_root_lift_m": 0.0,
      "joint_lookahead_s": 0.02,
      "joint_margin_rad": 0.001,
      "joint_lookahead_rejected": 0,
      "geometric_eligible": 112,
      "velocity_rejected": 2,
      "tracking_eligible": 0,
      "normal_speed_max_mps": 0.1128801129645981,
      "eligible_frames": [
        82,
        83,
        84,
        87,
        88,
        89,
        90,
        91,
        92,
        93,
        94,
        95,
        96,
        97,
        98,
        99,
        100,
        101,
        102,
        103,
        104,
        105,
        106,
        107,
        108,
        109,
        110,
        111,
        112,
        113,
        114,
        115,
        116,
        117,
        118,
        119,
        120,
        121,
        122,
        123,
        124,
        125,
        126,
        127,
        128,
        129,
        130,
        131,
        132,
        133,
        134,
        135,
        136,
        137,
        138,
        139,
        140,
        141,
        142,
        143,
        144,
        145,
        146,
        147,
        148,
        149,
        150,
        151,
        152,
        153,
        154,
        155,
        156,
        157,
        158,
        159,
        160,
        161,
        162,
        163,
        164,
        165,
        166,
        167,
        168,
        169,
        170,
        171,
        172,
        173,
        174,
        175,
        176,
        177,
        178,
        179,
        180,
        181,
        182,
        183,
        184,
        185,
        186,
        187,
        188,
        189,
        190,
        191,
        192,
        193
      ]
    },
    {
      "clip": "rpo_turn_l",
      "source": "official_rpo",
      "eligible": 120,
      "original_starts": 332,
      "max_root_lift_m": 0.004752741195261478,
      "joint_lookahead_s": 0.02,
      "joint_margin_rad": 0.001,
      "joint_lookahead_rejected": 7,
      "geometric_eligible": 142,
      "velocity_rejected": 15,
      "tracking_eligible": 70,
      "normal_speed_max_mps": 0.13837550343998825,
      "eligible_frames": [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        59,
        60,
        61,
        62,
        63,
        64,
        65,
        66,
        67,
        68,
        69,
        70,
        71,
        72,
        73,
        74,
        75,
        76,
        82,
        83,
        84,
        85,
        86,
        87,
        88,
        89,
        90,
        91,
        92,
        93,
        111,
        112,
        113,
        114,
        115,
        116,
        117,
        118,
        119,
        120,
        121,
        122,
        123,
        124,
        128,
        130,
        131,
        132,
        133,
        134,
        135,
        136,
        137,
        138,
        139,
        140,
        141,
        158,
        159,
        160,
        161,
        162,
        163,
        164,
        165,
        166,
        167,
        168,
        169,
        170,
        171,
        172,
        173,
        174,
        175,
        176
      ]
    },
    {
      "clip": "rpo_turn_r",
      "source": "official_rpo",
      "eligible": 139,
      "original_starts": 231,
      "max_root_lift_m": 0.005222457926720381,
      "joint_lookahead_s": 0.02,
      "joint_margin_rad": 0.001,
      "joint_lookahead_rejected": 8,
      "geometric_eligible": 167,
      "velocity_rejected": 20,
      "tracking_eligible": 0,
      "normal_speed_max_mps": 0.1317806010563765,
      "eligible_frames": [
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        60,
        61,
        62,
        63,
        64,
        65,
        66,
        78,
        79,
        80,
        81,
        82,
        83,
        84,
        85,
        86,
        87,
        88,
        89,
        90,
        91,
        92,
        93,
        94,
        95,
        100,
        101,
        103,
        104,
        105,
        106,
        107,
        108,
        109,
        110,
        111,
        112,
        113,
        114,
        115,
        116,
        117,
        118,
        119,
        135,
        136,
        137,
        138,
        139,
        140,
        141,
        142,
        143,
        144,
        145,
        146,
        147,
        148,
        151,
        156,
        158,
        159,
        160,
        161,
        162,
        163,
        164,
        165,
        166,
        167,
        168,
        169,
        170,
        171,
        180,
        181,
        182,
        183,
        184,
        185,
        186,
        187,
        188,
        189,
        190,
        191,
        192,
        193,
        194,
        195,
        196,
        201,
        202,
        203,
        204,
        205,
        207,
        208,
        209,
        210,
        211,
        212,
        213,
        214,
        215,
        216,
        217,
        218,
        219,
        220,
        224,
        225,
        226,
        227,
        228,
        229,
        230
      ]
    }
  ],
  "qualification": "not_requested"
}
```

### 阶段100→600：新增500真实联合更新

Train trn_287ae01b97e6403abd30；Job job_8b575c408625492ea7e2；Run run_bc7df2383d8f498bb08d；平台状态succeeded。
完整checkpoint与计数核验：
```json
{
  "step": 600,
  "counters": {
    "learner_joint_updates": 600,
    "sampling_rounds": 40,
    "online_transitions": 983040,
    "next_trigger": 993280
  },
  "six_adam_steps": {
    "forward_optimizer": 600,
    "backward_optimizer": 600,
    "discriminator_optimizer": 600,
    "actor_optimizer": 600,
    "critic_optimizer": 600,
    "aux_critic_optimizer": 600
  },
  "hashes_verified": true,
  "state_coverage": {
    "model": "restored",
    "optimizer": "restored",
    "scheduler": "not_applicable",
    "scaler": "not_applicable",
    "rng_per_rank": "restored",
    "replay": "restored",
    "expert_sampler": "restored",
    "z_pool": "restored",
    "rollout": "restored",
    "environment": "not_available"
  },
  "fingerprint": "sha256:29724c802c07204bdea91f0cc890492a202925ad5bd079547b3d163d6836d558",
  "execution": {
    "schema_version": "eb/training-binding/v1",
    "recipe_id": "rpo_bfm_ufo_agent_rlinf_mjlab_4gpu_viability_v13",
    "algorithm_id": "bfm_fb_cpr_aux",
    "provider_id": "ufo",
    "runtime_id": "rlinf",
    "simulator_id": "mjlab",
    "progress_unit": "learner_joint_update",
    "checkpoint_format": "ufo_agent_state_v2",
    "topology_id": "one_learner_three_samplers"
  }
}
```
实际恢复/采样状态同步回执：
```json
{
  "schema_version": "eb/resume-request/v1",
  "status": "validated",
  "source_run_id": "run_72deb358859f4098bb80",
  "step": 100,
  "mode": "trainer_state",
  "checkpoint": "/workspace/resume/global_step_100",
  "compatibility_fingerprint": "sha256:29724c802c07204bdea91f0cc890492a202925ad5bd079547b3d163d6836d558"
}
```
实际恢复/采样状态同步回执：
```json
{
  "schema_version": "eb/ufo-sampler-sync/v1",
  "identity": {
    "state_sha256": "e9874fc46534f130b29f735e13b52490b83ed3f0b8e522ce9d95551f831c62f1",
    "version": 596,
    "online_transitions": 958464
  },
  "samplers": [
    {
      "rank": 0,
      "state_sha256": "e9874fc46534f130b29f735e13b52490b83ed3f0b8e522ce9d95551f831c62f1",
      "version": 596,
      "online_transitions": 958464
    },
    {
      "rank": 1,
      "state_sha256": "e9874fc46534f130b29f735e13b52490b83ed3f0b8e522ce9d95551f831c62f1",
      "version": 596,
      "online_transitions": 958464
    },
    {
      "rank": 2,
      "state_sha256": "e9874fc46534f130b29f735e13b52490b83ed3f0b8e522ce9d95551f831c62f1",
      "version": 596,
      "online_transitions": 958464
    }
  ],
  "sampling_rounds": 39
}
```
每30秒doctor采样的整卡已用显存最大值MiB（包含驱动/Warp与常驻占用，并非连续测量的真正峰值）：
```json
{
  "0": 16710,
  "1": 4537,
  "2": 4537,
  "3": 4537
}
```
原始指标标签数149；联合更新记录500，范围101–600；逐行trace 786432条。
| 窗口 | 环境点数 | 训练episode_len步 | 严格影子步 | 严格影子秒 |
|---|---:|---:|---:|---:|
| 101–200 | 6 | 4.431478 | 3.524973 | 0.07049945 |
| 201–300 | 6 | 6.108557 | 4.572177 | 0.09144353 |
| 301–400 | 6 | 7.70005 | 5.448849 | 0.108977 |
| 401–500 | 7 | 9.869158 | 6.36294 | 0.1272588 |
| 501–600 | 7 | 12.71072 | 7.302666 | 0.1460533 |

末点训练13.85121步、严格影子7.879843步。首步训练终止10/105291=0.009497%；严格首步唯一违规1566例（1.487307%），位置/速度/力矩计数[1539, 27, 0]。子步全局峰值42.57798rad/s、5.322248倍额定值；垂直净力峰值4701.443N。

完整Replay/初态/奖励/上下文与事故审计（报告passed仅表示核验一致，不表示行走达标）：
```json
{
  "status": "passed",
  "rows": 786432,
  "rounds": 32,
  "round_first": 9,
  "round_last": 40,
  "replay_count": 983040,
  "replay_capacity": 4000000,
  "fifo_evictions": 0,
  "requested_saturation_fraction": 0.37726410575534985,
  "applied_saturation_fraction": 0.11310096409009851,
  "strict_position_velocity_torque_counts": [
    243700,
    47980,
    0
  ],
  "training_position_velocity_torque_counts": [
    86364,
    10989,
    0
  ],
  "macro_contact_height_tilt_counts": [
    0,
    1506,
    4960
  ],
  "boundary_max_radps": 28.494556427001953,
  "physics_max_radps": 42.577980041503906,
  "physics_max_rated_ratio": 5.322247505187988,
  "boundary_max_event": {
    "round": 18,
    "row": 11478,
    "episode_id_run_local": 62770193417,
    "episode_step": 6,
    "joint": "left_ankle_pitch_joint",
    "rated_radps": 8.0,
    "boundary_dq_radps": 28.494556427001953,
    "physics_signed_dq_radps": 28.607463836669922,
    "physics_abs_dq_radps": 28.607463836669922,
    "peak_substep_0_to_7": 6,
    "q_at_joint_peak_rad": -0.07812625169754028,
    "torque_at_joint_peak_Nm": -27.0,
    "target_at_joint_peak_rad": -0.22080975770950317,
    "requested_action": 0.019292939454317093,
    "applied_action": -0.36801624298095703,
    "foot_peak_vertical_net_force_N": [
      3645.22900390625,
      0.0
    ],
    "foot_vertical_net_force_integral_Ns": [
      23.600439071655273,
      0.0
    ],
    "global_peak_joint": "left_ankle_pitch_joint",
    "feet_at_global_abs_dq_peak_origin_pose_wxyz_linear_angular": [
      [
        -11.286864280700684,
        4.1721625328063965,
        0.05817355215549469,
        0.6552661657333374,
        0.10196654498577118,
        -0.3768531382083893,
        0.6466922163963318,
        -0.25906288623809814,
        1.3384475708007812,
        -0.0034685134887695312,
        -20.343753814697266,
        -1.703470230102539,
        -0.45923957228660583
      ],
      [
        -11.203392028808594,
        3.76015567779541,
        0.1139901876449585,
        0.5421223044395447,
        -0.3705368936061859,
        0.20152397453784943,
        0.7267695069313049,
        -1.0781806707382202,
        2.5102524757385254,
        0.3304697275161743,
        2.274794816970825,
        1.138234257698059,
        0.8062452077865601
      ]
    ],
    "feet_vertical_net_force_at_global_peak_N": [
      1063.52734375,
      0.0
    ],
    "strict_flags_pos_vel_torque": [
      0,
      1,
      0
    ],
    "training_flags_pos_vel_torque": [
      0,
      1,
      0
    ],
    "macro_contact_height_tilt": [
      0,
      0,
      0
    ]
  },
  "physics_max_event": {
    "round": 30,
    "row": 2293,
    "episode_id_run_local": 2247133673484,
    "episode_step": 16,
    "joint": "right_ankle_pitch_joint",
    "rated_radps": 8.0,
    "boundary_dq_radps": -24.769874572753906,
    "physics_signed_dq_radps": -42.577980041503906,
    "physics_abs_dq_radps": 42.577980041503906,
    "peak_substep_0_to_7": 6,
    "q_at_joint_peak_rad": -0.3085629343986511,
    "torque_at_joint_peak_Nm": 27.0,
    "target_at_joint_peak_rad": 0.2552996873855591,
    "requested_action": -0.3214556574821472,
    "applied_action": 0.42549943923950195,
    "foot_peak_vertical_net_force_N": [
      0.0,
      2403.103759765625
    ],
    "foot_vertical_net_force_integral_Ns": [
      0.0,
      20.74536895751953
    ],
    "global_peak_joint": "right_ankle_pitch_joint",
    "feet_at_global_abs_dq_peak_origin_pose_wxyz_linear_angular": [
      [
        42.72224044799805,
        -5.579956531524658,
        0.10554677248001099,
        0.3653241991996765,
        0.16609160602092743,
        0.1763065755367279,
        0.8988146781921387,
        0.9951435327529907,
        -1.9540014266967773,
        0.41498279571533203,
        -5.6118998527526855,
        1.7469964027404785,
        2.8595850467681885
      ],
      [
        42.589271545410156,
        -5.023991584777832,
        0.03996506333351135,
        -0.014479607343673706,
        0.022754032164812088,
        -0.16647380590438843,
        0.9856769442558289,
        1.9251670837402344,
        0.22587323188781738,
        -1.7764301300048828,
        -3.537712812423706,
        35.48981857299805,
        -0.2434592843055725
      ]
    ],
    "feet_vertical_net_force_at_global_peak_N": [
      0.0,
      1219.9078369140625
    ],
    "strict_flags_pos_vel_torque": [
      0,
      1,
      0
    ],
    "training_flags_pos_vel_torque": [
      0,
      1,
      0
    ],
    "macro_contact_height_tilt": [
      0,
      0,
      0
    ]
  },
  "normalized_max_event": {
    "round": 30,
    "row": 2293,
    "episode_id_run_local": 2247133673484,
    "episode_step": 16,
    "joint": "right_ankle_pitch_joint",
    "rated_radps": 8.0,
    "boundary_dq_radps": -24.769874572753906,
    "physics_signed_dq_radps": -42.577980041503906,
    "physics_abs_dq_radps": 42.577980041503906,
    "peak_substep_0_to_7": 6,
    "q_at_joint_peak_rad": -0.3085629343986511,
    "torque_at_joint_peak_Nm": 27.0,
    "target_at_joint_peak_rad": 0.2552996873855591,
    "requested_action": -0.3214556574821472,
    "applied_action": 0.42549943923950195,
    "foot_peak_vertical_net_force_N": [
      0.0,
      2403.103759765625
    ],
    "foot_vertical_net_force_integral_Ns": [
      0.0,
      20.74536895751953
    ],
    "global_peak_joint": "right_ankle_pitch_joint",
    "feet_at_global_abs_dq_peak_origin_pose_wxyz_linear_angular": [
      [
        42.72224044799805,
        -5.579956531524658,
        0.10554677248001099,
        0.3653241991996765,
        0.16609160602092743,
        0.1763065755367279,
        0.8988146781921387,
        0.9951435327529907,
        -1.9540014266967773,
        0.41498279571533203,
        -5.6118998527526855,
        1.7469964027404785,
        2.8595850467681885
      ],
      [
        42.589271545410156,
        -5.023991584777832,
        0.03996506333351135,
        -0.014479607343673706,
        0.022754032164812088,
        -0.16647380590438843,
        0.9856769442558289,
        1.9251670837402344,
        0.22587323188781738,
        -1.7764301300048828,
        -3.537712812423706,
        35.48981857299805,
        -0.2434592843055725
      ]
    ],
    "feet_vertical_net_force_at_global_peak_N": [
      0.0,
      1219.9078369140625
    ],
    "strict_flags_pos_vel_torque": [
      0,
      1,
      0
    ],
    "training_flags_pos_vel_torque": [
      0,
      1,
      0
    ],
    "macro_contact_height_tilt": [
      0,
      0,
      0
    ]
  },
  "foot_vertical_net_force_peak_N": 4701.443359375,
  "first_step_foot_vertical_net_force_N": {
    "n": 105291,
    "mean": 56.7226448059082,
    "median": 0.0,
    "p95": 354.832763671875,
    "p99": 610.5845092773455,
    "max": 1660.192626953125
  },
  "first_step_foot_absolute_vertical_net_force_integral_Ns": {
    "n": 105291,
    "mean": 0.4160936176776886,
    "median": 0.0,
    "p95": 3.133272647857666,
    "p99": 4.765342569351197,
    "max": 10.962077140808105
  },
  "reset_lift_m": {
    "n": 105291,
    "mean": 0.0014551373897120357,
    "median": 0.0,
    "p95": 0.014100168831646442,
    "p99": 0.01963331736624241,
    "max": 0.021992022171616554
  },
  "resets": 105291,
  "first_step_terminated": 10,
  "first_step_termination_fraction": 9.49748791444663e-05,
  "first_step_strict_any": 1566,
  "first_step_strict_any_fraction": 0.01487306607402342,
  "first_step_strict_counts": [
    1539,
    27,
    0
  ],
  "first_step_macro_counts": [
    0,
    0,
    0
  ],
  "raw_velocity_penalty_mean": 0.010422811707636962,
  "weighted_velocity_penalty_mean": -0.2605702926909241,
  "strict_velocity_substep_joint_exposure_fraction": 0.001136309858681499,
  "velocity_reward_coverage_miss_count": 0,
  "latched_velocity_terms_without_boundary_violation": 28316,
  "reset_q_preservation_max_error_rad": 8.821487429422348e-08,
  "reset_dq_preservation_max_error_radps": 0.0,
  "decoder_max_error_rad": 2.813339232687895e-07,
  "infinite_plane_trace_margin_sentinel_m": 1000000.0,
  "ground_outside_count": 0,
  "mixed_nonperiodic_z_equal_pairs": 333305,
  "mixed_nonperiodic_cosine": {
    "n": 333305,
    "mean": 1.0,
    "median": 1.0,
    "p95": 1.0000001192092896,
    "p99": 1.0000001192092896,
    "max": 1.0000001192092896
  },
  "tracking_unexpected_refreshes": 0,
  "tracking_phase_mismatches": 0,
  "tracking_window_stable_pairs": 347836,
  "tracking_adjacent_phase_cosine": {
    "n": 347836,
    "mean": 0.9974445700645447,
    "median": 0.9978035092353821,
    "p95": 0.9999883770942688,
    "p99": 0.9999957084655762,
    "max": 0.9999988675117493
  },
  "tracking_window_renewals": 0,
  "terminal_following_new_episode_pairs": 99147,
  "tracking_envs_per_rank": [
    1024,
    1024,
    1024
  ],
  "mixed_envs_per_rank": [
    1024,
    1024,
    1024
  ],
  "completed_episode_len": {
    "n": 99547,
    "mean": 7.342531668458116,
    "median": 6.0,
    "p95": 17.0,
    "p99": 26.0,
    "max": 57.0
  },
  "completed_strict_shadow_len": {
    "n": 99547,
    "mean": 5.057831978864255,
    "median": 4.0,
    "p95": 11.0,
    "p99": 15.0,
    "max": 33.0
  },
  "completed_tracking_shadow_len": {
    "n": 42442,
    "mean": 5.847698034965364,
    "median": 5.0,
    "p95": 12.0,
    "p99": 16.0,
    "max": 33.0
  },
  "completed_mixed_shadow_len": {
    "n": 57105,
    "mean": 4.470781893004116,
    "median": 4.0,
    "p95": 9.0,
    "p99": 12.0,
    "max": 28.0
  },
  "unfinished_train_episodes_right_censored": 5744,
  "unfinished_strict_shadow_right_censored": 3091,
  "unfinished_train_observed_lengths": {
    "n": 5744,
    "mean": 9.663126740947074,
    "median": 8.0,
    "p95": 26.0,
    "p99": 33.0,
    "max": 51.0
  },
  "RSI_source_by_rollout_mode": {
    "official_rpo/mixed": {
      "rows": 192932,
      "resets": 29943,
      "first_terms": 7,
      "strict_velocity_rows": 17192,
      "completed_episode_len": {
        "n": 28557,
        "mean": 6.368140911160136,
        "median": 5.0,
        "p95": 14.0,
        "p99": 21.0,
        "max": 41.0
      },
      "completed_strict_shadow_len": {
        "n": 28557,
        "mean": 4.593479707252162,
        "median": 4.0,
        "p95": 9.0,
        "p99": 12.0,
        "max": 28.0
      },
      "first_term_fraction": 0.00023377751060347994,
      "strict_velocity_substep_joint_exposure_fraction": 0.0017987865064420085,
      "raw_velocity_penalty_mean": 0.016615050408410777
    },
    "official_rpo/tracking": {
      "rows": 194135,
      "resets": 22495,
      "first_terms": 1,
      "strict_velocity_rows": 5115,
      "completed_episode_len": {
        "n": 21184,
        "mean": 8.529078549848943,
        "median": 7.0,
        "p95": 19.0,
        "p99": 27.0,
        "max": 57.0
      },
      "completed_strict_shadow_len": {
        "n": 21184,
        "mean": 6.032902190332326,
        "median": 6.0,
        "p95": 12.0,
        "p99": 15.0,
        "max": 31.0
      },
      "first_term_fraction": 4.445432318292954e-05,
      "strict_velocity_substep_joint_exposure_fraction": 0.0005228600447245922,
      "raw_velocity_penalty_mean": 0.005525572183839746
    },
    "g1/mixed": {
      "rows": 200284,
      "resets": 29968,
      "first_terms": 2,
      "strict_velocity_rows": 16243,
      "completed_episode_len": {
        "n": 28548,
        "mean": 6.578674513100743,
        "median": 5.0,
        "p95": 15.0,
        "p99": 23.0,
        "max": 41.0
      },
      "completed_strict_shadow_len": {
        "n": 28548,
        "mean": 4.348045397225725,
        "median": 4.0,
        "p95": 9.0,
        "p99": 13.0,
        "max": 28.0
      },
      "first_term_fraction": 6.673785371062466e-05,
      "strict_velocity_substep_joint_exposure_fraction": 0.0014193160928872306,
      "raw_velocity_penalty_mean": 0.01243060493823502
    },
    "g1/tracking": {
      "rows": 199081,
      "resets": 22885,
      "first_terms": 0,
      "strict_velocity_rows": 9430,
      "completed_episode_len": {
        "n": 21258,
        "mean": 8.49487251858124,
        "median": 7.0,
        "p95": 20.0,
        "p99": 29.0,
        "max": 46.0
      },
      "completed_strict_shadow_len": {
        "n": 21258,
        "mean": 5.663138583121649,
        "median": 5.0,
        "p95": 12.0,
        "p99": 17.0,
        "max": 33.0
      },
      "first_term_fraction": 0.0,
      "strict_velocity_substep_joint_exposure_fraction": 0.0008077878722294159,
      "raw_velocity_penalty_mean": 0.007177477934280013
    }
  },
  "per_joint_physical_diagnostics": {
    "left_thigh_yaw_joint": {
      "rated_radps": 25.0,
      "max_physics_abs_radps": 20.573762893676758,
      "max_boundary_abs_radps": 20.317298889160156,
      "max_rated_ratio": 0.8229505157470703,
      "substep_exposure_fraction": 0.0
    },
    "right_thigh_yaw_joint": {
      "rated_radps": 25.0,
      "max_physics_abs_radps": 22.553632736206055,
      "max_boundary_abs_radps": 21.591859817504883,
      "max_rated_ratio": 0.9021453094482422,
      "substep_exposure_fraction": 0.0
    },
    "torso_joint": {
      "rated_radps": 25.0,
      "max_physics_abs_radps": 18.471818923950195,
      "max_boundary_abs_radps": 17.278486251831055,
      "max_rated_ratio": 0.7388727569580078,
      "substep_exposure_fraction": 0.0
    },
    "left_thigh_roll_joint": {
      "rated_radps": 25.0,
      "max_physics_abs_radps": 25.097200393676758,
      "max_boundary_abs_radps": 24.065439224243164,
      "max_rated_ratio": 1.0038880157470702,
      "substep_exposure_fraction": 3.178914388020833e-07
    },
    "right_thigh_roll_joint": {
      "rated_radps": 25.0,
      "max_physics_abs_radps": 25.276395797729492,
      "max_boundary_abs_radps": 24.846921920776367,
      "max_rated_ratio": 1.0110558319091796,
      "substep_exposure_fraction": 7.947285970052084e-07
    },
    "left_arm_pitch_joint": {
      "rated_radps": 8.0,
      "max_physics_abs_radps": 13.854570388793945,
      "max_boundary_abs_radps": 13.854570388793945,
      "max_rated_ratio": 1.7318212985992432,
      "substep_exposure_fraction": 0.0035947163899739585
    },
    "right_arm_pitch_joint": {
      "rated_radps": 8.0,
      "max_physics_abs_radps": 13.368700981140137,
      "max_boundary_abs_radps": 13.368700981140137,
      "max_rated_ratio": 1.671087622642517,
      "substep_exposure_fraction": 0.0027414957682291665
    },
    "left_thigh_pitch_joint": {
      "rated_radps": 25.0,
      "max_physics_abs_radps": 26.282794952392578,
      "max_boundary_abs_radps": 24.660242080688477,
      "max_rated_ratio": 1.0513117980957032,
      "substep_exposure_fraction": 9.5367431640625e-07
    },
    "right_thigh_pitch_joint": {
      "rated_radps": 25.0,
      "max_physics_abs_radps": 23.695539474487305,
      "max_boundary_abs_radps": 23.187942504882812,
      "max_rated_ratio": 0.9478215789794922,
      "substep_exposure_fraction": 0.0
    },
    "left_arm_roll_joint": {
      "rated_radps": 8.0,
      "max_physics_abs_radps": 14.24984359741211,
      "max_boundary_abs_radps": 14.24984359741211,
      "max_rated_ratio": 1.7812304496765137,
      "substep_exposure_fraction": 0.0013167063395182292
    },
    "right_arm_roll_joint": {
      "rated_radps": 8.0,
      "max_physics_abs_radps": 12.964090347290039,
      "max_boundary_abs_radps": 12.964090347290039,
      "max_rated_ratio": 1.6205112934112549,
      "substep_exposure_fraction": 0.0008794466654459635
    },
    "left_knee_joint": {
      "rated_radps": 25.0,
      "max_physics_abs_radps": 22.57154655456543,
      "max_boundary_abs_radps": 20.63555908203125,
      "max_rated_ratio": 0.9028618621826172,
      "substep_exposure_fraction": 0.0
    },
    "right_knee_joint": {
      "rated_radps": 25.0,
      "max_physics_abs_radps": 19.8154296875,
      "max_boundary_abs_radps": 19.8154296875,
      "max_rated_ratio": 0.7926171875,
      "substep_exposure_fraction": 0.0
    },
    "left_arm_yaw_joint": {
      "rated_radps": 8.0,
      "max_physics_abs_radps": 6.411949157714844,
      "max_boundary_abs_radps": 5.726081371307373,
      "max_rated_ratio": 0.8014936447143555,
      "substep_exposure_fraction": 0.0
    },
    "right_arm_yaw_joint": {
      "rated_radps": 8.0,
      "max_physics_abs_radps": 6.621391296386719,
      "max_boundary_abs_radps": 6.058536052703857,
      "max_rated_ratio": 0.8276739120483398,
      "substep_exposure_fraction": 0.0
    },
    "left_ankle_pitch_joint": {
      "rated_radps": 8.0,
      "max_physics_abs_radps": 28.776344299316406,
      "max_boundary_abs_radps": 28.494556427001953,
      "max_rated_ratio": 3.597043037414551,
      "substep_exposure_fraction": 0.00820144017537435
    },
    "right_ankle_pitch_joint": {
      "rated_radps": 8.0,
      "max_physics_abs_radps": 42.577980041503906,
      "max_boundary_abs_radps": 27.628990173339844,
      "max_rated_ratio": 5.322247505187988,
      "substep_exposure_fraction": 0.0072142283121744795
    },
    "left_elbow_pitch_joint": {
      "rated_radps": 8.0,
      "max_physics_abs_radps": 8.086335182189941,
      "max_boundary_abs_radps": 7.730127811431885,
      "max_rated_ratio": 1.0107918977737427,
      "substep_exposure_fraction": 4.76837158203125e-07
    },
    "right_elbow_pitch_joint": {
      "rated_radps": 8.0,
      "max_physics_abs_radps": 7.685504913330078,
      "max_boundary_abs_radps": 7.506246089935303,
      "max_rated_ratio": 0.9606881141662598,
      "substep_exposure_fraction": 0.0
    },
    "left_ankle_roll_joint": {
      "rated_radps": 8.0,
      "max_physics_abs_radps": 32.13508605957031,
      "max_boundary_abs_radps": 22.835481643676758,
      "max_rated_ratio": 4.016885757446289,
      "substep_exposure_fraction": 0.0011502901713053386
    },
    "right_ankle_roll_joint": {
      "rated_radps": 8.0,
      "max_physics_abs_radps": 27.737089157104492,
      "max_boundary_abs_radps": 27.737089157104492,
      "max_rated_ratio": 3.4671361446380615,
      "substep_exposure_fraction": 0.0010342597961425781
    },
    "left_elbow_yaw_joint": {
      "rated_radps": 8.0,
      "max_physics_abs_radps": 4.2518720626831055,
      "max_boundary_abs_radps": 3.8067963123321533,
      "max_rated_ratio": 0.5314840078353882,
      "substep_exposure_fraction": 0.0
    },
    "right_elbow_yaw_joint": {
      "rated_radps": 8.0,
      "max_physics_abs_radps": 4.262666702270508,
      "max_boundary_abs_radps": 3.8115618228912354,
      "max_rated_ratio": 0.5328333377838135,
      "substep_exposure_fraction": 0.0
    }
  },
  "right_ankle_pitch_physics_peak_event": {
    "round": 30,
    "row": 2293,
    "episode_id_run_local": 2247133673484,
    "episode_step": 16,
    "joint": "right_ankle_pitch_joint",
    "rated_radps": 8.0,
    "boundary_dq_radps": -24.769874572753906,
    "physics_signed_dq_radps": -42.577980041503906,
    "physics_abs_dq_radps": 42.577980041503906,
    "peak_substep_0_to_7": 6,
    "q_at_joint_peak_rad": -0.3085629343986511,
    "torque_at_joint_peak_Nm": 27.0,
    "target_at_joint_peak_rad": 0.2552996873855591,
    "requested_action": -0.3214556574821472,
    "applied_action": 0.42549943923950195,
    "foot_peak_vertical_net_force_N": [
      0.0,
      2403.103759765625
    ],
    "foot_vertical_net_force_integral_Ns": [
      0.0,
      20.74536895751953
    ],
    "global_peak_joint": "right_ankle_pitch_joint",
    "feet_at_global_abs_dq_peak_origin_pose_wxyz_linear_angular": [
      [
        42.72224044799805,
        -5.579956531524658,
        0.10554677248001099,
        0.3653241991996765,
        0.16609160602092743,
        0.1763065755367279,
        0.8988146781921387,
        0.9951435327529907,
        -1.9540014266967773,
        0.41498279571533203,
        -5.6118998527526855,
        1.7469964027404785,
        2.8595850467681885
      ],
      [
        42.589271545410156,
        -5.023991584777832,
        0.03996506333351135,
        -0.014479607343673706,
        0.022754032164812088,
        -0.16647380590438843,
        0.9856769442558289,
        1.9251670837402344,
        0.22587323188781738,
        -1.7764301300048828,
        -3.537712812423706,
        35.48981857299805,
        -0.2434592843055725
      ]
    ],
    "feet_vertical_net_force_at_global_peak_N": [
      0.0,
      1219.9078369140625
    ],
    "strict_flags_pos_vel_torque": [
      0,
      1,
      0
    ],
    "training_flags_pos_vel_torque": [
      0,
      1,
      0
    ],
    "macro_contact_height_tilt": [
      0,
      0,
      0
    ]
  },
  "right_ankle_roll_physics_peak_event": {
    "round": 37,
    "row": 21989,
    "episode_id_run_local": 1167828446734,
    "episode_step": 36,
    "joint": "right_ankle_roll_joint",
    "rated_radps": 8.0,
    "boundary_dq_radps": 27.737089157104492,
    "physics_signed_dq_radps": 27.737089157104492,
    "physics_abs_dq_radps": 27.737089157104492,
    "peak_substep_0_to_7": 7,
    "q_at_joint_peak_rad": 0.3331005871295929,
    "torque_at_joint_peak_Nm": -27.0,
    "target_at_joint_peak_rad": 0.007029742002487183,
    "requested_action": -0.9337453842163086,
    "applied_action": 0.014059484004974365,
    "foot_peak_vertical_net_force_N": [
      0.0,
      2615.968505859375
    ],
    "foot_vertical_net_force_integral_Ns": [
      0.0,
      14.82856273651123
    ],
    "global_peak_joint": "right_ankle_roll_joint",
    "feet_at_global_abs_dq_peak_origin_pose_wxyz_linear_angular": [
      [
        -26.219205856323242,
        5.128350734710693,
        0.06483888626098633,
        0.7436100244522095,
        0.38771823048591614,
        -0.26944392919540405,
        -0.47341176867485046,
        -0.18596339225769043,
        0.738359272480011,
        0.14885860681533813,
        1.2767304182052612,
        -0.6910907626152039,
        0.4201328158378601
      ],
      [
        -26.757802963256836,
        4.134298801422119,
        0.04712657630443573,
        0.9187236428260803,
        -0.2537994682788849,
        0.09105543792247772,
        -0.28851625323295593,
        -0.892003059387207,
        -1.6186270713806152,
        -0.13242053985595703,
        19.53075408935547,
        -14.38428783416748,
        1.5891695022583008
      ]
    ],
    "feet_vertical_net_force_at_global_peak_N": [
      0.0,
      1358.1951904296875
    ],
    "strict_flags_pos_vel_torque": [
      1,
      1,
      0
    ],
    "training_flags_pos_vel_torque": [
      1,
      1,
      0
    ],
    "macro_contact_height_tilt": [
      0,
      1,
      0
    ]
  },
  "semantics": "Run-local episode IDs; original physical thresholds shadow; interval mean of exactly 8 completed substeps; vertical NET force not contact point normal impulse; unfinished trajectories are right-censored, excluded from completed means"
}
```

所有实际指标的全段统计与窗口均值：

| tag | n | first | last | mean | min | max | median | p95 | 按更新窗口均值(n) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| env/episode_len | 32 | 3.242604 | 13.85121 | 8.359365 | 3.242604 | 14.25906 | 7.966063 | 13.62902 | 101–200: 4.431478(6); 201–300: 6.108557(6); 301–400: 7.70005(6); 401–500: 9.869158(7); 501–600: 12.71072(7) |
| env/num_trajectories | 32 | 4394 | 1781 | 3110.844 | 1683 | 5206 | 2865 | 5099.35 | 101–200: 4851.5(6); 201–300: 3894.833(6); 301–400: 2999(6); 401–500: 2369.571(7); 501–600: 1784(7) |
| env/return | 32 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 101–200: 0(6); 201–300: 0(6); 301–400: 0(6); 401–500: 0(7); 501–600: 0(7) |
| env/reward | 32 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 101–200: 0(6); 201–300: 0(6); 301–400: 0(6); 401–500: 0(7); 501–600: 0(7) |
| env/success_once | 32 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 101–200: 0(6); 201–300: 0(6); 301–400: 0(6); 401–500: 0(7); 501–600: 0(7) |
| rollout/ufo/received_transitions | 32 | 24576 | 24576 | 24576 | 24576 | 24576 | 24576 | 24576 | 101–200: 24576(6); 201–300: 24576(6); 301–400: 24576(6); 401–500: 24576(7); 501–600: 24576(7) |
| time/actor_training | 32 | 15.29896 | 4.226684 | 12.21757 | 4.226684 | 15.29896 | 12.37197 | 12.5084 | 101–200: 12.8361(6); 201–300: 12.36156(6); 301–400: 12.39026(6); 401–500: 12.38931(7); 501–600: 11.24421(7) |
| time/cal_adv_and_returns | 32 | 0.003736734 | 0.007782936 | 0.01165468 | 0.003037691 | 0.02229452 | 0.01168346 | 0.02096823 | 101–200: 0.01149122(6); 201–300: 0.0114218(6); 301–400: 0.01226016(6); 401–500: 0.01163323(7); 501–600: 0.01149685(7) |
| time/env/compute_bootstrap_rewards | 32 | 0.0002817598 | 0.0002703411 | 0.0002657451 | 0.000229599 | 0.0003099728 | 0.0002658315 | 0.0002886151 | 101–200: 0.000275013(6); 201–300: 0.0002619648(6); 301–400: 0.0002538651(6); 401–500: 0.0002735493(7); 501–600: 0.0002634199(7) |
| time/env/env/bootstrap_step | 32 | 0.01228534 | 0.0002840832 | 0.0006744018 | 0.0002341098 | 0.01228534 | 0.0002676471 | 0.0005811247 | 101–200: 0.002277779(6); 201–300: 0.0002652931(6); 301–400: 0.0002916097(6); 401–500: 0.0003138627(7); 501–600: 0.0003393897(7) |
| time/env/env/send_rollout_trajectories | 32 | 0.4897398 | 0.3806092 | 0.3939691 | 0.3637441 | 0.4897398 | 0.3856683 | 0.4468369 | 101–200: 0.4110707(6); 201–300: 0.3856845(6); 301–400: 0.3839292(6); 401–500: 0.3893745(7); 501–600: 0.3996118(7) |
| time/env/env_interact_step | 32 | 1.400514 | 1.072223 | 1.083868 | 1.04419 | 1.400514 | 1.072492 | 1.145196 | 101–200: 1.119938(6); 201–300: 1.066151(6); 301–400: 1.064598(6); 401–500: 1.099282(7); 501–600: 1.06924(7) |
| time/env/interact | 32 | 4.344017 | 2.573646 | 2.711378 | 2.411273 | 4.344017 | 2.665089 | 2.952438 | 101–200: 3.115513(6); 201–300: 2.692768(6); 301–400: 2.628339(6); 401–500: 2.612413(7); 501–600: 2.55107(7) |
| time/env/run_interact_once | 32 | 4.343966 | 2.573624 | 2.711351 | 2.41124 | 4.343966 | 2.665064 | 2.952406 | 101–200: 3.115481(6); 201–300: 2.692743(6); 301–400: 2.628312(6); 401–500: 2.612388(7); 501–600: 2.551045(7) |
| time/generate_rollouts | 32 | 4.72887 | 2.807786 | 3.01006 | 2.689937 | 4.72887 | 2.961863 | 3.299592 | 101–200: 3.460549(6); 201–300: 2.953656(6); 301–400: 2.916612(6); 401–500: 2.896532(7); 501–600: 2.865897(7) |
| time/rollout/generate_one_epoch | 32 | 3.881552 | 2.193349 | 2.315733 | 2.01288 | 3.881552 | 2.283192 | 2.533605 | 101–200: 2.703222(6); 201–300: 2.300309(6); 301–400: 2.238806(6); 401–500: 2.224729(7); 501–600: 2.15376(7) |
| time/rollout/predict | 32 | 1.24796 | 0.1943472 | 0.3074453 | 0.1662803 | 1.24796 | 0.2753576 | 0.4273583 | 101–200: 0.5422292(6); 201–300: 0.3320219(6); 301–400: 0.2811633(6); 401–500: 0.2306726(7); 501–600: 0.1844366(7) |
| time/rollout/rollout/generate | 32 | 3.884616 | 2.194211 | 2.316038 | 2.012984 | 3.884616 | 2.283314 | 2.533733 | 101–200: 2.703836(6); 201–300: 2.30056(6); 301–400: 2.239066(6); 401–500: 2.224948(7); 501–600: 2.153973(7) |
| time/rollout/sync_model_from_actor | 32 | 14.835 | 13.83198 | 13.15576 | 12.52954 | 14.835 | 13.11372 | 13.69849 | 101–200: 13.62397(6); 201–300: 13.17394(6); 301–400: 13.04614(6); 401–500: 12.95489(7); 501–600: 13.03366(7) |
| time/step | 32 | 38.2514 | 24.36583 | 31.93577 | 24.36583 | 38.2514 | 31.90559 | 32.77963 | 101–200: 33.50038(6); 201–300: 32.04629(6); 301–400: 31.9036(6); 401–500: 31.8132(7); 501–600: 30.6501(7) |
| time/sync_weights | 32 | 18.2191 | 17.32301 | 16.69592 | 16.08788 | 18.2191 | 16.64627 | 17.26917 | 101–200: 17.19164(6); 201–300: 16.71907(6); 301–400: 16.5839(6); 401–500: 16.51519(7); 501–600: 16.52793(7) |
| train/ufo/B | 500 | -0.003752882 | -0.003030163 | -0.005121013 | -0.0115095 | 0.0002558373 | -0.005042979 | -0.001595022 | 101–200: -0.006910169(100); 201–300: -0.006624122(100); 301–400: -0.005211142(100); 401–500: -0.003815053(100); 501–600: -0.003044579(100) |
| train/ufo/B_norm | 500 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 101–200: 16(100); 201–300: 16(100); 301–400: 16(100); 401–500: 16(100); 501–600: 16(100) |
| train/ufo/F1 | 500 | 0.02611887 | 0.03037979 | 0.001123629 | -0.01916555 | 0.0305694 | 0.0004370099 | 0.01610596 | 101–200: 0.006127152(100); 201–300: -0.002804787(100); 301–400: -0.005428436(100); 401–500: -0.0005962045(100); 501–600: 0.008320419(100) |
| train/ufo/M1 | 500 | 4.050819 | 0.8485443 | 1.088313 | 0.619064 | 4.050819 | 0.940945 | 2.05457 | 101–200: 1.791211(100); 201–300: 1.037068(100); 301–400: 0.8976095(100); 401–500: 0.8463556(100); 501–600: 0.8693199(100) |
| train/ufo/Q1 | 500 | -2.256174 | -3.374846 | -2.43507 | -3.630041 | -1.506117 | -2.339548 | -1.814444 | 101–200: -2.053107(100); 201–300: -1.99213(100); 301–400: -2.280071(100); 401–500: -2.683802(100); 501–600: -3.166238(100) |
| train/ufo/Q_aux | 500 | -3.028059 | -4.628915 | -3.821355 | -4.833945 | -2.842614 | -3.802889 | -3.084964 | 101–200: -3.229767(100); 201–300: -3.392373(100); 301–400: -3.787952(100); 401–500: -4.172418(100); 501–600: -4.524264(100) |
| train/ufo/Q_discriminator | 500 | -2.236839 | -4.00221 | -2.680546 | -4.00221 | -1.758704 | -2.562246 | -2.008743 | 101–200: -2.21415(100); 201–300: -2.246269(100); 301–400: -2.552591(100); 401–500: -2.937487(100); 501–600: -3.452234(100) |
| train/ufo/Q_fb | 500 | 9.913292 | 63.62383 | 38.35339 | 9.913292 | 67.54675 | 36.99663 | 62.97806 | 101–200: 18.30743(100); 201–300: 28.31091(100); 301–400: 37.19727(100); 401–500: 47.83543(100); 501–600: 60.11592(100) |
| train/ufo/action_abs_mean | 32 | 0.9229497 | 0.7611718 | 0.8603879 | 0.7611718 | 0.9229497 | 0.8741862 | 0.9169558 | 101–200: 0.9139814(6); 201–300: 0.8990771(6); 301–400: 0.877556(6); 401–500: 0.8419926(7); 501–600: 0.7849683(7) |
| train/ufo/action_saturation_fraction | 32 | 0.471294 | 0.2419593 | 0.3772641 | 0.2419593 | 0.471294 | 0.3958448 | 0.4616475 | 101–200: 0.4561054(6); 201–300: 0.4324008(6); 301–400: 0.4011115(6); 401–500: 0.3474629(7); 501–600: 0.2717864(7) |
| train/ufo/actor_loss | 500 | -7.178048 | -48.9894 | -29.70509 | -53.49673 | -7.178048 | -28.5696 | -11.9592 | 101–200: -14.3499(100); 201–300: -22.32196(100); 301–400: -28.51065(100); 401–500: -36.38189(100); 501–600: -46.96107(100) |
| train/ufo/actor_optimizer/grad_clip_limit | 500 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 101–200: 1(100); 201–300: 1(100); 301–400: 1(100); 401–500: 1(100); 501–600: 1(100) |
| train/ufo/actor_optimizer/grad_norm_after_clip_l2 | 500 | 0.9999995 | 0.9999999 | 0.9999998 | 0.9999995 | 1 | 0.9999998 | 0.9999999 | 101–200: 0.9999997(100); 201–300: 0.9999998(100); 301–400: 0.9999998(100); 401–500: 0.9999999(100); 501–600: 0.9999999(100) |
| train/ufo/actor_optimizer/grad_norm_l2 | 500 | 2.002782 | 17.05101 | 7.44105 | 2.002782 | 18.25937 | 6.992539 | 14.18074 | 101–200: 3.316471(100); 201–300: 5.302758(100); 301–400: 7.041692(100); 401–500: 9.515894(100); 501–600: 12.02843(100) |
| train/ufo/actor_optimizer/lr | 500 | 0.0003 | 0.0003 | 0.0003 | 0.0003 | 0.0003 | 0.0003 | 0.0003 | 101–200: 0.0003(100); 201–300: 0.0003(100); 301–400: 0.0003(100); 401–500: 0.0003(100); 501–600: 0.0003(100) |
| train/ufo/actor_optimizer/parameter_delta_l2 | 500 | 0.3607438 | 0.2518502 | 0.3186183 | 0.2474861 | 0.3607438 | 0.3234618 | 0.3523352 | 101–200: 0.3460351(100); 201–300: 0.3312391(100); 301–400: 0.3226679(100); 401–500: 0.3073011(100); 501–600: 0.2858483(100) |
| train/ufo/actor_reg_weight_applied | 500 | 15.8654 | 50 | 39.62363 | 15.8654 | 50 | 42.5963 | 50 | 101–200: 22.67271(100); 201–300: 33.14592(100); 301–400: 42.58769(100); 401–500: 49.71182(100); 501–600: 50(100) |
| train/ufo/actor_reg_weight_raw | 500 | 15.8654 | 70.3367 | 43.74014 | 15.8654 | 74.12204 | 42.5963 | 69.22968 | 101–200: 22.67271(100); 201–300: 33.14592(100); 301–400: 42.58769(100); 401–500: 53.82916(100); 501–600: 66.4652(100) |
| train/ufo/applied_action_abs_mean | 32 | 0.5030661 | 0.5262619 | 0.5338524 | 0.5030661 | 0.5464312 | 0.5344045 | 0.5438964 | 101–200: 0.5277466(6); 201–300: 0.5429228(6); 301–400: 0.5372738(6); 401–500: 0.5321991(7); 501–600: 0.5300321(7) |
| train/ufo/applied_action_saturation_fraction | 32 | 0.08476987 | 0.09381192 | 0.113101 | 0.08476987 | 0.1298863 | 0.115674 | 0.12758 | 101–200: 0.1074446(6); 201–300: 0.127123(6); 301–400: 0.1229401(6); 401–500: 0.1115091(7); 501–600: 0.09908874(7) |
| train/ufo/auxQ1 | 500 | -3.231126 | -4.236522 | -3.527 | -4.510396 | -2.749203 | -3.469421 | -2.891666 | 101–200: -3.051108(100); 201–300: -3.145599(100); 301–400: -3.45883(100); 401–500: -3.810719(100); 501–600: -4.168744(100) |
| train/ufo/aux_critic_loss | 500 | 1.08306 | 1.113316 | 0.9939108 | 0.636988 | 5.823526 | 0.955118 | 1.30223 | 101–200: 0.9443219(100); 201–300: 0.8736012(100); 301–400: 0.9521758(100); 401–500: 1.018237(100); 501–600: 1.181218(100) |
| train/ufo/aux_critic_optimizer/grad_clip_limit | 500 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 101–200: 5(100); 201–300: 5(100); 301–400: 5(100); 401–500: 5(100); 501–600: 5(100) |
| train/ufo/aux_critic_optimizer/grad_norm_after_clip_l2 | 500 | 4.999999 | 5 | 4.712766 | 0.8362148 | 5.000001 | 5 | 5 | 101–200: 4.887138(100); 201–300: 4.757774(100); 301–400: 4.463047(100); 401–500: 4.642627(100); 501–600: 4.813245(100) |
| train/ufo/aux_critic_optimizer/grad_norm_l2 | 500 | 9.182232 | 9.39742 | 9.459491 | 0.8362148 | 28.36359 | 8.91344 | 18.42572 | 101–200: 11.38386(100); 201–300: 9.016221(100); 301–400: 8.393044(100); 401–500: 10.12815(100); 501–600: 8.37618(100) |
| train/ufo/aux_critic_optimizer/lr | 500 | 0.0003 | 0.0003 | 0.0003 | 0.0003 | 0.0003 | 0.0003 | 0.0003 | 101–200: 0.0003(100); 201–300: 0.0003(100); 301–400: 0.0003(100); 401–500: 0.0003(100); 501–600: 0.0003(100) |
| train/ufo/aux_critic_optimizer/parameter_delta_l2 | 500 | 0.3021123 | 0.5872442 | 0.4719096 | 0.2500322 | 1.100245 | 0.4473951 | 0.7222209 | 101–200: 0.3610166(100); 201–300: 0.4678277(100); 301–400: 0.5403325(100); 401–500: 0.4428115(100); 501–600: 0.5475597(100) |
| train/ufo/aux_rew/limits_dof_pos | 500 | 0.05447216 | 0.1095039 | 0.08955544 | 0.05302569 | 0.1207071 | 0.09187176 | 0.1113461 | 101–200: 0.06559756(100); 201–300: 0.08222381(100); 301–400: 0.09200464(100); 401–500: 0.09951015(100); 501–600: 0.1084411(100) |
| train/ufo/aux_rew/limits_dof_vel | 500 | 0.02834612 | 0.01295904 | 0.01682565 | 0.01009393 | 0.02932038 | 0.0162892 | 0.02338433 | 101–200: 0.02137506(100); 201–300: 0.01800133(100); 301–400: 0.01608453(100); 401–500: 0.01487474(100); 501–600: 0.01379261(100) |
| train/ufo/aux_rew/limits_torque | 500 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 101–200: 0(100); 201–300: 0(100); 301–400: 0(100); 401–500: 0(100); 501–600: 0(100) |
| train/ufo/aux_rew/penalty_action_rate | 500 | 0.6349595 | 0.5175284 | 0.5404414 | 0.497712 | 0.6416278 | 0.521504 | 0.6224792 | 101–200: 0.6079164(100); 201–300: 0.5518325(100); 301–400: 0.5204024(100); 401–500: 0.5110002(100); 501–600: 0.5110553(100) |
| train/ufo/aux_rew/penalty_ankle_roll | 500 | 0.02964583 | 0.07275838 | 0.04939779 | 0.02887909 | 0.07638972 | 0.04847399 | 0.0703838 | 101–200: 0.03287829(100); 201–300: 0.03912889(100); 301–400: 0.04841833(100); 401–500: 0.05847421(100); 501–600: 0.06808924(100) |
| train/ufo/aux_rew/penalty_feet_ori | 500 | 0.08685932 | 0.144565 | 0.1174715 | 0.07503118 | 0.1568418 | 0.1193067 | 0.1477394 | 101–200: 0.08823876(100); 201–300: 0.105063(100); 301–400: 0.1192025(100); 401–500: 0.1315784(100); 501–600: 0.1432747(100) |
| train/ufo/aux_rew/penalty_slippage | 500 | 0.3045602 | 0.2488583 | 0.2710527 | 0.2268167 | 0.3150393 | 0.2707258 | 0.2961736 | 101–200: 0.2886938(100); 201–300: 0.2783703(100); 301–400: 0.2714683(100); 401–500: 0.2622812(100); 501–600: 0.2544497(100) |
| train/ufo/aux_rew/penalty_torques | 500 | 4912.815 | 4152.821 | 4532.229 | 4038.002 | 5139.657 | 4544.807 | 4887.163 | 101–200: 4803.553(100); 201–300: 4660.784(100); 301–400: 4553.515(100); 401–500: 4397.55(100); 501–600: 4245.744(100) |
| train/ufo/aux_rew/penalty_undesired_contact | 500 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 101–200: 0(100); 201–300: 0(100); 301–400: 0(100); 401–500: 0(100); 501–600: 0(100) |
| train/ufo/backward_optimizer/grad_clip_limit | 500 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 101–200: 5(100); 201–300: 5(100); 301–400: 5(100); 401–500: 5(100); 501–600: 5(100) |
| train/ufo/backward_optimizer/grad_norm_after_clip_l2 | 500 | 5 | 5 | 5 | 4.999999 | 5.000001 | 5 | 5 | 101–200: 5(100); 201–300: 5(100); 301–400: 5(100); 401–500: 5(100); 501–600: 5(100) |
| train/ufo/backward_optimizer/grad_norm_l2 | 500 | 168245.7 | 16078.31 | 48446.53 | 15849.33 | 168245.7 | 31282.6 | 136233 | 101–200: 115640.7(100); 201–300: 54227.76(100); 301–400: 31802.58(100); 401–500: 22758.2(100); 501–600: 17803.44(100) |
| train/ufo/backward_optimizer/lr | 500 | 1e-05 | 1e-05 | 1e-05 | 1e-05 | 1e-05 | 1e-05 | 1e-05 | 101–200: 1e-05(100); 201–300: 1e-05(100); 301–400: 1e-05(100); 401–500: 1e-05(100); 501–600: 1e-05(100) |
| train/ufo/backward_optimizer/parameter_delta_l2 | 500 | 0.004200207 | 0.00338075 | 0.004042536 | 0.00338075 | 0.004305919 | 0.004220569 | 0.004288338 | 101–200: 0.004245063(100); 201–300: 0.004277805(100); 301–400: 0.004221147(100); 401–500: 0.003924747(100); 501–600: 0.003543918(100) |
| train/ufo/collected_transitions | 32 | 24576 | 24576 | 24576 | 24576 | 24576 | 24576 | 24576 | 101–200: 24576(6); 201–300: 24576(6); 301–400: 24576(6); 401–500: 24576(7); 501–600: 24576(7) |
| train/ufo/contact_termination_count | 32 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 101–200: 0(6); 201–300: 0(6); 301–400: 0(6); 401–500: 0(7); 501–600: 0(7) |
| train/ufo/critic_loss | 500 | 0.6908125 | 0.5522153 | 0.4162964 | 0.2608267 | 0.7919729 | 0.3936682 | 0.6003597 | 101–200: 0.4747312(100); 201–300: 0.3689543(100); 301–400: 0.3423416(100); 401–500: 0.3915764(100); 501–600: 0.5038787(100) |
| train/ufo/critic_optimizer/grad_clip_limit | 500 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 101–200: 5(100); 201–300: 5(100); 301–400: 5(100); 401–500: 5(100); 501–600: 5(100) |
| train/ufo/critic_optimizer/grad_norm_after_clip_l2 | 500 | 5 | 5 | 4.571395 | 0.6622573 | 5.000001 | 4.999999 | 5 | 101–200: 4.943462(100); 201–300: 4.542338(100); 301–400: 4.800797(100); 401–500: 4.206958(100); 501–600: 4.363422(100) |
| train/ufo/critic_optimizer/grad_norm_l2 | 500 | 9.691519 | 7.08228 | 9.006649 | 0.6622573 | 39.58413 | 8.108312 | 19.82121 | 101–200: 11.58574(100); 201–300: 10.54382(100); 301–400: 10.17911(100); 401–500: 5.712071(100); 501–600: 7.0125(100) |
| train/ufo/critic_optimizer/lr | 500 | 0.0003 | 0.0003 | 0.0003 | 0.0003 | 0.0003 | 0.0003 | 0.0003 | 101–200: 0.0003(100); 201–300: 0.0003(100); 301–400: 0.0003(100); 401–500: 0.0003(100); 501–600: 0.0003(100) |
| train/ufo/critic_optimizer/parameter_delta_l2 | 500 | 0.2774117 | 0.4895035 | 0.3726945 | 0.1982467 | 0.5983702 | 0.3609281 | 0.5213162 | 101–200: 0.2694086(100); 201–300: 0.3460142(100); 301–400: 0.3193459(100); 401–500: 0.4550747(100); 501–600: 0.4736289(100) |
| train/ufo/disc_expert_loss | 500 | 0.4734844 | 0.2973211 | 0.3634283 | 0.2530535 | 0.4816959 | 0.3647211 | 0.4635872 | 101–200: 0.4432005(100); 201–300: 0.3947397(100); 301–400: 0.3643453(100); 401–500: 0.3210024(100); 501–600: 0.2938537(100) |
| train/ufo/disc_loss | 500 | 2.223361 | 0.929069 | 1.204499 | 0.8784869 | 2.223361 | 1.140881 | 1.857124 | 101–200: 1.660407(100); 201–300: 1.232911(100); 301–400: 1.143386(100); 401–500: 1.037337(100); 501–600: 0.948453(100) |
| train/ufo/disc_reward_clip_fraction | 500 | 0 | 0.1445312 | 0.05435156 | 0 | 0.1518555 | 0.05078125 | 0.13479 | 101–200: 0.0003662109(100); 201–300: 0.02116211(100); 301–400: 0.04982422(100); 401–500: 0.07830078(100); 501–600: 0.1221045(100) |
| train/ufo/disc_train_loss | 500 | 0.6989821 | 0.3176414 | 0.4483449 | 0.3075635 | 0.6992869 | 0.4268102 | 0.6485462 | 101–200: 0.6113211(100); 201–300: 0.4914913(100); 301–400: 0.4270707(100); 401–500: 0.3759956(100); 501–600: 0.3358458(100) |
| train/ufo/disc_wgan_gp_loss | 500 | 0.1050895 | 0.03141065 | 0.03927255 | 0.02921583 | 0.1050895 | 0.03459486 | 0.07503742 | 101–200: 0.06058854(100); 201–300: 0.03466795(100); 301–400: 0.03519704(100); 401–500: 0.03403388(100); 501–600: 0.03187535(100) |
| train/ufo/discriminator_optimizer/grad_clip_limit | 500 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 101–200: 5(100); 201–300: 5(100); 301–400: 5(100); 401–500: 5(100); 501–600: 5(100) |
| train/ufo/discriminator_optimizer/grad_norm_after_clip_l2 | 500 | 5 | 1.018647 | 1.940722 | 0.9349693 | 5 | 1.291891 | 4.999999 | 101–200: 4.445369(100); 201–300: 1.780873(100); 301–400: 1.29649(100); 401–500: 1.15085(100); 501–600: 1.03003(100) |
| train/ufo/discriminator_optimizer/grad_norm_l2 | 500 | 7.067091 | 1.018647 | 2.059824 | 0.9349693 | 7.103013 | 1.291891 | 6.144655 | 101–200: 5.040879(100); 201–300: 1.780873(100); 301–400: 1.29649(100); 401–500: 1.15085(100); 501–600: 1.03003(100) |
| train/ufo/discriminator_optimizer/lr | 500 | 3e-06 | 3e-06 | 3e-06 | 3e-06 | 3e-06 | 3e-06 | 3e-06 | 101–200: 3e-06(100); 201–300: 3e-06(100); 301–400: 3e-06(100); 401–500: 3e-06(100); 501–600: 3e-06(100) |
| train/ufo/discriminator_optimizer/parameter_delta_l2 | 500 | 0.005133048 | 0.002515278 | 0.003323803 | 0.002513823 | 0.005399954 | 0.00291711 | 0.00532863 | 101–200: 0.005054321(100); 201–300: 0.00330004(100); 301–400: 0.00293224(100); 401–500: 0.002751152(100); 501–600: 0.002581264(100) |
| train/ufo/environment_control_steps | 32 | 36 | 160 | 98 | 36 | 160 | 98 | 153.8 | 101–200: 46(6); 201–300: 70(6); 301–400: 94(6); 401–500: 120(7); 501–600: 148(7) |
| train/ufo/expert_draws/g1 | 32 | 119344 | 613376 | 372455.5 | 119344 | 613376 | 372728 | 600424.4 | 101–200: 160425.3(6); 201–300: 258517.3(6); 301–400: 356518.7(6); 401–500: 462862.9(7); 501–600: 575109.7(7) |
| train/ufo/expert_draws/official_rpo | 32 | 118224 | 615424 | 372248.5 | 118224 | 615424 | 372744 | 602161.2 | 101–200: 159062.7(6); 201–300: 257578.7(6); 301–400: 356185.3(6); 401–500: 462833.1(7); 501–600: 576451.4(7) |
| train/ufo/fb_diag | 500 | -66.65331 | -226.7705 | -178.13 | -241.0792 | -63.88692 | -193.5123 | -82.29746 | 101–200: -98.64795(100); 201–300: -157.705(100); 301–400: -193.5188(100); 401–500: -214.7622(100); 501–600: -226.0162(100) |
| train/ufo/fb_loss | 500 | 112061.8 | 21845.84 | 46227.01 | 21730.94 | 112470.7 | 37710.39 | 98864.22 | 101–200: 86793.73(100); 201–300: 52620.38(100); 301–400: 37948.84(100); 401–500: 29685.09(100); 501–600: 24087(100) |
| train/ufo/fb_offdiag | 500 | 103.5017 | 233.5758 | 163.4328 | 90.9829 | 243.8882 | 162.8043 | 229.3809 | 101–200: 105.3748(100); 201–300: 133.3773(100); 301–400: 163.0258(100); 401–500: 193.3789(100); 501–600: 222.0072(100) |
| train/ufo/first_step_samples | 32 | 8901 | 1785 | 3290.344 | 1696 | 8901 | 2924 | 5274.55 | 101–200: 5674.667(6); 201–300: 3945.833(6); 301–400: 3042.667(6); 401–500: 2387.714(7); 501–600: 1799.714(7) |
| train/ufo/first_step_terminated | 32 | 1 | 1 | 0.3125 | 0 | 2 | 0 | 1 | 101–200: 0.5(6); 201–300: 0.6666667(6); 301–400: 0(6); 401–500: 0.1428571(7); 501–600: 0.2857143(7) |
| train/ufo/foot_contact_peak_N | 32 | 3009.024 | 4238.275 | 4053.077 | 2992.887 | 4701.443 | 4189.402 | 4672.796 | 101–200: 4325.728(6); 201–300: 4344.217(6); 301–400: 4058.46(6); 401–500: 3898.729(7); 501–600: 3719.562(7) |
| train/ufo/forward_optimizer/grad_clip_limit | 500 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 101–200: 5(100); 201–300: 5(100); 301–400: 5(100); 401–500: 5(100); 501–600: 5(100) |
| train/ufo/forward_optimizer/grad_norm_after_clip_l2 | 500 | 5 | 5 | 5 | 4.999999 | 5.000001 | 5 | 5 | 101–200: 5(100); 201–300: 5(100); 301–400: 5(100); 401–500: 5(100); 501–600: 5(100) |
| train/ufo/forward_optimizer/grad_norm_l2 | 500 | 770.3595 | 93.7794 | 216.909 | 93.51153 | 1052.286 | 139.7805 | 606.8245 | 101–200: 494.6453(100); 201–300: 224.9755(100); 301–400: 144.2076(100); 401–500: 116.2356(100); 501–600: 104.4812(100) |
| train/ufo/forward_optimizer/lr | 500 | 0.0003 | 0.0003 | 0.0003 | 0.0003 | 0.0003 | 0.0003 | 0.0003 | 101–200: 0.0003(100); 201–300: 0.0003(100); 301–400: 0.0003(100); 401–500: 0.0003(100); 501–600: 0.0003(100) |
| train/ufo/forward_optimizer/parameter_delta_l2 | 500 | 0.3562471 | 0.9072555 | 0.7520809 | 0.3322026 | 0.9757722 | 0.8303409 | 0.9388285 | 101–200: 0.4590304(100); 201–300: 0.6697501(100); 301–400: 0.8320324(100); 401–500: 0.8870234(100); 501–600: 0.9125685(100) |
| train/ufo/hard_position_count | 32 | 6209 | 7607 | 7615.625 | 6209 | 8550 | 7637 | 8448.25 | 101–200: 7847.5(6); 201–300: 7636.167(6); 301–400: 7224.333(6); 401–500: 7656.286(7); 501–600: 7694(7) |
| train/ufo/hard_torque_count | 32 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 101–200: 0(6); 201–300: 0(6); 301–400: 0(6); 401–500: 0(7); 501–600: 0(7) |
| train/ufo/hard_velocity_count | 32 | 2140 | 996 | 1499.375 | 897 | 2176 | 1604.5 | 1975 | 101–200: 1866.833(6); 201–300: 1717.5(6); 301–400: 1632.167(6); 401–500: 1357.143(7); 501–600: 1025.857(7) |
| train/ufo/height_termination_count | 32 | 0 | 93 | 47.0625 | 0 | 103 | 53 | 89.7 | 101–200: 7.833333(6); 201–300: 21.5(6); 301–400: 45.83333(6); 401–500: 67.42857(7); 501–600: 83.28571(7) |
| train/ufo/latched_velocity_terms_without_boundary_violation | 32 | 1338 | 574 | 884.875 | 512 | 1338 | 935 | 1204.5 | 101–200: 1125.667(6); 201–300: 1025(6); 301–400: 968.8333(6); 401–500: 776.1429(7); 501–600: 595.1429(7) |
| train/ufo/learner_joint_updates | 500 | 101 | 600 | 350.5 | 101 | 600 | 350.5 | 575.05 | 101–200: 150.5(100); 201–300: 250.5(100); 301–400: 350.5(100); 401–500: 450.5(100); 501–600: 550.5(100) |
| train/ufo/mean_aux_reward | 500 | -0.6625607 | -0.8661015 | -0.7669827 | -0.9160435 | -0.5935928 | -0.7719667 | -0.6355036 | 101–200: -0.6596965(100); 201–300: -0.7225218(100); 301–400: -0.7712716(100); 401–500: -0.8236739(100); 501–600: -0.8577497(100) |
| train/ufo/mean_disc_reward | 500 | 0.04387326 | -0.8932413 | -0.4757202 | -0.9061682 | 0.04387326 | -0.5035466 | -0.03425678 | 101–200: -0.08382826(100); 201–300: -0.3177849(100); 301–400: -0.503465(100); 401–500: -0.6603967(100); 501–600: -0.8131261(100) |
| train/ufo/mean_disc_reward_raw | 500 | 0.04387326 | -0.9767947 | -0.5044368 | -0.9938735 | 0.04387326 | -0.529011 | -0.03425678 | 101–200: -0.0838727(100); 201–300: -0.3247514(100); 301–400: -0.5280363(100); 401–500: -0.703592(100); 501–600: -0.8819315(100) |
| train/ufo/mean_next_Q | 500 | -1.656039 | -2.862684 | -1.979059 | -2.909339 | -1.452947 | -1.877697 | -1.503151 | 101–200: -1.554101(100); 201–300: -1.602683(100); 301–400: -1.881135(100); 401–500: -2.227407(100); 501–600: -2.629968(100) |
| train/ufo/mean_next_auxQ | 500 | -2.706313 | -3.965687 | -3.140793 | -4.003119 | -2.567678 | -3.091642 | -2.603071 | 101–200: -2.642098(100); 201–300: -2.76157(100); 301–400: -3.089868(100); 401–500: -3.430414(100); 501–600: -3.780017(100) |
| train/ufo/online_transitions | 500 | 221184 | 983040 | 592969.7 | 221184 | 983040 | 589824 | 933888 | 101–200: 286064.6(100); 201–300: 439418.9(100); 301–400: 592773.1(100); 401–500: 746127.4(100); 501–600: 900464.6(100) |
| train/ufo/orientation_termination_count | 32 | 2 | 184 | 155 | 2 | 415 | 124 | 360.8 | 101–200: 211.8333(6); 201–300: 95.66667(6); 301–400: 134.6667(6); 401–500: 120(7); 501–600: 209.5714(7) |
| train/ufo/orth_loss | 500 | 1120.249 | 218.3903 | 462.417 | 217.2722 | 1124.328 | 377.4268 | 988.521 | 101–200: 867.87(100); 201–300: 526.4471(100); 301–400: 379.7933(100); 401–500: 297.0647(100); 501–600: 240.91(100) |
| train/ufo/orth_loss_diag | 500 | -256 | -256 | -256 | -256 | -256 | -256 | -256 | 101–200: -256(100); 201–300: -256(100); 301–400: -256(100); 401–500: -256(100); 501–600: -256(100) |
| train/ufo/orth_loss_offdiag | 500 | 1376.249 | 474.3903 | 718.417 | 473.2722 | 1380.328 | 633.4268 | 1244.521 | 101–200: 1123.87(100); 201–300: 782.4471(100); 301–400: 635.7933(100); 401–500: 553.0647(100); 501–600: 496.91(100) |
| train/ufo/physics_peak_joint_velocity | 32 | 22.71591 | 26.28279 | 25.97281 | 21.82097 | 42.57798 | 25.1795 | 30.4092 | 101–200: 26.36774(6); 201–300: 26.20629(6); 301–400: 23.1346(6); 401–500: 28.84098(7); 501–600: 24.99874(7) |
| train/ufo/physics_peak_normalized_velocity | 32 | 2.839489 | 2.94907 | 3.186527 | 2.510626 | 5.322248 | 3.096552 | 3.801149 | 101–200: 3.295967(6); 201–300: 3.271533(6); 301–400: 2.795667(6); 401–500: 3.571091(7); 501–600: 2.97032(7) |
| train/ufo/q_loss | 500 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 101–200: 0(100); 201–300: 0(100); 301–400: 0(100); 401–500: 0(100); 501–600: 0(100) |
| train/ufo/replay_draw_new_ratio | 32 | 1.074074 | 1.25 | 1.218752 | 1.074074 | 1.273504 | 1.238056 | 1.271017 | 101–200: 1.125781(6); 201–300: 1.198709(6); 301–400: 1.233513(6); 401–500: 1.255207(7); 501–600: 1.266514(7) |
| train/ufo/replay_draws | 32 | 237568 | 1228800 | 744704 | 237568 | 1228800 | 745472 | 1202586 | 101–200: 319488(6); 201–300: 516096(6); 301–400: 712704(6); 401–500: 925696(7); 501–600: 1151561(7) |
| train/ufo/replay_policy_age_max_updates | 32 | 116 | 600 | 363.625 | 116 | 600 | 364 | 587.2 | 101–200: 156(6); 201–300: 252(6); 301–400: 348(6); 401–500: 452(7); 501–600: 562.2857(7) |
| train/ufo/replay_policy_age_min_updates | 32 | 16 | 4 | 15.625 | 4 | 16 | 16 | 16 | 101–200: 16(6); 201–300: 16(6); 301–400: 16(6); 401–500: 16(7); 501–600: 14.28571(7) |
| train/ufo/replay_retained_fraction | 32 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 101–200: 1(6); 201–300: 1(6); 301–400: 1(6); 401–500: 1(7); 501–600: 1(7) |
| train/ufo/replay_size | 32 | 221184 | 983040 | 602112 | 221184 | 983040 | 602112 | 944947.2 | 101–200: 282624(6); 201–300: 430080(6); 301–400: 577536(6); 401–500: 737280(7); 501–600: 909312(7) |
| train/ufo/requested_action_abs_mean | 32 | 0.9229496 | 0.7611718 | 0.8603879 | 0.7611718 | 0.9229496 | 0.8741862 | 0.9169559 | 101–200: 0.9139814(6); 201–300: 0.899077(6); 301–400: 0.8775561(6); 401–500: 0.8419926(7); 501–600: 0.7849683(7) |
| train/ufo/requested_action_saturation_fraction | 32 | 0.471294 | 0.2419593 | 0.3772641 | 0.2419593 | 0.471294 | 0.3958448 | 0.4616475 | 101–200: 0.4561054(6); 201–300: 0.4324008(6); 301–400: 0.4011115(6); 401–500: 0.3474629(7); 501–600: 0.2717864(7) |
| train/ufo/requested_applied_max_diff | 32 | 1.941952 | 1.9528 | 1.952732 | 1.941952 | 1.966028 | 1.9528 | 1.96384 | 101–200: 1.949099(6); 201–300: 1.9528(6); 301–400: 1.953551(6); 401–500: 1.95469(7); 501–600: 1.953128(7) |
| train/ufo/reset_control_samples | 32 | 8901 | 1785 | 3290.344 | 1696 | 8901 | 2924 | 5274.55 | 101–200: 5674.667(6); 201–300: 3945.833(6); 301–400: 3042.667(6); 401–500: 2387.714(7); 501–600: 1799.714(7) |
| train/ufo/reset_projection_outside_fraction | 32 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 101–200: 0(6); 201–300: 0(6); 301–400: 0(6); 401–500: 0(7); 501–600: 0(7) |
| train/ufo/reset_projection_residual_max_rad | 32 | 1.788139e-07 | 1.788139e-07 | 1.788139e-07 | 1.788139e-07 | 1.788139e-07 | 1.788139e-07 | 1.788139e-07 | 101–200: 1.788139e-07(6); 201–300: 1.788139e-07(6); 301–400: 1.788139e-07(6); 401–500: 1.788139e-07(7); 501–600: 1.788139e-07(7) |
| train/ufo/reset_projection_residual_mean_rad | 32 | 6.397043e-08 | 6.407486e-08 | 6.397484e-08 | 6.314674e-08 | 6.484763e-08 | 6.39858e-08 | 6.454622e-08 | 101–200: 6.407953e-08(6); 201–300: 6.402719e-08(6); 301–400: 6.412725e-08(6); 401–500: 6.375987e-08(7); 501–600: 6.392458e-08(7) |
| train/ufo/reset_projection_residual_p95_rad | 32 | 1.192093e-07 | 1.192093e-07 | 1.192093e-07 | 1.192093e-07 | 1.192093e-07 | 1.192093e-07 | 1.192093e-07 | 101–200: 1.192093e-07(6); 201–300: 1.192093e-07(6); 301–400: 1.192093e-07(6); 401–500: 1.192093e-07(7); 501–600: 1.192093e-07(7) |
| train/ufo/reset_root_lift_max_m | 32 | 0.02199202 | 0.02194565 | 0.02197173 | 0.02194565 | 0.02199202 | 0.02199202 | 0.02199202 | 101–200: 0.02199202(6); 201–300: 0.02199202(6); 301–400: 0.02194565(6); 401–500: 0.0219854(7); 501–600: 0.02194565(7) |
| train/ufo/sampling_rounds | 500 | 9 | 40 | 24.128 | 9 | 40 | 24 | 38 | 101–200: 11.64(100); 201–300: 17.88(100); 301–400: 24.12(100); 401–500: 30.36(100); 501–600: 36.64(100) |
| train/ufo/strict_shadow_episode_len | 32 | 2.866636 | 7.879843 | 5.529226 | 2.866636 | 7.879843 | 5.597899 | 7.802168 | 101–200: 3.524973(6); 201–300: 4.572177(6); 301–400: 5.448849(6); 401–500: 6.36294(7); 501–600: 7.302666(7) |
| train/ufo/strict_shadow_mixed_episode_len | 32 | 2.837564 | 6.508042 | 4.743075 | 2.837564 | 6.508042 | 4.715094 | 6.264711 | 101–200: 3.386667(6); 201–300: 4.050781(6); 301–400: 4.612407(6); 401–500: 5.26958(7); 501–600: 6.084604(7) |
| train/ufo/strict_shadow_tracking_episode_len | 32 | 2.894926 | 9.882597 | 6.746812 | 2.894926 | 10.2986 | 6.899429 | 10.03444 | 101–200: 3.681051(6); 201–300: 5.24738(6); 301–400: 6.69004(6); 401–500: 8.051608(7); 501–600: 9.403699(7) |
| train/ufo/strict_velocity_exposure_fraction | 32 | 0.001210973 | 0.0007417098 | 0.00113631 | 0.0006408691 | 0.001487179 | 0.00120843 | 0.001426575 | 101–200: 0.001296481(6); 201–300: 0.001347602(6); 301–400: 0.001253137(6); 401–500: 0.001068747(7); 501–600: 0.000785338(7) |
| train/ufo/strict_velocity_transition_fraction | 32 | 0.08707682 | 0.04052734 | 0.06100972 | 0.03649902 | 0.08854166 | 0.06528727 | 0.08036295 | 101–200: 0.07596164(6); 201–300: 0.06988525(6); 301–400: 0.06641303(6); 401–500: 0.05522228(7); 501–600: 0.04174223(7) |
| train/ufo/target_M | 500 | 2.397962 | 0.5298274 | 0.6696919 | 0.3010613 | 2.397962 | 0.5941019 | 1.408453 | 101–200: 1.080257(100); 201–300: 0.5837345(100); 301–400: 0.5715456(100); 401–500: 0.5502527(100); 501–600: 0.5626701(100) |
| train/ufo/target_Q | 500 | -2.284226 | -3.455785 | -2.432329 | -3.506976 | -1.839875 | -2.289121 | -1.900273 | 101–200: -2.040919(100); 201–300: -1.989478(100); 301–400: -2.277086(100); 401–500: -2.681816(100); 501–600: -3.172346(100) |
| train/ufo/target_auxQ | 500 | -3.144984 | -4.365604 | -3.526793 | -4.451354 | -2.957711 | -3.466153 | -3.007667 | 101–200: -3.050601(100); 201–300: -3.144145(100); 301–400: -3.452252(100); 401–500: -3.812214(100); 501–600: -4.174751(100) |
| train/ufo/terminated_transitions | 32 | 4394 | 1781 | 3110.844 | 1683 | 5206 | 2865 | 5099.35 | 101–200: 4851.5(6); 201–300: 3894.833(6); 301–400: 2999(6); 401–500: 2369.571(7); 501–600: 1784(7) |
| train/ufo/tracking_episode_starts | 32 | 4495 | 735 | 1418.125 | 555 | 4495 | 1164 | 2564.1 | 101–200: 2725.5(6); 201–300: 1736.333(6); 301–400: 1239.167(6); 401–500: 936.4286(7); 501–600: 659.8571(7) |
| train/ufo/tracking_q_rmse_rad | 32 | 0.06090361 | 0.2573837 | 0.1807684 | 0.06090361 | 0.2723455 | 0.1745787 | 0.2719995 | 101–200: 0.1043651(6); 201–300: 0.1369551(6); 301–400: 0.16874(6); 401–500: 0.214735(7); 501–600: 0.2601547(7) |
| train/ufo/tracking_unique_env_fraction | 32 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 101–200: 0.5(6); 201–300: 0.5(6); 301–400: 0.5(6); 401–500: 0.5(7); 501–600: 0.5(7) |
| train/ufo/tracking_windows_completed | 32 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 101–200: 0(6); 201–300: 0(6); 301–400: 0(6); 401–500: 0(7); 501–600: 0(7) |
| train/ufo/training_velocity_count | 32 | 581 | 216 | 343.4062 | 190 | 581 | 341 | 467.35 | 101–200: 453(6); 201–300: 417(6); 301–400: 354.3333(6); 401–500: 297.4286(7); 501–600: 223(7) |
| train/ufo/truncated_transitions | 32 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 101–200: 0(6); 201–300: 0(6); 301–400: 0(6); 401–500: 0(7); 501–600: 0(7) |
| train/ufo/unc_Q | 500 | 2.237383 | 0.1415226 | 0.5634407 | 0.1389695 | 2.237383 | 0.3360159 | 1.726145 | 101–200: 1.449124(100); 201–300: 0.6651754(100); 301–400: 0.3445786(100); 401–500: 0.2053814(100); 501–600: 0.1529436(100) |
| train/ufo/unc_auxQ | 500 | 0.5659184 | 0.1491136 | 0.2532258 | 0.14318 | 0.5692017 | 0.205754 | 0.4859417 | 101–200: 0.4429275(100); 201–300: 0.2927705(100); 301–400: 0.2073019(100); 401–500: 0.168941(100); 501–600: 0.1541881(100) |
| train/ufo/velocity_reward_coverage_miss_count | 32 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 101–200: 0(6); 201–300: 0(6); 301–400: 0(6); 401–500: 0(7); 501–600: 0(7) |
| train/ufo/wall_time_s | 500 | 70.15152 | 1065.73 | 562.0614 | 70.15152 | 1065.73 | 560.5268 | 1005.604 | 101–200: 161.5999(100); 201–300: 363.2305(100); 301–400: 562.8461(100); 401–500: 761.738(100); 501–600: 960.8926(100) |
| train/ufo/z_norm | 500 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 101–200: 16(100); 201–300: 16(100); 301–400: 16(100); 401–500: 16(100); 501–600: 16(100) |

按实际采样轮记录的关键原始数值（—表示该step没有记录该指标，未插值）：

| step | env/episode_len | train/ufo/strict_shadow_episode_len | train/ufo/physics_peak_normalized_velocity | train/ufo/first_step_samples | train/ufo/first_step_terminated | train/ufo/hard_position_count | train/ufo/training_position_count | train/ufo/critic_loss | train/ufo/Q_fb | train/ufo/mean_disc_reward | train/ufo/applied_action_saturation_fraction |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 116 | 3.242604 | 2.866636 | 2.839489 | 8901 | 1 | 6209 | — | 0.6747029 | 14.31711 | -0.005469165 | 0.08476987 |
| 132 | 4.389223 | 3.353566 | 3.092317 | 5142 | 0 | 8185 | — | 0.5302256 | 14.30789 | -0.06213149 | 0.1085382 |
| 148 | 4.67519 | 3.635956 | 3.453624 | 5285 | 1 | 7929 | — | 0.4755211 | 19.36636 | -0.06846077 | 0.1090796 |
| 164 | 4.735113 | 3.761237 | 3.168693 | 5266 | 0 | 8383 | — | 0.3935734 | 19.35192 | -0.1041548 | 0.1103763 |
| 180 | 4.732874 | 3.774213 | 3.597043 | 5068 | 1 | 8462 | — | 0.3428875 | 20.68488 | -0.1653017 | 0.1114414 |
| 196 | 4.813862 | 3.758229 | 3.624638 | 4386 | 0 | 7917 | — | 0.3112259 | 22.7746 | -0.1737385 | 0.1204622 |
| 212 | 5.418267 | 4.183564 | 3.439057 | 4567 | 1 | 8437 | — | 0.3490038 | 23.69078 | -0.2780433 | 0.1273866 |
| 228 | 5.738749 | 4.268227 | 3.046757 | 4434 | 2 | 8273 | — | 0.6365554 | 25.96496 | -0.2782426 | 0.123797 |
| 244 | 5.650683 | 4.267345 | 3.46406 | 3941 | 0 | 7163 | — | 0.3374805 | 26.92186 | -0.3598989 | 0.1273158 |
| 260 | 6.416116 | 4.69247 | 3.575933 | 3776 | 1 | 7864 | — | 0.4247847 | 29.31008 | -0.3514102 | 0.1298863 |
| 276 | 6.67714 | 4.952871 | 2.795925 | 3713 | 0 | 7203 | — | 0.3439524 | 31.32071 | -0.4100676 | 0.1271743 |
| 292 | 6.750386 | 5.068582 | 3.307463 | 3244 | 0 | 6877 | — | 0.2693776 | 32.85064 | -0.4028611 | 0.1271778 |
| 308 | 7.241399 | 5.272362 | 2.868822 | 3521 | 0 | 7360 | — | 0.2917394 | 33.93788 | -0.4472496 | 0.1229938 |
| 324 | 7.204011 | 5.126902 | 2.580658 | 2978 | 0 | 7036 | — | 0.3205314 | 33.27315 | -0.4652422 | 0.125046 |
| 340 | 7.777085 | 5.599146 | 2.727621 | 3043 | 0 | 7367 | — | 0.353238 | 36.01199 | -0.5076241 | 0.1278165 |
| 356 | 8.184786 | 5.874346 | 2.510626 | 3266 | 0 | 7263 | — | 0.5451517 | 38.52302 | -0.5169647 | 0.1209151 |
| 372 | 7.63798 | 5.223684 | 2.985488 | 2870 | 0 | 6955 | — | 0.3080416 | 39.15378 | -0.5273315 | 0.1191919 |
| 388 | 8.155042 | 5.596652 | 3.100787 | 2578 | 0 | 7365 | — | 0.4183443 | 40.70386 | -0.5807793 | 0.1216776 |
| 404 | 9.007048 | 6.12685 | 3.117105 | 2833 | 0 | 7910 | — | 0.3382444 | 45.28226 | -0.608907 | 0.1183746 |
| 420 | 9.182823 | 6.000425 | 3.443945 | 2399 | 0 | 7239 | — | 0.3471578 | 44.87133 | -0.626411 | 0.1177908 |
| 436 | 9.576562 | 6.181707 | 4.016886 | 2465 | 0 | 7599 | — | 0.3789234 | 46.9728 | -0.6628789 | 0.1195033 |
| 452 | 9.78824 | 6.308479 | 5.322248 | 2357 | 1 | 7748 | — | 0.3751495 | 48.14783 | -0.6354347 | 0.1135572 |
| 468 | 9.932696 | 6.464756 | 2.86421 | 2170 | 0 | 7674 | — | 0.4305226 | 52.72331 | -0.6965017 | 0.1105285 |
| 484 | 10.10987 | 6.295781 | 3.0046 | 2228 | 0 | 7757 | — | 0.4285131 | 51.25752 | -0.7264801 | 0.1013838 |
| 500 | 11.48686 | 7.162583 | 3.228644 | 2262 | 0 | 7667 | — | 0.445709 | 51.56657 | -0.7141971 | 0.09942538 |
| 516 | 11.02647 | 6.517385 | 3.005218 | 1991 | 0 | 7095 | — | 0.4539869 | 60.30227 | -0.74732 | 0.1013608 |
| 532 | 11.80224 | 6.955742 | 2.681978 | 1820 | 1 | 7312 | — | 0.4318179 | 57.08454 | -0.7940394 | 0.0971202 |
| 548 | 11.98919 | 7.056915 | 3.462106 | 1717 | 0 | 7807 | — | 0.5825205 | 61.19158 | -0.7983723 | 0.1023781 |
| 564 | 12.59965 | 7.098494 | 3.467136 | 1774 | 0 | 8111 | — | 0.5247236 | 60.83096 | -0.80589 | 0.09966952 |
| 580 | 13.44724 | 7.834882 | 2.520187 | 1815 | 0 | 8550 | — | 0.5210966 | 66.22386 | -0.8620299 | 0.09887165 |
| 596 | 14.25906 | 7.775401 | 2.706543 | 1696 | 0 | 7376 | — | 0.5637919 | 67.42114 | -0.9025782 | 0.100409 |
| 600 | 13.85121 | 7.879843 | 2.94907 | 1785 | 1 | 7607 | — | 0.5522153 | 63.62383 | -0.8932413 | 0.09381192 |

本阶段冻结的实际解析配置：
```yaml
actor:
  model:
    model_type: mlp_policy
    model_path: ''
    policy_setup: rpo-v2-bfm-ufo
    obs_dim: 375
    action_dim: 23
    num_action_chunks: 1
    hidden_dim: 2048
    precision: '32'
    add_value_head: false
    is_lora: false
    lora_rank: 32
    critic_obs_dim: 733
    bfm_profile: ufo_native_bringup
    motion_root: /workspace/data/ds_fb0ecda9e70644f2afdd
    asset_sha256: b58868a782d8863ac38059eb4eab89d27ee1d70565bf58f5a055f44510c8e842
    batch_size: 2048
    replay_capacity: 4000000
    hidden_layers: 6
    topology_id: one_learner_three_samplers
    learning_profile: physical_limits_stable_slew_v4
    architecture_profile: native_fb_v1
    rollout_expert_trajectories: true
    auxiliary_profile: soft_velocity_v7
    scaling_profile: env6144_batch2048_v6
    control_profile: reachable_v2
    optimizer_profile: stabilized_v7
    context_profile: environment_tracking_v10
    contact_profile: native_mjcf_viability_v13
    physics_profile: native_mjcf_viability_v13
    codec_sha256: dce25d554ceb7fa69287515946d2f5cd9d41d14f1b1ef38715b819d7e962b6ef
  group_name: ActorGroup
  training_backend: ufo_agent
  micro_batch_size: 2048
  global_batch_size: 2048
  seed: 42
  enable_offload: false
  optim: null
  fsdp_config: null
weight_syncer:
  type: patch
  patch:
    snapshot_device: cpu
    delta_encoding: true
    compression: none
    init_sync:
      enabled: true
      prefixes: null
      bucket_size: 134217728
cluster:
  num_nodes: 1
  component_placement:
    actor: 0
    env, rollout: 1-3
runner:
  task_type: embodied
  logger:
    log_path: /workspace/output
    project_name: rlinf
    experiment_name: RPO UFO MJLab v13 RSI viability 500-update observation
    logger_backends:
    - tensorboard
  max_epochs: 4096
  max_steps: -1
  only_eval: false
  use_training_pipeline: false
  overlap_env_bootstrap: false
  val_check_interval: -1
  save_interval: -1
  resume_dir: /workspace/resume/global_step_100
  ckpt_path: null
algorithm:
  loss_type: ufo_fb_cpr_aux
  update_epoch: 1
  normalize_advantages: false
  kl_penalty: kl
  group_size: 1
  reward_type: action_level
  logprob_type: action_level
  entropy_type: action_level
  adv_type: gae
  loss_agg_func: token-mean
  bootstrap_type: standard
  gamma: 0.98
  gae_lambda: 1.0
env:
  group_name: EnvGroup
  train:
    env_type: mjlab
    total_num_envs: 6144
    auto_reset: true
    ignore_terminations: false
    use_rel_reward: false
    seed: 42
    group_size: 1
    reward_coef: 1.0
    use_fixed_reset_state_ids: false
    rollout_epoch: 1
    max_episode_steps: 1000
    max_steps_per_rollout_epoch: 4
    video_cfg:
      save_video: false
      info_on_video: false
      fps: 50
      video_base_dir: /workspace/output/video/train
    init_params:
      id: RPO-UFO-MJLab-RSI-viability-v13
      num_envs: null
      max_episode_steps: 1000
      task_description: RPO V2 BFM-Zero online interaction with pinned observation-only
        expert motion.
  eval:
    env_type: mjlab
    total_num_envs: 3
    auto_reset: true
    ignore_terminations: false
    use_rel_reward: false
    seed: 42
    group_size: 1
    reward_coef: 1.0
    use_fixed_reset_state_ids: true
    rollout_epoch: 1
    max_episode_steps: 1000
    max_steps_per_rollout_epoch: 1000
    video_cfg:
      save_video: false
      info_on_video: false
      fps: 50
      video_base_dir: /workspace/output/video/eval
    init_params:
      id: RPO-UFO-MJLab-RSI-viability-v13
      num_envs: null
      max_episode_steps: 1000
      task_description: RPO V2 BFM-Zero online interaction with pinned observation-only
        expert motion.
rollout:
  group_name: RolloutGroup
  backend: huggingface
  enable_offload: false
  pipeline_stage_num: 1
  enable_torch_compile: false
  enable_cuda_graph: false
  collect_transitions: true
  collect_prev_infos: true
  model:
    model_path: ''
    precision: '32'
reward:
  use_reward_model: false
critic:
  use_critic_model: false

```

## 最终判定与后续建议

```json
{
  "decision": "stop at frozen 600; no long training, no independent evaluation",
  "workflow_qualified": true,
  "walking_qualified": false,
  "gate": {
    "strict_shadow_last_two_ge35": false,
    "global_physics_rated_ratio_le2_5": false,
    "first_step_training_termination_le0_1pct": true,
    "independent_eval_episode_last_two_ge100": false
  },
  "train_id": "trn_287ae01b97e6403abd30",
  "job_id": "job_8b575c408625492ea7e2",
  "run_id": "run_bc7df2383d8f498bb08d",
  "model_version_id": "mv_5db6825607364c9fb1cc",
  "segment_updates": 500,
  "cumulative_updates": 600,
  "new_transitions": 786432,
  "cumulative_transitions": 983040,
  "started_at_utc": "2026-10-12T16:50:40.593261+00:00",
  "finished_at_utc": "2026-10-12T17:13:29.742077+00:00",
  "episode_five_windows": [
    4.431477586428325,
    6.108556906382243,
    7.700050433476766,
    9.869157654898506,
    12.710721833365303
  ],
  "strict_shadow_five_windows": [
    3.5249727169672647,
    4.572176535924275,
    5.448848724365234,
    6.362940311431885,
    7.302665982927595
  ],
  "first_step_training_terms": 10,
  "first_step_samples": 105291,
  "first_step_training_fraction": 9.49748791444663e-05,
  "first_step_strict_any": 1566,
  "first_step_strict_fraction": 0.01487306607402342,
  "strict_last_window_change_vs_v12_fraction": -0.02230081056113442,
  "physics_peak_change_vs_v12_fraction": 0.1901358837822149,
  "physics_peak_radps": 42.577980041503906,
  "physics_peak_rated_ratio": 5.322247505187988,
  "engineering_analysis": "finite149 metrics, identical restore fingerprint, six Adam600; all substep rewards cover violations, references/context/replay consistent; GPUs released",
  "quality_analysis": "training windows improve but position tolerance increases survival; strict last window 7.30 is slightly below v12 7.47; right ankle peak42.58 exceeds v12 35.78; no normal walking claim",
  "next_recommended": "freeze v13; diagnose joint-limit overshoot and coupled effective inertia/contact moment with paired replay before choosing further controller or armature changes; rotor inertia/reduction remain unavailable",
  "resume_candidate": {
    "available": false,
    "reason": "v13流程验证尚未确认或100→600观察预算已完成"
  },
  "rejected_continuation_preview": {
    "status": 422,
    "body": {
      "detail": "MJLab v13 permits only fresh 100 updates and complete-state 100→600 observation"
    }
  }
}
```
