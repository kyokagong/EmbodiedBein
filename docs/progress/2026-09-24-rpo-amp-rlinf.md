# RPO V2 AMP / RLinf 接入进度与运行交接

日期：2026-09-24。设计真源为 [AMP 接入设计](../design/15-amp-rpo-rlinf-integration.md)，实施步骤见[执行计划](../superpowers/plans/2026-09-24-rpo-amp-rlinf.md)。本文只记录已获得的运行证据和未完成的验收。**接入、训练、模型登记和双仿真评估已跑通；目前没有能正常直立和行走的合格模型。**

## 硬件判断纠正

主机实际有 **4 张 RTX 4090**，EmbodiedBein API 在 `127.0.0.1:31800`，ebd 正常。此前把当前执行沙箱不能调用 `nvidia-smi`，以及错误地检查 `:8000` 端口，推断成主机没有训练环境，这是错误判断。随后在主机确认四卡，并由 ebd 动态创建的 Isaac/RLinf 任务容器实证 GPU 可用；未绕过平台直接运行训练或评估。
本轮两条各请求 1 张 GPU 的训练没有并发，是调度器按每条 64 GiB 主机内存预算准入的结果；其余三张 4090 空闲不等于整机内存也可承载第二条任务。

## 已完成的受管链路

| 阶段 | 实际证据 | 范围 |
|---|---|---|
| 官方动作数据 | v1 `ds_4acffb5b17264907bbd2` 关节错列，v2 `ds_5c5b9a5138b84a179839` 混入腾空慢速段，均已标 `invalid`；v3 Job `job_873d9e73e5bf4ddc95f9`、Run `run_847fa6972c1b4eff9c4f` 生成 `ds_3df5ce9ac7a0453089cb`：12 段、1441 帧，站立 3.82 s、接地步行 25.0 s | 受管诊断数据，来源、3× 时间缩放、接地和转角筛选已记录；原始动作权利链和动力学可执行性仍待独立核验 |
| Isaac 数值探针 | Job `job_7f156fcc482e4d2f95b8`、Run `run_3a0638da132e42d2bd9c` 成功；2 环境、actor 780、critic 1390、动作 23、AMP 帧 74、3 帧窗口、局部 reset | 证明目标镜像、GPU 和基础接口可用；不是策略收敛证据 |
| Isaac 动作运动学回放 | Job `job_34a23c7cf9ec42c2b671`、Run `run_28597746236940d4be88` 成功；12 段动作中间帧，目标 RPO 资产六关键点最大 FK 误差 `0.000051 m` | 证明数据关节序和资产运动学匹配；不证明 PD/接触下可持续执行 |
| Isaac PD 动作跟踪诊断 | Job `job_9280291c64a841ba88d5`、Run `run_f3e602b876f44e76aedb`；站立片段完整跟踪 190/190 步；11 段步行全部在 28–64 步提前终止，关节 RMSE `0.059–0.109 rad` | 当前时间缩放的步行参考尚未证明动力学可执行；这个测试从片段首帧重置并直接给定关节目标，不能单独判定原数据还是 PD/接触参数是根因 |
| 官方替代片段 v4 | 转换 Job `job_137254f4ff6643a099f9`、Run `run_61c14fc53904494190ec` 生成 `ds_518dad79f5d3482e8db2`：A1 站立＋`36_11` 时间缩放/接地筛选，6 段、875 帧，其中步行 13.68 s。重新以源重力方向和根速度初始化的 PD 探针 Job `job_5c417cb2212d43398026`、Run `run_835818efc0d548808bb9`：5 段步行全部在 30–63 步提前终止 | 结构和 FK 通过；更公平的动态探针仍未通过，v4 仅保留为诊断数据，不能作为合格步行先验 |
| 默认姿态零动作基准 | Isaac Job `job_9039dfd0a263471cb7e9`、Run `run_224d480feafa4a0e8265`，同一资产/PD/50 Hz：站立指令 59 步终止、漂移 `0.529 m`；0.3 m/s 行走指令 76 步终止、漂移 `0.537 m` | 默认姿态自身无法被零动作维持；模型约 50–55 步跌倒不是单纯的 ONNX 导出或跨仿真错误，必须学出主动平衡 |
| 官方站姿初态诊断 | Isaac Job `job_a513c55bbb9b432fa1c7`、Run `run_24040658490343f0b967`：同一 A1 站姿恒定 PD 目标，从默认初态 52 步跌倒；从 A1 关节姿态/0.711 m 高度起步完整存活 1000 步 | 证实源站姿提供稳定起点；这只是初态诊断，不证明从默认姿态起步或主动行走。先前 `job_bd54b16e371d455992fc` 默认初态单独测试为 58 步 |
| v3 50 轮训练 | `trn_922f51e65efb4c80bdd2`、Job `job_ff5a245daa03427a8215`、Run `run_240e69fd4b3d45d5b799`、模型 `mv_3ed9c060841e42adb7c5` 成功；512 环境×24 步×50 轮；第 50 轮内部评估平均存活 55.8 步，D loss `0.0020`，style reward 均值 `1.03e-4` | 管线和模型登记成功，但判别器饱和、基础直立没有改善，不能算学会 |
| v3 MuJoCo 独立评估 | Job `job_ddca72c364a341f3938b`、Run `run_9df3abf5bab446d6b44b`、报告 `ev_7650582f7b5b4c7ea515`；实际 ONNX 3 种子，0/3 成功、3 次跌倒，58–73 步，速度 RMSE `0.454 m/s` | `qualification=fail` |
| v3 Isaac 独立评估 | Job `job_336229fd669f49a48c46`、Run `run_3596a189a48849b295fe`、报告 `ev_9f0e8d31690944fa8478`；实际 ONNX 3 种子，0/3 成功、3 次跌倒，50–55 步，速度 RMSE `0.514 m/s` | `qualification=fail`；跨仿真一致地未达到直立/行走 |
| 纯任务 PPO 对照独立评估 | step-50 checkpoint 经平台恢复到 51 并登记模型 `mv_640250fac99b48acbe2d`，训练 `trn_e95329d043954ee08328`、Run `run_c928d86fedd2496ca792`。默认初态 Isaac `ev_eccfce3dc1d14ff6b69a`、MuJoCo `ev_b6f73019086343098f04`：均 0/3 成功、3/3 跌倒 | 说明失败并非 AMP 独有；style=0 控制组同样未学会 |
| 源姿态 Profile 接线 | 发布训练 `env_2b6cd350f88940d48606`、Isaac 评估 `env_676c92ff44514b2f85aa`、MuJoCo 评估 `env_2f7dbe79b9894c8a8034`。旧模型按新 Profile 评估仍均 0/3：Isaac `ev_3d857b052e174baab65f`、MuJoCo `ev_5945cecd116f49f3ae1e`。实际环境复核 Job `job_01bf74afd187478b8306`、Run `run_26f3e55e60ef4e969156`：初始关节与 AMP 首帧误差均为 0，根高 `0.710682 m` | 初态设置写入环境快照/评估计划，并已在受管 Isaac 容器中测得生效；旧权重不会因修改初态自动稳定 |
| A1 起点 25 轮 AMP 诊断 | 训练 `trn_8c5a1ced47eb43b088c8`、Job `job_06cd107901244ac5a01d`、Run `run_8f64d9a8c71245838180`、模型 `mv_935e5b12a3204ee9bdc4` 成功；512 环境×24 步×25 轮；内部确定性评估平均存活 `113.77` 步，随机训练 rollout 约 68–74 步 | 比默认起点的 50–55 步改善，但远不足 1000 步；不能视为站立/行走成功 |
| A1 起点独立评估 | Isaac Job `job_3f0cbe8c681b49efb4a6`、Run `run_0a521ba179064cbf9db8`、报告 `ev_149314aa34d640218e31`：0/3、3 跌倒，75–195 步。MuJoCo Job `job_5750d73d13104042a7d1`、Run `run_3d6ce7094d24462c98de`、报告 `ev_7a74af3867a04c1a9a68`：0/3、3 跌倒，62–110 步 | 两套件均 `qualification=fail`；尚未获得能直立和行走的 actor |
| A1 起点低噪声 25 轮 | Profile `env_ea6f17cb14864856bc8d` 固定初始 actor `logstd=-2`；训练 `trn_3dd513a58a1540028491`、Run `run_f8aefb63f2a640bfa02e`、模型 `mv_7fdefbc3bda8493cac5d`。内部确定性评估平均 `244.94` 步；独立 Isaac 报告 `ev_386cdd50711447d8b6e9`：站立种子 0 存活 1000 步且漂移 `0.045 m`，另一站立和步行种子跌倒，1/3 成功 | 已有局部直立进展；步行仍失败，不满足套件，也不满足默认初态验收 |
| 低学习率对照 | 同一数据/Profile、actor LR `1e-4` 的训练 `trn_80a6a63a6c894bf78de7`、Run `run_5f25cb5933114d38bc02`、模型 `mv_7f7f0b8078fa4e1baf62` 完成 25 轮；内部平均 `143.61` 步。独立 Isaac `ev_128176e1f8f1475295ba`、MuJoCo `ev_b436a0e22d73461bac2e` 均 0/3 | 本次低学习率未优于首个低噪声模型；不据内部指标宣称会走 |
| 同初态 100 轮独立评估 | 累计 100 轮模型 `mv_a309fe67335b44f5927e`；Isaac 报告 `ev_22b9c16f82bb45c9a897`：种子 0 站立 1000 步成功；步行种子 1 存活 1000 步但仅前进 `0.012 m`，未达到 4 m；站立种子 2 于 117 步跌倒，整体 1/3。MuJoCo 报告 `ev_bf86d06a12054aa09fd6`：0/3、全跌倒 | 内部平均存活高不能证明会走；速度奖励容差允许“站着不走”骗分，跨仿真仍未通过 |
| 同初态 50 轮独立评估 | 从原 step-25 恢复到累计 50 轮的训练 `trn_f5c844f1596d4a3fb157`、Run `run_08cb1838ad2a4afd8570` 登记模型 `mv_35d45dabec8e4bdc9097`；内部确定性平均存活 `742.92` 步。独立 Isaac 报告 `ev_b4b12845f0434277a145`：1/3，仅种子 0 站立成功，步行种子 1 于 235 步跌倒；MuJoCo 报告 `ev_ae0e35bb610e4c19830f`：0/3 | 候选 checkpoint 的内部指标仍不能替代独立站/走病例 |
| 速度奖励修订训练与评估 | 原速度奖励函数 `exp(-||v-v_cmd||²/std²)` 且 `std=0.5`，0.3 m/s 指令下零速也有约 `0.697` 倍满分。新训练 Profile `env_be6aa9e7a6ea4e0fa3a2` 冻结 `velocity_tracking_std=0.2`，旧 Profile 不变；受管训练 `trn_5b5e306e6c834f7caa0d`、Run `run_f91ec71a94d440e78e23` 登记 step-100 模型 `mv_31f294f2c4934b04aac5`。独立 Isaac `ev_8648009d57224a6482d3`：1/3；步行种子 **前进 0.495 m、121 步跌倒**；MuJoCo `ev_00e828aa2d6141eaa0d3`：0/3 | 奖励修订纠正了“站着不走”，但尚无持续稳定行走；新配置不是原训练的 exact resume |
| 低学习率 PPO 候选独立评估 | 从 `trn_83e21c28931d47bf8f5f` 的 step 75 完整 checkpoint 显式恢复一轮，训练 `trn_36aa97a0c2634a98b943`、Run `run_e19640063e43452db70d` 登记实际 step-76 模型 `mv_e71131b812cf4005919e`。独立 Isaac Job `job_11ca291a34e543058ca1`、Run `run_7d2ad8d5740a430dbc29`、报告 `ev_0161c06c07a4481cb091`：1/3；站立 seed 0 存活 1000 步，步行 seed 1 存活 1000 步却仅前进 **0.027 m**，站立 seed 2 于 85 步跌倒。MuJoCo Job `job_23afdded3b2949bfb71c`、Run `run_57c8f73bf7194bf0b3f3`、报告 `ev_8c3cce74293b4dcf9f47`：0/3、全跌倒 | 低学习率和选取早期候选没有解决“站着不走”或跨仿真稳定性；不能将内部 step-75 均值冒充独立验收 |
| 高速度权重 AMP 诊断 | 新训练 Profile `env_6b5ac6cea02a4caca093` 固定 `velocity_reward_weight=8`、`velocity_tracking_std=0.2`、A1 初态、`logstd=-2`；训练 `trn_e269ecdbed714fabbdc3`、Run `run_f99d2f217e6640de99f8` 完成 150 轮并登记模型 `mv_730b60a9fcda4188a9b3`。内部确定性评估 step 25/50/75/100/125/150 平均存活约 **200/114/140/184/280/205 步**。独立 Isaac Job `job_95558e6c60e04edeb5be`、Run `run_ef02568be3a84aeda4a1`、报告 `ev_1dc665c42f434d9597af`：1/3；步行 seed 1 在 104 步跌倒并**后退 0.542 m**，站立 seed 2 于 78 步跌倒。独立 MuJoCo Job `job_b4928b8a730c4dba9490`、Run `run_06b9b0b53aab4bda8633`、报告 `ev_fa9ea2d8768e459f9b4d`：0/3；步行 seed 1 在 106 步跌倒并后退 0.421 m | 更高速度奖励没有换来受控前进，反而出现后退/跌倒；该修订不继续长训 |
| Isaac 执行器延迟对照 | 独立评估 `ev_4f9ad8d0cafb4572a998` 记录种子 0 腿/踝延迟 1/1 且站满，种子 2 为 2/2 且跌倒；单独重放种子 2 `ev_438a503f35ee4e32a153` 也跌倒。新评估 Profile `env_cba914331c274a9b9ca5` 固定四组延迟 0，种子 2 仍 76 步跌倒，报告 `ev_eb484ec25fe748c6b64d` | 延迟与失败相关但不是单独根因；不据此启动零延迟训练 |
| MuJoCo 对照校准 | 修正原评估器的关节顺序为目标 MJCF 的实际顺序，并把 PD 力矩上限与 Isaac 资产对齐；修正后首个低噪声模型 MuJoCo 仍 0/3，报告 `ev_f141a0b445d14c398f7a`。同一 MJCF 的源 A1 站姿恒定 PD 诊断，评估 `ev_b2832f541e33478eb4ac`：5 ms 物理步长 **97 步**过度倾斜跌倒；MJCF 原生 1 ms 步长 **32 步**跌倒 | 与 Isaac 源站姿恒定 PD 1000 步相比，静态基线本身存在跨仿真差异；MuJoCo 失败不能全归因于 ONNX actor，也不能只靠缩小步长解决 |
| MuJoCo 接触诊断 | 独立受管评估 Job `job_ec22d7092d2b4e92bc70`、Run `run_1ca3f24f2d3a48ac9f52`、报告 `ev_75eff4cd9e15407db532`：静态 PD 探针分别以源地面 `condim=1` 和临时 `condim=3` 运行，在 5 ms 与 1 ms 物理步长下结果逐项相同，仍分别在 97 和 32 策略步跌倒。后续只读探针 Job `job_c7c29609a8674773b7b2`、Run `run_8dfbf3981365441bb25e` 测得初始有 **15 个实际接触、维度 4** | 地面 `condim` 不是根因，且并非初始无接触；临时接触参数改动未进入正式评估路径，已从代码撤回。跨仿真动力学差异仍待定位 |
| 平台元数据 | 发布 `ee_rpo_v2_amp_locomotion_23dof_v1`、`tsk_rpo_amp_stand_walk`、`env_train_rpo_v2_amp_stand_walk_ppo`；数据、镜像、Run、Artifact、ModelVersion 和血缘按 EmbodiedBein 路径登记 | 未申请部署或真机放行 |
| 两轮短训 | 训练 `trn_3f7322ffc99d41c0b60e`、Run `run_7c03af64f2924f079bea` 成功；256 环境×24 步，PPO actor/critic 与 AMP 判别器均更新；完整 AMP checkpoint 和恢复清单生成；模型 `mv_ba6ada6752234e968f42` | 仅用于诊断；`amp/style_reward_mean=0` 是预定前 10 轮 warmup，不能证明风格奖励有效或模型会走 |
| 恢复 | 训练 `trn_157abf996dc448a69abb` 从 step 2 恢复至 step 3 成功，Run `run_088b7c355c0d4b15a142`、模型 `mv_08f1305eb5c74d80b5a0`；平台标明 `trainer_state`，环境/rollout 状态不可精确恢复 | 已证明进程恢复并继续更新；尚未做固定 batch 等价性检查 |
| MuJoCo 独立评估 | Job `job_9351a1e8db7545a79979`、Run `run_a7f605315ee54d7ebc76`、EvalReport `ev_c72278a689ac4919989c`；实际 ONNX 闭环 3 个诊断种子，**0/3 成功、3 次跌倒** | `qualification=fail` |
| Isaac 独立评估 | Job `job_4b49207ddb934a9da839`、Run `run_2a6ee7d9854947fab78e`、EvalReport `ev_011bf5aa32974ea181df`；实际 ONNX 闭环 3 个诊断种子，**0/3 成功、3 次跌倒**；ONNX/PyTorch 最大误差 `3.7e-8` | `qualification=fail`；两轮模型约 61–72 步跌倒 |

前两次短训失败记录保留：`job_074a6701aa804d47bbfd` 暴露 RLinf `EnvOutput.to_dict()` 丢弃特权 critic 观测，已在镜像补丁修复；`job_5912c6650a07428da9c2` 实际完成两轮 PPO 后因 AMP 状态写在 `actor/amp/`、与平台要求的 checkpoint 根目录 `amp/` 不一致而拒绝发布，已修复并重跑。失败不能算成功模型。

## 训练结果和后续验收

速度奖励 `std=0.2` 的 step-100 模型已通过 EmbodiedBein 的 `trainer_state` 续训预览，按**累计 300 轮**提交受管训练 `trn_61031d76b7c146a7a41d`、Job `job_ebc4302ef77d43e4b17e`、Run `run_f645df2ed64843169f06`。续训沿用源模型、训练 Profile、动作数据和固定镜像 digest。内部确定性评估在 step 125/150/175/200 的平均存活约为 **362/215/166/145 步**；连续退化后按早停规则通过平台取消，最后完整 checkpoint 为 step 200，取消前运行至 step 211。由于未成功完成最终登记，这条 Run **没有新的 ModelVersion 或独立 ONNX 结论**；之前 step-100 模型独立步行只前进 0.495 m 后跌倒。

同一 A1 初态、低噪声、收紧速度奖励和 512×24 rollout，另通过平台提交 `amp_discriminator.style_weight_max=0` 的 PPO 对照 `trn_64b127e9476b4417878b`、Job `job_df163e1dad6d4da3b5be`、Run `run_b76fba8699504314b24a`，目标累计 300 轮。该对照保留同一模板和数据身份以便比对，只让风格奖励权重为零；AMP 取消释放主机内存预留后，平台已将它调度到 GPU 0。具体效果仍须独立评估。
对照实际日志中 `amp/style_reward_mean=0`；内部确定性评估 step 25/50/75/100/125/150 平均存活约 **185/315/499/86/81/73 步**，AMP 相同设置 step 25/50/75/100 则约 **272/167/199/332 步**。PPO 对照在三个后续检查点连续退化，按早停规则通过平台取消；取消前约完成 step 155，最近完整 checkpoint 为 step 150。这既显示两条曲线波动，也说明单点内部存活不能证明 AMP 带来增益；取消的 Run 没有新 ModelVersion 或独立 ONNX 结论。

这两条长诊断暴露了选模缺口：旧镜像在每次 AMP 保存后只保留最新 checkpoint，较好的 step 75 在后续运行中已被清理，不能追认其模型表现。现已将 AMP 诊断镜像改为保留最近 4 个完整 checkpoint，并为平台恢复请求增加已保留 step 的显式选择与校验；其他训练栈仍按原保留策略。该修订只作用于之后启动的新镜像，不能恢复已删文件，也不是长期 best 引用租约。

受管端到端预览还查出链式恢复指纹错误：`resume_source_image` 被第二次恢复时的新 digest 覆盖，导致合法的 step 200 预览报“恢复配置改变了训练兼容性指纹”。已修复为沿用最初镜像身份；API 重建后对 `trn_61031d76b7c146a7a41d` 的显式 `checkpoint_step=200` 预览成功，已删除的 step 175 正确拒绝。低学习率 PPO 对照 `actor.optim.lr=1e-4` 已按相同数据、环境、512×24、style=0 提交：训练 `trn_83e21c28931d47bf8f5f`、Job `job_756022768de84a93a9e6`、Run `run_c97efe330a764f069c57`。其内部确定性评估 step 25/50/75/100/125/150 平均存活约 **176/334/430/280/128/81 步**，连续退化后也通过平台早停；取消时最近四个完整 checkpoint 为 75/100/125/150。只有这条新 Run 采用 4 点保留策略。

首次指定 step 75 的恢复预览发现 RLinf adapter 仍默认选最新 step 150，导致回执不一致；已让 adapter 按已验证的 `resume_step` 挂载，并做回归测试。重新预览成功后，通过平台提交候选 `trn_36aa97a0c2634a98b943`、Job `job_2e600aee69e0488da38f`、Run `run_e19640063e43452db70d`，从 step 75 恢复到累计 76 轮以登记新模型。**这会多执行一次 PPO 更新，不能把最终权重冒充原 step-75 权重**；评估结果只归于实际发布的 step-76 模型。

step-76 模型已登记为 `mv_e71131b812cf4005919e`；独立 Isaac 报告 `ev_0161c06c07a4481cb091` 为 1/3：站立种子 0 满 1000 步，步行种子 1 满 1000 步却只前进 0.027 m，站立种子 2 在 85 步跌倒。独立 MuJoCo 报告 `ev_8c3cce74293b4dcf9f47` 为 0/3。仅收紧速度容差仍不足以迫使 PPO 学走。新诊断训练 Profile `env_6b5ac6cea02a4caca093` 将 AMP 的 `velocity_reward_weight` 从默认 3 显式提高到 8，保留 `std=0.2`、A1 初态和低探索噪声；训练 `trn_e269ecdbed714fabbdc3`、Job `job_b45f9946f5e448148187`、Run `run_f99d2f217e6640de99f8`，512×24、actor LR `1e-4`，完成 150 轮。受管容器环境已只读核对收到 `EB_RPO_VELOCITY_REWARD_WEIGHT=8.0`。它的独立 Isaac/MuJoCo 分别为 1/3、0/3，步行病例在两仿真器均后退并跌倒；此实验仍使用 PD 回放失败的诊断动作先验，不能宣称 AMP 风格或自然步态有效。

首次 1000 轮训练 `trn_ef5716852f0b43fdaf54`、Run `run_6084fbada5d34a239075` 在 step 50 后通过平台取消：512 环境×24 步；第 50 轮内部验证平均存活约 52 步，AMP 风格奖励均值 `2.34e-4`，判别器损失约 396。74 个特征中有 8 个恒定，旧 `1e-4` 最小标准差把偏差放大数百倍；现已改为按物理单位设最小尺度。step 50 checkpoint 和诊断保留在原 Run。

复核官方 `gmr_to_lab.py`、`rpo.yaml` 与目标 `_amp_frame()` 后发现：官方 `rpo_lab/*.pkl` 的 `dof_pos` 依 Isaac Lab 交错顺序保存，旧转换器没有映射就声称是平台 `JOINT_ORDER`。已显式重排 q/dq，manifest 记录原始顺序和转换版本；loader 与训练准入拒绝旧 manifest。对应的物理尺度对照 `trn_9f9c3616d4494e2ebcd3` 已通过平台取消。

随后用受管审计 Job `job_e5b0333a08024305a883`、`job_26ab649490be46af9462`、`job_e5fb51ebf67f49b1a85f` 检查官方 30 个 PKL：原速库没有足量 0.3 m/s 连续接地直行动作；旧 `114_08` 低速片段双脚腾空率高。v2 短训 `trn_dd57cfe765df414c9894`、Job `job_1a73dbc0a2bf4f35ac4d` 已取消。v3 使用官方左右转步行、时间统一延长 3× 并同步重算全部速度，按至少一脚近地、低速、横移、偏航及片段累计转角筛选。它仍是诊断数据，不宣称自然慢速直行或动力学可执行。

v3 短训 `trn_922f51e65efb4c80bdd2` 已完成：512 环境×24 步×50 轮、seed 1234、PPO batch 4096/micro-batch 1024。Isaac 运动学探针通过，但判别器损失下降至 `0.0020`、风格奖励均值只有 `1.03e-4`；训练回合长度在前 50 轮约 55–75 步间波动，没有持续改善。实际 ONNX 的 Isaac 与 MuJoCo 闭环均 0/3，通过两个独立仿真确认尚未达到冻结的 1000 步站立和 0.3 m/s 直行条件。PD 跟踪诊断又显示所有时间缩放步行片段早停；不能把这些片段标为已通过动力学回放。

按设计 §8.2 运行同接口 PPO `w_style=0` 对照：`trn_b69cae484d004d5ea200`、Job `job_56d0d8f716b849028af9`、Run `run_b0fefe0077864469a340`，原预算 200 轮，其他环境/actor/seed/rollout 设置沿用 v3。`amp_discriminator.style_weight_max=0` 作为唯一新增 Hydra 数值覆盖写入预览、Run 与模型来源；仍由 EmbodiedBein 调度 RLinf。第 50 轮内部验证平均存活 `50.54` 步（AMP-on 为 `55.78` 步），训练存活未持续改善，完整 step-50 checkpoint 已保存；按设计的无改善早停规则通过平台取消后续轮次。之后从 step 50 受管恢复到 51 并登记模型，独立 Isaac/MuJoCo 均 0/3，不能据此声称 AMP 优于 PPO。

首个低噪声模型的 `trainer_state` checkpoint 已通过平台预览并恢复为 `trn_2737c8e836a84f47bd5d`、Job `job_d927ec2903d24b7ea0c8`、Run `run_bec420a9f4814776884a`，目标是**累计** 100 轮；沿用原训练镜像、数据、Profile 和超参数。已成功登记 step-100 模型 `mv_a309fe67335b44f5927e`；内部确定性评估 step 50/75/100 平均存活分别为 `634.17/365.02/219.59` 步，明显非单调。独立 Isaac/MuJoCo 结果分别为 1/3、0/3；Isaac 步行 case “站满全程却不走”暴露了奖励漏洞。另从同一 step-25 完整 checkpoint 恢复累计 50 轮候选 `trn_f5c844f1596d4a3fb157`、Job `job_f434a04ca11b4117b558`，内部平均存活虽升至 `742.92` 步，独立 Isaac 仍仅 1/3。MuJoCo 静态 PD 探针提示跨仿真接触/动力学差异，后续应先查清资产接触和控制协议，不能把调整 MuJoCo 判定门限当作修复。

只有诊断指标改善才按设计将**总训练预算**扩到 1000，继而 3000–5000 轮，并建立至少三 seed、AMP-on/off 对照、独立 held-out 动作参考及自然度/足滑/能耗的正式验收套件。v3 官方片段约 28.8 s（其中站立 3.82 s），足量性、慢速直行覆盖与动力学可行性仍是重大不确定性；需要在授权和来源可核验前提下扩充数据。诊断 EvalSuite 不提供部署资格，真机未触及。

本轮进一步反思后，**不扩大同一数据/奖励组合的训练预算**：现有官方慢速步行片段在目标 Isaac 资产上的 PD 动态回放全部提前跌倒，AMP 判别器易饱和且风格奖励极小；任务奖励从 `std=0.5` 改到 0.2 再把速度权重升到 8，分别出现原地不走、短暂前进后跌倒、反向跌倒。独立 MuJoCo 的源 A1 站姿恒定 PD 自身也不能维持 20 秒，跨仿真控制/资产差异尚未定位。下一次正式训练前，应先取得并通过目标资产动力学跟踪的授权低速直行动作，或单独把当前动作的跟踪失效归因并修复；同时把 MuJoCo 静态 PD/接触基线校准到可信状态。用户已说明没有新增原始动作，故当前只能保留诊断结果，不能伪造合格先验或降低冻结的评估标准。

## 来源、验证和边界

来源是官方 [RoboParty training workspace](https://github.com/Roboparty/roboparty_train) 指向的固定 [`robolab` 动作目录](https://github.com/Roboparty/robolab/tree/6b1c3d9988497c8961dcba77892de32edc1770e1/data/motions/rpo_lab)。仓库含 [BSD-3-Clause LICENSE](https://github.com/Roboparty/robolab/blob/6b1c3d9988497c8961dcba77892de32edc1770e1/LICENSE)；逐片段原始权利链尚未单独证实，Dataset 元数据明确记录该限制。v3 固定源为 `A1-_Stand_stageii.pkl`、`B10_-__Walk_turn_left_45_stageii.pkl`、`B14_-__Walk_turn_right_45_t2_stageii.pkl`，每个源的 SHA 和时间缩放在 manifest。宿主只做固定哈希和静态结构检查；PKL 正式加载仅在隔离、无网络的受管工具容器中进行。

`backend/.venv/bin/python -m pytest -q backend/tests/unit backend/tests/adapters backend/tests/agent`：**378 passed，1 skipped，1 warning**。这包括 RLinf 补丁、AMP checkpoint 契约与平台登记测试；实际 GPU/仿真证据以上述 Job/Run 和产物为准。工作区其他改动均保留，未暂存或提交 Git。
