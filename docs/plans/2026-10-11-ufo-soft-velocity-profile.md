# UFO/RPO 原生模型速度软惩罚候选 E

用户已授权按合理建议优化并对齐原生模型，不提交代码。此候选接续 [原实施计划](/data/kk/EmbodiedBein/docs/plans/2026-10-10-ufo-rlinf-bfm-integration.md) 的不可变配方、资格与分段闭环。当前 D 的 immutable 镜像与活动训练保持原样；先完成 E 的容器行为验证，D1024 独立快评和实际资源预算核验后才决定 E GPU 资格。

## 依据与固定语义

D 前32个新采样 round、24,576条真实转移中，23.10% 存在关节速度超过0.8额定上限，记录速度硬限位325次；训练请求/执行饱和分别45.28%/27.14%。限速器控制 PD target 增量，不能保证实际 dq。原生八项 aux 没有速度软惩罚，torque 两项权重均为0。证据见 [完整优化记录](/data/kk/EmbodiedBein/docs/progress/2026-10-11-ufo-native-alignment.md)，不是 E 效果证明。

E 保持 D 的全部原生网络、六 Adam、FB/CPR/aux 更新方法、八项原奖励与权重、batch1024、CPU replay100万、trackingFalse、192环境、0.5×Vmax target slew、RSI/hard terminal。通过原生 FBcprAuxAgent 已有 aux_rewards / aux_rewards_scaling 接口增加第九项 limits_dof_vel：

`raw = sum_j relu(abs(dq_j)/Vmax_j - 0.8)^2`，固定权重 `-5.0`，不乘dt，不在环境预加权，原生 agent 加权一次。阈值与23个Vmax沿用冻结物理上限；硬终止与验收判据不改变。惩罚在80%上限处连续且一阶导数连续，作用于实际物理 dq；它不保证 requested saturation 降到15%，也不代替正常行走验收。

aux critic 仍是原生标量，不增加网络或优化器。新的固定 auxiliary_profile 为 soft_velocity_v5，Gym/Profile/recipe/source/image/fingerprint 单独派生，禁止跨 D 完整恢复或回填第九项到 D replay。原八项路径默认行为保持；新候选不宣称完整原生默认训练复现。

实现复核选择：E保留D的 `actor.model.learning_profile=physical_limits_stable_slew_v4` 与 `EB_UFO_LEARNING_PROFILE=physical_limits_slew_v4`，另冻结 `actor.model.auxiliary_profile=soft_velocity_v5` / `EB_UFO_AUXILIARY_PROFILE=soft_velocity_v5`。这样复用已验证的reset诊断、控制器与checkpoint同model bind位置；新recipe、Gym、EnvironmentProfile、9项native cfg、replay-v2与source hash仍明确区分目标身份。生产与eval父镜像实际回执不同：eval未应用两个训练专用alias patch，按真实回执单独记录，不补造来源字段。

## 必须传播的范围

- 原生边界 [bfm_ufo_agent.py](/data/kk/EmbodiedBein/backend/eb/runners/bfm_ufo_agent.py)：固定第九项列表/权重、配置白名单与 native 配置回读；保持原生类/网络/损失。
- 环境 [rpo_ufo_isaac.py](/data/kk/EmbodiedBein/backend/eb/runners/rpo_ufo_isaac.py)、[rpo_bfm_extension.py](/data/kk/EmbodiedBein/backend/eb/runners/rpo_bfm_extension.py)：实际dq与硬Vmax、9列raw aux、cold/partial RSI reset、共享controller、真实terminal trace。
- 转移和恢复 [bfm_ufo_worker.py](/data/kk/EmbodiedBein/backend/eb/runners/bfm_ufo_worker.py)、[bfm_ufo_replay.py](/data/kk/EmbodiedBein/backend/eb/runners/bfm_ufo_replay.py)、[bfm_ufo_checkpoint.py](/data/kk/EmbodiedBein/backend/eb/runners/bfm_ufo_checkpoint.py)：8/9明确布局、严格shape/name、native配置与新fingerprint、同model bind发表完整state。
- 平台 [training_recipe.py](/data/kk/EmbodiedBein/backend/eb/domain/training_recipe.py)、[rlinf_form.py](/data/kk/EmbodiedBein/backend/eb/domain/rlinf_form.py)、[environment_catalog.py](/data/kk/EmbodiedBein/backend/eb/environment_catalog.py)、[estimator.py](/data/kk/EmbodiedBein/backend/eb/capacity/estimator.py)、[rlinf.py](/data/kk/EmbodiedBein/backend/eb/adapters/rlinf.py)：新固定模板/环境、9项replay内存、正确环境flag、只有已核验source/image可资格，资格未闭合不能长训。
- 推理/评估 [rpo_bfm.py](/data/kk/EmbodiedBein/backend/eb/rpo_bfm.py)、[rpo_bfm_eval.py](/data/kk/EmbodiedBein/backend/eb/runners/rpo_bfm_eval.py) 及已有原生export/collector/held-out选择：E与D共用v1动作codec及同一个限速器，原生cfg保留9项学习语义，推理导出actor仍raw。不得修改已发表模型。
- 受管 image-local Gym alias / source receipt、producer与eval镜像、Make入口；父镜像固定为 D88aee 与 D9129。外部 RLinf 仓库只读。

## 最小验证与阶段出口

1. 先留失败行为检查：0.8以下惩罚0、上界处已出现平滑惩罚、正负速度对称、dq/Vmax非法拒绝；原8列逐值不变。确认原生九项配置只增名字/权重，模型结构仍与完整 preset 对齐。
2. 9项 transition/replay round-trip、缺第九项与未知项拒绝；8项旧状态仍通过。tiny CPU 原生更新真正消费第九项、指标有 aux_rew/limits_dof_vel、加权一次、六 Adam 计数一致。原生配置改变在加载任何旧权重/Adam/replay前拒绝跨 D 恢复。
3. 平台固定source/Profile/Hydra/flag/资源与资格预算；独立导出/velocity/held-out兼容选择和共享controller。现有 C/D 相关回归按改动范围完成。
4. 第三轮只读审查后保留全部CPU证据。新candidate保持未资格状态。GPU仍只走 API→Job/Run→ebd；预算足够且 D 窗口决策支持调整后，才 E fresh16→24、完整回读、独立速度/held-out/媒体/取消，然后每1000更新决策。不能以loss下降或实现存在宣布80/100通过。

## 本段执行结果

D已完成1024更新及独立速度/held-out闭环；E镜像、固定配置、9列真实terminal传递、下一更新完整恢复、ONNX与发表、平台容器验证已完成，部署状态dependency_ready。完整后端524 passed/17 skipped，最后原生核心与导出14 passed。E没有GPU资格证据；实际存储快照65.612GiB加下一checkpoint与既定预留超过80GiB，本段决定budget_end。配置、指标、逐例结果、媒体与剩余问题见[自包含交接](/data/kk/EmbodiedBein/docs/progress/2026-10-11-ufo-optimization-summary.md)。
