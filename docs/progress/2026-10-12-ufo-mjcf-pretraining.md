# RPO 23DoF 原生 MJCF / MJLab 预训练实验

日期：2026-10-12。状态：原生MJCF/MJLab接入、16→24完整恢复与24→524新增500更新完成；训练门槛未达标；停止本段，不独立评估、不自动长训。

## 用户目标与执行边界

将现有 UFO + RLinf + IsaacLab/PhysX 链路接入原生 MJCF + MJLab/MuJoCo-Warp，并做一轮有界预训练。GPU0 为 learner；GPU1、2、3 各2048环境，总6144；Batch2048、Replay4M，完整原生网络。通过新后端16→24完整恢复后新增500次联合更新，24→524，最多2小时，无自动重试。524后不开放本轮追加训练。训练未达标不启动独立评估。不暂存、提交或推送代码。

## 已完成的接入

使用受管 RPO 原有 MJCF 原生运行，不将 MJCF 再导入 PhysX。RLinf 继续负责 actor/env/rollout 生命周期；UFO agent 独占六个 Adam 优化器、四组 target 网络、运行归一化与 Forward–Backward/CPR/aux 损失。

新训练身份：`rpo_bfm_ufo_agent_rlinf_mjlab_4gpu_v12`；环境 Profile：`rpo_bfm_ufo_mjlab_4gpu_rsi_v12`；运行环境：`RPO-UFO-MJLab-RSI-v12`；execution.simulator_id=`mjlab`，topology_id=`one_learner_three_samplers`，checkpoint_format=`ufo_agent_state_v2`。旧 Isaac 配方、原始 MJCF/URDF 和已发布接口保持不可变。新实验从头初始化，不把 PhysX trainer state 恢复到新后端。

冻结训练镜像：`eb/rlinf:rpo-bfm-ufo-mjlab-v12`，摘要 `sha256:071d5aacb6660f0bb48144fa2c6c0af34a8cfff9ef90819909dd4d2b4b63c986`。固定依赖：Python3.11.16、Torch2.7.0+cu128、MJLab1.4.0、MuJoCo3.8.0、MuJoCo-Warp3.8.0.3、Warp1.12.0。UFO固定提交 `0b89378de02f77712072d983bc43ec2a46455dc0`，RLinf固定提交 `018e6a9e0da60f96eeda8b11008543911027d414`；外部仓只读，分派补丁仅作用于 owned 镜像。UFO Python3.10包元数据未作为兼容证明，实际采用经过验证的 Python3.11 源码导入。

## 完整物理、动作与数据契约

| 参数 | 冻结设置 |
|---|---|
| 本体 | RPO V2，23受控关节，24个codec body |
| 宿主受管资产根 | `/data/kk/assets/roboparty/rpo-v2` |
| 物理资产根（运行容器） | `/workspace/rpo-assets` |
| 资产 manifest SHA256 | `b58868a782d8863ac38059eb4eab89d27ee1d70565bf58f5a055f44510c8e842` |
| 模型来源 | 原 RPO MJCF 质量、惯量、STL碰撞网格；移除旧XML电机/传感器和旧floor，由MJLab创建执行器/传感器/无限平面 |
| 物理/控制 | dt=0.0025s、decimation8、400Hz物理、50Hz控制 |
| 求解器 | Newton、100iterations、implicitfast；原XML的PGS不用于GPU运行 |
| 接触材料 | 原MJCF solref=`0.001 2`、原solimp、friction=`0.9 0.2 0.2`；实际编译参数保存到 startup receipt |
| 自碰撞 | 保留原MJCF contype1/conaffinity0；自碰撞关闭，与旧PhysX开启自碰撞不同 |
| 驱动 | MJLab显式DC motor PD，零延迟，armature=0.01kg·m²，frictionloss=0.01 |
| thigh PD | Kp100、Kd3.3、120Nm、25rad/s |
| knee/torso PD | Kp150、Kd5、120Nm、25rad/s |
| ankle PD | Kp40、Kd2.5、27Nm、8rad/s |
| arm PD | Kp40、Kd2、27Nm、8rad/s |
| elbow_pitch PD | Kp30、Kd1.5、27Nm、8rad/s |
| elbow_yaw PD | Kp20、Kd1、27Nm、8rad/s |
| 速度边界 | DC motor torque-speed曲线；不直接夹持qvel。原额定速度仍作为严格影子标准 |
| 动作 | URDF有限位置范围的midpoint/halfwidth映射，target slew每控制步≤0.5×额定Vmax×0.02s；reset anchor由实际q推导 |
| 观测 | actor375=states52+last_action23+history300；FB/critic733再加privileged358；wxyz、world root link原点速度，4帧oldest-first/zero-pad |
| codec SHA256 | `dce25d554ceb7fa69287515946d2f5cd9d41d14f1b1ef38715b819d7e962b6ef`，历史physics_hz=200字段未改；实际400Hz由本Profile明确 |
| 数据集 | `ds_fb0ecda9e70644f2afdd`，RobotState v2，仅train初态与expert窗口 |
| 数据 manifest SHA256 | `dc0c6585887507df59f675963b2733d25c9836d805e9bb89f2402398f07bd288` |
| RSI | 与运行时同一prepare_spec碰撞模型；足底间隙2mm、初始最低足点≤20mm、root lift≤30mm、近地足点\|v_normal\|≤0.15m/s、dq≤0.70额定；不投影改写原q/dq |
| Tracking | 50%环境独立绑定250步train专家窗口，RSI起点与轨迹相位绑定；其余mixed上下文；不同rank种子42/43/44 |
| 终止 | 每物理子步位置/力矩限位、1.3×额定速度和protected接触锁存；root height<0.53m或tilt>0.70rad；1000控制步timeout保留truncated |
| 严格影子 | 所有物理子步按原1.0×位置/速度/力矩与宏观跌倒标准记录第一次违规存活步数 |
| 物理容量 | 128 contacts/world共享总池；1024 constraints/world；达到容量直接失败，不静默丢弃碰撞 |

## 优化与实验配置

完整435,116,484个可训练参数；F/Actor/Critic/Aux Critic宽2048、深6，B输出潜空间256，潜向量范数16。Batch2048，CPU streaming Replay4,000,000条。每24,576新增在线转移触发16次learner联合更新；warmup10,240条。网络与optimizer state由GPU0持有，sampler仅持有推理所需权重；这不是DDP。

沿用stabilized_v7：Actor梯度裁剪1，其他五优化器5；自适应正则尺度w≤50，判别器lr=3e-6、reward截[-2,2]，gamma0.98。九项raw aux仅在UFO侧加权一次；速度项为8个已完成物理子步的sum_j relu(|dq|/Vmax−0.70)²均值，权重−25；没有将v11附加高阶速度罚或屈膝RSI过滤带入本实验。完整原生Agent配置已在本文嵌入；真实执行的resolved config与最终checkpoint已在本文追加核对。

资源请求：4张GPU独占、24CPU、96GB RAM、32GB shm。真实任务通过API→Job→Run→ebd启动；诊断容器也由平台管理。CPU验证和测试在隔离容器内执行。

## 三轮复核与已修正问题

1. 本体/数据：质量、惯量、网格与codec哈希检查通过；发现旧RSI扫描开启MJCF自碰撞，原XML实际关闭，新路径统一口径；root四元数、link原点速度、joint/body/sensor顺序显式映射。
2. 时序/控制：真实CPU足地接触证明原生sensor force Z为负，现已转为施加在robot上的支撑+Z；读取qfrc_actuator而非未实现的joint_torques接口。独立复核发现step后forward覆盖实际积分接触力：现先复制实际力/力矩，再刷新post-integration FK；真实MuJoCo碰撞测试RED→GREEN。原始probe的force峰值不能充当该修正的最终冲量证据。
3. 生命周期/预算：独立复核发现可以重复追加500更新，现正式训练锁定24→524且必须trainer_state恢复，524后关闭候选。接触/约束容量耗尽检查RED→GREEN。控制面完整测试555通过、30跳过；原生同步检查覆盖新后端及复用tracking、checkpoint、真实更新计数；API异步检查由控制面带pytest-asyncio镜像执行。

部分reset的全局forward可能重算未reset环境的力矩诊断；当前RLinf在任意done时使用整批reset前final_obs采集transition，未证实replay污染。保留现有采集协议；solver warmstart与其他调用者的诊断影响未额外验证。

## 已执行物理实验

首次诊断 Job=`job_64a0f77b123d4439a052`，Run=`run_69bac540803a4a13bee1`，正常完成。其镜像摘要为`sha256:07543dfa1b0b54bcd2532eda1388b21592068d3e37d719ae37ed7858587bb6de`，早于force符号与实际积分力时序修正。每卡2048环境、32控制步，固定train RSI和anchor target；未加载训练policy，不属于独立评估。

| sampler/seed | 首步终止/2048 | 32步仍存活的初始回合 | 总终止 | 原诊断最大额定速度比 | 原诊断最大足力N |
|---|---:|---:|---:|---:|---:|
| 0/42 | 23 | 1022 | 1545 | 2.81918 | 2292.63 |
| 1/43 | 48 | 984 | 1663 | 2.81925 | 2676.74 |
| 2/44 | 28 | 1043 | 1537 | 2.78676 | 2676.50 |

首步终止合计99/6144=1.6113%，尚未达到≤0.1%。Torch peak allocated仅80,162,304bytes/card、reserved111,149,056bytes/card，**不包含Warp自身分配**，不能当作整卡VRAM峰值。

修正后物理复验 Job=`job_b141c3a526f440bb87ed`，使用阶段诊断镜像`sha256:4897bb2f2aed1bdeac39dad458e186b8c9649255b82d16de7f3840654fd68399`，Run=`run_4a0d372b41d7438c8e64`，成功完成。三个sampler均2048环境、32步；实际积分力时序正确，接触/约束容量未耗尽。

| sampler/seed | 首步终止 | 初始回合32步存活 | 总终止 | 最大额定速度比 | 最大实际足力N |
|---|---:|---:|---:|---:|---:|
|0/42|23|1022|1533|2.8191998|2362.646|
|1/43|48|983|1663|2.8191690|2685.480|
|2/44|28|1042|1522|2.7865829|2685.281|

首步99/6144=1.6113%；Torch峰值allocated79,302,144bytes/card、reserved111,149,056bytes/card，不包含Warp。此probe没有训练policy，不等于预训练收益或独立行走验收。

## 首次资格失败及边界修复

Train=`trn_b707f0de1d3b43a4b55b`、Job=`job_1a69b52ff91b40c697b5`、Run=`run_7029a4b00f5b4149b33c`；退出255，未产生实际learner联合更新或完整checkpoint。RLinf交互层传入numpy.ndarray，环境step直接torch.isfinite导致TypeError。入口统一torch.as_tensor(float32,device)后，新行为检查RED→GREEN；6个原生契约检查通过。资格前镜像由4897…派生为d7bc…，保留失败Job原始摘要/来源，不覆盖旧Run或伪造成功。

## 冻结的完整原生Agent配置

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

## 资格失败记录与当前进度

|Train|Run|结果与修正|
|---|---|---|
|trn_b707f0de1d3b43a4b55b|run_7029a4b00f5b4149b33c|学习更新前NumPy动作边界TypeError；已修正float32/device转换|
|trn_dfa94ade1a6d49f49d56|run_277e4895c92e4fd9971d|固定镜像摘要未同步新来源凭证；冻结校验在初始化前拒绝，已同步|
|trn_bf7b2207901c46ad8972|run_c6da581f8ec043ec80fd|首个采样轮的131列trace被旧分支当128列校验；学习更新0、无完整checkpoint，已修正|

第三次任务的取消请求被自动审批拒绝（可能丢失未保存进度）；未绕过限制，任务随后因真实trace校验自然失败，四卡释放。还修正了诊断trace把放宽1.3×标志当作严格1.0×标志的问题；两个新增行为检查均RED→GREEN。8项新原生契约与复用worker/tracking共44项通过。

修正后的16更新资格Train=`trn_8d5de58d8a334c3e82f2`预览已断言固定摘要与完整来源逐字段一致；后因位置范围复核而取消，记录到14更新，无完整模型。所有失败均保存真实Job/Run，不自动重试、不覆盖成功状态。


## 位置范围与软惩罚对齐

原始MJCF质量、惯量、STL和碰撞掩码保留，但其15个关节位置范围与冻结RPO接口不同。派生的运行MjSpec将23个范围统一为已发布接口范围；RSI扫描使用同一prepare_spec。原始资产文件、manifest和codec哈希不修改。位置软惩罚使用MJLab的90% soft_joint_pos_limits，物理约束与硬终止使用完整范围。两个一致性检查RED→GREEN，相关44项和最终镜像8项契约检查通过，平台40项通过。

|关节|原始MJCF范围rad|运行模型/冻结接口范围rad|
|---|---|---|
|torso_joint|[-2.62, 2.62]|[-3.14, 3.14]|
|left_thigh_roll_joint|[-0.2, 2.0]|[-0.2, 1.0]|
|right_thigh_roll_joint|[-2.0, 0.2]|[-1.0, 0.2]|
|left_arm_pitch_joint|[-3.14, 2.57]|[-3.14, 1.57]|
|right_arm_pitch_joint|[-3.14, 2.57]|[-3.14, 1.57]|
|left_thigh_pitch_joint|[-2.5, 0.8]|[-2.094, 0.7854]|
|right_thigh_pitch_joint|[-2.5, 0.8]|[-2.094, 0.7854]|
|left_knee_joint|[0.0, 2.5]|[-0.2, 2.5]|
|right_knee_joint|[0.0, 2.5]|[-0.2, 2.5]|
|left_arm_yaw_joint|[-2.6, 2.6]|[-1.57, 1.57]|
|right_arm_yaw_joint|[-2.6, 2.6]|[-1.57, 1.57]|
|left_ankle_pitch_joint|[-0.7, 0.7]|[-0.6, 0.6]|
|right_ankle_pitch_joint|[-0.7, 0.7]|[-0.6, 0.6]|
|left_elbow_pitch_joint|[-1.0, 3.57]|[-0.6, 1.57]|
|right_elbow_pitch_joint|[-1.0, 3.57]|[-0.6, 1.57]|

范围对齐前的Train trn_8d5de58d8a334c3e82f2 / Run run_59f0856f82c54fd7b5e0经平台取消，实际原生指标记录到14次更新/49152累计转移，无完整checkpoint、无发布模型；不作为后续初值。取消后四卡释放。取消前warmup episode均值3.1246、严格3.0309；首步94/6640=1.4157%；额定速度比2.3590；覆盖漏记0。旧anchor probe保存的768个首步样本中19个硬位置违规、速度/力矩/宏观违规0；其首步最大速度6.7266rad/s、足部竖直净力458.33N。这只覆盖每rank前256环境，不能用19/768替代全6144计数；首步失效不能全部归因于接触超速。

最终16资格Train trn_7e1e42b773cf49d5934a / Run run_c041f9646fe94376b812使用071d…镜像，已核对完整source receipt并成功完成。新后端从头训练。

## 更新数与Tracking预算的解释

16次联合更新对应每采样卡4个控制步、24576条新transition。新增500更新需要32个采样轮，128个控制步/环境=2.56s；完整恢复保留算法状态，但物理环境重建，不能把16→24→524三Run的短轨迹拼接为一个连续episode。因此本段单个新episode的最大可观测跨度128步，小于250步Tracking窗口，完成数0本身不能证明Tracking实现或政策失败。存活统计区分已经结束的episode与结束时仍存活的右删失轨迹，后者另报数量、已观察长度，不混入已完成均值。

## 新镜像完整来源凭证

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
  "scope": "native UFO agent + platform RPO MJCF/MJLab bridge; tested Python3.11 import",
  "topology_patch_hash": "sha256:3078506628924464cb42d0b2f108b3fab502a43fd75e091f358d504565294d23",
  "learning_profile_patch_hash": "sha256:d47623e48b4cc757d5e63336a1fa7d49234c3833a2f1c6bbdbee4fe7ffd9a272",
  "rollout_profile_patch_hash": "sha256:777456ce171734326cf8c55eaa434fc5a69a27d0b2d0b725608dd54c190f4a77",
  "action_control_source_sha256": "sha256:0cad3bac9e3efb3f1f37d6cac873f789548143b9cd597ed90a93a3045339cd24",
  "action_control_profile_patch_hash": "sha256:548260ca5e3eedd53ae739c00672c7e273e7835faed734778039cf5977b56e0d",
  "auxiliary_profile": "soft_velocity_v7",
  "auxiliary_source_sha256": "sha256:0bdd6b97862fe21dea301b957f55ad0c25b62e07a489675efb7eb4705b7e05f2",
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
  "physics_profile": "native_mjcf_v12",
  "contact_profile": "native_mjcf_v12",
  "mjlab_patch_hash": "sha256:3b06c0e0883c415c910fee686422f7c25827b047ec37528083ddf87b0a6a8eb4"
}
```


## 预训练结果与决策

完整500新增更新25–524连续存在，全部149项标量有限，32个采样轮产出786432条新transition，累计860160条。训练episode五窗均值：3.2383 → 4.13525 → 5.21094 → 6.53911 → 8.49378；严格影子：2.98339 → 3.83911 → 4.79267 → 5.83511 → 7.46924。全段物理峰值35.7757rad/s，最高额定比4.4720；首步终止5478/155862=3.5146%。

|冻结条件|实际|结果|
|---|---|---|
|严格影子末两个窗口≥35|5.83511、7.46924|未通过|
|全段物理额定速度比≤2.5|4.47197|未通过|
|首步终止≤0.1%|3.5146%|未通过|
|独立行走评估前训练末两窗≥100|6.53911、8.49378|未通过|
|实际更新、完整checkpoint、有限标量|500新增，六Adam524，SHA通过|通过|

本轮完成原生后端预训练测试，运行成功与正常行走合格分别判定。未启动独立validation/acceptance或真机测试，不提供成功率/录像，不把训练episode均值当成行走成功率。524预算结束，API恢复候选关闭；后续实验需新的明确计划。

## 未达标原因与下一步建议

位置硬限位涉及138343条转移；完成episode共150339个，数量比为92.02%，终止原因可能重叠。全部首步终止5478次均涉及位置违规，首步速度违规仅28次。本轮主要探索阻塞已转为位置边界，不应继续把每次早停都归因于足地角速度飞车。速度峰值和足地冲击仍未达到严格标准。

重置姿态-速度诊断：

```json
{
  "resets": 155862,
  "first_terms": 5478,
  "first_term_fraction": 0.03514647572852908,
  "any_outward_near_limit_fraction": 0.12378899282698798,
  "term_fraction_given_any_outward_near_limit": 0.1879340727687364,
  "term_fraction_without_outward_near_limit": 0.01356100989983012,
  "linear_20ms_prediction_violation_fraction": 0.033927448640464,
  "first_terms_with_boundary_position_violation": 5159,
  "first_terms_without_boundary_position_violation": 319,
  "per_joint": [
    {
      "joint": "left_thigh_yaw_joint",
      "resets": 155862,
      "start_within_0_02rad_fraction": 0.010457969229189925,
      "outward_velocity_near_limit_fraction": 0.007153764227329304,
      "first_term_given_outward_near_limit": 0.46367713004484307,
      "first_boundary_outside_count": 812,
      "first_boundary_max_violation_rad": 0.03260866105556487
    },
    {
      "joint": "right_thigh_yaw_joint",
      "resets": 155862,
      "start_within_0_02rad_fraction": 0.0031951341571390076,
      "outward_velocity_near_limit_fraction": 0.0023610629916207926,
      "first_term_given_outward_near_limit": 0.4429347826086957,
      "first_boundary_outside_count": 178,
      "first_boundary_max_violation_rad": 0.027434992790222157
    },
    {
      "joint": "torso_joint",
      "resets": 155862,
      "start_within_0_02rad_fraction": 0.0,
      "outward_velocity_near_limit_fraction": 0.0,
      "first_term_given_outward_near_limit": null,
      "first_boundary_outside_count": 0,
      "first_boundary_max_violation_rad": 0.0
    },
    {
      "joint": "left_thigh_roll_joint",
      "resets": 155862,
      "start_within_0_02rad_fraction": 0.06286330215190361,
      "outward_velocity_near_limit_fraction": 0.04158807149914668,
      "first_term_given_outward_near_limit": 0.25300833076211043,
      "first_boundary_outside_count": 2390,
      "first_boundary_max_violation_rad": 0.019980031251907338
    },
    {
      "joint": "right_thigh_roll_joint",
      "resets": 155862,
      "start_within_0_02rad_fraction": 0.030527004657966664,
      "outward_velocity_near_limit_fraction": 0.026286073577908662,
      "first_term_given_outward_near_limit": 0.29192091774469126,
      "first_boundary_outside_count": 1461,
      "first_boundary_max_violation_rad": 0.020624908804893483
    },
    {
      "joint": "left_arm_pitch_joint",
      "resets": 155862,
      "start_within_0_02rad_fraction": 0.0,
      "outward_velocity_near_limit_fraction": 0.0,
      "first_term_given_outward_near_limit": null,
      "first_boundary_outside_count": 0,
      "first_boundary_max_violation_rad": 0.0
    },
    {
      "joint": "right_arm_pitch_joint",
      "resets": 155862,
      "start_within_0_02rad_fraction": 0.0,
      "outward_velocity_near_limit_fraction": 0.0,
      "first_term_given_outward_near_limit": null,
      "first_boundary_outside_count": 0,
      "first_boundary_max_violation_rad": 0.0
    },
    {
      "joint": "left_thigh_pitch_joint",
      "resets": 155862,
      "start_within_0_02rad_fraction": 0.0,
      "outward_velocity_near_limit_fraction": 0.0,
      "first_term_given_outward_near_limit": null,
      "first_boundary_outside_count": 0,
      "first_boundary_max_violation_rad": 0.0
    },
    {
      "joint": "right_thigh_pitch_joint",
      "resets": 155862,
      "start_within_0_02rad_fraction": 0.0,
      "outward_velocity_near_limit_fraction": 0.0,
      "first_term_given_outward_near_limit": null,
      "first_boundary_outside_count": 0,
      "first_boundary_max_violation_rad": 0.0
    },
    {
      "joint": "left_arm_roll_joint",
      "resets": 155862,
      "start_within_0_02rad_fraction": 0.0011741155637679487,
      "outward_velocity_near_limit_fraction": 0.0005389382915656157,
      "first_term_given_outward_near_limit": 0.19047619047619047,
      "first_boundary_outside_count": 16,
      "first_boundary_max_violation_rad": 0.0028509223461151145
    },
    {
      "joint": "right_arm_roll_joint",
      "resets": 155862,
      "start_within_0_02rad_fraction": 0.0017451335155458034,
      "outward_velocity_near_limit_fraction": 0.0007763277771361846,
      "first_term_given_outward_near_limit": 0.049586776859504134,
      "first_boundary_outside_count": 7,
      "first_boundary_max_violation_rad": 0.005581113100051882
    },
    {
      "joint": "left_knee_joint",
      "resets": 155862,
      "start_within_0_02rad_fraction": 0.0,
      "outward_velocity_near_limit_fraction": 0.0,
      "first_term_given_outward_near_limit": null,
      "first_boundary_outside_count": 0,
      "first_boundary_max_violation_rad": 0.0
    },
    {
      "joint": "right_knee_joint",
      "resets": 155862,
      "start_within_0_02rad_fraction": 0.0,
      "outward_velocity_near_limit_fraction": 0.0,
      "first_term_given_outward_near_limit": null,
      "first_boundary_outside_count": 0,
      "first_boundary_max_violation_rad": 0.0
    },
    {
      "joint": "left_arm_yaw_joint",
      "resets": 155862,
      "start_within_0_02rad_fraction": 0.0,
      "outward_velocity_near_limit_fraction": 0.0,
      "first_term_given_outward_near_limit": null,
      "first_boundary_outside_count": 0,
      "first_boundary_max_violation_rad": 0.0
    },
    {
      "joint": "right_arm_yaw_joint",
      "resets": 155862,
      "start_within_0_02rad_fraction": 0.0,
      "outward_velocity_near_limit_fraction": 0.0,
      "first_term_given_outward_near_limit": null,
      "first_boundary_outside_count": 0,
      "first_boundary_max_violation_rad": 0.0
    },
    {
      "joint": "left_ankle_pitch_joint",
      "resets": 155862,
      "start_within_0_02rad_fraction": 0.03366439542672364,
      "outward_velocity_near_limit_fraction": 0.02213496554644493,
      "first_term_given_outward_near_limit": 0.1063768115942029,
      "first_boundary_outside_count": 102,
      "first_boundary_max_violation_rad": 0.0021843671798705833
    },
    {
      "joint": "right_ankle_pitch_joint",
      "resets": 155862,
      "start_within_0_02rad_fraction": 0.029141163336797938,
      "outward_velocity_near_limit_fraction": 0.023860851265863393,
      "first_term_given_outward_near_limit": 0.13767141704759345,
      "first_boundary_outside_count": 113,
      "first_boundary_max_violation_rad": 0.002043342590332009
    },
    {
      "joint": "left_elbow_pitch_joint",
      "resets": 155862,
      "start_within_0_02rad_fraction": 0.02921815452130731,
      "outward_velocity_near_limit_fraction": 0.010573456005953986,
      "first_term_given_outward_near_limit": 0.11589805825242719,
      "first_boundary_outside_count": 93,
      "first_boundary_max_violation_rad": 0.008406515121459956
    },
    {
      "joint": "right_elbow_pitch_joint",
      "resets": 155862,
      "start_within_0_02rad_fraction": 0.00807124250939934,
      "outward_velocity_near_limit_fraction": 0.005402214779741053,
      "first_term_given_outward_near_limit": 0.1828978622327791,
      "first_boundary_outside_count": 61,
      "first_boundary_max_violation_rad": 0.006870384216308589
    },
    {
      "joint": "left_ankle_roll_joint",
      "resets": 155862,
      "start_within_0_02rad_fraction": 0.0,
      "outward_velocity_near_limit_fraction": 0.0,
      "first_term_given_outward_near_limit": null,
      "first_boundary_outside_count": 0,
      "first_boundary_max_violation_rad": 0.0
    },
    {
      "joint": "right_ankle_roll_joint",
      "resets": 155862,
      "start_within_0_02rad_fraction": 0.0,
      "outward_velocity_near_limit_fraction": 0.0,
      "first_term_given_outward_near_limit": null,
      "first_boundary_outside_count": 0,
      "first_boundary_max_violation_rad": 0.0
    },
    {
      "joint": "left_elbow_yaw_joint",
      "resets": 155862,
      "start_within_0_02rad_fraction": 0.0,
      "outward_velocity_near_limit_fraction": 0.0,
      "first_term_given_outward_near_limit": null,
      "first_boundary_outside_count": 0,
      "first_boundary_max_violation_rad": 0.0
    },
    {
      "joint": "right_elbow_yaw_joint",
      "resets": 155862,
      "start_within_0_02rad_fraction": 0.0,
      "outward_velocity_near_limit_fraction": 0.0,
      "first_term_given_outward_near_limit": null,
      "first_boundary_outside_count": 0,
      "first_boundary_max_violation_rad": 0.0
    }
  ],
  "interpretation": "Measured reset pose and velocity versus first post-control-step state. Linear prediction is a diagnostic, not a proof of causality; substep position may recover before boundary."
}
```

初始关节距离边界<0.02rad且速度朝外的组，首步终止率显著更高；这是观测关联，尚未以单因素介入验证因果。下一步优先检查RSI的q/dq刹停兼容性，过滤贴边且朝外运动的初态；同时核对MuJoCo柔性关节约束与1mrad子步硬终止之间的容差，探索终止和严格影子应分别定义。不要通过更改硬件验收限位来声称达标。

参考速度一致性复核：

```json
{
  "frames_checked": 153,
  "point_frame": "body link origin, world",
  "error_columns": [
    "linear_mps",
    "angular_radps"
  ],
  "root": {
    "median": [
      0.0,
      1.3857486047541556e-08
    ],
    "p95": [
      0.0,
      9.695186506952787e-08
    ],
    "max": [
      0.0,
      5.835687844069071e-07
    ]
  },
  "all_body_values": {
    "median": [
      0.0005479013254948151,
      0.0020526913078298126
    ],
    "p95": [
      0.047559993268282516,
      0.11675831937447986
    ],
    "max": [
      2.582384761271655,
      3.431806471335128
    ]
  },
  "sources": {
    "official_rpo": {
      "median": [
        0.0004231098865725702,
        0.0015010965370504878
      ],
      "p95": [
        0.04173856410715706,
        0.09305925772437756
      ],
      "max": [
        2.582384761271655,
        3.431806471335128
      ]
    },
    "g1": {
      "median": [
        0.0008071208198350138,
        0.003082515490622467
      ],
      "p95": [
        0.053293835436268464,
        0.1442381539282004
      ],
      "max": [
        1.267596685013478,
        1.0677002491754435
      ]
    }
  },
  "max_events": {
    "linear": {
      "max": 2.582384761271655,
      "clip": "rpo_C3_-_run_stageii",
      "source": "official_rpo",
      "frame": 1,
      "body": "right_ankle_pitch_link",
      "analytic": [
        -6.055637772537912,
        3.8388481474222456,
        5.677699502406298
      ],
      "reference": [
        -6.383109092712402,
        4.0726094245910645,
        3.0953147411346436
      ]
    },
    "angular": {
      "max": 3.431806471335128,
      "clip": "rpo_C17_-_run_change_direction_stageii",
      "source": "official_rpo",
      "frame": 1,
      "body": "right_knee_link",
      "analytic": [
        -6.646458045460758,
        3.225543525206283,
        -6.242397986349227
      ],
      "reference": [
        -3.6800646781921387,
        -0.20626294612884521,
        -7.728714942932129
      ]
    }
  },
  "interpretation": "Source body velocities use 50Hz finite differences; compared with analytic Jacobian times raw qvel. This is not proof of a runtime velocity-frame bug."
}
```

root线速度误差为0、角速度最大误差约5.84e-7rad/s，未发现root世界/局部坐标系错位。body速度中位差很小，但部分片段边缘帧有较大离群差异。源数据按50Hz位姿差分构造body速度，当前检查按解析Jacobian×原始qvel计算；二者差异不等同于运行时速度API错误。建议在新的不可变数据视图中核对边缘帧、转折点q/dq与body速度的一致性；本轮没有改写已发布数据或重启这500更新。

后续长程试验还需提供≥250个连续控制步；当前500更新只推进128步。应明确采样控制步与learner更新比例，保持每卡2048/Batch2048/Replay4M并先验证数据与边界调整，再决定是否延长。此处为后续建议，本轮未实施或追加训练。

预算关闭的真实API响应：

```json
{
  "available": false,
  "reason": "本次 MJLab 500 更新实验预算已完成；后续训练需新的执行计划"
}
```

## 任务身份、完整恢复与产物

|阶段|Train|Job|Run|目标/实际|ModelVersion|
|---|---|---|---|---|---|
|16|trn_7e1e42b773cf49d5934a|job_1e117c966f374fd6b35f|run_c041f9646fe94376b812|16/16|mv_cfc36a49d5dd496d841d|
|24|trn_262331436fad4334bcca|job_323deb5671d5464588a5|run_e16c3dde3996450bb7ed|24/24|mv_9ec1ccb51c0d441e8348|
|524|trn_68a5fc34919446c5adf4|job_1d8704651d3d4f86bfff|run_4627d50438b84a07bc24|524/524|mv_d6e45769375c410ba4d9|

trainer_state恢复包含模型、六Adam、四target、normalizer、replay、expert sampler、z pool、各rank RNG及rollout状态；物理环境标记not_available，由RSI重建，不能称物理轨迹逐位复现。

### 第16次完整checkpoint核验

```json
{
  "step": 16,
  "counters": {
    "learner_joint_updates": 16,
    "sampling_rounds": 2,
    "online_transitions": 49152,
    "next_trigger": 59392
  },
  "six_adam_steps": {
    "forward_optimizer": 16,
    "backward_optimizer": 16,
    "discriminator_optimizer": 16,
    "actor_optimizer": 16,
    "critic_optimizer": 16,
    "aux_critic_optimizer": 16
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
  "fingerprint": "sha256:cedee2cb6d888e4932dfe6068ccc2382e8bda5ad8a0e4d1248bf7848d091dcc4",
  "execution": {
    "schema_version": "eb/training-binding/v1",
    "recipe_id": "rpo_bfm_ufo_agent_rlinf_mjlab_4gpu_v12",
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

### 第24次完整checkpoint核验

```json
{
  "step": 24,
  "counters": {
    "learner_joint_updates": 24,
    "sampling_rounds": 3,
    "online_transitions": 73728,
    "next_trigger": 83968
  },
  "six_adam_steps": {
    "forward_optimizer": 24,
    "backward_optimizer": 24,
    "discriminator_optimizer": 24,
    "actor_optimizer": 24,
    "critic_optimizer": 24,
    "aux_critic_optimizer": 24
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
  "fingerprint": "sha256:cedee2cb6d888e4932dfe6068ccc2382e8bda5ad8a0e4d1248bf7848d091dcc4",
  "execution": {
    "schema_version": "eb/training-binding/v1",
    "recipe_id": "rpo_bfm_ufo_agent_rlinf_mjlab_4gpu_v12",
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

### 第524次完整checkpoint核验

```json
{
  "step": 524,
  "counters": {
    "learner_joint_updates": 524,
    "sampling_rounds": 35,
    "online_transitions": 860160,
    "next_trigger": 870400
  },
  "six_adam_steps": {
    "forward_optimizer": 524,
    "backward_optimizer": 524,
    "discriminator_optimizer": 524,
    "actor_optimizer": 524,
    "critic_optimizer": 524,
    "aux_critic_optimizer": 524
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
  "fingerprint": "sha256:cedee2cb6d888e4932dfe6068ccc2382e8bda5ad8a0e4d1248bf7848d091dcc4",
  "execution": {
    "schema_version": "eb/training-binding/v1",
    "recipe_id": "rpo_bfm_ufo_agent_rlinf_mjlab_4gpu_v12",
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

## 完整执行配置与输入

平台表单kind=ppo、配置名称中的isaac字样为复用字段；实际training_backend=ufo_agent、loss_type=ufo_fb_cpr_aux、simulator_id=mjlab。最终模型根为`/data/kk/models/runs/run_4627d50438b84a07bc24`。

```json
{
  "training_request": {
    "name": "RPO UFO native MJCF MJLab step524 3x2048",
    "model_name": "RPO-UFO-MJLab-v12-step524",
    "checkpoint_step": 24,
    "target_learner_updates": 524
  },
  "execution": {
    "recipe_id": "rpo_bfm_ufo_agent_rlinf_mjlab_4gpu_v12",
    "runtime_id": "rlinf",
    "provider_id": "ufo",
    "topology_id": "one_learner_three_samplers",
    "algorithm_id": "bfm_fb_cpr_aux",
    "simulator_id": "mjlab",
    "progress_unit": "learner_joint_update",
    "schema_version": "eb/training-binding/v1",
    "checkpoint_format": "ufo_agent_state_v2"
  },
  "inputs": {
    "datasets": [
      {
        "id": "ds_fb0ecda9e70644f2afdd",
        "path": "/data/kk/data/ds_fb0ecda9e70644f2afdd/content",
        "purpose": "behavior_prior",
        "source_uri": "ds_92718168bfec4da6a01d",
        "validation": {
          "scope": "rpo_ufo_robot_state_v2",
          "status": "validated",
          "unchecked": [
            "RSI physics probe",
            "native learning qualification"
          ],
          "asset_sha256": "b58868a782d8863ac38059eb4eab89d27ee1d70565bf58f5a055f44510c8e842",
          "codec_sha256": "dce25d554ceb7fa69287515946d2f5cd9d41d14f1b1ef38715b819d7e962b6ef",
          "window_counts": {
            "8": {
              "g1": 6716,
              "official_rpo": 5206
            },
            "250": {
              "g1": 931,
              "official_rpo": 1279
            }
          },
          "training_ready": false,
          "manifest_sha256": "dc0c6585887507df59f675963b2733d25c9836d805e9bb89f2402398f07bd288"
        }
      }
    ],
    "base_model": null,
    "resume_from": "run_e16c3dde3996450bb7ed"
  },
  "overrides": [
    "actor.seed=42"
  ],
  "resources": {
    "cpus": 24,
    "gpus": 4,
    "shm_gb": 32,
    "devices": [],
    "gpu_mode": "exclusive",
    "memory_gb": 96
  },
  "policy": {
    "priority": 50,
    "retry_on": [
      "infra_failed",
      "oom"
    ],
    "timeout_h": 2.0,
    "max_retries": 0,
    "preemptible": false
  },
  "environment_profile": {
    "id": "rpo_bfm_ufo_mjlab_4gpu_rsi_v12",
    "hash": "sha256:99e62e1cb60def4718c86ec3b346641602a3d4d28ca96fbd1146bb9c879b6edb",
    "definition": {
      "task": "RPO23DoF原生MJCF/MJLab/MuJoCo-Warp；GPU0 learner，GPU1–3各2048环境；435M UFO六优化器，Batch2048、Replay4M；独立环境250步Tracking。",
      "reward": "九项raw aux；八个物理子步速度罚均值，权重-25；Actor裁剪1，其余5，w<=50，D lr3e-6/reward[-2,2]。",
      "failure": "400Hz原生物理/50Hz控制；MJCF质量惯量碰撞掩码，零延迟DC motor PD，踝Kd2.5、armature0.01；RSI与运行时几何一致，足法向速度<=0.15m/s；1.3倍训练速度终止，原额定限位严格影子。",
      "success": "新后端从头16→24完整恢复后新增500更新，最多2小时/无重试；训练未达标不独立评估。原physics_hz=200为历史codec元数据，实际物理频率由本Profile明确。",
      "settings": {}
    }
  },
  "image_digest": "sha256:071d5aacb6660f0bb48144fa2c6c0af34a8cfff9ef90819909dd4d2b4b63c986"
}
```

### 真实运行的完整Hydra配置

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
    contact_profile: native_mjcf_v12
    physics_profile: native_mjcf_v12
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
    experiment_name: RPO UFO native MJCF MJLab step524 3x2048
    logger_backends:
    - tensorboard
  max_epochs: 4096
  max_steps: -1
  only_eval: false
  use_training_pipeline: false
  overlap_env_bootstrap: false
  val_check_interval: -1
  save_interval: -1
  resume_dir: /workspace/resume/global_step_24
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
      id: RPO-UFO-MJLab-RSI-v12
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
      id: RPO-UFO-MJLab-RSI-v12
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

### 实际编译的物理参数与RSI候选池

三rank的质量/惯量/位置范围/solver/接触/候选池逐字段相同，以下完整参数仅打印一次；种子与Tracking分区另列，未丢弃独立rank身份。

```json
{
  "shared_parameters": {
    "schema_version": "eb/ufo-mjlab-physics/v1",
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
    "rsi_pool": [
      {
        "clip": "g1_dance1_subject1_clip3",
        "source": "g1",
        "eligible": 155,
        "original_starts": 320,
        "max_root_lift_m": 0.0019999786745756865,
        "geometric_eligible": 183,
        "velocity_rejected": 28,
        "tracking_eligible": 0,
        "normal_speed_max_mps": 0.14995892522437004
      },
      {
        "clip": "g1_dance1_subject2_clip3",
        "source": "g1",
        "eligible": 1,
        "original_starts": 207,
        "max_root_lift_m": 0.0,
        "geometric_eligible": 8,
        "velocity_rejected": 7,
        "tracking_eligible": 0,
        "normal_speed_max_mps": 0.016850794033747507
      },
      {
        "clip": "g1_dance1_subject3_clip3",
        "source": "g1",
        "eligible": 11,
        "original_starts": 157,
        "max_root_lift_m": 0.0,
        "geometric_eligible": 11,
        "velocity_rejected": 0,
        "tracking_eligible": 0,
        "normal_speed_max_mps": 0.08903391599250743
      },
      {
        "clip": "g1_fallAndGetUp1_subject1_clip3",
        "source": "g1",
        "eligible": 98,
        "original_starts": 188,
        "max_root_lift_m": 0.0019290782511234283,
        "geometric_eligible": 137,
        "velocity_rejected": 39,
        "tracking_eligible": 0,
        "normal_speed_max_mps": 0.14749255044349804
      },
      {
        "clip": "g1_fallAndGetUp1_subject4_clip3",
        "source": "g1",
        "eligible": 0,
        "original_starts": 153,
        "max_root_lift_m": null,
        "geometric_eligible": 0,
        "velocity_rejected": 0,
        "tracking_eligible": 0,
        "normal_speed_max_mps": null
      },
      {
        "clip": "g1_fallAndGetUp1_subject5_clip3",
        "source": "g1",
        "eligible": 5,
        "original_starts": 130,
        "max_root_lift_m": 0.0,
        "geometric_eligible": 8,
        "velocity_rejected": 3,
        "tracking_eligible": 0,
        "normal_speed_max_mps": 0.13500343275066987
      },
      {
        "clip": "g1_fight1_subject2_clip3",
        "source": "g1",
        "eligible": 2,
        "original_starts": 340,
        "max_root_lift_m": 0.0020000170916318893,
        "geometric_eligible": 9,
        "velocity_rejected": 7,
        "tracking_eligible": 0,
        "normal_speed_max_mps": 0.10905021868275344
      },
      {
        "clip": "g1_fight1_subject3_clip3",
        "source": "g1",
        "eligible": 0,
        "original_starts": 337,
        "max_root_lift_m": null,
        "geometric_eligible": 0,
        "velocity_rejected": 0,
        "tracking_eligible": 0,
        "normal_speed_max_mps": null
      },
      {
        "clip": "g1_fight1_subject5_clip3",
        "source": "g1",
        "eligible": 8,
        "original_starts": 376,
        "max_root_lift_m": 0.0,
        "geometric_eligible": 16,
        "velocity_rejected": 8,
        "tracking_eligible": 0,
        "normal_speed_max_mps": 0.14350248123616954
      },
      {
        "clip": "g1_jumps1_subject1_clip3",
        "source": "g1",
        "eligible": 32,
        "original_starts": 313,
        "max_root_lift_m": 0.0,
        "geometric_eligible": 42,
        "velocity_rejected": 10,
        "tracking_eligible": 0,
        "normal_speed_max_mps": 0.11807671882510415
      },
      {
        "clip": "g1_jumps1_subject2_clip3",
        "source": "g1",
        "eligible": 3,
        "original_starts": 358,
        "max_root_lift_m": 0.0019999647047370672,
        "geometric_eligible": 8,
        "velocity_rejected": 5,
        "tracking_eligible": 0,
        "normal_speed_max_mps": 0.14521208615639442
      },
      {
        "clip": "g1_jumps1_subject5_clip3",
        "source": "g1",
        "eligible": 5,
        "original_starts": 344,
        "max_root_lift_m": 0.0,
        "geometric_eligible": 6,
        "velocity_rejected": 1,
        "tracking_eligible": 0,
        "normal_speed_max_mps": 0.14246651897605273
      },
      {
        "clip": "g1_run1_subject2_clip3",
        "source": "g1",
        "eligible": 15,
        "original_starts": 344,
        "max_root_lift_m": 0.0,
        "geometric_eligible": 29,
        "velocity_rejected": 14,
        "tracking_eligible": 0,
        "normal_speed_max_mps": 0.1402887629296184
      },
      {
        "clip": "g1_run1_subject5_clip3",
        "source": "g1",
        "eligible": 4,
        "original_starts": 127,
        "max_root_lift_m": 0.0,
        "geometric_eligible": 5,
        "velocity_rejected": 1,
        "tracking_eligible": 0,
        "normal_speed_max_mps": 0.029606838280292536
      },
      {
        "clip": "g1_sprint1_subject2_clip3",
        "source": "g1",
        "eligible": 165,
        "original_starts": 306,
        "max_root_lift_m": 0.001999995205551386,
        "geometric_eligible": 178,
        "velocity_rejected": 13,
        "tracking_eligible": 0,
        "normal_speed_max_mps": 0.14655554343129124
      },
      {
        "clip": "g1_sprint1_subject4_clip3",
        "source": "g1",
        "eligible": 160,
        "original_starts": 379,
        "max_root_lift_m": 0.0019999679643660784,
        "geometric_eligible": 178,
        "velocity_rejected": 18,
        "tracking_eligible": 88,
        "normal_speed_max_mps": 0.14864236534598466
      },
      {
        "clip": "g1_walk1_subject1_clip3",
        "source": "g1",
        "eligible": 160,
        "original_starts": 392,
        "max_root_lift_m": 0.0,
        "geometric_eligible": 173,
        "velocity_rejected": 13,
        "tracking_eligible": 44,
        "normal_speed_max_mps": 0.1430332513635609
      },
      {
        "clip": "g1_walk1_subject2_clip3",
        "source": "g1",
        "eligible": 358,
        "original_starts": 392,
        "max_root_lift_m": 0.0,
        "geometric_eligible": 380,
        "velocity_rejected": 22,
        "tracking_eligible": 130,
        "normal_speed_max_mps": 0.14876935331450417
      },
      {
        "clip": "g1_walk1_subject5_clip3",
        "source": "g1",
        "eligible": 90,
        "original_starts": 392,
        "max_root_lift_m": 0.0018110821256414056,
        "geometric_eligible": 130,
        "velocity_rejected": 40,
        "tracking_eligible": 24,
        "normal_speed_max_mps": 0.14463494078018915
      },
      {
        "clip": "g1_walk2_subject1_clip3",
        "source": "g1",
        "eligible": 176,
        "original_starts": 392,
        "max_root_lift_m": 0.0,
        "geometric_eligible": 206,
        "velocity_rejected": 30,
        "tracking_eligible": 88,
        "normal_speed_max_mps": 0.1481767628218937
      },
      {
        "clip": "g1_walk2_subject3_clip3",
        "source": "g1",
        "eligible": 140,
        "original_starts": 392,
        "max_root_lift_m": 0.0,
        "geometric_eligible": 153,
        "velocity_rejected": 13,
        "tracking_eligible": 62,
        "normal_speed_max_mps": 0.145671936610597
      },
      {
        "clip": "g1_walk2_subject4_clip3",
        "source": "g1",
        "eligible": 251,
        "original_starts": 377,
        "max_root_lift_m": 0.0019999807700514793,
        "geometric_eligible": 271,
        "velocity_rejected": 20,
        "tracking_eligible": 0,
        "normal_speed_max_mps": 0.1487947890569247
      },
      {
        "clip": "rpo_114_08",
        "source": "official_rpo",
        "eligible": 13,
        "original_starts": 435,
        "max_root_lift_m": 0.02023807168006897,
        "geometric_eligible": 23,
        "velocity_rejected": 10,
        "tracking_eligible": 11,
        "normal_speed_max_mps": 0.12476524495724414
      },
      {
        "clip": "rpo_114_09",
        "source": "official_rpo",
        "eligible": 84,
        "original_starts": 774,
        "max_root_lift_m": 0.0,
        "geometric_eligible": 94,
        "velocity_rejected": 10,
        "tracking_eligible": 25,
        "normal_speed_max_mps": 0.14718967526243504
      },
      {
        "clip": "rpo_127_04",
        "source": "official_rpo",
        "eligible": 0,
        "original_starts": 1,
        "max_root_lift_m": null,
        "geometric_eligible": 0,
        "velocity_rejected": 0,
        "tracking_eligible": 0,
        "normal_speed_max_mps": null
      },
      {
        "clip": "rpo_36_01",
        "source": "official_rpo",
        "eligible": 46,
        "original_starts": 113,
        "max_root_lift_m": 0.021992022171616554,
        "geometric_eligible": 63,
        "velocity_rejected": 17,
        "tracking_eligible": 0,
        "normal_speed_max_mps": 0.13336111476340784
      },
      {
        "clip": "rpo_36_11",
        "source": "official_rpo",
        "eligible": 103,
        "original_starts": 1066,
        "max_root_lift_m": 0.0219456497579813,
        "geometric_eligible": 185,
        "velocity_rejected": 82,
        "tracking_eligible": 14,
        "normal_speed_max_mps": 0.1487074913323966
      },
      {
        "clip": "rpo_B10_-__Walk_turn_left_45_stageii",
        "source": "official_rpo",
        "eligible": 6,
        "original_starts": 13,
        "max_root_lift_m": 0.0036081799771636724,
        "geometric_eligible": 13,
        "velocity_rejected": 7,
        "tracking_eligible": 0,
        "normal_speed_max_mps": 0.10923854966130817
      },
      {
        "clip": "rpo_B11_-__Walk_turn_left_135_stageii",
        "source": "official_rpo",
        "eligible": 10,
        "original_starts": 14,
        "max_root_lift_m": 0.02008538879454136,
        "geometric_eligible": 14,
        "velocity_rejected": 4,
        "tracking_eligible": 0,
        "normal_speed_max_mps": 0.13797726363764667
      },
      {
        "clip": "rpo_B13_-__Walk_turn_right_90_stageii",
        "source": "official_rpo",
        "eligible": 159,
        "original_starts": 229,
        "max_root_lift_m": 0.021529648452997208,
        "geometric_eligible": 209,
        "velocity_rejected": 50,
        "tracking_eligible": 0,
        "normal_speed_max_mps": 0.1446504917512562
      },
      {
        "clip": "rpo_B14_-__Walk_turn_right_45_t2_stageii",
        "source": "official_rpo",
        "eligible": 104,
        "original_starts": 143,
        "max_root_lift_m": 0.020397944375872612,
        "geometric_eligible": 142,
        "velocity_rejected": 38,
        "tracking_eligible": 0,
        "normal_speed_max_mps": 0.14825249313373162
      },
      {
        "clip": "rpo_B15_-__Walk_turn_around_stageii",
        "source": "official_rpo",
        "eligible": 5,
        "original_starts": 7,
        "max_root_lift_m": 0.018899673596024513,
        "geometric_eligible": 7,
        "velocity_rejected": 2,
        "tracking_eligible": 0,
        "normal_speed_max_mps": 0.1271776023096718
      },
      {
        "clip": "rpo_B4_-_Stand_to_Walk_backwards_stageii",
        "source": "official_rpo",
        "eligible": 0,
        "original_starts": 1,
        "max_root_lift_m": null,
        "geometric_eligible": 1,
        "velocity_rejected": 1,
        "tracking_eligible": 0,
        "normal_speed_max_mps": null
      },
      {
        "clip": "rpo_B5_-__Walk_backwards_stageii",
        "source": "official_rpo",
        "eligible": 33,
        "original_starts": 108,
        "max_root_lift_m": 0.021808993071317673,
        "geometric_eligible": 101,
        "velocity_rejected": 68,
        "tracking_eligible": 0,
        "normal_speed_max_mps": 0.14196069787896384
      },
      {
        "clip": "rpo_B9_-__Walk_turn_left_90_stageii",
        "source": "official_rpo",
        "eligible": 5,
        "original_starts": 7,
        "max_root_lift_m": 0.018748613074421883,
        "geometric_eligible": 7,
        "velocity_rejected": 2,
        "tracking_eligible": 0,
        "normal_speed_max_mps": 0.1279541372804039
      },
      {
        "clip": "rpo_C12_-_run_turn_left_45_stageii",
        "source": "official_rpo",
        "eligible": 0,
        "original_starts": 11,
        "max_root_lift_m": null,
        "geometric_eligible": 2,
        "velocity_rejected": 2,
        "tracking_eligible": 0,
        "normal_speed_max_mps": null
      },
      {
        "clip": "rpo_C17_-_run_change_direction_stageii",
        "source": "official_rpo",
        "eligible": 1,
        "original_starts": 67,
        "max_root_lift_m": 0.02094338834285736,
        "geometric_eligible": 13,
        "velocity_rejected": 12,
        "tracking_eligible": 0,
        "normal_speed_max_mps": 0.08836470190685453
      },
      {
        "clip": "rpo_C1_-_stand_to_run_stageii",
        "source": "official_rpo",
        "eligible": 100,
        "original_starts": 136,
        "max_root_lift_m": 0.021625429391860962,
        "geometric_eligible": 107,
        "velocity_rejected": 7,
        "tracking_eligible": 0,
        "normal_speed_max_mps": 0.14340187104558202
      },
      {
        "clip": "rpo_C3_-_run_stageii",
        "source": "official_rpo",
        "eligible": 1,
        "original_starts": 36,
        "max_root_lift_m": 0.012029066681861877,
        "geometric_eligible": 6,
        "velocity_rejected": 5,
        "tracking_eligible": 0,
        "normal_speed_max_mps": 0.0775231785581496
      },
      {
        "clip": "rpo_C4_-_run_to_walk_a_stageii",
        "source": "official_rpo",
        "eligible": 14,
        "original_starts": 72,
        "max_root_lift_m": 0.01279268879443407,
        "geometric_eligible": 48,
        "velocity_rejected": 34,
        "tracking_eligible": 0,
        "normal_speed_max_mps": 0.1406593510426886
      },
      {
        "clip": "rpo_C5_-_walk_to_run_stageii",
        "source": "official_rpo",
        "eligible": 9,
        "original_starts": 47,
        "max_root_lift_m": 0.01850222982466221,
        "geometric_eligible": 25,
        "velocity_rejected": 16,
        "tracking_eligible": 0,
        "normal_speed_max_mps": 0.14396789904815765
      },
      {
        "clip": "rpo_move_back",
        "source": "official_rpo",
        "eligible": 271,
        "original_starts": 304,
        "max_root_lift_m": 0.0038758968003094196,
        "geometric_eligible": 298,
        "velocity_rejected": 27,
        "tracking_eligible": 62,
        "normal_speed_max_mps": 0.14790810872584087
      },
      {
        "clip": "rpo_move_l",
        "source": "official_rpo",
        "eligible": 97,
        "original_starts": 272,
        "max_root_lift_m": 0.00035552948247641325,
        "geometric_eligible": 111,
        "velocity_rejected": 14,
        "tracking_eligible": 30,
        "normal_speed_max_mps": 0.13400432594949468
      },
      {
        "clip": "rpo_move_r",
        "source": "official_rpo",
        "eligible": 113,
        "original_starts": 242,
        "max_root_lift_m": 0.004947061184793711,
        "geometric_eligible": 128,
        "velocity_rejected": 15,
        "tracking_eligible": 0,
        "normal_speed_max_mps": 0.1487628561516832
      },
      {
        "clip": "rpo_run_start_180_R_001__A345",
        "source": "official_rpo",
        "eligible": 3,
        "original_starts": 114,
        "max_root_lift_m": 0.0,
        "geometric_eligible": 5,
        "velocity_rejected": 2,
        "tracking_eligible": 0,
        "normal_speed_max_mps": 0.13470170948044125
      },
      {
        "clip": "rpo_run_start_180_R_001__A345_M",
        "source": "official_rpo",
        "eligible": 4,
        "original_starts": 115,
        "max_root_lift_m": 0.0,
        "geometric_eligible": 6,
        "velocity_rejected": 2,
        "tracking_eligible": 0,
        "normal_speed_max_mps": 0.13790660873340807
      },
      {
        "clip": "rpo_run_stop_180_R_001__A345",
        "source": "official_rpo",
        "eligible": 110,
        "original_starts": 158,
        "max_root_lift_m": 0.0,
        "geometric_eligible": 112,
        "velocity_rejected": 2,
        "tracking_eligible": 0,
        "normal_speed_max_mps": 0.10800908574350976
      },
      {
        "clip": "rpo_run_stop_180_R_001__A345_M",
        "source": "official_rpo",
        "eligible": 110,
        "original_starts": 158,
        "max_root_lift_m": 0.0,
        "geometric_eligible": 112,
        "velocity_rejected": 2,
        "tracking_eligible": 0,
        "normal_speed_max_mps": 0.1128801129645981
      },
      {
        "clip": "rpo_turn_l",
        "source": "official_rpo",
        "eligible": 127,
        "original_starts": 332,
        "max_root_lift_m": 0.004752741195261478,
        "geometric_eligible": 142,
        "velocity_rejected": 15,
        "tracking_eligible": 73,
        "normal_speed_max_mps": 0.13837550343998825
      },
      {
        "clip": "rpo_turn_r",
        "source": "official_rpo",
        "eligible": 147,
        "original_starts": 231,
        "max_root_lift_m": 0.005222457926720381,
        "geometric_eligible": 167,
        "velocity_rejected": 20,
        "tracking_eligible": 0,
        "normal_speed_max_mps": 0.1317806010563765
      }
    ],
    "qualification": "not_requested"
  },
  "ranks": [
    {
      "seed": 42,
      "num_envs": 2048,
      "tracking_count": 1024,
      "tracking_partition_sha256": "5b653aad83c9abf3cbcb14748009093b85939c3186041e13b527d8370463637d"
    },
    {
      "seed": 43,
      "num_envs": 2048,
      "tracking_count": 1024,
      "tracking_partition_sha256": "4aedbad59e9db02006a65da4702725c2f8aee6c4fe1e1cef3a7d516366d8e0c9"
    },
    {
      "seed": 44,
      "num_envs": 2048,
      "tracking_count": 1024,
      "tracking_partition_sha256": "9bcdb64f3a3805ddf44b0aa6f6581dec9b67da4cfc80e0e2336669b4b9399fc6"
    }
  ]
}
```

参考源概率official_rpo=0.5、g1=0.5；仅train clip用于expert、RSI与Tracking，validation/test/acceptance不参与拟合。153帧body位姿对齐实测：
```json
{
  "frames_checked": 153,
  "sources": [
    "g1",
    "official_rpo"
  ],
  "max_body_position_difference_m": 0.0009958325013881275,
  "max_quaternion_angle_rad": 0.0011664758917460482
}
```

## 核心训练指标与五窗趋势

|指标|点数|首值|末值|全段均值|最大值|5个100更新窗口均值|
|---|---:|---:|---:|---:|---:|---|
|env/episode_len|32|2.54764|9.53861|5.64804|9.53861|3.2383 → 4.13525 → 5.21094 → 6.53911 → 8.49378|
|train/ufo/strict_shadow_episode_len|32|2.4331|8.35549|5.08817|8.35549|2.98339 → 3.83911 → 4.79267 → 5.83511 → 7.46924|
|train/ufo/physics_peak_normalized_velocity|32|2.47071|3.3247|3.14606|4.47197|3.15234 → 3.30115 → 3.35142 → 3.15565 → 2.82215|
|train/ufo/strict_velocity_exposure_fraction|32|0.00147148|0.000844099|0.00124434|0.00180761|0.00158658 → 0.00130371 → 0.00123475 → 0.00120257 → 0.00095012|
|train/ufo/first_step_terminated|32|1028|22|171.188|1028|477.167 → 209.333 → 122.5 → 62.1429 → 27|
|train/ufo/first_step_samples|32|11506|2507|4870.69|11506|7892 → 5918.33 → 4668.5 → 3637 → 2790|
|train/ufo/applied_action_saturation_fraction|32|0.0443115|0.0798711|0.0921252|0.113308|0.0640468 → 0.10283 → 0.108241 → 0.10294 → 0.0823884|
|train/ufo/requested_action_saturation_fraction|32|0.482466|0.278055|0.397377|0.482466|0.466516 → 0.446353 → 0.417918 → 0.377337 → 0.29857|
|train/ufo/hard_position_count|32|6962|2179|4323.22|8499|6702.33 → 5542.33 → 4224.5 → 3195 → 2451.86|
|train/ufo/hard_velocity_count|32|2599|1145|1791.81|3013|2626.33 → 1855 → 1733.67 → 1588.29 → 1275.71|
|train/ufo/training_velocity_count|32|518|243|447.438|818|633.667 → 498.833 → 473.333 → 397.286 → 271.714|
|train/ufo/critic_loss|500|3.16118|0.451438|0.604403|4.3053|1.66448 → 0.386455 → 0.289756 → 0.307572 → 0.373756|
|train/ufo/Q_fb|500|-4.52322|52.7632|27.5138|54.1374|6.96761 → 18.4169 → 27.7906 → 36.661 → 47.733|
|train/ufo/fb_loss|500|284417|27287.7|73868.1|284417|168303 → 81707.2 → 51123.3 → 37985.4 → 30221.5|
|train/ufo/actor_loss|500|7.1505|-39.5853|-21.0327|7.1505|-4.0145 → -14.6703 → -22.0017 → -28.1297 → -36.3473|
|train/ufo/mean_disc_reward|500|0.101914|-0.7044|-0.296731|0.112646|0.0461355 → -0.112957 → -0.301209 → -0.478484 → -0.637142|

窗口为25–124、125–224、225–324、325–424、425–524。learner指标每更新一记录，环境/动力学指标每采样轮一记录；窗口均值对日志点等权。所有终止episode的加权分布、首步比例和最大值以逐行trace核验为准。指标末50点均值可能覆盖较大更新跨度，不能当成末50更新。

## 全部标量指标统计

|指标|点数|首值|末值|全段均值|最大值|5个100更新窗口均值|
|---|---:|---:|---:|---:|---:|---|
|env/episode_len|32|2.54764|9.53861|5.64804|9.53861|3.2383 → 4.13525 → 5.21094 → 6.53911 → 8.49378|
|env/num_trajectories|32|7063|2512|4698.09|8757|7104.5 → 5911 → 4634.67 → 3584.86 → 2763.43|
|env/return|32|0|0|0|0|0 → 0 → 0 → 0 → 0|
|env/reward|32|0|0|0|0|0 → 0 → 0 → 0 → 0|
|env/success_once|32|0|0|0|0|0 → 0 → 0 → 0 → 0|
|rollout/ufo/received_transitions|32|24576|24576|24576|24576|24576 → 24576 → 24576 → 24576 → 24576|
|time/actor_training|32|16.5662|4.32541|12.2293|16.5662|13.0276 → 12.3389 → 12.3313 → 12.3595 → 11.2332|
|time/cal_adv_and_returns|32|0.00399375|0.00835562|0.0113964|0.0275557|0.0123433 → 0.0157122 → 0.00885646 → 0.00970326 → 0.0107556|
|time/env/compute_bootstrap_rewards|32|0.000268436|0.000260742|0.000268112|0.000314437|0.000267398 → 0.000259644 → 0.000266996 → 0.000272846 → 0.000272204|
|time/env/env/bootstrap_step|32|0.0156202|0.000316415|0.000764726|0.0156202|0.00283145 → 0.000278572 → 0.000275973 → 0.000299785 → 0.000293827|
|time/env/env/send_rollout_trajectories|32|0.448549|0.389538|0.405278|0.448549|0.407799 → 0.400868 → 0.406991 → 0.404536 → 0.406171|
|time/env/env_interact_step|32|1.52489|1.07247|1.09097|1.52489|1.15788 → 1.07164 → 1.08929 → 1.07903 → 1.06359|
|time/env/interact|32|4.58114|2.81774|2.99361|4.58114|3.35414 → 3.04474 → 2.93786 → 2.85853 → 2.82362|
|time/env/run_interact_once|32|4.5811|2.81771|2.99358|4.5811|3.35411 → 3.04471 → 2.93784 → 2.8585 → 2.82359|
|time/generate_rollouts|32|5.03623|3.16245|3.3121|5.03623|3.73379 → 3.37352 → 3.2447 → 3.14109 → 3.12681|
|time/rollout/generate_one_epoch|32|4.13996|2.41138|2.58902|4.13996|2.95277 → 2.65954 → 2.5212 → 2.44823 → 2.41569|
|time/rollout/predict|32|1.46036|0.214879|0.410961|1.46036|0.712649 → 0.451166 → 0.374985 → 0.297488 → 0.262218|
|time/rollout/rollout/generate|32|4.14308|2.41148|2.58926|4.14308|2.95349 → 2.65965 → 2.52132 → 2.44844 → 2.41579|
|time/rollout/sync_model_from_actor|32|14.3636|13.0632|12.9606|14.3636|13.3108 → 12.9857 → 12.8339 → 12.8056 → 12.9027|
|time/step|32|39.917|24.0957|32.0444|39.917|33.7343 → 32.3118 → 31.9016 → 31.798 → 30.7354|
|time/sync_weights|32|18.3098|16.5988|16.491|18.3098|16.96 → 16.5831 → 16.3161 → 16.2871 → 16.364|
|train/ufo/B|500|-0.00634608|-0.00097147|-0.00528608|0.00077509|-0.0092336 → -0.00623641 → -0.00434117 → -0.00369192 → -0.00292732|
|train/ufo/B_norm|500|16|16|16|16|16 → 16 → 16 → 16 → 16|
|train/ufo/F1|500|0.0325462|0.00394637|0.0100795|0.0382783|0.0200213 → 0.0147366 → 0.00868022 → 0.00436346 → 0.00259607|
|train/ufo/M1|500|-2.49519|0.923884|1.26608|6.68437|2.46028 → 1.33816 → 0.839621 → 0.806473 → 0.885883|
|train/ufo/Q1|500|-4.24979|-2.94566|-2.23568|-1.39871|-3.00597 → -1.74209 → -1.78549 → -2.11857 → -2.52631|
|train/ufo/Q_aux|500|-4.13476|-4.49437|-3.82563|-2.88954|-3.90211 → -3.37543 → -3.66711 → -3.97515 → -4.20836|
|train/ufo/Q_discriminator|500|-4.00238|-3.47342|-2.555|-1.63468|-3.29323 → -1.9927 → -2.11074 → -2.47569 → -2.90265|
|train/ufo/Q_fb|500|-4.52322|52.7632|27.5138|54.1374|6.96761 → 18.4169 → 27.7906 → 36.661 → 47.733|
|train/ufo/action_abs_mean|32|0.928865|0.796306|0.875467|0.928865|0.919951 → 0.908952 → 0.890021 → 0.86242 → 0.809207|
|train/ufo/action_saturation_fraction|32|0.482466|0.278055|0.397377|0.482466|0.466516 → 0.446353 → 0.417918 → 0.377337 → 0.29857|
|train/ufo/actor_loss|500|7.1505|-39.5853|-21.0327|7.1505|-4.0145 → -14.6703 → -22.0017 → -28.1297 → -36.3473|
|train/ufo/actor_optimizer/grad_clip_limit|500|1|1|1|1|1 → 1 → 1 → 1 → 1|
|train/ufo/actor_optimizer/grad_norm_after_clip_l2|500|0.999999|1|0.998586|1|0.992929 → 1 → 1 → 1 → 1|
|train/ufo/actor_optimizer/grad_norm_l2|500|1.06358|14.1334|6.49326|20.1754|1.76074 → 3.75623 → 6.56337 → 8.45033 → 11.9356|
|train/ufo/actor_optimizer/lr|500|0.0003|0.0003|0.0003|0.0003|0.0003 → 0.0003 → 0.0003 → 0.0003 → 0.0003|
|train/ufo/actor_optimizer/parameter_delta_l2|500|0.448045|0.274046|0.327016|0.492189|0.383945 → 0.347021 → 0.309038 → 0.311121 → 0.283957|
|train/ufo/actor_reg_weight_applied|500|9.28975|50|31.7682|50|12.6238 → 22.4546 → 32.2813 → 41.8467 → 49.6347|
|train/ufo/actor_reg_weight_raw|500|9.28975|58.4687|32.5089|60.5829|12.6238 → 22.4546 → 32.2813 → 41.8467 → 53.3379|
|train/ufo/applied_action_abs_mean|32|0.423863|0.511644|0.507268|0.529875|0.464941 → 0.523752 → 0.519708 → 0.517988 → 0.508035|
|train/ufo/applied_action_saturation_fraction|32|0.0443115|0.0798711|0.0921252|0.113308|0.0640468 → 0.10283 → 0.108241 → 0.10294 → 0.0823884|
|train/ufo/auxQ1|500|-4.14917|-3.86214|-3.42805|-2.44255|-3.54577 → -3.0762 → -3.28586 → -3.51028 → -3.72215|
|train/ufo/aux_critic_loss|500|2.95023|0.950194|1.15254|3.92026|1.6144 → 1.05108 → 0.980675 → 1.08953 → 1.027|
|train/ufo/aux_critic_optimizer/grad_clip_limit|500|5|5|5|5|5 → 5 → 5 → 5 → 5|
|train/ufo/aux_critic_optimizer/grad_norm_after_clip_l2|500|5|5|4.75751|5|4.97699 → 4.55544 → 4.51679 → 4.88695 → 4.85141|
|train/ufo/aux_critic_optimizer/grad_norm_l2|500|18.5648|9.95915|11.443|47.6119|15.1809 → 9.68487 → 8.95125 → 13.2152 → 10.1827|
|train/ufo/aux_critic_optimizer/lr|500|0.0003|0.0003|0.0003|0.0003|0.0003 → 0.0003 → 0.0003 → 0.0003 → 0.0003|
|train/ufo/aux_critic_optimizer/parameter_delta_l2|500|0.457356|0.574467|0.418999|0.824958|0.33789 → 0.424461 → 0.463325 → 0.408031 → 0.461285|
|train/ufo/aux_rew/limits_dof_pos|500|0.0347833|0.0679235|0.0604568|0.0745533|0.0442568 → 0.0598303 → 0.0648822 → 0.066211 → 0.0671036|
|train/ufo/aux_rew/limits_dof_vel|500|0.0126406|0.0116309|0.012249|0.0178357|0.012635 → 0.0126774 → 0.012184 → 0.0121481 → 0.0116005|
|train/ufo/aux_rew/limits_torque|500|0|0|0|0|0 → 0 → 0 → 0 → 0|
|train/ufo/aux_rew/penalty_action_rate|500|0.74059|0.551886|0.60324|0.748201|0.714368 → 0.626976 → 0.577441 → 0.553149 → 0.544265|
|train/ufo/aux_rew/penalty_ankle_roll|500|0.014137|0.0473747|0.030098|0.0518412|0.0173659 → 0.023818 → 0.0291647 → 0.0359166 → 0.044225|
|train/ufo/aux_rew/penalty_feet_ori|500|0.0810567|0.120906|0.09548|0.128496|0.0756994 → 0.085361 → 0.0953182 → 0.105137 → 0.115885|
|train/ufo/aux_rew/penalty_slippage|500|0.302768|0.255432|0.273928|0.3304|0.294267 → 0.277574 → 0.271402 → 0.265725 → 0.260674|
|train/ufo/aux_rew/penalty_torques|500|4728.69|4290.36|4397.43|5113.25|4707.93 → 4270.08 → 4341.65 → 4377.51 → 4290|
|train/ufo/aux_rew/penalty_undesired_contact|500|0|0|0|0|0 → 0 → 0 → 0 → 0|
|train/ufo/backward_optimizer/grad_clip_limit|500|5|5|5|5|5 → 5 → 5 → 5 → 5|
|train/ufo/backward_optimizer/grad_norm_after_clip_l2|500|5|5|5|5|5 → 5 → 5 → 5 → 5|
|train/ufo/backward_optimizer/grad_norm_l2|500|557655|19611.2|94832.3|587070|274955 → 98535.8 → 47813.6 → 30341.2 → 22516.1|
|train/ufo/backward_optimizer/lr|500|1e-05|1e-05|1e-05|1e-05|1e-05 → 1e-05 → 1e-05 → 1e-05 → 1e-05|
|train/ufo/backward_optimizer/parameter_delta_l2|500|0.00344269|0.00365229|0.00417533|0.00444762|0.00392679 → 0.00441489 → 0.00442032 → 0.0042504 → 0.00386425|
|train/ufo/collected_transitions|32|24576|24576|24576|24576|24576 → 24576 → 24576 → 24576 → 24576|
|train/ufo/contact_termination_count|32|0|0|0|0|0 → 0 → 0 → 0 → 0|
|train/ufo/critic_loss|500|3.16118|0.451438|0.604403|4.3053|1.66448 → 0.386455 → 0.289756 → 0.307572 → 0.373756|
|train/ufo/critic_optimizer/grad_clip_limit|500|5|5|5|5|5 → 5 → 5 → 5 → 5|
|train/ufo/critic_optimizer/grad_norm_after_clip_l2|500|5|5|4.84384|5|5 → 4.84654 → 4.85943 → 4.64581 → 4.86744|
|train/ufo/critic_optimizer/grad_norm_l2|500|31.6397|10.8967|10.2812|59.0686|17.0413 → 8.87921 → 8.60086 → 7.8669 → 9.01789|
|train/ufo/critic_optimizer/lr|500|0.0003|0.0003|0.0003|0.0003|0.0003 → 0.0003 → 0.0003 → 0.0003 → 0.0003|
|train/ufo/critic_optimizer/parameter_delta_l2|500|0.378404|0.329845|0.339215|0.538558|0.351617 → 0.304101 → 0.307448 → 0.359903 → 0.373004|
|train/ufo/disc_expert_loss|500|0.589835|0.321389|0.40898|0.589835|0.534663 → 0.435287 → 0.387054 → 0.359424 → 0.32847|
|train/ufo/disc_loss|500|3.43611|1.03256|1.48632|3.43611|2.54721 → 1.47115 → 1.22035 → 1.14039 → 1.05251|
|train/ufo/disc_reward_clip_fraction|500|0|0.078125|0.0311836|0.0957031|0 → 0.00404785 → 0.0296875 → 0.0508691 → 0.0713135|
|train/ufo/disc_train_loss|500|0.747852|0.367726|0.506974|0.758333|0.700718 → 0.559855 → 0.474887 → 0.421094 → 0.378317|
|train/ufo/disc_wgan_gp_loss|500|0.209843|0.0343447|0.0570366|0.209843|0.131183 → 0.0476006 → 0.0358404 → 0.0359876 → 0.0345719|
|train/ufo/discriminator_optimizer/grad_clip_limit|500|5|5|5|5|5 → 5 → 5 → 5 → 5|
|train/ufo/discriminator_optimizer/grad_norm_after_clip_l2|500|5|0.999071|2.52875|5|5 → 3.76218 → 1.54887 → 1.23089 → 1.10182|
|train/ufo/discriminator_optimizer/grad_norm_l2|500|8.49824|0.999071|3.05797|8.50389|7.48684 → 3.92143 → 1.54887 → 1.23089 → 1.10182|
|train/ufo/discriminator_optimizer/lr|500|3e-06|3e-06|3e-06|3e-06|3e-06 → 3e-06 → 3e-06 → 3e-06 → 3e-06|
|train/ufo/discriminator_optimizer/parameter_delta_l2|500|0.00445429|0.0026095|0.00365335|0.00541391|0.00491069 → 0.0046909 → 0.0031115 → 0.00285843 → 0.00269525|
|train/ufo/environment_control_steps|32|16|140|78|140|26 → 50 → 74 → 100 → 128|
|train/ufo/expert_draws/g1|32|41104|536088|294789|536088|82226.7 → 181020 → 278861 → 385026 → 497918|
|train/ufo/expert_draws/official_rpo|32|40816|537064|294267|537064|81613.3 → 179428 → 278195 → 385022 → 497995|
|train/ufo/fb_diag|500|-7.61286|-219.948|-147.255|-2.32972|-39.9295 → -110.127 → -165.269 → -200.393 → -220.555|
|train/ufo/fb_loss|500|284417|27287.7|73868.1|284417|168303 → 81707.2 → 51123.3 → 37985.4 → 30221.5|
|train/ufo/fb_offdiag|500|264.952|198.944|136.494|286.773|101.507 → 100.812 → 130.937 → 159.914 → 189.299|
|train/ufo/first_step_samples|32|11506|2507|4870.69|11506|7892 → 5918.33 → 4668.5 → 3637 → 2790|
|train/ufo/first_step_terminated|32|1028|22|171.188|1028|477.167 → 209.333 → 122.5 → 62.1429 → 27|
|train/ufo/foot_contact_peak_N|32|2979.11|3389.51|3733.7|5212.73|3368.95 → 4064.49 → 4200.56 → 3684.32 → 3412.04|
|train/ufo/forward_optimizer/grad_clip_limit|500|5|5|5|5|5 → 5 → 5 → 5 → 5|
|train/ufo/forward_optimizer/grad_norm_after_clip_l2|500|5|5|5|5|5 → 5 → 5 → 5 → 5|
|train/ufo/forward_optimizer/grad_norm_l2|500|4798.85|111.625|395.761|5124.61|1125.08 → 403.374 → 208.677 → 132.633 → 109.041|
|train/ufo/forward_optimizer/lr|500|0.0003|0.0003|0.0003|0.0003|0.0003 → 0.0003 → 0.0003 → 0.0003 → 0.0003|
|train/ufo/forward_optimizer/parameter_delta_l2|500|0.281503|0.882718|0.654559|0.971708|0.292078 → 0.501374 → 0.707093 → 0.864964 → 0.907285|
|train/ufo/hard_position_count|32|6962|2179|4323.22|8499|6702.33 → 5542.33 → 4224.5 → 3195 → 2451.86|
|train/ufo/hard_torque_count|32|0|0|0|0|0 → 0 → 0 → 0 → 0|
|train/ufo/hard_velocity_count|32|2599|1145|1791.81|3013|2626.33 → 1855 → 1733.67 → 1588.29 → 1275.71|
|train/ufo/height_termination_count|32|0|51|15.2812|58|3.33333 → 2.5 → 10.1667 → 15.7143 → 40.4286|
|train/ufo/latched_velocity_terms_without_boundary_violation|32|1355|710|1086.78|2135|1625.17 → 1133.67 → 1050.5 → 932.714 → 770.286|
|train/ufo/learner_joint_updates|500|25|524|274.5|524|74.5 → 174.5 → 274.5 → 374.5 → 474.5|
|train/ufo/mean_aux_reward|500|-0.591551|-0.786669|-0.731422|-0.583685|-0.654637 → -0.726371 → -0.754035 → -0.750603 → -0.771464|
|train/ufo/mean_disc_reward|500|0.101914|-0.7044|-0.296731|0.112646|0.0461355 → -0.112957 → -0.301209 → -0.478484 → -0.637142|
|train/ufo/mean_disc_reward_raw|500|0.101914|-0.748292|-0.314224|0.112646|0.0461355 → -0.113858 → -0.315461 → -0.508452 → -0.679486|
|train/ufo/mean_next_Q|500|-2.5911|-2.31723|-1.81703|-1.34575|-2.18821 → -1.46487 → -1.52829 → -1.78796 → -2.11582|
|train/ufo/mean_next_auxQ|500|-3.46485|-3.56589|-3.15111|-2.668|-3.24374 → -2.76435 → -3.01128 → -3.26251 → -3.47367|
|train/ufo/online_transitions|500|98304|860160|470090|860160|163185 → 316539 → 469893 → 623247 → 777585|
|train/ufo/orientation_termination_count|32|1|146|108.094|749|155 → 128.833 → 69.5 → 91.1429 → 100.143|
|train/ufo/orth_loss|500|2841.6|273.087|738.789|2841.6|1682.41 → 817.165 → 511.577 → 380.259 → 302.528|
|train/ufo/orth_loss_diag|500|-256|-256|-256|-256|-256 → -256 → -256 → -256 → -256|
|train/ufo/orth_loss_offdiag|500|3097.6|529.087|994.789|3097.6|1938.41 → 1073.17 → 767.577 → 636.259 → 558.528|
|train/ufo/physics_peak_joint_velocity|32|19.7657|26.5976|25.4235|35.7757|25.2991 → 26.4967 → 26.8113 → 25.291 → 23.5529|
|train/ufo/physics_peak_normalized_velocity|32|2.47071|3.3247|3.14606|4.47197|3.15234 → 3.30115 → 3.35142 → 3.15565 → 2.82215|
|train/ufo/q_loss|500|0|0|0|0|0 → 0 → 0 → 0 → 0|
|train/ufo/replay_draw_new_ratio|32|0.833333|1.24762|1.18785|1.27451|1.00146 → 1.17025 → 1.22429 → 1.25282 → 1.26651|
|train/ufo/replay_draws|32|81920|1.07315e+06|589056|1.07315e+06|163840 → 360448 → 557056 → 770048 → 995913|
|train/ufo/replay_policy_age_max_updates|32|40|524|287.625|524|80 → 176 → 272 → 376 → 486.286|
|train/ufo/replay_policy_age_min_updates|32|16|4|15.625|16|16 → 16 → 16 → 16 → 14.2857|
|train/ufo/replay_retained_fraction|32|1|1|1|1|1 → 1 → 1 → 1 → 1|
|train/ufo/replay_size|32|98304|860160|479232|860160|159744 → 307200 → 454656 → 614400 → 786432|
|train/ufo/requested_action_abs_mean|32|0.928865|0.796306|0.875467|0.928865|0.919951 → 0.908952 → 0.890021 → 0.86242 → 0.809207|
|train/ufo/requested_action_saturation_fraction|32|0.482466|0.278055|0.397377|0.482466|0.466516 → 0.446353 → 0.417918 → 0.377337 → 0.29857|
|train/ufo/requested_applied_max_diff|32|1.92627|1.9528|1.95079|1.9528|1.94838 → 1.9514 → 1.95087 → 1.9528 → 1.95026|
|train/ufo/reset_control_samples|32|11506|2507|4870.69|11506|7892 → 5918.33 → 4668.5 → 3637 → 2790|
|train/ufo/reset_projection_outside_fraction|32|0|0|0|0|0 → 0 → 0 → 0 → 0|
|train/ufo/reset_projection_residual_max_rad|32|1.78814e-07|1.78814e-07|1.78814e-07|1.78814e-07|1.78814e-07 → 1.78814e-07 → 1.78814e-07 → 1.78814e-07 → 1.78814e-07|
|train/ufo/reset_projection_residual_mean_rad|32|6.39169e-08|6.45956e-08|6.40736e-08|6.47676e-08|6.42075e-08 → 6.41289e-08 → 6.39066e-08 → 6.41605e-08 → 6.39679e-08|
|train/ufo/reset_projection_residual_p95_rad|32|1.19209e-07|1.19209e-07|1.19209e-07|1.19209e-07|1.19209e-07 → 1.19209e-07 → 1.19209e-07 → 1.19209e-07 → 1.19209e-07|
|train/ufo/reset_root_lift_max_m|32|0.0219456|0.021992|0.0219761|0.021992|0.0219843 → 0.0219766 → 0.0219688 → 0.0219655 → 0.0219854|
|train/ufo/sampling_rounds|500|4|35|19.128|35|6.64 → 12.88 → 19.12 → 25.36 → 31.64|
|train/ufo/strict_shadow_episode_len|32|2.4331|8.35549|5.08817|8.35549|2.98339 → 3.83911 → 4.79267 → 5.83511 → 7.46924|
|train/ufo/strict_shadow_mixed_episode_len|32|2.42334|6.47895|4.32347|6.56691|2.9526 → 3.49386 → 3.99884 → 4.8693 → 5.94199|
|train/ufo/strict_shadow_tracking_episode_len|32|2.44267|11.2308|6.21325|11.2308|3.03612 → 4.27481 → 5.91547 → 7.21474 → 9.85175|
|train/ufo/strict_velocity_exposure_fraction|32|0.00147148|0.000844099|0.00124434|0.00180761|0.00158658 → 0.00130371 → 0.00123475 → 0.00120257 → 0.00095012|
|train/ufo/strict_velocity_transition_fraction|32|0.105754|0.0465902|0.072909|0.122599|0.106866 → 0.0754801 → 0.0705431 → 0.0646275 → 0.0519089|
|train/ufo/target_M|500|3.44364|0.555114|0.845074|3.47368|2.09614 → 0.620353 → 0.418126 → 0.504207 → 0.586544|
|train/ufo/target_Q|500|-3.71873|-2.77542|-2.23848|-1.56153|-3.00334 → -1.73795 → -1.79168 → -2.12593 → -2.5335|
|train/ufo/target_auxQ|500|-4.00761|-3.84975|-3.42796|-2.98528|-3.54842 → -3.07754 → -3.28165 → -3.50255 → -3.72967|
|train/ufo/terminated_transitions|32|7063|2512|4698.09|8757|7104.5 → 5911 → 4634.67 → 3584.86 → 2763.43|
|train/ufo/tracking_episode_starts|32|5754|962|2173.53|5754|3920 → 2693.33 → 1945 → 1513 → 1087.43|
|train/ufo/tracking_q_rmse_rad|32|0.0417452|0.201078|0.122864|0.201078|0.0799447 → 0.102695 → 0.11501 → 0.13126 → 0.175277|
|train/ufo/tracking_unique_env_fraction|32|0.5|0.5|0.5|0.5|0.5 → 0.5 → 0.5 → 0.5 → 0.5|
|train/ufo/tracking_windows_completed|32|0|0|0|0|0 → 0 → 0 → 0 → 0|
|train/ufo/training_velocity_count|32|518|243|447.438|818|633.667 → 498.833 → 473.333 → 397.286 → 271.714|
|train/ufo/truncated_transitions|32|0|0|0|0|0 → 0 → 0 → 0 → 0|
|train/ufo/unc_Q|500|4.11938|0.173247|1.05421|4.83734|3.30527 → 1.02275 → 0.470693 → 0.278418 → 0.193914|
|train/ufo/unc_auxQ|500|1.19015|0.15714|0.365907|1.20481|0.780064 → 0.41688 → 0.277512 → 0.193571 → 0.161509|
|train/ufo/velocity_reward_coverage_miss_count|32|0|0|0|0|0 → 0 → 0 → 0 → 0|
|train/ufo/wall_time_s|500|54.12|1052.78|548.389|1052.78|146.7 → 349.32 → 549.496 → 748.595 → 947.833|
|train/ufo/z_norm|500|16|16|16|16|16 → 16 → 16 → 16 → 16|

### 全部指标的分位与范围

|指标|最小值|中位数|p95|末50记录均值|
|---|---:|---:|---:|---:|
|env/episode_len|2.54764|5.59343|8.99872|5.64804|
|env/num_trajectories|2512|4368|7540.35|4698.09|
|env/return|0|0|0|0|
|env/reward|0|0|0|0|
|env/success_once|0|0|0|0|
|rollout/ufo/received_transitions|24576|24576|24576|24576|
|time/actor_training|4.32541|12.3622|12.4843|12.2293|
|time/cal_adv_and_returns|0.00274777|0.00958645|0.0242808|0.0113964|
|time/env/compute_bootstrap_rewards|0.000217562|0.000268365|0.000293031|0.000268112|
|time/env/env/bootstrap_step|0.000238993|0.000262823|0.00040744|0.000764726|
|time/env/env/send_rollout_trajectories|0.366175|0.404005|0.429561|0.405278|
|time/env/env_interact_step|1.0318|1.07946|1.12038|1.09097|
|time/env/interact|2.69257|2.93105|3.27211|2.99361|
|time/env/run_interact_once|2.69254|2.93102|3.27208|2.99358|
|time/generate_rollouts|2.92631|3.23922|3.64777|3.3121|
|time/rollout/generate_one_epoch|2.25514|2.52082|2.88531|2.58902|
|time/rollout/predict|0.214879|0.352483|0.6663|0.410961|
|time/rollout/rollout/generate|2.25524|2.52093|2.88575|2.58926|
|time/rollout/sync_model_from_actor|12.4922|12.9108|13.4637|12.9606|
|time/step|24.0957|31.9966|32.8547|32.0444|
|time/sync_weights|15.944|16.4595|17.0338|16.491|
|train/ufo/B|-0.01624|-0.00483532|-0.0013185|-0.00274857|
|train/ufo/B_norm|16|16|16|16|
|train/ufo/F1|-0.00845659|0.00985081|0.0250565|0.00275328|
|train/ufo/M1|-2.49519|0.930279|3.20031|0.887937|
|train/ufo/Q1|-4.88108|-2.07979|-1.56087|-2.63137|
|train/ufo/Q_aux|-5.39369|-3.78443|-3.2365|-4.27295|
|train/ufo/Q_discriminator|-5.40422|-2.36888|-1.83733|-3.01671|
|train/ufo/Q_fb|-4.52322|27.7395|50.5682|50.6422|
|train/ufo/action_abs_mean|0.79199|0.885465|0.922029|0.875467|
|train/ufo/action_saturation_fraction|0.274598|0.412788|0.470252|0.397377|
|train/ufo/actor_loss|-42.072|-21.9312|-1.50774|-38.8274|
|train/ufo/actor_optimizer/grad_clip_limit|1|1|1|1|
|train/ufo/actor_optimizer/grad_norm_after_clip_l2|0.825037|1|1|1|
|train/ufo/actor_optimizer/grad_norm_l2|0.825037|6.17348|13.2746|13.0203|
|train/ufo/actor_optimizer/lr|0.0003|0.0003|0.0003|0.0003|
|train/ufo/actor_optimizer/parameter_delta_l2|0.230402|0.324058|0.40312|0.273479|
|train/ufo/actor_reg_weight_applied|8.67967|32.1098|50|50|
|train/ufo/actor_reg_weight_raw|8.67967|32.1098|56.4979|56.3992|
|train/ufo/applied_action_abs_mean|0.423863|0.517561|0.524701|0.507268|
|train/ufo/applied_action_saturation_fraction|0.0443115|0.0981366|0.110369|0.0921252|
|train/ufo/auxQ1|-4.62276|-3.40718|-2.96145|-3.79325|
|train/ufo/aux_critic_loss|0.761263|0.995434|2.13187|1.05212|
|train/ufo/aux_critic_optimizer/grad_clip_limit|5|5|5|5|
|train/ufo/aux_critic_optimizer/grad_norm_after_clip_l2|1.44192|5|5|4.74507|
|train/ufo/aux_critic_optimizer/grad_norm_l2|1.44192|9.92658|25.6192|8.79998|
|train/ufo/aux_critic_optimizer/lr|0.0003|0.0003|0.0003|0.0003|
|train/ufo/aux_critic_optimizer/parameter_delta_l2|0.25709|0.412132|0.564595|0.488826|
|train/ufo/aux_rew/limits_dof_pos|0.0326519|0.0642268|0.0687287|0.0670803|
|train/ufo/aux_rew/limits_dof_vel|0.00812645|0.0120928|0.0148518|0.0114654|
|train/ufo/aux_rew/limits_torque|0|0|0|0|
|train/ufo/aux_rew/penalty_action_rate|0.532179|0.577871|0.740365|0.544311|
|train/ufo/aux_rew/penalty_ankle_roll|0.0136426|0.0293053|0.0462745|0.0463613|
|train/ufo/aux_rew/penalty_feet_ori|0.0631424|0.0947675|0.118753|0.118035|
|train/ufo/aux_rew/penalty_slippage|0.235466|0.270776|0.306699|0.257706|
|train/ufo/aux_rew/penalty_torques|4015.64|4349.82|4878.58|4259.63|
|train/ufo/aux_rew/penalty_undesired_contact|0|0|0|0|
|train/ufo/backward_optimizer/grad_clip_limit|5|5|5|5|
|train/ufo/backward_optimizer/grad_norm_after_clip_l2|5|5|5|5|
|train/ufo/backward_optimizer/grad_norm_l2|19611.2|46718.4|343409|20958|
|train/ufo/backward_optimizer/lr|1e-05|1e-05|1e-05|1e-05|
|train/ufo/backward_optimizer/parameter_delta_l2|0.00340622|0.0043007|0.00443972|0.0037591|
|train/ufo/collected_transitions|24576|24576|24576|24576|
|train/ufo/contact_termination_count|0|0|0|0|
|train/ufo/critic_loss|0.247535|0.341202|2.34946|0.412285|
|train/ufo/critic_optimizer/grad_clip_limit|5|5|5|5|
|train/ufo/critic_optimizer/grad_norm_after_clip_l2|1.36803|5|5|4.73489|
|train/ufo/critic_optimizer/grad_norm_l2|1.36803|8.99278|19.4827|10.2229|
|train/ufo/critic_optimizer/lr|0.0003|0.0003|0.0003|0.0003|
|train/ufo/critic_optimizer/parameter_delta_l2|0.231857|0.330395|0.447198|0.386458|
|train/ufo/disc_expert_loss|0.284379|0.387763|0.555868|0.321854|
|train/ufo/disc_loss|0.971032|1.22249|2.88606|1.03016|
|train/ufo/disc_reward_clip_fraction|0|0.0310059|0.0781494|0.0787695|
|train/ufo/disc_train_loss|0.336044|0.473693|0.737027|0.368477|
|train/ufo/disc_wgan_gp_loss|0.0315364|0.0365398|0.159995|0.0339828|
|train/ufo/discriminator_optimizer/grad_clip_limit|5|5|5|5|
|train/ufo/discriminator_optimizer/grad_norm_after_clip_l2|0.976767|1.48848|5|1.07654|
|train/ufo/discriminator_optimizer/grad_norm_l2|0.976767|1.48848|8.1095|1.07654|
|train/ufo/discriminator_optimizer/lr|3e-06|3e-06|3e-06|3e-06|
|train/ufo/discriminator_optimizer/parameter_delta_l2|0.0026095|0.00303932|0.00533472|0.00265304|
|train/ufo/environment_control_steps|16|78|133.8|78|
|train/ufo/expert_draws/g1|41104|295080|523470|294789|
|train/ufo/expert_draws/official_rpo|40816|294744|523467|294267|
|train/ufo/fb_diag|-236.631|-165.889|-21.1465|-223.982|
|train/ufo/fb_loss|26896.1|50421.1|196142|28612|
|train/ufo/fb_offdiag|74.3513|134.868|196.968|196.557|
|train/ufo/first_step_samples|2480|4431.5|8120.55|4870.69|
|train/ufo/first_step_terminated|21|117.5|464.35|171.188|
|train/ufo/foot_contact_peak_N|2888.61|3637.83|4686.85|3733.7|
|train/ufo/forward_optimizer/grad_clip_limit|5|5|5|5|
|train/ufo/forward_optimizer/grad_norm_after_clip_l2|5|5|5|5|
|train/ufo/forward_optimizer/grad_norm_l2|87.3903|202.928|1221.02|105.568|
|train/ufo/forward_optimizer/lr|0.0003|0.0003|0.0003|0.0003|
|train/ufo/forward_optimizer/parameter_delta_l2|0.203585|0.696376|0.936988|0.916275|
|train/ufo/hard_position_count|2179|3970.5|7258.1|4323.22|
|train/ufo/hard_torque_count|0|0|0|0|
|train/ufo/hard_velocity_count|1034|1654.5|2906|1791.81|
|train/ufo/height_termination_count|0|9|49.9|15.2812|
|train/ufo/latched_velocity_terms_without_boundary_violation|616|986|1875.3|1086.78|
|train/ufo/learner_joint_updates|25|274.5|499.05|499.5|
|train/ufo/mean_aux_reward|-0.822286|-0.74274|-0.632155|-0.780777|
|train/ufo/mean_disc_reward|-0.736457|-0.300099|0.0836261|-0.677278|
|train/ufo/mean_disc_reward_raw|-0.787958|-0.314963|0.0836261|-0.723302|
|train/ufo/mean_next_Q|-2.67738|-1.7349|-1.42187|-2.20435|
|train/ufo/mean_next_auxQ|-3.69345|-3.15752|-2.73078|-3.52591|
|train/ufo/online_transitions|98304|466944|811008|815923|
|train/ufo/orientation_termination_count|1|84.5|251.75|108.094|
|train/ufo/orth_loss|269.146|504.554|1960.74|286.395|
|train/ufo/orth_loss_diag|-256|-256|-256|-256|
|train/ufo/orth_loss_offdiag|525.146|760.554|2216.74|542.395|
|train/ufo/physics_peak_joint_velocity|19.7657|25.1049|30.0551|25.4235|
|train/ufo/physics_peak_normalized_velocity|2.44595|3.13811|3.75688|3.14606|
|train/ufo/q_loss|0|0|0|0|
|train/ufo/replay_draw_new_ratio|0.833333|1.2307|1.27169|1.18785|
|train/ufo/replay_draws|81920|589824|1.04694e+06|589056|
|train/ufo/replay_policy_age_max_updates|40|288|511.2|287.625|
|train/ufo/replay_policy_age_min_updates|4|16|16|15.625|
|train/ufo/replay_retained_fraction|1|1|1|1|
|train/ufo/replay_size|98304|479232|822067|479232|
|train/ufo/requested_action_abs_mean|0.79199|0.885465|0.922029|0.875467|
|train/ufo/requested_action_saturation_fraction|0.274598|0.412788|0.470252|0.397377|
|train/ufo/requested_applied_max_diff|1.92627|1.9528|1.9528|1.95079|
|train/ufo/reset_control_samples|2480|4431.5|8120.55|4870.69|
|train/ufo/reset_projection_outside_fraction|0|0|0|0|
|train/ufo/reset_projection_residual_max_rad|1.78814e-07|1.78814e-07|1.78814e-07|1.78814e-07|
|train/ufo/reset_projection_residual_mean_rad|6.32898e-08|6.39585e-08|6.47206e-08|6.40736e-08|
|train/ufo/reset_projection_residual_p95_rad|1.19209e-07|1.19209e-07|1.19209e-07|1.19209e-07|
|train/ufo/reset_root_lift_max_m|0.0219456|0.021992|0.021992|0.0219761|
|train/ufo/sampling_rounds|4|19|33|33.2|
|train/ufo/strict_shadow_episode_len|2.4331|5.09826|7.86659|5.08817|
|train/ufo/strict_shadow_mixed_episode_len|2.42334|4.15783|6.28391|4.32347|
|train/ufo/strict_shadow_tracking_episode_len|2.44267|6.2639|10.4235|6.21325|
|train/ufo/strict_velocity_exposure_fraction|0.000743921|0.00122015|0.00170393|0.00124434|
|train/ufo/strict_velocity_transition_fraction|0.0420736|0.0673218|0.118245|0.072909|
|train/ufo/target_M|0.251992|0.550293|2.81529|0.585607|
|train/ufo/target_Q|-4.09301|-2.09827|-1.65721|-2.645|
|train/ufo/target_auxQ|-4.19197|-3.39744|-3.04025|-3.79215|
|train/ufo/terminated_transitions|2512|4368|7540.35|4698.09|
|train/ufo/tracking_episode_starts|951|1807.5|4244.05|2173.53|
|train/ufo/tracking_q_rmse_rad|0.0417452|0.123506|0.18585|0.122864|
|train/ufo/tracking_unique_env_fraction|0.5|0.5|0.5|0.5|
|train/ufo/tracking_windows_completed|0|0|0|0|
|train/ufo/training_velocity_count|223|434|700.75|447.438|
|train/ufo/truncated_transitions|0|0|0|0|
|train/ufo/unc_Q|0.166164|0.462348|4.28487|0.180833|
|train/ufo/unc_auxQ|0.151447|0.273035|0.922673|0.158047|
|train/ufo/velocity_reward_coverage_miss_count|0|0|0|0|
|train/ufo/wall_time_s|54.12|547.134|993.117|997.669|
|train/ufo/z_norm|16|16|16|16|

## 全部真实采样轮的关键指标

|真实更新|env/episode_len|train/ufo/strict_shadow_episode_len|train/ufo/physics_peak_normalized_velocity|train/ufo/first_step_terminated|train/ufo/first_step_samples|train/ufo/hard_position_count|train/ufo/training_velocity_count|
|---|---:|---:|---:|---:|---:|---:|---:|
|40|2.54764|2.4331|2.47071|1028|11506|6962|518|
|56|3.09625|2.87423|3.7956|488|7394|7620|731|
|72|2.78326|2.6388|2.76911|445|8846|8499|818|
|88|3.10244|2.88843|3.57466|361|7527|6936|594|
|104|3.62197|3.26557|3.51153|253|6200|5237|676|
|120|4.27825|3.8002|2.79244|288|5879|4960|465|
|136|4.29002|3.97773|3.08457|249|5681|5572|463|
|152|3.7838|3.52529|2.9242|281|6952|6699|476|
|168|3.73384|3.52856|3.23309|162|6315|5759|533|
|184|4.07944|3.81839|4.47197|261|5774|5330|595|
|200|4.37188|4.04302|2.90141|143|5417|4983|508|
|216|4.55255|4.14166|3.19165|160|5371|4911|418|
|232|4.69536|4.36365|3.69011|129|5299|4993|418|
|248|4.64058|4.27872|3.0102|169|5067|4526|492|
|264|5.06944|4.65981|3.42407|143|4479|4027|440|
|280|5.64589|5.21715|3.27595|85|4457|3914|428|
|296|5.6734|5.18258|3.72521|74|4303|3822|553|
|312|5.54098|5.05412|2.98296|135|4406|4065|509|
|328|5.74085|5.1424|2.98147|106|4140|3627|396|
|344|6.1172|5.49507|3.61551|60|3733|3254|425|
|360|6.53074|5.88315|3.28138|74|3788|3313|454|
|376|6.7681|5.98644|2.77471|41|3693|3343|373|
|392|6.51742|5.84966|3.35268|73|3595|3246|407|
|408|6.94935|6.19889|2.86149|45|3404|2832|395|
|424|7.1501|6.29019|3.22234|36|3106|2750|331|
|440|7.54182|6.68829|2.62655|28|3036|2598|353|
|456|8.3021|7.43383|3.31515|35|3074|2909|286|
|472|8.04485|7.01121|2.44595|21|3107|2595|295|
|488|8.00376|7.0459|2.59258|34|2706|2356|265|
|504|8.87322|7.7912|2.80587|25|2480|2272|223|
|520|9.15212|7.95873|2.64426|24|2620|2254|237|
|524|9.53861|8.35549|3.3247|22|2507|2179|243|

## 786432条transition、replay与控制时序逐行核验

```json
{
  "status": "passed",
  "rows": 786432,
  "rounds": 32,
  "round_first": 4,
  "round_last": 35,
  "replay_count": 860160,
  "replay_capacity": 4000000,
  "fifo_evictions": 0,
  "requested_saturation_fraction": 0.39737723530202673,
  "applied_saturation_fraction": 0.09212521539218184,
  "strict_position_velocity_torque_counts": [
    138343,
    57338,
    0
  ],
  "training_position_velocity_torque_counts": [
    138343,
    14318,
    0
  ],
  "macro_contact_height_tilt_counts": [
    0,
    489,
    3459
  ],
  "boundary_max_radps": 28.924074172973633,
  "physics_max_radps": 35.77573013305664,
  "physics_max_rated_ratio": 4.47196626663208,
  "boundary_max_event": {
    "round": 23,
    "row": 23459,
    "episode_id_run_local": 2258642843671,
    "episode_step": 6,
    "joint": "right_ankle_roll_joint",
    "rated_radps": 8.0,
    "boundary_dq_radps": 28.924074172973633,
    "physics_signed_dq_radps": 28.924074172973633,
    "physics_abs_dq_radps": 28.924074172973633,
    "peak_substep_0_to_7": 7,
    "q_at_joint_peak_rad": 0.3769920766353607,
    "torque_at_joint_peak_Nm": -27.0,
    "target_at_joint_peak_rad": 0.23999999463558197,
    "requested_action": -0.9999989867210388,
    "applied_action": 0.47999998927116394,
    "foot_peak_vertical_net_force_N": [
      0.0,
      1983.32275390625
    ],
    "foot_vertical_net_force_integral_Ns": [
      0.0,
      9.635289192199707
    ],
    "global_peak_joint": "right_ankle_roll_joint",
    "feet_at_global_abs_dq_peak_origin_pose_wxyz_linear_angular": [
      [
        7.8445587158203125,
        22.822845458984375,
        0.11897310614585876,
        0.46153637766838074,
        0.35300585627555847,
        0.023442715406417847,
        0.813524067401886,
        -0.003919124603271484,
        1.2096716165542603,
        0.6985129117965698,
        1.2694498300552368,
        4.60024356842041,
        -1.3607075214385986
      ],
      [
        8.71145248413086,
        22.689632415771484,
        0.047463104128837585,
        0.5204658508300781,
        -0.036120351403951645,
        -0.057796720415353775,
        0.8511582016944885,
        2.238025665283203,
        0.49048757553100586,
        -0.5398216247558594,
        -13.891681671142578,
        21.82993507385254,
        1.3894774913787842
      ]
    ],
    "feet_vertical_net_force_at_global_peak_N": [
      0.0,
      1870.79296875
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
    "round": 13,
    "row": 9439,
    "episode_id_run_local": 2263944443914,
    "episode_step": 5,
    "joint": "right_ankle_pitch_joint",
    "rated_radps": 8.0,
    "boundary_dq_radps": 16.67308807373047,
    "physics_signed_dq_radps": 35.77573013305664,
    "physics_abs_dq_radps": 35.77573013305664,
    "peak_substep_0_to_7": 4,
    "q_at_joint_peak_rad": 0.3985496759414673,
    "torque_at_joint_peak_Nm": -27.0,
    "target_at_joint_peak_rad": 0.3307030498981476,
    "requested_action": 0.9303691387176514,
    "applied_action": 0.5511717200279236,
    "foot_peak_vertical_net_force_N": [
      0.0,
      1320.1539306640625
    ],
    "foot_vertical_net_force_integral_Ns": [
      0.0,
      8.615104675292969
    ],
    "global_peak_joint": "right_ankle_pitch_joint",
    "feet_at_global_abs_dq_peak_origin_pose_wxyz_linear_angular": [
      [
        -9.742863655090332,
        27.652971267700195,
        0.16773787140846252,
        0.6109530329704285,
        -0.27431946992874146,
        0.4576384723186493,
        0.584852397441864,
        2.9726219177246094,
        -2.7894649505615234,
        2.042330741882324,
        -12.199236869812012,
        5.582192420959473,
        16.177011489868164
      ],
      [
        -9.035820960998535,
        27.84461784362793,
        0.06368209421634674,
        0.9643663763999939,
        0.17318566143512726,
        -0.18157343566417694,
        0.08387727290391922,
        2.5813732147216797,
        0.36068546772003174,
        -0.4839057922363281,
        -3.0381569862365723,
        32.499107360839844,
        0.8433094024658203
      ]
    ],
    "feet_vertical_net_force_at_global_peak_N": [
      0.0,
      499.4773864746094
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
  "normalized_max_event": {
    "round": 13,
    "row": 9439,
    "episode_id_run_local": 2263944443914,
    "episode_step": 5,
    "joint": "right_ankle_pitch_joint",
    "rated_radps": 8.0,
    "boundary_dq_radps": 16.67308807373047,
    "physics_signed_dq_radps": 35.77573013305664,
    "physics_abs_dq_radps": 35.77573013305664,
    "peak_substep_0_to_7": 4,
    "q_at_joint_peak_rad": 0.3985496759414673,
    "torque_at_joint_peak_Nm": -27.0,
    "target_at_joint_peak_rad": 0.3307030498981476,
    "requested_action": 0.9303691387176514,
    "applied_action": 0.5511717200279236,
    "foot_peak_vertical_net_force_N": [
      0.0,
      1320.1539306640625
    ],
    "foot_vertical_net_force_integral_Ns": [
      0.0,
      8.615104675292969
    ],
    "global_peak_joint": "right_ankle_pitch_joint",
    "feet_at_global_abs_dq_peak_origin_pose_wxyz_linear_angular": [
      [
        -9.742863655090332,
        27.652971267700195,
        0.16773787140846252,
        0.6109530329704285,
        -0.27431946992874146,
        0.4576384723186493,
        0.584852397441864,
        2.9726219177246094,
        -2.7894649505615234,
        2.042330741882324,
        -12.199236869812012,
        5.582192420959473,
        16.177011489868164
      ],
      [
        -9.035820960998535,
        27.84461784362793,
        0.06368209421634674,
        0.9643663763999939,
        0.17318566143512726,
        -0.18157343566417694,
        0.08387727290391922,
        2.5813732147216797,
        0.36068546772003174,
        -0.4839057922363281,
        -3.0381569862365723,
        32.499107360839844,
        0.8433094024658203
      ]
    ],
    "feet_vertical_net_force_at_global_peak_N": [
      0.0,
      499.4773864746094
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
  "foot_vertical_net_force_peak_N": 5212.72900390625,
  "first_step_foot_vertical_net_force_N": {
    "n": 155862,
    "mean": 54.10807800292969,
    "median": 0.0,
    "p95": 359.413195800781,
    "p99": 608.5360693359371,
    "max": 1985.40087890625
  },
  "first_step_foot_absolute_vertical_net_force_integral_Ns": {
    "n": 155862,
    "mean": 0.3853159546852112,
    "median": 0.0,
    "p95": 2.888686168193811,
    "p99": 4.725857677459706,
    "max": 11.91213607788086
  },
  "reset_lift_m": {
    "n": 155862,
    "mean": 0.001417264575138688,
    "median": 0.0,
    "p95": 0.013961445540189743,
    "p99": 0.019589802250266075,
    "max": 0.021992022171616554
  },
  "resets": 155862,
  "first_step_terminated": 5478,
  "first_step_termination_fraction": 0.03514647572852908,
  "first_step_strict_counts": [
    5478,
    28,
    0
  ],
  "first_step_macro_counts": [
    0,
    0,
    0
  ],
  "raw_velocity_penalty_mean": 0.011124609833738456,
  "weighted_velocity_penalty_mean": -0.2781152458434614,
  "strict_velocity_substep_joint_exposure_fraction": 0.0012443445730900419,
  "velocity_reward_coverage_miss_count": 0,
  "latched_velocity_terms_without_boundary_violation": 34777,
  "reset_q_preservation_max_error_rad": 8.821487429422348e-08,
  "reset_dq_preservation_max_error_radps": 0.0,
  "decoder_max_error_rad": 2.1934509275567393e-07,
  "infinite_plane_trace_margin_sentinel_m": 1000000.0,
  "ground_outside_count": 0,
  "mixed_nonperiodic_z_equal_pairs": 306907,
  "mixed_nonperiodic_cosine": {
    "n": 306907,
    "mean": 1.0,
    "median": 1.0,
    "p95": 1.0000001192092896,
    "p99": 1.0000001192092896,
    "max": 1.0000001192092896
  },
  "tracking_unexpected_refreshes": 0,
  "tracking_phase_mismatches": 0,
  "tracking_window_stable_pairs": 323663,
  "tracking_adjacent_phase_cosine": {
    "n": 323663,
    "mean": 0.9974987506866455,
    "median": 0.9979077577590942,
    "p95": 0.9999899864196777,
    "p99": 0.9999957084655762,
    "max": 0.9999988079071045
  },
  "tracking_window_renewals": 0,
  "terminal_following_new_episode_pairs": 149718,
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
    "n": 150339,
    "mean": 5.002913415680562,
    "median": 4.0,
    "p95": 11.0,
    "p99": 15.0,
    "max": 32.0
  },
  "completed_strict_shadow_len": {
    "n": 150339,
    "mean": 4.54206825906784,
    "median": 4.0,
    "p95": 10.0,
    "p99": 14.0,
    "max": 32.0
  },
  "completed_tracking_shadow_len": {
    "n": 66730,
    "mean": 5.188865577701184,
    "median": 4.0,
    "p95": 12.0,
    "p99": 16.0,
    "max": 32.0
  },
  "completed_mixed_shadow_len": {
    "n": 83609,
    "mean": 4.025846499778732,
    "median": 3.0,
    "p95": 9.0,
    "p99": 12.0,
    "max": 28.0
  },
  "unfinished_train_episodes_right_censored": 5523,
  "unfinished_strict_shadow_right_censored": 4725,
  "unfinished_train_observed_lengths": {
    "n": 5523,
    "mean": 6.2102118413905485,
    "median": 5.0,
    "p95": 15.0,
    "p99": 20.0,
    "max": 32.0
  },
  "RSI_source_by_rollout_mode": {
    "official_rpo/mixed": {
      "rows": 203379,
      "resets": 42763,
      "first_terms": 595,
      "strict_velocity_rows": 17869,
      "completed_episode_len": {
        "n": 41376,
        "mean": 4.750749226604795,
        "median": 4.0,
        "p95": 10.0,
        "p99": 12.0,
        "max": 24.0
      },
      "completed_strict_shadow_len": {
        "n": 41376,
        "mean": 4.295291956689868,
        "median": 4.0,
        "p95": 9.0,
        "p99": 11.0,
        "max": 24.0
      },
      "first_term_fraction": 0.013913897528237027,
      "strict_velocity_substep_joint_exposure_fraction": 0.0015908401470204375,
      "raw_velocity_penalty_mean": 0.014848951038368868
    },
    "official_rpo/tracking": {
      "rows": 203589,
      "resets": 34763,
      "first_terms": 1185,
      "strict_velocity_rows": 8422,
      "completed_episode_len": {
        "n": 33216,
        "mean": 5.765414258188825,
        "median": 5.0,
        "p95": 12.0,
        "p99": 16.0,
        "max": 31.0
      },
      "completed_strict_shadow_len": {
        "n": 33216,
        "mean": 5.480581647398844,
        "median": 5.0,
        "p95": 11.0,
        "p99": 16.0,
        "max": 27.0
      },
      "first_term_fraction": 0.034087967091447804,
      "strict_velocity_substep_joint_exposure_fraction": 0.0006490057654520073,
      "raw_velocity_penalty_mean": 0.005675604209103382
    },
    "g1/mixed": {
      "rows": 189837,
      "resets": 43546,
      "first_terms": 2625,
      "strict_velocity_rows": 18730,
      "completed_episode_len": {
        "n": 42233,
        "mean": 4.334122605545427,
        "median": 4.0,
        "p95": 9.0,
        "p99": 13.0,
        "max": 29.0
      },
      "completed_strict_shadow_len": {
        "n": 42233,
        "mean": 3.761868680889352,
        "median": 3.0,
        "p95": 8.0,
        "p99": 12.0,
        "max": 28.0
      },
      "first_term_fraction": 0.06028108207412851,
      "strict_velocity_substep_joint_exposure_fraction": 0.0016318633537100822,
      "raw_velocity_penalty_mean": 0.013680819927256107
    },
    "g1/tracking": {
      "rows": 189627,
      "resets": 34790,
      "first_terms": 1073,
      "strict_velocity_rows": 12317,
      "completed_episode_len": {
        "n": 33514,
        "mean": 5.401294981201886,
        "median": 4.0,
        "p95": 12.0,
        "p99": 16.0,
        "max": 32.0
      },
      "completed_strict_shadow_len": {
        "n": 33514,
        "mean": 4.8997433908217465,
        "median": 4.0,
        "p95": 12.0,
        "p99": 16.0,
        "max": 32.0
      },
      "first_term_fraction": 0.030842196033342914,
      "strict_velocity_substep_joint_exposure_fraction": 0.0011239456131384702,
      "raw_velocity_penalty_mean": 0.010421342445433121
    }
  },
  "per_joint_physical_diagnostics": {
    "left_thigh_yaw_joint": {
      "rated_radps": 25.0,
      "max_physics_abs_radps": 23.406757354736328,
      "max_boundary_abs_radps": 22.513578414916992,
      "max_rated_ratio": 0.9362702941894532,
      "substep_exposure_fraction": 0.0
    },
    "right_thigh_yaw_joint": {
      "rated_radps": 25.0,
      "max_physics_abs_radps": 22.63115119934082,
      "max_boundary_abs_radps": 20.910018920898438,
      "max_rated_ratio": 0.9052460479736328,
      "substep_exposure_fraction": 0.0
    },
    "torso_joint": {
      "rated_radps": 25.0,
      "max_physics_abs_radps": 19.7150821685791,
      "max_boundary_abs_radps": 17.24617576599121,
      "max_rated_ratio": 0.788603286743164,
      "substep_exposure_fraction": 0.0
    },
    "left_thigh_roll_joint": {
      "rated_radps": 25.0,
      "max_physics_abs_radps": 23.17833709716797,
      "max_boundary_abs_radps": 23.17833709716797,
      "max_rated_ratio": 0.9271334838867188,
      "substep_exposure_fraction": 0.0
    },
    "right_thigh_roll_joint": {
      "rated_radps": 25.0,
      "max_physics_abs_radps": 24.18840217590332,
      "max_boundary_abs_radps": 24.053241729736328,
      "max_rated_ratio": 0.9675360870361328,
      "substep_exposure_fraction": 0.0
    },
    "left_arm_pitch_joint": {
      "rated_radps": 8.0,
      "max_physics_abs_radps": 12.564691543579102,
      "max_boundary_abs_radps": 12.437161445617676,
      "max_rated_ratio": 1.5705864429473877,
      "substep_exposure_fraction": 0.0024460156758626304
    },
    "right_arm_pitch_joint": {
      "rated_radps": 8.0,
      "max_physics_abs_radps": 13.502949714660645,
      "max_boundary_abs_radps": 13.502949714660645,
      "max_rated_ratio": 1.6878687143325806,
      "substep_exposure_fraction": 0.0024677912394205728
    },
    "left_thigh_pitch_joint": {
      "rated_radps": 25.0,
      "max_physics_abs_radps": 25.622941970825195,
      "max_boundary_abs_radps": 25.622941970825195,
      "max_rated_ratio": 1.0249176788330079,
      "substep_exposure_fraction": 3.178914388020833e-07
    },
    "right_thigh_pitch_joint": {
      "rated_radps": 25.0,
      "max_physics_abs_radps": 23.918909072875977,
      "max_boundary_abs_radps": 23.294536590576172,
      "max_rated_ratio": 0.9567563629150391,
      "substep_exposure_fraction": 0.0
    },
    "left_arm_roll_joint": {
      "rated_radps": 8.0,
      "max_physics_abs_radps": 13.150837898254395,
      "max_boundary_abs_radps": 13.150837898254395,
      "max_rated_ratio": 1.6438547372817993,
      "substep_exposure_fraction": 0.0004380544026692708
    },
    "right_arm_roll_joint": {
      "rated_radps": 8.0,
      "max_physics_abs_radps": 13.221084594726562,
      "max_boundary_abs_radps": 13.221084594726562,
      "max_rated_ratio": 1.6526355743408203,
      "substep_exposure_fraction": 0.0006200472513834635
    },
    "left_knee_joint": {
      "rated_radps": 25.0,
      "max_physics_abs_radps": 19.373008728027344,
      "max_boundary_abs_radps": 19.012571334838867,
      "max_rated_ratio": 0.7749203491210938,
      "substep_exposure_fraction": 0.0
    },
    "right_knee_joint": {
      "rated_radps": 25.0,
      "max_physics_abs_radps": 20.12640953063965,
      "max_boundary_abs_radps": 19.769914627075195,
      "max_rated_ratio": 0.805056381225586,
      "substep_exposure_fraction": 0.0
    },
    "left_arm_yaw_joint": {
      "rated_radps": 8.0,
      "max_physics_abs_radps": 6.521249771118164,
      "max_boundary_abs_radps": 6.078492164611816,
      "max_rated_ratio": 0.8151562213897705,
      "substep_exposure_fraction": 0.0
    },
    "right_arm_yaw_joint": {
      "rated_radps": 8.0,
      "max_physics_abs_radps": 6.043422698974609,
      "max_boundary_abs_radps": 5.529430866241455,
      "max_rated_ratio": 0.7554278373718262,
      "substep_exposure_fraction": 0.0
    },
    "left_ankle_pitch_joint": {
      "rated_radps": 8.0,
      "max_physics_abs_radps": 28.8846435546875,
      "max_boundary_abs_radps": 28.102872848510742,
      "max_rated_ratio": 3.6105804443359375,
      "substep_exposure_fraction": 0.009197870890299479
    },
    "right_ankle_pitch_joint": {
      "rated_radps": 8.0,
      "max_physics_abs_radps": 35.77573013305664,
      "max_boundary_abs_radps": 28.597278594970703,
      "max_rated_ratio": 4.47196626663208,
      "substep_exposure_fraction": 0.011243025461832682
    },
    "left_elbow_pitch_joint": {
      "rated_radps": 8.0,
      "max_physics_abs_radps": 8.622459411621094,
      "max_boundary_abs_radps": 7.134142875671387,
      "max_rated_ratio": 1.0778074264526367,
      "substep_exposure_fraction": 6.357828776041666e-07
    },
    "right_elbow_pitch_joint": {
      "rated_radps": 8.0,
      "max_physics_abs_radps": 8.591360092163086,
      "max_boundary_abs_radps": 8.118785858154297,
      "max_rated_ratio": 1.0739200115203857,
      "substep_exposure_fraction": 9.5367431640625e-07
    },
    "left_ankle_roll_joint": {
      "rated_radps": 8.0,
      "max_physics_abs_radps": 25.7786922454834,
      "max_boundary_abs_radps": 24.701616287231445,
      "max_rated_ratio": 3.222336530685425,
      "substep_exposure_fraction": 0.000865777333577474
    },
    "right_ankle_roll_joint": {
      "rated_radps": 8.0,
      "max_physics_abs_radps": 29.801660537719727,
      "max_boundary_abs_radps": 28.924074172973633,
      "max_rated_ratio": 3.725207567214966,
      "substep_exposure_fraction": 0.0013394355773925781
    },
    "left_elbow_yaw_joint": {
      "rated_radps": 8.0,
      "max_physics_abs_radps": 4.242692947387695,
      "max_boundary_abs_radps": 3.7940139770507812,
      "max_rated_ratio": 0.5303366184234619,
      "substep_exposure_fraction": 0.0
    },
    "right_elbow_yaw_joint": {
      "rated_radps": 8.0,
      "max_physics_abs_radps": 4.270328044891357,
      "max_boundary_abs_radps": 3.8076581954956055,
      "max_rated_ratio": 0.5337910056114197,
      "substep_exposure_fraction": 0.0
    }
  },
  "right_ankle_pitch_physics_peak_event": {
    "round": 13,
    "row": 9439,
    "episode_id_run_local": 2263944443914,
    "episode_step": 5,
    "joint": "right_ankle_pitch_joint",
    "rated_radps": 8.0,
    "boundary_dq_radps": 16.67308807373047,
    "physics_signed_dq_radps": 35.77573013305664,
    "physics_abs_dq_radps": 35.77573013305664,
    "peak_substep_0_to_7": 4,
    "q_at_joint_peak_rad": 0.3985496759414673,
    "torque_at_joint_peak_Nm": -27.0,
    "target_at_joint_peak_rad": 0.3307030498981476,
    "requested_action": 0.9303691387176514,
    "applied_action": 0.5511717200279236,
    "foot_peak_vertical_net_force_N": [
      0.0,
      1320.1539306640625
    ],
    "foot_vertical_net_force_integral_Ns": [
      0.0,
      8.615104675292969
    ],
    "global_peak_joint": "right_ankle_pitch_joint",
    "feet_at_global_abs_dq_peak_origin_pose_wxyz_linear_angular": [
      [
        -9.742863655090332,
        27.652971267700195,
        0.16773787140846252,
        0.6109530329704285,
        -0.27431946992874146,
        0.4576384723186493,
        0.584852397441864,
        2.9726219177246094,
        -2.7894649505615234,
        2.042330741882324,
        -12.199236869812012,
        5.582192420959473,
        16.177011489868164
      ],
      [
        -9.035820960998535,
        27.84461784362793,
        0.06368209421634674,
        0.9643663763999939,
        0.17318566143512726,
        -0.18157343566417694,
        0.08387727290391922,
        2.5813732147216797,
        0.36068546772003174,
        -0.4839057922363281,
        -3.0381569862365723,
        32.499107360839844,
        0.8433094024658203
      ]
    ],
    "feet_vertical_net_force_at_global_peak_N": [
      0.0,
      499.4773864746094
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
    "round": 20,
    "row": 19958,
    "episode_id_run_local": 67602031633,
    "episode_step": 7,
    "joint": "right_ankle_roll_joint",
    "rated_radps": 8.0,
    "boundary_dq_radps": 6.493529796600342,
    "physics_signed_dq_radps": 29.801660537719727,
    "physics_abs_dq_radps": 29.801660537719727,
    "peak_substep_0_to_7": 3,
    "q_at_joint_peak_rad": 0.49292176961898804,
    "torque_at_joint_peak_Nm": -27.0,
    "target_at_joint_peak_rad": 0.4935023784637451,
    "requested_action": 0.9870047569274902,
    "applied_action": 0.9870047569274902,
    "foot_peak_vertical_net_force_N": [
      0.0,
      2918.458984375
    ],
    "foot_vertical_net_force_integral_Ns": [
      0.0,
      30.079063415527344
    ],
    "global_peak_joint": "right_ankle_roll_joint",
    "feet_at_global_abs_dq_peak_origin_pose_wxyz_linear_angular": [
      [
        -26.604753494262695,
        47.255855560302734,
        0.1690341681241989,
        0.4441048204898834,
        0.11382746696472168,
        0.44071879982948303,
        0.7717390656471252,
        -3.1599977016448975,
        0.35649651288986206,
        -0.03691530227661133,
        6.089191436767578,
        12.401277542114258,
        3.949326276779175
      ],
      [
        -25.81160545349121,
        47.47355270385742,
        0.04288575053215027,
        0.9196546673774719,
        0.01240491308271885,
        0.003913640044629574,
        0.39251261949539185,
        1.5292768478393555,
        -1.504908561706543,
        -0.5277409553527832,
        19.823402404785156,
        19.338138580322266,
        -4.03154993057251
      ]
    ],
    "feet_vertical_net_force_at_global_peak_N": [
      0.0,
      2220.57666015625
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
  "semantics": "Run-local episode IDs; original physical thresholds shadow; interval mean of exactly 8 completed substeps; vertical NET force not contact point normal impulse; unfinished trajectories are right-censored, excluded from completed means"
}
```

原生episode_id含seed×1e9初始偏移；learner还加入rank×2^40和env×2^24。逐行核验已扣除seed偏移解码rank/env，未套用旧Isaac的小整数计数假设。竖直净力包含该foot body全部接触合力；积分为8个已完成子步的绝对竖直净力积分，不等于某个接触点的法向冲量。自碰撞保持关闭，foot/ground与关节限位约束仍存在；本轮不能将收益单独归因于求解器或文件格式。

## 实际资源、预算与证据保全

```json
[
  {
    "stage": 16,
    "started_at": "2026-10-12T14:43:50.089513+00:00",
    "finished_at": "2026-10-12T14:49:17.349933+00:00",
    "sampled_GPU_peak_used_MiB": {
      "0": 16726,
      "1": 4539,
      "2": 4539,
      "3": 4539
    },
    "checkpoint_bytes": 8074904276
  },
  {
    "stage": 24,
    "started_at": "2026-10-12T14:51:05.783752+00:00",
    "finished_at": "2026-10-12T14:56:31.432923+00:00",
    "sampled_GPU_peak_used_MiB": {
      "0": 16710,
      "1": 4535,
      "2": 4535,
      "3": 4535
    },
    "checkpoint_bytes": 8247972840
  },
  {
    "stage": 524,
    "started_at": "2026-10-12T15:01:10.648475+00:00",
    "finished_at": "2026-10-12T15:23:33.430860+00:00",
    "sampled_GPU_peak_used_MiB": {
      "0": 16710,
      "1": 4535,
      "2": 4537,
      "3": 4537
    },
    "checkpoint_bytes": 13786027024
  }
]
```

显存为Doctor约30秒采样的整卡已用MiB观测峰值，包含CUDA/Warp；并非连续真实峰值。Torch probe峰值不含Warp，不能直接比较。CPU请求24、RAM96GB、shm32GB；本轮未额外采集连续RSS/CPU利用率/磁盘吞吐，未补造数值。原始TensorBoard、两类trace、源凭证、模型和checkpoint由平台保全；未删除原有资产或模型，无Git提交。

## 三轮最终复核与限制

第一轮：原生MJCF与冻结接口范围、375/733布局、原始惯量/几何、153帧坐标系与train-only数据；第二轮：实际400Hz、三卡各2048/seed42–44、六Adam16→24→524、哈希/指纹/恢复与预算；第三轮：500更新连续、所有有限标量、32采样轮与786432trace/replay逐行对账、删失与分组、资源释放/产物完整性和训练结论。
容器控制面全量555 passed/30 skipped；后续平台40 passed、原生相关44 passed、最终8契约通过。跳过不作为通过，未声称全仓lint或独立评估通过。旧physics_hz=200元数据仍是历史问题；当前Profile与实测400Hz一致。部分reset全局forward对solver warmstart/其他调用者诊断的影响未单独验证。
