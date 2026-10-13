# UFO/RPO 原生模型对齐与四卡优化：隔离分析摘要

本文件包含本轮实现、完整配置、全部指标统计和逐例结果，可独立交给其他模型分析。详细历史与全部逐点指标另在[完整记录](/data/kk/EmbodiedBein/docs/progress/2026-10-11-ufo-native-alignment.md)内嵌保存；文档引用均为绝对路径。JSON/YAML中的相对成员名属于原始checkpoint/receipt协议字段，不是文档引用，保留原值以维持证据真实性。没有提交代码。

## 当前结论

- 对接RoboParty RPO/RPO-v2的23DoF IsaacLab仿真本体，接口ee_rpo_v2_bfm_23dof_v1；没有真机放行或真机训练。
- C/D已实际对齐固定UFO原生完整模型：F/actor/critic/aux critic 2048×6，discriminator1024×3，B256×1，latent256。六Adam拥有435,116,484参数，原精简模型57,392,580，约7.58倍。不能把“原生对齐”解释为所有网络都2048×6。
- D完成16→24资格后，从完整24状态继续四卡训练到1024，新增1000联合更新、64,512在线转移；独立验证仍0/16，中位存活35步/0.70秒，比D24的7步提高。正常行走尚未通过，新100例acceptance未运行。
- E在D上增加固定实际速度软惩罚，通过原生既有aux接口消费第九项limits_dof_vel，原八项和网络/损失方法不改。E源码、producer/eval镜像、CPU恢复/导出/发表及平台门禁已经实现与部署，状态dependency_ready；没有E GPU资格或质量改善证据。
- 本轮暂不追加GPU：真实受管产物与近期Docker块计量65.612GiB，另有未测量/未来writable/缓存等8GiB预留；下一D checkpoint另预留8GiB，合计81.612>80；E16+24约13GiB合计86.612>80。主机空闲空间不能替代冻结实验上限；不删pinned模型、不自动扩大预算。

## 对外部分析的取舍与剩余问题

1. Tracking覆盖问题有真实证据，关闭rollout_expert_trajectories合理且已经运行。D1024三个sampler分别23,872/23,865/23,912个继续episode配对，排除原生100步提示边界后被动latent变化均为0，方向余弦中位数1。expert连续帧/判别器学习保持，原生每100步提示更新仍保留。
2. 请求饱和与真实超速属实。共享限速器约束PD target增量0.5×Vmax×0.02，训练replay用requested，last_action/history/action-rate用applied，reset按实测q投影anchor并清历史。D1024全部64,512新转移控制器/PD解码/硬终止校验通过；训练请求/执行饱和45.96%/29.62%，最高dq42.28rad/s，速度硬限位759次，仍未达到建议的饱和<15%。独立验证的执行饱和更高47.40%，不能混用两个分母。
3. native网络与batch1024已对齐，replay容量从10万提高至100万。但D1024实际仅存76,800条，尚未超过旧10万容量，因此不能将此段改善归因于更大的replay。CPU flat transition replay100万、192环境、关闭compile/cudagraphs等仍不同于上游默认trajectory replay512万和完整原生运行方式。
4. 2cm逐帧ground-valid过滤原本已存在；manifest clip最大穿透可能来自已无效帧，不按clip最大值删除整片段。真实512状态接触对照显示保持参考target比立即默认target的首步dq/足部竖向净力更小，但此净力是proxy而非独立normal contact pair impulse。D1024第一控制步终止11/4285=0.257%，硬限位7/4285=0.163%；82.03% reset实际q仍在旧动作target盒外，最大投影残差1.729rad。尚未解决全部初态动力学冲击。
5. FB loss下降和短回合不能单独证明“虚假收敛”的唯一因果。新D训练episode_len有所改善但非稳步单调上升，独立存活变长仍不足20秒。FB/orth优化与物理行走质量需联合判断，不能按loss宣布验收。
6. 新E负成本与原生aux normalizer改变有效尺度；D posthoc第九项平均贡献约−0.051，尾部最大绝对贡献45.55，尚未喂给D或回填replay。强惩罚可能促使少动/早终止，需验证。原生actor同时最大化Q_FB、Q_discriminator和Q_aux，scale_reg按mean(abs(Q_FB)) detach加权；discount=.98，reg_coeff=.05，reg_coeff_aux=.02，D末期Q_FB=83.7768、Q_discriminator=−16.8967、Q_aux=−8.8925。仅凭这些均值不能证明reward hacking，当前不修改原生损失或故意丢失terminated。
7. IsaacLab日志提示enable_external_forces_every_iteration=False及速度迭代设置可能造成噪声；这是待核验的数值动力学假设，没有为提升曲线修改solver或验收阈值。

## 固定训练与评估设置

四卡是一个learner+三个sampler，每sampler64环境，共192；不是四learner DDP。actor/global/micro/native batch均1024，连续expert序列8帧，expert来源official_rpo与g1固定50/50。在线warmup阈值10,240样本后，每新增1024在线转移触发16联合更新，六Adam每次一起计数；每rollout4控制步。actor_std=.05，latent norm16，全部native LR/target/FB/CPR/normalizer设置见下面原生config，无按GPU卡数线性加LR。

控制50Hz、physics200Hz，23关节动作[-1,1]，解码default_q+0.25×ACTION_GAIN×action；actor输入375=state52+last_action23+history300，privileged358，总critic733，z256。硬位置/速度/力矩越限terminated，容差.001；未丢失terminal observation，超时truncated单独保留。

训练平地、无噪声、仅train RobotState RSI；独立velocity采用官方70%flat/30%rough、原生noise/commands、固定16seeds61000..61015、4GPU×1env。每例要求1000控制步/20s，无fall/NaN/hard limit，速度向量RMSE≤.20m/s、yawRMSE≤.30rad/s。acceptance套件suite_rpo_ufo_isaac_velocity_acceptance_v2，100新seeds66000..66099，至少80成功；未用于调参或执行。

全局冻结：24h，2026-10-10 15:31至2026-10-11 15:31 UTC；最多20,000新联合更新、2M在线转移、600个≤20s视频、80GiB存储增量；配置上限经原生模型/控制/速度优化请求派生至5，目前GPU实际A/B/C/D四个，E未启动。已完成新更新7072，在线累计上界526336。固定每段最多1000新更新，独立快评后才能继续；前2000更新质量观察热身，其后3窗口无改善停止。成功相同时中位存活至少比best+10步算改善。

完整恢复为trainer_state；native模型/六Adam/四target/RNG/replay/expert/z pool/三个sampler状态恢复，物理环境重建，不能称exact simulation resume。最后sampler同步版本1016、online76032，最后learner1024；这是结束前最后广播状态，恢复时先重新同步。

## 本段真实身份和计数

```json
{
  "train_id": "trn_5fe87af0fa6a4f158569",
  "job_id": "job_9dca508223f44a88a4ec",
  "run_id": "run_d1037ca56dce4f8ebaef",
  "model_version_id": "mv_521b1bfec83a4b4887a2",
  "producer_image": "sha256:88aee59f8d0f62d9e99affd231b34511f4624a9203bf7f9205c5f2f4008c1fe4",
  "evaluation_image": "sha256:9129bc326a1e1d68122bc630dff29e362066c2fdc4f0285eda7f61d211072d99",
  "source_train": "trn_a17b23c11cc74de7845b",
  "source_model": "mv_a1e1addefb774a6eaec5",
  "source_updates": 24,
  "final_updates": 1024,
  "new_updates": 1000,
  "new_online_transitions": 64512,
  "dataset_id": "ds_92718168bfec4da6a01d",
  "motion_manifest_sha256": "7e47773bf53588ee200cccc5bbde3999df172a11593d7359cbeca2a00ac4c5d8",
  "asset_sha256": "b58868a782d8863ac38059eb4eab89d27ee1d70565bf58f5a055f44510c8e842",
  "codec_contract_sha256": "fff6cf3e8f05c88fc7b552f65e5089dc47315c343bd9a205a1f59539cb2811c4",
  "codec_source_sha256": "sha256:1b0bbc0a7a350581ef1939faf93a3e27d824bde62107ecfe4e9ff988bbc088fb",
  "native_checkpoint": {
    "step": 1024,
    "counters": {
      "learner_joint_updates": 1024,
      "sampling_rounds": 100,
      "online_transitions": 76800,
      "next_trigger": 77824
    },
    "six_adam_steps": {
      "forward_optimizer": 1024,
      "backward_optimizer": 1024,
      "discriminator_optimizer": 1024,
      "actor_optimizer": 1024,
      "critic_optimizer": 1024,
      "aux_critic_optimizer": 1024
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
    "fingerprint": "sha256:9850261fefe3c937bc66d476f72f7ae63d4b28e45f1ed5930026781a44761d56",
    "execution": {
      "schema_version": "eb/training-binding/v1",
      "recipe_id": "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_slew_v4",
      "algorithm_id": "bfm_fb_cpr_aux",
      "provider_id": "ufo",
      "runtime_id": "rlinf",
      "simulator_id": "isaaclab",
      "progress_unit": "learner_joint_update",
      "checkpoint_format": "ufo_agent_state_v1",
      "topology_id": "one_learner_three_samplers"
    }
  },
  "start_utc": "2026-10-10T21:28:39.038457+00:00",
  "end_utc": "2026-10-10T22:02:22.006880+00:00",
  "sampler_sync": {
    "identity": {
      "version": 1016,
      "state_sha256": "6548ff1997cfd5855d5b138d964afa4db83609494812ac72b9cb3289298217e1",
      "online_transitions": 76032
    },
    "samplers": [
      {
        "rank": 0,
        "version": 1016,
        "state_sha256": "6548ff1997cfd5855d5b138d964afa4db83609494812ac72b9cb3289298217e1",
        "online_transitions": 76032
      },
      {
        "rank": 1,
        "version": 1016,
        "state_sha256": "6548ff1997cfd5855d5b138d964afa4db83609494812ac72b9cb3289298217e1",
        "online_transitions": 76032
      },
      {
        "rank": 2,
        "version": 1016,
        "state_sha256": "6548ff1997cfd5855d5b138d964afa4db83609494812ac72b9cb3289298217e1",
        "online_transitions": 76032
      }
    ],
    "schema_version": "eb/ufo-sampler-sync/v1",
    "sampling_rounds": 99
  }
}
```

## D实际完整原生配置

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
    "batch_size": 1024,
    "discount": 0.98,
    "use_mix_rollout": true,
    "update_z_every_step": 100,
    "z_buffer_size": 8192,
    "rollout_expert_trajectories": false,
    "rollout_expert_trajectories_length": 250,
    "rollout_expert_trajectories_percentage": 0.5,
    "lr_discriminator": 1e-05,
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
    "penalty_slippage"
  ],
  "aux_rewards_scaling": {
    "penalty_action_rate": -0.1,
    "penalty_feet_ori": -0.4,
    "penalty_ankle_roll": -4.0,
    "limits_dof_pos": -10.0,
    "penalty_slippage": -2.0,
    "penalty_undesired_contact": -1.0,
    "penalty_torques": 0.0,
    "limits_torque": 0.0
  },
  "cudagraphs": false,
  "compile": false
}
```

## D实际resolved Hydra配置

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
    motion_root: /workspace/data/ds_92718168bfec4da6a01d
    asset_sha256: b58868a782d8863ac38059eb4eab89d27ee1d70565bf58f5a055f44510c8e842
    batch_size: 1024
    replay_capacity: 1000000
    hidden_layers: 6
    topology_id: one_learner_three_samplers
    learning_profile: physical_limits_stable_slew_v4
    architecture_profile: native_fb_v1
    rollout_expert_trajectories: false
  group_name: ActorGroup
  training_backend: ufo_agent
  micro_batch_size: 1024
  global_batch_size: 1024
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
    experiment_name: RPO UFO native target-slew first interval to1024
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
    env_type: isaaclab
    total_num_envs: 192
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
      id: RPO-UFO-Isaac-RSI-slew-v4
      num_envs: null
      max_episode_steps: 1000
      task_description: RPO V2 BFM-Zero online interaction with pinned observation-only
        expert motion.
  eval:
    env_type: isaaclab
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
      id: RPO-UFO-Isaac-RSI-v1
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

## 动作解码、物理限速和速度辅助信号

```json
{
  "action_gain": [
    1,
    2,
    2,
    1,
    1,
    2,
    2,
    6,
    4,
    2,
    1,
    9,
    7,
    3,
    2,
    2,
    3,
    2,
    3,
    1,
    1,
    1,
    1
  ],
  "default_q": [
    0,
    0,
    0,
    0,
    0,
    0.18,
    0.18,
    -0.1,
    -0.1,
    0.06,
    -0.06,
    0.3,
    0.3,
    0,
    0,
    -0.2,
    -0.2,
    0.78,
    0.78,
    0,
    0,
    0,
    0
  ],
  "rated_velocity_rad_s": [
    25,
    25,
    25,
    25,
    25,
    8,
    8,
    25,
    25,
    8,
    8,
    25,
    25,
    8,
    8,
    8,
    8,
    8,
    8,
    8,
    8,
    8,
    8
  ],
  "target_slew_fraction": 0.5,
  "train_trace_audit": {
    "rounds": 84,
    "rows": 64512,
    "replay_is_requested_action": true,
    "applied_matches_shared_controller": true,
    "target_delta_cap_verified": true,
    "reset_anchor_verified": true,
    "reset_samples": 4285,
    "reset_outside_target_box_fraction": 0.8203033838973163,
    "reset_projection_residual_rad": {
      "median": 0.35675978660583496,
      "p95": 0.8799999952316284,
      "max": 1.7291607856750488
    },
    "first_control_terminated_fraction": 0.0025670945157526253,
    "first_control_hard_limit_fraction": 0.001633605600933489,
    "requested_saturation_fraction": 0.45959767512077293,
    "applied_saturation_fraction": 0.2962347416321601,
    "hard_limit_transitions": 1695,
    "every_hard_limit_terminated": true,
    "terminated_transitions": 4097,
    "violations": [
      1089.0,
      759.0,
      0.0
    ],
    "applied_action_abs_max": 0.9999989867210388,
    "joint_velocity_abs_max": 42.276954650878906,
    "maximum_target_delta_rad": 0.2500000521540642,
    "pd_decoder_max_error": 0.0
  },
  "posthoc_D_early_velocity_signal": {
    "run_id": "run_d1037ca56dce4f8ebaef",
    "round_range": [
      17,
      48
    ],
    "rows": 24576,
    "requested_saturation_fraction": 0.45283309273097827,
    "applied_saturation_fraction": 0.27141007133152173,
    "any_joint_over_80pct_velocity_limit_fraction": 0.23103841145833334,
    "hard_position_count": 394,
    "hard_velocity_count": 325,
    "per_step_9th_raw_penalty": {
      "mean": 0.010197989580673106,
      "p95": 0.04885663449572108,
      "max": 9.10994124122479
    },
    "proposed_minus5_contribution_mean": -0.05098994790336553,
    "diagnostic_only": "post-hoc scalar on actual D trace; never fed to D or used to relabel replay; not a causal effect or E training proof"
  }
}
```

## 全部指标统计及其限制

以下111项均来自原始Run指标；原始值在完整记录逐点内嵌。first/last、min/max/median/p95、前后最多50个实际记录点均保留。合并范围D24的17..24更新与D1024的25..1024；env/episode_len只有64点，两个50点窗口重叠36点，不代表独立统计窗口。未对缺失点补值。

```json
{
  "runs": [
    {
      "run_id": "run_b83cbc1f513141e48b51",
      "first": 17,
      "last": 24
    },
    {
      "run_id": "run_d1037ca56dce4f8ebaef",
      "first": 25,
      "last": 1024
    }
  ],
  "window": "up to 50 actual recorded points; no padding",
  "statistics": {
    "env/episode_len": {
      "n": 64,
      "first_step": 24,
      "last_step": 1024,
      "first": 3.75,
      "last": 18.517240524291992,
      "mean": 15.281531728804111,
      "min": 3.75,
      "max": 21.93181800842285,
      "median": 15.284495830535889,
      "p95": 20.101586914062498,
      "first_window_mean": 14.323750772476195,
      "last_window_mean": 16.658128986358644,
      "window_points": 50
    },
    "env/num_trajectories": {
      "n": 64,
      "first_step": 24,
      "last_step": 1024,
      "first": 4.0,
      "last": 29.0,
      "mean": 48.5625,
      "min": 4.0,
      "max": 92.0,
      "median": 47.5,
      "p95": 76.54999999999998,
      "first_window_mean": 51.28,
      "last_window_mean": 45.58,
      "window_points": 50
    },
    "env/return": {
      "n": 64,
      "first_step": 24,
      "last_step": 1024,
      "first": 0.0,
      "last": 0.0,
      "mean": 0.0,
      "min": 0.0,
      "max": 0.0,
      "median": 0.0,
      "p95": 0.0,
      "first_window_mean": 0.0,
      "last_window_mean": 0.0,
      "window_points": 50
    },
    "env/reward": {
      "n": 64,
      "first_step": 24,
      "last_step": 1024,
      "first": 0.0,
      "last": 0.0,
      "mean": 0.0,
      "min": 0.0,
      "max": 0.0,
      "median": 0.0,
      "p95": 0.0,
      "first_window_mean": 0.0,
      "last_window_mean": 0.0,
      "window_points": 50
    },
    "env/success_once": {
      "n": 64,
      "first_step": 24,
      "last_step": 1024,
      "first": 0.0,
      "last": 0.0,
      "mean": 0.0,
      "min": 0.0,
      "max": 0.0,
      "median": 0.0,
      "p95": 0.0,
      "first_window_mean": 0.0,
      "last_window_mean": 0.0,
      "window_points": 50
    },
    "rollout/ufo/received_transitions": {
      "n": 64,
      "first_step": 24,
      "last_step": 1024,
      "first": 768.0,
      "last": 768.0,
      "mean": 768.0,
      "min": 768.0,
      "max": 768.0,
      "median": 768.0,
      "p95": 768.0,
      "first_window_mean": 768.0,
      "last_window_mean": 768.0,
      "window_points": 50
    },
    "time/actor_training": {
      "n": 64,
      "first_step": 24,
      "last_step": 1024,
      "first": 1.8217658996582031,
      "last": 3.109604597091675,
      "mean": 6.055472526699305,
      "min": 1.8217658996582031,
      "max": 8.216520309448242,
      "median": 6.12638783454895,
      "p95": 6.2293678522109985,
      "first_window_mean": 6.094408569335937,
      "last_window_mean": 6.080201468467712,
      "window_points": 50
    },
    "time/cal_adv_and_returns": {
      "n": 64,
      "first_step": 24,
      "last_step": 1024,
      "first": 0.0020394325256347656,
      "last": 0.0018961429595947266,
      "mean": 0.002442389726638794,
      "min": 0.0016481876373291016,
      "max": 0.013873100280761719,
      "median": 0.00209200382232666,
      "p95": 0.002799248695373535,
      "first_window_mean": 0.002229161262512207,
      "last_window_mean": 0.0025568532943725584,
      "window_points": 50
    },
    "time/env/compute_bootstrap_rewards": {
      "n": 64,
      "first_step": 24,
      "last_step": 1024,
      "first": 0.00020368979312479496,
      "last": 0.0001787981018424034,
      "mean": 0.00019269902259111404,
      "min": 0.0001499189529567957,
      "max": 0.00024466216564178467,
      "median": 0.00018927990458905697,
      "p95": 0.00023201513104140756,
      "first_window_mean": 0.00019357193261384964,
      "last_window_mean": 0.00019323200918734074,
      "window_points": 50
    },
    "time/env/env/bootstrap_step": {
      "n": 64,
      "first_step": 24,
      "last_step": 1024,
      "first": 0.008203918114304543,
      "last": 0.00034375605173408985,
      "mean": 0.0004302548841224052,
      "min": 0.00023011211305856705,
      "max": 0.008203918114304543,
      "median": 0.00031743559520691633,
      "p95": 0.00039901173440739515,
      "first_window_mean": 0.00046040909830480814,
      "last_window_mean": 0.0003131436416879296,
      "window_points": 50
    },
    "time/env/env/send_rollout_trajectories": {
      "n": 64,
      "first_step": 24,
      "last_step": 1024,
      "first": 0.20046474039554596,
      "last": 0.18934307992458344,
      "mean": 0.19489741697907448,
      "min": 0.1769053041934967,
      "max": 0.21600717306137085,
      "median": 0.19453638792037964,
      "p95": 0.210122837126255,
      "first_window_mean": 0.19425175786018373,
      "last_window_mean": 0.19598080784082414,
      "window_points": 50
    },
    "time/env/env_interact_step": {
      "n": 64,
      "first_step": 24,
      "last_step": 1024,
      "first": 0.6767593622207642,
      "last": 0.5941898226737976,
      "mean": 0.5892384238541126,
      "min": 0.49130523204803467,
      "max": 1.8377044200897217,
      "median": 0.5650085806846619,
      "p95": 0.6727521419525145,
      "first_window_mean": 0.5941357624530792,
      "last_window_mean": 0.56712209045887,
      "window_points": 50
    },
    "time/env/interact": {
      "n": 64,
      "first_step": 24,
      "last_step": 1024,
      "first": 1.524786114692688,
      "last": 0.9433251023292542,
      "mean": 0.9444671692326665,
      "min": 0.8375372886657715,
      "max": 2.1474809646606445,
      "median": 0.9118475914001465,
      "p95": 1.0262677788734436,
      "first_window_mean": 0.9523949086666107,
      "last_window_mean": 0.9137277698516846,
      "window_points": 50
    },
    "time/env/run_interact_once": {
      "n": 64,
      "first_step": 24,
      "last_step": 1024,
      "first": 1.5247297286987305,
      "last": 0.9432827234268188,
      "mean": 0.944437931291759,
      "min": 0.8374930620193481,
      "max": 2.1474549770355225,
      "median": 0.9118238985538483,
      "p95": 1.0262267947196961,
      "first_window_mean": 0.9523658466339111,
      "last_window_mean": 0.9136984527111054,
      "window_points": 50
    },
    "time/generate_rollouts": {
      "n": 64,
      "first_step": 24,
      "last_step": 1024,
      "first": 1.685633659362793,
      "last": 0.9986450672149658,
      "mean": 0.9943511076271534,
      "min": 0.8830087184906006,
      "max": 2.1767966747283936,
      "median": 0.9644007682800293,
      "p95": 1.0592726588249206,
      "first_window_mean": 1.0029714822769165,
      "last_window_mean": 0.9621652507781983,
      "window_points": 50
    },
    "time/rollout/generate_one_epoch": {
      "n": 64,
      "first_step": 24,
      "last_step": 1024,
      "first": 1.3160351514816284,
      "last": 0.7315098643302917,
      "mean": 0.734948581084609,
      "min": 0.622591495513916,
      "max": 1.9597398042678833,
      "median": 0.7021030187606812,
      "p95": 0.817041316628456,
      "first_window_mean": 0.7427278184890747,
      "last_window_mean": 0.7040347397327423,
      "window_points": 50
    },
    "time/rollout/predict": {
      "n": 64,
      "first_step": 24,
      "last_step": 1024,
      "first": 0.2784363627433777,
      "last": 0.03689722716808319,
      "mean": 0.03453511546831578,
      "min": 0.02377730794250965,
      "max": 0.2784363627433777,
      "median": 0.030384535901248455,
      "p95": 0.039885256625711914,
      "first_window_mean": 0.036187258921563624,
      "last_window_mean": 0.030830656960606575,
      "window_points": 50
    },
    "time/rollout/rollout/generate": {
      "n": 64,
      "first_step": 24,
      "last_step": 1024,
      "first": 1.3196438550949097,
      "last": 0.7316048741340637,
      "mean": 0.7351987874135375,
      "min": 0.6226921677589417,
      "max": 1.9598543643951416,
      "median": 0.702420711517334,
      "p95": 0.8178011387586593,
      "first_window_mean": 0.7430067646503449,
      "last_window_mean": 0.704216730594635,
      "window_points": 50
    },
    "time/rollout/sync_model_from_actor": {
      "n": 64,
      "first_step": 24,
      "last_step": 1024,
      "first": 14.208600997924805,
      "last": 12.779870986938477,
      "mean": 12.84639298915863,
      "min": 12.464377403259277,
      "max": 14.208600997924805,
      "median": 12.843361854553223,
      "p95": 13.078665113449096,
      "first_window_mean": 12.846545276641846,
      "last_window_mean": 12.821691379547119,
      "window_points": 50
    },
    "time/step": {
      "n": 64,
      "first_step": 24,
      "last_step": 1024,
      "first": 21.177371978759766,
      "last": 20.668766021728516,
      "mean": 23.639618545770645,
      "min": 20.668766021728516,
      "max": 27.091062545776367,
      "median": 23.677263259887695,
      "p95": 23.92426471710205,
      "first_window_mean": 23.67748394012451,
      "last_window_mean": 23.63100025177002,
      "window_points": 50
    },
    "time/sync_weights": {
      "n": 64,
      "first_step": 24,
      "last_step": 1024,
      "first": 17.66724967956543,
      "last": 16.55817222595215,
      "mean": 16.586841613054276,
      "min": 16.229225158691406,
      "max": 17.66724967956543,
      "median": 16.56768798828125,
      "p95": 16.7956805229187,
      "first_window_mean": 16.577357940673828,
      "last_window_mean": 16.585572052001954,
      "window_points": 50
    },
    "train/ufo/B": {
      "n": 1008,
      "first_step": 17,
      "last_step": 1024,
      "first": -0.020892322063446045,
      "last": 0.00027838919777423143,
      "mean": -0.0019125256029159543,
      "min": -0.02429279312491417,
      "max": 0.0058679888024926186,
      "median": -0.0009234582248609513,
      "p95": 0.002574805333279072,
      "first_window_mean": -0.01658657571300864,
      "last_window_mean": 0.00020032577507663517,
      "window_points": 50
    },
    "train/ufo/B_norm": {
      "n": 1008,
      "first_step": 17,
      "last_step": 1024,
      "first": 16.0,
      "last": 16.0,
      "mean": 16.0,
      "min": 16.0,
      "max": 16.0,
      "median": 16.0,
      "p95": 16.0,
      "first_window_mean": 16.0,
      "last_window_mean": 16.0,
      "window_points": 50
    },
    "train/ufo/F1": {
      "n": 1008,
      "first_step": 17,
      "last_step": 1024,
      "first": 0.015925180166959763,
      "last": 0.10002335906028748,
      "mean": 0.0560591522523845,
      "min": 0.004195090848952532,
      "max": 0.1256732940673828,
      "median": 0.05170718394219875,
      "p95": 0.10277053713798523,
      "first_window_mean": 0.020099465688690542,
      "last_window_mean": 0.10538638383150101,
      "window_points": 50
    },
    "train/ufo/M1": {
      "n": 1008,
      "first_step": 17,
      "last_step": 1024,
      "first": -1.1429533958435059,
      "last": 1.8854531049728394,
      "mean": 1.0773115098986419,
      "min": -2.657503128051758,
      "max": 2.348270893096924,
      "median": 1.1057624220848083,
      "p95": 1.6967881858348846,
      "first_window_mean": -0.46923263125121595,
      "last_window_mean": 1.6537498927116394,
      "window_points": 50
    },
    "train/ufo/Q1": {
      "n": 1008,
      "first_step": 17,
      "last_step": 1024,
      "first": 0.11389666795730591,
      "last": -17.18871307373047,
      "mean": -7.726526407356326,
      "min": -17.236061096191406,
      "max": 0.33636945486068726,
      "median": -7.3644859790802,
      "p95": -0.3315701067447663,
      "first_window_mean": -0.01520863669924438,
      "last_window_mean": -16.4648380279541,
      "window_points": 50
    },
    "train/ufo/Q_aux": {
      "n": 1008,
      "first_step": 17,
      "last_step": 1024,
      "first": -2.3869056701660156,
      "last": -8.89253044128418,
      "mean": -5.399791884516913,
      "min": -9.090442657470703,
      "max": -1.3924294710159302,
      "median": -5.335014820098877,
      "p95": -2.4556902170181276,
      "first_window_mean": -2.128623342514038,
      "last_window_mean": -8.62877820968628,
      "window_points": 50
    },
    "train/ufo/Q_discriminator": {
      "n": 1008,
      "first_step": 17,
      "last_step": 1024,
      "first": -0.14319702982902527,
      "last": -16.896656036376953,
      "mean": -7.81801968192338,
      "min": -17.40843391418457,
      "max": 0.032047417014837265,
      "median": -7.422753810882568,
      "p95": -0.5314171999692917,
      "first_window_mean": -0.26817034190520644,
      "last_window_mean": -16.46777883529663,
      "window_points": 50
    },
    "train/ufo/Q_fb": {
      "n": 1008,
      "first_step": 17,
      "last_step": 1024,
      "first": -6.592137813568115,
      "last": 83.77679443359375,
      "mean": 41.20002066258282,
      "min": -7.621039390563965,
      "max": 91.2486343383789,
      "median": 41.35234260559082,
      "p95": 79.53553085327148,
      "first_window_mean": -1.7966719776391984,
      "last_window_mean": 81.7948013305664,
      "window_points": 50
    },
    "train/ufo/action_abs_mean": {
      "n": 64,
      "first_step": 24,
      "last_step": 1024,
      "first": 0.885714054107666,
      "last": 0.9147884249687195,
      "mean": 0.9137543328106403,
      "min": 0.8717551231384277,
      "max": 0.922569751739502,
      "median": 0.9158909916877747,
      "p95": 0.9216488867998123,
      "first_window_mean": 0.9142167377471924,
      "last_window_mean": 0.9160782587528229,
      "window_points": 50
    },
    "train/ufo/action_saturation_fraction": {
      "n": 64,
      "first_step": 24,
      "last_step": 1024,
      "first": 0.4099864065647125,
      "last": 0.45646512508392334,
      "mean": 0.45848724618554115,
      "min": 0.37879303097724915,
      "max": 0.47554346919059753,
      "median": 0.46300384402275085,
      "p95": 0.47226561456918714,
      "first_window_mean": 0.4590715569257736,
      "last_window_mean": 0.4630196988582611,
      "window_points": 50
    },
    "train/ufo/actor_loss": {
      "n": 1008,
      "first_step": 17,
      "last_step": 1024,
      "first": 7.2515974044799805,
      "last": 7.985015869140625,
      "mean": -10.5513914362897,
      "min": -21.994121551513672,
      "max": 10.579788208007812,
      "median": -12.333804607391357,
      "p95": 3.868770980834961,
      "first_window_mean": 2.348183835744858,
      "last_window_mean": 4.771694183349609,
      "window_points": 50
    },
    "train/ufo/actor_optimizer/grad_norm_l2": {
      "n": 1008,
      "first_step": 17,
      "last_step": 1024,
      "first": 3.7974395751953125,
      "last": 5.879215240478516,
      "mean": 4.711604757441415,
      "min": 2.565969467163086,
      "max": 6.686490058898926,
      "median": 4.8769371509552,
      "p95": 6.12190351486206,
      "first_window_mean": 2.9693395614624025,
      "last_window_mean": 6.1065380668640135,
      "window_points": 50
    },
    "train/ufo/actor_optimizer/lr": {
      "n": 1008,
      "first_step": 17,
      "last_step": 1024,
      "first": 0.0003000000142492354,
      "last": 0.0003000000142492354,
      "mean": 0.0003000000142492354,
      "min": 0.0003000000142492354,
      "max": 0.0003000000142492354,
      "median": 0.0003000000142492354,
      "p95": 0.0003000000142492354,
      "first_window_mean": 0.0003000000142492354,
      "last_window_mean": 0.0003000000142492354,
      "window_points": 50
    },
    "train/ufo/actor_optimizer/parameter_delta_l2": {
      "n": 1008,
      "first_step": 17,
      "last_step": 1024,
      "first": 0.3602486252784729,
      "last": 0.4281977117061615,
      "mean": 0.3652777010722766,
      "min": 0.2453058958053589,
      "max": 0.44386914372444153,
      "median": 0.3722044825553894,
      "p95": 0.42453020960092547,
      "first_window_mean": 0.26980050444602965,
      "last_window_mean": 0.4268161255121231,
      "window_points": 50
    },
    "train/ufo/applied_action_abs_mean": {
      "n": 64,
      "first_step": 24,
      "last_step": 1024,
      "first": 0.5229846835136414,
      "last": 0.7335072159767151,
      "mean": 0.7410990251228213,
      "min": 0.5229846835136414,
      "max": 0.7900278568267822,
      "median": 0.7523009181022644,
      "p95": 0.7740025877952575,
      "first_window_mean": 0.7404035103321075,
      "last_window_mean": 0.7557063722610473,
      "window_points": 50
    },
    "train/ufo/applied_action_saturation_fraction": {
      "n": 64,
      "first_step": 24,
      "last_step": 1024,
      "first": 0.09799592196941376,
      "last": 0.2998754382133484,
      "mean": 0.2956870263442397,
      "min": 0.09799592196941376,
      "max": 0.3426743745803833,
      "median": 0.30932970345020294,
      "p95": 0.32641246616840364,
      "first_window_mean": 0.29278079748153685,
      "last_window_mean": 0.3119950175285339,
      "window_points": 50
    },
    "train/ufo/auxQ1": {
      "n": 1008,
      "first_step": 17,
      "last_step": 1024,
      "first": -2.6976451873779297,
      "last": -9.08116340637207,
      "mean": -5.308183874047938,
      "min": -9.08116340637207,
      "max": -1.074152946472168,
      "median": -5.224926948547363,
      "p95": -2.244580364227295,
      "first_window_mean": -1.9185668349266052,
      "last_window_mean": -8.592174797058105,
      "window_points": 50
    },
    "train/ufo/aux_critic_loss": {
      "n": 1008,
      "first_step": 17,
      "last_step": 1024,
      "first": 2.3439722061157227,
      "last": 1.2051866054534912,
      "mean": 0.5660342975475249,
      "min": 0.24464447796344757,
      "max": 2.3439722061157227,
      "median": 0.5039214789867401,
      "p95": 1.010355186462402,
      "first_window_mean": 1.24248455286026,
      "last_window_mean": 0.9143365955352784,
      "window_points": 50
    },
    "train/ufo/aux_critic_optimizer/grad_norm_l2": {
      "n": 1008,
      "first_step": 17,
      "last_step": 1024,
      "first": 13.907383918762207,
      "last": 6.889871597290039,
      "mean": 3.6286677980706807,
      "min": 1.8908780813217163,
      "max": 13.907383918762207,
      "median": 3.2575321197509766,
      "p95": 6.256592631340027,
      "first_window_mean": 6.079298610687256,
      "last_window_mean": 4.883796653747559,
      "window_points": 50
    },
    "train/ufo/aux_critic_optimizer/lr": {
      "n": 1008,
      "first_step": 17,
      "last_step": 1024,
      "first": 0.0003000000142492354,
      "last": 0.0003000000142492354,
      "mean": 0.0003000000142492354,
      "min": 0.0003000000142492354,
      "max": 0.0003000000142492354,
      "median": 0.0003000000142492354,
      "p95": 0.0003000000142492354,
      "first_window_mean": 0.0003000000142492354,
      "last_window_mean": 0.0003000000142492354,
      "window_points": 50
    },
    "train/ufo/aux_critic_optimizer/parameter_delta_l2": {
      "n": 1008,
      "first_step": 17,
      "last_step": 1024,
      "first": 0.4378751516342163,
      "last": 0.41929253935813904,
      "mean": 0.24004419834633905,
      "min": 0.10699933767318726,
      "max": 0.4493442177772522,
      "median": 0.22263743728399277,
      "p95": 0.3968663260340691,
      "first_window_mean": 0.16648985981941222,
      "last_window_mean": 0.40336279571056366,
      "window_points": 50
    },
    "train/ufo/aux_rew/limits_dof_pos": {
      "n": 1008,
      "first_step": 17,
      "last_step": 1024,
      "first": 0.043834175914525986,
      "last": 0.25314417481422424,
      "mean": 0.18839424758750414,
      "min": 0.04325011000037193,
      "max": 0.26019343733787537,
      "median": 0.21100246906280518,
      "p95": 0.2482865273952484,
      "first_window_mean": 0.05154443554580212,
      "last_window_mean": 0.24663176596164704,
      "window_points": 50
    },
    "train/ufo/aux_rew/limits_torque": {
      "n": 1008,
      "first_step": 17,
      "last_step": 1024,
      "first": 0.0,
      "last": 0.0,
      "mean": 0.0,
      "min": 0.0,
      "max": 0.0,
      "median": 0.0,
      "p95": 0.0,
      "first_window_mean": 0.0,
      "last_window_mean": 0.0,
      "window_points": 50
    },
    "train/ufo/aux_rew/penalty_action_rate": {
      "n": 1008,
      "first_step": 17,
      "last_step": 1024,
      "first": 2.3308327198028564,
      "last": 1.1714353561401367,
      "mean": 1.4735470836361249,
      "min": 1.080564260482788,
      "max": 2.3608031272888184,
      "median": 1.3451288938522339,
      "p95": 2.19976863861084,
      "first_window_mean": 2.2622322368621828,
      "last_window_mean": 1.152942316532135,
      "window_points": 50
    },
    "train/ufo/aux_rew/penalty_ankle_roll": {
      "n": 1008,
      "first_step": 17,
      "last_step": 1024,
      "first": 0.014144237153232098,
      "last": 0.05788026750087738,
      "mean": 0.04510722805501982,
      "min": 0.013551747426390648,
      "max": 0.06125456839799881,
      "median": 0.048733558505773544,
      "p95": 0.05845059044659137,
      "first_window_mean": 0.017434771806001663,
      "last_window_mean": 0.05848139941692352,
      "window_points": 50
    },
    "train/ufo/aux_rew/penalty_feet_ori": {
      "n": 1008,
      "first_step": 17,
      "last_step": 1024,
      "first": 0.16693642735481262,
      "last": 0.1941828727722168,
      "mean": 0.16553157779373348,
      "min": 0.1326635777950287,
      "max": 0.20516176521778107,
      "median": 0.16503353416919708,
      "p95": 0.18644266501069068,
      "first_window_mean": 0.16509485989809036,
      "last_window_mean": 0.18261738896369933,
      "window_points": 50
    },
    "train/ufo/aux_rew/penalty_slippage": {
      "n": 1008,
      "first_step": 17,
      "last_step": 1024,
      "first": 0.40437155961990356,
      "last": 0.3124353587627411,
      "mean": 0.33260512275118675,
      "min": 0.26083043217658997,
      "max": 0.45210421085357666,
      "median": 0.3255321681499481,
      "p95": 0.3994630008935928,
      "first_window_mean": 0.405168731212616,
      "last_window_mean": 0.3098850286006927,
      "window_points": 50
    },
    "train/ufo/aux_rew/penalty_torques": {
      "n": 1008,
      "first_step": 17,
      "last_step": 1024,
      "first": 3771.4326171875,
      "last": 12428.67578125,
      "mean": 9924.163823445639,
      "min": 3771.4326171875,
      "max": 12833.2880859375,
      "median": 10828.630859375,
      "p95": 12183.073193359374,
      "first_window_mean": 4205.888627929688,
      "last_window_mean": 12092.21099609375,
      "window_points": 50
    },
    "train/ufo/aux_rew/penalty_undesired_contact": {
      "n": 1008,
      "first_step": 17,
      "last_step": 1024,
      "first": 0.0,
      "last": 0.0,
      "mean": 4.940941220238095e-05,
      "min": 0.0,
      "max": 0.0009765625,
      "median": 0.0,
      "p95": 0.0006347656249999778,
      "first_window_mean": 0.0,
      "last_window_mean": 3.90625e-05,
      "window_points": 50
    },
    "train/ufo/backward_optimizer/grad_norm_l2": {
      "n": 1008,
      "first_step": 17,
      "last_step": 1024,
      "first": 195551.75,
      "last": 18148.728515625,
      "mean": 41907.863279312376,
      "min": 17434.46875,
      "max": 206184.09375,
      "median": 29190.6298828125,
      "p95": 100697.59882812499,
      "first_window_mean": 136190.93875,
      "last_window_mean": 18020.79046875,
      "window_points": 50
    },
    "train/ufo/backward_optimizer/lr": {
      "n": 1008,
      "first_step": 17,
      "last_step": 1024,
      "first": 9.999999747378752e-06,
      "last": 9.999999747378752e-06,
      "mean": 9.999999747378752e-06,
      "min": 9.999999747378752e-06,
      "max": 9.999999747378752e-06,
      "median": 9.999999747378752e-06,
      "p95": 9.999999747378752e-06,
      "first_window_mean": 9.999999747378752e-06,
      "last_window_mean": 9.999999747378752e-06,
      "window_points": 50
    },
    "train/ufo/backward_optimizer/parameter_delta_l2": {
      "n": 1008,
      "first_step": 17,
      "last_step": 1024,
      "first": 0.0035896720364689827,
      "last": 0.0013593054609373212,
      "mean": 0.0021229155074309796,
      "min": 0.0013530612923204899,
      "max": 0.0035929034929722548,
      "median": 0.001842249184846878,
      "p95": 0.003436644480098039,
      "first_window_mean": 0.003302025776356459,
      "last_window_mean": 0.0013604968739673495,
      "window_points": 50
    },
    "train/ufo/collected_transitions": {
      "n": 64,
      "first_step": 24,
      "last_step": 1024,
      "first": 768.0,
      "last": 768.0,
      "mean": 768.0,
      "min": 768.0,
      "max": 768.0,
      "median": 768.0,
      "p95": 768.0,
      "first_window_mean": 768.0,
      "last_window_mean": 768.0,
      "window_points": 50
    },
    "train/ufo/critic_loss": {
      "n": 1008,
      "first_step": 17,
      "last_step": 1024,
      "first": 1.487947940826416,
      "last": 4.050329685211182,
      "mean": 1.3065070265105792,
      "min": 0.20130346715450287,
      "max": 4.724181652069092,
      "median": 0.8224163055419922,
      "p95": 3.609397661685944,
      "first_window_mean": 0.6671771490573883,
      "last_window_mean": 3.651258430480957,
      "window_points": 50
    },
    "train/ufo/critic_optimizer/grad_norm_l2": {
      "n": 1008,
      "first_step": 17,
      "last_step": 1024,
      "first": 11.598573684692383,
      "last": 13.498269081115723,
      "mean": 6.828197713645678,
      "min": 1.5303393602371216,
      "max": 26.317049026489258,
      "median": 5.736822843551636,
      "p95": 15.93949155807495,
      "first_window_mean": 5.031745681762695,
      "last_window_mean": 15.53081756591797,
      "window_points": 50
    },
    "train/ufo/critic_optimizer/lr": {
      "n": 1008,
      "first_step": 17,
      "last_step": 1024,
      "first": 0.0003000000142492354,
      "last": 0.0003000000142492354,
      "mean": 0.0003000000142492354,
      "min": 0.0003000000142492354,
      "max": 0.0003000000142492354,
      "median": 0.0003000000142492354,
      "p95": 0.0003000000142492354,
      "first_window_mean": 0.0003000000142492354,
      "last_window_mean": 0.0003000000142492354,
      "window_points": 50
    },
    "train/ufo/critic_optimizer/parameter_delta_l2": {
      "n": 1008,
      "first_step": 17,
      "last_step": 1024,
      "first": 0.40139153599739075,
      "last": 0.6756578087806702,
      "mean": 0.3585225496115902,
      "min": 0.08700723201036453,
      "max": 0.7609537243843079,
      "median": 0.33702316880226135,
      "p95": 0.702822357416153,
      "first_window_mean": 0.15044689893722535,
      "last_window_mean": 0.7090363264083862,
      "window_points": 50
    },
    "train/ufo/disc_expert_loss": {
      "n": 1008,
      "first_step": 17,
      "last_step": 1024,
      "first": 0.3986208438873291,
      "last": 0.06117326766252518,
      "mean": 0.11350222034675497,
      "min": 0.052237845957279205,
      "max": 0.3986208438873291,
      "median": 0.09068560227751732,
      "p95": 0.21623197272419928,
      "first_window_mean": 0.2803023743629456,
      "last_window_mean": 0.062014605328440664,
      "window_points": 50
    },
    "train/ufo/disc_loss": {
      "n": 1008,
      "first_step": 17,
      "last_step": 1024,
      "first": 2.8667421340942383,
      "last": 0.27606990933418274,
      "mean": 0.5110014188916437,
      "min": 0.26527994871139526,
      "max": 2.8667421340942383,
      "median": 0.40955260396003723,
      "p95": 0.9055110633373258,
      "first_window_mean": 1.5910134363174437,
      "last_window_mean": 0.2878260111808777,
      "window_points": 50
    },
    "train/ufo/disc_train_loss": {
      "n": 1008,
      "first_step": 17,
      "last_step": 1024,
      "first": 0.6521362662315369,
      "last": 0.071867436170578,
      "mean": 0.16009946843417036,
      "min": 0.059072885662317276,
      "max": 0.6521362662315369,
      "median": 0.1093619056046009,
      "p95": 0.4292061358690261,
      "first_window_mean": 0.5226812845468521,
      "last_window_mean": 0.07458166502416134,
      "window_points": 50
    },
    "train/ufo/disc_wgan_gp_loss": {
      "n": 1008,
      "first_step": 17,
      "last_step": 1024,
      "first": 0.1815984845161438,
      "last": 0.014302920550107956,
      "mean": 0.023739972986125697,
      "min": 0.013518051244318485,
      "max": 0.1815984845161438,
      "median": 0.020897286012768745,
      "p95": 0.028928011562675236,
      "first_window_mean": 0.07880297739058734,
      "last_window_mean": 0.015122973825782537,
      "window_points": 50
    },
    "train/ufo/discriminator_optimizer/grad_norm_l2": {
      "n": 1008,
      "first_step": 17,
      "last_step": 1024,
      "first": 8.418952941894531,
      "last": 0.345083087682724,
      "mean": 0.8974851210498147,
      "min": 0.3243160843849182,
      "max": 8.449532508850098,
      "median": 0.5408653914928436,
      "p95": 2.2761567831039407,
      "first_window_mean": 5.7432102727890015,
      "last_window_mean": 0.3917819583415985,
      "window_points": 50
    },
    "train/ufo/discriminator_optimizer/lr": {
      "n": 1008,
      "first_step": 17,
      "last_step": 1024,
      "first": 9.999999747378752e-06,
      "last": 9.999999747378752e-06,
      "mean": 9.999999747378752e-06,
      "min": 9.999999747378752e-06,
      "max": 9.999999747378752e-06,
      "median": 9.999999747378752e-06,
      "p95": 9.999999747378752e-06,
      "first_window_mean": 9.999999747378752e-06,
      "last_window_mean": 9.999999747378752e-06,
      "window_points": 50
    },
    "train/ufo/discriminator_optimizer/parameter_delta_l2": {
      "n": 1008,
      "first_step": 17,
      "last_step": 1024,
      "first": 0.01506988424807787,
      "last": 0.0034426534548401833,
      "mean": 0.005670753311105294,
      "min": 0.0033833873458206654,
      "max": 0.015088734216988087,
      "median": 0.00515864766202867,
      "p95": 0.010138024901971217,
      "first_window_mean": 0.013733752481639385,
      "last_window_mean": 0.0035073854448273776,
      "window_points": 50
    },
    "train/ufo/environment_control_steps": {
      "n": 64,
      "first_step": 24,
      "last_step": 1024,
      "first": 68.0,
      "last": 400.0,
      "mean": 233.375,
      "min": 68.0,
      "max": 400.0,
      "median": 234.0,
      "p95": 383.4,
      "first_window_mean": 196.08,
      "last_window_mean": 270.64,
      "window_points": 50
    },
    "train/ufo/expert_draws/g1": {
      "n": 64,
      "first_step": 24,
      "last_step": 1024,
      "first": 11800.0,
      "last": 525432.0,
      "mean": 270821.0,
      "min": 11800.0,
      "max": 525432.0,
      "median": 271096.0,
      "p95": 503780.79999999993,
      "first_window_mean": 213395.68,
      "last_window_mean": 328324.0,
      "window_points": 50
    },
    "train/ufo/expert_draws/official_rpo": {
      "n": 64,
      "first_step": 24,
      "last_step": 1024,
      "first": 12776.0,
      "last": 523144.0,
      "mean": 269723.0,
      "min": 12776.0,
      "max": 523144.0,
      "median": 269576.0,
      "p95": 501377.6,
      "first_window_mean": 212588.32,
      "last_window_mean": 326872.16,
      "window_points": 50
    },
    "train/ufo/fb_diag": {
      "n": 1008,
      "first_step": 17,
      "last_step": 1024,
      "first": -23.424514770507812,
      "last": -206.64553833007812,
      "mean": -150.028999900061,
      "min": -211.8548126220703,
      "max": -23.424514770507812,
      "median": -168.76849365234375,
      "p95": -38.86622505187989,
      "first_window_mean": -30.520785293579102,
      "last_window_mean": -202.8880810546875,
      "window_points": 50
    },
    "train/ufo/fb_loss": {
      "n": 1008,
      "first_step": 17,
      "last_step": 1024,
      "first": 144971.625,
      "last": 18113.21484375,
      "mean": 39054.65073358445,
      "min": 17830.041015625,
      "max": 144971.625,
      "median": 29943.216796875,
      "p95": 91826.593359375,
      "first_window_mean": 113397.325625,
      "last_window_mean": 18459.6026171875,
      "window_points": 50
    },
    "train/ufo/fb_offdiag": {
      "n": 1008,
      "first_step": 17,
      "last_step": 1024,
      "first": 263.1612243652344,
      "last": 214.24082946777344,
      "mean": 154.69650520597185,
      "min": 96.89938354492188,
      "max": 263.1612243652344,
      "median": 157.42952728271484,
      "p95": 206.87734451293946,
      "first_window_mean": 157.2190071105957,
      "last_window_mean": 205.1061212158203,
      "window_points": 50
    },
    "train/ufo/forward_optimizer/grad_norm_l2": {
      "n": 1008,
      "first_step": 17,
      "last_step": 1024,
      "first": 622.9271850585938,
      "last": 148.5118865966797,
      "mean": 157.97486693518502,
      "min": 113.4677734375,
      "max": 662.11328125,
      "median": 140.70076751708984,
      "p95": 235.3386238098144,
      "first_window_mean": 392.87721954345704,
      "last_window_mean": 144.48674377441407,
      "window_points": 50
    },
    "train/ufo/forward_optimizer/lr": {
      "n": 1008,
      "first_step": 17,
      "last_step": 1024,
      "first": 0.0003000000142492354,
      "last": 0.0003000000142492354,
      "mean": 0.0003000000142492354,
      "min": 0.0003000000142492354,
      "max": 0.0003000000142492354,
      "median": 0.0003000000142492354,
      "p95": 0.0003000000142492354,
      "first_window_mean": 0.0003000000142492354,
      "last_window_mean": 0.0003000000142492354,
      "window_points": 50
    },
    "train/ufo/forward_optimizer/parameter_delta_l2": {
      "n": 1008,
      "first_step": 17,
      "last_step": 1024,
      "first": 0.45710912346839905,
      "last": 0.5024192929267883,
      "mean": 0.33930639115472633,
      "min": 0.14975090324878693,
      "max": 0.508054256439209,
      "median": 0.3460565060377121,
      "p95": 0.4944998398423195,
      "first_window_mean": 0.21326299369335175,
      "last_window_mean": 0.49756530463695525,
      "window_points": 50
    },
    "train/ufo/hard_position_count": {
      "n": 64,
      "first_step": 24,
      "last_step": 1024,
      "first": 1.0,
      "last": 10.0,
      "mean": 12.640625,
      "min": 1.0,
      "max": 20.0,
      "median": 13.0,
      "p95": 18.0,
      "first_window_mean": 12.86,
      "last_window_mean": 13.2,
      "window_points": 50
    },
    "train/ufo/hard_torque_count": {
      "n": 64,
      "first_step": 24,
      "last_step": 1024,
      "first": 0.0,
      "last": 0.0,
      "mean": 0.0,
      "min": 0.0,
      "max": 0.0,
      "median": 0.0,
      "p95": 0.0,
      "first_window_mean": 0.0,
      "last_window_mean": 0.0,
      "window_points": 50
    },
    "train/ufo/hard_velocity_count": {
      "n": 64,
      "first_step": 24,
      "last_step": 1024,
      "first": 3.0,
      "last": 8.0,
      "mean": 9.078125,
      "min": 2.0,
      "max": 17.0,
      "median": 9.0,
      "p95": 14.849999999999994,
      "first_window_mean": 9.56,
      "last_window_mean": 8.86,
      "window_points": 50
    },
    "train/ufo/learner_joint_updates": {
      "n": 1008,
      "first_step": 17,
      "last_step": 1024,
      "first": 17.0,
      "last": 1024.0,
      "mean": 520.5,
      "min": 17.0,
      "max": 1024.0,
      "median": 520.5,
      "p95": 973.65,
      "first_window_mean": 41.5,
      "last_window_mean": 999.5,
      "window_points": 50
    },
    "train/ufo/mean_aux_reward": {
      "n": 1008,
      "first_step": 17,
      "last_step": 1024,
      "first": -1.1354396343231201,
      "last": -1.465177297592163,
      "mean": -1.3726436760690477,
      "min": -1.5127170085906982,
      "max": -1.1246895790100098,
      "median": -1.3816468715667725,
      "p95": -1.2527877032756807,
      "first_window_mean": -1.2026945018768311,
      "last_window_mean": -1.4367640686035157,
      "window_points": 50
    },
    "train/ufo/mean_disc_reward": {
      "n": 1008,
      "first_step": 17,
      "last_step": 1024,
      "first": 0.20410257577896118,
      "last": -3.7636842727661133,
      "mean": -2.532038527008447,
      "min": -3.8657593727111816,
      "max": 0.24826538562774658,
      "median": -3.0022884607315063,
      "p95": 0.046264264360069744,
      "first_window_mean": 0.16664417058229447,
      "last_window_mean": -3.7304034757614137,
      "window_points": 50
    },
    "train/ufo/mean_next_Q": {
      "n": 1008,
      "first_step": 17,
      "last_step": 1024,
      "first": -0.01336921751499176,
      "last": -13.753725051879883,
      "mean": -5.313999633378689,
      "min": -13.753725051879883,
      "max": 0.4417334198951721,
      "median": -4.51494026184082,
      "p95": 0.013929260149597979,
      "first_window_mean": 0.2627707113325596,
      "last_window_mean": -13.217436294555664,
      "window_points": 50
    },
    "train/ufo/mean_next_auxQ": {
      "n": 1008,
      "first_step": 17,
      "last_step": 1024,
      "first": -0.2662714421749115,
      "last": -7.724252700805664,
      "mean": -3.9713855502565227,
      "min": -7.724252700805664,
      "max": -0.13959765434265137,
      "median": -3.968439221382141,
      "p95": -0.6855727136135104,
      "first_window_mean": -0.33637055575847624,
      "last_window_mean": -7.35188720703125,
      "window_points": 50
    },
    "train/ufo/online_transitions": {
      "n": 1008,
      "first_step": 17,
      "last_step": 1024,
      "first": 12288.0,
      "last": 76800.0,
      "mean": 44800.76190476191,
      "min": 12288.0,
      "max": 76800.0,
      "median": 44928.0,
      "p95": 73728.0,
      "first_window_mean": 14146.56,
      "last_window_mean": 75448.32,
      "window_points": 50
    },
    "train/ufo/orth_loss": {
      "n": 1008,
      "first_step": 17,
      "last_step": 1024,
      "first": 1447.31884765625,
      "last": 181.05618286132812,
      "mean": 390.4998315478128,
      "min": 178.26052856445312,
      "max": 1447.31884765625,
      "median": 299.54644775390625,
      "p95": 917.5432861328125,
      "first_window_mean": 1132.7062622070312,
      "last_window_mean": 184.5738446044922,
      "window_points": 50
    },
    "train/ufo/orth_loss_diag": {
      "n": 1008,
      "first_step": 17,
      "last_step": 1024,
      "first": -256.0,
      "last": -256.0,
      "mean": -256.0,
      "min": -256.0,
      "max": -256.0,
      "median": -256.0,
      "p95": -256.0,
      "first_window_mean": -256.0,
      "last_window_mean": -256.0,
      "window_points": 50
    },
    "train/ufo/orth_loss_offdiag": {
      "n": 1008,
      "first_step": 17,
      "last_step": 1024,
      "first": 1703.31884765625,
      "last": 437.0561828613281,
      "mean": 646.4998315478128,
      "min": 434.2605285644531,
      "max": 1703.31884765625,
      "median": 555.5464477539062,
      "p95": 1173.5432861328125,
      "first_window_mean": 1388.7062622070312,
      "last_window_mean": 440.5738446044922,
      "window_points": 50
    },
    "train/ufo/q_loss": {
      "n": 1008,
      "first_step": 17,
      "last_step": 1024,
      "first": 0.0,
      "last": 0.0,
      "mean": 0.0,
      "min": 0.0,
      "max": 0.0,
      "median": 0.0,
      "p95": 0.0,
      "first_window_mean": 0.0,
      "last_window_mean": 0.0,
      "window_points": 50
    },
    "train/ufo/replay_draw_new_ratio": {
      "n": 64,
      "first_step": 24,
      "last_step": 1024,
      "first": 1.8823529481887817,
      "last": 13.65333366394043,
      "mean": 10.989829452708364,
      "min": 1.8823529481887817,
      "max": 13.683501243591309,
      "median": 12.05649709701538,
      "p95": 13.602520513534547,
      "first_window_mean": 10.291056768894196,
      "last_window_mean": 12.287729167938233,
      "window_points": 50
    },
    "train/ufo/replay_draws": {
      "n": 64,
      "first_step": 24,
      "last_step": 1024,
      "first": 24576.0,
      "last": 1048576.0,
      "mean": 540544.0,
      "min": 24576.0,
      "max": 1048576.0,
      "median": 540672.0,
      "p95": 1005158.3999999999,
      "first_window_mean": 425984.0,
      "last_window_mean": 655196.16,
      "window_points": 50
    },
    "train/ufo/replay_policy_age_max_updates": {
      "n": 64,
      "first_step": 24,
      "last_step": 1024,
      "first": 24.0,
      "last": 1024.0,
      "mean": 527.875,
      "min": 24.0,
      "max": 1024.0,
      "median": 528.0,
      "p95": 981.5999999999999,
      "first_window_mean": 416.0,
      "last_window_mean": 639.84,
      "window_points": 50
    },
    "train/ufo/replay_policy_age_min_updates": {
      "n": 64,
      "first_step": 24,
      "last_step": 1024,
      "first": 0.0,
      "last": 8.0,
      "mean": 15.625,
      "min": 0.0,
      "max": 16.0,
      "median": 16.0,
      "p95": 16.0,
      "first_window_mean": 15.68,
      "last_window_mean": 15.84,
      "window_points": 50
    },
    "train/ufo/replay_retained_fraction": {
      "n": 64,
      "first_step": 24,
      "last_step": 1024,
      "first": 1.0,
      "last": 1.0,
      "mean": 1.0,
      "min": 1.0,
      "max": 1.0,
      "median": 1.0,
      "p95": 1.0,
      "first_window_mean": 1.0,
      "last_window_mean": 1.0,
      "window_points": 50
    },
    "train/ufo/replay_size": {
      "n": 64,
      "first_step": 24,
      "last_step": 1024,
      "first": 13056.0,
      "last": 76800.0,
      "mean": 44808.0,
      "min": 13056.0,
      "max": 76800.0,
      "median": 44928.0,
      "p95": 73612.79999999999,
      "first_window_mean": 37647.36,
      "last_window_mean": 51962.88,
      "window_points": 50
    },
    "train/ufo/requested_action_abs_mean": {
      "n": 64,
      "first_step": 24,
      "last_step": 1024,
      "first": 0.885714054107666,
      "last": 0.9147884845733643,
      "mean": 0.9137543374672532,
      "min": 0.8717551231384277,
      "max": 0.9225698113441467,
      "median": 0.9158910512924194,
      "p95": 0.9216488778591156,
      "first_window_mean": 0.9142167413234711,
      "last_window_mean": 0.9160782694816589,
      "window_points": 50
    },
    "train/ufo/requested_action_saturation_fraction": {
      "n": 64,
      "first_step": 24,
      "last_step": 1024,
      "first": 0.4099864065647125,
      "last": 0.45646512508392334,
      "mean": 0.45848724618554115,
      "min": 0.37879303097724915,
      "max": 0.47554346919059753,
      "median": 0.46300384402275085,
      "p95": 0.47226561456918714,
      "first_window_mean": 0.4590715569257736,
      "last_window_mean": 0.4630196988582611,
      "window_points": 50
    },
    "train/ufo/requested_applied_max_diff": {
      "n": 64,
      "first_step": 24,
      "last_step": 1024,
      "first": 1.8933322429656982,
      "last": 1.8933322429656982,
      "mean": 1.8931175507605076,
      "min": 1.8795928955078125,
      "max": 1.8933322429656982,
      "median": 1.8933322429656982,
      "p95": 1.8933322429656982,
      "first_window_mean": 1.8933322429656982,
      "last_window_mean": 1.8930574369430542,
      "window_points": 50
    },
    "train/ufo/reset_control_samples": {
      "n": 64,
      "first_step": 24,
      "last_step": 1024,
      "first": 193.0,
      "last": 36.0,
      "mean": 52.21875,
      "min": 17.0,
      "max": 193.0,
      "median": 49.0,
      "p95": 76.85,
      "first_window_mean": 55.86,
      "last_window_mean": 46.18,
      "window_points": 50
    },
    "train/ufo/reset_projection_outside_fraction": {
      "n": 64,
      "first_step": 24,
      "last_step": 1024,
      "first": 0.787564754486084,
      "last": 0.75,
      "mean": 0.8203688701614738,
      "min": 0.7027027010917664,
      "max": 0.9411764740943909,
      "median": 0.8240454196929932,
      "p95": 0.892879244685173,
      "first_window_mean": 0.8242511308193207,
      "last_window_mean": 0.820681185722351,
      "window_points": 50
    },
    "train/ufo/reset_projection_residual_max_rad": {
      "n": 64,
      "first_step": 24,
      "last_step": 1024,
      "first": 1.4034202098846436,
      "last": 1.1026034355163574,
      "mean": 1.1258680522441864,
      "min": 0.8795425891876221,
      "max": 1.7291607856750488,
      "median": 1.0700000524520874,
      "p95": 1.4842845737934112,
      "first_window_mean": 1.1321829771995544,
      "last_window_mean": 1.1055713844299317,
      "window_points": 50
    },
    "train/ufo/reset_projection_residual_mean_rad": {
      "n": 64,
      "first_step": 24,
      "last_step": 1024,
      "first": 0.36163192987442017,
      "last": 0.3364742398262024,
      "mean": 0.3801990377251059,
      "min": 0.23102863132953644,
      "max": 0.5013399720191956,
      "median": 0.3820967972278595,
      "p95": 0.47550826519727707,
      "first_window_mean": 0.384086846113205,
      "last_window_mean": 0.38518272548913957,
      "window_points": 50
    },
    "train/ufo/reset_projection_residual_p95_rad": {
      "n": 64,
      "first_step": 24,
      "last_step": 1024,
      "first": 0.8799999952316284,
      "last": 0.9745765924453735,
      "mean": 0.8938528373837471,
      "min": 0.8154979944229126,
      "max": 1.0914385318756104,
      "median": 0.8799999952316284,
      "p95": 0.9897753208875655,
      "first_window_mean": 0.8929175901412963,
      "last_window_mean": 0.8937686169147492,
      "window_points": 50
    },
    "train/ufo/sampling_rounds": {
      "n": 1008,
      "first_step": 17,
      "last_step": 1024,
      "first": 16.0,
      "last": 100.0,
      "mean": 58.3343253968254,
      "min": 16.0,
      "max": 100.0,
      "median": 58.5,
      "p95": 96.0,
      "first_window_mean": 18.42,
      "last_window_mean": 98.24,
      "window_points": 50
    },
    "train/ufo/target_M": {
      "n": 1008,
      "first_step": 17,
      "last_step": 1024,
      "first": -2.056060791015625,
      "last": 1.6744836568832397,
      "mean": 0.7455948341271973,
      "min": -2.71756911277771,
      "max": 2.11360502243042,
      "median": 0.8515777587890625,
      "p95": 1.4655028104782104,
      "first_window_mean": -1.505100117623806,
      "last_window_mean": 1.447259373664856,
      "window_points": 50
    },
    "train/ufo/target_Q": {
      "n": 1008,
      "first_step": 17,
      "last_step": 1024,
      "first": -0.4484046697616577,
      "last": -16.947078704833984,
      "mean": -7.734461769145647,
      "min": -17.069435119628906,
      "max": 0.16366145014762878,
      "median": -7.363377094268799,
      "p95": -0.36216563284397135,
      "first_window_mean": -0.04911110509186983,
      "last_window_mean": -16.468464069366455,
      "window_points": 50
    },
    "train/ufo/target_auxQ": {
      "n": 1008,
      "first_step": 17,
      "last_step": 1024,
      "first": -2.000433921813965,
      "last": -8.918633460998535,
      "mean": -5.3141891781299835,
      "min": -8.96221923828125,
      "max": -1.7604517936706543,
      "median": -5.2383153438568115,
      "p95": -2.2338439941406256,
      "first_window_mean": -1.9634960889816284,
      "last_window_mean": -8.594278678894042,
      "window_points": 50
    },
    "train/ufo/terminated_transitions": {
      "n": 64,
      "first_step": 24,
      "last_step": 1024,
      "first": 4.0,
      "last": 29.0,
      "mean": 48.5625,
      "min": 4.0,
      "max": 92.0,
      "median": 47.5,
      "p95": 76.54999999999998,
      "first_window_mean": 51.28,
      "last_window_mean": 45.58,
      "window_points": 50
    },
    "train/ufo/truncated_transitions": {
      "n": 64,
      "first_step": 24,
      "last_step": 1024,
      "first": 0.0,
      "last": 0.0,
      "mean": 0.0,
      "min": 0.0,
      "max": 0.0,
      "median": 0.0,
      "p95": 0.0,
      "first_window_mean": 0.0,
      "last_window_mean": 0.0,
      "window_points": 50
    },
    "train/ufo/unc_Q": {
      "n": 1008,
      "first_step": 17,
      "last_step": 1024,
      "first": 1.3422057628631592,
      "last": 0.45735663175582886,
      "mean": 0.4552342565582385,
      "min": 0.3311373293399811,
      "max": 1.3422057628631592,
      "median": 0.4018319845199585,
      "p95": 0.7626259803771973,
      "first_window_mean": 0.9841370785236359,
      "last_window_mean": 0.4532763761281967,
      "window_points": 50
    },
    "train/ufo/unc_auxQ": {
      "n": 1008,
      "first_step": 17,
      "last_step": 1024,
      "first": 1.2780282497406006,
      "last": 0.3616941571235657,
      "mean": 0.4346759615259038,
      "min": 0.31506770849227905,
      "max": 1.2780282497406006,
      "median": 0.36520491540431976,
      "p95": 0.7626506954431531,
      "first_window_mean": 0.9522501575946808,
      "last_window_mean": 0.3588866001367569,
      "window_points": 50
    },
    "train/ufo/wall_time_s": {
      "n": 1008,
      "first_step": 17,
      "last_step": 1024,
      "first": 32.54937744140625,
      "last": 1878.7857666015625,
      "mean": 951.2839654948976,
      "min": 30.115720748901367,
      "max": 1878.7857666015625,
      "median": 953.1942443847656,
      "p95": 1789.1408386230469,
      "first_window_mean": 69.20117656707764,
      "last_window_mean": 1838.531865234375,
      "window_points": 50
    },
    "train/ufo/z_norm": {
      "n": 1008,
      "first_step": 17,
      "last_step": 1024,
      "first": 16.0,
      "last": 16.0,
      "mean": 16.0,
      "min": 16.0,
      "max": 16.0,
      "median": 16.0,
      "p95": 16.0,
      "first_window_mean": 16.0,
      "last_window_mean": 16.0,
      "window_points": 50
    }
  }
}
```

![D训练时序图](/data/kk/EmbodiedBein/docs/progress/artifacts/2026-10-11-ufo-walking-repair/slew1024-training.png)

## 独立16case完整结果

```json
{
  "cases": 16,
  "successes": 0,
  "median_steps": 35.0,
  "mean_steps": 32.5,
  "min_steps": 2,
  "max_steps": 50,
  "steps_by_seed": [
    25,
    35,
    28,
    38,
    46,
    40,
    33,
    30,
    50,
    48,
    15,
    35,
    2,
    48,
    36,
    11
  ],
  "median_physical_s": 0.7000000000000001,
  "fall_count": 11,
  "hard_limit_count": 5
}
```

| Seed | 步数 | 物理秒 | 原因 | 速度RMSE m/s | yaw RMSE rad/s |
|---|---:|---:|---|---:|---:|
| 61000 | 25 | 0.50 | physical_hard_limit | 0.706339 | 1.808334 |
| 61001 | 35 | 0.70 | fall | 0.858038 | 1.560684 |
| 61002 | 28 | 0.56 | fall | 0.566601 | 1.654044 |
| 61003 | 38 | 0.76 | fall | 1.187349 | 1.467422 |
| 61004 | 46 | 0.92 | fall | 1.154030 | 2.105813 |
| 61005 | 40 | 0.80 | fall | 1.001721 | 2.340656 |
| 61006 | 33 | 0.66 | fall | 0.768676 | 1.854872 |
| 61007 | 30 | 0.60 | fall | 1.115733 | 2.151694 |
| 61008 | 50 | 1.00 | fall | 1.096955 | 1.864095 |
| 61009 | 48 | 0.96 | fall | 0.798465 | 1.683123 |
| 61010 | 15 | 0.30 | physical_hard_limit | 1.007364 | 2.130249 |
| 61011 | 35 | 0.70 | fall | 0.895425 | 1.911903 |
| 61012 | 2 | 0.04 | physical_hard_limit | 0.680256 | 3.308536 |
| 61013 | 48 | 0.96 | fall | 0.543209 | 1.962331 |
| 61014 | 36 | 0.72 | physical_hard_limit | 0.356892 | 2.072587 |
| 61015 | 11 | 0.22 | physical_hard_limit | 1.145456 | 2.604302 |

![D1024逐例存活与D24对照](/data/kk/EmbodiedBein/docs/progress/artifacts/2026-10-11-ufo-walking-repair/slew1024-validation.png)

每例因提前失败不满足1000步。速度/yaw RMSE由实际短片段计算；误差变化不能当作完整20秒跟踪改善。0成功，无成功录像；16失败录像全部hash/解码与HTTP Range206验证通过。

```json
{
  "report_id": "ev_f31fcc52ce1d4dda9ea0",
  "job_id": "job_c11e3a285cfd4c1b990d",
  "run_id": "run_bf9c6e29de7f42578791",
  "plan_hash": "sha256:083d4f8fb1699e87d513468694643a008114705a6f3c170d6769966da0774024",
  "controller_all_520_steps": {
    "cases": [
      {
        "seed": 61000,
        "steps": 25,
        "shared_controller_verified": true
      },
      {
        "seed": 61001,
        "steps": 35,
        "shared_controller_verified": true
      },
      {
        "seed": 61002,
        "steps": 28,
        "shared_controller_verified": true
      },
      {
        "seed": 61003,
        "steps": 38,
        "shared_controller_verified": true
      },
      {
        "seed": 61004,
        "steps": 46,
        "shared_controller_verified": true
      },
      {
        "seed": 61005,
        "steps": 40,
        "shared_controller_verified": true
      },
      {
        "seed": 61006,
        "steps": 33,
        "shared_controller_verified": true
      },
      {
        "seed": 61007,
        "steps": 30,
        "shared_controller_verified": true
      },
      {
        "seed": 61008,
        "steps": 50,
        "shared_controller_verified": true
      },
      {
        "seed": 61009,
        "steps": 48,
        "shared_controller_verified": true
      },
      {
        "seed": 61010,
        "steps": 15,
        "shared_controller_verified": true
      },
      {
        "seed": 61011,
        "steps": 35,
        "shared_controller_verified": true
      },
      {
        "seed": 61012,
        "steps": 2,
        "shared_controller_verified": true
      },
      {
        "seed": 61013,
        "steps": 48,
        "shared_controller_verified": true
      },
      {
        "seed": 61014,
        "steps": 36,
        "shared_controller_verified": true
      },
      {
        "seed": 61015,
        "steps": 11,
        "shared_controller_verified": true
      }
    ],
    "steps": 520,
    "requested_saturation_fraction": 0.7112876254180602,
    "applied_saturation_fraction": 0.47399665551839465,
    "requested_applied_abs_max_diff": 1.8393425941467285
  },
  "cases": [
    {
      "seed": 61000,
      "case_id": "rpo-bfm-seed-61000",
      "task_id": "tsk_rpo_bfm_pretrain",
      "completed": true,
      "success": false,
      "actual_steps": 25,
      "episode_len": 25,
      "fall": false,
      "nan": false,
      "hard_limit": true,
      "reason": "physical_hard_limit",
      "env_index": 0,
      "terrain_seed": 64000,
      "initial_state": {
        "path": "traces/seed_61000.initial.json",
        "sha256": "8fd42de79fb962d6dc900e53bd9b6be622f5f1507d2a738d1a4cd42599ed0b7f"
      },
      "video_rel": "seed_61000/0.mp4",
      "metric_steps": 25,
      "metrics_available": true,
      "velocity_rmse": 0.7063394354324458,
      "yaw_rmse": 1.8083336741802396,
      "status": "task_failure",
      "termination_reason": "physical_hard_limit",
      "trace": {
        "path": "traces/seed_61000.jsonl.gz",
        "sha256": "7e11099cb3f847b82a191a073bc5eb39bfbda63396217c8416cb2477fa8a238e",
        "policy_steps": 25,
        "metric_steps": 25
      },
      "video_integrity": {
        "schema_version": "eb/bfm-case-video/v1",
        "sha256": "28589392a0f6f25c8529f9aac7ff81cf606ded1331ffd240d777a3f73abcf764",
        "decoded_frames": 6,
        "resolution": [
          320,
          240
        ],
        "fps": 10
      },
      "worker_rank": 0,
      "run_id": "run_bf9c6e29de7f42578791",
      "plan_hash": "sha256:083d4f8fb1699e87d513468694643a008114705a6f3c170d6769966da0774024",
      "video_status": "available"
    },
    {
      "seed": 61001,
      "case_id": "rpo-bfm-seed-61001",
      "task_id": "tsk_rpo_bfm_pretrain",
      "completed": true,
      "success": false,
      "actual_steps": 35,
      "episode_len": 35,
      "fall": true,
      "nan": false,
      "hard_limit": false,
      "reason": "fall",
      "env_index": 0,
      "terrain_seed": 64000,
      "initial_state": {
        "path": "traces/seed_61001.initial.json",
        "sha256": "97d456e1f29fe698eee04ef76b3c246e7b4be4e493573b7971eaac4a27b1bc8e"
      },
      "video_rel": "seed_61001/0.mp4",
      "metric_steps": 35,
      "metrics_available": true,
      "velocity_rmse": 0.8580381012225213,
      "yaw_rmse": 1.5606835609716083,
      "status": "task_failure",
      "termination_reason": "fall",
      "trace": {
        "path": "traces/seed_61001.jsonl.gz",
        "sha256": "a4f8a58e830d29c438d65e13b023370076d5e4362226b9334cd1899745f3e9e1",
        "policy_steps": 35,
        "metric_steps": 35
      },
      "video_integrity": {
        "schema_version": "eb/bfm-case-video/v1",
        "sha256": "0638799e32b68998e4737e78f682cee6226444e52bb89c260184ff7d2d1fb422",
        "decoded_frames": 8,
        "resolution": [
          320,
          240
        ],
        "fps": 10
      },
      "worker_rank": 1,
      "run_id": "run_bf9c6e29de7f42578791",
      "plan_hash": "sha256:083d4f8fb1699e87d513468694643a008114705a6f3c170d6769966da0774024",
      "video_status": "available"
    },
    {
      "seed": 61002,
      "case_id": "rpo-bfm-seed-61002",
      "task_id": "tsk_rpo_bfm_pretrain",
      "completed": true,
      "success": false,
      "actual_steps": 28,
      "episode_len": 28,
      "fall": true,
      "nan": false,
      "hard_limit": false,
      "reason": "fall",
      "env_index": 0,
      "terrain_seed": 64000,
      "initial_state": {
        "path": "traces/seed_61002.initial.json",
        "sha256": "770dd8c19384f14bdd7c960cb4d1da0c5c1ad91bfa0e82bcafcad61ebef0ad32"
      },
      "video_rel": "seed_61002/0.mp4",
      "metric_steps": 28,
      "metrics_available": true,
      "velocity_rmse": 0.5666010064717211,
      "yaw_rmse": 1.654043678029252,
      "status": "task_failure",
      "termination_reason": "fall",
      "trace": {
        "path": "traces/seed_61002.jsonl.gz",
        "sha256": "0722e4b42033e4b4acdaaf81b8a4f541cc6c050891b50edb2b72443076c8b723",
        "policy_steps": 28,
        "metric_steps": 28
      },
      "video_integrity": {
        "schema_version": "eb/bfm-case-video/v1",
        "sha256": "8ecc240e2fa7a22361242ccd1c6c9d64ff0fc991549304c65d1ef9bfd0f8b3fa",
        "decoded_frames": 7,
        "resolution": [
          320,
          240
        ],
        "fps": 10
      },
      "worker_rank": 2,
      "run_id": "run_bf9c6e29de7f42578791",
      "plan_hash": "sha256:083d4f8fb1699e87d513468694643a008114705a6f3c170d6769966da0774024",
      "video_status": "available"
    },
    {
      "seed": 61003,
      "case_id": "rpo-bfm-seed-61003",
      "task_id": "tsk_rpo_bfm_pretrain",
      "completed": true,
      "success": false,
      "actual_steps": 38,
      "episode_len": 38,
      "fall": true,
      "nan": false,
      "hard_limit": false,
      "reason": "fall",
      "env_index": 0,
      "terrain_seed": 64000,
      "initial_state": {
        "path": "traces/seed_61003.initial.json",
        "sha256": "482630c9dc02ca963ea0e8e61a33763de5ef83d766fc0d75c0762dcfb13b7a96"
      },
      "video_rel": "seed_61003/0.mp4",
      "metric_steps": 38,
      "metrics_available": true,
      "velocity_rmse": 1.1873486438684282,
      "yaw_rmse": 1.4674216397274367,
      "status": "task_failure",
      "termination_reason": "fall",
      "trace": {
        "path": "traces/seed_61003.jsonl.gz",
        "sha256": "055722c175d2077de75dbbda74b27069f4edf212365ff748b45f1d85a34c16da",
        "policy_steps": 38,
        "metric_steps": 38
      },
      "video_integrity": {
        "schema_version": "eb/bfm-case-video/v1",
        "sha256": "ea7ad61351245a3ff9af7f57c87aa4b77496570dc9a2355e17310b08e0a36dd0",
        "decoded_frames": 9,
        "resolution": [
          320,
          240
        ],
        "fps": 10
      },
      "worker_rank": 3,
      "run_id": "run_bf9c6e29de7f42578791",
      "plan_hash": "sha256:083d4f8fb1699e87d513468694643a008114705a6f3c170d6769966da0774024",
      "video_status": "available"
    },
    {
      "seed": 61004,
      "case_id": "rpo-bfm-seed-61004",
      "task_id": "tsk_rpo_bfm_pretrain",
      "completed": true,
      "success": false,
      "actual_steps": 46,
      "episode_len": 46,
      "fall": true,
      "nan": false,
      "hard_limit": false,
      "reason": "fall",
      "env_index": 0,
      "terrain_seed": 64000,
      "initial_state": {
        "path": "traces/seed_61004.initial.json",
        "sha256": "04c6a979be46023b21b0e9a678f66f67b753f9edc91dec16fcad79eca82039af"
      },
      "video_rel": "seed_61004/0.mp4",
      "metric_steps": 46,
      "metrics_available": true,
      "velocity_rmse": 1.1540300000877075,
      "yaw_rmse": 2.1058133285250027,
      "status": "task_failure",
      "termination_reason": "fall",
      "trace": {
        "path": "traces/seed_61004.jsonl.gz",
        "sha256": "39b3bf36cf652672dbf9342d83ab42ce94880c0662e1ddbd4a579d465e618af7",
        "policy_steps": 46,
        "metric_steps": 46
      },
      "video_integrity": {
        "schema_version": "eb/bfm-case-video/v1",
        "sha256": "4fbe1a3fd4bc7db128f57f4db55456c4aa7b9e96e5f3ba12c643abfd4d44c5c9",
        "decoded_frames": 11,
        "resolution": [
          320,
          240
        ],
        "fps": 10
      },
      "worker_rank": 0,
      "run_id": "run_bf9c6e29de7f42578791",
      "plan_hash": "sha256:083d4f8fb1699e87d513468694643a008114705a6f3c170d6769966da0774024",
      "video_status": "available"
    },
    {
      "seed": 61005,
      "case_id": "rpo-bfm-seed-61005",
      "task_id": "tsk_rpo_bfm_pretrain",
      "completed": true,
      "success": false,
      "actual_steps": 40,
      "episode_len": 40,
      "fall": true,
      "nan": false,
      "hard_limit": false,
      "reason": "fall",
      "env_index": 0,
      "terrain_seed": 64000,
      "initial_state": {
        "path": "traces/seed_61005.initial.json",
        "sha256": "1413389dfe8c22d5e619e7615fa9c5b734dc6b7ca9a0c5aa1dff19f293911695"
      },
      "video_rel": "seed_61005/0.mp4",
      "metric_steps": 40,
      "metrics_available": true,
      "velocity_rmse": 1.0017205671013723,
      "yaw_rmse": 2.3406555343665927,
      "status": "task_failure",
      "termination_reason": "fall",
      "trace": {
        "path": "traces/seed_61005.jsonl.gz",
        "sha256": "5420bbc418df8e1cc96fb96e01184b0e75ae402e75a8c946bf5ee5cc2aa053e3",
        "policy_steps": 40,
        "metric_steps": 40
      },
      "video_integrity": {
        "schema_version": "eb/bfm-case-video/v1",
        "sha256": "585fd941c8cb913259a345d25c1c128df20680066b078da2d1423dda44d090a6",
        "decoded_frames": 9,
        "resolution": [
          320,
          240
        ],
        "fps": 10
      },
      "worker_rank": 1,
      "run_id": "run_bf9c6e29de7f42578791",
      "plan_hash": "sha256:083d4f8fb1699e87d513468694643a008114705a6f3c170d6769966da0774024",
      "video_status": "available"
    },
    {
      "seed": 61006,
      "case_id": "rpo-bfm-seed-61006",
      "task_id": "tsk_rpo_bfm_pretrain",
      "completed": true,
      "success": false,
      "actual_steps": 33,
      "episode_len": 33,
      "fall": true,
      "nan": false,
      "hard_limit": false,
      "reason": "fall",
      "env_index": 0,
      "terrain_seed": 64000,
      "initial_state": {
        "path": "traces/seed_61006.initial.json",
        "sha256": "f8a7fc9b647d1a573de0bbc5f32dd29777cd59166c8e0f8cd38223e775ccaec8"
      },
      "video_rel": "seed_61006/0.mp4",
      "metric_steps": 33,
      "metrics_available": true,
      "velocity_rmse": 0.7686761226303411,
      "yaw_rmse": 1.8548721008720503,
      "status": "task_failure",
      "termination_reason": "fall",
      "trace": {
        "path": "traces/seed_61006.jsonl.gz",
        "sha256": "482e8afdf35710f5cdae2e9e5de2e891c686b1be20059304e4962a8fdfeb040c",
        "policy_steps": 33,
        "metric_steps": 33
      },
      "video_integrity": {
        "schema_version": "eb/bfm-case-video/v1",
        "sha256": "1752d62d5e5fe6fa5618a43751a6c4fdef3c8a5dda90d8671cf0e7b97a2b7989",
        "decoded_frames": 8,
        "resolution": [
          320,
          240
        ],
        "fps": 10
      },
      "worker_rank": 2,
      "run_id": "run_bf9c6e29de7f42578791",
      "plan_hash": "sha256:083d4f8fb1699e87d513468694643a008114705a6f3c170d6769966da0774024",
      "video_status": "available"
    },
    {
      "seed": 61007,
      "case_id": "rpo-bfm-seed-61007",
      "task_id": "tsk_rpo_bfm_pretrain",
      "completed": true,
      "success": false,
      "actual_steps": 30,
      "episode_len": 30,
      "fall": true,
      "nan": false,
      "hard_limit": false,
      "reason": "fall",
      "env_index": 0,
      "terrain_seed": 64000,
      "initial_state": {
        "path": "traces/seed_61007.initial.json",
        "sha256": "b5e263d99bb3679cab4e5e8d139b79362b4f5c52f6dcc7d84ed5cc1da9c2258e"
      },
      "video_rel": "seed_61007/0.mp4",
      "metric_steps": 30,
      "metrics_available": true,
      "velocity_rmse": 1.1157325566792635,
      "yaw_rmse": 2.1516936007093386,
      "status": "task_failure",
      "termination_reason": "fall",
      "trace": {
        "path": "traces/seed_61007.jsonl.gz",
        "sha256": "814f6be29b6fe3b103e33ff9b0af033b27553e09d35e95ecc3e221290e9e9042",
        "policy_steps": 30,
        "metric_steps": 30
      },
      "video_integrity": {
        "schema_version": "eb/bfm-case-video/v1",
        "sha256": "44b9b3fc1f244d55408282672d1683d16ed0935029dd372d5223339040db24af",
        "decoded_frames": 7,
        "resolution": [
          320,
          240
        ],
        "fps": 10
      },
      "worker_rank": 3,
      "run_id": "run_bf9c6e29de7f42578791",
      "plan_hash": "sha256:083d4f8fb1699e87d513468694643a008114705a6f3c170d6769966da0774024",
      "video_status": "available"
    },
    {
      "seed": 61008,
      "case_id": "rpo-bfm-seed-61008",
      "task_id": "tsk_rpo_bfm_pretrain",
      "completed": true,
      "success": false,
      "actual_steps": 50,
      "episode_len": 50,
      "fall": true,
      "nan": false,
      "hard_limit": false,
      "reason": "fall",
      "env_index": 0,
      "terrain_seed": 64000,
      "initial_state": {
        "path": "traces/seed_61008.initial.json",
        "sha256": "e8f1583ee2f913cd14eace8d0d8d02e52fc08a0237590d80062dbfa1670a94ef"
      },
      "video_rel": "seed_61008/0.mp4",
      "metric_steps": 50,
      "metrics_available": true,
      "velocity_rmse": 1.0969549075149765,
      "yaw_rmse": 1.8640949587646156,
      "status": "task_failure",
      "termination_reason": "fall",
      "trace": {
        "path": "traces/seed_61008.jsonl.gz",
        "sha256": "0c9fd6ddabf827854b64ceb57bebbd136e413c13ee8aae0c21fa5b9c9c128d69",
        "policy_steps": 50,
        "metric_steps": 50
      },
      "video_integrity": {
        "schema_version": "eb/bfm-case-video/v1",
        "sha256": "a2d7a525ec1ad9e344e234f7a48aa1be808b32b7116414ab0ae2044d470cc94c",
        "decoded_frames": 11,
        "resolution": [
          320,
          240
        ],
        "fps": 10
      },
      "worker_rank": 0,
      "run_id": "run_bf9c6e29de7f42578791",
      "plan_hash": "sha256:083d4f8fb1699e87d513468694643a008114705a6f3c170d6769966da0774024",
      "video_status": "available"
    },
    {
      "seed": 61009,
      "case_id": "rpo-bfm-seed-61009",
      "task_id": "tsk_rpo_bfm_pretrain",
      "completed": true,
      "success": false,
      "actual_steps": 48,
      "episode_len": 48,
      "fall": true,
      "nan": false,
      "hard_limit": false,
      "reason": "fall",
      "env_index": 0,
      "terrain_seed": 64000,
      "initial_state": {
        "path": "traces/seed_61009.initial.json",
        "sha256": "2534af0884ae55d39617097a786228605e1cb7e69389458b8b4dea74f97a8dc8"
      },
      "video_rel": "seed_61009/0.mp4",
      "metric_steps": 48,
      "metrics_available": true,
      "velocity_rmse": 0.7984653979670564,
      "yaw_rmse": 1.6831225160617722,
      "status": "task_failure",
      "termination_reason": "fall",
      "trace": {
        "path": "traces/seed_61009.jsonl.gz",
        "sha256": "114150fd60ed830d17c82c5d7c892d2001e72ed309e2bf647aab98dd18b47cf0",
        "policy_steps": 48,
        "metric_steps": 48
      },
      "video_integrity": {
        "schema_version": "eb/bfm-case-video/v1",
        "sha256": "e413dd486a089e47e1413ffcea9494ec85e8df591065023c7e2ce4418fb5fc7a",
        "decoded_frames": 11,
        "resolution": [
          320,
          240
        ],
        "fps": 10
      },
      "worker_rank": 1,
      "run_id": "run_bf9c6e29de7f42578791",
      "plan_hash": "sha256:083d4f8fb1699e87d513468694643a008114705a6f3c170d6769966da0774024",
      "video_status": "available"
    },
    {
      "seed": 61010,
      "case_id": "rpo-bfm-seed-61010",
      "task_id": "tsk_rpo_bfm_pretrain",
      "completed": true,
      "success": false,
      "actual_steps": 15,
      "episode_len": 15,
      "fall": false,
      "nan": false,
      "hard_limit": true,
      "reason": "physical_hard_limit",
      "env_index": 0,
      "terrain_seed": 64000,
      "initial_state": {
        "path": "traces/seed_61010.initial.json",
        "sha256": "af3731e48774c211287ed2c5262a29e88bc8e45904525dd0d6524ebfc54a8fc6"
      },
      "video_rel": "seed_61010/0.mp4",
      "metric_steps": 15,
      "metrics_available": true,
      "velocity_rmse": 1.0073643484112742,
      "yaw_rmse": 2.1302488171208696,
      "status": "task_failure",
      "termination_reason": "physical_hard_limit",
      "trace": {
        "path": "traces/seed_61010.jsonl.gz",
        "sha256": "f51e7f6fbc0c39b099f0de558da7d242d5c1d1e98bd9ef12804e8db0c3e03aa8",
        "policy_steps": 15,
        "metric_steps": 15
      },
      "video_integrity": {
        "schema_version": "eb/bfm-case-video/v1",
        "sha256": "f0b2dbfd87ea8da75a824ebcea9cabcfc4a8a9388eb1016a2d71f0c7d98c9e2f",
        "decoded_frames": 4,
        "resolution": [
          320,
          240
        ],
        "fps": 10
      },
      "worker_rank": 2,
      "run_id": "run_bf9c6e29de7f42578791",
      "plan_hash": "sha256:083d4f8fb1699e87d513468694643a008114705a6f3c170d6769966da0774024",
      "video_status": "available"
    },
    {
      "seed": 61011,
      "case_id": "rpo-bfm-seed-61011",
      "task_id": "tsk_rpo_bfm_pretrain",
      "completed": true,
      "success": false,
      "actual_steps": 35,
      "episode_len": 35,
      "fall": true,
      "nan": false,
      "hard_limit": false,
      "reason": "fall",
      "env_index": 0,
      "terrain_seed": 64000,
      "initial_state": {
        "path": "traces/seed_61011.initial.json",
        "sha256": "d55045b6671629afab055e8c07a1e7717131ea6c1087df3a0085fbfc8c698a10"
      },
      "video_rel": "seed_61011/0.mp4",
      "metric_steps": 35,
      "metrics_available": true,
      "velocity_rmse": 0.8954252733796513,
      "yaw_rmse": 1.911902638133691,
      "status": "task_failure",
      "termination_reason": "fall",
      "trace": {
        "path": "traces/seed_61011.jsonl.gz",
        "sha256": "68bb0d8d2c02e8f8719722833b954754857d2af1b1e81f49400192c463260daa",
        "policy_steps": 35,
        "metric_steps": 35
      },
      "video_integrity": {
        "schema_version": "eb/bfm-case-video/v1",
        "sha256": "4d487a65b4ce55a14141037489ca0b2e46355d527a0c356c9f59e554dccd2d4c",
        "decoded_frames": 8,
        "resolution": [
          320,
          240
        ],
        "fps": 10
      },
      "worker_rank": 3,
      "run_id": "run_bf9c6e29de7f42578791",
      "plan_hash": "sha256:083d4f8fb1699e87d513468694643a008114705a6f3c170d6769966da0774024",
      "video_status": "available"
    },
    {
      "seed": 61012,
      "case_id": "rpo-bfm-seed-61012",
      "task_id": "tsk_rpo_bfm_pretrain",
      "completed": true,
      "success": false,
      "actual_steps": 2,
      "episode_len": 2,
      "fall": false,
      "nan": false,
      "hard_limit": true,
      "reason": "physical_hard_limit",
      "env_index": 0,
      "terrain_seed": 64000,
      "initial_state": {
        "path": "traces/seed_61012.initial.json",
        "sha256": "0ca7c15c19854635c6e3f0a82e45b6f9a713237a08b22396e980ce181aa75a9e"
      },
      "video_rel": "seed_61012/0.mp4",
      "metric_steps": 2,
      "metrics_available": true,
      "velocity_rmse": 0.6802555970173816,
      "yaw_rmse": 3.308535600544121,
      "status": "task_failure",
      "termination_reason": "physical_hard_limit",
      "trace": {
        "path": "traces/seed_61012.jsonl.gz",
        "sha256": "b9b96b46d3039682f0ea26991f494adc17a3acb6ad9be5d4f958f2038f9f5e0c",
        "policy_steps": 2,
        "metric_steps": 2
      },
      "video_integrity": {
        "schema_version": "eb/bfm-case-video/v1",
        "sha256": "3c3a1816ef044beaa3f67d03888b80cfeacf386d6ced6197d16b8cf09653a8ee",
        "decoded_frames": 2,
        "resolution": [
          320,
          240
        ],
        "fps": 10
      },
      "worker_rank": 0,
      "run_id": "run_bf9c6e29de7f42578791",
      "plan_hash": "sha256:083d4f8fb1699e87d513468694643a008114705a6f3c170d6769966da0774024",
      "video_status": "available"
    },
    {
      "seed": 61013,
      "case_id": "rpo-bfm-seed-61013",
      "task_id": "tsk_rpo_bfm_pretrain",
      "completed": true,
      "success": false,
      "actual_steps": 48,
      "episode_len": 48,
      "fall": true,
      "nan": false,
      "hard_limit": false,
      "reason": "fall",
      "env_index": 0,
      "terrain_seed": 64000,
      "initial_state": {
        "path": "traces/seed_61013.initial.json",
        "sha256": "639503162918c79b65151c955f18a9d2f34e9cf5d344ab955490b163c73a774a"
      },
      "video_rel": "seed_61013/0.mp4",
      "metric_steps": 48,
      "metrics_available": true,
      "velocity_rmse": 0.5432086450775832,
      "yaw_rmse": 1.9623310190782506,
      "status": "task_failure",
      "termination_reason": "fall",
      "trace": {
        "path": "traces/seed_61013.jsonl.gz",
        "sha256": "3e661fd103025ae1d6502512aa0955dae035ce735940e35e7a52d1751519bf02",
        "policy_steps": 48,
        "metric_steps": 48
      },
      "video_integrity": {
        "schema_version": "eb/bfm-case-video/v1",
        "sha256": "579d33395cfd540cf883ab7e4077f211d40f944a865963280308c3f43f632a63",
        "decoded_frames": 11,
        "resolution": [
          320,
          240
        ],
        "fps": 10
      },
      "worker_rank": 1,
      "run_id": "run_bf9c6e29de7f42578791",
      "plan_hash": "sha256:083d4f8fb1699e87d513468694643a008114705a6f3c170d6769966da0774024",
      "video_status": "available"
    },
    {
      "seed": 61014,
      "case_id": "rpo-bfm-seed-61014",
      "task_id": "tsk_rpo_bfm_pretrain",
      "completed": true,
      "success": false,
      "actual_steps": 36,
      "episode_len": 36,
      "fall": false,
      "nan": false,
      "hard_limit": true,
      "reason": "physical_hard_limit",
      "env_index": 0,
      "terrain_seed": 64000,
      "initial_state": {
        "path": "traces/seed_61014.initial.json",
        "sha256": "af54b5aa2e1dd258b9cea4631d294aedc3ecf1b89669d778e9761020c9190ce4"
      },
      "video_rel": "seed_61014/0.mp4",
      "metric_steps": 36,
      "metrics_available": true,
      "velocity_rmse": 0.3568921927932474,
      "yaw_rmse": 2.0725865985770766,
      "status": "task_failure",
      "termination_reason": "physical_hard_limit",
      "trace": {
        "path": "traces/seed_61014.jsonl.gz",
        "sha256": "b3ae8a1832e3196a833efb42c010f2363066b3941afb2079068a6a1da5fe66c7",
        "policy_steps": 36,
        "metric_steps": 36
      },
      "video_integrity": {
        "schema_version": "eb/bfm-case-video/v1",
        "sha256": "c50b6a8fc5adeae9173cd4b7ac1ef03fb103ed84f921cab6e3ef1f9dc8b96eff",
        "decoded_frames": 9,
        "resolution": [
          320,
          240
        ],
        "fps": 10
      },
      "worker_rank": 2,
      "run_id": "run_bf9c6e29de7f42578791",
      "plan_hash": "sha256:083d4f8fb1699e87d513468694643a008114705a6f3c170d6769966da0774024",
      "video_status": "available"
    },
    {
      "seed": 61015,
      "case_id": "rpo-bfm-seed-61015",
      "task_id": "tsk_rpo_bfm_pretrain",
      "completed": true,
      "success": false,
      "actual_steps": 11,
      "episode_len": 11,
      "fall": false,
      "nan": false,
      "hard_limit": true,
      "reason": "physical_hard_limit",
      "env_index": 0,
      "terrain_seed": 64000,
      "initial_state": {
        "path": "traces/seed_61015.initial.json",
        "sha256": "2464b046e5ef3a59d7b81a50da66048e305146c137ed2dac9f7634077a4f342d"
      },
      "video_rel": "seed_61015/0.mp4",
      "metric_steps": 11,
      "metrics_available": true,
      "velocity_rmse": 1.1454558012208444,
      "yaw_rmse": 2.6043019347062826,
      "status": "task_failure",
      "termination_reason": "physical_hard_limit",
      "trace": {
        "path": "traces/seed_61015.jsonl.gz",
        "sha256": "a82c5689f5fe0f3df60cc3083f6763c24f9b77bbb7563fffb6b5ac5d6b820bc1",
        "policy_steps": 11,
        "metric_steps": 11
      },
      "video_integrity": {
        "schema_version": "eb/bfm-case-video/v1",
        "sha256": "4361df600343e43967d9bac1f1444e2154cbab5fa8cb551e63228d01adecf5e1",
        "decoded_frames": 4,
        "resolution": [
          320,
          240
        ],
        "fps": 10
      },
      "worker_rank": 3,
      "run_id": "run_bf9c6e29de7f42578791",
      "plan_hash": "sha256:083d4f8fb1699e87d513468694643a008114705a6f3c170d6769966da0774024",
      "video_status": "available"
    }
  ],
  "video_access": [
    {
      "seed": 61000,
      "http_status": 206,
      "sha256": "28589392a0f6f25c8529f9aac7ff81cf606ded1331ffd240d777a3f73abcf764",
      "decoded_frames": 6,
      "path": "61000/0"
    },
    {
      "seed": 61001,
      "http_status": 206,
      "sha256": "0638799e32b68998e4737e78f682cee6226444e52bb89c260184ff7d2d1fb422",
      "decoded_frames": 8,
      "path": "61001/0"
    },
    {
      "seed": 61002,
      "http_status": 206,
      "sha256": "8ecc240e2fa7a22361242ccd1c6c9d64ff0fc991549304c65d1ef9bfd0f8b3fa",
      "decoded_frames": 7,
      "path": "61002/0"
    },
    {
      "seed": 61003,
      "http_status": 206,
      "sha256": "ea7ad61351245a3ff9af7f57c87aa4b77496570dc9a2355e17310b08e0a36dd0",
      "decoded_frames": 9,
      "path": "61003/0"
    },
    {
      "seed": 61004,
      "http_status": 206,
      "sha256": "4fbe1a3fd4bc7db128f57f4db55456c4aa7b9e96e5f3ba12c643abfd4d44c5c9",
      "decoded_frames": 11,
      "path": "61004/0"
    },
    {
      "seed": 61005,
      "http_status": 206,
      "sha256": "585fd941c8cb913259a345d25c1c128df20680066b078da2d1423dda44d090a6",
      "decoded_frames": 9,
      "path": "61005/0"
    },
    {
      "seed": 61006,
      "http_status": 206,
      "sha256": "1752d62d5e5fe6fa5618a43751a6c4fdef3c8a5dda90d8671cf0e7b97a2b7989",
      "decoded_frames": 8,
      "path": "61006/0"
    },
    {
      "seed": 61007,
      "http_status": 206,
      "sha256": "44b9b3fc1f244d55408282672d1683d16ed0935029dd372d5223339040db24af",
      "decoded_frames": 7,
      "path": "61007/0"
    },
    {
      "seed": 61008,
      "http_status": 206,
      "sha256": "a2d7a525ec1ad9e344e234f7a48aa1be808b32b7116414ab0ae2044d470cc94c",
      "decoded_frames": 11,
      "path": "61008/0"
    },
    {
      "seed": 61009,
      "http_status": 206,
      "sha256": "e413dd486a089e47e1413ffcea9494ec85e8df591065023c7e2ce4418fb5fc7a",
      "decoded_frames": 11,
      "path": "61009/0"
    },
    {
      "seed": 61010,
      "http_status": 206,
      "sha256": "f0b2dbfd87ea8da75a824ebcea9cabcfc4a8a9388eb1016a2d71f0c7d98c9e2f",
      "decoded_frames": 4,
      "path": "61010/0"
    },
    {
      "seed": 61011,
      "http_status": 206,
      "sha256": "4d487a65b4ce55a14141037489ca0b2e46355d527a0c356c9f59e554dccd2d4c",
      "decoded_frames": 8,
      "path": "61011/0"
    },
    {
      "seed": 61012,
      "http_status": 206,
      "sha256": "3c3a1816ef044beaa3f67d03888b80cfeacf386d6ced6197d16b8cf09653a8ee",
      "decoded_frames": 2,
      "path": "61012/0"
    },
    {
      "seed": 61013,
      "http_status": 206,
      "sha256": "579d33395cfd540cf883ab7e4077f211d40f944a865963280308c3f43f632a63",
      "decoded_frames": 11,
      "path": "61013/0"
    },
    {
      "seed": 61014,
      "http_status": 206,
      "sha256": "c50b6a8fc5adeae9173cd4b7ac1ef03fb103ed84f921cab6e3ef1f9dc8b96eff",
      "decoded_frames": 9,
      "path": "61014/0"
    },
    {
      "seed": 61015,
      "http_status": 206,
      "sha256": "4361df600343e43967d9bac1f1444e2154cbab5fa8cb551e63228d01adecf5e1",
      "decoded_frames": 4,
      "path": "61015/0"
    }
  ]
}
```

## held-out参考跟踪诊断

同一1024模型，不另训：dance17步/0.34s、qRMSE.24249rad、hard_limit；walk87步/1.74s、qRMSE.34019rad、hard_limit。D24分别23步fall/.50044与9步fall/.31020；新dance存活更短但q误差更小，walk存活更长但q误差略大，不能说全面改善。两视频实际解码与hash、全部104步共享控制器校验通过；冻结split无official_rpo heldout，保持unavailable。

```json
{
  "job": "job_9022d09d7f564dbc875a",
  "run": "run_de5c7bdfe5db44129a50",
  "receipt": {
    "plan": {
      "hash": "sha256:03ee620e080541831ace9846c853b1f513ac6a1f4e4b93bd1c8b76c24fb219d6",
      "suite": {
        "id": "suite_rpo_ufo_isaac_heldout_tracking_v2",
        "split": "validation",
        "frames": 250,
        "policy": "native_tracking_inference_mean_actor",
        "case_ids": [
          63000,
          63001
        ],
        "selection": "first_and_last_fully_valid_validation_clip_by_id",
        "scene_seed": 63000,
        "qualification": "diagnostic_only"
      },
      "clip_ids": [
        "g1_dance2_subject2_clip3",
        "g1_walk3_subject4_clip3"
      ],
      "asset_sha256": "b58868a782d8863ac38059eb4eab89d27ee1d70565bf58f5a055f44510c8e842",
      "image_digest": "sha256:9129bc326a1e1d68122bc630dff29e362066c2fdc4f0285eda7f61d211072d99",
      "source_run_id": "run_d1037ca56dce4f8ebaef",
      "action_control": {
        "reset": "project measured q into v1 target box; last_action=anchor; clear history",
        "guarantee": "target increments only; actual dq and infeasible reset residual remain measured",
        "control_dt_s": 0.02,
        "replay_action": "requested",
        "schema_version": "eb/rpo-target-slew/v1",
        "observation_last_action": "applied",
        "joint_velocity_limits_rad_s": [
          25.0,
          25.0,
          25.0,
          25.0,
          25.0,
          8.0,
          8.0,
          25.0,
          25.0,
          8.0,
          8.0,
          25.0,
          25.0,
          8.0,
          8.0,
          8.0,
          8.0,
          8.0,
          8.0,
          8.0,
          8.0,
          8.0,
          8.0
        ],
        "target_velocity_limit_fraction": 0.5
      },
      "weights_sha256": "08c68093e96e6b7aee27513b96f7f2cc2388127cc7fdd19b8a5de90e1b0db639",
      "checkpoint_step": 1024,
      "motion_manifest_sha256": "7e47773bf53588ee200cccc5bbde3999df172a11593d7359cbeca2a00ac4c5d8"
    },
    "cases": [
      {
        "split": "validation",
        "start": 0,
        "steps": 17,
        "trace": "tracking-63000.jsonl",
        "video": "tracking-63000.mp4",
        "reason": "hard_limit",
        "source": "g1",
        "case_id": 63000,
        "clip_id": "g1_dance2_subject2_clip3",
        "initial": {
          "seed": 63000,
          "state": {
            "joint_pos": [
              0.10742152482271194,
              -0.04892759770154953,
              -0.245329350233078,
              -0.14100255072116852,
              0.027546362951397896,
              -0.031033411622047424,
              0.20592018961906433,
              -0.006458901334553957,
              -0.32395684719085693,
              -0.021512849256396294,
              0.006657715421169996,
              0.018925629556179047,
              -0.03266697749495506,
              0.15887676179409027,
              0.013584383763372898,
              -0.03679205849766731,
              0.04059223830699921,
              -0.10518943518400192,
              -0.04857902601361275,
              -0.01638425514101982,
              0.010864009149372578,
              -5.15163155796472e-05,
              1.1562417057575658e-05
            ],
            "joint_vel": [
              0.34670570492744446,
              -0.012177228927612305,
              -2.461834192276001,
              -0.4387304186820984,
              -0.1338970810174942,
              0.26940852403640747,
              2.2680282592773438,
              -0.2741324007511139,
              -2.1669254302978516,
              0.48474541306495667,
              0.016313418745994568,
              1.1738395690917969,
              3.1189393997192383,
              4.661955833435059,
              -0.021598022431135178,
              -1.3564820289611816,
              1.0458266735076904,
              -3.153118848800659,
              -2.271662950515747,
              -0.4017891585826874,
              0.3273966312408447,
              -0.0015205582603812218,
              0.0005137825501151383
            ],
            "root_pos_w": [
              -0.34654700756073,
              0.6690345406532288,
              0.7514374852180481
            ],
            "root_quat_w": [
              0.8401289582252502,
              -0.009509999305009842,
              0.0878949984908104,
              0.5351330041885376
            ],
            "applied_torque": [
              -0.23059161007404327,
              -0.2518518269062042,
              -0.5014876127243042,
              0.4051766097545624,
              0.3754017949104309,
              0.34028807282447815,
              -0.14575931429862976,
              -0.4553878903388977,
              -0.9894275069236755,
              0.21776162087917328,
              0.1773398369550705,
              -0.9943281412124634,
              -0.12963080406188965,
              -0.9239127039909363,
              0.20540812611579895,
              0.35958749055862427,
              0.05543842539191246,
              13.525477409362793,
              12.132038116455078,
              -0.19192518293857574,
              0.3163147568702698,
              -0.00018803388229571283,
              -3.716998890013201e-06
            ],
            "root_ang_vel_b": [
              -0.17079153656959534,
              -0.110317662358284,
              0.1368255615234375
            ],
            "root_ang_vel_w": [
              0.047858040779829025,
              -0.18534092605113983,
              0.15303371846675873
            ],
            "root_lin_vel_b": [
              0.15520256757736206,
              0.22412124276161194,
              0.028727121651172638
            ],
            "root_lin_vel_w": [
              -0.1340309977531433,
              0.23817284405231476,
              0.02127910777926445
            ],
            "projected_gravity_b": [
              0.1578644961118698,
              -0.07809177786111832,
              -0.984368085861206
            ]
          },
          "command": [
            0.246636763215065,
            0.37335705757141113,
            1.5214855670928955
          ],
          "env_origin": [
            1.25,
            0.0,
            0.0
          ],
          "hard_limits": {
            "joint_pos": [
              [
                -0.9999999403953552,
                0.19999998807907104
              ],
              [
                -0.19999998807907104,
                0.9999999403953552
              ],
              [
                -3.1399998664855957,
                3.1399998664855957
              ],
              [
                -0.19999998807907104,
                0.9999999403953552
              ],
              [
                -0.9999999403953552,
                0.19999998807907104
              ],
              [
                -3.1399998664855957,
                1.5699999332427979
              ],
              [
                -3.1399998664855957,
                1.5699999332427979
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
                -0.2499999850988388,
                3.1399998664855957
              ],
              [
                -3.1399998664855957,
                0.2499999850988388
              ],
              [
                -0.19999998807907104,
                2.499999761581421
              ],
              [
                -0.19999998807907104,
                2.499999761581421
              ],
              [
                -1.5699999332427979,
                1.5699999332427979
              ],
              [
                -1.5699999332427979,
                1.5699999332427979
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
                1.5699999332427979
              ],
              [
                -0.6000000238418579,
                1.5699999332427979
              ],
              [
                -0.4999999701976776,
                0.4999999701976776
              ],
              [
                -0.4999999701976776,
                0.4999999701976776
              ],
              [
                -1.5699999332427979,
                1.5699999332427979
              ],
              [
                -1.5699999332427979,
                1.5699999332427979
              ]
            ],
            "joint_vel": [
              25.0,
              25.0,
              25.0,
              25.0,
              25.0,
              8.0,
              8.0,
              25.0,
              25.0,
              8.0,
              8.0,
              25.0,
              25.0,
              8.0,
              8.0,
              8.0,
              8.0,
              8.0,
              8.0,
              8.0,
              8.0,
              8.0,
              8.0
            ],
            "applied_torque": [
              120.0,
              120.0,
              120.0,
              120.0,
              120.0,
              27.0,
              27.0,
              120.0,
              120.0,
              27.0,
              27.0,
              120.0,
              120.0,
              27.0,
              27.0,
              27.0,
              27.0,
              27.0,
              27.0,
              27.0,
              27.0,
              27.0,
              27.0
            ]
          },
          "terrain_seed": 63000,
          "terrain_level": null,
          "actuator_gains": {
            "arms": {
              "damping": [
                1.6437726020812988,
                1.4847837686538696,
                0.9565516114234924,
                1.0891530513763428
              ],
              "stiffness": [
                30.803661346435547,
                32.349063873291016,
                21.63111686706543,
                21.014842987060547
              ]
            },
            "feet": {
              "damping": [
                2.0425937175750732,
                2.199343681335449,
                2.0604615211486816,
                1.8856925964355469
              ],
              "stiffness": [
                40.855865478515625,
                42.168128967285156,
                42.02286148071289,
                37.10384750366211
              ]
            },
            "legs": {
              "damping": [
                3.0756280422210693,
                3.1756396293640137,
                4.873538017272949,
                3.5468978881835938,
                3.075746774673462,
                3.6219284534454346,
                3.2907302379608154,
                5.277044773101807,
                4.7272796630859375
              ],
              "stiffness": [
                92.44561004638672,
                103.2003173828125,
                158.58753967285156,
                100.29654693603516,
                100.1108169555664,
                91.97803497314453,
                95.90167999267578,
                162.1211395263672,
                146.7464141845703
              ]
            },
            "shoulders": {
              "damping": [
                2.0774006843566895,
                1.8790806531906128,
                1.8417327404022217,
                2.1494808197021484,
                1.9970283508300781,
                1.9828617572784424
              ],
              "stiffness": [
                37.150394439697266,
                38.77536392211914,
                39.22160720825195,
                37.671241760253906,
                40.97926330566406,
                43.7481689453125
              ]
            }
          },
          "terrain_column": null,
          "startup_parameters": {
            "coms": [
              [
                -0.024436334148049355,
                -0.00762091763317585,
                -0.044149912893772125,
                -1.6202104234253056e-05,
                0.315790593624115,
                1.7049482266884297e-05,
                0.948828935623169
              ],
              [
                -0.03847656026482582,
                -1.4199999895936344e-05,
                -0.056123290210962296,
                0.0016626018332317472,
                0.2687978446483612,
                -0.000640812620986253,
                0.9631949663162231
              ],
              [
                -0.03847656026482582,
                1.4199999895936344e-05,
                -0.056123290210962296,
                -3.7299345422070473e-05,
                0.2691715359687805,
                1.7442502212361433e-05,
                0.9630922675132751
              ],
              [
                0.020223606377840042,
                -0.007623187731951475,
                0.06292270123958588,
                -0.00037474121199920774,
                -0.03675074130296707,
                -0.001695423386991024,
                0.9993229508399963
              ],
              [
                0.05748777836561203,
                -0.006019010208547115,
                -0.03308920934796333,
                -0.00306421285495162,
                0.2610301673412323,
                -0.012036540545523167,
                0.9652507305145264
              ],
              [
                0.05748777836561203,
                0.006019010208547115,
                -0.03308920934796333,
                0.0032282336615025997,
                0.2609482705593109,
                0.0116127235814929,
                0.9652776122093201
              ],
              [
                -0.004365229979157448,
                0.0499514602124691,
                0.0,
                0.0,
                0.0,
                0.13336403667926788,
                0.991067111492157
              ],
              [
                -0.004365229979157448,
                -0.0499514602124691,
                0.0,
                0.0,
                0.0,
                -0.13336403667926788,
                0.991067111492157
              ],
              [
                4.5870001486036927e-05,
                -0.01904764026403427,
                -0.1985814869403839,
                -0.05691227316856384,
                -0.0003159357875119895,
                0.0014507799642160535,
                0.9983781576156616
              ],
              [
                4.5870001486036927e-05,
                0.01904764026403427,
                -0.1985814869403839,
                0.05691226199269295,
                -0.0003164053487125784,
                -0.0014481309335678816,
                0.9983781576156616
              ],
              [
                -0.0029007000848650932,
                0.0,
                -0.021971119567751884,
                0.0,
                0.26533544063568115,
                0.0,
                0.9641561508178711
              ],
              [
                -0.0029007000848650932,
                0.0,
                -0.021971119567751884,
                0.0,
                0.26533544063568115,
                0.0,
                0.9641561508178711
              ],
              [
                -0.00035292000393383205,
                -0.013279800303280354,
                -0.15208271145820618,
                -0.01299212034791708,
                -0.0013886223314329982,
                0.03272939845919609,
                0.9993789196014404
              ],
              [
                -0.00035292000393383205,
                0.013279800303280354,
                -0.15208271145820618,
                0.012992119416594505,
                0.0013886203523725271,
                0.032729558646678925,
                0.9993789196014404
              ],
              [
                -4.3280000681988895e-05,
                -0.0018211000133305788,
                -0.10615894943475723,
                -0.015526726841926575,
                6.655540346400812e-05,
                6.0692276747431606e-05,
                0.9998795390129089
              ],
              [
                -4.3280000681988895e-05,
                0.0018211000133305788,
                -0.10615894943475723,
                0.015265019610524178,
                -8.227960279327817e-06,
                0.0009966748766601086,
                0.9998829960823059
              ],
              [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0
              ],
              [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0
              ],
              [
                0.02378500998020172,
                -0.00433075986802578,
                0.0,
                0.0,
                0.0,
                -0.2751936912536621,
                0.9613888263702393
              ],
              [
                0.02378500998020172,
                0.00433075986802578,
                0.0,
                0.0,
                0.0,
                0.2751936912536621,
                0.9613888263702393
              ],
              [
                0.014542819932103157,
                0.0028482500929385424,
                -0.029086079448461533,
                0.1610575169324875,
                0.029140209779143333,
                -0.02334902621805668,
                0.9862384796142578
              ],
              [
                0.014542819932103157,
                -0.0028482500929385424,
                -0.029086079448461533,
                -0.16106057167053223,
                0.02914167381823063,
                0.023337095975875854,
                0.9862381219863892
              ],
              [
                0.053176190704107285,
                -3.4500000765547156e-05,
                -1.9920000340789557e-05,
                0.23636305332183838,
                -5.414546831161715e-05,
                5.12967053509783e-05,
                0.9716647863388062
              ],
              [
                0.053176190704107285,
                3.4500000765547156e-05,
                -1.9920000340789557e-05,
                0.23636282980442047,
                5.41454937774688e-05,
                -5.129670171299949e-05,
                0.9716649651527405
              ]
            ],
            "masses": [
              5.512955188751221,
              1.711920976638794,
              1.6210618019104004,
              8.124547004699707,
              1.765224814414978,
              1.7324010133743286,
              0.4898262023925781,
              0.4292874336242676,
              2.4946794509887695,
              2.1540346145629883,
              0.08728164434432983,
              0.09124477207660675,
              1.580350399017334,
              1.7165347337722778,
              1.0783905982971191,
              1.1481728553771973,
              0.07750018686056137,
              0.0817917138338089,
              0.09505625814199448,
              0.08643028885126114,
              0.34616053104400635,
              0.34229597449302673,
              0.7308517694473267,
              0.69977205991745
            ],
            "materials": [
              [
                0.4794110953807831,
                0.4794110953807831,
                0.06193619966506958
              ],
              [
                1.2090952396392822,
                0.7537564039230347,
                0.05769467353820801
              ],
              [
                0.7040688395500183,
                0.3974241316318512,
                0.3158239424228668
              ],
              [
                1.5534241199493408,
                0.9439371824264526,
                0.27535945177078247
              ],
              [
                0.664222240447998,
                0.664222240447998,
                0.4014032185077667
              ],
              [
                0.8553846478462219,
                0.6159341335296631,
                0.2739356756210327
              ],
              [
                0.754753589630127,
                0.754753589630127,
                0.48388612270355225
              ],
              [
                1.0813511610031128,
                0.9456270337104797,
                0.02358299493789673
              ],
              [
                1.3778901100158691,
                0.7172248959541321,
                0.36801350116729736
              ],
              [
                1.1001620292663574,
                1.0345749855041504,
                0.19786012172698975
              ],
              [
                1.1551988124847412,
                0.6262463927268982,
                0.23825979232788086
              ],
              [
                1.3934311866760254,
                0.8071075081825256,
                0.36501455307006836
              ],
              [
                1.1228675842285156,
                0.7925946712493896,
                0.13509050011634827
              ],
              [
                1.2090952396392822,
                0.7537564039230347,
                0.05769467353820801
              ],
              [
                0.34174537658691406,
                0.34174537658691406,
                0.12865614891052246
              ],
              [
                1.3934311866760254,
                0.8071075081825256,
                0.36501455307006836
              ]
            ],
            "joint_damping": [
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            "joint_armature": [
              0.008333805948495865,
              0.01012714859098196,
              0.009203548543155193,
              0.011422732844948769,
              0.00838932953774929,
              0.011110135354101658,
              0.009054404683411121,
              0.009742881171405315,
              0.009143668226897717,
              0.008458951488137245,
              0.01171178836375475,
              0.01036433782428503,
              0.00971306674182415,
              0.008961028419435024,
              0.011473285034298897,
              0.009078443050384521,
              0.011772281490266323,
              0.009667211212217808,
              0.009433036670088768,
              0.011082270182669163,
              0.01020065788179636,
              0.008855944499373436,
              0.00966416671872139
            ],
            "joint_stiffness": [
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            "joint_friction_coeff": [
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0
            ]
          },
          "actuator_delay_steps": {
            "arms": 0,
            "feet": 0,
            "legs": 0,
            "shoulders": 0
          }
        },
        "q_rmse_rad": 0.24248955958158713,
        "scene_seed": 63000,
        "trace_sha256": "c8c9bb368c87b2236badfe0722e503d65727752aeef041e275ca54e91d415648",
        "video_integrity": {
          "fps": 10,
          "sha256": "276a405604967da61c2580ef2c4aef17c85b36c12f6ff53cb246d12b1d712700",
          "resolution": [
            320,
            240
          ],
          "decoded_frames": 5,
          "schema_version": "eb/bfm-case-video/v1"
        }
      },
      {
        "split": "validation",
        "start": 0,
        "steps": 87,
        "trace": "tracking-63001.jsonl",
        "video": "tracking-63001.mp4",
        "reason": "hard_limit",
        "source": "g1",
        "case_id": 63001,
        "clip_id": "g1_walk3_subject4_clip3",
        "initial": {
          "seed": 63000,
          "state": {
            "joint_pos": [
              0.05141235142946243,
              -0.010788017883896828,
              0.1066763624548912,
              -0.05681707710027695,
              0.028885554522275925,
              0.0806150734424591,
              -0.021717162802815437,
              -0.10438023507595062,
              0.2947237193584442,
              -0.023503951728343964,
              0.04386427626013756,
              0.03977415710687637,
              0.12589527666568756,
              0.04234668239951134,
              -0.11227357387542725,
              0.013288579881191254,
              -0.07376706600189209,
              -0.006926596164703369,
              -0.03221946209669113,
              0.001448440714739263,
              -0.0009191203280352056,
              -4.121133770240704e-06,
              -6.593851594516309e-06
            ],
            "joint_vel": [
              0.3482937812805176,
              0.29122769832611084,
              1.1016775369644165,
              -0.486002117395401,
              -0.39769867062568665,
              1.5810579061508179,
              0.6004852652549744,
              -0.4481706917285919,
              1.620085597038269,
              -0.22985628247261047,
              -0.056404247879981995,
              2.2229135036468506,
              0.3370799124240875,
              1.2957109212875366,
              -3.019347906112671,
              -0.20609042048454285,
              -3.018254041671753,
              -0.6266817450523376,
              -1.099671483039856,
              0.23292680084705353,
              -0.1162007600069046,
              -0.0002437959483359009,
              -0.0002055102668236941
            ],
            "root_pos_w": [
              -5.913344860076904,
              2.325208902359009,
              0.7609716653823853
            ],
            "root_quat_w": [
              0.9998173713684082,
              0.009662002325057983,
              -0.015908006578683853,
              -0.0043310015462338924
            ],
            "applied_torque": [
              -0.46642041206359863,
              0.40238437056541443,
              1.0298912525177002,
              0.7116038799285889,
              0.1060817688703537,
              -0.6033617854118347,
              -0.2966286242008209,
              -0.14885593950748444,
              -1.1044893264770508,
              0.35301879048347473,
              -0.45720627903938293,
              -0.5804672837257385,
              3.584282875061035,
              0.06682170927524567,
              1.8079301118850708,
              0.1984107494354248,
              -0.7652433514595032,
              15.03182315826416,
              14.959432601928711,
              -0.39367207884788513,
              0.21919389069080353,
              -0.0005767500260844827,
              9.033414244186133e-05
            ],
            "root_ang_vel_b": [
              0.39260333776474,
              0.09319345653057098,
              -0.1674448698759079
            ],
            "root_ang_vel_w": [
              0.39850881695747375,
              0.09286380559206009,
              -0.15305954217910767
            ],
            "root_lin_vel_b": [
              0.7488530874252319,
              0.07364263385534286,
              -0.028031926602125168
            ],
            "root_lin_vel_w": [
              0.7499551773071289,
              0.06744826585054398,
              -0.0028210515156388283
            ],
            "projected_gravity_b": [
              -0.03172650933265686,
              -0.019458269700407982,
              -0.9993071556091309
            ]
          },
          "command": [
            0.5684600472450256,
            0.013298332691192627,
            -0.2880822718143463
          ],
          "env_origin": [
            -1.25,
            0.0,
            0.0
          ],
          "hard_limits": {
            "joint_pos": [
              [
                -0.9999999403953552,
                0.19999998807907104
              ],
              [
                -0.19999998807907104,
                0.9999999403953552
              ],
              [
                -3.1399998664855957,
                3.1399998664855957
              ],
              [
                -0.19999998807907104,
                0.9999999403953552
              ],
              [
                -0.9999999403953552,
                0.19999998807907104
              ],
              [
                -3.1399998664855957,
                1.5699999332427979
              ],
              [
                -3.1399998664855957,
                1.5699999332427979
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
                -0.2499999850988388,
                3.1399998664855957
              ],
              [
                -3.1399998664855957,
                0.2499999850988388
              ],
              [
                -0.19999998807907104,
                2.499999761581421
              ],
              [
                -0.19999998807907104,
                2.499999761581421
              ],
              [
                -1.5699999332427979,
                1.5699999332427979
              ],
              [
                -1.5699999332427979,
                1.5699999332427979
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
                1.5699999332427979
              ],
              [
                -0.6000000238418579,
                1.5699999332427979
              ],
              [
                -0.4999999701976776,
                0.4999999701976776
              ],
              [
                -0.4999999701976776,
                0.4999999701976776
              ],
              [
                -1.5699999332427979,
                1.5699999332427979
              ],
              [
                -1.5699999332427979,
                1.5699999332427979
              ]
            ],
            "joint_vel": [
              25.0,
              25.0,
              25.0,
              25.0,
              25.0,
              8.0,
              8.0,
              25.0,
              25.0,
              8.0,
              8.0,
              25.0,
              25.0,
              8.0,
              8.0,
              8.0,
              8.0,
              8.0,
              8.0,
              8.0,
              8.0,
              8.0,
              8.0
            ],
            "applied_torque": [
              120.0,
              120.0,
              120.0,
              120.0,
              120.0,
              27.0,
              27.0,
              120.0,
              120.0,
              27.0,
              27.0,
              120.0,
              120.0,
              27.0,
              27.0,
              27.0,
              27.0,
              27.0,
              27.0,
              27.0,
              27.0,
              27.0,
              27.0
            ]
          },
          "terrain_seed": 63000,
          "terrain_level": null,
          "actuator_gains": {
            "arms": {
              "damping": [
                1.495827555656433,
                1.4944145679473877,
                1.0659939050674438,
                0.9655811190605164
              ],
              "stiffness": [
                32.47827911376953,
                31.758577346801758,
                21.431636810302734,
                18.335224151611328
              ]
            },
            "feet": {
              "damping": [
                1.9406613111495972,
                2.0268983840942383,
                2.0120415687561035,
                1.8811606168746948
              ],
              "stiffness": [
                38.29948043823242,
                42.16834259033203,
                38.18552780151367,
                40.69554901123047
              ]
            },
            "legs": {
              "damping": [
                3.0588200092315674,
                3.0423243045806885,
                4.824323654174805,
                3.3312835693359375,
                3.4676637649536133,
                3.1536662578582764,
                3.5345377922058105,
                4.917845726013184,
                5.449957847595215
              ],
              "stiffness": [
                107.30628967285156,
                107.43846893310547,
                144.4141387939453,
                104.33283996582031,
                98.32749938964844,
                90.17626190185547,
                95.6202163696289,
                140.5286865234375,
                149.7949676513672
              ]
            },
            "shoulders": {
              "damping": [
                1.9003756046295166,
                1.9364495277404785,
                2.144157648086548,
                2.0033230781555176,
                2.049323081970215,
                2.0778656005859375
              ],
              "stiffness": [
                40.93000793457031,
                39.69049835205078,
                37.38105010986328,
                42.54682922363281,
                40.74794006347656,
                39.93252182006836
              ]
            }
          },
          "terrain_column": null,
          "startup_parameters": {
            "coms": [
              [
                -0.04195932671427727,
                -0.02286790870130062,
                -0.006718686781823635,
                -1.6202104234253056e-05,
                0.315790593624115,
                1.7049482266884297e-05,
                0.948828935623169
              ],
              [
                -0.03847656026482582,
                -1.4199999895936344e-05,
                -0.056123290210962296,
                0.0016626018332317472,
                0.2687978446483612,
                -0.0006408119224943221,
                0.9631949663162231
              ],
              [
                -0.03847656026482582,
                1.4199999895936344e-05,
                -0.056123290210962296,
                -3.729931631824002e-05,
                0.2691715359687805,
                1.744246947055217e-05,
                0.9630922675132751
              ],
              [
                0.00270061451010406,
                -0.022870179265737534,
                0.10035392642021179,
                -0.00037474153214134276,
                -0.03675074130296707,
                -0.001695422106422484,
                0.9993229508399963
              ],
              [
                0.05748777836561203,
                -0.006019010208547115,
                -0.03308920934796333,
                -0.003064215648919344,
                0.2610299587249756,
                -0.012036539614200592,
                0.9652507901191711
              ],
              [
                0.05748777836561203,
                0.006019010208547115,
                -0.03308920934796333,
                0.0032282331958413124,
                0.26094850897789,
                0.011612734757363796,
                0.9652775526046753
              ],
              [
                -0.004365229979157448,
                0.0499514602124691,
                0.0,
                0.0,
                0.0,
                0.13336403667926788,
                0.991067111492157
              ],
              [
                -0.004365229979157448,
                -0.0499514602124691,
                0.0,
                0.0,
                0.0,
                -0.13336403667926788,
                0.991067111492157
              ],
              [
                4.5870001486036927e-05,
                -0.01904764026403427,
                -0.1985814869403839,
                -0.05691227316856384,
                -0.00031593587482348084,
                0.0014507801970466971,
                0.9983781576156616
              ],
              [
                4.5870001486036927e-05,
                0.01904764026403427,
                -0.1985814869403839,
                0.05691226199269295,
                -0.0003164056106470525,
                -0.0014481268590316176,
                0.9983781576156616
              ],
              [
                -0.0029007000848650932,
                0.0,
                -0.021971119567751884,
                0.0,
                0.26533544063568115,
                0.0,
                0.9641561508178711
              ],
              [
                -0.0029007000848650932,
                0.0,
                -0.021971119567751884,
                0.0,
                0.26533544063568115,
                0.0,
                0.9641561508178711
              ],
              [
                -0.00035292000393383205,
                -0.013279800303280354,
                -0.15208271145820618,
                -0.01299212034791708,
                -0.0013886201195418835,
                0.032729558646678925,
                0.9993789196014404
              ],
              [
                -0.00035292000393383205,
                0.013279800303280354,
                -0.15208271145820618,
                0.012992119416594505,
                0.001388623146340251,
                0.03272935748100281,
                0.9993789196014404
              ],
              [
                -4.3280000681988895e-05,
                -0.0018211000133305788,
                -0.10615894943475723,
                -0.015526728704571724,
                6.655539618805051e-05,
                6.069216760806739e-05,
                0.9998795390129089
              ],
              [
                -4.3280000681988895e-05,
                0.0018211000133305788,
                -0.10615894943475723,
                0.015265017747879028,
                -8.228191290982068e-06,
                0.0009966898942366242,
                0.9998829960823059
              ],
              [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0
              ],
              [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0
              ],
              [
                0.02378500998020172,
                -0.00433075986802578,
                0.0,
                0.0,
                0.0,
                -0.2751936912536621,
                0.9613888263702393
              ],
              [
                0.02378500998020172,
                0.00433075986802578,
                0.0,
                0.0,
                0.0,
                0.2751936912536621,
                0.9613888263702393
              ],
              [
                0.014542819932103157,
                0.0028482500929385424,
                -0.029086079448461533,
                0.16105782985687256,
                0.029140207916498184,
                -0.023349033668637276,
                0.986238420009613
              ],
              [
                0.014542819932103157,
                -0.0028482500929385424,
                -0.029086079448461533,
                -0.161060631275177,
                0.029141675680875778,
                0.023337094113230705,
                0.9862381219863892
              ],
              [
                0.053176190704107285,
                -3.4500000765547156e-05,
                -1.9920000340789557e-05,
                0.23634660243988037,
                -5.4146337788552046e-05,
                5.129579221829772e-05,
                0.9716688394546509
              ],
              [
                0.053176190704107285,
                3.4500000765547156e-05,
                -1.9920000340789557e-05,
                0.2363518923521042,
                5.414606130216271e-05,
                -5.12960905325599e-05,
                0.9716675281524658
              ]
            ],
            "masses": [
              5.512955188751221,
              1.7435755729675293,
              1.651986837387085,
              8.006284713745117,
              1.587217092514038,
              1.5294945240020752,
              0.472698450088501,
              0.5169083476066589,
              2.448643445968628,
              2.464855670928955,
              0.08315767347812653,
              0.07700209319591522,
              1.7147549390792847,
              1.550467610359192,
              1.1745911836624146,
              1.2124613523483276,
              0.08989439904689789,
              0.08732220530509949,
              0.08537430316209793,
              0.088210329413414,
              0.3489702343940735,
              0.3284491002559662,
              0.6901547908782959,
              0.6186112761497498
            ],
            "materials": [
              [
                0.30846264958381653,
                0.30846264958381653,
                0.4814665913581848
              ],
              [
                0.754753589630127,
                0.754753589630127,
                0.48388612270355225
              ],
              [
                1.3643221855163574,
                1.165284276008606,
                0.0018328428268432617
              ],
              [
                0.521917462348938,
                0.521917462348938,
                0.03335854411125183
              ],
              [
                0.8358306884765625,
                0.8358306884765625,
                0.2220456600189209
              ],
              [
                1.081646203994751,
                0.889136791229248,
                0.2628234922885895
              ],
              [
                1.07847261428833,
                0.8393118977546692,
                0.2101098597049713
              ],
              [
                1.1228675842285156,
                0.7925946712493896,
                0.13509050011634827
              ],
              [
                0.7788393497467041,
                0.7788393497467041,
                0.2808488607406616
              ],
              [
                1.3361382484436035,
                0.37216901779174805,
                0.20714911818504333
              ],
              [
                1.4124853610992432,
                0.4278773367404938,
                0.05549207329750061
              ],
              [
                0.7788393497467041,
                0.7788393497467041,
                0.2808488607406616
              ],
              [
                0.41061434149742126,
                0.41061434149742126,
                0.29464390873908997
              ],
              [
                1.2457685470581055,
                0.3472864627838135,
                0.019394606351852417
              ],
              [
                1.2457685470581055,
                0.3472864627838135,
                0.019394606351852417
              ],
              [
                0.6376650929450989,
                0.6376650929450989,
                0.2472410500049591
              ]
            ],
            "joint_damping": [
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            "joint_armature": [
              0.00982633139938116,
              0.011548714712262154,
              0.011928418651223183,
              0.010323451831936836,
              0.011220159009099007,
              0.01096391212195158,
              0.008579136803746223,
              0.010041543282568455,
              0.01152980886399746,
              0.009514507837593555,
              0.011995592154562473,
              0.009529626928269863,
              0.009279388003051281,
              0.009952090680599213,
              0.010876135900616646,
              0.011512220837175846,
              0.010961214080452919,
              0.00985111016780138,
              0.00820247083902359,
              0.010159232653677464,
              0.008682084269821644,
              0.00854026060551405,
              0.011850157752633095
            ],
            "joint_stiffness": [
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0
            ],
            "joint_friction_coeff": [
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0
            ]
          },
          "actuator_delay_steps": {
            "arms": 0,
            "feet": 0,
            "legs": 0,
            "shoulders": 0
          }
        },
        "q_rmse_rad": 0.3401874046376152,
        "scene_seed": 63000,
        "trace_sha256": "6e8419430cf3b0a610aeff646ce40aa73ba57ec063d1075ebdd3c96f3a4ad06f",
        "video_integrity": {
          "fps": 10,
          "sha256": "2dcaa641ac8b334f11c3457a73474b7abe2bba5f9552201e895c30a4680ee57d",
          "resolution": [
            320,
            240
          ],
          "decoded_frames": 19,
          "schema_version": "eb/bfm-case-video/v1"
        }
      }
    ],
    "suite": {
      "id": "suite_rpo_ufo_isaac_heldout_tracking_v2",
      "split": "validation",
      "frames": 250,
      "policy": "native_tracking_inference_mean_actor",
      "case_ids": [
        63000,
        63001
      ],
      "selection": "first_and_last_fully_valid_validation_clip_by_id",
      "scene_seed": 63000,
      "qualification": "diagnostic_only"
    },
    "qualification": "not_requested",
    "schema_version": "eb/ufo-heldout-tracking/v1",
    "official_rpo_heldout": "unavailable_in_frozen_parent_splits"
  },
  "media_and_control_verification": [
    {
      "case_id": 63000,
      "trace_sha_verified": true,
      "shared_controller_steps": 17,
      "video": {
        "schema_version": "eb/bfm-case-video/v1",
        "sha256": "276a405604967da61c2580ef2c4aef17c85b36c12f6ff53cb246d12b1d712700",
        "decoded_frames": 5,
        "resolution": [
          320,
          240
        ],
        "fps": 10
      }
    },
    {
      "case_id": 63001,
      "trace_sha_verified": true,
      "shared_controller_steps": 87,
      "video": {
        "schema_version": "eb/bfm-case-video/v1",
        "sha256": "2dcaa641ac8b334f11c3457a73474b7abe2bba5f9552201e895c30a4680ee57d",
        "decoded_frames": 19,
        "resolution": [
          320,
          240
        ],
        "fps": 10
      }
    }
  ],
  "http_media": [
    {
      "case_id": 63000,
      "http_status": 206,
      "content_range": "bytes 0-1023/6587",
      "url": "http://194.0.106:31800/media/27/276a405604967da61c2580ef2c4aef17c85b36c12f6ff53cb246d12b1d712700"
    },
    {
      "case_id": 63001,
      "http_status": 206,
      "content_range": "bytes 0-1023/17488",
      "url": "http://194.0.106:31800/media/2d/2dcaa641ac8b334f11c3457a73474b7abe2bba5f9552201e895c30a4680ee57d"
    }
  ]
}
```

两项heldout媒体均另经HTTP Range206验证；URL在上方索引中完整保存。

## GPU/RSS、退出与吞吐

```json
{
  "slew1024": {
    "gpu_peak_mib_by_physical_index": {
      "0": 12466,
      "1": 5778,
      "2": 5825,
      "3": 5778
    },
    "rss_peak_gib": 42.31,
    "scope": "periodic observed maxima, not guaranteed instantaneous peaks"
  },
  "slew1024-eval": {
    "gpu_peak_mib_by_physical_index": {
      "0": 8441,
      "1": 8363,
      "2": 8410,
      "3": 7721
    },
    "rss_peak_gib": 52.15,
    "scope": "periodic observed maxima, not guaranteed instantaneous peaks"
  },
  "slew1024-tracking": {
    "gpu_peak_mib_by_physical_index": {
      "0": 7222
    },
    "rss_peak_gib": 12.27,
    "scope": "periodic observed maxima, not guaranteed instantaneous peaks"
  }
}
```

slew1024退出状态（均exit0/OOMfalse，自有结束容器已移除）：

```text
{"Status":"exited","Running":false,"Paused":false,"Restarting":false,"OOMKilled":false,"Dead":false,"Pid":0,"ExitCode":0,"Error":"","StartedAt":"2026-10-10T21:28:40.665978463Z","FinishedAt":"2026-10-10T22:02:18.848961668Z"} 168117996

```

slew1024-eval退出状态（均exit0/OOMfalse，自有结束容器已移除）：

```text
{"Status":"exited","Running":false,"Paused":false,"Restarting":false,"OOMKilled":false,"Dead":false,"Pid":0,"ExitCode":0,"Error":"","StartedAt":"2026-10-10T22:03:46.191932872Z","FinishedAt":"2026-10-10T22:08:32.027662103Z"} 2528238368

```

slew1024-tracking退出状态（均exit0/OOMfalse，自有结束容器已移除）：

```text
{"Status":"exited","Running":false,"Paused":false,"Restarting":false,"OOMKilled":false,"Dead":false,"Pid":0,"ExitCode":0,"Error":"","StartedAt":"2026-10-10T22:09:06.469711526Z","FinishedAt":"2026-10-10T22:16:53.286778664Z"} 235861045

```

## E已实现的固定候选

原生现有aux接口追加第九项：`sum_j relu(abs(dq_j)/Vmax_j-.8)^2`，原生加权−5一次，不乘dt、不预加权；全部原八项权重保留。改变整体aux normalizer分布，需后续关注尾部。共享SLEW_CONTROL、同一codec、同一D learning_profile/checkpoint model bind保持；独立recipe/Profile/Gym、auxiliary_profile、source/image/fingerprint与replay v2区分E，拒绝D↔E完整恢复（包括空replay），没有跨版本回填旧样本。

```json
{
  "recipe": "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_slew_velocity_v5",
  "profile": "rpo_bfm_ufo_isaac_4gpu_rsi_slew_velocity_v5",
  "gym": "RPO-UFO-Isaac-RSI-slew-velocity-v5",
  "learning_profile": "physical_limits_stable_slew_v4",
  "auxiliary_profile": "soft_velocity_v5",
  "env_flags": {
    "EB_UFO_LEARNING_PROFILE": "physical_limits_slew_v4",
    "EB_UFO_AUXILIARY_PROFILE": "soft_velocity_v5"
  },
  "producer": "sha256:e1382226cd5f0bd3795ee6a187f62555ba77f591e342c862a64bf1825a463dee",
  "evaluator": "sha256:3094f020480ce099ff281ac76c4812bd47985b7d4a489a15a0550c49e6e9fbca",
  "status": "dependency_ready",
  "owned_parameters": {
    "forward_optimizer": 134241604,
    "backward_optimizer": 171520,
    "discriminator_optimizer": 2785281,
    "actor_optimizer": 31524851,
    "critic_optimizer": 133196614,
    "aux_critic_optimizer": 133196614
  },
  "owned_parameter_total": 435116484,
  "cpu_replay_max_bytes": 7042000000,
  "producer_actual_source": {
    "schema_version": "eb/ufo-image-source/v1",
    "ufo": {
      "commit": "0b89378de02f77712072d983bc43ec2a46455dc0",
      "tree": "3a6b8b0d6a04287f37c1d7834e51561a8e014fc9",
      "content_hash": "sha256:2191cf26c77262f7ef2e3525ca23878a8f08c46fd02a20b7776a791f8f5e7582",
      "license": "CC-BY-NC-4.0",
      "scope": "agent Python source; binary assets excluded"
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
      "tensordict": "0.11.0"
    },
    "scope": "agent-only; Python3.10 package metadata bypassed by tested source import",
    "topology_patch_hash": "sha256:3078506628924464cb42d0b2f108b3fab502a43fd75e091f358d504565294d23",
    "learning_profile_patch_hash": "sha256:d47623e48b4cc757d5e63336a1fa7d49234c3833a2f1c6bbdbee4fe7ffd9a272",
    "rollout_profile_patch_hash": "sha256:777456ce171734326cf8c55eaa434fc5a69a27d0b2d0b725608dd54c190f4a77",
    "action_control_source_sha256": "sha256:1b0bbc0a7a350581ef1939faf93a3e27d824bde62107ecfe4e9ff988bbc088fb",
    "action_control_profile_patch_hash": "sha256:548260ca5e3eedd53ae739c00672c7e273e7835faed734778039cf5977b56e0d",
    "auxiliary_profile": "soft_velocity_v5",
    "auxiliary_source_sha256": "sha256:58c495c65bd5831b001832ddc1a73cf0ef79c9437907d2d084c048eeb5a45c33",
    "auxiliary_profile_patch_hash": "sha256:30e7b4ea7a3ec0e45cdbe0a9dbf718eb219c2c6fd36d07f9b18bd5222183a03a"
  },
  "evaluator_actual_source": {
    "schema_version": "eb/ufo-image-source/v1",
    "ufo": {
      "commit": "0b89378de02f77712072d983bc43ec2a46455dc0",
      "tree": "3a6b8b0d6a04287f37c1d7834e51561a8e014fc9",
      "content_hash": "sha256:2191cf26c77262f7ef2e3525ca23878a8f08c46fd02a20b7776a791f8f5e7582",
      "license": "CC-BY-NC-4.0",
      "scope": "agent Python source; binary assets excluded"
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
      "tensordict": "0.11.0"
    },
    "scope": "agent-only; Python3.10 package metadata bypassed by tested source import",
    "topology_patch_hash": "sha256:3078506628924464cb42d0b2f108b3fab502a43fd75e091f358d504565294d23",
    "action_control_source_sha256": "sha256:1b0bbc0a7a350581ef1939faf93a3e27d824bde62107ecfe4e9ff988bbc088fb",
    "action_control_profile_patch_hash": "sha256:548260ca5e3eedd53ae739c00672c7e273e7835faed734778039cf5977b56e0d",
    "auxiliary_profile": "soft_velocity_v5",
    "auxiliary_source_sha256": "sha256:58c495c65bd5831b001832ddc1a73cf0ef79c9437907d2d084c048eeb5a45c33",
    "auxiliary_profile_patch_hash": "sha256:30e7b4ea7a3ec0e45cdbe0a9dbf718eb219c2c6fd36d07f9b18bd5222183a03a"
  },
  "all14_owned_sources_sha256": {
    "bfm_ufo_agent.py": "1e40b3f979eb6d9a5ebb72534fa54d6c784edd5686c65c6990c1052932e0c5a6",
    "bfm_ufo_replay.py": "7ac18fed333bfe32f10f8d6555cd306b73b6895e8c4dbe437bc2df2ebf585505",
    "bfm_ufo_worker.py": "b6d7b0e91af8199ecf093033738913297f9e833833c49cc7c09b380597dc23cd",
    "bfm_ufo_checkpoint.py": "05a01bda1b32b1c028aeb45213e3ca697fabb848c0f3d971fcc7f3d8c08cfe0f",
    "bfm_ufo_policy.py": "5abf648ee8f28c74a53b124fa82f92ff45525b202126fa8c94d9facee08f55cb",
    "bfm_ufo_artifact.py": "665e9a1d31c75332c4d8aefcab144e257fd8a6300994f8d9841a33545b9535b4",
    "rpo_ufo_isaac.py": "fc537ceebf7b379688d7e31b9754fe59023725cf59ca556bde7397bbefdc55d3",
    "rpo_bfm.py": "5b03f1f8fdc5e47847a267f857c007a54c7714b8233e91854819b3d3c759d112",
    "rpo_bfm_codec.py": "1b0bbc0a7a350581ef1939faf93a3e27d824bde62107ecfe4e9ff988bbc088fb",
    "rpo_bfm_extension.py": "2b380550bc8e289e64064273053bb6f2254c7612dbd30eeb760b284176f19722",
    "rpo_bfm_eval.py": "c19d15fda8e61633d5de71feaf36263f7eb0e75512d934090d24b46151105814",
    "rpo_bfm_parallel_eval.py": "c6348ebb6279909f71212665fd1a9ee889b65dbfcf9e262e839b3c5c38f8a009",
    "rpo_flat_artifact.py": "ea9ec58ca1b97dbcbabff2d188032e6c676dbead3e77dbeb2d4ebec75a6d711f",
    "rpo_ufo_tracking_diagnostic.py": "9c2e27ad459c8cce11fbd81b20b14d4b4b7b4dd9c44600176c43d75643bca769"
  },
  "config": {
    "name": "FBcprAuxAgent",
    "model": {
      "name": "FBcprAuxModel",
      "device": "cpu",
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
      "batch_size": 1024,
      "discount": 0.98,
      "use_mix_rollout": true,
      "update_z_every_step": 100,
      "z_buffer_size": 8192,
      "rollout_expert_trajectories": false,
      "rollout_expert_trajectories_length": 250,
      "rollout_expert_trajectories_percentage": 0.5,
      "lr_discriminator": 1e-05,
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
      "limits_dof_vel": -5.0
    },
    "cudagraphs": false,
    "compile": false
  }
}
```

E eval父层未应用两个training-only alias patch，回执按实际分开，14个owned源完全一致；没有补造缺少的hash。实现覆盖Physics cold/partial reset/terminal9列、strict rollout/replay、native更新/配置恢复、bootstrap、Torch/ONNX、serial/parallel/heldout、collector、capacity、recipe/Profile、镜像/Make与平台门禁。

## 验证、三轮复核与未验证项

- 行为检查先失败后实现：原8逐值保持、速度80%边界/正负对称/导数/非法limit、9列shape/missing/empty replay版本拒绝。
- 新native tiny CPU真正消费第九项：raw=.04，归一化前实际weighted=−.2，六Adam=1；完整checkpoint恢复后下一update六Adam=2，全部metrics/模型/四target与连续更新一致。真实auto-reset→EnvOutput→builder→rollout→replay保留非零terminal第九项；E full save/load/ONNX/CLI export/same-bind hardlink/publication/external ModelsRoot/resume fallback与跨版本拒绝均过。tiny只缩测试网络，生产结构独立meta census为435,116,484，未放宽生产配置约束。
- 相关native容器回归80 passed；后续强化next-update11 passed；最新核心/导出14 passed。backend完整容器pytest524 passed/17 skipped；API额外资格门禁27 passed；核心Ruff与git diff --check通过。跳过项没有当作通过，native专用目标镜像另行执行。
- 第一轮：本体/codec/原生网络/数据split与q-target残差；第二轮：真实六Adam/恢复环境覆盖、GPU/终止/预算与实际storage；第三轮：全部16case/两个heldout/初态trace/图表/解码/hash/HTTP可访问性。独立只读review检查14源和完整config；此前指出的9aux真实builder和完整恢复/导出两项CPU证据缺口已补齐。
- 未验证：E GPU16→24资格、E模型质量、E真实速度下降/饱和<15%、原生512万replay/完整native workspace复现、全部RSI substep normal contact和solver敏感性、真机/目标设备、新100case acceptance。没有正常行走合格结论。

## 本段决定与预算

质量指标符合“零成功时median≥best+10”的改善条件（7→35）；仍在前2000更新观察范围，但下一段超出冻结资源上限，不能跳过预算继续。决定budget_end，保全D1024最新候选和全部历史证据，E保持dependency_ready。存储计量为平台最后部署前的只读观测快照，包含同期宿主工作作为保守上界；不等于完整任务归因的精确增量。旧层变化、CAS、临时文件、文档及未来writable峰值沿用8GiB预留。未来若预算重新明确，再按E fresh16→24或D同配方恢复的合规流程决定；此报告没有授权自动扩额。

```json
{
  "ledger": {
    "start_utc": "2026-10-10T15:31:00Z",
    "deadline_utc": "2026-10-11T15:31:00Z",
    "max_configurations": 5,
    "configurations_used": 4,
    "max_new_joint_updates": 20000,
    "completed_new_joint_updates": 7072,
    "max_online_transitions": 2000000,
    "completed_online_upper_bound": 526336,
    "max_new_video_count": 600,
    "max_disk_increment_gib": 80,
    "target": 1024,
    "source_step": 24,
    "best_safe_step": 24,
    "best_safe_median": 9,
    "best_safe_successes": 0,
    "post_warmup_strikes": 3,
    "next_required": "budget_end: next GPU segment projected above frozen80GiB with existing reserves; E dependency_ready, not GPU qualified",
    "final_suite": "suite_rpo_ufo_isaac_velocity_acceptance_v2",
    "final_seeds": "66000..66099",
    "final_acceptance_used": false,
    "latest_safe_train": "trn_602c852d5db243afa6ce",
    "latest_safe_run": "run_2375cb7b95504d99bf72",
    "latest_safe_model": "mv_8bfdd837d60e40a7ba5f",
    "latest_safe_eval": "ev_486def3c7a5c4b7e8aef",
    "user_requested_handoff": false,
    "continuation_reason": "User requested native architecture and reasonable recommendations",
    "previous_max_configurations": 3,
    "protocol_revision": "User explicitly requested native model alignment plus staged action/RSI/velocity optimization; allow at most C(native+stable), D(aligned control), E(soft velocity if justified), preserving global update/online/time/media/disk caps",
    "candidate_recipe": "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_slew_velocity_v5",
    "latest_native_train": "trn_affcd87370df42199698",
    "latest_native_run": "run_1fa049bcea314d86875e",
    "latest_native_model": "mv_0164d7f84f214e24af27",
    "latest_native_eval": "ev_a10a4ac19f114fb2b9d0",
    "latest_native_cancel_run": "run_9b4180c3547242919727",
    "cancelled_native_updates": 0,
    "cancelled_native_online_upper_bound": 12288,
    "latest_contact_run": "run_c4bd4473cb9b4f96b215",
    "latest_slew_train": "trn_5fe87af0fa6a4f158569",
    "latest_slew_run": "run_d1037ca56dce4f8ebaef",
    "latest_slew_model": "mv_521b1bfec83a4b4887a2",
    "latest_slew_eval": "ev_f31fcc52ce1d4dda9ea0",
    "latest_slew_cancel_run": "run_09be27c11145466dbf2f",
    "cancelled_slew_updates": 0,
    "cancelled_slew_online_upper_bound": 12288,
    "latest_slew_median": 35,
    "latest_slew_tracking_run": "run_de5c7bdfe5db44129a50",
    "last_decision": "budget_end"
  },
  "actual_storage": {
    "schema_version": "eb/ufo-storage-actual-block-upper-bound/v1",
    "start_utc": "2026-10-10T15:31:00Z",
    "managed_run_model_gib": 57.133392333984375,
    "recent_overlay_allocated_gib": 8.245555877685547,
    "recent_buildkit_allocated_gib": 0.2095947265625,
    "recent_image_metadata_gib": 0.023468017578125,
    "layer_count": 145,
    "cache_files": 205,
    "metadata_files": 739,
    "scope": "All recent host overlay directory inode blocks, excluding merged mount views; recent BuildKit/image inodes. Includes unrelated recent host work conservatively. File payloads never read. Old-root changes, assets/tmp/docs/CAS and future writable peaks need separate reserve.",
    "counted_managed_docker_gib": 65.61201095581055
  },
  "existing_unmeasured_and_future_writable_reserve_gib": 8,
  "next_D_checkpoint_reserve_gib": 8,
  "projected_D_next_gib": 81.61201095581055,
  "projected_E16_and24_gib": 86.61201095581055,
  "cap_gib": 80
}
```

## 可访问入口与证据索引

[四卡1024训练](http://194.0.106:31800/train/trn_5fe87af0fa6a4f158569) · [16case报告](http://194.0.106:31800/eval/reports/ev_f31fcc52ce1d4dda9ea0) · [heldout Run](http://194.0.106:31800/train/runs/run_de5c7bdfe5db44129a50)

```json
[
  {
    "path": "/data/kk/EmbodiedBein/docs/progress/artifacts/2026-10-11-ufo-walking-repair/slew1024-native.json",
    "bytes": 1223,
    "sha256": "8a20c66ff8d38f293a130d03e85a48f8e385403d12d5c7ff8d7d52c5b3009a0d"
  },
  {
    "path": "/data/kk/EmbodiedBein/docs/progress/artifacts/2026-10-11-ufo-walking-repair/slew1024-trace.json",
    "bytes": 980,
    "sha256": "a10ca42a9437a807cdfd5906ccc1f0110b97f5c9995b5fc0f8f9963b66486cc5"
  },
  {
    "path": "/data/kk/EmbodiedBein/docs/progress/artifacts/2026-10-11-ufo-walking-repair/slew1024-context.json",
    "bytes": 795,
    "sha256": "01593b5208696a1b4e4ee08283b7600dbde8f226ff3f65d2c30be993e7ab9ed9"
  },
  {
    "path": "/data/kk/EmbodiedBein/docs/progress/artifacts/2026-10-11-ufo-walking-repair/slew1024-final-metrics.json",
    "bytes": 2523194,
    "sha256": "0c7b1a769871f8a0d76bd782db499c8575ff76b6992044906c40f0e2584b0180"
  },
  {
    "path": "/data/kk/EmbodiedBein/docs/progress/artifacts/2026-10-11-ufo-walking-repair/slew1024-training-statistics.json",
    "bytes": 49094,
    "sha256": "168a2337996320ec598d589db7a01096b5b146ea056d64e1b7c83cd96e00bcae"
  },
  {
    "path": "/data/kk/EmbodiedBein/docs/progress/artifacts/2026-10-11-ufo-walking-repair/slew1024-training.png",
    "bytes": 465253,
    "sha256": "20c1900f2f0caaa865e5b868ac20c01a83377ff1570e2a33f91e46aca7943246"
  },
  {
    "path": "/data/kk/EmbodiedBein/docs/progress/artifacts/2026-10-11-ufo-walking-repair/slew1024-validation.png",
    "bytes": 73230,
    "sha256": "0c6ce42e094d679e776bb7289b2081d557bf4fc141dff4f152b9ef147db8e533"
  },
  {
    "path": "/data/kk/EmbodiedBein/docs/progress/artifacts/2026-10-11-ufo-walking-repair/slew1024-train-current.json",
    "bytes": 6044,
    "sha256": "897293bf1ad3c913164aa6ec14b77e579038ca94c3b9a4e5b1ca70b182684143"
  },
  {
    "path": "/data/kk/EmbodiedBein/docs/progress/artifacts/2026-10-11-ufo-walking-repair/slew1024-job-final.json",
    "bytes": 5650,
    "sha256": "a0f3a7da564b2d22b6e52d46de4723a3223f3b08330cb92bcc88375b8f01cb81"
  },
  {
    "path": "/data/kk/EmbodiedBein/docs/progress/artifacts/2026-10-11-ufo-walking-repair/slew1024-eval-preview.json",
    "bytes": 5426,
    "sha256": "b7edbabb34c400214b54593cc147bf7381d95612e5471cfb46c6a589d834760c"
  },
  {
    "path": "/data/kk/EmbodiedBein/docs/progress/artifacts/2026-10-11-ufo-walking-repair/slew1024-eval-report.json",
    "bytes": 200370,
    "sha256": "4d5065de41158941a23d4297d26ed1ff4a7d8557a43b769b75d94e7339861219"
  },
  {
    "path": "/data/kk/EmbodiedBein/docs/progress/artifacts/2026-10-11-ufo-walking-repair/slew1024-eval-cases.json",
    "bytes": 17401,
    "sha256": "3bac451282610012fe938db2c4766d1de9d1272f79686d5dd5d3fc56328497cc"
  },
  {
    "path": "/data/kk/EmbodiedBein/docs/progress/artifacts/2026-10-11-ufo-walking-repair/slew1024-eval-summary.json",
    "bytes": 358,
    "sha256": "4d5c03dcb79376a7c073d498526c6ed3db1d110ba71bd8fec8f573d83366ce5d"
  },
  {
    "path": "/data/kk/EmbodiedBein/docs/progress/artifacts/2026-10-11-ufo-walking-repair/slew1024-eval-control.json",
    "bytes": 1703,
    "sha256": "1a100bb42b732957d33e977603e5b8cc4a2aeab7ad144f14704bb23717bf3fa1"
  },
  {
    "path": "/data/kk/EmbodiedBein/docs/progress/artifacts/2026-10-11-ufo-walking-repair/slew1024-eval-verified.json",
    "bytes": 6271,
    "sha256": "49fec7f302772c1d6522da7870ad25c545e4292224603e52fb012d23522d5e45"
  },
  {
    "path": "/data/kk/EmbodiedBein/docs/progress/artifacts/2026-10-11-ufo-walking-repair/slew1024-eval-media.json",
    "bytes": 2902,
    "sha256": "9460eec32bcb8abc4ac6e916a5704a8973c915aec111d9f68aa99666e86eddaa"
  },
  {
    "path": "/data/kk/EmbodiedBein/docs/progress/artifacts/2026-10-11-ufo-walking-repair/slew1024-tracking-run.json",
    "bytes": 24087,
    "sha256": "8d07e9ab85e57d509b37d6762f5c50310d95c1962b4865fac308357f5ec216d3"
  },
  {
    "path": "/data/kk/EmbodiedBein/docs/progress/artifacts/2026-10-11-ufo-walking-repair/slew1024-tracking-verified.json",
    "bytes": 703,
    "sha256": "a4e3d6a9e2a899cf7fbb1295651ecb83a5d8b4285a04929101c57a3250d7c08e"
  },
  {
    "path": "/data/kk/EmbodiedBein/docs/progress/artifacts/2026-10-11-ufo-walking-repair/slew1024-tracking-http-media.json",
    "bytes": 417,
    "sha256": "05df50444181c1420f28cb2c4f4b6c328b4929a9fd7f2c6e3e945d6436d70d93"
  },
  {
    "path": "/data/kk/EmbodiedBein/docs/progress/artifacts/2026-10-11-ufo-walking-repair/slew1024-container-final.txt",
    "bytes": 234,
    "sha256": "ba00fe3a89c069114ebf0b43de7414e01bd7693aff1d8c8a5f00dd14f709f6a5"
  },
  {
    "path": "/data/kk/EmbodiedBein/docs/progress/artifacts/2026-10-11-ufo-walking-repair/slew1024-eval-container-final.txt",
    "bytes": 235,
    "sha256": "ac2bc3262656a1eb0e4cd9ea6310781f397df30c37f317c656e61b85cbe19ddc"
  },
  {
    "path": "/data/kk/EmbodiedBein/docs/progress/artifacts/2026-10-11-ufo-walking-repair/slew1024-tracking-container-final.txt",
    "bytes": 234,
    "sha256": "ab9f2efb7d6f335129a54653c3ec1f82d4df9fb7ef86b38722536d6f13c6dca2"
  },
  {
    "path": "/data/kk/EmbodiedBein/docs/progress/artifacts/2026-10-11-ufo-walking-repair/slew1024-resources.log",
    "bytes": 46861,
    "sha256": "3daf091a2243bbd9f78f8ba891e3f78e8697ccbfba2b18d68a40d9bc792b0f18"
  },
  {
    "path": "/data/kk/EmbodiedBein/docs/progress/artifacts/2026-10-11-ufo-walking-repair/slew1024-eval-resources.log",
    "bytes": 4684,
    "sha256": "71a1f1fec256513f0e1574ffb2f353368a9eb37c6d3097527d8bcabadfd0f50a"
  },
  {
    "path": "/data/kk/EmbodiedBein/docs/progress/artifacts/2026-10-11-ufo-walking-repair/slew1024-tracking-resources.log",
    "bytes": 735,
    "sha256": "67634835208101f634984d0b1d1997a03c61c9d40c2a9005af4e09fe160c71b6"
  },
  {
    "path": "/data/kk/EmbodiedBein/docs/progress/artifacts/2026-10-11-ufo-walking-repair/slew1024-velocity-margin-early.json",
    "bytes": 672,
    "sha256": "6ac1921e96f3e84ab96caa2dc2ecace52681fee1080dde764f7eef70f2140ef9"
  },
  {
    "path": "/data/kk/EmbodiedBein/docs/progress/artifacts/2026-10-11-ufo-walking-repair/soft-velocity-producer-verified.json",
    "bytes": 9002,
    "sha256": "e0140576d6124c19e6df9bc498ed5baa8ef7cd5b6f12d8e406b28dfaf0c41d1c"
  },
  {
    "path": "/data/kk/EmbodiedBein/docs/progress/artifacts/2026-10-11-ufo-walking-repair/soft-velocity-evaluator-verified.json",
    "bytes": 8783,
    "sha256": "324cc4b297ce8f4325680e5b7c04ce7f29897140c4d0983851737f06a2292fdb"
  },
  {
    "path": "/data/kk/EmbodiedBein/docs/progress/artifacts/2026-10-11-ufo-walking-repair/soft-velocity-core-red.log",
    "bytes": 4572,
    "sha256": "59c6a60590c19b8ffa347446d0cc0e45bd04dcbd0d6b83ba0b1052e53e6201c6"
  },
  {
    "path": "/data/kk/EmbodiedBein/docs/progress/artifacts/2026-10-11-ufo-walking-repair/soft-velocity-core-green.log",
    "bytes": 599,
    "sha256": "d17964bf873fa503ca389b3b6617e98faffad9d3c57f7b5faaf476aaab95f0e4"
  },
  {
    "path": "/data/kk/EmbodiedBein/docs/progress/artifacts/2026-10-11-ufo-walking-repair/soft-velocity-replay-red.log",
    "bytes": 4899,
    "sha256": "9de36aaaed21628db9ad4a73d466fc9eaf61ea1382465bd1c0125fba579ba8de"
  },
  {
    "path": "/data/kk/EmbodiedBein/docs/progress/artifacts/2026-10-11-ufo-walking-repair/soft-velocity-replay-green.log",
    "bytes": 600,
    "sha256": "bc2d50ddf84c1405b2946224015e145c3f0e9dc044567584d9d062805455f8ca"
  },
  {
    "path": "/data/kk/EmbodiedBein/docs/progress/artifacts/2026-10-11-ufo-walking-repair/soft-velocity-bootstrap-red.log",
    "bytes": 4138,
    "sha256": "e3c2f5b3d35fd0da1e175425225f505c2b9a81ee108a95283bfd8931659a5043"
  },
  {
    "path": "/data/kk/EmbodiedBein/docs/progress/artifacts/2026-10-11-ufo-walking-repair/soft-velocity-bootstrap-green.log",
    "bytes": 599,
    "sha256": "1ba885017e04de63d959b7960e530d78ba439e2bf427d6c2d8262e34e6716373"
  },
  {
    "path": "/data/kk/EmbodiedBein/docs/progress/artifacts/2026-10-11-ufo-walking-repair/soft-velocity-platform-red.log",
    "bytes": 1666,
    "sha256": "6f367e299f0b07a75f00306a07ba2624428b96875d8d2bea1072aebf8fd938db"
  },
  {
    "path": "/data/kk/EmbodiedBein/docs/progress/artifacts/2026-10-11-ufo-walking-repair/soft-velocity-platform-green.log",
    "bytes": 557,
    "sha256": "2375a082aee6f80d34ee45b578b32c5d6eb54e16203025613a1908373127c57d"
  },
  {
    "path": "/data/kk/EmbodiedBein/docs/progress/artifacts/2026-10-11-ufo-walking-repair/soft-velocity-cpu-closure.log",
    "bytes": 963,
    "sha256": "8961aa21a108c42f99e4bb75b4f0c4b509c68df95095b38c1d7c5e4ed00f5ce2"
  },
  {
    "path": "/data/kk/EmbodiedBein/docs/progress/artifacts/2026-10-11-ufo-walking-repair/soft-velocity-next-update.log",
    "bytes": 600,
    "sha256": "b7e3fc823473571d07367fe7539babda6b3e8f0fadc7b156b5f2c1997955a744"
  },
  {
    "path": "/data/kk/EmbodiedBein/docs/progress/artifacts/2026-10-11-ufo-walking-repair/soft-velocity-final-native.log",
    "bytes": 883,
    "sha256": "f88ccd7cd04891e4958ced57d7faa6db3a19ef01224a7ab734073228ae36d1b3"
  },
  {
    "path": "/data/kk/EmbodiedBein/docs/progress/artifacts/2026-10-11-ufo-walking-repair/soft-velocity-backend-suite.log",
    "bytes": 1119,
    "sha256": "cd86ed7e94c87711b4ebd7246aefc508f9c8d12d1ddccd48fbcf6d1d929ede83"
  },
  {
    "path": "/data/kk/EmbodiedBein/docs/progress/artifacts/2026-10-11-ufo-walking-repair/soft-velocity-final-qualification-gate.log",
    "bytes": 557,
    "sha256": "413a1af0d7cb8c7cd1204ad2f2a9572c397525f22cecb62229570f0ca6ab09a0"
  },
  {
    "path": "/data/kk/EmbodiedBein/docs/progress/artifacts/2026-10-11-ufo-walking-repair/soft-velocity-image-build.log",
    "bytes": 3265,
    "sha256": "98680aedcaf022cafe91cc6ca2090a0e4b8616dc79ef311b6a80839835889dcd"
  },
  {
    "path": "/data/kk/EmbodiedBein/docs/progress/artifacts/2026-10-11-ufo-walking-repair/soft-velocity-platform-deploy.log",
    "bytes": 7556,
    "sha256": "2089872b386ba0aa2f47244adab20cdf8da2ee4ac3975c59fdf4e4ab05d6dd07"
  },
  {
    "path": "/data/kk/EmbodiedBein/docs/progress/artifacts/2026-10-11-ufo-walking-repair/soft-velocity-deployed-templates.json",
    "bytes": 22522,
    "sha256": "58b94289471995283468a40814ea1ea2d32ae492ac7c0fb9c45e98381910e473"
  },
  {
    "path": "/data/kk/EmbodiedBein/docs/progress/artifacts/2026-10-11-ufo-walking-repair/soft-velocity-v5-review.json",
    "bytes": 11754,
    "sha256": "a7b106711eb8171f266d9597d519c5ac77a1487a2f3148e84cfbd44c187e7346"
  },
  {
    "path": "/data/kk/EmbodiedBein/docs/progress/artifacts/2026-10-11-ufo-walking-repair/soft-velocity-v5-review.md",
    "bytes": 4455,
    "sha256": "4dfdd947f81789d820940d35366a6960ac70bd275d2551ea5c13ac171c7fb349"
  },
  {
    "path": "/data/kk/EmbodiedBein/docs/progress/artifacts/2026-10-11-ufo-walking-repair/soft-velocity-implementation-review.json",
    "bytes": 11316,
    "sha256": "b706fe7d224b123db494412389ab915a15bf5f18bdb62c4380b52b1fd18d2538"
  },
  {
    "path": "/data/kk/EmbodiedBein/docs/progress/artifacts/2026-10-11-ufo-walking-repair/soft-velocity-implementation-review.md",
    "bytes": 4310,
    "sha256": "a1d51ee0b75385f08149d76699fbd142e6ce4a2f7103bd8fefaa90e8afe9f8a0"
  },
  {
    "path": "/data/kk/EmbodiedBein/docs/progress/artifacts/2026-10-11-ufo-walking-repair/storage-actual-docker-blocks.json",
    "bytes": 18326,
    "sha256": "34d78808b2db19da344f5d8a854501b946e4f3b31451845dae84a3b4c38ddc63"
  },
  {
    "path": "/data/kk/EmbodiedBein/docs/progress/artifacts/2026-10-11-ufo-walking-repair/repair-storage-current.json",
    "bytes": 9302,
    "sha256": "6f5cc4039171770f2c2833b958fb8dcea24d559c46acf826e7b02b9664bb4c77"
  },
  {
    "path": "/data/kk/EmbodiedBein/docs/progress/artifacts/2026-10-11-ufo-walking-repair/repair-session.json",
    "bytes": 2314,
    "sha256": "480270f6ac5d9a64a390b5d4329fdf4951094b245c9e0fcb0b669d05797b9d83"
  },
  {
    "path": "/data/kk/EmbodiedBein/docs/progress/artifacts/2026-10-11-ufo-walking-repair/slew1024-eval-raw-traces.json",
    "bytes": 4620105,
    "sha256": "cc48027ef9e054ac8ffd0f3a654c55dda8b33e5c89d02ec1cd743b6d0349c3bc"
  },
  {
    "path": "/data/kk/EmbodiedBein/docs/progress/artifacts/2026-10-11-ufo-walking-repair/slew1024-survival-orthogonality.png",
    "bytes": 232434,
    "sha256": "8157bb69464eb594a79ee8f5f56644a240a706bfa1f0628ef9ddff7d1d201e86"
  },
  {
    "path": "/data/kk/EmbodiedBein/docs/progress/artifacts/2026-10-11-ufo-walking-repair/slew1024-start-doctor.json",
    "bytes": 617,
    "sha256": "133294df3b1a064d07cc68b93d8c56941381f8b2f1f3b99a34a04734fd1feed6"
  }
]
```

## 补充：存活与正交原始曲线、实际硬件

![存活、完成轨迹数、正交损失与辅助归一化原始曲线](/data/kk/EmbodiedBein/docs/progress/artifacts/2026-10-11-ufo-walking-repair/slew1024-survival-orthogonality.png)

曲线直接使用实际记录点，没有补值或插值；episode_len为控制步数，每步0.02秒。该训练指标统计随机latent与RSI轨迹，不能替代独立指令行走的存活结果。D24至D1024中，critic/aux critic损失与actor梯度后期上升，也需在后续窗口对照检查。

实际宿主eai4090x4m1：4×RTX4090，每卡24564MiB，宿主内存128644MiB；训练请求96GiB/24CPU/32GiB共享内存，独立四卡速度评估96GiB/64CPU/32GiB共享内存。资源观测的逐卡峰值和RSS见上方记录，采样最大值不保证捕捉瞬时峰值。以下是D1024启动时平台doctor原始结果：

```json
{
  "hostname": "eai4090x4m1",
  "gpus": [
    {
      "index": 0,
      "name": "NVIDIA GeForce RTX 4090",
      "memory_total_mb": 24564,
      "memory_free_mb": 24080,
      "run_id": "run_d1037ca56dce4f8ebaef"
    },
    {
      "index": 1,
      "name": "NVIDIA GeForce RTX 4090",
      "memory_total_mb": 24564,
      "memory_free_mb": 24080,
      "run_id": "run_d1037ca56dce4f8ebaef"
    },
    {
      "index": 2,
      "name": "NVIDIA GeForce RTX 4090",
      "memory_total_mb": 24564,
      "memory_free_mb": 24080,
      "run_id": "run_d1037ca56dce4f8ebaef"
    },
    {
      "index": 3,
      "name": "NVIDIA GeForce RTX 4090",
      "memory_total_mb": 24564,
      "memory_free_mb": 24080,
      "run_id": "run_d1037ca56dce4f8ebaef"
    }
  ],
  "memory_total_mb": 128644,
  "has_dri": true,
  "memory_available_mb": 106349
}
```
