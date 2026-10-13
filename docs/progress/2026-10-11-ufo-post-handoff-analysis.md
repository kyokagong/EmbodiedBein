# UFO四卡交接后的续段与提示敏感性证据

记录：北京时间2026-10-11 02:20。先前完整交接文档保持截至3024的冻结快照。随后继续原有界目标，完成3024→4024第二观察窗口；同safe-v2镜像/Profile、seed42、batch128、1024宽/3层、192env、一个learner三个sampler，未改变UFO数学、数据、物理失败条件或新100例验收。没有提交代码。

本段Train `trn_57ba63fb6b994979be47` / Job `job_9cbda63459814cd7b467` / Run `run_a4b6ef0d96584b29b20c` / Model `mv_758c24169168471a80ed` 已正常结束。全部六Adam4024/hash/相同fingerprint和三sampler同步通过。累计online270336，新增64512；新增4064hard全terminated（position1242/velocity3166/重叠344），总7254终止，无torque越限，dq峰42.958，PD解码最大误差1.669e−7。本轮累计新增5024joint updates、online含取消上界347136；尚待本段独立快评才能记第二strike或改变best。

独立快评Job `job_3e78e61168934b6fb2f3` / Run `run_e4df98d9c2d64a408f85`：冻结16case/4GPU×1env，已启动，资源记录及时开启。已正常结束，Report ev_c11f14b2b4914e199922：0/16、2fall/14hard、无NaN、171实际步，median8、范围1–35，速度.597459/yaw3.853483；全部16初态/trace/video hash、解码/HTTP与4export核验通过。相比best24 median9未改善，第二后热身窗口strike2；只允许再进入既定第三窗口5024，若仍无改善则stop_adjust。

**CPU提示敏感性分析。** 使用同一最终2024模型和同一当前观测，比较实际当前z与相邻上一z。分别从23462个changed和35546个unchanged相邻pair中均匀选1024个；排除episode变化与step%100定期更新，实际行为统计另排除policy_version切换。无GPU、无网络、权重与trainer-state SHA核对通过，没有修改模型或训练数据。

| 统计 | z改变组 | z不变组 |
|---|---:|---:|
| 同观测平均abs动作差 | 0.8442001342773438 | 0.0 |
| 同观测平均abs PD目标差(rad) | 0.529042409404808 | 0.0 |
| 同观测任一目标差>1rad的样本数/1024 | 988 | 0 |
| 实际行为同policy-version的pair数 | 19135 | 28965 |
| 实际行为平均abs PD目标差(rad) | 0.6277268682959395 | 0.17184585189461946 |
| 实际行为任一目标差>1rad的比例 | 0.9875097987980141 | 0.6675643017434835 |

同观测反事实显示z方向更换足以引起大控制目标变化；实际行为比较还包含物理观测变化，不能把组间差异当随机对照因果。目标变化不是实际关节速度，不能直接把rad差除控制dt冒报dq。它加强关闭原生rollout_expert_trajectories的候选依据，但没有证明该候选能通过行走。第三配置仍未实施，仍待冻结三个后热身窗口决策。

**本段关键原始指标统计。** env指标按最多10个实际记录点的首/末窗口，其他最多50个；无补点，单位保持源tag。

| tag | n | first | last | mean | 首窗口 | 末窗口 |
|---|---:|---:|---:|---:|---:|---:|
| env/episode_len | 64 | 3.090909004 | 9.083333015 | 8.678193495 | 7.896437836 | 9.001949692 |
| env/num_trajectories | 64 | 44 | 84 | 85.03125 | 80.9 | 83.5 |
| train/ufo/fb_loss | 1000 | 11345.19727 | 7434.452148 | 8915.604181 | 10277.21779 | 7609.972471 |
| train/ufo/actor_loss | 1000 | 62.10575867 | 60.91621399 | 71.21163363 | 69.60384064 | 74.77143997 |
| train/ufo/action_saturation_fraction | 64 | 0.4173460007 | 0.4164968431 | 0.4214964034 | 0.4222758132 | 0.4214073807 |
| train/ufo/z_norm | 1000 | 15.99708176 | 16 | 16.00000109 | 16.0001446 | 16.00031111 |

**完整新Train配置/恢复关系**

```json
{
  "id": "trn_57ba63fb6b994979be47",
  "run_id": "run_a4b6ef0d96584b29b20c",
  "kind": "ppo",
  "name": "RPO UFO safe-v2 post-warmup interval to4024",
  "model_name": "RPO-UFO-safe-v2-step4024",
  "status": "succeeded",
  "template_id": "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_safe_v2",
  "image_digest": "sha256:adbca4ec9cf899e18da848b7e2a47d2df50d1f8e74e9015655b6deac193b0f99",
  "embodiment_id": "emb_roboparty_rpo",
  "dataset_ids": [
    "ds_92718168bfec4da6a01d"
  ],
  "base_model_id": null,
  "gpus_allocated": [
    0,
    1,
    2,
    3
  ],
  "ray_port": 6396,
  "params": {
    "stack": "rlinf",
    "purpose": "training",
    "execution": {
      "recipe_id": "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_safe_v2",
      "runtime_id": "rlinf",
      "provider_id": "ufo",
      "topology_id": "one_learner_three_samplers",
      "algorithm_id": "bfm_fb_cpr_aux",
      "simulator_id": "isaaclab",
      "progress_unit": "learner_joint_update",
      "schema_version": "eb/training-binding/v1",
      "checkpoint_format": "ufo_agent_state_v1"
    },
    "model_name": "RPO-UFO-safe-v2-step4024",
    "config_name": "rpo_bfm_ufo_isaac",
    "resume_mode": "trainer_state",
    "resume_step": 3024,
    "template_id": "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_safe_v2",
    "source_receipt": {
      "ufo": {
        "tree": "3a6b8b0d6a04287f37c1d7834e51561a8e014fc9",
        "scope": "agent Python source; binary assets excluded",
        "commit": "0b89378de02f77712072d983bc43ec2a46455dc0",
        "license": "CC-BY-NC-4.0",
        "content_hash": "sha256:2191cf26c77262f7ef2e3525ca23878a8f08c46fd02a20b7776a791f8f5e7582"
      },
      "rlinf": {
        "commit": "018e6a9e0da60f96eeda8b11008543911027d414",
        "license": "Apache-2.0",
        "content_hash": "sha256:dbaa8bed56be412f6d012ee34030822941945786cede6bf9bc3de0fec5445d48",
        "base_patch_hash": "sha256:df5e404b711161d7068894890d1a90bdcf61f064c8c9bfea60210899ecf787a5"
      },
      "scope": "agent-only; Python3.10 package metadata bypassed by tested source import",
      "packages": {
        "torch": "2.7.0+cu128",
        "python": "3.11.16",
        "pydantic": "2.11.5",
        "gymnasium": "1.2.0",
        "tensordict": "0.11.0",
        "safetensors": "0.6.2"
      },
      "patch_hash": "sha256:f99aa821d9ccdf436f91010a7b175209a2bda0f32f40afe8e0a6b49ddaf7d5b5",
      "schema_version": "eb/ufo-image-source/v1",
      "topology_patch_hash": "sha256:3078506628924464cb42d0b2f108b3fab502a43fd75e091f358d504565294d23",
      "learning_profile_patch_hash": "sha256:d47623e48b4cc757d5e63336a1fa7d49234c3833a2f1c6bbdbee4fe7ffd9a272"
    },
    "environment_profile": {
      "id": "rpo_bfm_ufo_isaac_4gpu_rsi_safe_v2",
      "hash": "sha256:58ca340bbcca2301ef3f40b5fc14f20f91102b94498d49a0342fd9c2dacf1591",
      "definition": {
        "task": "RPO v1 codec、仅train RobotState RSI；原生UFO agent及四卡拓扑不变。",
        "reward": "八项原生raw aux与权重不变；flat/无噪声保持单因素对照。",
        "failure": "继承高度/倾斜/接触终止；追加与验收相同的位置/速度/力矩硬限位(容差0.001)，RSI拒绝已越限参考帧。",
        "success": "先16→24完整恢复资格；训练正常结束不代表行走合格。每步记录requested/applied/target/速度/力矩及越限。",
        "settings": {}
      }
    },
    "resume_source_image": "eb/rlinf:rpo-bfm-ufo-agent-4gpu-safe-v2",
    "source_train_job_id": "trn_8f995f148c7347398519",
    "target_learner_updates": 4024
  },
  "inputs": {
    "datasets": [
      {
        "id": "ds_92718168bfec4da6a01d",
        "path": "/data/kk/data/ds_92718168bfec4da6a01d/content",
        "purpose": "behavior_prior",
        "source_uri": "ds_27fbaa34696e4f26b2c7",
        "validation": {
          "scope": "rpo_ufo_robot_state_v1",
          "status": "validated",
          "unchecked": [
            "RSI physics probe",
            "native learning qualification"
          ],
          "asset_sha256": "b58868a782d8863ac38059eb4eab89d27ee1d70565bf58f5a055f44510c8e842",
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
          "manifest_sha256": "7e47773bf53588ee200cccc5bbde3999df172a11593d7359cbeca2a00ac4c5d8"
        }
      }
    ],
    "base_model": null,
    "resume_from": "run_f96169ae70bc4012be13"
  },
  "resumed_from": "trn_8f995f148c7347398519",
  "resume_step": 3024,
  "resume_mode": "trainer_state",
  "hydra_overrides": [
    "env.train.total_num_envs=192",
    "env.eval.total_num_envs=3",
    "+actor.model.topology_id=one_learner_three_samplers",
    "+actor.model.learning_profile=physical_limits_v2",
    "env.train.init_params.id=RPO-UFO-Isaac-RSI-safe-v2",
    "actor.seed=42",
    "~cluster.component_placement",
    "+cluster.component_placement={actor:0,env\\,\\ rollout:1-3}",
    "runner.logger.log_path=/workspace/output",
    "hydra.run.dir=/workspace/output/hydra",
    "runner.logger.experiment_name=\"RPO UFO safe-v2 post-warmup interval to4024\"",
    "runner.resume_dir=/workspace/resume/global_step_3024"
  ],
  "run_dir": "/data/kk/embodiedbein/runs/run_a4b6ef0d96584b29b20c",
  "auto_register_latest": false,
  "metrics_summary": {
    "checkpoint": {
      "path": "/data/kk/models/runs/run_a4b6ef0d96584b29b20c",
      "step": 4024,
      "bytes": 1648568057
    },
    "sampler_sync": {
      "identity": {
        "version": 4016,
        "state_sha256": "14ba0b401e5fc23d54fa31982d997ae806038108a6e684734b59e7345dcdf7f5",
        "online_transitions": 269568
      },
      "samplers": [
        {
          "rank": 0,
          "version": 4016,
          "state_sha256": "14ba0b401e5fc23d54fa31982d997ae806038108a6e684734b59e7345dcdf7f5",
          "online_transitions": 269568
        },
        {
          "rank": 1,
          "version": 4016,
          "state_sha256": "14ba0b401e5fc23d54fa31982d997ae806038108a6e684734b59e7345dcdf7f5",
          "online_transitions": 269568
        },
        {
          "rank": 2,
          "version": 4016,
          "state_sha256": "14ba0b401e5fc23d54fa31982d997ae806038108a6e684734b59e7345dcdf7f5",
          "online_transitions": 269568
        }
      ],
      "schema_version": "eb/ufo-sampler-sync/v1",
      "sampling_rounds": 351
    },
    "model_version_id": "mv_758c24169168471a80ed"
  },
  "error_message": null,
  "job_id": "job_9cbda63459814cd7b467",
  "started_at": "2026-10-10T18:07:31.144884+00:00",
  "finished_at": "2026-10-10T18:15:45.409944+00:00",
  "created_at": "2026-10-10T18:07:28.360388+00:00"
}
```

**完整4024 checkpoint回读**

```json
{
  "step": 4024,
  "counters": {
    "learner_joint_updates": 4024,
    "sampling_rounds": 352,
    "online_transitions": 270336,
    "next_trigger": 271360
  },
  "six_adam_steps": {
    "forward_optimizer": 4024,
    "backward_optimizer": 4024,
    "discriminator_optimizer": 4024,
    "actor_optimizer": 4024,
    "critic_optimizer": 4024,
    "aux_critic_optimizer": 4024
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
  "fingerprint": "sha256:f5b2a521243bda9ad31e8ed78510d84bd0953254b7ebaed95152b7148270c0e1",
  "execution": {
    "schema_version": "eb/training-binding/v1",
    "recipe_id": "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_safe_v2",
    "algorithm_id": "bfm_fb_cpr_aux",
    "provider_id": "ufo",
    "runtime_id": "rlinf",
    "simulator_id": "isaaclab",
    "progress_unit": "learner_joint_update",
    "checkpoint_format": "ufo_agent_state_v1",
    "topology_id": "one_learner_three_samplers"
  }
}
```

**完整新增trace核验**

```json
{
  "rounds": 84,
  "rows": 64512,
  "trace_action_identity_match": true,
  "hard_limit_transitions": 4064,
  "every_hard_limit_terminated": true,
  "terminated_transitions": 7254,
  "violations": [
    1242.0,
    3166.0,
    0.0
  ],
  "applied_action_abs_max": 0.9999989867210388,
  "joint_velocity_abs_max": 42.957969665527344,
  "pd_decoder_max_error": 1.6689300541550267e-07
}
```

**完整CPU提示敏感性设置/结果**

```json
{
  "schema_version": "eb/ufo-frozen-z-action-sensitivity/v1",
  "source_run_id": "run_298bb473985642c29436",
  "model_step": 2024,
  "weights_sha256": "c6ce77c4842d4e50223a8521cf9833c58b15df16b88351a26f96606464831a77",
  "trainer_state_sha256": "b65d5d5c511baa258f01beb4ac241625b9ac4b36dbcc46a3135384756f1f2b6c",
  "cuda_available": false,
  "results": {
    "changed": {
      "eligible_pairs": 23462,
      "measured_pairs": 1024,
      "sample_selection": "deterministic evenly spaced by rank/time, no GPU training seed change",
      "mean_abs_action_delta": 0.8442001342773438,
      "maximum_abs_action_delta": 2.0,
      "action_delta_l2_quantiles": [
        0.00932975672185421,
        5.9097442626953125,
        7.028026390075683,
        7.949216365814209
      ],
      "target_delta_abs_mean_rad": 0.529042409404808,
      "target_delta_abs_max_rad": 4.4999997317790985,
      "samples_with_any_target_jump_over_half_rad": 1020,
      "samples_with_any_target_jump_over_one_rad": 988,
      "actual_recorded_behavior": {
        "same_policy_version_pairs": 19135,
        "mean_abs_action_delta": 0.9719471335411072,
        "target_delta_abs_mean_rad": 0.6277268682959395,
        "target_delta_abs_max_rad": 4.499995440244675,
        "any_target_jump_over_half_rad_fraction": 0.9993206166710217,
        "any_target_jump_over_one_rad_fraction": 0.9875097987980141,
        "interpretation": "Actual consecutive behavior, both observation and z may differ; excludes policy broadcasts, cannot isolate z causality"
      },
      "per_joint_mean_target_delta_rad": {
        "left_thigh_yaw_joint": 0.25275005455114297,
        "right_thigh_yaw_joint": 0.42014699056744576,
        "torso_joint": 0.43512910519348225,
        "left_thigh_roll_joint": 0.24207456148724305,
        "right_thigh_roll_joint": 0.22001148594608821,
        "left_arm_pitch_joint": 0.4418160101704416,
        "right_arm_pitch_joint": 0.46241778207331663,
        "left_thigh_pitch_joint": 1.0285483555671817,
        "right_thigh_pitch_joint": 0.6602251845397404,
        "left_arm_roll_joint": 0.3029794947942719,
        "right_arm_roll_joint": 0.18008013782309717,
        "left_knee_joint": 1.729106255719671,
        "right_knee_joint": 1.5150528017838951,
        "left_arm_yaw_joint": 0.7279085102018144,
        "right_arm_yaw_joint": 0.4359086975164246,
        "left_ankle_pitch_joint": 0.45348644359910395,
        "right_ankle_pitch_joint": 0.6958998422924196,
        "left_elbow_pitch_joint": 0.42858986045757774,
        "right_elbow_pitch_joint": 0.7297341203084216,
        "left_ankle_roll_joint": 0.22213162051775726,
        "right_ankle_roll_joint": 0.253233940951759,
        "left_elbow_yaw_joint": 0.14819777465891093,
        "right_elbow_yaw_joint": 0.18254638558937586
      }
    },
    "unchanged": {
      "eligible_pairs": 35546,
      "measured_pairs": 1024,
      "sample_selection": "deterministic evenly spaced by rank/time, no GPU training seed change",
      "mean_abs_action_delta": 0.0,
      "maximum_abs_action_delta": 0.0,
      "action_delta_l2_quantiles": [
        0.0,
        0.0,
        0.0,
        0.0
      ],
      "target_delta_abs_mean_rad": 0.0,
      "target_delta_abs_max_rad": 0.0,
      "samples_with_any_target_jump_over_half_rad": 0,
      "samples_with_any_target_jump_over_one_rad": 0,
      "actual_recorded_behavior": {
        "same_policy_version_pairs": 28965,
        "mean_abs_action_delta": 0.20833800733089447,
        "target_delta_abs_mean_rad": 0.17184585189461946,
        "target_delta_abs_max_rad": 4.499995440244675,
        "any_target_jump_over_half_rad_fraction": 0.8685655100983947,
        "any_target_jump_over_one_rad_fraction": 0.6675643017434835,
        "interpretation": "Actual consecutive behavior, both observation and z may differ; excludes policy broadcasts, cannot isolate z causality"
      },
      "per_joint_mean_target_delta_rad": {
        "left_thigh_yaw_joint": 0.0,
        "right_thigh_yaw_joint": 0.0,
        "torso_joint": 0.0,
        "left_thigh_roll_joint": 0.0,
        "right_thigh_roll_joint": 0.0,
        "left_arm_pitch_joint": 0.0,
        "right_arm_pitch_joint": 0.0,
        "left_thigh_pitch_joint": 0.0,
        "right_thigh_pitch_joint": 0.0,
        "left_arm_roll_joint": 0.0,
        "right_arm_roll_joint": 0.0,
        "left_knee_joint": 0.0,
        "right_knee_joint": 0.0,
        "left_arm_yaw_joint": 0.0,
        "right_arm_yaw_joint": 0.0,
        "left_ankle_pitch_joint": 0.0,
        "right_ankle_pitch_joint": 0.0,
        "left_elbow_pitch_joint": 0.0,
        "right_elbow_pitch_joint": 0.0,
        "left_ankle_roll_joint": 0.0,
        "right_ankle_roll_joint": 0.0,
        "left_elbow_yaw_joint": 0.0,
        "right_elbow_yaw_joint": 0.0
      }
    }
  },
  "interpretation": "Same frozen final2024 policy and same current observation; change only recorded z versus previous z. This measures action sensitivity, not actual historical behavior, causal training quality, or a GPU ablation."
}
```

完整原始指标与图：[4024 metrics](/data/kk/EmbodiedBein/docs/progress/artifacts/2026-10-11-ufo-walking-repair/safe4024-metrics.json)、[曲线](/data/kk/EmbodiedBein/docs/progress/artifacts/2026-10-11-ufo-walking-repair/safe4024-metrics.png)。原始交接资料：[完整文档](/data/kk/EmbodiedBein/docs/progress/2026-10-11-ufo-walking-handoff.md)、[逐点文档](/data/kk/EmbodiedBein/docs/progress/2026-10-11-ufo-walking-raw-metrics.md)。仍未达到行走标准，新的66000–66099未运行。

**4024完整独立评估case结果及核验。**

**safe4024-eval-cases.json**

```json
{
  "items": [
    {
      "seed": 61000,
      "case_id": "rpo-bfm-seed-61000",
      "task_id": "tsk_rpo_bfm_pretrain",
      "completed": true,
      "success": false,
      "actual_steps": 14,
      "episode_len": 14,
      "fall": true,
      "nan": false,
      "hard_limit": false,
      "reason": "fall",
      "env_index": 0,
      "terrain_seed": 64000,
      "initial_state": {
        "path": "traces/seed_61000.initial.json",
        "sha256": "9355713f3c11b4d99028451c26fcdcb5550f15f269fac2c9bd1b1d87c68c8776"
      },
      "video_rel": "seed_61000/0.mp4",
      "metric_steps": 14,
      "metrics_available": true,
      "velocity_rmse": 0.5782077428932866,
      "yaw_rmse": 3.8444218356400763,
      "status": "task_failure",
      "termination_reason": "fall",
      "trace": {
        "path": "traces/seed_61000.jsonl.gz",
        "sha256": "a786569da736afea6f481c8a4e04826a8e41176e76aaa59a6177b98360e47d40",
        "policy_steps": 14,
        "metric_steps": 14
      },
      "video_integrity": {
        "schema_version": "eb/bfm-case-video/v1",
        "sha256": "1ee217a9f2514ce137b94cf198c75fea68aacdcc5c70a9bff94a7bd07cbf1f02",
        "decoded_frames": 4,
        "resolution": [
          320,
          240
        ],
        "fps": 10
      },
      "worker_rank": 0,
      "run_id": "run_e4df98d9c2d64a408f85",
      "plan_hash": "sha256:5544b07474387bdbc7e6cdfe9c4b22fecd2e81072421b69423f7451e2dbdab0a",
      "video_status": "available"
    },
    {
      "seed": 61001,
      "case_id": "rpo-bfm-seed-61001",
      "task_id": "tsk_rpo_bfm_pretrain",
      "completed": true,
      "success": false,
      "actual_steps": 24,
      "episode_len": 24,
      "fall": false,
      "nan": false,
      "hard_limit": true,
      "reason": "physical_hard_limit",
      "env_index": 0,
      "terrain_seed": 64000,
      "initial_state": {
        "path": "traces/seed_61001.initial.json",
        "sha256": "4f903e8bb0e113b96b36b6c3d2faeb352cff494646a9f9741ae65d54b9c8e815"
      },
      "video_rel": "seed_61001/0.mp4",
      "metric_steps": 24,
      "metrics_available": true,
      "velocity_rmse": 0.44248953594254115,
      "yaw_rmse": 3.1355990877370257,
      "status": "task_failure",
      "termination_reason": "physical_hard_limit",
      "trace": {
        "path": "traces/seed_61001.jsonl.gz",
        "sha256": "c8b16252b276bb9424ad7248290423e11b4126e7c29ef19888f0ae1081e7e6c1",
        "policy_steps": 24,
        "metric_steps": 24
      },
      "video_integrity": {
        "schema_version": "eb/bfm-case-video/v1",
        "sha256": "1d32bf08fda78db5a9a478cf0eac578640ea30a3d917da280d6fd1dc37b3a0cf",
        "decoded_frames": 6,
        "resolution": [
          320,
          240
        ],
        "fps": 10
      },
      "worker_rank": 1,
      "run_id": "run_e4df98d9c2d64a408f85",
      "plan_hash": "sha256:5544b07474387bdbc7e6cdfe9c4b22fecd2e81072421b69423f7451e2dbdab0a",
      "video_status": "available"
    },
    {
      "seed": 61002,
      "case_id": "rpo-bfm-seed-61002",
      "task_id": "tsk_rpo_bfm_pretrain",
      "completed": true,
      "success": false,
      "actual_steps": 4,
      "episode_len": 4,
      "fall": false,
      "nan": false,
      "hard_limit": true,
      "reason": "physical_hard_limit",
      "env_index": 0,
      "terrain_seed": 64000,
      "initial_state": {
        "path": "traces/seed_61002.initial.json",
        "sha256": "09774f8cdf88e0e99362e9b080df959962e49ebb889b7f730bd451dfac73d14a"
      },
      "video_rel": "seed_61002/0.mp4",
      "metric_steps": 4,
      "metrics_available": true,
      "velocity_rmse": 0.21468241863010548,
      "yaw_rmse": 3.5195926140806075,
      "status": "task_failure",
      "termination_reason": "physical_hard_limit",
      "trace": {
        "path": "traces/seed_61002.jsonl.gz",
        "sha256": "38b5b3870e8436ae8205c058a0e718e886731899b12a1b88fd97865c841e7775",
        "policy_steps": 4,
        "metric_steps": 4
      },
      "video_integrity": {
        "schema_version": "eb/bfm-case-video/v1",
        "sha256": "98c677112c1f6214d855b29fa281ad17f455ff4271122dd854e61def6a3e663d",
        "decoded_frames": 2,
        "resolution": [
          320,
          240
        ],
        "fps": 10
      },
      "worker_rank": 2,
      "run_id": "run_e4df98d9c2d64a408f85",
      "plan_hash": "sha256:5544b07474387bdbc7e6cdfe9c4b22fecd2e81072421b69423f7451e2dbdab0a",
      "video_status": "available"
    },
    {
      "seed": 61003,
      "case_id": "rpo-bfm-seed-61003",
      "task_id": "tsk_rpo_bfm_pretrain",
      "completed": true,
      "success": false,
      "actual_steps": 3,
      "episode_len": 3,
      "fall": false,
      "nan": false,
      "hard_limit": true,
      "reason": "physical_hard_limit",
      "env_index": 0,
      "terrain_seed": 64000,
      "initial_state": {
        "path": "traces/seed_61003.initial.json",
        "sha256": "dc8f314089ef02fcb831b9eda02283c4a9d776a442c935dccf567a24f4469360"
      },
      "video_rel": "seed_61003/0.mp4",
      "metric_steps": 3,
      "metrics_available": true,
      "velocity_rmse": 0.7220099444433037,
      "yaw_rmse": 2.4965607784600508,
      "status": "task_failure",
      "termination_reason": "physical_hard_limit",
      "trace": {
        "path": "traces/seed_61003.jsonl.gz",
        "sha256": "993421384f66f0f64769b111bcb3d3897d174bf0ee16dd1595564169c2665042",
        "policy_steps": 3,
        "metric_steps": 3
      },
      "video_integrity": {
        "schema_version": "eb/bfm-case-video/v1",
        "sha256": "a371734dfc596cbde94c9b740e3139d8cf58fa01291d0fa1c50cc2b3f2dbbd8b",
        "decoded_frames": 2,
        "resolution": [
          320,
          240
        ],
        "fps": 10
      },
      "worker_rank": 3,
      "run_id": "run_e4df98d9c2d64a408f85",
      "plan_hash": "sha256:5544b07474387bdbc7e6cdfe9c4b22fecd2e81072421b69423f7451e2dbdab0a",
      "video_status": "available"
    },
    {
      "seed": 61004,
      "case_id": "rpo-bfm-seed-61004",
      "task_id": "tsk_rpo_bfm_pretrain",
      "completed": true,
      "success": false,
      "actual_steps": 9,
      "episode_len": 9,
      "fall": true,
      "nan": false,
      "hard_limit": false,
      "reason": "fall",
      "env_index": 0,
      "terrain_seed": 64000,
      "initial_state": {
        "path": "traces/seed_61004.initial.json",
        "sha256": "52ac5aafb143a7f8ad48cc403cf7c8e467a6d6279171fd964d3bf60609835d5a"
      },
      "video_rel": "seed_61004/0.mp4",
      "metric_steps": 9,
      "metrics_available": true,
      "velocity_rmse": 0.439733584647206,
      "yaw_rmse": 3.3926006053971176,
      "status": "task_failure",
      "termination_reason": "fall",
      "trace": {
        "path": "traces/seed_61004.jsonl.gz",
        "sha256": "b0fe5c1d5d93296147429b9829d6a12aa55217a7e63a47e77a42578eb3b2c2a3",
        "policy_steps": 9,
        "metric_steps": 9
      },
      "video_integrity": {
        "schema_version": "eb/bfm-case-video/v1",
        "sha256": "08570a7db92a25432e5f751859cc5938e3919daa1e5119b853bd8e8ad2c8cebe",
        "decoded_frames": 3,
        "resolution": [
          320,
          240
        ],
        "fps": 10
      },
      "worker_rank": 0,
      "run_id": "run_e4df98d9c2d64a408f85",
      "plan_hash": "sha256:5544b07474387bdbc7e6cdfe9c4b22fecd2e81072421b69423f7451e2dbdab0a",
      "video_status": "available"
    },
    {
      "seed": 61005,
      "case_id": "rpo-bfm-seed-61005",
      "task_id": "tsk_rpo_bfm_pretrain",
      "completed": true,
      "success": false,
      "actual_steps": 8,
      "episode_len": 8,
      "fall": false,
      "nan": false,
      "hard_limit": true,
      "reason": "physical_hard_limit",
      "env_index": 0,
      "terrain_seed": 64000,
      "initial_state": {
        "path": "traces/seed_61005.initial.json",
        "sha256": "c2d1ea2b47ef3bcc4cd1db8a8c058ff337d0842a09279b881935dd8bb5b3c743"
      },
      "video_rel": "seed_61005/0.mp4",
      "metric_steps": 8,
      "metrics_available": true,
      "velocity_rmse": 0.9957370847326084,
      "yaw_rmse": 3.2552818475535217,
      "status": "task_failure",
      "termination_reason": "physical_hard_limit",
      "trace": {
        "path": "traces/seed_61005.jsonl.gz",
        "sha256": "d7e166922c886c61b05bdeac5610955d1b8e52df3123c5b649710e9cc61e13ff",
        "policy_steps": 8,
        "metric_steps": 8
      },
      "video_integrity": {
        "schema_version": "eb/bfm-case-video/v1",
        "sha256": "3b9bbed5257d910a02c3d5c709cf83246d90fc532964eec601b9eed8ce8b870a",
        "decoded_frames": 3,
        "resolution": [
          320,
          240
        ],
        "fps": 10
      },
      "worker_rank": 1,
      "run_id": "run_e4df98d9c2d64a408f85",
      "plan_hash": "sha256:5544b07474387bdbc7e6cdfe9c4b22fecd2e81072421b69423f7451e2dbdab0a",
      "video_status": "available"
    },
    {
      "seed": 61006,
      "case_id": "rpo-bfm-seed-61006",
      "task_id": "tsk_rpo_bfm_pretrain",
      "completed": true,
      "success": false,
      "actual_steps": 35,
      "episode_len": 35,
      "fall": false,
      "nan": false,
      "hard_limit": true,
      "reason": "physical_hard_limit",
      "env_index": 0,
      "terrain_seed": 64000,
      "initial_state": {
        "path": "traces/seed_61006.initial.json",
        "sha256": "91671228f15a66f3b198ae5728ec1da109b561fe674462cc9a7d473504267f3a"
      },
      "video_rel": "seed_61006/0.mp4",
      "metric_steps": 35,
      "metrics_available": true,
      "velocity_rmse": 0.3489858053696962,
      "yaw_rmse": 3.381213512570117,
      "status": "task_failure",
      "termination_reason": "physical_hard_limit",
      "trace": {
        "path": "traces/seed_61006.jsonl.gz",
        "sha256": "2b0263ba285b51aa4be690efdb858ba523461a4d664de082d7cdf32d9e6a2c2f",
        "policy_steps": 35,
        "metric_steps": 35
      },
      "video_integrity": {
        "schema_version": "eb/bfm-case-video/v1",
        "sha256": "4605222678aed6718a5b2da3fbb0db9d9aac486b6116d0c0f81bfa8c1200101f",
        "decoded_frames": 8,
        "resolution": [
          320,
          240
        ],
        "fps": 10
      },
      "worker_rank": 2,
      "run_id": "run_e4df98d9c2d64a408f85",
      "plan_hash": "sha256:5544b07474387bdbc7e6cdfe9c4b22fecd2e81072421b69423f7451e2dbdab0a",
      "video_status": "available"
    },
    {
      "seed": 61007,
      "case_id": "rpo-bfm-seed-61007",
      "task_id": "tsk_rpo_bfm_pretrain",
      "completed": true,
      "success": false,
      "actual_steps": 6,
      "episode_len": 6,
      "fall": false,
      "nan": false,
      "hard_limit": true,
      "reason": "physical_hard_limit",
      "env_index": 0,
      "terrain_seed": 64000,
      "initial_state": {
        "path": "traces/seed_61007.initial.json",
        "sha256": "081dc63584d40e2102c1df67c578b0be7bf85265e25cc07abb12cec270e50616"
      },
      "video_rel": "seed_61007/0.mp4",
      "metric_steps": 6,
      "metrics_available": true,
      "velocity_rmse": 0.6946357080236758,
      "yaw_rmse": 6.962753459051482,
      "status": "task_failure",
      "termination_reason": "physical_hard_limit",
      "trace": {
        "path": "traces/seed_61007.jsonl.gz",
        "sha256": "96fb7b4ebc349078f1b530b0d119c39c64d0f5b559407385bf04e8d058c1f44c",
        "policy_steps": 6,
        "metric_steps": 6
      },
      "video_integrity": {
        "schema_version": "eb/bfm-case-video/v1",
        "sha256": "07210dd0b8b5c7d8d2bf262edfee05f98fe2f9470ebf123fdcb923f24535d767",
        "decoded_frames": 3,
        "resolution": [
          320,
          240
        ],
        "fps": 10
      },
      "worker_rank": 3,
      "run_id": "run_e4df98d9c2d64a408f85",
      "plan_hash": "sha256:5544b07474387bdbc7e6cdfe9c4b22fecd2e81072421b69423f7451e2dbdab0a",
      "video_status": "available"
    },
    {
      "seed": 61008,
      "case_id": "rpo-bfm-seed-61008",
      "task_id": "tsk_rpo_bfm_pretrain",
      "completed": true,
      "success": false,
      "actual_steps": 19,
      "episode_len": 19,
      "fall": false,
      "nan": false,
      "hard_limit": true,
      "reason": "physical_hard_limit",
      "env_index": 0,
      "terrain_seed": 64000,
      "initial_state": {
        "path": "traces/seed_61008.initial.json",
        "sha256": "bc19f0cd49212100689a30db90ec50a263f7439a83438084023a2a8d2a3be0a4"
      },
      "video_rel": "seed_61008/0.mp4",
      "metric_steps": 19,
      "metrics_available": true,
      "velocity_rmse": 0.7830652531819361,
      "yaw_rmse": 3.14173488948136,
      "status": "task_failure",
      "termination_reason": "physical_hard_limit",
      "trace": {
        "path": "traces/seed_61008.jsonl.gz",
        "sha256": "05062666b08ab90b9ee2e9723541e19413a6e36c4ab84a31855716e0d88511f1",
        "policy_steps": 19,
        "metric_steps": 19
      },
      "video_integrity": {
        "schema_version": "eb/bfm-case-video/v1",
        "sha256": "ec47c152732e14ed28c67c971d3e03e76cc72af4f8d7eaabb550ca81020a17b4",
        "decoded_frames": 5,
        "resolution": [
          320,
          240
        ],
        "fps": 10
      },
      "worker_rank": 0,
      "run_id": "run_e4df98d9c2d64a408f85",
      "plan_hash": "sha256:5544b07474387bdbc7e6cdfe9c4b22fecd2e81072421b69423f7451e2dbdab0a",
      "video_status": "available"
    },
    {
      "seed": 61009,
      "case_id": "rpo-bfm-seed-61009",
      "task_id": "tsk_rpo_bfm_pretrain",
      "completed": true,
      "success": false,
      "actual_steps": 5,
      "episode_len": 5,
      "fall": false,
      "nan": false,
      "hard_limit": true,
      "reason": "physical_hard_limit",
      "env_index": 0,
      "terrain_seed": 64000,
      "initial_state": {
        "path": "traces/seed_61009.initial.json",
        "sha256": "474ef3818d71404b545da320d6b6f322723f20c36eca2849f0d3f62235d6124c"
      },
      "video_rel": "seed_61009/0.mp4",
      "metric_steps": 5,
      "metrics_available": true,
      "velocity_rmse": 0.44311589790817324,
      "yaw_rmse": 3.4618933800562486,
      "status": "task_failure",
      "termination_reason": "physical_hard_limit",
      "trace": {
        "path": "traces/seed_61009.jsonl.gz",
        "sha256": "5522e9583e31278ec2df784b5aa6b9fd56afe056212830b50b90c2f53775484a",
        "policy_steps": 5,
        "metric_steps": 5
      },
      "video_integrity": {
        "schema_version": "eb/bfm-case-video/v1",
        "sha256": "78de03f5044be179ecb0bc4295da1521869a66891cd5a5eff2fe19550e8b169c",
        "decoded_frames": 2,
        "resolution": [
          320,
          240
        ],
        "fps": 10
      },
      "worker_rank": 1,
      "run_id": "run_e4df98d9c2d64a408f85",
      "plan_hash": "sha256:5544b07474387bdbc7e6cdfe9c4b22fecd2e81072421b69423f7451e2dbdab0a",
      "video_status": "available"
    },
    {
      "seed": 61010,
      "case_id": "rpo-bfm-seed-61010",
      "task_id": "tsk_rpo_bfm_pretrain",
      "completed": true,
      "success": false,
      "actual_steps": 8,
      "episode_len": 8,
      "fall": false,
      "nan": false,
      "hard_limit": true,
      "reason": "physical_hard_limit",
      "env_index": 0,
      "terrain_seed": 64000,
      "initial_state": {
        "path": "traces/seed_61010.initial.json",
        "sha256": "ba593ba53376269d7c0dcd7b14d69a12063208d858817963b60b5d276373fd97"
      },
      "video_rel": "seed_61010/0.mp4",
      "metric_steps": 8,
      "metrics_available": true,
      "velocity_rmse": 0.932185879219352,
      "yaw_rmse": 3.962191203218409,
      "status": "task_failure",
      "termination_reason": "physical_hard_limit",
      "trace": {
        "path": "traces/seed_61010.jsonl.gz",
        "sha256": "a0ff3cd7babfc5276ea55800080714825cc62f773518ce11eba5f48648d6f128",
        "policy_steps": 8,
        "metric_steps": 8
      },
      "video_integrity": {
        "schema_version": "eb/bfm-case-video/v1",
        "sha256": "64e3705b353921f28c24b03d8a8f49b13410cd50fa30bbc2d7bdb9185ae729ea",
        "decoded_frames": 3,
        "resolution": [
          320,
          240
        ],
        "fps": 10
      },
      "worker_rank": 2,
      "run_id": "run_e4df98d9c2d64a408f85",
      "plan_hash": "sha256:5544b07474387bdbc7e6cdfe9c4b22fecd2e81072421b69423f7451e2dbdab0a",
      "video_status": "available"
    },
    {
      "seed": 61011,
      "case_id": "rpo-bfm-seed-61011",
      "task_id": "tsk_rpo_bfm_pretrain",
      "completed": true,
      "success": false,
      "actual_steps": 3,
      "episode_len": 3,
      "fall": false,
      "nan": false,
      "hard_limit": true,
      "reason": "physical_hard_limit",
      "env_index": 0,
      "terrain_seed": 64000,
      "initial_state": {
        "path": "traces/seed_61011.initial.json",
        "sha256": "2e17e83e748754629804fbd3130b0fbb63cdd06ddc8638dc7a442a16199543e3"
      },
      "video_rel": "seed_61011/0.mp4",
      "metric_steps": 3,
      "metrics_available": true,
      "velocity_rmse": 0.5409708051633503,
      "yaw_rmse": 3.8283910385751954,
      "status": "task_failure",
      "termination_reason": "physical_hard_limit",
      "trace": {
        "path": "traces/seed_61011.jsonl.gz",
        "sha256": "55c3165b5bfd8440335850b55f11a99f62db660ad8a759130d01a5cc102cfb38",
        "policy_steps": 3,
        "metric_steps": 3
      },
      "video_integrity": {
        "schema_version": "eb/bfm-case-video/v1",
        "sha256": "a74ef6a579c822eb774a5e4eec8c980fcf34d856c4e5eac1b56ecc985c4e6d67",
        "decoded_frames": 2,
        "resolution": [
          320,
          240
        ],
        "fps": 10
      },
      "worker_rank": 3,
      "run_id": "run_e4df98d9c2d64a408f85",
      "plan_hash": "sha256:5544b07474387bdbc7e6cdfe9c4b22fecd2e81072421b69423f7451e2dbdab0a",
      "video_status": "available"
    },
    {
      "seed": 61012,
      "case_id": "rpo-bfm-seed-61012",
      "task_id": "tsk_rpo_bfm_pretrain",
      "completed": true,
      "success": false,
      "actual_steps": 1,
      "episode_len": 1,
      "fall": false,
      "nan": false,
      "hard_limit": true,
      "reason": "physical_hard_limit",
      "env_index": 0,
      "terrain_seed": 64000,
      "initial_state": {
        "path": "traces/seed_61012.initial.json",
        "sha256": "a7d3a6a5b6e84a334db8a40dc158fc1233f7541ad5e556425e304f381016b31e"
      },
      "video_rel": "seed_61012/0.mp4",
      "metric_steps": 1,
      "metrics_available": true,
      "velocity_rmse": 0.3520595841758019,
      "yaw_rmse": 1.735783040523529,
      "status": "task_failure",
      "termination_reason": "physical_hard_limit",
      "trace": {
        "path": "traces/seed_61012.jsonl.gz",
        "sha256": "658c4b387c925c1fa7c0143bccc910da1f8669c5abe097d728b61ef8e9d17241",
        "policy_steps": 1,
        "metric_steps": 1
      },
      "video_integrity": {
        "schema_version": "eb/bfm-case-video/v1",
        "sha256": "958002b45700959a1367b9cbf331b3c9c5c3d441e9a11d3b520b770ea920e403",
        "decoded_frames": 2,
        "resolution": [
          320,
          240
        ],
        "fps": 10
      },
      "worker_rank": 0,
      "run_id": "run_e4df98d9c2d64a408f85",
      "plan_hash": "sha256:5544b07474387bdbc7e6cdfe9c4b22fecd2e81072421b69423f7451e2dbdab0a",
      "video_status": "available"
    },
    {
      "seed": 61013,
      "case_id": "rpo-bfm-seed-61013",
      "task_id": "tsk_rpo_bfm_pretrain",
      "completed": true,
      "success": false,
      "actual_steps": 8,
      "episode_len": 8,
      "fall": false,
      "nan": false,
      "hard_limit": true,
      "reason": "physical_hard_limit",
      "env_index": 0,
      "terrain_seed": 64000,
      "initial_state": {
        "path": "traces/seed_61013.initial.json",
        "sha256": "9c6cc532ba02ac780b50852c6296aeb13291485d0e74fb9feec10e3078903652"
      },
      "video_rel": "seed_61013/0.mp4",
      "metric_steps": 8,
      "metrics_available": true,
      "velocity_rmse": 0.3126066342382911,
      "yaw_rmse": 2.5071205037808317,
      "status": "task_failure",
      "termination_reason": "physical_hard_limit",
      "trace": {
        "path": "traces/seed_61013.jsonl.gz",
        "sha256": "c1cdb2bb2bd8f444ab3ec3279d9ff7444aa9a0b9fc4f7534ce0747795c19e9f2",
        "policy_steps": 8,
        "metric_steps": 8
      },
      "video_integrity": {
        "schema_version": "eb/bfm-case-video/v1",
        "sha256": "6280d856360365db55f08681f7fa7df2edafb0c15ad998f5dfb57c77a1b203fc",
        "decoded_frames": 3,
        "resolution": [
          320,
          240
        ],
        "fps": 10
      },
      "worker_rank": 1,
      "run_id": "run_e4df98d9c2d64a408f85",
      "plan_hash": "sha256:5544b07474387bdbc7e6cdfe9c4b22fecd2e81072421b69423f7451e2dbdab0a",
      "video_status": "available"
    },
    {
      "seed": 61014,
      "case_id": "rpo-bfm-seed-61014",
      "task_id": "tsk_rpo_bfm_pretrain",
      "completed": true,
      "success": false,
      "actual_steps": 14,
      "episode_len": 14,
      "fall": false,
      "nan": false,
      "hard_limit": true,
      "reason": "physical_hard_limit",
      "env_index": 0,
      "terrain_seed": 64000,
      "initial_state": {
        "path": "traces/seed_61014.initial.json",
        "sha256": "87abd3c6c554db6d9b9d312ccbaadf736b95a34d104a12f0972c2c37c271c43c"
      },
      "video_rel": "seed_61014/0.mp4",
      "metric_steps": 14,
      "metrics_available": true,
      "velocity_rmse": 0.31674494250697216,
      "yaw_rmse": 6.5233211013245045,
      "status": "task_failure",
      "termination_reason": "physical_hard_limit",
      "trace": {
        "path": "traces/seed_61014.jsonl.gz",
        "sha256": "ee35fa8fbff6d4adfd1a36313856cdd167e58b28a7840a7de07d9cfc4591d50a",
        "policy_steps": 14,
        "metric_steps": 14
      },
      "video_integrity": {
        "schema_version": "eb/bfm-case-video/v1",
        "sha256": "1baf7040482603312f91911615c4919351588fed594ddba9f62ad214a588971f",
        "decoded_frames": 4,
        "resolution": [
          320,
          240
        ],
        "fps": 10
      },
      "worker_rank": 2,
      "run_id": "run_e4df98d9c2d64a408f85",
      "plan_hash": "sha256:5544b07474387bdbc7e6cdfe9c4b22fecd2e81072421b69423f7451e2dbdab0a",
      "video_status": "available"
    },
    {
      "seed": 61015,
      "case_id": "rpo-bfm-seed-61015",
      "task_id": "tsk_rpo_bfm_pretrain",
      "completed": true,
      "success": false,
      "actual_steps": 10,
      "episode_len": 10,
      "fall": false,
      "nan": false,
      "hard_limit": true,
      "reason": "physical_hard_limit",
      "env_index": 0,
      "terrain_seed": 64000,
      "initial_state": {
        "path": "traces/seed_61015.initial.json",
        "sha256": "b0e1a7969fd980d8fc219399972810fbeaa4569d5342b64543ca61cdce1ba6dd"
      },
      "video_rel": "seed_61015/0.mp4",
      "metric_steps": 10,
      "metrics_available": true,
      "velocity_rmse": 1.0153466914908629,
      "yaw_rmse": 2.874854538301144,
      "status": "task_failure",
      "termination_reason": "physical_hard_limit",
      "trace": {
        "path": "traces/seed_61015.jsonl.gz",
        "sha256": "3d7e6651887c676705fa9cac3c8eff10ef4afca938957cddc93e9905147d5ce7",
        "policy_steps": 10,
        "metric_steps": 10
      },
      "video_integrity": {
        "schema_version": "eb/bfm-case-video/v1",
        "sha256": "8ff241a196aaaaadf9853e63afe28c903ee1036103be7351c779571a0566b75d",
        "decoded_frames": 3,
        "resolution": [
          320,
          240
        ],
        "fps": 10
      },
      "worker_rank": 3,
      "run_id": "run_e4df98d9c2d64a408f85",
      "plan_hash": "sha256:5544b07474387bdbc7e6cdfe9c4b22fecd2e81072421b69423f7451e2dbdab0a",
      "video_status": "available"
    }
  ],
  "available": true,
  "total": 16
}
```

**safe4024-eval-statistics.json**

```json
{
  "median_steps": 8.0,
  "min_steps": 1,
  "max_steps": 35,
  "successes": 0,
  "steps": [
    14,
    24,
    4,
    3,
    9,
    8,
    35,
    6,
    19,
    5,
    8,
    3,
    1,
    8,
    14,
    10
  ]
}
```

**safe4024-eval-verified.json**

```json
{
  "model_step": 4024,
  "cases": [
    {
      "seed": 61000,
      "initial_state_hash": true,
      "trace_hash": true,
      "video_hash": true
    },
    {
      "seed": 61001,
      "initial_state_hash": true,
      "trace_hash": true,
      "video_hash": true
    },
    {
      "seed": 61002,
      "initial_state_hash": true,
      "trace_hash": true,
      "video_hash": true
    },
    {
      "seed": 61003,
      "initial_state_hash": true,
      "trace_hash": true,
      "video_hash": true
    },
    {
      "seed": 61004,
      "initial_state_hash": true,
      "trace_hash": true,
      "video_hash": true
    },
    {
      "seed": 61005,
      "initial_state_hash": true,
      "trace_hash": true,
      "video_hash": true
    },
    {
      "seed": 61006,
      "initial_state_hash": true,
      "trace_hash": true,
      "video_hash": true
    },
    {
      "seed": 61007,
      "initial_state_hash": true,
      "trace_hash": true,
      "video_hash": true
    },
    {
      "seed": 61008,
      "initial_state_hash": true,
      "trace_hash": true,
      "video_hash": true
    },
    {
      "seed": 61009,
      "initial_state_hash": true,
      "trace_hash": true,
      "video_hash": true
    },
    {
      "seed": 61010,
      "initial_state_hash": true,
      "trace_hash": true,
      "video_hash": true
    },
    {
      "seed": 61011,
      "initial_state_hash": true,
      "trace_hash": true,
      "video_hash": true
    },
    {
      "seed": 61012,
      "initial_state_hash": true,
      "trace_hash": true,
      "video_hash": true
    },
    {
      "seed": 61013,
      "initial_state_hash": true,
      "trace_hash": true,
      "video_hash": true
    },
    {
      "seed": 61014,
      "initial_state_hash": true,
      "trace_hash": true,
      "video_hash": true
    },
    {
      "seed": 61015,
      "initial_state_hash": true,
      "trace_hash": true,
      "video_hash": true
    }
  ],
  "export": {
    "schema_version": "eb/rpo-bfm-export/v1",
    "status": "verified",
    "readiness": "diagnostic_smoke",
    "model_weights_sha256": "51dab511d72ddd0a157534b31b1286e2ed9905831b539a9ba58562950958d9d2",
    "resolved_config_sha256": "c7e4c495eaf5aee824d1f63e2b38f8de29d8885bdf526c240805674ee79d8766",
    "codec_sha256": "fff6cf3e8f05c88fc7b552f65e5089dc47315c343bd9a205a1f59539cb2811c4",
    "asset_sha256": "b58868a782d8863ac38059eb4eab89d27ee1d70565bf58f5a055f44510c8e842",
    "motion_manifest_sha256": "7e47773bf53588ee200cccc5bbde3999df172a11593d7359cbeca2a00ac4c5d8",
    "onnx": {
      "path": "policy.onnx",
      "sha256": "4dfe25fefd16f59a14f0786e5ba7739a0778829d4149dc5130c15640417b49c0",
      "obs": 375,
      "latent": 256,
      "act": 23,
      "opset": 17,
      "inputs": [
        "observation",
        "z"
      ],
      "batch_sizes": [
        1,
        4
      ],
      "max_abs_error": 1.7583370208740234e-06,
      "max_joint_target_error_rad": 3.077089786529541e-06,
      "normalization": "embedded"
    },
    "prompts": {
      "path": "prompts.npz",
      "sha256": "6204631fbe1aadd4d88a81edea2f12f96da8a62e6c5940282f42948575039864",
      "manifest": "prompt-manifest.json",
      "names": [
        "goal_train_reference",
        "tracking_train_sequence",
        "reward_stand",
        "reward_walk_0p3mps"
      ],
      "manifest_sha256": "8e8b3ce0d625025a154e574c581626844c0c1a0a624ad265c408f6f40d63c23e"
    },
    "command_latent": {
      "path": "command-to-latent.onnx",
      "sha256": "49a02818a8b404ddd49bdc4097f1728674d19fabac6e229ec877cb5acf0bf78f",
      "command_dim": 3,
      "latent_dim": 256,
      "opset": 17,
      "reference_split": "train",
      "reference_count": 4096,
      "reference_limit": 4096,
      "batch_sizes": [
        1,
        6
      ],
      "max_abs_error": 5.125999450683594e-06
    },
    "golden": {
      "path": "golden-traces.npz",
      "sha256": "e1a6bb579cd25501a24dce827498fa0eeac6fb0a6dd052d3505f282e92a80c86",
      "n_inputs": 4,
      "batch_sizes": [
        1,
        4
      ],
      "n_command_inputs": 6,
      "command_batch_sizes": [
        1,
        6
      ]
    },
    "training_profile": "ufo_native_bringup",
    "execution": {
      "schema_version": "eb/training-binding/v1",
      "recipe_id": "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_safe_v2",
      "algorithm_id": "bfm_fb_cpr_aux",
      "provider_id": "ufo",
      "runtime_id": "rlinf",
      "simulator_id": "isaaclab",
      "progress_unit": "learner_joint_update",
      "checkpoint_format": "ufo_agent_state_v1",
      "topology_id": "one_learner_three_samplers"
    },
    "checkpoint_format": "ufo_agent_state_v1",
    "learner_joint_updates": 4024,
    "native_manifest_sha256": "4a137eeed0c512f6f4dc92ddf0a481391f90970dab05182f77f0db608f632a21",
    "primary_unit": "learner_joint_update",
    "io_signature": {
      "actor_fields": [
        {
          "name": "state",
          "width": 52,
          "layout": "q-minus-default23,dq23,gravity_b3,omega_b/4"
        },
        {
          "name": "last_action",
          "width": 23,
          "units": "policy [-1,1]"
        },
        {
          "name": "history_actor",
          "width": 300,
          "layout": "4x75 oldest-first, frame-major"
        }
      ],
      "privileged_width": 358,
      "latent_dim": 256,
      "action_dim": 23,
      "normalization": "frozen native batch norm embedded in ONNX",
      "action_decoder": "RPO v1 default_q + .25*ACTION_GAIN*action"
    }
  }
}
```

**safe4024-eval-media.json**

```json
[
  {
    "seed": 61000,
    "http_status": 206,
    "sha256": "1ee217a9f2514ce137b94cf198c75fea68aacdcc5c70a9bff94a7bd07cbf1f02",
    "decoded_frames": 4,
    "path": "61000/0"
  },
  {
    "seed": 61001,
    "http_status": 206,
    "sha256": "1d32bf08fda78db5a9a478cf0eac578640ea30a3d917da280d6fd1dc37b3a0cf",
    "decoded_frames": 6,
    "path": "61001/0"
  },
  {
    "seed": 61002,
    "http_status": 206,
    "sha256": "98c677112c1f6214d855b29fa281ad17f455ff4271122dd854e61def6a3e663d",
    "decoded_frames": 2,
    "path": "61002/0"
  },
  {
    "seed": 61003,
    "http_status": 206,
    "sha256": "a371734dfc596cbde94c9b740e3139d8cf58fa01291d0fa1c50cc2b3f2dbbd8b",
    "decoded_frames": 2,
    "path": "61003/0"
  },
  {
    "seed": 61004,
    "http_status": 206,
    "sha256": "08570a7db92a25432e5f751859cc5938e3919daa1e5119b853bd8e8ad2c8cebe",
    "decoded_frames": 3,
    "path": "61004/0"
  },
  {
    "seed": 61005,
    "http_status": 206,
    "sha256": "3b9bbed5257d910a02c3d5c709cf83246d90fc532964eec601b9eed8ce8b870a",
    "decoded_frames": 3,
    "path": "61005/0"
  },
  {
    "seed": 61006,
    "http_status": 206,
    "sha256": "4605222678aed6718a5b2da3fbb0db9d9aac486b6116d0c0f81bfa8c1200101f",
    "decoded_frames": 8,
    "path": "61006/0"
  },
  {
    "seed": 61007,
    "http_status": 206,
    "sha256": "07210dd0b8b5c7d8d2bf262edfee05f98fe2f9470ebf123fdcb923f24535d767",
    "decoded_frames": 3,
    "path": "61007/0"
  },
  {
    "seed": 61008,
    "http_status": 206,
    "sha256": "ec47c152732e14ed28c67c971d3e03e76cc72af4f8d7eaabb550ca81020a17b4",
    "decoded_frames": 5,
    "path": "61008/0"
  },
  {
    "seed": 61009,
    "http_status": 206,
    "sha256": "78de03f5044be179ecb0bc4295da1521869a66891cd5a5eff2fe19550e8b169c",
    "decoded_frames": 2,
    "path": "61009/0"
  },
  {
    "seed": 61010,
    "http_status": 206,
    "sha256": "64e3705b353921f28c24b03d8a8f49b13410cd50fa30bbc2d7bdb9185ae729ea",
    "decoded_frames": 3,
    "path": "61010/0"
  },
  {
    "seed": 61011,
    "http_status": 206,
    "sha256": "a74ef6a579c822eb774a5e4eec8c980fcf34d856c4e5eac1b56ecc985c4e6d67",
    "decoded_frames": 2,
    "path": "61011/0"
  },
  {
    "seed": 61012,
    "http_status": 206,
    "sha256": "958002b45700959a1367b9cbf331b3c9c5c3d441e9a11d3b520b770ea920e403",
    "decoded_frames": 2,
    "path": "61012/0"
  },
  {
    "seed": 61013,
    "http_status": 206,
    "sha256": "6280d856360365db55f08681f7fa7df2edafb0c15ad998f5dfb57c77a1b203fc",
    "decoded_frames": 3,
    "path": "61013/0"
  },
  {
    "seed": 61014,
    "http_status": 206,
    "sha256": "1baf7040482603312f91911615c4919351588fed594ddba9f62ad214a588971f",
    "decoded_frames": 4,
    "path": "61014/0"
  },
  {
    "seed": 61015,
    "http_status": 206,
    "sha256": "8ff241a196aaaaadf9853e63afe28c903ee1036103be7351c779571a0566b75d",
    "decoded_frames": 3,
    "path": "61015/0"
  }
]
```

