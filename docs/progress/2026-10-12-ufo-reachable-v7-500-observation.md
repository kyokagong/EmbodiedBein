# UFO 可达控制v7：6144环境、500新增更新观测报告

后续复核补充（2026-10-12）：本段和H接触修复段使用同一100m×100m有限地面，而每采样卡2048、spacing2.5m的实际网格中心范围±56.25/±55m；408/2048中心位于地面±50m范围外，参考初态XY还存在偏移。原段指标和模型保留，但不能视为全部环境都有完整地面支撑的有效实验。后续已派生512m平面与实际grid+RSI足网格包络断言，未修改此历史镜像、profile或模型。此发现不能单独证明旧右踝68.68rad/s由地面边缘引起。


本报告直接列出任务配置、原始证据摘要与全部指标，不要求读者访问其他文件。日期按Asia/Shanghai；更新于2026-10-12T01:30:09.953126+08:00。未进行Git提交。

G配方从新初始化完成16更新资格、恢复至24，再恢复至524，本段新增500次真实UFO agent联合更新；这是保留原生网络和损失主体的平台注册稳定化变体，不是官方配方无修改复现。训练观测门槛通过，可供下一段有限研究决策；本轮不自动追加训练或评估。

五窗episode_len均值（控制步）：8.52139382 → 10.8671797 → 11.9711922 → 13.0027414 → 14.1548762。末窗均值对应0.283097523秒，末记录点14.1907654步。日志是完成回合的聚合均值，最大日志均值不是单案例最大存活时间。平均值仍远低于100步的评估触发门槛，正常行走未验证；本轮在既定500更新观测段结束，不自动扩展长训。

所有标量为有限值；裁剪后范数门槛=True，RSI盒外率门槛=True，末窗episode增长且硬速度率下降=True。这不是独立评估通过或模型收敛结论。

## 任务、改动与固定配置


目的：修复v1控制目标盒与URDF可达区间不匹配，并在完整原生网络上观察优化稳定性。左肘v1目标[0.28,1.28]rad、物理范围[-0.6,1.57]rad，0.88rad残差来自目标盒，不能据此修改关节轴、符号或零点。v2使用有限物理区间中点/半宽映射；原始q/dq、DEFAULT_Q观测编码、70段NPZ字节、来源与训练/验证/测试分组保持一致。G1有效训练7510帧，v1全关节目标盒可达仅4帧；官方6134帧中可达2204帧，不能用过滤少量可达帧替代控制语义修复。

recipe=rpo_bfm_ufo_agent_rlinf_isaac_4gpu_reachable_v7；v2 codec hash=dce25d554ceb7fa69287515946d2f5cd9d41d14f1b1ef38715b819d7e962b6ef；数据ds_fb0ecda9e70644f2afdd由ds_92718168bfec4da6a01d派生，manifest SHA256=dc0c6585887507df59f675963b2733d25c9836d805e9bb89f2402398f07bd288。镜像SHA256=2359c60a3f40ef9f6a47b59d9c02194d1aa14afa603789960c5807e988277fb7。控制版本、数据版本、源码版本分别冻结；跨recipe完整恢复不支持，G未恢复F权重。

完整435116484参数网络：F/Actor/Critic/辅助Critic为2048×6，D为1024×3，B为256×1，latent256。GPU0单learner，GPU1–3各2048 sampler环境，共6144，非DDP；batch2048，CPU replay100万；每轮4控制步=24576在线转移，每触发16联合更新，sample_interval24576，warmup10240。累计轮数和在线量以下列checkpoint实际计数为准。

六优化器step前裁剪：Actor L2≤1，其余五项≤5；同时保留原始/裁剪后范数。native clip_grad_norm=0表示由平台注册hook执行裁剪，不是取消裁剪。Actor detach(mean(abs(Q_fb)))上界50；D回报截到[-2,2]并记录raw，D学习率3e-6。速度raw=sum(relu(abs(dq)/Vmax−0.70)^2)，权重−25只应用一次，再经过RewardNormalizer(scale=True)。TD真实terminated令bootstrap为0，truncated保留；tracking=False仍允许每环境100步正常z更新。没有加入原生站立奖励，也没有未经物理证据的midpoint Actor bias初始化。

资格每段1小时，正式500更新段2小时，零自动重试，新证据/模型最多64GiB，不自动扩展长训。冻结训练门槛：非有限值0；裁剪后范数不超过上限；RSI盒外率<10%；末100更新episode均值超过首100且硬速度越限率下降。独立行走评估另需episode≥100且持续稳定。用户要求训练不足时不安排独立评估；本段未满足行走触发门槛，G evaluator也尚不具备资格，不能填写0/100、成功率或验收通过。

env/reward、env/return与env/success_once在本recipe为占位零值；真实优化使用FB、判别奖励和具名辅助奖励，不能从这些零值推导没有更新或验收失败。裁剪是梯度上界，不直接限制Adam参数步长；浮点范数按1e-5容差核验。

完整TrainJob、Job、Run（其中路径只作为实际配置值）：

```json
{
  "id": "trn_e82cee642c564d74bb49",
  "run_id": "run_aeb47af29c104e63ba42",
  "kind": "ppo",
  "name": "RPO UFO reachable v7 stabilized 500-update observation",
  "model_name": "RPO-UFO-reachable-v7-step524",
  "status": "succeeded",
  "template_id": "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_reachable_v7",
  "image_digest": "sha256:2359c60a3f40ef9f6a47b59d9c02194d1aa14afa603789960c5807e988277fb7",
  "embodiment_id": "emb_roboparty_rpo",
  "dataset_ids": [
    "ds_fb0ecda9e70644f2afdd"
  ],
  "base_model_id": null,
  "gpus_allocated": [
    0,
    1,
    2,
    3
  ],
  "ray_port": 6418,
  "params": {
    "stack": "rlinf",
    "purpose": "training",
    "execution": {
      "recipe_id": "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_reachable_v7",
      "runtime_id": "rlinf",
      "provider_id": "ufo",
      "topology_id": "one_learner_three_samplers",
      "algorithm_id": "bfm_fb_cpr_aux",
      "simulator_id": "isaaclab",
      "progress_unit": "learner_joint_update",
      "schema_version": "eb/training-binding/v1",
      "checkpoint_format": "ufo_agent_state_v1"
    },
    "model_name": "RPO-UFO-reachable-v7-step524",
    "config_name": "rpo_bfm_ufo_isaac",
    "resume_mode": "trainer_state",
    "resume_step": 24,
    "template_id": "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_reachable_v7",
    "source_receipt": {
      "ufo": {
        "tree": "3a6b8b0d6a04287f37c1d7834e51561a8e014fc9",
        "scope": "pinned native agent plus measured v7 regularization/reward bounds",
        "commit": "0b89378de02f77712072d983bc43ec2a46455dc0",
        "license": "CC-BY-NC-4.0",
        "content_hash": "sha256:2ec3a9015dda0fd4cfe6861e6ebb35fbf55c78bf43c22544a82ab2368a222e84",
        "upstream_content_hash": "sha256:2191cf26c77262f7ef2e3525ca23878a8f08c46fd02a20b7776a791f8f5e7582"
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
      "control_profile": "reachable_v2",
      "scaling_profile": "env6144_batch2048_v6",
      "auxiliary_profile": "soft_velocity_v7",
      "optimizer_profile": "stabilized_v7",
      "topology_patch_hash": "sha256:3078506628924464cb42d0b2f108b3fab502a43fd75e091f358d504565294d23",
      "auxiliary_source_sha256": "sha256:4612307d49eacdfab44e7bea5981daa1c4634b0ead9e6b51626f6603f88e6938",
      "stabilization_patch_hash": "sha256:564f5452e12e3179112e6aa3020efba4cbcedc0ad90dd26b98aed78a50dd85b1",
      "rollout_profile_patch_hash": "sha256:777456ce171734326cf8c55eaa434fc5a69a27d0b2d0b725608dd54c190f4a77",
      "scaling_profile_patch_hash": "sha256:58a2dca8377ed2ede804c7101b39bc6992ec669ef95e610ddec4b2a59bfc6b20",
      "learning_profile_patch_hash": "sha256:d47623e48b4cc757d5e63336a1fa7d49234c3833a2f1c6bbdbee4fe7ffd9a272",
      "action_control_source_sha256": "sha256:1b0bbc0a7a350581ef1939faf93a3e27d824bde62107ecfe4e9ff988bbc088fb",
      "auxiliary_profile_patch_hash": "sha256:30e7b4ea7a3ec0e45cdbe0a9dbf718eb219c2c6fd36d07f9b18bd5222183a03a",
      "action_control_profile_patch_hash": "sha256:548260ca5e3eedd53ae739c00672c7e273e7835faed734778039cf5977b56e0d"
    },
    "environment_profile": {
      "id": "rpo_bfm_ufo_isaac_4gpu_rsi_reachable_v7",
      "hash": "sha256:7fa42e02c127deee753c646fa86bf5d4929ec8cb551f37ca064fda7e8673dc57",
      "definition": {
        "task": "RPO v2有限URDF动作目标，原始q/dq与观察零点保持；6144环境，每个sampler2048；435M原生网络、batch2048、replay100万，每24576新转移触发16次联合更新。",
        "reward": "九项raw aux，limits_dof_vel=sum relu(abs(dq)/Vmax-0.70)^2，权重-25仅加权一次；判别奖励截[-2,2]，D lr=3e-6；Actor正则尺度上限50。",
        "failure": "真实硬位置/速度/力矩终止保持；target slew每步0.5×Vmax；Actor裁剪1，其余五Adam裁剪5，非有限梯度失败。",
        "success": "训练/完整恢复资格16→24后新增500联合更新，最多2小时且不自动重试；训练指标不足不启动独立评估。正常行走尚未验证。",
        "settings": {}
      }
    },
    "resume_source_image": "eb/rlinf:rpo-bfm-ufo-agent-4gpu-reachable-v7",
    "source_train_job_id": "trn_f70ff2974dc04cf0a7f6",
    "target_learner_updates": 524
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
    "resume_from": "run_27bcf1e60b4e424fa6b0"
  },
  "resumed_from": "trn_f70ff2974dc04cf0a7f6",
  "resume_step": 24,
  "resume_mode": "trainer_state",
  "hydra_overrides": [
    "env.train.total_num_envs=6144",
    "env.eval.total_num_envs=3",
    "+actor.model.topology_id=one_learner_three_samplers",
    "+actor.model.learning_profile=physical_limits_stable_slew_v4",
    "+actor.model.architecture_profile=native_fb_v1",
    "+actor.model.rollout_expert_trajectories=False",
    "actor.model.hidden_dim=2048",
    "actor.model.hidden_layers=6",
    "actor.model.batch_size=2048",
    "actor.micro_batch_size=2048",
    "actor.global_batch_size=2048",
    "actor.model.replay_capacity=1000000",
    "env.train.init_params.id=RPO-UFO-Isaac-RSI-reachable-v7",
    "+actor.model.auxiliary_profile=soft_velocity_v7",
    "+actor.model.scaling_profile=env6144_batch2048_v6",
    "+actor.model.control_profile=reachable_v2",
    "+actor.model.optimizer_profile=stabilized_v7",
    "actor.seed=42",
    "~cluster.component_placement",
    "+cluster.component_placement={actor:0,env\\,\\ rollout:1-3}",
    "runner.logger.log_path=/workspace/output",
    "hydra.run.dir=/workspace/output/hydra",
    "runner.logger.experiment_name=\"RPO UFO reachable v7 stabilized 500-update observation\"",
    "runner.resume_dir=/workspace/resume/global_step_24"
  ],
  "run_dir": "/data/kk/embodiedbein/runs/run_aeb47af29c104e63ba42",
  "auto_register_latest": false,
  "metrics_summary": {
    "checkpoint": {
      "path": "/data/kk/models/runs/run_aeb47af29c104e63ba42",
      "step": 524,
      "bytes": 12922255031
    },
    "sampler_sync": {
      "identity": {
        "version": 520,
        "state_sha256": "36c3d7c4cdd19f585f445432c999edd0f641a9aa187aa85ec47ecd9a777a95e6",
        "online_transitions": 835584
      },
      "samplers": [
        {
          "rank": 0,
          "version": 520,
          "state_sha256": "36c3d7c4cdd19f585f445432c999edd0f641a9aa187aa85ec47ecd9a777a95e6",
          "online_transitions": 835584
        },
        {
          "rank": 1,
          "version": 520,
          "state_sha256": "36c3d7c4cdd19f585f445432c999edd0f641a9aa187aa85ec47ecd9a777a95e6",
          "online_transitions": 835584
        },
        {
          "rank": 2,
          "version": 520,
          "state_sha256": "36c3d7c4cdd19f585f445432c999edd0f641a9aa187aa85ec47ecd9a777a95e6",
          "online_transitions": 835584
        }
      ],
      "schema_version": "eb/ufo-sampler-sync/v1",
      "sampling_rounds": 34
    },
    "model_version_id": "mv_76b9f82e409249fabe85"
  },
  "error_message": null,
  "job_id": "job_94648a5de5d14a95ae4a",
  "started_at": "2026-10-11T16:49:09.052251+00:00",
  "finished_at": "2026-10-11T17:22:31.417376+00:00",
  "created_at": "2026-10-11T16:49:03.155768+00:00"
}
```


```json
{
  "id": "job_94648a5de5d14a95ae4a",
  "kind": "train.rl",
  "status": "succeeded",
  "embodiment_env_id": "ee_rpo_v2_bfm_23dof_v2",
  "priority": 50,
  "created_by": "user:kk",
  "spec": {
    "kind": "train.rl",
    "name": "RPO UFO reachable v7 stabilized 500-update observation",
    "gates": [],
    "image": "sha256:2359c60a3f40ef9f6a47b59d9c02194d1aa14afa603789960c5807e988277fb7",
    "fanout": null,
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
      "resume_from": "run_27bcf1e60b4e424fa6b0"
    },
    "labels": {},
    "params": {
      "stack": "rlinf",
      "purpose": "training",
      "execution": {
        "recipe_id": "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_reachable_v7",
        "runtime_id": "rlinf",
        "provider_id": "ufo",
        "topology_id": "one_learner_three_samplers",
        "algorithm_id": "bfm_fb_cpr_aux",
        "simulator_id": "isaaclab",
        "progress_unit": "learner_joint_update",
        "schema_version": "eb/training-binding/v1",
        "checkpoint_format": "ufo_agent_state_v1"
      },
      "model_name": "RPO-UFO-reachable-v7-step524",
      "config_name": "rpo_bfm_ufo_isaac",
      "resume_mode": "trainer_state",
      "resume_step": 24,
      "template_id": "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_reachable_v7",
      "source_receipt": {
        "ufo": {
          "tree": "3a6b8b0d6a04287f37c1d7834e51561a8e014fc9",
          "scope": "pinned native agent plus measured v7 regularization/reward bounds",
          "commit": "0b89378de02f77712072d983bc43ec2a46455dc0",
          "license": "CC-BY-NC-4.0",
          "content_hash": "sha256:2ec3a9015dda0fd4cfe6861e6ebb35fbf55c78bf43c22544a82ab2368a222e84",
          "upstream_content_hash": "sha256:2191cf26c77262f7ef2e3525ca23878a8f08c46fd02a20b7776a791f8f5e7582"
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
        "control_profile": "reachable_v2",
        "scaling_profile": "env6144_batch2048_v6",
        "auxiliary_profile": "soft_velocity_v7",
        "optimizer_profile": "stabilized_v7",
        "topology_patch_hash": "sha256:3078506628924464cb42d0b2f108b3fab502a43fd75e091f358d504565294d23",
        "auxiliary_source_sha256": "sha256:4612307d49eacdfab44e7bea5981daa1c4634b0ead9e6b51626f6603f88e6938",
        "stabilization_patch_hash": "sha256:564f5452e12e3179112e6aa3020efba4cbcedc0ad90dd26b98aed78a50dd85b1",
        "rollout_profile_patch_hash": "sha256:777456ce171734326cf8c55eaa434fc5a69a27d0b2d0b725608dd54c190f4a77",
        "scaling_profile_patch_hash": "sha256:58a2dca8377ed2ede804c7101b39bc6992ec669ef95e610ddec4b2a59bfc6b20",
        "learning_profile_patch_hash": "sha256:d47623e48b4cc757d5e63336a1fa7d49234c3833a2f1c6bbdbee4fe7ffd9a272",
        "action_control_source_sha256": "sha256:1b0bbc0a7a350581ef1939faf93a3e27d824bde62107ecfe4e9ff988bbc088fb",
        "auxiliary_profile_patch_hash": "sha256:30e7b4ea7a3ec0e45cdbe0a9dbf718eb219c2c6fd36d07f9b18bd5222183a03a",
        "action_control_profile_patch_hash": "sha256:548260ca5e3eedd53ae739c00672c7e273e7835faed734778039cf5977b56e0d"
      },
      "environment_profile": {
        "id": "rpo_bfm_ufo_isaac_4gpu_rsi_reachable_v7",
        "hash": "sha256:7fa42e02c127deee753c646fa86bf5d4929ec8cb551f37ca064fda7e8673dc57",
        "definition": {
          "task": "RPO v2有限URDF动作目标，原始q/dq与观察零点保持；6144环境，每个sampler2048；435M原生网络、batch2048、replay100万，每24576新转移触发16次联合更新。",
          "reward": "九项raw aux，limits_dof_vel=sum relu(abs(dq)/Vmax-0.70)^2，权重-25仅加权一次；判别奖励截[-2,2]，D lr=3e-6；Actor正则尺度上限50。",
          "failure": "真实硬位置/速度/力矩终止保持；target slew每步0.5×Vmax；Actor裁剪1，其余五Adam裁剪5，非有限梯度失败。",
          "success": "训练/完整恢复资格16→24后新增500联合更新，最多2小时且不自动重试；训练指标不足不启动独立评估。正常行走尚未验证。",
          "settings": {}
        }
      },
      "resume_source_image": "eb/rlinf:rpo-bfm-ufo-agent-4gpu-reachable-v7",
      "source_train_job_id": "trn_f70ff2974dc04cf0a7f6",
      "target_learner_updates": 524
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
    "outputs": {},
    "runtime": {},
    "task_id": "tsk_rpo_bfm_pretrain",
    "code_refs": {
      "stack": {
        "kind": "content_snapshot",
        "path": "/data/kk/embodiedbein/snapshots/dbaa8bed56be412f6d012ee34030822941945786cede6bf9bc3de0fec5445d48/code",
        "branch": null,
        "commit": "sha256:dbaa8bed56be412f6d012ee34030822941945786cede6bf9bc3de0fec5445d48"
      },
      "embodiment": {
        "kind": "content_snapshot",
        "path": "/data/kk/embodiedbein/snapshots/04e20ece2e3e3f3603a98b24e04aba42023fb31a7395beddfdac2a5d522de54d/code",
        "branch": null,
        "commit": "sha256:04e20ece2e3e3f3603a98b24e04aba42023fb31a7395beddfdac2a5d522de54d"
      }
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
    "experiment": null,
    "api_version": "eb/v1",
    "embodiment_env_id": "ee_rpo_v2_bfm_23dof_v2"
  },
  "error": null,
  "runs": [
    {
      "id": "run_aeb47af29c104e63ba42",
      "attempt": 1,
      "status": "finished",
      "abort_reason": null,
      "metrics_summary": {
        "checkpoint": {
          "path": "/data/kk/models/runs/run_aeb47af29c104e63ba42",
          "step": 524,
          "bytes": 12922255031
        },
        "sampler_sync": {
          "identity": {
            "version": 520,
            "state_sha256": "36c3d7c4cdd19f585f445432c999edd0f641a9aa187aa85ec47ecd9a777a95e6",
            "online_transitions": 835584
          },
          "samplers": [
            {
              "rank": 0,
              "version": 520,
              "state_sha256": "36c3d7c4cdd19f585f445432c999edd0f641a9aa187aa85ec47ecd9a777a95e6",
              "online_transitions": 835584
            },
            {
              "rank": 1,
              "version": 520,
              "state_sha256": "36c3d7c4cdd19f585f445432c999edd0f641a9aa187aa85ec47ecd9a777a95e6",
              "online_transitions": 835584
            },
            {
              "rank": 2,
              "version": 520,
              "state_sha256": "36c3d7c4cdd19f585f445432c999edd0f641a9aa187aa85ec47ecd9a777a95e6",
              "online_transitions": 835584
            }
          ],
          "schema_version": "eb/ufo-sampler-sync/v1",
          "sampling_rounds": 34
        },
        "model_version_id": "mv_76b9f82e409249fabe85"
      }
    }
  ]
}
```


```json
{
  "id": "run_aeb47af29c104e63ba42",
  "job_id": "job_94648a5de5d14a95ae4a",
  "attempt": 1,
  "status": "finished",
  "abort_reason": null,
  "metrics_summary": {
    "checkpoint": {
      "path": "/data/kk/models/runs/run_aeb47af29c104e63ba42",
      "step": 524,
      "bytes": 12922255031
    },
    "sampler_sync": {
      "identity": {
        "version": 520,
        "state_sha256": "36c3d7c4cdd19f585f445432c999edd0f641a9aa187aa85ec47ecd9a777a95e6",
        "online_transitions": 835584
      },
      "samplers": [
        {
          "rank": 0,
          "version": 520,
          "state_sha256": "36c3d7c4cdd19f585f445432c999edd0f641a9aa187aa85ec47ecd9a777a95e6",
          "online_transitions": 835584
        },
        {
          "rank": 1,
          "version": 520,
          "state_sha256": "36c3d7c4cdd19f585f445432c999edd0f641a9aa187aa85ec47ecd9a777a95e6",
          "online_transitions": 835584
        },
        {
          "rank": 2,
          "version": 520,
          "state_sha256": "36c3d7c4cdd19f585f445432c999edd0f641a9aa187aa85ec47ecd9a777a95e6",
          "online_transitions": 835584
        }
      ],
      "schema_version": "eb/ufo-sampler-sync/v1",
      "sampling_rounds": 34
    },
    "model_version_id": "mv_76b9f82e409249fabe85"
  },
  "gpu_ids": [
    0,
    1,
    2,
    3
  ],
  "provenance": "{\"code_ref\": \"sha256:04e20ece2e3e3f3603a98b24e04aba42023fb31a7395beddfdac2a5d522de54d\", \"config_hash\": \"05171fd6ee20428567034dbd40150e0959a74409130055f1dba70aa8df5b5dfb\", \"data_version\": null, \"image_digest\": \"sha256:2359c60a3f40ef9f6a47b59d9c02194d1aa14afa603789960c5807e988277fb7\", \"seed\": null}",
  "artifacts": [
    {
      "id": "art_80e7d8bdf1064d0ba6ac",
      "kind": "rlinf_checkpoint",
      "sha256": "sha256:7a33a0fd0cdc5427cae82818ff655a32368d689e1cc18dfc9a012bf187967ca1",
      "size_bytes": 12922255031,
      "storage": "/data/kk/models/runs/run_aeb47af29c104e63ba42"
    }
  ]
}
```


最终模型版本及产物登记：
```json
{
  "id": "mv_76b9f82e409249fabe85",
  "family_id": "mf_rpo_v2_bfm_zero",
  "name": "RPO-UFO-reachable-v7-step524",
  "embodiment_env_id": "ee_rpo_v2_bfm_23dof_v2",
  "io_signature": {
    "act": 23,
    "obs": 375,
    "latent": 256,
    "critic_obs": 733,
    "action_profile": "rpo_v2_joint23_position_v2",
    "observation_profile": "rpo_bfm_actor_history4_z256_v1"
  },
  "mem_profile": {},
  "card": {
    "step": 524,
    "run_id": "run_aeb47af29c104e63ba42",
    "network": {
      "components": [
        "actor",
        "forward_map",
        "backward_map",
        "discriminator",
        "critic",
        "aux_critic"
      ],
      "latent_dim": 256,
      "history_length": 4
    },
    "training": {
      "kind": "train.rl",
      "image": "sha256:2359c60a3f40ef9f6a47b59d9c02194d1aa14afa603789960c5807e988277fb7",
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
      }
    },
    "execution": {
      "recipe_id": "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_reachable_v7",
      "runtime_id": "rlinf",
      "provider_id": "ufo",
      "topology_id": "one_learner_three_samplers",
      "algorithm_id": "bfm_fb_cpr_aux",
      "simulator_id": "isaaclab",
      "progress_unit": "learner_joint_update",
      "schema_version": "eb/training-binding/v1",
      "checkpoint_format": "ufo_agent_state_v1"
    },
    "readiness": "simulation_unverified",
    "template_id": "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_reachable_v7",
    "codec_sha256": "dce25d554ceb7fa69287515946d2f5cd9d41d14f1b1ef38715b819d7e962b6ef",
    "motion_prior": {
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
    },
    "train_job_id": "job_94648a5de5d14a95ae4a",
    "normalization": "frozen_running_stats",
    "progress_unit": "learner_joint_update",
    "action_control": {
      "reset": "clamp measured q to physical finite bounds; normalize anchor; clear history",
      "guarantee": "target increments only; actual dq and infeasible reset residual remain measured",
      "codec_sha256": "dce25d554ceb7fa69287515946d2f5cd9d41d14f1b1ef38715b819d7e962b6ef",
      "control_dt_s": 0.02,
      "interface_id": "ee_rpo_v2_bfm_23dof_v2",
      "replay_action": "requested",
      "schema_version": "eb/rpo-target-slew/v2",
      "control_profile": "reachable_v2",
      "action_midpoint_rad": [
        -0.4,
        0.4,
        0.0,
        0.4,
        -0.4,
        -0.785,
        -0.785,
        -0.6542999999999999,
        -0.6542999999999999,
        1.445,
        -1.445,
        1.15,
        1.15,
        0.0,
        0.0,
        0.0,
        0.0,
        0.48500000000000004,
        0.48500000000000004,
        0.0,
        0.0,
        0.0,
        0.0
      ],
      "action_halfwidth_rad": [
        0.6,
        0.6,
        3.14,
        0.6,
        0.6,
        2.355,
        2.355,
        1.4397,
        1.4397,
        1.695,
        1.695,
        1.35,
        1.35,
        1.57,
        1.57,
        0.6,
        0.6,
        1.085,
        1.085,
        0.5,
        0.5,
        1.57,
        1.57
      ],
      "joint_position_max_rad": [
        0.2,
        1.0,
        3.14,
        1.0,
        0.2,
        1.57,
        1.57,
        0.7854,
        0.7854,
        3.14,
        0.25,
        2.5,
        2.5,
        1.57,
        1.57,
        0.6,
        0.6,
        1.57,
        1.57,
        0.5,
        0.5,
        1.57,
        1.57
      ],
      "joint_position_min_rad": [
        -1.0,
        -0.2,
        -3.14,
        -0.2,
        -1.0,
        -3.14,
        -3.14,
        -2.094,
        -2.094,
        -0.25,
        -3.14,
        -0.2,
        -0.2,
        -1.57,
        -1.57,
        -0.6,
        -0.6,
        -0.6,
        -0.6,
        -0.5,
        -0.5,
        -1.57,
        -1.57
      ],
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
      "target_velocity_limit_fraction": 0.5,
      "observation_default_joint_pos_rad": [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.18,
        0.18,
        -0.1,
        -0.1,
        0.06,
        -0.06,
        0.3,
        0.3,
        0.0,
        0.0,
        -0.2,
        -0.2,
        0.78,
        0.78,
        0.0,
        0.0,
        0.0,
        0.0
      ]
    },
    "schema_version": "eb/rpo-bfm-model/v1",
    "artifact_sha256": "sha256:7a33a0fd0cdc5427cae82818ff655a32368d689e1cc18dfc9a012bf187967ca1",
    "checkpoint_path": "/data/kk/models/runs/run_aeb47af29c104e63ba42",
    "learner_progress": {
      "next_trigger": 870400,
      "sampling_rounds": 35,
      "online_transitions": 860160,
      "learner_joint_updates": 524
    },
    "auxiliary_profile": "soft_velocity_v7",
    "checkpoint_format": "ufo_agent_state_v1",
    "hardware_verified": false,
    "environment_profile": {
      "id": "rpo_bfm_ufo_isaac_4gpu_rsi_reachable_v7",
      "hash": "sha256:7fa42e02c127deee753c646fa86bf5d4929ec8cb551f37ca064fda7e8673dc57",
      "definition": {
        "task": "RPO v2有限URDF动作目标，原始q/dq与观察零点保持；6144环境，每个sampler2048；435M原生网络、batch2048、replay100万，每24576新转移触发16次联合更新。",
        "reward": "九项raw aux，limits_dof_vel=sum relu(abs(dq)/Vmax-0.70)^2，权重-25仅加权一次；判别奖励截[-2,2]，D lr=3e-6；Actor正则尺度上限50。",
        "failure": "真实硬位置/速度/力矩终止保持；target slew每步0.5×Vmax；Actor裁剪1，其余五Adam裁剪5，非有限梯度失败。",
        "success": "训练/完整恢复资格16→24后新增500联合更新，最多2小时且不自动重试；训练指标不足不启动独立评估。正常行走尚未验证。",
        "settings": {}
      }
    }
  },
  "tags": [
    "roboparty",
    "rpo-v2",
    "bfm-zero",
    "rlinf",
    "checkpoint"
  ],
  "provenance": "full",
  "created_at": "2026-10-11T17:22:31.299084Z",
  "train_job_name": "RPO UFO reachable v7 stabilized 500-update observation",
  "train_page_id": "trn_e82cee642c564d74bb49",
  "artifact": {
    "id": "art_80e7d8bdf1064d0ba6ac",
    "kind": "rlinf_checkpoint",
    "sha256": "sha256:7a33a0fd0cdc5427cae82818ff655a32368d689e1cc18dfc9a012bf187967ca1",
    "size_bytes": 12922255031,
    "storage": "/data/kk/models/runs/run_aeb47af29c104e63ba42",
    "meta": {
      "step": 524,
      "manifest": {
        "step": 524,
        "files": [
          {
            "path": "model/actor-normalization.json",
            "bytes": 158,
            "sha256": "e74511417e96ffb17c38b219617502245b37679bf7ab2342465aaa8df6f4b3a6"
          },
          {
            "path": "model/native/global_step_524/config.json",
            "bytes": 6250,
            "sha256": "f7f67e6c591a432bf4aa465039ed9f2bc8b97d7571d36aa94740d08892ef86a0"
          },
          {
            "path": "model/native/global_step_524/init_kwargs.json",
            "bytes": 45883,
            "sha256": "20dba3350b336a6a89608645695618bd883df4c02aadbecd437e2971423616ed"
          },
          {
            "path": "model/native/global_step_524/model/config.json",
            "bytes": 3807,
            "sha256": "52f94d2946ed8912fc12ac9c25b4bf0e68ccdc669a05ea104e8b6c178e91fb46"
          },
          {
            "path": "model/native/global_step_524/model/init_kwargs.json",
            "bytes": 45883,
            "sha256": "20dba3350b336a6a89608645695618bd883df4c02aadbecd437e2971423616ed"
          },
          {
            "path": "model/native/global_step_524/model/model.safetensors",
            "bytes": 3343756272,
            "sha256": "969f60e2448b9d61e2e2c18abb7bdd26e36ebdc27f9e400c9033c10d636340ce"
          },
          {
            "path": "model/native/global_step_524/optimizers.pth",
            "bytes": 3481181041,
            "sha256": "6758d911cd38692aeea0e257c2e2e87cebd99d4ddd42d0fc4e3752e7983b6a2d"
          },
          {
            "path": "model/native/global_step_524/resume-manifest.json",
            "bytes": 25058,
            "sha256": "2adba63649c9059a1a8aa1fbddf4b18b4f6759de391e92085060df0130933d1d"
          },
          {
            "path": "model/native/global_step_524/trainer-state.pt",
            "bytes": 6097184508,
            "sha256": "f66a2e53b7606bb0987a5ac727d80f93096cf1de4aa06fb72bbf0c886865368c"
          },
          {
            "path": "model/native-checkpoint.json",
            "bytes": 88,
            "sha256": "24414cc2fa26deb57f6445d810747cdb6906f32b41b6b6102e7eb39cbc5debcc"
          },
          {
            "path": "model/resolved-config.yaml",
            "bytes": 3810,
            "sha256": "fae47387124f8a5459bff1a2b3345465c775d88d9691e3f7908ce64414fd4542"
          },
          {
            "path": "model/source-receipt.json",
            "bytes": 2273,
            "sha256": "8cface6a5f9b0c99f7e2b20e79fc67d7659526da87ed6e076b4f400a6f5b6bab"
          }
        ],
        "format": "ufo_agent_state_v1",
        "run_id": "run_aeb47af29c104e63ba42",
        "status": "complete",
        "execution": {
          "recipe_id": "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_reachable_v7",
          "runtime_id": "rlinf",
          "provider_id": "ufo",
          "topology_id": "one_learner_three_samplers",
          "algorithm_id": "bfm_fb_cpr_aux",
          "simulator_id": "isaaclab",
          "progress_unit": "learner_joint_update",
          "schema_version": "eb/training-binding/v1",
          "checkpoint_format": "ufo_agent_state_v1"
        },
        "model_path": "model",
        "template_id": "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_reachable_v7",
        "bfm_progress": {
          "next_trigger": 870400,
          "sampling_rounds": 35,
          "online_transitions": 860160,
          "learner_joint_updates": 524
        },
        "primary_unit": "learner_joint_update",
        "weights_path": "model/native/global_step_524/model/model.safetensors",
        "schema_version": "eb/checkpoint/v1",
        "source_checkpoint": "model/.checkpoint-source/global_step_524",
        "native_checkpoint_path": "model/native/global_step_524"
      }
    }
  },
  "eval_reports": [],
  "env_check": {
    "passed": true,
    "violations": []
  },
  "env": {
    "obs_dim": 375,
    "action_dim": 23,
    "policy_hz": 50
  }
}
```


TrainJob.kind=ppo是现有通用元数据字段，实际执行binding=bfm_fb_cpr_aux、provider=ufo；本轮没有PPO优化。解析配置中的env.eval.total_num_envs也不表示启动过评估。

## 恢复、源码与数据证据


16更新完整checkpoint核验：
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
  "fingerprint": "sha256:ae65e46d37525ee5e70ae3f8b974322e0f4ae036bbdbcc55607fd3e6b7ad080d",
  "execution": {
    "schema_version": "eb/training-binding/v1",
    "recipe_id": "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_reachable_v7",
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


24更新完整checkpoint核验：
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
  "fingerprint": "sha256:ae65e46d37525ee5e70ae3f8b974322e0f4ae036bbdbcc55607fd3e6b7ad080d",
  "execution": {
    "schema_version": "eb/training-binding/v1",
    "recipe_id": "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_reachable_v7",
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


524更新完整checkpoint核验：
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
  "fingerprint": "sha256:ae65e46d37525ee5e70ae3f8b974322e0f4ae036bbdbcc55607fd3e6b7ad080d",
  "execution": {
    "schema_version": "eb/training-binding/v1",
    "recipe_id": "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_reachable_v7",
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


模型、六Adam、RNG、replay、expert sampler、z pool与rollout恢复以回执为准；物理内部state不可用，环境通过RSI重建，不能称逐位一致的物理轨迹恢复。梯度裁剪也不等于Adam参数位移的硬上界。

本段新增在线转移=786432，除以6144环境为每环境128控制步，即2.56秒物理探索。并发环境数不等于单环境探索跨度；累计计数包括资格段，物理episode在各次恢复边界重新初始化。当前replay尚未填满，不足以验证原F段缓冲快速淘汰之后的长程稳定性。

测量镜像源码回执：
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
    "tensordict": "0.11.0"
  },
  "scope": "agent-only; Python3.10 package metadata bypassed by tested source import",
  "topology_patch_hash": "sha256:3078506628924464cb42d0b2f108b3fab502a43fd75e091f358d504565294d23",
  "learning_profile_patch_hash": "sha256:d47623e48b4cc757d5e63336a1fa7d49234c3833a2f1c6bbdbee4fe7ffd9a272",
  "rollout_profile_patch_hash": "sha256:777456ce171734326cf8c55eaa434fc5a69a27d0b2d0b725608dd54c190f4a77",
  "action_control_source_sha256": "sha256:1b0bbc0a7a350581ef1939faf93a3e27d824bde62107ecfe4e9ff988bbc088fb",
  "action_control_profile_patch_hash": "sha256:548260ca5e3eedd53ae739c00672c7e273e7835faed734778039cf5977b56e0d",
  "auxiliary_profile": "soft_velocity_v7",
  "auxiliary_source_sha256": "sha256:4612307d49eacdfab44e7bea5981daa1c4634b0ead9e6b51626f6603f88e6938",
  "auxiliary_profile_patch_hash": "sha256:30e7b4ea7a3ec0e45cdbe0a9dbf718eb219c2c6fd36d07f9b18bd5222183a03a",
  "scaling_profile": "env6144_batch2048_v6",
  "scaling_profile_patch_hash": "sha256:58a2dca8377ed2ede804c7101b39bc6992ec669ef95e610ddec4b2a59bfc6b20",
  "control_profile": "reachable_v2",
  "optimizer_profile": "stabilized_v7",
  "stabilization_patch_hash": "sha256:564f5452e12e3179112e6aa3020efba4cbcedc0ad90dd26b98aed78a50dd85b1"
}
```


数据隔离与字节一致审计：
```json
{
  "clips": 70,
  "raw_arrays_identical": true,
  "source_refs_identical": true,
  "splits_identical": true,
  "manifest_sha256": "dc0c6585887507df59f675963b2733d25c9836d805e9bb89f2402398f07bd288"
}
```


v1可达性定位：
```json
{
  "manifest_sha256": "7e47773bf53588ee200cccc5bbde3999df172a11593d7359cbeca2a00ac4c5d8",
  "joint_data": {
    "left_elbow_pitch_joint": {
      "default": 0.78,
      "target_box": [
        0.28,
        1.28
      ],
      "urdf": {
        "axis": "0 1 0",
        "lower": -0.6,
        "upper": 1.57
      }
    },
    "right_elbow_pitch_joint": {
      "default": 0.78,
      "target_box": [
        0.030000000000000027,
        1.53
      ],
      "urdf": {
        "axis": "0 1 0",
        "lower": -0.6,
        "upper": 1.57
      }
    },
    "right_arm_yaw_joint": {
      "default": 0.0,
      "target_box": [
        -0.5,
        0.5
      ],
      "urdf": {
        "axis": "0 0 -1",
        "lower": -1.57,
        "upper": 1.57
      }
    }
  },
  "clips": [
    {
      "id": "g1_dance1_subject1_clip3",
      "source": "g1",
      "valid": 368,
      "target_feasible": 0,
      "first_frame": {
        "left_elbow_pitch_joint": -0.10589626431465149,
        "right_elbow_pitch_joint": -0.07242888957262039,
        "right_arm_yaw_joint": -0.024287600070238113
      },
      "joint_q_range": {
        "left_elbow_pitch_joint": [
          -0.6000000238418579,
          0.5406454801559448
        ],
        "right_elbow_pitch_joint": [
          -0.6000000238418579,
          0.16735032200813293
        ],
        "right_arm_yaw_joint": [
          -1.215348243713379,
          0.3843398690223694
        ]
      }
    },
    {
      "id": "g1_dance1_subject2_clip3",
      "source": "g1",
      "valid": 291,
      "target_feasible": 3,
      "first_frame": {
        "left_elbow_pitch_joint": -0.06331004947423935,
        "right_elbow_pitch_joint": 0.09750233590602875,
        "right_arm_yaw_joint": -0.1168246790766716
      },
      "joint_q_range": {
        "left_elbow_pitch_joint": [
          -0.6000000238418579,
          0.44002294540405273
        ],
        "right_elbow_pitch_joint": [
          -0.6000000238418579,
          0.39340534806251526
        ],
        "right_arm_yaw_joint": [
          -1.5699999332427979,
          0.20981350541114807
        ]
      }
    },
    {
      "id": "g1_dance1_subject3_clip3",
      "source": "g1",
      "valid": 228,
      "target_feasible": 0,
      "first_frame": {
        "left_elbow_pitch_joint": 0.05127726122736931,
        "right_elbow_pitch_joint": 0.02554195187985897,
        "right_arm_yaw_joint": -0.19435571134090424
      },
      "joint_q_range": {
        "left_elbow_pitch_joint": [
          -0.6000000238418579,
          0.11480704694986343
        ],
        "right_elbow_pitch_joint": [
          -0.6000000238418579,
          0.08326476067304611
        ],
        "right_arm_yaw_joint": [
          -1.5700000524520874,
          -0.19435571134090424
        ]
      }
    },
    {
      "id": "g1_fallAndGetUp1_subject1_clip3",
      "source": "g1",
      "valid": 212,
      "target_feasible": 0,
      "first_frame": {
        "left_elbow_pitch_joint": -0.06781751662492752,
        "right_elbow_pitch_joint": -0.10978677868843079,
        "right_arm_yaw_joint": -0.01201283186674118
      },
      "joint_q_range": {
        "left_elbow_pitch_joint": [
          -0.5695106387138367,
          0.010824490338563919
        ],
        "right_elbow_pitch_joint": [
          -0.5825061798095703,
          0.06549104303121567
        ],
        "right_arm_yaw_joint": [
          -1.4504340887069702,
          0.41993027925491333
        ]
      }
    },
    {
      "id": "g1_fallAndGetUp1_subject4_clip3",
      "source": "g1",
      "valid": 202,
      "target_feasible": 1,
      "first_frame": {
        "left_elbow_pitch_joint": 0.31025636196136475,
        "right_elbow_pitch_joint": 0.2762433588504791,
        "right_arm_yaw_joint": -0.11461912095546722
      },
      "joint_q_range": {
        "left_elbow_pitch_joint": [
          -0.13462898135185242,
          0.5104895830154419
        ],
        "right_elbow_pitch_joint": [
          -0.284780353307724,
          0.5755999684333801
        ],
        "right_arm_yaw_joint": [
          -1.5700000524520874,
          -0.11461912095546722
        ]
      }
    },
    {
      "id": "g1_fallAndGetUp1_subject5_clip3",
      "source": "g1",
      "valid": 172,
      "target_feasible": 0,
      "first_frame": {
        "left_elbow_pitch_joint": -0.13103316724300385,
        "right_elbow_pitch_joint": -0.1475047469139099,
        "right_arm_yaw_joint": -0.16870976984500885
      },
      "joint_q_range": {
        "left_elbow_pitch_joint": [
          -0.6000000238418579,
          0.657721996307373
        ],
        "right_elbow_pitch_joint": [
          -0.6000000238418579,
          -0.005481489468365908
        ],
        "right_arm_yaw_joint": [
          -1.5700000524520874,
          0.011232215911149979
        ]
      }
    },
    {
      "id": "g1_fight1_subject2_clip3",
      "source": "g1",
      "valid": 381,
      "target_feasible": 0,
      "first_frame": {
        "left_elbow_pitch_joint": -0.11562875658273697,
        "right_elbow_pitch_joint": -0.07422518730163574,
        "right_arm_yaw_joint": -0.02713475376367569
      },
      "joint_q_range": {
        "left_elbow_pitch_joint": [
          -0.6000000238418579,
          0.07547963410615921
        ],
        "right_elbow_pitch_joint": [
          -0.443204790353775,
          0.1731077879667282
        ],
        "right_arm_yaw_joint": [
          -0.7510197162628174,
          -0.024325024336576462
        ]
      }
    },
    {
      "id": "g1_fight1_subject3_clip3",
      "source": "g1",
      "valid": 371,
      "target_feasible": 0,
      "first_frame": {
        "left_elbow_pitch_joint": -0.09069566428661346,
        "right_elbow_pitch_joint": -0.07434480637311935,
        "right_arm_yaw_joint": -0.12909284234046936
      },
      "joint_q_range": {
        "left_elbow_pitch_joint": [
          -0.6000000238418579,
          0.02356463484466076
        ],
        "right_elbow_pitch_joint": [
          -0.6000000238418579,
          0.1752500981092453
        ],
        "right_arm_yaw_joint": [
          -1.4822437763214111,
          -0.12909284234046936
        ]
      }
    },
    {
      "id": "g1_fight1_subject5_clip3",
      "source": "g1",
      "valid": 392,
      "target_feasible": 0,
      "first_frame": {
        "left_elbow_pitch_joint": -0.1586950570344925,
        "right_elbow_pitch_joint": 6.0290512919891626e-05,
        "right_arm_yaw_joint": -0.0877179428935051
      },
      "joint_q_range": {
        "left_elbow_pitch_joint": [
          -0.6000000238418579,
          -0.1586950570344925
        ],
        "right_elbow_pitch_joint": [
          -0.6000000238418579,
          6.0290512919891626e-05
        ],
        "right_arm_yaw_joint": [
          -1.4073219299316406,
          -0.0877179428935051
        ]
      }
    },
    {
      "id": "g1_jumps1_subject1_clip3",
      "source": "g1",
      "valid": 356,
      "target_feasible": 0,
      "first_frame": {
        "left_elbow_pitch_joint": -0.15482769906520844,
        "right_elbow_pitch_joint": -0.0487314835190773,
        "right_arm_yaw_joint": -0.08869112282991409
      },
      "joint_q_range": {
        "left_elbow_pitch_joint": [
          -0.6000000238418579,
          -0.015245804563164711
        ],
        "right_elbow_pitch_joint": [
          -0.6000000238418579,
          -0.0487314835190773
        ],
        "right_arm_yaw_joint": [
          -0.7949004173278809,
          0.18142551183700562
        ]
      }
    },
    {
      "id": "g1_jumps1_subject2_clip3",
      "source": "g1",
      "valid": 390,
      "target_feasible": 0,
      "first_frame": {
        "left_elbow_pitch_joint": -0.07544052600860596,
        "right_elbow_pitch_joint": 0.03547888249158859,
        "right_arm_yaw_joint": -0.05122430622577667
      },
      "joint_q_range": {
        "left_elbow_pitch_joint": [
          -0.6000000238418579,
          -0.07544052600860596
        ],
        "right_elbow_pitch_joint": [
          -0.6000000238418579,
          0.03547888249158859
        ],
        "right_arm_yaw_joint": [
          -0.7432587742805481,
          -0.05122430622577667
        ]
      }
    },
    {
      "id": "g1_jumps1_subject5_clip3",
      "source": "g1",
      "valid": 378,
      "target_feasible": 0,
      "first_frame": {
        "left_elbow_pitch_joint": -0.1614433079957962,
        "right_elbow_pitch_joint": -0.1170719563961029,
        "right_arm_yaw_joint": -0.1367676556110382
      },
      "joint_q_range": {
        "left_elbow_pitch_joint": [
          -0.6000000238418579,
          -0.1614433079957962
        ],
        "right_elbow_pitch_joint": [
          -0.6000000238418579,
          -0.1170719563961029
        ],
        "right_arm_yaw_joint": [
          -0.8052541017532349,
          0.0839824229478836
        ]
      }
    },
    {
      "id": "g1_run1_subject2_clip3",
      "source": "g1",
      "valid": 387,
      "target_feasible": 0,
      "first_frame": {
        "left_elbow_pitch_joint": -0.06658478081226349,
        "right_elbow_pitch_joint": 0.01990044303238392,
        "right_arm_yaw_joint": -0.0327557809650898
      },
      "joint_q_range": {
        "left_elbow_pitch_joint": [
          -0.6000000238418579,
          -0.06658478081226349
        ],
        "right_elbow_pitch_joint": [
          -0.4726267457008362,
          0.01990044303238392
        ],
        "right_arm_yaw_joint": [
          -0.6961184740066528,
          -0.0327557809650898
        ]
      }
    },
    {
      "id": "g1_run1_subject5_clip3",
      "source": "g1",
      "valid": 223,
      "target_feasible": 0,
      "first_frame": {
        "left_elbow_pitch_joint": -0.11664905399084091,
        "right_elbow_pitch_joint": 0.01723378896713257,
        "right_arm_yaw_joint": -0.1507043093442917
      },
      "joint_q_range": {
        "left_elbow_pitch_joint": [
          -0.6000000238418579,
          -0.11664905399084091
        ],
        "right_elbow_pitch_joint": [
          -0.6000000238418579,
          0.01723378896713257
        ],
        "right_arm_yaw_joint": [
          -1.000701665878296,
          -0.1507043093442917
        ]
      }
    },
    {
      "id": "g1_sprint1_subject2_clip3",
      "source": "g1",
      "valid": 371,
      "target_feasible": 0,
      "first_frame": {
        "left_elbow_pitch_joint": 0.019909091293811798,
        "right_elbow_pitch_joint": 0.13409121334552765,
        "right_arm_yaw_joint": -0.0683622881770134
      },
      "joint_q_range": {
        "left_elbow_pitch_joint": [
          -0.6000000238418579,
          0.04583283141255379
        ],
        "right_elbow_pitch_joint": [
          -0.5450780391693115,
          0.25960931181907654
        ],
        "right_arm_yaw_joint": [
          -1.1207321882247925,
          0.27865105867385864
        ]
      }
    },
    {
      "id": "g1_sprint1_subject4_clip3",
      "source": "g1",
      "valid": 395,
      "target_feasible": 0,
      "first_frame": {
        "left_elbow_pitch_joint": 0.02816898003220558,
        "right_elbow_pitch_joint": -0.027675388380885124,
        "right_arm_yaw_joint": -0.15513119101524353
      },
      "joint_q_range": {
        "left_elbow_pitch_joint": [
          -0.4761897623538971,
          0.4036882519721985
        ],
        "right_elbow_pitch_joint": [
          -0.6000000238418579,
          -0.027675388380885124
        ],
        "right_arm_yaw_joint": [
          -1.1957192420959473,
          -0.09730236977338791
        ]
      }
    },
    {
      "id": "g1_walk1_subject1_clip3",
      "source": "g1",
      "valid": 400,
      "target_feasible": 0,
      "first_frame": {
        "left_elbow_pitch_joint": -0.04907841980457306,
        "right_elbow_pitch_joint": -0.03786123916506767,
        "right_arm_yaw_joint": -0.09283493459224701
      },
      "joint_q_range": {
        "left_elbow_pitch_joint": [
          -0.3923731744289398,
          -0.04907841980457306
        ],
        "right_elbow_pitch_joint": [
          -0.45130133628845215,
          -0.03786123916506767
        ],
        "right_arm_yaw_joint": [
          -0.47379010915756226,
          -0.09283493459224701
        ]
      }
    },
    {
      "id": "g1_walk1_subject2_clip3",
      "source": "g1",
      "valid": 400,
      "target_feasible": 0,
      "first_frame": {
        "left_elbow_pitch_joint": -0.0812138170003891,
        "right_elbow_pitch_joint": -0.049070343375205994,
        "right_arm_yaw_joint": -0.009908233769237995
      },
      "joint_q_range": {
        "left_elbow_pitch_joint": [
          -0.4341455101966858,
          -0.06983013451099396
        ],
        "right_elbow_pitch_joint": [
          -0.45922696590423584,
          -0.049070343375205994
        ],
        "right_arm_yaw_joint": [
          -0.13875985145568848,
          0.20441216230392456
        ]
      }
    },
    {
      "id": "g1_walk1_subject5_clip3",
      "source": "g1",
      "valid": 400,
      "target_feasible": 0,
      "first_frame": {
        "left_elbow_pitch_joint": -0.13529755175113678,
        "right_elbow_pitch_joint": 0.0536690279841423,
        "right_arm_yaw_joint": -0.059852298349142075
      },
      "joint_q_range": {
        "left_elbow_pitch_joint": [
          -0.5999999046325684,
          0.2913557291030884
        ],
        "right_elbow_pitch_joint": [
          -0.6000000238418579,
          0.06776145845651627
        ],
        "right_arm_yaw_joint": [
          -0.6080084443092346,
          -0.04721621423959732
        ]
      }
    },
    {
      "id": "g1_walk2_subject1_clip3",
      "source": "g1",
      "valid": 400,
      "target_feasible": 0,
      "first_frame": {
        "left_elbow_pitch_joint": -0.010582692921161652,
        "right_elbow_pitch_joint": -0.02721426822245121,
        "right_arm_yaw_joint": -0.041179243475198746
      },
      "joint_q_range": {
        "left_elbow_pitch_joint": [
          -0.5805358290672302,
          0.058169204741716385
        ],
        "right_elbow_pitch_joint": [
          -0.5999999642372131,
          -0.02721426822245121
        ],
        "right_arm_yaw_joint": [
          -0.6953646540641785,
          0.184219628572464
        ]
      }
    },
    {
      "id": "g1_walk2_subject3_clip3",
      "source": "g1",
      "valid": 400,
      "target_feasible": 0,
      "first_frame": {
        "left_elbow_pitch_joint": -0.10791340470314026,
        "right_elbow_pitch_joint": -0.14592479169368744,
        "right_arm_yaw_joint": -0.06758217513561249
      },
      "joint_q_range": {
        "left_elbow_pitch_joint": [
          -0.6000000238418579,
          -0.10791340470314026
        ],
        "right_elbow_pitch_joint": [
          -0.5999999642372131,
          -0.055361777544021606
        ],
        "right_arm_yaw_joint": [
          -0.543958306312561,
          0.050447773188352585
        ]
      }
    },
    {
      "id": "g1_walk2_subject4_clip3",
      "source": "g1",
      "valid": 393,
      "target_feasible": 0,
      "first_frame": {
        "left_elbow_pitch_joint": 0.09170354157686234,
        "right_elbow_pitch_joint": 0.1681629866361618,
        "right_arm_yaw_joint": -0.14926598966121674
      },
      "joint_q_range": {
        "left_elbow_pitch_joint": [
          -0.35919830203056335,
          0.21192389726638794
        ],
        "right_elbow_pitch_joint": [
          -0.40667012333869934,
          0.2875871956348419
        ],
        "right_arm_yaw_joint": [
          -1.104311466217041,
          -0.14926598966121674
        ]
      }
    },
    {
      "id": "rpo_114_08",
      "source": "official_rpo",
      "valid": 460,
      "target_feasible": 208,
      "first_frame": {
        "left_elbow_pitch_joint": 0.7027087807655334,
        "right_elbow_pitch_joint": 0.16432012617588043,
        "right_arm_yaw_joint": -0.5725463032722473
      },
      "joint_q_range": {
        "left_elbow_pitch_joint": [
          0.7272393107414246,
          1.1489121913909912
        ],
        "right_elbow_pitch_joint": [
          0.54011070728302,
          1.2854284048080444
        ],
        "right_arm_yaw_joint": [
          -0.49631479382514954,
          -0.08552668243646622
        ]
      }
    },
    {
      "id": "rpo_114_09",
      "source": "official_rpo",
      "valid": 782,
      "target_feasible": 134,
      "first_frame": {
        "left_elbow_pitch_joint": -0.012356306426227093,
        "right_elbow_pitch_joint": 1.0627245903015137,
        "right_arm_yaw_joint": -0.358586847782135
      },
      "joint_q_range": {
        "left_elbow_pitch_joint": [
          -0.012356306426227093,
          1.1631572246551514
        ],
        "right_elbow_pitch_joint": [
          0.240297332406044,
          1.2251944541931152
        ],
        "right_arm_yaw_joint": [
          -0.6157090067863464,
          0.15366825461387634
        ]
      }
    },
    {
      "id": "rpo_127_04",
      "source": "official_rpo",
      "valid": 15,
      "target_feasible": 0,
      "first_frame": {
        "left_elbow_pitch_joint": 1.1006102561950684,
        "right_elbow_pitch_joint": 1.1850022077560425,
        "right_arm_yaw_joint": -0.2879284918308258
      },
      "joint_q_range": {
        "left_elbow_pitch_joint": [
          -0.5936942100524902,
          0.31295594573020935
        ],
        "right_elbow_pitch_joint": [
          -0.5346862077713013,
          0.48241668939590454
        ],
        "right_arm_yaw_joint": [
          -0.3744443356990814,
          0.21998277306556702
        ]
      }
    },
    {
      "id": "rpo_127_06",
      "source": "official_rpo",
      "valid": 13,
      "target_feasible": 0,
      "first_frame": {
        "left_elbow_pitch_joint": -0.2839474380016327,
        "right_elbow_pitch_joint": 0.7637654542922974,
        "right_arm_yaw_joint": -0.22775162756443024
      },
      "joint_q_range": {
        "left_elbow_pitch_joint": [
          -0.526894748210907,
          0.10349047183990479
        ],
        "right_elbow_pitch_joint": [
          -0.5626831650733948,
          0.27068179845809937
        ],
        "right_arm_yaw_joint": [
          -0.1757388859987259,
          -0.03569521754980087
        ]
      }
    },
    {
      "id": "rpo_36_01",
      "source": "official_rpo",
      "valid": 145,
      "target_feasible": 135,
      "first_frame": {
        "left_elbow_pitch_joint": 0.9120858907699585,
        "right_elbow_pitch_joint": 1.112180471420288,
        "right_arm_yaw_joint": -0.29382121562957764
      },
      "joint_q_range": {
        "left_elbow_pitch_joint": [
          0.6743707060813904,
          1.204448938369751
        ],
        "right_elbow_pitch_joint": [
          0.9357580542564392,
          1.305586576461792
        ],
        "right_arm_yaw_joint": [
          -0.49507400393486023,
          0.14799030125141144
        ]
      }
    },
    {
      "id": "rpo_36_11",
      "source": "official_rpo",
      "valid": 1159,
      "target_feasible": 952,
      "first_frame": {
        "left_elbow_pitch_joint": 1.043128490447998,
        "right_elbow_pitch_joint": 1.1567739248275757,
        "right_arm_yaw_joint": -0.25090688467025757
      },
      "joint_q_range": {
        "left_elbow_pitch_joint": [
          0.6565167307853699,
          1.277398943901062
        ],
        "right_elbow_pitch_joint": [
          0.6557829976081848,
          1.3000651597976685
        ],
        "right_arm_yaw_joint": [
          -0.473092645406723,
          0.006215346977114677
        ]
      }
    },
    {
      "id": "rpo_A1-_Stand_stageii",
      "source": "official_rpo",
      "valid": 0,
      "target_feasible": 0,
      "first_frame": {
        "left_elbow_pitch_joint": 1.2044469118118286,
        "right_elbow_pitch_joint": 1.1636255979537964,
        "right_arm_yaw_joint": -0.3556744456291199
      },
      "joint_q_range": {}
    },
    {
      "id": "rpo_B10_-__Walk_turn_left_45_stageii",
      "source": "official_rpo",
      "valid": 21,
      "target_feasible": 11,
      "first_frame": {
        "left_elbow_pitch_joint": 1.2069960832595825,
        "right_elbow_pitch_joint": 1.1555862426757812,
        "right_arm_yaw_joint": -0.26047825813293457
      },
      "joint_q_range": {
        "left_elbow_pitch_joint": [
          1.0603015422821045,
          1.303269386291504
        ],
        "right_elbow_pitch_joint": [
          0.5029951333999634,
          1.0095974206924438
        ],
        "right_arm_yaw_joint": [
          -0.3844238519668579,
          -0.27376672625541687
        ]
      }
    },
    {
      "id": "rpo_B11_-__Walk_turn_left_135_stageii",
      "source": "official_rpo",
      "valid": 45,
      "target_feasible": 29,
      "first_frame": {
        "left_elbow_pitch_joint": 1.1432098150253296,
        "right_elbow_pitch_joint": 1.1562868356704712,
        "right_arm_yaw_joint": -0.34752336144447327
      },
      "joint_q_range": {
        "left_elbow_pitch_joint": [
          0.7280954718589783,
          1.2433195114135742
        ],
        "right_elbow_pitch_joint": [
          0.5340481400489807,
          1.2298223972320557
        ],
        "right_arm_yaw_joint": [
          -0.30049270391464233,
          0.032828252762556076
        ]
      }
    },
    {
      "id": "rpo_B13_-__Walk_turn_right_90_stageii",
      "source": "official_rpo",
      "valid": 253,
      "target_feasible": 163,
      "first_frame": {
        "left_elbow_pitch_joint": 1.1293983459472656,
        "right_elbow_pitch_joint": 1.1153520345687866,
        "right_arm_yaw_joint": -0.3751285672187805
      },
      "joint_q_range": {
        "left_elbow_pitch_joint": [
          0.8034533262252808,
          1.2697181701660156
        ],
        "right_elbow_pitch_joint": [
          0.6247826814651489,
          1.1813740730285645
        ],
        "right_arm_yaw_joint": [
          -0.47876593470573425,
          -0.09112849831581116
        ]
      }
    },
    {
      "id": "rpo_B14_-__Walk_turn_right_45_t2_stageii",
      "source": "official_rpo",
      "valid": 191,
      "target_feasible": 151,
      "first_frame": {
        "left_elbow_pitch_joint": 1.1644805669784546,
        "right_elbow_pitch_joint": 1.0957014560699463,
        "right_arm_yaw_joint": -0.2512586712837219
      },
      "joint_q_range": {
        "left_elbow_pitch_joint": [
          0.7524382472038269,
          1.2662220001220703
        ],
        "right_elbow_pitch_joint": [
          0.5784949660301208,
          1.1848580837249756
        ],
        "right_arm_yaw_joint": [
          -0.4100404977798462,
          0.014806346036493778
        ]
      }
    },
    {
      "id": "rpo_B15_-__Walk_turn_around_stageii",
      "source": "official_rpo",
      "valid": 33,
      "target_feasible": 24,
      "first_frame": {
        "left_elbow_pitch_joint": 1.1241776943206787,
        "right_elbow_pitch_joint": 1.0950669050216675,
        "right_arm_yaw_joint": -0.2594263255596161
      },
      "joint_q_range": {
        "left_elbow_pitch_joint": [
          0.5832140445709229,
          1.2358952760696411
        ],
        "right_elbow_pitch_joint": [
          0.4004038870334625,
          1.1792477369308472
        ],
        "right_arm_yaw_joint": [
          -0.3296028673648834,
          -0.025925982743501663
        ]
      }
    },
    {
      "id": "rpo_B4_-_Stand_to_Walk_backwards_stageii",
      "source": "official_rpo",
      "valid": 21,
      "target_feasible": 13,
      "first_frame": {
        "left_elbow_pitch_joint": 1.1887145042419434,
        "right_elbow_pitch_joint": 1.1385579109191895,
        "right_arm_yaw_joint": -0.3854866027832031
      },
      "joint_q_range": {
        "left_elbow_pitch_joint": [
          0.845918595790863,
          1.2503846883773804
        ],
        "right_elbow_pitch_joint": [
          0.7610013484954834,
          1.0619440078735352
        ],
        "right_arm_yaw_joint": [
          -0.42725273966789246,
          -0.21770688891410828
        ]
      }
    },
    {
      "id": "rpo_B5_-__Walk_backwards_stageii",
      "source": "official_rpo",
      "valid": 202,
      "target_feasible": 109,
      "first_frame": {
        "left_elbow_pitch_joint": 0.5582261681556702,
        "right_elbow_pitch_joint": 0.5649946928024292,
        "right_arm_yaw_joint": -0.39457228779792786
      },
      "joint_q_range": {
        "left_elbow_pitch_joint": [
          0.7397725582122803,
          1.2916183471679688
        ],
        "right_elbow_pitch_joint": [
          0.6968631148338318,
          1.3067072629928589
        ],
        "right_arm_yaw_joint": [
          -0.5315547585487366,
          -0.04232010990381241
        ]
      }
    },
    {
      "id": "rpo_B9_-__Walk_turn_left_90_stageii",
      "source": "official_rpo",
      "valid": 16,
      "target_feasible": 16,
      "first_frame": {
        "left_elbow_pitch_joint": 1.1593220233917236,
        "right_elbow_pitch_joint": 1.137406826019287,
        "right_arm_yaw_joint": -0.33571338653564453
      },
      "joint_q_range": {
        "left_elbow_pitch_joint": [
          0.7498645782470703,
          1.2751423120498657
        ],
        "right_elbow_pitch_joint": [
          0.8228517174720764,
          1.1867250204086304
        ],
        "right_arm_yaw_joint": [
          -0.2460823655128479,
          -0.0459846556186676
        ]
      }
    },
    {
      "id": "rpo_C12_-_run_turn_left_45_stageii",
      "source": "official_rpo",
      "valid": 66,
      "target_feasible": 0,
      "first_frame": {
        "left_elbow_pitch_joint": 0.18488886952400208,
        "right_elbow_pitch_joint": 0.04767969995737076,
        "right_arm_yaw_joint": -0.21537016332149506
      },
      "joint_q_range": {
        "left_elbow_pitch_joint": [
          -0.4326065480709076,
          0.585983395576477
        ],
        "right_elbow_pitch_joint": [
          -0.5843568444252014,
          0.7086511850357056
        ],
        "right_arm_yaw_joint": [
          -0.28516027331352234,
          0.20966364443302155
        ]
      }
    },
    {
      "id": "rpo_C17_-_run_change_direction_stageii",
      "source": "official_rpo",
      "valid": 117,
      "target_feasible": 0,
      "first_frame": {
        "left_elbow_pitch_joint": -0.1907922923564911,
        "right_elbow_pitch_joint": 0.41551756858825684,
        "right_arm_yaw_joint": -0.13380306959152222
      },
      "joint_q_range": {
        "left_elbow_pitch_joint": [
          -0.5361177325248718,
          0.935261607170105
        ],
        "right_elbow_pitch_joint": [
          -0.5466627478599548,
          0.5228297710418701
        ],
        "right_arm_yaw_joint": [
          -0.33231472969055176,
          0.37228938937187195
        ]
      }
    },
    {
      "id": "rpo_C1_-_stand_to_run_stageii",
      "source": "official_rpo",
      "valid": 169,
      "target_feasible": 113,
      "first_frame": {
        "left_elbow_pitch_joint": 1.1842454671859741,
        "right_elbow_pitch_joint": 1.063921570777893,
        "right_arm_yaw_joint": -0.28836408257484436
      },
      "joint_q_range": {
        "left_elbow_pitch_joint": [
          -0.2970103919506073,
          1.2025325298309326
        ],
        "right_elbow_pitch_joint": [
          -0.4445802569389343,
          1.063921570777893
        ],
        "right_arm_yaw_joint": [
          -0.30128705501556396,
          0.0932452604174614
        ]
      }
    },
    {
      "id": "rpo_C3_-_run_stageii",
      "source": "official_rpo",
      "valid": 98,
      "target_feasible": 0,
      "first_frame": {
        "left_elbow_pitch_joint": 0.07380982488393784,
        "right_elbow_pitch_joint": 0.4732973575592041,
        "right_arm_yaw_joint": -0.050006791949272156
      },
      "joint_q_range": {
        "left_elbow_pitch_joint": [
          -0.40680697560310364,
          0.2162727266550064
        ],
        "right_elbow_pitch_joint": [
          -0.5953333377838135,
          0.3924627900123596
        ],
        "right_arm_yaw_joint": [
          -0.28304046392440796,
          0.3993236720561981
        ]
      }
    },
    {
      "id": "rpo_C4_-_run_to_walk_a_stageii",
      "source": "official_rpo",
      "valid": 120,
      "target_feasible": 35,
      "first_frame": {
        "left_elbow_pitch_joint": 0.2566046416759491,
        "right_elbow_pitch_joint": -0.1362576186656952,
        "right_arm_yaw_joint": -0.3307928144931793
      },
      "joint_q_range": {
        "left_elbow_pitch_joint": [
          -0.1933874487876892,
          1.4080758094787598
        ],
        "right_elbow_pitch_joint": [
          -0.37953656911849976,
          1.288273811340332
        ],
        "right_arm_yaw_joint": [
          -0.44735151529312134,
          0.18027430772781372
        ]
      }
    },
    {
      "id": "rpo_C5_-_walk_to_run_stageii",
      "source": "official_rpo",
      "valid": 119,
      "target_feasible": 33,
      "first_frame": {
        "left_elbow_pitch_joint": 0.7576902508735657,
        "right_elbow_pitch_joint": 0.3570346236228943,
        "right_arm_yaw_joint": -0.3931472897529602
      },
      "joint_q_range": {
        "left_elbow_pitch_joint": [
          -0.34836703538894653,
          1.3301986455917358
        ],
        "right_elbow_pitch_joint": [
          -0.5151633620262146,
          1.2805590629577637
        ],
        "right_arm_yaw_joint": [
          -0.4528947174549103,
          0.08325531333684921
        ]
      }
    },
    {
      "id": "rpo_move_back",
      "source": "official_rpo",
      "valid": 312,
      "target_feasible": 0,
      "first_frame": {
        "left_elbow_pitch_joint": 1.3223261833190918,
        "right_elbow_pitch_joint": 1.3639198541641235,
        "right_arm_yaw_joint": -0.4668909013271332
      },
      "joint_q_range": {
        "left_elbow_pitch_joint": [
          1.2784181833267212,
          1.4012733697891235
        ],
        "right_elbow_pitch_joint": [
          1.3263986110687256,
          1.446001410484314
        ],
        "right_arm_yaw_joint": [
          -0.5513309836387634,
          -0.29136279225349426
        ]
      }
    },
    {
      "id": "rpo_move_l",
      "source": "official_rpo",
      "valid": 280,
      "target_feasible": 59,
      "first_frame": {
        "left_elbow_pitch_joint": 1.3124127388000488,
        "right_elbow_pitch_joint": 1.2281315326690674,
        "right_arm_yaw_joint": -0.6306757926940918
      },
      "joint_q_range": {
        "left_elbow_pitch_joint": [
          1.1410045623779297,
          1.3170452117919922
        ],
        "right_elbow_pitch_joint": [
          1.1502751111984253,
          1.2692091464996338
        ],
        "right_arm_yaw_joint": [
          -0.6349427103996277,
          -0.3818787634372711
        ]
      }
    },
    {
      "id": "rpo_move_r",
      "source": "official_rpo",
      "valid": 250,
      "target_feasible": 8,
      "first_frame": {
        "left_elbow_pitch_joint": 1.2506691217422485,
        "right_elbow_pitch_joint": 1.2320599555969238,
        "right_arm_yaw_joint": -0.49922049045562744
      },
      "joint_q_range": {
        "left_elbow_pitch_joint": [
          1.2506691217422485,
          1.3648223876953125
        ],
        "right_elbow_pitch_joint": [
          1.2136211395263672,
          1.3030754327774048
        ],
        "right_arm_yaw_joint": [
          -0.5630363821983337,
          -0.46002328395843506
        ]
      }
    },
    {
      "id": "rpo_run_start_180_R_001__A345",
      "source": "official_rpo",
      "valid": 143,
      "target_feasible": 0,
      "first_frame": {
        "left_elbow_pitch_joint": 1.003098726272583,
        "right_elbow_pitch_joint": 1.0961918830871582,
        "right_arm_yaw_joint": -0.9454082250595093
      },
      "joint_q_range": {
        "left_elbow_pitch_joint": [
          -0.5999207496643066,
          1.0349937677383423
        ],
        "right_elbow_pitch_joint": [
          -0.5858049988746643,
          1.3401142358779907
        ],
        "right_arm_yaw_joint": [
          -1.0014212131500244,
          0.5771026611328125
        ]
      }
    },
    {
      "id": "rpo_run_start_180_R_001__A345_M",
      "source": "official_rpo",
      "valid": 145,
      "target_feasible": 0,
      "first_frame": {
        "left_elbow_pitch_joint": 1.121012568473816,
        "right_elbow_pitch_joint": 0.9810613393783569,
        "right_arm_yaw_joint": -0.8014596700668335
      },
      "joint_q_range": {
        "left_elbow_pitch_joint": [
          -0.5860040187835693,
          1.3568679094314575
        ],
        "right_elbow_pitch_joint": [
          -0.5949754118919373,
          1.0180425643920898
        ],
        "right_arm_yaw_joint": [
          -1.2011404037475586,
          -0.05106934532523155
        ]
      }
    },
    {
      "id": "rpo_run_stop_180_R_001__A345",
      "source": "official_rpo",
      "valid": 190,
      "target_feasible": 0,
      "first_frame": {
        "left_elbow_pitch_joint": 0.2033587396144867,
        "right_elbow_pitch_joint": -0.07249374687671661,
        "right_arm_yaw_joint": -0.47894105315208435
      },
      "joint_q_range": {
        "left_elbow_pitch_joint": [
          -0.5833985209465027,
          1.1679672002792358
        ],
        "right_elbow_pitch_joint": [
          -0.5978000164031982,
          1.5577946901321411
        ],
        "right_arm_yaw_joint": [
          -0.8816852569580078,
          0.30827775597572327
        ]
      }
    },
    {
      "id": "rpo_run_stop_180_R_001__A345_M",
      "source": "official_rpo",
      "valid": 190,
      "target_feasible": 0,
      "first_frame": {
        "left_elbow_pitch_joint": -0.05957012623548508,
        "right_elbow_pitch_joint": 0.18910814821720123,
        "right_arm_yaw_joint": -0.6165137887001038
      },
      "joint_q_range": {
        "left_elbow_pitch_joint": [
          -0.5957021713256836,
          1.4912320375442505
        ],
        "right_elbow_pitch_joint": [
          -0.5158136487007141,
          1.1699061393737793
        ],
        "right_arm_yaw_joint": [
          -0.7815667986869812,
          0.11435545980930328
        ]
      }
    },
    {
      "id": "rpo_turn_l",
      "source": "official_rpo",
      "valid": 340,
      "target_feasible": 6,
      "first_frame": {
        "left_elbow_pitch_joint": 1.2924429178237915,
        "right_elbow_pitch_joint": 1.2839926481246948,
        "right_arm_yaw_joint": -0.568901777267456
      },
      "joint_q_range": {
        "left_elbow_pitch_joint": [
          1.221816897392273,
          1.378539800643921
        ],
        "right_elbow_pitch_joint": [
          1.2080758810043335,
          1.3586169481277466
        ],
        "right_arm_yaw_joint": [
          -0.625886082649231,
          -0.36871397495269775
        ]
      }
    },
    {
      "id": "rpo_turn_r",
      "source": "official_rpo",
      "valid": 239,
      "target_feasible": 5,
      "first_frame": {
        "left_elbow_pitch_joint": 1.32046377658844,
        "right_elbow_pitch_joint": 1.3113319873809814,
        "right_arm_yaw_joint": -0.5252022743225098
      },
      "joint_q_range": {
        "left_elbow_pitch_joint": [
          1.1957013607025146,
          1.3732247352600098
        ],
        "right_elbow_pitch_joint": [
          1.1493333578109741,
          1.3317859172821045
        ],
        "right_arm_yaw_joint": [
          -0.5930209755897522,
          -0.3778374493122101
        ]
      }
    }
  ],
  "by_source": {
    "official_rpo": {
      "valid": 6134,
      "target_feasible": 2204
    },
    "g1": {
      "valid": 7510,
      "target_feasible": 4
    }
  }
}
```


真实小步更新行为核验：
```json
{
  "status": "passed",
  "six_adam_steps": {
    "forward_optimizer": 1,
    "backward_optimizer": 1,
    "discriminator_optimizer": 1,
    "actor_optimizer": 1,
    "critic_optimizer": 1,
    "aux_critic_optimizer": 1
  },
  "metrics": {
    "disc_loss": 3.3286056518554688,
    "disc_expert_loss": 0.7785646915435791,
    "disc_train_loss": 0.6742761135101318,
    "disc_wgan_gp_loss": 0.1875765025615692,
    "target_M": -0.48164236545562744,
    "M1": 25.50908660888672,
    "F1": 1.6647038459777832,
    "B": 0.03200933337211609,
    "B_norm": 16.0,
    "z_norm": 16.0,
    "fb_loss": 1355173.125,
    "fb_diag": -336.4302978515625,
    "fb_offdiag": 1370669.875,
    "orth_loss": -151.60391235351562,
    "orth_loss_diag": -255.99998474121094,
    "orth_loss_offdiag": 104.39606475830078,
    "q_loss": 0.0,
    "target_Q": -2.3109960556030273,
    "Q1": -0.22287428379058838,
    "mean_next_Q": 0.2652205526828766,
    "unc_Q": 1.1651270389556885,
    "critic_loss": 6.804961204528809,
    "mean_disc_reward": -2.0,
    "mean_disc_reward_raw": -10.0,
    "disc_reward_clip_fraction": 1.0,
    "aux_rew/penalty_torques": 0.6681990027427673,
    "aux_rew/penalty_action_rate": 0.4848378598690033,
    "aux_rew/limits_dof_pos": 0.5348075032234192,
    "aux_rew/limits_torque": 0.3868333399295807,
    "aux_rew/penalty_undesired_contact": 0.6390904784202576,
    "aux_rew/penalty_feet_ori": 0.4724477231502533,
    "aux_rew/penalty_ankle_roll": 0.5019434690475464,
    "aux_rew/penalty_slippage": 0.5858249068260193,
    "aux_rew/limits_dof_vel": 0.42562592029571533,
    "target_auxQ": -3.404588222503662,
    "auxQ1": -0.02961719036102295,
    "mean_next_auxQ": 0.1060590073466301,
    "unc_auxQ": 1.697151780128479,
    "aux_critic_loss": 15.298188209533691,
    "mean_aux_reward": -2.6769216060638428,
    "actor_loss": 640.1534423828125,
    "actor_reg_weight_raw": 840.4491577148438,
    "actor_reg_weight_applied": 50.0,
    "Q_discriminator": -1.495289921760559,
    "Q_aux": -1.6831313371658325,
    "Q_fb": -634.7320556640625
  },
  "optimizer_metrics": {
    "discriminator_optimizer/lr": 3e-06,
    "discriminator_optimizer/grad_norm_l2": 9.09282398223877,
    "discriminator_optimizer/grad_norm_after_clip_l2": 4.999999523162842,
    "discriminator_optimizer/grad_clip_limit": 5.0,
    "discriminator_optimizer/parameter_delta_l2": 0.0006506613572128117,
    "forward_optimizer/lr": 0.0003,
    "forward_optimizer/grad_norm_l2": 4401968.0,
    "forward_optimizer/grad_norm_after_clip_l2": 5.000001430511475,
    "forward_optimizer/grad_clip_limit": 5.0,
    "forward_optimizer/parameter_delta_l2": 0.16158534586429596,
    "backward_optimizer/lr": 1e-05,
    "backward_optimizer/grad_norm_l2": 3988921.25,
    "backward_optimizer/grad_norm_after_clip_l2": 5.000004291534424,
    "backward_optimizer/grad_clip_limit": 5.0,
    "backward_optimizer/parameter_delta_l2": 0.004141205921769142,
    "critic_optimizer/lr": 0.0003,
    "critic_optimizer/grad_norm_l2": 44.2822151184082,
    "critic_optimizer/grad_norm_after_clip_l2": 5.00000524520874,
    "critic_optimizer/grad_clip_limit": 5.0,
    "critic_optimizer/parameter_delta_l2": 0.15207302570343018,
    "aux_critic_optimizer/lr": 0.0003,
    "aux_critic_optimizer/grad_norm_l2": 77.69927215576172,
    "aux_critic_optimizer/grad_norm_after_clip_l2": 5.000002861022949,
    "aux_critic_optimizer/grad_clip_limit": 5.0,
    "aux_critic_optimizer/parameter_delta_l2": 0.15207336843013763,
    "actor_optimizer/lr": 0.0003,
    "actor_optimizer/grad_norm_l2": 1255.4189453125,
    "actor_optimizer/grad_norm_after_clip_l2": 1.000000238418579,
    "actor_optimizer/grad_clip_limit": 1.0,
    "actor_optimizer/parameter_delta_l2": 0.08545085042715073
  }
}
```


恢复资格段控制trace核验：
```json
{
  "rows": 24576,
  "rounds": 1,
  "round_first": 3,
  "round_last": 3,
  "requested_saturation_fraction": 0.37526360110960144,
  "applied_saturation_fraction": 0.036233653192934784,
  "hard_position_velocity_torque_counts": [
    53,
    154,
    0
  ],
  "max_joint_velocity_radps": 25.0364990234375,
  "soft_velocity_raw_mean": 0.0027588258186976113,
  "any_joint_over_soft_limit_fraction": 0.0579833984375,
  "retained_replay_rows_checked": 24576,
  "hard_terminal_episode_changes_checked": 85,
  "pd_decoder_max_error": 1.9572377185639311e-07,
  "reset_in_retained_replay": 6402,
  "first_step_terms_in_retained_replay": 139,
  "first_step_term_fraction": 0.021711965010934083,
  "reset_outside_box_fraction": 0.0,
  "replay_actions_are_requested": true,
  "ninth_reward_matches_actual_dq": true
}
```


全段控制trace核验：
```json
{
  "rows": 786432,
  "rounds": 32,
  "round_first": 4,
  "round_last": 35,
  "requested_saturation_fraction": 0.4419359953507133,
  "applied_saturation_fraction": 0.16147906538369,
  "hard_position_velocity_torque_counts": [
    10692,
    6675,
    0
  ],
  "max_joint_velocity_radps": 68.68045806884766,
  "soft_velocity_raw_mean": 0.015343798654309163,
  "any_joint_over_soft_limit_fraction": 0.33545811971028644,
  "retained_replay_rows_checked": 786432,
  "hard_terminal_episode_changes_checked": 15853,
  "pd_decoder_max_error": 2.828240392993564e-07,
  "reset_in_retained_replay": 66449,
  "first_step_terms_in_retained_replay": 963,
  "first_step_term_fraction": 0.01449231741636443,
  "reset_outside_box_fraction": 0.0,
  "replay_actions_are_requested": true,
  "ninth_reward_matches_actual_dq": true
}
```


速度极值关节与回合阶段定位：
```json
{
  "scope": "all 32 formal sampling rounds, max(abs(dq)/Vmax) per transition",
  "rows": 786432,
  "peak_absolute_velocity": {
    "absolute_velocity_radps": 68.68045806884766,
    "joint": "right_ankle_roll_joint",
    "nominal_limit_radps": 8.0,
    "ratio_to_limit": 8.585057258605957,
    "round": 7,
    "step_in_episode": 13,
    "hard_position_velocity_torque": [
      1,
      1,
      0
    ],
    "requested_action": -0.19165153801441193,
    "applied_action": -0.19165153801441193,
    "pd_target_rad": -0.09582576900720596
  },
  "ratio_quantiles": {
    "0.5": 0.601872056722641,
    "0.95": 0.9928547769784927,
    "0.99": 1.0000008344650269,
    "0.999": 1.2870153053999036,
    "1": 8.585057258605957
  },
  "any_hard_velocity_ratio_fraction": 0.01996612548828125,
  "actual_hard_velocity_tolerance_radps": 0.001,
  "actual_hard_velocity_rows": 6675,
  "actual_hard_velocity_fraction": 0.008487701416015625,
  "first_step_rows": 66449,
  "first_step_hard_rows": 62,
  "later_hard_rows": 15900
}
```


最终replay、RSI和latent核验：
```json
{
  "scope": "all final retained replay rows; chronological complete 6144-row blocks",
  "same_episode_nonperiodic_pairs": 734449,
  "passive_z_changes": 0,
  "native_period_100_pairs_excluded": 0,
  "cosine_median": 1.0,
  "reset_rows": 79752,
  "reset_joint_residuals": [
    {
      "joint": "left_thigh_yaw_joint",
      "outside_count": 0,
      "outside_fraction": 0.0,
      "mean_residual_rad": 8.60228531960343e-11,
      "p95_residual_rad": 0.0,
      "max_residual_rad": 2.980232227667301e-09
    },
    {
      "joint": "right_thigh_yaw_joint",
      "outside_count": 0,
      "outside_fraction": 0.0,
      "mean_residual_rad": 3.0081840496441185e-11,
      "p95_residual_rad": 0.0,
      "max_residual_rad": 2.980232227667301e-09
    },
    {
      "joint": "torso_joint",
      "outside_count": 0,
      "outside_fraction": 0.0,
      "mean_residual_rad": 0.0,
      "p95_residual_rad": 0.0,
      "max_residual_rad": 0.0
    },
    {
      "joint": "left_thigh_roll_joint",
      "outside_count": 0,
      "outside_fraction": 0.0,
      "mean_residual_rad": 1.646093259463645e-10,
      "p95_residual_rad": 2.980232227667301e-09,
      "max_residual_rad": 2.980232227667301e-09
    },
    {
      "joint": "right_thigh_roll_joint",
      "outside_count": 0,
      "outside_fraction": 0.0,
      "mean_residual_rad": 1.5960191398795067e-10,
      "p95_residual_rad": 2.980232227667301e-09,
      "max_residual_rad": 2.980232227667301e-09
    },
    {
      "joint": "left_arm_pitch_joint",
      "outside_count": 0,
      "outside_fraction": 0.0,
      "mean_residual_rad": 0.0,
      "p95_residual_rad": 0.0,
      "max_residual_rad": 0.0
    },
    {
      "joint": "right_arm_pitch_joint",
      "outside_count": 0,
      "outside_fraction": 0.0,
      "mean_residual_rad": 0.0,
      "p95_residual_rad": 0.0,
      "max_residual_rad": 0.0
    },
    {
      "joint": "left_thigh_pitch_joint",
      "outside_count": 0,
      "outside_fraction": 0.0,
      "mean_residual_rad": 0.0,
      "p95_residual_rad": 0.0,
      "max_residual_rad": 0.0
    },
    {
      "joint": "right_thigh_pitch_joint",
      "outside_count": 0,
      "outside_fraction": 0.0,
      "mean_residual_rad": 0.0,
      "p95_residual_rad": 0.0,
      "max_residual_rad": 0.0
    },
    {
      "joint": "left_arm_roll_joint",
      "outside_count": 0,
      "outside_fraction": 0.0,
      "mean_residual_rad": 2.331809758657006e-12,
      "p95_residual_rad": 0.0,
      "max_residual_rad": 2.384185793236071e-09
    },
    {
      "joint": "right_arm_roll_joint",
      "outside_count": 0,
      "outside_fraction": 0.0,
      "mean_residual_rad": 0.0,
      "p95_residual_rad": 0.0,
      "max_residual_rad": 0.0
    },
    {
      "joint": "left_knee_joint",
      "outside_count": 0,
      "outside_fraction": 0.0,
      "mean_residual_rad": 1.4050648480325323e-10,
      "p95_residual_rad": 0.0,
      "max_residual_rad": 4.7683715642676816e-08
    },
    {
      "joint": "right_knee_joint",
      "outside_count": 0,
      "outside_fraction": 0.0,
      "mean_residual_rad": 4.6456824975373514e-10,
      "p95_residual_rad": 0.0,
      "max_residual_rad": 4.7683715642676816e-08
    },
    {
      "joint": "left_arm_yaw_joint",
      "outside_count": 0,
      "outside_fraction": 0.0,
      "mean_residual_rad": 1.7165707187010607e-10,
      "p95_residual_rad": 0.0,
      "max_residual_rad": 5.245208734017126e-08
    },
    {
      "joint": "right_arm_yaw_joint",
      "outside_count": 0,
      "outside_fraction": 0.0,
      "mean_residual_rad": 2.828066701308261e-10,
      "p95_residual_rad": 0.0,
      "max_residual_rad": 5.245208734017126e-08
    },
    {
      "joint": "left_ankle_pitch_joint",
      "outside_count": 0,
      "outside_fraction": 0.0,
      "mean_residual_rad": 3.0694688025446352e-09,
      "p95_residual_rad": 3.5762786843029915e-08,
      "max_residual_rad": 3.5762786843029915e-08
    },
    {
      "joint": "right_ankle_pitch_joint",
      "outside_count": 0,
      "outside_fraction": 0.0,
      "mean_residual_rad": 3.162143292824726e-09,
      "p95_residual_rad": 3.5762786843029915e-08,
      "max_residual_rad": 3.5762786843029915e-08
    },
    {
      "joint": "left_elbow_pitch_joint",
      "outside_count": 0,
      "outside_fraction": 0.0,
      "mean_residual_rad": 0.0,
      "p95_residual_rad": 0.0,
      "max_residual_rad": 0.0
    },
    {
      "joint": "right_elbow_pitch_joint",
      "outside_count": 0,
      "outside_fraction": 0.0,
      "mean_residual_rad": 0.0,
      "p95_residual_rad": 0.0,
      "max_residual_rad": 0.0
    },
    {
      "joint": "left_ankle_roll_joint",
      "outside_count": 0,
      "outside_fraction": 0.0,
      "mean_residual_rad": 0.0,
      "p95_residual_rad": 0.0,
      "max_residual_rad": 0.0
    },
    {
      "joint": "right_ankle_roll_joint",
      "outside_count": 0,
      "outside_fraction": 0.0,
      "mean_residual_rad": 0.0,
      "p95_residual_rad": 0.0,
      "max_residual_rad": 0.0
    },
    {
      "joint": "left_elbow_yaw_joint",
      "outside_count": 0,
      "outside_fraction": 0.0,
      "mean_residual_rad": 0.0,
      "p95_residual_rad": 0.0,
      "max_residual_rad": 0.0
    },
    {
      "joint": "right_elbow_yaw_joint",
      "outside_count": 0,
      "outside_fraction": 0.0,
      "mean_residual_rad": 0.0,
      "p95_residual_rad": 0.0,
      "max_residual_rad": 0.0
    }
  ]
}
```


最终存储与预算：
```json
{
  "scope": "G data conversion, qualification16, restore24 and formal524; unique inode accounting",
  "allocated_unique_bytes": 27936763904,
  "unique_inode_bytes": 27936309650,
  "apparent_bytes": 55453878220,
  "roots_apparent_bytes": {
    "/data/kk/models/runs/run_27bcf1e60b4e424fa6b0": 14768394791,
    "/data/kk/models/runs/run_74d708d0e22240588383": 14422257561,
    "/data/kk/models/runs/run_aeb47af29c104e63ba42": 25844503733,
    "/data/kk/models/runs/run_fe16dd1ed5d94564b505": 0,
    "/data/kk/embodiedbein/runs/run_27bcf1e60b4e424fa6b0": 10095044,
    "/data/kk/embodiedbein/runs/run_74d708d0e22240588383": 21069418,
    "/data/kk/embodiedbein/runs/run_aeb47af29c104e63ba42": 306890012,
    "/data/kk/embodiedbein/runs/run_fe16dd1ed5d94564b505": 11962,
    "/data/kk/data/ds_fb0ecda9e70644f2afdd": 80655699
  },
  "additional_work_bytes": 9852020,
  "image_and_staging_reserved_bytes": 12884901888,
  "budget_bytes": 68719476736,
  "within_budget": true,
  "disk_free_bytes": 972810952704
}
```


最终GPU租约与空闲显存：
```json
{
  "hostname": "eai4090x4m1",
  "gpus": [
    {
      "index": 0,
      "name": "NVIDIA GeForce RTX 4090",
      "memory_total_mb": 24564,
      "memory_free_mb": 24080,
      "run_id": null
    },
    {
      "index": 1,
      "name": "NVIDIA GeForce RTX 4090",
      "memory_total_mb": 24564,
      "memory_free_mb": 24080,
      "run_id": null
    },
    {
      "index": 2,
      "name": "NVIDIA GeForce RTX 4090",
      "memory_total_mb": 24564,
      "memory_free_mb": 24080,
      "run_id": null
    },
    {
      "index": 3,
      "name": "NVIDIA GeForce RTX 4090",
      "memory_total_mb": 24564,
      "memory_free_mb": 24080,
      "run_id": null
    }
  ],
  "memory_total_mb": 128644,
  "has_dri": true,
  "memory_available_mb": 103433
}
```


正式任务started_at至finished_at墙钟=2002.36513秒；训练循环末wall_time_s=1064.75232秒，该指标不覆盖完整启动、保存、发布和收集过程。末次指标之后checkpoint保存耗时约12分钟，最终12922255031字节完整产物已原子发布、哈希回读并登记，未取消任务。缺少保存各子阶段计时，不能认定耗时根因。四卡最终租约均为空，剩余显存每卡24080MiB；未采集本段逐卡峰值/利用率完整时序，不能用运行结束时显存替代峰值。

## 五个100更新窗口与观测结论


首/末窗episode日志均值=8.52139382/14.1548762步，硬速度flags每轮均值=275.333333/184.857143。每采样轮固定24576转移，flags/24576是转移越限率，多个关节/原因可重叠，不是episode失败概率。窗口日志点等权，不能冒充所有episode加权均值。

FB loss下降、Actor loss正负、Q符号或价值损失变化均不能独立证明或否定收敛；应结合存活趋势、真实dq越限尾部、原始/裁剪后梯度及完整恢复判断。v1与v2改变目标codec，applied saturation比例不能直接比较为控制质量。D旧均值约−4.5不是数学绝对下界。速度罚贡献只能除以原始加权辅助总奖励，不能与归一化后的mean_aux_reward相比。

F旧5000段仅作背景对照：后程episode缩短、硬速度/位置增加；G此次改变控制语义和稳定性参数，并非单变量因果实验。F保留replay100万占累计7766016转移的12.87%，不是14.3%；F段新增7692288转移，G本段新增786432转移，缓冲刷新时间也不是统计半衰期。

### 剩余动力学风险与下一步判断


全段786432条正式转移中，硬位置10692、硬速度6675、硬力矩0；多个原因可以重叠。最大dq为68.680458rad/s，高于F旧段64.574234rad/s；极值发生在round7、episode第13控制步的right_ankle_roll_joint，额定8rad/s，requested/applied动作均约−0.19165，PD目标约−0.09583rad，同时位置和速度终止。这次极值不是动作饱和，也不是reset首步；不能把它归因于单一饱和动作，更不能宣称速度飞车已经根除。trace只提供控制步观测，尚不能确定接触脉冲、求解器或PD动量各自贡献。硬速度终止实际使用Vmax+0.001rad/s容差，直接统计dq/Vmax>1会把浮点边界误差算进去，两个口径不混用。

正式段66449次reset中963次在首步终止，比例1.44923%；其中硬限位仅62次，其余901次为非硬限位真终止，尚未采集更细终止原因，不能全部命名为fall。G24资格段首步终止2.171%，比F旧最终保留replay的0.27565%仍高；采样覆盖与codec不同，不能作严格因果对照。所有重置目标盒外率0只证明位置可达，不能证明RSI姿态/接触/初速度动态可行。原始参考帧未因本次控制视图派生而过滤，接触力和地面穿透问题尚未闭合。

Actor原始梯度本段最大6.1683且裁剪后约1，六Adam计数524；Critic loss末窗0.5783、末点0.6229，尚未重现F旧长段18量级发散，但末几窗存在回升。D原始奖励末窗−1.3813、末点−1.4951，截断后仍须结合raw/截断占比观察，不能用bounded奖励证明判别器不再过拟合。稳定化只由500更新证据支持。

本轮结论：结束已冻结的500更新观察段，不启动独立评估或追加长训。下一轮首先做训练内的首步终止原因、RSI接触/穿透与踝部极值诊断，区分完整参考初态及PD目标变化的影响；确认后按单因素派生配置继续有限训练。v2动作0对应URDF中点而非DEFAULT_Q，扩大了可请求目标范围，应核验中立姿态、默认增益与实际target变化；不要未经实验就修改关节符号或扩大硬限位。当前replay860160未满100万，也没有足够训练跨度验证F旧2000更新后的退化是否消失。

| 指标 | F旧段末1000窗均值 | G本段首100窗 | G本段末100窗 |
|---|---:|---:|---:|
| env/episode_len | 15.4368712 | 8.52139382 | 14.1548762 |
| train/ufo/hard_velocity_count | 438.761905 | 275.333333 | 184.857143 |
| train/ufo/reset_projection_outside_fraction | 0.815685003 | 0 | 0 |
| train/ufo/fb_loss | 1272.75307 | 100982.622 | 17461.2352 |
| train/ufo/critic_loss | 18.0167173 | 0.789665318 | 0.578277404 |
| train/ufo/actor_loss | 247.649292 | -3.20858538 | -35.0825002 |

## 每采样轮实际观测


| 实际联合更新 | env/episode_len | train/ufo/online_transitions | train/ufo/hard_velocity_count | train/ufo/hard_position_count | train/ufo/terminated_transitions | train/ufo/reset_projection_outside_fraction | train/ufo/applied_action_saturation_fraction | train/ufo/aux_rew/limits_dof_vel |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 40 | 2.42814374 | 98304 | 123 | 68 | 334 | 0 | 0.0478515625 | 0.00352928601 |
| 56 | 6.81145573 | 122880 | 531 | 345 | 1257 | 0 | 0.0807043985 | 0.00702476641 |
| 72 | 9.70162678 | 147456 | 359 | 394 | 2889 | 0 | 0.103335172 | 0.00804542936 |
| 88 | 10.6766081 | 172032 | 233 | 400 | 2146 | 0 | 0.122790352 | 0.00870869681 |
| 104 | 11.0205603 | 196608 | 193 | 392 | 2286 | 0 | 0.141192898 | 0.0106737241 |
| 120 | 10.4899683 | 221184 | 213 | 358 | 2492 | 0 | 0.144308344 | 0.0129393889 |
| 136 | 10.5533848 | 245760 | 201 | 394 | 2201 | 0 | 0.147057578 | 0.0125368414 |
| 152 | 10.619194 | 270336 | 198 | 394 | 2282 | 0 | 0.155015141 | 0.0123288408 |
| 168 | 10.7353468 | 294912 | 196 | 367 | 2218 | 0 | 0.152948797 | 0.0159083381 |
| 184 | 10.8986197 | 319488 | 185 | 329 | 2101 | 0 | 0.159413218 | 0.012696771 |
| 200 | 11.1031408 | 344064 | 204 | 366 | 2133 | 0 | 0.162916094 | 0.0148159983 |
| 216 | 11.2933922 | 368640 | 203 | 363 | 2028 | 0 | 0.165008992 | 0.0133242635 |
| 232 | 11.6115627 | 393216 | 198 | 350 | 2093 | 0 | 0.167268172 | 0.0170099884 |
| 248 | 11.6886969 | 417792 | 206 | 374 | 2088 | 0 | 0.169905946 | 0.0145556992 |
| 264 | 11.8937693 | 442368 | 195 | 321 | 1958 | 0 | 0.170760438 | 0.0161377676 |
| 280 | 11.9775105 | 466944 | 184 | 326 | 1912 | 0 | 0.172683492 | 0.0163097195 |
| 296 | 12.3350677 | 491520 | 207 | 343 | 1922 | 0 | 0.174707383 | 0.0137052108 |
| 312 | 12.3205462 | 516096 | 206 | 327 | 1903 | 0 | 0.175788328 | 0.0142343743 |
| 328 | 12.2991629 | 540672 | 207 | 344 | 1912 | 0 | 0.176451758 | 0.0155695006 |
| 344 | 12.5755472 | 565248 | 212 | 317 | 1873 | 0 | 0.179183289 | 0.0146871014 |
| 360 | 12.9275446 | 589824 | 201 | 308 | 1808 | 0 | 0.179568961 | 0.0139622428 |
| 376 | 12.9633598 | 614400 | 173 | 337 | 1774 | 0 | 0.18094182 | 0.0162831433 |
| 392 | 13.3604088 | 638976 | 191 | 355 | 1859 | 0 | 0.179913953 | 0.0157180391 |
| 408 | 13.3022842 | 663552 | 200 | 321 | 1707 | 0 | 0.181286797 | 0.0167509783 |
| 424 | 13.5908823 | 688128 | 162 | 299 | 1689 | 0 | 0.181500867 | 0.0124679115 |
| 440 | 13.7982197 | 712704 | 181 | 302 | 1685 | 0 | 0.184492469 | 0.013579689 |
| 456 | 13.7975779 | 737280 | 189 | 315 | 1734 | 0 | 0.185992703 | 0.0136124315 |
| 472 | 13.8100662 | 761856 | 197 | 337 | 1669 | 0 | 0.185550407 | 0.0160595737 |
| 488 | 14.4829445 | 786432 | 169 | 314 | 1671 | 0 | 0.187528312 | 0.0150059341 |
| 504 | 14.3531141 | 811008 | 197 | 306 | 1702 | 0 | 0.184653461 | 0.0194316506 |
| 520 | 14.6514454 | 835584 | 177 | 326 | 1730 | 0 | 0.181608781 | 0.0140636154 |
| 524 | 14.1907654 | 860160 | 184 | 300 | 1646 | 0 | 0.185000211 | 0.0127029875 |

同一step对齐；未记录不采用插值或前值补齐。episode日志可能对应完成采样后最近learner更新，实际物理/更新计数口径应结合原始计数，不强行把不同行为频率视为同一时间轴。

## 全部指标统计


n为实际记录点数；统计覆盖新增段>24，若某tag仅含初始配置则沿用原始全部点，first/last step明确展示。最后50点均值不是最后50更新。所有tag完整列出，不使用UI抽样限制。

| tag | n | first_step | last_step | first | last | mean | min | max | median | p95 | last50_mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| env/episode_len | 32 | 40 | 524 | 2.42814374 | 14.1907654 | 11.8206849 | 2.42814374 | 14.6514454 | 12.1383367 | 14.4115378 | 11.8206849 |
| env/num_trajectories | 32 | 40 | 524 | 334 | 1646 | 1896.9375 | 334 | 2889 | 1907.5 | 2378.7 | 1896.9375 |
| env/return | 32 | 40 | 524 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| env/reward | 32 | 40 | 524 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| env/success_once | 32 | 40 | 524 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| rollout/ufo/received_transitions | 32 | 40 | 524 | 24576 | 24576 | 24576 | 24576 | 24576 | 24576 | 24576 | 24576 |
| time/actor_training | 32 | 40 | 524 | 15.6808567 | 3.43380928 | 11.4299252 | 3.43380928 | 15.6808567 | 11.5322547 | 11.6832426 | 11.4299252 |
| time/cal_adv_and_returns | 32 | 40 | 524 | 0.00260877609 | 0.0069873333 | 0.0111200437 | 0.00260877609 | 0.0310020447 | 0.0111268759 | 0.0222138405 | 0.0111200437 |
| time/env/compute_bootstrap_rewards | 32 | 40 | 524 | 0.000250172103 | 0.000211709877 | 0.000224648313 | 0.000200734707 | 0.000326581765 | 0.00021678058 | 0.000250065944 | 0.000224648313 |
| time/env/env/bootstrap_step | 32 | 40 | 524 | 0.0190295503 | 0.000256040134 | 0.000849796634 | 0.000235648127 | 0.0190295503 | 0.000255175051 | 0.000340351299 | 0.000849796634 |
| time/env/env/send_rollout_trajectories | 32 | 40 | 524 | 0.255087942 | 0.213264972 | 0.221176866 | 0.19759059 | 0.255087942 | 0.219406493 | 0.24033325 | 0.221176866 |
| time/env/env_interact_step | 32 | 40 | 524 | 2.56656194 | 2.57939506 | 2.8809403 | 1.84303391 | 3.79075956 | 2.85365117 | 3.37907442 | 2.8809403 |
| time/env/interact | 32 | 40 | 524 | 3.99613714 | 3.5078733 | 3.67242926 | 2.7918334 | 4.4911828 | 3.59093106 | 4.14355576 | 3.67242926 |
| time/env/run_interact_once | 32 | 40 | 524 | 3.99609494 | 3.50785041 | 3.67240198 | 2.79180384 | 4.49115467 | 3.59090054 | 4.14353232 | 3.67240198 |
| time/generate_rollouts | 32 | 40 | 524 | 4.34607124 | 3.67354703 | 3.86179733 | 3.06032276 | 4.66599941 | 3.7955693 | 4.39321783 | 3.86179733 |
| time/rollout/generate_one_epoch | 32 | 40 | 524 | 3.72362947 | 3.2810595 | 3.4391466 | 2.54229975 | 4.26474142 | 3.35595119 | 3.90704165 | 3.4391466 |
| time/rollout/predict | 32 | 40 | 524 | 0.32138288 | 0.0362194888 | 0.048746967 | 0.0352478586 | 0.32138288 | 0.0397918634 | 0.048835774 | 0.048746967 |
| time/rollout/rollout/generate | 32 | 40 | 524 | 3.72832155 | 3.28185797 | 3.43993502 | 2.54320002 | 4.2655077 | 3.35667944 | 3.90797369 | 3.43993502 |
| time/rollout/sync_model_from_actor | 32 | 40 | 524 | 14.1754265 | 12.913825 | 13.0726173 | 12.6864653 | 14.1754265 | 13.0150471 | 13.3555983 | 13.0726173 |
| time/step | 32 | 40 | 524 | 37.8462257 | 23.7876396 | 32.125011 | 23.7876396 | 37.8462257 | 32.1557903 | 32.8961258 | 32.125011 |
| time/sync_weights | 32 | 40 | 524 | 17.8159561 | 16.6727905 | 16.8216507 | 16.4595108 | 17.8159561 | 16.8014784 | 17.0828323 | 16.8216507 |
| train/ufo/B | 500 | 25 | 524 | -0.00536779314 | 0.000509028963 | -0.00343889839 | -0.0112563614 | 0.00216611009 | -0.00311399472 | 0.000181381484 | -0.000555474288 |
| train/ufo/B_norm | 500 | 25 | 524 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 |
| train/ufo/F1 | 500 | 25 | 524 | 0.0713853166 | 0.0245060455 | 0.0299434456 | 0.00817772932 | 0.0727545172 | 0.0295808855 | 0.0418021427 | 0.0313565566 |
| train/ufo/M1 | 500 | 25 | 524 | 4.46553993 | 0.840999007 | 1.12446082 | 0.490111262 | 4.7163949 | 0.89380163 | 2.29536479 | 0.813526698 |
| train/ufo/Q1 | 500 | 25 | 524 | -3.76895285 | -4.66551781 | -3.23395538 | -4.84632778 | -1.71410942 | -3.15192699 | -2.22791629 | -4.53325212 |
| train/ufo/Q_aux | 500 | 25 | 524 | -5.23387527 | -5.38964462 | -4.26009158 | -5.41836929 | -2.96642447 | -4.21657372 | -3.44141127 | -5.20220433 |
| train/ufo/Q_discriminator | 500 | 25 | 524 | -3.92493725 | -4.98353577 | -3.33274493 | -4.98353577 | -1.84981263 | -3.21378303 | -2.30342567 | -4.62778743 |
| train/ufo/Q_fb | 500 | 25 | 524 | -3.03376341 | 58.8232536 | 29.8153045 | -3.03376341 | 58.8232536 | 30.3525944 | 53.3481014 | 53.7206481 |
| train/ufo/action_abs_mean | 32 | 40 | 524 | 0.887031436 | 0.899126291 | 0.903998733 | 0.887031436 | 0.913123012 | 0.90488559 | 0.910914549 | 0.903998733 |
| train/ufo/action_saturation_fraction | 32 | 40 | 524 | 0.406218141 | 0.436794102 | 0.441935995 | 0.406218141 | 0.455221415 | 0.443961769 | 0.452004614 | 0.441935995 |
| train/ufo/actor_loss | 500 | 25 | 524 | 5.91742229 | -40.9747696 | -20.5946053 | -41.5045776 | 5.91742229 | -21.9552574 | 0.0861400962 | -36.9489752 |
| train/ufo/actor_optimizer/grad_clip_limit | 500 | 25 | 524 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| train/ufo/actor_optimizer/grad_norm_after_clip_l2 | 500 | 25 | 524 | 0.999999344 | 0.999999821 | 0.999999746 | 0.999999344 | 1 | 0.999999762 | 0.999999881 | 0.999999831 |
| train/ufo/actor_optimizer/grad_norm_l2 | 500 | 25 | 524 | 1.45490348 | 6.16829205 | 4.3270091 | 1.45490348 | 6.16829205 | 4.44115019 | 5.80282581 | 5.74591098 |
| train/ufo/actor_optimizer/lr | 500 | 25 | 524 | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 |
| train/ufo/actor_optimizer/parameter_delta_l2 | 500 | 25 | 524 | 0.43285352 | 0.350665331 | 0.341938183 | 0.316799134 | 0.442046493 | 0.337437704 | 0.3672617 | 0.34607677 |
| train/ufo/actor_reg_weight_applied | 500 | 25 | 524 | 9.58267021 | 50 | 35.0915074 | 9.58267021 | 50 | 37.2646294 | 50 | 50 |
| train/ufo/actor_reg_weight_raw | 500 | 25 | 524 | 9.58267021 | 65.8285522 | 36.9459535 | 9.58267021 | 65.8285522 | 37.2646294 | 60.8895479 | 61.1602344 |
| train/ufo/applied_action_abs_mean | 32 | 40 | 524 | 0.44774732 | 0.61529541 | 0.589281554 | 0.44774732 | 0.617911279 | 0.601367116 | 0.614850256 | 0.589281554 |
| train/ufo/applied_action_saturation_fraction | 32 | 40 | 524 | 0.0478515625 | 0.185000211 | 0.161479065 | 0.0478515625 | 0.187528312 | 0.173695438 | 0.18574944 | 0.161479065 |
| train/ufo/auxQ1 | 500 | 25 | 524 | -4.2377491 | -5.14971018 | -4.18992406 | -5.3606472 | -2.97140622 | -4.18642211 | -3.31695105 | -5.1213992 |
| train/ufo/aux_critic_loss | 500 | 25 | 524 | 2.17887115 | 0.852182031 | 1.08841419 | 0.457870066 | 52.518055 | 0.792859703 | 1.81183991 | 1.08087669 |
| train/ufo/aux_critic_optimizer/grad_clip_limit | 500 | 25 | 524 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 |
| train/ufo/aux_critic_optimizer/grad_norm_after_clip_l2 | 500 | 25 | 524 | 4.99999952 | 4.99999857 | 4.78350563 | 0.924871564 | 5.00000048 | 4.99999952 | 5 | 4.65088829 |
| train/ufo/aux_critic_optimizer/grad_norm_l2 | 500 | 25 | 524 | 20.5641422 | 5.35753155 | 10.8198032 | 0.924871564 | 41.5266838 | 9.75809002 | 21.6242082 | 6.15602203 |
| train/ufo/aux_critic_optimizer/lr | 500 | 25 | 524 | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 |
| train/ufo/aux_critic_optimizer/parameter_delta_l2 | 500 | 25 | 524 | 0.40913853 | 0.611987531 | 0.398769317 | 0.168088183 | 1.1297307 | 0.371586874 | 0.629568577 | 0.585394997 |
| train/ufo/aux_rew/limits_dof_pos | 500 | 25 | 524 | 0.0351371132 | 0.231370986 | 0.170926894 | 0.033974193 | 0.251702309 | 0.187830456 | 0.240299471 | 0.23878876 |
| train/ufo/aux_rew/limits_dof_vel | 500 | 25 | 524 | 0.00420617778 | 0.0127029875 | 0.0137175999 | 0.00233350764 | 0.0475441702 | 0.0142148174 | 0.0183859184 | 0.0146298764 |
| train/ufo/aux_rew/limits_torque | 500 | 25 | 524 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| train/ufo/aux_rew/penalty_action_rate | 500 | 25 | 524 | 0.718162715 | 0.428738952 | 0.503918341 | 0.412620187 | 0.723014832 | 0.476969123 | 0.682245886 | 0.422901879 |
| train/ufo/aux_rew/penalty_ankle_roll | 500 | 25 | 524 | 0.0356984064 | 0.161565483 | 0.124275597 | 0.0351185463 | 0.170612484 | 0.134129375 | 0.164429774 | 0.16396687 |
| train/ufo/aux_rew/penalty_feet_ori | 500 | 25 | 524 | 0.0631782189 | 0.140883505 | 0.112547657 | 0.0631782189 | 0.152217478 | 0.112039991 | 0.140075097 | 0.138117052 |
| train/ufo/aux_rew/penalty_slippage | 500 | 25 | 524 | 0.263244361 | 0.287178129 | 0.293388313 | 0.236641884 | 0.344851524 | 0.292669982 | 0.323797923 | 0.280637659 |
| train/ufo/aux_rew/penalty_torques | 500 | 25 | 524 | 3727.05103 | 4673.51709 | 4902.8194 | 3590.46558 | 5357.91309 | 4948.31128 | 5227.12007 | 4682.90531 |
| train/ufo/aux_rew/penalty_undesired_contact | 500 | 25 | 524 | 0.0048828125 | 0.0068359375 | 0.00651660156 | 0.0009765625 | 0.01171875 | 0.00634765625 | 0.0102539062 | 0.00719726563 |
| train/ufo/backward_optimizer/grad_clip_limit | 500 | 25 | 524 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 |
| train/ufo/backward_optimizer/grad_norm_after_clip_l2 | 500 | 25 | 524 | 5.00000095 | 5.00000048 | 4.99999999 | 4.99999905 | 5.00000095 | 5 | 5.00000048 | 5.00000001 |
| train/ufo/backward_optimizer/grad_norm_l2 | 500 | 25 | 524 | 260877.953 | 15489.3145 | 54398.216 | 15099.2803 | 273938.562 | 33665.4297 | 160342.381 | 16157.7673 |
| train/ufo/backward_optimizer/lr | 500 | 25 | 524 | 9.99999975e-06 | 9.99999975e-06 | 9.99999975e-06 | 9.99999975e-06 | 9.99999975e-06 | 9.99999975e-06 | 9.99999975e-06 | 9.99999975e-06 |
| train/ufo/backward_optimizer/parameter_delta_l2 | 500 | 25 | 524 | 0.00367757352 | 0.00344471307 | 0.00400520319 | 0.00344471307 | 0.00440436229 | 0.00403679186 | 0.00438661531 | 0.00353580023 |
| train/ufo/collected_transitions | 32 | 40 | 524 | 24576 | 24576 | 24576 | 24576 | 24576 | 24576 | 24576 | 24576 |
| train/ufo/critic_loss | 500 | 25 | 524 | 1.52833819 | 0.622903824 | 0.49423647 | 0.227421179 | 1.52833819 | 0.422356695 | 0.995204481 | 0.611498291 |
| train/ufo/critic_optimizer/grad_clip_limit | 500 | 25 | 524 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 |
| train/ufo/critic_optimizer/grad_norm_after_clip_l2 | 500 | 25 | 524 | 4.99999905 | 4.99999905 | 4.7154531 | 1.03483272 | 5.00000048 | 4.99999905 | 5 | 4.99999931 |
| train/ufo/critic_optimizer/grad_norm_l2 | 500 | 25 | 524 | 16.868763 | 6.69641161 | 8.99283221 | 1.03483272 | 30.2362995 | 8.52976036 | 16.8720771 | 8.62775811 |
| train/ufo/critic_optimizer/lr | 500 | 25 | 524 | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 |
| train/ufo/critic_optimizer/parameter_delta_l2 | 500 | 25 | 524 | 0.379958093 | 0.440555066 | 0.348255217 | 0.212814242 | 0.546914101 | 0.347822979 | 0.454609744 | 0.410235109 |
| train/ufo/disc_expert_loss | 500 | 25 | 524 | 0.556768537 | 0.248346537 | 0.301970092 | 0.225142717 | 0.556768537 | 0.271624029 | 0.477539442 | 0.23879415 |
| train/ufo/disc_loss | 500 | 25 | 524 | 3.27214098 | 0.806309581 | 1.34589816 | 0.765356541 | 3.27214098 | 1.03786755 | 2.81550785 | 0.798033406 |
| train/ufo/disc_reward_clip_fraction | 500 | 25 | 524 | 0 | 0.382324219 | 0.130044922 | 0 | 0.406738281 | 0.0729980469 | 0.371118164 | 0.370625 |
| train/ufo/disc_train_loss | 500 | 25 | 524 | 0.690336883 | 0.329250991 | 0.504742447 | 0.314035118 | 0.690336883 | 0.495039761 | 0.663521075 | 0.331009958 |
| train/ufo/disc_wgan_gp_loss | 500 | 25 | 524 | 0.202503562 | 0.0228712112 | 0.0539185623 | 0.0212370716 | 0.203451797 | 0.0274015423 | 0.168261891 | 0.0228229293 |
| train/ufo/discriminator_optimizer/grad_clip_limit | 500 | 25 | 524 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 |
| train/ufo/discriminator_optimizer/grad_norm_after_clip_l2 | 500 | 25 | 524 | 4.99999905 | 0.929819107 | 2.67943372 | 0.906871319 | 5 | 1.84525406 | 4.99999952 | 1.00777851 |
| train/ufo/discriminator_optimizer/grad_norm_l2 | 500 | 25 | 524 | 8.52457333 | 0.929819107 | 3.27403272 | 0.906871319 | 8.54720592 | 1.84525406 | 8.20184908 | 1.00777851 |
| train/ufo/discriminator_optimizer/lr | 500 | 25 | 524 | 3.00000011e-06 | 3.00000011e-06 | 3.00000011e-06 | 3.00000011e-06 | 3.00000011e-06 | 3.00000011e-06 | 3.00000011e-06 | 3.00000011e-06 |
| train/ufo/discriminator_optimizer/parameter_delta_l2 | 500 | 25 | 524 | 0.00456860708 | 0.0023692823 | 0.00356148284 | 0.00236654747 | 0.00526137697 | 0.00302151486 | 0.00521608726 | 0.00241562165 |
| train/ufo/environment_control_steps | 32 | 40 | 524 | 16 | 140 | 78 | 16 | 140 | 78 | 133.8 | 78 |
| train/ufo/expert_draws/g1 | 32 | 40 | 524 | 41104 | 536088 | 294789.25 | 41104 | 536088 | 295080 | 523470.4 | 294789.25 |
| train/ufo/expert_draws/official_rpo | 32 | 40 | 524 | 40816 | 537064 | 294266.75 | 40816 | 537064 | 294744 | 523467.2 | 294266.75 |
| train/ufo/fb_diag | 500 | 25 | 524 | -20.4293823 | -216.459015 | -161.767816 | -223.496109 | -17.5648918 | -183.680183 | -42.8892218 | -215.478483 |
| train/ufo/fb_loss | 500 | 25 | 524 | 175292.641 | 15657.6055 | 44489.8849 | 15341.0303 | 175292.641 | 31644.3584 | 119646.694 | 16316.4188 |
| train/ufo/fb_offdiag | 500 | 25 | 524 | 185.263168 | 227.596085 | 166.857354 | 107.995239 | 235.169861 | 166.965103 | 223.894843 | 221.941949 |
| train/ufo/forward_optimizer/grad_clip_limit | 500 | 25 | 524 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 |
| train/ufo/forward_optimizer/grad_norm_after_clip_l2 | 500 | 25 | 524 | 5.00000048 | 5 | 5 | 4.99999905 | 5.00000095 | 5 | 5.00000048 | 4.99999997 |
| train/ufo/forward_optimizer/grad_norm_l2 | 500 | 25 | 524 | 947.790039 | 109.022301 | 211.563771 | 100.746368 | 1067.24219 | 138.8535 | 567.512619 | 113.558086 |
| train/ufo/forward_optimizer/lr | 500 | 25 | 524 | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 |
| train/ufo/forward_optimizer/parameter_delta_l2 | 500 | 25 | 524 | 0.387180477 | 0.763620019 | 0.656739567 | 0.308053344 | 0.810535371 | 0.719819129 | 0.77635726 | 0.756114484 |
| train/ufo/hard_position_count | 32 | 40 | 524 | 68 | 300 | 334.125 | 68 | 400 | 337 | 394 | 334.125 |
| train/ufo/hard_torque_count | 32 | 40 | 524 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| train/ufo/hard_velocity_count | 32 | 40 | 524 | 123 | 184 | 208.59375 | 123 | 531 | 197.5 | 289.7 | 208.59375 |
| train/ufo/learner_joint_updates | 500 | 25 | 524 | 25 | 524 | 274.5 | 25 | 524 | 274.5 | 499.05 | 499.5 |
| train/ufo/mean_aux_reward | 500 | 25 | 524 | -0.676600039 | -0.987104356 | -0.864634553 | -1.09904861 | -0.404771388 | -0.918225199 | -0.518178049 | -1.01398702 |
| train/ufo/mean_disc_reward | 500 | 25 | 524 | 0.104753725 | -1.16069746 | -0.540751972 | -1.19801629 | 0.120927013 | -0.610988677 | 0.0926075861 | -1.14863062 |
| train/ufo/mean_disc_reward_raw | 500 | 25 | 524 | 0.104753725 | -1.4950664 | -0.625306805 | -1.55896068 | 0.120927013 | -0.63325429 | 0.0926075861 | -1.45964464 |
| train/ufo/mean_next_Q | 500 | 25 | 524 | -2.82458067 | -3.79060984 | -2.60430441 | -3.79060984 | -1.92763889 | -2.51253474 | -1.9945991 | -3.5837151 |
| train/ufo/mean_next_auxQ | 500 | 25 | 524 | -3.37073374 | -4.49886703 | -3.40541498 | -4.51237297 | -2.74129105 | -3.25974727 | -2.789177 | -4.38192238 |
| train/ufo/online_transitions | 500 | 25 | 524 | 98304 | 860160 | 470089.728 | 98304 | 860160 | 466944 | 811008 | 815923.2 |
| train/ufo/orth_loss | 500 | 25 | 524 | 1751.27808 | 156.464691 | 444.847952 | 153.331024 | 1751.27808 | 316.616577 | 1195.71948 | 163.099555 |
| train/ufo/orth_loss_diag | 500 | 25 | 524 | -256 | -256 | -256 | -256 | -256 | -256 | -256 | -256 |
| train/ufo/orth_loss_offdiag | 500 | 25 | 524 | 2007.27808 | 412.464691 | 700.847952 | 409.331024 | 2007.27808 | 572.616577 | 1451.71948 | 419.099555 |
| train/ufo/q_loss | 500 | 25 | 524 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| train/ufo/replay_draw_new_ratio | 32 | 40 | 524 | 0.833333313 | 1.24761903 | 1.18784997 | 0.833333313 | 1.27450979 | 1.23070174 | 1.27168562 | 1.18784997 |
| train/ufo/replay_draws | 32 | 40 | 524 | 81920 | 1073152 | 589056 | 81920 | 1073152 | 589824 | 1046937.6 | 589056 |
| train/ufo/replay_policy_age_max_updates | 32 | 40 | 524 | 40 | 524 | 287.625 | 40 | 524 | 288 | 511.2 | 287.625 |
| train/ufo/replay_policy_age_min_updates | 32 | 40 | 524 | 16 | 4 | 15.625 | 4 | 16 | 16 | 16 | 15.625 |
| train/ufo/replay_retained_fraction | 32 | 40 | 524 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| train/ufo/replay_size | 32 | 40 | 524 | 98304 | 860160 | 479232 | 98304 | 860160 | 479232 | 822067.2 | 479232 |
| train/ufo/requested_action_abs_mean | 32 | 40 | 524 | 0.887031257 | 0.89912641 | 0.903998742 | 0.887031257 | 0.913122952 | 0.90488565 | 0.910914588 | 0.903998742 |
| train/ufo/requested_action_saturation_fraction | 32 | 40 | 524 | 0.406218141 | 0.436794102 | 0.441935995 | 0.406218141 | 0.455221415 | 0.443961769 | 0.452004614 | 0.441935995 |
| train/ufo/requested_applied_max_diff | 32 | 40 | 524 | 1.95280123 | 1.95280123 | 1.95822231 | 1.95280027 | 1.96602774 | 1.95411462 | 1.96602774 | 1.95822231 |
| train/ufo/reset_control_samples | 32 | 40 | 524 | 6388 | 1704 | 2076.53125 | 803 | 6388 | 1917 | 2592.4 | 2076.53125 |
| train/ufo/reset_projection_outside_fraction | 32 | 40 | 524 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| train/ufo/reset_projection_residual_max_rad | 32 | 40 | 524 | 2.38418579e-07 | 2.38418579e-07 | 2.38418579e-07 | 2.38418579e-07 | 2.38418579e-07 | 2.38418579e-07 | 2.38418579e-07 | 2.38418579e-07 |
| train/ufo/reset_projection_residual_mean_rad | 32 | 40 | 524 | 6.70394087e-08 | 6.63941364e-08 | 6.6689904e-08 | 6.52485141e-08 | 6.92391353e-08 | 6.65982789e-08 | 6.77169744e-08 | 6.6689904e-08 |
| train/ufo/reset_projection_residual_p95_rad | 32 | 40 | 524 | 1.1920929e-07 | 1.1920929e-07 | 1.1920929e-07 | 1.1920929e-07 | 1.1920929e-07 | 1.1920929e-07 | 1.1920929e-07 | 1.1920929e-07 |
| train/ufo/sampling_rounds | 500 | 25 | 524 | 4 | 35 | 19.128 | 4 | 35 | 19 | 33 | 33.2 |
| train/ufo/target_M | 500 | 25 | 524 | 2.20443583 | 0.643817782 | 0.729085259 | 0.395201951 | 2.30787086 | 0.639405668 | 1.40202573 | 0.631536301 |
| train/ufo/target_Q | 500 | 25 | 524 | -3.73159695 | -4.76942825 | -3.24139078 | -4.76942825 | -2.19079733 | -3.16702271 | -2.27770025 | -4.53615292 |
| train/ufo/target_auxQ | 500 | 25 | 524 | -4.53296661 | -5.23501396 | -4.19810805 | -5.27130222 | -3.33921862 | -4.15843248 | -3.40218457 | -5.11938587 |
| train/ufo/terminated_transitions | 32 | 40 | 524 | 334 | 1646 | 1896.9375 | 334 | 2889 | 1907.5 | 2378.7 | 1896.9375 |
| train/ufo/truncated_transitions | 32 | 40 | 524 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| train/ufo/unc_Q | 500 | 25 | 524 | 2.2868371 | 0.154340506 | 0.524141836 | 0.148449719 | 2.40854955 | 0.291311964 | 1.94474733 | 0.155636817 |
| train/ufo/unc_auxQ | 500 | 25 | 524 | 1.24145687 | 0.128524587 | 0.354131856 | 0.128439978 | 1.26004481 | 0.244346641 | 0.996107945 | 0.133370453 |
| train/ufo/wall_time_s | 500 | 25 | 524 | 61.3879623 | 1064.75232 | 558.060941 | 61.3879623 | 1064.75232 | 557.155212 | 1005.63762 | 1010.18169 |
| train/ufo/z_norm | 500 | 25 | 524 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 |

## 全部指标五窗均值


| tag | 25–124 | 125–224 | 225–324 | 325–424 | 425–524 |
|---|---:|---:|---:|---:|---:|
| env/episode_len | 8.52139382 | 10.8671797 | 11.9711922 | 13.0027414 | 14.1548762 |
| env/num_trajectories | 1900.66667 | 2160.5 | 1979.33333 | 1803.14286 | 1691 |
| env/return | 0 | 0 | 0 | 0 | 0 |
| env/reward | 0 | 0 | 0 | 0 | 0 |
| env/success_once | 0 | 0 | 0 | 0 | 0 |
| rollout/ufo/received_transitions | 24576 | 24576 | 24576 | 24576 | 24576 |
| time/actor_training | 12.2547949 | 11.5381913 | 11.5474537 | 11.54341 | 10.4158711 |
| time/cal_adv_and_returns | 0.00917903582 | 0.00894316037 | 0.0131765207 | 0.0138845784 | 0.0101224354 |
| time/env/compute_bootstrap_rewards | 0.000228101155 | 0.00022016031 | 0.000215976266 | 0.000242715336 | 0.000214901753 |
| time/env/env/bootstrap_step | 0.00338678178 | 0.00026171084 | 0.000254786224 | 0.000279906871 | 0.000259210156 |
| time/env/env/send_rollout_trajectories | 0.223738876 | 0.212821707 | 0.215272648 | 0.228734991 | 0.223645057 |
| time/env/env_interact_step | 3.00179682 | 3.11335496 | 2.90134172 | 2.84590721 | 2.59568259 |
| time/env/interact | 3.87942545 | 3.88323991 | 3.66318397 | 3.59746334 | 3.3971996 |
| time/env/run_interact_once | 3.87939608 | 3.88321161 | 3.66315929 | 3.59743459 | 3.39717419 |
| time/generate_rollouts | 4.09830956 | 4.05913035 | 3.86196331 | 3.77456471 | 3.57702034 |
| time/rollout/generate_one_epoch | 3.64054592 | 3.65836696 | 3.4274689 | 3.36352617 | 3.16424533 |
| time/rollout/predict | 0.0865561298 | 0.0409371238 | 0.0396656202 | 0.0406935268 | 0.0388707162 |
| time/rollout/rollout/generate | 3.64204979 | 3.65905333 | 3.42804305 | 3.36421221 | 3.16479404 |
| time/rollout/sync_model_from_actor | 13.3448032 | 13.1638306 | 13.031494 | 12.8884694 | 12.9805288 |
| time/step | 33.3767732 | 32.5260537 | 32.2291517 | 31.9810535 | 30.7630152 |
| time/sync_weights | 17.0139364 | 16.9192591 | 16.8060948 | 16.6486803 | 16.7594741 |
| train/ufo/B | -0.00701965403 | -0.00450480175 | -0.00308460238 | -0.00173538571 | -0.000850048094 |
| train/ufo/B_norm | 16 | 16 | 16 | 16 | 16 |
| train/ufo/F1 | 0.0397107732 | 0.0301742975 | 0.0257500082 | 0.0251775453 | 0.0289046036 |
| train/ufo/M1 | 2.06134431 | 1.09850543 | 0.813631999 | 0.830073285 | 0.818749054 |
| train/ufo/Q1 | -3.09723112 | -2.32472101 | -2.82687848 | -3.579249 | -4.34169729 |
| train/ufo/Q_aux | -4.14528907 | -3.58262549 | -3.95482801 | -4.52876471 | -5.08895059 |
| train/ufo/Q_discriminator | -3.25543213 | -2.40742985 | -2.89945201 | -3.66565663 | -4.43575401 |
| train/ufo/Q_fb | 6.64586781 | 19.5596317 | 30.457686 | 41.1525012 | 51.2608357 |
| train/ufo/action_abs_mean | 0.904628962 | 0.908459475 | 0.90621985 | 0.902332979 | 0.899396982 |
| train/ufo/action_saturation_fraction | 0.439501479 | 0.447695225 | 0.44524468 | 0.441124707 | 0.43706151 |
| train/ufo/actor_loss | -3.20858538 | -14.5295256 | -22.0560467 | -28.0963688 | -35.0825002 |
| train/ufo/actor_optimizer/grad_clip_limit | 1 | 1 | 1 | 1 | 1 |
| train/ufo/actor_optimizer/grad_norm_after_clip_l2 | 0.99999959 | 0.999999752 | 0.999999764 | 0.999999805 | 0.999999821 |
| train/ufo/actor_optimizer/grad_norm_l2 | 2.57513551 | 4.00689179 | 4.42856317 | 5.01140789 | 5.61304715 |
| train/ufo/actor_optimizer/lr | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 |
| train/ufo/actor_optimizer/parameter_delta_l2 | 0.369856555 | 0.332328745 | 0.329523186 | 0.335669156 | 0.342313274 |
| train/ufo/actor_reg_weight_applied | 14.4653719 | 26.1524201 | 37.2996577 | 47.5400874 | 50 |
| train/ufo/actor_reg_weight_raw | 14.4653719 | 26.1524201 | 37.2996577 | 48.2188991 | 58.5934184 |
| train/ufo/applied_action_abs_mean | 0.531528915 | 0.588781655 | 0.599920511 | 0.60560309 | 0.61377166 |
| train/ufo/applied_action_saturation_fraction | 0.106697121 | 0.15705997 | 0.171852293 | 0.179835349 | 0.184975192 |
| train/ufo/auxQ1 | -4.03907269 | -3.51989282 | -3.94090863 | -4.44200516 | -5.00774099 |
| train/ufo/aux_critic_loss | 1.81875333 | 0.700372398 | 0.772702276 | 1.04003568 | 1.11020728 |
| train/ufo/aux_critic_optimizer/grad_clip_limit | 5 | 5 | 5 | 5 | 5 |
| train/ufo/aux_critic_optimizer/grad_norm_after_clip_l2 | 4.96956595 | 4.8323965 | 4.67593369 | 4.83257444 | 4.60705754 |
| train/ufo/aux_critic_optimizer/grad_norm_l2 | 15.4212621 | 10.8265887 | 11.5252972 | 9.98322432 | 6.34264384 |
| train/ufo/aux_critic_optimizer/lr | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 |
| train/ufo/aux_critic_optimizer/parameter_delta_l2 | 0.292218556 | 0.303981573 | 0.350692279 | 0.481263794 | 0.565690381 |
| train/ufo/aux_rew/limits_dof_pos | 0.0737803058 | 0.147415844 | 0.186460176 | 0.213128119 | 0.233850025 |
| train/ufo/aux_rew/limits_dof_vel | 0.00899564836 | 0.0143072883 | 0.0154550156 | 0.0150360676 | 0.0147939794 |
| train/ufo/aux_rew/limits_torque | 0 | 0 | 0 | 0 | 0 |
| train/ufo/aux_rew/penalty_action_rate | 0.637722351 | 0.52879133 | 0.478497798 | 0.447763461 | 0.426816762 |
| train/ufo/aux_rew/penalty_ankle_roll | 0.0669454329 | 0.109474842 | 0.133390367 | 0.150115038 | 0.161452304 |
| train/ufo/aux_rew/penalty_feet_ori | 0.0890479876 | 0.102541313 | 0.11130238 | 0.124177567 | 0.135669037 |
| train/ufo/aux_rew/penalty_slippage | 0.301463077 | 0.304845734 | 0.291931237 | 0.286198625 | 0.282502891 |
| train/ufo/aux_rew/penalty_torques | 4639.0429 | 5116.03449 | 5082.24403 | 4913.73396 | 4763.04161 |
| train/ufo/aux_rew/penalty_undesired_contact | 0.00429199219 | 0.00637207031 | 0.00737792969 | 0.00728515625 | 0.00725585937 |
| train/ufo/backward_optimizer/grad_clip_limit | 5 | 5 | 5 | 5 | 5 |
| train/ufo/backward_optimizer/grad_norm_after_clip_l2 | 4.99999998 | 5 | 4.99999997 | 5.00000003 | 4.99999998 |
| train/ufo/backward_optimizer/grad_norm_l2 | 137238.109 | 60964.1259 | 33959.1329 | 22653.2722 | 17176.4404 |
| train/ufo/backward_optimizer/lr | 9.99999975e-06 | 9.99999975e-06 | 9.99999975e-06 | 9.99999975e-06 | 9.99999975e-06 |
| train/ufo/backward_optimizer/parameter_delta_l2 | 0.00404553866 | 0.00432963389 | 0.00413824331 | 0.00390788764 | 0.00360471244 |
| train/ufo/collected_transitions | 24576 | 24576 | 24576 | 24576 | 24576 |
| train/ufo/critic_loss | 0.789665318 | 0.355886813 | 0.312660514 | 0.434692299 | 0.578277404 |
| train/ufo/critic_optimizer/grad_clip_limit | 5 | 5 | 5 | 5 | 5 |
| train/ufo/critic_optimizer/grad_norm_after_clip_l2 | 4.77964332 | 4.7552055 | 4.74353897 | 4.35535762 | 4.9435201 |
| train/ufo/critic_optimizer/grad_norm_l2 | 9.9849408 | 9.37265122 | 8.10789278 | 8.49228733 | 9.00638892 |
| train/ufo/critic_optimizer/lr | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 |
| train/ufo/critic_optimizer/parameter_delta_l2 | 0.346155548 | 0.288570567 | 0.308574267 | 0.393236505 | 0.404739197 |
| train/ufo/disc_expert_loss | 0.432433607 | 0.303600048 | 0.268513559 | 0.261409382 | 0.243893863 |
| train/ufo/disc_loss | 2.51341332 | 1.44601447 | 1.04050848 | 0.912144431 | 0.81741011 |
| train/ufo/disc_reward_clip_fraction | 0 | 0.00395019531 | 0.0757080078 | 0.22609375 | 0.344472656 |
| train/ufo/disc_train_loss | 0.657954403 | 0.620308247 | 0.497071065 | 0.405401956 | 0.342976563 |
| train/ufo/disc_wgan_gp_loss | 0.142302531 | 0.0522106174 | 0.0274923858 | 0.0245333099 | 0.023053968 |
| train/ufo/discriminator_optimizer/grad_clip_limit | 5 | 5 | 5 | 5 | 5 |
| train/ufo/discriminator_optimizer/grad_norm_after_clip_l2 | 4.9999993 | 4.14955808 | 1.8791692 | 1.3137894 | 1.05465262 |
| train/ufo/discriminator_optimizer/grad_norm_l2 | 7.66619668 | 4.45635571 | 1.8791692 | 1.3137894 | 1.05465262 |
| train/ufo/discriminator_optimizer/lr | 3.00000011e-06 | 3.00000011e-06 | 3.00000011e-06 | 3.00000011e-06 | 3.00000011e-06 |
| train/ufo/discriminator_optimizer/parameter_delta_l2 | 0.00490244511 | 0.00474415337 | 0.00307639283 | 0.00263961432 | 0.00244480856 |
| train/ufo/environment_control_steps | 26 | 50 | 74 | 100 | 128 |
| train/ufo/expert_draws/g1 | 82226.6667 | 181020 | 278861.333 | 385026.286 | 497917.714 |
| train/ufo/expert_draws/official_rpo | 81613.3333 | 179428 | 278194.667 | 385021.714 | 497995.429 |
| train/ufo/fb_diag | -66.2398717 | -143.787377 | -183.663784 | -202.131266 | -213.016782 |
| train/ufo/fb_loss | 100982.622 | 49271.4579 | 31775.3533 | 22958.7562 | 17461.2352 |
| train/ufo/fb_offdiag | 123.193235 | 137.120791 | 166.15059 | 191.451982 | 216.370172 |
| train/ufo/forward_optimizer/grad_clip_limit | 5 | 5 | 5 | 5 | 5 |
| train/ufo/forward_optimizer/grad_norm_after_clip_l2 | 5.00000003 | 4.99999997 | 5.00000001 | 5 | 4.99999999 |
| train/ufo/forward_optimizer/grad_norm_l2 | 477.393489 | 208.245232 | 138.600738 | 119.685798 | 113.893599 |
| train/ufo/forward_optimizer/lr | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 |
| train/ufo/forward_optimizer/parameter_delta_l2 | 0.423559441 | 0.627177202 | 0.722632699 | 0.75138315 | 0.758945341 |
| train/ufo/hard_position_count | 326.166667 | 368.833333 | 340.166667 | 325.857143 | 314.285714 |
| train/ufo/hard_torque_count | 0 | 0 | 0 | 0 | 0 |
| train/ufo/hard_velocity_count | 275.333333 | 197.833333 | 199.333333 | 192.285714 | 184.857143 |
| train/ufo/learner_joint_updates | 74.5 | 174.5 | 274.5 | 374.5 | 474.5 |
| train/ufo/mean_aux_reward | -0.781700698 | -0.680872195 | -0.894392023 | -0.945959334 | -1.02024852 |
| train/ufo/mean_disc_reward | 0.0734647047 | -0.147420337 | -0.59645531 | -0.92267749 | -1.11067143 |
| train/ufo/mean_disc_reward_raw | 0.0734647047 | -0.147756894 | -0.621578231 | -1.04933391 | -1.3813297 |
| train/ufo/mean_next_Q | -2.59460974 | -2.0288884 | -2.22135044 | -2.76206505 | -3.41460842 |
| train/ufo/mean_next_auxQ | -3.03405798 | -2.83585011 | -3.20052993 | -3.70432717 | -4.25230973 |
| train/ufo/online_transitions | 163184.64 | 316538.88 | 469893.12 | 623247.36 | 777584.64 |
| train/ufo/orth_loss | 1009.25668 | 492.781245 | 317.928665 | 229.694354 | 174.578819 |
| train/ufo/orth_loss_diag | -256 | -256 | -256 | -256 | -256 |
| train/ufo/orth_loss_offdiag | 1265.25668 | 748.781245 | 573.928665 | 485.694354 | 430.578819 |
| train/ufo/q_loss | 0 | 0 | 0 | 0 | 0 |
| train/ufo/replay_draw_new_ratio | 1.00145505 | 1.17024642 | 1.22429011 | 1.2528155 | 1.26650587 |
| train/ufo/replay_draws | 163840 | 360448 | 557056 | 770048 | 995913.143 |
| train/ufo/replay_policy_age_max_updates | 80 | 176 | 272 | 376 | 486.285714 |
| train/ufo/replay_policy_age_min_updates | 16 | 16 | 16 | 16 | 14.2857143 |
| train/ufo/replay_retained_fraction | 1 | 1 | 1 | 1 | 1 |
| train/ufo/replay_size | 159744 | 307200 | 454656 | 614400 | 786432 |
| train/ufo/requested_action_abs_mean | 0.904628923 | 0.908459514 | 0.9062199 | 0.902332962 | 0.899396999 |
| train/ufo/requested_action_saturation_fraction | 0.439501479 | 0.447695225 | 0.44524468 | 0.441124707 | 0.43706151 |
| train/ufo/requested_applied_max_diff | 1.95897768 | 1.96205413 | 1.95409817 | 1.95764714 | 1.95840057 |
| train/ufo/reset_control_samples | 2816 | 2182.33333 | 1986.16667 | 1810.42857 | 1695.57143 |
| train/ufo/reset_projection_outside_fraction | 0 | 0 | 0 | 0 | 0 |
| train/ufo/reset_projection_residual_max_rad | 2.38418579e-07 | 2.38418579e-07 | 2.38418579e-07 | 2.38418579e-07 | 2.38418579e-07 |
| train/ufo/reset_projection_residual_mean_rad | 6.74368212e-08 | 6.65583085e-08 | 6.60861872e-08 | 6.66034075e-08 | 6.67664537e-08 |
| train/ufo/reset_projection_residual_p95_rad | 1.1920929e-07 | 1.1920929e-07 | 1.1920929e-07 | 1.1920929e-07 | 1.1920929e-07 |
| train/ufo/sampling_rounds | 6.64 | 12.88 | 19.12 | 25.36 | 31.64 |
| train/ufo/target_M | 1.17197347 | 0.58673604 | 0.58603463 | 0.656335608 | 0.644346546 |
| train/ufo/target_Q | -3.10170834 | -2.33070017 | -2.83432044 | -3.59286447 | -4.34736046 |
| train/ufo/target_auxQ | -4.04685939 | -3.52043202 | -3.97252052 | -4.44370083 | -5.00702748 |
| train/ufo/terminated_transitions | 1900.66667 | 2160.5 | 1979.33333 | 1803.14286 | 1691 |
| train/ufo/truncated_transitions | 0 | 0 | 0 | 0 | 0 |
| train/ufo/unc_Q | 1.45165695 | 0.505205476 | 0.295738944 | 0.206684812 | 0.161423005 |
| train/ufo/unc_auxQ | 0.808334897 | 0.402051086 | 0.248918088 | 0.172790385 | 0.138564822 |
| train/ufo/wall_time_s | 153.404853 | 357.169905 | 559.644217 | 759.741028 | 960.3447 |
| train/ufo/z_norm | 16 | 16 | 16 | 16 | 16 |

## 完整原生配置
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
    "rollout_expert_trajectories": false,
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


## 完整23关节控制与观察协议
```json
{
  "codec": {
    "schema_version": "eb/rpo-bfm-codec/v2",
    "interface_id": "ee_rpo_v2_bfm_23dof_v2",
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
    "observation_dims": {
      "states": 52,
      "last_action": 23,
      "history_actor": 300,
      "privileged_state": 358
    },
    "state_layout": [
      "q_minus_default",
      "dq_radps",
      "gravity_body",
      "omega_body_div4"
    ],
    "privileged_layout": [
      "root_height",
      "heading_local_positions_without_root",
      "heading_local_rotation_x_z",
      "heading_local_lin_vel",
      "heading_local_ang_vel"
    ],
    "quaternion": "wxyz",
    "history_layout": "frame_major_oldest_first_zero_pad",
    "history_frame": [
      "states",
      "last_action"
    ],
    "history_length": 4,
    "default_joint_pos": [
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.18,
      0.18,
      -0.1,
      -0.1,
      0.06,
      -0.06,
      0.3,
      0.3,
      0.0,
      0.0,
      -0.2,
      -0.2,
      0.78,
      0.78,
      0.0,
      0.0,
      0.0,
      0.0
    ],
    "action_gain": [
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0
    ],
    "action_scale": [
      0.6,
      0.6,
      3.14,
      0.6,
      0.6,
      2.355,
      2.355,
      1.4397,
      1.4397,
      1.695,
      1.695,
      1.35,
      1.35,
      1.57,
      1.57,
      0.6,
      0.6,
      1.085,
      1.085,
      0.5,
      0.5,
      1.57,
      1.57
    ],
    "policy_action_bounds": [
      -1.0,
      1.0
    ],
    "actuator_delay_steps": 0,
    "policy_hz": 50,
    "physics_hz": 200,
    "z_dim": 256,
    "privileged_velocity_point": "link_origin",
    "prompt_command": {
      "field": "commands",
      "dim": 3,
      "units": [
        "m/s",
        "m/s",
        "rad/s"
      ],
      "minimum": [
        -0.6,
        -0.5,
        -1.57
      ],
      "maximum": [
        1.0,
        0.5,
        1.57
      ],
      "consumer": "command_to_latent"
    },
    "action_offset": [
      -0.4,
      0.4,
      0.0,
      0.4,
      -0.4,
      -0.785,
      -0.785,
      -0.6542999999999999,
      -0.6542999999999999,
      1.445,
      -1.445,
      1.15,
      1.15,
      0.0,
      0.0,
      0.0,
      0.0,
      0.48500000000000004,
      0.48500000000000004,
      0.0,
      0.0,
      0.0,
      0.0
    ],
    "joint_position_min": [
      -1.0,
      -0.2,
      -3.14,
      -0.2,
      -1.0,
      -3.14,
      -3.14,
      -2.094,
      -2.094,
      -0.25,
      -3.14,
      -0.2,
      -0.2,
      -1.57,
      -1.57,
      -0.6,
      -0.6,
      -0.6,
      -0.6,
      -0.5,
      -0.5,
      -1.57,
      -1.57
    ],
    "joint_position_max": [
      0.2,
      1.0,
      3.14,
      1.0,
      0.2,
      1.57,
      1.57,
      0.7854,
      0.7854,
      3.14,
      0.25,
      2.5,
      2.5,
      1.57,
      1.57,
      0.6,
      0.6,
      1.57,
      1.57,
      0.5,
      0.5,
      1.57,
      1.57
    ],
    "action_mapping": "urdf_finite_midpoint_halfwidth"
  },
  "action_control": {
    "schema_version": "eb/rpo-target-slew/v2",
    "control_dt_s": 0.02,
    "target_velocity_limit_fraction": 0.5,
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
    "reset": "clamp measured q to physical finite bounds; normalize anchor; clear history",
    "replay_action": "requested",
    "observation_last_action": "applied",
    "guarantee": "target increments only; actual dq and infeasible reset residual remain measured",
    "control_profile": "reachable_v2",
    "interface_id": "ee_rpo_v2_bfm_23dof_v2",
    "codec_sha256": "dce25d554ceb7fa69287515946d2f5cd9d41d14f1b1ef38715b819d7e962b6ef",
    "joint_position_min_rad": [
      -1.0,
      -0.2,
      -3.14,
      -0.2,
      -1.0,
      -3.14,
      -3.14,
      -2.094,
      -2.094,
      -0.25,
      -3.14,
      -0.2,
      -0.2,
      -1.57,
      -1.57,
      -0.6,
      -0.6,
      -0.6,
      -0.6,
      -0.5,
      -0.5,
      -1.57,
      -1.57
    ],
    "joint_position_max_rad": [
      0.2,
      1.0,
      3.14,
      1.0,
      0.2,
      1.57,
      1.57,
      0.7854,
      0.7854,
      3.14,
      0.25,
      2.5,
      2.5,
      1.57,
      1.57,
      0.6,
      0.6,
      1.57,
      1.57,
      0.5,
      0.5,
      1.57,
      1.57
    ],
    "action_midpoint_rad": [
      -0.4,
      0.4,
      0.0,
      0.4,
      -0.4,
      -0.785,
      -0.785,
      -0.6542999999999999,
      -0.6542999999999999,
      1.445,
      -1.445,
      1.15,
      1.15,
      0.0,
      0.0,
      0.0,
      0.0,
      0.48500000000000004,
      0.48500000000000004,
      0.0,
      0.0,
      0.0,
      0.0
    ],
    "action_halfwidth_rad": [
      0.6,
      0.6,
      3.14,
      0.6,
      0.6,
      2.355,
      2.355,
      1.4397,
      1.4397,
      1.695,
      1.695,
      1.35,
      1.35,
      1.57,
      1.57,
      0.6,
      0.6,
      1.085,
      1.085,
      0.5,
      0.5,
      1.57,
      1.57
    ],
    "observation_default_joint_pos_rad": [
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.18,
      0.18,
      -0.1,
      -0.1,
      0.06,
      -0.06,
      0.3,
      0.3,
      0.0,
      0.0,
      -0.2,
      -0.2,
      0.78,
      0.78,
      0.0,
      0.0,
      0.0,
      0.0
    ]
  }
}
```


## 完整解析后RLinf配置
通用框架默认字段不代表原生UFO采用PPO损失；原生agent及真实联合更新计数决定优化行为。
```json
{
  "actor": {
    "model": {
      "model_type": "mlp_policy",
      "model_path": "",
      "policy_setup": "rpo-v2-bfm-ufo",
      "obs_dim": 375,
      "action_dim": 23,
      "num_action_chunks": 1,
      "hidden_dim": 2048,
      "precision": "32",
      "add_value_head": false,
      "is_lora": false,
      "lora_rank": 32,
      "critic_obs_dim": 733,
      "bfm_profile": "ufo_native_bringup",
      "motion_root": "/workspace/data/ds_fb0ecda9e70644f2afdd",
      "asset_sha256": "b58868a782d8863ac38059eb4eab89d27ee1d70565bf58f5a055f44510c8e842",
      "batch_size": 2048,
      "replay_capacity": 1000000,
      "hidden_layers": 6,
      "topology_id": "one_learner_three_samplers",
      "learning_profile": "physical_limits_stable_slew_v4",
      "architecture_profile": "native_fb_v1",
      "rollout_expert_trajectories": false,
      "auxiliary_profile": "soft_velocity_v7",
      "scaling_profile": "env6144_batch2048_v6",
      "control_profile": "reachable_v2",
      "optimizer_profile": "stabilized_v7",
      "codec_sha256": "dce25d554ceb7fa69287515946d2f5cd9d41d14f1b1ef38715b819d7e962b6ef"
    },
    "group_name": "ActorGroup",
    "training_backend": "ufo_agent",
    "micro_batch_size": 2048,
    "global_batch_size": 2048,
    "seed": 42,
    "enable_offload": false,
    "optim": null,
    "fsdp_config": null
  },
  "weight_syncer": {
    "type": "patch",
    "patch": {
      "snapshot_device": "cpu",
      "delta_encoding": true,
      "compression": "none",
      "init_sync": {
        "enabled": true,
        "prefixes": null,
        "bucket_size": 134217728
      }
    }
  },
  "cluster": {
    "num_nodes": 1,
    "component_placement": {
      "actor": 0,
      "env, rollout": "1-3"
    }
  },
  "runner": {
    "task_type": "embodied",
    "logger": {
      "log_path": "/workspace/output",
      "project_name": "rlinf",
      "experiment_name": "RPO UFO reachable v7 stabilized 500-update observation",
      "logger_backends": [
        "tensorboard"
      ]
    },
    "max_epochs": 4096,
    "max_steps": -1,
    "only_eval": false,
    "use_training_pipeline": false,
    "overlap_env_bootstrap": false,
    "val_check_interval": -1,
    "save_interval": -1,
    "resume_dir": "/workspace/resume/global_step_24",
    "ckpt_path": null
  },
  "algorithm": {
    "loss_type": "ufo_fb_cpr_aux",
    "update_epoch": 1,
    "normalize_advantages": false,
    "kl_penalty": "kl",
    "group_size": 1,
    "reward_type": "action_level",
    "logprob_type": "action_level",
    "entropy_type": "action_level",
    "adv_type": "gae",
    "loss_agg_func": "token-mean",
    "bootstrap_type": "standard",
    "gamma": 0.98,
    "gae_lambda": 1.0
  },
  "env": {
    "group_name": "EnvGroup",
    "train": {
      "env_type": "isaaclab",
      "total_num_envs": 6144,
      "auto_reset": true,
      "ignore_terminations": false,
      "use_rel_reward": false,
      "seed": 42,
      "group_size": 1,
      "reward_coef": 1.0,
      "use_fixed_reset_state_ids": false,
      "rollout_epoch": 1,
      "max_episode_steps": 1000,
      "max_steps_per_rollout_epoch": 4,
      "video_cfg": {
        "save_video": false,
        "info_on_video": false,
        "fps": 50,
        "video_base_dir": "/workspace/output/video/train"
      },
      "init_params": {
        "id": "RPO-UFO-Isaac-RSI-reachable-v7",
        "num_envs": null,
        "max_episode_steps": 1000,
        "task_description": "RPO V2 BFM-Zero online interaction with pinned observation-only expert motion."
      }
    },
    "eval": {
      "env_type": "isaaclab",
      "total_num_envs": 3,
      "auto_reset": true,
      "ignore_terminations": false,
      "use_rel_reward": false,
      "seed": 42,
      "group_size": 1,
      "reward_coef": 1.0,
      "use_fixed_reset_state_ids": true,
      "rollout_epoch": 1,
      "max_episode_steps": 1000,
      "max_steps_per_rollout_epoch": 1000,
      "video_cfg": {
        "save_video": false,
        "info_on_video": false,
        "fps": 50,
        "video_base_dir": "/workspace/output/video/eval"
      },
      "init_params": {
        "id": "RPO-UFO-Isaac-RSI-v1",
        "num_envs": null,
        "max_episode_steps": 1000,
        "task_description": "RPO V2 BFM-Zero online interaction with pinned observation-only expert motion."
      }
    }
  },
  "rollout": {
    "group_name": "RolloutGroup",
    "backend": "huggingface",
    "enable_offload": false,
    "pipeline_stage_num": 1,
    "enable_torch_compile": false,
    "enable_cuda_graph": false,
    "collect_transitions": true,
    "collect_prev_infos": true,
    "model": {
      "model_path": "",
      "precision": "32"
    }
  },
  "reward": {
    "use_reward_model": false
  },
  "critic": {
    "use_critic_model": false
  }
}
```


## 三轮复核、验证范围与限制


第一轮复核设计/本体语义与原始数据隔离；第二轮核对GPU拓扑、真实step、六Adam、裁剪hook与16→24→524完整恢复；第三轮核对全部tag、五窗、trace、replay及存储回收。平台容器测试536 passed/24 skipped；原生44 passed/1 skipped，G配置guard另1 passed；真实6 Adam一更新核验Actor raw1255→1、Actor权重840.449→50、D raw−10→−2。后续新增测试结果若未列出则未据此声明通过，skip不算通过。

未进行独立validation、held-out、100例验收或评估录像，不声明站立、行走或模型质量通过。物理resume非逐位复现，平均奖励/饱和降低不保证真实动力学安全；未新增接触力、穿透或真实机器人实验。
