# RPO UFO 接触顺应 v11：500 次新增更新完整观察记录

日期：2026-10-12。本文包含任务说明、冻结配置、真实训练/物理指标、checkpoint与决策，不依赖其它文档。仅仿真训练诊断；没有独立模型评估，没有Git提交。

## 1. 结果与决策

实际正式段：Train trn_b7e194580be0485da296，Job job_c8b6fe43587b4e84a522，Run run_4bc5e02a54b14f1ba0e9；状态 succeeded。真实learner24→524，新增500联合更新；三张采样卡各2048环境。

冻结3000更新准入结果：False。末两个100更新窗口按已完成episode数量加权的严格影子均值为 [12.804603951422875, 13.083483638237567] 步；全段物理最大额定速度比 6.2884321212768555；首步终止比例 0.0007216742843396681。

准入要求严格影子两窗口都>35、全段速度比<2.5、首步终止率≤0.1%、标量有限和诊断零错误、完整checkpoint可回读；2.0速度比为目标。跨窗口采样批次按其落入更新窗口的比例分配episode总长度和数量；这是episode加权口径，不是每批均值等权平均。

门禁摘要与原始记录分别进入完整checkpoint文件清单及SHA；平台重新计算原始记录再核对摘要。仅24→524或通过后的524→3524训练被允许，禁止拆成多个500小段绕过门禁；恢复候选页面同样拒绝失败门禁。

本次实际未创建后续3000更新Job，也没有新增独立评估任务。

本段未达到正常行走标准；独立评估门槛仍为末两窗口存活≥100步且满足原严格物理边界。短段执行成功和loss有限不能证明策略会正常行走，也不能证明柔顺地面向刚性地面或真机迁移。

关键结论是混合效果：训练存活末窗口只有小幅提高，严格影子存活反而降低；接触力全局峰值和首步冲击减轻，但严格速度违规更频繁，右踝pitch极值较基线恶化。不能将本次称为正常行走或整体动力学达标。

| 指标 | v10 | v11 |
| --- | --- | --- |
| 训练末100窗口日志均值/步 | 22.313360486711776 | 22.636510848999023 |
| 严格影子末100窗口日志均值/步 | 16.014141082763672 | 13.07493795667376 |
| 全部完成严格影子episode中位数/步 | 10.0 | 8.0 |
| 全局物理子步速度峰/radps | 55.763160705566406 | 50.307456970214844 |
| 右踝pitch物理子步峰/radps | 44.59560012817383 | 50.307456970214844 |
| 全段足垂直净力峰/N | 20959.25 | 12074.6611328125 |
| 首步足垂直净力峰/N | 4507.1708984375 | 1760.511962890625 |
| 首步终止比例 | 0.0008803462695326829 | 0.0007216742843396681 |
| 严格速度违规转移数/786432 | 26588 | 41825 |
| 速度子步-关节曝光比例 | 0.00030117449553116507 | 0.0006034270576808764 |

```json
{
  "schema_version": "eb/ufo-contact-long-gate/v1",
  "start": 24,
  "target": 524,
  "complete": true,
  "passed": false,
  "strict_shadow_last_windows": [
    12.804603951422875,
    13.083483638237567
  ],
  "physics_max_rated_ratio": 6.2884321212768555,
  "first_step_termination_fraction": 0.0007216742843396681,
  "error_count": 0
}
```

```json
{
  "candidate": {
    "available": false,
    "reason": "Long training gate did not pass"
  },
  "preview_status": 422,
  "preview": {
    "detail": "Long training gate did not pass"
  },
  "job_created": false
}
```

## 2. 任务与固定配置

| 项目 | 实际设置 |
| --- | --- |
| 本体 | RPO/RP0 v2 23DoF，IsaacLab/PhysX仿真；不是G1本体或真机训练 |
| 算法/生命周期 | 固定UFO FBcprAuxAgent，RLinf RPC采样/同步；平台API→Job/Run→ebd |
| 拓扑 | GPU0一个learner，GPU1/2/3各一个2048环境sampler，共6144；不是DDP |
| 网络 | 完整435116484参数；F/Actor/Qdisc/Qaux2048宽6层，D1024宽3层，B256宽1层；z256、actor输入375、FB输入733、动作23 |
| 采样/优化 | 每round4控制步×6144=24576转移；每24576新转移触发16次原生联合更新；batch2048；warmup10240 |
| Replay | CPU FIFO 4000000容量，24数组流式checkpoint v2；段内总860160/新增786432，未填满且没有FIFO淘汰 |
| 学习率 | F/Actor/Qdisc/Qaux3e-4、B1e-5、D3e-6；六Adam；discount0.98、FBtau0.01、Qtau0.005、orth100 |
| 保护 | Actor grad clip1，其它五优化器5；scale_reg w≤50；判别reward[-2,2]；AMP/compile/cudagraphs关闭 |
| 物理/控制 | physics dt0.0025s，decimation8，control dt0.02s；episode horizon1000；512×512m离线地面 |
| 驱动 | 显式DelayedPD，延迟0；踝Kp40/Kd2.5，DR×[0.9,1.1]，踝力矩27Nm；腿120Nm；没有新增LPF |
| 动作 | v2完整URDF盒，requested进replay、applied进观测；target slew0.5×额定速度，仅限制目标变化率 |
| 接触 | 地面implicit compliant spring stiffness60000N/m、damping600Ns/m、restitution0、multiply；仅地面改变 |
| 速度惩罚 | r=abs(dq)/Vmax；p=relu(r−0.70)^2+8relu(r−1)^2；对8个完成子步平均、23关节求和、系数−25一次加权 |
| 终止/影子 | 训练速度1.3×，原位置/力矩/保护部位接触/高度0.50/倾角0.80保持；原1.0×影子首次违规独立记录 |
| RSI | 仅train原始root姿态/速度、q/dq；近地碰撞点abs(vnormal)≤0.15m/s、2mm间隙/root lift≤3cm；支撑腿膝≥0.15rad、髋pitch≤0.10rad，摆动腿不限 |
| Tracking | 固定50%独立环境跟随实际RSI绑定的250转移/251状态窗口；每相位B编码8帧前向平均；其它环境mixed，非100步边界z不变 |
| 资源/预算 | 4卡独占、24CPU、96GiB RAM、32GiB shm；500段≤2h、新转移≤1M、无自动重试；条件3000段≤4h、新转移≤5M |
| 种子 | learner42，sampler42/43/44；实际分区hash核验；旧版本种子语义不变 |
| 恢复语义 | 新MDP从相同seed重新16→24→524；没有从旧v10步524暖启动；完整模型/六Adam/targets/norm/replay/每rank随机与context恢复，物理仿真状态重新初始化 |

速度罚在r=2时为旧二次罚5.7349倍，在r=1处连续且一阶连续；采用有限多项式，未引入指数放大。罚值并不直接对PhysX反向传播，也不保证策略会主动制动。native aux_rew/limits_dof_vel日志是正的raw罚，权重−25仅用一次；所有aux加权和共同经过RewardNormalizer，因此没有可唯一拆出的“第九项独立归一化贡献”。

环境启动打印的29项RewardManager项是原环境日志；UFO更新实际使用native FB目标、判别reward和九项raw aux。不能把PPO环境的termination_penalty−200当作已进入UFO actor目标的独立终止奖惩。

RSI=Reference State Initialization（参考状态初始化）。筛选拒绝不合格起点，保留原q/dq及root速度；改变起点分布，不代表改写或清理全部专家轨迹。

Train DTO的kind=ppo是既有表单/路由分类；实际execution.algorithm_id=bfm_fb_cpr_aux、training_backend=ufo_agent、六Adam native更新，不是PPO优化。输入Dataset.validation.training_ready=false保留已发布转换元数据；本配方的真实运行资格单独验证，不原地修改该数据版本。

## 3. 受管输入、来源及执行身份

```json
{
  "job_spec": {
    "kind": "train.rl",
    "name": "RPO UFO compliance v11 native step524 3x2048",
    "gates": [],
    "image": "sha256:6a709e3a277f19675e7ef464967e37f2ab77eae56388fc8dc3b4f083a62223ff",
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
      "resume_from": "run_86528d2756de4a489532"
    },
    "labels": {},
    "params": {
      "stack": "rlinf",
      "purpose": "training",
      "execution": {
        "recipe_id": "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_compliance_v11",
        "runtime_id": "rlinf",
        "provider_id": "ufo",
        "topology_id": "one_learner_three_samplers",
        "algorithm_id": "bfm_fb_cpr_aux",
        "simulator_id": "isaaclab",
        "progress_unit": "learner_joint_update",
        "schema_version": "eb/training-binding/v1",
        "checkpoint_format": "ufo_agent_state_v2"
      },
      "model_name": "RPO-UFO-compliance-v11-step524",
      "config_name": "rpo_bfm_ufo_isaac",
      "resume_mode": "trainer_state",
      "resume_step": 24,
      "template_id": "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_compliance_v11",
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
        "ground_profile": "wide_plane_v9",
        "schema_version": "eb/ufo-image-source/v1",
        "contact_profile": "contact_stable_v8",
        "context_profile": "environment_tracking_v10",
        "control_profile": "reachable_v2",
        "scaling_profile": "env6144_batch2048_v6",
        "dynamics_profile": "contact_compliance_v11",
        "auxiliary_profile": "soft_velocity_v7",
        "ground_patch_hash": "sha256:554a18609c9b8a420811fa6b1ded908a4bdf181e52e8d513c8772de6084eff59",
        "optimizer_profile": "stabilized_v7",
        "contact_patch_hash": "sha256:e20a0ff930b3b9430ece70c21ab316a06cef92b679b638c0a35d7fed62d6d814",
        "topology_patch_hash": "sha256:3078506628924464cb42d0b2f108b3fab502a43fd75e091f358d504565294d23",
        "ankle_control_profile": "kd25",
        "auxiliary_source_sha256": "sha256:ce37467a97a287c6c63825b53b69906d16ee3a79c375914ef7049105cd1a5249",
        "stabilization_patch_hash": "sha256:564f5452e12e3179112e6aa3020efba4cbcedc0ad90dd26b98aed78a50dd85b1",
        "rollout_profile_patch_hash": "sha256:777456ce171734326cf8c55eaa434fc5a69a27d0b2d0b725608dd54c190f4a77",
        "scaling_profile_patch_hash": "sha256:58a2dca8377ed2ede804c7101b39bc6992ec669ef95e610ddec4b2a59bfc6b20",
        "learning_profile_patch_hash": "sha256:d47623e48b4cc757d5e63336a1fa7d49234c3833a2f1c6bbdbee4fe7ffd9a272",
        "tracking_contact_patch_hash": "sha256:d9b075574bf38e19cd74b39db52edf92573a019ba24d06e08fb82861f07dbf15",
        "action_control_source_sha256": "sha256:0cad3bac9e3efb3f1f37d6cac873f789548143b9cd597ed90a93a3045339cd24",
        "auxiliary_profile_patch_hash": "sha256:30e7b4ea7a3ec0e45cdbe0a9dbf718eb219c2c6fd36d07f9b18bd5222183a03a",
        "action_control_profile_patch_hash": "sha256:548260ca5e3eedd53ae739c00672c7e273e7835faed734778039cf5977b56e0d"
      },
      "environment_profile": {
        "id": "rpo_bfm_ufo_isaac_4gpu_rsi_compliance_v11",
        "hash": "sha256:e9c499d8ca3be1420cfa0f3a0fb00f6540772ea682d4df3c6b79f4387105485b",
        "definition": {
          "task": "RPO23DoF完整435M网络；GPU0 learner、GPU1–3各2048环境；batch2048、CPU replay400万。固定50%独立环境跟随250步train专家片段，与RSI起点绑定；512m地面覆盖。 配对接触诊断选定踝关节Kd2.5，原Kp40/27Nm限幅/目标slew保持，无新增低通。 柔顺地面接触60000N/m、600Ns/m；RSI支撑腿膝>=0.15rad、髋pitch<=0.10rad，不限制摆动腿，只筛选不改q/dq。",
          "reward": "九项UFO aux；速度罚为8个已完成物理子步的归一化均值，软阈值0.70、系数-25。原生六优化器；Actor梯度裁剪1，其余5；w<=50，D lr3e-6/reward[-2,2]。 子步速度raw罚relu(r-0.7)^2+8relu(r-1)^2，保留8帧均值与-25一次加权。",
          "failure": "train接触点|v_normal|<=0.15m/s；物理400Hz、控制50Hz。探索速度终止1.3倍，原位置/力矩/宏观跌倒约束保持；原严格标准影子存活与所有子步违规独立记录。",
          "success": "新16→24完整恢复后新增500更新；末两100更新窗口严格影子>35、全段物理速度比<2.5等门禁通过，才允许同配方524→3524新增3000更新/4h。训练未达标不独立评估。",
          "settings": {}
        }
      },
      "resume_source_image": "eb/rlinf:rpo-bfm-ufo-compliance-v11",
      "source_train_job_id": "trn_ad1f43879bf040f7a928",
      "target_learner_updates": 524
    },
    "policy": {
      "priority": 50,
      "retry_on": [
        "infra_failed",
        "oom"
      ],
      "timeout_h": 1.0,
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
  "image_digest": "sha256:6a709e3a277f19675e7ef464967e37f2ab77eae56388fc8dc3b4f083a62223ff",
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
    "ground_profile": "wide_plane_v9",
    "schema_version": "eb/ufo-image-source/v1",
    "contact_profile": "contact_stable_v8",
    "context_profile": "environment_tracking_v10",
    "control_profile": "reachable_v2",
    "scaling_profile": "env6144_batch2048_v6",
    "dynamics_profile": "contact_compliance_v11",
    "auxiliary_profile": "soft_velocity_v7",
    "ground_patch_hash": "sha256:554a18609c9b8a420811fa6b1ded908a4bdf181e52e8d513c8772de6084eff59",
    "optimizer_profile": "stabilized_v7",
    "contact_patch_hash": "sha256:e20a0ff930b3b9430ece70c21ab316a06cef92b679b638c0a35d7fed62d6d814",
    "topology_patch_hash": "sha256:3078506628924464cb42d0b2f108b3fab502a43fd75e091f358d504565294d23",
    "ankle_control_profile": "kd25",
    "auxiliary_source_sha256": "sha256:ce37467a97a287c6c63825b53b69906d16ee3a79c375914ef7049105cd1a5249",
    "stabilization_patch_hash": "sha256:564f5452e12e3179112e6aa3020efba4cbcedc0ad90dd26b98aed78a50dd85b1",
    "rollout_profile_patch_hash": "sha256:777456ce171734326cf8c55eaa434fc5a69a27d0b2d0b725608dd54c190f4a77",
    "scaling_profile_patch_hash": "sha256:58a2dca8377ed2ede804c7101b39bc6992ec669ef95e610ddec4b2a59bfc6b20",
    "learning_profile_patch_hash": "sha256:d47623e48b4cc757d5e63336a1fa7d49234c3833a2f1c6bbdbee4fe7ffd9a272",
    "tracking_contact_patch_hash": "sha256:d9b075574bf38e19cd74b39db52edf92573a019ba24d06e08fb82861f07dbf15",
    "action_control_source_sha256": "sha256:0cad3bac9e3efb3f1f37d6cac873f789548143b9cd597ed90a93a3045339cd24",
    "auxiliary_profile_patch_hash": "sha256:30e7b4ea7a3ec0e45cdbe0a9dbf718eb219c2c6fd36d07f9b18bd5222183a03a",
    "action_control_profile_patch_hash": "sha256:548260ca5e3eedd53ae739c00672c7e273e7835faed734778039cf5977b56e0d"
  },
  "native_checkpoint": {
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
    "fingerprint": "sha256:8eb36b81219a6ce57a274e8f0fcf3b158350170de5c0876d1640ef4748b8b482",
    "execution": {
      "schema_version": "eb/training-binding/v1",
      "recipe_id": "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_compliance_v11",
      "algorithm_id": "bfm_fb_cpr_aux",
      "provider_id": "ufo",
      "runtime_id": "rlinf",
      "simulator_id": "isaaclab",
      "progress_unit": "learner_joint_update",
      "checkpoint_format": "ufo_agent_state_v2",
      "topology_id": "one_learner_three_samplers"
    }
  },
  "control_and_interface": {
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
    "execution_action_control": {
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
      ],
      "physics": {
        "physics_dt_s": 0.0025,
        "decimation": 8,
        "policy_dt_s": 0.02,
        "sole_clearance_m": 0.002,
        "max_initial_sole_height_m": 0.02,
        "max_root_lift_m": 0.03,
        "max_self_penetration_m": 0.002,
        "min_root_height_m": 0.53,
        "max_root_tilt_rad": 0.7,
        "reset_velocity_limit_fraction": 0.7,
        "pd": "unchanged explicit gains/effort/velocity limits; twice the integration resolution",
        "hard_limits": "latch every physics substep; retain original 0.001 tolerance",
        "protected_contact": "latch protected-body net force >1N every physics substep"
      },
      "ground": {
        "profile": "wide_plane_v9",
        "size_m": [
          512.0,
          512.0
        ],
        "motion_reserve_m": 20.0
      },
      "tracking_context": "environment_tracking_v10",
      "training_velocity_termination_multiplier": 1.3,
      "velocity_reward_aggregation": "mean_eight_completed_physics_frames",
      "rsi_support_normal_velocity_max_mps": 0.15,
      "ankle_control_profile": "kd25",
      "ankle_nominal_damping": 2.5,
      "dynamics_profile": "contact_compliance_v11",
      "contact_stiffness_N_per_m": 60000.0,
      "contact_damping_Ns_per_m": 600.0,
      "min_rsi_knee_flexion_rad": 0.15,
      "max_rsi_hip_pitch_rad": 0.1,
      "rsi_flexion_scope": "near-ground support legs only",
      "velocity_penalty": "relu(r-.7)^2+8*relu(r-1)^2"
    }
  }
}
```

```json
{
  "python": "3.11.16 (main, Sep  1 2026, 00:14:22) [GCC 12.2.0]",
  "torch": "2.7.0+cu128",
  "isaaclab": "2.3.2.post1",
  "isaacsim": "5.1.0.0",
  "mujoco": "3.3.5",
  "numpy": "1.26.0",
  "scipy": "1.15.3",
  "ray": "2.47.0",
  "omegaconf": "2.3.0",
  "hydra-core": "1.3.2",
  "tensorboard": "2.20.0",
  "isaac_distributions": {
    "isaaclab": "2.3.2.post1",
    "isaaclab_tasks": "0.11.14",
    "isaacsim-core": "5.1.0.0",
    "isaacsim-storage": "5.1.0.0",
    "isaacsim-ros1": "5.1.0.0",
    "isaacsim-sensor": "5.1.0.0",
    "isaacsim-rl": "5.1.0.0",
    "isaacsim": "5.1.0.0",
    "isaacsim-test": "5.1.0.0",
    "isaacsim-utils": "5.1.0.0",
    "isaacsim-extscache-kit": "5.1.0.0",
    "isaacsim-extscache-physics": "5.1.0.0",
    "isaacsim-example": "5.1.0.0",
    "isaacsim-kernel": "5.1.0.0",
    "isaacsim-robot-motion": "5.1.0.0",
    "isaacsim-robot": "5.1.0.0",
    "isaacsim-robot-setup": "5.1.0.0",
    "isaacsim-gui": "5.1.0.0",
    "isaacsim-app": "5.1.0.0",
    "isaacsim-ros2": "5.1.0.0",
    "isaacsim-cortex": "5.1.0.0",
    "isaacsim-asset": "5.1.0.0",
    "isaacsim-replicator": "5.1.0.0",
    "isaacsim-extscache-kit-sdk": "5.1.0.0",
    "isaacsim-code-editor": "5.1.0.0",
    "isaacsim-benchmark": "5.1.0.0",
    "isaacsim-template": "5.1.0.0"
  }
}
```

所有数据/模型/资产路径是实际绝对输入或输出参数。已发布旧v10/interface/dataset/asset保持；新recipe、环境profile和image有独立身份。两个qualification的ModelVersion可能为空，完整native状态仍作为恢复产物验证；正式ModelVersion见任务信息。

| 阶段 | Train | Run | 实际结果 |
| --- | --- | --- | --- |
| 物理配对诊断 | asset.validate job_5e423bfd86394723a7ea | run_8bf45eafcc214993b4d9 | 三相×三卡×2048×32，Job succeeded，无策略评估 |
| 16资格 | trn_abf9fe8f8def45229ed3 | run_8f08d3a6c57c48d49c67 | sixAdam16/49152转移/round2/allSHA通过 |
| 16→24资格 | trn_ad1f43879bf040f7a928 | run_86528d2756de4a489532 | sixAdam24/73728/round3/allSHA/同fingerprint通过 |
| 24→524正式 | trn_b7e194580be0485da296 | run_4bc5e02a54b14f1ba0e9 | 新增500联合更新，allSHA/sixAdam524/全部原始指标与trace核验 |

```json
{
  "id": "trn_b7e194580be0485da296",
  "name": "RPO UFO compliance v11 native step524 3x2048",
  "status": "succeeded",
  "job_id": "job_c8b6fe43587b4e84a522",
  "run_id": "run_4bc5e02a54b14f1ba0e9",
  "model_version_id": null,
  "started_at": "2026-10-12T08:27:05.495512+00:00",
  "finished_at": "2026-10-12T08:50:55.888148+00:00",
  "error_message": null
}
```

```json
{
  "registered_model_version_id": "mv_423df77e068640589972",
  "model_registration": {
    "id": "mv_423df77e068640589972",
    "family_id": "mf_rpo_v2_bfm_zero",
    "name": "RPO-UFO-compliance-v11-step524",
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
      "run_id": "run_4bc5e02a54b14f1ba0e9",
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
        "image": "sha256:6a709e3a277f19675e7ef464967e37f2ab77eae56388fc8dc3b4f083a62223ff",
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
        "recipe_id": "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_compliance_v11",
        "runtime_id": "rlinf",
        "provider_id": "ufo",
        "topology_id": "one_learner_three_samplers",
        "algorithm_id": "bfm_fb_cpr_aux",
        "simulator_id": "isaaclab",
        "progress_unit": "learner_joint_update",
        "schema_version": "eb/training-binding/v1",
        "checkpoint_format": "ufo_agent_state_v2"
      },
      "readiness": "simulation_unverified",
      "template_id": "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_compliance_v11",
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
      "train_job_id": "job_c8b6fe43587b4e84a522",
      "normalization": "frozen_running_stats",
      "progress_unit": "learner_joint_update",
      "action_control": {
        "reset": "clamp measured q to physical finite bounds; normalize anchor; clear history",
        "ground": {
          "size_m": [
            512.0,
            512.0
          ],
          "profile": "wide_plane_v9",
          "motion_reserve_m": 20.0
        },
        "physics": {
          "pd": "unchanged explicit gains/effort/velocity limits; twice the integration resolution",
          "decimation": 8,
          "hard_limits": "latch every physics substep; retain original 0.001 tolerance",
          "policy_dt_s": 0.02,
          "physics_dt_s": 0.0025,
          "max_root_lift_m": 0.03,
          "sole_clearance_m": 0.002,
          "max_root_tilt_rad": 0.7,
          "min_root_height_m": 0.53,
          "protected_contact": "latch protected-body net force >1N every physics substep",
          "max_self_penetration_m": 0.002,
          "max_initial_sole_height_m": 0.02,
          "reset_velocity_limit_fraction": 0.7
        },
        "guarantee": "target increments only; actual dq and infeasible reset residual remain measured",
        "codec_sha256": "dce25d554ceb7fa69287515946d2f5cd9d41d14f1b1ef38715b819d7e962b6ef",
        "control_dt_s": 0.02,
        "interface_id": "ee_rpo_v2_bfm_23dof_v2",
        "replay_action": "requested",
        "schema_version": "eb/rpo-target-slew/v2",
        "control_profile": "reachable_v2",
        "dynamics_profile": "contact_compliance_v11",
        "tracking_context": "environment_tracking_v10",
        "velocity_penalty": "relu(r-.7)^2+8*relu(r-1)^2",
        "rsi_flexion_scope": "near-ground support legs only",
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
        "ankle_control_profile": "kd25",
        "ankle_nominal_damping": 2.5,
        "max_rsi_hip_pitch_rad": 0.1,
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
        "contact_damping_Ns_per_m": 600.0,
        "min_rsi_knee_flexion_rad": 0.15,
        "contact_stiffness_N_per_m": 60000.0,
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
        "velocity_reward_aggregation": "mean_eight_completed_physics_frames",
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
        ],
        "rsi_support_normal_velocity_max_mps": 0.15,
        "training_velocity_termination_multiplier": 1.3
      },
      "schema_version": "eb/rpo-bfm-model/v1",
      "artifact_sha256": "sha256:dbdd64bd7d6c6fa02a97539aa281d2bf3e4c2c8857be4b807f05c93c55e509a9",
      "checkpoint_path": "/data/kk/models/runs/run_4bc5e02a54b14f1ba0e9",
      "learner_progress": {
        "next_trigger": 870400,
        "sampling_rounds": 35,
        "online_transitions": 860160,
        "learner_joint_updates": 524
      },
      "auxiliary_profile": "soft_velocity_v7",
      "checkpoint_format": "ufo_agent_state_v2",
      "hardware_verified": false,
      "environment_profile": {
        "id": "rpo_bfm_ufo_isaac_4gpu_rsi_compliance_v11",
        "hash": "sha256:e9c499d8ca3be1420cfa0f3a0fb00f6540772ea682d4df3c6b79f4387105485b",
        "definition": {
          "task": "RPO23DoF完整435M网络；GPU0 learner、GPU1–3各2048环境；batch2048、CPU replay400万。固定50%独立环境跟随250步train专家片段，与RSI起点绑定；512m地面覆盖。 配对接触诊断选定踝关节Kd2.5，原Kp40/27Nm限幅/目标slew保持，无新增低通。 柔顺地面接触60000N/m、600Ns/m；RSI支撑腿膝>=0.15rad、髋pitch<=0.10rad，不限制摆动腿，只筛选不改q/dq。",
          "reward": "九项UFO aux；速度罚为8个已完成物理子步的归一化均值，软阈值0.70、系数-25。原生六优化器；Actor梯度裁剪1，其余5；w<=50，D lr3e-6/reward[-2,2]。 子步速度raw罚relu(r-0.7)^2+8relu(r-1)^2，保留8帧均值与-25一次加权。",
          "failure": "train接触点|v_normal|<=0.15m/s；物理400Hz、控制50Hz。探索速度终止1.3倍，原位置/力矩/宏观跌倒约束保持；原严格标准影子存活与所有子步违规独立记录。",
          "success": "新16→24完整恢复后新增500更新；末两100更新窗口严格影子>35、全段物理速度比<2.5等门禁通过，才允许同配方524→3524新增3000更新/4h。训练未达标不独立评估。",
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
    "created_at": "2026-10-12T08:50:55.798977Z",
    "train_job_name": "RPO UFO compliance v11 native step524 3x2048",
    "train_page_id": "trn_b7e194580be0485da296",
    "artifact": {
      "id": "art_05bec4a1a47a4263885d",
      "kind": "rlinf_checkpoint",
      "sha256": "sha256:dbdd64bd7d6c6fa02a97539aa281d2bf3e4c2c8857be4b807f05c93c55e509a9",
      "size_bytes": 13786159383,
      "storage": "/data/kk/models/runs/run_4bc5e02a54b14f1ba0e9",
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
              "bytes": 6249,
              "sha256": "5b28dbd03d1e252100d69c44dd1a0c473963a290f596b688f3e5785d348c963f"
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
              "sha256": "af3347fb0dde00b5181e9b5d476f770059cfad655833174c41b683aef26edb3a"
            },
            {
              "path": "model/native/global_step_524/optimizers.pth",
              "bytes": 3481181041,
              "sha256": "e0c0f26823680a55f16ef681426bc3739a383c0fb844f26ba5e382d5ea2cce9b"
            },
            {
              "path": "model/native/global_step_524/replay/array_000.npy",
              "bytes": 79134848,
              "sha256": "ed57625e3d5cb577ba4e98f6ef70a5be1f6165cb53c8793d2af2b33a07e0504c"
            },
            {
              "path": "model/native/global_step_524/replay/array_001.npy",
              "bytes": 3440768,
              "sha256": "f04af74e0899e899ecf8cb13ef7bad8ca672ebc07745e2b226f0a24e8dfbf096"
            },
            {
              "path": "model/native/global_step_524/replay/array_002.npy",
              "bytes": 3440768,
              "sha256": "0caa1aa789fec0bc6b9796e0f04aebd91d960bb2771544d9608d906d76062853"
            },
            {
              "path": "model/native/global_step_524/replay/array_003.npy",
              "bytes": 3440768,
              "sha256": "1d11af5c57ba70849e0143fb20d9ab15e3bb932a626014b6027d0ac584dabd7f"
            },
            {
              "path": "model/native/global_step_524/replay/array_004.npy",
              "bytes": 3440768,
              "sha256": "01c55c4aa242f92c11a101c2cf88562d241cda151c58b9ff622e15f16ae8cea8"
            },
            {
              "path": "model/native/global_step_524/replay/array_005.npy",
              "bytes": 3440768,
              "sha256": "848a1e201a5f18702e64c69202666c3a9fc567dd3e128f1190aa2d8cf0a6094d"
            },
            {
              "path": "model/native/global_step_524/replay/array_006.npy",
              "bytes": 3440768,
              "sha256": "08b3c377368d36c7afed0649670efdfa9de899f67648a52a938ae8c7c2943050"
            },
            {
              "path": "model/native/global_step_524/replay/array_007.npy",
              "bytes": 3440768,
              "sha256": "eb1915b590d3c09420bb85bb31d6ba49571be05c48f51d866f0f23657880dc39"
            },
            {
              "path": "model/native/global_step_524/replay/array_008.npy",
              "bytes": 3440768,
              "sha256": "4f6f5550289ebd33e3f89457b0fefd365c6ac84326c83c0b5f088e4b7813d70a"
            },
            {
              "path": "model/native/global_step_524/replay/array_009.npy",
              "bytes": 3440768,
              "sha256": "7ba405ba3af4eda0318150824a942e14e5b3f57d156faabe3ee9117d27ebc27a"
            },
            {
              "path": "model/native/global_step_524/replay/array_010.npy",
              "bytes": 6881408,
              "sha256": "c34eb99e84a3400379f9309137098ef3d08687780a31b184bcf3d9c9777cd6c5"
            },
            {
              "path": "model/native/global_step_524/replay/array_011.npy",
              "bytes": 1032192128,
              "sha256": "dc67877da9230fd883892cc43316453622117a6a953418102732ce1c22fa3a2b"
            },
            {
              "path": "model/native/global_step_524/replay/array_012.npy",
              "bytes": 79134848,
              "sha256": "7d281e3396666043957e94df6222c7eb600d505f7483f017b822d6dfa1a160b9"
            },
            {
              "path": "model/native/global_step_524/replay/array_013.npy",
              "bytes": 1231749248,
              "sha256": "0b795d2369e75ed570e35b5501b8c16d5716ac416d38ca6bfcbb27309b487e45"
            },
            {
              "path": "model/native/global_step_524/replay/array_014.npy",
              "bytes": 178913408,
              "sha256": "d66997d053ffa276d39425703c6692d23cf6a0e15debc7ef9b7d51c8d146af7b"
            },
            {
              "path": "model/native/global_step_524/replay/array_015.npy",
              "bytes": 860288,
              "sha256": "0a4682ddef272ec8c1816cf6823b534b33962234ea5cdaac379f18ddb486087e"
            },
            {
              "path": "model/native/global_step_524/replay/array_016.npy",
              "bytes": 860288,
              "sha256": "31ba486699f2858be0234f3e959120e2c30c2eb7a9f8ff4d60dc548f01cb4e90"
            },
            {
              "path": "model/native/global_step_524/replay/array_017.npy",
              "bytes": 1032192128,
              "sha256": "19523bf5c086234f0ba30144739c147187d0605e31df3e117bebee0c68d43330"
            },
            {
              "path": "model/native/global_step_524/replay/array_018.npy",
              "bytes": 79134848,
              "sha256": "cb1089ed84cdddda22532f4d99643072eaebb7bed17be000cec0c733fa385fac"
            },
            {
              "path": "model/native/global_step_524/replay/array_019.npy",
              "bytes": 1231749248,
              "sha256": "2441339cce8f83c28df1e4ef1d4a9f2dc2c05bf7aa0dd6b970f8f2038242fb40"
            },
            {
              "path": "model/native/global_step_524/replay/array_020.npy",
              "bytes": 178913408,
              "sha256": "c7d24cb4dc1418a0637fad085aaa1983107ecafefc954f9d7057ceec2969b0c9"
            },
            {
              "path": "model/native/global_step_524/replay/array_021.npy",
              "bytes": 6881408,
              "sha256": "25805036cef312d11ecfb37fc73778886f63caf007db4724d99576cd1045ada9"
            },
            {
              "path": "model/native/global_step_524/replay/array_022.npy",
              "bytes": 6881408,
              "sha256": "c6f2021f6f3a33fd94549c6e7e9636101c16c8834640d0382797ca0e18e662e4"
            },
            {
              "path": "model/native/global_step_524/replay/array_023.npy",
              "bytes": 880803968,
              "sha256": "0f134bf97f5764ffaaaf80daace8b301129bfa588f11157a51395776b05ee0b0"
            },
            {
              "path": "model/native/global_step_524/resume-manifest.json",
              "bytes": 29185,
              "sha256": "f01ef783dacaf1f46c20e1a75610b76334d01662acd82fe25538ace6ba97c4c1"
            },
            {
              "path": "model/native/global_step_524/trainer-state.pt",
              "bytes": 903712401,
              "sha256": "2b8fa0d2c25b6f41fec3df04416d35883da6dd614303192445cb5594aee7b193"
            },
            {
              "path": "model/native/global_step_524/training-gate-records.json",
              "bytes": 121525,
              "sha256": "3873d77e3e39c269096f1de8c9894b9b068951002bd5cbbf22882ea334fda174"
            },
            {
              "path": "model/native/global_step_524/training-gate.json",
              "bytes": 307,
              "sha256": "38853623ed00c18c598927c4d253f1c20a908f7847188d8bcb118e9050e90477"
            },
            {
              "path": "model/native-checkpoint.json",
              "bytes": 88,
              "sha256": "47caad91a80d1cbebbb2a750326d52642be4d75fd37c559f3c8a1958ed2c0b6b"
            },
            {
              "path": "model/resolved-config.yaml",
              "bytes": 4003,
              "sha256": "c14a18af39579bbed76fff0cc92784a6add8c5f07792a97ff6d56556b894ae7e"
            },
            {
              "path": "model/source-receipt.json",
              "bytes": 2789,
              "sha256": "017fb009f40aaea6022c0bed060b12c2db141adde25c2876f97f2331c01319f9"
            }
          ],
          "format": "ufo_agent_state_v2",
          "run_id": "run_4bc5e02a54b14f1ba0e9",
          "status": "complete",
          "execution": {
            "recipe_id": "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_compliance_v11",
            "runtime_id": "rlinf",
            "provider_id": "ufo",
            "topology_id": "one_learner_three_samplers",
            "algorithm_id": "bfm_fb_cpr_aux",
            "simulator_id": "isaaclab",
            "progress_unit": "learner_joint_update",
            "schema_version": "eb/training-binding/v1",
            "checkpoint_format": "ufo_agent_state_v2"
          },
          "model_path": "model",
          "template_id": "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_compliance_v11",
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
}
```

## 4. 配对接触对照与选择依据

安装版本地面本来restitution=0，multiply合并；不是完全弹性碰撞。仅设0不会构成修复。官方PhysX接口提供implicit spring+damping柔顺接触；本次实际USD碰撞网格/material绑定已检查，不用配置文本替代实值。物理说明依据：[NVIDIA刚体接触文档](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/rigid_bodies_articulations/rigid_bodies.html)。

```json
{
  "schema_version": "eb/ufo-contact-compliance-probe/v1",
  "phases": [
    "baseline",
    "compliant",
    "compliant_kd30"
  ],
  "samplers": 3,
  "envs_per_sampler": 2048,
  "steps": 32,
  "detail_envs": 256,
  "seeds": [
    42,
    43,
    44
  ],
  "split": "train",
  "qualification": "not_requested",
  "requested_action": "anchor + ankle .25sin(2Hz), hip/knee .12sin(2Hz), clipped to [-1,1]",
  "pairing": "same RNG seeds, flexion-filtered RSI references and requested actions; fresh simulator each phase/rank"
}
```

| 相位 | 活跃首episode转移 | 首步终止/6144 | 速度暴露率 | 物理最大额定比 | 最大rank力p99 N | 最大足垂直净力N |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 154813 | 5 | 2.671526012167836e-05 | 1.7080796957015991 | 2120.945246582033 | 5022.982421875 |
| compliant | 173656 | 5 | 2.7446816394631264e-05 | 1.3771494626998901 | 1045.930639648439 | 2281.342041015625 |
| compliant_kd30 | 174030 | 5 | 2.260979491292106e-05 | 1.465437412261963 | 1046.7274438476568 | 2223.428955078125 |

选择compliant/Kd2.5：相对刚性/Kd2.5，力峰约−54.58%、速度比峰约−19.37%；Kd3.0暴露占比更低但速度极值更高，保留较小控制改动。Kd2.5柔顺组暴露占比略升，不能声称所有超速指标改善。所有相位首步5/6144=0.08138%，不是0。动作固定为anchor加2Hz踝0.25/髋膝0.12正弦并限幅，6144起点/seed配对；每卡256保留完整细节。与真实学习策略的事故不是同一动作分布。

```json
{
  "method": "collision hull transformed by foot link pose at global-|dq|-peak substep; event-only, not all-substep clearance",
  "groups": {
    "baseline": {
      "rows": 19615,
      "event_sole_min_z_m": -0.003707787738855592,
      "below_minus_20mm": 0,
      "below_minus_10mm": 0
    },
    "compliant": {
      "rows": 21809,
      "event_sole_min_z_m": -0.006300470562970298,
      "below_minus_20mm": 0,
      "below_minus_10mm": 0
    },
    "compliant_kd30": {
      "rows": 22116,
      "event_sole_min_z_m": -0.0071411809414530375,
      "below_minus_20mm": 0,
      "below_minus_10mm": 0
    }
  }
}
```

网格高度复核只覆盖全局abs(dq)峰值子步事件；柔顺组最低约−6.30mm、Kd3组−7.14mm，未出现该事件样本低于−10mm。不是全部物理子步无穿透证明；柔顺接触允许压缩，也不是消除仿真接触差异的承诺。

## 5. RSI候选与运行实值

```json
{
  "status": "passed",
  "actual_RSI_cases_checked": 1977,
  "unique_frames_checked": 1977,
  "normal_velocity_max_mps": 0.1499589254347166,
  "lifted_sole_minimum_m": 0.0019999999999999983,
  "source_counts": {
    "official_rpo": 378,
    "g1": 1599
  },
  "tracking_source_counts": {
    "official_rpo": 16,
    "g1": 356
  },
  "tracking_cases": 372,
  "finite_difference_cases": 32,
  "finite_difference_max_error_mps": 2.4385145207528325e-06,
  "support_phases_by_source_and_eligibility": {
    "g1/mixed_only": {
      "left_only": 403,
      "right_only": 493,
      "both": 347
    },
    "g1/tracking_eligible": {
      "left_only": 160,
      "right_only": 64,
      "both": 132
    },
    "official_rpo/tracking_eligible": {
      "left_only": 9,
      "right_only": 7,
      "both": 0
    },
    "official_rpo/mixed_only": {
      "left_only": 97,
      "right_only": 234,
      "both": 31
    }
  },
  "clips": {
    "g1_dance1_subject1_clip3": {
      "source": "g1",
      "total": 112,
      "tracking": 0,
      "support_phases": {
        "left_only": 43,
        "right_only": 37,
        "both": 32
      }
    },
    "g1_dance1_subject2_clip3": {
      "source": "g1",
      "total": 1,
      "tracking": 0,
      "support_phases": {
        "left_only": 0,
        "right_only": 1,
        "both": 0
      }
    },
    "g1_dance1_subject3_clip3": {
      "source": "g1",
      "total": 11,
      "tracking": 0,
      "support_phases": {
        "left_only": 0,
        "right_only": 11,
        "both": 0
      }
    },
    "g1_fallAndGetUp1_subject1_clip3": {
      "source": "g1",
      "total": 76,
      "tracking": 0,
      "support_phases": {
        "left_only": 26,
        "right_only": 40,
        "both": 10
      }
    },
    "g1_fallAndGetUp1_subject5_clip3": {
      "source": "g1",
      "total": 5,
      "tracking": 0,
      "support_phases": {
        "left_only": 3,
        "right_only": 2,
        "both": 0
      }
    },
    "g1_fight1_subject5_clip3": {
      "source": "g1",
      "total": 8,
      "tracking": 0,
      "support_phases": {
        "left_only": 0,
        "right_only": 8,
        "both": 0
      }
    },
    "g1_jumps1_subject1_clip3": {
      "source": "g1",
      "total": 28,
      "tracking": 0,
      "support_phases": {
        "left_only": 4,
        "right_only": 14,
        "both": 10
      }
    },
    "g1_jumps1_subject2_clip3": {
      "source": "g1",
      "total": 3,
      "tracking": 0,
      "support_phases": {
        "left_only": 0,
        "right_only": 3,
        "both": 0
      }
    },
    "g1_jumps1_subject5_clip3": {
      "source": "g1",
      "total": 5,
      "tracking": 0,
      "support_phases": {
        "left_only": 5,
        "right_only": 0,
        "both": 0
      }
    },
    "g1_run1_subject2_clip3": {
      "source": "g1",
      "total": 15,
      "tracking": 0,
      "support_phases": {
        "left_only": 4,
        "right_only": 11,
        "both": 0
      }
    },
    "g1_sprint1_subject2_clip3": {
      "source": "g1",
      "total": 156,
      "tracking": 0,
      "support_phases": {
        "left_only": 29,
        "right_only": 66,
        "both": 61
      }
    },
    "g1_sprint1_subject4_clip3": {
      "source": "g1",
      "total": 138,
      "tracking": 70,
      "support_phases": {
        "left_only": 47,
        "right_only": 33,
        "both": 58
      },
      "tracking_support_phases": {
        "left_only": 25,
        "right_only": 1,
        "both": 44
      }
    },
    "g1_walk1_subject1_clip3": {
      "source": "g1",
      "total": 158,
      "tracking": 42,
      "support_phases": {
        "left_only": 61,
        "right_only": 75,
        "both": 22
      },
      "tracking_support_phases": {
        "left_only": 25,
        "right_only": 17,
        "both": 0
      }
    },
    "g1_walk1_subject2_clip3": {
      "source": "g1",
      "total": 358,
      "tracking": 130,
      "support_phases": {
        "left_only": 79,
        "right_only": 55,
        "both": 224
      },
      "tracking_support_phases": {
        "left_only": 32,
        "right_only": 19,
        "both": 79
      }
    },
    "g1_walk1_subject5_clip3": {
      "source": "g1",
      "total": 90,
      "tracking": 24,
      "support_phases": {
        "left_only": 47,
        "right_only": 42,
        "both": 1
      },
      "tracking_support_phases": {
        "left_only": 11,
        "right_only": 13,
        "both": 0
      }
    },
    "g1_walk2_subject1_clip3": {
      "source": "g1",
      "total": 75,
      "tracking": 29,
      "support_phases": {
        "left_only": 34,
        "right_only": 38,
        "both": 3
      },
      "tracking_support_phases": {
        "left_only": 12,
        "right_only": 14,
        "both": 3
      }
    },
    "g1_walk2_subject3_clip3": {
      "source": "g1",
      "total": 124,
      "tracking": 61,
      "support_phases": {
        "left_only": 102,
        "right_only": 15,
        "both": 7
      },
      "tracking_support_phases": {
        "left_only": 55,
        "right_only": 0,
        "both": 6
      }
    },
    "g1_walk2_subject4_clip3": {
      "source": "g1",
      "total": 236,
      "tracking": 0,
      "support_phases": {
        "left_only": 79,
        "right_only": 106,
        "both": 51
      }
    },
    "rpo_114_08": {
      "source": "official_rpo",
      "total": 6,
      "tracking": 4,
      "support_phases": {
        "left_only": 5,
        "right_only": 1,
        "both": 0
      },
      "tracking_support_phases": {
        "left_only": 4,
        "right_only": 0,
        "both": 0
      }
    },
    "rpo_114_09": {
      "source": "official_rpo",
      "total": 49,
      "tracking": 8,
      "support_phases": {
        "left_only": 29,
        "right_only": 20,
        "both": 0
      },
      "tracking_support_phases": {
        "left_only": 5,
        "right_only": 3,
        "both": 0
      }
    },
    "rpo_36_01": {
      "source": "official_rpo",
      "total": 8,
      "tracking": 0,
      "support_phases": {
        "left_only": 0,
        "right_only": 8,
        "both": 0
      }
    },
    "rpo_36_11": {
      "source": "official_rpo",
      "total": 28,
      "tracking": 4,
      "support_phases": {
        "left_only": 0,
        "right_only": 13,
        "both": 15
      },
      "tracking_support_phases": {
        "left_only": 0,
        "right_only": 4,
        "both": 0
      }
    },
    "rpo_B10_-__Walk_turn_left_45_stageii": {
      "source": "official_rpo",
      "total": 6,
      "tracking": 0,
      "support_phases": {
        "left_only": 6,
        "right_only": 0,
        "both": 0
      }
    },
    "rpo_B11_-__Walk_turn_left_135_stageii": {
      "source": "official_rpo",
      "total": 2,
      "tracking": 0,
      "support_phases": {
        "left_only": 1,
        "right_only": 1,
        "both": 0
      }
    },
    "rpo_B13_-__Walk_turn_right_90_stageii": {
      "source": "official_rpo",
      "total": 77,
      "tracking": 0,
      "support_phases": {
        "left_only": 31,
        "right_only": 46,
        "both": 0
      }
    },
    "rpo_B14_-__Walk_turn_right_45_t2_stageii": {
      "source": "official_rpo",
      "total": 35,
      "tracking": 0,
      "support_phases": {
        "left_only": 12,
        "right_only": 23,
        "both": 0
      }
    },
    "rpo_B5_-__Walk_backwards_stageii": {
      "source": "official_rpo",
      "total": 11,
      "tracking": 0,
      "support_phases": {
        "left_only": 5,
        "right_only": 6,
        "both": 0
      }
    },
    "rpo_B9_-__Walk_turn_left_90_stageii": {
      "source": "official_rpo",
      "total": 1,
      "tracking": 0,
      "support_phases": {
        "left_only": 1,
        "right_only": 0,
        "both": 0
      }
    },
    "rpo_C17_-_run_change_direction_stageii": {
      "source": "official_rpo",
      "total": 1,
      "tracking": 0,
      "support_phases": {
        "left_only": 1,
        "right_only": 0,
        "both": 0
      }
    },
    "rpo_C1_-_stand_to_run_stageii": {
      "source": "official_rpo",
      "total": 7,
      "tracking": 0,
      "support_phases": {
        "left_only": 0,
        "right_only": 0,
        "both": 7
      }
    },
    "rpo_C3_-_run_stageii": {
      "source": "official_rpo",
      "total": 1,
      "tracking": 0,
      "support_phases": {
        "left_only": 0,
        "right_only": 1,
        "both": 0
      }
    },
    "rpo_C4_-_run_to_walk_a_stageii": {
      "source": "official_rpo",
      "total": 5,
      "tracking": 0,
      "support_phases": {
        "left_only": 3,
        "right_only": 2,
        "both": 0
      }
    },
    "rpo_C5_-_walk_to_run_stageii": {
      "source": "official_rpo",
      "total": 2,
      "tracking": 0,
      "support_phases": {
        "left_only": 0,
        "right_only": 2,
        "both": 0
      }
    },
    "rpo_move_back": {
      "source": "official_rpo",
      "total": 3,
      "tracking": 0,
      "support_phases": {
        "left_only": 0,
        "right_only": 3,
        "both": 0
      }
    },
    "rpo_move_l": {
      "source": "official_rpo",
      "total": 34,
      "tracking": 0,
      "support_phases": {
        "left_only": 1,
        "right_only": 33,
        "both": 0
      }
    },
    "rpo_move_r": {
      "source": "official_rpo",
      "total": 83,
      "tracking": 0,
      "support_phases": {
        "left_only": 7,
        "right_only": 71,
        "both": 5
      }
    },
    "rpo_run_start_180_R_001__A345": {
      "source": "official_rpo",
      "total": 1,
      "tracking": 0,
      "support_phases": {
        "left_only": 1,
        "right_only": 0,
        "both": 0
      }
    },
    "rpo_run_start_180_R_001__A345_M": {
      "source": "official_rpo",
      "total": 1,
      "tracking": 0,
      "support_phases": {
        "left_only": 0,
        "right_only": 1,
        "both": 0
      }
    },
    "rpo_run_stop_180_R_001__A345": {
      "source": "official_rpo",
      "total": 5,
      "tracking": 0,
      "support_phases": {
        "left_only": 0,
        "right_only": 3,
        "both": 2
      }
    },
    "rpo_run_stop_180_R_001__A345_M": {
      "source": "official_rpo",
      "total": 5,
      "tracking": 0,
      "support_phases": {
        "left_only": 3,
        "right_only": 0,
        "both": 2
      }
    },
    "rpo_turn_r": {
      "source": "official_rpo",
      "total": 7,
      "tracking": 0,
      "support_phases": {
        "left_only": 0,
        "right_only": 7,
        "both": 0
      }
    }
  },
  "method": "all eligible frames in identical three sampler candidate pools; train-only point velocity from MuJoCo body-origin Jacobian and omega cross offset; finite integrated qpos check; static geometry, no GPU policy evaluation"
}
```

原v10普通官方1675/G11767、跟踪官方215/G1433；新普通官方378/G11599、跟踪官方16/G1356。官方跟踪起点覆盖显著收窄，且没有双足支撑起点；G1保留双足支撑。专家学习仍使用原train清单。tracking/mixed起点池不同，组间存活差异不能单独解释为tracking因果收益。

```json
{
  "status": "passed",
  "records": [
    {
      "contact_material": {
        "compliant_materials": [
          {
            "path": "/World/ground/terrain/physicsMaterial",
            "stiffness_N_per_m": 60000.0,
            "damping_Ns_per_m": 600.0,
            "restitution": 0.0,
            "restitution_combine": "multiply"
          }
        ],
        "ground_bindings": [
          {
            "collision": "/World/ground/terrain/mesh",
            "material": "/World/ground/terrain/physicsMaterial"
          }
        ]
      },
      "seed": 42,
      "num_envs": 2048,
      "tracking_envs": 1024,
      "actual_mass_kg": {
        "min": 32.143978118896484,
        "mean": 33.76920121349394,
        "max": 35.75967788696289
      },
      "body_weight_N": {
        "min": 315.3324253463745,
        "mean": 331.27586390437557,
        "max": 350.802440071106
      },
      "nominal_mass_kg": 33.756893157958984,
      "partition_sha256": "5b653aad83c9abf3cbcb14748009093b85939c3186041e13b527d8370463637d",
      "actuator_gains": {
        "legs": {
          "stiffness": [
            90.00133514404297,
            164.9881591796875
          ],
          "damping": [
            2.970017671585083,
            5.499892711639404
          ],
          "effort_limit": [
            120.0,
            120.0
          ]
        },
        "feet": {
          "stiffness": [
            36.00110626220703,
            43.998382568359375
          ],
          "damping": [
            2.250110149383545,
            2.749948501586914
          ],
          "effort_limit": [
            27.0,
            27.0
          ]
        },
        "shoulders": {
          "stiffness": [
            36.00059127807617,
            43.99980545043945
          ],
          "damping": [
            1.8000330924987793,
            2.199772357940674
          ],
          "effort_limit": [
            27.0,
            27.0
          ]
        },
        "arms": {
          "stiffness": [
            18.000343322753906,
            32.99989700317383
          ],
          "damping": [
            0.9000027179718018,
            1.6499016284942627
          ],
          "effort_limit": [
            27.0,
            27.0
          ]
        }
      }
    },
    {
      "contact_material": {
        "compliant_materials": [
          {
            "path": "/World/ground/terrain/physicsMaterial",
            "stiffness_N_per_m": 60000.0,
            "damping_Ns_per_m": 600.0,
            "restitution": 0.0,
            "restitution_combine": "multiply"
          }
        ],
        "ground_bindings": [
          {
            "collision": "/World/ground/terrain/mesh",
            "material": "/World/ground/terrain/physicsMaterial"
          }
        ]
      },
      "seed": 44,
      "num_envs": 2048,
      "tracking_envs": 1024,
      "actual_mass_kg": {
        "min": 32.017066955566406,
        "mean": 33.76246723160148,
        "max": 35.603492736816406
      },
      "body_weight_N": {
        "min": 314.08742683410645,
        "mean": 331.2098035420105,
        "max": 349.270263748169
      },
      "nominal_mass_kg": 33.756893157958984,
      "partition_sha256": "9bcdb64f3a3805ddf44b0aa6f6581dec9b67da4cfc80e0e2336669b4b9399fc6",
      "actuator_gains": {
        "legs": {
          "stiffness": [
            90.00326538085938,
            164.9942626953125
          ],
          "damping": [
            2.970034599304199,
            5.49940824508667
          ],
          "effort_limit": [
            120.0,
            120.0
          ]
        },
        "feet": {
          "stiffness": [
            36.000938415527344,
            43.99989700317383
          ],
          "damping": [
            2.2500736713409424,
            2.7499606609344482
          ],
          "effort_limit": [
            27.0,
            27.0
          ]
        },
        "shoulders": {
          "stiffness": [
            36.00021743774414,
            43.99967956542969
          ],
          "damping": [
            1.8000295162200928,
            2.199985980987549
          ],
          "effort_limit": [
            27.0,
            27.0
          ]
        },
        "arms": {
          "stiffness": [
            18.000181198120117,
            32.99878692626953
          ],
          "damping": [
            0.9000183939933777,
            1.6499254703521729
          ],
          "effort_limit": [
            27.0,
            27.0
          ]
        }
      }
    },
    {
      "contact_material": {
        "compliant_materials": [
          {
            "path": "/World/ground/terrain/physicsMaterial",
            "stiffness_N_per_m": 60000.0,
            "damping_Ns_per_m": 600.0,
            "restitution": 0.0,
            "restitution_combine": "multiply"
          }
        ],
        "ground_bindings": [
          {
            "collision": "/World/ground/terrain/mesh",
            "material": "/World/ground/terrain/physicsMaterial"
          }
        ]
      },
      "seed": 43,
      "num_envs": 2048,
      "tracking_envs": 1024,
      "actual_mass_kg": {
        "min": 31.873842239379883,
        "mean": 33.737000557594,
        "max": 35.469852447509766
      },
      "body_weight_N": {
        "min": 312.68239236831664,
        "mean": 330.9599754699972,
        "max": 347.9592525100708
      },
      "nominal_mass_kg": 33.756893157958984,
      "partition_sha256": "4aedbad59e9db02006a65da4702725c2f8aee6c4fe1e1cef3a7d516366d8e0c9",
      "actuator_gains": {
        "legs": {
          "stiffness": [
            90.0009994506836,
            164.99708557128906
          ],
          "damping": [
            2.970015525817871,
            5.499982833862305
          ],
          "effort_limit": [
            120.0,
            120.0
          ]
        },
        "feet": {
          "stiffness": [
            36.00077819824219,
            43.996585845947266
          ],
          "damping": [
            2.2501516342163086,
            2.749945640563965
          ],
          "effort_limit": [
            27.0,
            27.0
          ]
        },
        "shoulders": {
          "stiffness": [
            36.00245666503906,
            43.999820709228516
          ],
          "damping": [
            1.8000295162200928,
            2.199965238571167
          ],
          "effort_limit": [
            27.0,
            27.0
          ]
        },
        "arms": {
          "stiffness": [
            18.000438690185547,
            32.99958419799805
          ],
          "damping": [
            0.9001254439353943,
            1.6499431133270264
          ],
          "effort_limit": [
            27.0,
            27.0
          ]
        }
      }
    }
  ],
  "total_envs": 6144,
  "tracking_unique_env_fraction": 0.5
}
```

## 6. 五个100更新窗口及基线对照

| 指标 | v10五窗口均值 | v11五窗口均值 | v11末值 | v11全段极大值 |
| --- | --- | --- | --- | --- |
| env/episode_len | [8.254514, 11.558163, 15.194496, 19.221669, 22.31336] | [7.799224, 12.087714, 15.311109, 19.654349, 22.636511] | 22.791828155517578 | 23.22370719909668 |
| train/ufo/strict_shadow_episode_len | [7.367988, 9.212812, 11.775847, 14.734676, 16.014141] | [5.563082, 8.086989, 10.619134, 12.869199, 13.074938] | 13.203307151794434 | 13.610203742980957 |
| train/ufo/critic_loss | [0.803934, 0.305527, 0.300615, 0.336857, 0.441099] | [0.808055, 0.284091, 0.243606, 0.29745, 0.381114] | 0.4776987135410309 | 2.647822141647339 |
| train/ufo/Q_fb | [6.171036, 16.52731, 27.027091, 38.958132, 52.201909] | [7.254135, 17.933306, 29.383052, 41.436655, 54.466805] | 61.627197265625 | 63.03118896484375 |
| train/ufo/mean_disc_reward | [0.034172, -0.150873, -0.506377, -0.772452, -0.95453] | [0.016213, -0.191453, -0.518128, -0.758183, -0.929651] | -0.9982694983482361 | 0.09344223886728287 |
| train/ufo/applied_action_saturation_fraction | [0.177575, 0.264612, 0.293047, 0.292409, 0.264446] | [0.17558, 0.257273, 0.28536, 0.284785, 0.270839] | 0.2664388120174408 | 0.2899930775165558 |
| train/ufo/physics_peak_normalized_velocity | [3.86503, 4.349701, 4.866824, 4.511645, 3.609909] | [2.067454, 2.962816, 2.282729, 3.379741, 2.375958] | 1.4925495386123657 | 6.2884321212768555 |

这里的曲线窗口均值是原始日志采样点均值，单次rollout记录点数量可能不同；与门禁按episode数加权的统计口径分开。两版本相同初始化种子，训练起点分布/接触/奖励联合变化，因此学习效果不是单因素因果比较。

逐采样round的原始记录（未插值；native loss取该真实更新点值）：

| step | env/episode_len | train/ufo/strict_shadow_episode_len | train/ufo/physics_peak_normalized_velocity | train/ufo/strict_velocity_exposure_fraction | train/ufo/first_step_samples | train/ufo/first_step_terminated | train/ufo/critic_loss | train/ufo/actor_loss | train/ufo/Q_fb | train/ufo/mean_disc_reward | train/ufo/aux_rew/limits_dof_vel | train/ufo/applied_action_saturation_fraction | train/ufo/hard_position_count | train/ufo/hard_velocity_count | train/ufo/foot_contact_peak_N |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 40 | 3.1857142448425293 | 2.671428680419922 | 1.833019733428955 | 0.001970152836292982 | 6184.0 | 4.0 | 1.0559478998184204 | -1.571272373199463 | 6.246171951293945 | 0.08839237689971924 | 0.00900826882570982 | 0.1082533672451973 | 29.0 | 3568.0 | 3510.235595703125 |
| 56 | 7.201652526855469 | 4.529047012329102 | 1.952716588973999 | 0.0002702353522181511 | 2505.0 | 2.0 | 0.7019549012184143 | 0.3482656478881836 | 3.9910402297973633 | 0.03617357090115547 | 0.009192977100610733 | 0.18603868782520294 | 222.0 | 575.0 | 5079.66552734375 |
| 72 | 9.118669509887695 | 7.333664417266846 | 1.6405922174453735 | 0.0014168559573590755 | 3165.0 | 1.0 | 0.6426645517349243 | -2.475203514099121 | 6.720450401306152 | 0.026944682002067566 | 0.011040829122066498 | 0.15283556282520294 | 378.0 | 2597.0 | 7319.94091796875 |
| 88 | 8.227216720581055 | 5.479175090789795 | 1.91213858127594 | 0.0007122979732230306 | 1817.0 | 0.0 | 0.48697414994239807 | -1.4965028762817383 | 5.570026874542236 | -0.006879772059619427 | 0.011446951888501644 | 0.21018914878368378 | 308.0 | 1510.0 | 7719.78515625 |
| 104 | 9.341777801513672 | 6.935215950012207 | 2.7381625175476074 | 0.0007282201549969614 | 2657.0 | 2.0 | 0.3751560151576996 | -5.993256568908691 | 10.114767074584961 | -0.010732224211096764 | 0.014348361641168594 | 0.18827134370803833 | 414.0 | 1575.0 | 7359.611328125 |
| 120 | 9.720314979553223 | 6.429963111877441 | 2.3280959129333496 | 0.0008160134893842041 | 2274.0 | 1.0 | 0.31656330823898315 | -8.54941177368164 | 13.273822784423828 | -0.04563060402870178 | 0.015140890143811703 | 0.20789282023906708 | 477.0 | 1755.0 | 8190.9990234375 |
| 136 | 9.612031936645508 | 6.550449848175049 | 2.6265759468078613 | 0.000933439820073545 | 2027.0 | 1.0 | 0.309021919965744 | -9.079723358154297 | 13.86994743347168 | -0.07379008829593658 | 0.014065753668546677 | 0.22337806224822998 | 481.0 | 1958.0 | 6785.62548828125 |
| 152 | 10.431313514709473 | 6.905050277709961 | 2.682100296020508 | 0.0007306527113541961 | 2001.0 | 3.0 | 0.3614759147167206 | -9.007698059082031 | 14.423483848571777 | -0.11812300980091095 | 0.015023276209831238 | 0.248052179813385 | 415.0 | 1579.0 | 7442.64013671875 |
| 168 | 11.2650728225708 | 7.841475963592529 | 2.075549602508545 | 0.0006048229988664389 | 2005.0 | 0.0 | 0.2667776346206665 | -10.967958450317383 | 16.826568603515625 | -0.18355002999305725 | 0.015107899904251099 | 0.25790271162986755 | 365.0 | 1314.0 | 5901.9130859375 |
| 184 | 13.460546493530273 | 8.982980728149414 | 3.512829303741455 | 0.0005205679917708039 | 1863.0 | 1.0 | 0.26052629947662354 | -11.277929306030273 | 17.442474365234375 | -0.23521851003170013 | 0.015908606350421906 | 0.26465728878974915 | 231.0 | 1187.0 | 6930.65283203125 |
| 200 | 13.923030853271484 | 8.821407318115234 | 2.1075563430786133 | 0.0005676712025888264 | 1754.0 | 0.0 | 0.25285276770591736 | -14.412653923034668 | 21.68156623840332 | -0.27122223377227783 | 0.015889044851064682 | 0.27199918031692505 | 343.0 | 1288.0 | 8154.666015625 |
| 216 | 13.834285736083984 | 9.420571327209473 | 4.772284507751465 | 0.0004843006900046021 | 1755.0 | 2.0 | 0.23377783596515656 | -14.012401580810547 | 21.78111457824707 | -0.3762426972389221 | 0.01607447676360607 | 0.27764803171157837 | 351.0 | 1097.0 | 8039.21875 |
| 232 | 13.70075798034668 | 9.85732364654541 | 2.077119827270508 | 0.0004829738463740796 | 1650.0 | 0.0 | 0.254332035779953 | -14.849370956420898 | 22.836410522460938 | -0.3794673681259155 | 0.015269629657268524 | 0.27828845381736755 | 285.0 | 1120.0 | 9361.689453125 |
| 248 | 14.301576614379883 | 10.160384178161621 | 3.4856796264648438 | 0.0004745704645756632 | 1504.0 | 2.0 | 0.2622002363204956 | -16.015350341796875 | 24.99292755126953 | -0.4385829567909241 | 0.01623043790459633 | 0.28929072618484497 | 275.0 | 1125.0 | 8028.587890625 |
| 264 | 15.009482383728027 | 10.207877159118652 | 1.6859996318817139 | 0.00043189007556065917 | 1373.0 | 1.0 | 0.2562481760978699 | -17.560148239135742 | 27.12323760986328 | -0.48648741841316223 | 0.01566389948129654 | 0.2852853834629059 | 257.0 | 943.0 | 7580.47314453125 |
| 280 | 15.450572967529297 | 10.573781967163086 | 2.059439182281494 | 0.00045311969006434083 | 1373.0 | 1.0 | 0.3031982183456421 | -20.164405822753906 | 30.89297103881836 | -0.5679488182067871 | 0.015499576926231384 | 0.2885512113571167 | 291.0 | 1104.0 | 8059.97314453125 |
| 296 | 15.579812049865723 | 10.73552417755127 | 2.380284070968628 | 0.0004449374391697347 | 1340.0 | 1.0 | 0.22299714386463165 | -20.797494888305664 | 32.120994567871094 | -0.5713784098625183 | 0.015364961698651314 | 0.28743666410446167 | 229.0 | 1033.0 | 12074.6611328125 |
| 312 | 17.824453353881836 | 12.179912567138672 | 2.007852554321289 | 0.00046085965004749596 | 1167.0 | 1.0 | 0.2064220905303955 | -19.98568344116211 | 31.098581314086914 | -0.6100582480430603 | 0.0160803385078907 | 0.2833074927330017 | 232.0 | 1053.0 | 8485.8125 |
| 328 | 18.96156883239746 | 13.600480079650879 | 6.2884321212768555 | 0.0004020359192509204 | 1205.0 | 0.0 | 0.29150885343551636 | -22.22825050354004 | 35.09458923339844 | -0.669844388961792 | 0.01529705710709095 | 0.2767210006713867 | 207.0 | 972.0 | 8192.8134765625 |
| 344 | 18.3632755279541 | 13.321939468383789 | 1.8982079029083252 | 0.00048076242092065513 | 1262.0 | 0.0 | 0.25082913041114807 | -23.292381286621094 | 37.97174072265625 | -0.7059298157691956 | 0.014926397241652012 | 0.27635833621025085 | 265.0 | 1142.0 | 9629.5908203125 |
| 360 | 18.573211669921875 | 12.780365943908691 | 3.5435667037963867 | 0.0006508204969577491 | 1216.0 | 1.0 | 0.25912395119667053 | -24.498435974121094 | 38.95376205444336 | -0.7411289215087891 | 0.016426702961325645 | 0.2891191244125366 | 321.0 | 1451.0 | 8989.3251953125 |
| 376 | 19.18707847595215 | 12.420443534851074 | 2.00091290473938 | 0.0004935886827297509 | 1117.0 | 1.0 | 0.3900824785232544 | -27.045297622680664 | 43.16084289550781 | -0.7795685529708862 | 0.0160063523799181 | 0.2899930775165558 | 241.0 | 1200.0 | 6618.95751953125 |
| 392 | 20.52988052368164 | 12.916335105895996 | 5.32218074798584 | 0.0005128280026838183 | 960.0 | 1.0 | 0.2913796901702881 | -26.96279525756836 | 43.60523223876953 | -0.7741951942443848 | 0.016079474240541458 | 0.28626903891563416 | 238.0 | 1141.0 | 6689.94287109375 |
| 408 | 20.367578506469727 | 12.620169639587402 | 2.9849610328674316 | 0.0005075205699540675 | 1091.0 | 1.0 | 0.434183806180954 | -27.0537109375 | 44.04436492919922 | -0.8176062703132629 | 0.01393628679215908 | 0.28663700819015503 | 241.0 | 1049.0 | 8428.9345703125 |
| 424 | 21.59784698486328 | 12.424657821655273 | 1.6199266910552979 | 0.0004462642828002572 | 962.0 | 0.0 | 0.27372002601623535 | -30.439407348632812 | 47.605003356933594 | -0.8442583084106445 | 0.015432368963956833 | 0.2884008288383484 | 196.0 | 977.0 | 7081.76953125 |
| 440 | 22.75337028503418 | 13.080095291137695 | 1.7014997005462646 | 0.0005002229008823633 | 1236.0 | 1.0 | 0.3954188823699951 | -35.23042297363281 | 52.06388854980469 | -0.8938184976577759 | 0.014059552922844887 | 0.2832261323928833 | 348.0 | 1131.0 | 6646.70654296875 |
| 456 | 22.276578903198242 | 13.545963287353516 | 2.2211244106292725 | 0.0005517489626072347 | 1270.0 | 2.0 | 0.3462464213371277 | -34.39678955078125 | 52.49021530151367 | -0.9459266662597656 | 0.014942956157028675 | 0.2754419445991516 | 267.0 | 1211.0 | 7636.841796875 |
| 472 | 23.22370719909668 | 13.316399574279785 | 1.849245309829712 | 0.0004425048828125 | 1118.0 | 4.0 | 0.3537561297416687 | -37.03905487060547 | 55.2464599609375 | -0.8956379890441895 | 0.0144658163189888 | 0.26478642225265503 | 198.0 | 1050.0 | 5958.4130859375 |
| 488 | 22.41391944885254 | 12.4981689453125 | 3.424905776977539 | 0.00045710025005973876 | 1120.0 | 2.0 | 0.33192265033721924 | -35.217185974121094 | 53.4409294128418 | -0.965218186378479 | 0.014927875250577927 | 0.2681000232696533 | 186.0 | 1087.0 | 8876.333984375 |
| 504 | 21.91147804260254 | 12.270427703857422 | 3.225222587585449 | 0.0005024343263357878 | 1061.0 | 0.0 | 0.5238779783248901 | -38.43212127685547 | 56.41753005981445 | -0.9553075432777405 | 0.01500930730253458 | 0.26479703187942505 | 200.0 | 1101.0 | 7700.474609375 |
| 520 | 23.084693908691406 | 13.610203742980957 | 2.7171597480773926 | 0.0004369763482827693 | 999.0 | 0.0 | 0.4008362591266632 | -40.469703674316406 | 59.06132507324219 | -0.9602982401847839 | 0.01498386263847351 | 0.27308014035224915 | 190.0 | 941.0 | 5947.80908203125 |
| 524 | 22.791828155517578 | 13.203307151794434 | 1.4925495386123657 | 0.00042127526830881834 | 1010.0 | 3.0 | 0.4776987135410309 | -42.602561950683594 | 61.627197265625 | -0.9982694983482361 | 0.016782240942120552 | 0.2664388120174408 | 232.0 | 991.0 | 6209.00146484375 |

native aux_rew是learner回放抽样批次统计；下节trace raw均值来自全部正式新转移，包含时间分布差异，两者不要求数值相等。

## 7. 物理事故、首步、存活、右删失与分组

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
  "requested_saturation_fraction": 0.6127929134645324,
  "applied_saturation_fraction": 0.25620772873145947,
  "strict_position_velocity_torque_counts": [
    8913,
    41825,
    0
  ],
  "training_position_velocity_torque_counts": [
    8913,
    651,
    0
  ],
  "macro_contact_height_tilt_counts": [
    6491,
    13309,
    21097
  ],
  "boundary_max_radps": 42.57744598388672,
  "physics_max_radps": 50.307456970214844,
  "physics_max_rated_ratio": 6.2884321212768555,
  "boundary_max_event": {
    "round": 26,
    "row": 6383,
    "episode_id_run_local": 4009754631,
    "episode_step": 22,
    "joint": "left_ankle_pitch_joint",
    "rated_radps": 8.0,
    "boundary_dq_radps": 42.57744598388672,
    "physics_signed_dq_radps": 42.57744598388672,
    "physics_abs_dq_radps": 42.57744598388672,
    "peak_substep_0_to_7": 7,
    "q_at_joint_peak_rad": 0.12332694977521896,
    "torque_at_joint_peak_Nm": 0.4442014694213867,
    "target_at_joint_peak_rad": -0.034565865993499756,
    "requested_action": -0.8527871370315552,
    "applied_action": -0.057609766721725464,
    "foot_peak_vertical_net_force_N": [
      1711.625244140625,
      0.0
    ],
    "foot_vertical_net_force_integral_Ns": [
      4.2790632247924805,
      0.0
    ],
    "global_peak_joint": "left_ankle_pitch_joint",
    "feet_at_global_abs_dq_peak_origin_pose_wxyz_linear_angular": [
      [
        43.4792366027832,
        -19.98497772216797,
        0.10328905284404755,
        0.016146453097462654,
        -0.4206344783306122,
        0.15310418605804443,
        0.8940721154212952,
        2.7455945014953613,
        1.0421054363250732,
        1.8044300079345703,
        4.975884437561035,
        -53.99490737915039,
        9.042741775512695
      ],
      [
        43.05767822265625,
        -19.970596313476562,
        0.10215520858764648,
        0.2094583958387375,
        0.5337865948677063,
        -0.05265013873577118,
        0.8175737261772156,
        -1.3410906791687012,
        -0.4202544093132019,
        0.6293606758117676,
        -1.4088149070739746,
        5.397313594818115,
        -0.9291974902153015
      ]
    ],
    "feet_vertical_net_force_at_global_peak_N": [
      1711.625244140625,
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
    "round": 22,
    "row": 23805,
    "episode_id_run_local": 2220447760393,
    "episode_step": 9,
    "joint": "right_ankle_pitch_joint",
    "rated_radps": 8.0,
    "boundary_dq_radps": -11.702563285827637,
    "physics_signed_dq_radps": 50.307456970214844,
    "physics_abs_dq_radps": 50.307456970214844,
    "peak_substep_0_to_7": 6,
    "q_at_joint_peak_rad": -0.4840656518936157,
    "torque_at_joint_peak_Nm": -1.0797102451324463,
    "target_at_joint_peak_rad": -0.47293347120285034,
    "requested_action": -0.7882224321365356,
    "applied_action": -0.7882224321365356,
    "foot_peak_vertical_net_force_N": [
      0.0,
      5520.546875
    ],
    "foot_vertical_net_force_integral_Ns": [
      0.0,
      26.96402359008789
    ],
    "global_peak_joint": "right_ankle_pitch_joint",
    "feet_at_global_abs_dq_peak_origin_pose_wxyz_linear_angular": [
      [
        -13.618943214416504,
        -10.397953033447266,
        0.2570202946662903,
        -0.08409515768289566,
        -0.8591628670692444,
        0.13077419996261597,
        0.4875093698501587,
        0.9607690572738647,
        0.8620430827140808,
        -2.439890146255493,
        0.7785590887069702,
        3.9524929523468018,
        4.046760559082031
      ],
      [
        -13.773435592651367,
        -10.280823707580566,
        0.0872160941362381,
        0.5941393375396729,
        -0.08909041434526443,
        0.3592461347579956,
        0.714145302772522,
        -1.048231601715088,
        -2.953230142593384,
        2.3656721115112305,
        -73.40380859375,
        7.416736125946045,
        -6.327365398406982
      ]
    ],
    "feet_vertical_net_force_at_global_peak_N": [
      0.0,
      5520.546875
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
    "round": 22,
    "row": 23805,
    "episode_id_run_local": 2220447760393,
    "episode_step": 9,
    "joint": "right_ankle_pitch_joint",
    "rated_radps": 8.0,
    "boundary_dq_radps": -11.702563285827637,
    "physics_signed_dq_radps": 50.307456970214844,
    "physics_abs_dq_radps": 50.307456970214844,
    "peak_substep_0_to_7": 6,
    "q_at_joint_peak_rad": -0.4840656518936157,
    "torque_at_joint_peak_Nm": -1.0797102451324463,
    "target_at_joint_peak_rad": -0.47293347120285034,
    "requested_action": -0.7882224321365356,
    "applied_action": -0.7882224321365356,
    "foot_peak_vertical_net_force_N": [
      0.0,
      5520.546875
    ],
    "foot_vertical_net_force_integral_Ns": [
      0.0,
      26.96402359008789
    ],
    "global_peak_joint": "right_ankle_pitch_joint",
    "feet_at_global_abs_dq_peak_origin_pose_wxyz_linear_angular": [
      [
        -13.618943214416504,
        -10.397953033447266,
        0.2570202946662903,
        -0.08409515768289566,
        -0.8591628670692444,
        0.13077419996261597,
        0.4875093698501587,
        0.9607690572738647,
        0.8620430827140808,
        -2.439890146255493,
        0.7785590887069702,
        3.9524929523468018,
        4.046760559082031
      ],
      [
        -13.773435592651367,
        -10.280823707580566,
        0.0872160941362381,
        0.5941393375396729,
        -0.08909041434526443,
        0.3592461347579956,
        0.714145302772522,
        -1.048231601715088,
        -2.953230142593384,
        2.3656721115112305,
        -73.40380859375,
        7.416736125946045,
        -6.327365398406982
      ]
    ],
    "feet_vertical_net_force_at_global_peak_N": [
      0.0,
      5520.546875
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
  "foot_vertical_net_force_peak_N": 12074.6611328125,
  "first_step_foot_vertical_net_force_N": {
    "n": 54041,
    "mean": 77.13948059082031,
    "median": 0.0,
    "p95": 484.6558532714844,
    "p99": 777.0605834960936,
    "max": 1760.511962890625
  },
  "first_step_foot_absolute_vertical_net_force_integral_Ns": {
    "n": 54041,
    "mean": 0.4921712577342987,
    "median": 0.0,
    "p95": 3.2138822078704834,
    "p99": 5.619148731231676,
    "max": 10.409579277038574
  },
  "reset_lift_m": {
    "n": 54041,
    "mean": 0.003020409494638443,
    "median": 0.0,
    "p95": 0.01811346784234047,
    "p99": 0.020991584286093712,
    "max": 0.021992022171616554
  },
  "resets": 54041,
  "first_step_terminated": 39,
  "first_step_termination_fraction": 0.0007216742843396681,
  "first_step_strict_counts": [
    0,
    97,
    0
  ],
  "first_step_macro_counts": [
    39,
    0,
    0
  ],
  "raw_velocity_penalty_mean": 0.015152744910058876,
  "weighted_velocity_penalty_mean": -0.3788186227514719,
  "strict_velocity_substep_joint_exposure_fraction": 0.0006034270576808764,
  "velocity_reward_coverage_miss_count": 0,
  "latched_velocity_terms_without_boundary_violation": 33248,
  "reset_q_preservation_max_error_rad": 8.821487429422348e-08,
  "reset_dq_preservation_max_error_radps": 0.0,
  "decoder_max_error_rad": 2.8342008562276533e-07,
  "ground_minimum_margin_m": 193.34695434570312,
  "ground_outside_count": 0,
  "mixed_nonperiodic_z_equal_pairs": 363974,
  "mixed_nonperiodic_cosine": {
    "n": 363974,
    "mean": 1.0,
    "median": 1.0,
    "p95": 1.0000001192092896,
    "p99": 1.0000001192092896,
    "max": 1.0000001192092896
  },
  "tracking_unexpected_refreshes": 0,
  "tracking_phase_mismatches": 0,
  "tracking_window_stable_pairs": 368417,
  "tracking_adjacent_phase_cosine": {
    "n": 368417,
    "mean": 0.9969742298126221,
    "median": 0.9973975419998169,
    "p95": 0.9996954202651978,
    "p99": 0.9998809695243835,
    "max": 0.9999691843986511
  },
  "tracking_window_renewals": 0,
  "terminal_following_new_episode_pairs": 47897,
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
    "n": 48141,
    "mean": 14.540059408819925,
    "median": 12.0,
    "p95": 33.0,
    "p99": 45.0,
    "max": 84.0
  },
  "completed_strict_shadow_len": {
    "n": 48141,
    "mean": 9.492262312789514,
    "median": 8.0,
    "p95": 26.0,
    "p99": 35.0,
    "max": 64.0
  },
  "completed_tracking_shadow_len": {
    "n": 21827,
    "mean": 10.414807348696568,
    "median": 8.0,
    "p95": 28.0,
    "p99": 36.0,
    "max": 62.0
  },
  "completed_mixed_shadow_len": {
    "n": 26314,
    "mean": 8.727027437865775,
    "median": 8.0,
    "p95": 22.0,
    "p99": 32.0,
    "max": 64.0
  },
  "unfinished_train_episodes_right_censored": 5900,
  "unfinished_strict_shadow_right_censored": 3232,
  "unfinished_train_observed_lengths": {
    "n": 5900,
    "mean": 14.65406779661017,
    "median": 13.0,
    "p95": 33.0,
    "p99": 50.0,
    "max": 88.0
  },
  "RSI_source_by_rollout_mode": {
    "official_rpo/mixed": {
      "rows": 200807,
      "resets": 14622,
      "first_terms": 0,
      "strict_velocity_rows": 10415,
      "completed_episode_len": {
        "n": 13227,
        "mean": 13.908369244726696,
        "median": 12.0,
        "p95": 28.0,
        "p99": 39.0,
        "max": 84.0
      },
      "completed_strict_shadow_len": {
        "n": 13227,
        "mean": 9.69902472215922,
        "median": 9.0,
        "p95": 22.0,
        "p99": 32.0,
        "max": 64.0
      },
      "first_term_fraction": 0.0,
      "strict_velocity_substep_joint_exposure_fraction": 0.0006321774249598522,
      "raw_velocity_penalty_mean": 0.01886333696634665
    },
    "official_rpo/tracking": {
      "rows": 216425,
      "resets": 12383,
      "first_terms": 0,
      "strict_velocity_rows": 9492,
      "completed_episode_len": {
        "n": 10761,
        "mean": 17.584239382956973,
        "median": 15.0,
        "p95": 35.0,
        "p99": 48.0,
        "max": 77.0
      },
      "completed_strict_shadow_len": {
        "n": 10761,
        "mean": 12.876405538518725,
        "median": 11.0,
        "p95": 29.0,
        "p99": 35.0,
        "max": 62.0
      },
      "first_term_fraction": 0.0,
      "strict_velocity_substep_joint_exposure_fraction": 0.0005155415823334722,
      "raw_velocity_penalty_mean": 0.016591235435462044
    },
    "g1/mixed": {
      "rows": 192409,
      "resets": 14620,
      "first_terms": 39,
      "strict_velocity_rows": 11195,
      "completed_episode_len": {
        "n": 13087,
        "mean": 13.059295484068159,
        "median": 10.0,
        "p95": 31.0,
        "p99": 42.0,
        "max": 75.0
      },
      "completed_strict_shadow_len": {
        "n": 13087,
        "mean": 7.744632077634294,
        "median": 6.0,
        "p95": 22.0,
        "p99": 33.0,
        "max": 60.0
      },
      "first_term_fraction": 0.002667578659370725,
      "strict_velocity_substep_joint_exposure_fraction": 0.0006225981022762426,
      "raw_velocity_penalty_mean": 0.01388288942353918
    },
    "g1/tracking": {
      "rows": 176791,
      "resets": 12416,
      "first_terms": 0,
      "strict_velocity_rows": 10723,
      "completed_episode_len": {
        "n": 11066,
        "mean": 14.086029278872221,
        "median": 9.0,
        "p95": 37.0,
        "p99": 49.0,
        "max": 84.0
      },
      "completed_strict_shadow_len": {
        "n": 11066,
        "mean": 8.021055485270198,
        "median": 5.0,
        "p95": 27.0,
        "p99": 37.0,
        "max": 60.0
      },
      "first_term_fraction": 0.0,
      "strict_velocity_substep_joint_exposure_fraction": 0.0006574946147416022,
      "raw_velocity_penalty_mean": 0.010559147734556016
    }
  },
  "RSI_source_mode_by_nominal_reset_support_phase": {
    "official_rpo/mixed/left_only": {
      "rows": 54571,
      "resets": 4109,
      "first_terms": 0,
      "completed_episode_len": {
        "n": 3725,
        "mean": 13.480805369127516,
        "median": 12.0,
        "p95": 26.0,
        "p99": 37.0,
        "max": 57.0
      },
      "completed_strict_shadow_len": {
        "n": 3725,
        "mean": 9.124832214765101,
        "median": 9.0,
        "p95": 20.0,
        "p99": 29.0,
        "max": 49.0
      },
      "first_term_fraction": 0.0
    },
    "official_rpo/mixed/right_only": {
      "rows": 128862,
      "resets": 9321,
      "first_terms": 0,
      "completed_episode_len": {
        "n": 8444,
        "mean": 13.972761724301279,
        "median": 12.0,
        "p95": 28.0,
        "p99": 39.0,
        "max": 84.0
      },
      "completed_strict_shadow_len": {
        "n": 8444,
        "mean": 9.716248223590716,
        "median": 9.0,
        "p95": 22.0,
        "p99": 32.0,
        "max": 64.0
      },
      "first_term_fraction": 0.0
    },
    "official_rpo/mixed/both": {
      "rows": 17374,
      "resets": 1192,
      "first_terms": 0,
      "completed_episode_len": {
        "n": 1058,
        "mean": 14.899810964083176,
        "median": 12.0,
        "p95": 31.0,
        "p99": 39.430000000000064,
        "max": 54.0
      },
      "completed_strict_shadow_len": {
        "n": 1058,
        "mean": 11.583175803402646,
        "median": 10.0,
        "p95": 26.0,
        "p99": 34.0,
        "max": 54.0
      },
      "first_term_fraction": 0.0
    },
    "official_rpo/tracking/left_only": {
      "rows": 109563,
      "resets": 6931,
      "first_terms": 0,
      "completed_episode_len": {
        "n": 5972,
        "mean": 15.562123241795044,
        "median": 13.0,
        "p95": 34.0,
        "p99": 44.0,
        "max": 77.0
      },
      "completed_strict_shadow_len": {
        "n": 5972,
        "mean": 10.860013395847288,
        "median": 10.0,
        "p95": 26.0,
        "p99": 35.0,
        "max": 62.0
      },
      "first_term_fraction": 0.0
    },
    "official_rpo/tracking/right_only": {
      "rows": 106862,
      "resets": 5452,
      "first_terms": 0,
      "completed_episode_len": {
        "n": 4789,
        "mean": 20.105867613280434,
        "median": 21.0,
        "p95": 36.0,
        "p99": 50.0,
        "max": 67.0
      },
      "completed_strict_shadow_len": {
        "n": 4789,
        "mean": 15.390895802881603,
        "median": 13.0,
        "p95": 29.0,
        "p99": 34.0,
        "max": 57.0
      },
      "first_term_fraction": 0.0
    },
    "official_rpo/tracking/both": {
      "rows": 0,
      "resets": 0,
      "first_terms": 0,
      "completed_episode_len": {
        "n": 0
      },
      "completed_strict_shadow_len": {
        "n": 0
      },
      "first_term_fraction": null
    },
    "g1/mixed/left_only": {
      "rows": 68470,
      "resets": 5177,
      "first_terms": 0,
      "completed_episode_len": {
        "n": 4625,
        "mean": 13.127567567567567,
        "median": 10.0,
        "p95": 31.0,
        "p99": 42.0,
        "max": 75.0
      },
      "completed_strict_shadow_len": {
        "n": 4625,
        "mean": 7.270918918918919,
        "median": 5.0,
        "p95": 21.0,
        "p99": 32.76000000000022,
        "max": 51.0
      },
      "first_term_fraction": 0.0
    },
    "g1/mixed/right_only": {
      "rows": 63854,
      "resets": 5015,
      "first_terms": 39,
      "completed_episode_len": {
        "n": 4514,
        "mean": 12.656180770934869,
        "median": 10.0,
        "p95": 30.0,
        "p99": 41.0,
        "max": 68.0
      },
      "completed_strict_shadow_len": {
        "n": 4514,
        "mean": 7.717323881258308,
        "median": 6.0,
        "p95": 21.0,
        "p99": 32.86999999999989,
        "max": 60.0
      },
      "first_term_fraction": 0.00777666999002991
    },
    "g1/mixed/both": {
      "rows": 60085,
      "resets": 4428,
      "first_terms": 0,
      "completed_episode_len": {
        "n": 3948,
        "mean": 13.440222897669706,
        "median": 11.0,
        "p95": 31.0,
        "p99": 41.0,
        "max": 57.0
      },
      "completed_strict_shadow_len": {
        "n": 3948,
        "mean": 8.33080040526849,
        "median": 7.0,
        "p95": 23.0,
        "p99": 33.529999999999745,
        "max": 55.0
      },
      "first_term_fraction": 0.0
    },
    "g1/tracking/left_only": {
      "rows": 76849,
      "resets": 5651,
      "first_terms": 0,
      "completed_episode_len": {
        "n": 5042,
        "mean": 13.362554541848473,
        "median": 9.0,
        "p95": 34.0,
        "p99": 46.590000000000146,
        "max": 78.0
      },
      "completed_strict_shadow_len": {
        "n": 5042,
        "mean": 7.225109083696946,
        "median": 4.0,
        "p95": 24.0,
        "p99": 34.590000000000146,
        "max": 57.0
      },
      "first_term_fraction": 0.0
    },
    "g1/tracking/right_only": {
      "rows": 26752,
      "resets": 2167,
      "first_terms": 0,
      "completed_episode_len": {
        "n": 1934,
        "mean": 12.047052740434333,
        "median": 8.0,
        "p95": 31.0,
        "p99": 43.0,
        "max": 74.0
      },
      "completed_strict_shadow_len": {
        "n": 1934,
        "mean": 5.247156153050672,
        "median": 3.0,
        "p95": 16.0,
        "p99": 26.0,
        "max": 45.0
      },
      "first_term_fraction": 0.0
    },
    "g1/tracking/both": {
      "rows": 73190,
      "resets": 4598,
      "first_terms": 0,
      "completed_episode_len": {
        "n": 4090,
        "mean": 15.942053789731052,
        "median": 10.0,
        "p95": 40.0,
        "p99": 53.0,
        "max": 84.0
      },
      "completed_strict_shadow_len": {
        "n": 4090,
        "mean": 10.313936430317849,
        "median": 7.0,
        "p95": 31.0,
        "p99": 41.0,
        "max": 60.0
      },
      "first_term_fraction": 0.0
    }
  },
  "per_joint_physical_diagnostics": {
    "left_thigh_yaw_joint": {
      "rated_radps": 25.0,
      "max_physics_abs_radps": 25.0577335357666,
      "max_boundary_abs_radps": 23.63583755493164,
      "max_rated_ratio": 1.002309341430664,
      "substep_exposure_fraction": 3.178914388020833e-07
    },
    "right_thigh_yaw_joint": {
      "rated_radps": 25.0,
      "max_physics_abs_radps": 25.191553115844727,
      "max_boundary_abs_radps": 25.000009536743164,
      "max_rated_ratio": 1.0076621246337891,
      "substep_exposure_fraction": 1.5894571940104166e-07
    },
    "torso_joint": {
      "rated_radps": 25.0,
      "max_physics_abs_radps": 25.01091957092285,
      "max_boundary_abs_radps": 25.00726318359375,
      "max_rated_ratio": 1.000436782836914,
      "substep_exposure_fraction": 1.7484029134114583e-06
    },
    "left_thigh_roll_joint": {
      "rated_radps": 25.0,
      "max_physics_abs_radps": 25.153715133666992,
      "max_boundary_abs_radps": 25.000032424926758,
      "max_rated_ratio": 1.0061486053466797,
      "substep_exposure_fraction": 1.5894571940104166e-07
    },
    "right_thigh_roll_joint": {
      "rated_radps": 25.0,
      "max_physics_abs_radps": 25.048295974731445,
      "max_boundary_abs_radps": 25.000015258789062,
      "max_rated_ratio": 1.0019318389892578,
      "substep_exposure_fraction": 1.5894571940104166e-07
    },
    "left_arm_pitch_joint": {
      "rated_radps": 8.0,
      "max_physics_abs_radps": 9.538993835449219,
      "max_boundary_abs_radps": 8.701135635375977,
      "max_rated_ratio": 1.1923742294311523,
      "substep_exposure_fraction": 0.00030310948689778644
    },
    "right_arm_pitch_joint": {
      "rated_radps": 8.0,
      "max_physics_abs_radps": 11.1644868850708,
      "max_boundary_abs_radps": 11.1644868850708,
      "max_rated_ratio": 1.39556086063385,
      "substep_exposure_fraction": 0.0003655751546223958
    },
    "left_thigh_pitch_joint": {
      "rated_radps": 25.0,
      "max_physics_abs_radps": 25.005313873291016,
      "max_boundary_abs_radps": 24.99817657470703,
      "max_rated_ratio": 1.0002125549316405,
      "substep_exposure_fraction": 1.5894571940104166e-07
    },
    "right_thigh_pitch_joint": {
      "rated_radps": 25.0,
      "max_physics_abs_radps": 25.000024795532227,
      "max_boundary_abs_radps": 25.000019073486328,
      "max_rated_ratio": 1.000000991821289,
      "substep_exposure_fraction": 0.0
    },
    "left_arm_roll_joint": {
      "rated_radps": 8.0,
      "max_physics_abs_radps": 9.075403213500977,
      "max_boundary_abs_radps": 8.12574291229248,
      "max_rated_ratio": 1.134425401687622,
      "substep_exposure_fraction": 0.0002044041951497396
    },
    "right_arm_roll_joint": {
      "rated_radps": 8.0,
      "max_physics_abs_radps": 9.584092140197754,
      "max_boundary_abs_radps": 8.40813159942627,
      "max_rated_ratio": 1.1980115175247192,
      "substep_exposure_fraction": 0.0001522699991861979
    },
    "left_knee_joint": {
      "rated_radps": 25.0,
      "max_physics_abs_radps": 25.349645614624023,
      "max_boundary_abs_radps": 25.349645614624023,
      "max_rated_ratio": 1.013985824584961,
      "substep_exposure_fraction": 1.1126200358072917e-06
    },
    "right_knee_joint": {
      "rated_radps": 25.0,
      "max_physics_abs_radps": 29.9786376953125,
      "max_boundary_abs_radps": 25.337039947509766,
      "max_rated_ratio": 1.1991455078125,
      "substep_exposure_fraction": 2.0662943522135415e-06
    },
    "left_arm_yaw_joint": {
      "rated_radps": 8.0,
      "max_physics_abs_radps": 8.111333847045898,
      "max_boundary_abs_radps": 8.055644989013672,
      "max_rated_ratio": 1.0139167308807373,
      "substep_exposure_fraction": 1.0172526041666666e-05
    },
    "right_arm_yaw_joint": {
      "rated_radps": 8.0,
      "max_physics_abs_radps": 8.131958961486816,
      "max_boundary_abs_radps": 8.131958961486816,
      "max_rated_ratio": 1.016494870185852,
      "substep_exposure_fraction": 2.3206075032552082e-05
    },
    "left_ankle_pitch_joint": {
      "rated_radps": 8.0,
      "max_physics_abs_radps": 42.57744598388672,
      "max_boundary_abs_radps": 42.57744598388672,
      "max_rated_ratio": 5.32218074798584,
      "substep_exposure_fraction": 0.005906899770100911
    },
    "right_ankle_pitch_joint": {
      "rated_radps": 8.0,
      "max_physics_abs_radps": 50.307456970214844,
      "max_boundary_abs_radps": 28.874065399169922,
      "max_rated_ratio": 6.2884321212768555,
      "substep_exposure_fraction": 0.004700183868408203
    },
    "left_elbow_pitch_joint": {
      "rated_radps": 8.0,
      "max_physics_abs_radps": 8.089740753173828,
      "max_boundary_abs_radps": 8.022608757019043,
      "max_rated_ratio": 1.0112175941467285,
      "substep_exposure_fraction": 5.1180521647135414e-05
    },
    "right_elbow_pitch_joint": {
      "rated_radps": 8.0,
      "max_physics_abs_radps": 8.365883827209473,
      "max_boundary_abs_radps": 8.067578315734863,
      "max_rated_ratio": 1.045735478401184,
      "substep_exposure_fraction": 8.153915405273438e-05
    },
    "left_ankle_roll_joint": {
      "rated_radps": 8.0,
      "max_physics_abs_radps": 28.348533630371094,
      "max_boundary_abs_radps": 22.814695358276367,
      "max_rated_ratio": 3.5435667037963867,
      "substep_exposure_fraction": 0.0011157989501953125
    },
    "right_ankle_roll_joint": {
      "rated_radps": 8.0,
      "max_physics_abs_radps": 28.10263442993164,
      "max_boundary_abs_radps": 15.44244384765625,
      "max_rated_ratio": 3.512829303741455,
      "substep_exposure_fraction": 0.0009586016337076823
    },
    "left_elbow_yaw_joint": {
      "rated_radps": 8.0,
      "max_physics_abs_radps": 4.586903095245361,
      "max_boundary_abs_radps": 4.175685882568359,
      "max_rated_ratio": 0.5733628869056702,
      "substep_exposure_fraction": 0.0
    },
    "right_elbow_yaw_joint": {
      "rated_radps": 8.0,
      "max_physics_abs_radps": 4.687984943389893,
      "max_boundary_abs_radps": 4.244753360748291,
      "max_rated_ratio": 0.5859981179237366,
      "substep_exposure_fraction": 0.0
    }
  },
  "right_ankle_pitch_physics_peak_event": {
    "round": 22,
    "row": 23805,
    "episode_id_run_local": 2220447760393,
    "episode_step": 9,
    "joint": "right_ankle_pitch_joint",
    "rated_radps": 8.0,
    "boundary_dq_radps": -11.702563285827637,
    "physics_signed_dq_radps": 50.307456970214844,
    "physics_abs_dq_radps": 50.307456970214844,
    "peak_substep_0_to_7": 6,
    "q_at_joint_peak_rad": -0.4840656518936157,
    "torque_at_joint_peak_Nm": -1.0797102451324463,
    "target_at_joint_peak_rad": -0.47293347120285034,
    "requested_action": -0.7882224321365356,
    "applied_action": -0.7882224321365356,
    "foot_peak_vertical_net_force_N": [
      0.0,
      5520.546875
    ],
    "foot_vertical_net_force_integral_Ns": [
      0.0,
      26.96402359008789
    ],
    "global_peak_joint": "right_ankle_pitch_joint",
    "feet_at_global_abs_dq_peak_origin_pose_wxyz_linear_angular": [
      [
        -13.618943214416504,
        -10.397953033447266,
        0.2570202946662903,
        -0.08409515768289566,
        -0.8591628670692444,
        0.13077419996261597,
        0.4875093698501587,
        0.9607690572738647,
        0.8620430827140808,
        -2.439890146255493,
        0.7785590887069702,
        3.9524929523468018,
        4.046760559082031
      ],
      [
        -13.773435592651367,
        -10.280823707580566,
        0.0872160941362381,
        0.5941393375396729,
        -0.08909041434526443,
        0.3592461347579956,
        0.714145302772522,
        -1.048231601715088,
        -2.953230142593384,
        2.3656721115112305,
        -73.40380859375,
        7.416736125946045,
        -6.327365398406982
      ]
    ],
    "feet_vertical_net_force_at_global_peak_N": [
      0.0,
      5520.546875
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
    "round": 13,
    "row": 1820,
    "episode_id_run_local": 30534533125,
    "episode_step": 16,
    "joint": "right_ankle_roll_joint",
    "rated_radps": 8.0,
    "boundary_dq_radps": 0.26433077454566956,
    "physics_signed_dq_radps": -28.10263442993164,
    "physics_abs_dq_radps": 28.10263442993164,
    "peak_substep_0_to_7": 2,
    "q_at_joint_peak_rad": 0.45163965225219727,
    "torque_at_joint_peak_Nm": 0.08593869209289551,
    "target_at_joint_peak_rad": 0.4999994933605194,
    "requested_action": 0.9999989867210388,
    "applied_action": 0.9999989867210388,
    "foot_peak_vertical_net_force_N": [
      0.0,
      6229.3408203125
    ],
    "foot_vertical_net_force_integral_Ns": [
      0.0,
      23.701278686523438
    ],
    "global_peak_joint": "right_ankle_roll_joint",
    "feet_at_global_abs_dq_peak_origin_pose_wxyz_linear_angular": [
      [
        -43.270355224609375,
        -4.0876312255859375,
        0.3527893126010895,
        0.6140121221542358,
        -0.17684447765350342,
        0.6083843111991882,
        0.4707266092300415,
        3.525275468826294,
        -0.5736948251724243,
        -1.0971323251724243,
        -2.5820770263671875,
        -6.816893100738525,
        6.6799092292785645
      ],
      [
        -42.41231918334961,
        -3.608973503112793,
        0.042846791446208954,
        0.9580430388450623,
        0.07568810880184174,
        0.03740387409925461,
        0.27390843629837036,
        2.416060447692871,
        -2.659963846206665,
        -0.02773863635957241,
        -30.282758712768555,
        -29.163766860961914,
        3.4991869926452637
      ]
    ],
    "feet_vertical_net_force_at_global_peak_N": [
      0.0,
      6229.3408203125
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

```json
{
  "method": "collision hull transformed by foot link pose at global-|dq|-peak substep; event-only, not all-substep clearance",
  "groups": {
    "training": {
      "rows": 786432,
      "event_sole_min_z_m": -0.019775397304142697,
      "below_minus_20mm": 0,
      "below_minus_10mm": 778
    }
  }
}
```

strict速度暴露率的分母是转移×关节×子步；strict_velocity_transition_fraction的分母是转移，二者不能互换。位置/速度/力矩与跌倒原因可重叠，不能把计数相加当作独立终止数。foot force是足部垂直净力，未测逐接触点法向力；积分是对净力绝对值dt积分，不能直接当作接触求解器真实冲量。

latched_velocity_terms_without_boundary_violation可仍非零：它统计子步违规、20ms末速度回落的物理现象。奖励是否漏检以velocity_reward_coverage_miss_count判定；mean8奖励覆盖并不要求速度毛刺现象已经消失。

显式PD力矩是在该子步积分前、根据前状态计算并实际作用的力矩；峰值q/dq是完成积分后的状态，不能把峰值后的dq代入PD公式再要求等同已记录τ。低τ与高接触净力支持被动冲击假设，尚未排除之前动作、其它关节驱动、自碰撞和接触对差异。

500更新每环境只有128控制步=2.56s总采样时间，单段本身不足以完成250步tracking窗口；完成数0不单独证明tracking实现失败。未结束episode右删失，排除在完成均值之外；重新启动仿真时episode ID是Run-local，不能拼接成连续物理轨迹。

## 8. 全部原始标量统计

原始TensorBoard无限size_guidance读取，共154个标量tag；全部有限。真实新增更新点为25～524共500点；未用RLinf epoch或累计样本代替更新计数。每个tag保留点数、首末值、分位数与窗口均值。

| tag | 点数 | 首step/末step | 首值 | 末值 | 均值 | 最小 | 最大 | 中位数 | p95 | 末50点均值 | 五窗口均值 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| env/episode_len | 32 | 40/524 | 3.1857142448425293 | 22.791828155517578 | 15.850759342312813 | 3.1857142448425293 | 23.22370719909668 | 15.51519250869751 | 22.9236177444458 | 15.850759342312813 | [7.799224297205607, 12.087713559468588, 15.31110922495524, 19.654348645891464, 22.636510848999023] |
| env/num_trajectories | 32 | 40/524 | 70.0 | 1028.0 | 1504.40625 | 70.0 | 3873.0 | 1269.5 | 2415.65 | 1504.40625 | [2157.6666666666665, 1894.5, 1372.1666666666667, 1119.0, 1108.857142857143] |
| env/return | 32 | 40/524 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | [0.0, 0.0, 0.0, 0.0, 0.0] |
| env/reward | 32 | 40/524 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | [0.0, 0.0, 0.0, 0.0, 0.0] |
| env/success_once | 32 | 40/524 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | [0.0, 0.0, 0.0, 0.0, 0.0] |
| rollout/ufo/received_transitions | 32 | 40/524 | 24576.0 | 24576.0 | 24576.0 | 24576.0 | 24576.0 | 24576.0 | 24576.0 | 24576.0 | [24576.0, 24576.0, 24576.0, 24576.0, 24576.0] |
| time/actor_training | 32 | 40/524 | 15.944266319274902 | 4.323446273803711 | 12.282304912805557 | 4.323446273803711 | 15.944266319274902 | 12.424657821655273 | 12.533731317520141 | 12.282304912805557 | [12.987703164418539, 12.4291623433431, 12.425888697306315, 12.433817046029228, 11.27721609388079] |
| time/cal_adv_and_returns | 32 | 40/524 | 0.0024895668029785156 | 0.007052421569824219 | 0.010045044124126434 | 0.002474546432495117 | 0.026110172271728516 | 0.010341286659240723 | 0.018569016456604 | 0.010045044124126434 | [0.010701855023701986, 0.010503093401590982, 0.009468992551167807, 0.009408065250941686, 0.010220187050955636] |
| time/env/compute_bootstrap_rewards | 32 | 40/524 | 0.0002666329964995384 | 0.00022841105237603188 | 0.00024086129269562662 | 0.00020481087267398834 | 0.00036191800609230995 | 0.00023150048218667507 | 0.0002683880622498691 | 0.00024086129269562662 | [0.00024732792129119235, 0.00025268349175651866, 0.00023727437170843282, 0.00023790718322353705, 0.0002312137678797756] |
| time/env/env/bootstrap_step | 32 | 40/524 | 0.017511505633592606 | 0.00029832380823791027 | 0.0008155545219779015 | 0.00024051102809607983 | 0.017511505633592606 | 0.0002611519303172827 | 0.00043827798217535016 | 0.0008155545219779015 | [0.003139309507484237, 0.00031079782638698816, 0.00025610012623171013, 0.0002778741264981883, 0.0002736258653125593] |
| time/env/env/send_rollout_trajectories | 32 | 40/524 | 0.2568044662475586 | 0.23392367362976074 | 0.23495030216872692 | 0.21200522780418396 | 0.28375640511512756 | 0.2318117991089821 | 0.25996815860271455 | 0.23495030216872692 | [0.25809162358442944, 0.22769317775964737, 0.23039764910936356, 0.22832432389259338, 0.23186352849006653] |
| time/env/env_interact_step | 32 | 40/524 | 3.3904929161071777 | 1.9145753383636475 | 1.9607322551310062 | 1.7434325218200684 | 3.3904929161071777 | 1.941879153251648 | 2.0534778356552126 | 1.9607322551310062 | [2.185156464576721, 1.9059612154960632, 1.8926138281822205, 1.9010747841426305, 1.933359946523394] |
| time/env/interact | 32 | 40/524 | 5.616068363189697 | 2.9488377571105957 | 3.106567859649658 | 2.7693066596984863 | 5.616068363189697 | 3.034589171409607 | 3.302821683883667 | 3.106567859649658 | [3.599123954772949, 3.072300672531128, 2.9255741039911904, 2.9727484498705183, 3.0027057102748325] |
| time/env/run_interact_once | 32 | 40/524 | 5.616028308868408 | 2.948814630508423 | 3.1065414398908615 | 2.7692782878875732 | 5.616028308868408 | 3.0345633029937744 | 3.3027890086174008 | 3.1065414398908615 | [3.599092880884806, 3.072275996208191, 2.925547242164612, 2.972723824637277, 3.0026803697858537] |
| time/generate_rollouts | 32 | 40/524 | 5.929497718811035 | 3.111374616622925 | 3.310854882001877 | 3.035081624984741 | 5.929497718811035 | 3.2189050912857056 | 3.523199760913849 | 3.310854882001877 | [3.8348934253056846, 3.2614593505859375, 3.146701534589132, 3.1654892648969377, 3.1900865009852817] |
| time/rollout/generate_one_epoch | 32 | 40/524 | 5.379324436187744 | 2.7126853466033936 | 2.8536696657538414 | 2.512136697769165 | 5.379324436187744 | 2.7879819869995117 | 2.9469110012054442 | 2.8536696657538414 | [3.300835887591044, 2.8225863774617515, 2.6817948818206787, 2.73091265133449, 2.7671054090772356] |
| time/rollout/predict | 32 | 40/524 | 0.996675968170166 | 0.10332246124744415 | 0.19211430451832712 | 0.10035991668701172 | 0.996675968170166 | 0.14662761986255646 | 0.30540360361337654 | 0.19211430451832712 | [0.3799049233396848, 0.20535254975159964, 0.15068234006563822, 0.12458118051290512, 0.12285008600779943] |
| time/rollout/rollout/generate | 32 | 40/524 | 5.382325172424316 | 2.7127912044525146 | 2.853870816528797 | 2.512239694595337 | 5.382325172424316 | 2.7880924940109253 | 2.947051453590393 | 2.853870816528797 | [3.30144202709198, 2.822697122891744, 2.681901494661967, 2.7310220514025008, 2.767209700175694] |
| time/rollout/sync_model_from_actor | 32 | 40/524 | 14.675507545471191 | 12.954425811767578 | 12.954959094524384 | 12.629284858703613 | 14.675507545471191 | 12.876195430755615 | 13.601199054718016 | 12.954959094524384 | [13.144458770751953, 12.82425308227539, 12.817449728647867, 13.10179615020752, 12.875592640468053] |
| time/step | 32 | 40/524 | 39.99635696411133 | 24.307607650756836 | 32.33524614572525 | 24.307607650756836 | 39.99635696411133 | 32.33347702026367 | 33.08476371765136 | 32.33524614572525 | [33.532544453938804, 32.295152028401695, 32.18354479471842, 32.5737680707659, 31.23486464364188] |
| time/sync_weights | 32 | 40/524 | 18.11937141418457 | 16.865184783935547 | 16.73150223493576 | 16.30554962158203 | 18.11937141418457 | 16.648768424987793 | 17.448491954803465 | 16.73150223493576 | [16.698702812194824, 16.593456268310547, 16.60099442799886, 16.964527947562082, 16.756779261997767] |
| train/ufo/B | 500 | 25/524 | -0.010226310230791569 | -0.002481282688677311 | -0.005419051396660507 | -0.01906270906329155 | 0.0010519367642700672 | -0.004295686259865761 | -0.0011858670739457013 | -0.002166681515518576 | [-0.011622576378285884, -0.005916013310197741, -0.004210882490733639, -0.003003401856403798, -0.0023423829476814715] |
| train/ufo/B_norm | 500 | 25/524 | 16.0 | 16.0 | 16.0 | 16.0 | 16.0 | 16.0 | 16.0 | 16.0 | [16.0, 16.0, 16.0, 16.0, 16.0] |
| train/ufo/F1 | 500 | 25/524 | 0.0674636960029602 | 0.039049264043569565 | 0.036299004841595886 | 0.01680080220103264 | 0.07737861573696136 | 0.03480536676943302 | 0.06217319797724485 | 0.03423567332327366 | [0.046583506278693676, 0.034218315836042165, 0.03596911804750562, 0.032408954799175264, 0.03231512924656272] |
| train/ufo/M1 | 500 | 25/524 | 0.49528467655181885 | 1.0091888904571533 | 1.3664601678848267 | 0.49528467655181885 | 6.799777984619141 | 1.0234403610229492 | 3.94892542362213 | 0.8638402163982392 | [2.8798693895339964, 1.205508101582527, 0.9862199902534485, 0.9198030459880829, 0.8409003120660782] |
| train/ufo/Q1 | 500 | 25/524 | -3.6897003650665283 | -4.503898620605469 | -3.382512068748474 | -4.968841552734375 | -2.3676562309265137 | -3.283723831176758 | -2.5715760707855226 | -4.33005316734314 | [-3.6350235176086425, -2.6823450183868407, -2.926790030002594, -3.496730260848999, -4.1716715168952945] |
| train/ufo/Q_aux | 500 | 25/524 | -4.473977088928223 | -7.664921760559082 | -5.872201619148254 | -7.751883029937744 | -4.4544291496276855 | -5.728756904602051 | -4.630104303359985 | -7.406116161346436 | [-4.857372121810913, -4.930201144218445, -5.713945202827453, -6.585213479995727, -7.274276146888733] |
| train/ufo/Q_discriminator | 500 | 25/524 | -5.228636741638184 | -4.543886184692383 | -3.4802357864379885 | -5.316997528076172 | -2.390211820602417 | -3.3421460390090942 | -2.607343912124634 | -4.434553756713867 | [-3.8145138955116273, -2.7115432953834535, -2.995691134929657, -3.599946539402008, -4.279484066963196] |
| train/ufo/Q_fb | 500 | 25/524 | 0.9666777849197388 | 61.627197265625 | 30.094790582343936 | -0.7241038084030151 | 63.03118896484375 | 29.596529006958008 | 57.30305843353271 | 57.81717216491699 | [7.254135158881545, 17.933306102752685, 29.383051643371584, 41.43665542602539, 54.466804580688475] |
| train/ufo/action_abs_mean | 32 | 40/524 | 0.9318665266036987 | 0.8553743362426758 | 0.9036657474935055 | 0.8553743362426758 | 0.9362102150917053 | 0.904106855392456 | 0.9328347504138946 | 0.9036657474935055 | [0.9304363330205282, 0.9216419756412506, 0.9064130286375681, 0.8928249308041164, 0.8737973400524685] |
| train/ufo/action_saturation_fraction | 32 | 40/524 | 0.7454922199249268 | 0.49290046095848083 | 0.612792911939323 | 0.49290046095848083 | 0.7454922199249268 | 0.612604558467865 | 0.6888439059257507 | 0.612792911939323 | [0.6870040396849314, 0.6524572670459747, 0.6151580115159353, 0.5863608292170933, 0.5395902097225189] |
| train/ufo/actor_loss | 500 | 25/524 | 2.9159226417541504 | -42.602561950683594 | -19.37378551054001 | -44.478485107421875 | 4.506507873535156 | -19.32904052734375 | -0.9914170265197831 | -39.32467185974121 | [-3.009517352581024, -11.918001146316529, -19.269779872894286, -26.177810859680175, -36.49381832122803] |
| train/ufo/actor_optimizer/grad_clip_limit | 500 | 25/524 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | [1.0, 1.0, 1.0, 1.0, 1.0] |
| train/ufo/actor_optimizer/grad_norm_after_clip_l2 | 500 | 25/524 | 0.9999990463256836 | 0.9999998807907104 | 0.9997192293405532 | 0.9199018478393555 | 0.9999999403953552 | 0.9999997615814209 | 0.9999998807907104 | 0.9999998509883881 | [0.998596979379654, 0.9999997174739838, 0.9999997866153717, 0.9999998235702514, 0.9999998396635056] |
| train/ufo/actor_optimizer/grad_norm_l2 | 500 | 25/524 | 1.1017838716506958 | 7.6237688064575195 | 4.58438438475132 | 0.9199018478393555 | 10.390420913696289 | 4.800204753875732 | 7.083274078369139 | 7.030235662460327 | [1.8631995874643326, 3.600414764881134, 4.840401225090027, 5.882842917442321, 6.735063428878784] |
| train/ufo/actor_optimizer/lr | 500 | 25/524 | 0.0003000000142492354 | 0.0003000000142492354 | 0.0003000000142492354 | 0.0003000000142492354 | 0.0003000000142492354 | 0.0003000000142492354 | 0.0003000000142492354 | 0.0003000000142492354 | [0.0003000000142492354, 0.0003000000142492354, 0.0003000000142492354, 0.0003000000142492354, 0.0003000000142492354] |
| train/ufo/actor_optimizer/parameter_delta_l2 | 500 | 25/524 | 0.44481322169303894 | 0.3266304135322571 | 0.34081267815828326 | 0.2934066653251648 | 0.44481322169303894 | 0.3375070244073868 | 0.37925004214048386 | 0.3271451884508133 | [0.36536835074424745, 0.3482022660970688, 0.3335008633136749, 0.32633424311876297, 0.33065766751766207] |
| train/ufo/actor_reg_weight_applied | 500 | 25/524 | 11.064332962036133 | 50.0 | 35.52902792167664 | 10.276827812194824 | 50.0 | 38.360626220703125 | 50.0 | 50.0 | [14.982503538131715, 25.641645545959474, 38.138437309265136, 48.88255321502685, 50.0] |
| train/ufo/actor_reg_weight_raw | 500 | 25/524 | 11.064332962036133 | 71.10066986083984 | 38.801511449813844 | 10.276827812194824 | 72.78044128417969 | 38.360626220703125 | 67.48071403503417 | 67.60880531311035 | [14.982503538131715, 25.641645545959474, 38.138437309265136, 51.00389846801758, 64.24107238769531] |
| train/ufo/applied_action_abs_mean | 32 | 40/524 | 0.46392595767974854 | 0.6118605732917786 | 0.6043008454144001 | 0.46392595767974854 | 0.6334840655326843 | 0.618164986371994 | 0.6317382007837296 | 0.6043008454144001 | [0.5398426552613577, 0.6093393564224243, 0.6241315801938375, 0.6272290434156146, 0.6153060282979693] |
| train/ufo/applied_action_saturation_fraction | 32 | 40/524 | 0.1082533672451973 | 0.2664388120174408 | 0.25620772619731724 | 0.1082533672451973 | 0.2899930775165558 | 0.2742610424757004 | 0.28919634521007537 | 0.25620772619731724 | [0.17558015510439873, 0.25727290908495587, 0.28535998860994977, 0.28478548782212393, 0.27083864382335116] |
| train/ufo/auxQ1 | 500 | 25/524 | -4.333232402801514 | -7.570643424987793 | -5.758412041664124 | -7.570643424987793 | -4.146952152252197 | -5.648586750030518 | -4.528618884086609 | -7.242484626770019 | [-4.670852627754211, -4.912626619338989, -5.665609078407288, -6.428285269737244, -7.114686613082886] |
| train/ufo/aux_critic_loss | 500 | 25/524 | 2.0069844722747803 | 2.069000482559204 | 1.0999043074846269 | 0.7364603281021118 | 11.267192840576172 | 1.0208181142807007 | 1.5531206488609313 | 1.389241338968277 | [1.2213255774974823, 0.8840277934074402, 0.9697508937120438, 1.1304379242658615, 1.2939793485403062] |
| train/ufo/aux_critic_optimizer/grad_clip_limit | 500 | 25/524 | 5.0 | 5.0 | 5.0 | 5.0 | 5.0 | 5.0 | 5.0 | 5.0 | [5.0, 5.0, 5.0, 5.0, 5.0] |
| train/ufo/aux_critic_optimizer/grad_norm_after_clip_l2 | 500 | 25/524 | 4.999999523162842 | 4.999999046325684 | 4.83491319322586 | 1.8552542924880981 | 5.000000476837158 | 4.999999523162842 | 5.0 | 4.6527714967727665 | [4.999999504089356, 4.8841240072250365, 4.717907058000565, 4.797439588308334, 4.775095808506012] |
| train/ufo/aux_critic_optimizer/grad_norm_l2 | 500 | 25/524 | 21.30470848083496 | 9.224893569946289 | 10.892003270626068 | 1.8552542924880981 | 39.28155517578125 | 9.5501389503479 | 24.339135742187494 | 8.844094948768616 | [13.246373877525329, 8.739638113975525, 7.306467505693436, 16.19866380572319, 8.96887305021286] |
| train/ufo/aux_critic_optimizer/lr | 500 | 25/524 | 0.0003000000142492354 | 0.0003000000142492354 | 0.0003000000142492354 | 0.0003000000142492354 | 0.0003000000142492354 | 0.0003000000142492354 | 0.0003000000142492354 | 0.0003000000142492354 | [0.0003000000142492354, 0.0003000000142492354, 0.0003000000142492354, 0.0003000000142492354, 0.0003000000142492354] |
| train/ufo/aux_critic_optimizer/parameter_delta_l2 | 500 | 25/524 | 0.4119172990322113 | 0.5968431234359741 | 0.43616681784391403 | 0.23162487149238586 | 0.6669462323188782 | 0.436299666762352 | 0.618875926733017 | 0.5381105440855026 | [0.3260089235007763, 0.43020265340805053, 0.5075645896792412, 0.38987110629677774, 0.5271868163347244] |
| train/ufo/aux_rew/limits_dof_pos | 500 | 25/524 | 0.03614848107099533 | 0.277969628572464 | 0.1852838257551193 | 0.03426898270845413 | 0.2835008502006531 | 0.20320259779691696 | 0.26742298156023026 | 0.2680749654769897 | [0.07521787703037262, 0.14959781169891356, 0.2016410008072853, 0.23678289830684662, 0.2631795409321785] |
| train/ufo/aux_rew/limits_dof_vel | 500 | 25/524 | 0.008886139839887619 | 0.016782240942120552 | 0.014703124916180968 | 0.008394215255975723 | 0.02812516689300537 | 0.015172887593507767 | 0.016920434962958097 | 0.015113313253968953 | [0.011603135438635946, 0.015265269409865142, 0.01594911511056125, 0.015526902787387371, 0.015171201834455132] |
| train/ufo/aux_rew/limits_torque | 500 | 25/524 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | [0.0, 0.0, 0.0, 0.0, 0.0] |
| train/ufo/aux_rew/penalty_action_rate | 500 | 25/524 | 0.6820842623710632 | 0.3978596329689026 | 0.4795042409300804 | 0.3825783133506775 | 0.6833406686782837 | 0.4521738737821579 | 0.633450385928154 | 0.39679870069026946 | [0.6125435209274293, 0.5179026129841805, 0.452887199819088, 0.415290804207325, 0.39889706671237946] |
| train/ufo/aux_rew/penalty_ankle_roll | 500 | 25/524 | 0.023734554648399353 | 0.15861894190311432 | 0.11331098582223058 | 0.023734554648399353 | 0.16424694657325745 | 0.12396223098039627 | 0.15648283585906028 | 0.15606590420007704 | [0.05089282164350152, 0.0976586952060461, 0.12333058081567287, 0.14081495732069016, 0.15385787412524224] |
| train/ufo/aux_rew/penalty_feet_ori | 500 | 25/524 | 0.11801569908857346 | 0.2169918268918991 | 0.16062916328012944 | 0.10438111424446106 | 0.22079220414161682 | 0.16034721583127975 | 0.20843563377857208 | 0.2071387243270874 | [0.11491849705576897, 0.13785450376570224, 0.16056635200977326, 0.1855487656593323, 0.20425769791007042] |
| train/ufo/aux_rew/penalty_slippage | 500 | 25/524 | 0.4680141508579254 | 0.3904952108860016 | 0.39288887709379194 | 0.3436899483203888 | 0.4865013659000397 | 0.3890358954668045 | 0.43491280674934385 | 0.3757189321517944 | [0.4248321267962456, 0.3972329220175743, 0.3843397161364555, 0.37887933492660525, 0.37916028559207915] |
| train/ufo/aux_rew/penalty_torques | 500 | 25/524 | 5740.03125 | 5451.94580078125 | 6013.369734375 | 5451.94580078125 | 6467.34326171875 | 6004.376953125 | 6321.875903320312 | 5790.825849609375 | [6106.89958984375, 6171.292016601563, 6011.276982421875, 5933.75353515625, 5843.6265478515625] |
| train/ufo/aux_rew/penalty_undesired_contact | 500 | 25/524 | 0.00048828125 | 0.0048828125 | 0.0056748046875 | 0.0 | 0.0126953125 | 0.005859375 | 0.009765625 | 0.005625 | [0.001748046875, 0.007060546875, 0.00732421875, 0.0064697265625, 0.005771484375] |
| train/ufo/backward_optimizer/grad_clip_limit | 500 | 25/524 | 5.0 | 5.0 | 5.0 | 5.0 | 5.0 | 5.0 | 5.0 | 5.0 | [5.0, 5.0, 5.0, 5.0, 5.0] |
| train/ufo/backward_optimizer/grad_norm_after_clip_l2 | 500 | 25/524 | 5.0 | 4.999999523162842 | 4.9999999952316285 | 4.999999046325684 | 5.000000953674316 | 5.0 | 5.000000476837158 | 5.0 | [5.0000000047683715, 4.999999976158142, 4.999999957084656, 5.000000038146973, 5.0] |
| train/ufo/backward_optimizer/grad_norm_l2 | 500 | 25/524 | 417287.6875 | 16874.1796875 | 79127.65570507813 | 16043.05078125 | 434283.09375 | 38274.720703125 | 287875.1593749998 | 17450.64095703125 | [235338.195546875, 77042.9409765625, 39116.436875, 25487.59849609375, 18653.106630859376] |
| train/ufo/backward_optimizer/lr | 500 | 25/524 | 9.999999747378752e-06 | 9.999999747378752e-06 | 9.999999747378752e-06 | 9.999999747378752e-06 | 9.999999747378752e-06 | 9.999999747378752e-06 | 9.999999747378752e-06 | 9.999999747378752e-06 | [9.999999747378752e-06, 9.999999747378752e-06, 9.999999747378752e-06, 9.999999747378752e-06, 9.999999747378752e-06] |
| train/ufo/backward_optimizer/parameter_delta_l2 | 500 | 25/524 | 0.0033988759387284517 | 0.00354715995490551 | 0.004068766467273235 | 0.003369143232703209 | 0.004430299159139395 | 0.004125529667362571 | 0.004410731163807213 | 0.0036328799277544023 | [0.003970019738189876, 0.004395084753632546, 0.004254629099741578, 0.004013653905130923, 0.003710444839671254] |
| train/ufo/collected_transitions | 32 | 40/524 | 24576.0 | 24576.0 | 24576.0 | 24576.0 | 24576.0 | 24576.0 | 24576.0 | 24576.0 | [24576.0, 24576.0, 24576.0, 24576.0, 24576.0] |
| train/ufo/contact_termination_count | 32 | 40/524 | 27.0 | 150.0 | 202.84375 | 24.0 | 559.0 | 168.5 | 517.25 | 202.84375 | [182.33333333333334, 393.3333333333333, 185.16666666666666, 111.71428571428571, 163.42857142857142] |
| train/ufo/context_error_count | 32 | 40/524 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | [0.0, 0.0, 0.0, 0.0, 0.0] |
| train/ufo/critic_loss | 500 | 25/524 | 1.581048607826233 | 0.4776987135410309 | 0.4028633179962635 | 0.2032177746295929 | 2.647822141647339 | 0.3129306584596634 | 1.0368886888027182 | 0.3984669828414917 | [0.8080549702048302, 0.2840913705527782, 0.24360636085271836, 0.2974502047896385, 0.3811136835813522] |
| train/ufo/critic_optimizer/grad_clip_limit | 500 | 25/524 | 5.0 | 5.0 | 5.0 | 5.0 | 5.0 | 5.0 | 5.0 | 5.0 | [5.0, 5.0, 5.0, 5.0, 5.0] |
| train/ufo/critic_optimizer/grad_norm_after_clip_l2 | 500 | 25/524 | 4.999999523162842 | 5.0 | 4.696313889980316 | 1.2363532781600952 | 5.000000476837158 | 4.999999046325684 | 5.0 | 4.299878356456756 | [4.978381135463715, 4.783748457431793, 4.620049679279328, 4.541373590230942, 4.558016587495803] |
| train/ufo/critic_optimizer/grad_norm_l2 | 500 | 25/524 | 18.511314392089844 | 8.177722930908203 | 9.33799832391739 | 1.2363532781600952 | 57.875770568847656 | 7.900599956512451 | 18.547468948364227 | 6.221947019100189 | [15.530564558506011, 8.085485379695893, 6.802077667713165, 8.608567861318589, 7.663296152353286] |
| train/ufo/critic_optimizer/lr | 500 | 25/524 | 0.0003000000142492354 | 0.0003000000142492354 | 0.0003000000142492354 | 0.0003000000142492354 | 0.0003000000142492354 | 0.0003000000142492354 | 0.0003000000142492354 | 0.0003000000142492354 | [0.0003000000142492354, 0.0003000000142492354, 0.0003000000142492354, 0.0003000000142492354, 0.0003000000142492354] |
| train/ufo/critic_optimizer/parameter_delta_l2 | 500 | 25/524 | 0.4622753858566284 | 0.4053543210029602 | 0.33834383749961855 | 0.19498378038406372 | 0.5529687404632568 | 0.3326013535261154 | 0.47221338003873825 | 0.4680368316173553 | [0.3173441517353058, 0.2843786345422268, 0.3246017897129059, 0.3416809086501598, 0.4237137028574944] |
| train/ufo/cuda_peak_allocated_bytes | 32 | 40/524 | 16248861696.0 | 16251819008.0 | 16251339008.0 | 16248861696.0 | 16251819008.0 | 16251819008.0 | 16251819008.0 | 16251339008.0 | [16249837909.333334, 16251240106.666666, 16251819008.0, 16251819008.0, 16251819008.0] |
| train/ufo/cuda_peak_reserved_bytes | 32 | 40/524 | 16456351744.0 | 16456351744.0 | 16456351744.0 | 16456351744.0 | 16456351744.0 | 16456351744.0 | 16456351744.0 | 16456351744.0 | [16456351744.0, 16456351744.0, 16456351744.0, 16456351744.0, 16456351744.0] |
| train/ufo/disc_expert_loss | 500 | 25/524 | 0.599824070930481 | 0.2322779893875122 | 0.3298961435854435 | 0.20685827732086182 | 0.599824070930481 | 0.2914711534976959 | 0.5573731243610383 | 0.23057065784931183 | [0.5016617399454116, 0.35777187258005144, 0.29275402665138245, 0.2608755527436733, 0.2364175260066986] |
| train/ufo/disc_loss | 500 | 25/524 | 3.5531814098358154 | 0.7647842168807983 | 1.3801713544130325 | 0.7399859428405762 | 3.553267240524292 | 1.0406888723373413 | 3.092177963256834 | 0.7805676424503326 | [2.7054447078704835, 1.456624858379364, 1.0380532789230346, 0.8988030987977982, 0.8019308280944825] |
| train/ufo/disc_reward_clip_fraction | 500 | 25/524 | 0.0 | 0.30712890625 | 0.101376953125 | 0.0 | 0.32666015625 | 0.0654296875 | 0.2880859375 | 0.287880859375 | [0.0, 0.005224609375, 0.0662890625, 0.1702978515625, 0.2650732421875] |
| train/ufo/disc_train_loss | 500 | 25/524 | 0.7632684707641602 | 0.295046865940094 | 0.47178297489881516 | 0.27524334192276 | 0.7675120234489441 | 0.4289858639240265 | 0.7388675510883331 | 0.29568557560443876 | [0.7104068249464035, 0.5550929564237594, 0.4320916745066643, 0.3555454099178314, 0.3057780086994171] |
| train/ufo/disc_wgan_gp_loss | 500 | 25/524 | 0.2190088927745819 | 0.023745929822325706 | 0.05784922358393669 | 0.023745929822325706 | 0.21998828649520874 | 0.031205547973513603 | 0.17969089597463586 | 0.02543114058673382 | [0.14933761477470398, 0.05437600307166576, 0.03132075751200318, 0.028238213397562505, 0.025973529163748024] |
| train/ufo/discriminator_optimizer/grad_clip_limit | 500 | 25/524 | 5.0 | 5.0 | 5.0 | 5.0 | 5.0 | 5.0 | 5.0 | 5.0 | [5.0, 5.0, 5.0, 5.0, 5.0] |
| train/ufo/discriminator_optimizer/grad_norm_after_clip_l2 | 500 | 25/524 | 5.0 | 0.9466066956520081 | 2.6310316964387894 | 0.9466066956520081 | 5.0 | 1.620998203754425 | 4.999999523162842 | 1.0487873542308808 | [4.999999313354492, 4.108927080631256, 1.6676116621494292, 1.2902032017707825, 1.0884172242879868] |
| train/ufo/discriminator_optimizer/grad_norm_l2 | 500 | 25/524 | 8.512041091918945 | 0.9466066956520081 | 3.3007920161485673 | 0.9466066956520081 | 8.552024841308594 | 1.620998203754425 | 8.235860300064086 | 1.0487873542308808 | [7.899394187927246, 4.558333804607392, 1.6676116621494292, 1.2902032017707825, 1.0884172242879868] |
| train/ufo/discriminator_optimizer/lr | 500 | 25/524 | 3.000000106112566e-06 | 3.000000106112566e-06 | 3.000000106112566e-06 | 3.000000106112566e-06 | 3.000000106112566e-06 | 3.000000106112566e-06 | 3.000000106112566e-06 | 3.000000106112566e-06 | [3.000000106112566e-06, 3.000000106112566e-06, 3.000000106112566e-06, 3.000000106112566e-06, 3.000000106112566e-06] |
| train/ufo/discriminator_optimizer/parameter_delta_l2 | 500 | 25/524 | 0.0044385637156665325 | 0.0024891344364732504 | 0.0036655338099226354 | 0.0024891344364732504 | 0.005346973892301321 | 0.003177270875312388 | 0.005252427258528769 | 0.00253980063367635 | [0.004859779006801545, 0.004835724993608892, 0.003224872355349362, 0.0028229007101617755, 0.002584391983691603] |
| train/ufo/environment_control_steps | 32 | 40/524 | 16.0 | 140.0 | 78.0 | 16.0 | 140.0 | 78.0 | 133.8 | 78.0 | [26.0, 50.0, 74.0, 100.0, 128.0] |
| train/ufo/expert_draws/g1 | 32 | 40/524 | 41104.0 | 536088.0 | 294789.25 | 41104.0 | 536088.0 | 295080.0 | 523470.39999999997 | 294789.25 | [82226.66666666667, 181020.0, 278861.3333333333, 385026.28571428574, 497917.71428571426] |
| train/ufo/expert_draws/official_rpo | 32 | 40/524 | 40816.0 | 537064.0 | 294266.75 | 40816.0 | 537064.0 | 294744.0 | 523467.2 | 294266.75 | [81613.33333333333, 179428.0, 278194.6666666667, 385021.71428571426, 497995.4285714286] |
| train/ufo/fb_diag | 500 | 25/524 | -20.963932037353516 | -243.4243927001953 | -176.50635841560364 | -249.11660766601562 | -14.8053617477417 | -208.01853942871094 | -36.0125654220581 | -241.1985061645508 | [-57.65922638893127, -151.13381050109862, -206.59584197998046, -228.17110061645508, -238.97181259155275] |
| train/ufo/fb_loss | 500 | 25/524 | 228903.296875 | 17392.404296875 | 56966.32880078125 | 17027.908203125 | 229236.140625 | 37483.646484375 | 168021.60468749996 | 18469.0709375 | [139402.10203125, 61080.5608984375, 37826.3386328125, 26695.53953125, 19827.10291015625] |
| train/ufo/fb_offdiag | 500 | 25/524 | 238.36273193359375 | 224.80953979492188 | 157.66959004211427 | 89.3569107055664 | 238.36273193359375 | 158.54044342041016 | 219.08348999023437 | 218.43224273681642 | [114.93129501342773, 121.607133102417, 155.75283508300782, 184.30778869628907, 211.7488983154297] |
| train/ufo/first_step_samples | 32 | 40/524 | 6184.0 | 1010.0 | 1688.78125 | 960.0 | 6184.0 | 1356.5 | 2885.5999999999995 | 1688.78125 | [3100.3333333333335, 1900.8333333333333, 1401.1666666666667, 1116.142857142857, 1116.2857142857142] |
| train/ufo/first_step_terminated | 32 | 40/524 | 4.0 | 3.0 | 1.21875 | 0.0 | 4.0 | 1.0 | 3.4499999999999993 | 1.21875 | [1.6666666666666667, 1.1666666666666667, 1.0, 0.5714285714285714, 1.7142857142857142] |
| train/ufo/foot_contact_peak_N | 32 | 40/524 | 3510.235595703125 | 6209.00146484375 | 7519.783302307129 | 3510.235595703125 | 12074.6611328125 | 7608.657470703125 | 9482.245068359374 | 7519.783302307129 | [6530.0395914713545, 7209.119384765625, 8931.8662109375, 7947.333426339285, 6996.511509486607] |
| train/ufo/forward_optimizer/grad_clip_limit | 500 | 25/524 | 5.0 | 5.0 | 5.0 | 5.0 | 5.0 | 5.0 | 5.0 | 5.0 | [5.0, 5.0, 5.0, 5.0, 5.0] |
| train/ufo/forward_optimizer/grad_norm_after_clip_l2 | 500 | 25/524 | 5.0 | 5.0 | 5.00000002861023 | 4.999999046325684 | 5.000000953674316 | 5.0 | 5.000000476837158 | 4.999999990463257 | [5.000000033378601, 5.00000002861023, 5.00000002861023, 5.000000033378601, 5.000000019073486] |
| train/ufo/forward_optimizer/grad_norm_l2 | 500 | 25/524 | 2498.95654296875 | 98.67385864257812 | 299.5144339599609 | 89.29103088378906 | 2498.95654296875 | 124.71846389770508 | 1156.2474365234366 | 103.41655899047852 | [943.5329339599609, 221.54456466674804, 124.08869468688965, 105.60835670471191, 102.79761978149413] |
| train/ufo/forward_optimizer/lr | 500 | 25/524 | 0.0003000000142492354 | 0.0003000000142492354 | 0.0003000000142492354 | 0.0003000000142492354 | 0.0003000000142492354 | 0.0003000000142492354 | 0.0003000000142492354 | 0.0003000000142492354 | [0.0003000000142492354, 0.0003000000142492354, 0.0003000000142492354, 0.0003000000142492354, 0.0003000000142492354] |
| train/ufo/forward_optimizer/parameter_delta_l2 | 500 | 25/524 | 0.3305702209472656 | 0.7804715037345886 | 0.6809246549606324 | 0.25958138704299927 | 0.8727094531059265 | 0.7689969837665558 | 0.839699974656105 | 0.7878299522399902 | [0.34828573286533354, 0.6660708957910537, 0.7867114788293839, 0.8100565588474273, 0.7934986084699631] |
| train/ufo/ground_foot_margin_min_m | 32 | 40/524 | 195.22433471679688 | 193.64068603515625 | 193.95327138900757 | 193.34695434570312 | 195.25938415527344 | 193.7163314819336 | 195.2148010253906 | 193.95327138900757 | [194.17183430989584, 194.19903310139975, 193.82720692952475, 193.9357430594308, 193.6808624267578] |
| train/ufo/ground_outside_count | 32 | 40/524 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | [0.0, 0.0, 0.0, 0.0, 0.0] |
| train/ufo/hard_position_count | 32 | 40/524 | 29.0 | 232.0 | 278.53125 | 29.0 | 481.0 | 261.0 | 442.9 | 278.53125 | [304.6666666666667, 364.3333333333333, 261.5, 244.14285714285714, 231.57142857142858] |
| train/ufo/hard_torque_count | 32 | 40/524 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | [0.0, 0.0, 0.0, 0.0, 0.0] |
| train/ufo/hard_velocity_count | 32 | 40/524 | 3568.0 | 991.0 | 1307.03125 | 575.0 | 3568.0 | 1128.0 | 2245.5499999999997 | 1307.03125 | [1930.0, 1403.8333333333333, 1063.0, 1133.142857142857, 1073.142857142857] |
| train/ufo/height_termination_count | 32 | 40/524 | 0.0 | 398.0 | 415.90625 | 0.0 | 632.0 | 443.5 | 566.25 | 415.90625 | [284.0, 486.5, 442.8333333333333, 429.2857142857143, 432.0] |
| train/ufo/latched_velocity_terms_without_boundary_violation | 32 | 40/524 | 2524.0 | 823.0 | 1039.0 | 462.0 | 2524.0 | 921.5 | 1734.7999999999997 | 1039.0 | [1446.8333333333333, 1135.8333333333333, 864.0, 924.7142857142857, 870.7142857142857] |
| train/ufo/learner_joint_updates | 500 | 25/524 | 25.0 | 524.0 | 274.5 | 25.0 | 524.0 | 274.5 | 499.04999999999995 | 499.5 | [74.5, 174.5, 274.5, 374.5, 474.5] |
| train/ufo/mean_aux_reward | 500 | 25/524 | -0.9537927508354187 | -1.3116027116775513 | -1.258650851368904 | -1.3890725374221802 | -0.8806980848312378 | -1.286170244216919 | -1.0103643536567695 | -1.2766572546958923 | [-1.1022631615400313, -1.2684957361221314, -1.291997230052948, -1.3220965099334716, -1.308401619195938] |
| train/ufo/mean_disc_reward | 500 | 25/524 | 0.09047820419073105 | -0.9982694983482361 | -0.4762400951145683 | -1.0231480598449707 | 0.09344223886728287 | -0.5175758004188538 | 0.034915583580732344 | -0.9645883476734162 | [0.016212883637053893, -0.19145250253379345, -0.518127606511116, -0.7581827491521835, -0.9296505010128021] |
| train/ufo/mean_disc_reward_raw | 500 | 25/524 | 0.09047820419073105 | -1.2371234893798828 | -0.5369179569690022 | -1.2505848407745361 | 0.09344223886728287 | -0.5459829568862915 | 0.034915583580732344 | -1.1747175979614257 | [0.016212883637053893, -0.19274589464068412, -0.5448431947827339, -0.8496663421392441, -1.113547236919403] |
| train/ufo/mean_next_Q | 500 | 25/524 | -2.8137617111206055 | -3.6858367919921875 | -2.744652467250824 | -3.6921005249023438 | -2.163283348083496 | -2.683862090110779 | -2.2240139722824095 | -3.508214931488037 | [-2.9118982434272764, -2.2837258768081665, -2.370007336139679, -2.7942793703079225, -3.3633515095710753] |
| train/ufo/mean_next_auxQ | 500 | 25/524 | -3.138868808746338 | -6.451302528381348 | -4.59865908908844 | -6.451302528381348 | -3.138868808746338 | -4.546119689941406 | -3.3332436084747314 | -6.2168359375 | [-3.38281005859375, -3.6695631551742554, -4.537072162628174, -5.34640528678894, -6.05744478225708] |
| train/ufo/online_transitions | 500 | 25/524 | 98304.0 | 860160.0 | 470089.728 | 98304.0 | 860160.0 | 466944.0 | 811008.0 | 815923.2 | [163184.64, 316538.88, 469893.12, 623247.36, 777584.64] |
| train/ufo/orientation_termination_count | 32 | 40/524 | 0.0 | 292.0 | 659.28125 | 0.0 | 3619.0 | 477.5 | 1565.3999999999996 | 659.28125 | [1453.0, 708.0, 538.8333333333334, 373.85714285714283, 325.85714285714283] |
| train/ufo/orth_loss | 500 | 25/524 | 2286.85888671875 | 174.11019897460938 | 569.8516552124023 | 170.48876953125 | 2290.670654296875 | 375.3540344238281 | 1679.3111633300778 | 184.91837463378906 | [1393.4482934570312, 611.1008782958984, 378.7718157958984, 267.39402954101564, 198.54325897216796] |
| train/ufo/orth_loss_diag | 500 | 25/524 | -256.0 | -256.0 | -256.0 | -256.0 | -256.0 | -256.0 | -256.0 | -256.0 | [-256.0, -256.0, -256.0, -256.0, -256.0] |
| train/ufo/orth_loss_offdiag | 500 | 25/524 | 2542.85888671875 | 430.1101989746094 | 825.8516552124023 | 426.48876953125 | 2546.670654296875 | 631.3540344238281 | 1935.3111633300778 | 440.9183746337891 | [1649.4482934570312, 867.1008782958984, 634.7718157958984, 523.3940295410156, 454.54325897216796] |
| train/ufo/physics_peak_joint_velocity | 32 | 40/524 | 21.175565719604492 | 25.002763748168945 | 27.13430267572403 | 21.175565719604492 | 50.307456970214844 | 25.013154983520508 | 40.157902526855466 | 27.13430267572403 | [24.48808256785075, 28.08420213063558, 25.5439510345459, 31.627552577427455, 25.458200454711914] |
| train/ufo/physics_peak_normalized_velocity | 32 | 40/524 | 1.833019733428955 | 1.4925495386123657 | 2.630246613174677 | 1.4925495386123657 | 6.2884321212768555 | 2.164340376853943 | 5.019737815856933 | 2.630246613174677 | [2.0674542586008706, 2.962815999984741, 2.282729148864746, 3.3797411578042165, 2.3759581531797136] |
| train/ufo/q_loss | 500 | 25/524 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | [0.0, 0.0, 0.0, 0.0, 0.0] |
| train/ufo/replay_draw_new_ratio | 32 | 40/524 | 0.8333333134651184 | 1.2476190328598022 | 1.187849972397089 | 0.8333333134651184 | 1.2745097875595093 | 1.230701744556427 | 1.2716856181621552 | 1.187849972397089 | [1.001455048720042, 1.170246422290802, 1.2242901126543682, 1.252815502030509, 1.2665058715002877] |
| train/ufo/replay_draws | 32 | 40/524 | 81920.0 | 1073152.0 | 589056.0 | 81920.0 | 1073152.0 | 589824.0 | 1046937.6 | 589056.0 | [163840.0, 360448.0, 557056.0, 770048.0, 995913.1428571428] |
| train/ufo/replay_policy_age_max_updates | 32 | 40/524 | 40.0 | 524.0 | 287.625 | 40.0 | 524.0 | 288.0 | 511.2 | 287.625 | [80.0, 176.0, 272.0, 376.0, 486.2857142857143] |
| train/ufo/replay_policy_age_min_updates | 32 | 40/524 | 16.0 | 4.0 | 15.625 | 4.0 | 16.0 | 16.0 | 16.0 | 15.625 | [16.0, 16.0, 16.0, 16.0, 14.285714285714286] |
| train/ufo/replay_retained_fraction | 32 | 40/524 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | [1.0, 1.0, 1.0, 1.0, 1.0] |
| train/ufo/replay_size | 32 | 40/524 | 98304.0 | 860160.0 | 479232.0 | 98304.0 | 860160.0 | 479232.0 | 822067.2 | 479232.0 | [159744.0, 307200.0, 454656.0, 614400.0, 786432.0] |
| train/ufo/requested_action_abs_mean | 32 | 40/524 | 0.9318664073944092 | 0.8553743958473206 | 0.9036657623946667 | 0.8553743958473206 | 0.9362102746963501 | 0.9041068255901337 | 0.9328347384929657 | 0.9036657623946667 | [0.930436392625173, 0.9216419756412506, 0.9064130385716757, 0.8928248967443194, 0.8737973826272147] |
| train/ufo/requested_action_saturation_fraction | 32 | 40/524 | 0.7454922199249268 | 0.49290046095848083 | 0.612792911939323 | 0.49290046095848083 | 0.7454922199249268 | 0.612604558467865 | 0.6888439059257507 | 0.612792911939323 | [0.6870040396849314, 0.6524572670459747, 0.6151580115159353, 0.5863608292170933, 0.5395902097225189] |
| train/ufo/requested_applied_max_diff | 32 | 40/524 | 1.9509069919586182 | 1.9659883975982666 | 1.9578820578753948 | 1.9507105350494385 | 1.9660286903381348 | 1.9528013467788696 | 1.9660270810127258 | 1.9578820578753948 | [1.9524856209754944, 1.9581259489059448, 1.9553038080533345, 1.9602992704936437, 1.9620912415640694] |
| train/ufo/reset_control_samples | 32 | 40/524 | 6184.0 | 1010.0 | 1688.78125 | 960.0 | 6184.0 | 1356.5 | 2885.5999999999995 | 1688.78125 | [3100.3333333333335, 1900.8333333333333, 1401.1666666666667, 1116.142857142857, 1116.2857142857142] |
| train/ufo/reset_projection_outside_fraction | 32 | 40/524 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | [0.0, 0.0, 0.0, 0.0, 0.0] |
| train/ufo/reset_projection_residual_max_rad | 32 | 40/524 | 1.7881393432617188e-07 | 1.7881393432617188e-07 | 1.7601996660232544e-07 | 1.4901161193847656e-07 | 1.7881393432617188e-07 | 1.7881393432617188e-07 | 1.7881393432617188e-07 | 1.7601996660232544e-07 | [1.7881393432617188e-07, 1.7881393432617188e-07, 1.7384688059488931e-07, 1.7455645969935825e-07, 1.7455645969935825e-07] |
| train/ufo/reset_projection_residual_mean_rad | 32 | 40/524 | 6.551456976922054e-08 | 6.53484164558904e-08 | 6.50594447204611e-08 | 6.347805481254909e-08 | 6.595600865466622e-08 | 6.521218054444944e-08 | 6.593980828029089e-08 | 6.50594447204611e-08 | [6.554188658469684e-08, 6.520731332670948e-08, 6.497011995065805e-08, 6.468560899617972e-08, 6.496957841558729e-08] |
| train/ufo/reset_projection_residual_p95_rad | 32 | 40/524 | 1.1920928955078125e-07 | 1.1920928955078125e-07 | 1.1920928955078125e-07 | 1.1920928955078125e-07 | 1.1920928955078125e-07 | 1.1920928955078125e-07 | 1.1920928955078125e-07 | 1.1920928955078125e-07 | [1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07] |
| train/ufo/reset_root_lift_max_m | 32 | 40/524 | 0.021992022171616554 | 0.021992022171616554 | 0.021992022171616554 | 0.021992022171616554 | 0.021992022171616554 | 0.021992022171616554 | 0.021992022171616554 | 0.021992022171616554 | [0.021992022171616554, 0.021992022171616554, 0.021992022171616554, 0.021992022171616554, 0.021992022171616554] |
| train/ufo/sampling_rounds | 500 | 25/524 | 4.0 | 35.0 | 19.128 | 4.0 | 35.0 | 19.0 | 33.0 | 33.2 | [6.64, 12.88, 19.12, 25.36, 31.64] |
| train/ufo/strict_shadow_episode_len | 32 | 40/524 | 2.671428680419922 | 13.203307151794434 | 10.225755959749222 | 2.671428680419922 | 13.610203742980957 | 10.654653072357178 | 13.570495843887329 | 10.225755959749222 | [5.563082377115886, 8.086989243825277, 10.619133949279785, 12.8691987991333, 13.07493795667376] |
| train/ufo/strict_shadow_mixed_episode_len | 32 | 40/524 | 2.762711763381958 | 11.173701286315918 | 8.957468919456005 | 2.762711763381958 | 11.62303638458252 | 9.28528356552124 | 11.14443039894104 | 8.957468919456005 | [5.8266342878341675, 8.049322525660196, 9.145216306050619, 10.167908532278878, 11.048086711338588] |
| train/ufo/strict_shadow_tracking_episode_len | 32 | 40/524 | 2.1818182468414307 | 16.237863540649414 | 12.181302957236767 | 2.1818182468414307 | 19.260038375854492 | 12.766324043273926 | 17.496849822998048 | 12.181302957236767 | [5.283561031023662, 8.108095010121664, 12.837622483571371, 17.287914276123047, 15.915803364345006] |
| train/ufo/strict_velocity_exposure_fraction | 32 | 40/524 | 0.001970152836292982 | 0.00042127526830881834 | 0.0006034270518284757 | 0.0002702353522181511 | 0.001970152836292982 | 0.0004969057918060571 | 0.0011509770818520335 | 0.0006034270518284757 | [0.0009856292939124007, 0.0006402425691097354, 0.00045805852763199556, 0.0004991171964710313, 0.0004731804198984589] |
| train/ufo/strict_velocity_transition_fraction | 32 | 40/524 | 0.1451822966337204 | 0.0403238944709301 | 0.05318450945196673 | 0.02339681051671505 | 0.1451822966337204 | 0.0458984375 | 0.09137166589498519 | 0.05318450945196673 | [0.07853190259387095, 0.057128905629118286, 0.043253581350048385, 0.04610770036067281, 0.043666294642857144] |
| train/ufo/target_M | 500 | 25/524 | 3.714249610900879 | 0.769761323928833 | 0.9010501910448074 | 0.27199235558509827 | 4.031633377075195 | 0.6341975331306458 | 3.224414694309233 | 0.5939834141731262 | [2.0463985547423365, 0.5993702614307403, 0.6619444105029106, 0.6239998817443848, 0.5735378468036652] |
| train/ufo/target_Q | 500 | 25/524 | -3.8018884658813477 | -4.542535781860352 | -3.38129735660553 | -4.556524276733398 | -2.5592315196990967 | -3.2861260175704956 | -2.6293530821800233 | -4.337822198867798 | [-3.6319090700149537, -2.6828088355064392, -2.927153263092041, -3.4939394330978395, -4.1706761813163755] |
| train/ufo/target_auxQ | 500 | 25/524 | -4.563880443572998 | -7.515530586242676 | -5.763062532424927 | -7.515530586242676 | -4.5445733070373535 | -5.675161838531494 | -4.628812599182129 | -7.236239194869995 | [-4.67917688369751, -4.916061992645264, -5.668080039024353, -6.440103106498718, -7.111890640258789] |
| train/ufo/terminated_transitions | 32 | 40/524 | 70.0 | 1028.0 | 1504.40625 | 70.0 | 3873.0 | 1269.5 | 2415.65 | 1504.40625 | [2157.6666666666665, 1894.5, 1372.1666666666667, 1119.0, 1108.857142857143] |
| train/ufo/tracking_episode_starts | 32 | 40/524 | 3082.0 | 416.0 | 774.96875 | 317.0 | 3082.0 | 565.5 | 1526.1999999999998 | 774.96875 | [1591.5, 930.0, 583.1666666666666, 420.85714285714283, 460.7142857142857] |
| train/ufo/tracking_q_rmse_rad | 32 | 40/524 | 0.08157125860452652 | 0.4050597846508026 | 0.3826612497214228 | 0.08157125860452652 | 0.46119555830955505 | 0.40516602993011475 | 0.45691797882318497 | 0.3826612497214228 | [0.255019919325908, 0.38947585225105286, 0.40335757037003833, 0.4392996685845511, 0.4118488941873823] |
| train/ufo/tracking_unique_env_fraction | 32 | 40/524 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | [0.5, 0.5, 0.5, 0.5, 0.5] |
| train/ufo/tracking_windows_completed | 32 | 40/524 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | [0.0, 0.0, 0.0, 0.0, 0.0] |
| train/ufo/training_velocity_count | 32 | 40/524 | 14.0 | 13.0 | 20.34375 | 6.0 | 46.0 | 18.5 | 33.14999999999999 | 20.34375 | [18.166666666666668, 23.5, 17.666666666666668, 23.0, 19.142857142857142] |
| train/ufo/truncated_transitions | 32 | 40/524 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | [0.0, 0.0, 0.0, 0.0, 0.0] |
| train/ufo/unc_Q | 500 | 25/524 | 2.350160598754883 | 0.1457522213459015 | 0.6312415745854377 | 0.1457522213459015 | 2.615459680557251 | 0.3325766623020172 | 2.3503166675567626 | 0.15593053191900252 | [1.8038828414678574, 0.6321482256054878, 0.3369461286067963, 0.21884088829159737, 0.16438978895545006] |
| train/ufo/unc_auxQ | 500 | 25/524 | 1.1221225261688232 | 0.1706915944814682 | 0.3748118170499802 | 0.16869699954986572 | 1.1374986171722412 | 0.27781830728054047 | 0.8896294802427284 | 0.17939026653766632 | [0.7717503464221954, 0.43364708960056303, 0.2805091604590416, 0.2049332354962826, 0.18321925327181815] |
| train/ufo/velocity_reward_coverage_miss_count | 32 | 40/524 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | [0.0, 0.0, 0.0, 0.0, 0.0] |
| train/ufo/wall_time_s | 500 | 25/524 | 81.58558654785156 | 1089.0428466796875 | 577.1268080902099 | 81.58558654785156 | 1089.0428466796875 | 573.6463317871094 | 1028.9013244628907 | 1033.4733129882814 | [172.89245597839354, 374.9580294799805, 576.1682540893555, 778.8347735595703, 982.78052734375] |
| train/ufo/z_norm | 500 | 25/524 | 16.0 | 16.0 | 16.0 | 16.0 | 16.0 | 16.0 | 16.0 | 16.0 | [16.0, 16.0, 16.0, 16.0, 16.0] |

## 9. Checkpoint、回放、资源与完整实际配置

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
  "fingerprint": "sha256:8eb36b81219a6ce57a274e8f0fcf3b158350170de5c0876d1640ef4748b8b482",
  "execution": {
    "schema_version": "eb/training-binding/v1",
    "recipe_id": "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_compliance_v11",
    "algorithm_id": "bfm_fb_cpr_aux",
    "provider_id": "ufo",
    "runtime_id": "rlinf",
    "simulator_id": "isaaclab",
    "progress_unit": "learner_joint_update",
    "checkpoint_format": "ufo_agent_state_v2",
    "topology_id": "one_learner_three_samplers"
  }
}
```

```json
{
  "schema_version": "eb/ufo-sampler-sync/v1",
  "identity": {
    "state_sha256": "ec8c113e5d9f65c05a6c3dcd2c0c36321fcf42e58498314be96ba9116562c170",
    "version": 520,
    "online_transitions": 835584
  },
  "samplers": [
    {
      "rank": 0,
      "state_sha256": "ec8c113e5d9f65c05a6c3dcd2c0c36321fcf42e58498314be96ba9116562c170",
      "version": 520,
      "online_transitions": 835584
    },
    {
      "rank": 1,
      "state_sha256": "ec8c113e5d9f65c05a6c3dcd2c0c36321fcf42e58498314be96ba9116562c170",
      "version": 520,
      "online_transitions": 835584
    },
    {
      "rank": 2,
      "state_sha256": "ec8c113e5d9f65c05a6c3dcd2c0c36321fcf42e58498314be96ba9116562c170",
      "version": 520,
      "online_transitions": 835584
    }
  ],
  "sampling_rounds": 34
}
```

```json
{
  "capacity": 4000000,
  "count": 860160,
  "received": 860160,
  "cursor": 860160,
  "retained_fraction": 1.0,
  "fifo_evictions": 0,
  "age_newer_transition_count": {
    "0": 0.0,
    "0.25": 215039.75,
    "0.5": 430079.5,
    "0.75": 645119.25,
    "0.95": 817151.0499999999,
    "0.99": 851557.41,
    "1": 860159.0
  },
  "age_environment_control_steps": {
    "0": 0.0,
    "0.25": 34.999959309895836,
    "0.5": 69.99991861979166,
    "0.75": 104.9998779296875,
    "0.95": 132.99984537760417,
    "0.99": 138.59983886718751,
    "1": 139.99983723958334
  },
  "age_equivalent_physical_seconds": {
    "0": 0.0,
    "0.25": 0.6999991861979167,
    "0.5": 1.3999983723958334,
    "0.75": 2.09999755859375,
    "0.95": 2.6599969075520833,
    "0.99": 2.77199677734375,
    "1": 2.7999967447916667
  },
  "policy_age_joint_updates": {
    "0": 4.0,
    "0.25": 132.0,
    "0.5": 276.0,
    "0.75": 420.0,
    "0.95": 524.0,
    "0.99": 524.0,
    "1": 524.0
  },
  "policy_version_counts": {
    "0": 49152,
    "16": 24576,
    "24": 24576,
    "40": 24576,
    "56": 24576,
    "72": 24576,
    "88": 24576,
    "104": 24576,
    "120": 24576,
    "136": 24576,
    "152": 24576,
    "168": 24576,
    "184": 24576,
    "200": 24576,
    "216": 24576,
    "232": 24576,
    "248": 24576,
    "264": 24576,
    "280": 24576,
    "296": 24576,
    "312": 24576,
    "328": 24576,
    "344": 24576,
    "360": 24576,
    "376": 24576,
    "392": 24576,
    "408": 24576,
    "424": 24576,
    "440": 24576,
    "456": 24576,
    "472": 24576,
    "488": 24576,
    "504": 24576,
    "520": 24576
  },
  "episode_current_step": {
    "0": 0.0,
    "0.25": 3.0,
    "0.5": 7.0,
    "0.75": 13.0,
    "0.95": 27.0,
    "0.99": 39.0,
    "1": 87.0
  },
  "terminated_fraction": 0.05693708147321429,
  "truncated_fraction": 0.0,
  "rollout_states": {},
  "wall_time_age": "Qualification and formal Run clocks differ and per-transition wall times are not stored; do not infer exact age_seconds from replay count.",
  "physical_age_semantics": "age/6144 is equivalent total collection time per environment, not continuous episode age or bitwise physical resume"
}
```

末次采样使用三个sampler均为version520/online835584且相同hash；随后learner完成最后4更新到524，差异符合“先采样再更新”的时序，不是三卡权重不一致。

```json
{
  "wall_seconds": 1430.392636,
  "native_learner_wall_last_s": 1089.0428466796875,
  "gpu_sampled_max_used_MB": {
    "0": 16710,
    "1": 7478,
    "2": 7525,
    "3": 7478
  },
  "learner_torch_peak_allocated_bytes_logged": {
    "n": 32,
    "first_step": 40,
    "last_step": 524,
    "first": 16248861696.0,
    "last": 16251819008.0,
    "mean": 16251339008.0,
    "min": 16248861696.0,
    "max": 16251819008.0,
    "median": 16251819008.0,
    "p95": 16251819008.0,
    "last50_mean": 16251339008.0,
    "windows": [
      {
        "start": 25,
        "end": 124,
        "n": 6,
        "mean": 16249837909.333334
      },
      {
        "start": 125,
        "end": 224,
        "n": 6,
        "mean": 16251240106.666666
      },
      {
        "start": 225,
        "end": 324,
        "n": 6,
        "mean": 16251819008.0
      },
      {
        "start": 325,
        "end": 424,
        "n": 7,
        "mean": 16251819008.0
      },
      {
        "start": 425,
        "end": 524,
        "n": 7,
        "mean": 16251819008.0
      }
    ]
  },
  "learner_torch_peak_reserved_bytes_logged": {
    "n": 32,
    "first_step": 40,
    "last_step": 524,
    "first": 16456351744.0,
    "last": 16456351744.0,
    "mean": 16456351744.0,
    "min": 16456351744.0,
    "max": 16456351744.0,
    "median": 16456351744.0,
    "p95": 16456351744.0,
    "last50_mean": 16456351744.0,
    "windows": [
      {
        "start": 25,
        "end": 124,
        "n": 6,
        "mean": 16456351744.0
      },
      {
        "start": 125,
        "end": 224,
        "n": 6,
        "mean": 16456351744.0
      },
      {
        "start": 225,
        "end": 324,
        "n": 6,
        "mean": 16456351744.0
      },
      {
        "start": 325,
        "end": 424,
        "n": 7,
        "mean": 16456351744.0
      },
      {
        "start": 425,
        "end": 524,
        "n": 7,
        "mean": 16456351744.0
      }
    ]
  },
  "container_memory_observations": [
    {
      "BlockIO": "1.53GB / 463MB",
      "CPUPerc": "229.78%",
      "Container": "2b60d9e47cb0",
      "ID": "2b60d9e47cb0",
      "MemPerc": "41.49%",
      "MemUsage": "39.83GiB / 96GiB",
      "Name": "eb-run-run_4bc5e02a54b14f1ba0e9",
      "NetIO": "936B / 126B",
      "PIDs": "5635"
    },
    {
      "BlockIO": "1.53GB / 1.35GB",
      "CPUPerc": "361.91%",
      "Container": "2b60d9e47cb0",
      "ID": "2b60d9e47cb0",
      "MemPerc": "45.34%",
      "MemUsage": "43.53GiB / 96GiB",
      "Name": "eb-run-run_4bc5e02a54b14f1ba0e9",
      "NetIO": "1.08kB / 126B",
      "PIDs": "5635"
    },
    {
      "BlockIO": "1.53GB / 914MB",
      "CPUPerc": "293.96%",
      "Container": "2b60d9e47cb0",
      "ID": "2b60d9e47cb0",
      "MemPerc": "44.61%",
      "MemUsage": "42.83GiB / 96GiB",
      "Name": "eb-run-run_4bc5e02a54b14f1ba0e9",
      "NetIO": "1.01kB / 126B",
      "PIDs": "5635"
    }
  ]
}
```

Doctor为整卡total−free抽样，含驱动/显示开销；learner新增torch allocated/reserved峰值日志是该进程观测，TensorBoard字节标量有float32舍入。未测每个sampler进程完整峰值或组合满4M replay保存/恢复真实RSS峰；容器MemUsage是观测，不等于kernel memory.peak。4M未填满，本段不能证明扩容的FIFO抗退化收益。

原生Agent实际配置：

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

完整已解析RLinf配置：

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
      "replay_capacity": 4000000,
      "hidden_layers": 6,
      "topology_id": "one_learner_three_samplers",
      "learning_profile": "physical_limits_stable_slew_v4",
      "architecture_profile": "native_fb_v1",
      "rollout_expert_trajectories": true,
      "auxiliary_profile": "soft_velocity_v7",
      "scaling_profile": "env6144_batch2048_v6",
      "control_profile": "reachable_v2",
      "optimizer_profile": "stabilized_v7",
      "contact_profile": "contact_stable_v8",
      "ground_profile": "wide_plane_v9",
      "context_profile": "environment_tracking_v10",
      "ankle_control_profile": "kd25",
      "dynamics_profile": "contact_compliance_v11",
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
      "experiment_name": "RPO UFO compliance v11 native step524 3x2048",
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
        "id": "RPO-UFO-Isaac-RSI-tracking-contact-v10",
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

## 10. 复核与尚未证明的事项

第一轮复核：23DoF顺序/左右膝髋轴、train隔离、原q/dq保真、支撑腿与点法向速度、实际USD绑定。第二轮：4GPU拓扑/seed42–44、六Adam真实更新、16→24全状态回读、预算和不能绕过的门禁。第三轮：全部原始指标/逐关节事件/首步分组/右删失/源hash/完整checkpoint与停止决策。一次独立全代码复核的2项P1与3项P2均修复并留下行为检查；随后补齐恢复候选UI检查。

验证：最终原生相关118 passed；后端最终结果见原始测试日志摘录如下。缺依赖或命令路径不正确的早期测试调用已经纠正，不冒充实现失败；未运行不受本次改动影响的前端/Rust全量验证。

```text
.....................................ss.s.....ss......................ss [ 12%]
ss.....s.s...s.s........................................................ [ 25%]
........................................................................ [ 37%]
........................................................................ [ 50%]
........................................................................ [ 63%]
.......................................................s.....s.......... [ 75%]
......s....s............................................................ [ 88%]
.....................................s.....s.....................        [100%]
=============================== warnings summary ===============================
../../app/backend/.venv/lib/python3.11/site-packages/passlib/utils/__init__.py:854
  /app/backend/.venv/lib/python3.11/site-packages/passlib/utils/__init__.py:854: DeprecationWarning: 'crypt' is deprecated and slated for removal in Python 3.13
    from crypt import crypt as _crypt

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
550 passed, 29 skipped, 1 warning in 5.39s

```

没有独立评估/录像、没有正常行走验收、没有真机试验。soft contact改变训练MDP，刚性地面迁移未知；官方tracking起点覆盖窄；没有单独消融学习试验分离接触/筛选/奖励三项贡献。若门禁失败，停止本次授权分段，不自动放量；后续应先对具体终止阶段与足踝事件做证据驱动的修正，再声明新实验门槛。

下一轮应优先复现真实右踝峰值病例，区分足-ground和足-body的接触力，补齐每子步根姿态/终止时间戳及PD积分前后状态，并核对真实physics惯量、armature和电机速度-力矩包络。当前不能仅凭固定正弦动作对照宣称真实策略接触问题已解决；也不应依据训练均值微增自动扩容或重定义本次35/2.5门槛。

审计曾在一条right-arm速度8.00100040435791rad/s记录出现float64与环境float32阈值舍入差异；全786432转移按实际float32阈值8+0.001复核零不一致，物理容差未修改。

```json
{
  "float32_mismatches": 0,
  "float64_only_differences": [
    {
      "file": "interval_14.npz",
      "row": 6222,
      "dq": [
        1.8529651165008545,
        6.695748805999756,
        1.5541993379592896,
        0.7090833187103271,
        11.962297439575195,
        3.083467721939087,
        8.00100040435791,
        0.27758505940437317,
        0.6710438132286072,
        4.8027825355529785,
        1.4652560949325562,
        1.1741724014282227,
        1.0142794847488403,
        1.7872506380081177,
        1.0018279552459717,
        0.5723040103912354,
        3.970503807067871,
        4.593352317810059,
        0.7580880522727966,
        1.324660301208496,
        1.6428455114364624,
        3.8165173530578613,
        0.8681902289390564
      ],
      "actual": false,
      "float64_expected": true,
      "float32_expected": false,
      "float32_thresholds": [
        25.000999450683594,
        25.000999450683594,
        25.000999450683594,
        25.000999450683594,
        25.000999450683594,
        8.00100040435791,
        8.00100040435791,
        25.000999450683594,
        25.000999450683594,
        8.00100040435791,
        8.00100040435791,
        25.000999450683594,
        25.000999450683594,
        8.00100040435791,
        8.00100040435791,
        8.00100040435791,
        8.00100040435791,
        8.00100040435791,
        8.00100040435791,
        8.00100040435791,
        8.00100040435791,
        8.00100040435791,
        8.00100040435791
      ]
    }
  ]
}
```
