# UFO tracking/contact Kd-v10：实现、四卡资格与500新增更新完整总结

日期：2026-10-12。本文展开完整任务身份、参数、原始指标统计、五窗趋势、RSI/PD诊断、恢复与结果；独立模型不需要读取其他文档。所有身份路径均为绝对路径；没有Git暂存、提交或推送。

**本轮工程实现及500次新增联合更新完成；正常行走训练门槛未达，独立validation/held-out/acceptance全部not_run。** GPU0一个learner，GPU1–3每张2048环境，总6144；batch2048、完整435,116,484参数，非DDP。24→524，每个原生优化器新增500次更新，不是500个sampler step或PPO epoch。

五个100更新窗口的训练episode_len：8.25451386 → 11.558163 → 15.1944955 → 19.2216685 → 22.3133605步；严格影子存活：7.36798751 → 9.21281242 → 11.7758474 → 14.7346755 → 16.0141411步。末点分别22.0957165 / 15.6891518步，50Hz下分别0.44191433 / 0.313783035秒。严格影子仅训练内诊断，不能替代独立评估。

正式段首步终止48/54524=0.088034627%；控制边界速度峰52.1282806rad/s，逐物理子步峰55.7631607rad/s，逐关节额定归一化峰6.97039509倍。覆盖漏记0、地面越界0；修复计算漏记并不等于消除瞬态动力学超速。

结论：停止本轮观察段，不自动追加长训、不收紧或放宽冻结参数、不启动独立评估。工程成功/有限损失/梯度受控不代表正常行走通过。本轮没有新的0/N验收成绩，也没有评估录像；历史0/100不是本模型新成绩。

## 1. 已实现的行为变化与保持的边界

| 部分 | 本轮最终行为 | 验证依据 |
|---|---|---|
| Tracking | 恢复rollout_expert_trajectories=True；每sampler固定1024 tracking/1024 mixed；每env独立reset/250步续段 | 源/片段/起帧与实际RSI同次采样；权重同步不重编码活动窗口；容器250边界测试及正式trace逐行核对 |
| 速度软罚 | mean八个已完成2.5ms子步的sum relu(abs(dq)/Vmax−0.70)^2；权重−25只用一次 | 回放与223列interval trace逐行相等；速度coverage_miss=0 |
| 探索终止 | 训练速度1.3×原额定值；原位置/力矩/保护触地/高度/倾角约束保持 | 1.0×原限制影子首次违规长度独立锁存；无验收阈值修改 |
| RSI | train-only近地碰撞点abs(v_normal)≤0.15m/s；保留原q/dq与root速度，有限向上root lift | MuJoCo碰撞hull/FK/角速度叉乘及有限差分；两个来源普通/250交集池均非空 |
| 踝控制 | 选定名义Kd2→2.5；Kp40/27Nm/slew保持；正式段没有新增LPF | 相同RSI和动作序列baseline/LPF15/Kd2.5单因素配对诊断 |
| Replay | CPU transition FIFO容量4,000,000；v2格式24数组64MiB流式保存/恢复 | 4M与5.12M满池CPU实测；16→24→524六Adam/模型/池/随机状态完整回读 |
| 观测运输 | 新增RSI参考、tracking mode和interval trace经真实RLinf EnvOutput白名单运输 | 实际镜像缺字段RED复现→修复GREEN；actor375/FB733不变 |
| 种子 | new context seed=42+sampler rank，legacy seed逻辑保持 | 真实镜像HF初始化第二次覆写1051造成首次资格失败；修复后42/43/44分区hash一致 |

RSI=Reference State Initialization（参考状态初始化）：每次episode重置，从train参考动作中取一帧，将root姿态/速度及23个关节q/dq写入仿真。它不等于固定站姿，也不等于完整物理状态恢复。Tracking额外将该起点绑定到251有效状态组成的250转移窗口；B编码并取最多8帧前向均值，逐相位投影为z。合法tracking相邻z可以变化；mixed在同一episode内非100步边界必须保持相同z。

保持原生UFO FBcprAuxAgent六优化器/核心loss、完整网络及train数据隔离；不改成教师/PPO、不写外部仓、不改URDF原点/符号、不原地修改发布接口或旧模型。新的recipe/profile/image/checkpoint格式有独立身份及fingerprint。跨ground-v9奖励/控制/容量语义不兼容，因此新候选从seed42初始化16资格，不把旧524状态冒充精确恢复。

## 2. 冻结基线与控制选择

| ground-v9基线 | 值 |
|---|---|
| Train / Job / Run | trn_8bdabc160cd04d2bb100 / job_5157c51b35464239b2b6 / run_13cb5781b3944be88906 |
| ModelVersion | mv_f1c2fd66a7f6402fa7a1 |
| 网络/拓扑 | 435116484参数；GPU0learner/GPU1–3各2048；batch2048 |
| episode_len五窗 | 7.130 → 10.087 → 11.034 → 12.026 → 13.271步；末点13.498 |
| 物理峰/边界峰 | 67.525 / 45.985rad/s；后者right_ankle_pitch、round13、episode step0 |
| 首步 | 366/73200=0.50%；首步足竖直净力最大10049.84N |
| 子步违规但步末不超速 | 21733条；旧软罚仅取20ms末速度 |
| tracking/replay | tracking关闭；replay1M，最终860160，尚无FIFO淘汰 |
| 物理/地面 | physics400Hz、control50Hz、512m地面；原dq8/25rad/s；显式PD已含力矩限幅 |

本轮联合修改了RSI筛选、时序context、子步奖励、课程终止和Kd；新策略学习效果与历史基线不是单因素因果对照。控制选择本身采用固定起点/序列的配对诊断，不据此承诺训练一定改善。基线仅全局67.525子步标量不能归到右踝；本轮新增逐关节事件可以归属关节。

| 配对profile | 初始6144首步终止 | 首episode活跃转移数 | 子步/关节额定速度暴露率 | 逐关节归一化全局峰 | 最大rank足竖直净力p99 N | 全局足竖直净力峰 N |
|---|---|---|---|---|---|---|
| baseline | 4/6144 | 175751 | 3.14179671e-05 | 2.43563199 | 2160.24687 | 7853.89941 |
| lpf15 | 4/6144 | 175644 | 3.103486e-05 | 2.44004416 | 2175.77621 | 7897.229 |
| kd25 | 4/6144 | 176463 | 2.80265675e-05 | 2.3423481 | 2098.52977 | 8110.27588 |

诊断Job job_896cb307da8749ef9464 / Run run_53f27bf937f842b3b248；3profile×3samplers×2048、32控制步；每card256固定case保留详细事件，所有6144保留初始参考身份。相同seed42/43/44、起点SHA一致、anchor+踝0.03sin(5Hz)固定命令，无已训练策略、无独立模型评估。动作第一步sin=0，LPF不会改变这一首步，不能把首步一致解读为LPF一般无效。后续活跃样本不同，极值/暴露下降不唯一证明Kd因果收益。

Kd2.5相对baseline速度暴露约−10.8%、归一化峰约−3.8%，p99力改善；全局力峰8110.28N相对7853.90N反而+3.3%。15Hz LPF效果较弱，故选Kd2.5、不叠加LPF。Probe旧mass_kg字段为33.756893kg名义质量；实际DR质量另读PhysX get_masses启动回执，不能将名义值标为每环境实值。2500N不是该34kg机体3–5倍体重。

### 完整配对结果与实测增益

```json
{
  "status": "complete",
  "phases": {
    "baseline": {
      "first_samples": 6144,
      "first_terminated": 4,
      "first_termination_fraction": 0.0006510416666666666,
      "active_first_episode_transition_rows": 175751,
      "velocity_exposure_fraction": 3.141796706951757e-05,
      "maximum_rank_normalized_velocity_p99": 0.9578679716587069,
      "maximum_normalized_velocity": 2.4356319904327393,
      "maximum_rank_force_p99_N": 2160.246874999997,
      "maximum_force_N": 7853.8994140625,
      "mass_kg_per_rank": [
        33.756893157958984,
        33.756893157958984,
        33.756893157958984
      ],
      "feet_actuator_gains_per_rank": [
        {
          "stiffness": [
            41.8367919921875,
            36.95480728149414,
            36.40849304199219,
            37.90756607055664
          ],
          "damping": [
            1.9305672645568848,
            2.198690176010132,
            1.8282575607299805,
            1.9157224893569946
          ],
          "effort_limit": [
            27.0,
            27.0,
            27.0,
            27.0
          ]
        },
        {
          "stiffness": [
            42.33475112915039,
            41.71537399291992,
            41.48829650878906,
            36.63763427734375
          ],
          "damping": [
            2.034041404724121,
            2.0188677310943604,
            2.101466655731201,
            1.8262290954589844
          ],
          "effort_limit": [
            27.0,
            27.0,
            27.0,
            27.0
          ]
        },
        {
          "stiffness": [
            39.33578872680664,
            40.314212799072266,
            42.07460403442383,
            43.133277893066406
          ],
          "damping": [
            2.173532247543335,
            1.958551287651062,
            1.9837111234664917,
            2.0026564598083496
          ],
          "effort_limit": [
            27.0,
            27.0,
            27.0,
            27.0
          ]
        }
      ]
    },
    "lpf15": {
      "first_samples": 6144,
      "first_terminated": 4,
      "first_termination_fraction": 0.0006510416666666666,
      "active_first_episode_transition_rows": 175644,
      "velocity_exposure_fraction": 3.1034860038041375e-05,
      "maximum_rank_normalized_velocity_p99": 0.9588760691881183,
      "maximum_normalized_velocity": 2.4400441646575928,
      "maximum_rank_force_p99_N": 2175.776213378906,
      "maximum_force_N": 7897.22900390625,
      "mass_kg_per_rank": [
        33.756893157958984,
        33.756893157958984,
        33.756893157958984
      ],
      "feet_actuator_gains_per_rank": [
        {
          "stiffness": [
            41.8367919921875,
            36.95480728149414,
            36.40849304199219,
            37.90756607055664
          ],
          "damping": [
            1.9305672645568848,
            2.198690176010132,
            1.8282575607299805,
            1.9157224893569946
          ],
          "effort_limit": [
            27.0,
            27.0,
            27.0,
            27.0
          ]
        },
        {
          "stiffness": [
            42.33475112915039,
            41.71537399291992,
            41.48829650878906,
            36.63763427734375
          ],
          "damping": [
            2.034041404724121,
            2.0188677310943604,
            2.101466655731201,
            1.8262290954589844
          ],
          "effort_limit": [
            27.0,
            27.0,
            27.0,
            27.0
          ]
        },
        {
          "stiffness": [
            39.33578872680664,
            40.314212799072266,
            42.07460403442383,
            43.133277893066406
          ],
          "damping": [
            2.173532247543335,
            1.958551287651062,
            1.9837111234664917,
            2.0026564598083496
          ],
          "effort_limit": [
            27.0,
            27.0,
            27.0,
            27.0
          ]
        }
      ]
    },
    "kd25": {
      "first_samples": 6144,
      "first_terminated": 4,
      "first_termination_fraction": 0.0006510416666666666,
      "active_first_episode_transition_rows": 176463,
      "velocity_exposure_fraction": 2.802656746124141e-05,
      "maximum_rank_normalized_velocity_p99": 0.9464883291721344,
      "maximum_normalized_velocity": 2.342348098754883,
      "maximum_rank_force_p99_N": 2098.529768066407,
      "maximum_force_N": 8110.27587890625,
      "mass_kg_per_rank": [
        33.756893157958984,
        33.756893157958984,
        33.756893157958984
      ],
      "feet_actuator_gains_per_rank": [
        {
          "stiffness": [
            41.8367919921875,
            36.95480728149414,
            36.40849304199219,
            37.90756607055664
          ],
          "damping": [
            2.4132089614868164,
            2.7483627796173096,
            2.2853219509124756,
            2.394653081893921
          ],
          "effort_limit": [
            27.0,
            27.0,
            27.0,
            27.0
          ]
        },
        {
          "stiffness": [
            42.33475112915039,
            41.71537399291992,
            41.48829650878906,
            36.63763427734375
          ],
          "damping": [
            2.5425517559051514,
            2.5235846042633057,
            2.626833438873291,
            2.2827863693237305
          ],
          "effort_limit": [
            27.0,
            27.0,
            27.0,
            27.0
          ]
        },
        {
          "stiffness": [
            39.33578872680664,
            40.314212799072266,
            42.07460403442383,
            43.133277893066406
          ],
          "damping": [
            2.7169153690338135,
            2.4481890201568604,
            2.4796388149261475,
            2.5033206939697266
          ],
          "effort_limit": [
            27.0,
            27.0,
            27.0,
            27.0
          ]
        }
      ]
    }
  }
}
```

## 3. GPU资格、失败记录与正式任务身份

| 阶段 | TrainJob / Job / Run | 结果 |
|---|---|---|
| 初次16资格 | trn_583366d6bb7c4b58908b / job_41324a19bd8744d5818f / run_69b0d90a9d0d453391c3 | 更新前退出255；非OOM，无训练checkpoint |
| 修复后16 | trn_f05ad644bca5442099e8 / job_366258b6d87d4d3ea5ea / run_85ae9116da5549e98143 | 16更新/49152转移/round2；六Adam与哈希通过 |
| 完整恢复24 | trn_4db44db408e14ba8805a / job_8acc776bf4d144118c85 / run_9b47b8e1134b4fb39c0a | 24更新/73728转移/round3；相同fingerprint/六Adam通过 |
| 正式+500 | trn_c4ab80368f5b40d1a203 / job_6594eb07b76a48129ebd / run_1299df81ef5740f99f23 | 累计524/860160转移/round35；正式新增786432/32rounds |

第一次失败根因是继承的HF初始化后段将42+rank再次覆写为42+1009×rank，rank1实际1051而非43；共享tracking分区一致性守卫因此拒绝启动。实际镜像初始化前序AST测试复现后，在新context profile的该覆写后改回42+rank，legacy保持原种子。不是换seed碰运气，平台无自动重试；旧Run digest4375762f及日志完整保留。fork遥测告警为伴随现象，未确认其为失败根因。诊断旧c2镜像仍只用于已完成配对诊断；原临时候选recipe标blocked，实际Kd派生recipe在16→24通过后smoke_passed。

最终镜像：sha256:1970d0dc894b0a80c5dc27bd6a5b94c8d8463c52f3ec5c3e8108dda0b1b073f9；recipe rpo_bfm_ufo_agent_rlinf_isaac_4gpu_tracking_contact_kd_v10；profile rpo_bfm_ufo_isaac_4gpu_rsi_tracking_contact_kd_v10；fingerprint sha256:67c040bea5a934869eb530ad28896299690c16675ff47024eaf67ac078aec1ff。

基线127 image layers派生时达到Docker最大深度，先将验证过的ground-v9只读rootfs扁平化并保留ENV/workdir，再构建新image；flat digest sha256:c91f37b08dd2f623df4823c7b6f6394802909c8b20de45b429ff0dd03d15611f。只更新本仓派生镜像，没有写外部源码或更改旧模型绑定。

### 16完整TrainJob参数

```json
{
  "id": "trn_f05ad644bca5442099e8",
  "run_id": "run_85ae9116da5549e98143",
  "kind": "ppo",
  "name": "RPO UFO tracking-contact Kd2.5 seedfix1 native16 qualification",
  "model_name": "RPO-UFO-tracking-contact-kd-v10-seedfix1-step16",
  "status": "succeeded",
  "template_id": "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_tracking_contact_kd_v10",
  "image_digest": "sha256:1970d0dc894b0a80c5dc27bd6a5b94c8d8463c52f3ec5c3e8108dda0b1b073f9",
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
  "ray_port": 6409,
  "params": {
    "stack": "rlinf",
    "purpose": "qualification",
    "execution": {
      "recipe_id": "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_tracking_contact_kd_v10",
      "runtime_id": "rlinf",
      "provider_id": "ufo",
      "topology_id": "one_learner_three_samplers",
      "algorithm_id": "bfm_fb_cpr_aux",
      "simulator_id": "isaaclab",
      "progress_unit": "learner_joint_update",
      "schema_version": "eb/training-binding/v1",
      "checkpoint_format": "ufo_agent_state_v2"
    },
    "model_name": "RPO-UFO-tracking-contact-kd-v10-seedfix1-step16",
    "config_name": "rpo_bfm_ufo_isaac",
    "template_id": "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_tracking_contact_kd_v10",
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
      "auxiliary_profile": "soft_velocity_v7",
      "ground_patch_hash": "sha256:554a18609c9b8a420811fa6b1ded908a4bdf181e52e8d513c8772de6084eff59",
      "optimizer_profile": "stabilized_v7",
      "contact_patch_hash": "sha256:e20a0ff930b3b9430ece70c21ab316a06cef92b679b638c0a35d7fed62d6d814",
      "topology_patch_hash": "sha256:3078506628924464cb42d0b2f108b3fab502a43fd75e091f358d504565294d23",
      "ankle_control_profile": "kd25",
      "auxiliary_source_sha256": "sha256:dc57a21a994cb639d3833b5fa3cb0eefa8630721795227eae2ad0d7aa4e20445",
      "stabilization_patch_hash": "sha256:564f5452e12e3179112e6aa3020efba4cbcedc0ad90dd26b98aed78a50dd85b1",
      "rollout_profile_patch_hash": "sha256:777456ce171734326cf8c55eaa434fc5a69a27d0b2d0b725608dd54c190f4a77",
      "scaling_profile_patch_hash": "sha256:58a2dca8377ed2ede804c7101b39bc6992ec669ef95e610ddec4b2a59bfc6b20",
      "learning_profile_patch_hash": "sha256:d47623e48b4cc757d5e63336a1fa7d49234c3833a2f1c6bbdbee4fe7ffd9a272",
      "tracking_contact_patch_hash": "sha256:a0db175687641801ae1a6b755ad5dae2258a0af16cc6db71da3ebcaa6f43e99d",
      "action_control_source_sha256": "sha256:0cad3bac9e3efb3f1f37d6cac873f789548143b9cd597ed90a93a3045339cd24",
      "auxiliary_profile_patch_hash": "sha256:30e7b4ea7a3ec0e45cdbe0a9dbf718eb219c2c6fd36d07f9b18bd5222183a03a",
      "action_control_profile_patch_hash": "sha256:548260ca5e3eedd53ae739c00672c7e273e7835faed734778039cf5977b56e0d"
    },
    "environment_profile": {
      "id": "rpo_bfm_ufo_isaac_4gpu_rsi_tracking_contact_kd_v10",
      "hash": "sha256:df3bee43c5f4d2fb8ed67105695d426fe2e0b3ac1ef2f1314fd990a2ed4d31fe",
      "definition": {
        "task": "RPO23DoF完整435M网络；GPU0 learner、GPU1–3各2048环境；batch2048、CPU replay400万。固定50%独立环境跟随250步train专家片段，与RSI起点绑定；512m地面覆盖。 配对接触诊断选定踝关节Kd2.5，原Kp40/27Nm限幅/目标slew保持，无新增低通。",
        "reward": "九项UFO aux；速度罚为8个已完成物理子步的归一化均值，软阈值0.70、系数-25。原生六优化器；Actor梯度裁剪1，其余5；w<=50，D lr3e-6/reward[-2,2]。",
        "failure": "train接触点|v_normal|<=0.15m/s；物理400Hz、控制50Hz。探索速度终止1.3倍，原位置/力矩/宏观跌倒约束保持；原严格标准影子存活与所有子步违规独立记录。",
        "success": "四卡16→24完整状态恢复后新增500联合更新；最多2小时/无自动重试。训练不足不独立评估，不改变正常行走验收门槛。",
        "settings": {}
      }
    },
    "target_learner_updates": 16
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
    "resume_from": null
  },
  "resumed_from": null,
  "resume_step": null,
  "resume_mode": null,
  "hydra_overrides": [
    "env.train.total_num_envs=6144",
    "env.eval.total_num_envs=3",
    "+actor.model.topology_id=one_learner_three_samplers",
    "+actor.model.learning_profile=physical_limits_stable_slew_v4",
    "+actor.model.architecture_profile=native_fb_v1",
    "+actor.model.rollout_expert_trajectories=True",
    "actor.model.hidden_dim=2048",
    "actor.model.hidden_layers=6",
    "actor.model.batch_size=2048",
    "actor.micro_batch_size=2048",
    "actor.global_batch_size=2048",
    "actor.model.replay_capacity=4000000",
    "env.train.init_params.id=RPO-UFO-Isaac-RSI-tracking-contact-v10",
    "+actor.model.auxiliary_profile=soft_velocity_v7",
    "+actor.model.scaling_profile=env6144_batch2048_v6",
    "+actor.model.control_profile=reachable_v2",
    "+actor.model.optimizer_profile=stabilized_v7",
    "+actor.model.contact_profile=contact_stable_v8",
    "+actor.model.ground_profile=wide_plane_v9",
    "+actor.model.context_profile=environment_tracking_v10",
    "env.train.seed=${actor.seed}",
    "+actor.model.ankle_control_profile=kd25",
    "actor.seed=42",
    "~cluster.component_placement",
    "+cluster.component_placement={actor:0,env\\,\\ rollout:1-3}",
    "runner.logger.log_path=/workspace/output",
    "hydra.run.dir=/workspace/output/hydra",
    "runner.logger.experiment_name=\"RPO UFO tracking-contact Kd2.5 seedfix1 native16 qualification\""
  ],
  "run_dir": "/data/kk/embodiedbein/runs/run_85ae9116da5549e98143",
  "auto_register_latest": false,
  "metrics_summary": {
    "checkpoint": {
      "path": "/data/kk/models/runs/run_85ae9116da5549e98143",
      "step": 16,
      "bytes": 8074873453
    },
    "sampler_sync": {
      "identity": {
        "version": 0,
        "state_sha256": "3b814378e2cb8b8bb7a6410a6054657c0a0ab30552d393655ee79a002ced4b82",
        "online_transitions": 24576
      },
      "samplers": [
        {
          "rank": 0,
          "version": 0,
          "state_sha256": "3b814378e2cb8b8bb7a6410a6054657c0a0ab30552d393655ee79a002ced4b82",
          "online_transitions": 24576
        },
        {
          "rank": 1,
          "version": 0,
          "state_sha256": "3b814378e2cb8b8bb7a6410a6054657c0a0ab30552d393655ee79a002ced4b82",
          "online_transitions": 24576
        },
        {
          "rank": 2,
          "version": 0,
          "state_sha256": "3b814378e2cb8b8bb7a6410a6054657c0a0ab30552d393655ee79a002ced4b82",
          "online_transitions": 24576
        }
      ],
      "schema_version": "eb/ufo-sampler-sync/v1",
      "sampling_rounds": 1
    },
    "model_version_id": "mv_0f22d1d1a2fc459ebc80"
  },
  "error_message": null,
  "job_id": "job_366258b6d87d4d3ea5ea",
  "started_at": "2026-10-12T05:15:43.890661+00:00",
  "finished_at": "2026-10-12T05:21:56.842228+00:00",
  "created_at": "2026-10-12T05:15:33.147590+00:00"
}
```

### 24完整TrainJob参数

```json
{
  "id": "trn_4db44db408e14ba8805a",
  "run_id": "run_9b47b8e1134b4fb39c0a",
  "kind": "ppo",
  "name": "RPO UFO tracking-contact Kd2.5 24 native updates 3x2048",
  "model_name": "RPO-UFO-tracking-contact-kd-v10-step24",
  "status": "succeeded",
  "template_id": "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_tracking_contact_kd_v10",
  "image_digest": "sha256:1970d0dc894b0a80c5dc27bd6a5b94c8d8463c52f3ec5c3e8108dda0b1b073f9",
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
  "ray_port": 6395,
  "params": {
    "stack": "rlinf",
    "purpose": "qualification",
    "execution": {
      "recipe_id": "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_tracking_contact_kd_v10",
      "runtime_id": "rlinf",
      "provider_id": "ufo",
      "topology_id": "one_learner_three_samplers",
      "algorithm_id": "bfm_fb_cpr_aux",
      "simulator_id": "isaaclab",
      "progress_unit": "learner_joint_update",
      "schema_version": "eb/training-binding/v1",
      "checkpoint_format": "ufo_agent_state_v2"
    },
    "model_name": "RPO-UFO-tracking-contact-kd-v10-step24",
    "config_name": "rpo_bfm_ufo_isaac",
    "resume_mode": "trainer_state",
    "resume_step": 16,
    "template_id": "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_tracking_contact_kd_v10",
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
      "auxiliary_profile": "soft_velocity_v7",
      "ground_patch_hash": "sha256:554a18609c9b8a420811fa6b1ded908a4bdf181e52e8d513c8772de6084eff59",
      "optimizer_profile": "stabilized_v7",
      "contact_patch_hash": "sha256:e20a0ff930b3b9430ece70c21ab316a06cef92b679b638c0a35d7fed62d6d814",
      "topology_patch_hash": "sha256:3078506628924464cb42d0b2f108b3fab502a43fd75e091f358d504565294d23",
      "ankle_control_profile": "kd25",
      "auxiliary_source_sha256": "sha256:dc57a21a994cb639d3833b5fa3cb0eefa8630721795227eae2ad0d7aa4e20445",
      "stabilization_patch_hash": "sha256:564f5452e12e3179112e6aa3020efba4cbcedc0ad90dd26b98aed78a50dd85b1",
      "rollout_profile_patch_hash": "sha256:777456ce171734326cf8c55eaa434fc5a69a27d0b2d0b725608dd54c190f4a77",
      "scaling_profile_patch_hash": "sha256:58a2dca8377ed2ede804c7101b39bc6992ec669ef95e610ddec4b2a59bfc6b20",
      "learning_profile_patch_hash": "sha256:d47623e48b4cc757d5e63336a1fa7d49234c3833a2f1c6bbdbee4fe7ffd9a272",
      "tracking_contact_patch_hash": "sha256:a0db175687641801ae1a6b755ad5dae2258a0af16cc6db71da3ebcaa6f43e99d",
      "action_control_source_sha256": "sha256:0cad3bac9e3efb3f1f37d6cac873f789548143b9cd597ed90a93a3045339cd24",
      "auxiliary_profile_patch_hash": "sha256:30e7b4ea7a3ec0e45cdbe0a9dbf718eb219c2c6fd36d07f9b18bd5222183a03a",
      "action_control_profile_patch_hash": "sha256:548260ca5e3eedd53ae739c00672c7e273e7835faed734778039cf5977b56e0d"
    },
    "environment_profile": {
      "id": "rpo_bfm_ufo_isaac_4gpu_rsi_tracking_contact_kd_v10",
      "hash": "sha256:df3bee43c5f4d2fb8ed67105695d426fe2e0b3ac1ef2f1314fd990a2ed4d31fe",
      "definition": {
        "task": "RPO23DoF完整435M网络；GPU0 learner、GPU1–3各2048环境；batch2048、CPU replay400万。固定50%独立环境跟随250步train专家片段，与RSI起点绑定；512m地面覆盖。 配对接触诊断选定踝关节Kd2.5，原Kp40/27Nm限幅/目标slew保持，无新增低通。",
        "reward": "九项UFO aux；速度罚为8个已完成物理子步的归一化均值，软阈值0.70、系数-25。原生六优化器；Actor梯度裁剪1，其余5；w<=50，D lr3e-6/reward[-2,2]。",
        "failure": "train接触点|v_normal|<=0.15m/s；物理400Hz、控制50Hz。探索速度终止1.3倍，原位置/力矩/宏观跌倒约束保持；原严格标准影子存活与所有子步违规独立记录。",
        "success": "四卡16→24完整状态恢复后新增500联合更新；最多2小时/无自动重试。训练不足不独立评估，不改变正常行走验收门槛。",
        "settings": {}
      }
    },
    "resume_source_image": "eb/rlinf:rpo-bfm-ufo-tracking-contact-kd-v10",
    "source_train_job_id": "trn_f05ad644bca5442099e8",
    "target_learner_updates": 24
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
    "resume_from": "run_85ae9116da5549e98143"
  },
  "resumed_from": "trn_f05ad644bca5442099e8",
  "resume_step": 16,
  "resume_mode": "trainer_state",
  "hydra_overrides": [
    "env.train.total_num_envs=6144",
    "env.eval.total_num_envs=3",
    "+actor.model.topology_id=one_learner_three_samplers",
    "+actor.model.learning_profile=physical_limits_stable_slew_v4",
    "+actor.model.architecture_profile=native_fb_v1",
    "+actor.model.rollout_expert_trajectories=True",
    "actor.model.hidden_dim=2048",
    "actor.model.hidden_layers=6",
    "actor.model.batch_size=2048",
    "actor.micro_batch_size=2048",
    "actor.global_batch_size=2048",
    "actor.model.replay_capacity=4000000",
    "env.train.init_params.id=RPO-UFO-Isaac-RSI-tracking-contact-v10",
    "+actor.model.auxiliary_profile=soft_velocity_v7",
    "+actor.model.scaling_profile=env6144_batch2048_v6",
    "+actor.model.control_profile=reachable_v2",
    "+actor.model.optimizer_profile=stabilized_v7",
    "+actor.model.contact_profile=contact_stable_v8",
    "+actor.model.ground_profile=wide_plane_v9",
    "+actor.model.context_profile=environment_tracking_v10",
    "env.train.seed=${actor.seed}",
    "+actor.model.ankle_control_profile=kd25",
    "actor.seed=42",
    "~cluster.component_placement",
    "+cluster.component_placement={actor:0,env\\,\\ rollout:1-3}",
    "runner.logger.log_path=/workspace/output",
    "hydra.run.dir=/workspace/output/hydra",
    "runner.logger.experiment_name=\"RPO UFO tracking-contact Kd2.5 24 native updates 3x2048\"",
    "runner.resume_dir=/workspace/resume/global_step_16"
  ],
  "run_dir": "/data/kk/embodiedbein/runs/run_9b47b8e1134b4fb39c0a",
  "auto_register_latest": false,
  "metrics_summary": {
    "checkpoint": {
      "path": "/data/kk/models/runs/run_9b47b8e1134b4fb39c0a",
      "step": 24,
      "bytes": 8247942020
    },
    "sampler_sync": {
      "identity": {
        "version": 16,
        "state_sha256": "29fcbe84ea13c652a9d545fefc260fc18ea3571ea7721c0423d85b268f92db18",
        "online_transitions": 49152
      },
      "samplers": [
        {
          "rank": 0,
          "version": 16,
          "state_sha256": "29fcbe84ea13c652a9d545fefc260fc18ea3571ea7721c0423d85b268f92db18",
          "online_transitions": 49152
        },
        {
          "rank": 1,
          "version": 16,
          "state_sha256": "29fcbe84ea13c652a9d545fefc260fc18ea3571ea7721c0423d85b268f92db18",
          "online_transitions": 49152
        },
        {
          "rank": 2,
          "version": 16,
          "state_sha256": "29fcbe84ea13c652a9d545fefc260fc18ea3571ea7721c0423d85b268f92db18",
          "online_transitions": 49152
        }
      ],
      "schema_version": "eb/ufo-sampler-sync/v1",
      "sampling_rounds": 2
    },
    "model_version_id": "mv_4a3caaf89cca4b45990a"
  },
  "error_message": null,
  "job_id": "job_8acc776bf4d144118c85",
  "started_at": "2026-10-12T05:27:36.935848+00:00",
  "finished_at": "2026-10-12T05:33:46.133191+00:00",
  "created_at": "2026-10-12T05:27:31.935012+00:00"
}
```

### 524完整TrainJob参数

```json
{
  "id": "trn_c4ab80368f5b40d1a203",
  "run_id": "run_1299df81ef5740f99f23",
  "kind": "ppo",
  "name": "RPO UFO tracking-contact Kd2.5 524 native updates 3x2048",
  "model_name": "RPO-UFO-tracking-contact-kd-v10-step524",
  "status": "succeeded",
  "template_id": "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_tracking_contact_kd_v10",
  "image_digest": "sha256:1970d0dc894b0a80c5dc27bd6a5b94c8d8463c52f3ec5c3e8108dda0b1b073f9",
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
  "ray_port": 6423,
  "params": {
    "stack": "rlinf",
    "purpose": "training",
    "execution": {
      "recipe_id": "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_tracking_contact_kd_v10",
      "runtime_id": "rlinf",
      "provider_id": "ufo",
      "topology_id": "one_learner_three_samplers",
      "algorithm_id": "bfm_fb_cpr_aux",
      "simulator_id": "isaaclab",
      "progress_unit": "learner_joint_update",
      "schema_version": "eb/training-binding/v1",
      "checkpoint_format": "ufo_agent_state_v2"
    },
    "model_name": "RPO-UFO-tracking-contact-kd-v10-step524",
    "config_name": "rpo_bfm_ufo_isaac",
    "resume_mode": "trainer_state",
    "resume_step": 24,
    "template_id": "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_tracking_contact_kd_v10",
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
      "auxiliary_profile": "soft_velocity_v7",
      "ground_patch_hash": "sha256:554a18609c9b8a420811fa6b1ded908a4bdf181e52e8d513c8772de6084eff59",
      "optimizer_profile": "stabilized_v7",
      "contact_patch_hash": "sha256:e20a0ff930b3b9430ece70c21ab316a06cef92b679b638c0a35d7fed62d6d814",
      "topology_patch_hash": "sha256:3078506628924464cb42d0b2f108b3fab502a43fd75e091f358d504565294d23",
      "ankle_control_profile": "kd25",
      "auxiliary_source_sha256": "sha256:dc57a21a994cb639d3833b5fa3cb0eefa8630721795227eae2ad0d7aa4e20445",
      "stabilization_patch_hash": "sha256:564f5452e12e3179112e6aa3020efba4cbcedc0ad90dd26b98aed78a50dd85b1",
      "rollout_profile_patch_hash": "sha256:777456ce171734326cf8c55eaa434fc5a69a27d0b2d0b725608dd54c190f4a77",
      "scaling_profile_patch_hash": "sha256:58a2dca8377ed2ede804c7101b39bc6992ec669ef95e610ddec4b2a59bfc6b20",
      "learning_profile_patch_hash": "sha256:d47623e48b4cc757d5e63336a1fa7d49234c3833a2f1c6bbdbee4fe7ffd9a272",
      "tracking_contact_patch_hash": "sha256:a0db175687641801ae1a6b755ad5dae2258a0af16cc6db71da3ebcaa6f43e99d",
      "action_control_source_sha256": "sha256:0cad3bac9e3efb3f1f37d6cac873f789548143b9cd597ed90a93a3045339cd24",
      "auxiliary_profile_patch_hash": "sha256:30e7b4ea7a3ec0e45cdbe0a9dbf718eb219c2c6fd36d07f9b18bd5222183a03a",
      "action_control_profile_patch_hash": "sha256:548260ca5e3eedd53ae739c00672c7e273e7835faed734778039cf5977b56e0d"
    },
    "environment_profile": {
      "id": "rpo_bfm_ufo_isaac_4gpu_rsi_tracking_contact_kd_v10",
      "hash": "sha256:df3bee43c5f4d2fb8ed67105695d426fe2e0b3ac1ef2f1314fd990a2ed4d31fe",
      "definition": {
        "task": "RPO23DoF完整435M网络；GPU0 learner、GPU1–3各2048环境；batch2048、CPU replay400万。固定50%独立环境跟随250步train专家片段，与RSI起点绑定；512m地面覆盖。 配对接触诊断选定踝关节Kd2.5，原Kp40/27Nm限幅/目标slew保持，无新增低通。",
        "reward": "九项UFO aux；速度罚为8个已完成物理子步的归一化均值，软阈值0.70、系数-25。原生六优化器；Actor梯度裁剪1，其余5；w<=50，D lr3e-6/reward[-2,2]。",
        "failure": "train接触点|v_normal|<=0.15m/s；物理400Hz、控制50Hz。探索速度终止1.3倍，原位置/力矩/宏观跌倒约束保持；原严格标准影子存活与所有子步违规独立记录。",
        "success": "四卡16→24完整状态恢复后新增500联合更新；最多2小时/无自动重试。训练不足不独立评估，不改变正常行走验收门槛。",
        "settings": {}
      }
    },
    "resume_source_image": "eb/rlinf:rpo-bfm-ufo-tracking-contact-kd-v10",
    "source_train_job_id": "trn_4db44db408e14ba8805a",
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
    "resume_from": "run_9b47b8e1134b4fb39c0a"
  },
  "resumed_from": "trn_4db44db408e14ba8805a",
  "resume_step": 24,
  "resume_mode": "trainer_state",
  "hydra_overrides": [
    "env.train.total_num_envs=6144",
    "env.eval.total_num_envs=3",
    "+actor.model.topology_id=one_learner_three_samplers",
    "+actor.model.learning_profile=physical_limits_stable_slew_v4",
    "+actor.model.architecture_profile=native_fb_v1",
    "+actor.model.rollout_expert_trajectories=True",
    "actor.model.hidden_dim=2048",
    "actor.model.hidden_layers=6",
    "actor.model.batch_size=2048",
    "actor.micro_batch_size=2048",
    "actor.global_batch_size=2048",
    "actor.model.replay_capacity=4000000",
    "env.train.init_params.id=RPO-UFO-Isaac-RSI-tracking-contact-v10",
    "+actor.model.auxiliary_profile=soft_velocity_v7",
    "+actor.model.scaling_profile=env6144_batch2048_v6",
    "+actor.model.control_profile=reachable_v2",
    "+actor.model.optimizer_profile=stabilized_v7",
    "+actor.model.contact_profile=contact_stable_v8",
    "+actor.model.ground_profile=wide_plane_v9",
    "+actor.model.context_profile=environment_tracking_v10",
    "env.train.seed=${actor.seed}",
    "+actor.model.ankle_control_profile=kd25",
    "actor.seed=42",
    "~cluster.component_placement",
    "+cluster.component_placement={actor:0,env\\,\\ rollout:1-3}",
    "runner.logger.log_path=/workspace/output",
    "hydra.run.dir=/workspace/output/hydra",
    "runner.logger.experiment_name=\"RPO UFO tracking-contact Kd2.5 524 native updates 3x2048\"",
    "runner.resume_dir=/workspace/resume/global_step_24"
  ],
  "run_dir": "/data/kk/embodiedbein/runs/run_1299df81ef5740f99f23",
  "auto_register_latest": false,
  "metrics_summary": {
    "checkpoint": {
      "path": "/data/kk/models/runs/run_1299df81ef5740f99f23",
      "step": 524,
      "bytes": 13785996204
    },
    "sampler_sync": {
      "identity": {
        "version": 520,
        "state_sha256": "caa089f02748a842d511462594f41e626aa8a441721427cc1e021fe2972dc007",
        "online_transitions": 835584
      },
      "samplers": [
        {
          "rank": 0,
          "version": 520,
          "state_sha256": "caa089f02748a842d511462594f41e626aa8a441721427cc1e021fe2972dc007",
          "online_transitions": 835584
        },
        {
          "rank": 1,
          "version": 520,
          "state_sha256": "caa089f02748a842d511462594f41e626aa8a441721427cc1e021fe2972dc007",
          "online_transitions": 835584
        },
        {
          "rank": 2,
          "version": 520,
          "state_sha256": "caa089f02748a842d511462594f41e626aa8a441721427cc1e021fe2972dc007",
          "online_transitions": 835584
        }
      ],
      "schema_version": "eb/ufo-sampler-sync/v1",
      "sampling_rounds": 34
    },
    "model_version_id": "mv_e95f1d9da4574b05b300"
  },
  "error_message": null,
  "job_id": "job_6594eb07b76a48129ebd",
  "started_at": "2026-10-12T05:52:45.930799+00:00",
  "finished_at": "2026-10-12T06:17:10.983341+00:00",
  "created_at": "2026-10-12T05:52:39.590744+00:00"
}
```

正式完整JobSpec和Run：
```json
{
  "id": "job_6594eb07b76a48129ebd",
  "kind": "train.rl",
  "status": "succeeded",
  "embodiment_env_id": "ee_rpo_v2_bfm_23dof_v2",
  "priority": 50,
  "created_by": "user:kk",
  "spec": {
    "kind": "train.rl",
    "name": "RPO UFO tracking-contact Kd2.5 524 native updates 3x2048",
    "gates": [],
    "image": "sha256:1970d0dc894b0a80c5dc27bd6a5b94c8d8463c52f3ec5c3e8108dda0b1b073f9",
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
      "resume_from": "run_9b47b8e1134b4fb39c0a"
    },
    "labels": {},
    "params": {
      "stack": "rlinf",
      "purpose": "training",
      "execution": {
        "recipe_id": "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_tracking_contact_kd_v10",
        "runtime_id": "rlinf",
        "provider_id": "ufo",
        "topology_id": "one_learner_three_samplers",
        "algorithm_id": "bfm_fb_cpr_aux",
        "simulator_id": "isaaclab",
        "progress_unit": "learner_joint_update",
        "schema_version": "eb/training-binding/v1",
        "checkpoint_format": "ufo_agent_state_v2"
      },
      "model_name": "RPO-UFO-tracking-contact-kd-v10-step524",
      "config_name": "rpo_bfm_ufo_isaac",
      "resume_mode": "trainer_state",
      "resume_step": 24,
      "template_id": "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_tracking_contact_kd_v10",
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
        "auxiliary_profile": "soft_velocity_v7",
        "ground_patch_hash": "sha256:554a18609c9b8a420811fa6b1ded908a4bdf181e52e8d513c8772de6084eff59",
        "optimizer_profile": "stabilized_v7",
        "contact_patch_hash": "sha256:e20a0ff930b3b9430ece70c21ab316a06cef92b679b638c0a35d7fed62d6d814",
        "topology_patch_hash": "sha256:3078506628924464cb42d0b2f108b3fab502a43fd75e091f358d504565294d23",
        "ankle_control_profile": "kd25",
        "auxiliary_source_sha256": "sha256:dc57a21a994cb639d3833b5fa3cb0eefa8630721795227eae2ad0d7aa4e20445",
        "stabilization_patch_hash": "sha256:564f5452e12e3179112e6aa3020efba4cbcedc0ad90dd26b98aed78a50dd85b1",
        "rollout_profile_patch_hash": "sha256:777456ce171734326cf8c55eaa434fc5a69a27d0b2d0b725608dd54c190f4a77",
        "scaling_profile_patch_hash": "sha256:58a2dca8377ed2ede804c7101b39bc6992ec669ef95e610ddec4b2a59bfc6b20",
        "learning_profile_patch_hash": "sha256:d47623e48b4cc757d5e63336a1fa7d49234c3833a2f1c6bbdbee4fe7ffd9a272",
        "tracking_contact_patch_hash": "sha256:a0db175687641801ae1a6b755ad5dae2258a0af16cc6db71da3ebcaa6f43e99d",
        "action_control_source_sha256": "sha256:0cad3bac9e3efb3f1f37d6cac873f789548143b9cd597ed90a93a3045339cd24",
        "auxiliary_profile_patch_hash": "sha256:30e7b4ea7a3ec0e45cdbe0a9dbf718eb219c2c6fd36d07f9b18bd5222183a03a",
        "action_control_profile_patch_hash": "sha256:548260ca5e3eedd53ae739c00672c7e273e7835faed734778039cf5977b56e0d"
      },
      "environment_profile": {
        "id": "rpo_bfm_ufo_isaac_4gpu_rsi_tracking_contact_kd_v10",
        "hash": "sha256:df3bee43c5f4d2fb8ed67105695d426fe2e0b3ac1ef2f1314fd990a2ed4d31fe",
        "definition": {
          "task": "RPO23DoF完整435M网络；GPU0 learner、GPU1–3各2048环境；batch2048、CPU replay400万。固定50%独立环境跟随250步train专家片段，与RSI起点绑定；512m地面覆盖。 配对接触诊断选定踝关节Kd2.5，原Kp40/27Nm限幅/目标slew保持，无新增低通。",
          "reward": "九项UFO aux；速度罚为8个已完成物理子步的归一化均值，软阈值0.70、系数-25。原生六优化器；Actor梯度裁剪1，其余5；w<=50，D lr3e-6/reward[-2,2]。",
          "failure": "train接触点|v_normal|<=0.15m/s；物理400Hz、控制50Hz。探索速度终止1.3倍，原位置/力矩/宏观跌倒约束保持；原严格标准影子存活与所有子步违规独立记录。",
          "success": "四卡16→24完整状态恢复后新增500联合更新；最多2小时/无自动重试。训练不足不独立评估，不改变正常行走验收门槛。",
          "settings": {}
        }
      },
      "resume_source_image": "eb/rlinf:rpo-bfm-ufo-tracking-contact-kd-v10",
      "source_train_job_id": "trn_4db44db408e14ba8805a",
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
      "id": "run_1299df81ef5740f99f23",
      "attempt": 1,
      "status": "finished",
      "abort_reason": null,
      "metrics_summary": {
        "checkpoint": {
          "path": "/data/kk/models/runs/run_1299df81ef5740f99f23",
          "step": 524,
          "bytes": 13785996204
        },
        "sampler_sync": {
          "identity": {
            "version": 520,
            "state_sha256": "caa089f02748a842d511462594f41e626aa8a441721427cc1e021fe2972dc007",
            "online_transitions": 835584
          },
          "samplers": [
            {
              "rank": 0,
              "version": 520,
              "state_sha256": "caa089f02748a842d511462594f41e626aa8a441721427cc1e021fe2972dc007",
              "online_transitions": 835584
            },
            {
              "rank": 1,
              "version": 520,
              "state_sha256": "caa089f02748a842d511462594f41e626aa8a441721427cc1e021fe2972dc007",
              "online_transitions": 835584
            },
            {
              "rank": 2,
              "version": 520,
              "state_sha256": "caa089f02748a842d511462594f41e626aa8a441721427cc1e021fe2972dc007",
              "online_transitions": 835584
            }
          ],
          "schema_version": "eb/ufo-sampler-sync/v1",
          "sampling_rounds": 34
        },
        "model_version_id": "mv_e95f1d9da4574b05b300"
      }
    }
  ]
}
```
```json
{
  "id": "run_1299df81ef5740f99f23",
  "job_id": "job_6594eb07b76a48129ebd",
  "attempt": 1,
  "status": "finished",
  "abort_reason": null,
  "metrics_summary": {
    "checkpoint": {
      "path": "/data/kk/models/runs/run_1299df81ef5740f99f23",
      "step": 524,
      "bytes": 13785996204
    },
    "sampler_sync": {
      "identity": {
        "version": 520,
        "state_sha256": "caa089f02748a842d511462594f41e626aa8a441721427cc1e021fe2972dc007",
        "online_transitions": 835584
      },
      "samplers": [
        {
          "rank": 0,
          "version": 520,
          "state_sha256": "caa089f02748a842d511462594f41e626aa8a441721427cc1e021fe2972dc007",
          "online_transitions": 835584
        },
        {
          "rank": 1,
          "version": 520,
          "state_sha256": "caa089f02748a842d511462594f41e626aa8a441721427cc1e021fe2972dc007",
          "online_transitions": 835584
        },
        {
          "rank": 2,
          "version": 520,
          "state_sha256": "caa089f02748a842d511462594f41e626aa8a441721427cc1e021fe2972dc007",
          "online_transitions": 835584
        }
      ],
      "schema_version": "eb/ufo-sampler-sync/v1",
      "sampling_rounds": 34
    },
    "model_version_id": "mv_e95f1d9da4574b05b300"
  },
  "gpu_ids": [
    0,
    1,
    2,
    3
  ],
  "provenance": "{\"code_ref\": \"sha256:04e20ece2e3e3f3603a98b24e04aba42023fb31a7395beddfdac2a5d522de54d\", \"config_hash\": \"faba1ddec0f70d468480d8bf39338e145d1f590dd9753d57a5f7d80cce67b3af\", \"data_version\": null, \"image_digest\": \"sha256:1970d0dc894b0a80c5dc27bd6a5b94c8d8463c52f3ec5c3e8108dda0b1b073f9\", \"seed\": null}",
  "artifacts": [
    {
      "id": "art_8032d8bf3bb5437ba5f0",
      "kind": "rlinf_checkpoint",
      "sha256": "sha256:19df777084a09f0449b2fb913acf701105274015f30f9721494bc8fb5d48d7a7",
      "size_bytes": 13785996204,
      "storage": "/data/kk/models/runs/run_1299df81ef5740f99f23"
    }
  ]
}
```

### 最终模型版本及空评估报告集合

```json
{
  "id": "mv_e95f1d9da4574b05b300",
  "family_id": "mf_rpo_v2_bfm_zero",
  "name": "RPO-UFO-tracking-contact-kd-v10-step524",
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
    "run_id": "run_1299df81ef5740f99f23",
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
      "image": "sha256:1970d0dc894b0a80c5dc27bd6a5b94c8d8463c52f3ec5c3e8108dda0b1b073f9",
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
      "recipe_id": "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_tracking_contact_kd_v10",
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
    "template_id": "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_tracking_contact_kd_v10",
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
    "train_job_id": "job_6594eb07b76a48129ebd",
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
      "tracking_context": "environment_tracking_v10",
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
    "artifact_sha256": "sha256:19df777084a09f0449b2fb913acf701105274015f30f9721494bc8fb5d48d7a7",
    "checkpoint_path": "/data/kk/models/runs/run_1299df81ef5740f99f23",
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
      "id": "rpo_bfm_ufo_isaac_4gpu_rsi_tracking_contact_kd_v10",
      "hash": "sha256:df3bee43c5f4d2fb8ed67105695d426fe2e0b3ac1ef2f1314fd990a2ed4d31fe",
      "definition": {
        "task": "RPO23DoF完整435M网络；GPU0 learner、GPU1–3各2048环境；batch2048、CPU replay400万。固定50%独立环境跟随250步train专家片段，与RSI起点绑定；512m地面覆盖。 配对接触诊断选定踝关节Kd2.5，原Kp40/27Nm限幅/目标slew保持，无新增低通。",
        "reward": "九项UFO aux；速度罚为8个已完成物理子步的归一化均值，软阈值0.70、系数-25。原生六优化器；Actor梯度裁剪1，其余5；w<=50，D lr3e-6/reward[-2,2]。",
        "failure": "train接触点|v_normal|<=0.15m/s；物理400Hz、控制50Hz。探索速度终止1.3倍，原位置/力矩/宏观跌倒约束保持；原严格标准影子存活与所有子步违规独立记录。",
        "success": "四卡16→24完整状态恢复后新增500联合更新；最多2小时/无自动重试。训练不足不独立评估，不改变正常行走验收门槛。",
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
  "created_at": "2026-10-12T06:17:10.670707Z",
  "train_job_name": "RPO UFO tracking-contact Kd2.5 524 native updates 3x2048",
  "train_page_id": "trn_c4ab80368f5b40d1a203",
  "artifact": {
    "id": "art_8032d8bf3bb5437ba5f0",
    "kind": "rlinf_checkpoint",
    "sha256": "sha256:19df777084a09f0449b2fb913acf701105274015f30f9721494bc8fb5d48d7a7",
    "size_bytes": 13785996204,
    "storage": "/data/kk/models/runs/run_1299df81ef5740f99f23",
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
            "sha256": "8308bae87e57c7a9add28109d7bdfcf33e9da90c49c5501453907a5c465e75fd"
          },
          {
            "path": "model/native/global_step_524/optimizers.pth",
            "bytes": 3481181041,
            "sha256": "5f11027f93401320337817b959507e66411be77d6f9ee10d7637b3630fcf0265"
          },
          {
            "path": "model/native/global_step_524/replay/array_000.npy",
            "bytes": 79134848,
            "sha256": "88434514aed73bed91cebc3b607827c7ac21222d70e669551c5c09a13bbed504"
          },
          {
            "path": "model/native/global_step_524/replay/array_001.npy",
            "bytes": 3440768,
            "sha256": "3c4e3a9223fc77f98adf782577599418af4a23ea3dcd4fa5085b96e6a99157da"
          },
          {
            "path": "model/native/global_step_524/replay/array_002.npy",
            "bytes": 3440768,
            "sha256": "862ad3c71076afbaec4a17841c8acdd23f184b6628eddfdcd797ecd33665a413"
          },
          {
            "path": "model/native/global_step_524/replay/array_003.npy",
            "bytes": 3440768,
            "sha256": "1d11af5c57ba70849e0143fb20d9ab15e3bb932a626014b6027d0ac584dabd7f"
          },
          {
            "path": "model/native/global_step_524/replay/array_004.npy",
            "bytes": 3440768,
            "sha256": "d382cd1e020e8ccd56c8aaac07170e1554163aeb55ecab5bf41147a52dbe999a"
          },
          {
            "path": "model/native/global_step_524/replay/array_005.npy",
            "bytes": 3440768,
            "sha256": "ab8e57d50f0f2d55a3c22e040919d4276753dfc30e727053a40fa9d01eab0cd8"
          },
          {
            "path": "model/native/global_step_524/replay/array_006.npy",
            "bytes": 3440768,
            "sha256": "66e8fe76ed750b3c5a40cf7030796b4e6c0669ecdf43b4575237695c44fe3c99"
          },
          {
            "path": "model/native/global_step_524/replay/array_007.npy",
            "bytes": 3440768,
            "sha256": "c0263159e5ca6d2ccbf05c54bf4b7de41c2d6692ec3a570d995d78aea5d42962"
          },
          {
            "path": "model/native/global_step_524/replay/array_008.npy",
            "bytes": 3440768,
            "sha256": "7480a31190b47f7c43afb92431e6ad8a2f0550479283c160ef995a7150ab6f96"
          },
          {
            "path": "model/native/global_step_524/replay/array_009.npy",
            "bytes": 3440768,
            "sha256": "14cd4be994aa58a9bb2169671ad21120aaef95c770186679e7760f6777d25109"
          },
          {
            "path": "model/native/global_step_524/replay/array_010.npy",
            "bytes": 6881408,
            "sha256": "36b9f69250b748a7f6a9b3f6c3bb31e501748bbde9d82732b4bbc02c9f557774"
          },
          {
            "path": "model/native/global_step_524/replay/array_011.npy",
            "bytes": 1032192128,
            "sha256": "dff429b75427d966f484466c2b9669c773685c348dad8a0e5d63fe06c0de67cc"
          },
          {
            "path": "model/native/global_step_524/replay/array_012.npy",
            "bytes": 79134848,
            "sha256": "884c8ad045b951ef9f8f29049fedc334cb44b1edbaec133b468e4172a339f7e9"
          },
          {
            "path": "model/native/global_step_524/replay/array_013.npy",
            "bytes": 1231749248,
            "sha256": "ca494535328782571018a1e4cf71d8702b512dc3468d5320708d39a9f8912d01"
          },
          {
            "path": "model/native/global_step_524/replay/array_014.npy",
            "bytes": 178913408,
            "sha256": "48b7cc938429ee7608348bf47da846aae1ea89e4cc2c092d527510f09f3a1510"
          },
          {
            "path": "model/native/global_step_524/replay/array_015.npy",
            "bytes": 860288,
            "sha256": "6d5ad5052bcbb42776fffe32e7dc5eebae89ffa5284ff052ec8c5442a01f88a3"
          },
          {
            "path": "model/native/global_step_524/replay/array_016.npy",
            "bytes": 860288,
            "sha256": "31ba486699f2858be0234f3e959120e2c30c2eb7a9f8ff4d60dc548f01cb4e90"
          },
          {
            "path": "model/native/global_step_524/replay/array_017.npy",
            "bytes": 1032192128,
            "sha256": "6983622e7e711bc3172c811115545a46535152f45cb85313f81e958a8fb420b8"
          },
          {
            "path": "model/native/global_step_524/replay/array_018.npy",
            "bytes": 79134848,
            "sha256": "992a324592200cd4267c70ad6c22bd5a593d17722e960e74b25a880adc471dbc"
          },
          {
            "path": "model/native/global_step_524/replay/array_019.npy",
            "bytes": 1231749248,
            "sha256": "6d9886c29ab007323ec808b8d30618b500e950010e763182ae2aaa1b45a32ce1"
          },
          {
            "path": "model/native/global_step_524/replay/array_020.npy",
            "bytes": 178913408,
            "sha256": "f08bd0be9275d6e6b19edd33f19e54789f2f1412fc44afed54196363ef4c7a2d"
          },
          {
            "path": "model/native/global_step_524/replay/array_021.npy",
            "bytes": 6881408,
            "sha256": "25805036cef312d11ecfb37fc73778886f63caf007db4724d99576cd1045ada9"
          },
          {
            "path": "model/native/global_step_524/replay/array_022.npy",
            "bytes": 6881408,
            "sha256": "0621d369a239c90009699ce7fa60a229c970af5d1b7304aee5f84ff5c7ef8cac"
          },
          {
            "path": "model/native/global_step_524/replay/array_023.npy",
            "bytes": 880803968,
            "sha256": "7decd6424b136c8e7665a1c03548a7f2aa6d3b70053ec4bcf91e92eeab364367"
          },
          {
            "path": "model/native/global_step_524/resume-manifest.json",
            "bytes": 28879,
            "sha256": "6a9b23fae91402a92ec2d7f679b7d8b54725ba4f8bcb6d465bc98562aa54e01e"
          },
          {
            "path": "model/native/global_step_524/trainer-state.pt",
            "bytes": 903671441,
            "sha256": "b2b62e16eb5ec1ea29e64ec3e8513d9a455f01038c2243474af07cf936c93561"
          },
          {
            "path": "model/native-checkpoint.json",
            "bytes": 88,
            "sha256": "47caad91a80d1cbebbb2a750326d52642be4d75fd37c559f3c8a1958ed2c0b6b"
          },
          {
            "path": "model/resolved-config.yaml",
            "bytes": 3970,
            "sha256": "c34f0ec37848e0c7820c05a5ca86f322a2de5b6dd8fc9c221b39ae79b07c0f09"
          },
          {
            "path": "model/source-receipt.json",
            "bytes": 2741,
            "sha256": "a635274be7115f364d92d8dc8815f98cadda95231a3daad4831d7bc37d646c89"
          }
        ],
        "format": "ufo_agent_state_v2",
        "run_id": "run_1299df81ef5740f99f23",
        "status": "complete",
        "execution": {
          "recipe_id": "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_tracking_contact_kd_v10",
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
        "template_id": "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_tracking_contact_kd_v10",
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

## 4. 完整网络、优化、采样和控制配置

F/Actor/Q_D/Q_aux宽2048、6个residual块；D1024×3；B256×1；z256。Actor375由52 state+4×75 history+23 last applied action组成；FB733由52 state+358 privileged+4×75 history+23 last applied组成。批2048；样本来自在线FIFO，expert独立train-only连续窗口。Actor/F/Q_D/Q_aux LR3e-4，B1e-5，D3e-6；discount.98、FB tau.01、Q tau.005、ortho_coef100、zpool8192、relabel_ratio.8、expert_asm_ratio.6、actor std.05；AMP/compile/cudagraphs=False。下面实际native/config为最终真值。

每采样round6144×4=24576 transitions，10240 warmup后每触发16联合更新；正常draw/new=16×2048/24576=1.33333，updates/new=0.000651042。正式500末次partial4更新，新增draw/new=500×2048/786432=1.30208333，updates/new=0.000635782878。每环境正式仅128控制步=2.56秒物理交互，小于250步专家完整窗口；500段中未出现250窗口完成的分母不足以判定实现失败，CPU测试已覆盖250到期，但GPU完整长周期仍未观察到。

Actor范数clip1，其余五Adam clip5；w=detach(mean(abs(Q_fb)))≤50；D奖励[-2,2]，其raw值也保留。native clip_grad_norm=0由owned hook实际裁剪，六组前/后范数均入全指标表。原八aux权重保持，第九为−25、阈值0.70。日志aux_rew/limits_dof_vel是RAW正惩罚；乘−25的贡献在原生agent中只计一次，然后所有aux加权和共同进入原生RewardNormalizer。mean_aux_reward是归一化总和，非第九归一化贡献；联合非线性normalizer下不能把它机械按权重拆分为独立速度项。

物理dt.0025、decimation8、控制dt.02；episode horizon1000。训练速度1.3×原额定8/25，strict shadow与验收仍原8/25；位置/力矩原边界不变，保护torso/左右thigh yaw/roll接触>1N、高度<.50m、倾角>.80rad保持。硬flags是八子步锁存，控制边界返回terminated，非第一次毛刺即停止剩余物理积分。512m离线mesh覆盖三卡实际网格，startup最低运动余量≥20m。

| PD组 | Kp名义 | Kd名义 | armature | 原额定速度rad/s | 力矩Nm | 训练delay |
|---|---|---|---|---|---|---|
| thigh yaw/roll/pitch | 100 | 3.3 | 0.01 | 25 | 120 | 0 |
| knee/torso | 150 | 5 | 0.01 | 25 | 120 | 0 |
| ankle pitch/roll | 40 | 2.5 | 0.01 | 8 | 27 | 0 |
| shoulders | 40 | 2 | 0.01 | 8 | 27 | 0 |
| elbow pitch | 30 | 1.5 | 0.01 | 8 | 27 | 0 |
| elbow yaw | 20 | 1 | 0.01 | 8 | 27 | 0 |

显式DelayedPD自带力矩限幅；URDF驱动Kp/Kd=0避免双重PD，sim effort/velocity限值已设置，但接触约束仍可能造成瞬时超过目标速度限值。solver position/velocity iterations8/4，self_collision=True，max_depenetration_velocity1m/s，rigid max linear/angular1000。地面静/动摩擦1.0，combine=multiply；contact sensor每物理帧更新/history3；观察噪声关闭，reset event的随机root/q/dq被RSI覆盖。

startup DR保留：Kp/Kd×[.9,1.1]、armature×[.8,1.2]、左右link质量×[.9,1.1]、torso增重[−1,+1]kg；静摩擦[.3,1.6]/动摩擦[.3,1.2]，64 buckets、make_consistent=True；restitution[0,.5]；base/torso COM x/y±.025m、z±.05m。这些为配置范围，逐环境DR实值仅质量和实际执行增益由启动回执记录；不声称所有body参数均采集。

另保留10–15秒间隔的push-by-root-velocity事件（vx/vy±.5、vz±.2m/s，roll/pitch±.52、yaw±.78rad/s）；正式段每环境2.56秒物理跨度不覆盖其首次触发。reset随机q±.15rad/dq0与随机root pose/vel被RSI覆盖。

### 完整RLinf解析配置

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
      "experiment_name": "RPO UFO tracking-contact Kd2.5 524 native updates 3x2048",
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

### 完整原生模型/优化/aux配置

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

### 本体编码/关节顺序/默认q/动作与物理控制协议

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
    "ankle_nominal_damping": 2.5
  }
}
```

### 镜像/官方源revision/补丁hash

```json
{
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
  "auxiliary_profile": "soft_velocity_v7",
  "ground_patch_hash": "sha256:554a18609c9b8a420811fa6b1ded908a4bdf181e52e8d513c8772de6084eff59",
  "optimizer_profile": "stabilized_v7",
  "contact_patch_hash": "sha256:e20a0ff930b3b9430ece70c21ab316a06cef92b679b638c0a35d7fed62d6d814",
  "topology_patch_hash": "sha256:3078506628924464cb42d0b2f108b3fab502a43fd75e091f358d504565294d23",
  "ankle_control_profile": "kd25",
  "auxiliary_source_sha256": "sha256:dc57a21a994cb639d3833b5fa3cb0eefa8630721795227eae2ad0d7aa4e20445",
  "stabilization_patch_hash": "sha256:564f5452e12e3179112e6aa3020efba4cbcedc0ad90dd26b98aed78a50dd85b1",
  "rollout_profile_patch_hash": "sha256:777456ce171734326cf8c55eaa434fc5a69a27d0b2d0b725608dd54c190f4a77",
  "scaling_profile_patch_hash": "sha256:58a2dca8377ed2ede804c7101b39bc6992ec669ef95e610ddec4b2a59bfc6b20",
  "learning_profile_patch_hash": "sha256:d47623e48b4cc757d5e63336a1fa7d49234c3833a2f1c6bbdbee4fe7ffd9a272",
  "tracking_contact_patch_hash": "sha256:a0db175687641801ae1a6b755ad5dae2258a0af16cc6db71da3ebcaa6f43e99d",
  "action_control_source_sha256": "sha256:0cad3bac9e3efb3f1f37d6cac873f789548143b9cd597ed90a93a3045339cd24",
  "auxiliary_profile_patch_hash": "sha256:30e7b4ea7a3ec0e45cdbe0a9dbf718eb219c2c6fd36d07f9b18bd5222183a03a",
  "action_control_profile_patch_hash": "sha256:548260ca5e3eedd53ae739c00672c7e273e7835faed734778039cf5977b56e0d"
}
```

本体codec历史physics_hz=200字段保持发布接口不变；实际新execution control和startup receipt明确400Hz。不得按历史codec字段把本轮物理误报200Hz。Native FBAgent来自UFO commit0b89378de02f77712072d983bc43ec2a46455dc0，RLinf018e6a9e0da60f96eeda8b11008543911027d414；算法对齐为agent/preset，尚未对齐MJLab接触求解、trajectory replay和RPO自定义history/保护项。

## 5. 数据隔离、RSI足点速度与支撑覆盖

Dataset ds_fb0ecda9e70644f2afdd；根/data/kk/data/ds_fb0ecda9e70644f2afdd/content；manifest SHA dc0c6585887507df59f675963b2733d25c9836d805e9bb89f2402398f07bd288；asset根/data/kk/assets/roboparty/rpo-v2，SHA b58868a782d8863ac38059eb4eab89d27ee1d70565bf58f5a055f44510c8e842；codec SHA dce25d554ceb7fa69287515946d2f5cd9d41d14f1b1ef38715b819d7e962b6ef。Dataset原validation status validated/scope rpo_ufo_robot_state_v2但training_ready=false未原地更新；本recipe独立资格证据不冒充全数据已完成正式验收。

普通RSI仅合法train 8帧窗口起点，官方1675/G11767；与250转移=251连续有效状态交集，官方215/G1433。来源权重各.5，source内候选抽样；validation/acceptance未进入筛选、编码、expert或replay。候选原有几何/root高度/倾角/物理q/dq/自碰撞条件继续生效；lift后近地≤20mm的碰撞点abs(dot(v_link+omega×offset,n))≤.15m/s，正常高摆动足不套支撑过滤。不把link原点vz等同碰撞点速度，不清零初始速度。

### 全部3442候选的真实几何与左右支撑覆盖

```json
{
  "status": "passed",
  "actual_RSI_cases_checked": 3442,
  "unique_frames_checked": 3442,
  "normal_velocity_max_mps": 0.1499589254347166,
  "lifted_sole_minimum_m": 0.0019999999999999983,
  "source_counts": {
    "official_rpo": 1675,
    "g1": 1767
  },
  "tracking_source_counts": {
    "official_rpo": 215,
    "g1": 433
  },
  "tracking_cases": 648,
  "finite_difference_cases": 32,
  "finite_difference_max_error_mps": 2.4385145207528325e-06,
  "support_phases_by_source_and_eligibility": {
    "g1/mixed_only": {
      "left_only": 445,
      "right_only": 527,
      "both": 362
    },
    "g1/tracking_eligible": {
      "left_only": 199,
      "right_only": 75,
      "both": 159
    },
    "official_rpo/tracking_eligible": {
      "left_only": 45,
      "right_only": 48,
      "both": 122
    },
    "official_rpo/mixed_only": {
      "left_only": 432,
      "right_only": 569,
      "both": 459
    }
  },
  "clips": {
    "g1_dance1_subject1_clip3": {
      "source": "g1",
      "total": 113,
      "tracking": 0,
      "support_phases": {
        "left_only": 44,
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
      "total": 88,
      "tracking": 0,
      "support_phases": {
        "left_only": 32,
        "right_only": 41,
        "both": 15
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
    "g1_fight1_subject2_clip3": {
      "source": "g1",
      "total": 1,
      "tracking": 0,
      "support_phases": {
        "left_only": 0,
        "right_only": 1,
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
      "total": 31,
      "tracking": 0,
      "support_phases": {
        "left_only": 4,
        "right_only": 17,
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
      "total": 155,
      "tracking": 85,
      "support_phases": {
        "left_only": 49,
        "right_only": 33,
        "both": 73
      },
      "tracking_support_phases": {
        "left_only": 25,
        "right_only": 1,
        "both": 59
      }
    },
    "g1_walk1_subject1_clip3": {
      "source": "g1",
      "total": 160,
      "tracking": 44,
      "support_phases": {
        "left_only": 61,
        "right_only": 75,
        "both": 24
      },
      "tracking_support_phases": {
        "left_only": 25,
        "right_only": 17,
        "both": 2
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
      "total": 176,
      "tracking": 88,
      "support_phases": {
        "left_only": 91,
        "right_only": 73,
        "both": 12
      },
      "tracking_support_phases": {
        "left_only": 51,
        "right_only": 25,
        "both": 12
      }
    },
    "g1_walk2_subject3_clip3": {
      "source": "g1",
      "total": 140,
      "tracking": 62,
      "support_phases": {
        "left_only": 113,
        "right_only": 18,
        "both": 9
      },
      "tracking_support_phases": {
        "left_only": 55,
        "right_only": 0,
        "both": 7
      }
    },
    "g1_walk2_subject4_clip3": {
      "source": "g1",
      "total": 251,
      "tracking": 0,
      "support_phases": {
        "left_only": 83,
        "right_only": 108,
        "both": 60
      }
    },
    "rpo_114_08": {
      "source": "official_rpo",
      "total": 13,
      "tracking": 11,
      "support_phases": {
        "left_only": 12,
        "right_only": 1,
        "both": 0
      },
      "tracking_support_phases": {
        "left_only": 11,
        "right_only": 0,
        "both": 0
      }
    },
    "rpo_114_09": {
      "source": "official_rpo",
      "total": 84,
      "tracking": 25,
      "support_phases": {
        "left_only": 42,
        "right_only": 42,
        "both": 0
      },
      "tracking_support_phases": {
        "left_only": 7,
        "right_only": 18,
        "both": 0
      }
    },
    "rpo_36_01": {
      "source": "official_rpo",
      "total": 46,
      "tracking": 0,
      "support_phases": {
        "left_only": 11,
        "right_only": 34,
        "both": 1
      }
    },
    "rpo_36_11": {
      "source": "official_rpo",
      "total": 103,
      "tracking": 14,
      "support_phases": {
        "left_only": 15,
        "right_only": 67,
        "both": 21
      },
      "tracking_support_phases": {
        "left_only": 1,
        "right_only": 13,
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
      "total": 10,
      "tracking": 0,
      "support_phases": {
        "left_only": 8,
        "right_only": 2,
        "both": 0
      }
    },
    "rpo_B13_-__Walk_turn_right_90_stageii": {
      "source": "official_rpo",
      "total": 159,
      "tracking": 0,
      "support_phases": {
        "left_only": 73,
        "right_only": 69,
        "both": 17
      }
    },
    "rpo_B14_-__Walk_turn_right_45_t2_stageii": {
      "source": "official_rpo",
      "total": 104,
      "tracking": 0,
      "support_phases": {
        "left_only": 31,
        "right_only": 34,
        "both": 39
      }
    },
    "rpo_B15_-__Walk_turn_around_stageii": {
      "source": "official_rpo",
      "total": 5,
      "tracking": 0,
      "support_phases": {
        "left_only": 3,
        "right_only": 2,
        "both": 0
      }
    },
    "rpo_B5_-__Walk_backwards_stageii": {
      "source": "official_rpo",
      "total": 33,
      "tracking": 0,
      "support_phases": {
        "left_only": 10,
        "right_only": 13,
        "both": 10
      }
    },
    "rpo_B9_-__Walk_turn_left_90_stageii": {
      "source": "official_rpo",
      "total": 5,
      "tracking": 0,
      "support_phases": {
        "left_only": 5,
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
      "total": 100,
      "tracking": 0,
      "support_phases": {
        "left_only": 2,
        "right_only": 0,
        "both": 98
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
      "total": 14,
      "tracking": 0,
      "support_phases": {
        "left_only": 6,
        "right_only": 8,
        "both": 0
      }
    },
    "rpo_C5_-_walk_to_run_stageii": {
      "source": "official_rpo",
      "total": 9,
      "tracking": 0,
      "support_phases": {
        "left_only": 5,
        "right_only": 4,
        "both": 0
      }
    },
    "rpo_move_back": {
      "source": "official_rpo",
      "total": 271,
      "tracking": 62,
      "support_phases": {
        "left_only": 106,
        "right_only": 87,
        "both": 78
      },
      "tracking_support_phases": {
        "left_only": 3,
        "right_only": 0,
        "both": 59
      }
    },
    "rpo_move_l": {
      "source": "official_rpo",
      "total": 97,
      "tracking": 30,
      "support_phases": {
        "left_only": 8,
        "right_only": 51,
        "both": 38
      },
      "tracking_support_phases": {
        "left_only": 0,
        "right_only": 0,
        "both": 30
      }
    },
    "rpo_move_r": {
      "source": "official_rpo",
      "total": 113,
      "tracking": 0,
      "support_phases": {
        "left_only": 28,
        "right_only": 71,
        "both": 14
      }
    },
    "rpo_run_start_180_R_001__A345": {
      "source": "official_rpo",
      "total": 3,
      "tracking": 0,
      "support_phases": {
        "left_only": 1,
        "right_only": 2,
        "both": 0
      }
    },
    "rpo_run_start_180_R_001__A345_M": {
      "source": "official_rpo",
      "total": 4,
      "tracking": 0,
      "support_phases": {
        "left_only": 1,
        "right_only": 3,
        "both": 0
      }
    },
    "rpo_run_stop_180_R_001__A345": {
      "source": "official_rpo",
      "total": 110,
      "tracking": 0,
      "support_phases": {
        "left_only": 0,
        "right_only": 3,
        "both": 107
      }
    },
    "rpo_run_stop_180_R_001__A345_M": {
      "source": "official_rpo",
      "total": 110,
      "tracking": 0,
      "support_phases": {
        "left_only": 3,
        "right_only": 0,
        "both": 107
      }
    },
    "rpo_turn_l": {
      "source": "official_rpo",
      "total": 127,
      "tracking": 73,
      "support_phases": {
        "left_only": 43,
        "right_only": 50,
        "both": 34
      },
      "tracking_support_phases": {
        "left_only": 23,
        "right_only": 17,
        "both": 33
      }
    },
    "rpo_turn_r": {
      "source": "official_rpo",
      "total": 147,
      "tracking": 0,
      "support_phases": {
        "left_only": 57,
        "right_only": 73,
        "both": 17
      }
    }
  },
  "reference_frame_support_phases": {
    "0:59": "right_only",
    "0:60": "right_only",
    "0:78": "left_only",
    "0:79": "left_only",
    "0:80": "left_only",
    "0:81": "left_only",
    "0:82": "left_only",
    "0:83": "left_only",
    "0:84": "left_only",
    "0:85": "left_only",
    "0:86": "left_only",
    "0:97": "right_only",
    "0:98": "right_only",
    "0:99": "right_only",
    "0:100": "right_only",
    "0:101": "right_only",
    "0:102": "right_only",
    "0:103": "right_only",
    "0:104": "right_only",
    "0:105": "right_only",
    "0:106": "right_only",
    "0:107": "right_only",
    "0:108": "right_only",
    "0:109": "right_only",
    "0:110": "right_only",
    "0:111": "right_only",
    "0:115": "both",
    "0:116": "both",
    "0:117": "both",
    "0:118": "both",
    "0:119": "both",
    "0:120": "both",
    "0:122": "left_only",
    "0:123": "left_only",
    "0:124": "left_only",
    "0:125": "left_only",
    "0:126": "left_only",
    "0:127": "left_only",
    "0:128": "left_only",
    "0:129": "left_only",
    "0:130": "left_only",
    "0:131": "left_only",
    "0:132": "both",
    "0:133": "both",
    "0:134": "both",
    "0:135": "both",
    "0:136": "both",
    "0:137": "both",
    "0:138": "both",
    "0:139": "both",
    "0:140": "both",
    "0:141": "both",
    "0:142": "both",
    "0:143": "both",
    "0:144": "both",
    "0:145": "both",
    "0:146": "both",
    "0:147": "both",
    "0:148": "both",
    "0:149": "both",
    "0:150": "both",
    "0:151": "both",
    "0:152": "both",
    "0:153": "both",
    "0:154": "both",
    "0:155": "both",
    "0:156": "both",
    "0:189": "both",
    "0:190": "right_only",
    "0:191": "right_only",
    "0:192": "right_only",
    "0:196": "right_only",
    "0:197": "right_only",
    "0:198": "right_only",
    "0:199": "right_only",
    "0:200": "right_only",
    "0:201": "right_only",
    "0:335": "right_only",
    "0:336": "right_only",
    "0:337": "right_only",
    "0:343": "left_only",
    "0:344": "left_only",
    "0:345": "left_only",
    "0:346": "left_only",
    "0:347": "left_only",
    "0:348": "left_only",
    "0:349": "left_only",
    "0:350": "left_only",
    "0:351": "left_only",
    "0:352": "left_only",
    "0:353": "left_only",
    "0:354": "left_only",
    "0:355": "left_only",
    "0:356": "left_only",
    "0:357": "left_only",
    "0:361": "left_only",
    "0:362": "left_only",
    "0:363": "left_only",
    "0:364": "left_only",
    "0:365": "left_only",
    "0:366": "left_only",
    "0:367": "left_only",
    "0:368": "left_only",
    "0:369": "left_only",
    "0:370": "left_only",
    "0:375": "right_only",
    "0:376": "right_only",
    "0:377": "right_only",
    "0:378": "right_only",
    "0:379": "right_only",
    "0:384": "right_only",
    "0:385": "right_only",
    "0:386": "right_only",
    "1:14": "right_only",
    "2:237": "right_only",
    "2:238": "right_only",
    "2:239": "right_only",
    "2:240": "right_only",
    "2:241": "right_only",
    "2:242": "right_only",
    "2:243": "right_only",
    "2:244": "right_only",
    "2:245": "right_only",
    "2:246": "right_only",
    "2:247": "right_only",
    "3:3": "both",
    "3:4": "both",
    "3:5": "both",
    "3:6": "both",
    "3:7": "both",
    "3:8": "both",
    "3:9": "both",
    "3:14": "both",
    "3:15": "both",
    "3:16": "both",
    "3:20": "right_only",
    "3:21": "right_only",
    "3:22": "right_only",
    "3:23": "right_only",
    "3:24": "right_only",
    "3:25": "right_only",
    "3:26": "right_only",
    "3:27": "right_only",
    "3:28": "right_only",
    "3:35": "right_only",
    "3:45": "left_only",
    "3:46": "left_only",
    "3:47": "left_only",
    "3:48": "left_only",
    "3:60": "left_only",
    "3:61": "left_only",
    "3:62": "left_only",
    "3:63": "left_only",
    "3:64": "left_only",
    "3:65": "left_only",
    "3:66": "both",
    "3:67": "both",
    "3:76": "right_only",
    "3:77": "right_only",
    "3:78": "right_only",
    "3:79": "right_only",
    "3:80": "right_only",
    "3:81": "right_only",
    "3:82": "right_only",
    "3:88": "right_only",
    "3:89": "right_only",
    "3:90": "right_only",
    "3:91": "right_only",
    "3:100": "left_only",
    "3:101": "left_only",
    "3:102": "left_only",
    "3:114": "left_only",
    "3:319": "left_only",
    "3:320": "left_only",
    "3:321": "left_only",
    "3:322": "left_only",
    "3:326": "both",
    "3:328": "right_only",
    "3:329": "right_only",
    "3:330": "right_only",
    "3:331": "right_only",
    "3:332": "right_only",
    "3:333": "right_only",
    "3:334": "right_only",
    "3:335": "right_only",
    "3:336": "right_only",
    "3:337": "right_only",
    "3:338": "right_only",
    "3:339": "right_only",
    "3:340": "right_only",
    "3:341": "right_only",
    "3:342": "right_only",
    "3:363": "both",
    "3:364": "both",
    "3:366": "left_only",
    "3:367": "left_only",
    "3:368": "left_only",
    "3:369": "left_only",
    "3:370": "left_only",
    "3:373": "left_only",
    "3:374": "left_only",
    "3:375": "left_only",
    "3:376": "left_only",
    "3:377": "left_only",
    "3:378": "left_only",
    "3:379": "left_only",
    "3:380": "left_only",
    "3:381": "left_only",
    "3:387": "right_only",
    "3:388": "right_only",
    "3:389": "right_only",
    "3:390": "right_only",
    "3:391": "right_only",
    "5:41": "left_only",
    "5:276": "left_only",
    "5:277": "left_only",
    "5:291": "right_only",
    "5:292": "right_only",
    "6:1": "right_only",
    "8:7": "right_only",
    "8:14": "right_only",
    "8:32": "right_only",
    "8:33": "right_only",
    "8:34": "right_only",
    "8:35": "right_only",
    "8:36": "right_only",
    "8:63": "right_only",
    "9:26": "both",
    "9:174": "left_only",
    "9:258": "left_only",
    "9:259": "left_only",
    "9:260": "left_only",
    "9:274": "right_only",
    "9:275": "right_only",
    "9:276": "right_only",
    "9:277": "right_only",
    "9:278": "right_only",
    "9:279": "right_only",
    "9:280": "right_only",
    "9:281": "right_only",
    "9:282": "right_only",
    "9:284": "right_only",
    "9:286": "right_only",
    "9:288": "right_only",
    "9:290": "right_only",
    "9:292": "right_only",
    "9:294": "right_only",
    "9:296": "right_only",
    "9:298": "right_only",
    "9:311": "both",
    "9:312": "both",
    "9:313": "both",
    "9:314": "both",
    "9:315": "both",
    "9:316": "both",
    "9:317": "both",
    "9:318": "both",
    "9:319": "both",
    "10:12": "right_only",
    "10:13": "right_only",
    "10:14": "right_only",
    "11:5": "left_only",
    "11:6": "left_only",
    "11:7": "left_only",
    "11:8": "left_only",
    "11:9": "left_only",
    "12:8": "right_only",
    "12:9": "right_only",
    "12:10": "right_only",
    "12:94": "right_only",
    "12:95": "right_only",
    "12:96": "right_only",
    "12:98": "right_only",
    "12:99": "right_only",
    "12:100": "right_only",
    "12:113": "left_only",
    "12:114": "left_only",
    "12:115": "left_only",
    "12:259": "right_only",
    "12:260": "right_only",
    "12:314": "left_only",
    "14:151": "right_only",
    "14:152": "right_only",
    "14:193": "right_only",
    "14:194": "right_only",
    "14:195": "right_only",
    "14:196": "right_only",
    "14:197": "right_only",
    "14:198": "right_only",
    "14:199": "right_only",
    "14:200": "right_only",
    "14:242": "right_only",
    "14:243": "right_only",
    "14:244": "right_only",
    "14:245": "right_only",
    "14:246": "right_only",
    "14:247": "right_only",
    "14:248": "right_only",
    "14:249": "right_only",
    "14:250": "right_only",
    "14:251": "right_only",
    "14:252": "right_only",
    "14:253": "right_only",
    "14:254": "right_only",
    "14:255": "right_only",
    "14:256": "right_only",
    "14:257": "both",
    "14:258": "both",
    "14:262": "left_only",
    "14:263": "left_only",
    "14:264": "left_only",
    "14:265": "left_only",
    "14:266": "left_only",
    "14:267": "left_only",
    "14:268": "left_only",
    "14:269": "left_only",
    "14:270": "left_only",
    "14:271": "left_only",
    "14:272": "left_only",
    "14:273": "left_only",
    "14:274": "left_only",
    "14:275": "left_only",
    "14:276": "left_only",
    "14:277": "left_only",
    "14:278": "left_only",
    "14:279": "left_only",
    "14:280": "left_only",
    "14:281": "left_only",
    "14:282": "both",
    "14:283": "both",
    "14:284": "both",
    "14:285": "both",
    "14:286": "right_only",
    "14:287": "right_only",
    "14:288": "right_only",
    "14:289": "right_only",
    "14:290": "right_only",
    "14:291": "right_only",
    "14:292": "right_only",
    "14:293": "right_only",
    "14:294": "right_only",
    "14:295": "right_only",
    "14:296": "right_only",
    "14:297": "right_only",
    "14:298": "right_only",
    "14:299": "right_only",
    "14:300": "right_only",
    "14:301": "right_only",
    "14:302": "right_only",
    "14:303": "right_only",
    "14:304": "right_only",
    "14:305": "right_only",
    "14:306": "right_only",
    "14:307": "both",
    "14:308": "both",
    "14:309": "both",
    "14:310": "both",
    "14:311": "both",
    "14:312": "both",
    "14:313": "both",
    "14:314": "both",
    "14:315": "both",
    "14:316": "both",
    "14:317": "both",
    "14:318": "both",
    "14:319": "both",
    "14:320": "both",
    "14:321": "both",
    "14:322": "both",
    "14:323": "both",
    "14:324": "both",
    "14:325": "both",
    "14:326": "both",
    "14:327": "both",
    "14:328": "both",
    "14:329": "both",
    "14:330": "both",
    "14:331": "both",
    "14:332": "both",
    "14:333": "both",
    "14:334": "both",
    "14:335": "both",
    "14:336": "both",
    "14:337": "both",
    "14:338": "both",
    "14:339": "both",
    "14:340": "both",
    "14:341": "both",
    "14:342": "both",
    "14:343": "both",
    "14:344": "both",
    "14:345": "both",
    "14:346": "both",
    "14:347": "both",
    "14:348": "both",
    "14:349": "both",
    "14:350": "both",
    "14:351": "both",
    "14:353": "left_only",
    "14:354": "left_only",
    "14:355": "left_only",
    "14:356": "left_only",
    "14:357": "left_only",
    "14:358": "left_only",
    "14:359": "left_only",
    "14:360": "left_only",
    "14:361": "left_only",
    "14:362": "both",
    "14:363": "both",
    "14:364": "both",
    "14:365": "both",
    "14:366": "both",
    "14:367": "both",
    "14:368": "both",
    "14:369": "both",
    "14:370": "both",
    "14:371": "both",
    "14:372": "right_only",
    "14:373": "right_only",
    "14:374": "right_only",
    "14:375": "right_only",
    "14:376": "right_only",
    "14:377": "right_only",
    "14:378": "right_only",
    "14:379": "right_only",
    "14:380": "right_only",
    "14:381": "right_only",
    "14:382": "right_only",
    "14:383": "right_only",
    "14:384": "right_only",
    "14:385": "right_only",
    "14:386": "right_only",
    "14:387": "right_only",
    "14:388": "right_only",
    "14:389": "right_only",
    "14:390": "right_only",
    "14:391": "right_only",
    "15:0": "right_only",
    "15:6": "left_only",
    "15:7": "left_only",
    "15:12": "both",
    "15:13": "both",
    "15:14": "both",
    "15:15": "both",
    "15:16": "both",
    "15:17": "both",
    "15:18": "right_only",
    "15:19": "right_only",
    "15:20": "right_only",
    "15:21": "right_only",
    "15:22": "right_only",
    "15:23": "right_only",
    "15:24": "right_only",
    "15:25": "right_only",
    "15:26": "right_only",
    "15:27": "right_only",
    "15:28": "right_only",
    "15:29": "right_only",
    "15:30": "right_only",
    "15:31": "right_only",
    "15:32": "right_only",
    "15:33": "right_only",
    "15:34": "right_only",
    "15:35": "right_only",
    "15:36": "right_only",
    "15:37": "right_only",
    "15:38": "right_only",
    "15:39": "right_only",
    "15:40": "right_only",
    "15:41": "right_only",
    "15:61": "left_only",
    "15:62": "left_only",
    "15:63": "left_only",
    "15:64": "left_only",
    "15:65": "left_only",
    "15:66": "left_only",
    "15:67": "left_only",
    "15:68": "left_only",
    "15:69": "left_only",
    "15:70": "left_only",
    "15:71": "left_only",
    "15:72": "left_only",
    "15:73": "left_only",
    "15:74": "left_only",
    "15:75": "left_only",
    "15:76": "left_only",
    "15:77": "left_only",
    "15:78": "left_only",
    "15:79": "left_only",
    "15:80": "left_only",
    "15:81": "left_only",
    "15:82": "left_only",
    "15:83": "left_only",
    "15:88": "both",
    "15:89": "both",
    "15:90": "both",
    "15:91": "both",
    "15:92": "both",
    "15:93": "both",
    "15:94": "both",
    "15:95": "both",
    "15:96": "both",
    "15:97": "both",
    "15:98": "both",
    "15:99": "right_only",
    "15:100": "both",
    "15:101": "both",
    "15:102": "both",
    "15:103": "both",
    "15:104": "both",
    "15:105": "both",
    "15:106": "both",
    "15:107": "both",
    "15:108": "both",
    "15:109": "both",
    "15:110": "both",
    "15:111": "both",
    "15:112": "both",
    "15:113": "both",
    "15:114": "both",
    "15:115": "both",
    "15:116": "both",
    "15:117": "both",
    "15:118": "both",
    "15:119": "both",
    "15:120": "both",
    "15:121": "both",
    "15:122": "both",
    "15:123": "both",
    "15:124": "both",
    "15:125": "both",
    "15:126": "both",
    "15:127": "both",
    "15:128": "both",
    "15:129": "both",
    "15:130": "both",
    "15:131": "both",
    "15:132": "both",
    "15:133": "both",
    "15:134": "both",
    "15:135": "both",
    "15:136": "both",
    "15:137": "both",
    "15:138": "both",
    "15:139": "both",
    "15:140": "both",
    "15:141": "both",
    "15:142": "both",
    "15:143": "both",
    "15:144": "both",
    "15:145": "both",
    "15:146": "both",
    "15:147": "both",
    "15:148": "left_only",
    "15:149": "left_only",
    "15:150": "left_only",
    "15:151": "left_only",
    "15:152": "left_only",
    "15:153": "left_only",
    "15:154": "left_only",
    "15:155": "left_only",
    "15:156": "left_only",
    "15:157": "left_only",
    "15:158": "left_only",
    "15:159": "left_only",
    "15:160": "left_only",
    "15:161": "left_only",
    "15:162": "left_only",
    "15:163": "left_only",
    "15:164": "left_only",
    "15:165": "left_only",
    "15:166": "left_only",
    "15:167": "left_only",
    "15:168": "left_only",
    "15:176": "left_only",
    "15:177": "left_only",
    "15:178": "left_only",
    "15:179": "both",
    "15:180": "both",
    "15:181": "both",
    "15:182": "both",
    "15:183": "both",
    "15:184": "both",
    "15:185": "both",
    "15:186": "both",
    "15:188": "right_only",
    "15:189": "right_only",
    "15:190": "right_only",
    "15:191": "right_only",
    "15:192": "right_only",
    "15:193": "right_only",
    "15:197": "right_only",
    "16:3": "both",
    "16:4": "both",
    "16:7": "left_only",
    "16:22": "left_only",
    "16:23": "left_only",
    "16:25": "left_only",
    "16:26": "left_only",
    "16:27": "left_only",
    "16:28": "left_only",
    "16:29": "left_only",
    "16:30": "left_only",
    "16:36": "left_only",
    "16:39": "left_only",
    "16:57": "right_only",
    "16:60": "right_only",
    "16:61": "right_only",
    "16:62": "right_only",
    "16:63": "right_only",
    "16:64": "right_only",
    "16:65": "right_only",
    "16:92": "left_only",
    "16:93": "left_only",
    "16:95": "left_only",
    "16:96": "left_only",
    "16:97": "left_only",
    "16:98": "left_only",
    "16:99": "left_only",
    "16:100": "left_only",
    "16:101": "left_only",
    "16:102": "left_only",
    "16:103": "left_only",
    "16:104": "left_only",
    "16:105": "left_only",
    "16:108": "left_only",
    "16:127": "right_only",
    "16:128": "right_only",
    "16:129": "right_only",
    "16:131": "right_only",
    "16:132": "right_only",
    "16:133": "right_only",
    "16:134": "right_only",
    "16:135": "right_only",
    "16:136": "right_only",
    "16:137": "right_only",
    "16:170": "left_only",
    "16:171": "left_only",
    "16:172": "left_only",
    "16:173": "left_only",
    "16:174": "left_only",
    "16:175": "left_only",
    "16:176": "left_only",
    "16:177": "left_only",
    "16:178": "left_only",
    "16:179": "left_only",
    "16:180": "left_only",
    "16:181": "left_only",
    "16:182": "left_only",
    "16:183": "left_only",
    "16:184": "left_only",
    "16:185": "both",
    "16:186": "both",
    "16:187": "both",
    "16:188": "both",
    "16:189": "both",
    "16:190": "both",
    "16:191": "both",
    "16:192": "both",
    "16:193": "both",
    "16:194": "both",
    "16:195": "both",
    "16:196": "both",
    "16:197": "both",
    "16:198": "both",
    "16:199": "both",
    "16:200": "both",
    "16:201": "both",
    "16:202": "both",
    "16:203": "both",
    "16:204": "both",
    "16:205": "both",
    "16:206": "right_only",
    "16:207": "both",
    "16:208": "right_only",
    "16:209": "right_only",
    "16:210": "right_only",
    "16:211": "right_only",
    "16:212": "right_only",
    "16:213": "right_only",
    "16:214": "right_only",
    "16:215": "right_only",
    "16:216": "right_only",
    "16:217": "right_only",
    "16:218": "right_only",
    "16:219": "right_only",
    "16:220": "right_only",
    "16:221": "right_only",
    "16:222": "right_only",
    "16:223": "right_only",
    "16:224": "right_only",
    "16:225": "right_only",
    "16:226": "right_only",
    "16:227": "right_only",
    "16:228": "right_only",
    "16:229": "right_only",
    "16:230": "right_only",
    "16:231": "right_only",
    "16:232": "right_only",
    "16:233": "right_only",
    "16:234": "right_only",
    "16:235": "right_only",
    "16:236": "right_only",
    "16:237": "right_only",
    "16:238": "right_only",
    "16:239": "right_only",
    "16:240": "right_only",
    "16:241": "right_only",
    "16:242": "right_only",
    "16:243": "right_only",
    "16:244": "right_only",
    "16:245": "right_only",
    "16:246": "right_only",
    "16:247": "right_only",
    "16:248": "right_only",
    "16:249": "right_only",
    "16:250": "right_only",
    "16:251": "right_only",
    "16:252": "right_only",
    "16:253": "right_only",
    "16:254": "right_only",
    "16:260": "left_only",
    "16:276": "left_only",
    "16:279": "left_only",
    "16:280": "left_only",
    "16:281": "left_only",
    "16:282": "left_only",
    "16:314": "right_only",
    "16:315": "right_only",
    "16:316": "right_only",
    "16:317": "right_only",
    "16:318": "right_only",
    "16:319": "right_only",
    "16:320": "right_only",
    "16:321": "right_only",
    "16:345": "left_only",
    "16:346": "left_only",
    "16:349": "left_only",
    "16:350": "left_only",
    "16:351": "left_only",
    "16:352": "left_only",
    "16:353": "left_only",
    "16:354": "left_only",
    "16:355": "left_only",
    "16:356": "left_only",
    "16:357": "left_only",
    "16:358": "left_only",
    "16:359": "left_only",
    "16:360": "left_only",
    "16:361": "left_only",
    "16:390": "right_only",
    "16:391": "right_only",
    "17:3": "right_only",
    "17:7": "both",
    "17:8": "both",
    "17:12": "right_only",
    "17:13": "right_only",
    "17:14": "right_only",
    "17:15": "right_only",
    "17:16": "right_only",
    "17:17": "right_only",
    "17:18": "right_only",
    "17:19": "right_only",
    "17:20": "right_only",
    "17:22": "left_only",
    "17:23": "left_only",
    "17:24": "left_only",
    "17:25": "left_only",
    "17:26": "left_only",
    "17:27": "left_only",
    "17:28": "left_only",
    "17:29": "left_only",
    "17:30": "left_only",
    "17:31": "left_only",
    "17:32": "left_only",
    "17:33": "left_only",
    "17:34": "left_only",
    "17:35": "left_only",
    "17:36": "left_only",
    "17:37": "left_only",
    "17:38": "left_only",
    "17:39": "left_only",
    "17:40": "left_only",
    "17:41": "left_only",
    "17:42": "left_only",
    "17:43": "left_only",
    "17:44": "both",
    "17:45": "both",
    "17:46": "both",
    "17:47": "both",
    "17:48": "both",
    "17:49": "both",
    "17:50": "both",
    "17:51": "both",
    "17:52": "both",
    "17:53": "both",
    "17:54": "both",
    "17:55": "both",
    "17:56": "both",
    "17:57": "both",
    "17:58": "both",
    "17:59": "both",
    "17:60": "both",
    "17:61": "both",
    "17:62": "both",
    "17:63": "both",
    "17:64": "both",
    "17:65": "both",
    "17:66": "both",
    "17:67": "both",
    "17:68": "both",
    "17:69": "both",
    "17:70": "both",
    "17:71": "both",
    "17:72": "both",
    "17:73": "both",
    "17:74": "both",
    "17:75": "both",
    "17:76": "both",
    "17:77": "both",
    "17:78": "both",
    "17:79": "both",
    "17:80": "both",
    "17:81": "both",
    "17:82": "both",
    "17:83": "both",
    "17:84": "both",
    "17:85": "both",
    "17:86": "both",
    "17:87": "both",
    "17:88": "both",
    "17:89": "both",
    "17:90": "both",
    "17:91": "both",
    "17:92": "both",
    "17:93": "both",
    "17:94": "both",
    "17:95": "both",
    "17:96": "both",
    "17:97": "both",
    "17:98": "both",
    "17:99": "both",
    "17:100": "both",
    "17:101": "both",
    "17:102": "both",
    "17:103": "both",
    "17:104": "both",
    "17:105": "both",
    "17:106": "both",
    "17:107": "both",
    "17:108": "both",
    "17:109": "both",
    "17:110": "both",
    "17:111": "both",
    "17:112": "both",
    "17:113": "both",
    "17:114": "both",
    "17:115": "both",
    "17:116": "both",
    "17:117": "both",
    "17:118": "both",
    "17:119": "both",
    "17:120": "both",
    "17:125": "left_only",
    "17:126": "left_only",
    "17:127": "left_only",
    "17:128": "left_only",
    "17:129": "left_only",
    "17:130": "left_only",
    "17:131": "left_only",
    "17:132": "left_only",
    "17:133": "left_only",
    "17:134": "left_only",
    "17:138": "right_only",
    "17:139": "right_only",
    "17:140": "right_only",
    "17:144": "right_only",
    "17:145": "right_only",
    "17:146": "right_only",
    "17:147": "right_only",
    "17:148": "right_only",
    "17:149": "right_only",
    "17:150": "right_only",
    "17:151": "right_only",
    "17:152": "right_only",
    "17:153": "right_only",
    "17:154": "right_only",
    "17:155": "right_only",
    "17:156": "right_only",
    "17:157": "right_only",
    "17:158": "right_only",
    "17:159": "right_only",
    "17:160": "right_only",
    "17:161": "right_only",
    "17:162": "right_only",
    "17:163": "right_only",
    "17:164": "both",
    "17:165": "both",
    "17:166": "both",
    "17:167": "both",
    "17:168": "both",
    "17:169": "both",
    "17:170": "both",
    "17:171": "both",
    "17:172": "both",
    "17:173": "both",
    "17:174": "both",
    "17:175": "both",
    "17:176": "left_only",
    "17:177": "left_only",
    "17:178": "left_only",
    "17:179": "left_only",
    "17:180": "left_only",
    "17:181": "left_only",
    "17:182": "left_only",
    "17:183": "left_only",
    "17:184": "left_only",
    "17:185": "both",
    "17:186": "both",
    "17:187": "both",
    "17:188": "both",
    "17:189": "both",
    "17:190": "both",
    "17:191": "both",
    "17:192": "both",
    "17:193": "both",
    "17:194": "both",
    "17:195": "both",
    "17:196": "both",
    "17:197": "both",
    "17:198": "both",
    "17:199": "both",
    "17:200": "both",
    "17:201": "both",
    "17:202": "both",
    "17:203": "both",
    "17:204": "both",
    "17:205": "both",
    "17:206": "both",
    "17:207": "both",
    "17:208": "both",
    "17:209": "both",
    "17:210": "both",
    "17:211": "both",
    "17:212": "both",
    "17:213": "both",
    "17:214": "both",
    "17:215": "both",
    "17:216": "both",
    "17:217": "both",
    "17:218": "both",
    "17:219": "both",
    "17:220": "both",
    "17:221": "both",
    "17:222": "both",
    "17:223": "both",
    "17:224": "both",
    "17:225": "both",
    "17:226": "both",
    "17:227": "both",
    "17:228": "both",
    "17:229": "both",
    "17:230": "both",
    "17:231": "both",
    "17:232": "both",
    "17:235": "left_only",
    "17:236": "left_only",
    "17:237": "left_only",
    "17:238": "left_only",
    "17:239": "left_only",
    "17:240": "left_only",
    "17:241": "left_only",
    "17:242": "left_only",
    "17:243": "left_only",
    "17:244": "left_only",
    "17:245": "left_only",
    "17:246": "left_only",
    "17:247": "left_only",
    "17:248": "left_only",
    "17:249": "left_only",
    "17:252": "right_only",
    "17:253": "right_only",
    "17:254": "right_only",
    "17:255": "right_only",
    "17:256": "right_only",
    "17:257": "right_only",
    "17:258": "right_only",
    "17:259": "right_only",
    "17:260": "right_only",
    "17:261": "right_only",
    "17:262": "right_only",
    "17:263": "right_only",
    "17:264": "both",
    "17:265": "both",
    "17:266": "both",
    "17:267": "left_only",
    "17:268": "left_only",
    "17:269": "both",
    "17:270": "both",
    "17:271": "both",
    "17:272": "right_only",
    "17:273": "right_only",
    "17:274": "right_only",
    "17:275": "right_only",
    "17:276": "right_only",
    "17:277": "right_only",
    "17:278": "right_only",
    "17:279": "right_only",
    "17:281": "both",
    "17:282": "both",
    "17:283": "both",
    "17:284": "both",
    "17:285": "both",
    "17:286": "both",
    "17:287": "both",
    "17:288": "both",
    "17:289": "left_only",
    "17:290": "left_only",
    "17:291": "left_only",
    "17:292": "left_only",
    "17:293": "left_only",
    "17:294": "left_only",
    "17:295": "left_only",
    "17:296": "left_only",
    "17:297": "left_only",
    "17:298": "left_only",
    "17:299": "left_only",
    "17:300": "left_only",
    "17:301": "left_only",
    "17:302": "left_only",
    "17:303": "left_only",
    "17:304": "left_only",
    "17:305": "left_only",
    "17:306": "left_only",
    "17:308": "both",
    "17:309": "both",
    "17:310": "both",
    "17:311": "both",
    "17:312": "both",
    "17:313": "both",
    "17:314": "both",
    "17:315": "both",
    "17:316": "both",
    "17:317": "both",
    "17:318": "both",
    "17:319": "both",
    "17:320": "both",
    "17:321": "both",
    "17:322": "both",
    "17:323": "both",
    "17:324": "both",
    "17:325": "both",
    "17:326": "both",
    "17:327": "both",
    "17:328": "both",
    "17:329": "both",
    "17:330": "both",
    "17:331": "both",
    "17:332": "both",
    "17:333": "both",
    "17:334": "both",
    "17:335": "both",
    "17:336": "both",
    "17:337": "both",
    "17:338": "both",
    "17:339": "right_only",
    "17:340": "both",
    "17:341": "right_only",
    "17:342": "both",
    "17:343": "both",
    "17:344": "both",
    "17:345": "both",
    "17:346": "both",
    "17:347": "both",
    "17:348": "both",
    "17:349": "both",
    "17:350": "both",
    "17:351": "both",
    "17:352": "both",
    "17:353": "both",
    "17:354": "both",
    "17:355": "both",
    "17:356": "both",
    "17:357": "both",
    "17:358": "both",
    "17:359": "both",
    "17:360": "both",
    "17:361": "both",
    "17:362": "both",
    "17:363": "both",
    "17:364": "both",
    "17:365": "both",
    "17:366": "both",
    "17:367": "both",
    "17:368": "both",
    "17:369": "both",
    "17:370": "both",
    "17:371": "both",
    "17:372": "both",
    "17:373": "both",
    "17:374": "both",
    "17:375": "both",
    "17:376": "both",
    "17:377": "both",
    "17:378": "both",
    "17:379": "both",
    "17:380": "both",
    "17:389": "left_only",
    "17:390": "left_only",
    "17:391": "left_only",
    "18:32": "left_only",
    "18:33": "left_only",
    "18:34": "left_only",
    "18:57": "right_only",
    "18:58": "right_only",
    "18:59": "right_only",
    "18:60": "right_only",
    "18:72": "left_only",
    "18:73": "left_only",
    "18:74": "left_only",
    "18:75": "left_only",
    "18:82": "left_only",
    "18:83": "left_only",
    "18:84": "left_only",
    "18:85": "left_only",
    "18:97": "right_only",
    "18:98": "right_only",
    "18:99": "right_only",
    "18:100": "right_only",
    "18:101": "right_only",
    "18:102": "right_only",
    "18:103": "right_only",
    "18:144": "right_only",
    "18:145": "right_only",
    "18:172": "left_only",
    "18:173": "left_only",
    "18:174": "left_only",
    "18:178": "right_only",
    "18:179": "right_only",
    "18:185": "right_only",
    "18:186": "right_only",
    "18:187": "right_only",
    "18:198": "right_only",
    "18:199": "right_only",
    "18:201": "right_only",
    "18:214": "left_only",
    "18:215": "left_only",
    "18:223": "left_only",
    "18:224": "left_only",
    "18:225": "left_only",
    "18:226": "left_only",
    "18:236": "right_only",
    "18:237": "right_only",
    "18:238": "right_only",
    "18:282": "left_only",
    "18:303": "right_only",
    "18:304": "right_only",
    "18:305": "right_only",
    "18:306": "right_only",
    "18:307": "right_only",
    "18:318": "left_only",
    "18:319": "left_only",
    "18:320": "left_only",
    "18:321": "left_only",
    "18:330": "left_only",
    "18:331": "left_only",
    "18:332": "left_only",
    "18:333": "left_only",
    "18:334": "left_only",
    "18:335": "both",
    "18:337": "right_only",
    "18:338": "right_only",
    "18:339": "right_only",
    "18:340": "right_only",
    "18:342": "right_only",
    "18:343": "right_only",
    "18:344": "right_only",
    "18:345": "right_only",
    "18:346": "right_only",
    "18:353": "right_only",
    "18:354": "right_only",
    "18:355": "right_only",
    "18:356": "right_only",
    "18:361": "left_only",
    "18:362": "left_only",
    "18:363": "left_only",
    "18:364": "left_only",
    "18:367": "left_only",
    "18:368": "left_only",
    "18:369": "left_only",
    "18:370": "left_only",
    "18:371": "left_only",
    "18:372": "left_only",
    "18:373": "left_only",
    "18:374": "left_only",
    "18:375": "left_only",
    "18:376": "left_only",
    "18:377": "left_only",
    "18:378": "left_only",
    "18:379": "left_only",
    "19:3": "both",
    "19:4": "both",
    "19:6": "both",
    "19:7": "both",
    "19:8": "both",
    "19:9": "left_only",
    "19:10": "left_only",
    "19:11": "left_only",
    "19:12": "left_only",
    "19:13": "left_only",
    "19:14": "left_only",
    "19:15": "left_only",
    "19:16": "left_only",
    "19:17": "left_only",
    "19:18": "left_only",
    "19:19": "left_only",
    "19:20": "left_only",
    "19:21": "left_only",
    "19:22": "left_only",
    "19:23": "left_only",
    "19:24": "left_only",
    "19:25": "left_only",
    "19:26": "left_only",
    "19:27": "left_only",
    "19:28": "left_only",
    "19:29": "left_only",
    "19:30": "left_only",
    "19:31": "left_only",
    "19:32": "left_only",
    "19:33": "left_only",
    "19:34": "both",
    "19:35": "left_only",
    "19:36": "both",
    "19:37": "both",
    "19:38": "both",
    "19:39": "left_only",
    "19:40": "left_only",
    "19:41": "left_only",
    "19:42": "both",
    "19:43": "left_only",
    "19:44": "left_only",
    "19:45": "left_only",
    "19:46": "left_only",
    "19:47": "left_only",
    "19:48": "left_only",
    "19:49": "left_only",
    "19:50": "left_only",
    "19:51": "left_only",
    "19:52": "left_only",
    "19:53": "left_only",
    "19:54": "left_only",
    "19:55": "left_only",
    "19:56": "left_only",
    "19:57": "left_only",
    "19:58": "both",
    "19:59": "both",
    "19:64": "right_only",
    "19:77": "right_only",
    "19:78": "right_only",
    "19:79": "right_only",
    "19:80": "right_only",
    "19:81": "right_only",
    "19:82": "right_only",
    "19:83": "right_only",
    "19:84": "right_only",
    "19:86": "right_only",
    "19:87": "right_only",
    "19:88": "right_only",
    "19:93": "left_only",
    "19:109": "left_only",
    "19:116": "left_only",
    "19:117": "left_only",
    "19:118": "left_only",
    "19:119": "left_only",
    "19:120": "left_only",
    "19:133": "right_only",
    "19:134": "right_only",
    "19:138": "right_only",
    "19:140": "right_only",
    "19:141": "right_only",
    "19:142": "right_only",
    "19:143": "right_only",
    "19:144": "right_only",
    "19:145": "right_only",
    "19:146": "right_only",
    "19:147": "right_only",
    "19:148": "right_only",
    "19:149": "right_only",
    "19:150": "right_only",
    "19:151": "right_only",
    "19:152": "right_only",
    "19:153": "right_only",
    "19:154": "right_only",
    "19:155": "right_only",
    "19:156": "right_only",
    "19:157": "right_only",
    "19:158": "right_only",
    "19:159": "right_only",
    "19:160": "right_only",
    "19:161": "right_only",
    "19:162": "right_only",
    "19:163": "right_only",
    "19:183": "left_only",
    "19:188": "left_only",
    "19:189": "left_only",
    "19:190": "left_only",
    "19:191": "left_only",
    "19:209": "right_only",
    "19:211": "right_only",
    "19:212": "right_only",
    "19:214": "right_only",
    "19:215": "right_only",
    "19:216": "right_only",
    "19:217": "right_only",
    "19:218": "right_only",
    "19:219": "right_only",
    "19:220": "right_only",
    "19:221": "right_only",
    "19:222": "right_only",
    "19:223": "right_only",
    "19:224": "right_only",
    "19:246": "left_only",
    "19:247": "left_only",
    "19:248": "left_only",
    "19:249": "left_only",
    "19:250": "left_only",
    "19:251": "left_only",
    "19:252": "left_only",
    "19:253": "left_only",
    "19:254": "left_only",
    "19:255": "left_only",
    "19:256": "left_only",
    "19:257": "left_only",
    "19:258": "left_only",
    "19:259": "left_only",
    "19:260": "left_only",
    "19:261": "left_only",
    "19:262": "left_only",
    "19:263": "left_only",
    "19:264": "left_only",
    "19:265": "left_only",
    "19:266": "left_only",
    "19:268": "left_only",
    "19:271": "left_only",
    "19:289": "right_only",
    "19:290": "right_only",
    "19:292": "right_only",
    "19:293": "right_only",
    "19:294": "right_only",
    "19:295": "right_only",
    "19:296": "right_only",
    "19:297": "right_only",
    "19:319": "left_only",
    "19:320": "left_only",
    "19:321": "left_only",
    "19:324": "left_only",
    "19:325": "left_only",
    "19:326": "left_only",
    "19:327": "left_only",
    "19:338": "right_only",
    "19:339": "right_only",
    "19:352": "right_only",
    "19:353": "right_only",
    "19:354": "right_only",
    "19:355": "right_only",
    "19:358": "right_only",
    "19:359": "right_only",
    "19:360": "right_only",
    "19:361": "right_only",
    "19:362": "right_only",
    "19:363": "right_only",
    "19:371": "left_only",
    "19:372": "left_only",
    "19:386": "left_only",
    "19:387": "left_only",
    "19:388": "left_only",
    "20:3": "both",
    "20:4": "both",
    "20:5": "both",
    "20:6": "both",
    "20:7": "both",
    "20:8": "both",
    "20:9": "left_only",
    "20:10": "both",
    "20:11": "left_only",
    "20:12": "left_only",
    "20:13": "left_only",
    "20:14": "left_only",
    "20:15": "left_only",
    "20:16": "left_only",
    "20:77": "left_only",
    "20:78": "left_only",
    "20:79": "left_only",
    "20:80": "left_only",
    "20:81": "left_only",
    "20:82": "left_only",
    "20:83": "left_only",
    "20:84": "left_only",
    "20:85": "left_only",
    "20:86": "left_only",
    "20:87": "left_only",
    "20:88": "left_only",
    "20:89": "left_only",
    "20:90": "left_only",
    "20:91": "left_only",
    "20:92": "left_only",
    "20:93": "left_only",
    "20:94": "left_only",
    "20:95": "left_only",
    "20:96": "left_only",
    "20:97": "left_only",
    "20:98": "left_only",
    "20:99": "left_only",
    "20:100": "left_only",
    "20:101": "left_only",
    "20:127": "left_only",
    "20:128": "left_only",
    "20:129": "left_only",
    "20:130": "left_only",
    "20:131": "left_only",
    "20:132": "left_only",
    "20:133": "left_only",
    "20:134": "left_only",
    "20:135": "left_only",
    "20:136": "left_only",
    "20:137": "left_only",
    "20:138": "left_only",
    "20:139": "left_only",
    "20:140": "left_only",
    "20:141": "left_only",
    "20:142": "left_only",
    "20:143": "left_only",
    "20:144": "left_only",
    "20:145": "left_only",
    "20:146": "left_only",
    "20:147": "left_only",
    "20:148": "left_only",
    "20:149": "left_only",
    "20:150": "left_only",
    "20:151": "left_only",
    "20:152": "left_only",
    "20:153": "left_only",
    "20:154": "left_only",
    "20:155": "left_only",
    "20:156": "left_only",
    "20:157": "left_only",
    "20:158": "left_only",
    "20:159": "left_only",
    "20:160": "left_only",
    "20:161": "left_only",
    "20:162": "left_only",
    "20:163": "left_only",
    "20:164": "left_only",
    "20:165": "left_only",
    "20:166": "left_only",
    "20:167": "left_only",
    "20:168": "both",
    "20:170": "right_only",
    "20:171": "right_only",
    "20:172": "right_only",
    "20:173": "right_only",
    "20:174": "right_only",
    "20:175": "right_only",
    "20:176": "right_only",
    "20:177": "right_only",
    "20:184": "right_only",
    "20:185": "right_only",
    "20:206": "right_only",
    "20:207": "right_only",
    "20:208": "right_only",
    "20:209": "left_only",
    "20:210": "left_only",
    "20:211": "left_only",
    "20:212": "left_only",
    "20:213": "left_only",
    "20:223": "left_only",
    "20:224": "left_only",
    "20:225": "left_only",
    "20:226": "left_only",
    "20:227": "left_only",
    "20:229": "left_only",
    "20:230": "left_only",
    "20:238": "left_only",
    "20:239": "left_only",
    "20:240": "left_only",
    "20:241": "left_only",
    "20:242": "left_only",
    "20:243": "left_only",
    "20:262": "right_only",
    "20:263": "right_only",
    "20:285": "left_only",
    "20:299": "left_only",
    "20:300": "left_only",
    "20:301": "left_only",
    "20:302": "left_only",
    "20:303": "left_only",
    "20:304": "left_only",
    "20:305": "left_only",
    "20:318": "left_only",
    "20:319": "left_only",
    "20:338": "right_only",
    "20:339": "right_only",
    "20:340": "right_only",
    "20:373": "left_only",
    "20:374": "left_only",
    "20:375": "left_only",
    "20:376": "left_only",
    "20:377": "left_only",
    "20:378": "left_only",
    "20:379": "left_only",
    "20:380": "left_only",
    "20:381": "left_only",
    "20:382": "left_only",
    "20:383": "left_only",
    "20:384": "both",
    "20:391": "left_only",
    "21:6": "right_only",
    "21:7": "right_only",
    "21:8": "right_only",
    "21:9": "right_only",
    "21:10": "right_only",
    "21:11": "right_only",
    "21:12": "right_only",
    "21:13": "right_only",
    "21:14": "right_only",
    "21:15": "both",
    "21:16": "both",
    "21:17": "right_only",
    "21:18": "right_only",
    "21:19": "right_only",
    "21:20": "right_only",
    "21:21": "right_only",
    "21:22": "right_only",
    "21:23": "both",
    "21:24": "left_only",
    "21:25": "left_only",
    "21:26": "left_only",
    "21:27": "left_only",
    "21:28": "left_only",
    "21:29": "left_only",
    "21:30": "left_only",
    "21:31": "left_only",
    "21:32": "left_only",
    "21:33": "left_only",
    "21:34": "left_only",
    "21:35": "left_only",
    "21:36": "left_only",
    "21:37": "left_only",
    "21:38": "left_only",
    "21:39": "left_only",
    "21:40": "left_only",
    "21:41": "left_only",
    "21:42": "left_only",
    "21:43": "left_only",
    "21:44": "left_only",
    "21:58": "left_only",
    "21:59": "left_only",
    "21:60": "left_only",
    "21:61": "left_only",
    "21:62": "left_only",
    "21:63": "left_only",
    "21:64": "left_only",
    "21:65": "both",
    "21:66": "both",
    "21:67": "both",
    "21:68": "both",
    "21:69": "both",
    "21:70": "both",
    "21:73": "both",
    "21:74": "both",
    "21:75": "both",
    "21:76": "both",
    "21:78": "right_only",
    "21:79": "right_only",
    "21:80": "right_only",
    "21:81": "right_only",
    "21:82": "right_only",
    "21:83": "right_only",
    "21:84": "right_only",
    "21:85": "right_only",
    "21:86": "right_only",
    "21:87": "right_only",
    "21:88": "right_only",
    "21:89": "right_only",
    "21:90": "right_only",
    "21:91": "right_only",
    "21:92": "right_only",
    "21:93": "right_only",
    "21:94": "right_only",
    "21:95": "right_only",
    "21:96": "right_only",
    "21:97": "right_only",
    "21:98": "right_only",
    "21:99": "right_only",
    "21:100": "right_only",
    "21:101": "right_only",
    "21:102": "right_only",
    "21:103": "right_only",
    "21:104": "right_only",
    "21:110": "right_only",
    "21:111": "right_only",
    "21:112": "right_only",
    "21:113": "right_only",
    "21:114": "right_only",
    "21:115": "right_only",
    "21:116": "right_only",
    "21:122": "both",
    "21:123": "both",
    "21:124": "both",
    "21:125": "both",
    "21:126": "both",
    "21:127": "both",
    "21:128": "both",
    "21:129": "both",
    "21:130": "both",
    "21:131": "both",
    "21:132": "both",
    "21:133": "both",
    "21:134": "both",
    "21:135": "both",
    "21:136": "both",
    "21:137": "both",
    "21:138": "both",
    "21:139": "both",
    "21:140": "both",
    "21:141": "both",
    "21:142": "both",
    "21:143": "both",
    "21:144": "both",
    "21:145": "both",
    "21:146": "both",
    "21:147": "both",
    "21:148": "both",
    "21:149": "both",
    "21:150": "left_only",
    "21:151": "left_only",
    "21:152": "left_only",
    "21:153": "left_only",
    "21:154": "left_only",
    "21:155": "left_only",
    "21:156": "left_only",
    "21:157": "left_only",
    "21:158": "left_only",
    "21:159": "left_only",
    "21:160": "left_only",
    "21:161": "left_only",
    "21:162": "left_only",
    "21:163": "left_only",
    "21:164": "left_only",
    "21:165": "left_only",
    "21:193": "right_only",
    "21:194": "right_only",
    "21:195": "right_only",
    "21:196": "right_only",
    "21:197": "right_only",
    "21:198": "right_only",
    "21:199": "right_only",
    "21:200": "right_only",
    "21:201": "right_only",
    "21:202": "right_only",
    "21:213": "right_only",
    "21:214": "right_only",
    "21:216": "right_only",
    "21:217": "right_only",
    "21:218": "right_only",
    "21:219": "right_only",
    "21:220": "right_only",
    "21:221": "right_only",
    "21:222": "right_only",
    "21:223": "right_only",
    "21:224": "right_only",
    "21:225": "both",
    "21:226": "both",
    "21:227": "both",
    "21:228": "both",
    "21:229": "right_only",
    "21:234": "left_only",
    "21:235": "left_only",
    "21:236": "left_only",
    "21:237": "left_only",
    "21:238": "left_only",
    "21:239": "left_only",
    "21:240": "left_only",
    "21:241": "left_only",
    "21:242": "left_only",
    "21:243": "left_only",
    "21:244": "left_only",
    "21:245": "left_only",
    "21:246": "left_only",
    "21:260": "left_only",
    "21:261": "left_only",
    "21:262": "left_only",
    "21:263": "left_only",
    "21:264": "left_only",
    "21:265": "left_only",
    "21:266": "left_only",
    "21:267": "left_only",
    "21:268": "both",
    "21:269": "both",
    "21:281": "right_only",
    "21:282": "right_only",
    "21:283": "right_only",
    "21:284": "right_only",
    "21:285": "right_only",
    "21:286": "right_only",
    "21:287": "right_only",
    "21:288": "right_only",
    "21:300": "right_only",
    "21:301": "right_only",
    "21:302": "right_only",
    "21:303": "right_only",
    "21:304": "right_only",
    "21:305": "right_only",
    "21:306": "right_only",
    "21:307": "right_only",
    "21:308": "right_only",
    "21:309": "right_only",
    "21:311": "both",
    "21:312": "both",
    "21:313": "both",
    "21:314": "both",
    "21:315": "both",
    "21:316": "both",
    "21:317": "both",
    "21:320": "left_only",
    "21:322": "left_only",
    "21:323": "left_only",
    "21:324": "left_only",
    "21:325": "left_only",
    "21:326": "left_only",
    "21:327": "left_only",
    "21:328": "left_only",
    "21:329": "left_only",
    "21:343": "left_only",
    "21:344": "left_only",
    "21:345": "left_only",
    "21:346": "left_only",
    "21:347": "left_only",
    "21:348": "left_only",
    "21:349": "left_only",
    "21:350": "left_only",
    "21:351": "both",
    "21:352": "both",
    "21:353": "left_only",
    "21:358": "right_only",
    "21:359": "right_only",
    "21:360": "right_only",
    "21:361": "right_only",
    "21:362": "right_only",
    "21:363": "right_only",
    "21:364": "right_only",
    "21:365": "right_only",
    "21:370": "right_only",
    "21:371": "right_only",
    "21:379": "right_only",
    "21:380": "right_only",
    "21:381": "right_only",
    "21:382": "right_only",
    "21:383": "right_only",
    "21:384": "right_only",
    "21:385": "right_only",
    "21:386": "right_only",
    "21:387": "right_only",
    "21:388": "both",
    "21:389": "both",
    "21:390": "both",
    "21:391": "both",
    "22:70": "left_only",
    "22:71": "left_only",
    "22:72": "left_only",
    "22:73": "left_only",
    "22:74": "left_only",
    "22:75": "left_only",
    "22:76": "left_only",
    "22:77": "left_only",
    "22:78": "left_only",
    "22:79": "left_only",
    "22:80": "left_only",
    "22:498": "right_only",
    "22:526": "left_only",
    "23:3": "left_only",
    "23:4": "left_only",
    "23:5": "left_only",
    "23:6": "left_only",
    "23:7": "left_only",
    "23:8": "left_only",
    "23:18": "left_only",
    "23:34": "right_only",
    "23:35": "right_only",
    "23:36": "right_only",
    "23:360": "right_only",
    "23:361": "right_only",
    "23:362": "right_only",
    "23:363": "right_only",
    "23:377": "right_only",
    "23:381": "right_only",
    "23:382": "right_only",
    "23:383": "right_only",
    "23:384": "right_only",
    "23:385": "right_only",
    "23:386": "right_only",
    "23:387": "right_only",
    "23:388": "right_only",
    "23:389": "right_only",
    "23:390": "right_only",
    "23:623": "right_only",
    "23:624": "right_only",
    "23:625": "right_only",
    "23:626": "right_only",
    "23:627": "right_only",
    "23:628": "right_only",
    "23:629": "right_only",
    "23:647": "left_only",
    "23:648": "left_only",
    "23:649": "left_only",
    "23:650": "left_only",
    "23:651": "left_only",
    "23:652": "left_only",
    "23:653": "left_only",
    "23:654": "left_only",
    "23:655": "left_only",
    "23:656": "left_only",
    "23:657": "left_only",
    "23:658": "left_only",
    "23:659": "left_only",
    "23:660": "left_only",
    "23:661": "left_only",
    "23:662": "left_only",
    "23:663": "left_only",
    "23:672": "right_only",
    "23:673": "right_only",
    "23:674": "right_only",
    "23:675": "right_only",
    "23:676": "right_only",
    "23:677": "right_only",
    "23:678": "right_only",
    "23:695": "left_only",
    "23:696": "left_only",
    "23:697": "left_only",
    "23:698": "left_only",
    "23:699": "left_only",
    "23:700": "left_only",
    "23:701": "left_only",
    "23:702": "left_only",
    "23:703": "left_only",
    "23:704": "left_only",
    "23:705": "left_only",
    "23:706": "left_only",
    "23:707": "left_only",
    "23:708": "left_only",
    "23:709": "left_only",
    "23:710": "left_only",
    "23:711": "left_only",
    "23:733": "right_only",
    "23:734": "right_only",
    "23:735": "right_only",
    "23:736": "right_only",
    "23:737": "right_only",
    "23:738": "right_only",
    "23:739": "right_only",
    "23:740": "right_only",
    "23:741": "right_only",
    "23:742": "right_only",
    "23:773": "left_only",
    "26:71": "right_only",
    "26:72": "right_only",
    "26:73": "right_only",
    "26:74": "right_only",
    "26:75": "right_only",
    "26:76": "right_only",
    "26:77": "right_only",
    "26:78": "right_only",
    "26:79": "right_only",
    "26:80": "right_only",
    "26:81": "right_only",
    "26:140": "right_only",
    "26:150": "right_only",
    "26:151": "right_only",
    "26:152": "right_only",
    "26:153": "right_only",
    "26:154": "right_only",
    "26:155": "right_only",
    "26:156": "right_only",
    "26:157": "right_only",
    "26:158": "right_only",
    "26:178": "left_only",
    "26:179": "left_only",
    "26:180": "left_only",
    "26:181": "left_only",
    "26:182": "left_only",
    "26:183": "left_only",
    "26:184": "left_only",
    "26:185": "left_only",
    "26:186": "left_only",
    "26:187": "left_only",
    "26:188": "left_only",
    "26:201": "both",
    "26:204": "right_only",
    "26:205": "right_only",
    "26:206": "right_only",
    "26:207": "right_only",
    "26:208": "right_only",
    "26:209": "right_only",
    "26:210": "right_only",
    "26:211": "right_only",
    "26:212": "right_only",
    "26:213": "right_only",
    "26:214": "right_only",
    "26:215": "right_only",
    "26:216": "right_only",
    "27:213": "right_only",
    "27:214": "right_only",
    "27:215": "right_only",
    "27:216": "right_only",
    "27:217": "right_only",
    "27:218": "right_only",
    "27:219": "right_only",
    "27:220": "right_only",
    "27:231": "right_only",
    "27:232": "right_only",
    "27:233": "right_only",
    "27:234": "right_only",
    "27:235": "right_only",
    "27:236": "right_only",
    "27:237": "right_only",
    "27:238": "right_only",
    "27:239": "right_only",
    "27:240": "right_only",
    "27:241": "right_only",
    "27:304": "right_only",
    "27:313": "right_only",
    "27:314": "right_only",
    "27:315": "right_only",
    "27:316": "right_only",
    "27:317": "right_only",
    "27:321": "right_only",
    "27:322": "right_only",
    "27:323": "right_only",
    "27:324": "right_only",
    "27:325": "right_only",
    "27:326": "right_only",
    "27:327": "right_only",
    "27:328": "right_only",
    "27:330": "both",
    "27:334": "both",
    "27:335": "both",
    "27:336": "both",
    "27:337": "both",
    "27:338": "both",
    "27:339": "both",
    "27:340": "both",
    "27:341": "both",
    "27:342": "both",
    "27:343": "both",
    "27:344": "both",
    "27:345": "both",
    "27:346": "both",
    "27:347": "both",
    "27:348": "both",
    "27:384": "both",
    "27:385": "both",
    "27:386": "both",
    "27:387": "both",
    "27:388": "both",
    "27:396": "right_only",
    "27:397": "right_only",
    "27:398": "right_only",
    "27:399": "right_only",
    "27:400": "right_only",
    "27:460": "right_only",
    "27:461": "right_only",
    "27:462": "right_only",
    "27:463": "right_only",
    "27:464": "right_only",
    "27:465": "right_only",
    "27:466": "right_only",
    "27:467": "right_only",
    "27:468": "right_only",
    "27:469": "right_only",
    "27:470": "right_only",
    "27:471": "right_only",
    "27:472": "right_only",
    "27:473": "right_only",
    "27:474": "right_only",
    "27:597": "left_only",
    "27:667": "left_only",
    "27:668": "left_only",
    "27:669": "left_only",
    "27:680": "left_only",
    "27:1027": "right_only",
    "27:1028": "right_only",
    "27:1029": "right_only",
    "27:1030": "right_only",
    "27:1031": "right_only",
    "27:1032": "right_only",
    "27:1033": "right_only",
    "27:1034": "right_only",
    "27:1051": "right_only",
    "27:1052": "right_only",
    "27:1053": "right_only",
    "27:1054": "right_only",
    "27:1055": "right_only",
    "27:1056": "right_only",
    "27:1314": "left_only",
    "27:1315": "left_only",
    "27:1316": "left_only",
    "27:1317": "left_only",
    "27:1655": "left_only",
    "27:1656": "left_only",
    "27:1657": "left_only",
    "27:1658": "left_only",
    "27:1659": "left_only",
    "27:1660": "left_only",
    "29:285": "left_only",
    "29:286": "left_only",
    "29:287": "left_only",
    "29:288": "left_only",
    "29:289": "left_only",
    "29:290": "left_only",
    "30:243": "left_only",
    "30:244": "left_only",
    "30:245": "left_only",
    "30:246": "left_only",
    "30:268": "right_only",
    "30:271": "right_only",
    "30:294": "left_only",
    "30:295": "left_only",
    "30:296": "left_only",
    "30:297": "left_only",
    "31:0": "both",
    "31:1": "both",
    "31:2": "both",
    "31:3": "both",
    "31:4": "both",
    "31:5": "both",
    "31:6": "both",
    "31:7": "both",
    "31:8": "both",
    "31:10": "left_only",
    "31:11": "left_only",
    "31:12": "left_only",
    "31:13": "left_only",
    "31:14": "left_only",
    "31:15": "left_only",
    "31:16": "left_only",
    "31:17": "left_only",
    "31:18": "left_only",
    "31:19": "left_only",
    "31:20": "left_only",
    "31:21": "left_only",
    "31:22": "left_only",
    "31:23": "left_only",
    "31:24": "left_only",
    "31:25": "left_only",
    "31:26": "left_only",
    "31:27": "left_only",
    "31:28": "left_only",
    "31:29": "left_only",
    "31:30": "both",
    "31:33": "right_only",
    "31:34": "right_only",
    "31:35": "right_only",
    "31:36": "right_only",
    "31:37": "right_only",
    "31:38": "right_only",
    "31:39": "right_only",
    "31:40": "right_only",
    "31:41": "right_only",
    "31:42": "right_only",
    "31:43": "right_only",
    "31:44": "right_only",
    "31:45": "right_only",
    "31:46": "right_only",
    "31:47": "right_only",
    "31:48": "right_only",
    "31:49": "right_only",
    "31:50": "right_only",
    "31:51": "right_only",
    "31:52": "right_only",
    "31:53": "right_only",
    "31:54": "right_only",
    "31:55": "right_only",
    "31:56": "right_only",
    "31:57": "right_only",
    "31:58": "right_only",
    "31:60": "right_only",
    "31:64": "left_only",
    "31:65": "left_only",
    "31:66": "left_only",
    "31:67": "left_only",
    "31:68": "left_only",
    "31:69": "left_only",
    "31:70": "left_only",
    "31:71": "left_only",
    "31:72": "left_only",
    "31:73": "left_only",
    "31:74": "left_only",
    "31:75": "left_only",
    "31:76": "left_only",
    "31:77": "left_only",
    "31:78": "left_only",
    "31:79": "left_only",
    "31:80": "left_only",
    "31:81": "left_only",
    "31:82": "both",
    "31:83": "both",
    "31:84": "left_only",
    "31:85": "left_only",
    "31:93": "right_only",
    "31:94": "right_only",
    "31:95": "right_only",
    "31:96": "right_only",
    "31:97": "right_only",
    "31:98": "right_only",
    "31:99": "right_only",
    "31:100": "right_only",
    "31:101": "right_only",
    "31:102": "right_only",
    "31:103": "right_only",
    "31:104": "right_only",
    "31:105": "right_only",
    "31:106": "right_only",
    "31:107": "right_only",
    "31:108": "right_only",
    "31:109": "right_only",
    "31:110": "both",
    "31:111": "both",
    "31:112": "both",
    "31:113": "right_only",
    "31:121": "left_only",
    "31:122": "left_only",
    "31:123": "left_only",
    "31:124": "left_only",
    "31:125": "left_only",
    "31:126": "left_only",
    "31:127": "left_only",
    "31:128": "left_only",
    "31:129": "left_only",
    "31:130": "left_only",
    "31:131": "left_only",
    "31:132": "left_only",
    "31:133": "left_only",
    "31:134": "left_only",
    "31:135": "left_only",
    "31:148": "right_only",
    "31:149": "right_only",
    "31:150": "right_only",
    "31:151": "right_only",
    "31:152": "right_only",
    "31:153": "right_only",
    "31:154": "right_only",
    "31:155": "right_only",
    "31:156": "right_only",
    "31:160": "right_only",
    "31:161": "right_only",
    "31:162": "right_only",
    "31:163": "right_only",
    "31:165": "both",
    "31:176": "left_only",
    "31:177": "left_only",
    "31:178": "left_only",
    "31:179": "left_only",
    "31:184": "left_only",
    "31:185": "left_only",
    "31:190": "both",
    "31:191": "left_only",
    "31:192": "left_only",
    "31:193": "left_only",
    "31:199": "right_only",
    "31:201": "right_only",
    "31:202": "right_only",
    "31:203": "right_only",
    "31:204": "right_only",
    "31:205": "right_only",
    "31:206": "right_only",
    "31:210": "right_only",
    "31:211": "right_only",
    "31:212": "right_only",
    "31:227": "left_only",
    "31:228": "left_only",
    "31:229": "left_only",
    "31:230": "left_only",
    "31:235": "left_only",
    "31:236": "left_only",
    "31:237": "left_only",
    "31:238": "left_only",
    "31:239": "left_only",
    "31:260": "right_only",
    "32:0": "both",
    "32:1": "both",
    "32:2": "both",
    "32:3": "both",
    "32:4": "both",
    "32:5": "both",
    "32:6": "both",
    "32:7": "both",
    "32:8": "both",
    "32:9": "both",
    "32:10": "both",
    "32:11": "both",
    "32:12": "both",
    "32:13": "both",
    "32:14": "both",
    "32:15": "both",
    "32:16": "both",
    "32:17": "both",
    "32:18": "both",
    "32:19": "both",
    "32:20": "both",
    "32:21": "both",
    "32:22": "both",
    "32:23": "both",
    "32:24": "both",
    "32:25": "both",
    "32:26": "both",
    "32:27": "both",
    "32:28": "both",
    "32:32": "left_only",
    "32:33": "left_only",
    "32:34": "left_only",
    "32:35": "left_only",
    "32:36": "left_only",
    "32:37": "left_only",
    "32:38": "left_only",
    "32:39": "left_only",
    "32:41": "both",
    "32:42": "both",
    "32:43": "both",
    "32:44": "both",
    "32:45": "both",
    "32:46": "both",
    "32:47": "both",
    "32:48": "both",
    "32:55": "right_only",
    "32:56": "right_only",
    "32:57": "right_only",
    "32:58": "right_only",
    "32:59": "right_only",
    "32:60": "right_only",
    "32:61": "right_only",
    "32:62": "right_only",
    "32:63": "right_only",
    "32:64": "right_only",
    "32:65": "right_only",
    "32:66": "right_only",
    "32:67": "right_only",
    "32:68": "right_only",
    "32:69": "right_only",
    "32:70": "right_only",
    "32:71": "right_only",
    "32:72": "right_only",
    "32:74": "both",
    "32:76": "both",
    "32:82": "left_only",
    "32:83": "left_only",
    "32:84": "left_only",
    "32:86": "left_only",
    "32:87": "left_only",
    "32:88": "left_only",
    "32:89": "left_only",
    "32:90": "left_only",
    "32:91": "left_only",
    "32:92": "left_only",
    "32:93": "left_only",
    "32:94": "left_only",
    "32:95": "left_only",
    "32:96": "left_only",
    "32:97": "left_only",
    "32:109": "right_only",
    "32:112": "right_only",
    "32:113": "right_only",
    "32:114": "right_only",
    "32:115": "right_only",
    "32:116": "right_only",
    "32:117": "right_only",
    "32:118": "right_only",
    "32:119": "right_only",
    "32:120": "right_only",
    "32:121": "right_only",
    "32:122": "right_only",
    "32:123": "right_only",
    "32:144": "left_only",
    "32:145": "left_only",
    "32:146": "left_only",
    "32:147": "left_only",
    "32:196": "left_only",
    "32:197": "left_only",
    "32:198": "left_only",
    "32:199": "left_only",
    "32:221": "right_only",
    "32:222": "right_only",
    "32:223": "right_only",
    "33:291": "left_only",
    "33:292": "left_only",
    "33:293": "left_only",
    "33:317": "right_only",
    "33:318": "right_only",
    "35:66": "right_only",
    "35:67": "right_only",
    "35:68": "right_only",
    "35:69": "right_only",
    "35:70": "right_only",
    "35:81": "right_only",
    "35:87": "both",
    "35:88": "both",
    "35:89": "both",
    "35:91": "left_only",
    "35:92": "left_only",
    "35:93": "left_only",
    "35:114": "both",
    "35:119": "right_only",
    "35:120": "right_only",
    "35:121": "right_only",
    "35:122": "right_only",
    "35:140": "both",
    "35:142": "both",
    "35:147": "left_only",
    "35:148": "left_only",
    "35:154": "left_only",
    "35:155": "left_only",
    "35:156": "left_only",
    "35:157": "left_only",
    "35:158": "left_only",
    "35:166": "both",
    "35:169": "right_only",
    "35:224": "right_only",
    "35:225": "right_only",
    "35:245": "both",
    "35:246": "both",
    "35:271": "both",
    "36:223": "left_only",
    "36:224": "left_only",
    "36:225": "left_only",
    "36:226": "left_only",
    "36:227": "left_only",
    "38:94": "left_only",
    "39:0": "both",
    "39:1": "both",
    "39:2": "both",
    "39:3": "both",
    "39:4": "both",
    "39:5": "both",
    "39:6": "both",
    "39:7": "both",
    "39:8": "both",
    "39:9": "both",
    "39:10": "both",
    "39:11": "both",
    "39:12": "both",
    "39:13": "both",
    "39:14": "both",
    "39:15": "both",
    "39:16": "both",
    "39:17": "both",
    "39:18": "both",
    "39:19": "both",
    "39:20": "both",
    "39:21": "both",
    "39:22": "both",
    "39:23": "both",
    "39:24": "both",
    "39:25": "both",
    "39:26": "both",
    "39:27": "both",
    "39:28": "both",
    "39:29": "both",
    "39:30": "both",
    "39:31": "both",
    "39:32": "both",
    "39:33": "both",
    "39:34": "both",
    "39:35": "both",
    "39:36": "both",
    "39:37": "both",
    "39:38": "both",
    "39:39": "both",
    "39:40": "both",
    "39:41": "both",
    "39:42": "both",
    "39:43": "both",
    "39:44": "both",
    "39:45": "both",
    "39:46": "both",
    "39:47": "both",
    "39:48": "both",
    "39:49": "both",
    "39:50": "both",
    "39:51": "both",
    "39:52": "both",
    "39:53": "both",
    "39:54": "both",
    "39:55": "both",
    "39:56": "both",
    "39:57": "both",
    "39:58": "both",
    "39:59": "both",
    "39:60": "both",
    "39:61": "both",
    "39:62": "both",
    "39:63": "both",
    "39:64": "both",
    "39:65": "both",
    "39:66": "both",
    "39:67": "both",
    "39:68": "both",
    "39:69": "both",
    "39:70": "both",
    "39:71": "both",
    "39:72": "both",
    "39:73": "both",
    "39:74": "both",
    "39:75": "both",
    "39:76": "both",
    "39:77": "both",
    "39:78": "both",
    "39:79": "both",
    "39:80": "both",
    "39:81": "both",
    "39:82": "both",
    "39:83": "both",
    "39:84": "both",
    "39:85": "both",
    "39:86": "both",
    "39:87": "both",
    "39:88": "both",
    "39:89": "both",
    "39:90": "both",
    "39:91": "both",
    "39:92": "both",
    "39:93": "both",
    "39:94": "both",
    "39:95": "both",
    "39:96": "both",
    "39:97": "both",
    "39:107": "left_only",
    "39:110": "left_only",
    "40:54": "right_only",
    "41:107": "left_only",
    "41:108": "left_only",
    "41:109": "left_only",
    "41:133": "right_only",
    "41:134": "right_only",
    "41:135": "right_only",
    "41:136": "right_only",
    "41:137": "right_only",
    "41:138": "right_only",
    "41:149": "left_only",
    "41:150": "left_only",
    "41:152": "left_only",
    "41:175": "right_only",
    "41:176": "right_only",
    "42:10": "left_only",
    "42:11": "left_only",
    "42:12": "left_only",
    "42:36": "right_only",
    "42:37": "right_only",
    "42:60": "left_only",
    "42:61": "left_only",
    "42:152": "right_only",
    "42:153": "right_only",
    "43:0": "both",
    "43:1": "both",
    "43:2": "both",
    "43:3": "both",
    "43:4": "both",
    "43:5": "both",
    "43:6": "both",
    "43:7": "both",
    "43:8": "both",
    "43:9": "both",
    "43:10": "both",
    "43:11": "both",
    "43:12": "both",
    "43:13": "both",
    "43:14": "both",
    "43:15": "both",
    "43:16": "both",
    "43:17": "both",
    "43:18": "both",
    "43:19": "both",
    "43:20": "both",
    "43:21": "both",
    "43:22": "both",
    "43:23": "both",
    "43:24": "both",
    "43:25": "both",
    "43:26": "both",
    "43:27": "both",
    "43:28": "both",
    "43:29": "both",
    "43:30": "both",
    "43:31": "both",
    "43:32": "both",
    "43:33": "both",
    "43:34": "both",
    "43:35": "both",
    "43:36": "both",
    "43:37": "both",
    "43:38": "both",
    "43:39": "both",
    "43:40": "both",
    "43:41": "both",
    "43:42": "both",
    "43:43": "both",
    "43:44": "both",
    "43:45": "both",
    "43:46": "both",
    "43:47": "both",
    "43:48": "both",
    "43:49": "both",
    "43:50": "both",
    "43:51": "both",
    "43:52": "both",
    "43:53": "both",
    "43:54": "both",
    "43:55": "both",
    "43:56": "both",
    "43:57": "both",
    "43:58": "both",
    "43:59": "left_only",
    "43:60": "left_only",
    "43:61": "left_only",
    "43:62": "left_only",
    "43:63": "left_only",
    "43:64": "left_only",
    "43:65": "left_only",
    "43:66": "left_only",
    "43:67": "left_only",
    "43:68": "left_only",
    "43:69": "left_only",
    "43:70": "left_only",
    "43:77": "left_only",
    "43:78": "left_only",
    "43:79": "left_only",
    "43:80": "left_only",
    "43:81": "left_only",
    "43:82": "left_only",
    "43:83": "left_only",
    "43:84": "left_only",
    "43:85": "left_only",
    "43:86": "left_only",
    "43:87": "left_only",
    "43:88": "left_only",
    "43:89": "left_only",
    "43:90": "left_only",
    "43:91": "left_only",
    "43:92": "left_only",
    "43:93": "both",
    "43:94": "both",
    "43:95": "right_only",
    "43:96": "right_only",
    "43:97": "right_only",
    "43:98": "right_only",
    "43:99": "right_only",
    "43:100": "right_only",
    "43:101": "right_only",
    "43:102": "right_only",
    "43:103": "right_only",
    "43:104": "right_only",
    "43:105": "right_only",
    "43:106": "right_only",
    "43:107": "right_only",
    "43:108": "right_only",
    "43:109": "right_only",
    "43:110": "right_only",
    "43:111": "right_only",
    "43:112": "right_only",
    "43:113": "right_only",
    "43:114": "right_only",
    "43:115": "right_only",
    "43:116": "right_only",
    "43:117": "right_only",
    "43:118": "right_only",
    "43:119": "right_only",
    "43:120": "right_only",
    "43:121": "right_only",
    "43:122": "right_only",
    "43:123": "right_only",
    "43:124": "right_only",
    "43:125": "right_only",
    "43:126": "right_only",
    "43:127": "right_only",
    "43:128": "right_only",
    "43:131": "left_only",
    "43:132": "left_only",
    "43:133": "left_only",
    "43:134": "left_only",
    "43:135": "left_only",
    "43:136": "left_only",
    "43:137": "left_only",
    "43:138": "left_only",
    "43:139": "left_only",
    "43:140": "left_only",
    "43:141": "left_only",
    "43:142": "left_only",
    "43:143": "left_only",
    "43:144": "left_only",
    "43:145": "left_only",
    "43:146": "left_only",
    "43:147": "left_only",
    "43:148": "left_only",
    "43:149": "left_only",
    "43:150": "left_only",
    "43:151": "left_only",
    "43:152": "left_only",
    "43:153": "left_only",
    "43:154": "left_only",
    "43:155": "left_only",
    "43:156": "left_only",
    "43:157": "left_only",
    "43:158": "left_only",
    "43:159": "left_only",
    "43:160": "left_only",
    "43:161": "left_only",
    "43:162": "left_only",
    "43:163": "both",
    "43:164": "both",
    "43:165": "both",
    "43:166": "both",
    "43:167": "both",
    "43:170": "both",
    "43:171": "both",
    "43:172": "both",
    "43:173": "both",
    "43:174": "right_only",
    "43:175": "right_only",
    "43:176": "right_only",
    "43:177": "right_only",
    "43:178": "right_only",
    "43:179": "right_only",
    "43:180": "right_only",
    "43:181": "right_only",
    "43:182": "right_only",
    "43:183": "right_only",
    "43:184": "right_only",
    "43:185": "right_only",
    "43:186": "right_only",
    "43:187": "right_only",
    "43:188": "right_only",
    "43:189": "right_only",
    "43:190": "right_only",
    "43:191": "right_only",
    "43:192": "right_only",
    "43:193": "right_only",
    "43:194": "right_only",
    "43:195": "right_only",
    "43:196": "right_only",
    "43:197": "right_only",
    "43:198": "right_only",
    "43:199": "right_only",
    "43:200": "both",
    "43:201": "both",
    "43:202": "both",
    "43:203": "both",
    "43:204": "both",
    "43:207": "left_only",
    "43:208": "left_only",
    "43:209": "left_only",
    "43:210": "left_only",
    "43:211": "left_only",
    "43:212": "left_only",
    "43:213": "left_only",
    "43:214": "left_only",
    "43:215": "left_only",
    "43:216": "left_only",
    "43:217": "left_only",
    "43:218": "left_only",
    "43:219": "left_only",
    "43:220": "left_only",
    "43:221": "left_only",
    "43:222": "left_only",
    "43:223": "left_only",
    "43:224": "left_only",
    "43:225": "left_only",
    "43:226": "left_only",
    "43:227": "left_only",
    "43:228": "left_only",
    "43:229": "left_only",
    "43:230": "left_only",
    "43:239": "both",
    "43:240": "right_only",
    "43:241": "right_only",
    "43:242": "right_only",
    "43:243": "right_only",
    "43:244": "right_only",
    "43:245": "right_only",
    "43:246": "right_only",
    "43:247": "right_only",
    "43:248": "right_only",
    "43:249": "right_only",
    "43:250": "right_only",
    "43:251": "right_only",
    "43:252": "right_only",
    "43:253": "right_only",
    "43:254": "right_only",
    "43:255": "right_only",
    "43:256": "right_only",
    "43:257": "right_only",
    "43:258": "right_only",
    "43:259": "right_only",
    "43:260": "right_only",
    "43:261": "right_only",
    "43:262": "right_only",
    "43:263": "right_only",
    "43:264": "right_only",
    "43:272": "left_only",
    "43:273": "left_only",
    "43:274": "left_only",
    "43:275": "left_only",
    "43:276": "left_only",
    "43:277": "left_only",
    "43:278": "left_only",
    "43:279": "left_only",
    "43:280": "left_only",
    "43:281": "left_only",
    "43:282": "left_only",
    "43:283": "left_only",
    "43:284": "left_only",
    "43:285": "left_only",
    "43:286": "left_only",
    "43:287": "left_only",
    "43:288": "left_only",
    "43:289": "left_only",
    "43:290": "left_only",
    "43:291": "left_only",
    "43:292": "left_only",
    "43:293": "left_only",
    "43:294": "both",
    "43:295": "both",
    "43:302": "right_only",
    "43:303": "right_only",
    "44:0": "both",
    "44:1": "both",
    "44:2": "both",
    "44:3": "both",
    "44:4": "both",
    "44:5": "both",
    "44:6": "both",
    "44:7": "both",
    "44:8": "both",
    "44:9": "both",
    "44:10": "both",
    "44:11": "both",
    "44:12": "both",
    "44:13": "both",
    "44:14": "both",
    "44:15": "both",
    "44:16": "both",
    "44:17": "both",
    "44:18": "both",
    "44:19": "both",
    "44:20": "both",
    "44:21": "both",
    "44:22": "both",
    "44:23": "both",
    "44:24": "both",
    "44:25": "both",
    "44:26": "both",
    "44:27": "both",
    "44:28": "both",
    "44:29": "both",
    "44:30": "both",
    "44:31": "both",
    "44:32": "both",
    "44:33": "both",
    "44:34": "both",
    "44:35": "both",
    "44:36": "both",
    "44:37": "both",
    "44:40": "right_only",
    "44:41": "right_only",
    "44:42": "right_only",
    "44:43": "right_only",
    "44:44": "right_only",
    "44:45": "right_only",
    "44:46": "right_only",
    "44:47": "right_only",
    "44:48": "right_only",
    "44:49": "right_only",
    "44:50": "right_only",
    "44:51": "right_only",
    "44:52": "right_only",
    "44:53": "right_only",
    "44:54": "right_only",
    "44:55": "right_only",
    "44:56": "right_only",
    "44:57": "right_only",
    "44:58": "right_only",
    "44:59": "right_only",
    "44:60": "right_only",
    "44:61": "right_only",
    "44:62": "right_only",
    "44:63": "right_only",
    "44:71": "left_only",
    "44:72": "left_only",
    "44:73": "left_only",
    "44:74": "left_only",
    "44:75": "left_only",
    "44:86": "right_only",
    "44:87": "right_only",
    "44:88": "right_only",
    "44:89": "right_only",
    "44:90": "right_only",
    "44:91": "right_only",
    "44:92": "right_only",
    "44:93": "right_only",
    "44:103": "right_only",
    "44:104": "right_only",
    "44:105": "right_only",
    "44:106": "right_only",
    "44:107": "right_only",
    "44:108": "right_only",
    "44:109": "right_only",
    "44:110": "right_only",
    "44:111": "right_only",
    "44:112": "right_only",
    "44:116": "left_only",
    "44:117": "left_only",
    "44:118": "left_only",
    "44:133": "right_only",
    "44:134": "right_only",
    "44:135": "right_only",
    "44:136": "right_only",
    "44:156": "right_only",
    "44:157": "right_only",
    "44:158": "right_only",
    "44:159": "right_only",
    "44:160": "right_only",
    "45:0": "right_only",
    "45:1": "right_only",
    "45:2": "right_only",
    "45:3": "right_only",
    "45:4": "right_only",
    "45:5": "right_only",
    "45:6": "right_only",
    "45:7": "right_only",
    "45:8": "right_only",
    "45:9": "right_only",
    "45:10": "right_only",
    "45:11": "right_only",
    "45:12": "both",
    "45:13": "both",
    "45:14": "both",
    "45:15": "both",
    "45:16": "both",
    "45:17": "both",
    "45:18": "both",
    "45:19": "both",
    "45:20": "both",
    "45:21": "left_only",
    "45:22": "left_only",
    "45:23": "left_only",
    "45:24": "left_only",
    "45:38": "left_only",
    "45:39": "left_only",
    "45:40": "left_only",
    "45:41": "left_only",
    "45:42": "left_only",
    "45:43": "left_only",
    "45:44": "left_only",
    "45:45": "left_only",
    "45:46": "left_only",
    "45:47": "left_only",
    "45:50": "right_only",
    "45:51": "right_only",
    "45:52": "right_only",
    "45:53": "right_only",
    "45:54": "right_only",
    "45:55": "right_only",
    "45:56": "right_only",
    "45:57": "right_only",
    "45:58": "right_only",
    "45:93": "left_only",
    "45:96": "right_only",
    "45:97": "right_only",
    "45:98": "right_only",
    "45:99": "right_only",
    "45:100": "right_only",
    "45:101": "right_only",
    "45:102": "right_only",
    "45:103": "right_only",
    "45:104": "right_only",
    "45:134": "left_only",
    "45:135": "left_only",
    "45:138": "right_only",
    "45:139": "right_only",
    "45:140": "right_only",
    "45:141": "right_only",
    "45:142": "right_only",
    "45:143": "right_only",
    "45:144": "right_only",
    "45:145": "right_only",
    "45:176": "left_only",
    "45:177": "left_only",
    "45:181": "right_only",
    "45:182": "right_only",
    "45:183": "right_only",
    "45:184": "right_only",
    "45:185": "right_only",
    "45:186": "right_only",
    "45:187": "right_only",
    "45:188": "right_only",
    "45:189": "right_only",
    "45:190": "right_only",
    "45:191": "right_only",
    "45:192": "right_only",
    "45:193": "right_only",
    "45:194": "right_only",
    "45:195": "right_only",
    "45:196": "right_only",
    "45:197": "right_only",
    "45:198": "right_only",
    "45:199": "right_only",
    "45:200": "both",
    "45:201": "both",
    "45:202": "left_only",
    "45:211": "left_only",
    "45:212": "left_only",
    "45:213": "left_only",
    "45:214": "left_only",
    "45:215": "left_only",
    "45:216": "left_only",
    "45:217": "left_only",
    "45:218": "left_only",
    "45:225": "right_only",
    "45:226": "right_only",
    "45:227": "right_only",
    "45:228": "right_only",
    "45:229": "right_only",
    "45:230": "right_only",
    "45:231": "right_only",
    "45:232": "right_only",
    "45:233": "right_only",
    "45:234": "right_only",
    "45:235": "right_only",
    "45:236": "right_only",
    "45:237": "right_only",
    "45:238": "right_only",
    "45:239": "both",
    "45:240": "both",
    "45:241": "both",
    "46:26": "right_only",
    "46:27": "right_only",
    "46:125": "left_only",
    "47:9": "right_only",
    "47:10": "right_only",
    "47:27": "left_only",
    "47:125": "right_only",
    "48:82": "right_only",
    "48:83": "right_only",
    "48:84": "right_only",
    "48:87": "both",
    "48:88": "both",
    "48:89": "both",
    "48:90": "both",
    "48:91": "both",
    "48:92": "both",
    "48:93": "both",
    "48:94": "both",
    "48:95": "both",
    "48:96": "both",
    "48:97": "both",
    "48:98": "both",
    "48:99": "both",
    "48:100": "both",
    "48:101": "both",
    "48:102": "both",
    "48:103": "both",
    "48:104": "both",
    "48:105": "both",
    "48:106": "both",
    "48:107": "both",
    "48:108": "both",
    "48:109": "both",
    "48:110": "both",
    "48:111": "both",
    "48:112": "both",
    "48:113": "both",
    "48:114": "both",
    "48:115": "both",
    "48:116": "both",
    "48:117": "both",
    "48:118": "both",
    "48:119": "both",
    "48:120": "both",
    "48:121": "both",
    "48:122": "both",
    "48:123": "both",
    "48:124": "both",
    "48:125": "both",
    "48:126": "both",
    "48:127": "both",
    "48:128": "both",
    "48:129": "both",
    "48:130": "both",
    "48:131": "both",
    "48:132": "both",
    "48:133": "both",
    "48:134": "both",
    "48:135": "both",
    "48:136": "both",
    "48:137": "both",
    "48:138": "both",
    "48:139": "both",
    "48:140": "both",
    "48:141": "both",
    "48:142": "both",
    "48:143": "both",
    "48:144": "both",
    "48:145": "both",
    "48:146": "both",
    "48:147": "both",
    "48:148": "both",
    "48:149": "both",
    "48:150": "both",
    "48:151": "both",
    "48:152": "both",
    "48:153": "both",
    "48:154": "both",
    "48:155": "both",
    "48:156": "both",
    "48:157": "both",
    "48:158": "both",
    "48:159": "both",
    "48:160": "both",
    "48:161": "both",
    "48:162": "both",
    "48:163": "both",
    "48:164": "both",
    "48:165": "both",
    "48:166": "both",
    "48:167": "both",
    "48:168": "both",
    "48:169": "both",
    "48:170": "both",
    "48:171": "both",
    "48:172": "both",
    "48:173": "both",
    "48:174": "both",
    "48:175": "both",
    "48:176": "both",
    "48:177": "both",
    "48:178": "both",
    "48:179": "both",
    "48:180": "both",
    "48:181": "both",
    "48:182": "both",
    "48:183": "both",
    "48:184": "both",
    "48:185": "both",
    "48:186": "both",
    "48:187": "both",
    "48:188": "both",
    "48:189": "both",
    "48:190": "both",
    "48:191": "both",
    "48:192": "both",
    "48:193": "both",
    "49:82": "left_only",
    "49:83": "left_only",
    "49:84": "left_only",
    "49:87": "both",
    "49:88": "both",
    "49:89": "both",
    "49:90": "both",
    "49:91": "both",
    "49:92": "both",
    "49:93": "both",
    "49:94": "both",
    "49:95": "both",
    "49:96": "both",
    "49:97": "both",
    "49:98": "both",
    "49:99": "both",
    "49:100": "both",
    "49:101": "both",
    "49:102": "both",
    "49:103": "both",
    "49:104": "both",
    "49:105": "both",
    "49:106": "both",
    "49:107": "both",
    "49:108": "both",
    "49:109": "both",
    "49:110": "both",
    "49:111": "both",
    "49:112": "both",
    "49:113": "both",
    "49:114": "both",
    "49:115": "both",
    "49:116": "both",
    "49:117": "both",
    "49:118": "both",
    "49:119": "both",
    "49:120": "both",
    "49:121": "both",
    "49:122": "both",
    "49:123": "both",
    "49:124": "both",
    "49:125": "both",
    "49:126": "both",
    "49:127": "both",
    "49:128": "both",
    "49:129": "both",
    "49:130": "both",
    "49:131": "both",
    "49:132": "both",
    "49:133": "both",
    "49:134": "both",
    "49:135": "both",
    "49:136": "both",
    "49:137": "both",
    "49:138": "both",
    "49:139": "both",
    "49:140": "both",
    "49:141": "both",
    "49:142": "both",
    "49:143": "both",
    "49:144": "both",
    "49:145": "both",
    "49:146": "both",
    "49:147": "both",
    "49:148": "both",
    "49:149": "both",
    "49:150": "both",
    "49:151": "both",
    "49:152": "both",
    "49:153": "both",
    "49:154": "both",
    "49:155": "both",
    "49:156": "both",
    "49:157": "both",
    "49:158": "both",
    "49:159": "both",
    "49:160": "both",
    "49:161": "both",
    "49:162": "both",
    "49:163": "both",
    "49:164": "both",
    "49:165": "both",
    "49:166": "both",
    "49:167": "both",
    "49:168": "both",
    "49:169": "both",
    "49:170": "both",
    "49:171": "both",
    "49:172": "both",
    "49:173": "both",
    "49:174": "both",
    "49:175": "both",
    "49:176": "both",
    "49:177": "both",
    "49:178": "both",
    "49:179": "both",
    "49:180": "both",
    "49:181": "both",
    "49:182": "both",
    "49:183": "both",
    "49:184": "both",
    "49:185": "both",
    "49:186": "both",
    "49:187": "both",
    "49:188": "both",
    "49:189": "both",
    "49:190": "both",
    "49:191": "both",
    "49:192": "both",
    "49:193": "both",
    "50:0": "both",
    "50:1": "both",
    "50:2": "both",
    "50:3": "both",
    "50:4": "both",
    "50:5": "both",
    "50:6": "both",
    "50:7": "both",
    "50:8": "both",
    "50:9": "both",
    "50:10": "both",
    "50:11": "both",
    "50:12": "both",
    "50:13": "both",
    "50:14": "both",
    "50:15": "both",
    "50:16": "both",
    "50:17": "both",
    "50:18": "both",
    "50:19": "both",
    "50:20": "both",
    "50:21": "both",
    "50:22": "both",
    "50:23": "both",
    "50:24": "both",
    "50:25": "both",
    "50:26": "both",
    "50:27": "both",
    "50:29": "both",
    "50:30": "both",
    "50:31": "both",
    "50:32": "left_only",
    "50:33": "left_only",
    "50:34": "left_only",
    "50:35": "left_only",
    "50:36": "left_only",
    "50:37": "left_only",
    "50:38": "left_only",
    "50:39": "left_only",
    "50:40": "left_only",
    "50:41": "left_only",
    "50:42": "left_only",
    "50:43": "left_only",
    "50:44": "left_only",
    "50:45": "left_only",
    "50:46": "left_only",
    "50:58": "both",
    "50:59": "right_only",
    "50:60": "right_only",
    "50:61": "right_only",
    "50:62": "right_only",
    "50:63": "right_only",
    "50:64": "right_only",
    "50:65": "right_only",
    "50:66": "right_only",
    "50:67": "right_only",
    "50:68": "right_only",
    "50:69": "right_only",
    "50:70": "right_only",
    "50:71": "right_only",
    "50:72": "right_only",
    "50:73": "right_only",
    "50:74": "right_only",
    "50:75": "right_only",
    "50:76": "both",
    "50:82": "left_only",
    "50:83": "left_only",
    "50:84": "left_only",
    "50:85": "left_only",
    "50:86": "left_only",
    "50:87": "left_only",
    "50:88": "left_only",
    "50:89": "left_only",
    "50:90": "left_only",
    "50:91": "left_only",
    "50:92": "left_only",
    "50:93": "left_only",
    "50:94": "left_only",
    "50:95": "left_only",
    "50:111": "right_only",
    "50:112": "right_only",
    "50:113": "right_only",
    "50:114": "right_only",
    "50:115": "right_only",
    "50:116": "right_only",
    "50:117": "right_only",
    "50:118": "right_only",
    "50:119": "right_only",
    "50:120": "right_only",
    "50:121": "right_only",
    "50:122": "right_only",
    "50:123": "right_only",
    "50:124": "right_only",
    "50:128": "both",
    "50:130": "left_only",
    "50:131": "left_only",
    "50:132": "left_only",
    "50:133": "left_only",
    "50:134": "left_only",
    "50:135": "left_only",
    "50:136": "left_only",
    "50:137": "left_only",
    "50:138": "left_only",
    "50:139": "left_only",
    "50:140": "left_only",
    "50:141": "left_only",
    "50:142": "left_only",
    "50:143": "left_only",
    "50:158": "right_only",
    "50:159": "right_only",
    "50:160": "right_only",
    "50:161": "right_only",
    "50:162": "right_only",
    "50:163": "right_only",
    "50:164": "right_only",
    "50:165": "right_only",
    "50:166": "right_only",
    "50:167": "right_only",
    "50:168": "right_only",
    "50:169": "right_only",
    "50:170": "right_only",
    "50:171": "right_only",
    "50:172": "right_only",
    "50:173": "right_only",
    "50:174": "right_only",
    "50:175": "right_only",
    "50:176": "right_only",
    "51:34": "left_only",
    "51:35": "left_only",
    "51:36": "left_only",
    "51:37": "left_only",
    "51:38": "left_only",
    "51:39": "left_only",
    "51:40": "left_only",
    "51:41": "left_only",
    "51:46": "right_only",
    "51:47": "right_only",
    "51:48": "right_only",
    "51:49": "right_only",
    "51:50": "right_only",
    "51:51": "right_only",
    "51:52": "right_only",
    "51:53": "right_only",
    "51:54": "right_only",
    "51:55": "right_only",
    "51:56": "right_only",
    "51:57": "right_only",
    "51:58": "right_only",
    "51:59": "right_only",
    "51:60": "right_only",
    "51:61": "right_only",
    "51:62": "right_only",
    "51:63": "right_only",
    "51:64": "right_only",
    "51:65": "right_only",
    "51:66": "right_only",
    "51:70": "right_only",
    "51:71": "right_only",
    "51:77": "left_only",
    "51:78": "left_only",
    "51:79": "left_only",
    "51:80": "left_only",
    "51:81": "left_only",
    "51:82": "left_only",
    "51:83": "left_only",
    "51:84": "left_only",
    "51:85": "left_only",
    "51:86": "left_only",
    "51:87": "left_only",
    "51:88": "left_only",
    "51:89": "left_only",
    "51:90": "left_only",
    "51:91": "left_only",
    "51:92": "left_only",
    "51:93": "left_only",
    "51:94": "left_only",
    "51:95": "left_only",
    "51:100": "both",
    "51:101": "both",
    "51:103": "right_only",
    "51:104": "right_only",
    "51:105": "right_only",
    "51:106": "right_only",
    "51:107": "right_only",
    "51:108": "right_only",
    "51:109": "right_only",
    "51:110": "right_only",
    "51:111": "right_only",
    "51:112": "right_only",
    "51:113": "right_only",
    "51:114": "right_only",
    "51:115": "right_only",
    "51:116": "right_only",
    "51:117": "right_only",
    "51:118": "right_only",
    "51:119": "right_only",
    "51:120": "right_only",
    "51:121": "right_only",
    "51:135": "left_only",
    "51:136": "left_only",
    "51:137": "left_only",
    "51:138": "left_only",
    "51:139": "left_only",
    "51:140": "left_only",
    "51:141": "left_only",
    "51:142": "left_only",
    "51:143": "left_only",
    "51:144": "left_only",
    "51:145": "left_only",
    "51:146": "left_only",
    "51:147": "left_only",
    "51:148": "left_only",
    "51:151": "both",
    "51:156": "both",
    "51:158": "right_only",
    "51:159": "right_only",
    "51:160": "right_only",
    "51:161": "right_only",
    "51:162": "right_only",
    "51:163": "right_only",
    "51:164": "right_only",
    "51:165": "right_only",
    "51:166": "right_only",
    "51:167": "right_only",
    "51:168": "right_only",
    "51:169": "right_only",
    "51:170": "right_only",
    "51:171": "right_only",
    "51:172": "right_only",
    "51:173": "right_only",
    "51:180": "both",
    "51:181": "left_only",
    "51:182": "left_only",
    "51:183": "left_only",
    "51:184": "left_only",
    "51:185": "left_only",
    "51:186": "left_only",
    "51:187": "left_only",
    "51:188": "left_only",
    "51:189": "left_only",
    "51:190": "left_only",
    "51:191": "left_only",
    "51:192": "left_only",
    "51:193": "left_only",
    "51:194": "left_only",
    "51:195": "left_only",
    "51:196": "left_only",
    "51:201": "both",
    "51:202": "both",
    "51:203": "both",
    "51:204": "both",
    "51:205": "both",
    "51:207": "right_only",
    "51:208": "right_only",
    "51:209": "right_only",
    "51:210": "right_only",
    "51:211": "right_only",
    "51:212": "right_only",
    "51:213": "right_only",
    "51:214": "right_only",
    "51:215": "right_only",
    "51:216": "right_only",
    "51:217": "right_only",
    "51:218": "right_only",
    "51:219": "right_only",
    "51:220": "right_only",
    "51:221": "right_only",
    "51:224": "both",
    "51:225": "both",
    "51:226": "both",
    "51:227": "both",
    "51:228": "both",
    "51:229": "both",
    "51:230": "both"
  },
  "method": "all eligible frames in identical three sampler candidate pools; train-only point velocity from MuJoCo body-origin Jacobian and omega cross offset; finite integrated qpos check; static geometry, no GPU policy evaluation"
}
```

### 配对6144实际起点及有限差分核验

```json
{
  "status": "passed",
  "actual_RSI_cases_checked": 6144,
  "unique_frames_checked": 2296,
  "normal_velocity_max_mps": 0.1499589254347166,
  "lifted_sole_minimum_m": 0.0019999999999999983,
  "source_counts": {
    "official_rpo": 3061,
    "g1": 3083
  },
  "tracking_source_counts": {
    "official_rpo": 1532,
    "g1": 1540
  },
  "tracking_cases": 3072,
  "finite_difference_cases": 32,
  "finite_difference_max_error_mps": 4.398409579331464e-06,
  "method": "actual paired reset-reference IDs; train-only point velocity from MuJoCo body-origin Jacobian and omega cross offset; finite integrated qpos check; static geometry, no GPU policy evaluation"
}
```

| train clip | source | 原候选 | 几何保留 | 法向速度拒绝 | 最终普通池 | 250交集 | 候选近地法向速度最大m/s |
|---|---|---|---|---|---|---|---|
| g1_dance1_subject1_clip3 | g1 | 320 | 141 | 28 | 113 | 0 | 0.149958925 |
| g1_dance1_subject2_clip3 | g1 | 207 | 8 | 7 | 1 | 0 | 0.016850794 |
| g1_dance1_subject3_clip3 | g1 | 157 | 11 | 0 | 11 | 0 | 0.089033916 |
| g1_fallAndGetUp1_subject1_clip3 | g1 | 188 | 123 | 35 | 88 | 0 | 0.14749255 |
| g1_fallAndGetUp1_subject4_clip3 | g1 | 153 | 0 | 0 | 0 | 0 | 未记录 |
| g1_fallAndGetUp1_subject5_clip3 | g1 | 130 | 8 | 3 | 5 | 0 | 0.135003433 |
| g1_fight1_subject2_clip3 | g1 | 340 | 8 | 7 | 1 | 0 | 0.109050219 |
| g1_fight1_subject3_clip3 | g1 | 337 | 0 | 0 | 0 | 0 | 未记录 |
| g1_fight1_subject5_clip3 | g1 | 376 | 16 | 8 | 8 | 0 | 0.143502481 |
| g1_jumps1_subject1_clip3 | g1 | 313 | 41 | 10 | 31 | 0 | 0.118076719 |
| g1_jumps1_subject2_clip3 | g1 | 358 | 8 | 5 | 3 | 0 | 0.145212086 |
| g1_jumps1_subject5_clip3 | g1 | 344 | 6 | 1 | 5 | 0 | 0.142466519 |
| g1_run1_subject2_clip3 | g1 | 344 | 27 | 12 | 15 | 0 | 0.140288763 |
| g1_run1_subject5_clip3 | g1 | 127 | 1 | 1 | 0 | 0 | 未记录 |
| g1_sprint1_subject2_clip3 | g1 | 306 | 168 | 12 | 156 | 0 | 0.146555543 |
| g1_sprint1_subject4_clip3 | g1 | 379 | 172 | 17 | 155 | 85 | 0.148642365 |
| g1_walk1_subject1_clip3 | g1 | 392 | 173 | 13 | 160 | 44 | 0.143033251 |
| g1_walk1_subject2_clip3 | g1 | 392 | 380 | 22 | 358 | 130 | 0.148769353 |
| g1_walk1_subject5_clip3 | g1 | 392 | 129 | 39 | 90 | 24 | 0.144634941 |
| g1_walk2_subject1_clip3 | g1 | 392 | 206 | 30 | 176 | 88 | 0.148176763 |
| g1_walk2_subject3_clip3 | g1 | 392 | 153 | 13 | 140 | 62 | 0.145671937 |
| g1_walk2_subject4_clip3 | g1 | 377 | 270 | 19 | 251 | 0 | 0.148794789 |
| rpo_114_08 | official_rpo | 435 | 23 | 10 | 13 | 11 | 0.124765245 |
| rpo_114_09 | official_rpo | 774 | 94 | 10 | 84 | 25 | 0.147189675 |
| rpo_127_04 | official_rpo | 1 | 0 | 0 | 0 | 0 | 未记录 |
| rpo_36_01 | official_rpo | 113 | 63 | 17 | 46 | 0 | 0.133361115 |
| rpo_36_11 | official_rpo | 1066 | 185 | 82 | 103 | 14 | 0.148707491 |
| rpo_B10_-__Walk_turn_left_45_stageii | official_rpo | 13 | 13 | 7 | 6 | 0 | 0.10923855 |
| rpo_B11_-__Walk_turn_left_135_stageii | official_rpo | 14 | 14 | 4 | 10 | 0 | 0.137977264 |
| rpo_B13_-__Walk_turn_right_90_stageii | official_rpo | 229 | 209 | 50 | 159 | 0 | 0.144650492 |
| rpo_B14_-__Walk_turn_right_45_t2_stageii | official_rpo | 143 | 142 | 38 | 104 | 0 | 0.148252493 |
| rpo_B15_-__Walk_turn_around_stageii | official_rpo | 7 | 7 | 2 | 5 | 0 | 0.127177602 |
| rpo_B4_-_Stand_to_Walk_backwards_stageii | official_rpo | 1 | 1 | 1 | 0 | 0 | 未记录 |
| rpo_B5_-__Walk_backwards_stageii | official_rpo | 108 | 101 | 68 | 33 | 0 | 0.141960698 |
| rpo_B9_-__Walk_turn_left_90_stageii | official_rpo | 7 | 7 | 2 | 5 | 0 | 0.127954137 |
| rpo_C12_-_run_turn_left_45_stageii | official_rpo | 11 | 2 | 2 | 0 | 0 | 未记录 |
| rpo_C17_-_run_change_direction_stageii | official_rpo | 67 | 13 | 12 | 1 | 0 | 0.0883647019 |
| rpo_C1_-_stand_to_run_stageii | official_rpo | 136 | 107 | 7 | 100 | 0 | 0.143401871 |
| rpo_C3_-_run_stageii | official_rpo | 36 | 6 | 5 | 1 | 0 | 0.0775231786 |
| rpo_C4_-_run_to_walk_a_stageii | official_rpo | 72 | 48 | 34 | 14 | 0 | 0.140659351 |
| rpo_C5_-_walk_to_run_stageii | official_rpo | 47 | 25 | 16 | 9 | 0 | 0.143967899 |
| rpo_move_back | official_rpo | 304 | 298 | 27 | 271 | 62 | 0.147908109 |
| rpo_move_l | official_rpo | 272 | 111 | 14 | 97 | 30 | 0.134004326 |
| rpo_move_r | official_rpo | 242 | 128 | 15 | 113 | 0 | 0.148762856 |
| rpo_run_start_180_R_001__A345 | official_rpo | 114 | 5 | 2 | 3 | 0 | 0.134701709 |
| rpo_run_start_180_R_001__A345_M | official_rpo | 115 | 6 | 2 | 4 | 0 | 0.137906609 |
| rpo_run_stop_180_R_001__A345 | official_rpo | 158 | 112 | 2 | 110 | 0 | 0.108009086 |
| rpo_run_stop_180_R_001__A345_M | official_rpo | 158 | 112 | 2 | 110 | 0 | 0.112880113 |
| rpo_turn_l | official_rpo | 332 | 142 | 15 | 127 | 73 | 0.138375503 |
| rpo_turn_r | official_rpo | 231 | 167 | 20 | 147 | 0 | 0.131780601 |

三卡候选清单完全相同；每card固定无放回seed分区hash与policy一致。两来源的tracking候选均保留左/右单足支撑和双支撑；G1 walk1/walk2等真实行走片段仍有合法250窗口。足点几何核验是CPU运动学诊断，不能代替PhysX动态可行性或策略验收。

### 正式三卡实际DR质量、体重及gain范围

```json
{
  "status": "passed",
  "records": [
    {
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

### 三卡地面覆盖回执

```json
[
  {
    "ground_profile": "wide_plane_v9",
    "plane_size_m": [
      512.0,
      512.0
    ],
    "num_envs": 2048,
    "origin_min_xy_m": [
      -56.25,
      -55.0
    ],
    "origin_max_xy_m": [
      56.25,
      55.0
    ],
    "reference_foot_bounds_xy_m": [
      [
        -6.096752880722514,
        -3.840407401985619
      ],
      [
        4.933901835662462,
        5.5889945061562365
      ]
    ],
    "combined_foot_bounds_xy_m": [
      [
        -62.34675288072251,
        -58.840407401985615
      ],
      [
        61.18390183566246,
        60.58899450615624
      ]
    ],
    "minimum_margin_m": 193.6532471192775,
    "required_motion_reserve_m": 20.0
  },
  {
    "ground_profile": "wide_plane_v9",
    "plane_size_m": [
      512.0,
      512.0
    ],
    "num_envs": 2048,
    "origin_min_xy_m": [
      -56.25,
      -55.0
    ],
    "origin_max_xy_m": [
      56.25,
      55.0
    ],
    "reference_foot_bounds_xy_m": [
      [
        -6.096752880722514,
        -3.840407401985619
      ],
      [
        4.933901835662462,
        5.5889945061562365
      ]
    ],
    "combined_foot_bounds_xy_m": [
      [
        -62.34675288072251,
        -58.840407401985615
      ],
      [
        61.18390183566246,
        60.58899450615624
      ]
    ],
    "minimum_margin_m": 193.6532471192775,
    "required_motion_reserve_m": 20.0
  },
  {
    "ground_profile": "wide_plane_v9",
    "plane_size_m": [
      512.0,
      512.0
    ],
    "num_envs": 2048,
    "origin_min_xy_m": [
      -56.25,
      -55.0
    ],
    "origin_max_xy_m": [
      56.25,
      55.0
    ],
    "reference_foot_bounds_xy_m": [
      [
        -6.096752880722514,
        -3.840407401985619
      ],
      [
        4.933901835662462,
        5.5889945061562365
      ]
    ],
    "combined_foot_bounds_xy_m": [
      [
        -62.34675288072251,
        -58.840407401985615
      ],
      [
        61.18390183566246,
        60.58899450615624
      ]
    ],
    "minimum_margin_m": 193.6532471192775,
    "required_motion_reserve_m": 20.0
  }
]
```

## 6. 500更新结果、分母与段间决策

| 指标 | 25–124 | 125–224 | 225–324 | 325–424 | 425–524 | 末点 |
|---|---|---|---|---|---|---|
| env/episode_len | 8.25451386 | 11.558163 | 15.1944955 | 19.2216685 | 22.3133605 | 22.0957165 |
| train/ufo/strict_shadow_episode_len | 7.36798751 | 9.21281242 | 11.7758474 | 14.7346755 | 16.0141411 | 15.6891518 |
| train/ufo/strict_shadow_tracking_episode_len | 7.423129 | 8.83360926 | 12.7018119 | 18.1704281 | 19.1313705 | 17.4733334 |
| train/ufo/strict_shadow_mixed_episode_len | 7.30444562 | 9.60746924 | 11.1257958 | 12.5257553 | 13.9167568 | 14.4482222 |
| train/ufo/strict_velocity_exposure_fraction | 0.00030569178 | 0.000378594879 | 0.000309119476 | 0.000268655779 | 0.000256650934 | 0.000255197723 |
| train/ufo/strict_velocity_transition_fraction | 0.0346883135 | 0.0422973627 | 0.0340169271 | 0.0303722561 | 0.029035296 | 0.0277913418 |
| train/ufo/training_velocity_count | 86.5 | 117.333333 | 86.5 | 71.8571429 | 58.2857143 | 57 |
| train/ufo/hard_position_count | 791.5 | 507.166667 | 438.833333 | 396.142857 | 354.714286 | 324 |
| train/ufo/hard_velocity_count | 852.5 | 1039.5 | 836 | 746.428571 | 713.571429 | 683 |
| train/ufo/physics_peak_normalized_velocity | 3.86503045 | 4.34970148 | 4.86682391 | 4.51164464 | 3.60990855 | 4.99384022 |
| train/ufo/tracking_q_rmse_rad | 0.297732711 | 0.36846746 | 0.38922449 | 0.42860353 | 0.42224626 | 0.425170273 |
| train/ufo/applied_action_saturation_fraction | 0.177575447 | 0.264612471 | 0.293046887 | 0.292408692 | 0.26444625 | 0.245350718 |
| train/ufo/critic_loss | 0.80393427 | 0.305526901 | 0.300615316 | 0.336857341 | 0.441098871 | 0.534224987 |
| train/ufo/Q_fb | 6.17103572 | 16.5273102 | 27.0270914 | 38.958132 | 52.201909 | 59.6927032 |
| train/ufo/mean_disc_reward | 0.0341719261 | -0.150872847 | -0.506376803 | -0.772451615 | -0.954529793 | -1.00685167 |

窗口按日志采样点等权，episode_len/strict_shadow每round记录、loss每联合更新记录；各窗口n在全统计表明确。episode_len与strict_shadow均针对已结束episode，早期短段存在右删失；尤其24资格物理只采4步，其2.88平均不能与长段直接判退化。末段未结束episode单列，不补造其最终长度。

预冻结决策检查：正向继续候选=False（要求末两严格均≥50且训练/严格较首两≥20%改善、速度与首步不恶化）；策略退化条件=False（严格均下降≥20%或暴露反弹≥25%）；独立评估门槛=False（末两训练及严格均≥100、首步≤.1%、无原位置/力矩/地面违规、全部子步速度原额定内）。无论loss如何，未满足该门槛不运行独立评估。

首步力诊断目标<2500N，实际最大=4507.1709N；首步终止目标≤.1%，实际=0.088034627%。这两项和后续落足超速分别判断，不能从RSI静态过滤通过推出接触脉冲已解决。原速度shadow与1.3×训练真实flags分列，位置/速度/接触原因允许重叠，不能直接相加得到terminated总数。

虽没有触发预声明晚期坍塌条件，末两严格影子均值仍只有14.7346755 / 16.0141411步，未达到下一段正向候选要求的50步。tracking q RMSE窗口0.297732711→0.42224626rad；更长episode下的参考误差仍大，不能以context恢复正确宣称步态跟踪已学会。Critic后程约0.441098871，未复现先前18量级发散；Q_fb仍增长、D奖励转负，保留监测，不从500点直接断言长期稳定。

首步48个真实terminated按来源/mode：G1 mixed45、official mixed3、两tracking组0。总体首步门槛虽达标，G1 mixed子组仍高于总体，候选覆盖/重置接触仍需关注；不同mode起点池本身不同，这不是tracking机制的纯因果收益。原速度flags26588/786432=3.38083903%转移，按八子步×23joint暴露比例则=0.0301174496%，两个分母不能互换。

| RSI来源/模式/候选支撑相位 | 转移数 | reset数 | 首步terminated | 完成episode数 | 训练mean/median步 | 严格mean/median步 |
|---|---|---|---|---|---|---|
| official_rpo/mixed/left_only | 57522 | 4109 | 1 | 3708 | 14.2971953 / 12 | 10.9290723 / 10 |
| official_rpo/mixed/right_only | 74318 | 5342 | 1 | 4808 | 14.0661398 / 12 | 10.9977121 / 10 |
| official_rpo/mixed/both | 74078 | 4986 | 1 | 4478 | 14.9933006 / 13 | 11.800134 / 10 |
| official_rpo/tracking/left_only | 44175 | 2730 | 0 | 2435 | 16.3979466 / 14 | 13.0390144 / 11 |
| official_rpo/tracking/right_only | 48568 | 2842 | 0 | 2540 | 17.3964567 / 16 | 12.9858268 / 10 |
| official_rpo/tracking/both | 130751 | 7196 | 0 | 6281 | 18.3150772 / 16 | 13.5567585 / 10 |
| g1/mixed/left_only | 67362 | 5342 | 3 | 4817 | 12.5461906 / 10 | 9.91986714 / 8 |
| g1/mixed/right_only | 62488 | 4993 | 42 | 4490 | 12.3685969 / 10 | 9.85434298 / 8 |
| g1/mixed/both | 57448 | 4320 | 0 | 3886 | 13.388317 / 11 | 10.8808543 / 9 |
| g1/tracking/left_only | 79279 | 5850 | 0 | 5206 | 13.2118709 / 11 | 10.7086055 / 8 |
| g1/tracking/right_only | 26536 | 2215 | 0 | 2009 | 11.6505724 / 8 | 8.60129418 / 6 |
| g1/tracking/both | 63907 | 4599 | 0 | 4029 | 13.2670638 / 9 | 10.9918094 / 7 |

支撑相位按实际RSI reference frame的名义MJCF碰撞hull、lift后20mm近地条件归类left_only/right_only/both；不是把reset尚未开始积分时的净力读数当接触相位。分组分位数均为已结束episode，段末未结束轨迹在下方单列右删失。

| joint | 原额定rad/s | 物理子步abs峰rad/s | 控制末abs峰rad/s | 峰/额定 | 子步暴露比例 |
|---|---|---|---|---|---|
| left_thigh_yaw_joint | 25 | 28.3955002 | 25.0000114 | 1.13582001 | 3.17891439e-07 |
| right_thigh_yaw_joint | 25 | 27.0865898 | 25.0029373 | 1.08346359 | 4.76837158e-07 |
| torso_joint | 25 | 25.0019875 | 25.0007038 | 1.0000795 | 3.17891439e-07 |
| left_thigh_roll_joint | 25 | 29.8551712 | 25.0000248 | 1.19420685 | 3.17891439e-07 |
| right_thigh_roll_joint | 25 | 26.5803738 | 25.0105343 | 1.06321495 | 1.58945719e-06 |
| left_arm_pitch_joint | 8 | 9.03009224 | 8.2750349 | 1.12876153 | 9.44137573e-05 |
| right_arm_pitch_joint | 8 | 11.0758009 | 11.0758009 | 1.38447511 | 0.00014034907 |
| left_thigh_pitch_joint | 25 | 25.0662727 | 25.0000095 | 1.00265091 | 1.58945719e-07 |
| right_thigh_pitch_joint | 25 | 25.0072708 | 25.0000114 | 1.00029083 | 1.58945719e-07 |
| left_arm_roll_joint | 8 | 8.4305563 | 8.08499908 | 1.05381954 | 7.48634338e-05 |
| right_arm_roll_joint | 8 | 12.4181013 | 12.4181013 | 1.55226266 | 5.91278076e-05 |
| left_knee_joint | 25 | 53.0160332 | 34.8754807 | 2.12064133 | 1.49408976e-05 |
| right_knee_joint | 25 | 36.1847687 | 34.759285 | 1.44739075 | 5.08626302e-06 |
| left_arm_yaw_joint | 8 | 8.18931198 | 8.03083038 | 1.023664 | 9.8546346e-06 |
| right_arm_yaw_joint | 8 | 8.0905962 | 8.06810379 | 1.01132452 | 2.3206075e-05 |
| left_ankle_pitch_joint | 8 | 55.7631607 | 52.1282806 | 6.97039509 | 0.00271193186 |
| right_ankle_pitch_joint | 8 | 44.5956001 | 36.2061882 | 5.57445002 | 0.00230693817 |
| left_elbow_pitch_joint | 8 | 8.84207916 | 8.08810329 | 1.1052599 | 1.90734863e-05 |
| right_elbow_pitch_joint | 8 | 9.69549084 | 8.05771255 | 1.21193635 | 5.69025675e-05 |
| left_ankle_roll_joint | 8 | 53.0206833 | 29.4530926 | 6.62758541 | 0.000825723012 |
| right_ankle_roll_joint | 8 | 52.809845 | 22.2354259 | 6.60123062 | 0.000581264496 |
| left_elbow_yaw_joint | 8 | 4.72797775 | 4.20682812 | 0.590997219 | 0 |
| right_elbow_yaw_joint | 8 | 4.61157274 | 4.20469904 | 0.576446593 | 0 |

### 正式786432转移逐行trace/replay/context/删失核验

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
  "requested_saturation_fraction": 0.5942724476689878,
  "applied_saturation_fraction": 0.2596685437188632,
  "strict_position_velocity_torque_counts": [
    15681,
    26588,
    0
  ],
  "training_position_velocity_torque_counts": [
    15681,
    2653,
    0
  ],
  "macro_contact_height_tilt_counts": [
    7002,
    11654,
    15126
  ],
  "boundary_max_radps": 52.12828063964844,
  "physics_max_radps": 55.763160705566406,
  "physics_max_rated_ratio": 6.970395088195801,
  "boundary_max_event": {
    "round": 17,
    "row": 20155,
    "episode_id_run_local": 28907143173,
    "episode_step": 10,
    "joint": "left_ankle_pitch_joint",
    "rated_radps": 8.0,
    "boundary_dq_radps": 52.12828063964844,
    "physics_signed_dq_radps": 52.12828063964844,
    "physics_abs_dq_radps": 52.12828063964844,
    "peak_substep_0_to_7": 7,
    "q_at_joint_peak_rad": 0.3390082120895386,
    "torque_at_joint_peak_Nm": -0.6712460517883301,
    "target_at_joint_peak_rad": 0.5100952982902527,
    "requested_action": 0.9999949932098389,
    "applied_action": 0.8501588106155396,
    "foot_peak_vertical_net_force_N": [
      12813.30859375,
      0.0
    ],
    "foot_vertical_net_force_integral_Ns": [
      32.03327178955078,
      0.0
    ],
    "global_peak_joint": "left_ankle_pitch_joint",
    "feet_at_global_abs_dq_peak_origin_pose_wxyz_linear_angular": [
      [
        -44.196990966796875,
        -20.364551544189453,
        0.055626560002565384,
        -0.9547846913337708,
        0.19040179252624512,
        0.022335899993777275,
        0.22723230719566345,
        -6.4749226570129395,
        2.6953892707824707,
        -3.275404930114746,
        44.51498031616211,
        67.63995361328125,
        6.6077728271484375
      ],
      [
        -44.81241989135742,
        -20.278244018554688,
        0.3404448628425598,
        -0.5758089423179626,
        -0.3148742914199829,
        -0.7011622190475464,
        0.2786928713321686,
        -1.2131296396255493,
        0.6931236982345581,
        5.162918567657471,
        8.950199127197266,
        14.054827690124512,
        2.0610337257385254
      ]
    ],
    "feet_vertical_net_force_at_global_peak_N": [
      12813.30859375,
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
    "round": 28,
    "row": 16937,
    "episode_id_run_local": 1108789428235,
    "episode_step": 2,
    "joint": "left_ankle_pitch_joint",
    "rated_radps": 8.0,
    "boundary_dq_radps": 2.850525379180908,
    "physics_signed_dq_radps": 55.763160705566406,
    "physics_abs_dq_radps": 55.763160705566406,
    "peak_substep_0_to_7": 6,
    "q_at_joint_peak_rad": -0.3544588088989258,
    "torque_at_joint_peak_Nm": 0.9681973457336426,
    "target_at_joint_peak_rad": -0.48245251178741455,
    "requested_action": -0.9999338984489441,
    "applied_action": -0.8040875196456909,
    "foot_peak_vertical_net_force_N": [
      9074.416015625,
      0.0
    ],
    "foot_vertical_net_force_integral_Ns": [
      22.686038970947266,
      0.0
    ],
    "global_peak_joint": "left_ankle_pitch_joint",
    "feet_at_global_abs_dq_peak_origin_pose_wxyz_linear_angular": [
      [
        26.0567684173584,
        -21.52837371826172,
        0.0681837871670723,
        0.7694054245948792,
        0.33010393381118774,
        -0.17686912417411804,
        0.5174592137336731,
        2.1087331771850586,
        -7.184350967407227,
        -3.3675625324249268,
        -99.44052124023438,
        -22.316295623779297,
        -9.055086135864258
      ],
      [
        26.325834274291992,
        -21.951431274414062,
        0.055551670491695404,
        0.644013524055481,
        -0.05907154083251953,
        -0.07780177146196365,
        0.7587516903877258,
        1.4337745904922485,
        0.8872319459915161,
        0.2830924987792969,
        0.42903807759284973,
        -7.213277816772461,
        -5.401115894317627
      ]
    ],
    "feet_vertical_net_force_at_global_peak_N": [
      9074.416015625,
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
  "normalized_max_event": {
    "round": 28,
    "row": 16937,
    "episode_id_run_local": 1108789428235,
    "episode_step": 2,
    "joint": "left_ankle_pitch_joint",
    "rated_radps": 8.0,
    "boundary_dq_radps": 2.850525379180908,
    "physics_signed_dq_radps": 55.763160705566406,
    "physics_abs_dq_radps": 55.763160705566406,
    "peak_substep_0_to_7": 6,
    "q_at_joint_peak_rad": -0.3544588088989258,
    "torque_at_joint_peak_Nm": 0.9681973457336426,
    "target_at_joint_peak_rad": -0.48245251178741455,
    "requested_action": -0.9999338984489441,
    "applied_action": -0.8040875196456909,
    "foot_peak_vertical_net_force_N": [
      9074.416015625,
      0.0
    ],
    "foot_vertical_net_force_integral_Ns": [
      22.686038970947266,
      0.0
    ],
    "global_peak_joint": "left_ankle_pitch_joint",
    "feet_at_global_abs_dq_peak_origin_pose_wxyz_linear_angular": [
      [
        26.0567684173584,
        -21.52837371826172,
        0.0681837871670723,
        0.7694054245948792,
        0.33010393381118774,
        -0.17686912417411804,
        0.5174592137336731,
        2.1087331771850586,
        -7.184350967407227,
        -3.3675625324249268,
        -99.44052124023438,
        -22.316295623779297,
        -9.055086135864258
      ],
      [
        26.325834274291992,
        -21.951431274414062,
        0.055551670491695404,
        0.644013524055481,
        -0.05907154083251953,
        -0.07780177146196365,
        0.7587516903877258,
        1.4337745904922485,
        0.8872319459915161,
        0.2830924987792969,
        0.42903807759284973,
        -7.213277816772461,
        -5.401115894317627
      ]
    ],
    "feet_vertical_net_force_at_global_peak_N": [
      9074.416015625,
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
  "foot_vertical_net_force_peak_N": 20959.25,
  "first_step_foot_vertical_net_force_N": {
    "n": 54524,
    "mean": 74.19508361816406,
    "median": 0.0,
    "p95": 456.67826690673814,
    "p99": 1061.8408020019522,
    "max": 4507.1708984375
  },
  "first_step_foot_absolute_vertical_net_force_integral_Ns": {
    "n": 54524,
    "mean": 0.4275542199611664,
    "median": 0.0,
    "p95": 3.076720654964447,
    "p99": 5.531707286834692,
    "max": 13.176518440246582
  },
  "reset_lift_m": {
    "n": 54524,
    "mean": 0.0014107656897976995,
    "median": 0.0,
    "p95": 0.013837347738444805,
    "p99": 0.019476622343063354,
    "max": 0.021992022171616554
  },
  "resets": 54524,
  "first_step_terminated": 48,
  "first_step_termination_fraction": 0.0008803462695326829,
  "first_step_strict_counts": [
    1,
    39,
    0
  ],
  "first_step_macro_counts": [
    42,
    0,
    0
  ],
  "raw_velocity_penalty_mean": 0.01386154528396825,
  "weighted_velocity_penalty_mean": -0.34653863209920627,
  "strict_velocity_substep_joint_exposure_fraction": 0.00030117449553116507,
  "velocity_reward_coverage_miss_count": 0,
  "latched_velocity_terms_without_boundary_violation": 22086,
  "reset_q_preservation_max_error_rad": 8.821487429422348e-08,
  "reset_dq_preservation_max_error_radps": 0.0,
  "decoder_max_error_rad": 2.8312206268310547e-07,
  "ground_minimum_margin_m": 193.23008728027344,
  "ground_outside_count": 0,
  "mixed_nonperiodic_z_equal_pairs": 364124,
  "mixed_nonperiodic_cosine": {
    "n": 364124,
    "mean": 1.0,
    "median": 1.0,
    "p95": 1.0000001192092896,
    "p99": 1.0000001192092896,
    "max": 1.0000001192092896
  },
  "tracking_unexpected_refreshes": 0,
  "tracking_phase_mismatches": 0,
  "tracking_window_stable_pairs": 367784,
  "tracking_adjacent_phase_cosine": {
    "n": 367784,
    "mean": 0.9978622794151306,
    "median": 0.9981222748756409,
    "p95": 0.9999926686286926,
    "p99": 0.9999972581863403,
    "max": 0.9999991059303284
  },
  "tracking_window_renewals": 0,
  "terminal_following_new_episode_pairs": 48380,
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
    "n": 48687,
    "mean": 14.389344178117362,
    "median": 12.0,
    "p95": 33.0,
    "p99": 45.0,
    "max": 75.0
  },
  "completed_strict_shadow_len": {
    "n": 48687,
    "mean": 11.250539158296876,
    "median": 10.0,
    "p95": 29.0,
    "p99": 39.0,
    "max": 75.0
  },
  "completed_tracking_shadow_len": {
    "n": 22500,
    "mean": 11.875511111111111,
    "median": 10.0,
    "p95": 31.0,
    "p99": 42.0,
    "max": 75.0
  },
  "completed_mixed_shadow_len": {
    "n": 26187,
    "mean": 10.713560163439874,
    "median": 9.0,
    "p95": 27.0,
    "p99": 36.0,
    "max": 75.0
  },
  "unfinished_train_episodes_right_censored": 5837,
  "unfinished_strict_shadow_right_censored": 4054,
  "unfinished_train_observed_lengths": {
    "n": 5837,
    "mean": 14.709268459825253,
    "median": 13.0,
    "p95": 34.0,
    "p99": 49.0,
    "max": 75.0
  },
  "RSI_source_by_rollout_mode": {
    "official_rpo/mixed": {
      "rows": 205918,
      "resets": 14437,
      "first_terms": 3,
      "strict_velocity_rows": 7692,
      "completed_episode_len": {
        "n": 12994,
        "mean": 14.451593042942896,
        "median": 12.0,
        "p95": 30.0,
        "p99": 40.0,
        "max": 75.0
      },
      "completed_strict_shadow_len": {
        "n": 12994,
        "mean": 11.25465599507465,
        "median": 10.0,
        "p95": 27.0,
        "p99": 36.0,
        "max": 75.0
      },
      "first_term_fraction": 0.00020779940430837432,
      "strict_velocity_substep_joint_exposure_fraction": 0.00037280035911297743,
      "raw_velocity_penalty_mean": 0.016741627884255705
    },
    "official_rpo/tracking": {
      "rows": 223494,
      "resets": 12768,
      "first_terms": 0,
      "strict_velocity_rows": 7253,
      "completed_episode_len": {
        "n": 11256,
        "mean": 17.693052594172,
        "median": 15.0,
        "p95": 37.0,
        "p99": 47.0,
        "max": 69.0
      },
      "completed_strict_shadow_len": {
        "n": 11256,
        "mean": 13.31592039800995,
        "median": 11.0,
        "p95": 32.0,
        "p99": 41.0,
        "max": 69.0
      },
      "first_term_fraction": 0.0,
      "strict_velocity_substep_joint_exposure_fraction": 0.00028436713212026703,
      "raw_velocity_penalty_mean": 0.01630766872849888
    },
    "g1/mixed": {
      "rows": 187298,
      "resets": 14655,
      "first_terms": 45,
      "strict_velocity_rows": 6051,
      "completed_episode_len": {
        "n": 13193,
        "mean": 12.733798226332146,
        "median": 10.0,
        "p95": 30.0,
        "p99": 41.0,
        "max": 69.0
      },
      "completed_strict_shadow_len": {
        "n": 13193,
        "mean": 10.180626089592966,
        "median": 8.0,
        "p95": 27.0,
        "p99": 37.0,
        "max": 62.0
      },
      "first_term_fraction": 0.0030706243602865915,
      "strict_velocity_substep_joint_exposure_fraction": 0.0002788221234981501,
      "raw_velocity_penalty_mean": 0.011965788848263988
    },
    "g1/tracking": {
      "rows": 169722,
      "resets": 12664,
      "first_terms": 0,
      "strict_velocity_rows": 5592,
      "completed_episode_len": {
        "n": 11244,
        "mean": 12.952685876912131,
        "median": 10.0,
        "p95": 34.0,
        "p99": 47.0,
        "max": 75.0
      },
      "completed_strict_shadow_len": {
        "n": 11244,
        "mean": 10.433564567769476,
        "median": 7.0,
        "p95": 29.0,
        "p99": 42.0,
        "max": 75.0
      },
      "first_term_fraction": 0.0,
      "strict_velocity_substep_joint_exposure_fraction": 0.00026107271071926827,
      "raw_velocity_penalty_mean": 0.00923820015110317
    }
  },
  "RSI_source_mode_by_nominal_reset_support_phase": {
    "official_rpo/mixed/left_only": {
      "rows": 57522,
      "resets": 4109,
      "first_terms": 1,
      "completed_episode_len": {
        "n": 3708,
        "mean": 14.297195253505933,
        "median": 12.0,
        "p95": 30.0,
        "p99": 40.929999999999836,
        "max": 73.0
      },
      "completed_strict_shadow_len": {
        "n": 3708,
        "mean": 10.929072276159655,
        "median": 10.0,
        "p95": 26.0,
        "p99": 36.0,
        "max": 60.0
      },
      "first_term_fraction": 0.0002433682161109759
    },
    "official_rpo/mixed/right_only": {
      "rows": 74318,
      "resets": 5342,
      "first_terms": 1,
      "completed_episode_len": {
        "n": 4808,
        "mean": 14.066139767054908,
        "median": 12.0,
        "p95": 29.0,
        "p99": 40.0,
        "max": 75.0
      },
      "completed_strict_shadow_len": {
        "n": 4808,
        "mean": 10.997712146422629,
        "median": 10.0,
        "p95": 25.0,
        "p99": 34.0,
        "max": 75.0
      },
      "first_term_fraction": 0.00018719580681392738
    },
    "official_rpo/mixed/both": {
      "rows": 74078,
      "resets": 4986,
      "first_terms": 1,
      "completed_episode_len": {
        "n": 4478,
        "mean": 14.993300580616346,
        "median": 13.0,
        "p95": 32.0,
        "p99": 41.0,
        "max": 60.0
      },
      "completed_strict_shadow_len": {
        "n": 4478,
        "mean": 11.800133988387673,
        "median": 10.0,
        "p95": 28.0,
        "p99": 38.0,
        "max": 56.0
      },
      "first_term_fraction": 0.00020056157240272763
    },
    "official_rpo/tracking/left_only": {
      "rows": 44175,
      "resets": 2730,
      "first_terms": 0,
      "completed_episode_len": {
        "n": 2435,
        "mean": 16.39794661190965,
        "median": 14.0,
        "p95": 34.0,
        "p99": 40.659999999999854,
        "max": 62.0
      },
      "completed_strict_shadow_len": {
        "n": 2435,
        "mean": 13.039014373716633,
        "median": 11.0,
        "p95": 31.299999999999727,
        "p99": 36.0,
        "max": 62.0
      },
      "first_term_fraction": 0.0
    },
    "official_rpo/tracking/right_only": {
      "rows": 48568,
      "resets": 2842,
      "first_terms": 0,
      "completed_episode_len": {
        "n": 2540,
        "mean": 17.396456692913386,
        "median": 16.0,
        "p95": 34.0,
        "p99": 46.0,
        "max": 65.0
      },
      "completed_strict_shadow_len": {
        "n": 2540,
        "mean": 12.985826771653544,
        "median": 10.0,
        "p95": 30.0,
        "p99": 39.61000000000013,
        "max": 61.0
      },
      "first_term_fraction": 0.0
    },
    "official_rpo/tracking/both": {
      "rows": 130751,
      "resets": 7196,
      "first_terms": 0,
      "completed_episode_len": {
        "n": 6281,
        "mean": 18.31507721700366,
        "median": 16.0,
        "p95": 39.0,
        "p99": 50.0,
        "max": 69.0
      },
      "completed_strict_shadow_len": {
        "n": 6281,
        "mean": 13.556758477949371,
        "median": 10.0,
        "p95": 34.0,
        "p99": 44.0,
        "max": 69.0
      },
      "first_term_fraction": 0.0
    },
    "g1/mixed/left_only": {
      "rows": 67362,
      "resets": 5342,
      "first_terms": 3,
      "completed_episode_len": {
        "n": 4817,
        "mean": 12.54619057504671,
        "median": 10.0,
        "p95": 30.0,
        "p99": 41.0,
        "max": 69.0
      },
      "completed_strict_shadow_len": {
        "n": 4817,
        "mean": 9.919867137222338,
        "median": 8.0,
        "p95": 25.0,
        "p99": 35.0,
        "max": 57.0
      },
      "first_term_fraction": 0.0005615874204417821
    },
    "g1/mixed/right_only": {
      "rows": 62488,
      "resets": 4993,
      "first_terms": 42,
      "completed_episode_len": {
        "n": 4490,
        "mean": 12.368596881959911,
        "median": 10.0,
        "p95": 30.0,
        "p99": 41.0,
        "max": 62.0
      },
      "completed_strict_shadow_len": {
        "n": 4490,
        "mean": 9.854342984409799,
        "median": 8.0,
        "p95": 26.0,
        "p99": 36.0,
        "max": 62.0
      },
      "first_term_fraction": 0.008411776487081914
    },
    "g1/mixed/both": {
      "rows": 57448,
      "resets": 4320,
      "first_terms": 0,
      "completed_episode_len": {
        "n": 3886,
        "mean": 13.388317035512095,
        "median": 11.0,
        "p95": 32.0,
        "p99": 41.15000000000009,
        "max": 65.0
      },
      "completed_strict_shadow_len": {
        "n": 3886,
        "mean": 10.880854348944931,
        "median": 9.0,
        "p95": 29.0,
        "p99": 38.0,
        "max": 61.0
      },
      "first_term_fraction": 0.0
    },
    "g1/tracking/left_only": {
      "rows": 79279,
      "resets": 5850,
      "first_terms": 0,
      "completed_episode_len": {
        "n": 5206,
        "mean": 13.211870918171341,
        "median": 11.0,
        "p95": 33.0,
        "p99": 45.0,
        "max": 72.0
      },
      "completed_strict_shadow_len": {
        "n": 5206,
        "mean": 10.70860545524395,
        "median": 8.0,
        "p95": 29.0,
        "p99": 40.0,
        "max": 69.0
      },
      "first_term_fraction": 0.0
    },
    "g1/tracking/right_only": {
      "rows": 26536,
      "resets": 2215,
      "first_terms": 0,
      "completed_episode_len": {
        "n": 2009,
        "mean": 11.650572424091587,
        "median": 8.0,
        "p95": 30.0,
        "p99": 40.0,
        "max": 57.0
      },
      "completed_strict_shadow_len": {
        "n": 2009,
        "mean": 8.601294176207068,
        "median": 6.0,
        "p95": 24.0,
        "p99": 35.0,
        "max": 53.0
      },
      "first_term_fraction": 0.0
    },
    "g1/tracking/both": {
      "rows": 63907,
      "resets": 4599,
      "first_terms": 0,
      "completed_episode_len": {
        "n": 4029,
        "mean": 13.267063787540332,
        "median": 9.0,
        "p95": 37.0,
        "p99": 51.0,
        "max": 75.0
      },
      "completed_strict_shadow_len": {
        "n": 4029,
        "mean": 10.99180938198064,
        "median": 7.0,
        "p95": 31.0,
        "p99": 47.0,
        "max": 75.0
      },
      "first_term_fraction": 0.0
    }
  },
  "per_joint_physical_diagnostics": {
    "left_thigh_yaw_joint": {
      "rated_radps": 25.0,
      "max_physics_abs_radps": 28.39550018310547,
      "max_boundary_abs_radps": 25.000011444091797,
      "max_rated_ratio": 1.1358200073242188,
      "substep_exposure_fraction": 3.178914388020833e-07
    },
    "right_thigh_yaw_joint": {
      "rated_radps": 25.0,
      "max_physics_abs_radps": 27.086589813232422,
      "max_boundary_abs_radps": 25.00293731689453,
      "max_rated_ratio": 1.0834635925292968,
      "substep_exposure_fraction": 4.76837158203125e-07
    },
    "torso_joint": {
      "rated_radps": 25.0,
      "max_physics_abs_radps": 25.00198745727539,
      "max_boundary_abs_radps": 25.000703811645508,
      "max_rated_ratio": 1.0000794982910157,
      "substep_exposure_fraction": 3.178914388020833e-07
    },
    "left_thigh_roll_joint": {
      "rated_radps": 25.0,
      "max_physics_abs_radps": 29.85517120361328,
      "max_boundary_abs_radps": 25.000024795532227,
      "max_rated_ratio": 1.1942068481445312,
      "substep_exposure_fraction": 3.178914388020833e-07
    },
    "right_thigh_roll_joint": {
      "rated_radps": 25.0,
      "max_physics_abs_radps": 26.580373764038086,
      "max_boundary_abs_radps": 25.010534286499023,
      "max_rated_ratio": 1.0632149505615234,
      "substep_exposure_fraction": 1.5894571940104167e-06
    },
    "left_arm_pitch_joint": {
      "rated_radps": 8.0,
      "max_physics_abs_radps": 9.030092239379883,
      "max_boundary_abs_radps": 8.27503490447998,
      "max_rated_ratio": 1.1287615299224854,
      "substep_exposure_fraction": 9.441375732421875e-05
    },
    "right_arm_pitch_joint": {
      "rated_radps": 8.0,
      "max_physics_abs_radps": 11.075800895690918,
      "max_boundary_abs_radps": 11.075800895690918,
      "max_rated_ratio": 1.3844751119613647,
      "substep_exposure_fraction": 0.00014034907023111978
    },
    "left_thigh_pitch_joint": {
      "rated_radps": 25.0,
      "max_physics_abs_radps": 25.066272735595703,
      "max_boundary_abs_radps": 25.000009536743164,
      "max_rated_ratio": 1.002650909423828,
      "substep_exposure_fraction": 1.5894571940104166e-07
    },
    "right_thigh_pitch_joint": {
      "rated_radps": 25.0,
      "max_physics_abs_radps": 25.00727081298828,
      "max_boundary_abs_radps": 25.000011444091797,
      "max_rated_ratio": 1.0002908325195312,
      "substep_exposure_fraction": 1.5894571940104166e-07
    },
    "left_arm_roll_joint": {
      "rated_radps": 8.0,
      "max_physics_abs_radps": 8.430556297302246,
      "max_boundary_abs_radps": 8.084999084472656,
      "max_rated_ratio": 1.0538195371627808,
      "substep_exposure_fraction": 7.486343383789062e-05
    },
    "right_arm_roll_joint": {
      "rated_radps": 8.0,
      "max_physics_abs_radps": 12.41810131072998,
      "max_boundary_abs_radps": 12.41810131072998,
      "max_rated_ratio": 1.5522626638412476,
      "substep_exposure_fraction": 5.91278076171875e-05
    },
    "left_knee_joint": {
      "rated_radps": 25.0,
      "max_physics_abs_radps": 53.01603317260742,
      "max_boundary_abs_radps": 34.87548065185547,
      "max_rated_ratio": 2.120641326904297,
      "substep_exposure_fraction": 1.4940897623697916e-05
    },
    "right_knee_joint": {
      "rated_radps": 25.0,
      "max_physics_abs_radps": 36.18476867675781,
      "max_boundary_abs_radps": 34.75928497314453,
      "max_rated_ratio": 1.4473907470703125,
      "substep_exposure_fraction": 5.086263020833333e-06
    },
    "left_arm_yaw_joint": {
      "rated_radps": 8.0,
      "max_physics_abs_radps": 8.189311981201172,
      "max_boundary_abs_radps": 8.030830383300781,
      "max_rated_ratio": 1.0236639976501465,
      "substep_exposure_fraction": 9.854634602864584e-06
    },
    "right_arm_yaw_joint": {
      "rated_radps": 8.0,
      "max_physics_abs_radps": 8.090596199035645,
      "max_boundary_abs_radps": 8.068103790283203,
      "max_rated_ratio": 1.0113245248794556,
      "substep_exposure_fraction": 2.3206075032552082e-05
    },
    "left_ankle_pitch_joint": {
      "rated_radps": 8.0,
      "max_physics_abs_radps": 55.763160705566406,
      "max_boundary_abs_radps": 52.12828063964844,
      "max_rated_ratio": 6.970395088195801,
      "substep_exposure_fraction": 0.0027119318644205728
    },
    "right_ankle_pitch_joint": {
      "rated_radps": 8.0,
      "max_physics_abs_radps": 44.59560012817383,
      "max_boundary_abs_radps": 36.2061882019043,
      "max_rated_ratio": 5.5744500160217285,
      "substep_exposure_fraction": 0.0023069381713867188
    },
    "left_elbow_pitch_joint": {
      "rated_radps": 8.0,
      "max_physics_abs_radps": 8.842079162597656,
      "max_boundary_abs_radps": 8.088103294372559,
      "max_rated_ratio": 1.105259895324707,
      "substep_exposure_fraction": 1.9073486328125e-05
    },
    "right_elbow_pitch_joint": {
      "rated_radps": 8.0,
      "max_physics_abs_radps": 9.695490837097168,
      "max_boundary_abs_radps": 8.05771255493164,
      "max_rated_ratio": 1.211936354637146,
      "substep_exposure_fraction": 5.6902567545572914e-05
    },
    "left_ankle_roll_joint": {
      "rated_radps": 8.0,
      "max_physics_abs_radps": 53.02068328857422,
      "max_boundary_abs_radps": 29.453092575073242,
      "max_rated_ratio": 6.627585411071777,
      "substep_exposure_fraction": 0.0008257230122884115
    },
    "right_ankle_roll_joint": {
      "rated_radps": 8.0,
      "max_physics_abs_radps": 52.809844970703125,
      "max_boundary_abs_radps": 22.23542594909668,
      "max_rated_ratio": 6.601230621337891,
      "substep_exposure_fraction": 0.0005812644958496094
    },
    "left_elbow_yaw_joint": {
      "rated_radps": 8.0,
      "max_physics_abs_radps": 4.727977752685547,
      "max_boundary_abs_radps": 4.2068281173706055,
      "max_rated_ratio": 0.5909972190856934,
      "substep_exposure_fraction": 0.0
    },
    "right_elbow_yaw_joint": {
      "rated_radps": 8.0,
      "max_physics_abs_radps": 4.611572742462158,
      "max_boundary_abs_radps": 4.2046990394592285,
      "max_rated_ratio": 0.5764465928077698,
      "substep_exposure_fraction": 0.0
    }
  },
  "right_ankle_pitch_physics_peak_event": {
    "round": 13,
    "row": 13177,
    "episode_id_run_local": 14914945029,
    "episode_step": 10,
    "joint": "right_ankle_pitch_joint",
    "rated_radps": 8.0,
    "boundary_dq_radps": -3.926098585128784,
    "physics_signed_dq_radps": 44.59560012817383,
    "physics_abs_dq_radps": 44.59560012817383,
    "peak_substep_0_to_7": 4,
    "q_at_joint_peak_rad": -0.1717906892299652,
    "torque_at_joint_peak_Nm": 0.055268287658691406,
    "target_at_joint_peak_rad": -0.38739943504333496,
    "requested_action": -0.9994906187057495,
    "applied_action": -0.64566570520401,
    "foot_peak_vertical_net_force_N": [
      0.0,
      10127.7294921875
    ],
    "foot_vertical_net_force_integral_Ns": [
      0.0,
      27.879276275634766
    ],
    "global_peak_joint": "right_ankle_pitch_joint",
    "feet_at_global_abs_dq_peak_origin_pose_wxyz_linear_angular": [
      [
        8.353196144104004,
        30.419742584228516,
        0.2786833643913269,
        0.9260918498039246,
        0.06762348115444183,
        0.16112379729747772,
        0.33439502120018005,
        -1.1925594806671143,
        -1.8710262775421143,
        2.1614694595336914,
        0.12556344270706177,
        6.218074321746826,
        3.394855260848999
      ],
      [
        9.018678665161133,
        30.745309829711914,
        0.06871508061885834,
        0.8218111395835876,
        0.1451767086982727,
        -0.37606897950172424,
        0.40264391899108887,
        -2.995957136154175,
        -4.110357761383057,
        -2.6205596923828125,
        -66.50651550292969,
        16.344682693481445,
        -5.481433391571045
      ]
    ],
    "feet_vertical_net_force_at_global_peak_N": [
      0.0,
      10127.7294921875
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
    "row": 6477,
    "episode_id_run_local": 5586812932,
    "episode_step": 12,
    "joint": "right_ankle_roll_joint",
    "rated_radps": 8.0,
    "boundary_dq_radps": 0.5990135669708252,
    "physics_signed_dq_radps": -52.809844970703125,
    "physics_abs_dq_radps": 52.809844970703125,
    "peak_substep_0_to_7": 5,
    "q_at_joint_peak_rad": -0.41515207290649414,
    "torque_at_joint_peak_Nm": -0.13814306259155273,
    "target_at_joint_peak_rad": -0.33993756771087646,
    "requested_action": 0.9982174038887024,
    "applied_action": -0.6798751354217529,
    "foot_peak_vertical_net_force_N": [
      0.0,
      10218.236328125
    ],
    "foot_vertical_net_force_integral_Ns": [
      0.0,
      25.545589447021484
    ],
    "global_peak_joint": "right_ankle_roll_joint",
    "feet_at_global_abs_dq_peak_origin_pose_wxyz_linear_angular": [
      [
        39.13017272949219,
        -13.310243606567383,
        0.10411424189805984,
        0.5188121795654297,
        0.047492172569036484,
        -0.07294906675815582,
        -0.850445032119751,
        -1.8657162189483643,
        -0.402870237827301,
        -2.379779577255249,
        -2.680335283279419,
        4.1294426918029785,
        2.66727614402771
      ],
      [
        38.34880447387695,
        -13.50918960571289,
        0.05032943934202194,
        0.5307120084762573,
        -0.20895452797412872,
        0.31649887561798096,
        -0.7579651474952698,
        -7.367401123046875,
        3.174307107925415,
        2.7495927810668945,
        50.861873626708984,
        60.43223190307617,
        4.8692307472229
      ]
    ],
    "feet_vertical_net_force_at_global_peak_N": [
      0.0,
      10218.236328125
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
  "semantics": "Run-local episode IDs; original physical thresholds shadow; interval mean of exactly 8 completed substeps; vertical NET force not contact point normal impulse; unfinished trajectories are right-censored, excluded from completed means"
}
```

### 恢复24的短段核验

```json
{
  "status": "passed",
  "rows": 24576,
  "rounds": 1,
  "round_first": 3,
  "round_last": 3,
  "replay_count": 73728,
  "replay_capacity": 4000000,
  "fifo_evictions": 0,
  "requested_saturation_fraction": 0.7907148720561594,
  "applied_saturation_fraction": 0.14938929461050723,
  "strict_position_velocity_torque_counts": [
    110,
    1382,
    0
  ],
  "training_position_velocity_torque_counts": [
    110,
    247,
    0
  ],
  "macro_contact_height_tilt_counts": [
    6,
    0,
    0
  ],
  "boundary_max_radps": 19.12569236755371,
  "physics_max_radps": 32.19276428222656,
  "physics_max_rated_ratio": 3.708996057510376,
  "boundary_max_event": {
    "round": 3,
    "row": 23598,
    "episode_id_run_local": 2216974876674,
    "episode_step": 3,
    "joint": "right_thigh_pitch_joint",
    "rated_radps": 25.0,
    "boundary_dq_radps": -19.12569236755371,
    "physics_signed_dq_radps": -19.535930633544922,
    "physics_abs_dq_radps": 19.535930633544922,
    "peak_substep_0_to_7": 5,
    "q_at_joint_peak_rad": -1.081801176071167,
    "torque_at_joint_peak_Nm": 12.018478393554688,
    "target_at_joint_peak_rad": -1.5225791931152344,
    "requested_action": -0.998568058013916,
    "applied_action": -0.6030972599983215,
    "foot_peak_vertical_net_force_N": [
      507.7763671875,
      0.0
    ],
    "foot_vertical_net_force_integral_Ns": [
      5.169576168060303,
      0.0
    ],
    "global_peak_joint": "right_thigh_pitch_joint",
    "feet_at_global_abs_dq_peak_origin_pose_wxyz_linear_angular": [
      [
        -1.5733237266540527,
        33.03351593017578,
        0.044932037591934204,
        0.9356293678283691,
        0.0018882309086620808,
        -0.0005274621653370559,
        -0.35297852754592896,
        0.02342662587761879,
        0.04150179773569107,
        -0.02601788565516472,
        -0.7810826897621155,
        0.6251821517944336,
        0.022727515548467636
      ],
      [
        -1.558713436126709,
        32.73719024658203,
        0.13631369173526764,
        0.9546669125556946,
        -0.10877294093370438,
        -0.27680298686027527,
        0.012629475444555283,
        1.3108863830566406,
        0.34294208884239197,
        2.1055235862731934,
        -2.615513324737549,
        -6.338859558105469,
        7.777263641357422
      ]
    ],
    "feet_vertical_net_force_at_global_peak_N": [
      274.93743896484375,
      0.0
    ],
    "strict_flags_pos_vel_torque": [
      0,
      0,
      0
    ],
    "training_flags_pos_vel_torque": [
      0,
      0,
      0
    ],
    "macro_contact_height_tilt": [
      0,
      0,
      0
    ]
  },
  "physics_max_event": {
    "round": 3,
    "row": 13516,
    "episode_id_run_local": 20602421250,
    "episode_step": 2,
    "joint": "right_knee_joint",
    "rated_radps": 25.0,
    "boundary_dq_radps": -1.7344861030578613,
    "physics_signed_dq_radps": -32.19276428222656,
    "physics_abs_dq_radps": 32.19276428222656,
    "peak_substep_0_to_7": 5,
    "q_at_joint_peak_rad": -0.2024448662996292,
    "torque_at_joint_peak_Nm": 17.659183502197266,
    "target_at_joint_peak_rad": -0.1999828815460205,
    "requested_action": -0.9999873042106628,
    "applied_action": -0.9999873042106628,
    "foot_peak_vertical_net_force_N": [
      580.1996459960938,
      6679.84521484375
    ],
    "foot_vertical_net_force_integral_Ns": [
      3.1028671264648438,
      19.700944900512695
    ],
    "global_peak_joint": "right_knee_joint",
    "feet_at_global_abs_dq_peak_origin_pose_wxyz_linear_angular": [
      [
        -11.128721237182617,
        -25.915029525756836,
        0.04653500020503998,
        0.6646313071250916,
        -0.005236510653048754,
        -0.018135391175746918,
        -0.7469329237937927,
        -0.5248092412948608,
        -0.8590789437294006,
        0.11172012984752655,
        1.853766679763794,
        -1.4886795282363892,
        -5.4565582275390625
      ],
      [
        -11.347105979919434,
        -25.87237548828125,
        0.05396449565887451,
        0.8127940893173218,
        -0.04524094983935356,
        -0.09386537969112396,
        -0.5731564164161682,
        5.554575443267822,
        -6.972264289855957,
        -0.30751296877861023,
        -11.815000534057617,
        -17.85586166381836,
        1.5451526641845703
      ]
    ],
    "feet_vertical_net_force_at_global_peak_N": [
      0.0,
      6679.84521484375
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
    "round": 3,
    "row": 23147,
    "episode_id_run_local": 2209408352258,
    "episode_step": 3,
    "joint": "right_ankle_pitch_joint",
    "rated_radps": 8.0,
    "boundary_dq_radps": -2.078796148300171,
    "physics_signed_dq_radps": 29.671968460083008,
    "physics_abs_dq_radps": 29.671968460083008,
    "peak_substep_0_to_7": 4,
    "q_at_joint_peak_rad": -0.1902025192975998,
    "torque_at_joint_peak_Nm": 0.4976167678833008,
    "target_at_joint_peak_rad": -0.35425665974617004,
    "requested_action": -1.0,
    "applied_action": -0.5904277563095093,
    "foot_peak_vertical_net_force_N": [
      0.0,
      7914.05712890625
    ],
    "foot_vertical_net_force_integral_Ns": [
      0.0,
      26.77709197998047
    ],
    "global_peak_joint": "right_knee_joint",
    "feet_at_global_abs_dq_peak_origin_pose_wxyz_linear_angular": [
      [
        21.224449157714844,
        30.083860397338867,
        0.24333864450454712,
        0.9546372890472412,
        0.06311266869306564,
        -0.2746344804763794,
        -0.09623058140277863,
        3.5676233768463135,
        -2.1227152347564697,
        2.1797356605529785,
        -0.004261502996087074,
        -11.649491310119629,
        -3.440045118331909
      ],
      [
        20.713722229003906,
        29.924070358276367,
        0.04941921681165695,
        0.996792197227478,
        -0.011612828820943832,
        -0.05113974213600159,
        0.06045915558934212,
        7.839982509613037,
        5.427298545837402,
        -1.7005118131637573,
        29.38065528869629,
        1.835230827331543,
        -4.384096622467041
      ]
    ],
    "feet_vertical_net_force_at_global_peak_N": [
      0.0,
      7914.05712890625
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
  "foot_vertical_net_force_peak_N": 12373.1640625,
  "first_step_foot_vertical_net_force_N": {
    "n": 6398,
    "mean": 91.90436553955078,
    "median": 0.0,
    "p95": 601.06994934082,
    "p99": 1223.027268066406,
    "max": 2412.094970703125
  },
  "first_step_foot_absolute_vertical_net_force_integral_Ns": {
    "n": 6398,
    "mean": 0.5374482870101929,
    "median": 0.0,
    "p95": 3.5074003219604486,
    "p99": 5.39100306510925,
    "max": 11.359224319458008
  },
  "reset_lift_m": {
    "n": 6398,
    "mean": 0.0013958984054625034,
    "median": 0.0,
    "p95": 0.013830908574163914,
    "p99": 0.019366327673196793,
    "max": 0.0219456497579813
  },
  "resets": 6398,
  "first_step_terminated": 6,
  "first_step_termination_fraction": 0.0009377930603313535,
  "first_step_strict_counts": [
    0,
    13,
    0
  ],
  "first_step_macro_counts": [
    6,
    0,
    0
  ],
  "raw_velocity_penalty_mean": 0.005095553273955981,
  "weighted_velocity_penalty_mean": -0.12738883184889951,
  "strict_velocity_substep_joint_exposure_fraction": 0.00038987311764039856,
  "velocity_reward_coverage_miss_count": 0,
  "latched_velocity_terms_without_boundary_violation": 1222,
  "reset_q_preservation_max_error_rad": 8.821487429422348e-08,
  "reset_dq_preservation_max_error_radps": 0.0,
  "decoder_max_error_rad": 1.9156932840047602e-07,
  "ground_minimum_margin_m": 195.19232177734375,
  "ground_outside_count": 0,
  "mixed_nonperiodic_z_equal_pairs": 9103,
  "mixed_nonperiodic_cosine": {
    "n": 9103,
    "mean": 1.0,
    "median": 1.0,
    "p95": 1.0000001192092896,
    "p99": 1.0000001192092896,
    "max": 1.0000001192092896
  },
  "tracking_unexpected_refreshes": 0,
  "tracking_phase_mismatches": 0,
  "tracking_window_stable_pairs": 9075,
  "tracking_adjacent_phase_cosine": {
    "n": 9075,
    "mean": 0.9979664087295532,
    "median": 0.9982756972312927,
    "p95": 0.9999957084655762,
    "p99": 0.9999975717067718,
    "max": 0.9999990463256836
  },
  "tracking_window_renewals": 0,
  "terminal_following_new_episode_pairs": 254,
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
    "n": 351,
    "mean": 2.8803418803418803,
    "median": 3.0,
    "p95": 4.0,
    "p99": 4.0,
    "max": 4.0
  },
  "completed_strict_shadow_len": {
    "n": 351,
    "mean": 2.826210826210826,
    "median": 3.0,
    "p95": 4.0,
    "p99": 4.0,
    "max": 4.0
  },
  "completed_tracking_shadow_len": {
    "n": 176,
    "mean": 2.715909090909091,
    "median": 3.0,
    "p95": 4.0,
    "p99": 4.0,
    "max": 4.0
  },
  "completed_mixed_shadow_len": {
    "n": 175,
    "mean": 2.9371428571428573,
    "median": 3.0,
    "p95": 4.0,
    "p99": 4.0,
    "max": 4.0
  },
  "unfinished_train_episodes_right_censored": 6047,
  "unfinished_strict_shadow_right_censored": 4995,
  "unfinished_train_observed_lengths": {
    "n": 6047,
    "mean": 3.8969737059699026,
    "median": 4.0,
    "p95": 4.0,
    "p99": 4.0,
    "max": 4.0
  },
  "semantics": "Run-local episode IDs; original physical thresholds shadow; interval mean of exactly 8 completed substeps; vertical NET force not contact point normal impulse; unfinished trajectories are right-censored, excluded from completed means"
}
```

本轮峰值事件已包含逐关节signed dq、q、输入力矩、PD target、0–7子步序号及requested/applied动作。足部13维origin位置/quaternion wxyz/线角速度和当帧净力仅对应该控制步全局abs(dq)峰子步；若额定归一化峰属于另一关节，不能把该足部事件冒称为其独有峰子步。同一episode物理相对时间=(step×.02)+(substep+1)×.0025。

131列control trace稳定前缀：0:23 requested，23:46 applied，46:69 target，69:92步末dq，92:115步末torque，115:118原strict flags，118:120双足垂直净力峰，120root高度，121倾角，122:125接触/高度/倾角终止，125lift，126global物理速度峰，127dt，128:130rootXY，130足地面余量。223列interval：0:23abs dq峰；23:46峰子步；46:69峰q；69:92峰torque；92:115峰target；115:138每joint mean8罚；138:161每joint原额定暴露子步比例；161:163双足积分N·s；163strict shadow长度；164完成子步数8；165:168strict flags；168:171train flags；171:194signed峰dq；194:220双足global峰13维状态；220:222该帧双足垂直净力；222global峰joint索引。以上切片均零起点、右开区间。

净力积分为每足abs(vertical NET force)对子步时间的积分，非逐碰撞点法向冲量；净力峰N与积分N·s不可混用。混合非周期相邻z逐值相等；tracking相邻z余弦仅描述合法相位变化。episode_id只在本Run内唯一；进程重建重新RSI，不能跨Run仅按ID拼成物理连续episode。

## 7. 恢复、replay容量与资源证据

### 16完整checkpoint

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
  "fingerprint": "sha256:67c040bea5a934869eb530ad28896299690c16675ff47024eaf67ac078aec1ff",
  "execution": {
    "schema_version": "eb/training-binding/v1",
    "recipe_id": "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_tracking_contact_kd_v10",
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

### 24完整checkpoint

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
  "fingerprint": "sha256:67c040bea5a934869eb530ad28896299690c16675ff47024eaf67ac078aec1ff",
  "execution": {
    "schema_version": "eb/training-binding/v1",
    "recipe_id": "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_tracking_contact_kd_v10",
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

### 524完整checkpoint

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
  "fingerprint": "sha256:67c040bea5a934869eb530ad28896299690c16675ff47024eaf67ac078aec1ff",
  "execution": {
    "schema_version": "eb/training-binding/v1",
    "recipe_id": "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_tracking_contact_kd_v10",
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

### 正式恢复回执

```json
{
  "schema_version": "eb/resume-request/v1",
  "status": "validated",
  "source_run_id": "run_9b47b8e1134b4fb39c0a",
  "step": 24,
  "mode": "trainer_state",
  "checkpoint": "/workspace/resume/global_step_24",
  "compatibility_fingerprint": "sha256:67c040bea5a934869eb530ad28896299690c16675ff47024eaf67ac078aec1ff"
}
```

### 三个sampler权重同步身份

```json
{
  "schema_version": "eb/ufo-sampler-sync/v1",
  "identity": {
    "state_sha256": "caa089f02748a842d511462594f41e626aa8a441721427cc1e021fe2972dc007",
    "version": 520,
    "online_transitions": 835584
  },
  "samplers": [
    {
      "rank": 0,
      "state_sha256": "caa089f02748a842d511462594f41e626aa8a441721427cc1e021fe2972dc007",
      "version": 520,
      "online_transitions": 835584
    },
    {
      "rank": 1,
      "state_sha256": "caa089f02748a842d511462594f41e626aa8a441721427cc1e021fe2972dc007",
      "version": 520,
      "online_transitions": 835584
    },
    {
      "rank": 2,
      "state_sha256": "caa089f02748a842d511462594f41e626aa8a441721427cc1e021fe2972dc007",
      "version": 520,
      "online_transitions": 835584
    }
  ],
  "sampling_rounds": 34
}
```

模型/target/六Adam/normalizer、每rank RNG、replay、expert sampler、zpool和rollout state完整恢复；scheduler/scaler不适用。PhysX内部状态不可恢复，每段新RSI并重新建立context，不承诺逐位物理轨迹连续。行为采样模型同步版本通常为最终更新之前一个batch（正式为520），不是缺失最终524learner权重；最后四次partial update发生在最后一批采样之后。

### v2checkpoint全部文件size/hash与schema

```json
{
  "schema_version": "eb/resume-checkpoint/v1",
  "status": "complete",
  "format": "ufo_agent_state_v2",
  "mode": "trainer_state",
  "step": 524,
  "primary_unit": "learner_joint_update",
  "run_id": "run_1299df81ef5740f99f23",
  "execution": {
    "schema_version": "eb/training-binding/v1",
    "recipe_id": "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_tracking_contact_kd_v10",
    "algorithm_id": "bfm_fb_cpr_aux",
    "provider_id": "ufo",
    "runtime_id": "rlinf",
    "simulator_id": "isaaclab",
    "progress_unit": "learner_joint_update",
    "checkpoint_format": "ufo_agent_state_v2",
    "topology_id": "one_learner_three_samplers"
  },
  "compatibility_fingerprint": "sha256:67c040bea5a934869eb530ad28896299690c16675ff47024eaf67ac078aec1ff",
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
  "optimizer_steps": {
    "forward_optimizer": 524,
    "backward_optimizer": 524,
    "discriminator_optimizer": 524,
    "actor_optimizer": 524,
    "critic_optimizer": 524,
    "aux_critic_optimizer": 524
  },
  "model_keys": [
    "_actor.embed_s.0.mlp.0.bias",
    "_actor.embed_s.0.mlp.0.weight",
    "_actor.embed_s.0.mlp.1.bias",
    "_actor.embed_s.0.mlp.1.weight",
    "_actor.embed_s.1.mlp.0.bias",
    "_actor.embed_s.1.mlp.0.weight",
    "_actor.embed_s.1.mlp.1.bias",
    "_actor.embed_s.1.mlp.1.weight",
    "_actor.embed_z.0.mlp.0.bias",
    "_actor.embed_z.0.mlp.0.weight",
    "_actor.embed_z.0.mlp.1.bias",
    "_actor.embed_z.0.mlp.1.weight",
    "_actor.embed_z.1.mlp.0.bias",
    "_actor.embed_z.1.mlp.0.weight",
    "_actor.embed_z.1.mlp.1.bias",
    "_actor.embed_z.1.mlp.1.weight",
    "_actor.policy.0.mlp.0.bias",
    "_actor.policy.0.mlp.0.weight",
    "_actor.policy.0.mlp.1.bias",
    "_actor.policy.0.mlp.1.weight",
    "_actor.policy.1.mlp.0.bias",
    "_actor.policy.1.mlp.0.weight",
    "_actor.policy.1.mlp.1.bias",
    "_actor.policy.1.mlp.1.weight",
    "_actor.policy.2.mlp.0.bias",
    "_actor.policy.2.mlp.0.weight",
    "_actor.policy.2.mlp.1.bias",
    "_actor.policy.2.mlp.1.weight",
    "_actor.policy.3.mlp.0.bias",
    "_actor.policy.3.mlp.0.weight",
    "_actor.policy.3.mlp.1.bias",
    "_actor.policy.3.mlp.1.weight",
    "_actor.policy.4.mlp.0.bias",
    "_actor.policy.4.mlp.0.weight",
    "_actor.policy.4.mlp.1.bias",
    "_actor.policy.4.mlp.1.weight",
    "_actor.policy.5.mlp.0.bias",
    "_actor.policy.5.mlp.0.weight",
    "_actor.policy.5.mlp.1.bias",
    "_actor.policy.5.mlp.1.weight",
    "_actor.policy.6.mlp.0.bias",
    "_actor.policy.6.mlp.0.weight",
    "_actor.policy.6.mlp.1.bias",
    "_actor.policy.6.mlp.1.weight",
    "_aux_critic.Fs.0.mlp.0.bias",
    "_aux_critic.Fs.0.mlp.0.weight",
    "_aux_critic.Fs.0.mlp.1.bias",
    "_aux_critic.Fs.0.mlp.1.weight",
    "_aux_critic.Fs.1.mlp.0.bias",
    "_aux_critic.Fs.1.mlp.0.weight",
    "_aux_critic.Fs.1.mlp.1.bias",
    "_aux_critic.Fs.1.mlp.1.weight",
    "_aux_critic.Fs.2.mlp.0.bias",
    "_aux_critic.Fs.2.mlp.0.weight",
    "_aux_critic.Fs.2.mlp.1.bias",
    "_aux_critic.Fs.2.mlp.1.weight",
    "_aux_critic.Fs.3.mlp.0.bias",
    "_aux_critic.Fs.3.mlp.0.weight",
    "_aux_critic.Fs.3.mlp.1.bias",
    "_aux_critic.Fs.3.mlp.1.weight",
    "_aux_critic.Fs.4.mlp.0.bias",
    "_aux_critic.Fs.4.mlp.0.weight",
    "_aux_critic.Fs.4.mlp.1.bias",
    "_aux_critic.Fs.4.mlp.1.weight",
    "_aux_critic.Fs.5.mlp.0.bias",
    "_aux_critic.Fs.5.mlp.0.weight",
    "_aux_critic.Fs.5.mlp.1.bias",
    "_aux_critic.Fs.5.mlp.1.weight",
    "_aux_critic.Fs.6.mlp.0.bias",
    "_aux_critic.Fs.6.mlp.0.weight",
    "_aux_critic.Fs.6.mlp.1.bias",
    "_aux_critic.Fs.6.mlp.1.weight",
    "_aux_critic.embed_sa.0.mlp.0.bias",
    "_aux_critic.embed_sa.0.mlp.0.weight",
    "_aux_critic.embed_sa.0.mlp.1.bias",
    "_aux_critic.embed_sa.0.mlp.1.weight",
    "_aux_critic.embed_sa.1.mlp.0.bias",
    "_aux_critic.embed_sa.1.mlp.0.weight",
    "_aux_critic.embed_sa.1.mlp.1.bias",
    "_aux_critic.embed_sa.1.mlp.1.weight",
    "_aux_critic.embed_sa.2.mlp.0.bias",
    "_aux_critic.embed_sa.2.mlp.0.weight",
    "_aux_critic.embed_sa.2.mlp.1.bias",
    "_aux_critic.embed_sa.2.mlp.1.weight",
    "_aux_critic.embed_sa.3.mlp.0.bias",
    "_aux_critic.embed_sa.3.mlp.0.weight",
    "_aux_critic.embed_sa.3.mlp.1.bias",
    "_aux_critic.embed_sa.3.mlp.1.weight",
    "_aux_critic.embed_sa.4.mlp.0.bias",
    "_aux_critic.embed_sa.4.mlp.0.weight",
    "_aux_critic.embed_sa.4.mlp.1.bias",
    "_aux_critic.embed_sa.4.mlp.1.weight",
    "_aux_critic.embed_sa.5.mlp.0.bias",
    "_aux_critic.embed_sa.5.mlp.0.weight",
    "_aux_critic.embed_sa.5.mlp.1.bias",
    "_aux_critic.embed_sa.5.mlp.1.weight",
    "_aux_critic.embed_z.0.mlp.0.bias",
    "_aux_critic.embed_z.0.mlp.0.weight",
    "_aux_critic.embed_z.0.mlp.1.bias",
    "_aux_critic.embed_z.0.mlp.1.weight",
    "_aux_critic.embed_z.1.mlp.0.bias",
    "_aux_critic.embed_z.1.mlp.0.weight",
    "_aux_critic.embed_z.1.mlp.1.bias",
    "_aux_critic.embed_z.1.mlp.1.weight",
    "_aux_critic.embed_z.2.mlp.0.bias",
    "_aux_critic.embed_z.2.mlp.0.weight",
    "_aux_critic.embed_z.2.mlp.1.bias",
    "_aux_critic.embed_z.2.mlp.1.weight",
    "_aux_critic.embed_z.3.mlp.0.bias",
    "_aux_critic.embed_z.3.mlp.0.weight",
    "_aux_critic.embed_z.3.mlp.1.bias",
    "_aux_critic.embed_z.3.mlp.1.weight",
    "_aux_critic.embed_z.4.mlp.0.bias",
    "_aux_critic.embed_z.4.mlp.0.weight",
    "_aux_critic.embed_z.4.mlp.1.bias",
    "_aux_critic.embed_z.4.mlp.1.weight",
    "_aux_critic.embed_z.5.mlp.0.bias",
    "_aux_critic.embed_z.5.mlp.0.weight",
    "_aux_critic.embed_z.5.mlp.1.bias",
    "_aux_critic.embed_z.5.mlp.1.weight",
    "_aux_reward_normalizer.counter",
    "_aux_reward_normalizer.mean",
    "_aux_reward_normalizer.mean_square",
    "_backward_map.net.0.bias",
    "_backward_map.net.0.weight",
    "_backward_map.net.1.bias",
    "_backward_map.net.1.weight",
    "_backward_map.net.3.bias",
    "_backward_map.net.3.weight",
    "_critic.Fs.0.mlp.0.bias",
    "_critic.Fs.0.mlp.0.weight",
    "_critic.Fs.0.mlp.1.bias",
    "_critic.Fs.0.mlp.1.weight",
    "_critic.Fs.1.mlp.0.bias",
    "_critic.Fs.1.mlp.0.weight",
    "_critic.Fs.1.mlp.1.bias",
    "_critic.Fs.1.mlp.1.weight",
    "_critic.Fs.2.mlp.0.bias",
    "_critic.Fs.2.mlp.0.weight",
    "_critic.Fs.2.mlp.1.bias",
    "_critic.Fs.2.mlp.1.weight",
    "_critic.Fs.3.mlp.0.bias",
    "_critic.Fs.3.mlp.0.weight",
    "_critic.Fs.3.mlp.1.bias",
    "_critic.Fs.3.mlp.1.weight",
    "_critic.Fs.4.mlp.0.bias",
    "_critic.Fs.4.mlp.0.weight",
    "_critic.Fs.4.mlp.1.bias",
    "_critic.Fs.4.mlp.1.weight",
    "_critic.Fs.5.mlp.0.bias",
    "_critic.Fs.5.mlp.0.weight",
    "_critic.Fs.5.mlp.1.bias",
    "_critic.Fs.5.mlp.1.weight",
    "_critic.Fs.6.mlp.0.bias",
    "_critic.Fs.6.mlp.0.weight",
    "_critic.Fs.6.mlp.1.bias",
    "_critic.Fs.6.mlp.1.weight",
    "_critic.embed_sa.0.mlp.0.bias",
    "_critic.embed_sa.0.mlp.0.weight",
    "_critic.embed_sa.0.mlp.1.bias",
    "_critic.embed_sa.0.mlp.1.weight",
    "_critic.embed_sa.1.mlp.0.bias",
    "_critic.embed_sa.1.mlp.0.weight",
    "_critic.embed_sa.1.mlp.1.bias",
    "_critic.embed_sa.1.mlp.1.weight",
    "_critic.embed_sa.2.mlp.0.bias",
    "_critic.embed_sa.2.mlp.0.weight",
    "_critic.embed_sa.2.mlp.1.bias",
    "_critic.embed_sa.2.mlp.1.weight",
    "_critic.embed_sa.3.mlp.0.bias",
    "_critic.embed_sa.3.mlp.0.weight",
    "_critic.embed_sa.3.mlp.1.bias",
    "_critic.embed_sa.3.mlp.1.weight",
    "_critic.embed_sa.4.mlp.0.bias",
    "_critic.embed_sa.4.mlp.0.weight",
    "_critic.embed_sa.4.mlp.1.bias",
    "_critic.embed_sa.4.mlp.1.weight",
    "_critic.embed_sa.5.mlp.0.bias",
    "_critic.embed_sa.5.mlp.0.weight",
    "_critic.embed_sa.5.mlp.1.bias",
    "_critic.embed_sa.5.mlp.1.weight",
    "_critic.embed_z.0.mlp.0.bias",
    "_critic.embed_z.0.mlp.0.weight",
    "_critic.embed_z.0.mlp.1.bias",
    "_critic.embed_z.0.mlp.1.weight",
    "_critic.embed_z.1.mlp.0.bias",
    "_critic.embed_z.1.mlp.0.weight",
    "_critic.embed_z.1.mlp.1.bias",
    "_critic.embed_z.1.mlp.1.weight",
    "_critic.embed_z.2.mlp.0.bias",
    "_critic.embed_z.2.mlp.0.weight",
    "_critic.embed_z.2.mlp.1.bias",
    "_critic.embed_z.2.mlp.1.weight",
    "_critic.embed_z.3.mlp.0.bias",
    "_critic.embed_z.3.mlp.0.weight",
    "_critic.embed_z.3.mlp.1.bias",
    "_critic.embed_z.3.mlp.1.weight",
    "_critic.embed_z.4.mlp.0.bias",
    "_critic.embed_z.4.mlp.0.weight",
    "_critic.embed_z.4.mlp.1.bias",
    "_critic.embed_z.4.mlp.1.weight",
    "_critic.embed_z.5.mlp.0.bias",
    "_critic.embed_z.5.mlp.0.weight",
    "_critic.embed_z.5.mlp.1.bias",
    "_critic.embed_z.5.mlp.1.weight",
    "_discriminator.trunk.0.bias",
    "_discriminator.trunk.0.weight",
    "_discriminator.trunk.1.bias",
    "_discriminator.trunk.1.weight",
    "_discriminator.trunk.3.bias",
    "_discriminator.trunk.3.weight",
    "_discriminator.trunk.5.bias",
    "_discriminator.trunk.5.weight",
    "_discriminator.trunk.7.bias",
    "_discriminator.trunk.7.weight",
    "_forward_map.Fs.0.mlp.0.bias",
    "_forward_map.Fs.0.mlp.0.weight",
    "_forward_map.Fs.0.mlp.1.bias",
    "_forward_map.Fs.0.mlp.1.weight",
    "_forward_map.Fs.1.mlp.0.bias",
    "_forward_map.Fs.1.mlp.0.weight",
    "_forward_map.Fs.1.mlp.1.bias",
    "_forward_map.Fs.1.mlp.1.weight",
    "_forward_map.Fs.2.mlp.0.bias",
    "_forward_map.Fs.2.mlp.0.weight",
    "_forward_map.Fs.2.mlp.1.bias",
    "_forward_map.Fs.2.mlp.1.weight",
    "_forward_map.Fs.3.mlp.0.bias",
    "_forward_map.Fs.3.mlp.0.weight",
    "_forward_map.Fs.3.mlp.1.bias",
    "_forward_map.Fs.3.mlp.1.weight",
    "_forward_map.Fs.4.mlp.0.bias",
    "_forward_map.Fs.4.mlp.0.weight",
    "_forward_map.Fs.4.mlp.1.bias",
    "_forward_map.Fs.4.mlp.1.weight",
    "_forward_map.Fs.5.mlp.0.bias",
    "_forward_map.Fs.5.mlp.0.weight",
    "_forward_map.Fs.5.mlp.1.bias",
    "_forward_map.Fs.5.mlp.1.weight",
    "_forward_map.Fs.6.mlp.0.bias",
    "_forward_map.Fs.6.mlp.0.weight",
    "_forward_map.Fs.6.mlp.1.bias",
    "_forward_map.Fs.6.mlp.1.weight",
    "_forward_map.embed_sa.0.mlp.0.bias",
    "_forward_map.embed_sa.0.mlp.0.weight",
    "_forward_map.embed_sa.0.mlp.1.bias",
    "_forward_map.embed_sa.0.mlp.1.weight",
    "_forward_map.embed_sa.1.mlp.0.bias",
    "_forward_map.embed_sa.1.mlp.0.weight",
    "_forward_map.embed_sa.1.mlp.1.bias",
    "_forward_map.embed_sa.1.mlp.1.weight",
    "_forward_map.embed_sa.2.mlp.0.bias",
    "_forward_map.embed_sa.2.mlp.0.weight",
    "_forward_map.embed_sa.2.mlp.1.bias",
    "_forward_map.embed_sa.2.mlp.1.weight",
    "_forward_map.embed_sa.3.mlp.0.bias",
    "_forward_map.embed_sa.3.mlp.0.weight",
    "_forward_map.embed_sa.3.mlp.1.bias",
    "_forward_map.embed_sa.3.mlp.1.weight",
    "_forward_map.embed_sa.4.mlp.0.bias",
    "_forward_map.embed_sa.4.mlp.0.weight",
    "_forward_map.embed_sa.4.mlp.1.bias",
    "_forward_map.embed_sa.4.mlp.1.weight",
    "_forward_map.embed_sa.5.mlp.0.bias",
    "_forward_map.embed_sa.5.mlp.0.weight",
    "_forward_map.embed_sa.5.mlp.1.bias",
    "_forward_map.embed_sa.5.mlp.1.weight",
    "_forward_map.embed_z.0.mlp.0.bias",
    "_forward_map.embed_z.0.mlp.0.weight",
    "_forward_map.embed_z.0.mlp.1.bias",
    "_forward_map.embed_z.0.mlp.1.weight",
    "_forward_map.embed_z.1.mlp.0.bias",
    "_forward_map.embed_z.1.mlp.0.weight",
    "_forward_map.embed_z.1.mlp.1.bias",
    "_forward_map.embed_z.1.mlp.1.weight",
    "_forward_map.embed_z.2.mlp.0.bias",
    "_forward_map.embed_z.2.mlp.0.weight",
    "_forward_map.embed_z.2.mlp.1.bias",
    "_forward_map.embed_z.2.mlp.1.weight",
    "_forward_map.embed_z.3.mlp.0.bias",
    "_forward_map.embed_z.3.mlp.0.weight",
    "_forward_map.embed_z.3.mlp.1.bias",
    "_forward_map.embed_z.3.mlp.1.weight",
    "_forward_map.embed_z.4.mlp.0.bias",
    "_forward_map.embed_z.4.mlp.0.weight",
    "_forward_map.embed_z.4.mlp.1.bias",
    "_forward_map.embed_z.4.mlp.1.weight",
    "_forward_map.embed_z.5.mlp.0.bias",
    "_forward_map.embed_z.5.mlp.0.weight",
    "_forward_map.embed_z.5.mlp.1.bias",
    "_forward_map.embed_z.5.mlp.1.weight",
    "_obs_normalizer._normalizers.history_actor._normalizer.num_batches_tracked",
    "_obs_normalizer._normalizers.history_actor._normalizer.running_mean",
    "_obs_normalizer._normalizers.history_actor._normalizer.running_var",
    "_obs_normalizer._normalizers.last_action._normalizer.num_batches_tracked",
    "_obs_normalizer._normalizers.last_action._normalizer.running_mean",
    "_obs_normalizer._normalizers.last_action._normalizer.running_var",
    "_obs_normalizer._normalizers.privileged_state._normalizer.num_batches_tracked",
    "_obs_normalizer._normalizers.privileged_state._normalizer.running_mean",
    "_obs_normalizer._normalizers.privileged_state._normalizer.running_var",
    "_obs_normalizer._normalizers.state._normalizer.num_batches_tracked",
    "_obs_normalizer._normalizers.state._normalizer.running_mean",
    "_obs_normalizer._normalizers.state._normalizer.running_var",
    "_target_aux_critic.Fs.0.mlp.0.bias",
    "_target_aux_critic.Fs.0.mlp.0.weight",
    "_target_aux_critic.Fs.0.mlp.1.bias",
    "_target_aux_critic.Fs.0.mlp.1.weight",
    "_target_aux_critic.Fs.1.mlp.0.bias",
    "_target_aux_critic.Fs.1.mlp.0.weight",
    "_target_aux_critic.Fs.1.mlp.1.bias",
    "_target_aux_critic.Fs.1.mlp.1.weight",
    "_target_aux_critic.Fs.2.mlp.0.bias",
    "_target_aux_critic.Fs.2.mlp.0.weight",
    "_target_aux_critic.Fs.2.mlp.1.bias",
    "_target_aux_critic.Fs.2.mlp.1.weight",
    "_target_aux_critic.Fs.3.mlp.0.bias",
    "_target_aux_critic.Fs.3.mlp.0.weight",
    "_target_aux_critic.Fs.3.mlp.1.bias",
    "_target_aux_critic.Fs.3.mlp.1.weight",
    "_target_aux_critic.Fs.4.mlp.0.bias",
    "_target_aux_critic.Fs.4.mlp.0.weight",
    "_target_aux_critic.Fs.4.mlp.1.bias",
    "_target_aux_critic.Fs.4.mlp.1.weight",
    "_target_aux_critic.Fs.5.mlp.0.bias",
    "_target_aux_critic.Fs.5.mlp.0.weight",
    "_target_aux_critic.Fs.5.mlp.1.bias",
    "_target_aux_critic.Fs.5.mlp.1.weight",
    "_target_aux_critic.Fs.6.mlp.0.bias",
    "_target_aux_critic.Fs.6.mlp.0.weight",
    "_target_aux_critic.Fs.6.mlp.1.bias",
    "_target_aux_critic.Fs.6.mlp.1.weight",
    "_target_aux_critic.embed_sa.0.mlp.0.bias",
    "_target_aux_critic.embed_sa.0.mlp.0.weight",
    "_target_aux_critic.embed_sa.0.mlp.1.bias",
    "_target_aux_critic.embed_sa.0.mlp.1.weight",
    "_target_aux_critic.embed_sa.1.mlp.0.bias",
    "_target_aux_critic.embed_sa.1.mlp.0.weight",
    "_target_aux_critic.embed_sa.1.mlp.1.bias",
    "_target_aux_critic.embed_sa.1.mlp.1.weight",
    "_target_aux_critic.embed_sa.2.mlp.0.bias",
    "_target_aux_critic.embed_sa.2.mlp.0.weight",
    "_target_aux_critic.embed_sa.2.mlp.1.bias",
    "_target_aux_critic.embed_sa.2.mlp.1.weight",
    "_target_aux_critic.embed_sa.3.mlp.0.bias",
    "_target_aux_critic.embed_sa.3.mlp.0.weight",
    "_target_aux_critic.embed_sa.3.mlp.1.bias",
    "_target_aux_critic.embed_sa.3.mlp.1.weight",
    "_target_aux_critic.embed_sa.4.mlp.0.bias",
    "_target_aux_critic.embed_sa.4.mlp.0.weight",
    "_target_aux_critic.embed_sa.4.mlp.1.bias",
    "_target_aux_critic.embed_sa.4.mlp.1.weight",
    "_target_aux_critic.embed_sa.5.mlp.0.bias",
    "_target_aux_critic.embed_sa.5.mlp.0.weight",
    "_target_aux_critic.embed_sa.5.mlp.1.bias",
    "_target_aux_critic.embed_sa.5.mlp.1.weight",
    "_target_aux_critic.embed_z.0.mlp.0.bias",
    "_target_aux_critic.embed_z.0.mlp.0.weight",
    "_target_aux_critic.embed_z.0.mlp.1.bias",
    "_target_aux_critic.embed_z.0.mlp.1.weight",
    "_target_aux_critic.embed_z.1.mlp.0.bias",
    "_target_aux_critic.embed_z.1.mlp.0.weight",
    "_target_aux_critic.embed_z.1.mlp.1.bias",
    "_target_aux_critic.embed_z.1.mlp.1.weight",
    "_target_aux_critic.embed_z.2.mlp.0.bias",
    "_target_aux_critic.embed_z.2.mlp.0.weight",
    "_target_aux_critic.embed_z.2.mlp.1.bias",
    "_target_aux_critic.embed_z.2.mlp.1.weight",
    "_target_aux_critic.embed_z.3.mlp.0.bias",
    "_target_aux_critic.embed_z.3.mlp.0.weight",
    "_target_aux_critic.embed_z.3.mlp.1.bias",
    "_target_aux_critic.embed_z.3.mlp.1.weight",
    "_target_aux_critic.embed_z.4.mlp.0.bias",
    "_target_aux_critic.embed_z.4.mlp.0.weight",
    "_target_aux_critic.embed_z.4.mlp.1.bias",
    "_target_aux_critic.embed_z.4.mlp.1.weight",
    "_target_aux_critic.embed_z.5.mlp.0.bias",
    "_target_aux_critic.embed_z.5.mlp.0.weight",
    "_target_aux_critic.embed_z.5.mlp.1.bias",
    "_target_aux_critic.embed_z.5.mlp.1.weight",
    "_target_backward_map.net.0.bias",
    "_target_backward_map.net.0.weight",
    "_target_backward_map.net.1.bias",
    "_target_backward_map.net.1.weight",
    "_target_backward_map.net.3.bias",
    "_target_backward_map.net.3.weight",
    "_target_critic.Fs.0.mlp.0.bias",
    "_target_critic.Fs.0.mlp.0.weight",
    "_target_critic.Fs.0.mlp.1.bias",
    "_target_critic.Fs.0.mlp.1.weight",
    "_target_critic.Fs.1.mlp.0.bias",
    "_target_critic.Fs.1.mlp.0.weight",
    "_target_critic.Fs.1.mlp.1.bias",
    "_target_critic.Fs.1.mlp.1.weight",
    "_target_critic.Fs.2.mlp.0.bias",
    "_target_critic.Fs.2.mlp.0.weight",
    "_target_critic.Fs.2.mlp.1.bias",
    "_target_critic.Fs.2.mlp.1.weight",
    "_target_critic.Fs.3.mlp.0.bias",
    "_target_critic.Fs.3.mlp.0.weight",
    "_target_critic.Fs.3.mlp.1.bias",
    "_target_critic.Fs.3.mlp.1.weight",
    "_target_critic.Fs.4.mlp.0.bias",
    "_target_critic.Fs.4.mlp.0.weight",
    "_target_critic.Fs.4.mlp.1.bias",
    "_target_critic.Fs.4.mlp.1.weight",
    "_target_critic.Fs.5.mlp.0.bias",
    "_target_critic.Fs.5.mlp.0.weight",
    "_target_critic.Fs.5.mlp.1.bias",
    "_target_critic.Fs.5.mlp.1.weight",
    "_target_critic.Fs.6.mlp.0.bias",
    "_target_critic.Fs.6.mlp.0.weight",
    "_target_critic.Fs.6.mlp.1.bias",
    "_target_critic.Fs.6.mlp.1.weight",
    "_target_critic.embed_sa.0.mlp.0.bias",
    "_target_critic.embed_sa.0.mlp.0.weight",
    "_target_critic.embed_sa.0.mlp.1.bias",
    "_target_critic.embed_sa.0.mlp.1.weight",
    "_target_critic.embed_sa.1.mlp.0.bias",
    "_target_critic.embed_sa.1.mlp.0.weight",
    "_target_critic.embed_sa.1.mlp.1.bias",
    "_target_critic.embed_sa.1.mlp.1.weight",
    "_target_critic.embed_sa.2.mlp.0.bias",
    "_target_critic.embed_sa.2.mlp.0.weight",
    "_target_critic.embed_sa.2.mlp.1.bias",
    "_target_critic.embed_sa.2.mlp.1.weight",
    "_target_critic.embed_sa.3.mlp.0.bias",
    "_target_critic.embed_sa.3.mlp.0.weight",
    "_target_critic.embed_sa.3.mlp.1.bias",
    "_target_critic.embed_sa.3.mlp.1.weight",
    "_target_critic.embed_sa.4.mlp.0.bias",
    "_target_critic.embed_sa.4.mlp.0.weight",
    "_target_critic.embed_sa.4.mlp.1.bias",
    "_target_critic.embed_sa.4.mlp.1.weight",
    "_target_critic.embed_sa.5.mlp.0.bias",
    "_target_critic.embed_sa.5.mlp.0.weight",
    "_target_critic.embed_sa.5.mlp.1.bias",
    "_target_critic.embed_sa.5.mlp.1.weight",
    "_target_critic.embed_z.0.mlp.0.bias",
    "_target_critic.embed_z.0.mlp.0.weight",
    "_target_critic.embed_z.0.mlp.1.bias",
    "_target_critic.embed_z.0.mlp.1.weight",
    "_target_critic.embed_z.1.mlp.0.bias",
    "_target_critic.embed_z.1.mlp.0.weight",
    "_target_critic.embed_z.1.mlp.1.bias",
    "_target_critic.embed_z.1.mlp.1.weight",
    "_target_critic.embed_z.2.mlp.0.bias",
    "_target_critic.embed_z.2.mlp.0.weight",
    "_target_critic.embed_z.2.mlp.1.bias",
    "_target_critic.embed_z.2.mlp.1.weight",
    "_target_critic.embed_z.3.mlp.0.bias",
    "_target_critic.embed_z.3.mlp.0.weight",
    "_target_critic.embed_z.3.mlp.1.bias",
    "_target_critic.embed_z.3.mlp.1.weight",
    "_target_critic.embed_z.4.mlp.0.bias",
    "_target_critic.embed_z.4.mlp.0.weight",
    "_target_critic.embed_z.4.mlp.1.bias",
    "_target_critic.embed_z.4.mlp.1.weight",
    "_target_critic.embed_z.5.mlp.0.bias",
    "_target_critic.embed_z.5.mlp.0.weight",
    "_target_critic.embed_z.5.mlp.1.bias",
    "_target_critic.embed_z.5.mlp.1.weight",
    "_target_forward_map.Fs.0.mlp.0.bias",
    "_target_forward_map.Fs.0.mlp.0.weight",
    "_target_forward_map.Fs.0.mlp.1.bias",
    "_target_forward_map.Fs.0.mlp.1.weight",
    "_target_forward_map.Fs.1.mlp.0.bias",
    "_target_forward_map.Fs.1.mlp.0.weight",
    "_target_forward_map.Fs.1.mlp.1.bias",
    "_target_forward_map.Fs.1.mlp.1.weight",
    "_target_forward_map.Fs.2.mlp.0.bias",
    "_target_forward_map.Fs.2.mlp.0.weight",
    "_target_forward_map.Fs.2.mlp.1.bias",
    "_target_forward_map.Fs.2.mlp.1.weight",
    "_target_forward_map.Fs.3.mlp.0.bias",
    "_target_forward_map.Fs.3.mlp.0.weight",
    "_target_forward_map.Fs.3.mlp.1.bias",
    "_target_forward_map.Fs.3.mlp.1.weight",
    "_target_forward_map.Fs.4.mlp.0.bias",
    "_target_forward_map.Fs.4.mlp.0.weight",
    "_target_forward_map.Fs.4.mlp.1.bias",
    "_target_forward_map.Fs.4.mlp.1.weight",
    "_target_forward_map.Fs.5.mlp.0.bias",
    "_target_forward_map.Fs.5.mlp.0.weight",
    "_target_forward_map.Fs.5.mlp.1.bias",
    "_target_forward_map.Fs.5.mlp.1.weight",
    "_target_forward_map.Fs.6.mlp.0.bias",
    "_target_forward_map.Fs.6.mlp.0.weight",
    "_target_forward_map.Fs.6.mlp.1.bias",
    "_target_forward_map.Fs.6.mlp.1.weight",
    "_target_forward_map.embed_sa.0.mlp.0.bias",
    "_target_forward_map.embed_sa.0.mlp.0.weight",
    "_target_forward_map.embed_sa.0.mlp.1.bias",
    "_target_forward_map.embed_sa.0.mlp.1.weight",
    "_target_forward_map.embed_sa.1.mlp.0.bias",
    "_target_forward_map.embed_sa.1.mlp.0.weight",
    "_target_forward_map.embed_sa.1.mlp.1.bias",
    "_target_forward_map.embed_sa.1.mlp.1.weight",
    "_target_forward_map.embed_sa.2.mlp.0.bias",
    "_target_forward_map.embed_sa.2.mlp.0.weight",
    "_target_forward_map.embed_sa.2.mlp.1.bias",
    "_target_forward_map.embed_sa.2.mlp.1.weight",
    "_target_forward_map.embed_sa.3.mlp.0.bias",
    "_target_forward_map.embed_sa.3.mlp.0.weight",
    "_target_forward_map.embed_sa.3.mlp.1.bias",
    "_target_forward_map.embed_sa.3.mlp.1.weight",
    "_target_forward_map.embed_sa.4.mlp.0.bias",
    "_target_forward_map.embed_sa.4.mlp.0.weight",
    "_target_forward_map.embed_sa.4.mlp.1.bias",
    "_target_forward_map.embed_sa.4.mlp.1.weight",
    "_target_forward_map.embed_sa.5.mlp.0.bias",
    "_target_forward_map.embed_sa.5.mlp.0.weight",
    "_target_forward_map.embed_sa.5.mlp.1.bias",
    "_target_forward_map.embed_sa.5.mlp.1.weight",
    "_target_forward_map.embed_z.0.mlp.0.bias",
    "_target_forward_map.embed_z.0.mlp.0.weight",
    "_target_forward_map.embed_z.0.mlp.1.bias",
    "_target_forward_map.embed_z.0.mlp.1.weight",
    "_target_forward_map.embed_z.1.mlp.0.bias",
    "_target_forward_map.embed_z.1.mlp.0.weight",
    "_target_forward_map.embed_z.1.mlp.1.bias",
    "_target_forward_map.embed_z.1.mlp.1.weight",
    "_target_forward_map.embed_z.2.mlp.0.bias",
    "_target_forward_map.embed_z.2.mlp.0.weight",
    "_target_forward_map.embed_z.2.mlp.1.bias",
    "_target_forward_map.embed_z.2.mlp.1.weight",
    "_target_forward_map.embed_z.3.mlp.0.bias",
    "_target_forward_map.embed_z.3.mlp.0.weight",
    "_target_forward_map.embed_z.3.mlp.1.bias",
    "_target_forward_map.embed_z.3.mlp.1.weight",
    "_target_forward_map.embed_z.4.mlp.0.bias",
    "_target_forward_map.embed_z.4.mlp.0.weight",
    "_target_forward_map.embed_z.4.mlp.1.bias",
    "_target_forward_map.embed_z.4.mlp.1.weight",
    "_target_forward_map.embed_z.5.mlp.0.bias",
    "_target_forward_map.embed_z.5.mlp.0.weight",
    "_target_forward_map.embed_z.5.mlp.1.bias",
    "_target_forward_map.embed_z.5.mlp.1.weight"
  ],
  "files": [
    {
      "path": "config.json",
      "bytes": 6249,
      "sha256": "5b28dbd03d1e252100d69c44dd1a0c473963a290f596b688f3e5785d348c963f"
    },
    {
      "path": "init_kwargs.json",
      "bytes": 45883,
      "sha256": "20dba3350b336a6a89608645695618bd883df4c02aadbecd437e2971423616ed"
    },
    {
      "path": "model/config.json",
      "bytes": 3807,
      "sha256": "52f94d2946ed8912fc12ac9c25b4bf0e68ccdc669a05ea104e8b6c178e91fb46"
    },
    {
      "path": "model/init_kwargs.json",
      "bytes": 45883,
      "sha256": "20dba3350b336a6a89608645695618bd883df4c02aadbecd437e2971423616ed"
    },
    {
      "path": "model/model.safetensors",
      "bytes": 3343756272,
      "sha256": "8308bae87e57c7a9add28109d7bdfcf33e9da90c49c5501453907a5c465e75fd"
    },
    {
      "path": "optimizers.pth",
      "bytes": 3481181041,
      "sha256": "5f11027f93401320337817b959507e66411be77d6f9ee10d7637b3630fcf0265"
    },
    {
      "path": "replay/array_000.npy",
      "bytes": 79134848,
      "sha256": "88434514aed73bed91cebc3b607827c7ac21222d70e669551c5c09a13bbed504"
    },
    {
      "path": "replay/array_001.npy",
      "bytes": 3440768,
      "sha256": "3c4e3a9223fc77f98adf782577599418af4a23ea3dcd4fa5085b96e6a99157da"
    },
    {
      "path": "replay/array_002.npy",
      "bytes": 3440768,
      "sha256": "862ad3c71076afbaec4a17841c8acdd23f184b6628eddfdcd797ecd33665a413"
    },
    {
      "path": "replay/array_003.npy",
      "bytes": 3440768,
      "sha256": "1d11af5c57ba70849e0143fb20d9ab15e3bb932a626014b6027d0ac584dabd7f"
    },
    {
      "path": "replay/array_004.npy",
      "bytes": 3440768,
      "sha256": "d382cd1e020e8ccd56c8aaac07170e1554163aeb55ecab5bf41147a52dbe999a"
    },
    {
      "path": "replay/array_005.npy",
      "bytes": 3440768,
      "sha256": "ab8e57d50f0f2d55a3c22e040919d4276753dfc30e727053a40fa9d01eab0cd8"
    },
    {
      "path": "replay/array_006.npy",
      "bytes": 3440768,
      "sha256": "66e8fe76ed750b3c5a40cf7030796b4e6c0669ecdf43b4575237695c44fe3c99"
    },
    {
      "path": "replay/array_007.npy",
      "bytes": 3440768,
      "sha256": "c0263159e5ca6d2ccbf05c54bf4b7de41c2d6692ec3a570d995d78aea5d42962"
    },
    {
      "path": "replay/array_008.npy",
      "bytes": 3440768,
      "sha256": "7480a31190b47f7c43afb92431e6ad8a2f0550479283c160ef995a7150ab6f96"
    },
    {
      "path": "replay/array_009.npy",
      "bytes": 3440768,
      "sha256": "14cd4be994aa58a9bb2169671ad21120aaef95c770186679e7760f6777d25109"
    },
    {
      "path": "replay/array_010.npy",
      "bytes": 6881408,
      "sha256": "36b9f69250b748a7f6a9b3f6c3bb31e501748bbde9d82732b4bbc02c9f557774"
    },
    {
      "path": "replay/array_011.npy",
      "bytes": 1032192128,
      "sha256": "dff429b75427d966f484466c2b9669c773685c348dad8a0e5d63fe06c0de67cc"
    },
    {
      "path": "replay/array_012.npy",
      "bytes": 79134848,
      "sha256": "884c8ad045b951ef9f8f29049fedc334cb44b1edbaec133b468e4172a339f7e9"
    },
    {
      "path": "replay/array_013.npy",
      "bytes": 1231749248,
      "sha256": "ca494535328782571018a1e4cf71d8702b512dc3468d5320708d39a9f8912d01"
    },
    {
      "path": "replay/array_014.npy",
      "bytes": 178913408,
      "sha256": "48b7cc938429ee7608348bf47da846aae1ea89e4cc2c092d527510f09f3a1510"
    },
    {
      "path": "replay/array_015.npy",
      "bytes": 860288,
      "sha256": "6d5ad5052bcbb42776fffe32e7dc5eebae89ffa5284ff052ec8c5442a01f88a3"
    },
    {
      "path": "replay/array_016.npy",
      "bytes": 860288,
      "sha256": "31ba486699f2858be0234f3e959120e2c30c2eb7a9f8ff4d60dc548f01cb4e90"
    },
    {
      "path": "replay/array_017.npy",
      "bytes": 1032192128,
      "sha256": "6983622e7e711bc3172c811115545a46535152f45cb85313f81e958a8fb420b8"
    },
    {
      "path": "replay/array_018.npy",
      "bytes": 79134848,
      "sha256": "992a324592200cd4267c70ad6c22bd5a593d17722e960e74b25a880adc471dbc"
    },
    {
      "path": "replay/array_019.npy",
      "bytes": 1231749248,
      "sha256": "6d9886c29ab007323ec808b8d30618b500e950010e763182ae2aaa1b45a32ce1"
    },
    {
      "path": "replay/array_020.npy",
      "bytes": 178913408,
      "sha256": "f08bd0be9275d6e6b19edd33f19e54789f2f1412fc44afed54196363ef4c7a2d"
    },
    {
      "path": "replay/array_021.npy",
      "bytes": 6881408,
      "sha256": "25805036cef312d11ecfb37fc73778886f63caf007db4724d99576cd1045ada9"
    },
    {
      "path": "replay/array_022.npy",
      "bytes": 6881408,
      "sha256": "0621d369a239c90009699ce7fa60a229c970af5d1b7304aee5f84ff5c7ef8cac"
    },
    {
      "path": "replay/array_023.npy",
      "bytes": 880803968,
      "sha256": "7decd6424b136c8e7665a1c03548a7f2aa6d3b70053ec4bcf91e92eeab364367"
    },
    {
      "path": "trainer-state.pt",
      "bytes": 903671441,
      "sha256": "b2b62e16eb5ec1ea29e64ec3e8513d9a455f01038c2243474af07cf936c93561"
    }
  ],
  "counters": {
    "learner_joint_updates": 524,
    "sampling_rounds": 35,
    "online_transitions": 860160,
    "next_trigger": 870400
  }
}
```

### 4M满池保存恢复RSS/时长

```json
{
  "status": "passed",
  "capacity": 4000000,
  "storage_bytes": 28168000000,
  "peak_rss_bytes": 28335329280,
  "wall_seconds": 85.49209015397355,
  "files": 24,
  "array_bytes": 28168003072,
  "next_sample_rng_match": true
}
```

### 5.12M满池保存恢复RSS/时长

```json
{
  "status": "passed",
  "capacity": 5120000,
  "storage_bytes": 36055040000,
  "peak_rss_bytes": 36208705536,
  "wall_seconds": 261.7296555240173,
  "files": 24,
  "array_bytes": 36055043072,
  "next_sample_rng_match": true
}
```

### 最终经验年龄/终止比例/版本分布

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
    "0.99": 38.0,
    "1": 74.0
  },
  "terminated_fraction": 0.05757068452380952,
  "truncated_fraction": 0.0,
  "rollout_states": {},
  "wall_time_age": "Qualification and formal Run clocks differ and per-transition wall times are not stored; do not infer exact age_seconds from replay count.",
  "physical_age_semantics": "age/6144 is equivalent total collection time per environment, not continuous episode age or bitwise physical resume"
}
```

### GPU与内存观测、时长

```json
{
  "train_started_at": "2026-10-12T05:52:45.930799+00:00",
  "train_finished_at": "2026-10-12T06:17:10.983341+00:00",
  "job_wall_seconds": 1465.052542,
  "learner_logged_wall_seconds": 1127.3336181640625,
  "cpu_limit": 24,
  "memory_limit_bytes": 103079215104,
  "shm_limit_bytes": 34359738368,
  "gpu_topology": {
    "0": "learner",
    "1": "2048env sampler seed42",
    "2": "2048env sampler seed43",
    "3": "2048env sampler seed44"
  },
  "gpu_observed_max_used_MB": {
    "0": 16710,
    "1": 7478,
    "2": 7525,
    "3": 7414
  },
  "gpu_observations_per_card": {
    "0": 37,
    "1": 37,
    "2": 37,
    "3": 37
  },
  "gpu_measurement_scope": "Doctor whole-card total-free, includes other pre-existing driver/display usage; sampled ~32 seconds; not torch allocated/reserved peaks",
  "container_memory_observations": [
    {
      "file": "/work/tracking-contact-kd524-resource-loaded.json",
      "working_set_bytes_approx": 38010460569.6,
      "measurement": {
        "BlockIO": "1.74GB / 324MB",
        "CPUPerc": "268.84%",
        "Container": "eb-run-run_1299df81ef5740f99f23",
        "ID": "cdc97ee9325d",
        "MemPerc": "36.88%",
        "MemUsage": "35.4GiB / 96GiB",
        "Name": "eb-run-run_1299df81ef5740f99f23",
        "NetIO": "936B / 126B",
        "PIDs": "4964"
      }
    },
    {
      "file": "/work/tracking-contact-kd524-resource-saving.json",
      "working_set_bytes_approx": 52377126174.72,
      "measurement": {
        "BlockIO": "1.84GB / 1.91GB",
        "CPUPerc": "233.35%",
        "Container": "eb-run-run_1299df81ef5740f99f23",
        "ID": "cdc97ee9325d",
        "MemPerc": "50.81%",
        "MemUsage": "48.78GiB / 96GiB",
        "Name": "eb-run-run_1299df81ef5740f99f23",
        "NetIO": "1.08kB / 126B",
        "PIDs": "5635"
      }
    },
    {
      "file": "/work/tracking-contact-kd524-resource-training1.json",
      "working_set_bytes_approx": 43024834887.68,
      "measurement": {
        "BlockIO": "1.84GB / 596MB",
        "CPUPerc": "453.43%",
        "Container": "eb-run-run_1299df81ef5740f99f23",
        "ID": "cdc97ee9325d",
        "MemPerc": "41.74%",
        "MemUsage": "40.07GiB / 96GiB",
        "Name": "eb-run-run_1299df81ef5740f99f23",
        "NetIO": "936B / 126B",
        "PIDs": "5635"
      }
    },
    {
      "file": "/work/tracking-contact-kd524-resource-training2.json",
      "working_set_bytes_approx": 50261854781.44,
      "measurement": {
        "BlockIO": "1.84GB / 1.6GB",
        "CPUPerc": "229.17%",
        "Container": "eb-run-run_1299df81ef5740f99f23",
        "ID": "cdc97ee9325d",
        "MemPerc": "48.76%",
        "MemUsage": "46.81GiB / 96GiB",
        "Name": "eb-run-run_1299df81ef5740f99f23",
        "NetIO": "1.08kB / 126B",
        "PIDs": "5636"
      }
    }
  ],
  "container_max_observed_working_set_bytes_approx": 52377126174.72,
  "full_pool_estimated_working_set_bytes": 74487879454.72,
  "full_pool_estimate_with20percent_margin_bytes": 89385455345.664,
  "full_pool_estimate_fits96GiB": true,
  "estimate_scope": "Observed combined run plus remaining 4M replay physical pages; conservative capacity estimate, not measured simultaneous full-pool GPU save/restore peak",
  "phase_timing_statistics": {
    "time/sync_weights": {
      "n": 32,
      "first_step": 40,
      "last_step": 524,
      "first": 18.360673904418945,
      "last": 16.87397575378418,
      "mean": 16.700024038553238,
      "min": 15.91318130493164,
      "max": 18.360673904418945,
      "median": 16.699763298034668,
      "p95": 17.3745397567749,
      "last50_mean": 16.700024038553238,
      "windows": [
        {
          "start": 25,
          "end": 124,
          "n": 6,
          "mean": 16.88003381093343
        },
        {
          "start": 125,
          "end": 224,
          "n": 6,
          "mean": 16.538954734802246
        },
        {
          "start": 225,
          "end": 324,
          "n": 6,
          "mean": 16.407757600148518
        },
        {
          "start": 325,
          "end": 424,
          "n": 7,
          "mean": 16.739424296787806
        },
        {
          "start": 425,
          "end": 524,
          "n": 7,
          "mean": 16.8949031829834
        }
      ]
    },
    "time/generate_rollouts": {
      "n": 32,
      "first_step": 40,
      "last_step": 524,
      "first": 6.112937927246094,
      "last": 3.1502182483673096,
      "mean": 3.2502525448799133,
      "min": 2.834174871444702,
      "max": 6.112937927246094,
      "median": 3.1868412494659424,
      "p95": 3.4105478286743165,
      "last50_mean": 3.2502525448799133,
      "windows": [
        {
          "start": 25,
          "end": 124,
          "n": 6,
          "mean": 3.733925779660543
        },
        {
          "start": 125,
          "end": 224,
          "n": 6,
          "mean": 3.120744268099467
        },
        {
          "start": 225,
          "end": 324,
          "n": 6,
          "mean": 3.21990700562795
        },
        {
          "start": 325,
          "end": 424,
          "n": 7,
          "mean": 3.0579220226832797
        },
        {
          "start": 425,
          "end": 524,
          "n": 7,
          "mean": 3.1650235652923584
        }
      ]
    },
    "time/cal_adv_and_returns": {
      "n": 32,
      "first_step": 40,
      "last_step": 524,
      "first": 0.002427339553833008,
      "last": 0.009396791458129883,
      "mean": 0.005837783217430115,
      "min": 0.002427339553833008,
      "max": 0.018366575241088867,
      "median": 0.003516674041748047,
      "p95": 0.011518454551696778,
      "last50_mean": 0.005837783217430115,
      "windows": [
        {
          "start": 25,
          "end": 124,
          "n": 6,
          "mean": 0.005168755849202474
        },
        {
          "start": 125,
          "end": 224,
          "n": 6,
          "mean": 0.005256255467732747
        },
        {
          "start": 225,
          "end": 324,
          "n": 6,
          "mean": 0.004845420519510905
        },
        {
          "start": 325,
          "end": 424,
          "n": 7,
          "mean": 0.006793430873325893
        },
        {
          "start": 425,
          "end": 524,
          "n": 7,
          "mean": 0.006804636546543666
        }
      ]
    },
    "time/actor_training": {
      "n": 32,
      "first_step": 40,
      "last_step": 524,
      "first": 23.511960983276367,
      "last": 4.317049026489258,
      "mean": 12.557043433189392,
      "min": 4.317049026489258,
      "max": 23.511960983276367,
      "median": 12.439062595367432,
      "p95": 12.691028499603272,
      "last50_mean": 12.557043433189392,
      "windows": [
        {
          "start": 25,
          "end": 124,
          "n": 6,
          "mean": 14.266781489054361
        },
        {
          "start": 125,
          "end": 224,
          "n": 6,
          "mean": 12.51085122426351
        },
        {
          "start": 225,
          "end": 324,
          "n": 6,
          "mean": 12.476491610209147
        },
        {
          "start": 325,
          "end": 424,
          "n": 7,
          "mean": 12.466438020978655
        },
        {
          "start": 425,
          "end": 524,
          "n": 7,
          "mean": 11.290796824863978
        }
      ]
    },
    "time/step": {
      "n": 32,
      "first_step": 40,
      "last_step": 524,
      "first": 47.988704681396484,
      "last": 24.35120391845703,
      "mean": 32.51367253065109,
      "min": 24.35120391845703,
      "max": 47.988704681396484,
      "median": 32.31250190734863,
      "p95": 32.9925142288208,
      "last50_mean": 32.51367253065109,
      "windows": [
        {
          "start": 25,
          "end": 124,
          "n": 6,
          "mean": 34.8864377339681
        },
        {
          "start": 125,
          "end": 224,
          "n": 6,
          "mean": 32.176317850748696
        },
        {
          "start": 225,
          "end": 324,
          "n": 6,
          "mean": 32.109496434529625
        },
        {
          "start": 325,
          "end": 424,
          "n": 7,
          "mean": 32.271069390433176
        },
        {
          "start": 425,
          "end": 524,
          "n": 7,
          "mean": 31.358074733189174
        }
      ]
    },
    "time/env/env/bootstrap_step": {
      "n": 32,
      "first_step": 40,
      "last_step": 524,
      "first": 0.023250972852110863,
      "last": 0.00024497206322848797,
      "mean": 0.0009994724750868045,
      "min": 0.0002376509364694357,
      "max": 0.023250972852110863,
      "median": 0.00027083803433924913,
      "p95": 0.00035197492688894273,
      "last50_mean": 0.0009994724750868045,
      "windows": [
        {
          "start": 25,
          "end": 124,
          "n": 6,
          "mean": 0.0041116179975991445
        },
        {
          "start": 125,
          "end": 224,
          "n": 6,
          "mean": 0.0002804376417770982
        },
        {
          "start": 225,
          "end": 324,
          "n": 6,
          "mean": 0.00028087229778369266
        },
        {
          "start": 325,
          "end": 424,
          "n": 7,
          "mean": 0.0002812779774623258
        },
        {
          "start": 425,
          "end": 524,
          "n": 7,
          "mean": 0.00028237224822597843
        }
      ]
    },
    "time/env/compute_bootstrap_rewards": {
      "n": 32,
      "first_step": 40,
      "last_step": 524,
      "first": 0.00025673210620880127,
      "last": 0.00023362995125353336,
      "mean": 0.00025237772206310183,
      "min": 0.00022323103621602058,
      "max": 0.00035252771340310574,
      "median": 0.00024841714184731245,
      "p95": 0.0002918244805186987,
      "last50_mean": 0.00025237772206310183,
      "windows": [
        {
          "start": 25,
          "end": 124,
          "n": 6,
          "mean": 0.00025334267411381006
        },
        {
          "start": 125,
          "end": 224,
          "n": 6,
          "mean": 0.00027097905209908885
        },
        {
          "start": 225,
          "end": 324,
          "n": 6,
          "mean": 0.00023431986725578705
        },
        {
          "start": 325,
          "end": 424,
          "n": 7,
          "mean": 0.00024949606241924424
        },
        {
          "start": 425,
          "end": 524,
          "n": 7,
          "mean": 0.000253966444038919
        }
      ]
    },
    "time/env/env_interact_step": {
      "n": 32,
      "first_step": 40,
      "last_step": 524,
      "first": 3.1464943885803223,
      "last": 1.7602533102035522,
      "mean": 1.773913148790598,
      "min": 1.473757266998291,
      "max": 3.1464943885803223,
      "median": 1.7242864966392517,
      "p95": 2.099747288227081,
      "last50_mean": 1.773913148790598,
      "windows": [
        {
          "start": 25,
          "end": 124,
          "n": 6,
          "mean": 1.8924379150072734
        },
        {
          "start": 125,
          "end": 224,
          "n": 6,
          "mean": 1.64904781182607
        },
        {
          "start": 225,
          "end": 324,
          "n": 6,
          "mean": 1.9441030025482178
        },
        {
          "start": 325,
          "end": 424,
          "n": 7,
          "mean": 1.7144658735820226
        },
        {
          "start": 425,
          "end": 524,
          "n": 7,
          "mean": 1.6929181814193726
        }
      ]
    },
    "time/env/env/send_rollout_trajectories": {
      "n": 32,
      "first_step": 40,
      "last_step": 524,
      "first": 0.28675612807273865,
      "last": 0.21294555068016052,
      "mean": 0.22611520905047655,
      "min": 0.20494399964809418,
      "max": 0.28675612807273865,
      "median": 0.22414454817771912,
      "p95": 0.24717386364936828,
      "last50_mean": 0.22611520905047655,
      "windows": [
        {
          "start": 25,
          "end": 124,
          "n": 6,
          "mean": 0.24177114913860956
        },
        {
          "start": 125,
          "end": 224,
          "n": 6,
          "mean": 0.23046727230151495
        },
        {
          "start": 225,
          "end": 324,
          "n": 6,
          "mean": 0.22477470835049948
        },
        {
          "start": 325,
          "end": 424,
          "n": 7,
          "mean": 0.21919154907975877
        },
        {
          "start": 425,
          "end": 524,
          "n": 7,
          "mean": 0.21703815247331346
        }
      ]
    },
    "time/env/run_interact_once": {
      "n": 32,
      "first_step": 40,
      "last_step": 524,
      "first": 5.719757556915283,
      "last": 3.0007362365722656,
      "mean": 3.0484673604369164,
      "min": 2.653312921524048,
      "max": 5.719757556915283,
      "median": 2.9918785095214844,
      "p95": 3.193458378314972,
      "last50_mean": 3.0484673604369164,
      "windows": [
        {
          "start": 25,
          "end": 124,
          "n": 6,
          "mean": 3.500538428624471
        },
        {
          "start": 125,
          "end": 224,
          "n": 6,
          "mean": 2.9182862838109336
        },
        {
          "start": 225,
          "end": 324,
          "n": 6,
          "mean": 3.0329221884409585
        },
        {
          "start": 325,
          "end": 424,
          "n": 7,
          "mean": 2.868745803833008
        },
        {
          "start": 425,
          "end": 524,
          "n": 7,
          "mean": 2.9656076431274414
        }
      ]
    },
    "time/env/interact": {
      "n": 32,
      "first_step": 40,
      "last_step": 524,
      "first": 5.71982479095459,
      "last": 3.0007615089416504,
      "mean": 3.0484967455267906,
      "min": 2.653339385986328,
      "max": 5.71982479095459,
      "median": 2.9919134378433228,
      "p95": 3.193484199047089,
      "last50_mean": 3.0484967455267906,
      "windows": [
        {
          "start": 25,
          "end": 124,
          "n": 6,
          "mean": 3.5005785624186196
        },
        {
          "start": 125,
          "end": 224,
          "n": 6,
          "mean": 2.9183130264282227
        },
        {
          "start": 225,
          "end": 324,
          "n": 6,
          "mean": 3.032948891321818
        },
        {
          "start": 325,
          "end": 424,
          "n": 7,
          "mean": 2.868774243763515
        },
        {
          "start": 425,
          "end": 524,
          "n": 7,
          "mean": 2.9656333242143904
        }
      ]
    },
    "time/rollout/sync_model_from_actor": {
      "n": 32,
      "first_step": 40,
      "last_step": 524,
      "first": 14.75439453125,
      "last": 13.418835639953613,
      "mean": 13.18848705291748,
      "min": 12.460832595825195,
      "max": 14.75439453125,
      "median": 13.201223850250244,
      "p95": 13.837196779251098,
      "last50_mean": 13.18848705291748,
      "windows": [
        {
          "start": 25,
          "end": 124,
          "n": 6,
          "mean": 13.322717825571695
        },
        {
          "start": 125,
          "end": 224,
          "n": 6,
          "mean": 13.061384995778402
        },
        {
          "start": 225,
          "end": 324,
          "n": 6,
          "mean": 12.830879052480062
        },
        {
          "start": 325,
          "end": 424,
          "n": 7,
          "mean": 13.253393036978585
        },
        {
          "start": 425,
          "end": 524,
          "n": 7,
          "mean": 13.423991884504046
        }
      ]
    },
    "time/rollout/predict": {
      "n": 32,
      "first_step": 40,
      "last_step": 524,
      "first": 1.1205333471298218,
      "last": 0.15980057418346405,
      "mean": 0.2112075868062675,
      "min": 0.11632955819368362,
      "max": 1.1205333471298218,
      "median": 0.17279541492462158,
      "p95": 0.2849878817796707,
      "last50_mean": 0.2112075868062675,
      "windows": [
        {
          "start": 25,
          "end": 124,
          "n": 6,
          "mean": 0.4086846436063449
        },
        {
          "start": 125,
          "end": 224,
          "n": 6,
          "mean": 0.22334039956331253
        },
        {
          "start": 225,
          "end": 324,
          "n": 6,
          "mean": 0.16891180103023848
        },
        {
          "start": 325,
          "end": 424,
          "n": 7,
          "mean": 0.1388257901583399
        },
        {
          "start": 425,
          "end": 524,
          "n": 7,
          "mean": 0.14017731164182937
        }
      ]
    },
    "time/rollout/generate_one_epoch": {
      "n": 32,
      "first_step": 40,
      "last_step": 524,
      "first": 5.437313079833984,
      "last": 2.785778522491455,
      "mean": 2.8093855753540993,
      "min": 2.406275987625122,
      "max": 5.437313079833984,
      "median": 2.744369626045227,
      "p95": 2.9622196793556212,
      "last50_mean": 2.8093855753540993,
      "windows": [
        {
          "start": 25,
          "end": 124,
          "n": 6,
          "mean": 3.22811222076416
        },
        {
          "start": 125,
          "end": 224,
          "n": 6,
          "mean": 2.6871721744537354
        },
        {
          "start": 225,
          "end": 324,
          "n": 6,
          "mean": 2.8077552318573
        },
        {
          "start": 325,
          "end": 424,
          "n": 7,
          "mean": 2.6289823395865306
        },
        {
          "start": 425,
          "end": 524,
          "n": 7,
          "mean": 2.737032038824899
        }
      ]
    },
    "time/rollout/rollout/generate": {
      "n": 32,
      "first_step": 40,
      "last_step": 524,
      "first": 5.443971157073975,
      "last": 2.785881996154785,
      "mean": 2.809846267104149,
      "min": 2.4064080715179443,
      "max": 5.443971157073975,
      "median": 2.7444803714752197,
      "p95": 2.9626498222351074,
      "last50_mean": 2.809846267104149,
      "windows": [
        {
          "start": 25,
          "end": 124,
          "n": 6,
          "mean": 3.229430158933004
        },
        {
          "start": 125,
          "end": 224,
          "n": 6,
          "mean": 2.687533696492513
        },
        {
          "start": 225,
          "end": 324,
          "n": 6,
          "mean": 2.807858943939209
        },
        {
          "start": 325,
          "end": 424,
          "n": 7,
          "mean": 2.629199062074934
        },
        {
          "start": 425,
          "end": 524,
          "n": 7,
          "mean": 2.7373929023742676
        }
      ]
    }
  },
  "limitations": [
    "No kernel memory.peak or complete process RSS peak",
    "No complete torch allocated/reserved time series",
    "No precisely isolated fsync/hash/publication I/O stages"
  ],
  "wall_budget_seconds": 7200,
  "formal_new_transition_budget": 1000000,
  "formal_new_transitions": 786432
}
```

### 新产物预算账目

```json
{
  "scope": "new diagnostic, failed seed qualification, corrected16/24 and formal500; old models and data preserved",
  "roots_apparent_bytes": {
    "/data/kk/embodiedbein/runs/run_1299df81ef5740f99f23": 758649938,
    "/data/kk/embodiedbein/runs/run_53f27bf937f842b3b248": 68380610,
    "/data/kk/embodiedbein/runs/run_69b0d90a9d0d453391c3": 256154,
    "/data/kk/embodiedbein/runs/run_85ae9116da5549e98143": 48959313,
    "/data/kk/embodiedbein/runs/run_9b47b8e1134b4fb39c0a": 23742440,
    "/data/kk/models/runs/run_1299df81ef5740f99f23": 27571985451,
    "/data/kk/models/runs/run_85ae9116da5549e98143": 16149739973,
    "/data/kk/models/runs/run_9b47b8e1134b4fb39c0a": 16495877086
  },
  "allocated_unique_bytes": 31009697792,
  "apparent_bytes": 61117590965,
  "scratch_file_bytes": 12625987,
  "image_and_staging_reserved_bytes": 51539607552,
  "budget_bytes": 206158430208,
  "within_budget": true,
  "full_pool_temporary_replay_storage_bytes_removed_after_verified_restore": 64223040000
}
```

正式新增786432，累计860160，4M使用约21.5%，FIFO淘汰0。资格和500均未超过旧1M容量；扩容主要完成后续资源/恢复准备，本轮不能证实保留早期样本或阻止淘汰的学习收益。4M数组28.168GB、5.12M36.05504GB，实际满池保存/读回峰RSS均实测且下次采样RNG一致；CPU测试临时64.22304GB数组在验证后清理，仅清自己临时文件，旧数据/模型保留。

GPU显存为定时doctor整卡占用观测（含原有驱动/显示基线），Docker stats为采样时的cgroup工作集，不等同进程RSS峰；未采集完整PyTorch allocated/reserved峰、kernel cgroup memory.peak和逐阶段精确I/O峰。Replay墙钟年龄不在逐转移记录中，跨资格/正式时钟不能机械等同采集秒数；提供transition/等效控制步/物理秒/策略版本年龄分位数，未虚构精确墙钟年龄。

## 8. 每round原始训练观测

| 累计联合更新 | env/episode_len | train/ufo/strict_shadow_episode_len | train/ufo/online_transitions | train/ufo/first_step_samples | train/ufo/first_step_terminated | train/ufo/hard_velocity_count | train/ufo/training_velocity_count | train/ufo/strict_velocity_exposure_fraction | train/ufo/physics_peak_joint_velocity | train/ufo/foot_contact_peak_N | train/ufo/tracking_q_rmse_rad |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 40 | 3.47992349 | 3.28871894 | 98304 | 6531 | 7 | 1147 | 147 | 0.000324857421 | 32.7114105 | 13535.9482 | 0.083344698 |
| 56 | 6.39905882 | 5.65141153 | 122880 | 2295 | 1 | 701 | 44 | 0.000333924225 | 25.0334682 | 12253.6123 | 0.29580763 |
| 72 | 9.02214432 | 8.35508919 | 147456 | 2888 | 2 | 709 | 80 | 0.000235958374 | 44.0882607 | 15465.3154 | 0.298297435 |
| 88 | 9.89025688 | 8.83641052 | 172032 | 1779 | 3 | 616 | 64 | 0.000215613327 | 31.1901169 | 16381.2832 | 0.348008543 |
| 104 | 10.1954699 | 8.96518421 | 196608 | 2239 | 5 | 809 | 80 | 0.000266033661 | 35.598381 | 14496.2295 | 0.398087978 |
| 120 | 10.5402298 | 9.11111069 | 221184 | 2495 | 2 | 1133 | 104 | 0.000457763672 | 34.8754807 | 14357.0068 | 0.362849981 |
| 136 | 10.0696812 | 8.49653912 | 245760 | 2173 | 3 | 1363 | 160 | 0.000561036926 | 36.4195442 | 16926.252 | 0.348961353 |
| 152 | 10.5594273 | 8.73031044 | 270336 | 2088 | 2 | 1169 | 131 | 0.000409333588 | 32.1656761 | 14509.9795 | 0.365412802 |
| 168 | 10.9599133 | 8.8591547 | 294912 | 1943 | 1 | 996 | 121 | 0.000352057861 | 53.0160332 | 14660.6064 | 0.36691913 |
| 184 | 11.8407984 | 9.18456554 | 319488 | 1851 | 2 | 908 | 91 | 0.000321097992 | 52.809845 | 15729.8447 | 0.387392759 |
| 200 | 12.5779819 | 9.75229359 | 344064 | 1840 | 2 | 956 | 107 | 0.000331270508 | 32.9265213 | 20959.25 | 0.374807864 |
| 216 | 13.341176 | 10.2540112 | 368640 | 1884 | 2 | 845 | 94 | 0.000296772399 | 27.9661961 | 17992.1191 | 0.367310852 |
| 232 | 13.0418406 | 10.000598 | 393216 | 1714 | 0 | 915 | 104 | 0.00031645401 | 51.3038254 | 15407.248 | 0.357886881 |
| 248 | 14.4517765 | 11.2360411 | 417792 | 1601 | 0 | 879 | 84 | 0.000330828247 | 52.1282806 | 16316.5898 | 0.370240211 |
| 264 | 14.7160139 | 11.2448397 | 442368 | 1449 | 1 | 818 | 80 | 0.000298099243 | 36.4800339 | 17753.7363 | 0.378615707 |
| 280 | 15.2240181 | 11.9037724 | 466944 | 1317 | 0 | 811 | 96 | 0.000300531799 | 39.0677299 | 16252.7803 | 0.401193231 |
| 296 | 16.4532928 | 12.6730471 | 491520 | 1291 | 0 | 814 | 74 | 0.000318886567 | 39.4236603 | 13125.4854 | 0.414422363 |
| 312 | 17.2800312 | 13.5967865 | 516096 | 1326 | 1 | 779 | 81 | 0.000289916992 | 44.3864136 | 16221.4277 | 0.412988544 |
| 328 | 17.181673 | 13.4908371 | 540672 | 1284 | 1 | 744 | 84 | 0.000275542785 | 37.2562027 | 13262.375 | 0.405352563 |
| 344 | 18.137989 | 13.7015791 | 565248 | 1202 | 0 | 760 | 80 | 0.000275985047 | 33.3593178 | 14301.3477 | 0.413368583 |
| 360 | 18.6919231 | 14.0774355 | 589824 | 1202 | 1 | 728 | 64 | 0.000262495392 | 42.2018814 | 13360.1045 | 0.424325973 |
| 376 | 19.6331844 | 14.836771 | 614400 | 1112 | 0 | 736 | 78 | 0.000270456512 | 34.9085388 | 14964.8896 | 0.434630245 |
| 392 | 20.2395649 | 15.5390196 | 638976 | 1107 | 2 | 744 | 69 | 0.000275985047 | 37.0790062 | 16828.5547 | 0.448624134 |
| 408 | 19.4064751 | 15.1519785 | 663552 | 1125 | 3 | 763 | 70 | 0.000260505127 | 34.9536591 | 15828.0596 | 0.437666416 |
| 424 | 21.26087 | 16.345108 | 688128 | 1135 | 0 | 750 | 58 | 0.000259620545 | 55.7631607 | 15327.9209 | 0.436256796 |
| 440 | 21.1206894 | 15.805172 | 712704 | 1133 | 0 | 702 | 59 | 0.000256303436 | 25.0010967 | 15824.251 | 0.430669874 |
| 456 | 22.2661133 | 15.7836094 | 737280 | 1095 | 0 | 758 | 57 | 0.000278859894 | 34.5116577 | 16563.6719 | 0.421893328 |
| 472 | 22.3339596 | 16.093132 | 761856 | 1087 | 0 | 672 | 54 | 0.000230429825 | 31.6435852 | 12977.1221 | 0.420851469 |
| 488 | 22.4058228 | 15.8732872 | 786432 | 1134 | 3 | 695 | 65 | 0.000244361756 | 28.4164066 | 16820.043 | 0.420977265 |
| 504 | 23.3187885 | 16.7791634 | 811008 | 1124 | 3 | 768 | 61 | 0.000279523316 | 34.0604973 | 11701.1162 | 0.421575069 |
| 520 | 22.6524334 | 16.0754719 | 835584 | 1046 | 0 | 717 | 55 | 0.000251880585 | 27.900341 | 14839.9844 | 0.414586544 |
| 524 | 22.0957165 | 15.6891518 | 860160 | 1034 | 1 | 683 | 57 | 0.000255197723 | 39.9507217 | 18838.4902 | 0.425170273 |

## 9. 全部151项标量统计

| tag | n | first_step | last_step | first | last | mean | min | max | median | p95 | last50_mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| env/episode_len | 32 | 40 | 524 | 3.47992349 | 22.0957165 | 15.6496324 | 3.47992349 | 23.3187885 | 15.8386555 | 22.5167975 | 15.6496324 |
| env/num_trajectories | 32 | 40 | 524 | 1046 | 1097 | 1521.46875 | 1007 | 2574 | 1302.5 | 2364.75 | 1521.46875 |
| env/return | 32 | 40 | 524 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| env/reward | 32 | 40 | 524 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| env/success_once | 32 | 40 | 524 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| rollout/ufo/received_transitions | 32 | 40 | 524 | 24576 | 24576 | 24576 | 24576 | 24576 | 24576 | 24576 | 24576 |
| time/actor_training | 32 | 40 | 524 | 23.511961 | 4.31704903 | 12.5570434 | 4.31704903 | 23.511961 | 12.4390626 | 12.6910285 | 12.5570434 |
| time/cal_adv_and_returns | 32 | 40 | 524 | 0.00242733955 | 0.00939679146 | 0.00583778322 | 0.00242733955 | 0.0183665752 | 0.00351667404 | 0.0115184546 | 0.00583778322 |
| time/env/compute_bootstrap_rewards | 32 | 40 | 524 | 0.000256732106 | 0.000233629951 | 0.000252377722 | 0.000223231036 | 0.000352527713 | 0.000248417142 | 0.000291824481 | 0.000252377722 |
| time/env/env/bootstrap_step | 32 | 40 | 524 | 0.0232509729 | 0.000244972063 | 0.000999472475 | 0.000237650936 | 0.0232509729 | 0.000270838034 | 0.000351974927 | 0.000999472475 |
| time/env/env/send_rollout_trajectories | 32 | 40 | 524 | 0.286756128 | 0.212945551 | 0.226115209 | 0.204944 | 0.286756128 | 0.224144548 | 0.247173864 | 0.226115209 |
| time/env/env_interact_step | 32 | 40 | 524 | 3.14649439 | 1.76025331 | 1.77391315 | 1.47375727 | 3.14649439 | 1.7242865 | 2.09974729 | 1.77391315 |
| time/env/interact | 32 | 40 | 524 | 5.71982479 | 3.00076151 | 3.04849675 | 2.65333939 | 5.71982479 | 2.99191344 | 3.1934842 | 3.04849675 |
| time/env/run_interact_once | 32 | 40 | 524 | 5.71975756 | 3.00073624 | 3.04846736 | 2.65331292 | 5.71975756 | 2.99187851 | 3.19345838 | 3.04846736 |
| time/generate_rollouts | 32 | 40 | 524 | 6.11293793 | 3.15021825 | 3.25025254 | 2.83417487 | 6.11293793 | 3.18684125 | 3.41054783 | 3.25025254 |
| time/rollout/generate_one_epoch | 32 | 40 | 524 | 5.43731308 | 2.78577852 | 2.80938558 | 2.40627599 | 5.43731308 | 2.74436963 | 2.96221968 | 2.80938558 |
| time/rollout/predict | 32 | 40 | 524 | 1.12053335 | 0.159800574 | 0.211207587 | 0.116329558 | 1.12053335 | 0.172795415 | 0.284987882 | 0.211207587 |
| time/rollout/rollout/generate | 32 | 40 | 524 | 5.44397116 | 2.785882 | 2.80984627 | 2.40640807 | 5.44397116 | 2.74448037 | 2.96264982 | 2.80984627 |
| time/rollout/sync_model_from_actor | 32 | 40 | 524 | 14.7543945 | 13.4188356 | 13.1884871 | 12.4608326 | 14.7543945 | 13.2012239 | 13.8371968 | 13.1884871 |
| time/step | 32 | 40 | 524 | 47.9887047 | 24.3512039 | 32.5136725 | 24.3512039 | 47.9887047 | 32.3125019 | 32.9925142 | 32.5136725 |
| time/sync_weights | 32 | 40 | 524 | 18.3606739 | 16.8739758 | 16.700024 | 15.9131813 | 18.3606739 | 16.6997633 | 17.3745398 | 16.700024 |
| train/ufo/B | 500 | 25 | 524 | -0.00417393912 | 0.000469153398 | -0.00390412197 | -0.0108742919 | 0.00191060104 | -0.00350500003 | -0.000562338869 | -0.00149980646 |
| train/ufo/B_norm | 500 | 25 | 524 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 |
| train/ufo/F1 | 500 | 25 | 524 | 0.0735809654 | 0.0704369694 | 0.0371293737 | 0.017245397 | 0.0774196535 | 0.034273101 | 0.0599378565 | 0.0511528381 |
| train/ufo/M1 | 500 | 25 | 524 | 3.5345645 | 0.867229819 | 1.35342878 | -1.18137813 | 9.35030746 | 1.03851426 | 3.53533278 | 0.821921856 |
| train/ufo/Q1 | 500 | 25 | 524 | -3.81878138 | -4.5559268 | -3.21028767 | -4.71804094 | -2.14978719 | -3.1155082 | -2.30448842 | -4.28581176 |
| train/ufo/Q_aux | 500 | 25 | 524 | -4.65010452 | -7.50958538 | -5.93624405 | -7.72749853 | -4.44705486 | -5.76362562 | -4.71312981 | -7.43219223 |
| train/ufo/Q_discriminator | 500 | 25 | 524 | -4.08714724 | -4.53286552 | -3.34626137 | -4.86706543 | -2.1657145 | -3.26386189 | -2.4134859 | -4.38633862 |
| train/ufo/Q_fb | 500 | 25 | 524 | 0.0497281402 | 59.6927032 | 28.1770957 | -2.91337848 | 61.0202675 | 26.5410576 | 55.2815086 | 55.40684 |
| train/ufo/action_abs_mean | 32 | 40 | 524 | 0.928150773 | 0.84094739 | 0.895692054 | 0.84094739 | 0.936651349 | 0.896075875 | 0.932814968 | 0.895692054 |
| train/ufo/action_saturation_fraction | 32 | 40 | 524 | 0.713479757 | 0.469071984 | 0.594272444 | 0.461678773 | 0.713479757 | 0.591050327 | 0.698713726 | 0.594272444 |
| train/ufo/actor_loss | 500 | 25 | 524 | 3.00655389 | -40.8509521 | -17.9336809 | -41.7497253 | 6.07306671 | -17.4907484 | -0.201991892 | -37.0088013 |
| train/ufo/actor_optimizer/grad_clip_limit | 500 | 25 | 524 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| train/ufo/actor_optimizer/grad_norm_after_clip_l2 | 500 | 25 | 524 | 0.999999404 | 0.999999881 | 0.990693474 | 0.662348509 | 1 | 0.999999762 | 0.999999881 | 0.999999852 |
| train/ufo/actor_optimizer/grad_norm_l2 | 500 | 25 | 524 | 1.46993196 | 6.66781616 | 4.49699843 | 0.662348509 | 11.4620256 | 4.60601425 | 7.72124324 | 7.05258699 |
| train/ufo/actor_optimizer/lr | 500 | 25 | 524 | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 |
| train/ufo/actor_optimizer/parameter_delta_l2 | 500 | 25 | 524 | 0.417830408 | 0.353093266 | 0.340917172 | 0.274857581 | 0.454400748 | 0.336748645 | 0.411609952 | 0.329213147 |
| train/ufo/actor_reg_weight_applied | 500 | 25 | 524 | 10.2780724 | 50 | 34.3416705 | 9.50122833 | 50 | 35.5301971 | 50 | 50 |
| train/ufo/actor_reg_weight_raw | 500 | 25 | 524 | 10.2780724 | 68.4230499 | 36.7876518 | 9.50122833 | 70.1167908 | 35.5301971 | 64.4637756 | 64.4931442 |
| train/ufo/applied_action_abs_mean | 32 | 40 | 524 | 0.472175717 | 0.608914971 | 0.615893805 | 0.472175717 | 0.647956848 | 0.63524586 | 0.645895192 | 0.615893805 |
| train/ufo/applied_action_saturation_fraction | 32 | 40 | 524 | 0.102615133 | 0.245350718 | 0.259668544 | 0.102615133 | 0.303542882 | 0.28015402 | 0.298620968 | 0.259668544 |
| train/ufo/auxQ1 | 500 | 25 | 524 | -4.60413074 | -7.21637058 | -5.78600468 | -7.6233778 | -4.12399101 | -5.64006686 | -4.53114619 | -7.27114471 |
| train/ufo/aux_critic_loss | 500 | 25 | 524 | 2.06466413 | 1.33265567 | 1.18073727 | 0.794096768 | 3.17895865 | 1.14197099 | 1.68496956 | 1.3541215 |
| train/ufo/aux_critic_optimizer/grad_clip_limit | 500 | 25 | 524 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 |
| train/ufo/aux_critic_optimizer/grad_norm_after_clip_l2 | 500 | 25 | 524 | 4.99999905 | 5 | 4.8202448 | 1.49385881 | 5.00000095 | 4.99999952 | 5 | 4.96957803 |
| train/ufo/aux_critic_optimizer/grad_norm_l2 | 500 | 25 | 524 | 23.1611252 | 9.97381496 | 11.5147637 | 1.49385881 | 53.0566673 | 9.9626503 | 25.2546474 | 8.97086674 |
| train/ufo/aux_critic_optimizer/lr | 500 | 25 | 524 | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 |
| train/ufo/aux_critic_optimizer/parameter_delta_l2 | 500 | 25 | 524 | 0.441335797 | 0.453075051 | 0.43949696 | 0.20916976 | 0.665640831 | 0.456599429 | 0.588592985 | 0.512391063 |
| train/ufo/aux_rew/limits_dof_pos | 500 | 25 | 524 | 0.0461485907 | 0.276194453 | 0.199270229 | 0.0442112274 | 0.299709022 | 0.212579153 | 0.288141674 | 0.287396663 |
| train/ufo/aux_rew/limits_dof_vel | 500 | 25 | 524 | 0.00501626916 | 0.0128023559 | 0.0134424606 | 0.00501626916 | 0.0181498863 | 0.0140680093 | 0.0159461817 | 0.0132136175 |
| train/ufo/aux_rew/limits_torque | 500 | 25 | 524 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| train/ufo/aux_rew/penalty_action_rate | 500 | 25 | 524 | 0.63536799 | 0.38758415 | 0.460939653 | 0.374922901 | 0.638697267 | 0.437435985 | 0.610250413 | 0.384763616 |
| train/ufo/aux_rew/penalty_ankle_roll | 500 | 25 | 524 | 0.0241785627 | 0.166928232 | 0.123173585 | 0.0241378807 | 0.174989551 | 0.140828833 | 0.169218714 | 0.167658659 |
| train/ufo/aux_rew/penalty_feet_ori | 500 | 25 | 524 | 0.113768257 | 0.199840635 | 0.157601859 | 0.0976313874 | 0.221331686 | 0.157972023 | 0.205511855 | 0.204688658 |
| train/ufo/aux_rew/penalty_slippage | 500 | 25 | 524 | 0.448643565 | 0.346823722 | 0.382839045 | 0.336302817 | 0.456962317 | 0.379903466 | 0.426021339 | 0.363528782 |
| train/ufo/aux_rew/penalty_torques | 500 | 25 | 524 | 4888.84424 | 5298.91455 | 5668.81927 | 4768.64551 | 6177.40186 | 5726.39331 | 6028.11968 | 5317.01745 |
| train/ufo/aux_rew/penalty_undesired_contact | 500 | 25 | 524 | 0.00146484375 | 0.00732421875 | 0.0070703125 | 0.00048828125 | 0.0151367188 | 0.0068359375 | 0.01171875 | 0.00590820312 |
| train/ufo/backward_optimizer/grad_clip_limit | 500 | 25 | 524 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 |
| train/ufo/backward_optimizer/grad_norm_after_clip_l2 | 500 | 25 | 524 | 5 | 5 | 5 | 4.99999905 | 5.00000095 | 5 | 5.00000048 | 5.00000001 |
| train/ufo/backward_optimizer/grad_norm_l2 | 500 | 25 | 524 | 548852.438 | 17359.9453 | 84724.5582 | 16794.3379 | 577446.062 | 41545.3906 | 324349.359 | 17994.0199 |
| train/ufo/backward_optimizer/lr | 500 | 25 | 524 | 9.99999975e-06 | 9.99999975e-06 | 9.99999975e-06 | 9.99999975e-06 | 9.99999975e-06 | 9.99999975e-06 | 9.99999975e-06 | 9.99999975e-06 |
| train/ufo/backward_optimizer/parameter_delta_l2 | 500 | 25 | 524 | 0.00332829379 | 0.0035523877 | 0.00407007324 | 0.00329788798 | 0.00444452232 | 0.00412114849 | 0.00442244513 | 0.00363995349 |
| train/ufo/collected_transitions | 32 | 40 | 524 | 24576 | 24576 | 24576 | 24576 | 24576 | 24576 | 24576 | 24576 |
| train/ufo/contact_termination_count | 32 | 40 | 524 | 372 | 104 | 218.8125 | 43 | 603 | 117 | 564.05 | 218.8125 |
| train/ufo/critic_loss | 500 | 25 | 524 | 1.60655117 | 0.534224987 | 0.43760654 | 0.21472998 | 1.86070204 | 0.367799938 | 1.02848197 | 0.462350021 |
| train/ufo/critic_optimizer/grad_clip_limit | 500 | 25 | 524 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 |
| train/ufo/critic_optimizer/grad_norm_after_clip_l2 | 500 | 25 | 524 | 5 | 4.99999952 | 4.81866378 | 1.76476312 | 5.00000095 | 4.99999952 | 5 | 4.95750554 |
| train/ufo/critic_optimizer/grad_norm_l2 | 500 | 25 | 524 | 15.8391218 | 7.96402025 | 11.0060637 | 1.76476312 | 54.6668243 | 9.04548883 | 25.4830115 | 8.27410788 |
| train/ufo/critic_optimizer/lr | 500 | 25 | 524 | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 |
| train/ufo/critic_optimizer/parameter_delta_l2 | 500 | 25 | 524 | 0.410245985 | 0.440660357 | 0.32812439 | 0.167622879 | 0.717133045 | 0.314761519 | 0.456584373 | 0.382310796 |
| train/ufo/disc_expert_loss | 500 | 25 | 524 | 0.585581899 | 0.243905634 | 0.333661779 | 0.223470032 | 0.585581899 | 0.311930716 | 0.544862762 | 0.242117458 |
| train/ufo/disc_loss | 500 | 25 | 524 | 3.54257536 | 0.800245881 | 1.40133804 | 0.772497892 | 3.54257536 | 1.1090436 | 3.04495721 | 0.80914809 |
| train/ufo/disc_reward_clip_fraction | 500 | 25 | 524 | 0 | 0.302734375 | 0.104598633 | 0 | 0.31640625 | 0.0749511719 | 0.287109375 | 0.288291016 |
| train/ufo/disc_train_loss | 500 | 25 | 524 | 0.747932732 | 0.314116925 | 0.488231089 | 0.289891839 | 0.752385199 | 0.468097523 | 0.729658645 | 0.31145251 |
| train/ufo/disc_wgan_gp_loss | 500 | 25 | 524 | 0.220906079 | 0.024222333 | 0.0579445176 | 0.0235670321 | 0.221511945 | 0.0320596062 | 0.17725674 | 0.0255578126 |
| train/ufo/discriminator_optimizer/grad_clip_limit | 500 | 25 | 524 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 |
| train/ufo/discriminator_optimizer/grad_norm_after_clip_l2 | 500 | 25 | 524 | 4.99999952 | 1.00927365 | 2.66648649 | 1.00927365 | 5.00000048 | 1.70077837 | 4.99999952 | 1.09625103 |
| train/ufo/discriminator_optimizer/grad_norm_l2 | 500 | 25 | 524 | 8.58319473 | 1.00927365 | 3.3449789 | 1.00927365 | 8.62033463 | 1.70077837 | 8.3055696 | 1.09625103 |
| train/ufo/discriminator_optimizer/lr | 500 | 25 | 524 | 3.00000011e-06 | 3.00000011e-06 | 3.00000011e-06 | 3.00000011e-06 | 3.00000011e-06 | 3.00000011e-06 | 3.00000011e-06 | 3.00000011e-06 |
| train/ufo/discriminator_optimizer/parameter_delta_l2 | 500 | 25 | 524 | 0.00442221994 | 0.00257814932 | 0.00373502419 | 0.00257814932 | 0.00536322268 | 0.00331137702 | 0.00529107261 | 0.00261248437 |
| train/ufo/environment_control_steps | 32 | 40 | 524 | 16 | 140 | 78 | 16 | 140 | 78 | 133.8 | 78 |
| train/ufo/expert_draws/g1 | 32 | 40 | 524 | 41104 | 536088 | 294789.25 | 41104 | 536088 | 295080 | 523470.4 | 294789.25 |
| train/ufo/expert_draws/official_rpo | 32 | 40 | 524 | 40816 | 537064 | 294266.75 | 40816 | 537064 | 294744 | 523467.2 | 294266.75 |
| train/ufo/fb_diag | 500 | 25 | 524 | -17.5880184 | -229.674301 | -162.852216 | -239.30426 | -6.42054367 | -191.919441 | -23.6527325 | -230.513388 |
| train/ufo/fb_loss | 500 | 25 | 524 | 270901.219 | 18504.043 | 61515.2803 | 17994.4316 | 271843.156 | 39624.9492 | 182727.445 | 19248.1123 |
| train/ufo/fb_offdiag | 500 | 25 | 524 | 257.70755 | 221.598267 | 153.931666 | 76.0381546 | 257.70755 | 155.444595 | 220.457865 | 218.532025 |
| train/ufo/first_step_samples | 32 | 40 | 524 | 6531 | 1034 | 1703.875 | 1034 | 6531 | 1321.5 | 2671.85 | 1703.875 |
| train/ufo/first_step_terminated | 32 | 40 | 524 | 7 | 1 | 1.5 | 0 | 7 | 1 | 3.9 | 1.5 |
| train/ufo/foot_contact_peak_N | 32 | 40 | 524 | 13535.9482 | 18838.4902 | 15430.7077 | 11701.1162 | 20959.25 | 15436.2817 | 18372.9861 | 15430.7077 |
| train/ufo/forward_optimizer/grad_clip_limit | 500 | 25 | 524 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 |
| train/ufo/forward_optimizer/grad_norm_after_clip_l2 | 500 | 25 | 524 | 5 | 5 | 5 | 4.99999905 | 5.00000095 | 5 | 5.00000048 | 5.00000007 |
| train/ufo/forward_optimizer/grad_norm_l2 | 500 | 25 | 524 | 4260.40674 | 112.814598 | 365.537166 | 93.9081879 | 4260.40674 | 149.706207 | 1451.99178 | 108.553583 |
| train/ufo/forward_optimizer/lr | 500 | 25 | 524 | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 |
| train/ufo/forward_optimizer/parameter_delta_l2 | 500 | 25 | 524 | 0.322039634 | 0.817753077 | 0.682004798 | 0.229204729 | 0.891043782 | 0.765199095 | 0.860128641 | 0.828434788 |
| train/ufo/ground_foot_margin_min_m | 32 | 40 | 524 | 193.832565 | 193.739441 | 193.671026 | 193.230087 | 193.973038 | 193.716545 | 193.920693 | 193.671026 |
| train/ufo/ground_outside_count | 32 | 40 | 524 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| train/ufo/hard_position_count | 32 | 40 | 524 | 579 | 324 | 490.03125 | 324 | 1035 | 413 | 969.9 | 490.03125 |
| train/ufo/hard_torque_count | 32 | 40 | 524 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| train/ufo/hard_velocity_count | 32 | 40 | 524 | 1147 | 683 | 830.875 | 616 | 1363 | 765.5 | 1156.9 | 830.875 |
| train/ufo/height_termination_count | 32 | 40 | 524 | 0 | 402 | 364.1875 | 0 | 636 | 348.5 | 598.75 | 364.1875 |
| train/ufo/latched_velocity_terms_without_boundary_violation | 32 | 40 | 524 | 996 | 570 | 690.1875 | 510 | 1106 | 627.5 | 990.5 | 690.1875 |
| train/ufo/learner_joint_updates | 500 | 25 | 524 | 25 | 524 | 274.5 | 25 | 524 | 274.5 | 499.05 | 499.5 |
| train/ufo/mean_aux_reward | 500 | 25 | 524 | -0.949797571 | -1.28823757 | -1.28103658 | -1.40221906 | -0.926295638 | -1.30862409 | -1.0473181 | -1.33609705 |
| train/ufo/mean_disc_reward | 500 | 25 | 524 | 0.0880542025 | -1.00685167 | -0.470011826 | -1.05394554 | 0.0973133966 | -0.512072712 | 0.0524512887 | -0.992658527 |
| train/ufo/mean_disc_reward_raw | 500 | 25 | 524 | 0.0880542025 | -1.25276661 | -0.53627217 | -1.29554617 | 0.0973133966 | -0.547993541 | 0.0524512887 | -1.20724374 |
| train/ufo/mean_next_Q | 500 | 25 | 524 | -2.72096777 | -3.62589598 | -2.61505797 | -3.66252184 | -1.99384582 | -2.53667784 | -2.06055179 | -3.44965821 |
| train/ufo/mean_next_auxQ | 500 | 25 | 524 | -3.22124743 | -6.34011269 | -4.62493312 | -6.42340374 | -3.21681142 | -4.53927803 | -3.35773501 | -6.23902013 |
| train/ufo/online_transitions | 500 | 25 | 524 | 98304 | 860160 | 470089.728 | 98304 | 860160 | 466944 | 811008 | 815923.2 |
| train/ufo/orientation_termination_count | 32 | 40 | 524 | 1 | 278 | 472.6875 | 1 | 1391 | 377.5 | 1084.95 | 472.6875 |
| train/ufo/orth_loss | 500 | 25 | 524 | 2706.61084 | 185.121185 | 615.242008 | 179.99408 | 2716.14014 | 396.652374 | 1826.56658 | 192.600937 |
| train/ufo/orth_loss_diag | 500 | 25 | 524 | -256 | -256 | -256 | -256 | -256 | -256 | -256 | -256 |
| train/ufo/orth_loss_offdiag | 500 | 25 | 524 | 2962.61084 | 441.121185 | 871.242008 | 435.99408 | 2972.14014 | 652.652374 | 2082.56658 | 448.600937 |
| train/ufo/physics_peak_joint_velocity | 32 | 40 | 524 | 32.7114105 | 39.9507217 | 37.4561547 | 25.0010967 | 55.7631607 | 35.2760201 | 52.9026297 | 37.4561547 |
| train/ufo/physics_peak_normalized_velocity | 32 | 40 | 524 | 4.08892632 | 4.99384022 | 4.22938148 | 2.4238286 | 6.97039509 | 3.96381867 | 6.55437307 | 4.22938148 |
| train/ufo/q_loss | 500 | 25 | 524 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| train/ufo/replay_draw_new_ratio | 32 | 40 | 524 | 0.833333313 | 1.24761903 | 1.18784997 | 0.833333313 | 1.27450979 | 1.23070174 | 1.27168562 | 1.18784997 |
| train/ufo/replay_draws | 32 | 40 | 524 | 81920 | 1073152 | 589056 | 81920 | 1073152 | 589824 | 1046937.6 | 589056 |
| train/ufo/replay_policy_age_max_updates | 32 | 40 | 524 | 40 | 524 | 287.625 | 40 | 524 | 288 | 511.2 | 287.625 |
| train/ufo/replay_policy_age_min_updates | 32 | 40 | 524 | 16 | 4 | 15.625 | 4 | 16 | 16 | 16 | 15.625 |
| train/ufo/replay_retained_fraction | 32 | 40 | 524 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| train/ufo/replay_size | 32 | 40 | 524 | 98304 | 860160 | 479232 | 98304 | 860160 | 479232 | 822067.2 | 479232 |
| train/ufo/requested_action_abs_mean | 32 | 40 | 524 | 0.928150773 | 0.840947568 | 0.895692077 | 0.840947568 | 0.93665123 | 0.896075845 | 0.932814986 | 0.895692077 |
| train/ufo/requested_action_saturation_fraction | 32 | 40 | 524 | 0.713479757 | 0.469071984 | 0.594272444 | 0.461678773 | 0.713479757 | 0.591050327 | 0.698713726 | 0.594272444 |
| train/ufo/requested_applied_max_diff | 32 | 40 | 524 | 1.94022894 | 1.96521783 | 1.96047322 | 1.94022894 | 1.96602845 | 1.96515417 | 1.96602798 | 1.96047322 |
| train/ufo/reset_control_samples | 32 | 40 | 524 | 6531 | 1034 | 1703.875 | 1034 | 6531 | 1321.5 | 2671.85 | 1703.875 |
| train/ufo/reset_projection_outside_fraction | 32 | 40 | 524 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| train/ufo/reset_projection_residual_max_rad | 32 | 40 | 524 | 1.78813934e-07 | 1.78813934e-07 | 1.76019967e-07 | 1.49011612e-07 | 1.78813934e-07 | 1.78813934e-07 | 1.78813934e-07 | 1.76019967e-07 |
| train/ufo/reset_projection_residual_mean_rad | 32 | 40 | 524 | 6.41934008e-08 | 6.39910454e-08 | 6.4045502e-08 | 6.25551237e-08 | 6.54890755e-08 | 6.40203019e-08 | 6.54232998e-08 | 6.4045502e-08 |
| train/ufo/reset_projection_residual_p95_rad | 32 | 40 | 524 | 1.1920929e-07 | 1.1920929e-07 | 1.1920929e-07 | 1.1920929e-07 | 1.1920929e-07 | 1.1920929e-07 | 1.1920929e-07 | 1.1920929e-07 |
| train/ufo/reset_root_lift_max_m | 32 | 40 | 524 | 0.0219456498 | 0.0219456498 | 0.0219804291 | 0.0219456498 | 0.0219920222 | 0.0219920222 | 0.0219920222 | 0.0219804291 |
| train/ufo/sampling_rounds | 500 | 25 | 524 | 4 | 35 | 19.128 | 4 | 35 | 19 | 33 | 33.2 |
| train/ufo/strict_shadow_episode_len | 32 | 40 | 524 | 3.28871894 | 15.6891518 | 12.043175 | 3.28871894 | 16.7791634 | 12.2884097 | 16.2065212 | 12.043175 |
| train/ufo/strict_shadow_mixed_episode_len | 32 | 40 | 524 | 3.22398591 | 14.4482222 | 11.0413703 | 3.22398591 | 14.7508898 | 11.3686895 | 14.4196688 | 11.0413703 |
| train/ufo/strict_shadow_tracking_episode_len | 32 | 40 | 524 | 3.36534452 | 17.4733334 | 13.5894966 | 3.36534452 | 21.072052 | 13.7066927 | 20.4191918 | 13.5894966 |
| train/ufo/strict_velocity_exposure_fraction | 32 | 40 | 524 | 0.000324857421 | 0.000255197723 | 0.000301174494 | 0.000215613327 | 0.000561036926 | 0.000279191605 | 0.000431127126 | 0.000301174494 |
| train/ufo/strict_velocity_transition_fraction | 32 | 40 | 524 | 0.0466715507 | 0.0277913418 | 0.0338083901 | 0.0250651035 | 0.0554606132 | 0.0311482744 | 0.0470743816 | 0.0338083901 |
| train/ufo/target_M | 500 | 25 | 524 | 3.47572684 | 0.611611962 | 0.906356782 | 0.245334029 | 4.19590759 | 0.709344298 | 2.90132638 | 0.566840035 |
| train/ufo/target_Q | 500 | 25 | 524 | -3.69620347 | -4.44878626 | -3.21513388 | -4.53898811 | -2.2891016 | -3.14620006 | -2.40114858 | -4.29236763 |
| train/ufo/target_auxQ | 500 | 25 | 524 | -4.59227943 | -7.31844616 | -5.79028886 | -7.50982237 | -4.57750463 | -5.66611362 | -4.69989891 | -7.27468995 |
| train/ufo/terminated_transitions | 32 | 40 | 524 | 1046 | 1097 | 1521.46875 | 1007 | 2574 | 1302.5 | 2364.75 | 1521.46875 |
| train/ufo/tracking_episode_starts | 32 | 40 | 524 | 3229 | 409 | 794.75 | 403 | 3229 | 580.5 | 1529.7 | 794.75 |
| train/ufo/tracking_q_rmse_rad | 32 | 40 | 524 | 0.083344698 | 0.425170273 | 0.384015515 | 0.083344698 | 0.448624134 | 0.403272897 | 0.436891125 | 0.384015515 |
| train/ufo/tracking_unique_env_fraction | 32 | 40 | 524 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 |
| train/ufo/tracking_windows_completed | 32 | 40 | 524 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| train/ufo/training_velocity_count | 32 | 40 | 524 | 147 | 57 | 82.90625 | 44 | 160 | 80 | 138.2 | 82.90625 |
| train/ufo/truncated_transitions | 32 | 40 | 524 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| train/ufo/unc_Q | 500 | 25 | 524 | 2.3536334 | 0.143360302 | 0.585362879 | 0.142590523 | 2.66353178 | 0.336406931 | 2.1598991 | 0.151184713 |
| train/ufo/unc_auxQ | 500 | 25 | 524 | 1.05287635 | 0.161964729 | 0.36919095 | 0.157077402 | 1.07364178 | 0.296801209 | 0.810656038 | 0.166303925 |
| train/ufo/velocity_reward_coverage_miss_count | 32 | 40 | 524 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| train/ufo/wall_time_s | 500 | 25 | 524 | 122.035652 | 1127.33362 | 616.494849 | 122.035652 | 1127.33362 | 613.760345 | 1066.9248 | 1071.67657 |
| train/ufo/z_norm | 500 | 25 | 524 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 |

## 10. 全指标五个100更新窗口

| tag | 25–124 | 125–224 | 225–324 | 325–424 | 425–524 |
|---|---|---|---|---|---|
| env/episode_len | 8.25451386 | 11.558163 | 15.1944955 | 19.2216685 | 22.3133605 |
| env/num_trajectories | 2106.83333 | 1947.33333 | 1427.66667 | 1156 | 1100.57143 |
| env/return | 0 | 0 | 0 | 0 | 0 |
| env/reward | 0 | 0 | 0 | 0 | 0 |
| env/success_once | 0 | 0 | 0 | 0 | 0 |
| rollout/ufo/received_transitions | 24576 | 24576 | 24576 | 24576 | 24576 |
| time/actor_training | 14.2667815 | 12.5108512 | 12.4764916 | 12.466438 | 11.2907968 |
| time/cal_adv_and_returns | 0.00516875585 | 0.00525625547 | 0.00484542052 | 0.00679343087 | 0.00680463655 |
| time/env/compute_bootstrap_rewards | 0.000253342674 | 0.000270979052 | 0.000234319867 | 0.000249496062 | 0.000253966444 |
| time/env/env/bootstrap_step | 0.004111618 | 0.000280437642 | 0.000280872298 | 0.000281277977 | 0.000282372248 |
| time/env/env/send_rollout_trajectories | 0.241771149 | 0.230467272 | 0.224774708 | 0.219191549 | 0.217038152 |
| time/env/env_interact_step | 1.89243792 | 1.64904781 | 1.944103 | 1.71446587 | 1.69291818 |
| time/env/interact | 3.50057856 | 2.91831303 | 3.03294889 | 2.86877424 | 2.96563332 |
| time/env/run_interact_once | 3.50053843 | 2.91828628 | 3.03292219 | 2.8687458 | 2.96560764 |
| time/generate_rollouts | 3.73392578 | 3.12074427 | 3.21990701 | 3.05792202 | 3.16502357 |
| time/rollout/generate_one_epoch | 3.22811222 | 2.68717217 | 2.80775523 | 2.62898234 | 2.73703204 |
| time/rollout/predict | 0.408684644 | 0.2233404 | 0.168911801 | 0.13882579 | 0.140177312 |
| time/rollout/rollout/generate | 3.22943016 | 2.6875337 | 2.80785894 | 2.62919906 | 2.7373929 |
| time/rollout/sync_model_from_actor | 13.3227178 | 13.061385 | 12.8308791 | 13.253393 | 13.4239919 |
| time/step | 34.8864377 | 32.1763179 | 32.1094964 | 32.2710694 | 31.3580747 |
| time/sync_weights | 16.8800338 | 16.5389547 | 16.4077576 | 16.7394243 | 16.8949032 |
| train/ufo/B | -0.00758521908 | -0.00440984685 | -0.00323536057 | -0.00249028358 | -0.00179989976 |
| train/ufo/B_norm | 16 | 16 | 16 | 16 | 16 |
| train/ufo/F1 | 0.048584002 | 0.0268841217 | 0.0269724116 | 0.034768324 | 0.0484380092 |
| train/ufo/M1 | 2.62671159 | 1.41317508 | 1.03540798 | 0.865660095 | 0.826189149 |
| train/ufo/Q1 | -3.26034205 | -2.45000442 | -2.81369621 | -3.42303902 | -4.10435666 |
| train/ufo/Q_aux | -5.01949474 | -5.06132613 | -5.78088722 | -6.55567375 | -7.26383841 |
| train/ufo/Q_discriminator | -3.52386193 | -2.54506982 | -2.93492735 | -3.51698391 | -4.21046385 |
| train/ufo/Q_fb | 6.17103572 | 16.5273102 | 27.0270914 | 38.958132 | 52.201909 |
| train/ufo/action_abs_mean | 0.931607157 | 0.922636787 | 0.90083167 | 0.878246069 | 0.854852796 |
| train/ufo/action_saturation_fraction | 0.689538042 | 0.654323409 | 0.601022909 | 0.55324792 | 0.496382373 |
| train/ufo/actor_loss | -2.26727295 | -10.9696855 | -17.6140071 | -24.4055283 | -34.411911 |
| train/ufo/actor_optimizer/grad_clip_limit | 1 | 1 | 1 | 1 | 1 |
| train/ufo/actor_optimizer/grad_norm_after_clip_l2 | 0.95346823 | 0.999999681 | 0.999999776 | 0.999999829 | 0.999999853 |
| train/ufo/actor_optimizer/grad_norm_l2 | 1.45158627 | 3.15169236 | 4.65332337 | 6.13912026 | 7.08926991 |
| train/ufo/actor_optimizer/lr | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 |
| train/ufo/actor_optimizer/parameter_delta_l2 | 0.385503453 | 0.351911208 | 0.332492818 | 0.313442565 | 0.321235817 |
| train/ufo/actor_reg_weight_applied | 14.40009 | 24.2924033 | 35.7294712 | 47.2863878 | 50 |
| train/ufo/actor_reg_weight_raw | 14.40009 | 24.2924033 | 35.7294712 | 48.1155305 | 61.4007642 |
| train/ufo/applied_action_abs_mean | 0.543634812 | 0.616977582 | 0.645220945 | 0.64312659 | 0.624530801 |
| train/ufo/applied_action_saturation_fraction | 0.177575447 | 0.264612471 | 0.293046887 | 0.292408692 | 0.26444625 |
| train/ufo/auxQ1 | -4.74971975 | -4.96499299 | -5.67534898 | -6.42491236 | -7.11504934 |
| train/ufo/aux_critic_loss | 1.33994628 | 0.979773325 | 1.0610733 | 1.17018375 | 1.35270971 |
| train/ufo/aux_critic_optimizer/grad_clip_limit | 5 | 5 | 5 | 5 | 5 |
| train/ufo/aux_critic_optimizer/grad_norm_after_clip_l2 | 4.91893308 | 4.83423481 | 4.77958727 | 4.81772115 | 4.7507477 |
| train/ufo/aux_critic_optimizer/grad_norm_l2 | 18.2138738 | 12.0902635 | 9.85748531 | 9.00264752 | 8.40954825 |
| train/ufo/aux_critic_optimizer/lr | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 |
| train/ufo/aux_critic_optimizer/parameter_delta_l2 | 0.314382348 | 0.39779031 | 0.453372624 | 0.504595522 | 0.527343995 |
| train/ufo/aux_rew/limits_dof_pos | 0.0856506795 | 0.160924526 | 0.212580162 | 0.254972849 | 0.282222931 |
| train/ufo/aux_rew/limits_dof_vel | 0.0095414308 | 0.0147126854 | 0.0150068865 | 0.0143765103 | 0.0135747902 |
| train/ufo/aux_rew/limits_torque | 0 | 0 | 0 | 0 | 0 |
| train/ufo/aux_rew/penalty_action_rate | 0.583135061 | 0.495990252 | 0.436617176 | 0.402241873 | 0.386713901 |
| train/ufo/aux_rew/penalty_ankle_roll | 0.0477757943 | 0.103405522 | 0.141048666 | 0.157643069 | 0.165994876 |
| train/ufo/aux_rew/penalty_feet_ori | 0.109712378 | 0.134409586 | 0.159275427 | 0.183459774 | 0.20115213 |
| train/ufo/aux_rew/penalty_slippage | 0.410305559 | 0.39141845 | 0.37855789 | 0.369402032 | 0.364511295 |
| train/ufo/aux_rew/penalty_torques | 5647.62959 | 5909.95561 | 5796.59029 | 5606.15873 | 5383.76213 |
| train/ufo/aux_rew/penalty_undesired_contact | 0.00455078125 | 0.00800292969 | 0.00916992188 | 0.00737792969 | 0.00625 |
| train/ufo/backward_optimizer/grad_clip_limit | 5 | 5 | 5 | 5 | 5 |
| train/ufo/backward_optimizer/grad_norm_after_clip_l2 | 4.99999998 | 5.00000001 | 5 | 4.99999996 | 5.00000006 |
| train/ufo/backward_optimizer/grad_norm_l2 | 255056.241 | 81340.6018 | 42092.8839 | 26000.6116 | 19132.4526 |
| train/ufo/backward_optimizer/lr | 9.99999975e-06 | 9.99999975e-06 | 9.99999975e-06 | 9.99999975e-06 | 9.99999975e-06 |
| train/ufo/backward_optimizer/parameter_delta_l2 | 0.00401576556 | 0.00437675722 | 0.00422403161 | 0.00401268021 | 0.00372113162 |
| train/ufo/collected_transitions | 24576 | 24576 | 24576 | 24576 | 24576 |
| train/ufo/contact_termination_count | 264.333333 | 490.333333 | 191.666667 | 91.2857143 | 97.8571429 |
| train/ufo/critic_loss | 0.80393427 | 0.305526901 | 0.300615316 | 0.336857341 | 0.441098871 |
| train/ufo/critic_optimizer/grad_clip_limit | 5 | 5 | 5 | 5 | 5 |
| train/ufo/critic_optimizer/grad_norm_after_clip_l2 | 4.95921664 | 4.68200248 | 4.87363576 | 4.72416585 | 4.85429816 |
| train/ufo/critic_optimizer/grad_norm_l2 | 20.0261298 | 8.98129029 | 9.50275314 | 8.45916243 | 8.06098298 |
| train/ufo/critic_optimizer/lr | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 |
| train/ufo/critic_optimizer/parameter_delta_l2 | 0.283187565 | 0.293020851 | 0.287689112 | 0.354228882 | 0.42249554 |
| train/ufo/disc_expert_loss | 0.484773338 | 0.350752926 | 0.313983449 | 0.271700678 | 0.247098505 |
| train/ufo/disc_loss | 2.66450851 | 1.4715674 | 1.10215732 | 0.937555813 | 0.830901186 |
| train/ufo/disc_reward_clip_fraction | 0 | 0.00942871094 | 0.0762841797 | 0.171499023 | 0.26578125 |
| train/ufo/disc_train_loss | 0.696291233 | 0.575557629 | 0.468247413 | 0.378944015 | 0.322115155 |
| train/ufo/disc_wgan_gp_loss | 0.148344393 | 0.0545256846 | 0.0319926459 | 0.0286911124 | 0.0261687527 |
| train/ufo/discriminator_optimizer/grad_clip_limit | 5 | 5 | 5 | 5 | 5 |
| train/ufo/discriminator_optimizer/grad_norm_after_clip_l2 | 4.99999938 | 4.09959005 | 1.73236704 | 1.36346668 | 1.13700931 |
| train/ufo/discriminator_optimizer/grad_norm_l2 | 7.94559263 | 4.54645886 | 1.73236704 | 1.36346668 | 1.13700931 |
| train/ufo/discriminator_optimizer/lr | 3.00000011e-06 | 3.00000011e-06 | 3.00000011e-06 | 3.00000011e-06 | 3.00000011e-06 |
| train/ufo/discriminator_optimizer/parameter_delta_l2 | 0.00487362688 | 0.00486463036 | 0.00334738001 | 0.00293329208 | 0.00265619162 |
| train/ufo/environment_control_steps | 26 | 50 | 74 | 100 | 128 |
| train/ufo/expert_draws/g1 | 82226.6667 | 181020 | 278861.333 | 385026.286 | 497917.714 |
| train/ufo/expert_draws/official_rpo | 81613.3333 | 179428 | 278194.667 | 385021.714 | 497995.429 |
| train/ufo/fb_diag | -45.575958 | -133.803807 | -190.773583 | -215.760249 | -228.34748 |
| train/ufo/fb_loss | 153033.38 | 66505.3425 | 40031.674 | 27454.6142 | 20551.3901 |
| train/ufo/fb_offdiag | 106.766142 | 116.896962 | 151.635988 | 182.626176 | 211.733061 |
| train/ufo/first_step_samples | 3037.83333 | 1963.16667 | 1449.66667 | 1166.71429 | 1093.28571 |
| train/ufo/first_step_terminated | 3.33333333 | 2 | 0.333333333 | 1 | 1 |
| train/ufo/foot_contact_peak_N | 14414.8993 | 16796.342 | 15846.2113 | 14839.036 | 15366.3827 |
| train/ufo/forward_optimizer/grad_clip_limit | 5 | 5 | 5 | 5 | 5 |
| train/ufo/forward_optimizer/grad_norm_after_clip_l2 | 5.00000001 | 4.99999999 | 4.99999999 | 4.99999994 | 5.00000008 |
| train/ufo/forward_optimizer/grad_norm_l2 | 1161.9292 | 289.44441 | 151.465751 | 114.999065 | 109.847406 |
| train/ufo/forward_optimizer/lr | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 |
| train/ufo/forward_optimizer/parameter_delta_l2 | 0.333016571 | 0.629451604 | 0.777160134 | 0.842670799 | 0.827724882 |
| train/ufo/ground_foot_margin_min_m | 193.500501 | 193.807571 | 193.653519 | 193.696745 | 193.68944 |
| train/ufo/ground_outside_count | 0 | 0 | 0 | 0 | 0 |
| train/ufo/hard_position_count | 791.5 | 507.166667 | 438.833333 | 396.142857 | 354.714286 |
| train/ufo/hard_torque_count | 0 | 0 | 0 | 0 | 0 |
| train/ufo/hard_velocity_count | 852.5 | 1039.5 | 836 | 746.428571 | 713.571429 |
| train/ufo/height_termination_count | 369 | 413.333333 | 353.833333 | 328 | 363 |
| train/ufo/latched_velocity_terms_without_boundary_violation | 709 | 875.666667 | 694.166667 | 615.285714 | 586.571429 |
| train/ufo/learner_joint_updates | 74.5 | 174.5 | 274.5 | 374.5 | 474.5 |
| train/ufo/mean_aux_reward | -1.14347814 | -1.28396951 | -1.3095343 | -1.3326833 | -1.33551763 |
| train/ufo/mean_disc_reward | 0.0341719261 | -0.150872847 | -0.506376803 | -0.772451615 | -0.954529793 |
| train/ufo/mean_disc_reward_raw | 0.0341719261 | -0.153700947 | -0.543167329 | -0.874447652 | -1.14421685 |
| train/ufo/mean_next_Q | -2.67556026 | -2.11333244 | -2.27489553 | -2.71959674 | -3.29190486 |
| train/ufo/mean_next_auxQ | -3.44452906 | -3.70490467 | -4.55376791 | -5.35135912 | -6.07010482 |
| train/ufo/online_transitions | 163184.64 | 316538.88 | 469893.12 | 623247.36 | 777584.64 |
| train/ufo/orientation_termination_count | 800.166667 | 541 | 439.833333 | 339.714286 | 294.571429 |
| train/ufo/orth_loss | 1529.7219 | 665.222495 | 400.708118 | 274.877484 | 205.680045 |
| train/ufo/orth_loss_diag | -256 | -256 | -256 | -256 | -256 |
| train/ufo/orth_loss_offdiag | 1785.7219 | 921.222495 | 656.708118 | 530.877484 | 461.680045 |
| train/ufo/physics_peak_joint_velocity | 33.9161863 | 39.2173026 | 43.7983239 | 39.3602524 | 31.6406152 |
| train/ufo/physics_peak_normalized_velocity | 3.86503045 | 4.34970148 | 4.86682391 | 4.51164464 | 3.60990855 |
| train/ufo/q_loss | 0 | 0 | 0 | 0 | 0 |
| train/ufo/replay_draw_new_ratio | 1.00145505 | 1.17024642 | 1.22429011 | 1.2528155 | 1.26650587 |
| train/ufo/replay_draws | 163840 | 360448 | 557056 | 770048 | 995913.143 |
| train/ufo/replay_policy_age_max_updates | 80 | 176 | 272 | 376 | 486.285714 |
| train/ufo/replay_policy_age_min_updates | 16 | 16 | 16 | 16 | 14.2857143 |
| train/ufo/replay_retained_fraction | 1 | 1 | 1 | 1 | 1 |
| train/ufo/replay_size | 159744 | 307200 | 454656 | 614400 | 786432 |
| train/ufo/requested_action_abs_mean | 0.931607107 | 0.922636787 | 0.90083168 | 0.878246112 | 0.854852889 |
| train/ufo/requested_action_saturation_fraction | 0.689538042 | 0.654323409 | 0.601022909 | 0.55324792 | 0.496382373 |
| train/ufo/requested_applied_max_diff | 1.95289469 | 1.96381301 | 1.9608102 | 1.9624449 | 1.96184591 |
| train/ufo/reset_control_samples | 3037.83333 | 1963.16667 | 1449.66667 | 1166.71429 | 1093.28571 |
| train/ufo/reset_projection_outside_fraction | 0 | 0 | 0 | 0 | 0 |
| train/ufo/reset_projection_residual_max_rad | 1.78813934e-07 | 1.78813934e-07 | 1.78813934e-07 | 1.70298985e-07 | 1.7455646e-07 |
| train/ufo/reset_projection_residual_mean_rad | 6.41770654e-08 | 6.41801113e-08 | 6.41367374e-08 | 6.39173874e-08 | 6.38672668e-08 |
| train/ufo/reset_projection_residual_p95_rad | 1.1920929e-07 | 1.1920929e-07 | 1.1920929e-07 | 1.1920929e-07 | 1.1920929e-07 |
| train/ufo/reset_root_lift_max_m | 0.0219765647 | 0.0219842934 | 0.0219920222 | 0.0219787729 | 0.0219721483 |
| train/ufo/sampling_rounds | 6.64 | 12.88 | 19.12 | 25.36 | 31.64 |
| train/ufo/strict_shadow_episode_len | 7.36798751 | 9.21281242 | 11.7758474 | 14.7346755 | 16.0141411 |
| train/ufo/strict_shadow_mixed_episode_len | 7.30444562 | 9.60746924 | 11.1257958 | 12.5257553 | 13.9167568 |
| train/ufo/strict_shadow_tracking_episode_len | 7.423129 | 8.83360926 | 12.7018119 | 18.1704281 | 19.1313705 |
| train/ufo/strict_velocity_exposure_fraction | 0.00030569178 | 0.000378594879 | 0.000309119476 | 0.000268655779 | 0.000256650934 |
| train/ufo/strict_velocity_transition_fraction | 0.0346883135 | 0.0422973627 | 0.0340169271 | 0.0303722561 | 0.029035296 |
| train/ufo/target_M | 1.78981498 | 0.832838383 | 0.721399766 | 0.611783684 | 0.575947101 |
| train/ufo/target_Q | -3.27795294 | -2.45179166 | -2.81289475 | -3.42259035 | -4.1104397 |
| train/ufo/target_auxQ | -4.75966614 | -4.95927304 | -5.68750231 | -6.43061532 | -7.11438748 |
| train/ufo/terminated_transitions | 2106.83333 | 1947.33333 | 1427.66667 | 1156 | 1100.57143 |
| train/ufo/tracking_episode_starts | 1563.16667 | 998.333333 | 625.666667 | 461 | 440.285714 |
| train/ufo/tracking_q_rmse_rad | 0.297732711 | 0.36846746 | 0.38922449 | 0.42860353 | 0.42224626 |
| train/ufo/tracking_unique_env_fraction | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 |
| train/ufo/tracking_windows_completed | 0 | 0 | 0 | 0 | 0 |
| train/ufo/training_velocity_count | 86.5 | 117.333333 | 86.5 | 71.8571429 | 58.2857143 |
| train/ufo/truncated_transitions | 0 | 0 | 0 | 0 | 0 |
| train/ufo/unc_Q | 1.5956933 | 0.612247214 | 0.343244351 | 0.21729351 | 0.158336025 |
| train/ufo/unc_auxQ | 0.716760792 | 0.449162436 | 0.298708373 | 0.209192643 | 0.172130505 |
| train/ufo/velocity_reward_coverage_miss_count | 0 | 0 | 0 | 0 | 0 |
| train/ufo/wall_time_s | 213.664867 | 415.278315 | 616.220723 | 816.595504 | 1020.71484 |
| train/ufo/z_norm | 16 | 16 | 16 | 16 | 16 |

统计直接读取无数量上限的两类原始TensorBoard事件；正式段采用>24点，只有配置初始点的tag保留其全点并明确step。缺点不插值、不补0，last50_mean是末50记录点，不是50更新；同step的sampler summary覆盖native同名最终summary点。环境占位reward/return/success=0不作质量判断；有效优化信号来自原生FB/D/aux。

## 11. 三轮复核、测试与尚未验证事项

第一轮：设计/本体语义与train-only隔离；3442全部候选和实际6144起点足点速度、左右支撑、251状态窗口、q/dq保持。第二轮：GPU0learner/三卡各2048、真实seed分区、16→24→524六Adam/哈希、子步奖励、严格影子、预算与恢复失败。第三轮：原始全部指标和五窗、两类trace/回放逐行对账、源/模式分组、删失、完整checkpoint与GPU释放；独立评估未达到门槛，未伪造评估媒体。

容器检查日志：
```text
........................................................................ [ 73%]
..........................                                               [100%]
=============================== warnings summary ===============================
../../workspace/isaaclab_extension_template/.venv/lib/python3.11/site-packages/_pytest/config/__init__.py:1428
  /workspace/isaaclab_extension_template/.venv/lib/python3.11/site-packages/_pytest/config/__init__.py:1428: PytestConfigWarning: Unknown config option: asyncio_mode
  
    self._warn_or_fail_if_strict(f"Unknown config option: {key}\n")

tests/unit/test_bfm_ufo_artifact.py: 12 warnings
  /workspace/isaaclab_extension_template/.venv/lib/python3.11/site-packages/torch/onnx/utils.py:185: DeprecationWarning: The feature will be removed. Please remove usage of this function
    setup_onnx_logging(verbose) as log_ctx,

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
98 passed, 13 warnings in 36.18s

```

容器检查日志：
```text
.....................................ss.s.....ss......................ss [ 12%]
ss.....s.s...s.s........................................................ [ 25%]
........................................................................ [ 38%]
........................................................................ [ 51%]
........................................................................ [ 64%]
.......................................................s.....s.......... [ 77%]
......s....s............................................................ [ 89%]
.........................................................                [100%]
=============================== warnings summary ===============================
../../app/backend/.venv/lib/python3.11/site-packages/passlib/utils/__init__.py:854
  /app/backend/.venv/lib/python3.11/site-packages/passlib/utils/__init__.py:854: DeprecationWarning: 'crypt' is deprecated and slated for removal in Python 3.13
    from crypt import crypt as _crypt

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
544 passed, 27 skipped, 1 warning in 5.65s

```

最终原生镜像98项通过，完整backend544项通过/27项跳过；平台镜像缺Torch/GPU可选依赖的跳过不冒充原生资格通过，原生镜像专项已实际执行。已有pytest asyncio_mode、crypt/ONNX导出弃用告警保留。代码整体fresh review发现运输漏字段和事件归因缺口，均RED复现后修复；真实GPU另发现二次seed覆写，保留失败并修复后从同seed资格重跑。没有测试前端/Rust全套（本轮未修改），没有真机、正常行走独立评估或物理逐位恢复。

后续建议需要单独冻结新实验：先根据峰事件接触/target/τ和source/mode存活确认被动接触冲击与位置越界的相对贡献，比较更柔顺的接触/显式PD方案与RPO/MJLab同起点物理对照；保留原严格影子和velocity mean8反馈。当前500每环境仅2.56秒，完整250步专家窗口的真实长周期尚缺；更长采样资格应先定义观测跨度/更新比例，不将replay扩容或loss下降自动当学习充分。若要恢复原生优化保护或重写trajectory replay，应另做单因素计划；本轮不自动执行。
