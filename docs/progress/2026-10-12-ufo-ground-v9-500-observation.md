# UFO ground-v9：RSI接触/PD与完整地面覆盖、6144环境500新增更新总结

本报告自包含任务参数、全部指标、源身份、恢复证据、物理诊断和历史对照，不需要读取任何其他文件。没有Git提交或暂存；没有启动独立validation/held-out/acceptance评估。

RSI=Reference State Initialization（参考状态初始化）：每次episode重置，从train参考动作中抽取一帧，将其root位置/四元数/速度及23个关节q/dq写入仿真。当前只从接触可行候选池采样，保留原始关节状态与速度，必要时向上修正root高度；不是固定站姿重置，也不是完整轨迹跟踪。

用户要求：重点排查右踝68.68rad/s和首步终止，修复后做500次新增原生UFO联合更新；GPU0 learner、GPU1–3每卡2048环境，batch2048/435116484参数；训练不足不独立评估。当前完成I新初始化16资格→完整恢复24→追加500至524；跨H→I物理recipe没有伪造完整恢复，实际物理状态仍不可恢复。

五个100更新窗口episode_len=7.13045224 → 10.0871313 → 11.0344256 → 12.0260083 → 13.2711236控制步；末点=13.4978046步（0.269956093秒）。至少末两窗口≥100步的训练触发门槛=False；这只是启动后续独立评估的必要观测，不是正常行走验收成功率。

正式786432条新增trace地面越界计数0，最小足部保守边界余量=193.260239m；首步终止=366/73200（0.5%），dq控制边界峰=45.985218rad/s、物理子步峰=67.5246201rad/s。剩余越速/接触/失衡风险按真实数据保留，不把数值有限、Job成功或loss下降当行走通过。

首步统计范围是正式500更新round4–35的trace且仍保留于最终replay的对应行，分母不包含资格段；最终完整replay860160行另含资格73728行。首步分子/分母与正式原始指标逐项相等。

六个优化器全部新增500次更新、裁剪后范数均满足Actor≤1/其余≤5且所有标量有限。训练触发门槛仍以持续存活为准，不以通过数值保护代替。

## 根因排查与修复范围

旧右踝峰值发生在episode第13控制步，q−0.1050→−0.0340rad、dq+0.896→−68.680rad/s，PD起始目标误差约0.0092rad、末输入力矩+0.2515Nm；该关节未位置越限，global位置flag来自右thigh pitch。右足垂直速度−3.282→+3.722m/s，离线足网格最低点+5.263cm→−0.785cm。支持落足/接触/约束附近速度突变，缺少旧PhysX子步冲量，未确定唯一因果原因。

H已修复train RSI足底穿透/高悬空候选、有限root lift、reset后传感器/FK刷新、控制首帧锁存上一物理帧旧力；保持显式PD和硬限位，物理5ms→2.5ms，decimation4→8仍50Hz，并锁存全部8帧物理越限。第三轮又发现100m×100m有限地面无法覆盖46×45实际网格：408/2048环境中心越界，参考root XY也有偏移。H内部/外部中心代理组足接触率60.52%/6.58%，高度终止率2.21%/7.59%；不能将该缺陷直接归因到旧右踝个例。

I只新增512m×512m离线有限mesh，seed42、1×1、无border、中心(0,0)；不修改奖励、PD增益/DR、URDF/惯量、solver或硬限位，不覆盖H/G发布对象。启动核对真实env_origins+全部接触候选足网格XY包络，必须保留20m运动余量；真实三sampler均2048，起点最小余量193.653247m。在线trace从128扩至131列，追加root world XY和足部保守覆盖余量；保守足半径约0.123323m包含实际网格，不只检查link原点。地面越界导致失败停止。

### 本段右踝首步45.99rad/s的完整真实初态/末态、PD目标与力矩

```json
{
  "scope": "I largest control-boundary dq transition, reset first step, CPU only",
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
  "peak_joint": "right_ankle_pitch_joint",
  "requested_action": -0.9567300081253052,
  "applied_action": -0.4177906811237335,
  "target_rad": -0.25067442655563354,
  "q_before_rad": -0.17067441046237947,
  "q_after_rad": -0.1885898381471634,
  "dq_before_radps": -1.138960599899292,
  "dq_after_radps": 45.9852180480957,
  "applied_last_substep_torque_Nm": 0.24385976791381836,
  "last_torque_times_boundary_dq_W": 11.213944435119629,
  "torque_velocity_sampling_limitation": "applied torque from final integration input; dq measured after step, product not integrated mechanical work",
  "pd_target_all_rad": [
    0.19999942183494568,
    -0.004766941070556641,
    0.08729410171508789,
    0.33338403701782227,
    0.16195687651634216,
    -0.02545332908630371,
    0.24306899309158325,
    0.7853986024856567,
    -0.3696732521057129,
    0.278971791267395,
    -0.2678852081298828,
    0.25,
    -0.061980366706848145,
    0.5116026997566223,
    -0.18411360681056976,
    -0.36458921432495117,
    -0.25067442655563354,
    0.9978975057601929,
    0.9562823176383972,
    -0.07999999821186066,
    -0.07999999821186066,
    0.07999999821186066,
    -0.07999999821186066
  ],
  "torque_all_Nm": [
    1.8321151733398438,
    -9.917403221130371,
    10.54227066040039,
    16.583993911743164,
    5.769474983215332,
    6.662247180938721,
    -5.627004623413086,
    4.183457374572754,
    6.116430282592773,
    -2.3181090354919434,
    -3.8488929271698,
    1.102304458618164,
    6.305755615234375,
    0.9055991172790527,
    0.843165397644043,
    -0.15019822120666504,
    0.24385976791381836,
    0.5091557502746582,
    -1.496353268623352,
    0.6331558227539062,
    0.422961950302124,
    -0.06868231296539307,
    -0.09335017204284668
  ],
  "history": [
    {
      "episode_step": 0,
      "before": {
        "q_rad": [
          0.04849814623594284,
          0.24523304402828217,
          -0.16270588338375092,
          0.08338404446840286,
          0.16946612298488617,
          -0.10545331954956055,
          0.3230688500404358,
          0.5672018766403198,
          -0.619673228263855,
          0.35897183179855346,
          -0.18788520753383636,
          -1.1920928966180355e-08,
          0.18801965117454528,
          0.5916026830673218,
          -0.1041136085987091,
          -0.4445892244577408,
          -0.17067441046237947,
          0.9178975510597229,
          1.0362823295593262,
          0.0,
          0.0,
          0.0,
          0.0
        ],
        "dq_radps": [
          0.3472062051296234,
          0.1982431858778,
          -0.26769378781318665,
          0.6351255178451538,
          0.021759793162345886,
          -0.37408360838890076,
          0.17034560441970825,
          2.3582727909088135,
          -0.9844139218330383,
          0.14467313885688782,
          0.1963738352060318,
          1.8895323790240254e-16,
          4.0841522216796875,
          -0.3591269254684448,
          0.21991786360740662,
          0.5583577156066895,
          -1.138960599899292,
          0.4417598247528076,
          0.652545690536499,
          0.0,
          0.0,
          0.0,
          0.0
        ],
        "gravity_body": [
          -0.03258451446890831,
          -0.0007212599739432335,
          -0.9994687438011169
        ],
        "root_omega_body_radps": [
          -0.04685944318771362,
          -0.10903556644916534,
          0.40197789669036865
        ],
        "root_height_m": 0.696290135383606,
        "foot_heading_local_lin_vel_mps": {
          "left": [
            0.5117188096046448,
            -0.08208660781383514,
            0.3368185758590698
          ],
          "right": [
            1.0797823667526245,
            -0.14957797527313232,
            -0.5241319537162781
          ]
        }
      },
      "after": {
        "q_rad": [
          0.1002330407500267,
          0.19564518332481384,
          -0.10565585643053055,
          0.11724952608346939,
          0.14316220581531525,
          -0.1291670572757721,
          0.34038528084754943,
          0.6236018776893616,
          -0.5579664766788482,
          0.35243171215057373,
          -0.18154826521873474,
          0.07889283001422881,
          0.09139502346515654,
          0.5616163611412048,
          -0.13202770054340363,
          -0.4222551852464676,
          -0.1885898381471634,
          0.9398594784736634,
          1.0316491413116455,
          -0.03251088783144951,
          -0.028017954900860786,
          0.021505815908312798,
          -0.018452389165759087
        ],
        "dq_radps": [
          -3.1459336280822754,
          -12.390783309936523,
          11.093596458435059,
          0.6971617937088013,
          -2.7757630348205566,
          -1.5703308582305908,
          -4.102105617523193,
          -0.43503475189208984,
          -9.439748764038086,
          0.345443457365036,
          1.4179049730300903,
          5.8859477043151855,
          35.634063720703125,
          -2.5587687492370605,
          -2.3345999717712402,
          1.5623152256011963,
          45.9852180480957,
          1.2012754678726196,
          2.6304049491882324,
          -1.5331929922103882,
          -13.076459884643555,
          1.2989532947540283,
          -1.238952398300171
        ],
        "gravity_body": [
          -0.047043997794389725,
          0.01783880963921547,
          -0.9987335205078125
        ],
        "root_omega_body_radps": [
          -4.626951217651367,
          1.2093009948730469,
          -10.455101013183594
        ],
        "root_height_m": 0.6891005039215088,
        "foot_heading_local_lin_vel_mps": {
          "left": [
            0.3816021978855133,
            0.7500699162483215,
            1.1811436414718628
          ],
          "right": [
            -5.401064872741699,
            1.0966053009033203,
            -2.4434616565704346
          ]
        }
      }
    }
  ],
  "missing": "vertical foot force and global physics peak recorded; no per-joint substep state or integrated solver impulse",
  "reset_target_slew_within_caps": true,
  "reset_target_delta_all_rad": [
    0.15150127559900284,
    -0.2499999850988388,
    0.2499999850988388,
    0.2499999925494194,
    -0.007509246468544006,
    0.07999999046325684,
    -0.07999985694885253,
    0.2181967258453369,
    0.24999997615814207,
    -0.08000004053115845,
    -0.08000000059604645,
    0.25000001192092897,
    -0.2500000178813934,
    -0.07999998331069946,
    -0.07999999821186066,
    0.08000001013278962,
    -0.08000001609325408,
    0.07999995470046994,
    -0.08000001192092898,
    -0.07999999821186066,
    -0.07999999821186066,
    0.07999999821186066,
    -0.07999999821186066
  ],
  "position_violation_joints": [],
  "before_root_height_m": 0.696290135383606,
  "after_root_height_m": 0.6891005039215088,
  "before_foot_velocity_mps": {
    "left": [
      0.5117188096046448,
      -0.08208660781383514,
      0.3368185758590698
    ],
    "right": [
      1.0797823667526245,
      -0.14957797527313232,
      -0.5241319537162781
    ]
  },
  "after_foot_velocity_mps": {
    "left": [
      0.3816021978855133,
      0.7500699162483215,
      1.1811436414718628
    ],
    "right": [
      -5.401064872741699,
      1.0966053009033203,
      -2.4434616565704346
    ]
  }
}
```

### 该右踝首步的碰撞足底初始/末态几何

```json
{
  "method": "offline official collision-mesh FK of I right-ankle first-step peak pre/post state, flat-ground yaw invariant; not Isaac contact force reconstruction",
  "sole_min_z_by_foot_m": [
    {
      "episode_step": 0,
      "before": {
        "left": 0.00942133681890141,
        "right": 0.007150745489455067
      },
      "after": {
        "left": 0.01229925038868638,
        "right": -0.0005075611119589041
      }
    }
  ]
}
```

本段最大控制边界dq为right_ankle_pitch_joint，round13、episode step0：dq−1.13896→+45.98522rad/s，q−0.170674→−0.188590rad；target−0.250674、变化−0.08rad符合slew，末输入力矩+0.24386Nm；右足垂直力峰10049.84N，无位置越限。初始右foot link heading-local速度[1.07978,−0.14958,−0.52413]m/s，已向下落足。这是与首步接触冲击一致的侧证：几何无初始穿透和关节dq可行并不能保证接触动态可行。末输入力矩与末dq有采样时差，不按其乘积推断积分功，也未采集逐子步接触冲量，不能唯一分离PD、全身耦合与接触求解。该45.99具体右踝案例与67.52全局子步标量不是同一极值。

### 原100m地面实际2048网格及代理组审计

```json
{
  "method": "frozen TerrainImporter grid formula and MeshPlaneTerrain extent; world origin group proxy, raw RSI xy offsets not applied per transition",
  "num_envs_per_sampler": 2048,
  "env_spacing_m": 2.5,
  "grid_rows": 46,
  "grid_cols": 45,
  "origin_xy_min_m": [
    -56.25,
    -55.0
  ],
  "origin_xy_max_m": [
    56.25,
    55.0
  ],
  "plane_size_m": [
    100,
    100
  ],
  "border_width_m": 0,
  "origins_outside_plane": 408,
  "outside_origin_fraction": 0.19921875,
  "train_valid_root_xy_min_m": [
    -6.971680164337158,
    -5.509981155395508
  ],
  "train_valid_root_xy_max_m": [
    7.098916053771973,
    7.317755699157715
  ],
  "formal_trace_group_statistics": {
    "origin_inside_100m": {
      "samples": 629760,
      "nonzero_foot_contact": 381111,
      "height_flags": 13930,
      "velocity_flags": 20798
    },
    "origin_outside_100m": {
      "samples": 156672,
      "nonzero_foot_contact": 10311,
      "height_flags": 11898,
      "velocity_flags": 511
    }
  },
  "limitation": "grouping uses grid origins; RSI xy offsets and foot placement must be included for exact per-case coverage"
}
```

### 三采样卡真实启动覆盖回执

```json
[
  {
    "run_id": "run_f7950a57b6e041b09023",
    "target_update": 16,
    "sampler_receipts": [
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
            -4.434485787549161
          ],
          [
            5.412482499340668,
            6.452038494132839
          ]
        ],
        "combined_foot_bounds_xy_m": [
          [
            -62.34675288072251,
            -59.434485787549164
          ],
          [
            61.66248249934067,
            61.452038494132836
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
            -4.434485787549161
          ],
          [
            5.412482499340668,
            6.452038494132839
          ]
        ],
        "combined_foot_bounds_xy_m": [
          [
            -62.34675288072251,
            -59.434485787549164
          ],
          [
            61.66248249934067,
            61.452038494132836
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
            -4.434485787549161
          ],
          [
            5.412482499340668,
            6.452038494132839
          ]
        ],
        "combined_foot_bounds_xy_m": [
          [
            -62.34675288072251,
            -59.434485787549164
          ],
          [
            61.66248249934067,
            61.452038494132836
          ]
        ],
        "minimum_margin_m": 193.6532471192775,
        "required_motion_reserve_m": 20.0
      }
    ]
  },
  {
    "run_id": "run_e077e5c07f794755aac9",
    "target_update": 24,
    "sampler_receipts": [
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
            -4.434485787549161
          ],
          [
            5.412482499340668,
            6.452038494132839
          ]
        ],
        "combined_foot_bounds_xy_m": [
          [
            -62.34675288072251,
            -59.434485787549164
          ],
          [
            61.66248249934067,
            61.452038494132836
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
            -4.434485787549161
          ],
          [
            5.412482499340668,
            6.452038494132839
          ]
        ],
        "combined_foot_bounds_xy_m": [
          [
            -62.34675288072251,
            -59.434485787549164
          ],
          [
            61.66248249934067,
            61.452038494132836
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
            -4.434485787549161
          ],
          [
            5.412482499340668,
            6.452038494132839
          ]
        ],
        "combined_foot_bounds_xy_m": [
          [
            -62.34675288072251,
            -59.434485787549164
          ],
          [
            61.66248249934067,
            61.452038494132836
          ]
        ],
        "minimum_margin_m": 193.6532471192775,
        "required_motion_reserve_m": 20.0
      }
    ]
  },
  {
    "run_id": "run_13cb5781b3944be88906",
    "target_update": 524,
    "sampler_receipts": [
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
            -4.434485787549161
          ],
          [
            5.412482499340668,
            6.452038494132839
          ]
        ],
        "combined_foot_bounds_xy_m": [
          [
            -62.34675288072251,
            -59.434485787549164
          ],
          [
            61.66248249934067,
            61.452038494132836
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
            -4.434485787549161
          ],
          [
            5.412482499340668,
            6.452038494132839
          ]
        ],
        "combined_foot_bounds_xy_m": [
          [
            -62.34675288072251,
            -59.434485787549164
          ],
          [
            61.66248249934067,
            61.452038494132836
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
            -4.434485787549161
          ],
          [
            5.412482499340668,
            6.452038494132839
          ]
        ],
        "combined_foot_bounds_xy_m": [
          [
            -62.34675288072251,
            -59.434485787549164
          ],
          [
            61.66248249934067,
            61.452038494132836
          ]
        ],
        "minimum_margin_m": 193.6532471192775,
        "required_motion_reserve_m": 20.0
      }
    ]
  }
]
```

### 修正后同一网格代理组的实际足部接触与终止

```json
{
  "method": "same origin proxy groups as affected H; all I feet covered by new512m plane; not exact old-foot coverage",
  "run_id": "run_13cb5781b3944be88906",
  "groups": {
    "old_origin_inside_100m": {
      "samples": 629760,
      "nonzero_foot_contact": 405639,
      "height_flags": 11935,
      "velocity_flags": 21733
    },
    "old_origin_outside_100m": {
      "samples": 156672,
      "nonzero_foot_contact": 101288,
      "height_flags": 2947,
      "velocity_flags": 5413
    }
  }
}
```

| 修正后原中心分组 | 转移数 | 足部垂直力>1N比例% | 高度终止比例% | 速度终止比例% |
|---|---|---|---|---|
| old_origin_inside_100m | 629760 | 64.4116806 | 1.89516641 | 3.45099721 |
| old_origin_outside_100m | 156672 | 64.6497141 | 1.8809998 | 3.45498877 |

原外部中心组足部接触由H6.58%恢复到I约64.65%，与I内部64.41%接近；高度与速度终止比例也接近。这支持大网格地面覆盖已修复。H外部中心组较少速度flags源于接触不足的可能性得到侧证，不能将旧低flags当好控制；仍未对单个旧68.68案例证明唯一因果。

独立32环境配对诊断已跑两次，每次最多15分钟；小网格本身无法暴露2048边缘覆盖缺陷。未再开第三次GPU probe。H受影响500完整保留；I资格16/24各≤1h、正式500≤2h、零自动重试。为保留两版全状态，对H+I新增证据总体预算派生96GiB，含12GiB镜像/暂存预留；这不是用户设定的上限。

## 完整I任务与配置

### I 16更新TrainJob

```json
{
  "id": "trn_c995d8844198488c9461",
  "run_id": "run_f7950a57b6e041b09023",
  "kind": "ppo",
  "name": "RPO UFO ground v9 native 16-update qualification",
  "model_name": "RPO-UFO-ground-v9-step16",
  "status": "succeeded",
  "template_id": "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_ground_v9",
  "image_digest": "sha256:1034a824a3782202b914275344f6c0985d0f1721fb423e5620b5a964dc16672e",
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
  "ray_port": 6436,
  "params": {
    "stack": "rlinf",
    "purpose": "qualification",
    "execution": {
      "recipe_id": "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_ground_v9",
      "runtime_id": "rlinf",
      "provider_id": "ufo",
      "topology_id": "one_learner_three_samplers",
      "algorithm_id": "bfm_fb_cpr_aux",
      "simulator_id": "isaaclab",
      "progress_unit": "learner_joint_update",
      "schema_version": "eb/training-binding/v1",
      "checkpoint_format": "ufo_agent_state_v1"
    },
    "model_name": "RPO-UFO-ground-v9-step16",
    "config_name": "rpo_bfm_ufo_isaac",
    "template_id": "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_ground_v9",
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
      "control_profile": "reachable_v2",
      "scaling_profile": "env6144_batch2048_v6",
      "auxiliary_profile": "soft_velocity_v7",
      "ground_patch_hash": "sha256:554a18609c9b8a420811fa6b1ded908a4bdf181e52e8d513c8772de6084eff59",
      "optimizer_profile": "stabilized_v7",
      "contact_patch_hash": "sha256:e20a0ff930b3b9430ece70c21ab316a06cef92b679b638c0a35d7fed62d6d814",
      "topology_patch_hash": "sha256:3078506628924464cb42d0b2f108b3fab502a43fd75e091f358d504565294d23",
      "auxiliary_source_sha256": "sha256:8359d881e7a3523a740b10fb01f4ee53e70a56ecaeb0f31ece3d0fb2e94c03d5",
      "stabilization_patch_hash": "sha256:564f5452e12e3179112e6aa3020efba4cbcedc0ad90dd26b98aed78a50dd85b1",
      "rollout_profile_patch_hash": "sha256:777456ce171734326cf8c55eaa434fc5a69a27d0b2d0b725608dd54c190f4a77",
      "scaling_profile_patch_hash": "sha256:58a2dca8377ed2ede804c7101b39bc6992ec669ef95e610ddec4b2a59bfc6b20",
      "learning_profile_patch_hash": "sha256:d47623e48b4cc757d5e63336a1fa7d49234c3833a2f1c6bbdbee4fe7ffd9a272",
      "action_control_source_sha256": "sha256:1b0bbc0a7a350581ef1939faf93a3e27d824bde62107ecfe4e9ff988bbc088fb",
      "auxiliary_profile_patch_hash": "sha256:30e7b4ea7a3ec0e45cdbe0a9dbf718eb219c2c6fd36d07f9b18bd5222183a03a",
      "action_control_profile_patch_hash": "sha256:548260ca5e3eedd53ae739c00672c7e273e7835faed734778039cf5977b56e0d"
    },
    "environment_profile": {
      "id": "rpo_bfm_ufo_isaac_4gpu_rsi_ground_v9",
      "hash": "sha256:fcf30a6403393bf675d7dc4985ab86267cfd5859c325391717758e8bd2206c1a",
      "definition": {
        "task": "RPO v2有限URDF动作目标，原始q/dq与观察零点保持；6144环境，每个sampler2048；435M原生网络、batch2048、replay100万，每24576新转移触发16次联合更新。 RSI仅train接触候选、足底2mm间隙、有限root上移；写入后刷新FK，专家数据不改。 512m离线地面；启动核对实际网格+所有RSI足碰撞网格包络与20m余量。",
        "reward": "九项raw aux，limits_dof_vel=sum relu(abs(dq)/Vmax-0.70)^2，权重-25仅加权一次；判别奖励截[-2,2]，D lr=3e-6；Actor正则尺度上限50。",
        "failure": "真实硬位置/速度/力矩终止保持；target slew每步0.5×Vmax；Actor裁剪1，其余五Adam裁剪5，非有限梯度失败。 显式PD增益保持，physics dt=2.5ms/decimation8/50Hz；物理子步硬限位锁存，接触/高度/倾角原因分列。 每步记录世界XY与足部保守覆盖余量，越界拒绝继续采样。",
        "success": "训练/完整恢复资格16→24后新增500联合更新，最多2小时且不自动重试；训练指标不足不启动独立评估。正常行走尚未验证。",
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
    "+actor.model.rollout_expert_trajectories=False",
    "actor.model.hidden_dim=2048",
    "actor.model.hidden_layers=6",
    "actor.model.batch_size=2048",
    "actor.micro_batch_size=2048",
    "actor.global_batch_size=2048",
    "actor.model.replay_capacity=1000000",
    "env.train.init_params.id=RPO-UFO-Isaac-RSI-ground-v9",
    "+actor.model.auxiliary_profile=soft_velocity_v7",
    "+actor.model.scaling_profile=env6144_batch2048_v6",
    "+actor.model.control_profile=reachable_v2",
    "+actor.model.optimizer_profile=stabilized_v7",
    "+actor.model.contact_profile=contact_stable_v8",
    "+actor.model.ground_profile=wide_plane_v9",
    "actor.seed=42",
    "~cluster.component_placement",
    "+cluster.component_placement={actor:0,env\\,\\ rollout:1-3}",
    "runner.logger.log_path=/workspace/output",
    "hydra.run.dir=/workspace/output/hydra",
    "runner.logger.experiment_name=\"RPO UFO ground v9 native 16-update qualification\""
  ],
  "run_dir": "/data/kk/embodiedbein/runs/run_f7950a57b6e041b09023",
  "auto_register_latest": false,
  "metrics_summary": {
    "checkpoint": {
      "path": "/data/kk/models/runs/run_f7950a57b6e041b09023",
      "step": 16,
      "bytes": 7211132970
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
    "model_version_id": "mv_6683a94fd64d43efbbb8"
  },
  "error_message": null,
  "job_id": "job_f4920aa7f7c54f61b41c",
  "started_at": "2026-10-12T00:18:51.462405+00:00",
  "finished_at": "2026-10-12T00:22:38.063337+00:00",
  "created_at": "2026-10-12T00:18:40.704264+00:00"
}
```

### I 24更新TrainJob

```json
{
  "id": "trn_518be1feb13e40b4b4ac",
  "run_id": "run_e077e5c07f794755aac9",
  "kind": "ppo",
  "name": "RPO UFO ground v9 native 24-update restore qualification",
  "model_name": "RPO-UFO-ground-v9-step24",
  "status": "succeeded",
  "template_id": "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_ground_v9",
  "image_digest": "sha256:1034a824a3782202b914275344f6c0985d0f1721fb423e5620b5a964dc16672e",
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
  "ray_port": 6436,
  "params": {
    "stack": "rlinf",
    "purpose": "qualification",
    "execution": {
      "recipe_id": "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_ground_v9",
      "runtime_id": "rlinf",
      "provider_id": "ufo",
      "topology_id": "one_learner_three_samplers",
      "algorithm_id": "bfm_fb_cpr_aux",
      "simulator_id": "isaaclab",
      "progress_unit": "learner_joint_update",
      "schema_version": "eb/training-binding/v1",
      "checkpoint_format": "ufo_agent_state_v1"
    },
    "model_name": "RPO-UFO-ground-v9-step24",
    "config_name": "rpo_bfm_ufo_isaac",
    "resume_mode": "trainer_state",
    "resume_step": 16,
    "template_id": "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_ground_v9",
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
      "control_profile": "reachable_v2",
      "scaling_profile": "env6144_batch2048_v6",
      "auxiliary_profile": "soft_velocity_v7",
      "ground_patch_hash": "sha256:554a18609c9b8a420811fa6b1ded908a4bdf181e52e8d513c8772de6084eff59",
      "optimizer_profile": "stabilized_v7",
      "contact_patch_hash": "sha256:e20a0ff930b3b9430ece70c21ab316a06cef92b679b638c0a35d7fed62d6d814",
      "topology_patch_hash": "sha256:3078506628924464cb42d0b2f108b3fab502a43fd75e091f358d504565294d23",
      "auxiliary_source_sha256": "sha256:8359d881e7a3523a740b10fb01f4ee53e70a56ecaeb0f31ece3d0fb2e94c03d5",
      "stabilization_patch_hash": "sha256:564f5452e12e3179112e6aa3020efba4cbcedc0ad90dd26b98aed78a50dd85b1",
      "rollout_profile_patch_hash": "sha256:777456ce171734326cf8c55eaa434fc5a69a27d0b2d0b725608dd54c190f4a77",
      "scaling_profile_patch_hash": "sha256:58a2dca8377ed2ede804c7101b39bc6992ec669ef95e610ddec4b2a59bfc6b20",
      "learning_profile_patch_hash": "sha256:d47623e48b4cc757d5e63336a1fa7d49234c3833a2f1c6bbdbee4fe7ffd9a272",
      "action_control_source_sha256": "sha256:1b0bbc0a7a350581ef1939faf93a3e27d824bde62107ecfe4e9ff988bbc088fb",
      "auxiliary_profile_patch_hash": "sha256:30e7b4ea7a3ec0e45cdbe0a9dbf718eb219c2c6fd36d07f9b18bd5222183a03a",
      "action_control_profile_patch_hash": "sha256:548260ca5e3eedd53ae739c00672c7e273e7835faed734778039cf5977b56e0d"
    },
    "environment_profile": {
      "id": "rpo_bfm_ufo_isaac_4gpu_rsi_ground_v9",
      "hash": "sha256:fcf30a6403393bf675d7dc4985ab86267cfd5859c325391717758e8bd2206c1a",
      "definition": {
        "task": "RPO v2有限URDF动作目标，原始q/dq与观察零点保持；6144环境，每个sampler2048；435M原生网络、batch2048、replay100万，每24576新转移触发16次联合更新。 RSI仅train接触候选、足底2mm间隙、有限root上移；写入后刷新FK，专家数据不改。 512m离线地面；启动核对实际网格+所有RSI足碰撞网格包络与20m余量。",
        "reward": "九项raw aux，limits_dof_vel=sum relu(abs(dq)/Vmax-0.70)^2，权重-25仅加权一次；判别奖励截[-2,2]，D lr=3e-6；Actor正则尺度上限50。",
        "failure": "真实硬位置/速度/力矩终止保持；target slew每步0.5×Vmax；Actor裁剪1，其余五Adam裁剪5，非有限梯度失败。 显式PD增益保持，physics dt=2.5ms/decimation8/50Hz；物理子步硬限位锁存，接触/高度/倾角原因分列。 每步记录世界XY与足部保守覆盖余量，越界拒绝继续采样。",
        "success": "训练/完整恢复资格16→24后新增500联合更新，最多2小时且不自动重试；训练指标不足不启动独立评估。正常行走尚未验证。",
        "settings": {}
      }
    },
    "resume_source_image": "eb/rlinf:rpo-bfm-ufo-agent-4gpu-ground-v9",
    "source_train_job_id": "trn_c995d8844198488c9461",
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
    "resume_from": "run_f7950a57b6e041b09023"
  },
  "resumed_from": "trn_c995d8844198488c9461",
  "resume_step": 16,
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
    "env.train.init_params.id=RPO-UFO-Isaac-RSI-ground-v9",
    "+actor.model.auxiliary_profile=soft_velocity_v7",
    "+actor.model.scaling_profile=env6144_batch2048_v6",
    "+actor.model.control_profile=reachable_v2",
    "+actor.model.optimizer_profile=stabilized_v7",
    "+actor.model.contact_profile=contact_stable_v8",
    "+actor.model.ground_profile=wide_plane_v9",
    "actor.seed=42",
    "~cluster.component_placement",
    "+cluster.component_placement={actor:0,env\\,\\ rollout:1-3}",
    "runner.logger.log_path=/workspace/output",
    "hydra.run.dir=/workspace/output/hydra",
    "runner.logger.experiment_name=\"RPO UFO ground v9 native 24-update restore qualification\"",
    "runner.resume_dir=/workspace/resume/global_step_16"
  ],
  "run_dir": "/data/kk/embodiedbein/runs/run_e077e5c07f794755aac9",
  "auto_register_latest": false,
  "metrics_summary": {
    "checkpoint": {
      "path": "/data/kk/models/runs/run_e077e5c07f794755aac9",
      "step": 24,
      "bytes": 7384201614
    },
    "sampler_sync": {
      "identity": {
        "version": 16,
        "state_sha256": "d63a72a20555cbd5670a427df21720731e159a68b662720cbc82f4728f798701",
        "online_transitions": 49152
      },
      "samplers": [
        {
          "rank": 0,
          "version": 16,
          "state_sha256": "d63a72a20555cbd5670a427df21720731e159a68b662720cbc82f4728f798701",
          "online_transitions": 49152
        },
        {
          "rank": 1,
          "version": 16,
          "state_sha256": "d63a72a20555cbd5670a427df21720731e159a68b662720cbc82f4728f798701",
          "online_transitions": 49152
        },
        {
          "rank": 2,
          "version": 16,
          "state_sha256": "d63a72a20555cbd5670a427df21720731e159a68b662720cbc82f4728f798701",
          "online_transitions": 49152
        }
      ],
      "schema_version": "eb/ufo-sampler-sync/v1",
      "sampling_rounds": 2
    },
    "model_version_id": "mv_bfbf570bbd004fad926f"
  },
  "error_message": null,
  "job_id": "job_df09329a407e4213860c",
  "started_at": "2026-10-12T00:25:20.213473+00:00",
  "finished_at": "2026-10-12T00:28:52.952684+00:00",
  "created_at": "2026-10-12T00:25:17.204086+00:00"
}
```

### I 524更新TrainJob

```json
{
  "id": "trn_8bdabc160cd04d2bb100",
  "run_id": "run_13cb5781b3944be88906",
  "kind": "ppo",
  "name": "RPO UFO ground v9 500-update observation",
  "model_name": "RPO-UFO-ground-v9-step524",
  "status": "succeeded",
  "template_id": "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_ground_v9",
  "image_digest": "sha256:1034a824a3782202b914275344f6c0985d0f1721fb423e5620b5a964dc16672e",
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
  "ray_port": 6437,
  "params": {
    "stack": "rlinf",
    "purpose": "training",
    "execution": {
      "recipe_id": "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_ground_v9",
      "runtime_id": "rlinf",
      "provider_id": "ufo",
      "topology_id": "one_learner_three_samplers",
      "algorithm_id": "bfm_fb_cpr_aux",
      "simulator_id": "isaaclab",
      "progress_unit": "learner_joint_update",
      "schema_version": "eb/training-binding/v1",
      "checkpoint_format": "ufo_agent_state_v1"
    },
    "model_name": "RPO-UFO-ground-v9-step524",
    "config_name": "rpo_bfm_ufo_isaac",
    "resume_mode": "trainer_state",
    "resume_step": 24,
    "template_id": "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_ground_v9",
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
      "control_profile": "reachable_v2",
      "scaling_profile": "env6144_batch2048_v6",
      "auxiliary_profile": "soft_velocity_v7",
      "ground_patch_hash": "sha256:554a18609c9b8a420811fa6b1ded908a4bdf181e52e8d513c8772de6084eff59",
      "optimizer_profile": "stabilized_v7",
      "contact_patch_hash": "sha256:e20a0ff930b3b9430ece70c21ab316a06cef92b679b638c0a35d7fed62d6d814",
      "topology_patch_hash": "sha256:3078506628924464cb42d0b2f108b3fab502a43fd75e091f358d504565294d23",
      "auxiliary_source_sha256": "sha256:8359d881e7a3523a740b10fb01f4ee53e70a56ecaeb0f31ece3d0fb2e94c03d5",
      "stabilization_patch_hash": "sha256:564f5452e12e3179112e6aa3020efba4cbcedc0ad90dd26b98aed78a50dd85b1",
      "rollout_profile_patch_hash": "sha256:777456ce171734326cf8c55eaa434fc5a69a27d0b2d0b725608dd54c190f4a77",
      "scaling_profile_patch_hash": "sha256:58a2dca8377ed2ede804c7101b39bc6992ec669ef95e610ddec4b2a59bfc6b20",
      "learning_profile_patch_hash": "sha256:d47623e48b4cc757d5e63336a1fa7d49234c3833a2f1c6bbdbee4fe7ffd9a272",
      "action_control_source_sha256": "sha256:1b0bbc0a7a350581ef1939faf93a3e27d824bde62107ecfe4e9ff988bbc088fb",
      "auxiliary_profile_patch_hash": "sha256:30e7b4ea7a3ec0e45cdbe0a9dbf718eb219c2c6fd36d07f9b18bd5222183a03a",
      "action_control_profile_patch_hash": "sha256:548260ca5e3eedd53ae739c00672c7e273e7835faed734778039cf5977b56e0d"
    },
    "environment_profile": {
      "id": "rpo_bfm_ufo_isaac_4gpu_rsi_ground_v9",
      "hash": "sha256:fcf30a6403393bf675d7dc4985ab86267cfd5859c325391717758e8bd2206c1a",
      "definition": {
        "task": "RPO v2有限URDF动作目标，原始q/dq与观察零点保持；6144环境，每个sampler2048；435M原生网络、batch2048、replay100万，每24576新转移触发16次联合更新。 RSI仅train接触候选、足底2mm间隙、有限root上移；写入后刷新FK，专家数据不改。 512m离线地面；启动核对实际网格+所有RSI足碰撞网格包络与20m余量。",
        "reward": "九项raw aux，limits_dof_vel=sum relu(abs(dq)/Vmax-0.70)^2，权重-25仅加权一次；判别奖励截[-2,2]，D lr=3e-6；Actor正则尺度上限50。",
        "failure": "真实硬位置/速度/力矩终止保持；target slew每步0.5×Vmax；Actor裁剪1，其余五Adam裁剪5，非有限梯度失败。 显式PD增益保持，physics dt=2.5ms/decimation8/50Hz；物理子步硬限位锁存，接触/高度/倾角原因分列。 每步记录世界XY与足部保守覆盖余量，越界拒绝继续采样。",
        "success": "训练/完整恢复资格16→24后新增500联合更新，最多2小时且不自动重试；训练指标不足不启动独立评估。正常行走尚未验证。",
        "settings": {}
      }
    },
    "resume_source_image": "eb/rlinf:rpo-bfm-ufo-agent-4gpu-ground-v9",
    "source_train_job_id": "trn_518be1feb13e40b4b4ac",
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
    "resume_from": "run_e077e5c07f794755aac9"
  },
  "resumed_from": "trn_518be1feb13e40b4b4ac",
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
    "env.train.init_params.id=RPO-UFO-Isaac-RSI-ground-v9",
    "+actor.model.auxiliary_profile=soft_velocity_v7",
    "+actor.model.scaling_profile=env6144_batch2048_v6",
    "+actor.model.control_profile=reachable_v2",
    "+actor.model.optimizer_profile=stabilized_v7",
    "+actor.model.contact_profile=contact_stable_v8",
    "+actor.model.ground_profile=wide_plane_v9",
    "actor.seed=42",
    "~cluster.component_placement",
    "+cluster.component_placement={actor:0,env\\,\\ rollout:1-3}",
    "runner.logger.log_path=/workspace/output",
    "hydra.run.dir=/workspace/output/hydra",
    "runner.logger.experiment_name=\"RPO UFO ground v9 500-update observation\"",
    "runner.resume_dir=/workspace/resume/global_step_24"
  ],
  "run_dir": "/data/kk/embodiedbein/runs/run_13cb5781b3944be88906",
  "auto_register_latest": false,
  "metrics_summary": {
    "checkpoint": {
      "path": "/data/kk/models/runs/run_13cb5781b3944be88906",
      "step": 524,
      "bytes": 12922256064
    },
    "sampler_sync": {
      "identity": {
        "version": 520,
        "state_sha256": "114d6cc10c56b477a66ff07fe5e21b4a470946bb4e0ac93e54a9a6da51692f7a",
        "online_transitions": 835584
      },
      "samplers": [
        {
          "rank": 0,
          "version": 520,
          "state_sha256": "114d6cc10c56b477a66ff07fe5e21b4a470946bb4e0ac93e54a9a6da51692f7a",
          "online_transitions": 835584
        },
        {
          "rank": 1,
          "version": 520,
          "state_sha256": "114d6cc10c56b477a66ff07fe5e21b4a470946bb4e0ac93e54a9a6da51692f7a",
          "online_transitions": 835584
        },
        {
          "rank": 2,
          "version": 520,
          "state_sha256": "114d6cc10c56b477a66ff07fe5e21b4a470946bb4e0ac93e54a9a6da51692f7a",
          "online_transitions": 835584
        }
      ],
      "schema_version": "eb/ufo-sampler-sync/v1",
      "sampling_rounds": 34
    },
    "model_version_id": "mv_f1c2fd66a7f6402fa7a1"
  },
  "error_message": null,
  "job_id": "job_5157c51b35464239b2b6",
  "started_at": "2026-10-12T00:33:39.546660+00:00",
  "finished_at": "2026-10-12T01:07:11.390022+00:00",
  "created_at": "2026-10-12T00:33:35.251877+00:00"
}
```

正式Job及Run完整元数据：
```json
{
  "id": "job_5157c51b35464239b2b6",
  "kind": "train.rl",
  "status": "succeeded",
  "embodiment_env_id": "ee_rpo_v2_bfm_23dof_v2",
  "priority": 50,
  "created_by": "user:kk",
  "spec": {
    "kind": "train.rl",
    "name": "RPO UFO ground v9 500-update observation",
    "gates": [],
    "image": "sha256:1034a824a3782202b914275344f6c0985d0f1721fb423e5620b5a964dc16672e",
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
      "resume_from": "run_e077e5c07f794755aac9"
    },
    "labels": {},
    "params": {
      "stack": "rlinf",
      "purpose": "training",
      "execution": {
        "recipe_id": "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_ground_v9",
        "runtime_id": "rlinf",
        "provider_id": "ufo",
        "topology_id": "one_learner_three_samplers",
        "algorithm_id": "bfm_fb_cpr_aux",
        "simulator_id": "isaaclab",
        "progress_unit": "learner_joint_update",
        "schema_version": "eb/training-binding/v1",
        "checkpoint_format": "ufo_agent_state_v1"
      },
      "model_name": "RPO-UFO-ground-v9-step524",
      "config_name": "rpo_bfm_ufo_isaac",
      "resume_mode": "trainer_state",
      "resume_step": 24,
      "template_id": "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_ground_v9",
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
        "control_profile": "reachable_v2",
        "scaling_profile": "env6144_batch2048_v6",
        "auxiliary_profile": "soft_velocity_v7",
        "ground_patch_hash": "sha256:554a18609c9b8a420811fa6b1ded908a4bdf181e52e8d513c8772de6084eff59",
        "optimizer_profile": "stabilized_v7",
        "contact_patch_hash": "sha256:e20a0ff930b3b9430ece70c21ab316a06cef92b679b638c0a35d7fed62d6d814",
        "topology_patch_hash": "sha256:3078506628924464cb42d0b2f108b3fab502a43fd75e091f358d504565294d23",
        "auxiliary_source_sha256": "sha256:8359d881e7a3523a740b10fb01f4ee53e70a56ecaeb0f31ece3d0fb2e94c03d5",
        "stabilization_patch_hash": "sha256:564f5452e12e3179112e6aa3020efba4cbcedc0ad90dd26b98aed78a50dd85b1",
        "rollout_profile_patch_hash": "sha256:777456ce171734326cf8c55eaa434fc5a69a27d0b2d0b725608dd54c190f4a77",
        "scaling_profile_patch_hash": "sha256:58a2dca8377ed2ede804c7101b39bc6992ec669ef95e610ddec4b2a59bfc6b20",
        "learning_profile_patch_hash": "sha256:d47623e48b4cc757d5e63336a1fa7d49234c3833a2f1c6bbdbee4fe7ffd9a272",
        "action_control_source_sha256": "sha256:1b0bbc0a7a350581ef1939faf93a3e27d824bde62107ecfe4e9ff988bbc088fb",
        "auxiliary_profile_patch_hash": "sha256:30e7b4ea7a3ec0e45cdbe0a9dbf718eb219c2c6fd36d07f9b18bd5222183a03a",
        "action_control_profile_patch_hash": "sha256:548260ca5e3eedd53ae739c00672c7e273e7835faed734778039cf5977b56e0d"
      },
      "environment_profile": {
        "id": "rpo_bfm_ufo_isaac_4gpu_rsi_ground_v9",
        "hash": "sha256:fcf30a6403393bf675d7dc4985ab86267cfd5859c325391717758e8bd2206c1a",
        "definition": {
          "task": "RPO v2有限URDF动作目标，原始q/dq与观察零点保持；6144环境，每个sampler2048；435M原生网络、batch2048、replay100万，每24576新转移触发16次联合更新。 RSI仅train接触候选、足底2mm间隙、有限root上移；写入后刷新FK，专家数据不改。 512m离线地面；启动核对实际网格+所有RSI足碰撞网格包络与20m余量。",
          "reward": "九项raw aux，limits_dof_vel=sum relu(abs(dq)/Vmax-0.70)^2，权重-25仅加权一次；判别奖励截[-2,2]，D lr=3e-6；Actor正则尺度上限50。",
          "failure": "真实硬位置/速度/力矩终止保持；target slew每步0.5×Vmax；Actor裁剪1，其余五Adam裁剪5，非有限梯度失败。 显式PD增益保持，physics dt=2.5ms/decimation8/50Hz；物理子步硬限位锁存，接触/高度/倾角原因分列。 每步记录世界XY与足部保守覆盖余量，越界拒绝继续采样。",
          "success": "训练/完整恢复资格16→24后新增500联合更新，最多2小时且不自动重试；训练指标不足不启动独立评估。正常行走尚未验证。",
          "settings": {}
        }
      },
      "resume_source_image": "eb/rlinf:rpo-bfm-ufo-agent-4gpu-ground-v9",
      "source_train_job_id": "trn_518be1feb13e40b4b4ac",
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
      "id": "run_13cb5781b3944be88906",
      "attempt": 1,
      "status": "finished",
      "abort_reason": null,
      "metrics_summary": {
        "checkpoint": {
          "path": "/data/kk/models/runs/run_13cb5781b3944be88906",
          "step": 524,
          "bytes": 12922256064
        },
        "sampler_sync": {
          "identity": {
            "version": 520,
            "state_sha256": "114d6cc10c56b477a66ff07fe5e21b4a470946bb4e0ac93e54a9a6da51692f7a",
            "online_transitions": 835584
          },
          "samplers": [
            {
              "rank": 0,
              "version": 520,
              "state_sha256": "114d6cc10c56b477a66ff07fe5e21b4a470946bb4e0ac93e54a9a6da51692f7a",
              "online_transitions": 835584
            },
            {
              "rank": 1,
              "version": 520,
              "state_sha256": "114d6cc10c56b477a66ff07fe5e21b4a470946bb4e0ac93e54a9a6da51692f7a",
              "online_transitions": 835584
            },
            {
              "rank": 2,
              "version": 520,
              "state_sha256": "114d6cc10c56b477a66ff07fe5e21b4a470946bb4e0ac93e54a9a6da51692f7a",
              "online_transitions": 835584
            }
          ],
          "schema_version": "eb/ufo-sampler-sync/v1",
          "sampling_rounds": 34
        },
        "model_version_id": "mv_f1c2fd66a7f6402fa7a1"
      }
    }
  ]
}
```
```json
{
  "id": "run_13cb5781b3944be88906",
  "job_id": "job_5157c51b35464239b2b6",
  "attempt": 1,
  "status": "finished",
  "abort_reason": null,
  "metrics_summary": {
    "checkpoint": {
      "path": "/data/kk/models/runs/run_13cb5781b3944be88906",
      "step": 524,
      "bytes": 12922256064
    },
    "sampler_sync": {
      "identity": {
        "version": 520,
        "state_sha256": "114d6cc10c56b477a66ff07fe5e21b4a470946bb4e0ac93e54a9a6da51692f7a",
        "online_transitions": 835584
      },
      "samplers": [
        {
          "rank": 0,
          "version": 520,
          "state_sha256": "114d6cc10c56b477a66ff07fe5e21b4a470946bb4e0ac93e54a9a6da51692f7a",
          "online_transitions": 835584
        },
        {
          "rank": 1,
          "version": 520,
          "state_sha256": "114d6cc10c56b477a66ff07fe5e21b4a470946bb4e0ac93e54a9a6da51692f7a",
          "online_transitions": 835584
        },
        {
          "rank": 2,
          "version": 520,
          "state_sha256": "114d6cc10c56b477a66ff07fe5e21b4a470946bb4e0ac93e54a9a6da51692f7a",
          "online_transitions": 835584
        }
      ],
      "schema_version": "eb/ufo-sampler-sync/v1",
      "sampling_rounds": 34
    },
    "model_version_id": "mv_f1c2fd66a7f6402fa7a1"
  },
  "gpu_ids": [
    0,
    1,
    2,
    3
  ],
  "provenance": "{\"code_ref\": \"sha256:04e20ece2e3e3f3603a98b24e04aba42023fb31a7395beddfdac2a5d522de54d\", \"config_hash\": \"af5dddd98989d8f59751d2d8ea3d1ec91e2c5633c745b56802c262850a7e2862\", \"data_version\": null, \"image_digest\": \"sha256:1034a824a3782202b914275344f6c0985d0f1721fb423e5620b5a964dc16672e\", \"seed\": null}",
  "artifacts": [
    {
      "id": "art_3df32c5c6b9440ef975b",
      "kind": "rlinf_checkpoint",
      "sha256": "sha256:90f82801d55d20c23aebd3d69500e1185526ba3b5d0256bdc12bb6c8886ab412",
      "size_bytes": 12922256064,
      "storage": "/data/kk/models/runs/run_13cb5781b3944be88906"
    }
  ]
}
```

### 最终不可变模型版本，无评估报告

```json
{
  "id": "mv_f1c2fd66a7f6402fa7a1",
  "family_id": "mf_rpo_v2_bfm_zero",
  "name": "RPO-UFO-ground-v9-step524",
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
    "run_id": "run_13cb5781b3944be88906",
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
      "image": "sha256:1034a824a3782202b914275344f6c0985d0f1721fb423e5620b5a964dc16672e",
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
      "recipe_id": "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_ground_v9",
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
    "template_id": "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_ground_v9",
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
    "train_job_id": "job_5157c51b35464239b2b6",
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
    "artifact_sha256": "sha256:90f82801d55d20c23aebd3d69500e1185526ba3b5d0256bdc12bb6c8886ab412",
    "checkpoint_path": "/data/kk/models/runs/run_13cb5781b3944be88906",
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
      "id": "rpo_bfm_ufo_isaac_4gpu_rsi_ground_v9",
      "hash": "sha256:fcf30a6403393bf675d7dc4985ab86267cfd5859c325391717758e8bd2206c1a",
      "definition": {
        "task": "RPO v2有限URDF动作目标，原始q/dq与观察零点保持；6144环境，每个sampler2048；435M原生网络、batch2048、replay100万，每24576新转移触发16次联合更新。 RSI仅train接触候选、足底2mm间隙、有限root上移；写入后刷新FK，专家数据不改。 512m离线地面；启动核对实际网格+所有RSI足碰撞网格包络与20m余量。",
        "reward": "九项raw aux，limits_dof_vel=sum relu(abs(dq)/Vmax-0.70)^2，权重-25仅加权一次；判别奖励截[-2,2]，D lr=3e-6；Actor正则尺度上限50。",
        "failure": "真实硬位置/速度/力矩终止保持；target slew每步0.5×Vmax；Actor裁剪1，其余五Adam裁剪5，非有限梯度失败。 显式PD增益保持，physics dt=2.5ms/decimation8/50Hz；物理子步硬限位锁存，接触/高度/倾角原因分列。 每步记录世界XY与足部保守覆盖余量，越界拒绝继续采样。",
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
  "created_at": "2026-10-12T01:07:11.307771Z",
  "train_job_name": "RPO UFO ground v9 500-update observation",
  "train_page_id": "trn_8bdabc160cd04d2bb100",
  "artifact": {
    "id": "art_3df32c5c6b9440ef975b",
    "kind": "rlinf_checkpoint",
    "sha256": "sha256:90f82801d55d20c23aebd3d69500e1185526ba3b5d0256bdc12bb6c8886ab412",
    "size_bytes": 12922256064,
    "storage": "/data/kk/models/runs/run_13cb5781b3944be88906",
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
            "sha256": "3edaf0ac918d14db889c779c684c40362c3bcc8138cc7aa5cc12213a008f1509"
          },
          {
            "path": "model/native/global_step_524/optimizers.pth",
            "bytes": 3481181041,
            "sha256": "30071766dd0ab47f1e93aec3a61595ca7f58c5d049a4f3210838ac3395256e5d"
          },
          {
            "path": "model/native/global_step_524/resume-manifest.json",
            "bytes": 25055,
            "sha256": "4d4ffee4423fd3eb0e48521bfe4252e44207e9b3ec799c6dabd2930b9c5460d3"
          },
          {
            "path": "model/native/global_step_524/trainer-state.pt",
            "bytes": 6097185212,
            "sha256": "c4cf327d1d42ac5e275c3ebebc6ee9a38d4e593d458e2d6621d2d5524caeec65"
          },
          {
            "path": "model/native-checkpoint.json",
            "bytes": 88,
            "sha256": "24414cc2fa26deb57f6445d810747cdb6906f32b41b6b6102e7eb39cbc5debcc"
          },
          {
            "path": "model/resolved-config.yaml",
            "bytes": 3866,
            "sha256": "41ab56fceadc35a2c1731adb7bd3cda82316d8e519823cdd61a99a9b7fb53052"
          },
          {
            "path": "model/source-receipt.json",
            "bytes": 2549,
            "sha256": "b43b47f3d41e9d5db4e853581d61ee38287771c7558f360045278db06547a5a9"
          }
        ],
        "format": "ufo_agent_state_v1",
        "run_id": "run_13cb5781b3944be88906",
        "status": "complete",
        "execution": {
          "recipe_id": "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_ground_v9",
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
        "template_id": "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_ground_v9",
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

### 镜像来源、原生UFO/RLinf与所有补丁哈希

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
  "auxiliary_source_sha256": "sha256:8359d881e7a3523a740b10fb01f4ee53e70a56ecaeb0f31ece3d0fb2e94c03d5",
  "auxiliary_profile_patch_hash": "sha256:30e7b4ea7a3ec0e45cdbe0a9dbf718eb219c2c6fd36d07f9b18bd5222183a03a",
  "scaling_profile": "env6144_batch2048_v6",
  "scaling_profile_patch_hash": "sha256:58a2dca8377ed2ede804c7101b39bc6992ec669ef95e610ddec4b2a59bfc6b20",
  "control_profile": "reachable_v2",
  "optimizer_profile": "stabilized_v7",
  "stabilization_patch_hash": "sha256:564f5452e12e3179112e6aa3020efba4cbcedc0ad90dd26b98aed78a50dd85b1",
  "contact_profile": "contact_stable_v8",
  "contact_patch_hash": "sha256:e20a0ff930b3b9430ece70c21ab316a06cef92b679b638c0a35d7fed62d6d814",
  "ground_profile": "wide_plane_v9",
  "ground_patch_hash": "sha256:554a18609c9b8a420811fa6b1ded908a4bdf181e52e8d513c8772de6084eff59"
}
```

### 完整本体编码和ground-v9物理控制协议

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
    }
  }
}
```

### 完整解析后RLinf配置

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
      "contact_profile": "contact_stable_v8",
      "ground_profile": "wide_plane_v9",
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
      "experiment_name": "RPO UFO ground v9 500-update observation",
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
        "id": "RPO-UFO-Isaac-RSI-ground-v9",
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

### 完整原生网络、优化器和九项aux设置

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

网络435116484参数：F/Actor/Critic/aux Critic 2048×6，D1024×3，B256×1，latent256；actor375、FB733、动作23。batch2048、CPU replay100万；每round6144×4=24576转移，每触发16联合更新。Formal500更新仅新增786432转移，每env128步=2.56秒物理探索，未靠大量并发弥补时间跨度。

继承稳定化保护：Actor范数裁剪1，其余五优化器5；native clip_grad_norm=0由注册hook实际裁剪，记录前后。Actor正则尺度detach(mean(abs(Q_fb)))≤50，D lr3e-6/reward[-2,2]同时记raw；第九raw速度罚=sum relu(abs(dq)/Vmax−.70)^2、固定权重−25，再由原生reward normalizer处理。第九仍取20ms末dq，不能反映全部子步峰值；真实硬终止是8个子步锁存。未接入默认RPO RewardManager的29项，其env reward为占位零，实际优化仅FB、D和9aux。

Tracking rollout=False，保留原生每环境100步latent周期更新；参考目标不因另一env reset随机跳变。所有state写入仅train候选，expert连续帧与train/validation/acceptance隔离和数据字节未修改。PD、摩擦、质量与COM随机化完整值及诊断均直接给在后面的历史对照附录中；I与H这些值完全相同。

## 恢复和物理证据

### 16完整checkpoint与六Adam

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
  "fingerprint": "sha256:fca3e90f09d5e6656bf0aef4d12402334e5d372c8e54ba69cd84b7f85b0c33f7",
  "execution": {
    "schema_version": "eb/training-binding/v1",
    "recipe_id": "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_ground_v9",
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

### 24完整checkpoint与六Adam

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
  "fingerprint": "sha256:fca3e90f09d5e6656bf0aef4d12402334e5d372c8e54ba69cd84b7f85b0c33f7",
  "execution": {
    "schema_version": "eb/training-binding/v1",
    "recipe_id": "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_ground_v9",
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

### 524完整checkpoint与六Adam

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
  "fingerprint": "sha256:fca3e90f09d5e6656bf0aef4d12402334e5d372c8e54ba69cd84b7f85b0c33f7",
  "execution": {
    "schema_version": "eb/training-binding/v1",
    "recipe_id": "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_ground_v9",
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

### 最终原生checkpoint全部文件大小和SHA256

```json
{
  "schema_version": "eb/resume-checkpoint/v1",
  "status": "complete",
  "format": "ufo_agent_state_v1",
  "mode": "trainer_state",
  "step": 524,
  "primary_unit": "learner_joint_update",
  "run_id": "run_13cb5781b3944be88906",
  "execution": {
    "schema_version": "eb/training-binding/v1",
    "recipe_id": "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_ground_v9",
    "algorithm_id": "bfm_fb_cpr_aux",
    "provider_id": "ufo",
    "runtime_id": "rlinf",
    "simulator_id": "isaaclab",
    "progress_unit": "learner_joint_update",
    "checkpoint_format": "ufo_agent_state_v1",
    "topology_id": "one_learner_three_samplers"
  },
  "compatibility_fingerprint": "sha256:fca3e90f09d5e6656bf0aef4d12402334e5d372c8e54ba69cd84b7f85b0c33f7",
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
      "bytes": 6250,
      "sha256": "f7f67e6c591a432bf4aa465039ed9f2bc8b97d7571d36aa94740d08892ef86a0"
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
      "sha256": "3edaf0ac918d14db889c779c684c40362c3bcc8138cc7aa5cc12213a008f1509"
    },
    {
      "path": "optimizers.pth",
      "bytes": 3481181041,
      "sha256": "30071766dd0ab47f1e93aec3a61595ca7f58c5d049a4f3210838ac3395256e5d"
    },
    {
      "path": "trainer-state.pt",
      "bytes": 6097185212,
      "sha256": "c4cf327d1d42ac5e275c3ebebc6ee9a38d4e593d458e2d6621d2d5524caeec65"
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

### 24恢复后的131列trace

```json
{
  "rows": 24576,
  "rounds": 1,
  "round_first": 3,
  "round_last": 3,
  "requested_saturation_fraction": 0.39311594202898553,
  "applied_saturation_fraction": 0.035152711730072464,
  "hard_position_velocity_torque_counts": [
    58,
    861,
    0
  ],
  "contact_height_orientation_counts": [
    13,
    0,
    2
  ],
  "max_boundary_joint_velocity_radps": 30.840190887451172,
  "max_physics_joint_velocity_radps": 47.55874252319336,
  "max_boundary_location": {
    "round": 3,
    "row": 10693,
    "joint": "right_knee_joint",
    "dq_radps": -30.840190887451172,
    "episode_step": 1,
    "requested": -0.9589251279830933,
    "applied": -0.9589251279830933,
    "pd_target_rad": -0.14454901218414307,
    "torque_Nm": 18.11861228942871,
    "hard_flags": [
      0,
      1,
      0
    ],
    "termination_reasons_contact_height_tilt": [
      0,
      0,
      0
    ],
    "foot_peak_N": [
      0.0,
      4559.41064453125
    ],
    "physics_peak_radps": 30.840190887451172
  },
  "max_foot_contact_force_N": 14583.4775390625,
  "max_reset_root_lift_m": 0.021992022171616554,
  "latched_velocity_terms_without_boundary_violation": 737,
  "soft_velocity_raw_mean": 0.003480361667264766,
  "any_joint_over_soft_limit_fraction": 0.0889892578125,
  "retained_replay_rows_checked": 24576,
  "terminal_episode_changes_checked": 660,
  "pd_decoder_max_error_rad": 1.941382885561893e-07,
  "reset_in_retained_replay": 6804,
  "first_step_terms_in_retained_replay": 39,
  "first_step_term_fraction": 0.005731922398589065,
  "first_step_hard_counts": [
    0,
    31,
    0
  ],
  "first_step_hard_union": 31,
  "first_step_contact_height_orientation_counts": [
    8,
    0,
    0
  ],
  "first_step_unexplained_terms": 0,
  "first_step_peak_foot_force_quantiles_N": {
    "0": 0.0,
    "0.5": 0.0,
    "0.95": 502.7730316162109,
    "0.99": 1214.273217773438,
    "1": 5669.33837890625
  },
  "reset_lift_quantiles_m": {
    "0": 0.0,
    "0.5": 0.0,
    "0.95": 0.016627678647637357,
    "1": 0.021992022171616554
  },
  "ground_minimum_foot_margin_m": 193.8170166015625,
  "ground_outside_count": 0,
  "trace_width": 131,
  "physics_dt_s": 0.0025,
  "policy_dt_s": 0.02,
  "hard_flags_are_physics_latched": true,
  "replay_actions_are_requested": true,
  "ninth_reward_matches_boundary_dq": true
}
```

### 正式500新增更新131列trace

```json
{
  "rows": 786432,
  "rounds": 32,
  "round_first": 4,
  "round_last": 35,
  "requested_saturation_fraction": 0.4400628684223562,
  "applied_saturation_fraction": 0.1581567404926687,
  "hard_position_velocity_torque_counts": [
    12372,
    27146,
    0
  ],
  "contact_height_orientation_counts": [
    4930,
    14882,
    14223
  ],
  "max_boundary_joint_velocity_radps": 45.9852180480957,
  "max_physics_joint_velocity_radps": 67.52462005615234,
  "max_boundary_location": {
    "round": 13,
    "row": 1058,
    "joint": "right_ankle_pitch_joint",
    "dq_radps": 45.9852180480957,
    "episode_step": 0,
    "requested": -0.9567300081253052,
    "applied": -0.4177906811237335,
    "pd_target_rad": -0.25067442655563354,
    "torque_Nm": 0.24385976791381836,
    "hard_flags": [
      0,
      1,
      0
    ],
    "termination_reasons_contact_height_tilt": [
      0,
      0,
      0
    ],
    "foot_peak_N": [
      0.0,
      10049.83984375
    ],
    "physics_peak_radps": 45.9852180480957
  },
  "max_foot_contact_force_N": 25652.88671875,
  "max_reset_root_lift_m": 0.021992022171616554,
  "latched_velocity_terms_without_boundary_violation": 21733,
  "soft_velocity_raw_mean": 0.013641493235039685,
  "any_joint_over_soft_limit_fraction": 0.3333180745442708,
  "retained_replay_rows_checked": 786432,
  "terminal_episode_changes_checked": 67056,
  "pd_decoder_max_error_rad": 2.801418306219716e-07,
  "reset_in_retained_replay": 73200,
  "first_step_terms_in_retained_replay": 366,
  "first_step_term_fraction": 0.005,
  "first_step_hard_counts": [
    1,
    286,
    0
  ],
  "first_step_hard_union": 287,
  "first_step_contact_height_orientation_counts": [
    79,
    0,
    0
  ],
  "first_step_unexplained_terms": 0,
  "first_step_peak_foot_force_quantiles_N": {
    "0": 0.0,
    "0.5": 0.0,
    "0.95": 573.691525268555,
    "0.99": 1319.1183508300721,
    "1": 10049.83984375
  },
  "reset_lift_quantiles_m": {
    "0": 0.0,
    "0.5": 0.0,
    "0.95": 0.016553590074181557,
    "1": 0.021992022171616554
  },
  "ground_minimum_foot_margin_m": 193.26023864746094,
  "ground_outside_count": 0,
  "trace_width": 131,
  "physics_dt_s": 0.0025,
  "policy_dt_s": 0.02,
  "hard_flags_are_physics_latched": true,
  "replay_actions_are_requested": true,
  "ninth_reward_matches_boundary_dq": true
}
```

### 最终潜目标与RSI关节残差

```json
{
  "scope": "all final retained replay rows; chronological complete 6144-row blocks",
  "same_episode_nonperiodic_pairs": 658009,
  "passive_z_changes": 0,
  "native_period_100_pairs_excluded": 0,
  "cosine_median": 1.0,
  "reset_rows": 87980,
  "reset_joint_residuals": [
    {
      "joint": "left_thigh_yaw_joint",
      "outside_count": 0,
      "outside_fraction": 0.0,
      "mean_residual_rad": 0.0,
      "p95_residual_rad": 0.0,
      "max_residual_rad": 0.0
    },
    {
      "joint": "right_thigh_yaw_joint",
      "outside_count": 0,
      "outside_fraction": 0.0,
      "mean_residual_rad": 0.0,
      "p95_residual_rad": 0.0,
      "max_residual_rad": 0.0
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
      "mean_residual_rad": 0.0,
      "p95_residual_rad": 0.0,
      "max_residual_rad": 0.0
    },
    {
      "joint": "right_thigh_roll_joint",
      "outside_count": 0,
      "outside_fraction": 0.0,
      "mean_residual_rad": 0.0,
      "p95_residual_rad": 0.0,
      "max_residual_rad": 0.0
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
      "mean_residual_rad": 2.6557195696423613e-12,
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
      "mean_residual_rad": 0.0,
      "p95_residual_rad": 0.0,
      "max_residual_rad": 0.0
    },
    {
      "joint": "right_knee_joint",
      "outside_count": 0,
      "outside_fraction": 0.0,
      "mean_residual_rad": 0.0,
      "p95_residual_rad": 0.0,
      "max_residual_rad": 0.0
    },
    {
      "joint": "left_arm_yaw_joint",
      "outside_count": 0,
      "outside_fraction": 0.0,
      "mean_residual_rad": 0.0,
      "p95_residual_rad": 0.0,
      "max_residual_rad": 0.0
    },
    {
      "joint": "right_arm_yaw_joint",
      "outside_count": 0,
      "outside_fraction": 0.0,
      "mean_residual_rad": 0.0,
      "p95_residual_rad": 0.0,
      "max_residual_rad": 0.0
    },
    {
      "joint": "left_ankle_pitch_joint",
      "outside_count": 0,
      "outside_fraction": 0.0,
      "mean_residual_rad": 0.0,
      "p95_residual_rad": 0.0,
      "max_residual_rad": 0.0
    },
    {
      "joint": "right_ankle_pitch_joint",
      "outside_count": 0,
      "outside_fraction": 0.0,
      "mean_residual_rad": 0.0,
      "p95_residual_rad": 0.0,
      "max_residual_rad": 0.0
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

### 1024真实reset足底网格几何

```json
{
  "method": "reconstruct first-step replay q, projected gravity and privileged root height; official collision-mesh convex-hull FK, flat-ground yaw invariant",
  "scope": "evenly selected retained RSI first-step states; offline geometry, not physical safety acceptance",
  "reset_rows": 87980,
  "samples": 1024,
  "sole_height_quantiles_m": {
    "0": 0.0019999114479741023,
    "0.05": 0.0019999874000109306,
    "0.5": 0.010686482792163388,
    "0.95": 0.018854764939003003,
    "1": 0.019999462959494917
  },
  "penetration_over_2mm": 0,
  "penetration_over_2cm": 0,
  "both_feet_airborne_over_5cm": 0,
  "sole_below_1mm": 0
}
```

### GPU启动快照

```json
{
  "hostname": "eai4090x4m1",
  "gpus": [
    {
      "index": 0,
      "name": "NVIDIA GeForce RTX 4090",
      "memory_total_mb": 24564,
      "memory_free_mb": 24080,
      "run_id": "run_13cb5781b3944be88906"
    },
    {
      "index": 1,
      "name": "NVIDIA GeForce RTX 4090",
      "memory_total_mb": 24564,
      "memory_free_mb": 24080,
      "run_id": "run_13cb5781b3944be88906"
    },
    {
      "index": 2,
      "name": "NVIDIA GeForce RTX 4090",
      "memory_total_mb": 24564,
      "memory_free_mb": 24080,
      "run_id": "run_13cb5781b3944be88906"
    },
    {
      "index": 3,
      "name": "NVIDIA GeForce RTX 4090",
      "memory_total_mb": 24564,
      "memory_free_mb": 24080,
      "run_id": "run_13cb5781b3944be88906"
    }
  ],
  "memory_total_mb": 128644,
  "has_dri": true,
  "memory_available_mb": 104463
}
```

### GPU模型加载快照

```json
{
  "hostname": "eai4090x4m1",
  "gpus": [
    {
      "index": 0,
      "name": "NVIDIA GeForce RTX 4090",
      "memory_total_mb": 24564,
      "memory_free_mb": 13840,
      "run_id": "run_13cb5781b3944be88906"
    },
    {
      "index": 1,
      "name": "NVIDIA GeForce RTX 4090",
      "memory_total_mb": 24564,
      "memory_free_mb": 17804,
      "run_id": "run_13cb5781b3944be88906"
    },
    {
      "index": 2,
      "name": "NVIDIA GeForce RTX 4090",
      "memory_total_mb": 24564,
      "memory_free_mb": 17757,
      "run_id": "run_13cb5781b3944be88906"
    },
    {
      "index": 3,
      "name": "NVIDIA GeForce RTX 4090",
      "memory_total_mb": 24564,
      "memory_free_mb": 17804,
      "run_id": "run_13cb5781b3944be88906"
    }
  ],
  "memory_total_mb": 128644,
  "has_dri": true,
  "memory_available_mb": 73501
}
```

### GPU训练中段快照

```json
{
  "hostname": "eai4090x4m1",
  "gpus": [
    {
      "index": 0,
      "name": "NVIDIA GeForce RTX 4090",
      "memory_total_mb": 24564,
      "memory_free_mb": 7854,
      "run_id": "run_13cb5781b3944be88906"
    },
    {
      "index": 1,
      "name": "NVIDIA GeForce RTX 4090",
      "memory_total_mb": 24564,
      "memory_free_mb": 17376,
      "run_id": "run_13cb5781b3944be88906"
    },
    {
      "index": 2,
      "name": "NVIDIA GeForce RTX 4090",
      "memory_total_mb": 24564,
      "memory_free_mb": 17329,
      "run_id": "run_13cb5781b3944be88906"
    },
    {
      "index": 3,
      "name": "NVIDIA GeForce RTX 4090",
      "memory_total_mb": 24564,
      "memory_free_mb": 17442,
      "run_id": "run_13cb5781b3944be88906"
    }
  ],
  "memory_total_mb": 128644,
  "has_dri": true,
  "memory_available_mb": 64708
}
```

### 结束GPU资源与租约

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
  "memory_available_mb": 104174
}
```

### H+I新增证据存储账目

```json
{
  "scope": "all new H two diagnostics/two qualification chains/H500 and I16/24/500; existing G/data excluded",
  "allocated_unique_bytes": 70406496256,
  "unique_inode_bytes": 70405714178,
  "apparent_bytes": 140036175071,
  "roots_apparent_bytes": {
    "/data/kk/models/runs/run_0f51028109cb4b2e9c12": 0,
    "/data/kk/embodiedbein/runs/run_0f51028109cb4b2e9c12": 9782834,
    "/data/kk/models/runs/run_7e3d481401804da2b688": 14422258506,
    "/data/kk/embodiedbein/runs/run_7e3d481401804da2b688": 22105426,
    "/data/kk/models/runs/run_c768a1338d1e46fe8092": 14768395628,
    "/data/kk/embodiedbein/runs/run_c768a1338d1e46fe8092": 10555471,
    "/data/kk/models/runs/run_ad47cf6f8cfd4355bb44": 25844504544,
    "/data/kk/embodiedbein/runs/run_ad47cf6f8cfd4355bb44": 323321218,
    "/data/kk/models/runs/run_a1ce3855d3894e97a17b": 0,
    "/data/kk/embodiedbein/runs/run_a1ce3855d3894e97a17b": 9783140,
    "/data/kk/models/runs/run_6fab081aacc04b11bee4": 14422258496,
    "/data/kk/embodiedbein/runs/run_6fab081aacc04b11bee4": 22123261,
    "/data/kk/models/runs/run_0c2d2f704e1a433db877": 14768395746,
    "/data/kk/embodiedbein/runs/run_0c2d2f704e1a433db877": 10549289,
    "/data/kk/models/runs/run_f7950a57b6e041b09023": 14422259301,
    "/data/kk/embodiedbein/runs/run_f7950a57b6e041b09023": 22697639,
    "/data/kk/models/runs/run_e077e5c07f794755aac9": 14768396553,
    "/data/kk/embodiedbein/runs/run_e077e5c07f794755aac9": 10870396,
    "/data/kk/models/runs/run_13cb5781b3944be88906": 25844505467,
    "/data/kk/embodiedbein/runs/run_13cb5781b3944be88906": 333412156
  },
  "additional_work_bytes": 23003624,
  "repo_artifact_bytes": 22840363,
  "image_and_staging_reserved_bytes": 12884901888,
  "budget_bytes": 103079215104,
  "within_budget": true
}
```

正式任务墙钟=2011.84336秒，learner日志墙钟=1077.88269秒；初始化、保存、发布、收集不在后者中。六Adam24→524一致，每个增加500次；模型、RNG、replay、expert sampler、z pool与rollout恢复，仿真内部物理状态不可恢复，边界重新RSI，非逐位轨迹恢复。

## 与受地面缺陷影响H500的观测对照

| 指标 | H首100 | H末100 | I首100 | I末100 |
|---|---|---|---|---|
| env/episode_len | 7.47538356 | 11.8562354 | 7.13045224 | 13.2711236 |
| train/ufo/hard_velocity_count | 744.833333 | 598 | 1131.5 | 690 |
| train/ufo/hard_position_count | 379.666667 | 289.714286 | 419.5 | 339.142857 |
| train/ufo/contact_termination_count | 188.166667 | 162.428571 | 136.166667 | 156 |
| train/ufo/height_termination_count | 621.666667 | 903.142857 | 325 | 524.857143 |
| train/ufo/critic_loss | 0.71755628 | 0.627521253 | 0.741080195 | 0.55125182 |
| train/ufo/applied_action_saturation_fraction | 0.10603399 | 0.163640182 | 0.0999690983 | 0.176634986 |

H和I新初始化且接触/轨迹随机过程不同，H大量边缘env没有地面支撑；修正后接触和越限计数变化不能当简单同分布质量差值。潜目标/FB loss改善不代替动态行走验收。原因flags允许重叠，分母和真实terminated见trace，不能直接求和。

## 每采样round原始观测

| 联合更新 | env/episode_len | train/ufo/online_transitions | train/ufo/first_step_samples | train/ufo/first_step_terminated | train/ufo/hard_velocity_count | train/ufo/contact_termination_count | train/ufo/height_termination_count | train/ufo/orientation_termination_count | train/ufo/physics_peak_joint_velocity | train/ufo/foot_contact_peak_N | train/ufo/ground_foot_margin_min_m | train/ufo/ground_outside_count |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 40 | 3.03896093 | 98304 | 6680 | 41 | 763 | 22 | 0 | 0 | 31.1331291 | 9481.13281 | 193.702866 | 0 |
| 56 | 6.15932226 | 122880 | 1477 | 7 | 1313 | 140 | 14 | 50 | 28.5076294 | 12904.0254 | 193.393799 | 0 |
| 72 | 8.72260857 | 147456 | 3190 | 16 | 1537 | 204 | 797 | 697 | 46.5948868 | 25652.8867 | 193.367722 | 0 |
| 88 | 7.86842108 | 172032 | 2962 | 16 | 1096 | 171 | 374 | 896 | 67.5246201 | 17608.3672 | 193.6754 | 0 |
| 104 | 8.22075653 | 196608 | 2744 | 19 | 1029 | 142 | 321 | 1048 | 30.6655102 | 13917.7686 | 193.68309 | 0 |
| 120 | 8.77264404 | 221184 | 2801 | 12 | 1051 | 138 | 444 | 818 | 33.8757706 | 13038.7266 | 193.739929 | 0 |
| 136 | 9.13325119 | 245760 | 2488 | 12 | 925 | 126 | 435 | 683 | 43.1940765 | 14830.5576 | 193.738785 | 0 |
| 152 | 9.48244476 | 270336 | 2311 | 9 | 861 | 123 | 479 | 626 | 41.9312363 | 17725.9121 | 193.724808 | 0 |
| 168 | 9.77954578 | 294912 | 2266 | 10 | 823 | 115 | 440 | 587 | 43.4629021 | 17954.7305 | 193.454941 | 0 |
| 184 | 10.522727 | 319488 | 2275 | 12 | 825 | 117 | 510 | 624 | 46.3026505 | 25049.293 | 193.266953 | 0 |
| 200 | 10.5713024 | 344064 | 2311 | 13 | 857 | 174 | 521 | 492 | 33.8797302 | 13113.5205 | 193.298615 | 0 |
| 216 | 11.0335169 | 368640 | 2178 | 6 | 789 | 137 | 488 | 525 | 39.6481209 | 13798.8281 | 193.260239 | 0 |
| 232 | 10.7999067 | 393216 | 2161 | 9 | 804 | 190 | 454 | 502 | 61.3483391 | 14194.916 | 193.534225 | 0 |
| 248 | 10.9572496 | 417792 | 2184 | 11 | 796 | 186 | 514 | 546 | 39.2002563 | 14569.1221 | 193.679321 | 0 |
| 264 | 10.7829313 | 442368 | 2179 | 15 | 822 | 164 | 486 | 512 | 36.1993561 | 17659.1211 | 193.697662 | 0 |
| 280 | 11.143054 | 466944 | 2196 | 6 | 796 | 216 | 456 | 515 | 34.0622673 | 22836.5273 | 193.605896 | 0 |
| 296 | 11.5072193 | 491520 | 2166 | 11 | 811 | 172 | 493 | 483 | 53.9670944 | 13824.8398 | 193.596695 | 0 |
| 312 | 11.0161924 | 516096 | 2071 | 10 | 829 | 181 | 486 | 364 | 55.3680267 | 15325.7373 | 193.679993 | 0 |
| 328 | 11.1684313 | 540672 | 2074 | 9 | 921 | 170 | 473 | 308 | 37.0650444 | 16121.5273 | 193.682037 | 0 |
| 344 | 11.3931236 | 565248 | 2030 | 12 | 878 | 146 | 482 | 318 | 40.7375336 | 16729.0215 | 193.693878 | 0 |
| 360 | 11.6181345 | 589824 | 1920 | 8 | 819 | 147 | 472 | 331 | 44.7268295 | 13597.5176 | 193.608307 | 0 |
| 376 | 12.298974 | 614400 | 1914 | 14 | 780 | 168 | 467 | 352 | 36.0065994 | 15007.2588 | 193.502075 | 0 |
| 392 | 12.5837812 | 638976 | 1910 | 10 | 734 | 157 | 498 | 336 | 35.7659912 | 18991.8926 | 193.495972 | 0 |
| 408 | 12.4919481 | 663552 | 1930 | 8 | 743 | 162 | 533 | 311 | 39.4929504 | 13256.0986 | 193.493423 | 0 |
| 424 | 12.6276655 | 688128 | 1897 | 8 | 714 | 170 | 571 | 290 | 34.0024452 | 15908.4561 | 193.441986 | 0 |
| 440 | 12.6775627 | 712704 | 1874 | 6 | 701 | 162 | 489 | 278 | 38.9689217 | 16153.6426 | 193.450119 | 0 |
| 456 | 13.271987 | 737280 | 1794 | 7 | 676 | 188 | 517 | 277 | 45.5115891 | 15901.6826 | 193.473404 | 0 |
| 472 | 13.5155821 | 761856 | 1852 | 6 | 663 | 153 | 530 | 289 | 35.0741959 | 21438.7734 | 193.494095 | 0 |
| 488 | 13.0827513 | 786432 | 1851 | 6 | 756 | 150 | 512 | 280 | 31.9620247 | 15211.0596 | 193.48996 | 0 |
| 504 | 13.1500273 | 811008 | 1831 | 11 | 695 | 161 | 508 | 285 | 46.233326 | 14808.667 | 193.475021 | 0 |
| 520 | 13.7021503 | 835584 | 1855 | 16 | 670 | 159 | 545 | 305 | 28.8960819 | 19881.9648 | 193.598602 | 0 |
| 524 | 13.4978046 | 860160 | 1828 | 10 | 669 | 119 | 573 | 295 | 35.6554413 | 13696.3135 | 193.591064 | 0 |

## 全部138项标量统计

| tag | n | first_step | last_step | first | last | mean | min | max | median | p95 | last50_mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| env/episode_len | 32 | 40 | 524 | 3.03896093 | 13.4978046 | 10.8309993 | 3.03896093 | 13.7021503 | 11.0882854 | 13.5058045 | 10.8309993 |
| env/num_trajectories | 32 | 40 | 524 | 847 | 1822 | 2109.6875 | 847 | 3450 | 2058 | 2787.5 | 2109.6875 |
| env/return | 32 | 40 | 524 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| env/reward | 32 | 40 | 524 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| env/success_once | 32 | 40 | 524 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| rollout/ufo/received_transitions | 32 | 40 | 524 | 24576 | 24576 | 24576 | 24576 | 24576 | 24576 | 24576 | 24576 |
| time/actor_training | 32 | 40 | 524 | 14.1629333 | 3.49031281 | 11.4221138 | 3.49031281 | 14.1629333 | 11.5657544 | 11.7159786 | 11.4221138 |
| time/cal_adv_and_returns | 32 | 40 | 524 | 0.00277471542 | 0.013684988 | 0.00548151135 | 0.00229001045 | 0.013684988 | 0.00377583504 | 0.0123087406 | 0.00548151135 |
| time/env/compute_bootstrap_rewards | 32 | 40 | 524 | 0.000234920764 | 0.00026100385 | 0.000249741868 | 0.000213108957 | 0.00029017264 | 0.000249573262 | 0.00028644976 | 0.000249741868 |
| time/env/env/bootstrap_step | 32 | 40 | 524 | 0.0157786608 | 0.000252563041 | 0.000759577568 | 0.000233171042 | 0.0157786608 | 0.000264268019 | 0.000370719016 | 0.000759577568 |
| time/env/env/send_rollout_trajectories | 32 | 40 | 524 | 0.272168905 | 0.231297851 | 0.226517146 | 0.209913269 | 0.272168905 | 0.222869679 | 0.257929043 | 0.226517146 |
| time/env/env_interact_step | 32 | 40 | 524 | 3.05884027 | 1.20410252 | 1.27819378 | 1.11902916 | 3.05884027 | 1.17718774 | 1.43218246 | 1.27819378 |
| time/env/interact | 32 | 40 | 524 | 4.54917336 | 2.24294925 | 2.2200084 | 2.03035307 | 4.54917336 | 2.1375463 | 2.24456898 | 2.2200084 |
| time/env/run_interact_once | 32 | 40 | 524 | 4.54912138 | 2.24292684 | 2.21998182 | 2.03032398 | 4.54912138 | 2.13751781 | 2.24454592 | 2.21998182 |
| time/generate_rollouts | 32 | 40 | 524 | 4.75713444 | 2.40608263 | 2.41771952 | 2.19372559 | 4.75713444 | 2.34284055 | 2.46766986 | 2.41771952 |
| time/rollout/generate_one_epoch | 32 | 40 | 524 | 4.30237722 | 2.0054903 | 1.98577713 | 1.79186797 | 4.30237722 | 1.9133864 | 2.00909745 | 1.98577713 |
| time/rollout/predict | 32 | 40 | 524 | 0.307204068 | 0.049801033 | 0.0542518642 | 0.0350163952 | 0.307204068 | 0.0465748888 | 0.0550169121 | 0.0542518642 |
| time/rollout/rollout/generate | 32 | 40 | 524 | 4.30625296 | 2.00559258 | 1.9862828 | 1.7919811 | 4.30625296 | 1.91412318 | 2.00920359 | 1.9862828 |
| time/rollout/sync_model_from_actor | 32 | 40 | 524 | 14.7542267 | 13.0030098 | 13.3786199 | 12.7762251 | 14.7542267 | 13.3426738 | 13.9560691 | 13.3786199 |
| time/step | 32 | 40 | 524 | 37.2954597 | 22.3845119 | 30.9135511 | 22.3845119 | 37.2954597 | 30.9500618 | 31.5439387 | 30.9135511 |
| time/sync_weights | 32 | 40 | 524 | 18.3719406 | 16.4738007 | 17.0677023 | 16.4738007 | 18.3719406 | 17.026783 | 17.6530441 | 17.0677023 |
| train/ufo/B | 500 | 25 | 524 | -0.0198900066 | -0.00209502503 | -0.00565088357 | -0.022220321 | 0.0013198232 | -0.00513838138 | -0.00124831582 | -0.00200883892 |
| train/ufo/B_norm | 500 | 25 | 524 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 |
| train/ufo/F1 | 500 | 25 | 524 | 0.0553178266 | 0.00868415087 | 0.0127092383 | -0.0101574752 | 0.0553178266 | 0.011966276 | 0.0297281645 | 0.0128612815 |
| train/ufo/M1 | 500 | 25 | 524 | 1.25925779 | 0.746983171 | 1.14587789 | -0.0954607129 | 7.110219 | 0.847457618 | 2.57779632 | 0.78810779 |
| train/ufo/Q1 | 500 | 25 | 524 | -3.68026423 | -4.12542105 | -2.54800515 | -4.22581005 | -1.41038227 | -2.42708886 | -1.54815942 | -3.87135755 |
| train/ufo/Q_aux | 500 | 25 | 524 | -5.39310503 | -6.2729845 | -4.78514101 | -6.2729845 | -3.72498107 | -4.63510036 | -3.94566101 | -5.87776372 |
| train/ufo/Q_discriminator | 500 | 25 | 524 | -3.33766317 | -4.31619883 | -2.60424121 | -4.31619883 | -1.4048264 | -2.48962569 | -1.56556405 | -3.93337044 |
| train/ufo/Q_fb | 500 | 25 | 524 | 0.622042298 | 53.9618759 | 27.8667447 | -4.49033308 | 54.3098984 | 28.2731113 | 50.4694183 | 50.5198543 |
| train/ufo/action_abs_mean | 32 | 40 | 524 | 0.885911345 | 0.897840202 | 0.902983485 | 0.885911345 | 0.91175139 | 0.902051747 | 0.91091516 | 0.902983485 |
| train/ufo/action_saturation_fraction | 32 | 40 | 524 | 0.405425578 | 0.433875054 | 0.44006287 | 0.405425578 | 0.452456266 | 0.439814031 | 0.451149757 | 0.44006287 |
| train/ufo/actor_loss | 500 | 25 | 524 | 2.25679874 | -36.8983955 | -19.7123943 | -38.604126 | 7.16370583 | -21.2296791 | -0.250153434 | -34.8086643 |
| train/ufo/actor_optimizer/grad_clip_limit | 500 | 25 | 524 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| train/ufo/actor_optimizer/grad_norm_after_clip_l2 | 500 | 25 | 524 | 0.999999344 | 0.999999881 | 0.999999726 | 0.999999225 | 0.99999994 | 0.999999762 | 0.999999881 | 0.999999826 |
| train/ufo/actor_optimizer/grad_norm_l2 | 500 | 25 | 524 | 1.35000324 | 6.27679682 | 4.17940629 | 1.18243361 | 6.27679682 | 4.24614286 | 5.90814102 | 5.87455791 |
| train/ufo/actor_optimizer/lr | 500 | 25 | 524 | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 |
| train/ufo/actor_optimizer/parameter_delta_l2 | 500 | 25 | 524 | 0.458068043 | 0.364992797 | 0.345114123 | 0.320893466 | 0.460575253 | 0.34057267 | 0.37411622 | 0.354498505 |
| train/ufo/actor_reg_weight_applied | 500 | 25 | 524 | 10.47822 | 50 | 34.0475445 | 9.47556114 | 50 | 35.3104248 | 50 | 50 |
| train/ufo/actor_reg_weight_raw | 500 | 25 | 524 | 10.47822 | 62.3759689 | 35.2883451 | 9.47556114 | 62.3759689 | 35.3104248 | 58.8020756 | 58.7824234 |
| train/ufo/applied_action_abs_mean | 32 | 40 | 524 | 0.431002468 | 0.60150373 | 0.583760682 | 0.431002468 | 0.607109904 | 0.597507417 | 0.605964977 | 0.583760682 |
| train/ufo/applied_action_saturation_fraction | 32 | 40 | 524 | 0.0346697383 | 0.175678641 | 0.158156741 | 0.0346697383 | 0.18081975 | 0.171130188 | 0.180227438 | 0.158156741 |
| train/ufo/auxQ1 | 500 | 25 | 524 | -5.13383389 | -5.83521557 | -4.71850215 | -6.06134033 | -3.78522062 | -4.60916996 | -3.9589304 | -5.74480122 |
| train/ufo/aux_critic_loss | 500 | 25 | 524 | 2.31073451 | 1.49936187 | 1.44180179 | 0.81059128 | 18.9774971 | 1.22890848 | 2.20597517 | 1.53335856 |
| train/ufo/aux_critic_optimizer/grad_clip_limit | 500 | 25 | 524 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 |
| train/ufo/aux_critic_optimizer/grad_norm_after_clip_l2 | 500 | 25 | 524 | 4.99999952 | 4.99999952 | 4.7693855 | 1.15698874 | 5.00000048 | 4.99999952 | 5 | 4.50405338 |
| train/ufo/aux_critic_optimizer/grad_norm_l2 | 500 | 25 | 524 | 29.4599419 | 8.42577744 | 11.1394668 | 1.15698874 | 35.6908913 | 10.9402051 | 21.5765506 | 8.04651932 |
| train/ufo/aux_critic_optimizer/lr | 500 | 25 | 524 | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 |
| train/ufo/aux_critic_optimizer/parameter_delta_l2 | 500 | 25 | 524 | 0.38899374 | 0.409199357 | 0.429836665 | 0.218117192 | 0.804991186 | 0.408226386 | 0.669265985 | 0.574432345 |
| train/ufo/aux_rew/limits_dof_pos | 500 | 25 | 524 | 0.0303898342 | 0.228907898 | 0.16036611 | 0.0274718516 | 0.238596752 | 0.177990764 | 0.227162524 | 0.226448522 |
| train/ufo/aux_rew/limits_dof_vel | 500 | 25 | 524 | 0.00432933262 | 0.0131899351 | 0.0125475466 | 0.00333478115 | 0.0261435937 | 0.0127854468 | 0.015568871 | 0.0128405767 |
| train/ufo/aux_rew/limits_torque | 500 | 25 | 524 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| train/ufo/aux_rew/penalty_action_rate | 500 | 25 | 524 | 0.73818469 | 0.430169404 | 0.51832031 | 0.423463225 | 0.745791376 | 0.48785837 | 0.710398471 | 0.435192791 |
| train/ufo/aux_rew/penalty_ankle_roll | 500 | 25 | 524 | 0.0202875286 | 0.151064679 | 0.105255894 | 0.0198836327 | 0.153679416 | 0.114884138 | 0.146266741 | 0.14625457 |
| train/ufo/aux_rew/penalty_feet_ori | 500 | 25 | 524 | 0.102473676 | 0.192678899 | 0.158051597 | 0.102273531 | 0.199865296 | 0.160988167 | 0.186457028 | 0.183927766 |
| train/ufo/aux_rew/penalty_slippage | 500 | 25 | 524 | 0.445068717 | 0.399229139 | 0.424500608 | 0.356294364 | 0.525922894 | 0.417488828 | 0.485056034 | 0.38531381 |
| train/ufo/aux_rew/penalty_torques | 500 | 25 | 524 | 4467.66309 | 5040.17188 | 5380.4055 | 4438.40918 | 5887.4502 | 5402.21899 | 5701.42002 | 5146.19854 |
| train/ufo/aux_rew/penalty_undesired_contact | 500 | 25 | 524 | 0 | 0.0029296875 | 0.00352734375 | 0 | 0.00927734375 | 0.00341796875 | 0.00634765625 | 0.00462890625 |
| train/ufo/backward_optimizer/grad_clip_limit | 500 | 25 | 524 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 |
| train/ufo/backward_optimizer/grad_norm_after_clip_l2 | 500 | 25 | 524 | 5 | 5 | 5.00000001 | 4.99999905 | 5.00000095 | 5 | 5.00000048 | 5.00000001 |
| train/ufo/backward_optimizer/grad_norm_l2 | 500 | 25 | 524 | 299352.25 | 15618.3467 | 57862.0424 | 15399.4326 | 312821.594 | 34199.1836 | 178722.867 | 16366.976 |
| train/ufo/backward_optimizer/lr | 500 | 25 | 524 | 9.99999975e-06 | 9.99999975e-06 | 9.99999975e-06 | 9.99999975e-06 | 9.99999975e-06 | 9.99999975e-06 | 9.99999975e-06 | 9.99999975e-06 |
| train/ufo/backward_optimizer/parameter_delta_l2 | 500 | 25 | 524 | 0.00369011471 | 0.00342828012 | 0.00402176283 | 0.00342828012 | 0.00444344385 | 0.00402216217 | 0.00442695739 | 0.00351434847 |
| train/ufo/collected_transitions | 32 | 40 | 524 | 24576 | 24576 | 24576 | 24576 | 24576 | 24576 | 24576 | 24576 |
| train/ufo/contact_termination_count | 32 | 40 | 524 | 22 | 119 | 154.0625 | 22 | 216 | 160 | 196.3 | 154.0625 |
| train/ufo/critic_loss | 500 | 25 | 524 | 1.58768535 | 0.63069129 | 0.447601053 | 0.216887534 | 1.7113421 | 0.371941552 | 0.952698767 | 0.599943777 |
| train/ufo/critic_optimizer/grad_clip_limit | 500 | 25 | 524 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 |
| train/ufo/critic_optimizer/grad_norm_after_clip_l2 | 500 | 25 | 524 | 4.99999857 | 4.99999905 | 4.80068975 | 0.992902279 | 5.00000095 | 4.99999928 | 5 | 4.98566079 |
| train/ufo/critic_optimizer/grad_norm_l2 | 500 | 25 | 524 | 17.4156647 | 6.03476954 | 8.70755857 | 0.992902279 | 29.8782749 | 8.40651846 | 14.4694676 | 8.32643948 |
| train/ufo/critic_optimizer/lr | 500 | 25 | 524 | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 |
| train/ufo/critic_optimizer/parameter_delta_l2 | 500 | 25 | 524 | 0.417566866 | 0.475493163 | 0.345225458 | 0.206821099 | 0.628436327 | 0.330303952 | 0.498691583 | 0.443148508 |
| train/ufo/disc_expert_loss | 500 | 25 | 524 | 0.555425644 | 0.261298537 | 0.319395902 | 0.236666292 | 0.555425644 | 0.291685402 | 0.477930808 | 0.250458185 |
| train/ufo/disc_loss | 500 | 25 | 524 | 3.217767 | 0.826917887 | 1.37609139 | 0.786532462 | 3.21935892 | 1.08661777 | 2.77529219 | 0.825523822 |
| train/ufo/disc_reward_clip_fraction | 500 | 25 | 524 | 0 | 0.356933594 | 0.106612305 | 0 | 0.379394531 | 0.0478515625 | 0.32668457 | 0.327617188 |
| train/ufo/disc_train_loss | 500 | 25 | 524 | 0.682288885 | 0.328028351 | 0.511443003 | 0.306534141 | 0.684329271 | 0.49932985 | 0.677238208 | 0.331263859 |
| train/ufo/disc_wgan_gp_loss | 500 | 25 | 524 | 0.198005244 | 0.0237591043 | 0.0545252485 | 0.022540262 | 0.199566364 | 0.0296142027 | 0.164787194 | 0.024380177 |
| train/ufo/discriminator_optimizer/grad_clip_limit | 500 | 25 | 524 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 |
| train/ufo/discriminator_optimizer/grad_norm_after_clip_l2 | 500 | 25 | 524 | 4.99999952 | 0.948140979 | 2.67213865 | 0.927807927 | 5 | 1.82089782 | 4.99999952 | 1.03038428 |
| train/ufo/discriminator_optimizer/grad_norm_l2 | 500 | 25 | 524 | 8.44700336 | 0.948140979 | 3.22199462 | 0.927807927 | 8.4507122 | 1.82089782 | 8.04044824 | 1.03038428 |
| train/ufo/discriminator_optimizer/lr | 500 | 25 | 524 | 3.00000011e-06 | 3.00000011e-06 | 3.00000011e-06 | 3.00000011e-06 | 3.00000011e-06 | 3.00000011e-06 | 3.00000011e-06 | 3.00000011e-06 |
| train/ufo/discriminator_optimizer/parameter_delta_l2 | 500 | 25 | 524 | 0.00457609771 | 0.00240259408 | 0.00359714356 | 0.00239133136 | 0.00527207041 | 0.00307887443 | 0.00522169676 | 0.0024445606 |
| train/ufo/environment_control_steps | 32 | 40 | 524 | 16 | 140 | 78 | 16 | 140 | 78 | 133.8 | 78 |
| train/ufo/expert_draws/g1 | 32 | 40 | 524 | 41104 | 536088 | 294789.25 | 41104 | 536088 | 295080 | 523470.4 | 294789.25 |
| train/ufo/expert_draws/official_rpo | 32 | 40 | 524 | 40816 | 537064 | 294266.75 | 40816 | 537064 | 294744 | 523467.2 | 294266.75 |
| train/ufo/fb_diag | 500 | 25 | 524 | -10.4976788 | -212.759186 | -156.700383 | -229.367493 | -6.63200045 | -180.518707 | -32.4833063 | -215.357541 |
| train/ufo/fb_loss | 500 | 25 | 524 | 194250.172 | 16107.4033 | 47292.3489 | 15959.7598 | 196839.172 | 32718.9014 | 129429.263 | 17123.6941 |
| train/ufo/fb_offdiag | 500 | 25 | 524 | 182.113525 | 231.013794 | 164.730495 | 102.867043 | 251.631683 | 162.063972 | 226.372935 | 227.590944 |
| train/ufo/first_step_samples | 32 | 40 | 524 | 6680 | 1828 | 2287.5 | 1477 | 6680 | 2117.5 | 3064.6 | 2287.5 |
| train/ufo/first_step_terminated | 32 | 40 | 524 | 41 | 10 | 11.4375 | 6 | 41 | 10 | 17.35 | 11.4375 |
| train/ufo/foot_contact_peak_N | 32 | 40 | 524 | 9481.13281 | 13696.3135 | 16255.934 | 9481.13281 | 25652.8867 | 15268.3984 | 23832.2719 | 16255.934 |
| train/ufo/forward_optimizer/grad_clip_limit | 500 | 25 | 524 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 |
| train/ufo/forward_optimizer/grad_norm_after_clip_l2 | 500 | 25 | 524 | 5 | 5.00000048 | 5 | 4.99999905 | 5.00000095 | 5 | 5.00000048 | 5.0000001 |
| train/ufo/forward_optimizer/grad_norm_l2 | 500 | 25 | 524 | 1269.88562 | 119.5214 | 266.290823 | 98.8053589 | 1707.19446 | 145.918129 | 776.425412 | 111.391506 |
| train/ufo/forward_optimizer/lr | 500 | 25 | 524 | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 |
| train/ufo/forward_optimizer/parameter_delta_l2 | 500 | 25 | 524 | 0.338382781 | 0.795063496 | 0.672747055 | 0.249461576 | 0.864873588 | 0.766777307 | 0.841530976 | 0.8190622 |
| train/ufo/ground_foot_margin_min_m | 32 | 40 | 524 | 193.702866 | 193.591064 | 193.549715 | 193.260239 | 193.739929 | 193.562645 | 193.731097 | 193.549715 |
| train/ufo/ground_outside_count | 32 | 40 | 524 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| train/ufo/hard_position_count | 32 | 40 | 524 | 82 | 320 | 386.625 | 82 | 658 | 377 | 474.35 | 386.625 |
| train/ufo/hard_torque_count | 32 | 40 | 524 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| train/ufo/hard_velocity_count | 32 | 40 | 524 | 763 | 669 | 848.3125 | 663 | 1537 | 807.5 | 1193.65 | 848.3125 |
| train/ufo/height_termination_count | 32 | 40 | 524 | 0 | 573 | 465.0625 | 0 | 797 | 487 | 571.9 | 465.0625 |
| train/ufo/learner_joint_updates | 500 | 25 | 524 | 25 | 524 | 274.5 | 25 | 524 | 274.5 | 499.05 | 499.5 |
| train/ufo/mean_aux_reward | 500 | 25 | 524 | -0.788443565 | -1.15446067 | -1.06064701 | -1.19532216 | -0.77312851 | -1.07410103 | -0.91019983 | -1.1282893 |
| train/ufo/mean_disc_reward | 500 | 25 | 524 | 0.104450002 | -1.16245139 | -0.516213208 | -1.19742441 | 0.123250082 | -0.567708164 | 0.102581317 | -1.12014274 |
| train/ufo/mean_disc_reward_raw | 500 | 25 | 524 | 0.104450002 | -1.45137262 | -0.577492059 | -1.48532331 | 0.123250082 | -0.580277473 | 0.102581317 | -1.35955494 |
| train/ufo/mean_next_Q | 500 | 25 | 524 | -2.16333294 | -3.15692663 | -1.86978079 | -3.15692663 | -1.22798538 | -1.6955902 | -1.2705382 | -2.94438689 |
| train/ufo/mean_next_auxQ | 500 | 25 | 524 | -3.53529024 | -5.16153097 | -3.81122877 | -5.16153097 | -2.80493951 | -3.68291187 | -2.9101781 | -4.97862644 |
| train/ufo/online_transitions | 500 | 25 | 524 | 98304 | 860160 | 470089.728 | 98304 | 860160 | 466944 | 811008 | 815923.2 |
| train/ufo/orientation_termination_count | 32 | 40 | 524 | 0 | 295 | 444.46875 | 0 | 1048 | 358 | 853.1 | 444.46875 |
| train/ufo/orth_loss | 500 | 25 | 524 | 1940.78564 | 160.891479 | 472.843187 | 159.367126 | 1966.68433 | 327.384949 | 1293.52283 | 171.114607 |
| train/ufo/orth_loss_diag | 500 | 25 | 524 | -256 | -256 | -256 | -256 | -256 | -256 | -256 | -256 |
| train/ufo/orth_loss_offdiag | 500 | 25 | 524 | 2196.78564 | 416.891479 | 728.843187 | 415.367126 | 2222.68433 | 583.384949 | 1549.52283 | 427.114607 |
| train/ufo/physics_peak_joint_velocity | 32 | 40 | 524 | 31.1331291 | 35.6554413 | 40.530143 | 28.5076294 | 67.5246201 | 39.084589 | 58.0591673 | 40.530143 |
| train/ufo/q_loss | 500 | 25 | 524 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| train/ufo/replay_draw_new_ratio | 32 | 40 | 524 | 0.833333313 | 1.24761903 | 1.18784997 | 0.833333313 | 1.27450979 | 1.23070174 | 1.27168562 | 1.18784997 |
| train/ufo/replay_draws | 32 | 40 | 524 | 81920 | 1073152 | 589056 | 81920 | 1073152 | 589824 | 1046937.6 | 589056 |
| train/ufo/replay_policy_age_max_updates | 32 | 40 | 524 | 40 | 524 | 287.625 | 40 | 524 | 288 | 511.2 | 287.625 |
| train/ufo/replay_policy_age_min_updates | 32 | 40 | 524 | 16 | 4 | 15.625 | 4 | 16 | 16 | 16 | 15.625 |
| train/ufo/replay_retained_fraction | 32 | 40 | 524 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| train/ufo/replay_size | 32 | 40 | 524 | 98304 | 860160 | 479232 | 98304 | 860160 | 479232 | 822067.2 | 479232 |
| train/ufo/requested_action_abs_mean | 32 | 40 | 524 | 0.885911345 | 0.897840202 | 0.902983485 | 0.885911345 | 0.911751568 | 0.902051717 | 0.910915214 | 0.902983485 |
| train/ufo/requested_action_saturation_fraction | 32 | 40 | 524 | 0.405425578 | 0.433875054 | 0.44006287 | 0.405425578 | 0.452456266 | 0.439814031 | 0.451149757 | 0.44006287 |
| train/ufo/requested_applied_max_diff | 32 | 40 | 524 | 1.95280135 | 1.95280027 | 1.95609733 | 1.94904256 | 1.96602774 | 1.95280135 | 1.96602774 | 1.95609733 |
| train/ufo/reset_control_samples | 32 | 40 | 524 | 6680 | 1828 | 2287.5 | 1477 | 6680 | 2117.5 | 3064.6 | 2287.5 |
| train/ufo/reset_projection_outside_fraction | 32 | 40 | 524 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| train/ufo/reset_projection_residual_max_rad | 32 | 40 | 524 | 1.78813934e-07 | 1.78813934e-07 | 1.78813934e-07 | 1.78813934e-07 | 1.78813934e-07 | 1.78813934e-07 | 1.78813934e-07 | 1.78813934e-07 |
| train/ufo/reset_projection_residual_mean_rad | 32 | 40 | 524 | 6.42993712e-08 | 6.49344329e-08 | 6.46944405e-08 | 6.36020516e-08 | 6.649433e-08 | 6.46557261e-08 | 6.59231016e-08 | 6.46944405e-08 |
| train/ufo/reset_projection_residual_p95_rad | 32 | 40 | 524 | 1.1920929e-07 | 1.1920929e-07 | 1.1920929e-07 | 1.1920929e-07 | 1.1920929e-07 | 1.1920929e-07 | 1.1920929e-07 | 1.1920929e-07 |
| train/ufo/reset_root_lift_max_m | 32 | 40 | 524 | 0.0219920222 | 0.0219513215 | 0.0219907503 | 0.0219513215 | 0.0219920222 | 0.0219920222 | 0.0219920222 | 0.0219907503 |
| train/ufo/sampling_rounds | 500 | 25 | 524 | 4 | 35 | 19.128 | 4 | 35 | 19 | 33 | 33.2 |
| train/ufo/target_M | 500 | 25 | 524 | 3.24742985 | 0.594288647 | 0.747654719 | 0.264537334 | 3.26152134 | 0.592351526 | 1.83277771 | 0.596193498 |
| train/ufo/target_Q | 500 | 25 | 524 | -3.3590951 | -4.10789394 | -2.54934724 | -4.10789394 | -1.54286528 | -2.43056881 | -1.60777212 | -3.86933418 |
| train/ufo/target_auxQ | 500 | 25 | 524 | -4.61953926 | -5.95526505 | -4.72088269 | -5.95526505 | -3.92981672 | -4.60994339 | -4.03460913 | -5.75268966 |
| train/ufo/terminated_transitions | 32 | 40 | 524 | 847 | 1822 | 2109.6875 | 847 | 3450 | 2058 | 2787.5 | 2109.6875 |
| train/ufo/truncated_transitions | 32 | 40 | 524 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| train/ufo/unc_Q | 500 | 25 | 524 | 2.9987812 | 0.144408062 | 0.669604417 | 0.138938636 | 3.16187 | 0.335171044 | 2.53296152 | 0.147122941 |
| train/ufo/unc_auxQ | 500 | 25 | 524 | 1.06036007 | 0.154729396 | 0.368491263 | 0.147684306 | 1.09377933 | 0.267994881 | 0.922045714 | 0.156281464 |
| train/ufo/wall_time_s | 500 | 25 | 524 | 114.348434 | 1077.88269 | 589.524217 | 114.348434 | 1077.88269 | 587.527527 | 1020.97386 | 1025.39401 |
| train/ufo/z_norm | 500 | 25 | 524 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 |

## 全部指标五个100更新窗口

| tag | 25–124 | 125–224 | 225–324 | 325–424 | 425–524 |
|---|---|---|---|---|---|
| env/episode_len | 7.13045224 | 10.0871313 | 11.0344256 | 12.0260083 | 13.2711236 |
| env/num_trajectories | 2395.33333 | 2285.5 | 2153 | 1953.57143 | 1833.14286 |
| env/return | 0 | 0 | 0 | 0 | 0 |
| env/reward | 0 | 0 | 0 | 0 | 0 |
| env/success_once | 0 | 0 | 0 | 0 | 0 |
| rollout/ufo/received_transitions | 24576 | 24576 | 24576 | 24576 | 24576 |
| time/actor_training | 12.0117213 | 11.6191169 | 11.5887173 | 11.5814423 | 10.4457447 |
| time/cal_adv_and_returns | 0.00641234716 | 0.00521965822 | 0.00576035182 | 0.00495576859 | 0.0051948343 |
| time/env/compute_bootstrap_rewards | 0.000254945441 | 0.000247445113 | 0.000250369582 | 0.000247229423 | 0.000249224715 |
| time/env/env/bootstrap_step | 0.00285956093 | 0.00025183983 | 0.000277099044 | 0.000291616323 | 0.00027630987 |
| time/env/env/send_rollout_trajectories | 0.241697108 | 0.221479374 | 0.227932125 | 0.218232051 | 0.224896095 |
| time/env/env_interact_step | 1.53127724 | 1.14887208 | 1.30462706 | 1.25864407 | 1.16900487 |
| time/env/interact | 2.53193287 | 2.1660676 | 2.12450969 | 2.1735659 | 2.12717806 |
| time/env/run_interact_once | 2.53189909 | 2.16604408 | 2.1244845 | 2.17354144 | 2.12715173 |
| time/generate_rollouts | 2.74054313 | 2.35853096 | 2.2992208 | 2.365712 | 2.34532448 |
| time/rollout/generate_one_epoch | 2.29337955 | 1.93721235 | 1.88174166 | 1.94910479 | 1.88959047 |
| time/rollout/predict | 0.090383842 | 0.0501897155 | 0.0439503218 | 0.0438844838 | 0.0459607132 |
| time/rollout/rollout/generate | 2.29435762 | 1.93744326 | 1.88228426 | 1.94960489 | 1.88990065 |
| time/rollout/sync_model_from_actor | 13.6651831 | 13.4600035 | 13.1912758 | 13.2070911 | 13.3953465 |
| time/step | 32.1509533 | 31.1679347 | 30.8368464 | 30.8422841 | 29.7718915 |
| time/sync_weights | 17.3917405 | 17.1845064 | 16.9426362 | 16.8896465 | 16.9750928 |
| train/ufo/B | -0.0104975824 | -0.00647297613 | -0.00527650274 | -0.00366203578 | -0.0023453208 |
| train/ufo/B_norm | 16 | 16 | 16 | 16 | 16 |
| train/ufo/F1 | 0.0236470816 | 0.0159528507 | 0.0056411546 | 0.00674458881 | 0.0115605159 |
| train/ufo/M1 | 2.37014846 | 1.05563955 | 0.704682938 | 0.799630432 | 0.799288079 |
| train/ufo/Q1 | -2.37756138 | -1.64799669 | -2.1492275 | -2.89083704 | -3.67440313 |
| train/ufo/Q_aux | -4.28680092 | -4.18136008 | -4.54251724 | -5.18078485 | -5.73424194 |
| train/ufo/Q_discriminator | -2.48364323 | -1.67117906 | -2.18772142 | -2.94799273 | -3.73066962 |
| train/ufo/Q_fb | 6.46434815 | 18.1351196 | 28.2846019 | 38.5106324 | 47.9390214 |
| train/ufo/action_abs_mean | 0.904442132 | 0.908720603 | 0.904410819 | 0.900247872 | 0.89832787 |
| train/ufo/action_saturation_fraction | 0.438618094 | 0.448023111 | 0.442551474 | 0.437744141 | 0.434663828 |
| train/ufo/actor_loss | -3.59252697 | -13.9718997 | -21.1724088 | -26.9470306 | -32.8781053 |
| train/ufo/actor_optimizer/grad_clip_limit | 1 | 1 | 1 | 1 | 1 |
| train/ufo/actor_optimizer/grad_norm_after_clip_l2 | 0.999999543 | 0.999999718 | 0.999999765 | 0.999999791 | 0.999999814 |
| train/ufo/actor_optimizer/grad_norm_l2 | 2.3149739 | 3.62763429 | 4.26645286 | 4.98240594 | 5.70556447 |
| train/ufo/actor_optimizer/lr | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 |
| train/ufo/actor_optimizer/parameter_delta_l2 | 0.36971997 | 0.334608838 | 0.333748963 | 0.336896214 | 0.35059663 |
| train/ufo/actor_reg_weight_applied | 14.1577687 | 24.8694526 | 35.3096964 | 45.9008047 | 50 |
| train/ufo/actor_reg_weight_raw | 14.1577687 | 24.8694526 | 35.3096964 | 46.0263923 | 56.0784154 |
| train/ufo/applied_action_abs_mean | 0.51968883 | 0.589509676 | 0.59637624 | 0.60329846 | 0.603400588 |
| train/ufo/applied_action_saturation_fraction | 0.0999690983 | 0.160489144 | 0.170032149 | 0.177375493 | 0.176634986 |
| train/ufo/auxQ1 | -4.27322369 | -4.12157442 | -4.50348538 | -5.08268529 | -5.61154196 |
| train/ufo/aux_critic_loss | 1.50856211 | 1.6433428 | 1.10943329 | 1.42808886 | 1.5195819 |
| train/ufo/aux_critic_optimizer/grad_clip_limit | 5 | 5 | 5 | 5 | 5 |
| train/ufo/aux_critic_optimizer/grad_norm_after_clip_l2 | 4.92206907 | 4.92772878 | 4.59041413 | 4.86828212 | 4.53843341 |
| train/ufo/aux_critic_optimizer/grad_norm_l2 | 12.3030158 | 12.8624762 | 9.55939639 | 12.5633174 | 8.40912834 |
| train/ufo/aux_critic_optimizer/lr | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 |
| train/ufo/aux_critic_optimizer/parameter_delta_l2 | 0.325061045 | 0.360046189 | 0.460177847 | 0.429134774 | 0.574763471 |
| train/ufo/aux_rew/limits_dof_pos | 0.0636832653 | 0.136295596 | 0.178180171 | 0.201475098 | 0.222196417 |
| train/ufo/aux_rew/limits_dof_vel | 0.00920373968 | 0.013396234 | 0.0135648579 | 0.0134080822 | 0.0131648192 |
| train/ufo/aux_rew/limits_torque | 0 | 0 | 0 | 0 | 0 |
| train/ufo/aux_rew/penalty_action_rate | 0.664956127 | 0.545866163 | 0.487280478 | 0.455395942 | 0.43810284 |
| train/ufo/aux_rew/penalty_ankle_roll | 0.047912056 | 0.0896038148 | 0.114333708 | 0.130745321 | 0.143684572 |
| train/ufo/aux_rew/penalty_feet_ori | 0.12553646 | 0.147153604 | 0.162481064 | 0.172848186 | 0.18223867 |
| train/ufo/aux_rew/penalty_slippage | 0.475799603 | 0.443502072 | 0.417166795 | 0.398149225 | 0.387885344 |
| train/ufo/aux_rew/penalty_torques | 5368.86111 | 5584.59881 | 5486.29068 | 5293.41635 | 5168.86053 |
| train/ufo/aux_rew/penalty_undesired_contact | 0.00186523437 | 0.00315429687 | 0.00391601562 | 0.00427246094 | 0.00442871094 |
| train/ufo/backward_optimizer/grad_clip_limit | 5 | 5 | 5 | 5 | 5 |
| train/ufo/backward_optimizer/grad_norm_after_clip_l2 | 5.00000003 | 4.99999999 | 4.99999996 | 5.00000004 | 5.00000002 |
| train/ufo/backward_optimizer/grad_norm_l2 | 151597.801 | 62243.5212 | 34878.8066 | 23166.3484 | 17423.7352 |
| train/ufo/backward_optimizer/lr | 9.99999975e-06 | 9.99999975e-06 | 9.99999975e-06 | 9.99999975e-06 | 9.99999975e-06 |
| train/ufo/backward_optimizer/parameter_delta_l2 | 0.0040511773 | 0.00436912509 | 0.00415968501 | 0.0039248493 | 0.00360397743 |
| train/ufo/collected_transitions | 24576 | 24576 | 24576 | 24576 | 24576 |
| train/ufo/contact_termination_count | 136.166667 | 132 | 184.833333 | 160 | 156 |
| train/ufo/critic_loss | 0.741080195 | 0.292190337 | 0.272795078 | 0.380687837 | 0.55125182 |
| train/ufo/critic_optimizer/grad_clip_limit | 5 | 5 | 5 | 5 | 5 |
| train/ufo/critic_optimizer/grad_norm_after_clip_l2 | 4.95328009 | 4.77837068 | 4.8328232 | 4.81058082 | 4.62839397 |
| train/ufo/critic_optimizer/grad_norm_l2 | 10.9167166 | 7.87616403 | 8.15101419 | 8.7190265 | 7.87487158 |
| train/ufo/critic_optimizer/lr | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 |
| train/ufo/critic_optimizer/parameter_delta_l2 | 0.2951346 | 0.2915511 | 0.298967286 | 0.374900782 | 0.465573521 |
| train/ufo/disc_expert_loss | 0.44456233 | 0.330385098 | 0.289738197 | 0.277320894 | 0.25497299 |
| train/ufo/disc_loss | 2.50309566 | 1.48834513 | 1.08966149 | 0.951634504 | 0.847720172 |
| train/ufo/disc_reward_clip_fraction | 0 | 0.00218261719 | 0.0510742187 | 0.179350586 | 0.300454102 |
| train/ufo/disc_train_loss | 0.668255837 | 0.634834514 | 0.502746147 | 0.40741717 | 0.343961345 |
| train/ufo/disc_wgan_gp_loss | 0.139027751 | 0.0523125514 | 0.0297177135 | 0.0266896434 | 0.0248785832 |
| train/ufo/discriminator_optimizer/grad_clip_limit | 5 | 5 | 5 | 5 | 5 |
| train/ufo/discriminator_optimizer/grad_norm_after_clip_l2 | 4.99999934 | 4.08992192 | 1.86036642 | 1.32718354 | 1.08322202 |
| train/ufo/discriminator_optimizer/grad_norm_l2 | 7.50794477 | 4.33125633 | 1.86036642 | 1.32718354 | 1.08322202 |
| train/ufo/discriminator_optimizer/lr | 3.00000011e-06 | 3.00000011e-06 | 3.00000011e-06 | 3.00000011e-06 | 3.00000011e-06 |
| train/ufo/discriminator_optimizer/parameter_delta_l2 | 0.00491402912 | 0.00472758791 | 0.0031402528 | 0.00271477255 | 0.00248907543 |
| train/ufo/environment_control_steps | 26 | 50 | 74 | 100 | 128 |
| train/ufo/expert_draws/g1 | 82226.6667 | 181020 | 278861.333 | 385026.286 | 497917.714 |
| train/ufo/expert_draws/official_rpo | 81613.3333 | 179428 | 278194.667 | 385021.714 | 497995.429 |
| train/ufo/fb_diag | -56.7866171 | -134.012008 | -178.866692 | -200.787124 | -213.049474 |
| train/ufo/fb_loss | 109202.504 | 51862.5057 | 33161.2178 | 23953.9878 | 18281.529 |
| train/ufo/fb_offdiag | 119.440868 | 132.551392 | 161.865027 | 190.112395 | 219.682794 |
| train/ufo/first_step_samples | 3309 | 2304.83333 | 2159.5 | 1953.57143 | 1840.71429 |
| train/ufo/first_step_terminated | 18.5 | 10.3333333 | 10.3333333 | 9.85714286 | 8.85714286 |
| train/ufo/foot_contact_peak_N | 15433.8179 | 17078.807 | 16401.7106 | 15658.8246 | 16727.4434 |
| train/ufo/forward_optimizer/grad_clip_limit | 5 | 5 | 5 | 5 | 5 |
| train/ufo/forward_optimizer/grad_norm_after_clip_l2 | 5.00000001 | 5.00000002 | 5 | 4.99999995 | 5 |
| train/ufo/forward_optimizer/grad_norm_l2 | 682.583389 | 268.151231 | 146.603958 | 119.452557 | 114.662981 |
| train/ufo/forward_optimizer/lr | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 |
| train/ufo/forward_optimizer/parameter_delta_l2 | 0.37269054 | 0.613104676 | 0.770736679 | 0.803788345 | 0.803415036 |
| train/ufo/ground_foot_margin_min_m | 193.593801 | 193.45739 | 193.632299 | 193.559668 | 193.510324 |
| train/ufo/ground_outside_count | 0 | 0 | 0 | 0 | 0 |
| train/ufo/hard_position_count | 419.5 | 445.166667 | 387 | 355.428571 | 339.142857 |
| train/ufo/hard_torque_count | 0 | 0 | 0 | 0 | 0 |
| train/ufo/hard_velocity_count | 1131.5 | 846.666667 | 809.666667 | 798.428571 | 690 |
| train/ufo/height_termination_count | 325 | 478.833333 | 481.5 | 499.428571 | 524.857143 |
| train/ufo/learner_joint_updates | 74.5 | 174.5 | 274.5 | 374.5 | 474.5 |
| train/ufo/mean_aux_reward | -1.00674761 | -1.03885947 | -1.03832924 | -1.09756967 | -1.12172908 |
| train/ufo/mean_disc_reward | 0.0819809661 | -0.130324462 | -0.561568714 | -0.888101057 | -1.08305278 |
| train/ufo/mean_disc_reward_raw | 0.0819809661 | -0.130531901 | -0.576594723 | -0.973993162 | -1.28832148 |
| train/ufo/mean_next_Q | -1.64426479 | -1.29444918 | -1.55481418 | -2.09039158 | -2.76498423 |
| train/ufo/mean_next_auxQ | -3.11308221 | -3.14686943 | -3.67773624 | -4.27749788 | -4.84095809 |
| train/ufo/online_transitions | 163184.64 | 316538.88 | 469893.12 | 623247.36 | 777584.64 |
| train/ufo/orientation_termination_count | 584.833333 | 589.5 | 487 | 320.857143 | 287 |
| train/ufo/orth_loss | 1091.3985 | 518.63966 | 331.782195 | 239.646626 | 182.748957 |
| train/ufo/orth_loss_diag | -256 | -256 | -256 | -256 | -256 |
| train/ufo/orth_loss_offdiag | 1347.3985 | 774.63966 | 587.782195 | 495.646626 | 438.748957 |
| train/ufo/physics_peak_joint_velocity | 39.7169243 | 41.4031194 | 46.69089 | 38.2567705 | 37.4716543 |
| train/ufo/q_loss | 0 | 0 | 0 | 0 | 0 |
| train/ufo/replay_draw_new_ratio | 1.00145505 | 1.17024642 | 1.22429011 | 1.2528155 | 1.26650587 |
| train/ufo/replay_draws | 163840 | 360448 | 557056 | 770048 | 995913.143 |
| train/ufo/replay_policy_age_max_updates | 80 | 176 | 272 | 376 | 486.285714 |
| train/ufo/replay_policy_age_min_updates | 16 | 16 | 16 | 16 | 14.2857143 |
| train/ufo/replay_retained_fraction | 1 | 1 | 1 | 1 | 1 |
| train/ufo/replay_size | 159744 | 307200 | 454656 | 614400 | 786432 |
| train/ufo/requested_action_abs_mean | 0.904442181 | 0.908720583 | 0.90441077 | 0.900247846 | 0.898327913 |
| train/ufo/requested_action_saturation_fraction | 0.438618094 | 0.448023111 | 0.442551474 | 0.437744141 | 0.434663828 |
| train/ufo/requested_applied_max_diff | 1.95290029 | 1.95280063 | 1.95720979 | 1.95793252 | 1.95887469 |
| train/ufo/reset_control_samples | 3309 | 2304.83333 | 2159.5 | 1953.57143 | 1840.71429 |
| train/ufo/reset_projection_outside_fraction | 0 | 0 | 0 | 0 | 0 |
| train/ufo/reset_projection_residual_max_rad | 1.78813934e-07 | 1.78813934e-07 | 1.78813934e-07 | 1.78813934e-07 | 1.78813934e-07 |
| train/ufo/reset_projection_residual_mean_rad | 6.46501105e-08 | 6.46323504e-08 | 6.44890186e-08 | 6.45356053e-08 | 6.51205687e-08 |
| train/ufo/reset_projection_residual_p95_rad | 1.1920929e-07 | 1.1920929e-07 | 1.1920929e-07 | 1.1920929e-07 | 1.1920929e-07 |
| train/ufo/reset_root_lift_max_m | 0.0219920222 | 0.0219920222 | 0.0219920222 | 0.0219920222 | 0.0219862078 |
| train/ufo/sampling_rounds | 6.64 | 12.88 | 19.12 | 25.36 | 31.64 |
| train/ufo/target_M | 1.53771833 | 0.5119877 | 0.454532325 | 0.620966448 | 0.613068793 |
| train/ufo/target_Q | -2.38193936 | -1.65128612 | -2.14991814 | -2.89117646 | -3.67241611 |
| train/ufo/target_auxQ | -4.26882001 | -4.11838051 | -4.51431157 | -5.08375454 | -5.61914682 |
| train/ufo/terminated_transitions | 2395.33333 | 2285.5 | 2153 | 1953.57143 | 1833.14286 |
| train/ufo/truncated_transitions | 0 | 0 | 0 | 0 | 0 |
| train/ufo/unc_Q | 1.97023565 | 0.677158443 | 0.340462816 | 0.206283675 | 0.153881506 |
| train/ufo/unc_auxQ | 0.782304484 | 0.433837434 | 0.272242081 | 0.192860988 | 0.161211326 |
| train/ufo/wall_time_s | 201.761268 | 396.542606 | 590.006846 | 782.396653 | 976.913712 |
| train/ufo/z_norm | 16 | 16 | 16 | 16 | 16 |

直接读取无数量上限的原始TensorBoard事件。统计取>24段，仅配置初始点的tag采用全点；n及first/last step明确。各日志点等权，无插值；last50_mean是最后50记录点而非50更新，episode_len为已结束episode日志聚合而非独立case存活统计。真实采样/learner更新频率不同。

## 三轮复核和检查

第一轮核对本体单位/关节语义、train-only数据与足碰撞网格；第二轮核对真实2048网格地面覆盖、4GPU租约、16→24恢复、500联合更新和预算；第三轮回读全部指标、131列trace/实际replay、六Adam/哈希、RSI几何与GPU释放。检查只读对照曾明确发现地面遗漏，已派生版本修正，未隐藏原H失败条件。

容器检查结果：
```text
...................................................                      [100%]
=============================== warnings summary ===============================
../../app/backend/.venv/lib/python3.11/site-packages/passlib/utils/__init__.py:854
  /app/backend/.venv/lib/python3.11/site-packages/passlib/utils/__init__.py:854: DeprecationWarning: 'crypt' is deprecated and slated for removal in Python 3.13
    from crypt import crypt as _crypt

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
51 passed, 1 skipped, 1 warning in 1.65s

```

容器检查结果：
```text
......................................                                   [100%]
=============================== warnings summary ===============================
../isaaclab_extension_template/.venv/lib/python3.11/site-packages/_pytest/config/__init__.py:1428
  /workspace/isaaclab_extension_template/.venv/lib/python3.11/site-packages/_pytest/config/__init__.py:1428: PytestConfigWarning: Unknown config option: asyncio_mode
  
    self._warn_or_fail_if_strict(f"Unknown config option: {key}\n")

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
38 passed, 1 warning in 13.03s

```

容器检查结果：
```text
...                                                                      [100%]
=============================== warnings summary ===============================
../isaaclab_extension_template/.venv/lib/python3.11/site-packages/_pytest/config/__init__.py:1428
  /workspace/isaaclab_extension_template/.venv/lib/python3.11/site-packages/_pytest/config/__init__.py:1428: PytestConfigWarning: Unknown config option: asyncio_mode
  
    self._warn_or_fail_if_strict(f"Unknown config option: {key}\n")

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
3 passed, 1 warning in 4.02s

```

容器检查结果：
```text
...s.ssss.s.s........................................................... [ 84%]
.............                                                            [100%]
=============================== warnings summary ===============================
../../app/backend/.venv/lib/python3.11/site-packages/passlib/utils/__init__.py:854
  /app/backend/.venv/lib/python3.11/site-packages/passlib/utils/__init__.py:854: DeprecationWarning: 'crypt' is deprecated and slated for removal in Python 3.13
    from crypt import crypt as _crypt

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
78 passed, 9 skipped, 1 warning in 1.99s

```

容器检查结果：
```text
..s.s.s..ssss.                                                           [100%]
=============================== warnings summary ===============================
tests/unit/test_bfm_ufo_scaling.py::test_scale_resume_moves_from_qualification_to_finite_training_budget
  /app/backend/.venv/lib/python3.11/site-packages/passlib/utils/__init__.py:854: DeprecationWarning: 'crypt' is deprecated and slated for removal in Python 3.13
    from crypt import crypt as _crypt

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
7 passed, 7 skipped, 1 warning in 1.31s

```

结束本轮500更新观测，不自动延长。正常行走未验收，用户指定训练不足不独立评估。剩余问题由首步原因、子步速度、PD target/torque和足部力数据确定；速度软罚仍只覆盖控制边界，接触冲量积分未采集，完整GPU峰值/利用率时序未采集。后续优先对齐子步软制动信号、RSI足部初速度与PD/接触实际瞬态，再讨论更长采样，避免仅增加learner更新。

诊断粒度限制：131列trace保留23个关节的控制边界dq，但物理子步峰只存全关节标量，未保存极值对应的关节/子步序号或逐关节峰值。能确认瞬时越限及完整锁存终止，不能将全局子步峰指定为右踝。后续应补齐逐关节瞬态与接触冲量，才能进一步区分PD与约束求解。

下一步推荐但本轮未实施：把RSI足网格表面的法向初速度纳入候选审计，比较保留低冲击起点的覆盖度，不直接抹掉原始q/dq或假定foot link速度等于网格接触点速度；记录逐关节子步峰/目标/实测PD输入与接触冲量，并将连续软制动罚对齐真实子步超速；在新profile中对照更保守的target slew与显式PD/接触参数。当前目标允许腿关节每20ms变化0.25rad、足/臂0.08rad，保证只是目标增量，不能保证真实速度。固定当前500结果后停止，不把新假设混进本段。

保存开销源码复核：860160行replay约6.06GB，state_dict先复制有效行，tensorize又复制一次再转换；连同原容量约7.04GB，仅replay可同时占约19.2GB。之后还有fsync、多轮完整SHA256验证与发布校验。实际保存最终成功，但没有阶段计时，不能断言全部延迟由内存或磁盘造成。

## 历史对照附录：H接触修复500段的完整独立内容

# UFO RSI接触/PD v8：6144环境、500新增联合更新总结


本报告直接给出任务描述、完整配置、全部标量统计与训练/物理证据，不需要访问其他文档。日期采用Asia/Shanghai；生成于2026-10-12T08:21:22.407135+08:00。本次未提交或暂存代码。

H新初始化完成16更新资格，完整恢复至24，再追加500次原生UFO agent联合更新至524。GPU0为单learner，GPU1–3各2048环境，共6144，batch2048；这是多卡采样，非DDP。物理配置改变后派生新recipe，未把旧G权重或replay伪装成跨recipe完整恢复。旧G模型及证据保留。

**第三轮复核确认地面覆盖缺陷，因此本段保留为受影响对照。** 当前100m×100m有限地面范围±50m，不克隆也非无限plane；实际2048网格中心范围±56.25/±55m，408/2048（19.921875%）中心在外，参考root XY还可偏移约7m。按中心代理分组的786432条trace：内部足接触率60.52%、高度终止率2.21%；外部足接触率6.58%、高度终止率7.59%。该分组未加入逐transition参考偏移，不能将右踝极值直接归因于地面边缘。即使优化器、恢复、曲线有效，全部环境有地面支撑的实验条件不成立；不据此决定扩大训练。

### 有限地面、真实2048网格与trace分组审计

```json
{
  "method": "frozen TerrainImporter grid formula and MeshPlaneTerrain extent; world origin group proxy, raw RSI xy offsets not applied per transition",
  "num_envs_per_sampler": 2048,
  "env_spacing_m": 2.5,
  "grid_rows": 46,
  "grid_cols": 45,
  "origin_xy_min_m": [
    -56.25,
    -55.0
  ],
  "origin_xy_max_m": [
    56.25,
    55.0
  ],
  "plane_size_m": [
    100,
    100
  ],
  "border_width_m": 0,
  "origins_outside_plane": 408,
  "outside_origin_fraction": 0.19921875,
  "train_valid_root_xy_min_m": [
    -6.971680164337158,
    -5.509981155395508
  ],
  "train_valid_root_xy_max_m": [
    7.098916053771973,
    7.317755699157715
  ],
  "formal_trace_group_statistics": {
    "origin_inside_100m": {
      "samples": 629760,
      "nonzero_foot_contact": 381111,
      "height_flags": 13930,
      "velocity_flags": 20798
    },
    "origin_outside_100m": {
      "samples": 156672,
      "nonzero_foot_contact": 10311,
      "height_flags": 11898,
      "velocity_flags": 511
    }
  },
  "limitation": "grouping uses grid origins; RSI xy offsets and foot placement must be included for exact per-case coverage"
}
```

派生修正I ground-v9：512m×512m离线平面，启动验证实际grid+所有候选RSI足碰撞网格XY包络+20m余量，逐step记录world root XY与足部保守覆盖余量，越界停止；原H镜像/profile/模型保持。新初始化16→24完整恢复后再做500更新，不能把跨地面recipe新初始化描述为H完整续训。额外保全总预算改为96GiB，原H自身64GiB账目仍单列。32环境诊断无法暴露2048边缘覆盖问题，这是本轮此前检查的遗漏。

五个100更新窗口episode_len日志均值：7.47538356 → 9.45686754 → 10.4291836 → 11.182853 → 11.8562354控制步；末窗口对应0.237124707秒，末记录点12.2290707步。正常行走训练触发门槛未满足，本次未启动独立validation、held-out、验收或评估录像；无成功率可报告。

本轮训练观测：六优化器裁剪后范数合格=True；末窗episode高于首窗且速度flags下降=True；至少末两窗episode≥100=False。500更新的趋势不能替代正常行走验收。

## 问题、证据与采用修复


旧G段最大dq=68.680458rad/s，关节right_ankle_roll_joint，round7、episode第13控制步；requested/applied约−0.19165、PD目标约−0.09583rad，同时触发硬位置/速度终止。该极值既不是饱和动作也不是reset首步，不能以单一动作饱和解释。旧G正式段963/66449次reset首步终止（1.44923%），其中硬限位62次，另外901次此前无细分原因。

进一步回读旧G该条transition：右踝q由−0.10503到−0.03397rad，dq由+0.89602骤变为−68.68046rad/s；起步目标误差仅约0.00920rad，末物理子步输入力矩仅+0.25154Nm，与末速度方向相反。q仍在踝roll物理范围内，global位置flag必须核对其他关节；末力矩×末速度不是同一时刻的积分机械功。证据不支持“右踝饱和PD目标直接打满”这一简单解释，更需关注接触/耦合/求解过程中的速度突变；旧记录无接触力和子步冲量，无法唯一归因。

### 68.68rad/s前后完整关节/根部/足部状态与PD力矩

```json
{
  "scope": "original G 68.68rad/s transition plus same-episode preceding retained states, CPU only",
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
  "peak_joint": "right_ankle_roll_joint",
  "requested_action": -0.19165153801441193,
  "applied_action": -0.19165153801441193,
  "target_rad": -0.09582576900720596,
  "q_before_rad": -0.10502716153860092,
  "q_after_rad": -0.033971063792705536,
  "dq_before_radps": 0.8960220813751221,
  "dq_after_radps": -68.68045806884766,
  "applied_last_substep_torque_Nm": 0.2515355348587036,
  "last_torque_times_boundary_dq_W": -17.275575637817383,
  "torque_velocity_sampling_limitation": "applied torque from final integration input; dq measured after step, product not integrated mechanical work",
  "pd_target_all_rad": [
    -0.8244942426681519,
    -0.03552946448326111,
    2.4293432235717773,
    0.5747939348220825,
    -0.49999940395355225,
    -0.8600509762763977,
    0.906171977519989,
    -0.2677854597568512,
    0.7038884162902832,
    -0.2122652530670166,
    0.1834319829940796,
    -0.19999873638153076,
    -0.19999873638153076,
    0.7765548825263977,
    -1.3495657444000244,
    0.5999994277954102,
    -0.5446184277534485,
    1.4630212783813477,
    0.7354453802108765,
    0.48866894841194153,
    -0.09582576900720596,
    -0.6399999856948853,
    0.31999996304512024
  ],
  "torque_all_Nm": [
    34.831024169921875,
    -2.1651878356933594,
    0.161865234375,
    -38.744712829589844,
    18.448837280273438,
    -4.771778583526611,
    5.569205284118652,
    -40.88298034667969,
    -4.316279411315918,
    -3.8194706439971924,
    -1.9236516952514648,
    -20.258167266845703,
    0.33945369720458984,
    0.21819210052490234,
    -0.06409883499145508,
    0.9669485092163086,
    0.4627366065979004,
    -1.9646681547164917,
    0.7814235687255859,
    -0.46504202485084534,
    0.2515355348587036,
    0.6023862361907959,
    0.25986456871032715
  ],
  "history": [
    {
      "episode_step": 9,
      "before": {
        "q_rad": [
          0.18580037355422974,
          0.5749382376670837,
          0.9329838156700134,
          -0.0485728420317173,
          -0.868030309677124,
          -0.5869687271118165,
          0.45143698930740356,
          0.1338343560695648,
          0.7853999972343445,
          0.062387888729572294,
          -0.0336437126994133,
          0.26138107776641845,
          -0.00582559108734132,
          0.6990739703178406,
          -0.7988757491111755,
          0.07930375933647155,
          -0.06251191794872285,
          1.305126338005066,
          1.2254485177993775,
          0.389300137758255,
          0.013354545459151268,
          -0.21593235433101654,
          0.41298240423202515
        ],
        "dq_radps": [
          0.647211492061615,
          -0.6273742318153381,
          16.037988662719727,
          6.079731464385986,
          -9.339241027832031,
          -2.9372291564941406,
          -0.7995242476463318,
          -3.08274245262146,
          3.147015377180651e-07,
          -1.0475363731384277,
          2.2536847591400146,
          5.056628704071045,
          -2.6821274757385254,
          3.0462467670440674,
          -3.3061869144439697,
          3.153482437133789,
          -2.4303085803985596,
          0.4492053985595703,
          -1.5522044897079468,
          1.2674765586853027,
          -2.1072046756744385,
          -0.40040820837020874,
          3.328299045562744
        ],
        "gravity_body": [
          -0.43643951416015625,
          -0.1852092146873474,
          -0.8804646730422974
        ],
        "root_omega_body_radps": [
          -0.6882439255714417,
          -1.2396326065063477,
          -6.774018287658691
        ],
        "root_height_m": 0.8147517442703247,
        "foot_heading_local_lin_vel_mps": {
          "left": [
            2.044888973236084,
            3.054388999938965,
            -1.4189279079437256
          ],
          "right": [
            -1.732090711593628,
            -1.6067925691604614,
            -0.00977034866809845
          ]
        }
      },
      "after": {
        "q_rad": [
          0.14584000408649445,
          0.4668697714805603,
          1.256048321723938,
          0.10369544476270676,
          -0.999996542930603,
          -0.6464609575271607,
          0.42161967635154723,
          0.12368708252906799,
          0.7854000568389893,
          0.05026196449995041,
          -0.003701062351465223,
          0.4199468374252319,
          -0.023977053165435802,
          0.7748696804046631,
          -0.8722755908966064,
          0.15440537333488463,
          -0.13243707716464997,
          1.3427671480178833,
          1.1772190618515015,
          0.4165458679199219,
          -0.04991926997900009,
          -0.2483801245689392,
          0.48705050349235535
        ],
        "dq_radps": [
          -2.849972724914551,
          -4.097993850708008,
          12.213210105895996,
          10.450984954833984,
          9.845962267718278e-08,
          -2.3242454528808594,
          -1.6528611183166504,
          0.7688685059547424,
          3.9130176787693927e-07,
          -0.14942818880081177,
          0.7357670068740845,
          6.877650260925293,
          -0.43281587958335876,
          3.23701548576355,
          -2.996281147003174,
          3.5366461277008057,
          -2.866387367248535,
          1.6612043380737305,
          -1.5098270177841187,
          1.2402288913726807,
          -2.678868532180786,
          -1.8107373714447021,
          3.470853805541992
        ],
        "gravity_body": [
          -0.42332395911216736,
          -0.21355479955673218,
          -0.8804494142532349
        ],
        "root_omega_body_radps": [
          -4.1505327224731445,
          -0.5161514282226562,
          -5.54843807220459
        ],
        "root_height_m": 0.7661814093589783,
        "foot_heading_local_lin_vel_mps": {
          "left": [
            -0.24247866868972778,
            3.5925042629241943,
            -0.8344228863716125
          ],
          "right": [
            -0.3911585211753845,
            -0.5520828366279602,
            -2.0263121128082275
          ]
        }
      }
    },
    {
      "episode_step": 10,
      "before": {
        "q_rad": [
          0.14584000408649445,
          0.4668697714805603,
          1.256048321723938,
          0.10369544476270676,
          -0.999996542930603,
          -0.6464609575271607,
          0.42161967635154723,
          0.12368708252906799,
          0.7854000568389893,
          0.05026196449995041,
          -0.003701062351465223,
          0.4199468374252319,
          -0.023977053165435802,
          0.7748696804046631,
          -0.8722755908966064,
          0.15440537333488463,
          -0.13243707716464997,
          1.3427671480178833,
          1.1772190618515015,
          0.4165458679199219,
          -0.04991926997900009,
          -0.2483801245689392,
          0.48705050349235535
        ],
        "dq_radps": [
          -2.849972724914551,
          -4.097993850708008,
          12.213210105895996,
          10.450984954833984,
          9.845962267718278e-08,
          -2.3242454528808594,
          -1.6528611183166504,
          0.7688685059547424,
          3.9130176787693927e-07,
          -0.14942818880081177,
          0.7357670068740845,
          6.877650260925293,
          -0.43281587958335876,
          3.23701548576355,
          -2.996281147003174,
          3.5366461277008057,
          -2.866387367248535,
          1.6612043380737305,
          -1.5098270177841187,
          1.2402288913726807,
          -2.678868532180786,
          -1.8107373714447021,
          3.470853805541992
        ],
        "gravity_body": [
          -0.42332395911216736,
          -0.21355479955673218,
          -0.8804494142532349
        ],
        "root_omega_body_radps": [
          -4.1505327224731445,
          -0.5161514282226562,
          -5.54843807220459
        ],
        "root_height_m": 0.7661814093589783,
        "foot_heading_local_lin_vel_mps": {
          "left": [
            -0.24247866868972778,
            3.5925042629241943,
            -0.8344228863716125
          ],
          "right": [
            -0.3911585211753845,
            -0.5520828366279602,
            -2.0263121128082275
          ]
        }
      },
      "after": {
        "q_rad": [
          0.05228911340236664,
          0.4004764258861542,
          1.5096789598464966,
          0.2955824136734009,
          -0.9985504150390625,
          -0.6887849044799805,
          0.4177973198890686,
          0.22564276456832885,
          0.7626037240028382,
          0.044196797013282774,
          0.014800677597522738,
          0.44073577523231505,
          -0.02684144377708436,
          0.8525059819221497,
          -0.9429015517234802,
          0.23944567441940307,
          -0.20946903228759767,
          1.3587672996520996,
          1.1111818838119507,
          0.44248560070991516,
          -0.11575013399124146,
          -0.3018747866153717,
          0.5223855376243591
        ],
        "dq_radps": [
          -4.3882269859313965,
          -1.9385006427764893,
          11.168930053710938,
          9.37771224975586,
          0.21550408005714417,
          -1.3797528743743896,
          0.40628963708877563,
          5.744168281555176,
          -1.4945203065872192,
          -0.16036583483219147,
          0.8825363516807556,
          0.977510929107666,
          -0.2629545032978058,
          3.28206729888916,
          -3.087055206298828,
          3.4732666015625,
          -3.3024048805236816,
          0.5475143790245056,
          -3.337096691131592,
          1.1545134782791138,
          -2.676269292831421,
          -2.668395757675171,
          1.151436448097229
        ],
        "gravity_body": [
          -0.4189990162849426,
          -0.1842501163482666,
          -0.8890960216522217
        ],
        "root_omega_body_radps": [
          -4.382093906402588,
          -1.3470768928527832,
          -5.290084362030029
        ],
        "root_height_m": 0.7161569595336914,
        "foot_heading_local_lin_vel_mps": {
          "left": [
            -0.6314171552658081,
            3.087799072265625,
            -0.503707766532898
          ],
          "right": [
            -0.1258639544248581,
            -0.493890643119812,
            -2.410926103591919
          ]
        }
      }
    },
    {
      "episode_step": 11,
      "before": {
        "q_rad": [
          0.05228911340236664,
          0.4004764258861542,
          1.5096789598464966,
          0.2955824136734009,
          -0.9985504150390625,
          -0.6887849044799805,
          0.4177973198890686,
          0.22564276456832885,
          0.7626037240028382,
          0.044196797013282774,
          0.014800677597522738,
          0.44073577523231505,
          -0.02684144377708436,
          0.8525059819221497,
          -0.9429015517234802,
          0.23944567441940307,
          -0.20946903228759767,
          1.3587672996520996,
          1.1111818838119507,
          0.44248560070991516,
          -0.11575013399124146,
          -0.3018747866153717,
          0.5223855376243591
        ],
        "dq_radps": [
          -4.3882269859313965,
          -1.9385006427764893,
          11.168930053710938,
          9.37771224975586,
          0.21550408005714417,
          -1.3797528743743896,
          0.40628963708877563,
          5.744168281555176,
          -1.4945203065872192,
          -0.16036583483219147,
          0.8825363516807556,
          0.977510929107666,
          -0.2629545032978058,
          3.28206729888916,
          -3.087055206298828,
          3.4732666015625,
          -3.3024048805236816,
          0.5475143790245056,
          -3.337096691131592,
          1.1545134782791138,
          -2.676269292831421,
          -2.668395757675171,
          1.151436448097229
        ],
        "gravity_body": [
          -0.4189990162849426,
          -0.1842501163482666,
          -0.8890960216522217
        ],
        "root_omega_body_radps": [
          -4.382093906402588,
          -1.3470768928527832,
          -5.290084362030029
        ],
        "root_height_m": 0.7161569595336914,
        "foot_heading_local_lin_vel_mps": {
          "left": [
            -0.6314171552658081,
            3.087799072265625,
            -0.503707766532898
          ],
          "right": [
            -0.1258639544248581,
            -0.493890643119812,
            -2.410926103591919
          ]
        }
      },
      "after": {
        "q_rad": [
          -0.17348189651966095,
          0.2687613368034363,
          1.7595336437225342,
          0.42058610916137695,
          -0.9999996423721313,
          -0.7101976346969605,
          0.44897674798965453,
          0.3127832055091858,
          0.7853999972343445,
          0.029355907291173933,
          0.03454908221960068,
          0.37720422744750975,
          -0.07583518028259278,
          0.8781847357749939,
          -1.0165646076202393,
          0.3255919098854065,
          -0.28505965769290925,
          1.3508264112472534,
          1.0331821727752686,
          0.46113699674606323,
          -0.12561090290546417,
          -0.36722877621650696,
          0.5168052315711975
        ],
        "dq_radps": [
          -10.434995651245117,
          -4.469593524932861,
          10.554770469665527,
          5.653179168701172,
          -9.275515822082525e-08,
          0.03439970314502716,
          2.4011342525482178,
          2.5760738849639893,
          -3.268710599968472e-07,
          -0.7286328077316284,
          0.9806066155433655,
          -2.6947498321533203,
          -2.1125810146331787,
          1.0638211965560913,
          -3.3352739810943604,
          3.5254268646240234,
          -3.2866766452789307,
          -0.4085853099822998,
          -3.501260280609131,
          0.7127089500427246,
          -0.4110598564147949,
          -3.163611650466919,
          -0.6476088762283325
        ],
        "gravity_body": [
          -0.4160000681877136,
          -0.15786361694335938,
          -0.8955574035644531
        ],
        "root_omega_body_radps": [
          -3.8746023178100586,
          -0.8819398880004883,
          -5.351299285888672
        ],
        "root_height_m": 0.66201251745224,
        "foot_heading_local_lin_vel_mps": {
          "left": [
            -0.13070990145206451,
            1.9574048519134521,
            -1.3440613746643066
          ],
          "right": [
            0.503102719783783,
            -0.5535951256752014,
            -2.618328809738159
          ]
        }
      }
    },
    {
      "episode_step": 12,
      "before": {
        "q_rad": [
          -0.17348189651966095,
          0.2687613368034363,
          1.7595336437225342,
          0.42058610916137695,
          -0.9999996423721313,
          -0.7101976346969605,
          0.44897674798965453,
          0.3127832055091858,
          0.7853999972343445,
          0.029355907291173933,
          0.03454908221960068,
          0.37720422744750975,
          -0.07583518028259278,
          0.8781847357749939,
          -1.0165646076202393,
          0.3255919098854065,
          -0.28505965769290925,
          1.3508264112472534,
          1.0331821727752686,
          0.46113699674606323,
          -0.12561090290546417,
          -0.36722877621650696,
          0.5168052315711975
        ],
        "dq_radps": [
          -10.434995651245117,
          -4.469593524932861,
          10.554770469665527,
          5.653179168701172,
          -9.275515822082525e-08,
          0.03439970314502716,
          2.4011342525482178,
          2.5760738849639893,
          -3.268710599968472e-07,
          -0.7286328077316284,
          0.9806066155433655,
          -2.6947498321533203,
          -2.1125810146331787,
          1.0638211965560913,
          -3.3352739810943604,
          3.5254268646240234,
          -3.2866766452789307,
          -0.4085853099822998,
          -3.501260280609131,
          0.7127089500427246,
          -0.4110598564147949,
          -3.163611650466919,
          -0.6476088762283325
        ],
        "gravity_body": [
          -0.4160000681877136,
          -0.15786361694335938,
          -0.8955574035644531
        ],
        "root_omega_body_radps": [
          -3.8746023178100586,
          -0.8819398880004883,
          -5.351299285888672
        ],
        "root_height_m": 0.66201251745224,
        "foot_heading_local_lin_vel_mps": {
          "left": [
            -0.13070990145206451,
            1.9574048519134521,
            -1.3440613746643066
          ],
          "right": [
            0.503102719783783,
            -0.5535951256752014,
            -2.618328809738159
          ]
        }
      },
      "after": {
        "q_rad": [
          -0.40758252143859863,
          0.13889667391777039,
          1.990387201309204,
          0.5614250302314758,
          -0.9692946076393127,
          -0.7133145999908448,
          0.4995599317550659,
          0.3594167172908783,
          0.7853574395179749,
          0.011259616762399671,
          0.0669785761833191,
          0.3130120813846588,
          -0.11677662730216981,
          0.8703212141990662,
          -1.0920695066452026,
          0.248408305644989,
          -0.3635674983263016,
          1.3682126092910767,
          0.9638556528091431,
          0.48291102051734924,
          -0.10502716153860092,
          -0.4395432472229004,
          0.4833061099052429
        ],
        "dq_radps": [
          -10.468816757202148,
          -5.374852657318115,
          9.660781860351562,
          8.078372955322266,
          2.7456700801849365,
          0.38125836849212646,
          2.868718147277832,
          1.0914506912231445,
          -5.804188549518585e-05,
          -0.6214860677719116,
          1.7939072847366333,
          -3.4831509590148926,
          -2.1243042945861816,
          -0.373867005109787,
          -3.4157235622406006,
          -0.3102928102016449,
          -3.425100326538086,
          1.0021835565567017,
          -3.2322661876678467,
          0.5341216921806335,
          0.8960220813751221,
          -3.434276819229126,
          -1.8165655136108398
        ],
        "gravity_body": [
          -0.41459736227989197,
          -0.129391610622406,
          -0.9007590413093567
        ],
        "root_omega_body_radps": [
          -4.392889499664307,
          -0.85135817527771,
          -5.086843013763428
        ],
        "root_height_m": 0.6092961430549622,
        "foot_heading_local_lin_vel_mps": {
          "left": [
            -0.3757646381855011,
            2.044487476348877,
            -0.4512869119644165
          ],
          "right": [
            0.8919354677200317,
            -0.10404549539089203,
            -3.282369375228882
          ]
        }
      }
    },
    {
      "episode_step": 13,
      "before": {
        "q_rad": [
          -0.40758252143859863,
          0.13889667391777039,
          1.990387201309204,
          0.5614250302314758,
          -0.9692946076393127,
          -0.7133145999908448,
          0.4995599317550659,
          0.3594167172908783,
          0.7853574395179749,
          0.011259616762399671,
          0.0669785761833191,
          0.3130120813846588,
          -0.11677662730216981,
          0.8703212141990662,
          -1.0920695066452026,
          0.248408305644989,
          -0.3635674983263016,
          1.3682126092910767,
          0.9638556528091431,
          0.48291102051734924,
          -0.10502716153860092,
          -0.4395432472229004,
          0.4833061099052429
        ],
        "dq_radps": [
          -10.468816757202148,
          -5.374852657318115,
          9.660781860351562,
          8.078372955322266,
          2.7456700801849365,
          0.38125836849212646,
          2.868718147277832,
          1.0914506912231445,
          -5.804188549518585e-05,
          -0.6214860677719116,
          1.7939072847366333,
          -3.4831509590148926,
          -2.1243042945861816,
          -0.373867005109787,
          -3.4157235622406006,
          -0.3102928102016449,
          -3.425100326538086,
          1.0021835565567017,
          -3.2322661876678467,
          0.5341216921806335,
          0.8960220813751221,
          -3.434276819229126,
          -1.8165655136108398
        ],
        "gravity_body": [
          -0.41459736227989197,
          -0.129391610622406,
          -0.9007590413093567
        ],
        "root_omega_body_radps": [
          -4.392889499664307,
          -0.85135817527771,
          -5.086843013763428
        ],
        "root_height_m": 0.6092961430549622,
        "foot_heading_local_lin_vel_mps": {
          "left": [
            -0.3757646381855011,
            2.044487476348877,
            -0.4512869119644165
          ],
          "right": [
            0.8919354677200317,
            -0.10404549539089203,
            -3.282369375228882
          ]
        }
      },
      "after": {
        "q_rad": [
          -0.6706504225730896,
          0.03508239611983299,
          2.171191453933716,
          0.7066859602928162,
          -0.9079880118370056,
          -0.7101512622833253,
          0.5848156738281249,
          0.3111933708190918,
          0.7876096606254578,
          -0.014118807911872866,
          0.12492257595062256,
          0.16022868752479552,
          -0.09356532692909242,
          0.8404168486595154,
          -1.1748207807540894,
          0.3281175374984741,
          -0.45574153661727906,
          1.4180046606063843,
          0.8971455979347229,
          0.5001569390296936,
          -0.033971063792705536,
          -0.5152983665466309,
          0.4324463903903961
        ],
        "dq_radps": [
          -2.2676820755004883,
          4.065006732940674,
          9.28255558013916,
          1.3128368854522705,
          -20.56983757019043,
          3.635037660598755,
          6.745122909545898,
          6.718378067016602,
          -9.557825088500977,
          1.0093169212341309,
          5.535492897033691,
          -8.462597846984863,
          -4.580289840698242,
          -1.1143438816070557,
          -5.025477886199951,
          1.3811969757080078,
          -32.08342361450195,
          2.865189552307129,
          -1.1796460151672363,
          0.19305406510829926,
          -68.68045806884766,
          -3.532449245452881,
          -2.5118355751037598
        ],
        "gravity_body": [
          -0.41959691047668457,
          -0.08437828719615936,
          -0.9037802219390869
        ],
        "root_omega_body_radps": [
          0.8222777843475342,
          -0.5967093110084534,
          -3.60252046585083
        ],
        "root_height_m": 0.5597235560417175,
        "foot_heading_local_lin_vel_mps": {
          "left": [
            0.2502298355102539,
            2.1967196464538574,
            -0.5334039330482483
          ],
          "right": [
            5.116176605224609,
            -5.961627006530762,
            3.722447395324707
          ]
        }
      }
    }
  ],
  "missing": "no original contact-force/substep/solver-impulse recording; cannot uniquely attribute contact versus PD",
  "position_violation_joints": [
    {
      "joint": "right_thigh_pitch_joint",
      "q_rad": 0.7876096606254578,
      "bounds_rad": [
        -2.094,
        0.7854
      ]
    }
  ],
  "before_root_height_m": 0.6092961430549622,
  "after_root_height_m": 0.5597235560417175,
  "before_foot_velocity_mps": {
    "left": [
      -0.3757646381855011,
      2.044487476348877,
      -0.4512869119644165
    ],
    "right": [
      0.8919354677200317,
      -0.10404549539089203,
      -3.282369375228882
    ]
  },
  "after_foot_velocity_mps": {
    "left": [
      0.2502298355102539,
      2.1967196464538574,
      -0.5334039330482483
    ],
    "right": [
      5.116176605224609,
      -5.961627006530762,
      3.722447395324707
    ]
  }
}
```

### 旧右踝极值前后足底碰撞网格高度

```json
{
  "method": "offline official collision-mesh FK of G peak pre/post state, flat-ground yaw invariant; not Isaac contact force reconstruction",
  "sole_min_z_by_foot_m": [
    {
      "episode_step": 9,
      "before": {
        "left": 0.08241001868774654,
        "right": 0.2055253056029257
      },
      "after": {
        "left": 0.04566694610877956,
        "right": 0.19785809365564674
      }
    },
    {
      "episode_step": 10,
      "before": {
        "left": 0.04566694610877956,
        "right": 0.19785809365564674
      },
      "after": {
        "left": 0.02285120864309964,
        "right": 0.15847616495463504
      }
    },
    {
      "episode_step": 11,
      "before": {
        "left": 0.02285120864309964,
        "right": 0.15847616495463504
      },
      "after": {
        "left": 0.0007421178184519317,
        "right": 0.11337915284484243
      }
    },
    {
      "episode_step": 12,
      "before": {
        "left": 0.0007421178184519317,
        "right": 0.11337915284484243
      },
      "after": {
        "left": -0.0008613280979217486,
        "right": 0.052633397247139516
      }
    },
    {
      "episode_step": 13,
      "before": {
        "left": -0.0008613280979217486,
        "right": 0.052633397247139516
      },
      "after": {
        "left": -0.00095092272778214,
        "right": -0.00785294977111213
      }
    }
  ]
}
```

同一旧案例的真实姿态离线FK显示，右脚最低碰撞网格点从+5.263cm降到−0.785cm；更早右脚由20.55cm持续下降，左脚先落地，随后右脚在该20ms控制步进入地面。结合右脚速度由向下−3.282m/s转为向上+3.722m/s，支持“落足接触/穿透附近速度突变”的动力学解释。这是几何与运动学侧证，缺少原Isaac子步接触冲量，不能把离线网格高度当已测量的PhysX穿透深度或唯一因果证明。

离线官方MJCF与真实碰撞网格审计：13644有效train帧中1284帧足底穿透超过2mm；1024个旧G实际reset样本中84个穿透超过2mm、240个双足最低点高于地面5cm；没有超过2cm的穿透。地面距离采用网格最低点，不能以脚link原点替代。原始manifest的大穿透统计与当前valid帧范围不同，未直接整段删除专家数据。

H仅在原train 8-frame合法窗口起点上建立RSI接触池：官方来源2141个、G1来源2049个，按来源0.5/0.5采样。候选需物理位置可达、初始dq≤0.70×Vmax、最小足底高度≤2cm、向上root修正≤3cm、非相邻自穿透≤2mm、root高度≥0.53m、倾角≤0.70rad、保护部位碰撞网格距地>2.5cm；需要时向上移动root，使足底最低点至少2mm。原始q/dq、root速度、专家连续帧与训练/验证/测试划分均不改。并未证明所有保留初态动态可行。

RSI后重置选中环境接触传感器，写入scene并sim.forward/robot.update(0)，随后使观察缓存失效；不额外推进物理时间。实际配对实验FK误差刷新前后均约0.996mm，未复现明显陈旧FK，因此刷新作为一致性处理，不宣称已证实其为68.68rad/s主因。

保持原显式零延迟PD、增益、关节惯量、力矩和速度边界；physics dt从5ms降为2.5ms、decimation从4升8，policy仍50Hz，slew仍每控制步0.5×Vmax。脚部名义Kp40/Kd2/armature0.01/effort27Nm/velocity8rad/s；原startup DR继续生效，实测Kp36.0101–43.8791、Kd1.80315–2.19737。未更换隐式PD、修改URDF惯量或放宽硬限位。

PD/PhysX字段核对：URDF drive Kp/Kd均0，动力来自显式DelayedPD，避免双重PD。资产已显式设置脚/肩/臂velocity_limit_sim=8rad/s、effort_limit_sim=27Nm，腿/躯干25rad/s与120Nm，并非漏设仿真速度限制。articulation solver position/velocity iterations=8/4，self_collisions=True，max_depenetration_velocity=1m/s，rigid max linear/angular velocity=1000。速度限制由求解器尝试制动，接触冲击下仍会瞬时超过，不能当数学硬截断；驱动力矩边界也不能限制所有接触约束冲量。

地面为离线生成100m×100m平面网格（seed42、1×1 terrain），静/动摩擦均1.0，friction/restitution combine=multiply；episode上限20秒=1000控制步。contact sensor update_period=physics dt、history_length=3，保护body为torso以及左右thigh yaw/roll links。环境关闭观察噪声、禁用base高度扫描，PD delay覆盖为0；没有依靠外部地面USD下载。

| PD组 | 名义Kp | 名义Kd | armature | 仿真速度rad/s | 仿真力矩Nm | 训练delay |
|---|---|---|---|---|---|---|
| thigh yaw/roll/pitch | 100 | 3.3 | 0.01 | 25 | 120 | 0 |
| knee/torso | 150 | 5 | 0.01 | 25 | 120 | 0 |
| feet/shoulders | 40 | 2 | 0.01 | 8 | 27 | 0 |
| elbow pitch | 30 | 1.5 | 0.01 | 8 | 27 | 0 |
| elbow yaw | 20 | 1 | 0.01 | 8 | 27 | 0 |

startup随机化保持：全joint Kp/Kd×[0.9,1.1]、armature×[0.8,1.2]、左右link质量×[0.9,1.1]、torso加质量[−1,+1]kg、所有body静摩擦[0.3,1.6]/动摩擦[0.3,1.2]（64 buckets、make_consistent=True）、restitution[0,0.5]、base/torso COM x/y±0.025m与z±0.05m；真实脚PD测量范围来自诊断回执，其余范围是冻结配置而非本轮实测逐样本分布。重置事件的随机root/q/dq随后被RSI参考状态覆盖，startup属性随机化保留。

每物理子步锁存位置/速度/力矩越限与保护部位接触>1N，控制步内任一越限均产生真实terminated。原接触history=3帧，在更细dt下仅覆盖7.5ms，新增锁存保留整个20ms控制间隔的检测，不以缩短历史隐藏失败。H硬flags口径因此严于旧G仅控制步末检测，直接跨版本比较counts须说明这一差异。height<0.50m、倾角>0.80rad仍终止。

## 受管GPU配对诊断（训练内物理检查）


Job job_5e90cb102463417a828f / Run run_0f51028109cb4b2e9c12；单GPU、CPU8/RAM24GiB/shm8GiB，最多15分钟、无自动重试。32环境×16轮×16控制步；固定train共同接触候选512个初态、相同requested动作序列与startup seed。前4步保持实测姿态anchor，之后使用固定随机动作；没有learner、held-out或模型验收。solver warm start未验证；各阶段终止不同，后续活跃样本覆盖不同，峰值下降不能归为纯单因素因果效应。

| 阶段 | physics dt | decimation | 首步终止/512 | 活跃物理dq峰值rad/s | 20ms边界dq峰值 | 足部力峰N | FK误差峰m |
|---|---|---|---|---|---|---|---|
| baseline | 0.005 | 4 | 0 | 29.4400616 | 29.4400616 | 3794.49854 | 0.000996204993 |
| grounded | 0.005 | 4 | 0 | 24.4792671 | 19.5618362 | 3794.49854 | 0.000996204993 |
| substeps | 0.0025 | 8 | 0 | 22.179285 | 16.8850517 | 7688.21191 | 0.000996204993 |

共同候选池已经筛选，三阶段0/512首步终止不等于旧全RSI分布首步问题已消除。新配置仍有越限，且接触力峰增大；力与dt有关但未记录力的时间积分，不能用峰值×dt冒充接触冲量，更不能宣布首步接触冲击彻底解决。此证据只支持进入有界500更新观测。

### 完整诊断协议、原始文件SHA与实测回执

```json
{
  "protocol": {
    "schema_version": "eb/ufo-contact-probe/v3",
    "phases": [
      "baseline",
      "grounded",
      "substeps"
    ],
    "envs": 32,
    "rounds": 16,
    "steps": 16,
    "seed": 64042,
    "split": "train",
    "pairing": "same reference state, requested action sequence and startup seed; solver warm start unverified",
    "scope": "contact-eligible reference starts; diagnostic only, no learner update or acceptance",
    "physics_samples": "new integration frames only; exclude prior interval/startup contact"
  },
  "status": "measured",
  "asset_sha256": "b58868a782d8863ac38059eb4eab89d27ee1d70565bf58f5a055f44510c8e842",
  "manifest_sha256": "dc0c6585887507df59f675963b2733d25c9836d805e9bb89f2402398f07bd288",
  "files": [
    {
      "name": "paired-inputs.npz",
      "bytes": 103660,
      "sha256": "f4e3681bde6c358a92465e1d2ba5d10f958a7abe9323f78578ae4dd48db3e3fc"
    },
    {
      "name": "baseline.npz",
      "bytes": 2989904,
      "sha256": "ba874d325864d86bb03d6ba0207adddd8dd188af6996734224a3760c159e18fc"
    },
    {
      "name": "grounded.npz",
      "bytes": 3329689,
      "sha256": "9e039adb9512ce4347a9fc2883c672886f92605d3fa67aa2b4d03592a79dad28"
    },
    {
      "name": "substeps.npz",
      "bytes": 3329182,
      "sha256": "fb66de3c550a7d7c096642a173d9a882b992932aa16f52e51b74835642ae61d9"
    }
  ],
  "phases": [
    {
      "phase": "baseline",
      "physics_dt_s": 0.005,
      "decimation": 4,
      "first_terminated": 0,
      "initial_states": 512,
      "max_dq_active": 29.440061569213867,
      "max_boundary_dq_active": 29.440061569213867,
      "max_foot_force_active_N": 3794.49853515625,
      "fk_before_max_m": 0.0009962049929651041,
      "fk_after_max_m": 0.0009962049929651041,
      "first_hard_flags": [
        0,
        0,
        0
      ],
      "feet_controller_stiffness_range": [
        36.01011657714844,
        43.87909698486328
      ],
      "feet_controller_damping_range": [
        1.8031455278396606,
        2.1973743438720703
      ]
    },
    {
      "phase": "grounded",
      "physics_dt_s": 0.005,
      "decimation": 4,
      "first_terminated": 0,
      "initial_states": 512,
      "max_dq_active": 24.479267120361328,
      "max_boundary_dq_active": 19.56183624267578,
      "max_foot_force_active_N": 3794.49853515625,
      "fk_before_max_m": 0.0009962049929651041,
      "fk_after_max_m": 0.0009962049929651041,
      "first_hard_flags": [
        0,
        0,
        0
      ],
      "feet_controller_stiffness_range": [
        36.01011657714844,
        43.87909698486328
      ],
      "feet_controller_damping_range": [
        1.8031455278396606,
        2.1973743438720703
      ]
    },
    {
      "phase": "substeps",
      "physics_dt_s": 0.0025,
      "decimation": 8,
      "first_terminated": 0,
      "initial_states": 512,
      "max_dq_active": 22.179285049438477,
      "max_boundary_dq_active": 16.885051727294922,
      "max_foot_force_active_N": 7688.2119140625,
      "fk_before_max_m": 0.0009962049929651041,
      "fk_after_max_m": 0.0009962049929651041,
      "first_hard_flags": [
        0,
        0,
        0
      ],
      "feet_controller_stiffness_range": [
        36.01011657714844,
        43.87909698486328
      ],
      "feet_controller_damping_range": [
        1.8031455278396606,
        2.1973743438720703
      ]
    }
  ]
}
```

### 诊断首步分布、保持姿态四步与活跃样本覆盖

```json
[
  {
    "phase": "baseline",
    "active_transitions": 7092,
    "terminated_active_total": 204,
    "hold_first4_terminated": 11,
    "first_foot_peak_quantiles_N": {
      "0.5": 0.0,
      "0.95": 694.0052490234375,
      "0.99": 1185.0955810546875,
      "1": 1814.0902099609375
    },
    "first_dq_peak_quantiles_radps": {
      "0.5": 1.4151268005371094,
      "0.95": 7.608170509338379,
      "0.99": 10.995383262634277,
      "1": 12.84577465057373
    },
    "protected_contact_terms": null
  },
  {
    "phase": "grounded",
    "active_transitions": 6329,
    "terminated_active_total": 316,
    "hold_first4_terminated": 32,
    "first_foot_peak_quantiles_N": {
      "0.5": 0.0,
      "0.95": 170.33514404296875,
      "0.99": 359.0414733886719,
      "1": 497.0079345703125
    },
    "first_dq_peak_quantiles_radps": {
      "0.5": 1.2930645942687988,
      "0.95": 4.343199729919434,
      "0.99": 6.3180155754089355,
      "1": 11.115426063537598
    },
    "protected_contact_terms": 0
  },
  {
    "phase": "substeps",
    "active_transitions": 6011,
    "terminated_active_total": 304,
    "hold_first4_terminated": 79,
    "first_foot_peak_quantiles_N": {
      "0.5": 0.0,
      "0.95": 133.20201110839844,
      "0.99": 637.371826171875,
      "1": 1087.808349609375
    },
    "first_dq_peak_quantiles_radps": {
      "0.5": 1.7638237476348877,
      "0.95": 5.031667232513428,
      "0.99": 7.108170509338379,
      "1": 11.951934814453125
    },
    "protected_contact_terms": 0
  }
]
```

保持初始anchor的前四步累计终止：baseline11/512、grounded32/512、substeps79/512；H检测了整个控制间隔的瞬时越限，口径严于baseline。即使首步0终止，参考初态配合静态保持也不保证数步后稳定；不能只报告下降的峰值而忽略这些终止。

### 原始数据/旧reset离线几何审计

```json
{
  "method": "offline official MJCF FK and foot mesh vertices; no Isaac rollout; yaw/xy invariant flat ground",
  "dataset_train": [
    {
      "clip": "g1_dance1_subject1_clip3",
      "source": "g1",
      "valid_frames": 368,
      "sole_min_height_m": 2.140661905722374e-08,
      "both_feet_airborne_fraction": 0.04076086956521739,
      "penetration_over_2mm": 0,
      "penetration_over_2cm": 0,
      "max_ground_contact_depth_m": 0.0,
      "near_base_height_50cm_fraction": 0.0
    },
    {
      "clip": "g1_dance1_subject2_clip3",
      "source": "g1",
      "valid_frames": 291,
      "sole_min_height_m": 0.0058914095696785615,
      "both_feet_airborne_fraction": 0.25773195876288657,
      "penetration_over_2mm": 0,
      "penetration_over_2cm": 0,
      "max_ground_contact_depth_m": 0.0,
      "near_base_height_50cm_fraction": 0.0
    },
    {
      "clip": "g1_dance1_subject3_clip3",
      "source": "g1",
      "valid_frames": 228,
      "sole_min_height_m": 0.007751257036604866,
      "both_feet_airborne_fraction": 0.17543859649122806,
      "penetration_over_2mm": 0,
      "penetration_over_2cm": 0,
      "max_ground_contact_depth_m": 0.0,
      "near_base_height_50cm_fraction": 0.0
    },
    {
      "clip": "g1_fallAndGetUp1_subject1_clip3",
      "source": "g1",
      "valid_frames": 212,
      "sole_min_height_m": 1.7022192284193993e-08,
      "both_feet_airborne_fraction": 0.0,
      "penetration_over_2mm": 0,
      "penetration_over_2cm": 0,
      "max_ground_contact_depth_m": 0.0,
      "near_base_height_50cm_fraction": 0.0047169811320754715
    },
    {
      "clip": "g1_fallAndGetUp1_subject4_clip3",
      "source": "g1",
      "valid_frames": 202,
      "sole_min_height_m": 0.004228276353337605,
      "both_feet_airborne_fraction": 0.995049504950495,
      "penetration_over_2mm": 0,
      "penetration_over_2cm": 0,
      "max_ground_contact_depth_m": 0.0,
      "near_base_height_50cm_fraction": 0.019801980198019802
    },
    {
      "clip": "g1_fallAndGetUp1_subject5_clip3",
      "source": "g1",
      "valid_frames": 172,
      "sole_min_height_m": 0.0027371106713417748,
      "both_feet_airborne_fraction": 0.1744186046511628,
      "penetration_over_2mm": 0,
      "penetration_over_2cm": 0,
      "max_ground_contact_depth_m": 0.0,
      "near_base_height_50cm_fraction": 0.023255813953488372
    },
    {
      "clip": "g1_fight1_subject2_clip3",
      "source": "g1",
      "valid_frames": 381,
      "sole_min_height_m": -1.7172374400936663e-08,
      "both_feet_airborne_fraction": 0.3228346456692913,
      "penetration_over_2mm": 0,
      "penetration_over_2cm": 0,
      "max_ground_contact_depth_m": 1.7172374397467216e-08,
      "near_base_height_50cm_fraction": 0.0
    },
    {
      "clip": "g1_fight1_subject3_clip3",
      "source": "g1",
      "valid_frames": 371,
      "sole_min_height_m": 2.117735842938062e-08,
      "both_feet_airborne_fraction": 0.9838274932614556,
      "penetration_over_2mm": 0,
      "penetration_over_2cm": 0,
      "max_ground_contact_depth_m": 0.0,
      "near_base_height_50cm_fraction": 0.0
    },
    {
      "clip": "g1_fight1_subject5_clip3",
      "source": "g1",
      "valid_frames": 392,
      "sole_min_height_m": -1.0522654310396451e-08,
      "both_feet_airborne_fraction": 0.14540816326530612,
      "penetration_over_2mm": 0,
      "penetration_over_2cm": 0,
      "max_ground_contact_depth_m": 1.0522654306927004e-08,
      "near_base_height_50cm_fraction": 0.0
    },
    {
      "clip": "g1_jumps1_subject1_clip3",
      "source": "g1",
      "valid_frames": 356,
      "sole_min_height_m": 9.197433273322453e-09,
      "both_feet_airborne_fraction": 0.33146067415730335,
      "penetration_over_2mm": 0,
      "penetration_over_2cm": 0,
      "max_ground_contact_depth_m": 0.0,
      "near_base_height_50cm_fraction": 0.0
    },
    {
      "clip": "g1_jumps1_subject2_clip3",
      "source": "g1",
      "valid_frames": 390,
      "sole_min_height_m": 3.537069863737696e-08,
      "both_feet_airborne_fraction": 0.31794871794871793,
      "penetration_over_2mm": 0,
      "penetration_over_2cm": 0,
      "max_ground_contact_depth_m": 0.0,
      "near_base_height_50cm_fraction": 0.0
    },
    {
      "clip": "g1_jumps1_subject5_clip3",
      "source": "g1",
      "valid_frames": 378,
      "sole_min_height_m": -2.8488182819697005e-09,
      "both_feet_airborne_fraction": 0.17724867724867724,
      "penetration_over_2mm": 0,
      "penetration_over_2cm": 0,
      "max_ground_contact_depth_m": 2.8488182819697005e-09,
      "near_base_height_50cm_fraction": 0.0
    },
    {
      "clip": "g1_run1_subject2_clip3",
      "source": "g1",
      "valid_frames": 387,
      "sole_min_height_m": -1.5592169812228551e-09,
      "both_feet_airborne_fraction": 0.23255813953488372,
      "penetration_over_2mm": 0,
      "penetration_over_2cm": 0,
      "max_ground_contact_depth_m": 1.5592169777534082e-09,
      "near_base_height_50cm_fraction": 0.0
    },
    {
      "clip": "g1_run1_subject5_clip3",
      "source": "g1",
      "valid_frames": 223,
      "sole_min_height_m": -3.857262009104856e-08,
      "both_feet_airborne_fraction": 0.484304932735426,
      "penetration_over_2mm": 0,
      "penetration_over_2cm": 0,
      "max_ground_contact_depth_m": 3.857262009104856e-08,
      "near_base_height_50cm_fraction": 0.0
    },
    {
      "clip": "g1_sprint1_subject2_clip3",
      "source": "g1",
      "valid_frames": 371,
      "sole_min_height_m": 4.754435862369677e-09,
      "both_feet_airborne_fraction": 0.1509433962264151,
      "penetration_over_2mm": 0,
      "penetration_over_2cm": 0,
      "max_ground_contact_depth_m": 0.0,
      "near_base_height_50cm_fraction": 0.0
    },
    {
      "clip": "g1_sprint1_subject4_clip3",
      "source": "g1",
      "valid_frames": 395,
      "sole_min_height_m": 3.2148584468150165e-08,
      "both_feet_airborne_fraction": 0.2582278481012658,
      "penetration_over_2mm": 0,
      "penetration_over_2cm": 0,
      "max_ground_contact_depth_m": 0.0,
      "near_base_height_50cm_fraction": 0.0
    },
    {
      "clip": "g1_walk1_subject1_clip3",
      "source": "g1",
      "valid_frames": 400,
      "sole_min_height_m": -2.8648633351030517e-08,
      "both_feet_airborne_fraction": 0.0,
      "penetration_over_2mm": 0,
      "penetration_over_2cm": 0,
      "max_ground_contact_depth_m": 2.8648633354499964e-08,
      "near_base_height_50cm_fraction": 0.0
    },
    {
      "clip": "g1_walk1_subject2_clip3",
      "source": "g1",
      "valid_frames": 400,
      "sole_min_height_m": -1.5970579088364634e-08,
      "both_feet_airborne_fraction": 0.0,
      "penetration_over_2mm": 0,
      "penetration_over_2cm": 0,
      "max_ground_contact_depth_m": 1.5970579088364634e-08,
      "near_base_height_50cm_fraction": 0.0
    },
    {
      "clip": "g1_walk1_subject5_clip3",
      "source": "g1",
      "valid_frames": 400,
      "sole_min_height_m": -2.137287536535526e-08,
      "both_feet_airborne_fraction": 0.0,
      "penetration_over_2mm": 0,
      "penetration_over_2cm": 0,
      "max_ground_contact_depth_m": 2.137287536535526e-08,
      "near_base_height_50cm_fraction": 0.0
    },
    {
      "clip": "g1_walk2_subject1_clip3",
      "source": "g1",
      "valid_frames": 400,
      "sole_min_height_m": 1.7727375612591745e-08,
      "both_feet_airborne_fraction": 0.0,
      "penetration_over_2mm": 0,
      "penetration_over_2cm": 0,
      "max_ground_contact_depth_m": 0.0,
      "near_base_height_50cm_fraction": 0.0
    },
    {
      "clip": "g1_walk2_subject3_clip3",
      "source": "g1",
      "valid_frames": 400,
      "sole_min_height_m": 3.9449295670401785e-09,
      "both_feet_airborne_fraction": 0.0,
      "penetration_over_2mm": 0,
      "penetration_over_2cm": 0,
      "max_ground_contact_depth_m": 0.0,
      "near_base_height_50cm_fraction": 0.0
    },
    {
      "clip": "g1_walk2_subject4_clip3",
      "source": "g1",
      "valid_frames": 393,
      "sole_min_height_m": 1.926423620624007e-08,
      "both_feet_airborne_fraction": 0.0,
      "penetration_over_2mm": 0,
      "penetration_over_2cm": 0,
      "max_ground_contact_depth_m": 0.0,
      "near_base_height_50cm_fraction": 0.0
    },
    {
      "clip": "rpo_114_08",
      "source": "official_rpo",
      "valid_frames": 460,
      "sole_min_height_m": -0.01971513332474169,
      "both_feet_airborne_fraction": 0.8173913043478261,
      "penetration_over_2mm": 68,
      "penetration_over_2cm": 0,
      "max_ground_contact_depth_m": 0.01971513332474169,
      "near_base_height_50cm_fraction": 0.0
    },
    {
      "clip": "rpo_114_09",
      "source": "official_rpo",
      "valid_frames": 782,
      "sole_min_height_m": 0.001985630822571135,
      "both_feet_airborne_fraction": 0.5191815856777494,
      "penetration_over_2mm": 0,
      "penetration_over_2cm": 0,
      "max_ground_contact_depth_m": 0.0,
      "near_base_height_50cm_fraction": 0.0
    },
    {
      "clip": "rpo_127_04",
      "source": "official_rpo",
      "valid_frames": 15,
      "sole_min_height_m": -0.014888435203392825,
      "both_feet_airborne_fraction": 0.0,
      "penetration_over_2mm": 9,
      "penetration_over_2cm": 0,
      "max_ground_contact_depth_m": 0.014888435203392825,
      "near_base_height_50cm_fraction": 0.0
    },
    {
      "clip": "rpo_127_06",
      "source": "official_rpo",
      "valid_frames": 13,
      "sole_min_height_m": -0.0192916483165141,
      "both_feet_airborne_fraction": 0.0,
      "penetration_over_2mm": 10,
      "penetration_over_2cm": 0,
      "max_ground_contact_depth_m": 0.0192916483165141,
      "near_base_height_50cm_fraction": 0.0
    },
    {
      "clip": "rpo_36_01",
      "source": "official_rpo",
      "valid_frames": 145,
      "sole_min_height_m": -0.019992022955008532,
      "both_feet_airborne_fraction": 0.2689655172413793,
      "penetration_over_2mm": 88,
      "penetration_over_2cm": 0,
      "max_ground_contact_depth_m": 0.019992022955008532,
      "near_base_height_50cm_fraction": 0.0
    },
    {
      "clip": "rpo_36_11",
      "source": "official_rpo",
      "valid_frames": 1159,
      "sole_min_height_m": -0.019958059953122363,
      "both_feet_airborne_fraction": 0.6721311475409836,
      "penetration_over_2mm": 293,
      "penetration_over_2cm": 0,
      "max_ground_contact_depth_m": 0.019958059953122363,
      "near_base_height_50cm_fraction": 0.0
    },
    {
      "clip": "rpo_B10_-__Walk_turn_left_45_stageii",
      "source": "official_rpo",
      "valid_frames": 21,
      "sole_min_height_m": -0.01827821432950359,
      "both_feet_airborne_fraction": 0.0,
      "penetration_over_2mm": 5,
      "penetration_over_2cm": 0,
      "max_ground_contact_depth_m": 0.018278214329503592,
      "near_base_height_50cm_fraction": 0.0
    },
    {
      "clip": "rpo_B11_-__Walk_turn_left_135_stageii",
      "source": "official_rpo",
      "valid_frames": 45,
      "sole_min_height_m": -0.018085388081173954,
      "both_feet_airborne_fraction": 0.0,
      "penetration_over_2mm": 45,
      "penetration_over_2cm": 0,
      "max_ground_contact_depth_m": 0.018085388081173954,
      "near_base_height_50cm_fraction": 0.0
    },
    {
      "clip": "rpo_B13_-__Walk_turn_right_90_stageii",
      "source": "official_rpo",
      "valid_frames": 253,
      "sole_min_height_m": -0.019998698286196763,
      "both_feet_airborne_fraction": 0.0,
      "penetration_over_2mm": 56,
      "penetration_over_2cm": 0,
      "max_ground_contact_depth_m": 0.019998698286196763,
      "near_base_height_50cm_fraction": 0.0
    },
    {
      "clip": "rpo_B14_-__Walk_turn_right_45_t2_stageii",
      "source": "official_rpo",
      "valid_frames": 191,
      "sole_min_height_m": -0.019784333708264122,
      "both_feet_airborne_fraction": 0.0,
      "penetration_over_2mm": 155,
      "penetration_over_2cm": 0,
      "max_ground_contact_depth_m": 0.019784333708264122,
      "near_base_height_50cm_fraction": 0.0
    },
    {
      "clip": "rpo_B15_-__Walk_turn_around_stageii",
      "source": "official_rpo",
      "valid_frames": 33,
      "sole_min_height_m": -0.019951321758314086,
      "both_feet_airborne_fraction": 0.0,
      "penetration_over_2mm": 33,
      "penetration_over_2cm": 0,
      "max_ground_contact_depth_m": 0.019951321758314086,
      "near_base_height_50cm_fraction": 0.0
    },
    {
      "clip": "rpo_B4_-_Stand_to_Walk_backwards_stageii",
      "source": "official_rpo",
      "valid_frames": 21,
      "sole_min_height_m": -0.01996637587289593,
      "both_feet_airborne_fraction": 0.0,
      "penetration_over_2mm": 21,
      "penetration_over_2cm": 0,
      "max_ground_contact_depth_m": 0.01996637587289593,
      "near_base_height_50cm_fraction": 0.0
    },
    {
      "clip": "rpo_B5_-__Walk_backwards_stageii",
      "source": "official_rpo",
      "valid_frames": 202,
      "sole_min_height_m": -0.01980899326651838,
      "both_feet_airborne_fraction": 0.0,
      "penetration_over_2mm": 144,
      "penetration_over_2cm": 0,
      "max_ground_contact_depth_m": 0.01980899326651838,
      "near_base_height_50cm_fraction": 0.0
    },
    {
      "clip": "rpo_B9_-__Walk_turn_left_90_stageii",
      "source": "official_rpo",
      "valid_frames": 16,
      "sole_min_height_m": -0.019863652487405417,
      "both_feet_airborne_fraction": 0.0,
      "penetration_over_2mm": 16,
      "penetration_over_2cm": 0,
      "max_ground_contact_depth_m": 0.019863652487405417,
      "near_base_height_50cm_fraction": 0.0
    },
    {
      "clip": "rpo_C12_-_run_turn_left_45_stageii",
      "source": "official_rpo",
      "valid_frames": 66,
      "sole_min_height_m": -0.019853686109623386,
      "both_feet_airborne_fraction": 0.10606060606060606,
      "penetration_over_2mm": 31,
      "penetration_over_2cm": 0,
      "max_ground_contact_depth_m": 0.019853686109623386,
      "near_base_height_50cm_fraction": 0.0
    },
    {
      "clip": "rpo_C17_-_run_change_direction_stageii",
      "source": "official_rpo",
      "valid_frames": 117,
      "sole_min_height_m": -0.01930529655223676,
      "both_feet_airborne_fraction": 0.1282051282051282,
      "penetration_over_2mm": 28,
      "penetration_over_2cm": 0,
      "max_ground_contact_depth_m": 0.01930529655223676,
      "near_base_height_50cm_fraction": 0.0
    },
    {
      "clip": "rpo_C1_-_stand_to_run_stageii",
      "source": "official_rpo",
      "valid_frames": 169,
      "sole_min_height_m": -0.019625430255605435,
      "both_feet_airborne_fraction": 0.07692307692307693,
      "penetration_over_2mm": 121,
      "penetration_over_2cm": 0,
      "max_ground_contact_depth_m": 0.019625430255605435,
      "near_base_height_50cm_fraction": 0.0
    },
    {
      "clip": "rpo_C3_-_run_stageii",
      "source": "official_rpo",
      "valid_frames": 98,
      "sole_min_height_m": -0.01961307176773797,
      "both_feet_airborne_fraction": 0.24489795918367346,
      "penetration_over_2mm": 21,
      "penetration_over_2cm": 0,
      "max_ground_contact_depth_m": 0.01961307176773797,
      "near_base_height_50cm_fraction": 0.0
    },
    {
      "clip": "rpo_C4_-_run_to_walk_a_stageii",
      "source": "official_rpo",
      "valid_frames": 120,
      "sole_min_height_m": -0.019968210963504762,
      "both_feet_airborne_fraction": 0.016666666666666666,
      "penetration_over_2mm": 53,
      "penetration_over_2cm": 0,
      "max_ground_contact_depth_m": 0.019968210963504762,
      "near_base_height_50cm_fraction": 0.0
    },
    {
      "clip": "rpo_C5_-_walk_to_run_stageii",
      "source": "official_rpo",
      "valid_frames": 119,
      "sole_min_height_m": -0.01994574637611489,
      "both_feet_airborne_fraction": 0.15966386554621848,
      "penetration_over_2mm": 46,
      "penetration_over_2cm": 0,
      "max_ground_contact_depth_m": 0.01994574637611489,
      "near_base_height_50cm_fraction": 0.0
    },
    {
      "clip": "rpo_move_back",
      "source": "official_rpo",
      "valid_frames": 312,
      "sole_min_height_m": -0.0018758968507121554,
      "both_feet_airborne_fraction": 0.0,
      "penetration_over_2mm": 0,
      "penetration_over_2cm": 0,
      "max_ground_contact_depth_m": 0.001875896850712152,
      "near_base_height_50cm_fraction": 0.0
    },
    {
      "clip": "rpo_move_l",
      "source": "official_rpo",
      "valid_frames": 280,
      "sole_min_height_m": 0.0016444705151495848,
      "both_feet_airborne_fraction": 0.0,
      "penetration_over_2mm": 0,
      "penetration_over_2cm": 0,
      "max_ground_contact_depth_m": 0.0,
      "near_base_height_50cm_fraction": 0.0
    },
    {
      "clip": "rpo_move_r",
      "source": "official_rpo",
      "valid_frames": 250,
      "sole_min_height_m": -0.0029470611600756584,
      "both_feet_airborne_fraction": 0.0,
      "penetration_over_2mm": 4,
      "penetration_over_2cm": 0,
      "max_ground_contact_depth_m": 0.0029470611600756584,
      "near_base_height_50cm_fraction": 0.0
    },
    {
      "clip": "rpo_run_start_180_R_001__A345",
      "source": "official_rpo",
      "valid_frames": 143,
      "sole_min_height_m": 0.01631435556028369,
      "both_feet_airborne_fraction": 0.09090909090909091,
      "penetration_over_2mm": 0,
      "penetration_over_2cm": 0,
      "max_ground_contact_depth_m": 0.0,
      "near_base_height_50cm_fraction": 0.0
    },
    {
      "clip": "rpo_run_start_180_R_001__A345_M",
      "source": "official_rpo",
      "valid_frames": 145,
      "sole_min_height_m": 0.01630484400335747,
      "both_feet_airborne_fraction": 0.09655172413793103,
      "penetration_over_2mm": 0,
      "penetration_over_2cm": 0,
      "max_ground_contact_depth_m": 0.0,
      "near_base_height_50cm_fraction": 0.0
    },
    {
      "clip": "rpo_run_stop_180_R_001__A345",
      "source": "official_rpo",
      "valid_frames": 190,
      "sole_min_height_m": 0.005743720280847096,
      "both_feet_airborne_fraction": 0.031578947368421054,
      "penetration_over_2mm": 0,
      "penetration_over_2cm": 0,
      "max_ground_contact_depth_m": 0.0,
      "near_base_height_50cm_fraction": 0.0
    },
    {
      "clip": "rpo_run_stop_180_R_001__A345_M",
      "source": "official_rpo",
      "valid_frames": 190,
      "sole_min_height_m": 0.005735622190172582,
      "both_feet_airborne_fraction": 0.031578947368421054,
      "penetration_over_2mm": 0,
      "penetration_over_2cm": 0,
      "max_ground_contact_depth_m": 0.0,
      "near_base_height_50cm_fraction": 0.0
    },
    {
      "clip": "rpo_turn_l",
      "source": "official_rpo",
      "valid_frames": 340,
      "sole_min_height_m": -0.0036675567345574485,
      "both_feet_airborne_fraction": 0.0,
      "penetration_over_2mm": 29,
      "penetration_over_2cm": 0,
      "max_ground_contact_depth_m": 0.003667556734557452,
      "near_base_height_50cm_fraction": 0.0
    },
    {
      "clip": "rpo_turn_r",
      "source": "official_rpo",
      "valid_frames": 239,
      "sole_min_height_m": -0.0032224578203292435,
      "both_feet_airborne_fraction": 0.0,
      "penetration_over_2mm": 8,
      "penetration_over_2cm": 0,
      "max_ground_contact_depth_m": 0.00322245782032924,
      "near_base_height_50cm_fraction": 0.0
    }
  ],
  "reset_replay_sample": {
    "samples": 1024,
    "sole_height_quantiles_m": {
      "0": -0.019945745538542583,
      "0.05": -0.009782728515164143,
      "0.5": 0.01997297328844717,
      "0.95": 0.3042136888792717,
      "1": 0.4557211641739127
    },
    "penetration_over_2mm": 84,
    "penetration_over_2cm": 0,
    "both_feet_airborne_over_5cm": 240,
    "max_ground_depth_m": 0.019945745538542583
  }
}
```

### 接触候选池逐clip覆盖

```json
{
  "source_candidates": {
    "official_rpo": 2141,
    "g1": 2049
  },
  "clips": [
    {
      "clip": "g1_dance1_subject1_clip3",
      "source": "g1",
      "eligible": 141,
      "original_starts": 320,
      "max_root_lift_m": 0.0010939905187115073
    },
    {
      "clip": "g1_dance1_subject2_clip3",
      "source": "g1",
      "eligible": 8,
      "original_starts": 207,
      "max_root_lift_m": 0.0
    },
    {
      "clip": "g1_dance1_subject3_clip3",
      "source": "g1",
      "eligible": 11,
      "original_starts": 157,
      "max_root_lift_m": 0.0
    },
    {
      "clip": "g1_fallAndGetUp1_subject1_clip3",
      "source": "g1",
      "eligible": 123,
      "original_starts": 188,
      "max_root_lift_m": 0.0019999828655272722
    },
    {
      "clip": "g1_fallAndGetUp1_subject4_clip3",
      "source": "g1",
      "eligible": 0,
      "original_starts": 153,
      "max_root_lift_m": null
    },
    {
      "clip": "g1_fallAndGetUp1_subject5_clip3",
      "source": "g1",
      "eligible": 8,
      "original_starts": 130,
      "max_root_lift_m": 0.0
    },
    {
      "clip": "g1_fight1_subject2_clip3",
      "source": "g1",
      "eligible": 8,
      "original_starts": 340,
      "max_root_lift_m": 0.0020000170916318893
    },
    {
      "clip": "g1_fight1_subject3_clip3",
      "source": "g1",
      "eligible": 0,
      "original_starts": 337,
      "max_root_lift_m": null
    },
    {
      "clip": "g1_fight1_subject5_clip3",
      "source": "g1",
      "eligible": 16,
      "original_starts": 376,
      "max_root_lift_m": 0.002000010572373867
    },
    {
      "clip": "g1_jumps1_subject1_clip3",
      "source": "g1",
      "eligible": 41,
      "original_starts": 313,
      "max_root_lift_m": 0.0
    },
    {
      "clip": "g1_jumps1_subject2_clip3",
      "source": "g1",
      "eligible": 8,
      "original_starts": 358,
      "max_root_lift_m": 0.0019999647047370672
    },
    {
      "clip": "g1_jumps1_subject5_clip3",
      "source": "g1",
      "eligible": 6,
      "original_starts": 344,
      "max_root_lift_m": 0.0
    },
    {
      "clip": "g1_run1_subject2_clip3",
      "source": "g1",
      "eligible": 27,
      "original_starts": 344,
      "max_root_lift_m": 0.0
    },
    {
      "clip": "g1_run1_subject5_clip3",
      "source": "g1",
      "eligible": 1,
      "original_starts": 127,
      "max_root_lift_m": 0.0
    },
    {
      "clip": "g1_sprint1_subject2_clip3",
      "source": "g1",
      "eligible": 168,
      "original_starts": 306,
      "max_root_lift_m": 0.001999995205551386
    },
    {
      "clip": "g1_sprint1_subject4_clip3",
      "source": "g1",
      "eligible": 172,
      "original_starts": 379,
      "max_root_lift_m": 0.0019999679643660784
    },
    {
      "clip": "g1_walk1_subject1_clip3",
      "source": "g1",
      "eligible": 173,
      "original_starts": 392,
      "max_root_lift_m": 0.002000028733164072
    },
    {
      "clip": "g1_walk1_subject2_clip3",
      "source": "g1",
      "eligible": 380,
      "original_starts": 392,
      "max_root_lift_m": 0.002000015927478671
    },
    {
      "clip": "g1_walk1_subject5_clip3",
      "source": "g1",
      "eligible": 129,
      "original_starts": 392,
      "max_root_lift_m": 0.0018110821256414056
    },
    {
      "clip": "g1_walk2_subject1_clip3",
      "source": "g1",
      "eligible": 206,
      "original_starts": 392,
      "max_root_lift_m": 0.0007657583919353783
    },
    {
      "clip": "g1_walk2_subject3_clip3",
      "source": "g1",
      "eligible": 153,
      "original_starts": 392,
      "max_root_lift_m": 0.0
    },
    {
      "clip": "g1_walk2_subject4_clip3",
      "source": "g1",
      "eligible": 270,
      "original_starts": 377,
      "max_root_lift_m": 0.0019999807700514793
    },
    {
      "clip": "rpo_114_08",
      "source": "official_rpo",
      "eligible": 23,
      "original_starts": 435,
      "max_root_lift_m": 0.021217873319983482
    },
    {
      "clip": "rpo_114_09",
      "source": "official_rpo",
      "eligible": 94,
      "original_starts": 774,
      "max_root_lift_m": 0.0
    },
    {
      "clip": "rpo_127_04",
      "source": "official_rpo",
      "eligible": 0,
      "original_starts": 1,
      "max_root_lift_m": null
    },
    {
      "clip": "rpo_36_01",
      "source": "official_rpo",
      "eligible": 63,
      "original_starts": 113,
      "max_root_lift_m": 0.021992022171616554
    },
    {
      "clip": "rpo_36_11",
      "source": "official_rpo",
      "eligible": 185,
      "original_starts": 1066,
      "max_root_lift_m": 0.0219456497579813
    },
    {
      "clip": "rpo_B10_-__Walk_turn_left_45_stageii",
      "source": "official_rpo",
      "eligible": 13,
      "original_starts": 13,
      "max_root_lift_m": 0.020278213545680046
    },
    {
      "clip": "rpo_B11_-__Walk_turn_left_135_stageii",
      "source": "official_rpo",
      "eligible": 14,
      "original_starts": 14,
      "max_root_lift_m": 0.02008538879454136
    },
    {
      "clip": "rpo_B13_-__Walk_turn_right_90_stageii",
      "source": "official_rpo",
      "eligible": 209,
      "original_starts": 229,
      "max_root_lift_m": 0.021529648452997208
    },
    {
      "clip": "rpo_B14_-__Walk_turn_right_45_t2_stageii",
      "source": "official_rpo",
      "eligible": 142,
      "original_starts": 143,
      "max_root_lift_m": 0.021784333512187004
    },
    {
      "clip": "rpo_B15_-__Walk_turn_around_stageii",
      "source": "official_rpo",
      "eligible": 7,
      "original_starts": 7,
      "max_root_lift_m": 0.02195132151246071
    },
    {
      "clip": "rpo_B4_-_Stand_to_Walk_backwards_stageii",
      "source": "official_rpo",
      "eligible": 1,
      "original_starts": 1,
      "max_root_lift_m": 0.01668175309896469
    },
    {
      "clip": "rpo_B5_-__Walk_backwards_stageii",
      "source": "official_rpo",
      "eligible": 101,
      "original_starts": 108,
      "max_root_lift_m": 0.021808993071317673
    },
    {
      "clip": "rpo_B9_-__Walk_turn_left_90_stageii",
      "source": "official_rpo",
      "eligible": 7,
      "original_starts": 7,
      "max_root_lift_m": 0.02166387438774109
    },
    {
      "clip": "rpo_C12_-_run_turn_left_45_stageii",
      "source": "official_rpo",
      "eligible": 2,
      "original_starts": 11,
      "max_root_lift_m": 0.004126798361539841
    },
    {
      "clip": "rpo_C17_-_run_change_direction_stageii",
      "source": "official_rpo",
      "eligible": 13,
      "original_starts": 67,
      "max_root_lift_m": 0.02094338834285736
    },
    {
      "clip": "rpo_C1_-_stand_to_run_stageii",
      "source": "official_rpo",
      "eligible": 107,
      "original_starts": 136,
      "max_root_lift_m": 0.021625429391860962
    },
    {
      "clip": "rpo_C3_-_run_stageii",
      "source": "official_rpo",
      "eligible": 6,
      "original_starts": 36,
      "max_root_lift_m": 0.01881994679570198
    },
    {
      "clip": "rpo_C4_-_run_to_walk_a_stageii",
      "source": "official_rpo",
      "eligible": 48,
      "original_starts": 72,
      "max_root_lift_m": 0.02129398100078106
    },
    {
      "clip": "rpo_C5_-_walk_to_run_stageii",
      "source": "official_rpo",
      "eligible": 25,
      "original_starts": 47,
      "max_root_lift_m": 0.02194574661552906
    },
    {
      "clip": "rpo_move_back",
      "source": "official_rpo",
      "eligible": 298,
      "original_starts": 304,
      "max_root_lift_m": 0.0038758968003094196
    },
    {
      "clip": "rpo_move_l",
      "source": "official_rpo",
      "eligible": 111,
      "original_starts": 272,
      "max_root_lift_m": 0.00035552948247641325
    },
    {
      "clip": "rpo_move_r",
      "source": "official_rpo",
      "eligible": 128,
      "original_starts": 242,
      "max_root_lift_m": 0.004947061184793711
    },
    {
      "clip": "rpo_run_start_180_R_001__A345",
      "source": "official_rpo",
      "eligible": 5,
      "original_starts": 114,
      "max_root_lift_m": 0.0
    },
    {
      "clip": "rpo_run_start_180_R_001__A345_M",
      "source": "official_rpo",
      "eligible": 6,
      "original_starts": 115,
      "max_root_lift_m": 0.0
    },
    {
      "clip": "rpo_run_stop_180_R_001__A345",
      "source": "official_rpo",
      "eligible": 112,
      "original_starts": 158,
      "max_root_lift_m": 0.0
    },
    {
      "clip": "rpo_run_stop_180_R_001__A345_M",
      "source": "official_rpo",
      "eligible": 112,
      "original_starts": 158,
      "max_root_lift_m": 0.0
    },
    {
      "clip": "rpo_turn_l",
      "source": "official_rpo",
      "eligible": 142,
      "original_starts": 332,
      "max_root_lift_m": 0.004752741195261478
    },
    {
      "clip": "rpo_turn_r",
      "source": "official_rpo",
      "eligible": 167,
      "original_starts": 231,
      "max_root_lift_m": 0.005222457926720381
    }
  ]
}
```

## 固定训练配方与完整任务身份


H recipe=rpo_bfm_ufo_agent_rlinf_isaac_4gpu_contact_v8；profile=rpo_bfm_ufo_isaac_4gpu_rsi_contact_v8；Gym=RPO-UFO-Isaac-RSI-contact-v8；interface=ee_rpo_v2_bfm_23dof_v2。数据ds_fb0ecda9e70644f2afdd，manifest SHA256=dc0c6585887507df59f675963b2733d25c9836d805e9bb89f2402398f07bd288；codec SHA256=dce25d554ceb7fa69287515946d2f5cd9d41d14f1b1ef38715b819d7e962b6ef。asset SHA256=b58868a782d8863ac38059eb4eab89d27ee1d70565bf58f5a055f44510c8e842。

冻结镜像sha256:288c9bc690f0b0cefce4529b9ae07d3d0c9163b201d08d228161b7818685b5e2，tag=eb/rlinf:rpo-bfm-ufo-agent-4gpu-contact-v8-poststep1；owned源码聚合sha256:46f51b0dc291a70d002afeef2f198eda2133c137b5b8658cb4996d05a241760d。原生UFO优化主体沿用G稳定化版本，未在本轮再改损失。原始70段NPZ字节与来源隔离沿用已核验v2数据，只有RSI候选池改变。

正式训练前额外复核发现首个_apply_action尚未推进物理就锁存上一帧接触力。Isaac传感器reset清零后标记outdated，下次data访问会重读PhysX上一物理帧；因此旧力存在跨RSI进入本步锁存的风险。修复为每控制间隔首_apply_action不采样，随后采样完成的1–7物理帧，_get_dones再采样第8帧。仍完整覆盖8帧，未降低终止阈值。诊断协议派生v3，同样排除区间起点旧力。

第一版H的诊断及16→24资格均成功，但未继续用于正式500段；修正时序后冻结新镜像，重新从头完成资格及恢复，旧镜像与已登记模型未改动。旧链Job/Run与诊断证据如下，不能与新链累计transition混计：

### 首版H保留证据 contact-probe-run-current.json

```json
{
  "id": "run_a1ce3855d3894e97a17b",
  "job_id": "job_92a915e3700d497cbe38",
  "attempt": 1,
  "status": "finished",
  "abort_reason": null,
  "metrics_summary": {
    "contact_probe": {
      "files": [
        {
          "name": "paired-inputs.npz",
          "bytes": 103660,
          "sha256": "f4e3681bde6c358a92465e1d2ba5d10f958a7abe9323f78578ae4dd48db3e3fc"
        },
        {
          "name": "baseline.npz",
          "bytes": 2988804,
          "sha256": "2d81133200e2432c62df4f22834561b8f3411b278c15c2159e112fd8034f903c"
        },
        {
          "name": "grounded.npz",
          "bytes": 3329345,
          "sha256": "74be172710267ada0a725ad29ce2cf43d8b483783aadbc89a3ac7edf7b1b6913"
        },
        {
          "name": "substeps.npz",
          "bytes": 3331024,
          "sha256": "e765df75ba0a64c638b7d5713482d774b2e462cd0d21a0982065e1133455b41e"
        }
      ],
      "phases": [
        {
          "phase": "baseline",
          "decimation": 4,
          "physics_dt_s": 0.005,
          "max_dq_active": 29.440061569213867,
          "fk_after_max_m": 0.0009962049929651041,
          "initial_states": 512,
          "fk_before_max_m": 0.0009962049929651041,
          "first_hard_flags": [
            0,
            0,
            0
          ],
          "first_terminated": 0,
          "max_boundary_dq_active": 29.440061569213867,
          "max_foot_force_active_N": 3794.49853515625,
          "feet_controller_damping_range": [
            1.8031455278396606,
            2.1973743438720703
          ],
          "feet_controller_stiffness_range": [
            36.01011657714844,
            43.87909698486328
          ]
        },
        {
          "phase": "grounded",
          "decimation": 4,
          "physics_dt_s": 0.005,
          "max_dq_active": 24.479267120361328,
          "fk_after_max_m": 0.0009962049929651041,
          "initial_states": 512,
          "fk_before_max_m": 0.0009962049929651041,
          "first_hard_flags": [
            0,
            0,
            0
          ],
          "first_terminated": 0,
          "max_boundary_dq_active": 19.56183624267578,
          "max_foot_force_active_N": 3794.49853515625,
          "feet_controller_damping_range": [
            1.8031455278396606,
            2.1973743438720703
          ],
          "feet_controller_stiffness_range": [
            36.01011657714844,
            43.87909698486328
          ]
        },
        {
          "phase": "substeps",
          "decimation": 8,
          "physics_dt_s": 0.0025,
          "max_dq_active": 22.179285049438477,
          "fk_after_max_m": 0.0009962049929651041,
          "initial_states": 512,
          "fk_before_max_m": 0.0009962049929651041,
          "first_hard_flags": [
            0,
            0,
            0
          ],
          "first_terminated": 0,
          "max_boundary_dq_active": 16.885051727294922,
          "max_foot_force_active_N": 7688.2119140625,
          "feet_controller_damping_range": [
            1.8031455278396606,
            2.1973743438720703
          ],
          "feet_controller_stiffness_range": [
            36.01011657714844,
            43.87909698486328
          ]
        }
      ],
      "status": "measured",
      "protocol": {
        "envs": 32,
        "seed": 64042,
        "scope": "contact-eligible reference starts; diagnostic only, no learner update or acceptance",
        "split": "train",
        "steps": 16,
        "phases": [
          "baseline",
          "grounded",
          "substeps"
        ],
        "rounds": 16,
        "pairing": "same reference state, requested action sequence and startup seed; solver warm start unverified",
        "schema_version": "eb/ufo-contact-probe/v2"
      },
      "asset_sha256": "b58868a782d8863ac38059eb4eab89d27ee1d70565bf58f5a055f44510c8e842",
      "manifest_sha256": "dc0c6585887507df59f675963b2733d25c9836d805e9bb89f2402398f07bd288"
    }
  },
  "gpu_ids": [
    0
  ],
  "provenance": "{\"code_ref\": null, \"config_hash\": \"d940190e21b4650f3b8b41efdbfc1610aa5ab60fb739c6b1f537cf306de6f11a\", \"data_version\": null, \"image_digest\": \"sha256:1b449f9b5194e1dc2e999e1f762a373dcffb88f4be3c435b36c5337facc16a52\", \"seed\": null}",
  "artifacts": []
}
```

### 首版H保留证据 contact16-run-current.json

```json
{
  "id": "run_6fab081aacc04b11bee4",
  "job_id": "job_0f8df98d897e4a709a1b",
  "attempt": 1,
  "status": "finished",
  "abort_reason": null,
  "metrics_summary": {
    "checkpoint": {
      "path": "/data/kk/models/runs/run_6fab081aacc04b11bee4",
      "step": 16,
      "bytes": 7211132484
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
    "model_version_id": "mv_a792742ed78f418dbdba"
  },
  "gpu_ids": [
    0,
    1,
    2,
    3
  ],
  "provenance": "{\"code_ref\": \"sha256:04e20ece2e3e3f3603a98b24e04aba42023fb31a7395beddfdac2a5d522de54d\", \"config_hash\": \"021fb69d34ae23cd573766a2f6dec61c0c16c1fef0b1b999753c927aa0214e4e\", \"data_version\": null, \"image_digest\": \"sha256:1b449f9b5194e1dc2e999e1f762a373dcffb88f4be3c435b36c5337facc16a52\", \"seed\": null}",
  "artifacts": [
    {
      "id": "art_f98c69463fbf4e4cbafe",
      "kind": "rlinf_checkpoint",
      "sha256": "sha256:4c54c8199351360e2d5fcd4a7726377713fe738d182eb5db3e67e9af1cd9beab",
      "size_bytes": 7211132484,
      "storage": "/data/kk/models/runs/run_6fab081aacc04b11bee4"
    }
  ]
}
```

### 首版H保留证据 contact24-run-current.json

```json
{
  "id": "run_0c2d2f704e1a433db877",
  "job_id": "job_deea57d036d6468daa13",
  "attempt": 1,
  "status": "finished",
  "abort_reason": null,
  "metrics_summary": {
    "checkpoint": {
      "path": "/data/kk/models/runs/run_0c2d2f704e1a433db877",
      "step": 24,
      "bytes": 7384201126
    },
    "sampler_sync": {
      "identity": {
        "version": 16,
        "state_sha256": "197daa39f58a26baf8a6225bb69a02dae69cb755d31eb654481c598f7bc9d7ea",
        "online_transitions": 49152
      },
      "samplers": [
        {
          "rank": 0,
          "version": 16,
          "state_sha256": "197daa39f58a26baf8a6225bb69a02dae69cb755d31eb654481c598f7bc9d7ea",
          "online_transitions": 49152
        },
        {
          "rank": 1,
          "version": 16,
          "state_sha256": "197daa39f58a26baf8a6225bb69a02dae69cb755d31eb654481c598f7bc9d7ea",
          "online_transitions": 49152
        },
        {
          "rank": 2,
          "version": 16,
          "state_sha256": "197daa39f58a26baf8a6225bb69a02dae69cb755d31eb654481c598f7bc9d7ea",
          "online_transitions": 49152
        }
      ],
      "schema_version": "eb/ufo-sampler-sync/v1",
      "sampling_rounds": 2
    },
    "model_version_id": "mv_982d3273e1d4491e955a"
  },
  "gpu_ids": [
    0,
    1,
    2,
    3
  ],
  "provenance": "{\"code_ref\": \"sha256:04e20ece2e3e3f3603a98b24e04aba42023fb31a7395beddfdac2a5d522de54d\", \"config_hash\": \"f0dd19f7626d05787fea5bf3c8f230116079205424bd76e8bea2b431973951eb\", \"data_version\": null, \"image_digest\": \"sha256:1b449f9b5194e1dc2e999e1f762a373dcffb88f4be3c435b36c5337facc16a52\", \"seed\": null}",
  "artifacts": [
    {
      "id": "art_9305fd76a4124a72a199",
      "kind": "rlinf_checkpoint",
      "sha256": "sha256:569f9896e120102d14f2da11b44c207daa91347013708a3483145f950b0be1a2",
      "size_bytes": 7384201126,
      "storage": "/data/kk/models/runs/run_0c2d2f704e1a433db877"
    }
  ]
}
```

### 首版H保留证据 contact-probe-result.json

```json
{
  "protocol": {
    "schema_version": "eb/ufo-contact-probe/v2",
    "phases": [
      "baseline",
      "grounded",
      "substeps"
    ],
    "envs": 32,
    "rounds": 16,
    "steps": 16,
    "seed": 64042,
    "split": "train",
    "pairing": "same reference state, requested action sequence and startup seed; solver warm start unverified",
    "scope": "contact-eligible reference starts; diagnostic only, no learner update or acceptance"
  },
  "status": "measured",
  "asset_sha256": "b58868a782d8863ac38059eb4eab89d27ee1d70565bf58f5a055f44510c8e842",
  "manifest_sha256": "dc0c6585887507df59f675963b2733d25c9836d805e9bb89f2402398f07bd288",
  "files": [
    {
      "name": "paired-inputs.npz",
      "bytes": 103660,
      "sha256": "f4e3681bde6c358a92465e1d2ba5d10f958a7abe9323f78578ae4dd48db3e3fc"
    },
    {
      "name": "baseline.npz",
      "bytes": 2988804,
      "sha256": "2d81133200e2432c62df4f22834561b8f3411b278c15c2159e112fd8034f903c"
    },
    {
      "name": "grounded.npz",
      "bytes": 3329345,
      "sha256": "74be172710267ada0a725ad29ce2cf43d8b483783aadbc89a3ac7edf7b1b6913"
    },
    {
      "name": "substeps.npz",
      "bytes": 3331024,
      "sha256": "e765df75ba0a64c638b7d5713482d774b2e462cd0d21a0982065e1133455b41e"
    }
  ],
  "phases": [
    {
      "phase": "baseline",
      "physics_dt_s": 0.005,
      "decimation": 4,
      "first_terminated": 0,
      "initial_states": 512,
      "max_dq_active": 29.440061569213867,
      "max_boundary_dq_active": 29.440061569213867,
      "max_foot_force_active_N": 3794.49853515625,
      "fk_before_max_m": 0.0009962049929651041,
      "fk_after_max_m": 0.0009962049929651041,
      "first_hard_flags": [
        0,
        0,
        0
      ],
      "feet_controller_stiffness_range": [
        36.01011657714844,
        43.87909698486328
      ],
      "feet_controller_damping_range": [
        1.8031455278396606,
        2.1973743438720703
      ]
    },
    {
      "phase": "grounded",
      "physics_dt_s": 0.005,
      "decimation": 4,
      "first_terminated": 0,
      "initial_states": 512,
      "max_dq_active": 24.479267120361328,
      "max_boundary_dq_active": 19.56183624267578,
      "max_foot_force_active_N": 3794.49853515625,
      "fk_before_max_m": 0.0009962049929651041,
      "fk_after_max_m": 0.0009962049929651041,
      "first_hard_flags": [
        0,
        0,
        0
      ],
      "feet_controller_stiffness_range": [
        36.01011657714844,
        43.87909698486328
      ],
      "feet_controller_damping_range": [
        1.8031455278396606,
        2.1973743438720703
      ]
    },
    {
      "phase": "substeps",
      "physics_dt_s": 0.0025,
      "decimation": 8,
      "first_terminated": 0,
      "initial_states": 512,
      "max_dq_active": 22.179285049438477,
      "max_boundary_dq_active": 16.885051727294922,
      "max_foot_force_active_N": 7688.2119140625,
      "fk_before_max_m": 0.0009962049929651041,
      "fk_after_max_m": 0.0009962049929651041,
      "first_hard_flags": [
        0,
        0,
        0
      ],
      "feet_controller_stiffness_range": [
        36.01011657714844,
        43.87909698486328
      ],
      "feet_controller_damping_range": [
        1.8031455278396606,
        2.1973743438720703
      ]
    }
  ]
}
```

网络435116484参数：F/Actor/Critic/辅助Critic为2048宽×6层，D为1024×3，B为256×1，latent256；Actor输入375、Actor FB输入733、动作23。batch2048，CPU replay容量100万；每采样round=6144环境×4步=24576转移，每触发16联合更新，warmup10240、sample_interval24576。实际累计计数以checkpoint为准。

沿用G：Actor梯度L2裁剪1，其余五优化器5；记录裁剪前后范数并拒绝非有限梯度。native clip_grad_norm=0表示由平台注册hook执行裁剪。Actor detach(mean(abs(Q_fb)))上界50；D学习率3e-6、reward截[-2,2]同时记录raw；第九速度raw=sum(relu(abs(dq)/Vmax−0.70)^2)，权重−25仅加权一次，再由RewardNormalizer归一化。第九项仍用控制步末dq，因此可能低估子步瞬时越限，物理峰值与锁存终止单列。

tracking=False；同episode非周期相邻步z应不被其他环境reset改写，仍允许每环境100步正常latent更新。真实terminated使TD bootstrap为0、truncated保留。env/reward、env/return、env/success_once为占位零，不用于推导训练或验收质量。TrainJob.kind=ppo为通用历史元数据，实际算法为bfm_fb_cpr_aux/provider=ufo，没有运行PPO。

环境初始化日志列出的29项RewardManager属于底层RPO默认配置；UFO Physics._get_rewards覆盖为零，未把那些standing/tracking/termination项送入原生UFO优化。实际辅助项是具名9项raw aux，FB与D奖励由原生agent生成；不能据29项日志声称已启用站立回报或−200终止罚。

资格16→24每段最多1h；新增500更新最多2h；零自动重试、64GiB新证据/模型预算，不自动长训。训练门槛：有限标量、六范数受限、RSI盒外<10%、末100窗episode优于首窗且硬速度率下降；独立行走评估另需episode≥100持续稳定。用户要求训练未达标不独立评估，H evaluator也未赋予qualification。

### 16更新完整TrainJob

```json
{
  "id": "trn_f16e6b774b0b4a0b8f19",
  "run_id": "run_7e3d481401804da2b688",
  "kind": "ppo",
  "name": "RPO UFO contact v8 poststep1 native 16-update qualification",
  "model_name": "RPO-UFO-contact-v8-poststep1-step16",
  "status": "succeeded",
  "template_id": "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_contact_v8",
  "image_digest": "sha256:288c9bc690f0b0cefce4529b9ae07d3d0c9163b201d08d228161b7818685b5e2",
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
  "ray_port": 6408,
  "params": {
    "stack": "rlinf",
    "purpose": "qualification",
    "execution": {
      "recipe_id": "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_contact_v8",
      "runtime_id": "rlinf",
      "provider_id": "ufo",
      "topology_id": "one_learner_three_samplers",
      "algorithm_id": "bfm_fb_cpr_aux",
      "simulator_id": "isaaclab",
      "progress_unit": "learner_joint_update",
      "schema_version": "eb/training-binding/v1",
      "checkpoint_format": "ufo_agent_state_v1"
    },
    "model_name": "RPO-UFO-contact-v8-poststep1-step16",
    "config_name": "rpo_bfm_ufo_isaac",
    "template_id": "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_contact_v8",
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
      "contact_profile": "contact_stable_v8",
      "control_profile": "reachable_v2",
      "scaling_profile": "env6144_batch2048_v6",
      "auxiliary_profile": "soft_velocity_v7",
      "optimizer_profile": "stabilized_v7",
      "contact_patch_hash": "sha256:e20a0ff930b3b9430ece70c21ab316a06cef92b679b638c0a35d7fed62d6d814",
      "topology_patch_hash": "sha256:3078506628924464cb42d0b2f108b3fab502a43fd75e091f358d504565294d23",
      "auxiliary_source_sha256": "sha256:46f51b0dc291a70d002afeef2f198eda2133c137b5b8658cb4996d05a241760d",
      "stabilization_patch_hash": "sha256:564f5452e12e3179112e6aa3020efba4cbcedc0ad90dd26b98aed78a50dd85b1",
      "rollout_profile_patch_hash": "sha256:777456ce171734326cf8c55eaa434fc5a69a27d0b2d0b725608dd54c190f4a77",
      "scaling_profile_patch_hash": "sha256:58a2dca8377ed2ede804c7101b39bc6992ec669ef95e610ddec4b2a59bfc6b20",
      "learning_profile_patch_hash": "sha256:d47623e48b4cc757d5e63336a1fa7d49234c3833a2f1c6bbdbee4fe7ffd9a272",
      "action_control_source_sha256": "sha256:1b0bbc0a7a350581ef1939faf93a3e27d824bde62107ecfe4e9ff988bbc088fb",
      "auxiliary_profile_patch_hash": "sha256:30e7b4ea7a3ec0e45cdbe0a9dbf718eb219c2c6fd36d07f9b18bd5222183a03a",
      "action_control_profile_patch_hash": "sha256:548260ca5e3eedd53ae739c00672c7e273e7835faed734778039cf5977b56e0d"
    },
    "environment_profile": {
      "id": "rpo_bfm_ufo_isaac_4gpu_rsi_contact_v8",
      "hash": "sha256:23fe3b6f8cd49a749591cf7ab85a75f3918d6ea1c4150b34a16791e7864d3eff",
      "definition": {
        "task": "RPO v2有限URDF动作目标，原始q/dq与观察零点保持；6144环境，每个sampler2048；435M原生网络、batch2048、replay100万，每24576新转移触发16次联合更新。 RSI仅train接触候选、足底2mm间隙、有限root上移；写入后刷新FK，专家数据不改。",
        "reward": "九项raw aux，limits_dof_vel=sum relu(abs(dq)/Vmax-0.70)^2，权重-25仅加权一次；判别奖励截[-2,2]，D lr=3e-6；Actor正则尺度上限50。",
        "failure": "真实硬位置/速度/力矩终止保持；target slew每步0.5×Vmax；Actor裁剪1，其余五Adam裁剪5，非有限梯度失败。 显式PD增益保持，physics dt=2.5ms/decimation8/50Hz；物理子步硬限位锁存，接触/高度/倾角原因分列。",
        "success": "训练/完整恢复资格16→24后新增500联合更新，最多2小时且不自动重试；训练指标不足不启动独立评估。正常行走尚未验证。",
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
    "+actor.model.rollout_expert_trajectories=False",
    "actor.model.hidden_dim=2048",
    "actor.model.hidden_layers=6",
    "actor.model.batch_size=2048",
    "actor.micro_batch_size=2048",
    "actor.global_batch_size=2048",
    "actor.model.replay_capacity=1000000",
    "env.train.init_params.id=RPO-UFO-Isaac-RSI-contact-v8",
    "+actor.model.auxiliary_profile=soft_velocity_v7",
    "+actor.model.scaling_profile=env6144_batch2048_v6",
    "+actor.model.control_profile=reachable_v2",
    "+actor.model.optimizer_profile=stabilized_v7",
    "+actor.model.contact_profile=contact_stable_v8",
    "actor.seed=42",
    "~cluster.component_placement",
    "+cluster.component_placement={actor:0,env\\,\\ rollout:1-3}",
    "runner.logger.log_path=/workspace/output",
    "hydra.run.dir=/workspace/output/hydra",
    "runner.logger.experiment_name=\"RPO UFO contact v8 poststep1 native 16-update qualification\""
  ],
  "run_dir": "/data/kk/embodiedbein/runs/run_7e3d481401804da2b688",
  "auto_register_latest": false,
  "metrics_summary": {
    "checkpoint": {
      "path": "/data/kk/models/runs/run_7e3d481401804da2b688",
      "step": 16,
      "bytes": 7211132494
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
    "model_version_id": "mv_816fbf65953d4db8aa58"
  },
  "error_message": null,
  "job_id": "job_c51c89f88e8b47f8a337",
  "started_at": "2026-10-11T23:28:18.852516+00:00",
  "finished_at": "2026-10-11T23:32:12.767697+00:00",
  "created_at": "2026-10-11T23:28:09.226881+00:00"
}
```

### 24更新完整TrainJob

```json
{
  "id": "trn_2536bcfa9c7d469e98b0",
  "run_id": "run_c768a1338d1e46fe8092",
  "kind": "ppo",
  "name": "RPO UFO contact v8 poststep1 native 16 to24 restore qualification",
  "model_name": "RPO-UFO-contact-v8-poststep1-step24",
  "status": "succeeded",
  "template_id": "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_contact_v8",
  "image_digest": "sha256:288c9bc690f0b0cefce4529b9ae07d3d0c9163b201d08d228161b7818685b5e2",
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
    "purpose": "qualification",
    "execution": {
      "recipe_id": "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_contact_v8",
      "runtime_id": "rlinf",
      "provider_id": "ufo",
      "topology_id": "one_learner_three_samplers",
      "algorithm_id": "bfm_fb_cpr_aux",
      "simulator_id": "isaaclab",
      "progress_unit": "learner_joint_update",
      "schema_version": "eb/training-binding/v1",
      "checkpoint_format": "ufo_agent_state_v1"
    },
    "model_name": "RPO-UFO-contact-v8-poststep1-step24",
    "config_name": "rpo_bfm_ufo_isaac",
    "resume_mode": "trainer_state",
    "resume_step": 16,
    "template_id": "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_contact_v8",
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
      "contact_profile": "contact_stable_v8",
      "control_profile": "reachable_v2",
      "scaling_profile": "env6144_batch2048_v6",
      "auxiliary_profile": "soft_velocity_v7",
      "optimizer_profile": "stabilized_v7",
      "contact_patch_hash": "sha256:e20a0ff930b3b9430ece70c21ab316a06cef92b679b638c0a35d7fed62d6d814",
      "topology_patch_hash": "sha256:3078506628924464cb42d0b2f108b3fab502a43fd75e091f358d504565294d23",
      "auxiliary_source_sha256": "sha256:46f51b0dc291a70d002afeef2f198eda2133c137b5b8658cb4996d05a241760d",
      "stabilization_patch_hash": "sha256:564f5452e12e3179112e6aa3020efba4cbcedc0ad90dd26b98aed78a50dd85b1",
      "rollout_profile_patch_hash": "sha256:777456ce171734326cf8c55eaa434fc5a69a27d0b2d0b725608dd54c190f4a77",
      "scaling_profile_patch_hash": "sha256:58a2dca8377ed2ede804c7101b39bc6992ec669ef95e610ddec4b2a59bfc6b20",
      "learning_profile_patch_hash": "sha256:d47623e48b4cc757d5e63336a1fa7d49234c3833a2f1c6bbdbee4fe7ffd9a272",
      "action_control_source_sha256": "sha256:1b0bbc0a7a350581ef1939faf93a3e27d824bde62107ecfe4e9ff988bbc088fb",
      "auxiliary_profile_patch_hash": "sha256:30e7b4ea7a3ec0e45cdbe0a9dbf718eb219c2c6fd36d07f9b18bd5222183a03a",
      "action_control_profile_patch_hash": "sha256:548260ca5e3eedd53ae739c00672c7e273e7835faed734778039cf5977b56e0d"
    },
    "environment_profile": {
      "id": "rpo_bfm_ufo_isaac_4gpu_rsi_contact_v8",
      "hash": "sha256:23fe3b6f8cd49a749591cf7ab85a75f3918d6ea1c4150b34a16791e7864d3eff",
      "definition": {
        "task": "RPO v2有限URDF动作目标，原始q/dq与观察零点保持；6144环境，每个sampler2048；435M原生网络、batch2048、replay100万，每24576新转移触发16次联合更新。 RSI仅train接触候选、足底2mm间隙、有限root上移；写入后刷新FK，专家数据不改。",
        "reward": "九项raw aux，limits_dof_vel=sum relu(abs(dq)/Vmax-0.70)^2，权重-25仅加权一次；判别奖励截[-2,2]，D lr=3e-6；Actor正则尺度上限50。",
        "failure": "真实硬位置/速度/力矩终止保持；target slew每步0.5×Vmax；Actor裁剪1，其余五Adam裁剪5，非有限梯度失败。 显式PD增益保持，physics dt=2.5ms/decimation8/50Hz；物理子步硬限位锁存，接触/高度/倾角原因分列。",
        "success": "训练/完整恢复资格16→24后新增500联合更新，最多2小时且不自动重试；训练指标不足不启动独立评估。正常行走尚未验证。",
        "settings": {}
      }
    },
    "resume_source_image": "eb/rlinf:rpo-bfm-ufo-agent-4gpu-contact-v8-poststep1",
    "source_train_job_id": "trn_f16e6b774b0b4a0b8f19",
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
    "resume_from": "run_7e3d481401804da2b688"
  },
  "resumed_from": "trn_f16e6b774b0b4a0b8f19",
  "resume_step": 16,
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
    "env.train.init_params.id=RPO-UFO-Isaac-RSI-contact-v8",
    "+actor.model.auxiliary_profile=soft_velocity_v7",
    "+actor.model.scaling_profile=env6144_batch2048_v6",
    "+actor.model.control_profile=reachable_v2",
    "+actor.model.optimizer_profile=stabilized_v7",
    "+actor.model.contact_profile=contact_stable_v8",
    "actor.seed=42",
    "~cluster.component_placement",
    "+cluster.component_placement={actor:0,env\\,\\ rollout:1-3}",
    "runner.logger.log_path=/workspace/output",
    "hydra.run.dir=/workspace/output/hydra",
    "runner.logger.experiment_name=\"RPO UFO contact v8 poststep1 native 16 to24 restore qualification\"",
    "runner.resume_dir=/workspace/resume/global_step_16"
  ],
  "run_dir": "/data/kk/embodiedbein/runs/run_c768a1338d1e46fe8092",
  "auto_register_latest": false,
  "metrics_summary": {
    "checkpoint": {
      "path": "/data/kk/models/runs/run_c768a1338d1e46fe8092",
      "step": 24,
      "bytes": 7384201072
    },
    "sampler_sync": {
      "identity": {
        "version": 16,
        "state_sha256": "c97f6007b91c7dbe9c5cc193992066fa9299d690c7a7a3f4216b27aa7a7bb878",
        "online_transitions": 49152
      },
      "samplers": [
        {
          "rank": 0,
          "version": 16,
          "state_sha256": "c97f6007b91c7dbe9c5cc193992066fa9299d690c7a7a3f4216b27aa7a7bb878",
          "online_transitions": 49152
        },
        {
          "rank": 1,
          "version": 16,
          "state_sha256": "c97f6007b91c7dbe9c5cc193992066fa9299d690c7a7a3f4216b27aa7a7bb878",
          "online_transitions": 49152
        },
        {
          "rank": 2,
          "version": 16,
          "state_sha256": "c97f6007b91c7dbe9c5cc193992066fa9299d690c7a7a3f4216b27aa7a7bb878",
          "online_transitions": 49152
        }
      ],
      "schema_version": "eb/ufo-sampler-sync/v1",
      "sampling_rounds": 2
    },
    "model_version_id": "mv_e75fc4446a09457b8aa6"
  },
  "error_message": null,
  "job_id": "job_3574aebace10470d9f31",
  "started_at": "2026-10-11T23:33:26.874857+00:00",
  "finished_at": "2026-10-11T23:36:57.849951+00:00",
  "created_at": "2026-10-11T23:33:21.470130+00:00"
}
```

### 524更新完整TrainJob

```json
{
  "id": "trn_48bbe5a5a1d94467b48a",
  "run_id": "run_ad47cf6f8cfd4355bb44",
  "kind": "ppo",
  "name": "RPO UFO contact v8 poststep1 500-update observation",
  "model_name": "RPO-UFO-contact-v8-poststep1-step524",
  "status": "succeeded",
  "template_id": "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_contact_v8",
  "image_digest": "sha256:288c9bc690f0b0cefce4529b9ae07d3d0c9163b201d08d228161b7818685b5e2",
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
  "ray_port": 6396,
  "params": {
    "stack": "rlinf",
    "purpose": "training",
    "execution": {
      "recipe_id": "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_contact_v8",
      "runtime_id": "rlinf",
      "provider_id": "ufo",
      "topology_id": "one_learner_three_samplers",
      "algorithm_id": "bfm_fb_cpr_aux",
      "simulator_id": "isaaclab",
      "progress_unit": "learner_joint_update",
      "schema_version": "eb/training-binding/v1",
      "checkpoint_format": "ufo_agent_state_v1"
    },
    "model_name": "RPO-UFO-contact-v8-poststep1-step524",
    "config_name": "rpo_bfm_ufo_isaac",
    "resume_mode": "trainer_state",
    "resume_step": 24,
    "template_id": "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_contact_v8",
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
      "contact_profile": "contact_stable_v8",
      "control_profile": "reachable_v2",
      "scaling_profile": "env6144_batch2048_v6",
      "auxiliary_profile": "soft_velocity_v7",
      "optimizer_profile": "stabilized_v7",
      "contact_patch_hash": "sha256:e20a0ff930b3b9430ece70c21ab316a06cef92b679b638c0a35d7fed62d6d814",
      "topology_patch_hash": "sha256:3078506628924464cb42d0b2f108b3fab502a43fd75e091f358d504565294d23",
      "auxiliary_source_sha256": "sha256:46f51b0dc291a70d002afeef2f198eda2133c137b5b8658cb4996d05a241760d",
      "stabilization_patch_hash": "sha256:564f5452e12e3179112e6aa3020efba4cbcedc0ad90dd26b98aed78a50dd85b1",
      "rollout_profile_patch_hash": "sha256:777456ce171734326cf8c55eaa434fc5a69a27d0b2d0b725608dd54c190f4a77",
      "scaling_profile_patch_hash": "sha256:58a2dca8377ed2ede804c7101b39bc6992ec669ef95e610ddec4b2a59bfc6b20",
      "learning_profile_patch_hash": "sha256:d47623e48b4cc757d5e63336a1fa7d49234c3833a2f1c6bbdbee4fe7ffd9a272",
      "action_control_source_sha256": "sha256:1b0bbc0a7a350581ef1939faf93a3e27d824bde62107ecfe4e9ff988bbc088fb",
      "auxiliary_profile_patch_hash": "sha256:30e7b4ea7a3ec0e45cdbe0a9dbf718eb219c2c6fd36d07f9b18bd5222183a03a",
      "action_control_profile_patch_hash": "sha256:548260ca5e3eedd53ae739c00672c7e273e7835faed734778039cf5977b56e0d"
    },
    "environment_profile": {
      "id": "rpo_bfm_ufo_isaac_4gpu_rsi_contact_v8",
      "hash": "sha256:23fe3b6f8cd49a749591cf7ab85a75f3918d6ea1c4150b34a16791e7864d3eff",
      "definition": {
        "task": "RPO v2有限URDF动作目标，原始q/dq与观察零点保持；6144环境，每个sampler2048；435M原生网络、batch2048、replay100万，每24576新转移触发16次联合更新。 RSI仅train接触候选、足底2mm间隙、有限root上移；写入后刷新FK，专家数据不改。",
        "reward": "九项raw aux，limits_dof_vel=sum relu(abs(dq)/Vmax-0.70)^2，权重-25仅加权一次；判别奖励截[-2,2]，D lr=3e-6；Actor正则尺度上限50。",
        "failure": "真实硬位置/速度/力矩终止保持；target slew每步0.5×Vmax；Actor裁剪1，其余五Adam裁剪5，非有限梯度失败。 显式PD增益保持，physics dt=2.5ms/decimation8/50Hz；物理子步硬限位锁存，接触/高度/倾角原因分列。",
        "success": "训练/完整恢复资格16→24后新增500联合更新，最多2小时且不自动重试；训练指标不足不启动独立评估。正常行走尚未验证。",
        "settings": {}
      }
    },
    "resume_source_image": "eb/rlinf:rpo-bfm-ufo-agent-4gpu-contact-v8-poststep1",
    "source_train_job_id": "trn_2536bcfa9c7d469e98b0",
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
    "resume_from": "run_c768a1338d1e46fe8092"
  },
  "resumed_from": "trn_2536bcfa9c7d469e98b0",
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
    "env.train.init_params.id=RPO-UFO-Isaac-RSI-contact-v8",
    "+actor.model.auxiliary_profile=soft_velocity_v7",
    "+actor.model.scaling_profile=env6144_batch2048_v6",
    "+actor.model.control_profile=reachable_v2",
    "+actor.model.optimizer_profile=stabilized_v7",
    "+actor.model.contact_profile=contact_stable_v8",
    "actor.seed=42",
    "~cluster.component_placement",
    "+cluster.component_placement={actor:0,env\\,\\ rollout:1-3}",
    "runner.logger.log_path=/workspace/output",
    "hydra.run.dir=/workspace/output/hydra",
    "runner.logger.experiment_name=\"RPO UFO contact v8 poststep1 500-update observation\"",
    "runner.resume_dir=/workspace/resume/global_step_24"
  ],
  "run_dir": "/data/kk/embodiedbein/runs/run_ad47cf6f8cfd4355bb44",
  "auto_register_latest": false,
  "metrics_summary": {
    "checkpoint": {
      "path": "/data/kk/models/runs/run_ad47cf6f8cfd4355bb44",
      "step": 524,
      "bytes": 12922255524
    },
    "sampler_sync": {
      "identity": {
        "version": 520,
        "state_sha256": "d01292fafe1893d0e0fb7cba47587091ba64159d2a7d805a5e8ec0651373397f",
        "online_transitions": 835584
      },
      "samplers": [
        {
          "rank": 0,
          "version": 520,
          "state_sha256": "d01292fafe1893d0e0fb7cba47587091ba64159d2a7d805a5e8ec0651373397f",
          "online_transitions": 835584
        },
        {
          "rank": 1,
          "version": 520,
          "state_sha256": "d01292fafe1893d0e0fb7cba47587091ba64159d2a7d805a5e8ec0651373397f",
          "online_transitions": 835584
        },
        {
          "rank": 2,
          "version": 520,
          "state_sha256": "d01292fafe1893d0e0fb7cba47587091ba64159d2a7d805a5e8ec0651373397f",
          "online_transitions": 835584
        }
      ],
      "schema_version": "eb/ufo-sampler-sync/v1",
      "sampling_rounds": 34
    },
    "model_version_id": "mv_2e471b1dde934793a5d0"
  },
  "error_message": null,
  "job_id": "job_9e78b2378fff41f29132",
  "started_at": "2026-10-11T23:41:28.024174+00:00",
  "finished_at": "2026-10-12T00:18:16.904328+00:00",
  "created_at": "2026-10-11T23:41:25.329626+00:00"
}
```

正式段完整Job/Run（路径仅作为实际配置值）：
```json
{
  "id": "job_9e78b2378fff41f29132",
  "kind": "train.rl",
  "status": "succeeded",
  "embodiment_env_id": "ee_rpo_v2_bfm_23dof_v2",
  "priority": 50,
  "created_by": "user:kk",
  "spec": {
    "kind": "train.rl",
    "name": "RPO UFO contact v8 poststep1 500-update observation",
    "gates": [],
    "image": "sha256:288c9bc690f0b0cefce4529b9ae07d3d0c9163b201d08d228161b7818685b5e2",
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
      "resume_from": "run_c768a1338d1e46fe8092"
    },
    "labels": {},
    "params": {
      "stack": "rlinf",
      "purpose": "training",
      "execution": {
        "recipe_id": "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_contact_v8",
        "runtime_id": "rlinf",
        "provider_id": "ufo",
        "topology_id": "one_learner_three_samplers",
        "algorithm_id": "bfm_fb_cpr_aux",
        "simulator_id": "isaaclab",
        "progress_unit": "learner_joint_update",
        "schema_version": "eb/training-binding/v1",
        "checkpoint_format": "ufo_agent_state_v1"
      },
      "model_name": "RPO-UFO-contact-v8-poststep1-step524",
      "config_name": "rpo_bfm_ufo_isaac",
      "resume_mode": "trainer_state",
      "resume_step": 24,
      "template_id": "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_contact_v8",
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
        "contact_profile": "contact_stable_v8",
        "control_profile": "reachable_v2",
        "scaling_profile": "env6144_batch2048_v6",
        "auxiliary_profile": "soft_velocity_v7",
        "optimizer_profile": "stabilized_v7",
        "contact_patch_hash": "sha256:e20a0ff930b3b9430ece70c21ab316a06cef92b679b638c0a35d7fed62d6d814",
        "topology_patch_hash": "sha256:3078506628924464cb42d0b2f108b3fab502a43fd75e091f358d504565294d23",
        "auxiliary_source_sha256": "sha256:46f51b0dc291a70d002afeef2f198eda2133c137b5b8658cb4996d05a241760d",
        "stabilization_patch_hash": "sha256:564f5452e12e3179112e6aa3020efba4cbcedc0ad90dd26b98aed78a50dd85b1",
        "rollout_profile_patch_hash": "sha256:777456ce171734326cf8c55eaa434fc5a69a27d0b2d0b725608dd54c190f4a77",
        "scaling_profile_patch_hash": "sha256:58a2dca8377ed2ede804c7101b39bc6992ec669ef95e610ddec4b2a59bfc6b20",
        "learning_profile_patch_hash": "sha256:d47623e48b4cc757d5e63336a1fa7d49234c3833a2f1c6bbdbee4fe7ffd9a272",
        "action_control_source_sha256": "sha256:1b0bbc0a7a350581ef1939faf93a3e27d824bde62107ecfe4e9ff988bbc088fb",
        "auxiliary_profile_patch_hash": "sha256:30e7b4ea7a3ec0e45cdbe0a9dbf718eb219c2c6fd36d07f9b18bd5222183a03a",
        "action_control_profile_patch_hash": "sha256:548260ca5e3eedd53ae739c00672c7e273e7835faed734778039cf5977b56e0d"
      },
      "environment_profile": {
        "id": "rpo_bfm_ufo_isaac_4gpu_rsi_contact_v8",
        "hash": "sha256:23fe3b6f8cd49a749591cf7ab85a75f3918d6ea1c4150b34a16791e7864d3eff",
        "definition": {
          "task": "RPO v2有限URDF动作目标，原始q/dq与观察零点保持；6144环境，每个sampler2048；435M原生网络、batch2048、replay100万，每24576新转移触发16次联合更新。 RSI仅train接触候选、足底2mm间隙、有限root上移；写入后刷新FK，专家数据不改。",
          "reward": "九项raw aux，limits_dof_vel=sum relu(abs(dq)/Vmax-0.70)^2，权重-25仅加权一次；判别奖励截[-2,2]，D lr=3e-6；Actor正则尺度上限50。",
          "failure": "真实硬位置/速度/力矩终止保持；target slew每步0.5×Vmax；Actor裁剪1，其余五Adam裁剪5，非有限梯度失败。 显式PD增益保持，physics dt=2.5ms/decimation8/50Hz；物理子步硬限位锁存，接触/高度/倾角原因分列。",
          "success": "训练/完整恢复资格16→24后新增500联合更新，最多2小时且不自动重试；训练指标不足不启动独立评估。正常行走尚未验证。",
          "settings": {}
        }
      },
      "resume_source_image": "eb/rlinf:rpo-bfm-ufo-agent-4gpu-contact-v8-poststep1",
      "source_train_job_id": "trn_2536bcfa9c7d469e98b0",
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
      "id": "run_ad47cf6f8cfd4355bb44",
      "attempt": 1,
      "status": "finished",
      "abort_reason": null,
      "metrics_summary": {
        "checkpoint": {
          "path": "/data/kk/models/runs/run_ad47cf6f8cfd4355bb44",
          "step": 524,
          "bytes": 12922255524
        },
        "sampler_sync": {
          "identity": {
            "version": 520,
            "state_sha256": "d01292fafe1893d0e0fb7cba47587091ba64159d2a7d805a5e8ec0651373397f",
            "online_transitions": 835584
          },
          "samplers": [
            {
              "rank": 0,
              "version": 520,
              "state_sha256": "d01292fafe1893d0e0fb7cba47587091ba64159d2a7d805a5e8ec0651373397f",
              "online_transitions": 835584
            },
            {
              "rank": 1,
              "version": 520,
              "state_sha256": "d01292fafe1893d0e0fb7cba47587091ba64159d2a7d805a5e8ec0651373397f",
              "online_transitions": 835584
            },
            {
              "rank": 2,
              "version": 520,
              "state_sha256": "d01292fafe1893d0e0fb7cba47587091ba64159d2a7d805a5e8ec0651373397f",
              "online_transitions": 835584
            }
          ],
          "schema_version": "eb/ufo-sampler-sync/v1",
          "sampling_rounds": 34
        },
        "model_version_id": "mv_2e471b1dde934793a5d0"
      }
    }
  ]
}
```
```json
{
  "id": "run_ad47cf6f8cfd4355bb44",
  "job_id": "job_9e78b2378fff41f29132",
  "attempt": 1,
  "status": "finished",
  "abort_reason": null,
  "metrics_summary": {
    "checkpoint": {
      "path": "/data/kk/models/runs/run_ad47cf6f8cfd4355bb44",
      "step": 524,
      "bytes": 12922255524
    },
    "sampler_sync": {
      "identity": {
        "version": 520,
        "state_sha256": "d01292fafe1893d0e0fb7cba47587091ba64159d2a7d805a5e8ec0651373397f",
        "online_transitions": 835584
      },
      "samplers": [
        {
          "rank": 0,
          "version": 520,
          "state_sha256": "d01292fafe1893d0e0fb7cba47587091ba64159d2a7d805a5e8ec0651373397f",
          "online_transitions": 835584
        },
        {
          "rank": 1,
          "version": 520,
          "state_sha256": "d01292fafe1893d0e0fb7cba47587091ba64159d2a7d805a5e8ec0651373397f",
          "online_transitions": 835584
        },
        {
          "rank": 2,
          "version": 520,
          "state_sha256": "d01292fafe1893d0e0fb7cba47587091ba64159d2a7d805a5e8ec0651373397f",
          "online_transitions": 835584
        }
      ],
      "schema_version": "eb/ufo-sampler-sync/v1",
      "sampling_rounds": 34
    },
    "model_version_id": "mv_2e471b1dde934793a5d0"
  },
  "gpu_ids": [
    0,
    1,
    2,
    3
  ],
  "provenance": "{\"code_ref\": \"sha256:04e20ece2e3e3f3603a98b24e04aba42023fb31a7395beddfdac2a5d522de54d\", \"config_hash\": \"0965e441bdf847b7c0348265a9684715643a448872f96ce612453709fb0f956b\", \"data_version\": null, \"image_digest\": \"sha256:288c9bc690f0b0cefce4529b9ae07d3d0c9163b201d08d228161b7818685b5e2\", \"seed\": null}",
  "artifacts": [
    {
      "id": "art_f612ba80360043a6a28b",
      "kind": "rlinf_checkpoint",
      "sha256": "sha256:44597698ab287af352fba32a3e0e2078265fcbb75af7381b080e60267b60ab03",
      "size_bytes": 12922255524,
      "storage": "/data/kk/models/runs/run_ad47cf6f8cfd4355bb44"
    }
  ]
}
```

最终模型版本登记（无评估报告）：
```json
{
  "id": "mv_2e471b1dde934793a5d0",
  "family_id": "mf_rpo_v2_bfm_zero",
  "name": "RPO-UFO-contact-v8-poststep1-step524",
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
    "run_id": "run_ad47cf6f8cfd4355bb44",
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
      "image": "sha256:288c9bc690f0b0cefce4529b9ae07d3d0c9163b201d08d228161b7818685b5e2",
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
      "recipe_id": "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_contact_v8",
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
    "template_id": "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_contact_v8",
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
    "train_job_id": "job_9e78b2378fff41f29132",
    "normalization": "frozen_running_stats",
    "progress_unit": "learner_joint_update",
    "action_control": {
      "reset": "clamp measured q to physical finite bounds; normalize anchor; clear history",
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
    "artifact_sha256": "sha256:44597698ab287af352fba32a3e0e2078265fcbb75af7381b080e60267b60ab03",
    "checkpoint_path": "/data/kk/models/runs/run_ad47cf6f8cfd4355bb44",
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
      "id": "rpo_bfm_ufo_isaac_4gpu_rsi_contact_v8",
      "hash": "sha256:23fe3b6f8cd49a749591cf7ab85a75f3918d6ea1c4150b34a16791e7864d3eff",
      "definition": {
        "task": "RPO v2有限URDF动作目标，原始q/dq与观察零点保持；6144环境，每个sampler2048；435M原生网络、batch2048、replay100万，每24576新转移触发16次联合更新。 RSI仅train接触候选、足底2mm间隙、有限root上移；写入后刷新FK，专家数据不改。",
        "reward": "九项raw aux，limits_dof_vel=sum relu(abs(dq)/Vmax-0.70)^2，权重-25仅加权一次；判别奖励截[-2,2]，D lr=3e-6；Actor正则尺度上限50。",
        "failure": "真实硬位置/速度/力矩终止保持；target slew每步0.5×Vmax；Actor裁剪1，其余五Adam裁剪5，非有限梯度失败。 显式PD增益保持，physics dt=2.5ms/decimation8/50Hz；物理子步硬限位锁存，接触/高度/倾角原因分列。",
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
  "created_at": "2026-10-12T00:18:16.141027Z",
  "train_job_name": "RPO UFO contact v8 poststep1 500-update observation",
  "train_page_id": "trn_48bbe5a5a1d94467b48a",
  "artifact": {
    "id": "art_f612ba80360043a6a28b",
    "kind": "rlinf_checkpoint",
    "sha256": "sha256:44597698ab287af352fba32a3e0e2078265fcbb75af7381b080e60267b60ab03",
    "size_bytes": 12922255524,
    "storage": "/data/kk/models/runs/run_ad47cf6f8cfd4355bb44",
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
            "sha256": "6822c4782a708a613f0d71f03ce14e39a30cf7be4e775433305f7d684f40de48"
          },
          {
            "path": "model/native/global_step_524/optimizers.pth",
            "bytes": 3481181041,
            "sha256": "53c9f7678d3d66e002a9952946a3d58666e4cd35f3a43de32d3cfe2f8e91d326"
          },
          {
            "path": "model/native/global_step_524/resume-manifest.json",
            "bytes": 25056,
            "sha256": "85b518a3998c3bc07dd30fc86e921130072bd72f2cfd369ef86b6c87416c8778"
          },
          {
            "path": "model/native/global_step_524/trainer-state.pt",
            "bytes": 6097184828,
            "sha256": "f2ff3410660ada51cac78e6ab14d1a83d75d1df8dc9368c25891a5df5a71224d"
          },
          {
            "path": "model/native-checkpoint.json",
            "bytes": 88,
            "sha256": "24414cc2fa26deb57f6445d810747cdb6906f32b41b6b6102e7eb39cbc5debcc"
          },
          {
            "path": "model/resolved-config.yaml",
            "bytes": 3844,
            "sha256": "38d3c6eca030290d72622c5155e6d0409d494cc12a8f7ee4b523cf4df73fefbf"
          },
          {
            "path": "model/source-receipt.json",
            "bytes": 2414,
            "sha256": "d223546a784e38619d42b3f7cb336efcb0c3431346ffdb077d57dd8e5a888442"
          }
        ],
        "format": "ufo_agent_state_v1",
        "run_id": "run_ad47cf6f8cfd4355bb44",
        "status": "complete",
        "execution": {
          "recipe_id": "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_contact_v8",
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
        "template_id": "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_contact_v8",
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

## checkpoint、恢复与原始训练证据


### 16更新checkpoint哈希、六Adam、累计计数

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
  "fingerprint": "sha256:1a6dc4f3d33c48f4554beb2fc6ede1979c5f6c7dabcc2578e476fb224c01f25f",
  "execution": {
    "schema_version": "eb/training-binding/v1",
    "recipe_id": "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_contact_v8",
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

### 24更新checkpoint哈希、六Adam、累计计数

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
  "fingerprint": "sha256:1a6dc4f3d33c48f4554beb2fc6ede1979c5f6c7dabcc2578e476fb224c01f25f",
  "execution": {
    "schema_version": "eb/training-binding/v1",
    "recipe_id": "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_contact_v8",
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

### 524更新checkpoint哈希、六Adam、累计计数

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
  "fingerprint": "sha256:1a6dc4f3d33c48f4554beb2fc6ede1979c5f6c7dabcc2578e476fb224c01f25f",
  "execution": {
    "schema_version": "eb/training-binding/v1",
    "recipe_id": "rpo_bfm_ufo_agent_rlinf_isaac_4gpu_contact_v8",
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

16→24→524同recipe fingerprint相等，完整模型、六Adam、RNG、replay、expert sampler、z pool和rollout按回执恢复。物理内部状态不可用，恢复边界通过RSI重建，不能声称逐位一致物理轨迹。H从头初始化是本轮变化的明确限制，不能当旧G继续优化500次的同权重对照。

正式段新增786432条在线转移，每个环境约128控制步，即2.56秒物理探索；总更新数不是单环境累计控制步。大并发不弥补单环境探索跨度，不能推导收敛。

### 镜像实际来源

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
  "auxiliary_source_sha256": "sha256:46f51b0dc291a70d002afeef2f198eda2133c137b5b8658cb4996d05a241760d",
  "auxiliary_profile_patch_hash": "sha256:30e7b4ea7a3ec0e45cdbe0a9dbf718eb219c2c6fd36d07f9b18bd5222183a03a",
  "scaling_profile": "env6144_batch2048_v6",
  "scaling_profile_patch_hash": "sha256:58a2dca8377ed2ede804c7101b39bc6992ec669ef95e610ddec4b2a59bfc6b20",
  "control_profile": "reachable_v2",
  "optimizer_profile": "stabilized_v7",
  "stabilization_patch_hash": "sha256:564f5452e12e3179112e6aa3020efba4cbcedc0ad90dd26b98aed78a50dd85b1",
  "contact_profile": "contact_stable_v8",
  "contact_patch_hash": "sha256:e20a0ff930b3b9430ece70c21ab316a06cef92b679b638c0a35d7fed62d6d814"
}
```

### H24恢复trace核验

```json
{
  "rows": 24576,
  "rounds": 1,
  "round_first": 3,
  "round_last": 3,
  "requested_saturation_fraction": 0.3849920742753623,
  "applied_saturation_fraction": 0.035281858582427536,
  "hard_position_velocity_torque_counts": [
    93,
    987,
    0
  ],
  "contact_height_orientation_counts": [
    11,
    0,
    1
  ],
  "max_boundary_joint_velocity_radps": 20.34412384033203,
  "max_physics_joint_velocity_radps": 34.232139587402344,
  "max_boundary_location": {
    "round": 3,
    "row": 16962,
    "joint": "left_ankle_roll_joint",
    "dq_radps": -20.34412384033203,
    "episode_step": 2,
    "requested": 0.7122913002967834,
    "applied": -0.1599999964237213,
    "pd_target_rad": -0.07999999821186066,
    "torque_Nm": 0.6639893651008606,
    "hard_flags": [
      0,
      1,
      0
    ],
    "termination_reasons_contact_height_tilt": [
      0,
      0,
      0
    ],
    "foot_peak_N": [
      9785.27734375,
      0.0
    ],
    "physics_peak_radps": 20.34412384033203
  },
  "max_foot_contact_force_N": 14091.0595703125,
  "max_reset_root_lift_m": 0.021992022171616554,
  "latched_velocity_terms_without_boundary_violation": 833,
  "soft_velocity_raw_mean": 0.0034238392449712843,
  "any_joint_over_soft_limit_fraction": 0.0767822265625,
  "retained_replay_rows_checked": 24576,
  "terminal_episode_changes_checked": 737,
  "pd_decoder_max_error_rad": 1.92224979267408e-07,
  "reset_in_retained_replay": 6881,
  "first_step_terms_in_retained_replay": 39,
  "first_step_term_fraction": 0.005667780845807296,
  "first_step_hard_counts": [
    0,
    31,
    0
  ],
  "first_step_hard_union": 31,
  "first_step_contact_height_orientation_counts": [
    8,
    0,
    0
  ],
  "first_step_unexplained_terms": 0,
  "first_step_peak_foot_force_quantiles_N": {
    "0": 0.0,
    "0.5": 0.0,
    "0.95": 517.6121826171875,
    "0.99": 1683.979125976562,
    "1": 5377.66552734375
  },
  "reset_lift_quantiles_m": {
    "0": 0.0,
    "0.5": 0.0,
    "0.95": 0.016594206914305687,
    "1": 0.021992022171616554
  },
  "physics_dt_s": 0.0025,
  "policy_dt_s": 0.02,
  "hard_flags_are_physics_latched": true,
  "replay_actions_are_requested": true,
  "ninth_reward_matches_boundary_dq": true
}
```

### 正式段128列trace与真实终止核验

```json
{
  "rows": 786432,
  "rounds": 32,
  "round_first": 4,
  "round_last": 35,
  "requested_saturation_fraction": 0.44171595918959466,
  "applied_saturation_fraction": 0.15119624483412591,
  "hard_position_velocity_torque_counts": [
    11113,
    21309,
    0
  ],
  "contact_height_orientation_counts": [
    5868,
    25828,
    12926
  ],
  "max_boundary_joint_velocity_radps": 43.27584457397461,
  "max_physics_joint_velocity_radps": 65.38127899169922,
  "max_boundary_location": {
    "round": 21,
    "row": 8976,
    "joint": "left_knee_joint",
    "dq_radps": -43.27584457397461,
    "episode_step": 17,
    "requested": -0.9999989867210388,
    "applied": -0.9999989867210388,
    "pd_target_rad": -0.19999873638153076,
    "torque_Nm": -5.561982154846191,
    "hard_flags": [
      0,
      1,
      0
    ],
    "termination_reasons_contact_height_tilt": [
      0,
      0,
      0
    ],
    "foot_peak_N": [
      5266.986328125,
      0.0
    ],
    "physics_peak_radps": 43.27584457397461
  },
  "max_foot_contact_force_N": 27570.041015625,
  "max_reset_root_lift_m": 0.021992022171616554,
  "latched_velocity_terms_without_boundary_violation": 17188,
  "soft_velocity_raw_mean": 0.012745186378108406,
  "any_joint_over_soft_limit_fraction": 0.3228136698404948,
  "retained_replay_rows_checked": 786432,
  "terminal_episode_changes_checked": 71115,
  "pd_decoder_max_error_rad": 2.8073787650129134e-07,
  "reset_in_retained_replay": 77259,
  "first_step_terms_in_retained_replay": 343,
  "first_step_term_fraction": 0.004439612213463804,
  "first_step_hard_counts": [
    0,
    253,
    0
  ],
  "first_step_hard_union": 253,
  "first_step_contact_height_orientation_counts": [
    90,
    0,
    0
  ],
  "first_step_unexplained_terms": 0,
  "first_step_peak_foot_force_quantiles_N": {
    "0": 0.0,
    "0.5": 0.0,
    "0.95": 472.02990722655795,
    "0.99": 1207.4452392578119,
    "1": 8799.15625
  },
  "reset_lift_quantiles_m": {
    "0": 0.0,
    "0.5": 0.0,
    "0.95": 0.016585223376750946,
    "1": 0.021992022171616554
  },
  "physics_dt_s": 0.0025,
  "policy_dt_s": 0.02,
  "hard_flags_are_physics_latched": true,
  "replay_actions_are_requested": true,
  "ninth_reward_matches_boundary_dq": true
}
```

### 最终latent/RSI关节核验

```json
{
  "scope": "all final retained replay rows; chronological complete 6144-row blocks",
  "same_episode_nonperiodic_pairs": 705129,
  "passive_z_changes": 0,
  "native_period_100_pairs_excluded": 0,
  "cosine_median": 1.0,
  "reset_rows": 91730,
  "reset_joint_residuals": [
    {
      "joint": "left_thigh_yaw_joint",
      "outside_count": 0,
      "outside_fraction": 0.0,
      "mean_residual_rad": 0.0,
      "p95_residual_rad": 0.0,
      "max_residual_rad": 0.0
    },
    {
      "joint": "right_thigh_yaw_joint",
      "outside_count": 0,
      "outside_fraction": 0.0,
      "mean_residual_rad": 0.0,
      "p95_residual_rad": 0.0,
      "max_residual_rad": 0.0
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
      "mean_residual_rad": 0.0,
      "p95_residual_rad": 0.0,
      "max_residual_rad": 0.0
    },
    {
      "joint": "right_thigh_roll_joint",
      "outside_count": 0,
      "outside_fraction": 0.0,
      "mean_residual_rad": 0.0,
      "p95_residual_rad": 0.0,
      "max_residual_rad": 0.0
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
      "mean_residual_rad": 3.0149956613472607e-12,
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
      "mean_residual_rad": 0.0,
      "p95_residual_rad": 0.0,
      "max_residual_rad": 0.0
    },
    {
      "joint": "right_knee_joint",
      "outside_count": 0,
      "outside_fraction": 0.0,
      "mean_residual_rad": 0.0,
      "p95_residual_rad": 0.0,
      "max_residual_rad": 0.0
    },
    {
      "joint": "left_arm_yaw_joint",
      "outside_count": 0,
      "outside_fraction": 0.0,
      "mean_residual_rad": 0.0,
      "p95_residual_rad": 0.0,
      "max_residual_rad": 0.0
    },
    {
      "joint": "right_arm_yaw_joint",
      "outside_count": 0,
      "outside_fraction": 0.0,
      "mean_residual_rad": 0.0,
      "p95_residual_rad": 0.0,
      "max_residual_rad": 0.0
    },
    {
      "joint": "left_ankle_pitch_joint",
      "outside_count": 0,
      "outside_fraction": 0.0,
      "mean_residual_rad": 0.0,
      "p95_residual_rad": 0.0,
      "max_residual_rad": 0.0
    },
    {
      "joint": "right_ankle_pitch_joint",
      "outside_count": 0,
      "outside_fraction": 0.0,
      "mean_residual_rad": 0.0,
      "p95_residual_rad": 0.0,
      "max_residual_rad": 0.0
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

### 最终真实reset足底几何核验

```json
{
  "method": "reconstruct first-step replay q, projected gravity and privileged root height; official collision-mesh convex-hull FK, flat-ground yaw invariant",
  "scope": "evenly selected retained RSI first-step states; offline geometry, not physical safety acceptance",
  "reset_rows": 91730,
  "samples": 1024,
  "sole_height_quantiles_m": {
    "0": 0.0019999164630541202,
    "0.05": 0.0019999843513903723,
    "0.5": 0.010930355929963068,
    "0.95": 0.01911937017457669,
    "1": 0.01998521075623195
  },
  "penetration_over_2mm": 0,
  "penetration_over_2cm": 0,
  "both_feet_airborne_over_5cm": 0,
  "sole_below_1mm": 0
}
```

### 资源与存储预算

```json
{
  "scope": "new H diagnostic,16 qualification,24 restore and524 formal; exclude immutable prior G/data, unique inodes",
  "allocated_unique_bytes": 42521608192,
  "unique_inode_bytes": 42521143339,
  "apparent_bytes": 84634033559,
  "roots_apparent_bytes": {
    "/data/kk/models/runs/run_0f51028109cb4b2e9c12": 0,
    "/data/kk/embodiedbein/runs/run_0f51028109cb4b2e9c12": 9782834,
    "/data/kk/models/runs/run_7e3d481401804da2b688": 14422258506,
    "/data/kk/embodiedbein/runs/run_7e3d481401804da2b688": 22105426,
    "/data/kk/models/runs/run_c768a1338d1e46fe8092": 14768395628,
    "/data/kk/embodiedbein/runs/run_c768a1338d1e46fe8092": 10555471,
    "/data/kk/models/runs/run_ad47cf6f8cfd4355bb44": 25844504544,
    "/data/kk/embodiedbein/runs/run_ad47cf6f8cfd4355bb44": 323321218,
    "/data/kk/models/runs/run_a1ce3855d3894e97a17b": 0,
    "/data/kk/embodiedbein/runs/run_a1ce3855d3894e97a17b": 9783140,
    "/data/kk/models/runs/run_6fab081aacc04b11bee4": 14422258496,
    "/data/kk/embodiedbein/runs/run_6fab081aacc04b11bee4": 22123261,
    "/data/kk/models/runs/run_0c2d2f704e1a433db877": 14768395746,
    "/data/kk/embodiedbein/runs/run_0c2d2f704e1a433db877": 10549289
  },
  "additional_work_bytes": 11375563,
  "image_and_staging_reserved_bytes": 12884901888,
  "budget_bytes": 68719476736,
  "within_budget": true
}
```

### 启动GPU快照

```json
{
  "hostname": "eai4090x4m1",
  "gpus": [
    {
      "index": 0,
      "name": "NVIDIA GeForce RTX 4090",
      "memory_total_mb": 24564,
      "memory_free_mb": 24077,
      "run_id": "run_ad47cf6f8cfd4355bb44"
    },
    {
      "index": 1,
      "name": "NVIDIA GeForce RTX 4090",
      "memory_total_mb": 24564,
      "memory_free_mb": 21533,
      "run_id": "run_ad47cf6f8cfd4355bb44"
    },
    {
      "index": 2,
      "name": "NVIDIA GeForce RTX 4090",
      "memory_total_mb": 24564,
      "memory_free_mb": 21486,
      "run_id": "run_ad47cf6f8cfd4355bb44"
    },
    {
      "index": 3,
      "name": "NVIDIA GeForce RTX 4090",
      "memory_total_mb": 24564,
      "memory_free_mb": 21533,
      "run_id": "run_ad47cf6f8cfd4355bb44"
    }
  ],
  "memory_total_mb": 128644,
  "has_dri": true,
  "memory_available_mb": 88766
}
```

### 训练前段GPU快照

```json
{
  "hostname": "eai4090x4m1",
  "gpus": [
    {
      "index": 0,
      "name": "NVIDIA GeForce RTX 4090",
      "memory_total_mb": 24564,
      "memory_free_mb": 7854,
      "run_id": "run_ad47cf6f8cfd4355bb44"
    },
    {
      "index": 1,
      "name": "NVIDIA GeForce RTX 4090",
      "memory_total_mb": 24564,
      "memory_free_mb": 17456,
      "run_id": "run_ad47cf6f8cfd4355bb44"
    },
    {
      "index": 2,
      "name": "NVIDIA GeForce RTX 4090",
      "memory_total_mb": 24564,
      "memory_free_mb": 17409,
      "run_id": "run_ad47cf6f8cfd4355bb44"
    },
    {
      "index": 3,
      "name": "NVIDIA GeForce RTX 4090",
      "memory_total_mb": 24564,
      "memory_free_mb": 17456,
      "run_id": "run_ad47cf6f8cfd4355bb44"
    }
  ],
  "memory_total_mb": 128644,
  "has_dri": true,
  "memory_available_mb": 56877
}
```

### 训练中段GPU快照

```json
{
  "hostname": "eai4090x4m1",
  "gpus": [
    {
      "index": 0,
      "name": "NVIDIA GeForce RTX 4090",
      "memory_total_mb": 24564,
      "memory_free_mb": 7854,
      "run_id": "run_ad47cf6f8cfd4355bb44"
    },
    {
      "index": 1,
      "name": "NVIDIA GeForce RTX 4090",
      "memory_total_mb": 24564,
      "memory_free_mb": 17372,
      "run_id": "run_ad47cf6f8cfd4355bb44"
    },
    {
      "index": 2,
      "name": "NVIDIA GeForce RTX 4090",
      "memory_total_mb": 24564,
      "memory_free_mb": 17389,
      "run_id": "run_ad47cf6f8cfd4355bb44"
    },
    {
      "index": 3,
      "name": "NVIDIA GeForce RTX 4090",
      "memory_total_mb": 24564,
      "memory_free_mb": 17372,
      "run_id": "run_ad47cf6f8cfd4355bb44"
    }
  ],
  "memory_total_mb": 128644,
  "has_dri": true,
  "memory_available_mb": 51261
}
```

### 最终GPU资源

```json
{
  "hostname": "eai4090x4m1",
  "gpus": [
    {
      "index": 0,
      "name": "NVIDIA GeForce RTX 4090",
      "memory_total_mb": 24564,
      "memory_free_mb": 24077,
      "run_id": "run_f7950a57b6e041b09023"
    },
    {
      "index": 1,
      "name": "NVIDIA GeForce RTX 4090",
      "memory_total_mb": 24564,
      "memory_free_mb": 21533,
      "run_id": "run_f7950a57b6e041b09023"
    },
    {
      "index": 2,
      "name": "NVIDIA GeForce RTX 4090",
      "memory_total_mb": 24564,
      "memory_free_mb": 21444,
      "run_id": "run_f7950a57b6e041b09023"
    },
    {
      "index": 3,
      "name": "NVIDIA GeForce RTX 4090",
      "memory_total_mb": 24564,
      "memory_free_mb": 21533,
      "run_id": "run_f7950a57b6e041b09023"
    }
  ],
  "memory_total_mb": 128644,
  "has_dri": true,
  "memory_available_mb": 88157
}
```

正式Job墙钟=2208.88015秒；learner日志wall_time_s末值=1010.81592秒。后者不包含完整初始化、checkpoint保存/发布/收集。最终完整产物的哈希与六Adam已回读；未将“更新到524”直接当作发布完成。逐GPU完整峰值/利用率时序未采集，结束显存不代表峰值。

## G与H正式500更新段对照


| 指标 | 旧G首100窗 | 旧G末100窗 | 新H首100窗 | 新H末100窗 |
|---|---|---|---|---|
| env/episode_len | 8.52139382 | 14.1548762 | 7.47538356 | 11.8562354 |
| train/ufo/hard_velocity_count | 275.333333 | 184.857143 | 744.833333 | 598 |
| train/ufo/hard_position_count | 326.166667 | 314.285714 | 379.666667 | 289.714286 |
| train/ufo/critic_loss | 0.789665318 | 0.578277404 | 0.71755628 | 0.627521253 |
| train/ufo/fb_loss | 100982.622 | 17461.2352 | 108008.635 | 19367.3987 |
| train/ufo/applied_action_saturation_fraction | 0.106697121 | 0.184975192 | 0.10603399 | 0.163640182 |

| 实测 | 旧G | 新H |
|---|---|---|
| 首步终止率 | 0.0144923174 | 0.00443961221 |
| 控制步末dq峰rad/s | 68.680458 | 43.2758446 |
| 物理子步dq峰rad/s | 未记录 | 65.381279 |
| applied饱和比例 | 0.161479 | 0.151196245 |
| req饱和比例 | 0.441936 | 0.441715959 |

H子步锁存更严格，同时改变RSI分布、root高度修正和物理dt；两段新初始化且RSI轨迹不同。该对照用于发现实际风险，不宣称单因素归因或总体行走泛化。训练窗口均值按日志点等权，最后50点均值不是最后50更新，episode平均不是单case最大存活。

首步终止=343/77259；位置/速度/力矩首步flags=[0, 253, 0]；接触/高度/倾角首步flags=[90, 0, 0]；未解释真终止=0。原因可重叠，不按列求和当独立失败数。脚部力是足部垂直合力绝对峰值，不是保护部位碰撞冲量；保护部位contact终止另记。

最终reset离线样本1024：足底穿透>2mm=0，双足高于5cm=0。FK几何检查不等于Isaac接触求解或真实机器人安全合格。

上述reset几何相对于假设的平坦地面Z=0重建，未结合其世界XY是否位于有限地面；因此“无穿透”不能证明该环境实际有地面支撑。H缺少逐transition世界XY，精确缺地case数未验证。

本段最大dq及对应关节、PD target/torque、episode阶段、足部力和原因已在trace回执完整列出；任何仍超额定边界的峰值都不能以episode增长或FB loss下降忽略。latched_velocity_terms_without_boundary_violation揭示仅看20ms步末观测会漏掉的瞬时越限。

## 每采样轮原始观测


| 联合更新 | env/episode_len | train/ufo/online_transitions | train/ufo/hard_velocity_count | train/ufo/first_step_samples | train/ufo/first_step_terminated | train/ufo/contact_termination_count | train/ufo/height_termination_count | train/ufo/orientation_termination_count | train/ufo/physics_peak_joint_velocity | train/ufo/foot_contact_peak_N |
|---|---|---|---|---|---|---|---|---|---|---|
| 40 | 2.98580432 | 98304 | 526 | 6562 | 25 | 44 | 0 | 2 | 33.8547745 | 12751.7783 |
| 56 | 6.03214264 | 122880 | 819 | 1203 | 8 | 257 | 44 | 65 | 42.8732605 | 14685.2725 |
| 72 | 9.26088238 | 147456 | 766 | 2862 | 17 | 304 | 1713 | 482 | 33.5951271 | 14276.4424 |
| 88 | 8.89984894 | 172032 | 845 | 2862 | 24 | 219 | 504 | 803 | 51.645237 | 21723.8828 |
| 104 | 8.6574564 | 196608 | 769 | 2790 | 10 | 161 | 537 | 1060 | 29.7059727 | 14992.0605 |
| 120 | 9.01616669 | 221184 | 744 | 3060 | 18 | 144 | 932 | 866 | 37.140274 | 14419.8047 |
| 136 | 8.73940372 | 245760 | 786 | 2546 | 12 | 166 | 646 | 663 | 50.4553299 | 14360.7891 |
| 152 | 9.26823425 | 270336 | 707 | 2529 | 8 | 159 | 878 | 555 | 53.0627213 | 16836.666 |
| 168 | 9.34714413 | 294912 | 672 | 2560 | 16 | 174 | 868 | 586 | 52.9503403 | 17879.8359 |
| 184 | 9.48950481 | 319488 | 712 | 2362 | 18 | 178 | 767 | 554 | 32.0113564 | 27570.041 |
| 200 | 9.96623802 | 344064 | 682 | 2457 | 12 | 153 | 954 | 521 | 46.7131348 | 14395.1152 |
| 216 | 9.93068027 | 368640 | 662 | 2411 | 9 | 213 | 803 | 487 | 35.0397911 | 21939.9453 |
| 232 | 9.97330379 | 393216 | 687 | 2258 | 5 | 221 | 828 | 393 | 63.1349792 | 20755.2207 |
| 248 | 10.3767805 | 417792 | 649 | 2288 | 8 | 178 | 847 | 467 | 38.0816269 | 15854.7656 |
| 264 | 10.6032305 | 442368 | 623 | 2289 | 12 | 218 | 829 | 446 | 37.6070786 | 16694.3242 |
| 280 | 10.496439 | 466944 | 686 | 2373 | 15 | 214 | 875 | 468 | 49.3727531 | 16291.4307 |
| 296 | 10.4988928 | 491520 | 640 | 2296 | 8 | 218 | 806 | 397 | 45.0026817 | 13759.7959 |
| 312 | 10.6264553 | 516096 | 656 | 2162 | 9 | 183 | 767 | 360 | 43.6522598 | 12516.0801 |
| 328 | 11.0543528 | 540672 | 606 | 2190 | 8 | 162 | 842 | 395 | 35.5311737 | 14121.1455 |
| 344 | 10.9567957 | 565248 | 667 | 2240 | 7 | 172 | 808 | 394 | 37.7786179 | 13846.5703 |
| 360 | 11.218935 | 589824 | 637 | 2197 | 4 | 190 | 871 | 323 | 41.8464355 | 17261.4492 |
| 376 | 11.2538319 | 614400 | 616 | 2086 | 8 | 197 | 860 | 270 | 29.6760902 | 24578.1406 |
| 392 | 11.3999996 | 638976 | 646 | 2169 | 7 | 227 | 853 | 282 | 65.381279 | 15388.4053 |
| 408 | 10.9995365 | 663552 | 639 | 2151 | 8 | 183 | 840 | 309 | 49.0345268 | 17200.7949 |
| 424 | 11.3965197 | 688128 | 681 | 2143 | 11 | 196 | 834 | 250 | 37.1565247 | 26403.3301 |
| 440 | 11.2480392 | 712704 | 637 | 2078 | 14 | 184 | 850 | 233 | 29.172966 | 16710.7871 |
| 456 | 11.8083954 | 737280 | 565 | 2022 | 9 | 169 | 909 | 251 | 36.5542641 | 13559.584 |
| 472 | 11.9308023 | 761856 | 578 | 2087 | 6 | 160 | 916 | 230 | 25.646759 | 17255.334 |
| 488 | 11.777668 | 786432 | 627 | 2057 | 9 | 147 | 886 | 221 | 25.7646484 | 13813.2207 |
| 504 | 11.9454727 | 811008 | 571 | 1964 | 4 | 175 | 931 | 196 | 30.9212208 | 17276.5078 |
| 520 | 12.0541992 | 835584 | 607 | 2046 | 7 | 165 | 946 | 193 | 45.6943817 | 13802.4746 |
| 524 | 12.2290707 | 860160 | 601 | 1959 | 7 | 137 | 884 | 204 | 40.7484932 | 11833.6982 |

不同记录频率未插值或沿用前值；空白记未记录。控制采样轮与learner更新频率不同，按实际日志step展示。

## 全部136项标量统计


覆盖新增段>24；仅有初始配置点的tag使用全部原始点，first/last step明确列出。直接解析无点数上限的原始TensorBoard事件，所有标量有限。

| tag | n | first_step | last_step | first | last | mean | min | max | median | p95 | last50_mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| env/episode_len | 32 | 40 | 524 | 2.98580432 | 12.2290707 | 10.1700696 | 2.98580432 | 12.2290707 | 10.5510616 | 11.9943996 | 10.1700696 |
| env/num_trajectories | 32 | 40 | 524 | 634 | 1947 | 2237.59375 | 634 | 3377 | 2209.5 | 2888.7 | 2237.59375 |
| env/return | 32 | 40 | 524 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| env/reward | 32 | 40 | 524 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| env/success_once | 32 | 40 | 524 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| rollout/ufo/received_transitions | 32 | 40 | 524 | 24576 | 24576 | 24576 | 24576 | 24576 | 24576 | 24576 | 24576 |
| time/actor_training | 32 | 40 | 524 | 15.671176 | 3.51280069 | 11.4827514 | 3.51280069 | 15.671176 | 11.6347151 | 11.6927455 | 11.4827514 |
| time/cal_adv_and_returns | 32 | 40 | 524 | 0.00256276131 | 0.0116345882 | 0.0105706006 | 0.00242567062 | 0.0211853981 | 0.0116648674 | 0.0164578319 | 0.0105706006 |
| time/env/compute_bootstrap_rewards | 32 | 40 | 524 | 0.000245762989 | 0.000248550903 | 0.000235681611 | 0.000204937765 | 0.000281042652 | 0.000235065585 | 0.000259206491 | 0.000235681611 |
| time/env/env/bootstrap_step | 32 | 40 | 524 | 0.014415673 | 0.00027491292 | 0.000703114332 | 0.000232920051 | 0.014415673 | 0.000254406943 | 0.000307000056 | 0.000703114332 |
| time/env/env/send_rollout_trajectories | 32 | 40 | 524 | 0.255454659 | 0.217330858 | 0.231245069 | 0.210841075 | 0.255454659 | 0.229832321 | 0.244618186 | 0.231245069 |
| time/env/env_interact_step | 32 | 40 | 524 | 3.01615834 | 1.34304047 | 1.33203026 | 1.10821676 | 3.01615834 | 1.32818133 | 1.41051363 | 1.33203026 |
| time/env/interact | 32 | 40 | 524 | 4.49398184 | 2.12341666 | 2.20090626 | 1.9660579 | 4.49398184 | 2.13288903 | 2.28671381 | 2.20090626 |
| time/env/run_interact_once | 32 | 40 | 524 | 4.49393702 | 2.1233871 | 2.20087948 | 1.9660331 | 4.49393702 | 2.13286221 | 2.28668512 | 2.20087948 |
| time/generate_rollouts | 32 | 40 | 524 | 4.67074585 | 2.29917836 | 2.41366369 | 2.17181087 | 4.67074585 | 2.34073341 | 2.45877551 | 2.41366369 |
| time/rollout/generate_one_epoch | 32 | 40 | 524 | 4.24288321 | 1.9000324 | 1.96351149 | 1.71993256 | 4.24288321 | 1.90074152 | 2.06729472 | 1.96351149 |
| time/rollout/predict | 32 | 40 | 524 | 0.284284532 | 0.0402934663 | 0.0489431665 | 0.0361655988 | 0.284284532 | 0.0411490723 | 0.0524128856 | 0.0489431665 |
| time/rollout/rollout/generate | 32 | 40 | 524 | 4.24609184 | 1.90013766 | 1.96377357 | 1.72003984 | 4.24609184 | 1.9008491 | 2.06739866 | 1.96377357 |
| time/rollout/sync_model_from_actor | 32 | 40 | 524 | 14.7875223 | 13.2418776 | 13.2986709 | 12.8318434 | 14.7875223 | 13.2450471 | 13.6495469 | 13.2986709 |
| time/step | 32 | 40 | 524 | 38.6026802 | 22.8023949 | 30.9513148 | 22.8023949 | 38.6026802 | 30.9657936 | 31.3343689 | 30.9513148 |
| time/sync_weights | 32 | 40 | 524 | 18.257513 | 16.9783554 | 17.0437977 | 16.5990372 | 18.257513 | 17.0483923 | 17.3825383 | 17.0437977 |
| train/ufo/B | 500 | 25 | 524 | -0.030087918 | -0.00203483272 | -0.00594122343 | -0.030087918 | 0.00076803274 | -0.00451833382 | -0.00154515558 | -0.00236689133 |
| train/ufo/B_norm | 500 | 25 | 524 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 |
| train/ufo/F1 | 500 | 25 | 524 | 0.0617445484 | 0.0174017344 | 0.0226372101 | 0.0028041231 | 0.0763371214 | 0.0207255147 | 0.0390552128 | 0.0242603802 |
| train/ufo/M1 | 500 | 25 | 524 | 3.96703553 | 0.903292835 | 1.15490522 | -0.383799136 | 7.02845097 | 0.901710629 | 2.59353557 | 0.870772934 |
| train/ufo/Q1 | 500 | 25 | 524 | -3.45448589 | -4.08699512 | -2.49933699 | -4.22694016 | -1.18435585 | -2.41196692 | -1.37461495 | -3.91725341 |
| train/ufo/Q_aux | 500 | 25 | 524 | -4.23068237 | -5.9731946 | -4.94686299 | -6.24734497 | -3.7246449 | -4.89746189 | -3.93771989 | -6.02473082 |
| train/ufo/Q_discriminator | 500 | 25 | 524 | -3.28118324 | -4.33158302 | -2.57564734 | -4.33158302 | -1.28467488 | -2.52541792 | -1.40289383 | -3.96052464 |
| train/ufo/Q_fb | 500 | 25 | 524 | -3.97614574 | 52.9578438 | 27.8927931 | -6.60834312 | 52.9578438 | 28.8569527 | 49.9871166 | 49.6140215 |
| train/ufo/action_abs_mean | 32 | 40 | 524 | 0.884790599 | 0.900920093 | 0.90407341 | 0.884790599 | 0.9135378 | 0.903509676 | 0.911531645 | 0.90407341 |
| train/ufo/action_saturation_fraction | 32 | 40 | 524 | 0.404443711 | 0.43744868 | 0.441715958 | 0.404443711 | 0.456038773 | 0.441018805 | 0.452251489 | 0.441715958 |
| train/ufo/actor_loss | 500 | 25 | 524 | 6.35247993 | -36.1556931 | -19.5783637 | -36.2408829 | 9.5708704 | -21.4948339 | 0.377257597 | -33.687979 |
| train/ufo/actor_optimizer/grad_clip_limit | 500 | 25 | 524 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| train/ufo/actor_optimizer/grad_norm_after_clip_l2 | 500 | 25 | 524 | 0.999999464 | 0.999999762 | 0.999999728 | 0.999999225 | 0.99999994 | 0.999999762 | 0.999999881 | 0.999999808 |
| train/ufo/actor_optimizer/grad_norm_l2 | 500 | 25 | 524 | 1.65320396 | 6.16909075 | 4.24482661 | 1.38237357 | 6.29210043 | 4.38418818 | 5.90729818 | 5.90167906 |
| train/ufo/actor_optimizer/lr | 500 | 25 | 524 | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 |
| train/ufo/actor_optimizer/parameter_delta_l2 | 500 | 25 | 524 | 0.448349804 | 0.351330608 | 0.342416274 | 0.317063421 | 0.459962159 | 0.337261036 | 0.378759037 | 0.346967655 |
| train/ufo/actor_reg_weight_applied | 500 | 25 | 524 | 9.55606842 | 50 | 34.0836438 | 9.43098736 | 50 | 35.6546326 | 50 | 50 |
| train/ufo/actor_reg_weight_raw | 500 | 25 | 524 | 9.55606842 | 61.4188881 | 35.1736681 | 9.43098736 | 61.4188881 | 35.6546326 | 57.9629112 | 57.6824353 |
| train/ufo/applied_action_abs_mean | 32 | 40 | 524 | 0.45246321 | 0.58882308 | 0.576090293 | 0.45246321 | 0.591052294 | 0.586881518 | 0.590629983 | 0.576090293 |
| train/ufo/applied_action_saturation_fraction | 32 | 40 | 524 | 0.0500329062 | 0.163844898 | 0.151196245 | 0.0500329062 | 0.168411031 | 0.163273469 | 0.167857736 | 0.151196245 |
| train/ufo/auxQ1 | 500 | 25 | 524 | -4.75325298 | -6.04815054 | -4.88396343 | -6.18373966 | -3.74871445 | -4.80616236 | -3.94576174 | -5.9389147 |
| train/ufo/aux_critic_loss | 500 | 25 | 524 | 2.6604805 | 1.79359472 | 1.33634545 | 0.875273347 | 5.41925764 | 1.26318032 | 1.85050806 | 1.56549496 |
| train/ufo/aux_critic_optimizer/grad_clip_limit | 500 | 25 | 524 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 |
| train/ufo/aux_critic_optimizer/grad_norm_after_clip_l2 | 500 | 25 | 524 | 4.99999905 | 4.99999905 | 4.80340505 | 1.30916345 | 5.00000048 | 4.99999952 | 5 | 4.68828501 |
| train/ufo/aux_critic_optimizer/grad_norm_l2 | 500 | 25 | 524 | 15.5169668 | 6.6890316 | 10.6104969 | 1.30916345 | 37.5395279 | 9.60940886 | 21.1239558 | 7.34942011 |
| train/ufo/aux_critic_optimizer/lr | 500 | 25 | 524 | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 |
| train/ufo/aux_critic_optimizer/parameter_delta_l2 | 500 | 25 | 524 | 0.437787443 | 0.674568117 | 0.437406138 | 0.178094909 | 0.720511496 | 0.440841168 | 0.587928423 | 0.580497313 |
| train/ufo/aux_rew/limits_dof_pos | 500 | 25 | 524 | 0.0300747864 | 0.205934018 | 0.152136823 | 0.0289100204 | 0.215892553 | 0.167948514 | 0.204056106 | 0.203805511 |
| train/ufo/aux_rew/limits_dof_vel | 500 | 25 | 524 | 0.00517010596 | 0.0111602778 | 0.0109212314 | 0.00257473299 | 0.0155326752 | 0.0116856168 | 0.0135562386 | 0.0119743048 |
| train/ufo/aux_rew/limits_torque | 500 | 25 | 524 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| train/ufo/aux_rew/penalty_action_rate | 500 | 25 | 524 | 0.71372956 | 0.442409903 | 0.518451252 | 0.430282354 | 0.724483192 | 0.48941341 | 0.687301371 | 0.441867677 |
| train/ufo/aux_rew/penalty_ankle_roll | 500 | 25 | 524 | 0.0192776211 | 0.142616436 | 0.106127703 | 0.0189836379 | 0.148378238 | 0.116678886 | 0.142792834 | 0.142173414 |
| train/ufo/aux_rew/penalty_feet_ori | 500 | 25 | 524 | 0.07203646 | 0.144504115 | 0.116683101 | 0.07203646 | 0.150553256 | 0.119295955 | 0.138730346 | 0.137572951 |
| train/ufo/aux_rew/penalty_slippage | 500 | 25 | 524 | 0.324098706 | 0.336494952 | 0.324406235 | 0.266375601 | 0.392257333 | 0.321185797 | 0.369228834 | 0.305140007 |
| train/ufo/aux_rew/penalty_torques | 500 | 25 | 524 | 3588.2832 | 4700.36426 | 4770.92855 | 3588.2832 | 5189.27637 | 4804.64551 | 5036.27583 | 4668.84631 |
| train/ufo/aux_rew/penalty_undesired_contact | 500 | 25 | 524 | 0 | 0.00537109375 | 0.00436035156 | 0 | 0.00927734375 | 0.00439453125 | 0.00732421875 | 0.00477539062 |
| train/ufo/backward_optimizer/grad_clip_limit | 500 | 25 | 524 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 |
| train/ufo/backward_optimizer/grad_norm_after_clip_l2 | 500 | 25 | 524 | 5 | 5 | 4.99999998 | 4.99999905 | 5.00000095 | 5 | 5.00000048 | 5.00000002 |
| train/ufo/backward_optimizer/grad_norm_l2 | 500 | 25 | 524 | 292690 | 16198.8691 | 59070.7895 | 15844.2598 | 307114.562 | 35606.4004 | 168129.148 | 16999.8756 |
| train/ufo/backward_optimizer/lr | 500 | 25 | 524 | 9.99999975e-06 | 9.99999975e-06 | 9.99999975e-06 | 9.99999975e-06 | 9.99999975e-06 | 9.99999975e-06 | 9.99999975e-06 | 9.99999975e-06 |
| train/ufo/backward_optimizer/parameter_delta_l2 | 500 | 25 | 524 | 0.00367672625 | 0.00345045072 | 0.00402689363 | 0.00345045072 | 0.00445008138 | 0.00404678751 | 0.00442874092 | 0.00352219935 |
| train/ufo/collected_transitions | 32 | 40 | 524 | 24576 | 24576 | 24576 | 24576 | 24576 | 24576 | 24576 | 24576 |
| train/ufo/contact_termination_count | 32 | 40 | 524 | 44 | 137 | 183.375 | 44 | 304 | 178 | 240.5 | 183.375 |
| train/ufo/critic_loss | 500 | 25 | 524 | 1.76303434 | 0.736615896 | 0.466034889 | 0.20119606 | 1.76303434 | 0.38312526 | 0.872438982 | 0.627570306 |
| train/ufo/critic_optimizer/grad_clip_limit | 500 | 25 | 524 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 |
| train/ufo/critic_optimizer/grad_norm_after_clip_l2 | 500 | 25 | 524 | 4.99999905 | 4.99999857 | 4.79814519 | 0.877620399 | 5.00000095 | 4.99999952 | 5 | 4.02615324 |
| train/ufo/critic_optimizer/grad_norm_l2 | 500 | 25 | 524 | 23.1710052 | 5.32845306 | 11.0983083 | 0.877620399 | 31.8198109 | 9.91461372 | 21.9793094 | 6.6618003 |
| train/ufo/critic_optimizer/lr | 500 | 25 | 524 | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 |
| train/ufo/critic_optimizer/parameter_delta_l2 | 500 | 25 | 524 | 0.390447885 | 0.529379129 | 0.345547646 | 0.19312796 | 0.699475348 | 0.320819676 | 0.599885193 | 0.566887358 |
| train/ufo/disc_expert_loss | 500 | 25 | 524 | 0.554197431 | 0.253996432 | 0.313237149 | 0.230359584 | 0.554197431 | 0.285409614 | 0.473437192 | 0.24481707 |
| train/ufo/disc_loss | 500 | 25 | 524 | 3.23688102 | 0.799194098 | 1.37120531 | 0.788055539 | 3.23964596 | 1.07219261 | 2.78747791 | 0.816953583 |
| train/ufo/disc_reward_clip_fraction | 500 | 25 | 524 | 0 | 0.348632812 | 0.110082031 | 0 | 0.369628906 | 0.0534667969 | 0.334960938 | 0.335097656 |
| train/ufo/disc_train_loss | 500 | 25 | 524 | 0.69165957 | 0.316856086 | 0.510624272 | 0.311786979 | 0.691893756 | 0.499356896 | 0.671652403 | 0.328833484 |
| train/ufo/disc_wgan_gp_loss | 500 | 25 | 524 | 0.199102402 | 0.0228341557 | 0.0547343889 | 0.022740595 | 0.201122731 | 0.0288516609 | 0.166660161 | 0.0243303024 |
| train/ufo/discriminator_optimizer/grad_clip_limit | 500 | 25 | 524 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 |
| train/ufo/discriminator_optimizer/grad_norm_after_clip_l2 | 500 | 25 | 524 | 4.99999952 | 0.925252199 | 2.67755772 | 0.915659368 | 5 | 1.83260787 | 4.99999952 | 1.02191785 |
| train/ufo/discriminator_optimizer/grad_norm_l2 | 500 | 25 | 524 | 8.47924614 | 0.925252199 | 3.2492863 | 0.915659368 | 8.50552082 | 1.83260787 | 8.10708733 | 1.02191785 |
| train/ufo/discriminator_optimizer/lr | 500 | 25 | 524 | 3.00000011e-06 | 3.00000011e-06 | 3.00000011e-06 | 3.00000011e-06 | 3.00000011e-06 | 3.00000011e-06 | 3.00000011e-06 | 3.00000011e-06 |
| train/ufo/discriminator_optimizer/parameter_delta_l2 | 500 | 25 | 524 | 0.00457450515 | 0.00238535576 | 0.00359385564 | 0.00238535576 | 0.00527813053 | 0.00308299763 | 0.00522864831 | 0.00243391411 |
| train/ufo/environment_control_steps | 32 | 40 | 524 | 16 | 140 | 78 | 16 | 140 | 78 | 133.8 | 78 |
| train/ufo/expert_draws/g1 | 32 | 40 | 524 | 41104 | 536088 | 294789.25 | 41104 | 536088 | 295080 | 523470.4 | 294789.25 |
| train/ufo/expert_draws/official_rpo | 32 | 40 | 524 | 40816 | 537064 | 294266.75 | 40816 | 537064 | 294744 | 523467.2 | 294266.75 |
| train/ufo/fb_diag | 500 | 25 | 524 | -16.3710022 | -223.166687 | -156.919021 | -224.310181 | -10.9476814 | -179.173927 | -35.6312962 | -214.537821 |
| train/ufo/fb_loss | 500 | 25 | 524 | 183333.203 | 16905.1523 | 48210.2801 | 16905.1523 | 183333.203 | 34055.7344 | 127827.171 | 18170.1827 |
| train/ufo/fb_offdiag | 500 | 25 | 524 | 177.379211 | 241.557709 | 163.506465 | 100.743607 | 241.557709 | 163.219551 | 223.418353 | 222.35152 |
| train/ufo/first_step_samples | 32 | 40 | 524 | 6562 | 1959 | 2414.34375 | 1203 | 6562 | 2249 | 2951.1 | 2414.34375 |
| train/ufo/first_step_terminated | 32 | 40 | 524 | 25 | 7 | 10.71875 | 4 | 25 | 9 | 20.7 | 10.71875 |
| train/ufo/foot_contact_peak_N | 32 | 40 | 524 | 12751.7783 | 11833.6982 | 16711.0842 | 11833.6982 | 27570.041 | 15621.5854 | 25399.4759 | 16711.0842 |
| train/ufo/forward_optimizer/grad_clip_limit | 500 | 25 | 524 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 |
| train/ufo/forward_optimizer/grad_norm_after_clip_l2 | 500 | 25 | 524 | 5 | 5.00000048 | 5 | 4.99999905 | 5.00000095 | 5 | 5.00000048 | 4.99999996 |
| train/ufo/forward_optimizer/grad_norm_l2 | 500 | 25 | 524 | 844.345032 | 112.928841 | 252.996727 | 104.034889 | 1647.64844 | 150.699783 | 740.924915 | 115.069654 |
| train/ufo/forward_optimizer/lr | 500 | 25 | 524 | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 |
| train/ufo/forward_optimizer/parameter_delta_l2 | 500 | 25 | 524 | 0.392923266 | 0.777011275 | 0.661068639 | 0.262930155 | 0.839725316 | 0.729507983 | 0.812679565 | 0.779756796 |
| train/ufo/hard_position_count | 32 | 40 | 524 | 70 | 270 | 347.28125 | 70 | 542 | 339.5 | 524.2 | 347.28125 |
| train/ufo/hard_torque_count | 32 | 40 | 524 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| train/ufo/hard_velocity_count | 32 | 40 | 524 | 526 | 601 | 665.90625 | 526 | 845 | 652.5 | 800.85 | 665.90625 |
| train/ufo/height_termination_count | 32 | 40 | 524 | 0 | 884 | 807.125 | 0 | 1713 | 848.5 | 949.6 | 807.125 |
| train/ufo/learner_joint_updates | 500 | 25 | 524 | 25 | 524 | 274.5 | 25 | 524 | 274.5 | 499.05 | 499.5 |
| train/ufo/mean_aux_reward | 500 | 25 | 524 | -0.691078246 | -1.15877533 | -1.09761246 | -1.214252 | -0.636559129 | -1.1336838 | -0.846565127 | -1.15203911 |
| train/ufo/mean_disc_reward | 500 | 25 | 524 | 0.107013956 | -1.17956591 | -0.513835499 | -1.18920374 | 0.133539483 | -0.568580508 | 0.114987742 | -1.13625201 |
| train/ufo/mean_disc_reward_raw | 500 | 25 | 524 | 0.107013956 | -1.45175099 | -0.578613259 | -1.47694111 | 0.133539483 | -0.581380159 | 0.114987742 | -1.38601327 |
| train/ufo/mean_next_Q | 500 | 25 | 524 | -2.13145494 | -3.16243911 | -1.86574027 | -3.19110036 | -1.1508441 | -1.70404893 | -1.198113 | -3.00385337 |
| train/ufo/mean_next_auxQ | 500 | 25 | 524 | -3.53119278 | -5.27678776 | -3.96594098 | -5.30413723 | -2.82573748 | -3.87705433 | -2.93348901 | -5.166075 |
| train/ufo/online_transitions | 500 | 25 | 524 | 98304 | 860160 | 470089.728 | 98304 | 860160 | 466944 | 811008 | 815923.2 |
| train/ufo/orientation_termination_count | 32 | 40 | 524 | 2 | 204 | 403.9375 | 2 | 1060 | 393.5 | 831.35 | 403.9375 |
| train/ufo/orth_loss | 500 | 25 | 524 | 1831.72192 | 168.867615 | 482.036927 | 168.867615 | 1831.72192 | 340.738373 | 1277.46789 | 181.623691 |
| train/ufo/orth_loss_diag | 500 | 25 | 524 | -256 | -256 | -256 | -256 | -256 | -256 | -256 | -256 |
| train/ufo/orth_loss_offdiag | 500 | 25 | 524 | 2087.72192 | 424.867615 | 738.036927 | 424.867615 | 2087.72192 | 596.738373 | 1533.46789 | 437.623691 |
| train/ufo/physics_peak_joint_velocity | 32 | 40 | 524 | 33.8547745 | 40.7484932 | 40.83769 | 25.646759 | 65.381279 | 37.9301224 | 57.5952374 | 40.83769 |
| train/ufo/q_loss | 500 | 25 | 524 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| train/ufo/replay_draw_new_ratio | 32 | 40 | 524 | 0.833333313 | 1.24761903 | 1.18784997 | 0.833333313 | 1.27450979 | 1.23070174 | 1.27168562 | 1.18784997 |
| train/ufo/replay_draws | 32 | 40 | 524 | 81920 | 1073152 | 589056 | 81920 | 1073152 | 589824 | 1046937.6 | 589056 |
| train/ufo/replay_policy_age_max_updates | 32 | 40 | 524 | 40 | 524 | 287.625 | 40 | 524 | 288 | 511.2 | 287.625 |
| train/ufo/replay_policy_age_min_updates | 32 | 40 | 524 | 16 | 4 | 15.625 | 4 | 16 | 16 | 16 | 15.625 |
| train/ufo/replay_retained_fraction | 32 | 40 | 524 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| train/ufo/replay_size | 32 | 40 | 524 | 98304 | 860160 | 479232 | 98304 | 860160 | 479232 | 822067.2 | 479232 |
| train/ufo/requested_action_abs_mean | 32 | 40 | 524 | 0.88479054 | 0.900920153 | 0.904073428 | 0.88479054 | 0.91353786 | 0.903509706 | 0.911531678 | 0.904073428 |
| train/ufo/requested_action_saturation_fraction | 32 | 40 | 524 | 0.404443711 | 0.43744868 | 0.441715958 | 0.404443711 | 0.456038773 | 0.441018805 | 0.452251489 | 0.441715958 |
| train/ufo/requested_applied_max_diff | 32 | 40 | 524 | 1.95207369 | 1.95573056 | 1.95521498 | 1.94904256 | 1.96602774 | 1.95280135 | 1.96602774 | 1.95521498 |
| train/ufo/reset_control_samples | 32 | 40 | 524 | 6562 | 1959 | 2414.34375 | 1203 | 6562 | 2249 | 2951.1 | 2414.34375 |
| train/ufo/reset_projection_outside_fraction | 32 | 40 | 524 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| train/ufo/reset_projection_residual_max_rad | 32 | 40 | 524 | 1.78813934e-07 | 1.78813934e-07 | 1.78813934e-07 | 1.78813934e-07 | 1.78813934e-07 | 1.78813934e-07 | 1.78813934e-07 | 1.78813934e-07 |
| train/ufo/reset_projection_residual_mean_rad | 32 | 40 | 524 | 6.43538485e-08 | 6.53098269e-08 | 6.46513916e-08 | 6.34298729e-08 | 6.61209327e-08 | 6.44895017e-08 | 6.57433887e-08 | 6.46513916e-08 |
| train/ufo/reset_projection_residual_p95_rad | 32 | 40 | 524 | 1.1920929e-07 | 1.1920929e-07 | 1.1920929e-07 | 1.1920929e-07 | 1.1920929e-07 | 1.1920929e-07 | 1.1920929e-07 | 1.1920929e-07 |
| train/ufo/reset_root_lift_max_m | 32 | 40 | 524 | 0.0219920222 | 0.0219920222 | 0.0219920222 | 0.0219920222 | 0.0219920222 | 0.0219920222 | 0.0219920222 | 0.0219920222 |
| train/ufo/sampling_rounds | 500 | 25 | 524 | 4 | 35 | 19.128 | 4 | 35 | 19 | 33 | 33.2 |
| train/ufo/target_M | 500 | 25 | 524 | 2.85923195 | 0.811191797 | 0.749122279 | 0.312265068 | 2.85923195 | 0.634746611 | 1.86174667 | 0.674887154 |
| train/ufo/target_Q | 500 | 25 | 524 | -3.30697393 | -4.0848856 | -2.50425991 | -4.18396902 | -1.37255836 | -2.38769221 | -1.45240638 | -3.93685606 |
| train/ufo/target_auxQ | 500 | 25 | 524 | -4.56019115 | -6.00156736 | -4.88306379 | -6.10711479 | -3.96630621 | -4.79409218 | -4.01337295 | -5.94793417 |
| train/ufo/terminated_transitions | 32 | 40 | 524 | 634 | 1947 | 2237.59375 | 634 | 3377 | 2209.5 | 2888.7 | 2237.59375 |
| train/ufo/truncated_transitions | 32 | 40 | 524 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| train/ufo/unc_Q | 500 | 25 | 524 | 2.91390181 | 0.140243024 | 0.586248656 | 0.135703444 | 3.06919289 | 0.301942229 | 2.35188859 | 0.145043054 |
| train/ufo/unc_auxQ | 500 | 25 | 524 | 1.08624637 | 0.155932218 | 0.348643591 | 0.147445023 | 1.09713876 | 0.259164244 | 0.867743284 | 0.154410634 |
| train/ufo/wall_time_s | 500 | 25 | 524 | 45.8407326 | 1010.81592 | 521.581778 | 45.8407326 | 1010.81592 | 518.680847 | 953.526028 | 957.870042 |
| train/ufo/z_norm | 500 | 25 | 524 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 |

## 全部指标五个100更新窗口


| tag | 25–124 | 125–224 | 225–324 | 325–424 | 425–524 |
|---|---|---|---|---|---|
| env/episode_len | 7.47538356 | 9.45686754 | 10.4291836 | 11.182853 | 11.8562354 |
| env/num_trajectories | 2309.83333 | 2458.33333 | 2280.66667 | 2163.71429 | 2023.42857 |
| env/return | 0 | 0 | 0 | 0 | 0 |
| env/reward | 0 | 0 | 0 | 0 | 0 |
| env/success_once | 0 | 0 | 0 | 0 | 0 |
| rollout/ufo/received_transitions | 24576 | 24576 | 24576 | 24576 | 24576 |
| time/actor_training | 12.2715693 | 11.6030273 | 11.6099776 | 11.6165393 | 10.4606893 |
| time/cal_adv_and_returns | 0.00636593501 | 0.0111596187 | 0.00997102261 | 0.0128797463 | 0.0118745055 |
| time/env/compute_bootstrap_rewards | 0.000239799381 | 0.000240666637 | 0.000229700274 | 0.000228195917 | 0.00024049177 |
| time/env/env/bootstrap_step | 0.00262259804 | 0.000277223337 | 0.000244198328 | 0.00026718557 | 0.00025217734 |
| time/env/env/send_rollout_trajectories | 0.241891655 | 0.230247604 | 0.22578576 | 0.233690762 | 0.22520811 |
| time/env/env_interact_step | 1.44619586 | 1.17102301 | 1.35352153 | 1.33647464 | 1.34931476 |
| time/env/interact | 2.49550599 | 2.09298704 | 2.17189519 | 2.12051582 | 2.14615147 |
| time/env/run_interact_once | 2.49547617 | 2.09296117 | 2.1718694 | 2.12048864 | 2.14612607 |
| time/generate_rollouts | 2.71428645 | 2.3172971 | 2.37417261 | 2.33183197 | 2.35426821 |
| time/rollout/generate_one_epoch | 2.24675832 | 1.84440269 | 1.94178593 | 1.88959137 | 1.91536379 |
| time/rollout/predict | 0.0861262207 | 0.0418328426 | 0.0406211137 | 0.0395603175 | 0.0396825778 |
| time/rollout/rollout/generate | 2.24752901 | 1.84452486 | 1.94190075 | 1.88970963 | 1.91557988 |
| time/rollout/sync_model_from_actor | 13.5606931 | 13.2432758 | 13.2178354 | 13.2394732 | 13.2500474 |
| time/step | 32.2782199 | 30.876764 | 31.0043939 | 30.9649211 | 29.8187656 |
| time/sync_weights | 17.2854544 | 16.9447056 | 17.0097602 | 17.0031338 | 16.9914382 |
| train/ufo/B | -0.0134612141 | -0.00528794667 | -0.00444431347 | -0.00391596285 | -0.00259668004 |
| train/ufo/B_norm | 16 | 16 | 16 | 16 | 16 |
| train/ufo/F1 | 0.0362850302 | 0.0228112639 | 0.0151506578 | 0.015219838 | 0.0237192606 |
| train/ufo/M1 | 2.22441707 | 1.10459314 | 0.742859663 | 0.835612036 | 0.867044175 |
| train/ufo/Q1 | -2.21494297 | -1.51640603 | -2.11428422 | -2.92846757 | -3.72258417 |
| train/ufo/Q_aux | -4.24566365 | -4.30297378 | -4.84188996 | -5.42193595 | -5.92185159 |
| train/ufo/Q_discriminator | -2.28996708 | -1.56096031 | -2.20865443 | -2.99744548 | -3.8212094 |
| train/ufo/Q_fb | 6.0361315 | 18.5281092 | 28.9202745 | 38.4668558 | 47.5125945 |
| train/ufo/action_abs_mean | 0.905625512 | 0.908509384 | 0.904324412 | 0.901963345 | 0.900835693 |
| train/ufo/action_saturation_fraction | 0.441285953 | 0.447390348 | 0.442359224 | 0.440003072 | 0.438382285 |
| train/ufo/actor_loss | -3.35089764 | -14.392709 | -21.4731441 | -26.6369724 | -32.0380953 |
| train/ufo/actor_optimizer/grad_clip_limit | 1 | 1 | 1 | 1 | 1 |
| train/ufo/actor_optimizer/grad_norm_after_clip_l2 | 0.999999539 | 0.999999727 | 0.999999761 | 0.999999802 | 0.999999812 |
| train/ufo/actor_optimizer/grad_norm_l2 | 2.28559998 | 3.751665 | 4.38418378 | 5.06446902 | 5.73821528 |
| train/ufo/actor_optimizer/lr | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 |
| train/ufo/actor_optimizer/parameter_delta_l2 | 0.370400155 | 0.334276229 | 0.329933833 | 0.33280991 | 0.34466124 |
| train/ufo/actor_reg_weight_applied | 13.9580604 | 25.0847302 | 35.7251604 | 45.6515776 | 49.9986901 |
| train/ufo/actor_reg_weight_raw | 13.9580604 | 25.0847302 | 35.7251604 | 45.7205904 | 55.3797988 |
| train/ufo/applied_action_abs_mean | 0.531591276 | 0.579303652 | 0.589137395 | 0.588733656 | 0.587651406 |
| train/ufo/applied_action_saturation_fraction | 0.10603399 | 0.150789859 | 0.164699388 | 0.16623702 | 0.163640182 |
| train/ufo/auxQ1 | -4.19567915 | -4.26565446 | -4.78978899 | -5.34092661 | -5.82776793 |
| train/ufo/aux_critic_loss | 1.46898656 | 1.10947863 | 1.20837368 | 1.37369835 | 1.52119002 |
| train/ufo/aux_critic_optimizer/grad_clip_limit | 5 | 5 | 5 | 5 | 5 |
| train/ufo/aux_critic_optimizer/grad_norm_after_clip_l2 | 4.89906503 | 4.87713706 | 4.63903552 | 4.8282541 | 4.77353352 |
| train/ufo/aux_critic_optimizer/grad_norm_l2 | 14.1791558 | 9.48526981 | 9.19126111 | 10.7563849 | 9.44041299 |
| train/ufo/aux_critic_optimizer/lr | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 |
| train/ufo/aux_critic_optimizer/parameter_delta_l2 | 0.316770244 | 0.400561146 | 0.483822794 | 0.477884843 | 0.507991662 |
| train/ufo/aux_rew/limits_dof_pos | 0.0700999688 | 0.134104514 | 0.16742266 | 0.188150109 | 0.20090686 |
| train/ufo/aux_rew/limits_dof_vel | 0.00674309921 | 0.0113372493 | 0.0122974664 | 0.0122330003 | 0.0119953417 |
| train/ufo/aux_rew/limits_torque | 0 | 0 | 0 | 0 | 0 |
| train/ufo/aux_rew/penalty_action_rate | 0.651722974 | 0.54423304 | 0.490696722 | 0.460930109 | 0.444673415 |
| train/ufo/aux_rew/penalty_ankle_roll | 0.0507658029 | 0.0933001156 | 0.116037904 | 0.130295029 | 0.140239664 |
| train/ufo/aux_rew/penalty_feet_ori | 0.093089098 | 0.107181773 | 0.119482616 | 0.128310334 | 0.135351683 |
| train/ufo/aux_rew/penalty_slippage | 0.355195549 | 0.333082074 | 0.320178499 | 0.307960053 | 0.305614998 |
| train/ufo/aux_rew/penalty_torques | 4638.22118 | 4929.63152 | 4850.23567 | 4756.49161 | 4680.06277 |
| train/ufo/aux_rew/penalty_undesired_contact | 0.00276367188 | 0.00407714844 | 0.00475097656 | 0.00505371094 | 0.00515625 |
| train/ufo/backward_optimizer/grad_clip_limit | 5 | 5 | 5 | 5 | 5 |
| train/ufo/backward_optimizer/grad_norm_after_clip_l2 | 4.99999992 | 5.00000003 | 4.99999997 | 4.99999999 | 4.99999999 |
| train/ufo/backward_optimizer/grad_norm_l2 | 150769.517 | 66051.4169 | 36359.7714 | 24047.6878 | 18125.554 |
| train/ufo/backward_optimizer/lr | 9.99999975e-06 | 9.99999975e-06 | 9.99999975e-06 | 9.99999975e-06 | 9.99999975e-06 |
| train/ufo/backward_optimizer/parameter_delta_l2 | 0.00407983884 | 0.00437900701 | 0.00414837638 | 0.00391930619 | 0.00360793971 |
| train/ufo/collected_transitions | 24576 | 24576 | 24576 | 24576 | 24576 |
| train/ufo/contact_termination_count | 188.166667 | 173.833333 | 205.333333 | 189.571429 | 162.428571 |
| train/ufo/critic_loss | 0.71755628 | 0.260926433 | 0.313359666 | 0.410810813 | 0.627521253 |
| train/ufo/critic_optimizer/grad_clip_limit | 5 | 5 | 5 | 5 | 5 |
| train/ufo/critic_optimizer/grad_norm_after_clip_l2 | 4.93262891 | 4.95407858 | 4.77197329 | 4.85377267 | 4.47827247 |
| train/ufo/critic_optimizer/grad_norm_l2 | 13.5906319 | 9.05033021 | 11.169804 | 10.3081403 | 11.3726349 |
| train/ufo/critic_optimizer/lr | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 |
| train/ufo/critic_optimizer/parameter_delta_l2 | 0.294531492 | 0.263081508 | 0.311709037 | 0.377970586 | 0.480445609 |
| train/ufo/disc_expert_loss | 0.437278292 | 0.324683145 | 0.284333271 | 0.270103066 | 0.249787969 |
| train/ufo/disc_loss | 2.51247793 | 1.48386845 | 1.08023359 | 0.939633409 | 0.839813167 |
| train/ufo/disc_reward_clip_fraction | 0 | 0.00258300781 | 0.0553808594 | 0.185791016 | 0.306655273 |
| train/ufo/disc_train_loss | 0.666165545 | 0.631725973 | 0.505630014 | 0.406410786 | 0.343189042 |
| train/ufo/disc_wgan_gp_loss | 0.140903409 | 0.0527459344 | 0.0290270302 | 0.026311956 | 0.0246836155 |
| train/ufo/discriminator_optimizer/grad_clip_limit | 5 | 5 | 5 | 5 | 5 |
| train/ufo/discriminator_optimizer/grad_norm_after_clip_l2 | 4.99999934 | 4.12557406 | 1.8644298 | 1.32409089 | 1.07369452 |
| train/ufo/discriminator_optimizer/grad_norm_l2 | 7.57855324 | 4.40566306 | 1.8644298 | 1.32409089 | 1.07369452 |
| train/ufo/discriminator_optimizer/lr | 3.00000011e-06 | 3.00000011e-06 | 3.00000011e-06 | 3.00000011e-06 | 3.00000011e-06 |
| train/ufo/discriminator_optimizer/parameter_delta_l2 | 0.00490987792 | 0.00475425316 | 0.00313008032 | 0.00270001747 | 0.00247504932 |
| train/ufo/environment_control_steps | 26 | 50 | 74 | 100 | 128 |
| train/ufo/expert_draws/g1 | 82226.6667 | 181020 | 278861.333 | 385026.286 | 497917.714 |
| train/ufo/expert_draws/official_rpo | 81613.3333 | 179428 | 278194.667 | 385021.714 | 497995.429 |
| train/ufo/fb_diag | -58.281833 | -136.292607 | -178.524933 | -199.401409 | -212.094325 |
| train/ufo/fb_loss | 108008.635 | 53891.5662 | 34650.0813 | 25133.7193 | 19367.3987 |
| train/ufo/fb_offdiag | 119.392432 | 131.594852 | 161.072999 | 189.204195 | 216.267848 |
| train/ufo/first_step_samples | 3223.16667 | 2477.5 | 2277.66667 | 2168 | 2030.42857 |
| train/ufo/first_step_terminated | 17 | 12.5 | 9.5 | 7.57142857 | 8 |
| train/ufo/foot_contact_peak_N | 15474.8735 | 18830.3988 | 15978.6029 | 18399.9766 | 14893.0866 |
| train/ufo/forward_optimizer/grad_clip_limit | 5 | 5 | 5 | 5 | 5 |
| train/ufo/forward_optimizer/grad_norm_after_clip_l2 | 4.99999996 | 5 | 5.00000002 | 5.00000005 | 4.99999997 |
| train/ufo/forward_optimizer/grad_norm_l2 | 635.228703 | 240.187057 | 150.821673 | 124.139044 | 114.60716 |
| train/ufo/forward_optimizer/lr | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 | 0.000300000014 |
| train/ufo/forward_optimizer/parameter_delta_l2 | 0.388457333 | 0.613478447 | 0.734988273 | 0.781438568 | 0.786980571 |
| train/ufo/hard_position_count | 379.666667 | 407.5 | 345.5 | 327 | 289.714286 |
| train/ufo/hard_torque_count | 0 | 0 | 0 | 0 | 0 |
| train/ufo/hard_velocity_count | 744.833333 | 703.5 | 656.833333 | 641.714286 | 598 |
| train/ufo/height_termination_count | 621.666667 | 819.333333 | 825.333333 | 844 | 903.142857 |
| train/ufo/learner_joint_updates | 74.5 | 174.5 | 274.5 | 374.5 | 474.5 |
| train/ufo/mean_aux_reward | -0.930349074 | -1.12462139 | -1.13775662 | -1.14409647 | -1.15123877 |
| train/ufo/mean_disc_reward | 0.0952740819 | -0.12548471 | -0.554411831 | -0.892331284 | -1.09222375 |
| train/ufo/mean_disc_reward_raw | 0.0952740819 | -0.125706186 | -0.571577414 | -0.984775124 | -1.30628165 |
| train/ufo/mean_next_Q | -1.59357101 | -1.22821751 | -1.54491701 | -2.13558904 | -2.82640676 |
| train/ufo/mean_next_auxQ | -3.13132149 | -3.2237085 | -3.89267812 | -4.53227539 | -5.04972139 |
| train/ufo/online_transitions | 163184.64 | 316538.88 | 469893.12 | 623247.36 | 777584.64 |
| train/ufo/orientation_termination_count | 546.333333 | 561 | 421.833333 | 317.571429 | 218.285714 |
| train/ufo/orth_loss | 1079.47525 | 538.962637 | 346.675331 | 251.439167 | 193.632252 |
| train/ufo/orth_loss_diag | -256 | -256 | -256 | -256 | -256 |
| train/ufo/orth_loss_offdiag | 1335.47525 | 794.962637 | 602.675331 | 507.439167 | 449.632252 |
| train/ufo/physics_peak_joint_velocity | 38.1357743 | 45.0387789 | 46.1418966 | 42.3435211 | 33.5003905 |
| train/ufo/q_loss | 0 | 0 | 0 | 0 | 0 |
| train/ufo/replay_draw_new_ratio | 1.00145505 | 1.17024642 | 1.22429011 | 1.2528155 | 1.26650587 |
| train/ufo/replay_draws | 163840 | 360448 | 557056 | 770048 | 995913.143 |
| train/ufo/replay_policy_age_max_updates | 80 | 176 | 272 | 376 | 486.285714 |
| train/ufo/replay_policy_age_min_updates | 16 | 16 | 16 | 16 | 14.2857143 |
| train/ufo/replay_retained_fraction | 1 | 1 | 1 | 1 | 1 |
| train/ufo/replay_size | 159744 | 307200 | 454656 | 614400 | 786432 |
| train/ufo/requested_action_abs_mean | 0.905625532 | 0.908509394 | 0.904324383 | 0.901963396 | 0.900835727 |
| train/ufo/requested_action_saturation_fraction | 0.441285953 | 0.447390348 | 0.442359224 | 0.440003072 | 0.438382285 |
| train/ufo/requested_applied_max_diff | 1.95267971 | 1.95276036 | 1.95721014 | 1.95410863 | 1.95888826 |
| train/ufo/reset_control_samples | 3223.16667 | 2477.5 | 2277.66667 | 2168 | 2030.42857 |
| train/ufo/reset_projection_outside_fraction | 0 | 0 | 0 | 0 | 0 |
| train/ufo/reset_projection_residual_max_rad | 1.78813934e-07 | 1.78813934e-07 | 1.78813934e-07 | 1.78813934e-07 | 1.78813934e-07 |
| train/ufo/reset_projection_residual_mean_rad | 6.4254993e-08 | 6.45921396e-08 | 6.45310022e-08 | 6.4577642e-08 | 6.52188896e-08 |
| train/ufo/reset_projection_residual_p95_rad | 1.1920929e-07 | 1.1920929e-07 | 1.1920929e-07 | 1.1920929e-07 | 1.1920929e-07 |
| train/ufo/reset_root_lift_max_m | 0.0219920222 | 0.0219920222 | 0.0219920222 | 0.0219920222 | 0.0219920222 |
| train/ufo/sampling_rounds | 6.64 | 12.88 | 19.12 | 25.36 | 31.64 |
| train/ufo/target_M | 1.35728357 | 0.571074336 | 0.494123623 | 0.645976776 | 0.677153089 |
| train/ufo/target_Q | -2.22394397 | -1.52115042 | -2.11592778 | -2.92930814 | -3.73096921 |
| train/ufo/target_auxQ | -4.19576042 | -4.25714802 | -4.78821894 | -5.3423034 | -5.83188816 |
| train/ufo/terminated_transitions | 2309.83333 | 2458.33333 | 2280.66667 | 2163.71429 | 2023.42857 |
| train/ufo/truncated_transitions | 0 | 0 | 0 | 0 | 0 |
| train/ufo/unc_Q | 1.74357115 | 0.532411802 | 0.303880679 | 0.200735126 | 0.150644525 |
| train/ufo/unc_auxQ | 0.735490377 | 0.405797108 | 0.260372851 | 0.184064189 | 0.157493429 |
| train/ufo/wall_time_s | 134.201199 | 327.726898 | 521.413982 | 715.168874 | 909.397938 |
| train/ufo/z_norm | 16 | 16 | 16 | 16 | 16 |

### 完整原生网络/优化配置

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

### 完整本体编码与H物理控制协议

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
    }
  }
}
```

### 完整解析后RLinf配置

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
      "contact_profile": "contact_stable_v8",
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
      "experiment_name": "RPO UFO contact v8 poststep1 500-update observation",
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
        "id": "RPO-UFO-Isaac-RSI-contact-v8",
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

冻结v2 codec中的physics_hz=200是原接口的历史基线描述，未原地修改已发布codec/hash；H实际执行频率以独立action_control.physics.physics_dt_s=0.0025与decimation=8及每条trace为准，即400Hz。H没有合格独立评估入口，不允许按该历史字段复用旧物理设置评估。

## 三轮复核、检查范围与结论


第一轮：本体动作/观察编码、train-only候选、不可变专家数据与真实足底碰撞几何。第二轮：4 GPU拓扑、每sampler2048、batch2048、真实500联合更新、16→24→524完整checkpoint和六Adam；速度与接触子步锁存防止监控漏报。第三轮：全标量/五窗、128列trace与replay逐项核对、latent、reset几何、存储预算和GPU释放。

首版容器平台回归540 passed/25 skipped；原生聚焦27 passed；scaled-counters测试先RED后GREEN（1 passed）；只读复核4 passed/1 skipped。时序行为测试RED→GREEN，相关原生回归36 passed。最终部署后的实际追加检查以下方日志为准，skip不算通过。配对GPU诊断与真实训练是另列运行证据，不能用CPU测试替代物理效果。未提交代码。

容器检查 contact-poststep-native-tests.log：
```text
....................................                                     [100%]
=============================== warnings summary ===============================
../isaaclab_extension_template/.venv/lib/python3.11/site-packages/_pytest/config/__init__.py:1428
  /workspace/isaaclab_extension_template/.venv/lib/python3.11/site-packages/_pytest/config/__init__.py:1428: PytestConfigWarning: Unknown config option: asyncio_mode
  
    self._warn_or_fail_if_strict(f"Unknown config option: {key}\n")

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
36 passed, 1 warning in 12.43s

```

容器检查 contact-poststep-platform-tests.log：
```text
.........................ss.s.....ss......................ssss.....s.s.. [ 12%]
.s.s.................................................................... [ 25%]
........................................................................ [ 38%]
........................................................................ [ 51%]
........................................................................ [ 64%]
...........................................s.....s..................s... [ 77%]
........................................................................ [ 90%]
....................................................                     [100%]
=============================== warnings summary ===============================
../../app/backend/.venv/lib/python3.11/site-packages/passlib/utils/__init__.py:854
  /app/backend/.venv/lib/python3.11/site-packages/passlib/utils/__init__.py:854: DeprecationWarning: 'crypt' is deprecated and slated for removal in Python 3.13
    from crypt import crypt as _crypt

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
540 passed, 25 skipped, 1 warning in 5.39s

```

容器检查 contact-promoted-tests.log：
```text
.........................s                                               [100%]
=============================== warnings summary ===============================
tests/unit/test_training_recipe.py::test_untrained_baseline_is_zero_update_qualification_only
  /app/backend/.venv/lib/python3.11/site-packages/passlib/utils/__init__.py:854: DeprecationWarning: 'crypt' is deprecated and slated for removal in Python 3.13
    from crypt import crypt as _crypt

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
25 passed, 1 skipped, 1 warning in 1.70s

```

结束冻结500更新观测，不自动追加长训。训练仍未达到持续100步门槛，按用户要求不独立评估。下一段应依据本报告的首步原因、子步越限、PD target/torque及接触峰证据，继续区分RSI动态可行性、接触求解和控制激励；本次不能宣称68.68rad/s根因已唯一定位，也不能宣称正常行走或真实机器人安全通过。

下一步推荐（本轮未实施）：优先把速度软罚与真实物理子步越限对齐，验证每关节子步峰值/积分罚与末边界罚的差异，避免“早已超速、末步回落”没有足够软制动信号；在保留原始q/dq的前提下核对候选足底法向初速度与连续落足动态，并用有界配对实验比较更保守的PD target slew、接触设置或隐式PD，不把增益/惯量当未经验证的修复；动力学尾部改善后增加在线采样跨度。本段6144环境每环境仅128新控制步=2.56秒，单凭500更新与下降的FB loss不能说明已获得足够行走探索。

