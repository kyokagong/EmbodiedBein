# RPO 单卡教师训练与录像诊断结果

> 用户后续指令禁止继续教师训练。本文保留实际历史结果，末尾教师v2及新增教师预算建议不再执行；当前按[原始参考重审](2026-10-09-bfm-reference-reassessment.md)推进。

单卡 RLinf 教师已跑通，完成 **416 个 PPO round、5,111,808 条 transition**，保存最终 checkpoint。预算内独立录像验证已执行 seed142 的四个三秒片段，**1/4通过**。计划共12case；seed143/144未完成，共8case没有结果。教师未通过正式动作数据资格，**未启动四卡 BFM1000 联合更新、未晋升 DatasetVersion**。

本报告是数据生产教师的诊断结果，不是 BFM 正式模型验收。已执行case的通过率25%，完整套件通过率不可计算；缺失case不算成功。

## 执行与模型证据

| 项目 | 实际值 |
|---|---|
| 训练 | `trn_61ab30c9a68a4ad39266` / `job_5f8348f9c42942be9f2e` / `run_4007d1abdd714446b8a0` |
| 状态与计数 | succeeded；5000为预算上界，2900秒训练循环主动保存退出，实际416 round |
| 计算 | 单RTX4090，512env，50Hz控制/200Hz物理；RLinf PPO、Actor547/Critic905、ELU512/256/128 |
| 稳定性 | 最终25个权重/统计tensor全部finite，actor/critic normalization count=5,111,808，std最小.039400 |
| 权重 SHA256 | `aab4b2a722a89a8ccba072d4f203167cd1dfab467ab0c360c83dfff44baae487` |
| 模型包 | `/data/kk/models/runs/run_4007d1abdd714446b8a0`，独立`mf_rpo_v2_tracking_teacher`，未声明质量/硬件合格 |
| 评估 | `job_2968bd09853a4daaa6d8` / `run_dec4741f49ab46a196b1`；单GPU、三种子分别新建进程/物理环境、实际startup参数witness |
| 评估终态 | failed：`Job exceeded timeout limit of 0.1h`；首轮带相机冷启动耗时较长，4case完整回执和录像保留 |
| 单卡预算 | 首个取消smoke109.5s + 成功smoke99.4s + 有限训练2993.6s + 验证361.6s = **3564.1s，59分24秒**（Run起止时间，包含启动/退出） |

![真实训练曲线](artifacts/2026-10-09-rpo-tracking-teacher/training.png)

曲线持续改善，但训练随机起始参考帧，episode长度不能代替从片段首帧开始的完整独立质量检查。此处416是教师PPO轮数，BFM新增联合更新为0。

## 已执行seed142结果

原门限保持不变：joint RMSE≤.15rad，root位置RMSE≤.15m，root角度RMSE≤.25rad，接触足速RMS≤.30m/s；不得提前跌倒、硬位置越限或非有限。每片段150帧，实际可执行149个控制步。

| 片段 | 步数 | q RMSE rad | root RMSE m | 角度RMSE rad | 接触足速RMS m/s | 硬越限帧 | 判定 |
|---|---:|---:|---:|---:|---:|---:|---|
| 站立 ID |149/149|.0586|.0254|.0489|.2339|0|通过 |
| 前行 FW |117/149|.1064|.3312|.2905|.3154|0|跌倒，最低base高度.4778m；位置/角度/足速也超标 |
| 后退 BW |149/149|.0843|.2139|.1469|.3771|1|位置、足速、硬越限失败 |
| 侧步 SW |149/149|.0839|.2544|.1453|.2518|1|位置、硬越限失败 |

四case状态均有限，初始MJCF/Isaac FK最大偏差约.001m。正常参考尾端truncation在原回执中记作`timeout`，149步且无termination的case按完成处理；FW是实际提前跌倒。

![资格指标及原门限](artifacts/2026-10-09-rpo-tracking-teacher/qualification.png)

## 可播放录像

四份录像已逐一核对SHA、实际解码、帧数与50fps。ID/BW/SW各150帧、3.0s；FW118帧、2.36s（包括初始帧）。下列路径为实际受管Run产物。

- [站立：通过](/data/kk/embodiedbein/runs/run_dec4741f49ab46a196b1/outputs/seed_142/Neutral_ID-seed-142.mp4)
- [前行：117步跌倒](/data/kk/embodiedbein/runs/run_dec4741f49ab46a196b1/outputs/seed_142/Neutral_FW-seed-142.mp4)
- [后退：位置/足速/越限失败](/data/kk/embodiedbein/runs/run_dec4741f49ab46a196b1/outputs/seed_142/Neutral_BW-seed-142.mp4)
- [侧步：位置/越限失败](/data/kk/embodiedbein/runs/run_dec4741f49ab46a196b1/outputs/seed_142/Neutral_SW-seed-142.mp4)

![站立最后一帧](artifacts/2026-10-09-rpo-tracking-teacher/stand-last-frame.png)
![前行最后一帧](artifacts/2026-10-09-rpo-tracking-teacher/forward-last-frame.png)

完整数字、缺失case清单和视频hash见[evaluation.json](artifacts/2026-10-09-rpo-tracking-teacher/evaluation.json)，实际曲线与模型版本见[metrics.json](artifacts/2026-10-09-rpo-tracking-teacher/metrics.json)。源回执、轨迹和startup物理参数位于评估Run的`outputs/seed_142/`，未修改原件。

## 多轮复核与下一阶段

1. 管线：完整训练经过API→Job/Run→ebd→RLinf；首个CPU/CUDA入口错误已修复，完整PPO checkpoint/normalization验证通过，无Git提交。
2. 输入与算法：接口547/905、参考和权重hash、残差logprob与实际动作区分、history和真实terminal语义检查通过；独立审查两项重要问题（startup seed不独立、非有限失败回执丢失）已修复，77项相关测试通过。
3. 质量：已执行case的关节误差全部合格，行走的整体根运动、平衡和足部约束仍失败；不能只凭q跟踪或正训练回报晋升数据。
4. 覆盖与预算：仅4/12case完成；标准通过率不发布。评估冷启动计入单卡预算，未延长作业或自动重试。原设计16的数据资格条件未达到，四卡容量/1000更新/正式BFM100case验收仍未执行。

推荐后续先派生教师v2：改用heading坐标下的根误差/速度目标，补充接触足滑与足端跟踪约束，增加根运动和软限位约束的权重，保留原23关节增益及真实物理。坐标语义变化派生新接口/环境版本，不能改已发布v1。先有限smoke与权重迁移验证，再追加教师训练，最终用新的独立seed完整验证。获得合格行走后再采集真实prior、验证祖先分组与来源、恢复原计划的四卡BFM容量探测和1000联合更新。新增教师GPU预算需要用户确认；不能挪用已授权的正式BFM预算。
