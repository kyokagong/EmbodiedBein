# 官方 RPO 原始参考重审与 UFO G1→RPO

本轮按用户最新指令停止教师分支。结论：**官方 RPO Lab 数据是直接 observation-only 参考候选，不能仅因固定 PD 跟踪失败而判定不可用。UFO 自身不执行重定向；可复用 hhtools R2R，但需要源状态导出桥接和目标校准。** 本轮未发布正式合格的新数据，未训练教师或BFM模型，没有Git提交。

设计：[18 · 原始参考与G1→RPO](../design/18-bfm-reference-prior-and-g1-rpo-retarget.md)。执行：[本轮计划](../superpowers/plans/2026-10-09-bfm-reference-reassessment.md)。详细逐文件数字与SHA：[reference-audit.json](artifacts/2026-10-09-bfm-reference-reassessment/reference-audit.json)；可复现审查脚本：[reference-audit.py](artifacts/2026-10-09-bfm-reference-reassessment/reference-audit.py)。

## 1. 实际执行与输入

开发维护CPU审查在镜像 `eb/rlinf:rpo-v2-bfm-zero` 中执行，image ID `sha256:397f8c1b840d746f3a15b0081d83f67a22b738145bca0428f1d79e07155bffdf`。network=none、2 CPU、4GiB内存、只读根文件系统；仓库/资产/既有数据只读，只有报告目录可写。没有GPU、梯度更新、数据下载或转换发布；本审查不是训练/转换Job。API/ebd容器健康，核查时没有运行的带 `eb.run_id` 标签任务，不需要取消新教师任务。

输入 `ds_1bfbc60982fd4e878fa0`，manifest SHA `000df06fe71ec6a4547353517f9c6533b79fef4996939995603df381ecf5966b`；资产manifest SHA `b58868a782d8863ac38059eb4eab89d27ee1d70565bf58f5a055f44510c8e842`。原始30文件SHA与inventory一致，全部转换NPZ与manifest SHA一致；没有复用旧错误四元数版本。

脚本挂载 `/repo`、`/assets`、`/view`、`/reports` 分别对应仓库、冻结RPO资产、既有数据content及本报告产物目录。初次容器因挂载遮盖镜像工作目录/虚拟环境未能启动，改用 `/repo`、`/tmp` 后解决，没有安装依赖；两次启动错误未执行数据审查，不是数据质量失败。

## 2. RPO Lab 的实际审查数字

所有30片段通过数组形状、有限值、四元数范数、50Hz时间轴检查。BFM `states` 的q/dq、身体线速度差分和358维privileged codec回读一致，最大差小于 `1e-4`。

| 项目 | 实际值 | 解释 |
|---|---:|---|
| 完整转换视图 |30 clips / 9431帧|没有新增或复制片段 |
| 既有保守mask |4351帧|包含PD目标范围，保持原样 |
| MJCF硬限位内 |9431帧|不能替代训练URDF限位 |
| 训练URDF硬限位内 |9277帧|与此前Isaac探针inventory汇总一致 |
| 去掉PD要求的静态候选 |9276帧|保留URDF限位、速度≤50rad/s、高度≥.5m、倾角≤.8rad |
| 位于至少9帧连续区间的帧 |9205帧|不是一条轨迹或新的独立动作量 |
| 连续9帧起点 |8765个|重叠窗口，不计为行为多样性 |
| 具有连续窗口的片段 |30个|原PD筛查为18个 |
| Lab抽帧FK |360帧|仅 `mj_forward`，没有控制器rollout |
| FK最大位置/姿态差 |.0009963m / .0000032rad|以固定RPO body名映射对照 |

9277→9276排除的是 `C3_-_run_stageii` 一帧约61.975rad/s的关节速度。50rad/s是当前静态筛查值，不是论文统一门槛，也不是全部关节已标定的硬件限值。

现有 `MotionView` 用原mask读取并采样16帧成功；另外在**内存副本**上应用新候选mask，通过连续8帧+下一帧的16帧采样断言。未修改原NPZ、mask或manifest。它证明采样器可处理候选，不证明新数据已发布或模型可跟踪。

## 3. 复核发现的资产与格式差异

初次使用MJCF得到9430帧候选；发现与旧inventory的硬限位统计不符后，按**实际训练URDF**重新计算为9276帧。JSON记录15处限制差异，包括髋roll/pitch、肩pitch/yaw、踝pitch、肘pitch等。没有使用较宽MJCF限位替代训练约束，也没有改任一模型。FK接近不代表限位/动力学一致。

26对GMR/Lab对应文件根位置完全相同，根旋转最大差约 `1.05e-7rad`。仅按joint name重排会留下1.57rad偏差；官方 `gmr_to_lab.py` 刻意将两个 `elbow_yaw` 清零，应用该规则后26对q差为0。因此两个目录不能计为新增独立样本，Lab中这部分肘yaw行为已丢失；`_M`变体同样需合并原始祖先组。

## 4. BeyondMimic 的可用内容与速度问题

三份均为50Hz、23关节/24body，全部数组有限，四元数范数误差小于 `8e-7`。已对全部1045帧静态FK，Lab顺序的数值假设与RPO相符；错误GMR顺序假设则有约.48–.71m偏差。

| 文件 | 帧数 | Lab顺序全帧FK最大位置差 | body线速度与位置差分最大差 |
|---|---:|---:|---:|
| getup_supin2prone.npz |225|.0009960m|1.5823m/s |
| yundong0.npz |435|.0009961m|.5530m/s |
| yundong1.npz |385|.0009962m|1.8512m/s |

官方 `scripts/tools/beyondmimic/csv_to_npz.py` 将输入SDK顺序映射到 `robot.data.joint_pos` 后记录body数组，支持Lab解释；NPZ自身无名字元数据，正式转换仍需冻结mapping回执。`joint_vel` 与中央差分最大差小于 `2e-6rad/s`；body速度明显不一致。生成器记录 `body_lin_vel_w`，BFM要求link-origin速度；本次只确认偏差，没有断言全部来自单一COM/滤波原因。导入时应重建link-origin线/角速度，不能直接复制旧body速度。

起身是地面行为，是否纳入直立行走课程需明确，不能普遍视为坏数据。三份尚未逐片段可视化，不从 `yundong0/1` 名字猜行为。

## 5. UFO重映射路线

实查：[UFO Robot Config](https://github.com/Roboparty/UFO/blob/main/docs/robot_config_training.md)明确不执行retargeting；[Import Wizard](https://github.com/Roboparty/UFO/blob/main/docs/import_wizard.md)只导入已适配目标的RobotState；[MotionLibBase](https://github.com/Roboparty/UFO/blob/main/humanoidverse/utils/motion_lib/motion_lib_base.py)使用聚合motion字典中的 `root_trans_offset/pose_aa/fps`，若有 `dof_pos`则覆盖FK关节状态。

[hhtools R2R](https://github.com/Roboparty/human-humanoid-tools/blob/main/hhtools/retarget/robot_to_robot.py)支持源FK→语义关键点→目标校准/IK。输入是单片段 `joint_q=(F,7+N)` 或特定root/dof格式，不能直接读取UFO聚合PKL。无教师路线为：

```text
G1 LAFAN + 匹配源模型 → 明确29关节/root/fps的单片段导出
→ hhtools G1 FK + RPO校准/受限IK
→ RPO23关节/50Hz/原生FK与速度 → 状态参考检查 → RLinf BFM
```

源 `pose_aa`不能直接reshape成29维角度；没有 `dof_pos`时需源hinge轴和FK对照。导出NPZ明确 `joint_q/dof_names/fps/root_quat_format`。目标绑定冻结RPO V2资产，不按 `rp1`名字猜本体。校准写受管副本，不在只读官方资产旁使用 `--init-zero-calibration`。29→23用任务空间IK，不能删6列。

hhtools要求Python≥3.12及retarget依赖，当前BFM镜像不能直接假定满足。UFO仅作数据/格式参考，训练仍是RLinf。下载入口/预期SHA见设计18；本次未下载真实G1数据或运行hhtools，因此尚未完成实际数值映射。

## 6. 进入下一阶段前的条件

现有manifest全部 `split=train`、`source_rights=repository_license_only`。来源使用资格、祖先分组、完整接触几何/自碰撞、全部Lab帧FK和BFM控制能力尚未完成。本轮只修订设计；生产validation仍是旧recipe，没有偷偷放宽代码或晋升旧版本。

后续顺序：派生reference-only converter/validation recipe并测试PD失败与坏参考的区别→固定来源/祖先split并发布新DatasetVersion→有限RLinf BFM单卡学习验证→单learner四卡容量/1000联合更新/独立评估。BM先重建body速度；参考与动作输出范围偏差继续记录，必要时派生新动作接口。G1作为扩充覆盖分支，明确用途后先做一个短片段的受管校准与映射试验。**全程没有教师训练。**

## 7. 多轮反思

1. 方法：原论文的状态参考与教师/PD控制器能力分开；教师被用户排除，设计14/16及历史教师入口说明已更新。
2. 数字：纠正MJCF9430到训练侧9276；区分连续帧9205和重叠窗口8765，不把“更多帧”当更多技能。
3. 格式：GMR肘yaw、BM速度、UFO/hhtools不同PKL都不能靠列数/字段名相似忽略。
4. 资格：没有将静态候选、loader消费或源码可行性冒充正式数据/模型通过；没有实际G1映射证据。

容器断言已通过，原始/已发布文件hash未改变，`git diff --check`通过。本轮未改产品Python代码，不为文档修改重跑训练或全量测试。
