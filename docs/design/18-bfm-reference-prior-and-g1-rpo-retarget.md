# 18 · BFM 原始运动先验与 G1→RPO 重映射

日期：2026-10-09。依据用户最新约束：重新审查官方 RPO 参考数据；**本阶段禁止训练教师，也不使用教师生成轨迹替代动捕参考**；调查 UFO 的 G1 LAFAN 数据如何重新映射到 RPO。最终训练仍为平台 API→Job/Run→ebd→RLinf。

本轮补充：用户确认先扩充多个G1动作、冻结分组，并要求同时使用官方RPO的30片段。派生混合参考recipe `isaac-fk-g1-rpo-mixed-pool-wxyz-v2`：40个motion/subject各固定8秒，同take所有subject同组；先固定15个take的train/validation/test再筛查。官方v5参考30片段全部train，保留未知原始祖先及使用条件说明，不用它们生成独立holdout。仅用于用户确认的本地非商业研究、不分发。mixed v1保留历史资格；v2对官方副本也应用根速度与根角速度异常mask，源版本不修改。

该实验的参考训练准入要求：所有片段源/校准/选择清单/官方manifest/资产/codec hash冻结；实际Isaac FK和独立MuJoCo FK误差≤2mm；硬限位、50rad/s关节速度、直立范围、2cm几何深度、连续9帧mask。G1至少5个可用train take、2000有效train帧及至少各1个可用validation/test take。新增root速度8m/s及角速度20rad/s异常筛查，报告低位足link-origin移动比例；此代理指标不宣称准确接触时序或物理足滑已验证。准入由受管转换及实际loader检查产生；旧单片段recipe继续training_ready=false，控制器资格另由独立物理评估产生。未经核验的原始官方运动权利不升级为商业或分发资格。

本轮网络使用bringup profile（1024宽、3层残差模块），正式联合expert/replay batch明确为128；两次smoke仍可用8。完整训练每轮4个控制步、1次联合更新，1000更新上限；容量探测采用相同128批次。更多环境改变采样/更新比，须报告实际数量。

评估派生专属Suite `suite_rpo_v2_bfm_isaac_velocity_commands_v2`：标准100个独立seed使用原Flat的完整指令、70/30地形、启动事件随机化与物理幅度的uniform传感噪声。噪声映射为BFM原生52维状态：q±0.03rad、dq±1.75rad/s、gravity±0.05、omega/4±0.05；同一episode/step重复读观测复用同一噪声测量，历史保存策略真实见过的测量。训练bringup原环境不因此改变。四种固定指令录像诊断继续使用平地和无噪声，资格不请求。BFM保持冻结codec的action gain与零执行延迟；这是新Suite与旧Flat直接数值比较的明确差别，不能声称通过旧Flat Suite。完整case计分沿用20秒、原机械限位、0.2线速度/0.3yaw RMSE、至少80/100判据。

## 1. 纠正数据资格与控制器资格的混用

[BFM-Zero 论文](https://arxiv.org/html/2511.04131v1) §3 使用重定向到 G1 的 LAFAN1；附录 B.1 将40条运动切成10秒片段，并按当前 BFM 的跟踪 EMD 更新采样优先级；算法1从 observation-only 参考序列初始化专家缓冲区。公开流程没有先训练 PPO 教师，也没有要求固定 PD 控制器逐条通过后才允许参考状态进入 BFM。

因此，设计16原先把 `dynamic_replay_verified=true` 当作全部行为先验的必要条件，是额外且过严的工程假设。本修订区分三类证据：

| 证据 | 必须核对的内容 | 能证明什么 |
|---|---|---|
| 参考数据资格 | 固定来源/内容/使用条件；原生23关节、body顺序、坐标、频率；有限值、四元数、FK与速度一致性；硬限位、明显穿地/自碰撞和异常跳变；连续有效窗口 | 这份数据可以作为 RPO observation-only 参考候选 |
| 采样与训练资格 | 冻结祖先分组及 train/validation/test；目标 loader 能消费；网络/codec/资产一致；有限 RLinf BFM 更新正常 | 对指定实验可消费，不代表策略已经学会动作 |
| 控制器能力 | BFM 在真实物理中的独立行走、跟踪、限位、足滑及录像指标 | 最终策略的实际质量，不能由静态数据检查代替 |

PD 开环失败继续保留为诊断证据，但不能单独否定动捕参考。也不能因为取消 PD 门槛，就接受错坐标、硬限位失配、无来源或没有连续窗口的数据。参考运动中足滑/接触时序的异常仍需分析；与某个控制器的滑动结果分别记录。

`prior_reference_ready` 与 `controller_evaluation_pass` 必须分别表示。正式 `training_ready` 仍须由受管验证回执和被冻结的来源/分组资格产生；本次研究报告不能直接修改已发布 DatasetVersion 的状态或 manifest。运行时实现该修订时派生新 converter/validation recipe；旧版本及27/291等历史结果保持原样。

## 2. 官方 RPO 数据优先路线

固定本机官方资产 revision `6b1c3d9988497c8961dcba77892de32edc1770e1`。审查：

- `robolab/data/motions/rpo_lab`：已转换为 Lab 关节序及 wxyz 根四元数；优先回读修正后的30片段 BFM 视图，逐帧区分硬限位、速度、直立任务范围和 PD 目标范围的排除原因。
- `rpo_gmr`：GMR关节序、xyzw；通过官方 `scripts/tools/retarget/config/rpo.yaml` 建立关节名映射，与 `rpo_lab` 对应片段核对。对应文件不能当成新增独立来源。
- `rpo_bm`：检查 NPZ 的关节/身体维数、频率、坐标、body元数据和FK。文件中有 body 状态不代表已知 body 顺序；缺少映射时须重建 RPO FK，不能默认沿用数组下标。

关节硬限位以当前训练绑定的URDF/实际Isaac probe为准；MJCF可作独立FK对照，但不能用其更宽限位替代训练侧约束。官方GMR→Lab转换还将左右 `elbow_yaw` 清零，不能仅按列排序就宣称两个目录的姿态完全相同。BM已有身体速度与link-origin位置差分的偏差需记录并重建，不能因为字段同名而直接复用。

不改原始文件、不裁剪关节角来制造通过率。`_M`变体、重映射版及窗口必须与原始动作同祖先组。高度/倾角是当前直立任务的选择规则，不宣称是原论文所有行为的普遍要求。现有保守 mask 不原地放宽。

直接路线：官方参考→RPO坐标/关节对齐→RPO FK及速度→冻结状态序列→RLinf BFM。没有教师阶段。

## 3. UFO 的能力边界

[UFO Robot Config 文档](https://github.com/Roboparty/UFO/blob/main/docs/robot_config_training.md) 明确不执行 retargeting。[Import Wizard](https://github.com/Roboparty/UFO/blob/main/docs/import_wizard.md) 的 `robot_inspect/data_inspect/data_build` 面向已适配目标机器人的 RobotState；换 `--robot-config` 不会把29维G1动作变成23维RPO。

[下载脚本](https://github.com/Roboparty/UFO/blob/main/scripts/download_data.sh)提供 `xuewang/UFO-MotionData` 的两个 G1 文件及预期SHA256：

- `g1/lafan_29dof_10s-clipped.pkl`：`7f5aa36957808ee2e972472b18add8510533742710ba312d8b8c6e6014f1c010`。
- `g1/lafan_29dof.pkl`：`f3a0c2810363f5c50bf4146fa2db33c1ff5b90d00cb7c0bc2aa4622696375e11`。

这些是下载入口中的身份要求；字节身份必须由受管获取回执验证。完整运动与10秒切片通常具有共同祖先，文件名称本身不构成独立 train/test 划分。

## 4. 无教师 G1→RPO 路线

复用官方 [hhtools R2R](https://github.com/Roboparty/human-humanoid-tools/blob/main/hhtools/retarget/robot_to_robot.py) 和 [batch_r2r_retarget](https://github.com/Roboparty/human-humanoid-tools/blob/main/scripts/batch_r2r_retarget.py)，不另造通用重映射框架：

```text
冻结的G1 LAFAN PKL + 对应G1 XML/URDF/关节顺序
→ 每条运动导出 root_xyz、root_xyzw、29维dof、真实fps及关节名
→ G1 FK / 语义关键点 / 身体比例和目标静态姿态校准
→ RPO23关节受限IK（hhtools R2R）
→ RPO原生关节序、wxyz、50Hz / RPO FK / 速度
→ observation-only数据检查与冻结分组
→ RLinf BFM 在线学习
```

UFO `MotionLibBase` 读取的是多运动字典，使用 `root_trans_offset`、`pose_aa`、`fps`；若有 `dof_pos`，直接用它覆盖FK的关节位置。hhtools R2R输入是单片段 `joint_q=(F,7+N)`，或 `root_pos/root_rot/dof_pos` 格式，**不能把UFO聚合PKL原样交给hhtools**。

桥接的必要检查：

1. 在隔离任务中固定G1文件hash后枚举真实字段；优先使用经关节名校验的 `dof_pos`。没有该字段时依照源模型hinge轴与 `pose_aa` 定义恢复角度，并用G1 FK对照；不能把轴角张量直接reshape成29维关节位置。
2. 从 `pose_aa[:,0]` / `root_trans_offset` 提取根姿态与位移时，核对源坐标和body定义；显式导出xyzw、fps、dof_names、原始clip ID、时间范围和来源hash。源G1的实际joint order以匹配XML为准，不猜URDF与MJCF顺序相同。
3. hhtools轻量PKL路径未显式解释 `root_quat_format`，优先导出带元数据的 `joint_q=(F,7+29)` NPZ/CSV；NPZ明确写 `dof_names`、`fps`、`root_quat_format`，不是任意带root/dof数组的RobotState NPZ，以避免wxyz静默误读及字段不兼容。
4. 目标绑定冻结的 RPO V2 23DoF URDF/XML，不因为预设名 `rp1` 相似就复用不同资产。标定文件派生到受管工作副本，并复核身体比例、静态姿态、肩/肘和足端语义；不在只读官方资产旁调用 `--init-zero-calibration`。
5. G1腰部和手臂比RPO多的自由度由任务空间IK重新求解，不能删6列或把其余23列简单改名。
6. 对转出的RPO轨迹重建完整FK/速度；参考动画与真实策略物理录像分开。参考坐标回放可用于几何核对，不冒充控制器成功。

hhtools要求Python≥3.12和其retarget依赖；当前BFM任务镜像不能据此假定已经具备运行条件。第一次实际映射应在独立受管工具镜像中只取1个短行走片段；固定依赖与校准后再扩大。来源条件未确认前不下载受限制的完整动捕库。UFO只提供数据/格式参考，最终训练栈保持RLinf。

## 5. 本阶段验收与复核

初次重审完成条件：记录官方RPO源和已转换视图的实际检查数字；明确候选、拒绝与未知项；核实UFO/hhtools输入边界；保存后续无教师路线。后续用户授权执行官方候选接入与一个 G1→RPO 数值试验，其实施边界见下一节。四卡1000更新与控制器质量仍另行验收。

反思一：取消教师和PD准入不等于取消数据校验。反思二：运动状态PKL与hhtools导出PKL不是同一schema。反思三：源/目标同为人形机器人，也不能删列复用关节角；源顺序、根坐标、body和身体比例都需要对齐。反思四：参考资格、训练可执行与模型评估合格分别由实际证据提升。

## 6. 授权实施边界

官方 `reference_pool` 派生独立版本：完整时间轴保留，训练 URDF 硬限位、速度、直立范围、独立 FK、地面及非相邻身体碰撞检查共同产生 valid mask，连续9帧才能形成8步状态序列。没有可靠祖先身份时，全池保守标为同一未核实祖先组，不构造伪独立验证集。参考候选可做有限 RLinf 接入试验，`training_ready` 和 `controller_evaluation_pass` 保持 false；正式实验还需冻结分组、来源资格及评估定义。

G1 仅处理用户确认的本地非商业、不分发派生数据用途。获取固定 hash 的切片库和 G1 XML；离线检查实际 `dof` 字段与每个 hinge 轴上的 `pose_aa` 一致后接受它。试验选择一个行走片段中的2秒，30Hz→50Hz：关节位置线性插值、根旋转 SLERP，输出100帧，不扩大到全库。

hhtools 固定 revision `1181cc98593dbef20b23f92ebbf9ad38ba6a1e7e`，CPU interaction-mesh 数值映射绑定实际 G1/RPO 模型及 RPO 训练 URDF 限位；源使用原 XML 碰撞基本体、目标保留官方网格。源/目标各自的 URDF/MJCF FK 对照不能替代跨机器人静态标定：本试验采用零关节参考姿态假设，`source_target_rest_pose_alignment` 和 `semantic_effector_offsets` 必须标为未验证，不能据此批量发布正式行为先验。

工具官方要求 Python≥3.12。独立 CPU 子集先在既有 Python3.11 科学运行时进行兼容试验，固定 NumPy2.2.6、yourdfpy0.0.60、lxml6.1.3；数值 Job 只能证明这一子集，不能宣称全应用或 Newton 后端可用。interaction-mesh 使用上游默认 SQP：普通帧3次、首帧5次，不把只作用于其他后端的 `ik_iterations` 当成实际次数。

试验输出格式 `rpo_g1_joint_trajectory` 是原生23关节/wxyz/50Hz的原始诊断参考及几何回执。它尚不是 `eb/rpo-bfm-motion/v1`：必须再经过 Isaac FK/codec 视图和训练资格检查，才能输入正式 RLinf BFM 训练。`retarget_candidate_ready` 只代表存在几何筛查后的连续窗口；`training_ready=false` 明确保留。失败输出、旧镜像和已发布版本不覆盖，修正派生新版本。最新证据在 progress，原始/派生 LAFAN 轨迹不进入文档仓库。

## 7. 标定与 codec 补齐

用户授权继续完成这两步。复用固定 hhtools 的受限链式标定助手与 SOMA 风格尺度/旋转/平移偏移；以 G1 零关节原生 FK 为参考，目标全23关节在实际训练URDF硬限位内求解。14关键点明确为源/目标原生 link origin，腕端槽代表最后一段手臂的关节原点，踝端槽代表踝关节原点；不等同手尖、脚底或抓取点，物理接触点对齐保持未验证。

验收：映射完整且无硬限位越界，8个手臂/腿骨段方向误差≤20°，左右镜像对称误差≤5°，左右踝高差≤1cm。另用独立FK核对每个原生关键点 `scale×source_relative_position + target_rotation×translation_offset = target_relative_position` 和 `source_rotation×rotation_offset = target_rotation`，最大位置闭合误差≤10µm、旋转≤0.001rad。闭合是标定变换一致性，不能代替身体方向检查或动态质量。

固定上游 interaction-mesh 默认只使用高度统一缩放，未消费逐关键点偏移。平台在该后端的 solver-target 构建步骤应用上述标定，根轨迹及地面缩放沿用上游；只支持本次无地形/交互物体的短片段。记录标定对求解目标的实际变化，未应用则失败。复用上游SQP，保留训练限位，关闭求解后额外hip clamp和仅用于viewer的踝overlay平移；不能裁剪最终关节角制造通过。

新 `POST /datasets/rpo-g1-bfm` 只接受受管、成功且已标定并实际应用标定、有连续窗口的原生RPO DatasetVersion。CPU重映射→GPU DATA_CONVERT：冻结retarget/calibration/NPZ/资产hash，实际Isaac articulation重建24身体FK、差分速度；复用同一个codec生成policy52维、privileged358维、6维reward features。源valid与训练Isaac限位/直立/速度/独立MuJoCo几何mask取交集；保留完整原始时间轴及拒绝帧，不跨洞采样。

新converter `isaac-fk-g1-calibrated-wxyz-v1` 绑定非商业本地用途、标定应用证据及来源组，派生immutable BFM视图与血缘；旧镜像拒绝新revision，新的专用RLinf镜像用于loader/有限更新验证。单动作100帧没有独立留出，这个有限配方始终 `training_ready=false`，不因伪造qualified/动态通过字段晋升正式训练资格。完成条件是实际codec抽样、RNG恢复、RLinf两次更新与完整checkpoint，不等于四卡1000更新或物理策略评估。
## 四卡共享场景评估补充（2026-10-09）

用户选择每卡一个仿真进程、多个独立 env。训练及其1000更新模型固定，评估派生 `suite_rpo_v2_bfm_isaac_velocity_commands_v3`，不修改已发布v2。每卡4env为本轮待实测档；四卡分片覆盖固定100种子，每个worker依批处理自己的25案例。

v3保留v2的任务分布、控制频率、20秒时长及80/100判据。每批共享官方GRAVEL网格与模型/资产；真实记录网格seed、各env地形位置。startup/reset按case seed独立调用，commands与sensor noise使用各自持续RNG；逐案例初态/轨迹/视频及源hash仍必需。共享网格与随机流绑定不同于v2逐例重建，必须以新协议报告，不能声称逐位相同随机轨迹。

已终止env从指标与录像采样中退出并冻结到初态；其他env继续真实物理step直到各自终止或1000控制步。先单卡4env检查数值/独立性/显存，再运行四卡；执行仍通过现有评估Job生命周期。完整100案例按同一分母重算，Job成功不等于模型合格。
