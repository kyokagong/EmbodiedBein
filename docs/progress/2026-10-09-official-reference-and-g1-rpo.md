# 官方 RPO 直接参考接入与 G1→RPO 数值试验

日期：2026-10-09。设计依据：[18](../design/18-bfm-reference-prior-and-g1-rpo-retarget.md)。执行计划：[官方参考与G1试验](../superpowers/plans/2026-10-09-official-reference-and-g1-rpo.md)。本次不训练教师、不提交 Git；最终学习仍通过平台 API→Job/Run→ebd→RLinf。

## 官方 RPO：直接作为 observation-only 候选

采用冻结的官方 `rpo_lab` 池，revision `6b1c3d9988497c8961dcba77892de32edc1770e1`。新 `reference_pool` 不使用 PD 输出范围筛选，不要求教师；保留完整时间轴，按实际训练 URDF 限位、速度、直立范围、FK与碰撞筛查生成有效窗口。源路径与资产只读。

| 项 | 实测结果 |
|---|---|
| DatasetVersion | `ds_11ba2ce4b69a4fc5a27d` |
| 转换 Job / Run | `job_d8ba312e5d9d49408ef8` / `run_6e2433f8649b4a388722` |
| converter / 镜像 | `isaac-fk-reference-pool-wxyz-v5` / `eb/rlinf:rpo-v2-bfm-reference-v5` |
| 数据量 | 30片段、9431帧，50Hz |
| 几何检查前 / 后有效帧 | 9276 / **6135** |
| 连续9帧窗口起点 / 有窗口片段 | **5206 / 28** |
| 穿地拒绝 / 非相邻自碰撞拒绝 | 3155 / 0；穿地上限2cm |
| 最大原始穿地深度 | 7.861cm，拒绝帧仍保留并标无效 |
| Isaac FK 与 MuJoCo 独立对照 | 最大0.9963mm，小于2mm门槛 |
| 资格 | `prior_candidate_ready=true`、`training_ready=false`、`controller_evaluation_pass=false` |

`A1-_Stand_stageii` 与 `127_06` 没有可采样的连续窗口，不纳入实际序列抽样。资产回执 SHA256 `b58868a782d8863ac38059eb4eab89d27ee1d70565bf58f5a055f44510c8e842`；codec SHA256 `fff6cf3e8f05c88fc7b552f65e5089dc47315c343bd9a205a1f59539cb2811c4`；manifest SHA256 `4b9064d261b9d3679a7500d914fbf3799f2d8e03a7657eec8cd383c20be952f7`。

实际 v5 镜像 loader 抽样256个状态：policy52维、privileged358维；序列与 RNG 恢复检查通过。未知上游原始动捕权利仍记 `repository_license_only`；全池保守使用同一个未核实祖先组，仅 train，不宣称独立留出。邻接/两跳身体碰撞、精确三角网格交叠、接触时序尚未完整验证。

原始诊断数据 `ds_1bfbc60982fd4e878fa0` 的 manifest SHA256 保持 `000df06fe71ec6a4547353517f9c6533b79fef4996939995603df381ecf5966b`。原 BFM 镜像也未覆盖。独立复核发现 v4 自碰撞筛查修改 geom mask 后没有刷新 MuJoCo 编译后的 body mask；负例复现后修正并派生 v5。`ds_d4d8582779f846f28835` 保留历史证据，不能作为已通过自碰撞检查的数据。

## 官方候选的真实 RLinf smoke

| 项 | 实测结果 |
|---|---|
| TrainJob | `trn_e836786bf1764bc28229` |
| Job / Run | `job_80e0cc60ca074e159e4a` / `run_1818b594ee6c4f9a9054` |
| 配置 | 1GPU、4env、2次learner更新；原生完整BFM网络 |
| 结果 | succeeded；32 transitions；完整 step-2 checkpoint |
| 最后 actor / discriminator loss | 0.396745 / 9.460634 |
| 最后 FB / critic / auxiliary critic loss | 394909.71875 / 102.539856 / 121.598801 |
| 模型登记 | `mv_9df6e50a3d7b4d7e83bd`；登记不代表质量合格 |

指标均有限，FB loss 仍很大，仅证明接入与更新路径可执行，不能证明收敛。实际指标接口回读33条series，step0、1；checkpoint有完整优化器、target、replay、expert sampler与learner状态。未执行独立控制器评估。

镜像 digest `sha256:64ff744228d0c06e12b3ef3ba066278eff40ffc044533e23260cfcdb95f85b29`。RLinf源快照 `sha256:dbaa8bed56be412f6d012ee34030822941945786cede6bf9bc3de0fec5445d48`；受管补丁 `sha256:4c46bbaff5adccf33f967237229f5615a244894ced1d338c1f13035725a9fe62`。

回执、manifest、真实指标及checkpoint元数据保存在 [official-reference-v5](artifacts/2026-10-09-official-reference-v5/)。目录不含模型权重或 LAFAN 动作数组。

## G1：固定来源、真实导出与任务空间 IK

用户确认：仅本地非商业研究/学习，不分发派生数据。固定获取与处理均走受管作业，反序列化只在 hash 核验后的离线任务中执行。

| 阶段 | Job | Run | 结果 |
|---|---|---|---|
| 源获取 | `job_67079773b7ac4497ac95` | `run_da621bce53e44584965d` | succeeded，PKL及XML尺寸/hash完全匹配 |
| schema检查 | `job_40ec71736089491595da` | `run_421fb9c6108e490abd24` | succeeded，862片段，实际 `dof` 字段，30Hz |
| 短片段导出 | `job_380e2fa7a1e842e6897d` | `run_3e21b88107034457afbe` | succeeded，100帧×36，50Hz、xyzw、29个显式关节名 |
| 首次数值映射v1 | `job_dd5bf801f4964658b5f6` | `run_6b572abb088348769acf` | failed：NumPy1不支持上游 `np.asarray(copy=True)` |
| 修复运行时v3 | `job_e58fa69841624b31b243` | `run_49c149f4227d4fc4a3ef` | 数值Job成功，质量拒绝；关节打包顺序根因已定位 |
| 原生顺序修正版v4 | `job_6d6f8dd40d434ec1a171` | `run_a9e874ec0fec41ba9d7c` | succeeded；87有效帧、79连续窗口；仅参考候选 |

来源切片文件205117479字节，SHA256 `7f5aa36957808ee2e972472b18add8510533742710ba312d8b8c6e6014f1c010`，HF revision `40faaedd0d75a024685cfc48bd8eaff2f6772727`。G1 XML25377字节，SHA256 `579c8d3d98afa207cc816f1f6bdbe2594c6153cd00d6cdf644a102a4794b5920`，UFO revision `0b89378de02f77712072d983bc43ec2a46455dc0`。

选 `walk1_subject1_clip3` 的原始2.00–3.98秒；关节线性插值、根姿态SLERP。逐轴验证 `pose_aa[:,1:]=dof×hinge_axis` 后才接受字段别名；真实 `root_rot` 为xyzw，与根轴角最大误差1.61×10⁻⁷rad。导出源FK与原G1 XML误差约4.44×10⁻¹⁶m。

工具固定 hhtools `1181cc98593dbef20b23f92ebbf9ad38ba6a1e7e` CPU interaction-mesh，语义关键点、比例缩放、受限IK重新求解29→23，不删除6列。目标实际 RPO V2 资产，训练限位来自 URDF；源保留原XML碰撞基本体，目标保留官方网格。零关节参考姿态只是试验假设；跨机器人静态对齐及效应器语义偏移没有合格标定。

官方 hhtools 要求Python≥3.12；两次全依赖下载超时保留日志。实际本次只验证 Python3.11 固定CPU子集，补 yourdfpy0.0.60/lxml6.1.3/NumPy2.2.6，未验证整个UI/Newton应用。上游实际SQP次数为普通帧3、首帧5，`ik_iterations`不影响这个后端。

v3 Dataset `ds_175408346b9945e3aa96` 虽形状验证成功，但100帧中有效帧与窗口均为0：98帧超硬限位，41帧穿地超2cm、最大15.015cm；不允许进入BFM训练。定位上游 `pack_joint_q_csv` 按MJCF hinge声明顺序打包，而不是preset给出的名字顺序；目标原MJCF顺序与Lab训练顺序不同。v4让solver/preset使用实际MJCF顺序，随后按名字重排到训练顺序，保留v3拒绝证据。23个各不相同的关节值构成旧错读反例及修复检查。

### 最终 G1→RPO 诊断参考

新 DatasetVersion **`ds_47e5c031c49a4cd7855a`**，格式 `rpo_g1_joint_trajectory`；API已纳管为validated，`retarget_candidate_ready=true`、`training_ready=false`。文件 `/data/kk/data/ds_47e5c031c49a4cd7855a/content/rpo_g1_walk.npz`：100帧、50Hz、原生23关节、根wxyz、完整24身体FK及有限差分速度；未裁剪关节角。

| 检查 | 实测 |
|---|---|
| 全部帧硬限位 / 速度 / 直立范围 | 通过 |
| 几何有效帧 / 连续9帧起点 | **87 / 79** |
| 穿地拒绝 / 非相邻自碰撞拒绝 | 13 / 5，有重叠；拒绝帧保留并标无效 |
| 最大原始穿地 / 自碰撞 | 5.039 / 3.192cm |
| 源 URDF/MJCF FK / 目标 URDF/MJCF FK | 约4.44×10⁻¹⁶m / 0.996mm |
| 保存身体位置与原目标MJCF重算误差 | 约4.10×10⁻⁸m |

实际独立容器回读检查：NPZ hash、关节/body顺序、100×23/100×24×3维数、50Hz、有限值、单位根四元数、dq差分、全帧训练硬限位、87有效/79窗口，与receipt一致。源高度估计约1.419m（碰撞体近似），目标网格约1.063m；两种几何精度不同，身体比例和跨机器人静态标定仍未独立验收。

镜像 `eb/tools:rpo-g1-r2r-v4` digest `sha256:b870ab43a024092dfadead62656f5bfbdb52fe2dd0a6a472e6bd509297e5d856`。映射回执 SHA256 `8adcb02364cc0eb9828984b8ef35294ff809d7864d8eb536531732c7167ac12d`。只保存元数据的证据见 [g1-rpo-trial](artifacts/2026-10-09-g1-rpo-trial/)；原始PKL、单片段NPZ和派生运动不复制进仓库或分发。

这不是已接入BFM的最终视图：Isaac FK/52维policy与358维privileged codec、来源留出和正式训练资格尚未完成；本次不把原始G1诊断数据交给RLinf learner。完整hhtools应用、Newton后端和策略物理能力亦未验证。

## 复核与下一阶段

复核一核对源/hash/坐标与PKL信任边界；复核二核对碰撞mask、来源组与连续窗口；复核三由独立 reviewer 检查模板/probe/export版本兼容、标定未知项及实际hhtools打包顺序。修复派生版本，未降低限位/几何门槛。

最终聚焦容器pytest **47 passed**；真实几何反例、NumPy2接口/solver默认值、23关节打包反例与最终NPZ实际回读均通过。`git diff --check`通过。只验证本次范围，没有将全仓测试或正式模型验收写成通过。

最终独立review另行重算v4 NPZ的hash、关节/body顺序、时间轴、四元数、速度、训练硬限位、FK和几何mask，复现87有效帧/79连续窗口。未发现剩余阻断问题。79个窗口相互重叠，来自同一2秒动作，不是79条独立运动；不足以证明动作多样性或泛化能力。

本次正式四卡1000次learner更新、env容量摸底和策略可视化评估未执行。官方候选已完成有限RLinf接入试验；G1原始诊断参考还须合格标定、几何窗口与Isaac FK/codec转换，不是正式BFM行为先验。不得由Job succeeded或Dataset validated推出策略合格。
