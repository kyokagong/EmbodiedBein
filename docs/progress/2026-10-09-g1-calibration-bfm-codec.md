# G1→RPO 标定与 BFM codec 接入

日期：2026-10-09。依据[设计18](../design/18-bfm-reference-prior-and-g1-rpo-retarget.md)、[实施计划](../superpowers/plans/2026-10-09-g1-calibration-bfm-codec.md)。不训练教师，不提交 Git；源/旧版本只读，计算经平台 Job/Run/ebd，最终 learner 使用 RLinf。源与派生运动仅用于本地非商业研究，不分发。

## 标定结果与实际应用

固定 hhtools revision `1181cc98593dbef20b23f92ebbf9ad38ba6a1e7e`，复用受限链式姿态提议/评估与 SOMA 风格逐关键点尺度、旋转、平移偏移。14槽明确采用冻结源/目标原生link origin；腕槽为最后一段手臂关节原点、踝槽为踝原点，不是物理手尖或接触脚底。

| 检查 | 实测结果 |
|---|---|
| 标定关节 / 映射槽 | 全部23关节 / 14槽，完整且无硬限位越界 |
| 非零标定关节数 | 8 |
| 上臂 / 前臂方向误差 | 左右各7.645° / 15.000° |
| 大腿 / 小腿方向误差 | 左右各9.139° / 3.959° |
| 最大对称误差 / 左右踝高差 | 约10⁻⁶° / 1.836×10⁻⁷m |
| 逐槽FK位置闭合 / 旋转闭合 | 3.399×10⁻⁸m / 0rad |
| 验收门槛 | 骨段≤20°、对称≤5°、踝高差≤1cm；位置闭合≤10µm、旋转≤0.001rad |

没有把静态闭合等同动作质量。头/脊柱的三条合成骨段不参与上述8条原生链方向验收；身体比例的源高度来自原G1碰撞基本体、目标来自官方网格，源可视化网格未获取。

发现并处理的实际问题：上游 CPU interaction-mesh 默认统一缩放，未使用逐关键点标定。首次 `ds_8dd03987d4604aea99bb` 只有文件标定通过，不能进入新codec。平台派生 v7 CPU镜像，在solver target构建处应用SOMA标定，根轨迹/地面缩放仍沿用上游；复用原SQP，没有另写IK或夹紧结果。关闭额外hip clamp和viewer踝overlay平移。只支持这个无地形/交互物体的短片段。

最终标定原生数据 **`ds_0ebd0bb63a044f018a51`**：Job **`job_52fc8e029c27428ab129`**，Run **`run_92ac20baa8d04a95ac30`**，succeeded。保留 `walk1_subject1_clip3` 原始2.00–3.98秒，100帧/50Hz；求解目标最大变化 **0.170859m**，`calibration_applied_to_solver=true`。全部100帧满足训练硬限位/速度/直立/几何筛查，形成 **92个连续9帧窗口**；穿地最大约1.57×10⁻⁸m、非相邻自碰撞0，无拒绝帧。原v4未标定版本87有效/79窗口仍保留。

CPU镜像 `eb/tools:rpo-g1-r2r-v7` digest `sha256:efc39886cd86b711e5565056c002c114e92c6ccacfb2016b14526e37b9c49a88`。标定回执 SHA256 `fa8d06510782eab81b7c6f23ce04001b00453d14842d7a1f8c1971a765067fd4`，retarget回执 `6848fcb366c089e1ebd110519b0afe414fbfc901711b9bf1489a01661ad64c9d`。

## 实际 Isaac FK 与 codec

新入口 `POST /api/v1/datasets/rpo-g1-bfm`，输入 `source_dataset_id`，只接受成功受管、标定已验证且实际应用、有连续窗口的原生RPO数据。旧未标定 `ds_47e5c031c49a4cd7855a` 实际请求返回409；未改变其发布内容。

新 BFM DatasetVersion **`ds_dd2c1854e9fd4dc0912c`**：Job **`job_f59e884fa8e14bb8ae34`**，Run **`run_7b35fa5c7225421a89f4`**，succeeded并纳管validated。格式 `rpo_bfm_motion` / schema `eb/rpo-bfm-motion/v1` / converter `isaac-fk-g1-calibrated-wxyz-v1`。

沿用现有转换器，1GPU/32 FK环境；实际Isaac articulation按原生23关节重建24身体的姿态与位置，再计算link-origin线/角速度和同一BFM codec。没有复用MuJoCo身体旋转或另写编码；没有控制器/教师rollout。

| 检查 | 实测结果 |
|---|---|
| 数组维数 | policy100×52、privileged100×358、reward features100×6 |
| 时间轴 / 关节姿态 | 原100帧/50Hz、根wxyz及原生关节名；独立回读精确保留 |
| mask / 窗口 | 源mask与Isaac硬限位/速度/直立/独立几何mask交集；100有效/92窗口 |
| Isaac / MuJoCo 最大FK位置差 | **0.996064mm**，小于2mm门槛 |
| 实际镜像 loader | batch256 policy52/privileged358及next state，连续序列通过 |
| expert sampler恢复 | RNG恢复后所有返回数组完全一致 |
| 资格 | `smoke_ready=true`、`prior_candidate_ready=true`、`training_ready=false`、`controller_evaluation_pass=false` |

源组 `walk1_subject1` 仅train，不构造伪独立留出。视图绑定源/标定/retarget/NPZ/资产hash；新增Dataset血缘。此固定单动作100帧配方始终不能晋升正式训练资格，伪造qualified/动态通过字段也不能改变它。

资产 SHA256 `b58868a782d8863ac38059eb4eab89d27ee1d70565bf58f5a055f44510c8e842`；codec `fff6cf3e8f05c88fc7b552f65e5089dc47315c343bd9a205a1f59539cb2811c4`；manifest `9480164ee4e33ec6c523c1b9d267ea20c863ff30e0b926fdead45d04d4ebc39b`。

训练/codec镜像 **`eb/rlinf:rpo-v2-bfm-g1-v3`** digest `sha256:0403a69576f09025589466f68ca46edb61b1ead22c2297f634f4896e66f4ecf7`；RLinf源及受管补丁与此前官方候选smoke一致。模板默认、probe、export同步使用新镜像；新G1视图拒绝旧consumer镜像。官方reference_pool转换继续使用旧v5专用镜像，不覆盖它。

## RLinf 验证

TrainJob **`trn_c7fa3159c9354d75bc47`**，Job **`job_05f70238563d4270a128`**，Run **`run_56f920caacd14d94b2a2`**：1GPU/4env、2次learner更新，**succeeded**。模型自动登记 `mv_df1eb27a74df43e0b8e6`，登记不代表质量合格。

实际指标接口33条series、step0/1，独立容器回读所有数值有限；update_count=2、target_update_count=1、replay_size=32。最后actor loss=-1.194972、discriminator=9.606257、FB=314391.625、critic=125.472267、auxiliary critic=100.774773。FB loss仍大，只有两次更新，不宣称收敛。

实际checkpoint manifest完整step-2，full_weights435419333字节，SHA256 `53a0627c586d26bef464e3fc8ee0ef2a297289875b73762dd3638e24441e7157`；API发布checkpoint标complete=true。完整训练状态resume-manifest包含DCP optimizer、模型、scheduler/RNG覆盖，以及BFM target/replay/expert sampler状态文件。平台resume-candidate返回 **available=true、mode=trainer_state、step=2**；没有另外执行续训，仿真environment状态明确not_available。回执和真实指标已保存到本次证据目录。

## 复核与边界

主执行器聚焦容器测试50项通过，最终镜像版本相关42项回归通过；独立review的40项容器测试通过。独立review另行核对SOMA数学顺序、实际URDF限位进入MPC、MJCF打包→名字重排、来源/hash/坐标、codec状态/q/timestamps/mask和loader/RNG恢复，未发现阻断缺陷。API负例409和真实CPU/Isaac作业提供实现测试之外的运行证据。

最终 `git diff --check`通过；实施计划所有任务完成。CPU标定→实际Isaac codec→实际RLinf learner与checkpoint闭环已跑通，旧镜像与数据均保持不变。

回执和manifest见[本次证据](artifacts/2026-10-09-g1-calibration-codec/)，不包含源PKL、派生运动数组或模型权重。92窗口重叠并且来自一个两秒动作，不能视为92条独立运动。

尚未验证物理手尖/脚底对齐、相邻/两跳身体与精确三角网格交叠、接触时序、其他运动泛化、整个hhtools应用/Newton或真实策略质量。此次补齐标定和codec，不代表完成全动作库扩充、四卡1000更新或独立策略可视化评估。
