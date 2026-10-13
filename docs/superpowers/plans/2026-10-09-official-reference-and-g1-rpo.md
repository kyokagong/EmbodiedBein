# 官方参考与 G1→RPO 实施计划

> 执行方式：superpowers:executing-plans；按下面顺序执行，不提交 Git。

**Goal:** 派生官方 RPO observation-only 候选数据；准备并尝试一个有明确使用条件的 G1 LAFAN 短片段重映射。

**Architecture:** 复用 DATA_CONVERT/API/ebd/collector，新增 reference_pool 配方，保留旧配方。源与资产只读，输出新 DatasetVersion；G1 先严格导出 hhtools 兼容轨迹，再做目标资产绑定的 R2R。

**Spec:** docs/design/18-bfm-reference-prior-and-g1-rpo-retarget.md。

## 约束与范围

- 不训练教师，不提交代码；最终学习仍由 RLinf 执行。
- 参考候选、正式训练资格、控制器能力分别记录；未知权利和几何检查不能伪造通过。
- 官方转换修改 rpo_bfm.py、rpo_bfm_motion.py、routes_data.py、data_convert.py 与对应测试；任务镜像重新构建，控制面 make up。
- G1 桥接只处理固定 hash 文件，保留真实 fps、根姿态和关节名，不删列替代 IK；现有镜像缺 hhtools，先核对运行依赖。
- 外部 LAFAN 数据获取/处理等待用途确认；代码、接口和测试工作继续。

## Task 1：官方参考候选

- [x] RED：添加 reference_pool 路由/adapter、训练硬限位 mask、祖先分组、连续窗口检查；旧版本不变。
- [x] GREEN：新增配方与 metadata，不把 repository_license_only 改为 verified；不使用 PD 输出范围作 mask。
- [x] 容器执行相关 pytest；Expected: 新旧配方检查全部通过。
- [x] 构建 BFM 镜像、make up；通过 API 创建新数据作业。
- [x] 回读 Job/Run、manifest/回执、实际 sampler；Expected: 新数据可加载、原数据 hash 未变，准确报告有效帧、窗口及未知项。
- [x] 新候选完成2轮单卡 RLinf BFM smoke，验证实际 consumer 版本接入；不宣称1000更新或模型合格。

## Task 2：G1 桥接与短片段试验

- [x] RED→GREEN：聚合 UFO 字典到单片段 joint_q NPZ，拒绝错维数/错顺序/非有限值/缺 DOF 来源；根轴角转换 xyzw，显式 fps 和 dof_names。
- [x] 核对并冻结工具与源模型；Expected: 用真实 G1 FK 和目标 RPO23 资产，不能用 rp1 名称替代目标校验。
- [x] 来源用途确认后，受管获取 hash-pinned G1 文件，枚举字段；运行至多一个短片段 R2R，派生新诊断数据。
- [x] 若用途或工具/数据有实证阻塞，保存明确失败/阻塞回执，完成不依赖它的官方路线；不得宣称已创建真实 LAFAN 派生数据。

## Task 3：复核与交付

- [x] 复核一：来源/hash/坐标/关节与旧配方兼容。
- [x] 复核二：来源组/连续窗口/候选资格与控制器资格。
- [x] 最终独立 review，修复重大问题并重复相关检查。
- [x] 更新 docs/progress，记录真实 Dataset/Job/Run、数值与未完成项。

## Pre-flight

- Task 1 route → converter：reference_pool 名称、converter revision、profile argv 必须一致。
- converter → validator/loader/collector：沿用 eb/rpo-bfm-motion/v1；新 revision 增加，不改旧状态。
- Task 2 → hhtools：UFO 聚合 PKL 不能直接消费；单片段 NPZ 必须为 F×(7+29)、xyzw 与显式 DOF 顺序。
- Ruling：用户禁止提交，所有技能中的 commit 步骤跳过；本地现有改动属于本次连续任务，不迁移/重置工作区。
- Task 1 Ruling：首次 API 提交因 capacity 未列新 operation，误按33亿参数训练估算63.5GB而失败；新增 estimator 白名单与三配方回归检查，不绕过准入。
- Task 1 Ruling：自动审批拒绝覆盖旧镜像，使用新增 eb/rlinf:rpo-v2-bfm-reference-v4；旧标签不改。
- Final Ruling：独立 review 找到 compiled body collision mask 未更新；反例复现并修正，派生 v5 converter/image/Dataset，v4 保持不可变且不作为已通过自碰撞检查的数据。新配方消费入口同步覆盖训练模板、probe、export；旧镜像显式加载新 manifest 时拒绝。
- Final Ruling：G1 float64 有限值可能在 norm/cast 后溢出；增加合理范围与输出四元数/有限值检查，两个反例 RED→GREEN。
- Task 2 Ruling：用户已明确“仅本地非商业研究/学习，不分发派生数据”；固定205117479字节源与G1 XML已由受管Job获取/核验。
- Task 2 Ruling：真实UFO库为862片段，30Hz，字段为dof而非dof_pos；桥接逐关节核对pose_aa=轴×DOF后才接受别名。
- Task 2 Ruling：Python3.12全依赖下载两次超过8分钟；保留失败记录。改为固定hhtools CPU R2R子集的Python3.11兼容实验，仅补yourdfpy/lxml；两个CPU模块实际import通过，后续数值FK/R2R Job证明兼容范围，不声称全应用受支持。
- Task 2 Ruling：G1小样本先发布原生RPO23关节参考诊断数据，不伪装已完成BFM Isaac FK/codec正式视图；形状验证、参考窗口与训练资格分别记录。
- Task 2 Ruling：实际v1 Job暴露上游NumPy2接口要求；v3固定NumPy2.2.6、保留源碰撞基本体/目标网格及LICENSE/NOTICE。独立模型FK检查不代表跨机器人标定，标注零姿态试验假设。
- Task 2 Ruling：v3数值完成但无有效帧；实际23个不同关节值反例定位hhtools IM按MJCF声明顺序打包而忽略preset顺序。v4采用真实MJCF solver顺序、最后按名字转Lab/native23；不裁剪角度、不放宽质量门槛。
- Evidence：官方v5 `ds_11ba2ce4b69a4fc5a27d`：6135有效帧/5206窗口；RLinf `trn_e836786bf1764bc28229`：2更新/完整checkpoint。G1 v4 `ds_47e5c031c49a4cd7855a`：100帧/87有效/79窗口，候选true、训练ready=false；源/派生轨迹只存受管私有目录。详见progress同日文档。
