# G1 标定与 BFM codec 实施计划

依据设计18及用户继续实施授权；沿用已有 API→DATA_CONVERT→ebd 和最终 RLinf。保留工作区、不提交 Git、不训练教师、不覆盖数据/镜像。

## Task 1：可复核标定

- [x] 为关键点尺度/平移/旋转闭合误差加入负例检查，容器观察 RED。
- [x] 复用固定 hhtools propose/assess；14关键点明确为原生link origin，不冒充手尖/脚底。
- [x] 保存全23关节、URDF硬限位、8骨段方向/对称/脚高及闭合误差；方向≤20°、对称≤5°、脚高差≤1cm、闭合位置≤10µm/旋转≤0.001rad。不合格不发布候选。
- [x] 新CPU镜像与受管单片段Job；记录标定后几何有效帧/窗口，原100帧保持原始时间轴。

## Task 2：Isaac FK / codec 新视图

- [x] RED：新converter revision及受管adapter输入绑定检查。
- [x] 在现有convert_official中增加G1已标定NPZ输入；复用实际Isaac FK/codec，不复制编码或复用MuJoCo身体旋转。
- [x] 固定来源、标定receipt/NPZ/资产hash、关节/body顺序、50Hz、原valid mask；质量mask取交集，不复活拒绝帧；不夹紧关节。
- [x] API新DatasetVersion与血缘；GPU1/32 FK环境容量；新不可变训练镜像及所有consumer兼容。

## Task 3：运行验证

- [x] 实际受管转换通过，loader抽样policy52/privileged358及RNG恢复；回读hash/FK/连续窗口。
- [x] RLinf 1卡4env/2次更新，完整checkpoint和真实metrics回读；不宣称收敛或正式1000训练。

## Task 4：多轮复核与交付

- [x] 复核来源/坐标/标定语义；复核mask/祖先分组/资格；最终独立review并修复重要问题。
- [x] 更新design18/progress及实际回执。单2秒动作仍无独立留出，training_ready=false；完成codec接入不等于物理控制器验收。

## 接口与预检

CPU retarget Job→新标定receipt→新G1 codec DATA_CONVERT，资产/hash一致；新revision需同时修改rpo_bfm validator、转换器、RLinf/export旧镜像拒绝及模板默认镜像。原official v5 converter/image保持可用。

Ruling：用户已明确继续标定与codec，不重复请求实施许可；原设计18说明两步必要检查。独立模型FK一致不能代替跨机器人标定，新增闭合检查不降低几何筛查。执行方法沿用native；技能的commit/worktree迁移步骤与用户要求冲突，保留本工作区并用计划复选框作为ledger。

Task 1 Ruling：控制面测试镜像不安装SciPy；闭合检查改用纯NumPy矩阵公式，运行时继续复用上游科学栈，不给控制面增加框架依赖。

Task 1 Ruling：固定上游CPU IM仅统一缩放、未使用逐关键点calibration；首次ds_8dd03987d4604aea99bb只证明文件标定，不允许进入codec。派生v7 CPU镜像，在solver target构建点使用SOMA标定偏移，复用原SQP；关闭UI overlay/额外hip clamp。实际ds_0ebd0bb63a044f018a51 / run_92ac20baa8d04a95ac30：目标变化0.170859m、100有效帧/92窗口，0穿地/自碰撞拒绝。

Pre-flight：新codec输入同时校验calibration verified/applied、receipt/NPZ/资产hash；只保留源mask交集。专用G1 revision始终不能晋升正式training_ready（此配方固定单动作100帧无独立留出）。模板/导出/probe使用新G1镜像；官方reference_pool转换仍用原v5镜像，不覆盖旧产物。

Task 2 complete：ds_dd2c1854e9fd4dc0912c / job_f59e884fa8e14bb8ae34 / run_7b35fa5c7225421a89f4真实Isaac转换成功，100有效帧/92窗口，FK max0.996064mm。实际G1 v3镜像loader抽样256×52/256×358和RNG恢复通过；未标定旧版本API请求409。

Final review：独立review无阻断缺陷，独立40项容器测试通过；另行回读新codec所有hash、q/timestamps/mask、loader形状/RNG，均一致。主执行器聚焦50项通过，最终版本42项回归通过。

Task 3 complete：trn_c7fa3159c9354d75bc47 / job_05f70238563d4270a128 / run_56f920caacd14d94b2a2 succeeded；2更新/32transitions，实际33指标series全有限，完整step2checkpoint与available trainer_state resume candidate。无实际续训/物理策略评估。

Task 4 complete：design18第7节与docs/progress/2026-10-09-g1-calibration-bfm-codec.md、metadata-only artifacts已更新；不复制派生运动/权重到仓库；git diff --check通过，未提交Git。
