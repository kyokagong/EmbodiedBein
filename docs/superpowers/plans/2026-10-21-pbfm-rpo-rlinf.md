# P-BFM / RPO / RLinf 执行计划

> **For agentic workers:** 使用 `superpowers:subagent-driven-development` 按任务实施、测试和复核；所有步骤禁止 Git commit/push。任务间只通过本文约定接口协作。

**Goal:** 实现设计26的RPO深度地形训练功能，经现有Job/Run/ebd与真实RLinf Worker承载P-BFM原生Workspace，补齐数据、恢复、导出与独立评估。

**Architecture:** 算法保留P-BFM原生实现，受管补丁仅适配RPO语义。平台复用native_rl生命周期，以recipe分发消除native_rl必然等于UFO的假设。运行证据按设计A→F逐级获取，未验证能力保持不可用。

**Tech Stack:** Python3.11控制面、Python3.10任务、RLinf scheduler/Ray、P-BFM/Torch2.7/MJLab1.4/MuJoCo3.8、Rust ebd、Vue3。

**Spec:** [设计26](../../design/26-pbfm-rpo-rlinf-integration.md)。该文件是约束，不因实施困难降低设计要求。

## Global Constraints

- 不提交Git，不修改外部P-BFM/UFO/RLinf源码；本仓受管补丁应用到任务私有副本或构建镜像。
- 保留现有JobKind、TrainJob/Job/Run/Artifact/ModelVersion；执行经过ebd，控制面不运行训练。
- 新接口`ee_rpo_v2_pbfm_depth_23dof_v1`，模板`rpo_pbfm_fb_depth_native`，任务`tsk_rpo_pbfm_terrain_locomotion_v1`，模型族`mf_rpo_pbfm_depth`。
- 每GPU一个RLinf基础Worker，共置完整Workspace；原生阶段DDP，无嵌套launcher，无PPO/SAC替换。
- 23DOF、400Hz物理/50Hz策略；标量观测380，depth8×18×32，z256；Actor禁止privileged输入。
- `prior_plane_envs=0`，QV开启；scratch课程起点0，默认间隔19200000 global transitions。
- 阶段112→512/1008→5008/10000联合更新，分段退出/全rank保存/独立评估/决策/新Run恢复；专属`pbfm_native_state`，非exact。
- 只读冻结来源、数据split、robot、相机、patch和镜像digest；新能力未取得相应证据不能标validated。
- 所有代码测试/运行在容器内；用户本轮授权运行时探测＋112更新，总墙钟最多1小时（2026-10-21 08:47:32–09:47:32 UTC，含准备/快评/录像），不进入512/1008。

## Review Focus

1. 新native_rl模板不得被UFO专用服务、codec、checkpoint或collector截获；旧UFO行为回归。
2. G1限制替换后仍拒绝未知机器人/缺参；RPO camera与foot几何、动作delay前后与角速度scale不得混淆。
3. 计数、原生恢复的有效agent配置、全rank Replay与缺target/optimizer必须真实核验，失败禁止发布。
4. 预览/提交和评估plan必须绑定相同不可变输入；路径逃逸、symlink、半写、holdout渗入训练必须拒绝。
5. 容器、通信、视频和证据失败应可见并有界收尾；程序退出0不代表模型通过。

## Task 1：RPO受管源码适配、契约与组合镜像（设计A/B）

**Files:** 新建`deploy/patches/pbfm/`、`deploy/images/rlinf-pbfm.Dockerfile`及精确依赖文件、`backend/eb/runners/pbfm_source.py`；新增`backend/tests/unit/test_pbfm_source.py`与runtime测试。

**Interfaces:** 提供`prepare_pbfm_source(source: Path, destination: Path) -> dict`（校验冻结上游文件后复制并应用受管适配，返回来源/patch/结果manifest）；`rpo_robot_config(source: Path) -> Path`返回私有副本的RPO配置；完整配置见Task2。镜像`eb/rlinf-pbfm:rpo-depth-v1`。

- [x] 先写失败检查：原生RPO拒绝、G1保留、未知robot拒绝、字段完整、外部源不变、重复/上游漂移拒绝、动作/深度/资产语义；容器验证完成。
- [x] 使用容器运行并确认失败；实现完整来源manifest、确定性受管patch与RPO robot/Hydra及虚拟相机配置。足部接触与sole/edge由冻结RPO碰撞网格派生，保留G1默认。
- [x] 独立镜像固定Python3.10/Torch2.7.1+cu128/MJLab1.4.0/MuJoCo3.8.0与最小RLinf scheduler；217项基础锁及pandas补充锁实际构建、导入Workspace/MotionLib/depth exporter/Worker和pip check通过，不继承UFO运行结论。
- [x] 目标镜像CPU验证RobotSpec/FK、原生字典state52/history300/actor380与privileged FK358、23执行器/接触/sole配置、PD缩放、crop/history与G1回归；78项通过、1项GPU reset检查跳过，保留依赖失败与修复记录。
- [ ] 真实GPU reset/step/观测空间、物理PD target与相机画面仍须经Job/Run/ebd诊断；CPU结构检查不替代这项证据。

## Task 2：P-BFM原生训练plan、RLinf运行与完整checkpoint（设计B/C）

**Files:** 新建`backend/eb/pbfm_training.py`、`pbfm_training_checkpoint.py`、`runners/rlinf_pbfm_{entry,worker,runtime}.py`及对应unit/runtime测试。

**Interfaces:** `pbfm_training`提供`TEMPLATE/ENV_ID/TASK_ID/IMAGE/SCHEMA`、`training_fingerprint(identity: dict)->str`、`segment_endpoint(target_updates,world_size,*,start_updates=0,start_local_time=0,num_envs=1024)->int`、`validate_train_job(spec)->dict`、`render_training(spec,run_id)->RenderedJob`。Job参数使用`native_train_plan`，schema=`eb/pbfm-native-train-plan/v1`；identity含image_digest/world_size/num_envs_per_rank/buffer_size/seed/dataset_id/dataset_manifest/contract_hash/environment_profile/pbfm_snapshot/rlinf_snapshot/asset_snapshot/camera_contract/terrain_contract。`source_refs`的pbfm与rlinf均含path/snapshot_hash/manifest_path。输出`outputs/native/checkpoint`、`checkpoint-manifest.json`与`outputs/receipt.json`，schema前缀`eb/pbfm-*`。checkpoint提供`read_pbfm_checkpoint(run_root: Path,spec: JobSpec,run_id: str,verify_hashes=False)->dict|None`与段末发布入口。

- [x] 计数失败与通过测试完成：112冷启18432local，112→512恢复45056local；拒绝100、错world_size、dedicated prior和配置失配。
- [x] plan校验、只读preview render、资源画像与完整source/asset manifest字节及成员集合核验完成；容器覆盖mesh篡改/新增文件/未冻结输入拒绝。
- [ ] RLinf Worker直接构建原生Workspace、base seed+rank、私有Ray和失败收尾代码及导入检查已完成；真实单卡来源/观测/深度/RGB及单卡异常退出已有成功证据，1024环境与Workspace已构建；完整更新、多rank/NCCL和恢复尚未通过。
- [ ] 8优化器完整paramgroups/state/实际step、model严格key/shape/target/normalizer/有限性、每rank Replay回读与独立原子发布代码完成；CPU小状态检查已测，完整原生多rank保存/恢复尚未运行。
- [x] 新Run/只读source binding/指纹不变及无状态降级的协议检查完成；容器覆盖缺shard/半写/symlink/hash错/配置漂移。真实恢复首更新仍归Task6。
- [x] 容器聚焦30项与目标镜像CPU contract probe完成；曲线使用真实联合更新轴并保留缺值和非有限标记。此处不含GPU更新。
- [ ] GPU runtime probe和112联合更新预算已获授权；真实数据派生Job已通过，继续经Job/Run/ebd取得GPU与更新证据。

## Task 3：平台训练与恢复接线（设计C）

**Files:** 新建`backend/eb/pbfm_training_service.py`、`native_training.py`；修改`domain/rlinf_form.py`、`adapters/rlinf.py`、`api/routes_train.py`、`checkpoints.py`、`orchestrator/{service,collect}.py`、`environment_catalog.py`、`settings.py`、`capacity/estimator.py`；对应测试。

**Interfaces:** `native_training.service_for(template_id)`只允许已注册recipe，旧UFO直接保留原服务，P-BFM走独立服务；`pbfm_training_service.build_spec(body,session)->JobSpec`、`validate_submission(spec,session)`、`build_resume_source_binding(session,source_run,checkpoint)`。P-BFM配置请求独立`pbfm_training`，不放开UFO的native_reconfigure参数。collector发布专属模型族和resume Artifact。

- [x] 写preview/submit/adapter/恢复分发失败测试，覆盖native_rl不能错误进入UFO（Docker聚焦196项通过，包含旧UFO与通用训练回归）。
- [x] 实现共享recipe选择与专属P-BFM输入校验、source snapshot/镜像/数据/契约冻结；独立`pbfm_training`保护参数与前端对齐。
- [x] Job提交、准入、取消沿用既有生命周期，失败/重复收集/专属模型发布、checkpoint候选与恢复完成代码接线；容器行为检查通过，实际GPU生命周期证据待Task6。
- [x] 初始模板ready=false、dependency_ready=false并提供明确原因；新增`asset.validate`/`probe_pbfm_runtime`诊断入口，允许冻结draft接口/Profile但仍要求真实已验证数据。默认runtime模式不发布模型；本轮受控segment112只发布显式diagnostic/incomplete候选，不开放普通模板。

## Task 4：数据派生、prompt/ONNX与独立地形评估（设计A/C/F）

**Files:** 新建`backend/eb/pbfm_data.py`、`pbfm_evaluation.py`、`api/routes_pbfm.py`、`runners/pbfm_{data,export,eval}.py`及测试；修改`api/app.py`、`adapters/data_convert.py`、现有eval分发与结果收集的必要位置。

**Interfaces:** 独立`data.convert` operation：`prepare_pbfm_rpo_motion`、`export_pbfm_policy`；数据manifest=`eb/pbfm-rpo-motion/v1`，只接受冻结RPO RobotState/UFO数据并重验FK，train/validation/acceptance分离。`pbfm_evaluation.build_eval_plan(...) -> dict`与`validate_eval_plan(plan)`冻结case/seed/geometry/输入hash；route前缀`/pbfm`，用已有JobSpec/Run执行。输出逐case/metrics/video index和完整receipt，训练内评估不可替代。

- [x] 先写数据split/骨架/来源路径/hash及case完整性失败测试；包含真实Dataset ORM→数据API→训练绑定。
- [x] 实现受管数据派生与原生loader/FK回执，训练目录仅含train；当前model绑定tracking z、完整depth ONNX、可携带RPO资产/源码/许可证/codec与策略ZIP均已实现。数值/真实深度trace与各自独立闭环的32step PyTorch/ONNX有限性及差异记录在执行Job时强制运行，尚未运行候选模型导出或这些GPU数值路径。
- [x] 实现独立RPO地形runner：14低难度locomotion固定+X/0.3m/s，2平地站立/停止回归，20秒上限；真实RGB/深度输入视频、原始depth/逐帧物理与tracking指标、显式时钟及逐case fsync日志。尚未执行GPU地形矩阵。
- [x] RPO足部几何由实际collision mesh推导并支持MJLab robot/命名空间；目标镜像CPU已验证RobotState→MotionLib与逐帧FK（3帧含关节/根姿态变化，误差约1.15e-7m），group0台阶射线正确且排除浮动robot碰撞。禁止非支撑足body接触≥5N的冻结约束及负例已覆盖。完整/中断case媒体完整性分离，缺失类别如实标注；未生成候选策略的实际评估媒体。
- [x] 服务路由/adapter/collector与模型/Artifact血缘接线；固定producer镜像/源码/模型与holdout hash，不修改原模型；失败/取消/超时/基础设施错误仅暴露已验证partial_files，报告保持incomplete。Task4及partial聚焦容器测试30通过，Ruff通过；未启动训练、部署或正式acceptance。

## Task 5：工作台、资产引导、数据面与构建入口（设计C）

**Files:** 新接口YAML；修改`bootstrap.py`、`Makefile`、前端`TrainNew/TrainJobDetail/ModelDetail.vue`及相关组件/测试；Rust source白名单和CSV能力仅在实际缺口处修改。

- [x] 写新模板与旧UFO数据/恢复页面不混淆的失败测试；加载新draft契约并验证modalities/shape。
- [x] 展示P-BFM真实可用状态、保护字段、联合更新/采样、camera/课程配置和恢复限制；模型页提供对应评估/证据入口。
- [x] 增加独立make镜像目标；ebd增加源码根、专属资源plan与P-BFM tracking CSV，并覆盖checkpoint发布前staging实时指标。Rust容器46项通过；目标镜像实际构建归Task1。
- [x] 后端契约、前端Node测试、vue-tsc/Vite容器检查通过。

## Task 6：集成验证、三轮复核与运行证据

**Files:** `docs/progress/2026-10-21-pbfm-rpo-integration.md`及进度索引；按实际Job/Run/Artifact追加证据。

- [x] 轮一：完成设计/robot/camera/动作/数据隔离的代码及CPU复核，修复发现的问题；完整相关后端387项通过。实际GPU相机画面仍归轮二。
- [ ] 轮二：真实镜像、GPU放置、计数/资源/恢复/失败退出；按用户预算与设计门槛运行112更新闭环，再决定是否进入512/1008及后续区间。
- [ ] 轮三：图表、raw指标、独立case、成功/失败视频、索引/HTTP播放、ONNX实包；不完整证据禁止提升能力。
- [x] 独立代码审查与针对性复查完成，修复重要问题；新增中断报告、group0地形射线及禁止接触负例，相关后端387项、前端77项、Rust46项通过。保留未完成运行验收的勾选状态。
- [x] 已交付功能代码、镜像和实际验证记录。本轮预算答复后数据与单卡A0通过，112诊断0更新失败，最后resolver修复CPU检查通过；多卡恢复/候选ONNX/完整媒体验收仍未通过，模板关闭。不提交Git。

## 执行记录

- 2026-10-21：工作区初始干净，已建立`codex/pbfm-rpo-integration`分支；本次禁止提交。按用户指令直接从计划进入实施，不增加等待计划批准的停点。
- 共享接口预检：Task1私有source供Task2/4使用；Task2 plan/checkpoint供Task3/5使用；Task4数据manifest供Task3使用；均以本文字段名对齐。
- 运行预算已询问，等待答复；静态实现、容器测试和镜像构建继续。
- Task5初次验证：新契约1项；前端训练/旧UFO恢复18项，新模型面板与P-BFM请求5项；vue-tsc/Vite通过。Rust先观察两个新测试失败，再实现CSV和资源plan，collector10项、ebd35项通过。
- 复核发现：A0不能依赖已开放训练模板，否则无法获取首次运行证据。补受控`asset.validate`诊断operation，经同一Job/Run/ebd，只允许冻结draft接口/环境与经过loader校验的数据，不生成TrainJob或模型合格状态。
- 独立复核发现并进入修复：原生数据属性/holdout绑定、MJLab body前缀、终止前状态、原生rank seed、资产网格哈希、RPO足底传感器/几何、旧模型与导出镜像绑定、发布前checkpoint字节核验、工作台逐例媒体。保留实际GPU/完整恢复/HTTP媒体验证的未完成状态。
- 最终收尾：后端387项、前端77项、Rust46项通过，新增Python文件Ruff通过。最终镜像`sha256:654f41b6134081c9df406bc4bdd339652258b9235f2f3ddd091fa8386554c957`完成baked代码的CPU契约、MotionLib/FK/真实地形射线与原生78通过/1GPU跳过检查。Task6轮二、轮三的真实GPU及完整Job证据尚未执行，不能据代码测试提升能力。

### 用户一小时预算内的执行细化

- [x] Doctor/健康/活动Job核对；4张4090空闲，单卡先行，不推导多卡可用性。
- [x] 用户确认既有UFO数据；新增冻结producer分组桥接，未知祖先排除。数据Job`job_5534d65cf6784364900a`、Run`run_8d240223d87041169742`成功，派生`ds_5bf835c98ccd4da4967c`，41/3/12条动作分别通过原生loader与全帧FK。
- [x] 修复原生Profile发布kind遗漏、精确草稿诊断G1接线、CPU转换和推理/渲染容量误分类；新增与既有容器检查保留其他门禁。
- [x] 默认runtime模式A0经真实RLinf Worker验证2环境reset/step/depth/EGL；失败Run保留。修复NVIDIA vendor清单、锁定MuJoCo-Warp UNKNOWN传感器编译错误及训练wrapper无RGB输出后，`job_8e940d9c2ef14b71992d`/`run_2654b9262178409a9081`成功，媒体206 Range/401鉴权已实测。仅W1，未证明完整容量/多卡/G1 GPU回归。
- [x] 新增受控segment112模式及针对性检查：固定W1/E1024/target112/buffer32768/no resume，无failure injection；真实Workspace与完整保存回读，继续使用asset.validate原Job身份、diagnostic候选与incomplete资格，不自动发布接口/Profile或开放模板。
- [x] 用户另行明确授权后，于2026-10-21 12:24 UTC通过正式操作者API发布新接口`ee_rpo_v2_pbfm_depth_23dof_v1`并回读确认active；训练Profile仍draft、模板ready/dependency_ready仍false。该发布不构成追加GPU预算或阶段C通过。
- [ ] A0已通过，112诊断`job_e4270df516994ceda057`/`run_3840fa8ccdd2435eb851`固定段超时254秒，镜像`sha256:f80c41d60a27d17c02e7d75415722d78932fb10db51e2bf81d9681cbe6d7c371`。实际在完整配置快照遇到缺失len resolver，于第一次train调用前失败，完成0更新；修复已CPU复现通过。剩余预算不足可靠完成重跑与快评，停止新增GPU作业，现有本轮Job均已终态。
- [ ] 完整模型候选出现后，使用其实际producer镜像完成prompt/ONNX和固定32rollout独立validation，核对raw图表、逐case、视频/depth、鉴权与Range；缺失仍标incomplete。
- [ ] 更新轮二/轮三证据及停止结论。112之后恢复首更新最少还需16次至128，超过本轮量级授权，保持未验证；不能把段末CPU loader回读写成新Run训练恢复已通过。
