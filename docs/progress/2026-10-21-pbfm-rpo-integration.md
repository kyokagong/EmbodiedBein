# P-BFM / RPO / RLinf 接入实施记录

设计真源：[设计26](../design/26-pbfm-rpo-rlinf-integration.md)。执行计划：[2026-10-21计划](../superpowers/plans/2026-10-21-pbfm-rpo-rlinf.md)。本记录只描述实际进度，不改变设计约束。

## 当前状态

功能代码已按计划实施，容器回归与CPU检查通过。真实数据派生与单卡RLinf深度/EGL/RGB探测已成功；112诊断在第一次更新前失败，实际更新数0，尚未完成训练闭环，普通模板保持不可训练。最后的配置resolver修复已通过CPU复现检查，未重跑GPU。用户本轮总预算最多1小时：从2026-10-21 08:47:32 UTC开始，截止09:47:32 UTC，含准备、探测、训练、快评与录像，不进入512/1008阶段。剩余时间不足可靠完成一次新冷启动和独立快评，因此停止新增GPU作业，保留失败现场，不把功能代码当成阶段C验收通过。

用户随后明确授权发布新接口；2026-10-21 12:24 UTC已通过正式操作者API发布`ee_rpo_v2_pbfm_depth_23dof_v1`为active，并回读核对。训练Profile仍draft，普通模板`ready=false/dependency_ready=false`；此次发布未增加GPU作业，也不改变112更新未完成的结论。

不Git提交；起点工作区干净，分支`codex/pbfm-rpo-integration`。外部源只读；平台只写本仓及Job受管输出。

## 已执行检查

- 旧UFO/RLinf表单基线：容器20项通过。
- 新契约：先因文件缺失失败，新增draft契约后容器1项通过；380标量＋8×18×32 uint8深度，23动作，400/50Hz，pre-delay命令和角速度0.25。
- 新前端数据/模型区分：先出现错误数据过滤/缺少P-BFM识别的失败；修复后连同旧训练与恢复18项通过。
- 模型导出/独立评估面板：先因面板缺失失败；实现后新前端5项通过，vue-tsc与Vite构建通过。
- Rust：新P-BFM tracking CSV与资源plan先出现未收集/未识别的失败；实现后`eb-collector`10项、`ebd`35项通过。随后复现并修复checkpoint发布前staging实时指标丢失，collector扩为11项通过。资源plan按文件名与schema绑定，保留租约/GPU/路径检查；源码快照新增P-BFM根。
- 整合回归：初次241项、扩大到UFO native后368项；最终P-BFM、全部UFO native/training、共享恢复/绑定/环境选择共**387项通过**。扩大检查首次发现测试镜像缺Pillow，补充仅用于测试的`eb/test:pbfm-pillow`后通过。最终完整前端**77项**、vue-tsc与Vite通过；Rust collector11项＋ebd35项通过。全部新增P-BFM代码及测试Ruff通过。
- 组合镜像：独立锁定217项基础依赖，补齐上游未声明的pandas2.2.3及其固定时区依赖；实际导入Workspace、MotionLib、完整depth exporter、RLinf Worker通过，pip check无冲突。依赖解析不代替GPU/EGL/NCCL验证。
- 目标镜像CPU数值检查：真实RobotState→原生MotionLib载入1条3帧motion，全帧Torch/MuJoCo FK最大绝对误差`1.1515356668e-7 m`；RPO名称前缀和碰撞网格足底几何检查通过。这是代码测试，不是已发布Dataset的运行验收。
- 发布与失败证据：复现并修复同长度checkpoint内容篡改仍能发布的问题，发布前逐文件由ebd核验；部分评估的取消、缺失/半写日志、重复/损坏证据聚焦检查通过，保留固定32次rollout分母，禁止不完整报告通过。
- 地形与成功函数：训练地形使用group0，已修复只查询group5的高度偏差；只分类静态碰撞地形供渲染/查询，排除动态机器人。原生关闭contact终止，独立成功函数因此显式禁止非支撑足body的≥5N接触，并有失败负例。0.3m/s/+X命令、独立跟踪/速度/足滑/净空/饱和指标及显式时钟已冻结。配对32step PyTorch/实际ONNX独立闭环与原始轨迹/summary已实现，候选GPU路径仍未运行。

## 三轮复核记录

| 轮次 | 实际复核与修复 | 验收边界 |
|---|---|---|
| 设计/本体/隔离 | RPO原生属性、holdout绑定、动作语义、G1默认保留；修复MJLab前缀与足底接触匹配，sole/edge改为RPO碰撞几何派生 | CPU构造与回归通过；虚拟相机实际视野需GPU画面 |
| 运行/计数/恢复 | 修复原生`seed+rank`，冻结资产网格与镜像/patch；模型、8优化器和Replay全包检查及发布前ebd哈希 | 多卡更新/同步、显存峰值、完整段末与恢复首更新尚未运行 |
| 图表/媒体/证据 | 专属曲线、完整导出包、逐例RGB/网络depth/原始轨迹；捕获自动reset前的真实末状态，中断仍保留案例 | 新面板与部分报告行为已测；真实ONNX闭环、录像/鉴权/Range整链仍待Job证据 |

独立审查提出的十类具体问题已完成针对性复核；不能据此把代码测试提升为模型质量合格。

## 实施分工与依赖

1. 受管源码/robot/Hydra/camera、组合依赖镜像、RLinf Worker和完整checkpoint。
2. native recipe分发、预览提交、恢复候选、只读source binding与专属collector。
3. 数据派生、tracking prompt/完整depth ONNX、独立地形与感知诊断、逐case/视频receipt。
4. 本体/任务引导、工作台、构建入口、Rust采集与最终集成复核。

正式训练和acceptance状态必须由相应Job/Run/Artifact及报告提升，测试通过不表示模型质量合格。

## 操作入口与运行边界

所有新接口位于`/api/v1/pbfm`，操作由登录工作台的operator发起，仍通过现有Job/Run/ebd执行。预览后提交同一个hash及`Idempotency-Key`。

1. 已注册的RPO本地源Dataset先提供`eb/pbfm-motion-source/v1`清单，明确23关节顺序、z-up/xyzw/rad、三个split与每个文件的hash/大小/原始祖先；调用`POST /datasets/{id}/sources/freeze`固定DatasetVersion。
2. `POST /datasets/prepare/preview`和`/prepare`接受`name,parent_version_id,source_manifest`，运行原生MotionLib与全帧FK检查，成功后生成可训练P-BFM Dataset。只有train目录挂载进训练；validation/acceptance清单单独绑定。
3. `POST /runtime-probe/preview`和`/runtime-probe`接受`name,dataset_id,environment_id,gpus,timeout_seconds`；允许冻结draft接口/Profile，输出真实rank/设备、reset/step、EGL和网络depth证据。可选`failure_rank`只用于失败诊断，不产生训练模型。
4. 运行时门槛通过后才开放模板`rpo_pbfm_fb_depth_native`。工作台训练表单使用`pbfm_training`，恢复使用专属完整`pbfm_native_state`，分段目标为累计真实联合更新。当前模板仍关闭。
5. 冻结模型页的P-BFM面板调用`/models/{id}/export`与`/evals`对应preview/submit，展示独立validation的逐例结果、视频、深度及轨迹；这些报告不具有正式acceptance资格。

新增代码和测试不等于阶段A–F已验收。原代码交付时尚未创建实际Job；本轮用户给出预算后，数据Job已经通过、GPU探测已执行，具体证据见下文。GPU容量、112更新/全rank保存恢复、数值导出与真实评估录像、HTTP鉴权/Range仍须按实际结果逐项核对。

## Task1/2 运行时与检查点证据

独立数值镜像以CUDA12.8/Ubuntu22.04/Python3.10构建，基础依赖锁217项；补充锁将上游CSVLogger漏声明的`pandas==2.2.3`以及numpy/dateutil/six/pytz/tzdata一并固定。实际导入`rlinf.scheduler.Worker/Cluster`、P-BFM builder/Workspace/MotionLibRobot/完整depth exporter和平台PbfmTrainWorker，版本断言与`pip check`通过。初次镜像为`sha256:9a63a188bc006c7752bb7aef810ef08bf037fc62fe6e132b3d3963233c61ce57`。最终通过`make images-rlinf-pbfm`纳入全部稳定代码后的镜像为 **`sha256:654f41b6134081c9df406bc4bdd339652258b9235f2f3ddd091fa8386554c957`**，三个CPU脚本均在该digest中重跑、退出0；原生回归结果仍为78通过/1GPU检查跳过。

目标镜像CPU probe通过：23DOF/24个body及实际MuJoCo FK、原生字典观测构造`state=52/history=300/actor=380`、原生全body观测函数输出358、400/50Hz配置、23个YAML执行器与`.6*effort/Kp`动作缩放、RPO接触geom/sole采样配置和base seed+rank。脚本使用实际FK及CPU观测结构测试，没有构建GPU环境、reset/step或生成深度画面。Hydra旧日志仍显示flattened actor732/critic357；MJLab字典adapter不消费这两个旧值，实际Worker仍须核对实时字典空间与358维特权状态。

最终受管patch候选在该固定数值runtime中的原生CPU回归：**78通过、1跳过**。覆盖G1校验/动作重排与delay、depth crop/history、紧凑深度Replay/稀疏后继、reset successor和完整状态验证；跳过项明确要求空闲GPU及`PBFM_TEST_RESET_GPU=1`。6条警告来自原生小模型ONNX导出的弃用/shape tracing/GRU batch提示；这不是RPO训练模型或真实trace的导出验收。首次原生回归发现G1错误文本回归，已恢复原文本并重跑通过；旧UFO镜像曾缺termcolor，精确P-BFM镜像已按其自身依赖安装，不把旧镜像视为本组合运行证据。

控制面与运行时协议聚焦检查30项通过，包含只读render、原生计数端点、完整source/asset字节与成员集合、mesh篡改及未登记文件拒绝、checkpoint缺shard/半写/symlink/hash与rank拒绝。小CPU状态检查核对8优化器完整paramgroups/state/真实step、持久化回读一致性、严格model key/shape和NaN/Inf；完整P-BFM/RPO段末保存、跨rank同步及恢复首更新仍未执行。完成manifest与receipt只在全rank验证及清理成功、独立快照与曲线生成后发布；原生训练写入`runtime/native-staging`，最终发布到`outputs/native`。

复现入口（从仓库根目录；测试容器不请求GPU）：

```bash
docker build --build-context pbfm=/data/kk/P-BFM --build-context rlinf=/data/kk/RLinf \
  -t eb/rlinf-pbfm:rpo-depth-v1 -f deploy/images/rlinf-pbfm.Dockerfile .
docker run --rm -v "$PWD:/workspace:ro" -v /data/kk/P-BFM:/sources/pbfm:ro \
  -v /data/kk/assets/roboparty/rpo-v2:/robot-assets:ro -e OMP_NUM_THREADS=1 \
  eb/rlinf-pbfm:rpo-depth-v1 python /workspace/backend/tests/runtime/pbfm_runtime_probe.py
docker run --rm -v "$PWD:/workspace:ro" -v /data/kk/P-BFM:/sources/pbfm:ro \
  -v /data/kk/assets/roboparty/rpo-v2:/robot-assets:ro -e OMP_NUM_THREADS=1 \
  eb/rlinf-pbfm:rpo-depth-v1 python /workspace/backend/tests/runtime/pbfm_cpu_regressions.py
docker run --rm -v "$PWD:/workspace:ro" -v /data/kk/P-BFM:/sources/pbfm:ro \
  -v /data/kk/assets/roboparty/rpo-v2:/robot-assets:ro -e OMP_NUM_THREADS=1 \
  eb/rlinf-pbfm:rpo-depth-v1 python /workspace/backend/tests/runtime/check_pbfm_evaluation_geometry.py
```

源码probe中的虚拟相机位置、实际EGL画面与GPU观测必须通过受保护的`asset.validate` Job继续取得；以上CPU检查没有放开训练模板，没有运行112次联合更新，也没有对模型质量或真机作出结论。

## 一小时运行验证：前置与复核

- 用户确认使用现有已验证UFO数据`ds_6541f726d7804007a93a`、版本`dsv_304fffb96b30417db5d8`，生产Run为`run_62698e4956174db1b697`。受控`/pbfm/datasets/from-ufo/preview`和提交入口只引用冻结producer字节，通过`data.convert`派生；API不反序列化pickle，不改已发布父数据。
- 训练选8个已知source groups共41片段；validation选walk1共3片段；保留集选dance1/fight1/walk4共12片段。17个`official_rpo_unknown_ancestry`片段明确排除；原holdout不得进入训练。该划分仅供流程诊断，不能据此宣称正式验收或来源权利复核完成。桥接33项容器检查通过。
- 预检平台健康，4张RTX4090各约24GiB、无活动Job；本轮先验证单卡，不把空闲卡数视为已验证DDP拓扑。实际GPU峰值仍待Job。
- 构建遇到Debian默认源HTTP连接失败、HTTPS大文件传输过慢，切换可达HTTPS签名镜像并限制下载超时。构建网络使用临时host override，仍由`make up`部署；任务运行网络不因此改变。
- API依赖安装也阻塞预算，因此部署复用现有API基础镜像；先后核对源/目标`uv.lock`均为`bbb8567f8469bf3f5d96695c09593a1c988f8aa9a0c7bba5b36ee37658cc5b1e`，`pyproject.toml`均为`5fdee343d71ebf8abf9d86e8c1f9e94c33b1e8269bc010c9b8f8a43c93a4bb45`，仅更新代码、contract和构建后的前端。临时Dockerfile与override位于`/tmp`。实际API/ebd已启动并通过嵌套健康检查。
- 发布Profile检查原先仅接受sft/ppo，导致native_rl即使已具证据仍不可发布；补native_rl分支并要求ready与dependency_ready，11项Profile容器检查通过。此修复没有发布当前PBFM Profile或接口。
- G1原先只认识旧AMP诊断，PBFM草稿接口A0会被拒绝；补精确`asset.validate/probe_pbfm_runtime`与PBFM接口匹配，仅draft例外，其他门禁保留。RED复现后相关9项通过。
- 数据转换preview成功，首个submit被容量估算误当3.3B训练而拒绝（63.5GiB/no_fit），未产生Job；正在修正CPU operation识别，保留此次前置失败记录。
- 第二轮复核明确：A0仅2环境reset/step/EGL/depth，不能证明完整Workspace/Replay/update/checkpoint容量。112诊断保持模板关闭；完整恢复首更新还需额外16更新至128，超过本轮112授权，不会静默追加，相关阶段C条件仍须如实列为未验证。
- CPU转换容量修复后，`job_5534d65cf6784364900a`/`run_8d240223d87041169742`已成功派生`ds_5bf835c98ccd4da4967c`。训练41条5569帧、validation3条1200帧、保留12条1818帧，全部通过实际原生loader及全帧FK；manifest为`b3f55370a3291d2b65a9e9e3dfd1099322dd69271d64a736d1052bdec3c530dd`。该Job镜像`sha256:412e864d1e65c09ea8e4d85063f087ff938b709f2ee1c3fecbdd532d88d0c2e4`。
- 首GPU诊断`job_ea5c53a1f9ae4838958c`/`run_b6674caf639c4b80a565`在真实RLinf Worker设备映射校验退出，更新数0。失败容器内已有`libEGL_nvidia.so.0`驱动库，但vendor目录仅Mesa；镜像新增NVIDIA GLVND清单，不改宿主机或放宽设备校验。
- 新诊断`job_5cfe9693a93d40558fb6`/`run_b595ac0ca5a149b9a838`使用`sha256:1d48cf1b68c95c557620c8f40d1c9291faa02efd28a893d31b41820b7f819604`已通过CUDA/EGL映射、RPO23执行器与地形构建，随后在MuJoCo-Warp传感器内核编译失败，更新数0。锁定组合为MuJoCo3.8.0/mujoco-warp3.8.1/Warp1.18.0/MJLab1.4.0。
- 已用CPU完整模块编译精确复现：`sensor._frame_axis`的UNKNOWN分支引用未定义`xmat`，并非配置猜测或GPU显存不足。独立构建补丁位于`deploy/patches/mujoco-warp/`，严格限定版本、原始SHA256`2ba411f182ba5fb7d88e85050c55ef4055fafefe2d26555f429ecd4e0d80c19e`与唯一替换；仅补UNKNOWN identity矩阵，修后SHA256`d23fe083134ce25da0998aa0f2deda0144e303a747cb03eff75463ef853b0260`。完整sensor CPU编译与BODY/XBODY/GEOM/SITE/CAMERA及UNKNOWN共18个轴的数值检查通过；依赖锁和PBFM数据适配身份不变。identity fallback与[上游当前矩阵辅助函数](https://mujoco.readthedocs.io/en/stable/_modules/mujoco_warp/_src/sensor.html)一致；当前冻结版本的GPU闭环仍待新Job证明。
- 新增segment112的模式/指纹/单卡/禁止恢复与failure injection/诊断发布及容量边界检查47项通过；root复核相关Profile、契约、数据、发布与媒体索引共70项通过。初次root测试漏挂contracts导致2项FileNotFound，补齐测试挂载后38项全过，没有改测试断言掩盖失败。segment112分析媒体复核8项通过，确认未索引文件与同长度内容篡改均拒绝。
- 扩大回归：PBFM全部、既有UFO全部、Profile与领域共600项容器检查通过，2条既有依赖弃用警告；本轮新增/修改运行时和依赖补丁Ruff通过。
- 传感器修复后`job_8e1fc66f65a24d77a302`/`run_ee3300cde3c4468eb527`使用`sha256:302e8e13bf87c5b87215b7aad14ed8f9a88f91168edbff1a89afa4cc1a7014fa`已完成RPO GPU环境构建、reset/step与非恒定uint8网络深度生成。最后因训练wrapper默认render无RGB输出退出；探测改用独立评估已有的`MujocoQposRenderer`，使用实际scene与当前qpos，不伪造RGB或替换物理状态。仍未把这个失败Job标记成功。
- 同镜像单卡故障注入`job_65e978a9a95f481aa34d`/`run_56151c7a486041d3a64c`按预期Worker异常退出；任务容器`85af1a4acd7e`已确认Exited(1)，未发布成功回执或模型。这仅证明单卡故障清理，不代表多rank/NCCL容错。
- RGB出口修复的新诊断`job_8e940d9c2ef14b71992d`/`run_2654b9262178409a9081`已成功，冻结镜像`sha256:bfc318c003f6ed38481a1299f98274a7eb429c53ab35886918bd8815eab115e2`。真实RLinf Worker/GPU0、2环境reset/step、23动作/380标量/8×18×32 uint8深度、EGL与RPO地形RGB全部通过；已实际查看RGB和网络depth。媒体鉴权请求Range返回206、`bytes 0-63/29081`及64字节，无凭据返回401。这里仍不包含完整Workspace更新、最大Replay或多卡容量证据。
- 112诊断镜像`sha256:f80c41d60a27d17c02e7d75415722d78932fb10db51e2bf81d9681cbe6d7c371`纳入同一修复及最后导入格式检查。受控112诊断`job_e4270df516994ceda057`/`run_3840fa8ccdd2435eb851`固定单卡/E1024/Replay32768/seed42/目标112，段超时254秒以给独立导出/快评保留时间；在超时前因实际错误退出。已完成1024环境、七类地形、41条动作及Workspace构建，保存`effective-environment.json`时遇到`UnsupportedInterpolationType: len`（`robot.num_key_bodies`），尚未调用`trainer.train()`，故实际更新数明确为0，没有完整checkpoint、训练ModelVersion或独立快评，不拿目标112代替完成数。
- 独立评估前所需的新接口发布被自动审批拒绝：发布会持久改变平台可用状态，被认为超出明确的探测/112诊断授权。接口/Profile与普通模板均未因此改状态；已向用户请求该具体POST发布动作的额外授权。112诊断不依赖发布，继续执行；未获授权前不绕过既有评估G1门禁。

## 本轮停止结论与未完成出口

- 根因修复：`resolved_environment_config`仅在缺失时注册标准`len` resolver，仍严格`resolve=True`，不改学习配置、不保留假称effective的未解析快照。目标镜像CPU脚本先复现原错误，再验证列表长度解析、既有resolver不覆盖及未知resolver继续拒绝；Ruff通过。该修复后的GPU更新与保存回读仍未验证。
- 最后修复已构建为`eb/rlinf-pbfm:rpo-depth-v1`，digest为`sha256:10ef8f48475e0196e8775ded90ad785fb6cbb2719fd67e2e19b93a9f7edadedb`；构建中原生导入、pip check、传感器CPU编译及18轴检查通过。实际成功GPU探测的producer仍是前述bfc318镜像；不能把它的通过状态移给此新镜像的未执行GPU更新路径。
- 数据数值证据：train/validation/保留集全帧FK最大绝对误差分别为`1.0519657864804799e-6 m`、`2.1415291606707854e-7 m`、`6.403445884650694e-7 m`。数据来源许可与策略质量仍不由FK证明。
- ebd实测资源：成功2环境A0共24条样本，租约GPU设备峰值1052MiB，容器内存峰值7193.6MiB；失败1024环境/Workspace段共24条样本，GPU设备峰值9852MiB、容器内存7690.24MiB。它们不是模型进程独占分配统计，也不涵盖第一次更新、八优化器状态展开、完整Replay或保存峰值，不能据此宣布24GiB训练容量已通过。
- 三轮结论：本体/来源/split的静态与实际数据检查已通过；单卡设备/深度/EGL/RGB与单卡异常退出已取得证据，但完整更新/恢复未过；媒体鉴权/Range已通过真实A0文件，训练时序、策略视频、实际ONNX候选与32rollout快评因没有训练候选而未执行。阶段C继续`incomplete`。
- 决策为**停止并调整，等待后续运行证据**。没有进入趋势/正式训练，没有提升模型或真机资格。后续在另获预算后应从已修复镜像重新执行112冷启动与完整保存回读，再做实际导出/validation；新Run恢复首更新至少再需16次至128，必须另行明确总更新预算。多卡/NCCL、G1 GPU回归、正式acceptance仍未验证。
- 本轮结束检查：平台API/ebd健康，Job查询无queued/running/admitted/collecting/starting任务，新PBFM接口仍draft；所有本轮GPU任务已终态，无需取消其他用户作业。不提交Git。
- 最终镜像baked代码的resolver检查已退出0；结束时Doctor healthy，4张4090均无run_id且各空闲24080MiB。预算截止前已停止所有本轮GPU工作，后续只整理文档与CPU回归证据。
- 2026-10-21 09:47:09 UTC最终确认：最后helper修复后再次运行相关回归，600项通过、2条既有弃用警告，`git diff --check`通过。本轮从08:47:32起共59分37秒；没有追加GPU重跑、趋势训练或Git提交。

## 用户授权后的接口发布

- 用户明确答复“允许发布该新接口”，解决此前自动审批提出的授权缺口。以人类操作者身份调用`POST /api/v1/embodiment-envs/ee_rpo_v2_pbfm_depth_23dof_v1/publish`，HTTP200；平台内Agent的ToolBus禁令保持不变。
- 发布前确认live接口与已审阅contract一致（仅排除状态），自检通过且23关节一致。发布后重新GET确认接口及其spec均为`active`，发布hash为`bfba8d167389d904ce4767e17d4af4f4fc00818455922a9ba116400a980add9d`。
- 同时回读确认`env_train_rpo_pbfm_fb_depth_native`仍为draft，`rpo_pbfm_fb_depth_native`仍为`ready=false/dependency_ready=false`。接口发布不等于训练闭环、模型评估或真机能力通过。未启动新训练/评估Job，未修改已发布接口内容，未提交Git。
