# 23 · 原生 UFO/RPO 模型导入、部署前仿真与交付

日期：2026-10-17（Asia/Shanghai）。状态：**本机推理、双后端 smoke 与仿真交付已接入并验证；正式验收/真机未验证**。实际证据及未实现项见 [接入进度](../progress/2026-10-17-ufo-native-deployment.md)。

遵循设计 [02](02-system-architecture.md)、[04](04-model-training.md)、[05](05-evaluation-delivery.md)、[11](11-rpo-stand-walk-phase-a.md)、[12](12-environment-catalog-and-selection.md)。本设计提供原生 UFO 推理产物入口，与其他训练栈独立。用户要求多轮复核、允许询问不确定项、禁止 Git 提交。

## 1. 目标与边界

把 `/data/kk/UFO/runs/rpo_experiment_20261016/final_model/` 接入工作台，形成可检查的闭环：**导入冻结模型 → 导出推理包 → 部署前闭环仿真 → 查看报告/录像 → 下载带证据的交付包**。模型表现不佳也应产生完整、诚实的失败报告；接入成功与模型合格分别验收。

用户已确认首期目标为本机 x86/NVIDIA 容器内推理、部署前仿真与可下载的仿真交付包。当前请求不包含重训；使用现有模型，不启动训练。首期仿真提供固定任务的按需运行与录像，不新增常驻推理服务、浏览器实时遥控、真机电机接口或远程放行。目标为 Orin 或其他端侧设备时，增加相应 TargetProfile 和离线实机证据，不能用本机测试代替。

## 2. 已核实事实

以下来自只读文件、现有服务 API 和 Docker 元数据，未运行模型推理或仿真。

| 项目 | 事实 |
|---|---|
| 模型 | `FBcprAuxModel`，原生 `checkpoint/model/model.safetensors`，612,798,672 bytes |
| 权重 SHA-256 | `2041d2d5bb90a679ce5913f6d820c2829fed0266c2bc021744b0fd882a767738` |
| 保存配置 SHA-256 | 根目录 `config.json`：`2d901ad74b5367c3131e44b784c7bf8ad023c5812fc14044cec040e44c58089b` |
| 解析配置 SHA-256 | 根目录 `config.yaml`：`ccac069ee4b1bee5ae4b51f8a439abab3fdd43f4483f4b1c8752cba928eaeb38` |
| actor 输入 | `state:52 + last_action:23 + history_actor:300`；另有 `z:256`；导出拼接输入共 631 维，输出 23 维 |
| 特权观测 | `privileged_state:358`；actor 不使用，不能作为目标设备传感器要求 |
| 控制 | 保存配置为 400 Hz 物理、8 倍 decimation、50 Hz 策略；MJCF 原始 timestep 不作为覆盖保存配置的依据 |
| 原生关节顺序 | 左腿 6 → 右腿 6 → 腰 1 → 左臂 5 → 右臂 5 |
| 历史 | 4 帧、不含当前帧；按 term 分组、各组最新帧在前 |
| 任务 embedding | `reward_locomotion.pkl` 可见三个任务键：`move-ego-0-0`、`move-ego-0-0.3`、`move-ego-0-0.7`；shape/有限性/范数尚待容器内验证 |
| embedding SHA-256 | `2c9bbd88b7645710e2e5981092a49f608c4e941e450efa3bdc39ff48f371e7c1`；带 `_150000` 后缀的文件内容相同，不能从文件名推断训练 step |
| 外部依赖 | 原生 MJCF 的 `meshdir` 指向 `/data/kk/assets/roboparty/rpo-v2/rpo_description/meshes`（约 31 MiB）；原生环境还读取约 5 MiB 的 `runs/rpo_reference_v4/train_clipped.pkl` |
| 原目录 | 约 8.4 GiB，包含部署不需要的 optimizer 与 replay；不能整体复制作为推理包 |
| 训练计数 | `train_status.json` 声明 `optimizer_steps=98048`、`global_time=25165824`、`world_size=4`；作为外部来源声明保存，不伪造平台训练 Run 或映射为已有 learner step |
| UFO 源码 | 当前 HEAD `bedef7f5898d20cf9a237aa44f1c4a23f93c937a`，存在已跟踪和未跟踪修改；commit 不能单独代表实际内容或历史训练源码 |
| 本地镜像 | `ufo-rpo:rpo-23dof-20261016`，image ID `sha256:8733fd6e6ed852b4687d43e5499a5bd4f6dec58b12e5ff23a3bc92ff233f46fd`；入口为训练命令，不能直接作为平台推理入口 |
| 镜像依赖核对 | PyTorch 2.7.1+cu128、MuJoCo/MuJoCo-Warp 3.11.0、MJLab 1.6.0、ONNX 1.16.1；**没有 ONNX Runtime**，需在派生任务镜像中固定安装并实测。四个关键 observation/core/config/export 源文件摘要与当前外部工作区一致，不代表历史训练身份已验证 |
| 来源许可 | 本地 UFO `LICENSE` 标题为 Attribution-NonCommercial 4.0 International；保留原许可及研究用途元数据，其他资产/训练数据权利按来源记录，不推导商业交付资格 |
| 平台 | API 实际宿主端口 31800，healthz 与 ebd 正常；Doctor 当时显示 4 张空闲 RTX 4090，运行前须重新准入 |

源码依据：UFO 的 `ufo/envs/{observations,core,config,actuators}.py`、`ufo/tools/export_onnx.py`、`ufo/inference/runtime.py`；平台的 `api/routes_model.py`、`api/routes_eval.py`、`orchestrator/collect.py`、`bundle/pack.py`、`adapters/__init__.py`。

## 3. 方案选择

| 方案 | 结果与代价 | 选择 |
|---|---|---|
| 原生 UFO 导入/导出适配，复用 Job/Run、ModelVersion、EvalSuite 和 Bundle | 保留训练语义；补有限的 runner、校验和工作台入口 | **推荐** |
| 新建独立推理/仿真服务 | 重复资源、状态、媒体和权限管理，增加常驻服务 | 不采用 |

平台对原生来源记录 `provider=ufo`、`serialization=safetensors`、`source_runtime=ufo_native`；当前训练仍走其已有框架。不要填充假的 RLinf execution binding 来复用旧验证分支。

## 4. 本体接口与推理语义

新增接口 `ee_rpo_v2_ufo_native_23dof_v2`，绑定现有本体 `emb_roboparty_rpo`。先经过现有草稿/校验/发布流程；平台 Agent 不获得接口发布工具。首次联调发现原生 v1 缺少 critic 布局，平台 G1 拒绝提交；运行记录保留，修正派生 v2，不改已登记 v1 内容。

接口与产物 codec 必须共同冻结：

- 原生 23 关节完整顺序、rad/rad/s/Nm 单位、默认位置、位置/速度/力矩范围、Kp/Kd、惯量/摩擦和电机模型参数。
- actor 主输入 `concat(q-q_default, dq, gravity_body, omega_body)` 共 52 维，本模型保存的对应 scale 均为 1；上一动作 23 维；history 300 维。
- history 按 `actions, base_ang_vel, dof_pos, dof_vel, projected_gravity` 分组，各组 4 帧最新在前、初始补零、不含当前帧。每个策略步只更新一次；episode/reset 时动作和历史全部清零。
- actor 的 running normalizer 必须由权重载入并包含在 ONNX 图中；输入为按源配置构造的 observation，禁止再次归一化。
- z 为独立的 256 维任务输入；ONNX 固定布局为上述 375 维 actor observation 后拼接 z，共 631 维。接口保存观测布局，推理 manifest 保存 prompt 布局及 hash，不能悄悄把 z 加入旧接口。
- 本模型动作先执行 `a_applied=clip(5*a_raw,-5,5)`，再逐关节计算 `q_target=q_default+0.6*(effort_limit/Kp)*a_applied`。上一动作与历史动作保存 `a_applied`。
- 源配置外部 root quaternion 为 xyzw，MuJoCo qpos 为 wxyz，重力和角速度取机体坐标；转换必须有数值样例验证。

契约中已有 schema 不能保留的顺序、history、prompt 或逐关节动作元数据，应进入有类型校验的 codec/推理 manifest，并纳入接口引用与内容 hash；禁止仅把未知字段写进 YAML 后让 Pydantic 静默丢弃。

## 5. 导入与冻结

工作台模型页增加“导入 UFO/RPO 模型”。输入只接受白名单内的本地目录、名称和目标接口；使用专用请求 schema，不能传任意 Python、镜像、CLI、网络或挂载。

复用 `data.convert` Job 作为导入操作，通过 API → Orchestrator → ebd 执行。`/data/kk/UFO` 当前不在默认路径白名单；部署配置需增加此次实际必需的只读来源前缀，不能放开整个 `/data/kk`。校验 resolve 后路径、内部引用、symlink 越界、文件数量/大小与输出写根。

导入 runner 执行以下操作：

1. 仅复制权重、model config/init_kwargs、根保存/解析配置、训练计数声明、选定任务 embedding 和许可证到受管新目录；保存原文件逐项 SHA-256。optimizer/replay 不进入推理依赖，也不声明支持训练恢复。
2. 冻结当前实际推理源码/依赖和镜像回执，记录其与历史训练源码的一致性为“未知”，除非找到真实归档凭证。已跟踪修改和必要的未跟踪 RPO 文件不能被仅 Git tree 的快照漏掉；源码快照排除 runs、replay、虚拟环境和缓存。
3. 从保存配置解析物理、控制和观测配置，与 `config.yaml/effective_config.json` 对账；不同来源不一致时明确失败，不能用当前仓库默认值静默覆盖。纯路径重定位保留原路径、转换记录及新 hash。
4. 冻结实际 MJCF、所有引用 mesh/include 和所需 robot YAML。保留原文件；派生自包含资产副本使用相对引用，不改变几何、质量和关节范围。逐项验证闭包，无运行时读取外部目录。
5. 复用原生 `load_model_from_checkpoint_dir` 在隔离任务容器内验证模型可加载、normalizer 完整、维度及输出有限。joblib/pickle 只在无网络、无特权、只读源挂载的任务容器解析；将验证后的三个 256 维 embedding 转为 JSON，后续推理不再反序列化 pickle。
6. embedding 没有绑定权重的生成回执时，记录 `derivation=external_unverified`，只声明当前提供的权重与 embedding 的冻结组合；不补造历史来源。独立仿真评估该组合，来源限制进入交付 manifest。
7. 产物落盘/hash/loader 检查成功后才写完整 receipt，由 collector 原子登记 Artifact + ModelVersion 和 import Run 血缘。卡片使用新的 `eb/rpo-ufo-native-model/v1`，初始仿真资格未验证。原生模型不进入现有训练恢复候选。

导入预览与提交共用校验，plan hash 覆盖来源元数据、接口、镜像和配置；实际复制前后复核每个输入 hash，源目录变化则失败。重复提交依赖现有幂等键返回相同 Job；同一冻结内容避免重复发布。取消或崩溃只留下未发布的临时产物，重试产生新 Run，不覆盖旧版本。

## 6. 导出与部署前仿真

复用 `export.policy`，由批准的原生适配操作分发，不落到默认 Isaac 导出。优先调用 UFO 已有 `export_policy_model`，输出仅写本 Run 的受管目录；不得调用其默认在外部 model-folder 下创建 exported 的入口。

产物至少为 `policy.onnx`、`prompts.json`、`deploy.yaml`、codec/资产 manifest、源/镜像/配置回执与等价性报告。站立、0.3 m/s、0.7 m/s 三项按名称选择冻结 latent；不做 latent 插值，不声称支持侧移、转向、上肢指令或恢复起身。

导出验证覆盖多组有限随机输入、零/非零历史、边界动作和从真实 rollout 获得的观测；对原生确定性 actor 与 ONNX 比较动作，再比较反解后的 q_target。初始 FP32 容差固定为动作 `atol=1e-5, rtol=1e-4`、关节目标最大误差 `1e-4 rad`，实际不满足则停止调查。ONNX Runtime、checker 或模型加载失败均为失败，不能沿用旧 packer 的“检查异常当作通过”。无证据不自动放宽容差。

仿真分两条明确标注的路径：

| 路径 | 执行内容 | 证据口径 |
|---|---|---|
| MJLab/MuJoCo-Warp | 冻结原生场景及电机配置，使用交付 ONNX 和相同 codec，单 GPU 独立 case | 训练同系环境中的部署包闭环验证 |
| 原生 MuJoCo CPU | 自包含资产、明确地面、400 Hz/50 Hz、对应 PD/力矩速度限制、摩擦/惯量与坐标变换，使用同一 ONNX/prompts | 跨物理后端迁移诊断；与 MJLab 同属 MuJoCo 系，不能声称独立物理引擎覆盖 |

两条路径先用黄金观测和逐关节测试核对前/后处理、历史、时间步和执行器，再进行闭环。原生 MJCF 自身没有可直接当作已验证训练地面的配置；生成地面是明确版本化的场景构建步骤。不能用只渲染 qpos 的 `MujocoQposRenderer` 冒充第二条物理仿真。

如需要 MJLab 初始化时依赖的 motion 文件，将它作为受管评估辅助资产冻结，保留在仿真依赖中，不放进固定 embedding 推理的交付 payload；评估初态由 Suite 定义，不抽取训练 motion 的“容易帧”作为测试。不得修改外部 motion 文件，也不为了部署重新读取 replay 或执行 reward relabel。

## 7. 套件、预算和报告

新增 Task/EnvironmentProfile/EvalSuite 版本，不改旧 Suite。首期复用设计11中站立/0.3 m/s 的任务判据，仿真 timestep 和接口单独冻结。正式目标为完成 20 秒（1000 策略步），基座高度 `<0.50 m` 或倾斜 `>0.80 rad` 时独立判倒并锁存失败，不因后续重新站起改判成功。

- smoke：每后端三个任务各 1 case，种子固定为 71000/71001/71002。检查加载、ONNX、闭环、记录和媒体；资格只记诊断，不晋升模型合格。
- acceptance：每后端 12 case，seed 72000–72011；偶数站立、奇数 0.3 m/s，沿设计11冻结初态扰动。站立要求漂移 ≤0.5 m、水平速度 RMSE ≤0.15 m/s；直行要求沿初始朝向位移 ≥4.0 m、水平速度 RMSE ≤0.20 m/s；两者均须全程直立。12 例全成功才满足该后端的任务资格。
- 0.7 m/s 先作为诊断；独立冻结判据之前不进入正式通过资格。保留足部接触、滑动和轨迹供步态检查，不用存活或速度均值单独证明“正常行走”。
- 每例记录关节位置/速度/力矩越限和持续时间、请求/执行动作、基座状态、yaw-only 机体水平速度及 COM 诊断、推理延迟、终止原因和实际步数。原始安全越限单列；任何硬件可部署结论仍需对应现场阈值和证据，任务通过不能覆盖安全失败。
- acceptance 候选由导出完成后冻结的 model/export/prompt/codec/asset hash 定义，不依据 acceptance 结果改配置再复用同一 Suite 记录。

首期运行预算提案：导入/导出与两后端 smoke 合计至多 1 张 GPU、30 GPU 分钟、CPU 8 核、RAM 16 GiB，新增受管产物预算 4 GiB；构建镜像另核算。先完成 smoke 再运行 acceptance，后者另设一次 30 GPU 分钟上限。每阶段运行前检查磁盘及资源准入；不占用其他作业，不为并行率拆四卡。达到预算、基础设施出错或证据不完整时停住，不自动重训。该预算是提案，不是已执行记录。

报告保留 planned/completed/valid/invalid/infra_failed/cancelled，逐 case JSONL、原始轨迹、指标时序图、失败分布、成功率分母及 Wilson 区间。每例都录制真实闭环录像，视频由轨迹渲染时标注物理执行和渲染来源；绑定 case/seed/model/export/suite hash，并核验解码、时长、FPS 和平台 GET/Range 播放。全失败则写“本轮无成功案例”，无失败则如实注明；不补造视频。缺少必需 case 或媒体为 incomplete，任务失败与基础设施失败分别记录。

## 8. 交付包与工作台

复用 Bundle/Artifact 表与 `bundle.build` Job，支持多文件 manifest。导出阶段先冻结包含全部可执行 payload 的候选，再对该候选进行仿真；最终归档加入评估报告引用时，payload hash 必须保持完全相同。重新导出或改提示/codec/动作参数均要求新版本与新评估。

`.ebp` 至少包含 ONNX、prompts、deploy/codec、必要本体资产、文件 hash、来源/许可证、镜像/runtime 要求、导出及仿真报告引用、目标设备覆盖状态。不得包含 optimizer/replay、训练数据副本或宿主绝对依赖。下载通过 ebd 的受控 Artifact 流式接口提供，Python 只鉴权/查元数据；不能复用当前整文件 `fs::read` 的媒体实现直接吞入大包内存。

部署页展示“导入检查 / 导出等价性 / MJLab 仿真 / MuJoCo 迁移 / 目标设备验证 / 交付包”及真实状态、失败原因和跳转。模型详情提供三项任务选择、预览/启动仿真、运行状态和报告录像入口；复用 Queue/Run/Eval 页面，不新建另一套作业中心。所有表单有加载/空/错误状态与标签，前端筛选不能替代后端兼容校验。

本机交付包标注 `simulation_only`、目标硬件未验证、`signed=false`；允许下载检查，不显示为真机可用。不能沿用部署页当前仅设置布尔值的“签名认证”或平台“阶段放行”操作来表示真机部署。证据核验与未来真正的密码学签名各自保留状态；平台 Agent 仍受真机激活/门禁豁免禁令约束。

## 9. 修改范围与验证要求

| 落点 | 最小必要改动 |
|---|---|
| `contracts/`、本体初始化/任务与环境目录 | 新原生接口/Task/Profile/Suite，旧版本不可变 |
| `api/routes_model.py` 与模型领域校验 | 专用导入预览/提交、来源/接口/路径检查，登记完整 ModelVersion |
| `adapters/{__init__,data_convert,mujoco}.py` 与现有导出/打包分发 | 明确批准的 operation、镜像 digest、argv、只读输入和受管输出；不使用默认训练入口 |
| `runners/` 与 `orchestrator/collect.py` | 原生导入/导出、共享 codec、闭环仿真和多文件打包；严格 receipt、原子发布及血缘 |
| `api/routes_eval.py`、通用 Orchestrator 校验 | 新卡片/候选和冻结 plan 支持，直接 `/jobs` 也执行相同校验；不强求伪造源训练 Run |
| `bundle/`、Bundle API、ebd Artifact 传输 | 候选 payload/证据绑定、受控下载；原生包隔离旧单 ONNX packer 的默认值与 fail-open 检查 |
| `frontend/src/views/{ModelList,ModelDetail,EvalNew,Bundles}.vue` | 导入入口、原生模型任务/仿真操作、完整证据和交付下载；修正这条链路中的误导性放行/签名展示 |
| `deploy/images/`、Makefile 与部署配置 | 原生 UFO 派生任务镜像、源码回执、只读路径前缀；API 镜像含 SPA，通过 `make up` 更新 |
| `backend/tests/`、`frontend/tests/`、Rust 对应模块测试 | 只补关键语义/信任边界/状态与传输行为；不为每个静态字段写镜像测试 |

复用现有 JSON metadata，当前不需要新建业务表；实现若发现必须改 schema，再提供真实的 Alembic 迁移与空库/旧库测试。不可用 `create_all` 升级生产库。

必须验证：缺失/截断权重、错误 joint/history/normalizer/latent、非有限输出、symlink 越界、来源在预览后变化、直接 Job 绕过、重复提交/收集、取消/重试、损坏或缺失视频、导出检查异常、payload 改变却引用旧报告、伪造合格报告、无效 Range 与未授权下载。CPU 行为测试和前端类型/构建在容器中运行；真实模型加载、导出和两种仿真一律通过 Job/Run/ebd。

## 10. 三轮复核记录

### 第一轮：设计、本体语义和数据来源

发现：23 维 shape 相同掩盖关节顺序、history 排列、角速度缩放与动作映射差异；z 不在旧 actor 输入中；直接纳管权重不等于可部署。

设计修正：新增独立接口和受检 codec；完整冻结 normalizer/prompts；不修改旧接口；用导入 Run 记录外部来源，训练计数只作声明。现有三项 embedding 不扩展成未验证连续命令或上身控制。

复核证据：保存配置、init_kwargs 和原生 observation/action 源码已逐项对照。未运行数值等价性测试。

### 第二轮：资源、运行、恢复和失败

发现：原目录约 8.4 GiB，部署只需其中一部分；原生镜像默认启动训练；外部路径不在平台白名单；模型外有 mesh/motion 依赖；已有 evaluator 要求 RLinf 训练 Run。

设计修正：只复制推理所需文件，显式覆盖受管执行入口；新原生来源分支复用生命周期；固定 image/content hash 和资产闭包；分阶段有界运行、失败关闭、取消不发布。推理导入不声明支持训练恢复，也不执行新训练。

复核证据：Docker 元数据、平台 health/模型/Bundle API、ebctl doctor、路径/挂载/collector 代码已检查；无网络只读容器中的依赖探测确认 ONNX Runtime 缺失，观察/控制/导出四个源码摘要与宿主匹配。GPU 状态仅代表检查时点，未取得运行租约。

### 第三轮：可视化、证据和交付出口

发现：原生训练大部分终止条件关闭，运行满时长不能证明直立；训练目录现有媒体不能代替平台新评估；两条后端同属 MuJoCo 系；现有 Bundle 单 ONNX、自检失败可跳过、签名只是布尔值；qpos renderer 只有渲染功能。

设计修正：使用独立跌倒/任务判据、全量逐 case 媒体、实际 ONNX 包闭环和黄金轨迹对照；迁移覆盖准确命名；新包明确仿真用途、来源与目标状态；不借用旧签名/放行状态。传输走 ebd，缺证据保持 incomplete。

复核证据：设计05/11、原生环境 termination、评估脚本、导出 helper、Bundle API/packer/UI 与 ebd 媒体实现已检查。未产生任何新模型/评估/交付物，不宣称设计或静态检查已完成接入。

## 11. 已确认范围

用户选择“先完成本机容器推理、部署前仿真和可下载交付包”。据此实施 [执行计划](../superpowers/plans/2026-10-17-ufo-native-deployment.md)，不做 Git commit/push，不修改外部 UFO/RLinf/BHL/数据仓，不启动额外训练。历史设计复核中的“未运行”描述记录设计时点，实际运行状态以接入进度中的 Job/Run/Artifact/报告证据为准。
