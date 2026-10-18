# UFO / RPO 部署方式的实时仿真验证

2026-10-18（Asia/Shanghai）。设计真源：[设计24](../design/24-ufo-live-simulation-validation.md)，继承[原生部署接入](2026-10-17-ufo-native-deployment.md)。用户确认连续切换 z，另提供重置。实现保留在当前工作区，无 Git 提交，无训练或任务向量生成。

**状态修订**：下文 v1 的模型/控制/产物验证仍是历史证据，但约5Hz的 HTTP 轮询与 Three.js 页面表现已被用户拒绝，不能再称为实时交互验收通过。现已部署 v2 原生 GPU 画面 WebSocket 推流；两环境实际浏览器约27–27.5fps，性能、运行和边界见[新版记录](2026-10-18-ufo-websocket-stream.md)。

入口：[实时仿真验证](http://1.94.0.106:31800/deploy/simulation?model_id=mv_94d72b2354f040838aea)，也可从交付导航及导出模型详情进入。下文“使用与实现”至旧性能记录描述 v1；v2 以设计24修订和新的运行证据为准。

## 使用与实现

选择已导出的原生模型、MuJoCo CPU 或 MJLab 环境、30–600秒时限和 seed，直接点击“启动仿真环境”；内部检查并冻结配置，默认从站立开始。也可单独点击“检查仿真配置”。顶部不再选择行为任务，环境运行后在右侧实时切换。页面展示真实 STL / 刚体位姿，支持三个已有任务、暂停、继续、独立重置和停止。切换任务保持物理状态、上一动作及历史；重置使用相同冻结 seed 并暂停，点击继续后开始下一轮。

会话走 `serve.policy` Job→Run→ebd，实际加载交付 payload 的 ONNX/runtime/prompts/codec。模拟时间仍为50Hz策略、400Hz物理；页面最多10Hz读取最新状态。推理、物理和浏览器绘制分开；不以离线视频充当实时画面，不改 dt 追赶慢机器。

模型、父模型、环境Profile/Suite、镜像、seed和预算均冻结进计划。每Job4CPU/16GiB、30分钟总超时；CPU不分配GPU，MJLab一GPU；无网络，输入只读。命令与租约由 ebd 写固定路径，经鉴权和 Run 活跃检查；全会话最多1024条命令，ID幂等，ACK记录实际step/episode。平台 Agent 不能启动或控制。

断线15秒结束实时阶段；全程轨迹每500步落盘。结束后生成5fps录像、图表、事件与文件摘要。CPU录像320px、MJLab480px，页面按浏览器视口绘制。完整记录登记诊断Artifact；异常/取消保留可下载的部分记录，`complete=false`，不创建正式 EvalReport 或通过资格。

## 冻结模型与实际运行

- 原生模型 `mv_ff284c75ef03498ca016`；导出模型 `mv_94d72b2354f040838aea`。
- payload：`sha256:5da9974b5bbf6701bb0719ae973e96318d48db5157d82b25bff3777b9493a2d2`。
- 原始权重仍为 `2041d2d5bb90a679ce5913f6d820c2829fed0266c2bc021744b0fd882a767738`；原始 `reward_locomotion_150000.pkl` 仍为 `2c9bbd88b7645710e2e5981092a49f608c4e941e450efa3bdc39ff48f371e7c1`，只读复核一致。
- 环境：`ep_rpo_ufo_native_mujoco_live_v1` / `ep_rpo_ufo_native_mjlab_live_v1`，新建版本，旧Suite/Profile未修改。
- 最终任务镜像 `sha256:48c2be4d19ec011edd1c860757c1630a37ada28a2f33623cdba6085744fc02e2`。MJLab实测使用前一镜像 `sha256:b6800c7c1c19bda03a3de0556d735d73b78170273d1bc260e4a17d53626a5783`；随后仅降低CPU收尾录像分辨率，MJLab物理/推理及录像路径不变。旧Run继续绑定其原镜像。

| 验证 | Job / Run | 实际结果 |
|---|---|---|
| MJLab连续切换 | `job_d643a3aa27c24631a7a5` / `run_54180e6282ad4138a535` | 360步，2轮，36帧/7.2秒视频，完整诊断；仅GPU 0 |
| 最终真实浏览器/MuJoCo | `job_41a45b2f349d43c19e9b` / `run_27002280884d4734aed4` | 页面完成全部控制，631步，2轮，64帧/12.8秒视频，7个文件入口；零GPU |
| 失联续租 | `job_79023c079d5149d786d0` / `run_7fce65158cd14bf59493` | 15.017秒触发收尾，736步，74帧/14.8秒视频；termination=`lease_expired` |
| 强制取消 | `job_7abe12d4d9c3449da1b5` / `run_911dfbca97a345b68dd7` | cancelled；5个部分文件可下载，含500步NPZ；complete=false，无完整视频声明 |

另保留三次调试对照：`job_1331110cb0da4796af32`（验收容器缺SwiftShader；真实CPU会话按30秒时限完成）、`job_2a5b24bb6b164cdb9a2f`（页面交互完成，脚本误匹配说明文字而提前检查文件；Job正常完成）、`job_cab4ec7260f345e692e1`（性能优化前完整浏览器对照）。首轮共7个Job，1次GPU会话，全部经平台准入；未绕过Job执行模型。后续HTTP现场复验另有1个CPU会话，见下文。

## 实际性能与证据

两次任务切换的事件证明 qpos 与历史摘要在应用前后相同；episode保持0，策略随后读取新任务向量。只有reset将episode增至1并清历史。MJLab原生观测与交付codec最大误差为0。暂停期间step不增加；停止后进入收尾，不继续控制物理。

| 指标 | MuJoCo最终会话 | MJLab会话 |
|---|---:|---:|
| 全轨迹推理 P50 / P95 | 1.143 / 1.668 ms | 1.075 / 1.484 ms |
| 最长单次推理 | 6.373 ms | 3.103 ms |
| 实时率（运行快照） | 约1.0× | 约0.68× |
| 位置 / 速度 / 力矩越限子步数 | 977 / 51 / 0 | 985 / 41 / 0 |

MJLab控制往返至实际ACK为253–361ms。真实浏览器采集320份状态、67组不同真实姿态，零JavaScript异常；软件渲染并截图时，运行区间平均状态更新5.13Hz、间隔中位数122ms。优化前为1.90Hz。页面轮询上限10Hz，不能将上限当成所有浏览器的实测帧率。普通GPU浏览器帧率未单独基准测试。

性能复核发现本地UDS每请求构建无用TLS上下文平均31.9ms；本地明文HTTP无需TLS，去除后构建0.36ms，最终API读取P50约10.7ms、P95约14.0ms。页面仅在新状态、相机变化或resize时绘制，并扣除请求耗时安排轮询；无限物理平面的显示随根位置移动，整米网格保持世界对齐。

独立回读三份完整会话的18个文件，GET/SHA/大小全部一致；NPZ的qpos30维、动作/目标23维、每步8×6限位证据、连续step及有限性均通过。三段视频均由worker全量FFmpeg解码，尾部Range返回206。取消会话5个文件均GET=200。匿名=401、Agent=403、非网格资产=404、已结束会话控制=409；未知任务=422。旧41,534,607字节交付包仍HEAD=200、Range=206。

原始记录：[MJLab](artifacts/2026-10-18-ufo-live/mjlab.json)、[最终浏览器](artifacts/2026-10-18-ufo-live/browser.json)、[优化前浏览器](artifacts/2026-10-18-ufo-live/browser-before-tuning.json)、[失联/取消](artifacts/2026-10-18-ufo-live/failure-paths.json)、[全部Job/Run、轨迹与文件复核](artifacts/2026-10-18-ufo-live/final-audit.json)。页面截图：[站立](artifacts/2026-10-18-ufo-live/mujoco-stand.png)、[前行](artifacts/2026-10-18-ufo-live/mujoco-forward.png)、[收尾完成](artifacts/2026-10-18-ufo-live/mujoco-finished.png)。这些大文件按仓库规则被gitignore，仍保存在本机。

## 三轮复核

1. **语义、模型与环境**：实际交付Policy按task索引已有z；仅reset清历史。核对原生23关节顺序、观测历史、50/400Hz、环境与源资产摘要。显示原始STL局部变换，刚体世界姿态从实际qpos做独立运动学，未重跑物理或补造动画。
2. **控制、资源与失败**：修复全局存储锁可阻断15秒续租的问题，live固定小文件使用独立短锁；补全命令全会话幂等和未知结果同ID重试。修复失败/取消证据原先不可访问的问题，部分诊断Artifact白名单提供下载。实际失联/取消验证通过。
3. **页面与完整证据**：独立代码审查、真实浏览器全部控制、原始轨迹/事件/媒体hash及Range复核。去除重复渲染和UDS开销；保留慢于目标实时率的实际数值。页面分别显示请求态、应用态、初始化、跌倒、断流、收尾及完成。

开发验证：后端相关88项通过；前端全部51项、vue-tsc和Vite构建通过；ebd相关21项通过；新增Python模块Ruff、差异空白检查通过。仅有现有crypt弃用和Vite大chunk提示。使用`make up`部署，任务镜像独立构建。未将这些结果扩大为全仓测试或模型质量验收。

## 验证边界

仍只支持原生UFO导出模型与两个兼容平地版本。最长600秒会话的长时稳定性、慢机器30分钟内完成收尾、实际跌倒后的页面重置和真实网络断流恢复未做长程/故障注入实测；跌倒/状态重连/幂等/过期界面逻辑有行为检查。本轮无实际跌倒，未补造该类录像。

当前仍有速度跟踪偏差和物理越限；交互会话不提供正式任务成功结论，既有离线评估0/6满足判据的事实不变。未验证真机、未放行硬件。未创建新速度z，也未更新网络参数。

## IP + HTTP 现场复验与启动流程修复

用户反馈`http://1.94.0.106:31800/deploy/simulation`点击检查时报`crypto.randomUUID is not a function`。此前localhost验收遗漏了这一来源差异。用完全相同URL的真实Chromium复现：`isSecureContext=false`、`crypto.randomUUID=undefined`、`crypto.getRandomValues=function`，未使用将不安全来源强制视为安全的浏览器选项。

新增统一`liveId()`：可用时使用randomUUID，否则用getRandomValues生成128位随机ID。覆盖配置幂等键、运行中任务/控制命令及离开页面stop三个调用点。入口改为直接启动一次环境，内部完成冻结预览；移除顶部“初始任务”，默认站立，所有行为通过运行中的控制区切换。后端API、策略、任务向量及任务镜像未变。

复验Job=`job_c3771036f7d9452ab11e`，Run=`run_45048da064eb4ddcba2d`，MuJoCo CPU、时限30秒。浏览器先检查配置，再通过启动按钮完成内部预览和唯一一次session提交。运行中实际切换`0.3 → 0.7 → 站立`，3次ACK均在同一Run、episode=0；每次应用前后的qpos/history摘要相同。捕获170份真实状态，零JavaScript异常，停止后录像与记录收集完成。

三轮补充核对：①按真实IP+HTTP复现而非localhost；②检查全部随机ID调用点与启动/重试/离开页面，并以行为测试覆盖无randomUUID与同Run切换；③在实际HTTP浏览器验证配置检查、唯一一次启动、三次连续切换与完整收尾。新增两项检查先失败后通过，前端全套53项、vue-tsc和Vite构建通过。已使用make up部署，无Git提交。

现场记录：[修复前复现](artifacts/2026-10-18-ufo-http-fix/http-baseline.json)、[修复后实际运行](artifacts/2026-10-18-ufo-http-fix/http-fixed.json)、[新页面实时切换截图](artifacts/2026-10-18-ufo-http-fix/http-live-switch.png)。
