# 24 · 部署方式的实时仿真验证

2026-10-18（Asia/Shanghai）。扩展设计23；用户要求给定模型和环境，在页面实时切换任务并观察真实闭环效果。不重训，不重新生成或覆盖 z，不提交 Git。

## 目标与选择

复用已冻结 UFO ONNX / prompts / codec / MJCF 及 Job→Run→ebd。启动一个有时限的 `serve.policy` 仿真会话，容器持续执行50Hz策略/400Hz物理，页面切换三个已有任务。默认连续更换 z，保持姿态、上一动作和历史；另提供暂停、继续、重置、停止。

2026-10-18 修订：旧版 HTTP 轮询加 Three.js 的浏览器实测仅约 5 Hz，用户明确拒绝其交互表现。新版使用服务端 EGL/GPU 原生 MuJoCo 渲染，WebSocket 二进制 JPEG 推流；浏览器解码显示，不重建机器人网格。任务和相机控制复用持久连接。既有 v1 记录保留，新会话派生 v2 Profile/Suite，不改已发布对象。

渲染目标 640×480、30fps，与 50Hz 策略循环分线程；只复制真实 qpos 最新快照，渲染线程独占自己的模型、数据和 GL 上下文，不修改物理状态。两种后端均申请一张 GPU 用于渲染，CPU MuJoCo 物理仍在 CPU。JPEG 编码在任务容器完成，ebd 负责固定路径和有界转发，API 只鉴权并透传字节；无新端口、无新守护进程，任务容器仍无网络。

选择 WebSocket JPEG 的原因是现有单端口 HTTP 部署可直接使用，不需要公网 UDP/TURN 或浏览器 WebCodecs。它有较高带宽成本；本次验收记录实际带宽和延迟。WebRTC 是高分辨率、弱网或多观看者场景的后续方案，不把 JPEG 推流称为 WebRTC。

首期仅支持 `ee_rpo_v2_ufo_native_23dof_v2` 兼容的已导出模型。提供版本化的MuJoCo CPU平地与MJLab平地环境，不接受任意MJCF/脚本/镜像/动作增益。其它模型显示不支持，不以相同动作维度推断兼容。

## 冻结输入与运行

- 新操作 `ufo_native_live`，JobKind=`serve.policy`；模型、payload hash、原生父模型、环境Profile/Suite hash、镜像、seed、初始task、最大秒数均进入预览hash。
- 新环境Profile使用 eval usage，绑定明确的 interactive diagnostic Suite；不修改已有Suite/Profile。该会话不生成正式 EvalReport 或 pass资格。
- 默认300秒、最多600秒；Job总超时30分钟含收尾录像。每Job4CPU/16GiB；v2 MuJoCo 与 MJLab 均一 GPU；v1 资源定义保持原样。无网络、输入只读、独占输出、无自动重试。
- `runtime.py` 从实际交付payload加载，固定ONNX CPU provider与已有三个256维prompts。task名仅索引已冻结z，每周期更新观测。
- 浏览器期望态与容器已应用态分开；所有控制回显命令ID、实际step、episode、task。切换任务不调用reset；重置才清历史和上一动作，使用同一冻结seed便于比较。
- 跌倒暂停并提示重置；位置/速度/力矩越限计数可见，不以存活或视觉效果宣称任务成功。非有限状态终止并保留原因。

## 控制与数据通道

API只鉴权、校验当前Job/Run与模型环境元数据。ebd负责固定路径的命令、租约及实时输出；不开放任意文件写入或HTTP转发。

内部控制：`POST /v1/runs/{rid}/live/control`，严格JSON `{id,action,task?}`；action仅task/pause/resume/reset/stop。命令原子写入本Run `live-control/command.json`，worker只读挂载；心跳 `POST /v1/runs/{rid}/live/heartbeat` 独立写lease，不能覆盖未消费命令。所有入口验证native-plan operation、Run存活、路径与symlink，无shell/网络/电机命令。

命令ID在本会话全程幂等：最多1024项，同ID同内容返回原接受结果，同ID不同内容拒绝；延迟重试不能再次reset。live小文件写入使用独立短锁，不等待无关Run的大目录清理。心跳不计入命令数。

新版媒体UDS WebSocket：`GET /v1/runs/{rid}/live/stream`；公开入口 `/api/v1/ufo-native/live/runs/{rid}/stream`，连接前验证用户、会话所有权、Run 与同源 Origin。API 仅透传二进制帧，不解码或解析图像。旧快照端点供诊断留存，页面不再轮询 state/scene。

二进制消息为 4 字节大端 metadata 长度、≤4096 字节 UTF-8 JSON、JPEG，整帧≤2MiB。metadata 含 seq、step、episode、源 updated_at、rendered_at、camera_seq、尺寸及渲染/编码耗时。worker 原子替换 `outputs/live/frame.bin`，保留一帧；ebd 每连接最多8张在途帧，仅保存它们的序号和发送时间。浏览器完成解码显示后发送累计 `{type:frame_ack,seq}`，确认序号必须属于实际发送窗口，不能确认未来或倒退。窗口满则跳过旧帧，有空位才读取最新帧；最旧未确认帧超过2秒或发送超时断开，无无限缓冲。浏览器最多一张正在解码和一张待显示最新帧，跳过的中间帧由下一张成功显示的累计 ACK 清除。

采用有限窗口而非停等协议：实际 IP+HTTP 浏览器试运行证明，一帧一确认虽有约30fps服务端渲染，但会被往返时间限制到约5fps，故此结果不算验收通过。8张在途允许网络持续传输，同时有明确内存/时间边界。

文本推送 `{type:state,state}` 最多10Hz，控制 `{type:control,command}` 复用既有幂等和实际应用 ack，结果 `{type:control_result,id,ok,status,detail}`。相机为独立 `{type:camera,seq,azimuth,elevation,distance}`，限制30Hz、角度与距离范围，写固定 camera.json；它不覆盖任务命令，也不改变策略输入。帧回显 camera_seq。浏览器拖动旋转、滚轮缩放、键盘操作和复位视角，始终跟随机器人根位置。

状态包含run/plan/model/payload身份、status、当前task、episode、策略step、sim/wall时间、实时率、命令ack、根位姿、刚体世界位姿、速度/高度/倾角、限位计数和推理时延。展示时检查身份、序号和时效，断流时明确停用控制与提示。

浏览器每3秒通过 WebSocket heartbeat 续租，worker就绪后15秒无心跳会话自动结束；服务端不能自行续租。命令发送后等待实际ack再接受下一项。断线重连复用同一 Run，不重新启动或重置；状态过期禁用控制。启动/控制仅操作者，平台Agent入口仍禁止。

“关闭交互环境”与“结束并保存录像”分别表示立即回收和正常收尾。关闭使用独立 `POST /sessions/{jid}/close`，不依赖 WebSocket、任务控制可用性或模型计划有效性。仅拥有该会话的用户可调用，同源检查在写入前完成；通过现有 Job cancel→ebd 确认容器删除后再标 cancelled，保留已落盘的部分诊断记录并明确 complete=false。排队、初始化、断流、暂停、跌倒、录像收尾都可关闭，重复关闭终态幂等。资源删除失败返回可重试错误，不显示已释放；已删除后诊断文件损坏不能回滚取消状态，应单独记录收集失败。已完成的完整产物不被重复关闭改变。

站内路由离开或切换会话，必须等待关闭确认才允许跳转；失败留在当前页并提供重试。刷新/关页使用不受页面 AbortController 影响的同源 cookie keepalive 关闭请求，不再依赖发出 WS stop 后立即断流。页面加载时提供 `GET /sessions` 的所有者活动会话列表，最多50项，可重连或关闭，解决遗失当前页面状态后的管理入口。

创建请求已发出但尚未拿到 Job ID 时，使用已知幂等键 `POST /sessions/close-pending`。创建与关闭按用户/键取得同一 PostgreSQL 事务 advisory lock；关闭先到则提交审计取消标记，迟到启动在创建 Job 前拒绝；创建已提交则查到并取消该 Job。标记及请求不跨用户/操作作用域。客户端不等待可能已丢失的创建响应才请求关闭，晚到的响应与旧轮询不得恢复旧会话。硬刷新请求本身仍依赖网络可达；浏览器崩溃/断网由现有就绪后租约及作业总超时兜底，不把初始化阶段误称为15秒内回收。

## 页面与资产

新页面 `/deploy/simulation`，模型详情及交付页可达。选择模型/兼容环境后点击“启动仿真环境”，内部先预览冻结输入再启动一次会话，默认从站立开始；“检查仿真配置”可单独使用，不作为启动前必点步骤。顶部不提供按行为启动的任务选择；环境运行后，只在右侧切换站立、0.3、0.7，复用同一Run，不重新部署或reset。另提供暂停/继续/重置/停止。显示Job/Run、初始化/运行/暂停/跌倒/收尾/完成状态及实际task、目标/实际速度、推理时延、实时率与警告。

页面支持IP+HTTP访问：幂等键、控制命令和离开页面的stop使用统一随机ID生成。安全上下文可用`crypto.randomUUID`；普通HTTP下使用`crypto.getRandomValues`生成128位随机ID，不要求操作者改用localhost或绕过浏览器安全限制。启动重试仍复用同一冻结预览的幂等键，命令重试仍复用原ID。

画面取自冻结 MJCF 和实际物理 qpos；渲染模型只新增可视地面、灯光及显示质量配置，不修改物理模型。浏览器不下载 STL，不执行 Three.js 渲染。缺帧、渲染失败、协议错误和断流明确展示；相机操作在暂停时也应生效。

## 证据与边界

命令事件逐项flush，保存任务切换、reset、pause/resume、终止原因；每10秒提交轨迹NPZ（真实qpos、动作、target、task/episode/step、限位指标），完成后保留图表和全程5fps真实轨迹录像。离线收尾渲染在会话结束后执行，不阻塞实时物理；UI标为整理录像，不能冒充仍在实时运行。

离线 CPU 录像使用320px、MJLab录像480px；实时流 v2 固定640×480，与证据录像分开。总超时仍严格执行，未完成收尾不标完整。失败/强制取消后，固定事件、快照与已落盘NPZ经ebd扫描登记partial诊断Artifact，允许下载并标complete=false；未落盘的末段不补造。

collector核对run/plan/model/payload/环境身份、文件hash、轨迹/事件/图表/视频完整性，登记诊断Artifact与模型血缘。失败/强制取消可保留已落盘轨迹，但证据不齐不标完整；无需伪造标准case分母或正式通过资格。

验收：指定真实模型在两环境运行、站立→0.3→0.7连续切换，ack真实生效且无重置；暂停/相机/重置/停止/失联租约验证；录像、图表、索引可鉴权访问。新版浏览器目标稳定25–30fps，记录实际 FPS、帧龄、命令及相机响应 p95、带宽和物理实时率；这些是目标，运行实测前不得宣称达标。保留真实50Hz策略与400Hz物理，实际慢于实时必须展示，不改dt追赶。实际 IP+HTTP 浏览器验证无状态轮询、无 STL 下载。

三轮复核：①语义/环境/资产身份；②命令竞态/失联/资源/失败；③实时页面/实际切换/完整证据。正式验收与真机仍单独处理。
