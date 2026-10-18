# UFO 原生 GPU 画面 WebSocket 推流

2026-10-18（Asia/Shanghai）。对应[设计24修订](../design/24-ufo-live-simulation-validation.md)。用户明确拒绝旧 HTTP 轮询加 Three.js 的卡顿表现，并要求可操作相机。本次已部署原生 GPU 画面、持久控制连接及现有模型的连续任务切换，无训练、无 z 生成、无 Git 提交。

入口：[实时仿真](http://1.94.0.106:31800/deploy/simulation?model_id=mv_94d72b2354f040838aea)。登录后选择 v2 环境，点击“启动仿真环境”，默认站立；在右侧切换任务。画面可拖动旋转、滚轮缩放，支持键盘和独立重置相机。切换行为不重启 Run，也不清姿态、上一动作或历史；“重置”才开始同 seed 的新 episode 并暂停。

## 实现与实测修正

任务容器用 NVIDIA EGL 渲染冻结 MJCF 和真实物理 qpos，输出640×480 JPEG；独立渲染线程最多30fps，复制最新状态，与50Hz策略/400Hz物理隔离。原生纹理地面、灯光和抗锯齿只作用于渲染模型。浏览器只解码显示，不运行 Three.js，不拉取 STL。

同源 `/api/v1/ufo-native/live/runs/{rid}/stream` 通过 API 鉴权后接 ebd 媒体 UDS；图像二进制原样转发。任务、相机、心跳和真实状态使用同一 WebSocket；仅 Job/产物元数据保留2秒查询。没有新增端口或服务，任务容器仍 `network=none`、输入只读；派生 v2 Suite/Profile，两后端都申请一张 GPU。v1 对象及记录保留，v1 不假装能建立新推流。

真实测试发现并修正三个影响性能的环节：

1. 基础镜像仅有 Mesa EGL 厂商清单，虽然分配 GPU 仍选择软件驱动。新增受管 NVIDIA 厂商清单；worker 核实实际 GL vendor/renderer，不允许把软件渲染当 GPU。首次失败保留为 `job_f1c1d134c4974022ad87` / `run_a6e7c7d159064a848962`。
2. 逐帧停等 ACK 让约30fps服务端输出在实际 IP 浏览器降到5.45fps。改为8帧有界窗口和累计 ACK，只存序号/时间，不建图像历史队列；满窗口跳过旧帧。单帧≤2MiB、最旧未确认帧2秒超时、单次发送1秒超时。浏览器最多一张正在解码和一张最新待显示帧。
3. ebd 收集器在 Tokio worker 同步扫描所有日志，每约3.6秒让控制/媒体 UDS 同时停顿1.27–1.72秒；API本地401最大39ms，定位到数据面。完整扫描移到 `spawn_blocking`，每轮等待结束再休眠2秒，不并行重扫。之后14秒探测 UDS p95约4.3ms，原周期停顿不再出现；另记录一次约714ms同步尖峰，未归因，不宣称无抖动。

完整保留[停等基线](artifacts/2026-10-18-ufo-websocket/mujoco-stop-and-wait-baseline.json)、[ebd停顿基线](artifacts/2026-10-18-ufo-websocket/ebd-stall-baseline.json)和[修复后探测](artifacts/2026-10-18-ufo-websocket/ebd-stall-fixed.json)。未带 WebSocket cookie 的早期测试被正确拒绝，其 Job `job_47c03636ff7445e2b78f` / `run_0e30a10dc17647d9bb8d` 在15秒失联后完整收尾；正式浏览器测试使用真实登录 cookie。

## 实际浏览器与证据

测试为本机容器内 Chromium149，经实际 `http://1.94.0.106:31800` 访问，安全上下文=false、randomUUID=undefined；使用正常登录 cookie。不是用户远端浏览器或弱网结果。FPS来自真实 JPEG 解码后 canvas 绘制，去除前30帧初始化，采样约12秒；拖动、缩放、暂停和截图包含在流程中。帧龄使用同一主机时钟，从真实物理快照到画布更新计算；页面显示的“接收后时长”不冒充跨机器端到端延迟。

| 项目 | MuJoCo v2 | MJLab v2 |
|---|---:|---:|
| 画布显示 FPS | 27.03 | 27.50 |
| 画面间隔 p95 | 61.1ms | 59.9ms |
| 源快照到显示帧龄 p95 | 88.84ms | 96.11ms |
| 三次任务点击到实际状态确认 | 153–255ms | 154–413ms |
| 拖动 / 缩放首次回显 | 243 / 32ms | 251 / 33ms |
| 图像载荷速率（不含状态/协议开销） | 4.53Mbps | 4.77Mbps |
| 结束前物理实时率 | 0.969× | 0.594× |
| 渲染耗时 p95 | 1.84ms | 1.76ms |
| 轨迹步数 / episode数 | 569 / 2 | 353 / 2 |
| 证据录像 | 57帧 / 11.4s / 320px | 36帧 / 7.2s / 480px |

- MuJoCo：`job_d2882a8c696f430caecd` / `run_80833a6583de43e09f38`；[浏览器记录](artifacts/2026-10-18-ufo-websocket/mujoco-browser.json)、[截图](artifacts/2026-10-18-ufo-websocket/mujoco-native-stream.png)。
- MJLab：`job_232fa2fa12904df79917` / `run_b6fb614d52a14c15ab83`；[浏览器记录](artifacts/2026-10-18-ufo-websocket/mjlab-browser.json)、[截图](artifacts/2026-10-18-ufo-websocket/mjlab-native-stream.png)。

每次仅创建一个会话，站立→0.3→0.7→站立，三个切换事件的 qpos/history 前后指纹相同且 episode=0。暂停期间 step不变但相机可操作；随后 reset 到 episode1、继续、停止。两次均成功登记完整诊断 Artifact，六个索引文件逐一 GET/SHA256验证，录像 Range返回206；[下载核验](artifacts/2026-10-18-ufo-websocket/artifact-audit.json)。两次均观察到0条 HTTP state/scene/control/heartbeat 请求、0条STL下载、0个JS异常。

源码模型与 prompts 未变化：权重 `2041d2d5bb90a679ce5913f6d820c2829fed0266c2bc021744b0fd882a767738`；prompts `2c9bbd88b7645710e2e5981092a49f608c4e941e450efa3bdc39ff48f371e7c1`；payload仍为 `sha256:5da9974b5bbf6701bb0719ae973e96318d48db5157d82b25bff3777b9493a2d2`。已有交付包 `bnd_9e68129a8faf4421a3f1` HEAD200，未被覆盖。

实际镜像：API `sha256:e942a317b200bb2dd00ea672cab2eace10ffbadecd6aab15fe0d1d2c3796f106`；ebd `sha256:ca84e91f001878061b4b57f8115fbd312176dad082da7dc2405faefe401d7f5f`；任务 `sha256:79ab792eeec399e1b979c2f4ddebaaa0bceaeecf3055f62a84df659e424b7adf`。渲染设备实测 NVIDIA GeForce RTX4090，平台通过 `make up` 部署。

## 三轮复核与边界

第一轮核对模型/环境/资产身份、v1不可变性、同Run z切换及相机与物理隔离。第二轮检查有界帧/累计ACK/超时、租约、权限/Origin、失败/关闭和扫描并发；独立复核发现渲染关闭异常原先未传播，已用失败检查复现并修复，不能在渲染失败后发布 complete=true。第三轮使用实际HTTP浏览器确认相机、真实帧率、同Run切换和完整媒体下载，保留所有失败及性能基线。

容器检查：相关后端120项、前端53项、Vue类型检查及Vite构建通过；Rust ebd28项、collector4项通过。WebSocket TestClient 关闭竞态已定位为测试提前取消ASGI清理：测试改为主动关闭并等待服务端正常关闭，重复100次、聚焦3轮及完整后端范围通过；没有吞掉异常或为此改生产服务。

这是交互基础设施验证，不是模型任务通过资格。两次分别累计597/574次关节位置越限、65/64次速度越限，力矩越限0；MJLab物理约0.59×，页面如实提示。未做长时间稳定性、多人观看、远端弱网、WebRTC或真机验证。WebSocket JPEG 带宽约4.5–4.8Mbps；更高分辨率或弱网场景需另测编码/传输方案。
