# UFO 原生画面 WebSocket 推流实施计划

> 执行使用 executing-plans；独立的前端和传输边界通过 dispatching-parallel-agents 分工。用户禁止 Git 提交，所有步骤不提交。

目标：用服务端 GPU 画面和持久控制连接取代卡顿的 HTTP 轮询与 Three.js，保持同 Run 连续切换 z 和可操作相机。

设计真源：[24](../../design/24-ufo-live-simulation-validation.md)。技术：MuJoCo EGL、Pillow JPEG、Rust axum WebSocket、Vue canvas；复用 Job/Run/ebd、鉴权、租约、证据收集。

- [x] 渲染与资源：先验证 v1/v2 不可变性、单帧编码和快照隔离；发布 v2 环境，独立渲染线程限30fps、有限容量、原子发布帧、错误终止；构建任务镜像。
- [x] 传输与控制：验证身份/Origin/路径/输入边界和帧 ACK；ebd 固定通道 WS，API 鉴权透传；任务幂等、心跳、相机分离；Rust/Python 聚焦检查。真实测试后将停等改为8帧窗口，并隔离阻塞日志扫描。
- [x] 页面：去除 pose/scene 轮询和 Three 渲染，二进制解码后 ACK、相机指针/滚轮/键盘、断流重连和实际 ack；Node/类型检查。
- [x] 部署与真实验收：make up；通过平台提交模型 mv_94d72b2354f040838aea 的有限时长会话；实际 IP+HTTP 浏览器测帧率、帧龄、控制/相机响应与带宽，连续切换与暂停/reset，检查录像/轨迹。
- [x] 三轮复核：模型/物理/相机语义隔离；慢端/断线/资源/失败/旧计划；浏览器可见性与完整证据。记录实际结果及未达目标，不用测试通过代替性能验收。

结果及未验证边界见[运行记录](../../progress/2026-10-18-ufo-websocket-stream.md)。不代表长期稳定性、远端弱网或模型任务资格通过。
