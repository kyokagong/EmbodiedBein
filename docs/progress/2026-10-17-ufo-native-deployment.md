# 原生 UFO RPO 模型部署与仿真接入

2026-10-17。设计真源：[23](../design/23-ufo-native-rpo-deployment.md)。用户确认本机容器推理、部署前仿真和仿真交付包；禁止 Git 提交。外部来源只读，无新增训练。

**所选范围已完成真实模型端到端验证。** 两后端各 3/3 case、6 段录像与全部轨迹/图表完整；模型 0/6 满足当前任务判据，报告为 diagnostic，不能据此放行真机。

## 当前运行证据

工作台：`http://localhost:31800`。通过模型列表的原生 UFO 导入面板进入；原生模型详情导出 ONNX，再在导出模型详情选择后端、规模并预览提交。所有计算均通过平台 Job → Run → ebd 动态任务容器，无网络。

| 阶段 | Job / Run | 结果 |
|---|---|---|
| 导入 | `job_268852016cfa416ca0bf` / `run_3fb9ae677a2647d39322` | 成功；`mv_ff284c75ef03498ca016`，`art_51c2993e182242368f2a`，649,955,986 bytes |
| 首次导出 | `job_08d47dee383c467c9f00` / `run_6b71bf2066834da09563` | 失败并保留：保存配置的派生 `obs_dims` 不能直接合并进 UFO struct 配置 |
| 修复后导出 | `job_2d786ffc5b1248a3be14` / `run_8a3af660ba4e475eba5d` | 成功；`mv_94d72b2354f040838aea`，`art_bcc4eeb654844567b3ae`，65,702,815 bytes |
| 首轮 MuJoCo smoke | `job_c821af5c2646401d85e1` / `run_82049ebaf2374b6e9f66` | 为补完整诊断取消；已完成 1/3 case 完整保存，报告 `ev_39839e1dc574425caba4` 为 incomplete |
| 首轮 MJLab smoke | `job_9b8be643297b4f248512` / `run_ad2a899f49de4d4b96ed` | 同轮取消，0/3 完成；报告 `ev_47ca2ba7029f4d42aa76` 为 incomplete |
| 第二轮 MuJoCo | `job_3614fb0fe3444a48b043` / `run_d296737bc0964e12afc1` | 发现旧调度器为 CPU 评估默认占卡，取消并释放；未篡改旧 Run 的实际资源记录 |
| 第二轮 MJLab | `job_7494efad31d14af7a916` / `run_f42fac5085364bfb95ff` | 质心读取依赖仅在可选奖励启用的缓存属性，失败并保留；改用实体真实 body 索引 |
| 最终 MuJoCo smoke | `job_ae74ca14ae1847d2bc2b` / `run_f973e42b0004437e8f1c` | Job succeeded，报告 `ev_025c2faa45834db9b473`，3/3 完整、0/3 成功、diagnostic；Docker DeviceRequests=null，CUDA 隐藏 |
| 最终 MJLab smoke | `job_ffc6c67139c842a19578` / `run_6d872c3165eb4b3cb1b8` | Job succeeded，报告 `ev_ae3631f5aa864f18b28e`，3/3 完整、0/3 成功、diagnostic；仅分配 GPU 0 |
| 仿真交付包 | `job_df858595093f4cd2936a` / `run_86581a02f9354eb5b7a3` | Job succeeded；`bnd_9e68129a8faf4421a3f1`，41,534,607 bytes，39 文件，simulation_only / signed=false |

导出 96 组黄金输入对比通过：动作 `atol=1e-5, rtol=1e-4`，实测最大绝对误差 `8.910894393920898e-6`；关节目标误差 `2.4059414863586424e-5 rad`，上限 `1e-4 rad`。闭环另外采集真实 rollout 输入进行 Torch/ONNX 差分。

原始权重 SHA-256：`2041d2d5bb90a679ce5913f6d820c2829fed0266c2bc021744b0fd882a767738`。导出镜像 `sha256:f86f261c22de90ada3c77d07c03c57a746a0f26eab26d6ff0c020b18eafaf5f2`；首轮仿真镜像 `sha256:7686d13d74c317607fb2b3d325c311074472d49088d7bb8a773fc7572cb9f17d`。每个 Job 冻结实际镜像 ID，后续标签更新不改变旧 Run。

最终 CPU 镜像 `sha256:23360d0b731ce7bc161cfec1e33f838828516a5fe985f0f6c21685a28c4fe84f`；最终 MJLab 镜像 `sha256:13c64437f59d27894828b6039bd87c2e64e9f1e5fb75e40ebaa1c1dbb2db4f58`；打包镜像 `sha256:33bcfb60b64a15390842092319a384d6516df2d359b4b584a3b30f63b99c434f`。已评估、已归档的 payload 均为 `sha256:5da9974b5bbf6701bb0719ae973e96318d48db5157d82b25bff3777b9493a2d2`；最终归档 SHA-256 `2bf0ae7c574bf52990a62ff5c5c4298679122b8bfbff4d7fb77922c9face0881`。

## 结果与可访问出口

登录工作台后打开 [导出模型与仿真面板](http://localhost:31800/models/mv_94d72b2354f040838aea)，查看两个后端的报告、图表与逐例录像；[交付包列表](http://localhost:31800/deploy/bundles) 提供下载。也可直接 [下载已核验 .ebp](http://localhost:31800/api/v1/ufo-native/bundles/bnd_9e68129a8faf4421a3f1/download)（需登录）。

所有案例均完成 1000 步 / 20 秒，没有跌倒；五例主要判因为物理越界，一例为 tracking。0.3 m/s 的前向位移未达 4 m，0.7 m/s 两例还有明显速度跟踪偏差；存活不能解释为正常行走。

| 案例 / 录像 | 速度 RMSE m/s | 前向位移 m | 位置越限峰 rad | 速度越限峰 rad/s | 结果 |
|---|---:|---:|---:|---:|---|
| [CPU 站立](http://localhost:31800/api/v1/ufo-native/runs/run_f973e42b0004437e8f1c/files/videos/smoke-0.mp4) | 0.0178 | -0.0184 | 0.00553 | 5.4047 | physical_limit |
| [CPU 0.3](http://localhost:31800/api/v1/ufo-native/runs/run_f973e42b0004437e8f1c/files/videos/smoke-1.mp4) | 0.1633 | -0.1653 | 0 | 0.4303 | physical_limit |
| [CPU 0.7](http://localhost:31800/api/v1/ufo-native/runs/run_f973e42b0004437e8f1c/files/videos/smoke-2.mp4) | 0.8679 | -4.2170 | 0.06661 | 7.7821 | physical_limit |
| [MJLab 站立](http://localhost:31800/api/v1/ufo-native/runs/run_6d872c3165eb4b3cb1b8/files/videos/smoke-0.mp4) | 0.0453 | 0.2512 | 0.00360 | 0.5250 | physical_limit |
| [MJLab 0.3](http://localhost:31800/api/v1/ufo-native/runs/run_6d872c3165eb4b3cb1b8/files/videos/smoke-1.mp4) | 0.1609 | 0.0500 | 0 | 0 | tracking |
| [MJLab 0.7](http://localhost:31800/api/v1/ufo-native/runs/run_6d872c3165eb4b3cb1b8/files/videos/smoke-2.mp4) | 0.8190 | -3.0345 | 0.05134 | 0.7563 | physical_limit |

本轮无成功案例，六段都为真实失败案例录像，没有补造成功媒体。每段 500 帧 / 25 fps / 20 秒。六例实际 rollout 共 60 组 Torch/ONNX 对比，最大动作误差 `1.70805e-6`，最大目标误差 `6.14896e-6 rad`；本机 4 线程 CPU 推理各例 P95 为 2.09–2.20 ms，6000 步中无超过 20 ms 的记录。MJLab 原生观测与交付 codec 对账通过。

原始可复核索引：[CPU 全部指标、文件摘要与访问检查](artifacts/2026-10-17-ufo-native/mujoco-evidence.json)、[MJLab 对应证据](artifacts/2026-10-17-ufo-native/mjlab-evidence.json)、[交付包独立检查](artifacts/2026-10-17-ufo-native/delivery-evidence.json)、[全部 Job/Run 与资源账本](artifacts/2026-10-17-ufo-native/job-ledger.json)。六份 NPZ 的 23 关节、8 子步、真实质心、双足运动/净力和实际观测差分 shape/有限性另外独立检查通过；两后端录像抽帧目视确认机器人/场景完整。

下载包为 gzip 压缩 tar；`manifest.json` 保存来源、逐文件摘要和两份报告回执，`payload/` 保存 ONNX、三个任务提示、codec、资产、许可证、运行依赖回执、黄金输入、示例输入和独立 `runtime.py`。完整原始录像/NPZ 经工作台访问，包内保留其报告引用与 hash。包内 README 给出本机固定镜像中的离线推理命令；连续使用时保留同一个 Policy 实例，episode 开始调用 `codec.reset()`。

## 资源与实施范围

10 个 Job 均已结束，无新增训练或自动重试。累计 Run 时长 39.24 分钟（并行 Run 相加），按实际分配卡数计 28.49 GPU 分钟，含修复前 CPU 操作的默认 GPU 保留，不等于 GPU 忙碌时间。最终 CPU Run 为 0 GPU，MJLab 为 1 GPU；每 Job 上限 4 CPU / 16 GiB，同时两 Job 的组合保留是 32 GiB，与初稿合计 16 GiB 的预算提案不同，未测全程组合内存峰值。十个 Run 的目录合计 878,167,556 bytes（约 0.818 GiB），包含失败/取消记录和 Warp 缓存，不计镜像及外部源。

没有修改数据库表结构、训练栈或外部 UFO/RLinf/BHL/数据仓；已有工作区差异保留，未提交 Git。

## 三轮复核与修复

1. **本体语义和来源隔离**：原生关节顺序、375 维观测的 term-major 四帧历史、256 维提示和 5 倍动作反馈独立冻结；载入原 normalizer。G1 发现原生接口 v1 缺 critic 布局，派生 `ee_rpo_v2_ufo_native_23dof_v2`，不原地修改旧接口。仅复制推理依赖及仿真辅助资产，未复制 optimizer/replay。
2. **运行、资源与失败**：修复旧调度器为 CPU eval/export 默认占一张卡的问题，仅为原生批准操作遵循显式 gpus=0，原有其他栈默认值保持；CPU 镜像同时隐藏 CUDA。MJLab 申请单 GPU。修复派生配置合并、电机延迟初值、被动阻尼和力矩边界；每个 400 Hz 子步记录越界，不用 50 Hz 末态覆盖瞬时越界。取消/失败报告保留已完成 case，逐项验证 Run/plan/model/payload 身份和媒体/轨迹 hash，保持 incomplete。
3. **证据与交付出口**：仿真直接加载交付 payload 中的 `runtime.py`；保存真实观测差分、足部轨迹/接触净力、真实刚体质量加权的质心、原始 NPZ、图表和每例录像。FFmpeg 全量解码后记录媒体证据；包绑定两个完整报告和同一 payload，并在顶层嵌入原始导入 manifest，离线可查源权重/配置/资产摘要，不复制原权重或 motion。旧打包/布尔签名入口拒绝本原生接口；下载走鉴权与 ebd 有界流式 Range。

## 已执行开发检查

- 后端聚焦检查（包含领域契约及调度、取消、失败回归）：87 passed；新增文件 Ruff 检查通过。仅已有 passlib/crypt 弃用警告。
- 前端全部 Node 测试：36 passed；`vue-tsc --noEmit`、Vite 生产构建通过；保留现有大 chunk 提示。
- ebd 容器化 Rust 测试：13 passed，包含新增 10 项路径、符号链接和 Range 测试；Rust fmt 通过。
- 已通过 `make up` 更新平台；任务镜像通过 `make images-ufo-native-deploy` 独立构建。
- 实际 API 拒绝路径：缺两后端完整报告的 bundle preview=400；过期 preview=409；平台 Agent 部署提交=403；通用 `/jobs` 伪造网络计划=400，均未启动计算。
- 已完成的取消案例：鉴权 HEAD=200、Range=206、越界 Range=416；匿名=401；浏览器会话 Cookie 登录后视频尾段 Range=206，模型面板 status=200。
- 六段最终视频、六份 NPZ、四张 SVG 和两份逐例索引：全部登录 HEAD、完整 GET/SHA 通过；全部视频前/后缀 Range=206、越界=416，且 worker 已全量 FFmpeg 解码。
- `.ebp` 完整 GET/SHA、HEAD、Range、39 文件逐项 hash 和源 manifest 链通过；匿名下载=401、旧签名接口=403。相同幂等键重提返回同一成功 Job，没有重复 Bundle。
- 最终平台 healthz 与嵌套 ebd 均 ok；两个模型详情 API 返回 200。
- 最终发布后再次通过 ebd 导入预览复核原始 612,798,672-byte 权重 SHA 未变，未创建第二个导入 Job；交付包 HEAD 仍返回 200 / 41,534,607 bytes。

## 边界与待验证

已完成所选本机导入、推理、双后端 smoke、媒体和交付出口。Smoke 只给诊断资格；12-case acceptance 配置/API 已实现，未运行完整验证，按当前逐例耗时不能证明其能在默认 30 分钟内完成，工作台暂禁用该选项。Orin/其他设备、硬件实时性和真机均未验证；本机推理时延不能代替实机实时性。两种后端同属 MuJoCo 系，未覆盖独立物理引擎。

工作台当前按固定套件一次覆盖三个任务；单任务 GUI 仿真尚未单独实现，离线 runtime 可按冻结任务名选择提示。未增加常驻推理服务或浏览器实时遥控。

质心/足部派生状态沿用原生时序，可能比积分 qpos 滞后一个 2.5 ms 物理子步；GPU 峰值字段仅为 Torch allocator，未覆盖 Warp 的全部显存。提示生成来源及历史训练源码身份只作外部声明，许可证随包保留；交付包限定 simulation_only、signed=false。未运行全仓 backend/Rust workspace 测试或真实浏览器自动化，已执行范围见上文。
