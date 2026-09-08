# 01 · 系统架构与数据面深度审查报告

- **对照设计**：`docs/design/01-architecture.html`
- **实现范围**：`deploy/`、`rust/crates/*`、`backend/eb/store/`、`backend/eb/worktree.py`
- **成熟度评分**：**65%**

---

## 1. 架构原则对照核查

| 原则 | 设计要求 | 当前实现状态 | 判定 |
|---|---|---|---|
| **1. 可复现优先于可用性** | 凭 `code_ref + config_hash + data_version + image_digest + seed` 五元组重跑 | JobSpec 记录了部分 hash；但 image digest 未强制钉死，外部 data version 存在 `unknown`，五元组对比 UI 较薄 | ⚠️ 部分符合 |
| **2. 外部仓库只读** | 绝不写 `/data/kk/*`，通过 worktree 快照 + 只读挂载 | 已实现 `git clone --no-hardlinks` 隔离，容器内 `logs/` 与 `outputs/` 打洞至 overlay，源仓 porcelain 干净 | ✅ 符合 |
| **3. 执行一律容器化** | 控制面永不宿主 `subprocess` 跑训练，所有计算走 Docker 容器 | 完全由 `ebd` 动态起容器，compose 内不塞任务容器，控制面零训练进程 | ✅ 符合 |
| **4. 控制面无状态** | FastAPI 与 ebd 随时可重启，靠 PG 与容器 label reconcile | 基本具备，调度器有 `reconcile` 逻辑，容器带 `eb.run_id` 标签；但宿主重启后的深层状态补偿待增强 | ⚠️ 基本符合 |
| **5. 热路径旁路** | 指标、日志、视频、大文件绝不流经 Python 请求循环，走 Rust 数据面 | Rust TSDB 已接收 tfevents，但媒体流反向代理仍在 Python 做 Range 转发；未实现 HLS 预切片 | ⚠️ 部分符合 |
| **6. 安全默认拒绝** | 真机、删除、覆盖已发布产物必须人工确认，Agent 权限白名单 | 真机 promote 403 阻断，Agent 禁止动作硬拦截生效，已发布环境不可原地修改 | ✅ 符合 |

---

## 2. 进程与通信拓扑审查 (IPC)

```
设计目标拓扑：
[ 浏览器 (Vue SPA) ]
       │ REST / WebSocket / Range
       ▼ (:8000 单一暴露端口)
[ eb-api (FastAPI) ] ─── gRPC over UDS (/run/eb/ebd.sock) ───► [ ebd (Rust) ]
       ▲                                                           │ Docker Engine API
       │ Redis Streams 事件总线 (解耦，断线可续读)                  │
[ Redis (:6380) ] ◄────────────────────────────────────────────────┴─► [ 任务容器 ]
```

### 现状与差距分析：

1. **FastAPI → ebd 控制通道**：
   - **设计**：`gRPC over Unix Domain Socket`，具备强类型、流式支持与低开销。
   - **现状**：当前采用 `HTTP-over-UDS` (FastAPI 经 `httpx.AsyncHTTPTransport(uds=...)` 发送 REST JSON 请求，`ebd` 侧用 `axum` 监听 UDS)。
   - **评价**：单机场景下 HTTP-over-UDS 足够轻量且调试方便，属于合理的务实取舍；但需要保持 API 版本协商，避免协议静默失步。

2. **ebd → FastAPI 事件回传通道**：
   - **设计**：`Redis Streams` 解耦回传（Rust 将指标聚合与容器事件写入 Streams，FastAPI 异步消费并更新 Job 状态，支持崩溃后 last-id 续读）。
   - **现状**：当前控制面热路径**尚未真正激活 Redis Streams 消费**。FastAPI 调度器通过轮询 ebd HTTP 接口及直接读取 PG/TSDB 来驱动状态机；Redis 仅作为依赖部署在 6380 端口。
   - **风险**：高频指标更新与状态事件会增加控制面轮询负载，缺乏事件回溯能力。

3. **媒体与大文件 Range 流**：
   - **设计**：`eb-artifact` 提供零拷贝 Range 分发与 HLS 切片。
   - **现状**：FastAPI `/media/*` 路由通过 UDS 流式反代 ebd，ebd 实现了基础的文件 Range 读取，但尚未集成 HLS 视频转封装及雪碧图（Sprites）生成。

---

## 3. Rust Workspace 与 Crate 划分审查

对照设计 10.3 要求的 9 个 crate 体系：

```
rust/crates/
├── eb-core/        ✅ 已拆分：共享 GpuInfo, HostProfile, sha256
├── eb-runner/      ✅ 已拆分：Docker 生命周期、日志泵、waiter、退出码判定
├── eb-collector/   ✅ 已拆分：tfevents 解析、指标归一化、写 TSDB
├── eb-scanner/     ✅ 已拆分：目录扫描、格式嗅探、哈希计算
├── eb-artifact/    ✅ 已拆分：CAS 基础读写
├── eb-tsdb/        ✅ 已拆分：append-only 段文件存储与查询
├── eb-probe/       ✅ 已拆分：硬件探测 (ebd doctor / NVML)
├── eb-cli/         ✅ 已拆分：ebctl 运维工具
└── ebd/            ✅ 已拆分：单二进制主入口，子命令路由
```

### 审查结论：
- **优点**：Rust Workspace 已经从早期的单一 crate 彻底解耦为 9 个职责分明的 crate，符合设计规范。
- **不足**：
  1. `eb-runner` 依然依赖 `std::process::Command` 调用宿主 `docker` CLI，尚未完全切换至 `bollard`（异步 Docker API 库）。在并发管理大量短时任务时可能产生进程衍生开销。
  2. `eb-cli (ebctl)` 功能较少，目前仅支持 `health`, `doctor`, `ps`，缺少设计中的 `ebctl rescue`, `ebctl replay`, `ebctl cas verify` 等救援运维子命令。

---

## 4. 目录布局与隔离机制审查

| 目录路径 | 设计用途 | 现状 | 改进建议 |
|---|---|---|---|
| `/var/lib/eb/repos/bhl/<sha>` | git worktree 快照 | 采用 `git clone --no-hardlinks` 替代原生 worktree，子模块通过 `cp -a` 复制 | 需实现 LRU 淘汰机制（保留最近 10 个 commit），并建立并发克隆文件锁 |
| `/var/lib/eb/runs/<id>/` | Run 独占可写目录 (logs/outputs/configs) | 结构完整，具备 `console.log`, `outputs/`, `configs/` | 缺少冻结的 `spec.json` 与 `cmd.sh`（用于一键复现和复制等价命令） |
| `/var/lib/eb/cas/` | 本地热 CAS 内容寻址 | 按 sha256 目录存储生效 | 需补充引用计数、GC 标记与定期清理水位线（Watermark） |
| `/var/lib/eb/tsdb/` | 时序段文件 | append-only `.jsonl` / 段结构生效 | 需补充服务端自适应降采样算法（Downsampling），支撑前端对比 8 曲线 × 6000 步 |
| `/var/lib/eb/cache/` | 跨 Run 共享缓存 (hf/torch/uv) | 已在 Runner 中声明挂载 | 需确保并发下载时的 `flock` 进程锁可靠 |

---

## 5. 改进计划与行动项

1. **[P0] 补齐 Run 目录复现工件**：在 `Runner::start` 时，在 `/var/lib/eb/runs/<run_id>/` 下显式写入冻结的 `spec.json` 与可直接执行的 `cmd.sh`。
2. **[P1] 打通 Redis Streams 事件链路**：在 `eb-collector` 与 `eb-runner` 中将容器状态变更和指标聚合推入 Redis Streams `eb:events`，FastAPI 启动后台协程消费事件并驱动状态机。
3. **[P2] 完善 TSDB 降采样与对齐**：在 `eb-tsdb` 中实现 LTTB (Largest Triangle Three Buckets) 降采样算法，使对比多条长曲线时响应延迟控制在 50ms 内。
4. **[P2] ebd 容器与 CLI 深度集成**：`eb-runner` 逐步迁移到 `bollard`；`ebctl` 补齐 `cas verify` 与 `doctor --json`。
