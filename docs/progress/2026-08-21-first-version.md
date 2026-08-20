# EmbodiedBein 第一版实现对照与后续计划

- 日期：2026-08-21
- 对照真源：`docs/design/`（尤其 `10-roadmap.html`、`09-ui-system.html`、`01-architecture.html`）
- 代码版本：仓库 `main` 上的可部署骨架（`make up` 已在本机通过）
- 判定口径见 [README](./README.md)

---

## 1. 一句话结论

第一版已经把设计里的 **M0–M5 模块边界、进程拓扑和领域内核** 搭起来，并且能 `make up` 打开浏览器。  
它是「可编译、可部署、可纳管、可提交占位任务」的平台，**还不是**设计验收意义上的训练/评估/上机闭环。

当前最有价值的已完成能力：

1. 四个 Embodiment Contract 落库并自洽校验。
2. 遗留 ONNX / tfevents 扫描入库，Showcase 可分享。
3. JobSpec → G1 → 容量估算 → 队列 → `ebd` 下发（镜像缺失时 stub 成功）。
4. Agent 禁止集生效（真机放行 / 改契约 / 门禁豁免），规则路径 `llm_calls == 0`。

当前最大的缺口：

1. 真实 Isaac / RLinf / MuJoCo 任务镜像未构建，训练不会真正跑起来。
2. `ebd` 仍是单一 crate 内的最小实现，设计中的 runner / collector / tsdb / scanner / artifact 尚未拆成独立 crate。
3. 31 个页面里，约三分之二仍是 `Page.vue` 的 JSON 回显，没有设计系统里的表格、曲线、DAG、视频墙。
4. GPU 探测、worktree 隔离、产物认领、实时日志、五元组复现都还没接到真路径。

---

## 2. 已验证的部署事实（2026-08-20/21）

在本机执行 `make up` 后：

| 检查 | 结果 |
|---|---|
| 后端单测 | `uv run pytest -q` → 16 passed |
| 前端构建 | `vite build` 成功，打进 api 镜像 `web/dist` |
| Compose | `api` / `ebd` / `postgres` / `redis` healthy；MinIO 运行中 |
| `/healthz` | `{"ok": true, "ebd": {"ok": true, "service": "ebd"}}` |
| 登录 | `kk` / `embodied` |
| 契约 | 4 个全部落库 |
| 遗留模型 | checkpoints 下 6 个 ONNX 已纳管；扫描上限内共约 80 个模型版本 |
| 历史 run | tfevents 导入约 20+ 条 Job |
| 提交 `train.rl` | 入队后因镜像缺失走 stub，状态 `succeeded` |
| Agent 护栏 | 「真机放行」返回 `guardrail=blocked`，`llm_calls=0` |
| Showcase | `/showcase/bhl-progress-demo` HTTP 200，无需登录 |

默认入口：http://localhost:8000

---

## 3. 按里程碑对照（M0–M5）

### 3.1 M0 骨架 + 只读纳管

设计验收：干净机器 `git clone && make up` 能打开；≥6 历史策略、≥10 条历史曲线、4 个契约；策略能做 I/O 与契约比对；Showcase 可发同事。

| 设计项 | 状态 | 现状 |
|---|---|---|
| compose + 两个 Dockerfile | **完成** | `deploy/docker-compose.yml`、`api.Dockerfile`、`ebd.Dockerfile` |
| FastAPI + Postgres schema + Vue 静态托管 | **完成** | Alembic `0001_init`；SPA 由 api 托管 |
| 四个契约落库 | **完成** | `contracts/*.yaml` → `embodiment_contracts` |
| 扫描 checkpoints ONNX | **完成** | `eb/ingest.py`，按 sha256 去重，文件名推断契约 |
| 扫描 logs tfevents | **可用骨架** | 有轻量 TFRecord 解析；曲线进 `metrics_summary.series`，不是独立 TSDB |
| berkeley_ego_data 目录清单 | **可用骨架** | 种子数据集 + 漏斗 155k→35；未做真实目录级完整性校验 |
| 策略 I/O 与契约比对 | **可用骨架** | 模型详情跑 G3；I/O 目前主要来自文件名启发式，不是 ONNX 实测 |
| Showcase | **可用骨架** | 公开 token 快照页可打开；无视频、无趋势图、无过期 UI |

M0 主路径已经可用。缺口是「看见」的质量：曲线/Showcase 还很薄，ONNX 维度还不是实测。

### 3.2 M1 任务下发闭环（Isaac / RSL-RL）

设计验收：UI 提交 `train.rl` 看到实时曲线与日志；产物自动入库；上游仓 `git status` 干净；同一 JobSpec 两次提交五元组一致；4 GPU 满载第 5 个任务排队。

| 设计项 | 状态 | 现状 |
|---|---|---|
| Isaac Lab 适配器（命令行快照） | **完成** | `isaaclab_rslrl.py` + `tests/adapters/test_isaac_snapshot.py` |
| 容量估算器 L2 | **完成** | OpenVLA 全参 4×24GB → `no_fit` 已有单测 |
| 单机调度器 | **可用骨架** | 3s tick；无 NVML 时虚拟 4 卡池；无老化/抢占 |
| `ebd` runner | **占位** | `docker run` 走 CLI；镜像不存在则 `stub_finished` |
| worktree 快照 + overlay | **未做** | 未隔离上游仓写路径 |
| eb-tsdb + 采集 | **占位** | 指标在 Postgres JSON；无独立时序库、无 tfevents 泵 |
| Run 实时曲线与日志 | **占位** | Run 页 SVG sparkline；无 WS 日志流 |
| 产物认领 | **未做** | stub 不认领 overlay 里的 checkpoint/onnx |
| 五元组复现 | **未做** | JobSpec 有 hash，但 image digest / config hash / data version 未钉死 |
| GPU 满载排队 | **占位** | 逻辑有 `need vs free`；`ebd doctor` 当前看不到真实 GPU |

M1 是下一阶段投入产出比最高的工作。没有真实镜像和 runner，平台只能「演示提交」。

### 3.3 M2 评估、门禁与对比

设计验收：固定 seed 两次成功率一致；门禁拦住已知坏策略并给出断言；对比页标「差异不显著」；6 个预置策略出基线榜。

| 设计项 | 状态 | 现状 |
|---|---|---|
| G1–G5 + 门禁 DSL | **完成** | `eb/gates/engine.py`，domain 单测覆盖 G1/G5/契约自洽 |
| EvalSuite + seed 注册表 | **可用骨架** | 表结构 + 种子套件 `suite_biped_locomotion_v2`（8 seeds，frozen） |
| `play.py --metrics_output` 解析 | **未做** | 适配器能渲染 play 命令，没有结果解析器 |
| 评估报告 / 失败视频墙 | **占位** | API 有 reports；UI 是 JSON |
| Run 对比 + 显著性 | **未做** | 路由存在，无 diff / Pareto / 统计 |
| 排行榜与基线 | **占位** | `/evals/leaderboard` 按 success_rate 排序；无 6 策略实测榜 |

### 3.4 M3 数据模块 + RLinf

设计验收：提交一次 RLinf SFT 并看到指标；漏斗展示 155k→35 与拒绝原因；Episode 预览音画同步；数据集版本哈希可复现。

| 设计项 | 状态 | 现状 |
|---|---|---|
| 漏斗审计视图 | **可用骨架** | Funnel 页展示种子 stages；无拒绝原因表、无敏感性滑块 |
| RLinf 适配器 | **可用骨架** | Hydra 覆盖 + ray_port 默认 6390；未实测容器内 Ray |
| ebd scanner | **占位** | `POST /v1/scan` 只数文件/字节，无格式嗅探 |
| 六层数据模型 | **占位** | Dataset / DatasetVersion 两层；缺 episode/slice/canonical 等 |
| Episode 预览器 | **未做** | 路由在，无 CAS 视频+动作曲线联动 |
| 数据转换/切片 Job | **占位** | JobKind 有枚举，tools 适配器是空命令 |
| 重定向 `data.retarget` | **未做** | 设计已标为可选高风险子项，第一版刻意不做 |
| 数据集版本哈希 | **未做** | 种子 `content_hash=unknown` |

### 3.5 M4 部署流水线与遥测

设计验收：checkpoint 一键签出策略包；MJCF 限位改错被 sim2sim 拦住并指出关节；真机 jitter/温度可见；离线可回滚。

| 设计项 | 状态 | 现状 |
|---|---|---|
| 策略包实体 + X-Confirm 签名 | **可用骨架** | `POST /bundles/{id}/sign` 要求头匹配；无打包、无签名密钥 |
| 导出自检 ONNX vs PyTorch | **未做** | |
| sim2sim 门禁 | **占位** | MujocoAdapter 能渲染 `play_mujoco.py`；无数值比对 |
| 资产一致性 | **占位** | `/assets` 列出契约里的 mjcf/usd 路径 |
| S0/S1/S2 放行 | **完成（拒绝）** | `POST /releases/{id}/promote` 固定 403，符合「人按按钮」 |
| Fleet / 机器人台账 | **占位** | 种子 `robot_bhl_01` |
| robot-agent 接包+上报 | **占位** | 二进制只打印身份；遥测 API 只计数 |
| 离线回滚 | **未做** | |

### 3.6 M5 Agent

设计验收：数字可点击溯源；禁止动作拦截并留审计；预算耗尽降级只读；关 LLM 后规则/剧本仍工作；磁带回放 `llm_calls==0`；权限矩阵快照测试。

| 设计项 | 状态 | 现状 |
|---|---|---|
| 禁止集 | **完成** | `deploy.release` / `contract.modify` / `gate.override` / `contract.create` |
| 7 个剧本注册 | **可用骨架** | 有目录与关键词分发；步骤是规则文本，不是真工具编排 |
| 权限矩阵由注册表生成 | **完成** | `GET /agent/tools`；单测覆盖 |
| 关 LLM 仍可用 | **完成** | 当前根本没有 LLM 客户端 |
| 审计 | **可用骨架** | `agent_steps` 落库；审计页 JSON |
| ToolBus 七道处理 | **占位** | 有 check/call/approval/forbidden；缺预算硬顶、速率令牌桶、补偿、磁带 |
| 场景磁带回放 | **未做** | 只有 playbook 单测，不是设计里的 tape fixture |
| 数字溯源 ref# | **未做** | |
| 预算耗尽降级 | **未做** | 设置了 `EB_AGENT_WEEKLY_GPU_HOURS`，未执行 |
| LLM 接入 / MCP 真导出 | **占位** | `GET /agent/mcp` 返回工具列表 JSON，不是 MCP server |
| 控制台 UI | **可用骨架** | 能发消息；无工具轨迹、待审批、预算条、WS 流式 |

---

## 4. 按子系统对照

### 4.1 控制面 Python（`backend/eb`）

已实现：

- domain：Contract、JobSpec、Job/Run 状态机、Violation/CheckResult（纯逻辑，无 IO）
- gates：G1 提交、G2 数据维、G3 模型 I/O、G4 导出 YAML、G5 真机、DSL
- adapters：isaaclab_rslrl / rlinf / mujoco / tools / native / deploy / lerobot（后两者极薄）
- capacity：L2 解析公式 + 凭据 + alternatives
- orchestrator：提交、Idempotency-Key、G1、no_fit 拒绝、admit_and_start
- scheduler：队列 + 虚拟 GPU 池
- ingest：ONNX/pt 去重、tfevents 尽力解析
- harness：工具注册、禁止集、审批表达式
- pluginhost：load/unload Context
- API：鉴权（Cookie 会话 + Bearer）、合约/数据/模型/任务/评估/包/Agent/Showcase/容量/扫描
- 种子：用户 `kk`、API Key、4 契约、ego 数据集、Showcase token `bhl-progress-demo`

未实现或过薄：

- API 未按资源分模块（设计要求 `api/` 分文件，现在几乎全在 `routes.py`）
- Redis 已部署但控制面几乎未用（无 Streams、无速率计数）
- MinIO 已部署但产物不走 S3/CAS
- 血缘 `lineage.add_edge` 存在，训练成功后不自动写边
- 编排无 DAG/Pipeline 执行器（只有表）
- 没有 APScheduler 真正挂剧本 cron（依赖在 pyproject 里，未用）
- 搜索是内存扫 20 条，不是设计中的检索

### 4.2 数据面 Rust（`rust/`）

设计要求 workspace 拆：`eb-core` / `eb-runner` / `eb-collector` / `eb-scanner` / `eb-artifact` / `eb-tsdb` / `eb-probe` / `eb-cli` / `ebd`。

当前实际：

```
rust/crates/
  eb-core/     # GpuInfo, HostProfile, sha256
  ebd/         # 单一 crate：serve + doctor + scan + docker run + ebctl
```

| 能力 | 状态 |
|---|---|
| 双 UDS（`/run/eb/ebd.sock` + `ebd-http.sock`） | 完成 |
| `ebctl health/doctor/ps` | 可用骨架 |
| doctor：hostname / 内存 / `/dev/dri` | 完成 |
| doctor：NVML / nvidia-smi | 未做（容器内无 nvidia-smi，返回空 GPU 列表） |
| docker run + label `eb.run_id` | 可用骨架（无 GPU/shm/overlay/worktree） |
| 镜像缺失 stub | 有意为之的第一版行为 |
| 日志泵、退出码 reconcile | 未做 |
| tfevents 采集写 TSDB | 未做（Python ingest 旁路了这条设计） |
| CAS 内容寻址 | 占位（`cas_put` 未接入路径） |
| media Range 代理 | API 有反向代理；ebd 读整个文件，无 Range |
| bollard | 未用，走 `docker` CLI |

这与架构原则「热路径旁路 Python」不一致：历史曲线现在是 Python 解析后塞进 Postgres JSON。

### 4.3 前端（`frontend/`）

设计：Vue 3 + TS + Vite + Pinia + Router + **自建设计系统** + ECharts + Vue Flow + three.js（懒加载）；31 页。

| 类型 | 页面 | 状态 |
|---|---|---|
| 专用视图 | `/` 总览、契约列表/详情、提交、队列、Run、模型详情、数据集详情、漏斗、Showcase、Agent、登录 | **可用骨架**（功能在，视觉与交互远低于 09 章原型） |
| JSON 回显 `Page.vue` | 数据列表、切片、转换向导、血缘、资产、3D viewer、实验、流水线、Run 对比、评估套件/报告/对比/失败库/排行榜、策略包、机器人、Agent 审计、设置 | **占位** |

仍缺的设计组件：虚拟表格、KPI 环、GPU 甘特、ECharts 网格、Vue Flow DAG、失败视频墙、Episode 双轴预览、three.js 关节树。

鉴权体验：前端写死 Bearer `eb-dev-key`，登录会话与 API Key 两条线没有统一；未登录其实也能靠内置 key 调 API。Showcase 数据请求同样带 key，只是 shares 接口本身免鉴权。

### 4.4 部署与执行镜像

| 项 | 状态 |
|---|---|
| 单机 compose 唯一入口 | 完成 |
| Redis 6380（避让 Ray 6379） | 完成 |
| 任务容器不进 compose | 完成 |
| 外部仓只读挂载 `/data/kk` | 完成 |
| 仅 ebd 持有 `docker.sock` | 完成 |
| Caddy `--profile tls` | 文件在，未作为默认 |
| Isaac / RLinf / MuJoCo / tools 镜像 | **busybox 占位**，`make images` 不会得到可训练镜像 |
| ebd NVIDIA runtime | 未配，故看不到 4×4090 |
| `Cargo.lock` | 未提交，每次 Docker 构建现场解析（clap 已钉死 4.5.32 以兼容 rustc 1.83） |

### 4.5 插件与端侧

| 包 | 状态 |
|---|---|
| `eb_plugin_isaaclab` / `rlinf` / `bhl` | 只有 Manifest + `setup` 注册字符串，未对接到 Adapter 加载 |
| `robot-agent` | 打印身份 JSON；未拉包、未上报、未离线缓存策略 |

---

## 5. 对照设计仓库树：结构债

第一版为了尽快 `make up`，把多个 Rust crate 和若干 Python 包**压进了最少文件**。后续应按设计拆回去，而不是继续在 `ebd/src/main.rs` 和 `api/routes.py` 上堆：

必须补齐的目录（设计 10.3，当前缺失或名不副实）：

- `rust/crates/eb-runner`
- `rust/crates/eb-collector`
- `rust/crates/eb-scanner`
- `rust/crates/eb-artifact`
- `rust/crates/eb-tsdb`
- `rust/crates/eb-probe`
- `rust/crates/eb-cli`（现为 `ebd` 第二 bin）
- `backend/eb/api/` 按资源拆路由
- `backend/eb/adapters/data_*.py`
- `docs/adr/`
- `backend/tests/agent/` 磁带夹具（现仅有 toolbus/playbook 单测）

这些拆分本身不增加用户可见功能，但会挡住 M1 采集与 M3 scanner 的正确归属（控制面不应解析 tfevents）。

---

## 6. 已知缺陷与风险（实现层）

按「会在下一轮咬人」排序：

1. **GPU 不可见**：ebd 镜像无 `nvidia-smi`、无 NVIDIA Container Toolkit；调度只能虚拟 4 卡。4 卡满载排队的验收做不到。
2. **stub 成功会误导**：镜像不存在时 Job 标 `succeeded` 并写假 `Train/mean_reward`。应改为 `infra_failed` 或显式 `stub` 状态，避免污染排行榜。
3. **ONNX I/O 是文件名猜的**：`policy_video.onnx` 归到 fixed-hand 只是启发式；G3「通过」可能是假阳性。
4. **ingest 扫 `.pt` 过宽**：rsl_rl 中间 checkpoint 也会进模型表（上限 80），M0「6 个历史策略」被稀释。
5. **控制面解析 tfevents** 违反「Python 不解析 tfevents」。
6. **idempotency 与复现混淆**：现在只有显式 `Idempotency-Key` 才去重；同一 JobSpec 两次提交会生成两个 Job，五元组也未钉死。
7. **WebSocket** `/ws/queue` 握手后立即关闭。
8. **会话 Cookie vs 写死 Bearer**：单用户模式能用，但 Showcase/审计里的「谁」不干净。
9. **MinIO 用户名**：MinIO 要求 root user ≥3 字符，已改为 `ebadmin`（`.env.example`）。
10. **api 镜像重复编译 Rust**：只为拷贝 `ebctl`，拉长 `make up`。

---

## 7. 后续优化计划

原则与设计 10.1 一致：**每个切片独立可用**；先记录后控制；Agent 的 LLM 放最后。  
不在本计划里改设计文档。冲突时改代码。

建议节奏：P0 先把已经能打开的系统变成「真的能训一条 biped」；P1 把评估接到现有 `play.py`；其后才是数据、上机、Agent 增强。

### 阶段 P0 — 把 M0/M1 做成真（优先，约 2–3 周）

目标：从 UI 提交一次真实 `train.rl`，在 4090 上看到曲线和日志，产物入库，上游仓保持干净。

#### P0.1 硬件与 runner（阻塞项）

1. ebd 容器接入 NVIDIA runtime / 挂载 `nvidia-smi`，`GET /v1/doctor` 返回 4×4090 的 index/name/vram。
2. 把 stub 成功改为：镜像不存在 → Job `infra_failed`，UI 明确提示跑 `make images`；禁止写假 reward。
3. runner 补全：`--gpus`、`--shm-size`、`--device /dev/dri`、label、退出码、`docker logs` 泵到文件、超时 kill。
4. 用 `docker ps --filter label=eb.run_id` 做重启 reconcile，避免控制面与容器脱节。

验收：`ebd doctor` 列出 4 张卡；故意提交不存在的镜像会失败而不是成功。

#### P0.2 真实 Isaac 镜像与一次手工容器

1. 把 `deploy/images/isaac.Dockerfile` 从 busybox 换成可跑 BHL `train.py --headless` 的镜像（先手工 `docker run` 通，再写进适配器）。
2. 固化 GPU 渲染/egl 参数，记入 ADR（新建 `docs/adr/`）。
3. 快照测试继续锁命令行，避免上游改参数名时无声失败。

验收：不经过平台、只 `docker run` 能完成一次短迭代 biped 训练。

#### P0.3 工作区隔离

1. 对 `/data/kk/Berkeley-Humanoid-Lite` 做 git worktree 快照，只读 bind 进容器。
2. `logs/`、`outputs/`、`configs/` 落到 `/var/lib/eb/runs/<run_id>/` overlay，禁止写回上游仓。
3. 跑完后宿主 `git -C Berkeley-Humanoid-Lite status` 必须干净。

验收：设计 M1 第三条。

#### P0.4 采集与 TSDB 最小版

1. 按设计把 collector 放到 Rust：跟踪容器内 tfevents，归一化 tag，写入 `/var/lib/eb/tsdb`。
2. `GET /api/v1/runs/{id}/metrics` 改读 TSDB，不再依赖 Postgres JSON。
3. Run 详情：ECharts 实时曲线 + 日志尾部（WS 或 SSE，热路径仍走 ebd）。

验收：一次真实训练过程中刷新 Run 页能看到 `Train/mean_reward` 增长。

#### P0.5 产物认领与契约实测

1. overlay 里的 `exported/policy.onnx` / `model_*.pt` 按 sha256 进 CAS（MinIO 或本地 CAS，二选一，优先本地 CAS 符合单机）。
2. 用 ONNX 图输入输出维做 G3，替换文件名启发式。
3. 模型列表默认只展示「策略」类产物，中间 ckpt 折叠。

验收：6 个 checkpoints ONNX 显示真实 obs/act，并能给出与四个契约的通过/失败矩阵。

#### P0.6 调度与五元组

1. doctor 的 `run_id` 回填到 GPU 占用；第 5 个 1-GPU 任务必须排队。
2. 提交时记录 `code_ref.commit`、`image_digest`、`config_hash`、`data_version`、`seed`。
3. 同一 JobSpec 两次提交：若带 Idempotency-Key 则返回同一 Job；若要复现实验则新 Job 但五元组可对比。

验收：设计 M1 第四、五条。

#### P0.7 M0 可视化补完（与 runner 可并行）

1. 总览：KPI、最近活动、队列摘要、资源环（不要再只显示四个数字）。
2. 契约页：观测 layout 条、引用该契约的模型/数据集。
3. 提交页：表单化 JobSpec + G1 预检 + `equivalent-command` 预览。
4. 队列页：GPU 条 + 排队列表，接 `/ws/queue`。
5. Showcase：用真实模型数/run 数生成快照；加曲线缩略图。

验收：P0 页面不再依赖 `Page.vue` JSON 回显。

---

### 阶段 P1 — 评估与门禁接到真脚本（约 2 周）

目标：现有 6 个 ONNX + 冻结 seed 套件跑出第一张基线榜。

1. 解析 BHL 已有 `play.py --metrics_output` / eval json（仓库 `logs/rsl_rl/wbc/.../eval_*.json` 已存在，可先离线导入再在线跑）。
2. `eval.sim` Job → 固定 seeds → EvalReport；两次提交成功率必须逐 seed 一致。
3. 门禁 DSL 接到报告：已知坏策略（标定过期、obs 维错误）必须被 G3/G5 拦住，断言 path 可读。
4. 评估报告页：门禁面板、成功率矩阵；失败视频墙可第二步再做。
5. Run 对比：参数 diff + 曲线叠加；显著性先做简单重叠/置信区间，文案「差异不显著」按设计出现。
6. 排行榜：6 个预置策略 + baseline 标记。

验收：设计 M2 四条。

---

### 阶段 P2 — 数据主线（不含重定向）+ RLinf 打通（约 3 周）

目标：漏斗可审计、Episode 能看、SFT 能提交。重定向保持可选，失败不挡主线。

1. 把 scanner 移到 Rust：格式嗅探、完整性、内容哈希；Python 只存元数据。
2. 补齐数据版本模型：raw → valid episode → scenario；漏斗每一级拒绝原因可查询。
3. Episode 预览：CAS 代理 Range 流视频 + 动作曲线时间轴对齐（09 章 P0 页）。
4. 数据集 `content_hash` 可复现。
5. 手工跑通一次 RLinf 容器（Ray 端口 6390+、与 Redis 6380 隔离），再固化适配器。
6. 提交 `train.sft` 看到指标；容量估算继续挡住 OpenVLA 全参 4×24GB。

验收：设计 M3 主线三条。重定向单独立项，不写入本阶段必达。

---

### 阶段 P3 — 部署与真机最小闭环（约 3 周）

1. `bundle.build`：从 checkpoint 打策略包（契约、ONNX、deploy.yaml、资产摘要）。
2. 导出 G4：ONNX vs PyTorch 数值比对；失败不得签名。
3. sim2sim：改 `play_mujoco.py` 输出关节误差；故意改 MJCF 限位必须指出关节名。
4. 资产一致性报告（URDF/MJCF/USD 关节名与 `joint_order`）。
5. robot-agent：拉已签名包、本地缓存、上报 jitter/温度/事件；平台展示时间线。
6. 回滚：端侧不依赖网络即可切回上一包。
7. 放行仍保持人按；Agent 永不可调 `deploy.release`。

验收：设计 M4 四条。没有真机时，用回放遥测文件做「平台能显示」的弱验收，真机作为加分项。

---

### 阶段 P4 — Agent 从剧本到可审计助手（约 2–3 周）

LLM 最后做。顺序不可颠倒。

1. ToolBus 补全七道：白名单、预算（周 GPU·h 硬顶）、速率、审批队列、补偿、审计、磁带。
2. 7 个剧本改为真正的多步 tool call，而不是一段中文。优先：**日报**、**失败诊断**、**资源巡检**（只读）。
3. 磁带目录 + 回放测试：禁止动作、关 LLM、`llm_calls==0`。
4. 权限矩阵快照测试（设计 8.4）：未声明的新工具 CI 失败。
5. 控制台：工具轨迹、ref 点击溯源、待审批、预算条。
6. 数字溯源：每个 KPI 带 `ref_id` 指向 Job/Run/Report。
7. 这之后才接 LLM；上下文只给规则/剧本失败后的残差。MCP 用注册表导出，不另做一套权限。

验收：设计 M5 六条。

---

### 阶段 P5 — 工程与体验（可与 P0–P4 穿插）

这些不单独构成里程碑，但会显著降低返工：

1. **拆 crate / 拆路由**：按 10.3 把 `ebd` 和 `routes.py` 切开；每拆一块配单测，禁止行为变化。
2. **提交 `Cargo.lock` + `uv.lock`**，镜像构建可复现。
3. **CI 最小四项**（设计 10.4）：ruff、适配器快照、domain 单测、`alembic upgrade head && downgrade -1`。不做 GPU 端到端。
4. **api 镜像不要编 Rust**：`ebctl` 只放 ebd 镜像，缩短 `make up`。
5. **前端设计系统**：先把 CSS 变量扩成 09 章组件（表格、徽标、KPI、日志），再引入 ECharts / Vue Flow，three.js 继续懒加载。
6. **WebSocket 协议**：`v, type, ts, seq, payload`；队列、日志、Agent 流共用。
7. **鉴权收口**：浏览器走 HttpOnly Cookie；脚本走 Bearer；Showcase 完全无 key；审计写 `principal`。
8. **插件真加载**：`pluginhost` 按 Manifest 注入 adapter/estimator，卸载时停掉 cron。
9. **搜索**：至少按 id/name/tag 打 Postgres `ILIKE` + 契约过滤。
10. **观测**：api/ebd 结构化日志；Job 失败带 `abort_reason` 枚举（OOM / 缺显示设备 / 契约不匹配），供 P4 诊断剧本使用。

---

## 8. 建议的执行顺序（一张图）

```
P0.1 GPU + 诚实失败
  → P0.2 手工 Isaac 容器
  → P0.3 worktree/overlay
  → P0.4 TSDB + Run 页
  → P0.5 产物 + ONNX 实测
  → P0.6 调度/五元组
  → P0.7 P0 页面（可与 0.4 并行）
      → P1 评估基线榜
          → P2 数据+RLinf（重定向可选）
              → P3 策略包+真机
                  → P4 Agent 磁带与 LLM
P5 工程债全程穿插，不单独排期超过半周
```

**明确不做（除非设计变更）：** Kubernetes、Kafka、Celery、ClickHouse、多租户 RBAC、OAuth、把任务镜像打进 `compose up`、让 Agent 拥有 shell/文件系统/SQL/真机放行。

---

## 9. 下一份进度应何时更新

完成 P0.2（第一次真实容器训练）或放弃 stub 成功语义时，应新增 `docs/progress/YYYY-MM-DD-*.md`，并改本目录 README 索引。不要在本文件上无限追加。
