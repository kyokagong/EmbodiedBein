# EmbodiedBein 给 Agent 的说明

本文是 Agent 的操作手册。设计真源是 `docs/design/`；实现进度见 `docs/progress/`。

## 硬约束（冲突时改代码，不改设计）

- 设计真源：`docs/design/`。实现必须服从设计，冲突时改代码而不是改设计（除非设计有明确错误）。
- 控制面 Python（`backend/eb`），数据面 Rust（`rust/`，单一二进制 `ebd` + `ebctl`）。
- 外部仓只读：`/data/kk/Berkeley-Humanoid-Lite`、`/data/kk/RLinf`、`/data/kk/berkeley_ego_data`。平台不得写入这三个目录。
- 执行一律容器化。`ebd` 是唯一持有 `docker.sock` 的服务。控制面禁止 `subprocess` 跑训练。
- 平台内 Agent 没有后门：工具走 ToolBus，禁止 shell / 文件系统 / 出网 / SQL / 真机放行 / 改已发布本体环境 / 门禁豁免。
- 部署：`make up`。前端打进 api 镜像的 `backend/web/dist`（由 `frontend/` Vite 构建，不进 git）。
- 单机、`docker compose` 是唯一入口。任务容器不进 compose，由 ebd 按 Job 动态创建。
- Redis 端口 **6380**（避让 Ray 默认 6379）。

## 项目是什么

EmbodiedBein 是面向 Berkeley-Humanoid-Lite 的单机在线管理平台：本体、本体环境、任务、数据、模型、训练、评估、部署与平台 Agent。

当前代码是 **v0.1 可部署骨架**（`make up` 可起）：领域内核（本体环境 / 门禁 / 估算器 / ToolBus）是真的；Isaac/RLinf 大镜像、真机、重定向、LLM 仍是占位。细节对照见 `docs/progress/2026-08-21-first-version.md`。

## 基本信息

| 项 | 值 |
|---|---|
| 版本 | 0.1.0 |
| 控制面 | Python 3.11 · FastAPI · SQLAlchemy 2 · Pydantic v2 · uv |
| 数据面 | Rust 2021 · axum 0.8 · clap 4.5.32 · 二进制 `ebd` / `ebctl` |
| 前端 | Vue 3 · TypeScript · Vite · Pinia · Vue Router |
| 存储 | PostgreSQL 16 · Redis 7（6380）· MinIO · 本地 `/var/lib/eb` |
| Compose 项目名 | `eb` |
| Web / API | http://localhost:8000 |
| 默认用户 | `kk` / `embodied`（`deploy/.env.example`） |
| 默认 API Key | `eb-dev-key`（Bearer）；Agent 内部 key `eb-agent-key` |
| Showcase | `/showcase/bhl-progress-demo`（免登录） |
| 本体 / 环境 | `emb_berkeley_humanoid_lite`；`ee_bhl_humanoid_22dof_v1`（75/22/25Hz）、`ee_bhl_biped_12dof_v1`（45/12/25）、`ee_bhl_wbc_22dof_v1`（85/22/50）、`ee_bhl_fixedhand_10dof_v1`（10+img/10/20，concat=false） |
| UDS | `/run/eb/ebd.sock`（控制）· `/run/eb/ebd-http.sock`（媒体） |
| 数据根 | `/var/lib/eb` |
| MinIO | 用户 `ebadmin`（≥3 字符限制），密码见 `.env` |
| 时区 | Asia/Shanghai |

环境变量前缀 `EB_`，见 `backend/eb/settings.py` 与 `deploy/.env.example`。

## 常用命令

```bash
cp deploy/.env.example deploy/.env   # 首次
make up                              # 构建并启动 api / ebd / postgres / redis / minio
make down
make logs
make test                            # backend pytest；rust cargo test
make dev-web                         # Vite :5173，代理 /api → :8000
make images                          # 任务镜像（当前 Dockerfile 仍是 busybox 占位）
cd backend && uv sync --extra dev && uv run pytest -q
```

## 进程与端口

| 服务 | 说明 | 端口 / 套接字 |
|---|---|---|
| api | FastAPI 控制面 + 托管 SPA | 8000 |
| ebd | 数据面；host network；持有 docker.sock | UDS 如上 |
| postgres | 业务库 `eb` | 127.0.0.1:5432 |
| redis | 队列/缓存预留 | 127.0.0.1:6380 |
| minio | S3 兼容（产物尚未全面接入） | 127.0.0.1:9000 / 9001 |
| caddy | 仅 `compose --profile tls` | 80 / 443 |
| 任务容器 | `ebd` 动态 `docker run`，label `eb.run_id` | 不进 compose |

## 当前目录结构

与设计 `docs/design/10-roadmap.html` 10.3 对齐的是**意图**；下面是仓库**现在实际有的**树。Rust 侧设计中的 `eb-runner` / `eb-collector` / `eb-scanner` / `eb-artifact` / `eb-tsdb` / `eb-probe` / `eb-cli` 尚未拆 crate，能力暂时都在 `rust/crates/ebd`。

```
EmbodiedBein/
├── AGENTS.md                 # 本文件（Agent 手册）
├── CLAUDE.md                 # Claude Code 入口，内容与本文件对齐
├── README.md
├── Makefile                  # up / down / logs / migrate / backup / images / test
├── .gitignore
├── .dockerignore
│
├── contracts/                # 四份本体环境 YAML（ee_*）
│   ├── ee_bhl_humanoid_22dof_v1.yaml
│   ├── ee_bhl_biped_12dof_v1.yaml
│   ├── ee_bhl_wbc_22dof_v1.yaml
│   └── ee_bhl_fixedhand_10dof_v1.yaml
│
├── backend/                  # 控制面（包名 eb）
│   ├── pyproject.toml        # uv / hatchling；pytest 在 extra dev
│   ├── alembic.ini
│   ├── migrations/           # Alembic；0001_init 按 ORM metadata 建表
│   ├── web/dist/             # 前端构建产物（gitignore；镜像内 /app/web/dist）
│   ├── eb/
│   │   ├── api/              # FastAPI：app.py 装配；routes.py 目前集中全部路由
│   │   ├── domain/           # 本体环境、JobSpec、状态机、CheckResult（禁止 IO）
│   │   ├── orchestrator/     # Job→Run、幂等、admit_and_start
│   │   ├── scheduler/        # 单机准入循环（无 NVML 时虚拟 4 卡）
│   │   ├── adapters/         # isaaclab_rslrl / rlinf / mujoco / tools / deploy / lerobot
│   │   ├── gates/            # G1–G5 与门禁 DSL
│   │   ├── lineage/          # 血缘边写入（训练成功后尚未自动接）
│   │   ├── harness/          # ToolBus、禁止集、注册表
│   │   ├── pluginhost/       # 插件 Context load/unload
│   │   ├── agent/            # tools + 7 个剧本（规则路径，无 LLM）
│   │   ├── capacity/         # 显存估算器 L2
│   │   ├── store/            # PG 模型、ebd HTTP-over-UDS 客户端
│   │   ├── ingest.py         # 遗留 ONNX / tfevents 纳管
│   │   ├── bootstrap.py      # 种子用户、本体/环境/任务、Showcase、扫描
│   │   └── settings.py       # EB_* 配置
│   └── tests/
│       ├── unit/             # 本体环境、门禁、容量、ingest、插件
│       ├── adapters/         # Isaac 命令行快照
│       └── agent/            # ToolBus / 剧本
│
├── frontend/                 # Vue 3 源码；31 路由；P0 有专用视图，其余 Page.vue JSON
│   ├── package.json
│   ├── vite.config.ts        # outDir 默认 ../backend/web/dist
│   └── src/
│       ├── api/client.ts     # Bearer eb-dev-key
│       ├── router/index.ts
│       └── views/
│
├── rust/                     # workspace；当前成员仅 eb-core + ebd
│   ├── Cargo.toml
│   └── crates/
│       ├── eb-core/          # GpuInfo / HostProfile / sha256
│       └── ebd/              # bin ebd（serve/doctor/scan）+ bin ebctl
│
├── plugins/                  # 设计中的栈插件；现仅 Manifest 占位
│   ├── eb_plugin_isaaclab/
│   ├── eb_plugin_rlinf/
│   └── eb_plugin_bhl/
│
├── robot-agent/              # 端侧薄 agent（独立部署）；现只打印身份
│
├── deploy/
│   ├── docker-compose.yml    # api / ebd / migrate / postgres / redis / minio；（profile tls）caddy
│   ├── api.Dockerfile        # node 构建前端 → rust 编 ebctl → python+uv
│   ├── ebd.Dockerfile        # rust 编 ebd/ebctl；拷贝 docker CLI
│   ├── Caddyfile
│   ├── .env.example          # 密码模板；实际 .env 不进 git
│   └── images/               # isaac / rlinf / mujoco / tools（当前 busybox 占位）
│
└── docs/
    ├── design/               # 设计真源（HTML）
    └── progress/             # 实现进度，不是设计
```

外部只读挂载（compose 把 `/data/kk` bind 进 api 与 ebd）：

```
/data/kk/Berkeley-Humanoid-Lite/   # 资产、checkpoints、logs
/data/kk/RLinf/
/data/kk/berkeley_ego_data/
```

## 给后续改动的提示

- 新增训练栈：只加 `backend/eb/adapters/` 一个文件 + 快照测试，不要改 domain。
- 热路径（日志、tfevents、视频、大文件）应进 `ebd`，不要在 FastAPI 里解析。
- 真机放行、改已发布本体环境、gate.override 必须保持结构禁止。
- 任务镜像用 `make images`，不要塞进 `make up`。
- 进度文档只记录对照设计的差距，不要反过来改 `docs/design/`。
