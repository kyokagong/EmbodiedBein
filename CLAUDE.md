# EmbodiedBein — Claude Code 说明

与 `AGENTS.md` 对齐。设计真源是 `docs/design/`；实现进度见 `docs/progress/`。改行为前先读这两处和本文件。

## 硬约束（冲突时改代码，不改设计）

- 设计真源：`docs/design/`。实现必须服从设计，冲突时改代码而不是改设计（除非设计有明确错误）。
- 控制面 Python（`backend/eb`），数据面 Rust（`rust/`，单一二进制 `ebd` + `ebctl`）。
- 外部仓只读：`/data/kk/Berkeley-Humanoid-Lite`、`/data/kk/RLinf`、`/data/kk/berkeley_ego_data`。平台不得写入这三个目录。
- 执行一律容器化。`ebd` 是唯一持有 `docker.sock` 的服务。控制面禁止 `subprocess` 跑训练。
- 平台内 Agent 没有后门：工具走 ToolBus，禁止 shell / 文件系统 / 出网 / SQL / 真机放行 / 改契约 / 门禁豁免。
- 部署：`make up`。前端打进 api 镜像的 `backend/web/dist`（由 `frontend/` Vite 构建，不进 git）。
- 单机、`docker compose` 是唯一入口。任务容器不进 compose，由 ebd 按 Job 动态创建。
- Redis 端口 **6380**（避让 Ray 默认 6379）。

## 项目基本信息

EmbodiedBein 是面向 Berkeley-Humanoid-Lite 的单机在线管理平台：契约、数据、模型、训练、评估、部署与平台 Agent。

当前是 **v0.1 可部署骨架**：`make up` 后浏览器可打开；契约/门禁/估算器/ToolBus 是真实现；Isaac/RLinf 大镜像、真机、重定向、LLM 为占位。对照表：`docs/progress/2026-08-21-first-version.md`。

| 项 | 值 |
|---|---|
| 版本 | 0.1.0 |
| 控制面 | Python 3.11 · FastAPI · SQLAlchemy 2 · Pydantic v2 · uv |
| 数据面 | Rust 2021 · axum 0.8 · 二进制 `ebd` / `ebctl` |
| 前端 | Vue 3 · TypeScript · Vite · Pinia · Vue Router |
| 存储 | PostgreSQL 16 · Redis 7（6380）· MinIO · `/var/lib/eb` |
| Compose 项目名 | `eb` |
| 入口 | http://localhost:8000 |
| 默认用户 | `kk` / `embodied` |
| 默认 API Key | Bearer `eb-dev-key`；Agent `eb-agent-key` |
| Showcase | `/showcase/bhl-progress-demo` |
| 契约 | `ec_bhl_humanoid_22dof_v1`（75/22/25Hz）、`ec_bhl_biped_12dof_v1`（45/12/25）、`ec_bhl_wbc_22dof_v1`（85/22/50）、`ec_bhl_fixedhand_10dof_v1`（10+img/10/20） |
| UDS | `/run/eb/ebd.sock`、`/run/eb/ebd-http.sock` |
| MinIO 用户 | `ebadmin` |
| 配置前缀 | `EB_`（`backend/eb/settings.py`、`deploy/.env.example`） |

```bash
cp deploy/.env.example deploy/.env
make up          # api / ebd / postgres / redis / minio
make test
make dev-web     # Vite :5173 → 代理 /api
make images      # 任务镜像；当前 Dockerfile 仍是 busybox 占位
```

| 服务 | 作用 | 暴露 |
|---|---|---|
| api | FastAPI + SPA | :8000 |
| ebd | 数据面；host net；docker.sock | 上述 UDS |
| postgres | 库 `eb` | 127.0.0.1:5432 |
| redis | 预留 | 127.0.0.1:6380 |
| minio | S3 | 127.0.0.1:9000 / 9001 |
| 任务容器 | ebd 动态创建 | 不进 compose |

## 当前目录结构

下面是仓库**实际**树，不是设计里尚未拆分的理想树。Rust 设计中的 runner/collector/scanner/artifact/tsdb/probe/cli 目前都挤在 `rust/crates/ebd`。

```
EmbodiedBein/
├── AGENTS.md
├── CLAUDE.md                 # 本文件
├── README.md
├── Makefile
├── contracts/                # 四个契约 YAML
│   ├── ec_bhl_humanoid_22dof_v1.yaml
│   ├── ec_bhl_biped_12dof_v1.yaml
│   ├── ec_bhl_wbc_22dof_v1.yaml
│   └── ec_bhl_fixedhand_10dof_v1.yaml
├── backend/                  # 控制面包 eb
│   ├── pyproject.toml
│   ├── alembic.ini
│   ├── migrations/
│   ├── web/dist/             # gitignore；镜像内静态资源
│   ├── eb/
│   │   ├── api/              # FastAPI（路由目前集中在 routes.py）
│   │   ├── domain/           # 纯逻辑：契约、JobSpec、FSM
│   │   ├── orchestrator/
│   │   ├── scheduler/
│   │   ├── adapters/         # isaaclab_rslrl / rlinf / mujoco / …
│   │   ├── gates/            # G1–G5 + DSL
│   │   ├── lineage/
│   │   ├── harness/          # ToolBus
│   │   ├── pluginhost/
│   │   ├── agent/            # tools + playbooks
│   │   ├── capacity/
│   │   ├── store/
│   │   ├── ingest.py
│   │   ├── bootstrap.py
│   │   └── settings.py
│   └── tests/{unit,adapters,agent}/
├── frontend/                 # Vue 3；构建写入 backend/web/dist
│   └── src/{api,router,views}/
├── rust/
│   └── crates/
│       ├── eb-core/
│       └── ebd/              # bin: ebd, ebctl
├── plugins/{eb_plugin_isaaclab,eb_plugin_rlinf,eb_plugin_bhl}/
├── robot-agent/
├── deploy/
│   ├── docker-compose.yml
│   ├── api.Dockerfile
│   ├── ebd.Dockerfile
│   ├── Caddyfile
│   ├── .env.example
│   └── images/               # isaac / rlinf / mujoco / tools 占位
└── docs/
    ├── design/               # 设计真源
    └── progress/             # 实现进度
```

外部只读：`/data/kk/Berkeley-Humanoid-Lite`、`/data/kk/RLinf`、`/data/kk/berkeley_ego_data`（compose bind `/data/kk`）。

## 改动时记住

- 新训练栈：只加 adapter 文件 + 快照测试。
- tfevents / 视频 / 大文件走 ebd，不要在 Python 里做热路径。
- 禁止给平台 Agent 开 shell、改契约、真机放行。
- 不要把任务大镜像绑进 `make up`。
- 不要为了迁就实现去改 `docs/design/`。
