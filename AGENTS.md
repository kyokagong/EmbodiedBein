# EmbodiedBein 给 Agent 的说明

本文是 Agent 的操作手册。设计真源是 `docs/design/`；实现进度见 `docs/progress/`。

## 工具与技能入口

开发操作手册见 [`.agents/README.md`](.agents/README.md)，按任务读取，不必一次加载全部参考。

| 任务 | 入口 |
|---|---|
| 接入本体、数据、模型、算法/训练栈 | [eb-integrate](.agents/skills/eb-integrate/SKILL.md) |
| API、排队、运行、产物、评估、页面排障 | [eb-debug](.agents/skills/eb-debug/SKILL.md) |
| 开发规范、迁移、测试、ToolBus 扩展 | [eb-develop](.agents/skills/eb-develop/SKILL.md) |
| 现有工具、容器化检查与部署命令 | [工具手册](.agents/tools.md) |

这里的开发 Agent 与平台内 Agent 不同：维护仓库可在授权范围内使用开发工具；平台内 Agent 始终受 ToolBus 结构性禁令约束，不能使用这些 shell/Docker 命令。

## 硬约束（冲突时改代码，不改设计）

- 设计真源：`docs/design/`。实现必须服从设计，冲突时改代码而不是改设计（除非设计有明确错误）。
- 控制面 Python（`backend/eb`），数据面 Rust（`rust/`，单一二进制 `ebd` + `ebctl`）。
- 外部仓只读：`/data/kk/Berkeley-Humanoid-Lite`、`/data/kk/RLinf`、`/data/kk/berkeley_ego_data`。平台不得写入这三个目录。
- 执行一律容器化。`ebd` 是唯一持有 `docker.sock` 的服务。控制面禁止 `subprocess` 跑训练。
- 平台内 Agent 没有后门：工具走 ToolBus，禁止 shell / 文件系统 / 出网 / SQL / 真机放行 / 改已发布本体环境 / 门禁豁免。
- 部署：`make up`。前端打进 api 镜像的 `backend/web/dist`（由 `frontend/` Vite 构建，不进 git）。
- 单机、`docker compose` 是唯一入口。任务容器不进 compose，由 ebd 按 Job 动态创建。
- Redis 端口 **6380**（避让 Ray 默认 6379）。
- 已发布接口、数据/模型版本、评估套件与交付包不可原地修改，变更派生新版本；Job 执行成功不代表模型评估合格。

## 项目是什么

EmbodiedBein 是单机具身学习工作台，连接本体、本体接口、任务、数据、模型、训练、评估、交付与平台 Agent；BHL 是已有接入之一。

版本仍为 0.1.0；代码已包含 LIBERO、RoboTwin、RPO WBC/AMP、真实任务镜像和分模块 API。可用范围必须逐项核对 [当前进度](docs/progress/README.md) 与对应代码/运行证据；2026-08 的骨架说明只代表历史基线。设计、代码存在、smoke 通过和正式验收是不同状态，不能推导真机或模型质量已验证。

## 基本信息

| 项 | 值 |
|---|---|
| 版本 | 0.1.0 |
| 控制面 | Python 3.11 · FastAPI · SQLAlchemy 2 · Pydantic v2 · uv |
| 数据面 | Rust 2021 · axum 0.8 · clap 4.5.32 · 二进制 `ebd` / `ebctl` |
| 前端 | Vue 3 · TypeScript · Vite · Pinia · Vue Router |
| 存储 | PostgreSQL 16 · Redis 7（6380）· MinIO · 母机可配置目录 |
| Compose 项目名 | `eb` |
| Web / API | 默认 http://localhost:8000；实际宿主端口取 `deploy/.env` / Compose `port api 8000` |
| 默认用户 | `kk` / `embodied`（`deploy/.env.example`） |
| 默认 API Key | `eb-dev-key`（Bearer）；Agent 内部 key `eb-agent-key` |
| Showcase | `/showcase/bhl-progress-demo`（免登录） |
| BHL 已发布接口 | `emb_berkeley_humanoid_lite`；`ee_bhl_humanoid_22dof_v1`（75/22/25Hz）、`ee_bhl_biped_12dof_v1`（45/12/25）、`ee_bhl_wbc_22dof_v1`（85/22/50）、`ee_bhl_fixedhand_10dof_v1`（10+img/10/20，concat=false）；其余见 `contracts/` |
| UDS | `/run/eb/ebd.sock`（控制）· `/run/eb/ebd-http.sock`（媒体） |
| 状态 / 模型 / 数据根 | `${EB_HOST_STATE_DIR:-/data/kk/embodiedbein}` · `${EB_HOST_MODELS_DIR:-/data/kk/models}` · `${EB_HOST_DATA_DIR:-/data/kk/data}`；容器内保持相同绝对路径 |
| 受管资产根 | `${EB_HOST_ASSETS_DIR:-/data/kk/assets}`；对应控制面 `EB_ASSETS_ROOT` |
| MinIO | 用户 `ebadmin`（≥3 字符限制），密码见 `.env` |
| 时区 | Asia/Shanghai |

环境变量前缀 `EB_`，见 `backend/eb/settings.py` 与 `deploy/.env.example`。

## 常用命令

```bash
make env                             # 首次配置与目录初始化，不覆盖已有 .env
make up                              # 构建并启动 api / ebd / postgres / redis / minio
make ps
make down
make logs
make test                            # 容器内 backend pytest 与 Rust cargo test；基础镜像需已构建
make images                          # 单独构建任务镜像；先核对 Makefile 中的基础镜像和 build context
```

聚焦测试、前端 Node 测试/类型检查/Vite 构建、迁移和只读排障命令见 [工具手册](.agents/tools.md)。当前 `make lint` / `make dev-web` 仍调用宿主工具，容器化操作使用手册中的替代命令。`make down` 不会自动停止 ebd 创建的任务容器；作业取消走平台 API。

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

目录职责以 `docs/design/02-system-architecture.md` 为准；下面是仓库结构示意。Rust workspace 已拆出 `eb-runner` / `eb-collector` / `eb-scanner` / `eb-artifact` / `eb-tsdb` / `eb-probe` / `eb-cli` 等内部 crate，部署仍只有 `ebd` / `ebctl` 二进制。

```
EmbodiedBein/
├── AGENTS.md                 # 本文件（Agent 手册）
├── CLAUDE.md                 # Claude Code 入口，内容与本文件对齐
├── README.md
├── Makefile                  # up / down / logs / migrate / backup / images / test
├── .gitignore
├── .dockerignore
│
├── .agents/                  # 开发工具手册、按需加载的技能与接入参考
├── contracts/                # 本体接口 YAML（ee_*）；下面仅列 BHL，另有 LIBERO/RoboTwin/RPO
│   ├── ee_bhl_humanoid_22dof_v1.yaml
│   ├── ee_bhl_biped_12dof_v1.yaml
│   ├── ee_bhl_wbc_22dof_v1.yaml
│   └── ee_bhl_fixedhand_10dof_v1.yaml
│
├── backend/                  # 控制面（包名 eb）
│   ├── pyproject.toml        # uv / hatchling；pytest 在 extra dev
│   ├── alembic.ini
│   ├── migrations/           # Alembic；已有多份迁移，schema 变化需考虑现有库升级
│   ├── web/dist/             # 前端构建产物（gitignore；镜像内 /app/web/dist）
│   ├── eb/
│   │   ├── api/              # app.py 装配；routes.py、embodiment_routes.py 和 routes_*.py 分领域
│   │   ├── domain/           # 本体环境、JobSpec、状态机、CheckResult（禁止 IO）
│   │   ├── orchestrator/     # Job→Run、幂等、admit_and_start
│   │   ├── scheduler/        # 单机准入；无 GPU 证据时跳过 GPU 准入，fake_gpus 仅用于测试
│   │   ├── adapters/         # isaaclab_rslrl / rlinf / mujoco / tools / deploy / lerobot
│   │   ├── gates/            # G1–G5 与门禁 DSL
│   │   ├── lineage/          # 血缘；各类产物登记还需核对 orchestrator/collect.py 分支
│   │   ├── runners/          # 容器内运行入口、模型桥和配置；不在 API 进程执行
│   │   ├── environment_catalog.py # 训练/评估环境 Profile，与本体接口分开
│   │   ├── checkpoints.py    # checkpoint 完整性、推理配置与恢复输入
│   │   ├── model_sources.py  # 模型下载来源与必需文件
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
│       ├── agent/            # ToolBus / 剧本
│       ├── runtime/          # 目标镜像/GPU 专项脚本，不等于 pytest 全量已覆盖
│       └── integration/      # PG 等专项集成检查
│
├── frontend/                 # Vue 3 工作台源码；路由以 router/index.ts 为准
│   ├── package.json
│   ├── vite.config.ts        # outDir 默认 ../backend/web/dist
│   └── src/
│       ├── api/client.ts     # Bearer eb-dev-key
│       ├── router/index.ts
│       └── views/
│
├── rust/                     # workspace；内部能力 crate + ebd/ebctl 二进制
│   ├── Cargo.toml
│   └── crates/
│       ├── eb-core/          # 公共类型与 hash
│       ├── eb-{probe,runner,tsdb,collector,scanner,artifact,cli}/
│       └── ebd/              # 数据面服务
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
│   ├── api.Dockerfile        # node 构建前端 → python+uv 控制面镜像
│   ├── ebd.Dockerfile        # rust 编 ebd/ebctl；拷贝 docker CLI
│   ├── Caddyfile
│   ├── .env.example          # 密码模板；实际 .env 不进 git
│   ├── images/               # Isaac / RLinf 多模型 / MuJoCo / tools 任务镜像
│   ├── patches/              # 外部训练栈的受管补丁，不修改外部仓
│   └── test.Dockerfile       # backend 测试镜像，基于 eb-api:latest
│
└── docs/
    ├── design/               # 设计真源（Markdown）
    └── progress/             # 实现进度，不是设计
```

外部只读挂载（compose 把 `/data/kk` bind 进 api 与 ebd）：

```
/data/kk/Berkeley-Humanoid-Lite/   # 资产、checkpoints、logs
/data/kk/RLinf/
/data/kk/berkeley_ego_data/
```

## 给后续改动的提示

- 新增训练栈优先复用 adapter + 渲染快照测试；按实际需要同步模板注册、runner/config、镜像、checkpoint/收集与评估。不要为栈差异复制领域生命周期；新增协议字段按设计修改现有 schema。
- 热路径（日志、tfevents、视频、大文件）应进 `ebd`，不要在 FastAPI 里解析。
- 真机放行、改已发布本体环境、gate.override 必须保持结构禁止。
- 任务镜像用 `make images`，不要塞进 `make up`。
- 进度文档只记录对照设计的差距，不要反过来改 `docs/design/`。
- 修改前核对已有工作区差异，不覆盖他人工作；是否提交服从当前用户要求，不把技能的提交建议当作授权。
- 非平凡逻辑保留最小行为检查；验证按改动范围运行，报告实际结果与未验证项。
