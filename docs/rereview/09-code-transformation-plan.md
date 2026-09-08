# 再审方案代码改造计划

> 对照真源：`docs/rereview/00`–`08`。本文件是**可执行**改造计划。不使用 git。旧评估/部署/Agent 路由保持可访问但不进主导航。

**Goal:** 把 v0.1 骨架收敛成 RLinf 训练工作台：六项导航、路径白名单、磁盘挂载、多 URDF、训练表单与模型血缘。

**Architecture:** 领域纯函数先落地（扫描 / Hydra / 路径 / 磁盘）；Postgres 用 Alembic **加列+新表**，不删旧表；新 API 按模块拆文件；训练提交双写 `train_jobs` + 现有 `jobs`，调度器不重写。

**Tech Stack:** Python 3.11 · FastAPI · SQLAlchemy 2 · Alembic · Vue 3 · 现有 ebd。

## Global Constraints

- 不得写入 `/data/kk/Berkeley-Humanoid-Lite`、`/data/kk/RLinf`、`/data/kk/berkeley_ego_data`
- 控制面不 `subprocess` 跑训练；Doctor 可读文件与 ebd `/v1/doctor`
- 路径必须 `realpath` 后落在允许前缀：默认 `/data/kk`、`/var/lib/eb`
- 代码/数据/基座挂载 `ro`；唯一可写 `/var/lib/eb/runs/<id>/`
- 强制 Hydra：`runner.logger.log_path=/workspace/output`
- Ray 端口：`6390 + (crc32(job_id) % 50)`
- 本版训练 UI 只提交 RLinf 模板；Isaac RSL-RL 不进新建表单
- 不改 `docs/design/`
- 不做 git commit / push

## 文件地图

| 新建 | 职责 |
|---|---|
| `backend/eb/paths.py` | 白名单路径校验 |
| `backend/eb/domain/disk.py` | statvfs 预检 |
| `backend/eb/domain/scan.py` | 本体目录扫描 |
| `backend/eb/domain/rlinf_form.py` | 模板、placement、Ray、overrides |
| `backend/eb/api/routes_data.py` | 数据接入/树/媒体 |
| `backend/eb/api/routes_env.py` | 镜像目录、Doctor |
| `backend/eb/api/routes_train.py` | 训练列表/提交/预览/产物登记 |
| `backend/eb/api/routes_model.py` | 基座/产物/DAG |
| `backend/eb/api/routes_settings.py` | 白名单与默认镜像 |
| `backend/migrations/versions/0003_workbench.py` | 加列+新表 |
| `frontend/src/views/EnvHub.vue` | 环境页 |
| `frontend/src/views/TrainList.vue` | 训练列表 |
| `frontend/src/views/TrainNew.vue` | RLinf 表单 |
| `frontend/src/views/TrainJobDetail.vue` | 训练详情 |
| `frontend/src/views/DatasetNew.vue` | 数据接入 |

| 修改 | 职责 |
|---|---|
| `backend/eb/settings.py` | `allowed_path_prefixes` |
| `backend/eb/store/models.py` | 新列/新表 |
| `backend/eb/domain/urdf.py` | 关节限位解析 |
| `backend/eb/adapters/rlinf.py` | 脚本/placement/强制 overrides/Nucleus 挂载 |
| `backend/eb/api/app.py` | 挂新路由 |
| `backend/eb/api/embodiment_routes.py` | 路径登记、扫描落库、变体 preview |
| `frontend/src/views/Shell.vue` + `router` | 六项导航 |
| 本体/数据/模型/总览 Vue | 对齐再审页面 |

## 任务

### Task 1: 路径白名单与磁盘预检
- 测试：`backend/tests/unit/test_paths.py`、`test_disk.py`
- 实现：`paths.py`、`disk.py`、`settings.allowed_path_prefixes`

### Task 2: URDF 关节树 + 本体扫描
- 测试：扩展 `test_urdf.py`，新增 `test_scan.py`
- 实现：`urdf.parse_urdf_joint_tree`、`scan.scan_embodiment`

### Task 3: RLinf 表单引擎
- 测试：`test_rlinf_form.py`
- 实现：三模板、placement、ray_port、forced overrides、hydra key 粗检

### Task 4: ORM + Alembic 0003
- 加列：embodiments / datasets
- 新表：`train_jobs`、`sim_images`、`system_settings`、`weight_models`（表名 `models` 若冲突则用 `weight_models`，API 仍暴露 `/models` 的 items）
- 保留旧表

### Task 5: API 拆分与双写训练
- 数据/环境/训练/模型/设置路由
- 提交：写 `train_jobs` + 现有 `Job`（模板映射 env/task），调度不改
- 资产：`GET /embodiments/{id}/assets/{path:path}`

### Task 6: RLinf Adapter
- embodiment 用 `train_embodied_agent.py`；SFT 用现脚本
- 不再写死 `actor_rollout={gpus}`
- 挂 Nucleus cache、output/tb

### Task 7: 前端工作台
- 导航：总览/本体/数据/环境/训练/模型 + 设置
- 旧页保留路由、移出导航
- 训练取代实验；表单 + 预览
- 本体多 URDF + 关节滑杆
- 数据接入 + 惰性树
- 模型 items + DAG

### Task 8: 回归
- `cd backend && uv run pytest -q`
- 浏览器核对主导航与训练/本体主路径
