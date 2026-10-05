# 工具手册

适用于开发者/维护仓库的 Agent。平台内 Agent 的工具入口是 `backend/eb/harness/` 与 `backend/eb/agent/tools.py`，不能把本页命令注册成通用 shell 工具。

## 先选已有入口

| 工具 | 用途与边界 |
|---|---|
| `rg` / `rg --files` | 查设计、接口、调用者、测试；先定位再改 |
| `Makefile` | 平台部署、状态、日志、测试、任务镜像；执行前读对应 target |
| `/api/openapi.json` | 当前服务的请求 schema；API 路由真源是 `backend/eb/api/` |
| `ebctl health / doctor / ps` | 仅这三个子命令；在 ebd 服务容器内使用 |
| `/api/v1/jobs`、`/runs` | 逻辑任务、每次尝试、资源、日志、指标和产物 |
| `/api/v1/train/preview`、`/evals/previews` | 验证冻结输入及执行计划；不是启动命令 |
| `/api/v1/environments` | 训练/评估环境版本；不同于 `/envs` 的镜像与 Doctor |
| `backend/eb/paths.py` | 路径白名单和 resolve 后的包含关系；可读不等于可写 |
| `backend/eb/checkpoints.py` | checkpoint 发现、完整性与推理配置；不要自行 glob 最大 step |
| `backend/eb/adapters/` | 生成 `RenderedJob`；展示的 equivalent-command 只供诊断 |

## 服务与只读诊断

从仓库根目录执行。`deploy/.env` 决定实际端口；8000 只是默认值。不要打印整个 `.env`、完整 Compose 展开配置或容器环境，里面可能有密钥。

```bash
make ps
docker compose -f deploy/docker-compose.yml --env-file deploy/.env port api 8000
docker compose -f deploy/docker-compose.yml --env-file deploy/.env logs --tail=100 api ebd
docker compose -f deploy/docker-compose.yml --env-file deploy/.env exec -T ebd ebctl health
docker compose -f deploy/docker-compose.yml --env-file deploy/.env exec -T ebd ebctl doctor
docker compose -f deploy/docker-compose.yml --env-file deploy/.env exec -T ebd ebctl ps
```

根据 `port` 输出设置地址；API key 使用操作者已有凭据，示例值仅适用于未改凭据的本地开发种子。不要将真实 token 写入文档、日志或 shell 跟踪。

```bash
EB_BASE_URL=http://127.0.0.1:8000   # 替换为查询到的实际端口
EB_API_TOKEN=eb-dev-key          # 仅开发默认值
curl --connect-timeout 5 --max-time 30 --fail-with-body -sS "$EB_BASE_URL/healthz"
curl --connect-timeout 5 --max-time 30 --fail-with-body -sS "$EB_BASE_URL/api/v1/envs/doctor" -H "Authorization: Bearer $EB_API_TOKEN"
curl --connect-timeout 5 --max-time 30 --fail-with-body -sS "$EB_BASE_URL/api/openapi.json"
```

复用上述地址/凭据查询 `GET /api/v1/jobs/{job_id}`、`/runs/{run_id}`、`/runs/{run_id}/logs?tail=100`、`/runs/{run_id}/metrics`。`trn_*` 是 TrainJob，不是 `job_*` 或 `run_*`；先从 `/train/{tid}` 获取关联 ID。日志接口返回 200 也要检查 `error`，空曲线还要核对回退数据和 provenance。

健康判断要读响应：`/healthz` 即使顶层 `ok=true`、HTTP 200，嵌套 `ebd.ok/error` 仍可能失败；Doctor 还需核对 `checks[].detail.error`。`ebctl ps` 在 UDS 失败时会回退 Docker，且未可靠传播回退失败退出码，空输出或退出 0 不能证明“没有任务”。沙箱访问受限时，记录检查受限，不能据此判定服务/GPU 故障。

## 构建与验证（容器化）

以下会构建镜像、下载依赖或使用缓存；按当前任务选择，不为文档改动启动训练。宿主只运行 Docker/Compose、文本检查等维护命令，Python、Rust、Node 测试在容器内执行。

```bash
# make test 的测试镜像以 eb-api:latest 为基础；需要时先构建 api（不重启服务）
docker compose -f deploy/docker-compose.yml --env-file deploy/.env build api
make test

# 复用 make test 构建的镜像，聚焦本次改动；按需替换测试文件
docker run --rm -v "$PWD:/workspace:ro" -v /data/kk:/data/kk:ro eb/test:local \
  python -m pytest -q -p no:cacheprovider tests/unit/test_domain.py
docker run --rm -v "$PWD:/workspace:ro" -v /data/kk:/data/kk:ro eb/test:local \
  python -m ruff check --no-cache eb tests

# 前端现有 Node 测试、类型检查；依赖装在临时容器内，不写宿主 node_modules
docker run --rm -v "$PWD/frontend:/src:ro" -w /tmp/web node:20-alpine sh -ec \
  'cp -R /src/. .; npm ci; node --test tests/*.test.mjs; npx vue-tsc --noEmit'
# 与实际部署共用的 Vite 构建阶段（不等于类型检查）
docker build --target web -f deploy/api.Dockerfile -t eb/web-check:local .
```

`make test` 覆盖 backend pytest 和 Rust workspace；`backend/tests/runtime/` 与 `integration/` 下的独立脚本未必被 pytest 自动收集，GPU/PG 场景要阅读脚本前置条件并单独安排。当前 `make lint`、`make dev-web` 仍调用宿主工具，不作为容器化验证入口；上面的容器命令用于 lint/前端检查。

## 有状态操作

| 命令 | 影响与使用条件 |
|---|---|
| `make env` | 首次配置（不覆盖已有 `.env`）及受管目录创建；不要反复 `cp` 覆盖配置 |
| `make up` | 构建并启动/更新平台服务；前端随 api 镜像发布 |
| `make images` / `make images-rlinf` | 单独构建任务镜像；依赖本地基础镜像及外部只读 build context，先读 target |
| `make migrate` | 修改数据库 schema；不会重建 migrate 镜像。新增迁移后先 `docker compose -f deploy/docker-compose.yml --env-file deploy/.env build migrate`，再核对目标库、备份及兼容性 |
| `make backup` | PG dump 写入 `backup/`；不包含受管数据、模型、MinIO 文件及密钥 |
| `make down` | 停止平台服务；ebd 创建的动态任务容器不随 Compose 自动消失 |

训练、下载、采集、评估通过 API → Job → ebd 启动；不要复制 equivalent-command 绕过准入执行。取消用 `POST /api/v1/jobs/{id}/cancel`，先确认目标，保留失败/取消证据。不要用 SQL 改成功状态或 `docker system prune` 处理排队问题。

迁移测试先明确隔离数据库连接，分别验证空库初始化和已有库升级；上表 `make migrate` 使用部署配置，不能当成默认的隔离测试命令。`build api` 不会同步刷新独立的 migrate 镜像。

## 源码检索

```bash
rg --files docs/design backend/eb backend/tests rust/crates frontend/src
rg -n 'template_id|claim_outputs|admit_and_start' backend/eb
rg -n '函数名或字段名' backend/eb backend/tests frontend/src rust/crates
git diff --check
```

最后一条是只读差异检查，不提交。具体排障顺序见 [eb-debug](skills/eb-debug/SKILL.md)。
