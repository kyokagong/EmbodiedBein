# EmbodiedBein

具身机器人的在线管理平台：数据、模型、训练、评估、部署与 Agent。
单机部署，唯一入口是 Docker Compose。

## 启动

```bash
cp deploy/.env.example deploy/.env   # 改密码；按需改端口
make up                              # 构建并拉起 api / ebd / postgres / redis / minio
```

浏览器打开入口（默认 `http://localhost:8000`）。默认账号见 `deploy/.env.example`。

设计文档：[`docs/design/index.html`](docs/design/index.html)。

## 部署端口

宿主机发布端口全部在 `deploy/.env` 里调，**不必改 compose 文件**。容器内部端口不变：Postgres `5432`、Redis `6380`、MinIO `9000/9001`、api 进程 `8000`。

Redis **容器内必须仍是 6380**（避让 Ray 默认 6379）。`EB_PORT_REDIS` 只改宿主机映射。`ebd` 使用 host network，它连 Redis 时走 `127.0.0.1:${EB_PORT_REDIS}`，因此改 Redis 宿主机端口后必须 `make down && make up`，让 ebd 一起读新值。

| 变量 | 默认 | 发布到 | 说明 |
|---|---|---|---|
| `EB_PORT_API` | `8000` | `0.0.0.0` | Web / API，浏览器入口 |
| `EB_PORT_POSTGRES` | `5432` | `127.0.0.1` | 仅本机 `psql` |
| `EB_PORT_REDIS` | `6380` | `127.0.0.1` | 仅本机；ebd host 网络依赖它 |
| `EB_PORT_MINIO` | `9000` | `127.0.0.1` | S3 API |
| `EB_PORT_MINIO_CONSOLE` | `9001` | `127.0.0.1` | MinIO 控制台 |
| `EB_PORT_HTTP` / `EB_PORT_HTTPS` | `80` / `443` | 全部接口 | 仅 `compose --profile tls` 的 Caddy |
| `EB_PORT_VITE` | `5173` | 本机 | `make dev-web` 热更新 |

改端口后：

```bash
make down
make up
```

`make smoke-job` 和 `make dev-web` 会读同一份 `deploy/.env`。

### 示例：躲开本机已被占用的端口

本机已有 Postgres 5432、某个服务占了 8000，把平台挪到另一组端口：

```bash
# deploy/.env（在 example 拷贝之后追加或覆盖）
EB_PORT_API=8080
EB_PORT_POSTGRES=15432
EB_PORT_REDIS=16380
EB_PORT_MINIO=19000
EB_PORT_MINIO_CONSOLE=19001
EB_PORT_VITE=5174
```

```bash
make down && make up
# 浏览器 http://localhost:8080
curl -fsS http://127.0.0.1:8080/healthz
make smoke-job          # 自动打到 :8080
make dev-web            # Vite :5174，代理 /api → :8080
```

默认（不改 `.env` 端口项）仍是：

```text
http://localhost:8000          Web / API
127.0.0.1:5432                 Postgres
127.0.0.1:6380                 Redis
127.0.0.1:9000 / 9001          MinIO
http://localhost:5173          仅 make dev-web
```

## 开发

```bash
make up          # 后端与依赖在 compose 里
make dev-web     # Vite 热更新，代理 /api 到 EB_PORT_API
cd backend && uv sync && uv run pytest
```

任务镜像（Isaac / RLinf）体积大，用 `make images` 单独构建，不跟 `compose up` 绑在一起。
