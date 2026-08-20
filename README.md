# EmbodiedBein

具身机器人的在线管理平台：数据、模型、训练、评估、部署与 Agent。
单机部署，唯一入口是 Docker Compose。

## 启动

```bash
cp deploy/.env.example deploy/.env   # 改密码
make up                              # 构建并拉起 api / ebd / postgres / redis / minio
```

浏览器打开 `http://localhost:8000`。默认账号见 `deploy/.env.example`。

设计文档：[`docs/design/index.html`](docs/design/index.html)。

## 开发

```bash
make up          # 后端与依赖在 compose 里
make dev-web     # Vite 热更新，代理 /api 到 :8000
cd backend && uv sync && uv run pytest
```

任务镜像（Isaac / RLinf）体积大，用 `make images` 单独构建，不跟 `compose up` 绑在一起。
