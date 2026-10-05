---
name: eb-develop
description: Use when changing EmbodiedBein Python/Rust/Vue code, API schemas, database migrations, runtime images, ToolBus tools, tests or developer documentation; 适用于基本开发规范和交付检查。
---

# 开发规范与交付检查

根目录 [AGENTS.md](../../../AGENTS.md) 约束优先；[设计](../../../docs/design/README.md) 规定目标，[进度](../../../docs/progress/README.md) 记录差距。先查已有实现，改最少必要位置，不复制业务生命周期。

## 改动前

用只读 `git status --short`、局部 diff 区分已有改动与本次工作；不要覆盖、还原或暂存别人的改动。用户禁止提交时不执行 commit/push，也不为了验证生成提交。阅读涉及的设计、调用者、数据流与测试；设计确有错误时先列证据和需要确认的语义，不用改设计掩盖实现缺口。

## 代码放在哪里

| 职责 | 位置与约束 |
|---|---|
| 领域规则/状态/契约 | `backend/eb/domain/`；新增逻辑保持纯计算，不添加 IO、训练框架依赖或 GPU 探测 |
| HTTP/鉴权/参数 | `backend/eb/api/`；路由已按领域拆分，`app.py` 装配；HTTP 层调用共享领域服务 |
| 作业/恢复/准入/收集 | `backend/eb/orchestrator/`、`scheduler/`、`pipelines.py`；维护 Job/Run、幂等和事务边界 |
| 栈差异与命令 | `backend/eb/adapters/`；复用 `RenderedJob`，命令参数不用未经校验的 shell 拼接 |
| Python 计算依赖 | `backend/eb/runners/` 或现有 worker；只在受管任务容器中执行 |
| 执行/大字节路径 | `rust/crates/` 中对应 crate；部署仍只有 ebd/ebctl，不新增常驻微服务 |
| 元数据/迁移 | `backend/eb/store/models.py`、`backend/migrations/versions/` |
| 页面/请求/路由 | `frontend/src/views/`、`api/client.ts`、`router/index.ts`；复用现有组件与错误展示 |

表中同列省略的路径延续所在模块前缀。新代码遵循周边 Python 类型、Pydantic v2、SQLAlchemy 2、Rust/Vue 风格；优先已有依赖/标准库。不要把个别机器人特例变成全局默认。

## 修改时核对

- **API 与 schema**：在信任边界校验 ID/用途/版本/范围/路径；预览与提交共用解析，冻结输入并拒绝过期预览。空数据、无权限、失配、历史对象缺新字段都有明确结果。先查 `/api/openapi.json` 与路由，不猜接口。
- **数据库**：模型变化若影响已有库，新增 Alembic 迁移；不重写已应用迁移或依赖 `create_all` 更新生产表。检查旧数据、索引/外键、回填和回滚可行性；迁移在隔离库验证，部署前备份。不可逆数据变化明确记录，不自动 downgrade。
- **异步与状态**：长任务用 Job；异常要保留原因，重复请求幂等；重试新 Run。完成、取消和恢复的并发不能重复发布产物。结果/血缘校验失败不得标成功。
- **路径与镜像**：外部源只读；白名单、真实路径解析、挂载和写根一起核对。代码快照/镜像 digest/输入 hash 可追溯；改 runner 要更新任务镜像，改 UI 要构建 api 镜像中的 SPA。
- **前端**：显示后端真实状态及错误，字段标签解释用户含义；保留键盘操作、表单标签和加载/空/失败状态。不要用前端筛选代替后端兼容性校验。
- **安全与不可变性**：已发布接口/数据/模型/Suite/Bundle 派生新版本；日志脱敏；不把许可未知或 target 未验证自动升级为可交付。

迁移的仓库特例：`0001_init.py` 从当前 ORM metadata 建表，空库可能已包含新字段/表，后续无条件新增会重复失败；同时验证空库初始化和旧库升级。新增迁移需先构建 Compose 的 `migrate` 镜像，`make migrate` 本身不重建，`build api` 也不更新它；隔离测试不能误连部署库。

## 新增平台 Agent 工具

先读 [Agent 安全设计](../../../docs/design/06-agent-security.md)，再查 `backend/eb/harness/toolbus.py` 和 `backend/eb/agent/tools.py` 的注册方式。复用领域服务，补参数校验、effect/速率/预算/审批/幂等/审计和结果 ref，按实际工具能力验证，不把设计中的护栏当作已全部实现。

永久禁令没有审批豁免：shell、任意文件/网络/SQL、docker.sock、发布或修改已发布接口、`gate.override`、`deploy.release`/真机激活。不要把 `.agents` 开发命令包装为平台 Agent 工具。增加工具需覆盖正常调用、越权/禁止项、重复请求和审计，复用 `backend/tests/agent/` 的磁带/测试。

工具注册变化还需维护 `backend/tests/agent/matrix_snapshot.json`；现有快照主要核对工具名及固定禁令，不能代替审批、越权、预算和幂等行为测试。

## 验证和报告

运行方式见 [工具手册](../../tools.md)。为非平凡逻辑留下能捕获错误的最小行为检查；bug 先复现再修。按改动范围选择已有测试：后端 pytest、Rust cargo、前端 Node tests/类型检查/Vite build。文档改动核对元数据、路径/链接、命令和场景，不为了文档改动运行训练。

例如增加恢复字段：核对请求模型 → Job 冻结快照 → runner 实际消费 → checkpoint 完整性 → collector，而非只给前端加输入框；测试缺字段的旧任务、错误 checkpoint 与重复恢复。

交付前逐轮检查：①设计边界与真实能力；②成功/失败/取消/重试及旧数据；③用户入口、错误可见性、命令和实际验证证据。报告改了什么、执行了哪些检查及结果、哪些运行验证未做；进度写入 `docs/progress/`，不要把计划或静态检查称为端到端完成。
