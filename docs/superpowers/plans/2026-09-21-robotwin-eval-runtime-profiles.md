# RoboTwin 评估运行档位与高级容器资源执行计划

**设计真源：** `docs/design/09-robotwin-evaluation.md` §7.2.1 及 `docs/rlinf/02-robotwin-single-policy-eval.md` §8.0。

**目标：** 页面展示节省显存、均衡异步、吞吐优先三个档位；提交只能选择服务端白名单中的精确资源变体。容器 memory/shm、拓扑、调度模式和 probe 身份一起冻结进 EvalPlan。未实现或未 probe 的档位可见但不可提交。

**全局约束：** 不修改外部 `/data/kk/RLinf`；通过固定镜像补丁接入。保留 Ray 内存保护。无 Git 提交。旧计划可读，旧 `gpus + parallel_envs` 请求只做同步白名单兼容。

## 文件职责

- `backend/eb/domain/robotwin_eval_runtime.py`：唯一运行档位目录、资源变体解析和兼容映射。
- `backend/eb/api/routes_eval.py`：目录 API、请求校验、预览/提交冻结。
- `frontend/src/views/EvalNew.vue`：档位卡片、未就绪 blocker、高级资源变体选择。
- `frontend/src/views/EvalJobDetail.vue`：展示冻结档位、容器 RAM/shm 和调度模式。
- `backend/tests/unit/test_robotwin_eval_runtime.py`、`test_eval_plan.py`：域与 API 契约。
- `frontend/tests/robotwin-workbench.test.mjs`：页面契约。

## Task 1：运行档位目录（TDD）

1. 先写失败测试：目录固定返回三个档位；同步档位含 1/2/4 卡资源变体；80 GiB 变体身份稳定；异步档位有 blocker；未知或未就绪组合拒绝。
2. 实现 `list_robotwin_runtime_profiles()` 与 `resolve_robotwin_runtime(profile_id, variant_id)`；扩展 runtime 字段 `profile_id`、`resource_variant_id`、`scheduling_mode`、`probe_receipt_sha256`。
3. 保留 `derive_robotwin_runtime(gpus, parallel_envs)` 作为旧同步映射。
4. 运行域测试，预期全部通过。

## Task 2：API 与冻结计划（TDD）

1. 先写失败测试：`EvalIn` 接受两个新 ID；目录 API 返回 blocker；预览来源 runtime 同时包含 resources/probe；客户端不能提交 memory_gb/shm_gb。
2. 新增 `GET /api/v1/evals/runtime-profiles`。
3. `_prepare_robotwin` 优先解析新 ID；仅在 ID 缺失时走旧字段兼容。JobSpec resources 必须来自解析结果。
4. 预览返回解析后的档位；计划 hash 随资源变体变化。
5. 运行 API、计划和 adapter 测试。

## Task 3：页面（TDD）

1. 先扩展前端契约测试，要求三个档位、blocker、高级配置、容器内存和 shm 字段存在。
2. 页面加载 runtime profile API；默认选择首个 ready 档位及推荐资源变体。
3. 三个档位以 radio/card 展示；未就绪项禁用并显示原因。
4. 高级配置只提供当前档位 `ready` 资源变体下拉框，展示 GPU、Policy、Env、并发、micro-batch、RAM、shm；请求只发送两个 ID。
5. 详情页显示冻结值。运行 Node 测试和 Vite build。

## Task 4：部署与验收

1. 运行完整后端测试。
2. 重建 API；调用目录 API确认三个档位与 blockers。
3. 对同步 4 卡/16 env/80 GiB 生成预览，确认 plan、JobSpec resources、镜像一致。
4. 尝试提交未就绪异步档位，确认服务端拒绝。
5. 不自动启动昂贵的 5000-case 任务；用户从页面重新预览后提交。

## 完成条件

- 三个产品档位都可见；未就绪档位无法提交。
- 高级配置没有任意数值输入；memory/shm 来自服务端白名单。
- 预览哈希覆盖档位、资源变体、memory/shm 与 probe 摘要。
- 旧同步 API 仍能解析；新页面不再发送 `gpus/parallel_envs`。
- 后端全套测试、前端测试和构建通过。
