---
name: eb-debug
description: Use when diagnosing EmbodiedBein API errors, queued or failed Jobs, missing outputs, data/model validation, evaluation evidence, GPU/runtime problems or frontend failures; 适用于 debug、故障定位和复现。
---

# 排障与根因定位

保留失败现场，先定位生命周期的断点，再修改共享根因。开发者可使用 [工具手册](../../tools.md)；平台内 Agent 只能查询 ToolBus 已开放的结构化结果。依据：[架构](../../../docs/design/02-system-architecture.md)、[模型训练](../../../docs/design/04-model-training.md)、[评估](../../../docs/design/05-evaluation-delivery.md)。

## 最小证据

记录发生时间/时区、实际 API 地址、请求或页面动作、预期/实际结果、Job/Run/TrainJob/PipelineRun ID、第一次错误和少量相关日志；再记录输入 hash、镜像 digest、资源和最近改动。凭据、完整环境变量及大文件不进入报告。

先用 `make ps` 和 Compose `port api 8000` 确认服务/端口，再查 `/healthz`、`/api/v1/envs/doctor`。沙箱不能访问 Docker/GPU 只说明检查受限；错误端口或宿主无 `nvidia-smi` 不能证明主机没 GPU。说明未完成的检查，不设置 `EB_FAKE_GPUS` 冒充设备。

`/healthz` 的 HTTP 200/顶层 `ok` 不足以判断健康，检查 `ebd.ok/error`；Doctor 核对 `checks[].detail.error`。`ebctl ps` 可能在 UDS 失败后回退 Docker，退出码 0/空输出也不能证明无任务。HTTP 查询设置连接和总超时；无法确认实际端口或执行环境网络隔离时，结论应是“服务可达性未验证”。

## 沿生命周期缩小范围

`页面 → API/领域校验 → Job → scheduler → Run → ebd/任务容器 → collector → Artifact/模型/评估 → 页面`

下表 Python 路径相对 `backend/eb/`；其他路径从仓库根目录算。

| 症状 | 首查证据 | 代码落点 |
|---|---|---|
| 401/403/422、提交被拒 | schema、principal、Task/接口/用途、preview 是否过期、结构化错误 | `api/deps.py`、对应 `api/routes_*.py`、`domain/`、`gates/` |
| 健康检查失败、API 503 | api/ebd 日志、控制/媒体 UDS、挂载/权限、PG迁移 | `deploy/docker-compose.yml`、`store/ebd.py`、`rust/crates/ebd/src/main.rs` |
| queued、没有 Run | doctor 实际 GPU/占用/显存、估算值、主存保留预算、优先级、准入异常 | `scheduler/loop.py`、`capacity/estimator.py`、`orchestrator/service.py` |
| Run 起不来/立刻失败 | abort_reason、镜像/digest、实际命令、源快照和输出挂载、设备/网络、依赖 | `adapters/`、`orchestrator/service.py`、`rust/crates/eb-runner/` |
| OOM、卡住、GPU 利用率低 | 区分 GPU OOM、主存 OOM、shm、图形缓冲、I/O/loader；保留第一条错误 | `runners/`、`scheduler/loop.py` |
| 日志/曲线/视频缺失 | logs 响应的 error、console.log、metrics tag/step、TSDB、媒体 UDS/Range/路径 | `rust/crates/eb-collector/`、`rust/crates/eb-tsdb/`、`api/routes.py` |
| 下载/验证失败、选不到数据 | 来源/revision、路径可见性、validate Job/receipt、用途/接口、training_ready | `api/routes_data.py`、`data_worker.py`、`data_validation.py`、`orchestrator/collect.py` |
| 成功无模型/恢复失败 | artifacts、完整 checkpoint/完成标记、norm/config、训练族登记分支 | `checkpoints.py`、`orchestrator/collect.py`、`api/routes_train.py` |
| 评估缺结果/失败 | EvalPlan hash、候选模型、逐 case 对账、receipt、qualification；区分任务与基础设施失败 | `api/routes_eval.py`、`orchestrator/collect.py`、`eval/` |
| 管线停住/恢复重复 | PipelineRun 当前 step、子 Job、幂等键、失败策略及冻结输入 | `pipelines.py`、`scheduler/loop.py` |
| 前端空白/旧内容/报错 | Network/Console、HTTP 错误、路由参数/关联 ID、构建产物是否进入 api 镜像 | `frontend/src/api/client.ts`、`frontend/src/router/index.ts`、对应视图、`deploy/api.Dockerfile` |

## 验证根因

1. 对失败函数先 `rg` 所有调用者，核对 API、后台调度、重试和平台工具的共享路径，不只修页面分支。
2. 用相同冻结输入做最小复现，先补能复现错误的检查。资源问题一次只改变一个已测量因素；估算值不当成实测。
3. 比对 preview、Job 快照、实际镜像里的 runner/config 和 receipt。只更新控制面而未更新任务镜像是常见版本错位。
4. 按真实收集分支判断完成：普通 `.pt` 不一定自动成为 ModelVersion；LIBERO 与 RoboTwin/RPO 的登记路径不同。模型质量差不能靠改状态或降低成功阈值解决。
5. 修复后重跑最小复现及相关回归，确认日志/产物/血缘；真实运行通过平台提交有限预算的新 Run，保留原失败 Run。

例如 `queued` 且 doctor 无 GPU，先区分 probe/权限失败与设备不可用；GPU 空闲则继续看主存预算和显存估算。不要直接改 DB 或绕过 ebd 执行 equivalent-command。日志接口返回 200 仍需检查 `error`；空曲线还需核对回退数据与 provenance。

## 交接

输出“症状与复现 → 根因与证据 → 修改 → 实际检查 → 未验证项/下一步”。`trn_*`、`job_*`、`run_*` 不可混用，从训练详情取得关联 ID。取消、重试、恢复走对应 API；不删除被引用资产、不放宽门禁，不用清理 Docker 或重启全部服务替代定位。
