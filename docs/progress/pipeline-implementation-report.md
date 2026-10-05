# 顺序流水线运行隔离（2026-09-09）

本轮在现有 steps 顺序执行器上增加运行快照，没有实现通用 DAG。

- 新增 PipelineRun，保存 template ID、spec 深拷贝、创建者、status、job_ids、error。启动不再写 Pipeline.spec；同一模板可多次独立执行。0004 migration 保留已有运行历史快照。
- POST /pipelines/{id}/start 返回新运行；相同 Idempotency-Key 在同一用户/模板内返回既有运行。GET /pipelines/{id}/runs、/pipeline-runs/{id} 和现有列表显示历史与冻结 spec。
- 每一步 Job 使用 pipeline-run:{run_id}:step:{index} 幂等键。推进时先查询已提交 Job，补写游标，覆盖 Job commit 后进程丢失的窗口；唯一键并发冲突回滚后下个 tick 重试。每个 run 用保存的创建者推进。
- 有限 steps 最大64，旧的直接 JobSpec 步骤保留。新增包装形式：

```json
{
  "job": {"kind": "train.rlft", "embodiment_env_id": "...", "task_id": "..."},
  "bindings": {"inputs.base_model": {"from_step": 0, "output": "model_directory"}}
}
```

只允许引用更早步骤的完整模型目录，禁止任意参数插值。生产 Job succeeded、Run finished，并且 checkpoint-manifest.json 完整且与生产 run 相符才绑定。

共享 `eb.checkpoints.checkpoint_model(run_root, run_id)` 读取 `run_root/output/checkpoint-manifest.json`，校验 eb/checkpoint/v1 回执、64KiB 上限、相对路径、全部文件在模型目录内、无 symlink/路径穿越、唯一文件条目、实际文件大小、sha256 格式和非空 full_weights.pt。生产 task 计算 payload hash，API 不重新读取大权重计算 hash。返回模型 path、总 bytes、step 和 manifest，供模型注册/训练 API 复用。

## 验收

```
docker exec eb-tests-rereview2 python -m pytest tests/unit/test_pipelines.py -q -p no:cacheprovider
```

11 passed。覆盖顺序状态机、重复 start、模板深拷贝隔离、Job commit 后模拟中断再推进不重复提交、错误绑定、无完成回执、错误生产者、路径穿越、symlink 和文件大小变化。F 类检查通过新 pipeline/data 模块及 scheduler/model imports。

测试使用内存 session 验证控制流，尚未运行真实 PostgreSQL 并发竞争、生产迁移或真实 SFT→onlineRL 管线。内容 hash 不在 API 重算；完整性依赖受管生产 task 的回执以及后续只读模型消费，当前未实现独立不可变 publication/GC。失败续跑现已支持保留连续成功前缀，细节见下文。列表展示历史，未加入通用图编辑器或调度层 DAG。


## 从失败阶段续跑

`POST /pipeline-runs/{rid}/resume` 仅接受failed运行，创建新PipelineRun。使用原run.spec深拷贝而非当前可变模板，记录resumed_from；只复用按顺序连续succeeded的Job前缀，失败及后续步骤使用新run的幂等Job键重新提交。旧run、旧Job状态和错误记录保持原样。复用的训练checkpoint及既有模型绑定来源须先通过bound_checkpoint检查，缺失或不完整时不创建新运行。

新run的spec与已复用job_ids在同次commit创建，避免先创建空游标再补前缀导致重跑SFT。相同resume Idempotency-Key返回既有新run。页面failed历史项提供“从失败阶段继续”，复用启动pending防重和幂等请求。

容器回归覆盖成功SFT不重跑、失败和后续步骤得到新Job、旧run/spec/Job保持不变、旧spec与当前模板隔离、重复请求不多建run、缺checkpoint拒绝；pipeline测试增至13项。生产续跑由主任务在部署后单独提交。

## 独立正确性复核：数据→训练→续跑→评估

本次发现并修复的具体缺陷：

1. LIBERO评估collector原先只要求metrics为dict，空指标、NaN或零完成轨迹也会进入成功报告路径；NaN还可能在PG提交时失败而不是形成明确终态。现在要求run匹配、not_qualified凭证、非空有限标量、正整数num_trajectories及[0,1]的success_once。按上游真实语义规范化success_once→success_rate、num_trajectories→n_trials，保留原始receipt；不把success_at_end或benchmark qualification混进来。
2. checkpoint成员检查原先遗漏run/output目录自身的symlink，因此目录整体指向其他位置时能绕过子路径的symlink规则。共享validator现拒绝run_root/output根alias，保护绑定、续跑和注册的共同入口。

指标语义证据来自只读RLinf源码：libero_env._record_metrics在episode内累计success_once；env_worker.env_evaluate_step只返回newly_done轨迹记录；compute_evaluate_metrics拼接这些记录求均值，并以count_trajectories求num_trajectories。此处success_rate只表示“已完成轨迹内至少成功一次”，不表示成功保持时长，也不表示未见场景/完整基准通过。报告另记录实际model_input路径，model_version_id未知时保持空，不用路径冒充模型ID。

```
docker exec eb-tests-rereview2 python -m pytest tests/unit/test_eval_receipt.py tests/unit/test_pipelines.py tests/unit/test_orchestrator_fail.py -q -p no:cacheprovider
```

21 passed；包括空评估不登记报告、无效指标拒绝、实测分母规范化、raw证据保留、output根symlink拒绝、成功SFT复用及失败后续新Job。

以下仍是已明确的实现缺口，不宣称本轮修复：数据未发布不可变DatasetVersion/冻结split；10Hz数据与20Hz接口不一致只在G2明确披露，实验训练未自动重采样；checkpoint消费端仅复核文件路径和大小，payload哈希由生产容器计算，未建立消费前全量哈希复验；管线模型未自动登记成ModelVersion，报告通过JobSpec/model_input路径追溯而不是已发布模型版本ID；聚合评估没有逐任务attempt ledger/固定验收集证据，因此qualification始终not_qualified。也未在此审计中运行新的训练或真机评估。

### 失败续跑的单阶段内存修正

resume接口可选JSON `{ "memory_gb": 72 }`，要求1..1024的整数。仅修改新run第一剩余阶段的memory_gb，保留原shm/GPU/CPU及后续阶段；重新通过ResourceSpec并要求shm_gb<=memory_gb。spec记录resource_adjustment的step_index/source_run_id/before/after。空请求继续使用原冻结spec，旧run/Job不变，相同幂等键不创建第二个新run。

页面失败运行提供可选内存框，留空不发送body。容器pipeline测试15项通过，覆盖96→72、连续成功前缀复用、后续资源不变、旧快照不变、默认不调整、幂等重复及无效/小于shm的值；实际API请求模型边界检查通过。真实PPO续跑由主任务部署后提交。

### 2026-09-09 准入与取消的真实 PostgreSQL 并发回归

准入先锁定并刷新 Job，确认仍为 queued；取消锁定同一 Job 且使用 populate_existing，再读取最新 Run。运行回收也锁定 Job 并重检 running，空操作和查询失败及时释放锁，防止覆盖已提交的取消。容器单元回归 test_scheduler_reap.py 与 test_worktree.py 共 12 passed。

真实验证脚本保留在 `backend/tests/integration/postgres_cancel_admission.py`。两条真实 AsyncSession 分别调用实际 admit_and_start 与 cancel_job，仅模拟 Docker 和 adapter。取消事务先缓存 queued 对象；准入持锁停在 Docker 提交处时，PostgreSQL 的 pg_blocking_pids 确认取消事务确实等待；准入提交后，取消刷新已有对象、读取新 Run，并持久化 Job=cancelled、Run=aborted/abort_reason=cancelled。脚本实际退出 0，输出 PASS。该证据仅覆盖准入/取消竞争，不等于 pipeline 全链路 PostgreSQL 并发验证。

验证使用已有隔离 `eb-migration-check` 容器内专用空数据库，未写生产数据库。脚本只接受显式 `EB_ROWLOCK_TEST_DSN`，且数据库名必须为 `eb_rowlock_check`；未设置直接输出 SKIP，不读取应用数据库配置。脚本总时限 30 秒。复现命令（测试凭据仅供该临时数据库）：

```bash
docker exec eb-migration-check psql -U postgres -v ON_ERROR_STOP=1 -c "CREATE ROLE eb_rowlock_check LOGIN PASSWORD 'rowlock-check-only'" -c "CREATE DATABASE eb_rowlock_check OWNER eb_rowlock_check"
docker run --rm --network container:eb-migration-check --cpus 1 --memory 512m -v /data/kk/EmbodiedBein:/workspace:ro -e PYTHONPATH=/workspace/backend -e EB_ROWLOCK_TEST_DSN=postgresql+asyncpg://eb_rowlock_check:rowlock-check-only@127.0.0.1/eb_rowlock_check --entrypoint python eb/test:local tests/integration/postgres_cancel_admission.py
docker exec eb-migration-check psql -U postgres -v ON_ERROR_STOP=1 -c "DROP DATABASE eb_rowlock_check" -c "DROP ROLE eb_rowlock_check"
```

临时数据库和角色已清理，任务容器自动删除；无部署操作。再次运行需要重新创建空数据库，脚本不会自动清空任何已有表。

### 管线后续 PPO 轮数独立配置

创建接口 `/train/libero-pipeline` 使用 `LiberoPipelineIn`，新增 `ppo_max_epochs`（严格整数 1..10000，默认 1）。只为新管线 PPO 阶段生成 `runner.max_epochs` 覆盖，经过原有 override 校验及各阶段容量检查；SFT 的 max_steps/max_epochs 不传入 PPO 或评估，已有冻结 PipelineRun 不变。

TrainNew 增加“管线后续 PPO max_epochs”，仅管线请求发送该字段，单作业和预览继续采用独立 SFT 参数。界面明确后续 PPO/评估仍使用模板 4 个环境、右侧预览仅展示 SFT。本次未开放新的环境并行数自由度，也未修改容量公式或模板配方。

容器验证：`docker exec eb-tests-rereview2 python -m pytest tests/unit/test_train_bindings.py tests/unit/test_pipelines.py -q -p no:cacheprovider`，25 passed；覆盖默认值、非法值、SFT2步/PPO2轮隔离及容量检查收到覆盖后的 spec。前端已有 `eb-web-pipeline-review` 镜像挂载变更 TrainNew.vue 与 train-workbench.test.mjs，执行 `node --test tests/train-workbench.test.mjs && npx vue-tsc --noEmit && npm run build`，4项回归、类型检查和构建通过（仅既有大chunk提示）。未部署或启动训练。
