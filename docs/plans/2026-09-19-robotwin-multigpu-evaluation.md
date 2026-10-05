# RoboTwin 多卡评估实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让一个 RoboTwin 评估 Job 显式申请 1/2/4 张 GPU，通过 RLInf 原生多卡 placement 完成确定性 case 分片和统一证据归并。

**Architecture:** 控制面冻结 `gpu_count`、`per_gpu_concurrency` 和派生的 `eval_batch_size`，现有调度器一次分配全部 GPU，ebd 启动一个容器。容器内不同任务分片串行执行，每个分片按 active GPU 数启动一个 RLInf Ray 集群；不创建平台子 Job，也不并发运行多个 Ray 集群。

**Tech Stack:** FastAPI、Pydantic v2、Vue 3、TypeScript、RLInf、Ray、RoboTwin、pytest、Node test runner。

**Spec:** `docs/design/09-robotwin-evaluation.md`

## Global Constraints

- `/data/kk/RLinf` 与 `/data/kk/robotwin` 保持只读；本功能不修改 RLInf 上游源码。
- 执行仍由 `eval.sim Job → Run → ebd → RLInf evaluation-only entry` 完成。
- 一个 Job、一个 Run、一个容器、一份最终报告；GPU 不足时整单排队。
- GPU 数只允许 `1/2/4`，默认 `1`；每卡并发只允许 `1–5`，默认 `4`。
- 5000 等计划 case 总数不能映射成一次并发环境数。
- 不执行 Git commit。

## Review Focus

- 请求 `gpus=3`、`concurrency_per_gpu=0/6` 时 API 必须在创建计划前拒绝。
- 任务尾批小于 GPU 数或不能整除 GPU 数时，必须拆成合法分片且 case 不重不漏。
- 历史计划缺少多卡字段时必须按单卡解释，不能破坏旧评估记录。
- 子分片必须移除父命令中的旧 placement，并只引用该分片可用的逻辑 GPU rank。
- 日志出现 `RuntimeError: cannot create buffer` 时，任务错误必须提示降低每卡并发，而不是只显示退出码 255。

---

### Task 1: 冻结多卡 API 与资源请求

**Files:**
- Modify: `backend/eb/api/routes_eval.py:32-190`
- Modify: `backend/tests/unit/test_eval_plan.py:60-75`

**Interfaces:**
- Consumes: `EvalIn` 请求与 `_prepare_robotwin()` 的现有模型、Suite、镜像冻结流程。
- Produces: `EvalIn.gpus: Literal[1, 2, 4]`、`EvalIn.concurrency_per_gpu: int`，以及 `plan.model.runtime.{gpu_count,per_gpu_concurrency,eval_batch_size}`。

- [ ] **Step 1: 写失败的请求边界测试**

```python
def test_robotwin_parallelism_defaults_and_bounds():
    body = {"name":"eval", "suite_id":"suite_1", "model_version_id":"mv_1"}
    assert EvalIn(**body).gpus == 1
    assert EvalIn(**body).concurrency_per_gpu == 4
    assert EvalIn(**body, gpus=4, concurrency_per_gpu=5).gpus == 4
    for invalid in (0, 3, 8):
        with pytest.raises(ValueError):
            EvalIn(**body, gpus=invalid)
    for invalid in (0, 6):
        with pytest.raises(ValueError):
            EvalIn(**body, concurrency_per_gpu=invalid)
```

- [ ] **Step 2: 运行测试并确认旧字段导致失败**

Run: `cd backend && uv run pytest -q tests/unit/test_eval_plan.py::test_robotwin_parallelism_defaults_and_bounds`

Expected: FAIL，因为 `gpus` 与 `concurrency_per_gpu` 尚不存在。

- [ ] **Step 3: 实现最小请求与资源画像**

在 `routes_eval.py` 中使用受限类型并删除旧 `concurrency`：

```python
gpus: Literal[1, 2, 4] = 1
concurrency_per_gpu: int = Field(default=4, ge=1, le=5)


ROBOTWIN_EVAL_RESOURCES = {
    1: {"gpus":1, "cpus":8, "memory_gb":64, "shm_gb":32},
    2: {"gpus":2, "cpus":16, "memory_gb":72, "shm_gb":48},
    4: {"gpus":4, "cpus":32, "memory_gb":92, "shm_gb":64},
}
```

`_prepare_robotwin()` 必须复制资源字典，并冻结：

```python
resources = dict(ROBOTWIN_EVAL_RESOURCES[body.gpus])
runtime.update(
    gpu_count=body.gpus,
    per_gpu_concurrency=body.concurrency_per_gpu,
    eval_batch_size=body.gpus * body.concurrency_per_gpu,
)
```

资源值是当前 125 GiB 母机的首轮画像；Task 6 的真实 smoke 若触发 cgroup OOM，只能按实测提高或降低对应画像并重新预览，不能绕过调度器。

- [ ] **Step 4: 添加 plan hash 输入测试并运行**

扩展测试，断言 GPU 数或每卡并发变化会改变 `plan.hash`，且运行时三个字段满足乘法关系。

Run: `cd backend && uv run pytest -q tests/unit/test_eval_plan.py`

Expected: PASS。

### Task 2: 生成确定性的多卡合法分片

**Files:**
- Modify: `backend/eb/runners/robotwin_eval.py:13-115,210-220`
- Modify: `backend/tests/unit/test_robotwin_eval.py:90-190`

**Interfaces:**
- Consumes: `runtime.gpu_count`、`runtime.per_gpu_concurrency`；旧计划仅有 `runtime.eval_batch_size`。
- Produces: `shard_plan(plan, gpu_count, per_gpu_concurrency) -> list[dict]`；每个分片含 `active_gpu_count` 与 `per_gpu_envs`。

- [ ] **Step 1: 写尾批分片失败测试**

```python
def test_multigpu_shards_are_divisible_and_exact():
    value = plan()
    value["cases"] = [
        {"case_id":f"adjust_bottle/clean/{i:06d}", "task_id":"adjust_bottle",
         "seed":1000 + i, "scenario":"clean"}
        for i in range(22)
    ]
    value["planned"] = 22
    shards = shard_plan(value, gpu_count=4, per_gpu_concurrency=4)
    assert [(s["planned"], s["active_gpu_count"], s["per_gpu_envs"])
            for s in shards] == [(16, 4, 4), (4, 4, 1), (2, 2, 1)]
    assert [c["case_id"] for s in shards for c in s["cases"]] == [
        c["case_id"] for c in value["cases"]]
    assert all(s["planned"] == s["active_gpu_count"] * s["per_gpu_envs"] for s in shards)
```

另加 1、3、5、6、15 个 case 以及两个 task 的参数化测试。

- [ ] **Step 2: 运行测试并确认当前固定 batch_size 算法失败**

Run: `cd backend && uv run pytest -q tests/unit/test_robotwin_eval.py -k 'multigpu_shards or full_plan_shards'`

Expected: FAIL，因为分片没有 active GPU 信息，也不能保证 world size 整除。

- [ ] **Step 3: 实现最小分片循环**

每个 task 内按冻结顺序执行：

```python
while offset < len(cases):
    remaining = len(cases) - offset
    active = min(gpu_count, remaining)
    per_gpu = min(per_gpu_concurrency, remaining // active)
    planned = active * per_gpu
    chunk = cases[offset:offset + planned]
    shards.append({**plan, "suite_spec":suite, "planned":planned, "cases":chunk,
                   "task_id":task_id, "max_episode_steps":limit,
                   "active_gpu_count":active, "per_gpu_envs":per_gpu})
    offset += planned
```

校验两个输入均为正整数，并继续保留总 case 数与原顺序检查。

- [ ] **Step 4: 增加历史计划兼容解析**

在 `evaluate()` 增加一个小型解析函数：新计划读取 `gpu_count/per_gpu_concurrency`；旧计划返回 `(1, eval_batch_size)`。对新计划校验 `eval_batch_size == gpu_count * per_gpu_concurrency`，不一致直接失败。

- [ ] **Step 5: 运行 runner 单元测试**

Run: `cd backend && uv run pytest -q tests/unit/test_robotwin_eval.py`

Expected: PASS。

### Task 3: 为每个分片渲染 RLInf 原生多卡 placement

**Files:**
- Modify: `backend/eb/runners/robotwin_eval.py:58-110`
- Modify: `backend/tests/unit/test_robotwin_eval.py:140-190`
- Modify: `backend/tests/unit/test_rlinf_adapter.py:175-220`

**Interfaces:**
- Consumes: Task 2 的 `active_gpu_count` 与 `per_gpu_envs`。
- Produces: 每个子进程唯一的 `cluster.component_placement` 和可整除的 `env.eval.total_num_envs`。

- [ ] **Step 1: 写子命令快照失败测试**

对 4 卡分片断言命令包含：

```python
assert "cluster.component_placement={env\\, rollout:0-3}" in command
assert "env.eval.total_num_envs=16" in command
```

对 2 卡尾批断言 placement 为 `0-1`、环境数为 `2`，并断言父命令中的 `cluster.component_placement={env\, rollout:0-0}` 不再出现。

- [ ] **Step 2: 运行测试确认当前命令仍固定为单卡**

Run: `cd backend && uv run pytest -q tests/unit/test_robotwin_eval.py::test_sharded_evaluation_merges_one_job_evidence`

Expected: FAIL，当前子分片未渲染 placement。

- [ ] **Step 3: 在共享渲染点替换 placement**

将 `cluster.component_placement=` 加入 `evaluate_shards()` 的 `replaced` 前缀，并为子命令追加：

```python
active = shard["active_gpu_count"]
placement = f"cluster.component_placement={{env\\, rollout:0-{active - 1}}}"
```

保持分片串行 `subprocess.run(command, env=env, check=True)`；不要引入线程池、多进程池或第二个 Ray 集群。

- [ ] **Step 4: 保持父级 adapter 命令可预览**

`RlinfAdapter` 继续输出评估入口和完整计划总数，但预览资源必须为所选 GPU 数。runner 子进程负责把总数替换成实际分片数和 placement；测试断言入口仍是 `evaluations/eval_embodied_agent.py`。

- [ ] **Step 5: 运行 adapter 与 runner 测试**

Run: `cd backend && uv run pytest -q tests/unit/test_robotwin_eval.py tests/unit/test_rlinf_adapter.py`

Expected: PASS。

### Task 4: 把图形缓冲区耗尽转成可操作错误

**Files:**
- Modify: `backend/eb/orchestrator/service.py:505-525`
- Modify: `backend/tests/unit/test_scheduler_reap.py`

**Interfaces:**
- Consumes: Run 的 `console.log`。
- Produces: `run.abort_reason="graphics_buffer_exhausted"` 和简短中文 `job.error`。

- [ ] **Step 1: 写失败日志分类测试**

构造包含以下调用链的 console：

```text
self.left_camera.take_picture()
RuntimeError: cannot create buffer
```

断言 reap 后：

```python
assert run.abort_reason == "graphics_buffer_exhausted"
assert "降低每卡并发环境数" in job.error
```

- [ ] **Step 2: 运行测试确认当前只返回 python_exception**

Run: `cd backend && uv run pytest -q tests/unit/test_scheduler_reap.py -k graphics_buffer`

Expected: FAIL。

- [ ] **Step 3: 在统一日志分类点添加一个优先分支**

当日志同时包含 `take_picture` 和 `RuntimeError: cannot create buffer` 时，设置结构化 abort reason 与提示；其他 Traceback 继续走现有逻辑。不要创建新的异常框架。

- [ ] **Step 4: 运行调度回收测试**

Run: `cd backend && uv run pytest -q tests/unit/test_scheduler_reap.py`

Expected: PASS。

### Task 5: 更新评估页面与静态前端检查

**Files:**
- Modify: `frontend/src/views/EvalNew.vue:1-100`
- Modify: `frontend/tests/robotwin-workbench.test.mjs`

**Interfaces:**
- Consumes: `POST /api/v1/evals/previews` 的 `gpus`、`concurrency_per_gpu` 请求字段和 runtime 预览字段。
- Produces: GPU 数下拉框、每卡并发输入框及最大同时环境数预览。

- [ ] **Step 1: 写失败的页面契约检查**

```javascript
test('RoboTwin evaluation exposes bounded multi-GPU controls', () => {
  const source = readFileSync(new URL('../src/views/EvalNew.vue', import.meta.url), 'utf8');
  assert.match(source, /concurrency_per_gpu/);
  assert.match(source, /每卡并发环境数/);
  assert.match(source, /最大同时环境数/);
  assert.match(source, /<option value="4">4 卡<\/option>/);
});
```

- [ ] **Step 2: 运行测试确认旧页面失败**

Run: `cd frontend && node --test tests/robotwin-workbench.test.mjs`

Expected: FAIL。

- [ ] **Step 3: 实现原生表单控件**

增加 `gpus=1`、`concurrencyPerGpu=4`；GPU 使用 `<select>` 固定 1/2/4，单卡并发使用 `<input type="number" min="1" max="5">`。`body()` 发送 `gpus` 和 `concurrency_per_gpu`，watch 列表包含两者。

预览显示：

```text
GPU 4 卡 · 每卡并发 4 · 最大同时环境数 16
```

删除旧的“并发环境数 1–16”语义。

- [ ] **Step 4: 运行前端测试和类型构建**

Run: `cd frontend && node --test tests/*.test.mjs && npm run build`

Expected: 所有测试 PASS，Vite 构建成功。

### Task 6: 回归验证、部署与真实双卡/四卡 smoke

**Files:**
- Modify only if measured evidence requires it: `backend/eb/api/routes_eval.py` 中 `ROBOTWIN_EVAL_RESOURCES`
- Record evidence in existing Run output; do not create a second reporting path.

**Interfaces:**
- Consumes: Tasks 1–5 的 API、runner、adapter 和页面。
- Produces: 双卡与四卡各一个完成的 smoke Job、逐 case evidence 和完整 EvalReport。

- [ ] **Step 1: 运行针对性后端回归**

Run:

```bash
cd backend && uv run pytest -q \
  tests/unit/test_eval_plan.py \
  tests/unit/test_robotwin_eval.py \
  tests/unit/test_rlinf_adapter.py \
  tests/unit/test_scheduler_reap.py
```

Expected: PASS。

- [ ] **Step 2: 运行全量仓库检查**

Run: `make test`

Expected: backend pytest 与 Rust workspace tests 全部 PASS。

- [ ] **Step 3: 构建并启动平台**

Run: `make up`

Expected: api、ebd、postgres、redis、minio 均 healthy，页面 `/eval/new` 可打开。

- [ ] **Step 4: 从页面提交双卡 smoke**

选择已验证 π0.5 ModelVersion、RoboTwin smoke Suite、`GPU 数=2`、`每卡并发=2`。预览必须显示 5 个计划 case、最大同时 4 个环境和 2 GPU；提交后等待 Job 终态。

验收：Run 的 `gpu_ids` 恰有两个不同 ID；分片为 4-case 双卡波次和 1-case 单卡尾批；最终报告 `valid=5/5`，case ID 不重不漏。

- [ ] **Step 5: 从页面提交四卡 smoke**

使用相同模型与 Suite，选择 `GPU 数=4`、`每卡并发=1`。资源不足时状态保持 queued；四张卡同时空闲后只能一次性准入。

验收：Run 的 `gpu_ids` 恰有四个不同 ID；分片为 4-case 四卡波次和 1-case 单卡尾批；最终报告 `valid=5/5`。

- [ ] **Step 6: 核对证据与资源错误**

列出最近两小时生成的归并证据：

```bash
state_dir=$(sed -n 's/^EB_HOST_STATE_DIR=//p' deploy/.env)
find "${state_dir:-/data/kk/embodiedbein}/runs" \
  -path '*/output/eval-episodes-rank0.jsonl' -mmin -120 -print
```

根据评估详情页显示的两个 Run ID 选择对应结果文件，检查每个计划 case 恰好一行、`plan_hash` 一致、policy receipt 一致。若出现 cgroup OOM，依据 peak RSS 调整 Task 1 的对应资源画像后重新预览；若出现 `cannot create buffer`，降低每卡并发，不能增加单卡上限。

- [ ] **Step 7: 最终复查工作区**

Run: `git diff --check && git status --short`

Expected: 无空白错误；仅保留本任务和进入任务前已经存在的改动，不执行 commit。
