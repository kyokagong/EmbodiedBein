# RoboTwin Single-Policy Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace RoboTwin evaluation’s per-GPU π0.5 replicas and per-wave RLInf subprocesses with one persistent Policy Worker, multiple Env Workers, bounded batched inference, and exact resumable full-suite evidence.

**Architecture:** One ebd-created container runs one Ray cluster. `rollout:0` owns the only model, `env:0-(N-1)` owns distributed offscreen environments, and a suite coordinator advances frozen cases in process-local waves while retaining the Policy Worker across waves and tasks. The runtime keeps EGL/Vulkan camera rendering, disables video, bounds observation traffic to one in-flight batch, and applies any RLInf changes to an immutable image copy rather than `/data/kk/RLinf`.

**Tech Stack:** Python 3.11, FastAPI/Pydantic v2, Vue 3/TypeScript, RLInf/Ray, PyTorch/OpenPI π0.5, RoboTwin/SAPIEN/Vulkan/EGL, Docker BuildKit, pytest and Node test scripts.

**Spec:** `docs/rlinf/02-robotwin-single-policy-eval.md`

## Global Constraints

- `docs/design/` remains the product design source; implementation must match `docs/design/09-robotwin-evaluation.md` and the RLInf detail spec.
- `/data/kk/RLinf` and RoboTwin source are read-only; no task may edit them.
- All evaluation execution remains containerized and launched only by `ebd`; FastAPI must not run evaluation subprocesses.
- One Eval Job creates one container, one Ray cluster, one Policy Worker, and one model load.
- Headless evaluation retains EGL/Vulkan/SAPIEN offscreen rendering for the three required cameras.
- Full evaluation defaults to `save_video=false` and `max_inflight_batches=1`.
- Existing frozen case identity, success function, action profile, evidence hashes and report denominator remain unchanged.
- No database migration is planned.
- Do not run `git commit`; the user explicitly prohibited git commits.

## Review Focus

- A tail wave with fewer cases than Env Workers must activate exact slots without padding or changing Policy placement; Task 3 pins this.
- A stale legacy preview containing `per_gpu_concurrency` must be readable but rejected for new submission; Task 1 pins this.
- Multiple Env ranks must receive actions in their original rank/slot order after micro-batching; Task 5 pins this.
- A task-switch cleanup timeout or stale Channel message must stop the suite before the next environment starts; Task 6 pins this.
- Recovery must ignore malformed or mismatched partial evidence and rerun only valid pending case IDs; Task 7 pins this.

---

## File Map

### Create

- `backend/eb/domain/robotwin_eval_runtime.py` — validated topology profiles and runtime derivation.
- `deploy/patches/rlinf/robotwin-single-policy-eval.patch` — minimal patch against the frozen RLInf revision.
- `backend/tests/unit/test_robotwin_eval_runtime.py` — topology/profile contract tests.
- `docs/progress/2026-09-20-robotwin-single-policy-eval.md` — implementation results and measured gaps after real smoke.

### Modify

- `backend/eb/api/routes_eval.py` — preview/submit schema and runtime/resource derivation.
- `backend/eb/domain/rlinf_form.py` — evaluation overrides and placement rendering.
- `backend/eb/adapters/rlinf.py` — immutable patched source, headless/video settings and mounts.
- `backend/eb/runners/robotwin_eval.py` — persistent suite coordinator and process-local waves.
- `backend/eb/runners/robotwin_env.py` — wave identity, slot evidence and lifecycle checks.
- `backend/eb/runners/rlinf_entry.py` — dispatch without recursive per-wave child processes.
- `backend/eb/capacity/estimator.py` — single-policy GPU estimate.
- `backend/eb/orchestrator/collect.py` — topology receipt and resumable evidence validation.
- `deploy/images/rlinf-robotwin.Dockerfile` — copy frozen RLInf source and apply tracked patch.
- `Makefile` — supply the read-only RLInf build context and revision/hash arguments.
- `frontend/src/views/EvalNew.vue` — total parallel environment profile selection and topology preview.
- `frontend/src/views/EvalJobDetail.vue` — task/wave/model-load diagnostics.
- Existing unit and frontend tests listed in the spec.

### External patch targets inside the image copy

- `rlinf/runners/embodied_eval_runner.py`
- `rlinf/workers/rollout/hf/huggingface_worker.py`
- `rlinf/workers/env/env_worker.py`
- `rlinf/envs/sim/robotwin/robotwin_env.py`
- `evaluations/robotwin/robotwin_adjust_bottle_openpi_pi05_eval.yaml`
- `rlinf/utils/placement.py` only if an executable placement probe proves the current parser cannot represent separate component ranges.

---

### Task 1: Freeze the single-policy runtime contract

**Files:**
- Create: `backend/eb/domain/robotwin_eval_runtime.py`
- Create: `backend/tests/unit/test_robotwin_eval_runtime.py`
- Modify: `backend/eb/api/routes_eval.py`
- Modify: `backend/tests/unit/test_eval_plan.py`

**Interfaces:**
- Produces: `RobotwinEvalRuntime` Pydantic model.
- Produces: `derive_robotwin_runtime(gpus: int, parallel_envs: int) -> tuple[RobotwinEvalRuntime, dict[str, int]]`.
- Produces: `read_robotwin_runtime(plan: dict, *, submission: bool) -> RobotwinEvalRuntime`.
- Later tasks consume `runtime.model_dump()` and the returned resource dictionary.

- [ ] **Step 1: Write failing topology tests**

```python
@pytest.mark.parametrize("gpus,parallel,workers,envs,micro", [
    (1, 4, 1, 4, 4),
    (2, 8, 2, 4, 4),
    (4, 16, 4, 4, 4),
])
def test_verified_profiles_use_one_policy(gpus, parallel, workers, envs, micro):
    runtime, resources = derive_robotwin_runtime(gpus, parallel)
    assert runtime.policy_worker_count == 1
    assert runtime.env_worker_count == workers
    assert runtime.envs_per_worker == envs
    assert runtime.parallel_envs == parallel
    assert runtime.inference_micro_batch_size == micro
    assert runtime.max_inflight_batches == 1
    assert runtime.headless and runtime.offscreen_rendering
    assert runtime.save_video is False
    assert resources["memory_gb"] >= resources["shm_gb"]


def test_legacy_runtime_is_readable_but_not_submittable():
    plan = {"model":{"runtime":{"gpu_count":4,"per_gpu_concurrency":4,
                                 "eval_batch_size":16}}}
    assert read_robotwin_runtime(plan, submission=False).parallel_envs == 16
    with pytest.raises(ValueError, match="重新预览"):
        read_robotwin_runtime(plan, submission=True)
```

- [ ] **Step 2: Run the focused tests and observe the missing module failure**

Run: `cd backend && uv run pytest -q tests/unit/test_robotwin_eval_runtime.py tests/unit/test_eval_plan.py`

Expected: import failure for `eb.domain.robotwin_eval_runtime` or assertions against the old runtime schema.

- [ ] **Step 3: Implement the minimal validated runtime model**

The module must define exactly three verified profiles:

```python
PROFILES = {
    (1, 4): {"env_worker_count":1, "envs_per_worker":4,
             "inference_micro_batch_size":4, "cpus":8, "memory_gb":48, "shm_gb":24},
    (2, 8): {"env_worker_count":2, "envs_per_worker":4,
             "inference_micro_batch_size":4, "cpus":16, "memory_gb":56, "shm_gb":32},
    (4, 16): {"env_worker_count":4, "envs_per_worker":4,
              "inference_micro_batch_size":4, "cpus":32, "memory_gb":64, "shm_gb":40},
}
```

The values are initial admission ceilings, not measured claims; Task 9 must replace or confirm them from real probes before exposing full evaluation. Validation must enforce all invariants from spec section 10.

- [ ] **Step 4: Replace `EvalIn.concurrency_per_gpu` with `parallel_envs`**

Keep a read-only response compatibility path for historical plans. New preview and submit requests accept only `parallel_envs`; unknown GPU/parallel combinations return a clear validation error.

- [ ] **Step 5: Run tests**

Run: `cd backend && uv run pytest -q tests/unit/test_robotwin_eval_runtime.py tests/unit/test_eval_plan.py`

Expected: all selected tests pass.

---

### Task 2: Render separated placement and immutable headless settings

**Files:**
- Modify: `backend/eb/domain/rlinf_form.py`
- Modify: `backend/eb/adapters/rlinf.py`
- Modify: `backend/tests/unit/test_rlinf_form.py`
- Modify: `backend/tests/unit/test_rlinf_adapter.py`

**Interfaces:**
- Consumes: `RobotwinEvalRuntime` from Task 1.
- Produces: `robotwin_eval_placement(runtime) -> dict[str, str]` with one rollout GPU and fixed Env Worker GPUs.
- Produces argv containing `runner.enable_decoupled_mode=False`, `rollout.enable_offload=False`, and `env.eval.video_cfg.save_video=False`.

- [ ] **Step 1: Write adapter snapshot tests**

```python
def test_robotwin_eval_uses_one_policy_and_four_env_workers(rendered_robotwin_job):
    args = rendered_robotwin_job.command
    assert r"cluster.component_placement.rollout=0" in args
    assert r"cluster.component_placement.env=0-3" in args
    assert "env.eval.video_cfg.save_video=False" in args
    assert "rollout.enable_offload=False" in args
    assert not any("env\\, rollout" in value for value in args)
```

Also assert the environment retains `NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics`, `MUJOCO_GL=egl`, and `PYOPENGL_PLATFORM=egl`.

- [ ] **Step 2: Run focused tests and confirm old collocated placement fails**

Run: `cd backend && uv run pytest -q tests/unit/test_rlinf_form.py tests/unit/test_rlinf_adapter.py`

Expected: assertions show the old combined placement or missing headless overrides.

- [ ] **Step 3: Implement separated placement rendering**

Use the placement syntax accepted by an executable RLInf config probe. Do not guess syntax in production code: add a small test that constructs `HybridComponentPlacement` with rollout GPU `[0]` and Env GPUs `[0, 1, 2, 3]`.

- [ ] **Step 4: Remove the external RLInf mount from the executable path for patched evaluation images**

For RoboTwin evaluation only, reserve `/workspace/rlinf` for the immutable image copy. Mount the frozen source snapshot at `/workspace/source-receipts/rlinf` read-only for provenance validation. Training and non-RoboTwin paths keep their current mounts.

- [ ] **Step 5: Run focused tests**

Run: `cd backend && uv run pytest -q tests/unit/test_rlinf_form.py tests/unit/test_rlinf_adapter.py`

Expected: all selected tests pass and rendered mounts contain no host source over `/workspace/rlinf` for RoboTwin eval.

---

### Task 3: Replace subprocess shards with deterministic in-process waves

**Files:**
- Modify: `backend/eb/runners/robotwin_eval.py`
- Modify: `backend/tests/unit/test_robotwin_eval.py`

**Interfaces:**
- Replaces: `runtime_parallelism`, `shard_plan`, and `evaluate_shards` for new plans.
- Produces: `wave_plan(plan: dict, runtime: RobotwinEvalRuntime) -> list[TaskWave]`.
- Produces: `TaskWave(task_id, wave_id, cases, active_slots, max_episode_steps)`.
- Historical helper functions may remain only for reading old evidence; new evaluation must never call `subprocess.run()`.

- [ ] **Step 1: Write wave planning tests**

```python
def test_tail_wave_uses_exact_slots_without_reducing_workers():
    waves = wave_plan(plan_with_cases(22), runtime_4x16())
    assert [len(w.cases) for w in waves] == [16, 6]
    assert waves[0].active_slots == tuple(range(16))
    assert waves[1].active_slots == tuple(range(6))
    assert all(w.env_worker_count == 4 for w in waves)


def test_new_evaluation_has_no_subprocess_boundary(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("subprocess"))
    coordinator = fake_suite_coordinator(two_wave_plan())
    assert coordinator.run()["num_trajectories"] == 22
```

- [ ] **Step 2: Run tests and verify failure against `evaluate_shards()`**

Run: `cd backend && uv run pytest -q tests/unit/test_robotwin_eval.py`

Expected: tests fail because existing code creates new subprocesses and shrinks active GPU count for tail shards.

- [ ] **Step 3: Implement immutable `TaskWave` planning**

Group by frozen task order, preserve frozen case order, keep the Env Worker placement constant, and mark unused tail slots inactive. Strip the duplicated top-level `suite_spec.cases` from wave payloads.

- [ ] **Step 4: Add a coordinator seam without launching Ray in unit tests**

Define a narrow backend protocol used only for testing:

```python
class EvalBackend(Protocol):
    def start_policy(self, cfg, runtime) -> dict: ...
    def start_task(self, task_id, max_episode_steps, runtime) -> None: ...
    def run_wave(self, wave: TaskWave) -> dict: ...
    def close_task(self) -> None: ...
    def close(self) -> None: ...
```

Production implementation remains RLInf-backed; tests use a fake. Do not add a plugin registry.

- [ ] **Step 5: Run focused tests**

Run: `cd backend && uv run pytest -q tests/unit/test_robotwin_eval.py`

Expected: wave tests pass and no new-plan test reaches `subprocess.run()`.

---

### Task 4: Build a traceable patched RLInf image copy

**Files:**
- Create: `deploy/patches/rlinf/robotwin-single-policy-eval.patch`
- Modify: `deploy/images/rlinf-robotwin.Dockerfile`
- Modify: `Makefile`
- Modify: `backend/eb/adapters/rlinf.py`
- Test: `backend/tests/unit/test_rlinf_adapter.py`

**Interfaces:**
- Produces image file `/workspace/rlinf/.eb-source-receipt.json` containing upstream content hash and patch SHA-256.
- Runner consumes `EB_RLINF_CONTENT_HASH` and `EB_RLINF_PATCH_SHA256` and compares them to the image receipt before loading RLInf.

- [ ] **Step 1: Add failing receipt and mount tests**

```python
def test_robotwin_eval_requires_patched_image_receipt(rendered_robotwin_job):
    assert rendered_robotwin_job.env["EB_RLINF_PATCH_SHA256"].startswith("sha256:")
    assert rendered_robotwin_job.env["EB_RLINF_CONTENT_HASH"] == FROZEN_RLINF_HASH
    assert not any(dst == "/workspace/rlinf" for _, dst, _ in rendered_robotwin_job.mounts)
```

- [ ] **Step 2: Add the read-only BuildKit context**

Change `images-rlinf` to invoke:

```make
+docker build --build-context rlinf=/data/kk/RLinf \
+  --build-arg RLINF_CONTENT_HASH=<validated-content-hash> \
+  -t eb/rlinf:robotwin-pi05 -f deploy/images/rlinf-robotwin.Dockerfile .
```

The Dockerfile copies `rlinf` into `/workspace/rlinf`, runs `git apply --check`, applies the tracked patch, writes the receipt, removes copied `.git` metadata if present, and leaves the tree immutable at runtime.

- [ ] **Step 3: Add pre-patch checks to the patch file**

The patch must apply only to the frozen upstream revision and must fail the image build if upstream context changed. It may touch only the external target paths listed in the file map.

- [ ] **Step 4: Validate receipt before import**

`rlinf_entry.py` or `robotwin_eval.py` must read the image receipt before importing patched modules. Hash or patch mismatch raises a deterministic configuration error.

- [ ] **Step 5: Run unit tests and a Docker build**

Run:

```bash
cd backend && uv run pytest -q tests/unit/test_rlinf_adapter.py tests/unit/test_rlinf_entry.py
make images-rlinf
```

Expected: tests pass; `eb/rlinf:robotwin-pi05` builds; `/data/kk/RLinf` remains unchanged.

---

### Task 5: Patch RLInf for one Policy Worker and bounded micro-batching

**Files represented in:**
- `deploy/patches/rlinf/robotwin-single-policy-eval.patch`

**External patch targets:**
- `rlinf/workers/rollout/hf/huggingface_worker.py`
- `rlinf/runners/embodied_eval_runner.py`
- `evaluations/robotwin/robotwin_adjust_bottle_openpi_pi05_eval.yaml`

**Interfaces:**
- Produces: `MultiStepRolloutWorker.evaluate_wave(input_channel, output_channel, steps, micro_batch_size)`.
- Produces: `_predict_eval_micro_batches(obs: dict, micro_batch_size: int) -> torch.Tensor`.
- Produces: `EmbodiedEvalRunner.evaluate_wave(steps: int) -> dict` without reinitializing workers.

- [ ] **Step 1: Add patch tests for merge, micro-batch and route split**

The patched RLInf test content must include an executable test equivalent to:

```python
def test_one_policy_merges_and_restores_four_env_routes(worker):
    incoming = [obs(rank, size=4) for rank in range(4)]
    merged = worker._merge_obs_batches(incoming)
    actions = worker._predict_eval_micro_batches(merged["obs"], 4)
    restored = worker._split_actions(actions, [4, 4, 4, 4])
    assert actions.shape == (16, 50, 14)
    assert [part.shape for part in restored] == [(4, 50, 14)] * 4
    assert route_markers(restored) == [0, 1, 2, 3]
```

Also test invalid micro-batch values and a non-divisible tail batch.

- [ ] **Step 2: Verify current upstream behavior fails the new topology test**

Run the tests against the unpatched read-only source in a disposable container or temporary copy. Expected: missing `evaluate_wave`/micro-batch interface or incorrect worker count.

- [ ] **Step 3: Implement micro-batch slicing in the Policy Worker**

Slice every tensor/list observation field on batch dimension, call the existing `_predict_rollout_actions()`, concatenate outputs in original order, and release each temporary output before the next slice. Keep train/generate paths unchanged.

- [ ] **Step 4: Make Policy lifecycle explicit**

`init_workers()` loads once. `evaluate_wave()` must not call `reload_model()` or `offload_model()`. Final `close()` performs model cleanup once. Add counters exposed in the receipt path so `model_load_count == 1` is measured rather than inferred.

- [ ] **Step 5: Keep the channel bounded**

Use synchronous request/response for the first implementation. No Env rank sends a second observation until its prior action arrives. Reject config with `max_inflight_batches != 1`.

- [ ] **Step 6: Run patched RLInf tests in the image**

Run the patch-specific test command inside `eb/rlinf:robotwin-pi05`.

Expected: merge/split, tail micro-batch and single-load lifecycle tests pass.

---

### Task 6: Add reusable task/wave environment lifecycle

**Files:**
- Modify: `backend/eb/runners/robotwin_env.py`
- Modify: `backend/eb/runners/robotwin_eval.py`
- Modify patch targets: `rlinf/workers/env/env_worker.py`, `rlinf/envs/sim/robotwin/robotwin_env.py`
- Modify: `backend/tests/unit/test_robotwin_eval.py`

**Interfaces:**
- Produces: `configure_eval_wave(cases: list[dict], active_slots: tuple[int, ...])`.
- Produces: `close_eval_task(timeout_s: float) -> dict` returning process/renderer/channel cleanup evidence.
- Coordinator consumes cleanup evidence before starting the next task.

- [ ] **Step 1: Write lifecycle failure tests**

```python
def test_task_switch_requires_clean_barrier(fake_backend):
    fake_backend.close_result = {"children_alive":1,"pending_messages":0}
    with pytest.raises(RuntimeError, match="清理"):
        run_two_task_suite(fake_backend)
    assert fake_backend.started_tasks == ["task_a"]


def test_tail_wave_writes_only_active_slots(worker):
    worker.configure_eval_wave(cases=six_cases(), active_slots=tuple(range(6)))
    rows = worker.finish_wave(fake_results_for_six())
    assert [row["env_slot"] for row in rows] == list(range(6))
```

- [ ] **Step 2: Run tests and confirm lifecycle methods are absent**

Run: `cd backend && uv run pytest -q tests/unit/test_robotwin_eval.py`

- [ ] **Step 3: Implement wave seed/slot configuration**

Each active slot receives exactly one case ID and seed. Inactive slots do not reset, render, send observations, or produce evidence. Evidence records worker rank, env slot, wave ID and the common model load ID.

- [ ] **Step 4: Implement idempotent environment close**

Close VectorEnv, renderer resources and child processes; wait with a finite timeout; return counts. Calling close twice must be safe. A timeout or pending Channel message prevents task transition.

- [ ] **Step 5: Remove unnecessary image copies and video buffers**

In the patched `robotwin_env.py`, bypass PIL when the image is already contiguous uint8 RGB and no crop is requested. Do not allocate frame history when video is disabled.

- [ ] **Step 6: Run unit and patched environment tests**

Expected: exact active slots, idempotent close, stale-message rejection and no-PIL fast path pass.

---

### Task 7: Preserve evidence, receipts and exact recovery

**Files:**
- Modify: `backend/eb/runners/robotwin_eval.py`
- Modify: `backend/eb/runners/robotwin_env.py`
- Modify: `backend/eb/orchestrator/collect.py`
- Modify: `backend/tests/unit/test_eval_receipt.py`
- Modify: `backend/tests/unit/test_eval_plan.py`

**Interfaces:**
- Produces: `validated_completed_case_ids(output: Path, plan: dict, receipt_sha256: str) -> set[str]`.
- Produces receipt fields `policy_worker_count`, `model_load_count`, headless/offscreen/video and micro-batch.

- [ ] **Step 1: Write recovery integrity tests**

```python
def test_resume_only_skips_valid_completed_cases(tmp_path):
    write_valid_case(tmp_path, "case-1", plan_hash=PLAN, receipt=RECEIPT)
    write_valid_case(tmp_path, "case-2", plan_hash="wrong", receipt=RECEIPT)
    assert validated_completed_case_ids(tmp_path, plan(), RECEIPT) == {"case-1"}


def test_complete_report_requires_single_model_load(receipt):
    receipt["model_load_count"] = 2
    with pytest.raises(ValueError, match="模型加载次数"):
        validate_robotwin_receipt(receipt)
```

- [ ] **Step 2: Run focused tests and verify failure**

Run: `cd backend && uv run pytest -q tests/unit/test_eval_receipt.py tests/unit/test_eval_plan.py`

- [ ] **Step 3: Implement atomic append and resume validation**

Each row must bind plan hash, task config hash, model receipt hash, wave/rank/slot and frozen case identity. Flush and `fsync` after each completed wave. Malformed, duplicate or mismatched rows remain diagnostic evidence but do not remove a case from the pending set.

- [ ] **Step 4: Enforce receipt topology during collection**

A complete report requires one Policy Worker, one model load, headless true, offscreen rendering true and video false. Partial reports retain measured cases but never receive complete qualification.

- [ ] **Step 5: Run focused tests**

Expected: recovery and receipt tests pass.

---

### Task 8: Update capacity, API preview and UI

**Files:**
- Modify: `backend/eb/capacity/estimator.py`
- Modify: `backend/tests/unit/test_capacity.py`
- Modify: `frontend/src/views/EvalNew.vue`
- Modify: `frontend/src/views/EvalJobDetail.vue`
- Modify: `frontend/tests/robotwin-workbench.test.mjs`

**Interfaces:**
- Consumes: verified runtime/resource profile from Task 1.
- UI submits `{gpus, parallel_envs}` and renders the derived topology; it never submits placement or Policy Worker count.

- [ ] **Step 1: Write capacity and frontend assertions**

```python
def test_more_eval_gpus_do_not_multiply_policy_weights():
    one = estimate(robotwin_eval_spec(gpus=1, parallel_envs=4))
    four = estimate(robotwin_eval_spec(gpus=4, parallel_envs=16))
    assert one.basis["inputs"]["policy_replicas"] == 1
    assert four.basis["inputs"]["policy_replicas"] == 1
```

Frontend test assertions must require the labels “模型副本 1”, “总并发环境”, “推理 micro-batch”, “Headless”, “离屏三相机” and “视频关闭”.

- [ ] **Step 2: Run tests and verify old UI/schema failures**

Run:

```bash
cd backend && uv run pytest -q tests/unit/test_capacity.py
node frontend/tests/robotwin-workbench.test.mjs
```

- [ ] **Step 3: Update the estimator**

Represent a single Policy model plus micro-batch activation and rendering reserve. Do not claim the initial CPU memory profile is measured; label it `provisional` until Task 9 writes measured values.

- [ ] **Step 4: Update form and preview**

Replace “每卡并发” with verified total environment profiles. Preview must show one model replica and the derived resources. Detail view shows current task/wave, completed/planned cases, model load count and measured peak memory when available.

- [ ] **Step 5: Run backend and frontend checks**

Run:

```bash
cd backend && uv run pytest -q tests/unit/test_capacity.py tests/unit/test_eval_plan.py
cd frontend && npm run build
node frontend/tests/robotwin-workbench.test.mjs
```

Expected: all commands pass.

---

### Task 9: Run real topology probes and freeze safe profiles

**Files:**
- Modify: `backend/eb/domain/robotwin_eval_runtime.py`
- Create: `docs/progress/2026-09-20-robotwin-single-policy-eval.md`

**Interfaces:**
- Consumes the deployed API/ebd/image stack.
- Produces measured CPU RAM, `/dev/shm`, object store and per-GPU VRAM for each admitted profile.

- [ ] **Step 1: Rebuild and restart the platform**

Run `make images-rlinf` followed by `make up`.

Expected: api, ebd, postgres, redis and minio are healthy; image receipt matches the frozen plan source hash.

- [ ] **Step 2: Run 1 GPU / 4 env / 5 case smoke**

Capture container peak RSS, Ray worker RSS, `/dev/shm`, object store, Policy VRAM, renderer VRAM, model load count and case completeness.

- [ ] **Step 3: Run 4 GPU / 16 env / 16 case topology smoke**

Acceptance for this step:

```text
Policy Worker == 1
model_load_count == 1
completed cases == 16
unknown/duplicate cases == 0
container not OOM-killed
```

- [ ] **Step 4: Run a two-wave single-task smoke**

Use more than 16 cases. Verify the second wave reuses the same model load ID and RSS does not rise monotonically.

- [ ] **Step 5: Run a two-task smoke**

Verify task A environment cleanup completes before task B starts, Policy identity remains unchanged, and stale Channel messages are zero.

- [ ] **Step 6: Replace provisional resources with measured safe limits**

Round measured peaks upward and retain at least 15% headroom. Keep any profile blocked if it misses the margin or cleanup criteria.

- [ ] **Step 7: Record evidence in the progress document**

Include run IDs, image digest, source and patch hashes, exact topology, peak measurements, completed cases and unresolved limitations. Do not describe an unrun profile as validated.

---

### Task 10: Full regression and full-suite readiness

**Files:**
- Modify only files required by failures found here.

**Interfaces:**
- Produces final verified platform state; no new feature interface.

- [ ] **Step 1: Run backend tests**

Run: `cd backend && uv run pytest -q`

Expected: zero failures.

- [ ] **Step 2: Run frontend build and tests**

Run:

```bash
cd frontend && npm run build
for test_file in tests/*.test.mjs; do node "$test_file"; done
```

Expected: build succeeds and all scripts exit zero.

- [ ] **Step 3: Run Rust workspace tests**

Run: `cd rust && cargo test --workspace`

Expected: zero failures.

- [ ] **Step 4: Verify external repositories are unchanged**

Record the before/after content hashes of `/data/kk/RLinf` and the RoboTwin source root. They must match. Verify no task mount exposes either path writable.

- [ ] **Step 5: Run 100-case and 150-case gates**

A 100-case single-task run and a 150-case multi-task run must satisfy exact case accounting, one model load and stable task-switch memory before enabling the 5000-case action in the UI.

- [ ] **Step 6: Start the full 5000-case evaluation only after all gates pass**

The full run must use the frozen model, suite, image digest, source/patch hashes and measured resource profile. Completion requires 5000 unique valid case records and one Policy receipt. If runtime is too long for the current work session, leave the Job running and report its Job/Run IDs without claiming completion.

---

## Plan Self-Review

- Spec coverage: topology, batching, headless/offscreen behavior, task/wave lifecycle, resource modeling, image patch provenance, evidence, recovery, UI and staged real validation each map to a task.
- Placeholder scan: no implementation step delegates unspecified validation or error handling.
- Type consistency: Tasks 1–8 consistently use `RobotwinEvalRuntime`, `parallel_envs`, `TaskWave`, `model_load_count` and one Policy Worker.
- Review-focus coverage: tail waves are in Task 3; legacy previews in Task 1; action route order in Task 5; cleanup barriers in Task 6; partial-evidence recovery in Task 7.
- User constraint: commit steps are deliberately omitted; implementation must not create git commits.
