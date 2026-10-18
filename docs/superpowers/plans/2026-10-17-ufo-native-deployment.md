# UFO Native RPO Deployment Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans; implement in the current workspace without Git commits, preserving all pre-existing changes.

**Goal:** Import the specified native UFO model, run its exported inference package in local deployment simulations, and make the package and evidence downloadable.

**Architecture:** Reuse Job/Run/ebd and existing Artifact/ModelVersion/EvalSuite/Bundle tables. Add one native adapter and focused metadata/runner/API modules; keep model bytes and simulation in task containers and streaming in ebd.

**Tech Stack:** Existing FastAPI/SQLAlchemy, Vue/TypeScript, Rust/axum, native UFO/MJLab/MuJoCo image plus ONNX Runtime.

**Spec:** `docs/design/23-ufo-native-rpo-deployment.md`; selected target: local container inference, deployment simulations, downloadable simulation-only bundle.

## Global Constraints

- No Git commit/push; no new training; external repositories read-only.
- Native joint order, 375 observation + 256 latent → 23 actions, newest-first term-major history; 400/50 Hz.
- Jobs use offline approved images, frozen inputs, bounded resources, managed output roots; no platform Agent backdoors.
- Task execution success, evidence completeness, model qualification, and hardware readiness remain distinct.

## Review Focus

- Preview/source changes and symlink escapes must fail at both submit and execution.
- History reset and applied-action feedback must agree with native UFO, including after a failed episode.
- Missing ONNX Runtime, invalid numbers, partial receipt/media must never produce pass.
- Repeated collection/submission must not duplicate published versions; dependent jobs bind immutable parents.
- Downloads require authentication and verified managed paths, handle Range without buffering full files.

### Task 1: Native contract, metadata and trust-boundary tests

**Files:** create `backend/eb/ufo_native.py`, `backend/tests/unit/test_ufo_native.py`, native YAML contract; extend settings/bootstrap.

**Interfaces:** `content_hash(value)`, `read_json(path)`, `safe_file(root,relative)`, `validate_native_config(...)`, `suite_spec(backend,profile)`, `validate_receipt(...)` and constants for approved operations/card/interface/task.

- [x] Write and run failing tests for source layout, wrong joint/history/actions, unsafe paths, strict receipt/case completeness and rejected forged jobs.
- [x] Implement minimal shared metadata validation, fixed suites and new contract seed.
- [x] Run focused containerized pytest and contract checks.

### Task 2: Native task adapter, runner, API and collection

**Files:** create `adapters/ufo_native.py`, `runners/ufo_native_package.py`, `runners/ufo_native_sim.py`, `api/routes_ufo_native.py`, `orchestrator/ufo_native.py`; extend dispatch, submit guard, collector and app.

**Interfaces:** four approved operations (`ufo_native_import/export/eval/bundle`) consume frozen `native_plan`; each publishes a versioned receipt under `outputs/`. Collectors bind receipt to Job/Run, parent artifacts, suites and image.

- [x] Test rendered commands/mounts and direct Job rejection before implementation.
- [x] Use existing ebd scan for large input SHA-256; copy only inference dependencies and assets; verify model/normalizer/latent inside Job.
- [x] Export via native UFO helper; compare ONNX with actor; share one tested codec across simulation paths.
- [x] Run ONNX in MJLab and native MuJoCo; independently classify falls, metrics and videos; package immutable payload with reports.
- [x] Test collector idempotency, invalid source/receipt/media, and failed report behavior.

### Task 3: Streaming and workbench

**Files:** extend ebd with bounded managed-artifact streaming and tests; add native model workbench component reused by ModelList/ModelDetail/Bundles.

- [x] Add tests for Range and path constraints; implement streaming via existing axum body support.
- [x] Add authenticated preview/submit/status/evidence/download routes; link existing Run and Eval pages.
- [x] Add import, export, fixed-task simulation, results/media and bundle actions with visible errors and honest simulation-only state.
- [x] Run scoped Python/Rust/Node tests, frontend type check/build; preserve unrelated existing changes.

### Task 4: Deploy and actual-model evidence

**Files:** native task Dockerfile/Makefile target, deployment configuration, `docs/progress/2026-10-17-ufo-native-deployment.md`.

- [x] Build native task image with ONNX Runtime and source/dependency receipt; update platform via `make up`.
- [x] Import the provided source through API and run export Job; verify immutable registered versions and actual model inference.
- [x] Run the frozen smoke cases on each backend, check actual metrics/charts and every video through authenticated GET/Range.
- [x] Create/download/verify `.ebp` using frozen payload and report IDs; no hardware/signature claim.
- [x] Review design/semantics, execution/failure, and full user evidence in three rounds; record actual checks, IDs, limitations and remaining qualification separately.

## Execution record

2026-10-17: user selected local container inference, deployment simulations and downloadable package. No training or Git commits authorized. Existing dirty workspace retained. Plan uses inline execution; progress and runtime evidence will be recorded in the progress document.

2026-10-17 completed: raw `mv_ff284c75ef03498ca016` → export `mv_94d72b2354f040838aea` → complete MJLab/MuJoCo smoke reports → `bnd_9e68129a8faf4421a3f1`. Both simulation Jobs succeeded with 3/3 cases and complete media; model task outcomes are 0/6, diagnostic. Downloaded 41,534,607-byte archive and all 39 file hashes/source chain verified. Backend 87, frontend 36, ebd 13 checks passed; full records and actual resource ledger are in `docs/progress/2026-10-17-ufo-native-deployment.md`. Formal acceptance remains unverified and disabled in the workbench; no hardware release, new training, or Git commits.
