# RoboTwin Platform Implementation Plan

> **For agentic workers:** Execute task-by-task with TDD and fresh verification. Do not commit; the user explicitly forbids git commits.

**Goal:** Implement the RoboTwin dataset acquisition/catalog UI and backend, register five native embodiments, and prepare six RLinf training recipes so the user can later launch supported training from the page.

**Architecture:** Reuse `Dataset`, `data.convert`, `TrainIn`, and `RlinfAdapter`. A bounded RoboTwin catalog produces immutable dataset registrations and task-container downloads; training recipes expose compatibility and readiness before submission. Training remains owned by RLinf.

**Tech Stack:** FastAPI, Pydantic v2, SQLAlchemy 2, Vue 3, TypeScript, RLinf/Hydra, Hugging Face Hub.

**Spec:** `docs/design/08-robotwin-integration.md`

## Global Constraints

- API is control plane; download/index work runs in a task container through `ebd`.
- External repositories are read-only and no task receives `docker.sock`.
- Five RoboTwin embodiments remain distinct; format conversion never changes embodiment identity.
- Every training job uses `stack=rlinf`; unavailable model workers are blocked before submission with a concrete reason.
- No model training is run in this implementation task.
- No git commit is created.

### Task 1: RoboTwin catalog and compatibility domain

**Files:**
- Create: `backend/eb/robotwin.py`
- Test: `backend/tests/unit/test_robotwin.py`

**Produces:** Fixed source revision, 50 tasks, five embodiments, official downloadable profiles, model readiness/compatibility records, and strict selection validation.

- [ ] Write failing tests for catalog completeness, safe HF patterns, and native embodiment binding.
- [ ] Implement immutable catalog constants and validation helpers.
- [ ] Run focused tests.

### Task 2: Dataset acquisition through existing jobs

**Files:**
- Modify: `backend/eb/api/routes_data.py`
- Modify: `backend/eb/adapters/data_convert.py`
- Create: `backend/eb/robotwin_worker.py`
- Modify: `deploy/images/tools.Dockerfile`
- Test: `backend/tests/unit/test_robotwin_data.py`

**Produces:** `GET /datasets/robotwin-catalog`, `POST /datasets/robotwin`, and existing `/datasets/{id}/validate` dispatching a bounded pinned download with progress/result manifest.

- [ ] Write failing route/adapter/worker tests.
- [ ] Add registration and reconciliation metadata without adding schema tables.
- [ ] Add worker download/extract/index behavior with allowlisted files, disk reserve, staging, and atomic publish.
- [ ] Run focused tests.

### Task 3: Embodiments and RLinf recipes

**Files:**
- Create: `contracts/ee_robotwin_aloha_joint14_v1.yaml`
- Modify: `backend/eb/bootstrap.py`
- Modify: `backend/eb/domain/rlinf_form.py`
- Modify: `backend/eb/api/routes_train.py`
- Modify: `backend/eb/adapters/rlinf.py`
- Test: `backend/tests/unit/test_robotwin_train.py`

**Produces:** Five catalog embodiments, the verified ALOHA draft environment, six model recipe entries, compatibility filtering, and pre-submit readiness gates. π0.5 uses the existing RLinf RoboTwin configuration; recipes lacking an implemented RLinf worker remain visible but blocked.

- [ ] Write failing catalog/template/preview tests.
- [ ] Seed distinct embodiments and the ALOHA contract.
- [ ] Add recipe metadata and compatibility checks to template listing and spec building.
- [ ] Render the ready π0.5 recipe through RLinf with immutable mounts.
- [ ] Run focused tests.

### Task 4: Dataset and training pages

**Files:**
- Modify: `frontend/src/views/DatasetNew.vue`
- Modify: `frontend/src/views/DatasetDetail.vue`
- Modify: `frontend/src/views/TrainNew.vue`
- Modify: `frontend/tests/train-workbench.test.mjs`
- Create: `frontend/tests/robotwin-workbench.test.mjs`

**Produces:** RoboTwin import form, task/scenario/embodiment/profile selection, download status and native compatibility display, and six-model training readiness/blocker display.

- [ ] Write failing component logic tests.
- [ ] Implement forms using existing routes and components.
- [ ] Run frontend tests and build.

### Task 5: Download the first training-ready dataset

**Files:** Runtime data only under the configured managed dataset root.

**Produces:** Pinned ALOHA LeRobot v2.1 data registered and downloaded for the π0.5 first vertical slice. The 1.53 TB full repository is not downloaded because only about 1.1 TB is free; other profiles remain page-selectable.

- [ ] Resolve and record the exact current HF revision.
- [ ] Build/start the platform task image if needed.
- [ ] Submit the same bounded acquisition used by the page.
- [ ] Verify manifest, size, source revision, embodiment, and page-visible status.

### Task 6: Multi-round review and full verification

- [ ] Review data identity/download failure paths and fix omissions.
- [ ] Review model/embodiment gating and fix false-positive readiness.
- [ ] Review UI lifecycle/idempotency and fix duplicate or stale actions.
- [ ] Run backend tests, frontend tests/build, `git diff --check`, and inspect the final diff without committing.
