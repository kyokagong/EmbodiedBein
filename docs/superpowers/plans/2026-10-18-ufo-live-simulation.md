# UFO Live Simulation Validation Implementation Plan

> Use executing-plans inline; delegate independent data-plane and frontend tasks. Preserve all existing changes. No Git commits.

**Goal:** Select a frozen model/environment, deploy its actual policy in a bounded Job, switch existing task z live and view physical behavior in the workbench.

**Architecture:** Reuse native deployment metadata/adapter/collector, serve.policy, ebd UDS and Three.js. Separate physics/control from browser rendering; save diagnostic traces, events, charts and video after the session.

**Tech Stack:** Python runners/FastAPI, Rust axum, Vue3/Three.js; existing dependencies.

**Spec:** `docs/design/24-ufo-live-simulation-validation.md`.

## Global Constraints

- No training, prompt regeneration, external repo writes, Git commits, new daemon or task ports.
- Model/env/interface/image/content hashes frozen; actor50Hz/physics400Hz, preserve history across task switches.
- Default300s/max600s; Job30min;4CPU/16GiB;CPU0GPU/MJLab1GPU; offline and managed outputs.
- Operator-only commands; strict task whitelist,15s lease, no platformAgent bypass; diagnostic only.

## Review Focus

- Task switch must change z without resetting physics/history; only reset clears them.
- Heartbeat must not overwrite an unconsumed command; ack must identify actual application.
- Unknown tasks/models/envs, stale preview, symlink/path escape, foreign Run and Agent commands fail closed.
- Browser failure/disconnect stops leaking resources; stale snapshots cannot appear live.
- Live scene must be frozen real meshes/poses; trace evidence must survive failures without fake qualification.

### Task 1: ebd fixed live control and state transport

Files: `rust/crates/ebd/src/native_live.rs`, `main.rs`; unit tests in module.

- [x] Red tests for strict native-plan/run checks, path/symlink bounds, bounded state, atomic commands and separate heartbeat.
- [x] Implement `/v1/runs/{id}/live/control`, `/heartbeat` on control UDS; `/v1/runs/{id}/live/{state|scene}` on media UDS.
- [x] Container cargo test and fmt; no Docker/model/physics execution by child.

### Task 2: native live worker and model/environment/API lifecycle

Files: `eb/ufo_live.py`, `runners/ufo_native_live.py`, `api/routes_ufo_live.py`; minimal hooks in native adapter/validation/collector/bootstrap/app/capacity.

- [x] Red tests for task/state/history, lease/pause/reset, scene metadata and frozen Job validation.
- [x] Add versioned interactive profiles, bounded plan/API submit/status/command/heartbeat/state/scene/assets; canonical direct Job validation and owner checks.
- [x] Implement50Hz policy loop with existing backend+deliveredPolicy, atomic latest state, applied command events, chunked traces,5fps post-session media and receipt.
- [x] Register diagnostic Artifact only with verified identity/files; preserve failed evidence honestly.
- [x] Focused containerized pytest and old native regression.

### Task 3: workbench live viewer and controls

Files: `views/SimulationValidation.vue`, `components/NativeSimulationStage.vue`, `tests/ufo-live.test.mjs`; router/Shell/model and bundle entry links.

- [x] Red behavior tests for preview/submit, pending command ack, stale stream, clean unmount/heartbeat and terminal state.
- [x] Reuse Three STL/OrbitControls, authenticated frozen assets, real body poses, model/env/task controls and metadata; avoid server-frame rendering in the live loop.
- [x] Node tests, vue-tsc and Vite container build.

### Task 4: actual-model validation and delivery

- [x] Build task image and `make up`; check actual idle resources and source/model hashes.
- [x] Real Job with standing→0.3→0.7, pause/reset/resume/stop and lease expiry; test both compatible environments within a bounded verification budget.
- [x] Verify live page/state cadence and command response, all output hash/video/chart access, unauthorized/bad command rejection and old deployment regression.
- [x] Independent three-round review; progress document with actual Jobs, model/task outcomes, rates/latency and limitations.

## Execution ledger

- Task 1 complete: strict UDS control/lease/snapshots, bounded atomic files and session-wide command IDs; ebd 21 checks. Review fixed shared-storage-lock heartbeat starvation with a dedicated short live lock.
- Task 2 complete: frozen profiles/plans, operator-owned APIs, deliveredPolicy worker, real qpos scene, traces/events/video and diagnostic collection; 88 scoped backend checks. Review fixed inaccessible interrupted traces with partial Artifact registration and download allowlists.
- Task 3 complete: real STL stage, task/request ACK separation, lifecycle/retry/stale handling, independent reset, navigation; 51 Node checks, vue-tsc and Vite. Real software-WebGL browser verified every control, 320 snapshots, 67 distinct body states and no JS exceptions.
- Task 4 complete: 7 bounded platform Jobs, one GPU session; native MJLab and CPU loop, source hash preservation, lease at15.017s, cancelled NPZ downloads,18 full evidence file hashes/shape/step continuity, media Range and old bundle access. Actual IDs and limits recorded in `docs/progress/2026-10-18-ufo-live-simulation.md`.
- Ruling: preserve this shared dirty workspace and leave all code uncommitted, following the user's explicit requirement; no branch/merge/PR operations.
- Ruling: browser renders frozen meshes from integrated qpos; post-session recording is separate, because synchronous software rendering would slow the physical loop. WebGL failure is explicit rather than an invented pose.
- Ruling: CPU recording uses320px to reduce post-session rendering cost; MJLab retains480px. Page resolution follows its viewport. Long600s/30min completion remains an unverified boundary, not an acceptance claim.
- Ruling: local UDS uses literal plain HTTP, so skip unused TLS-context construction; measured construction31.9ms→0.36ms. Renderer redraws only changed states/camera/size and polling accounts for elapsed time; software-browser observed1.90→5.13Hz, configured cap10Hz.
- No model retraining, prompt generation, external source writes or Git commits. No formal EvalReport/pass or hardware qualification created.

## HTTP follow-up

- Reproduced the user's exact `http://1.94.0.106:31800` context: randomUUID unavailable, getRandomValues available. Previous localhost verification missed this case.
- Fixed all three live-page ID creation paths with a secure-random HTTP fallback. Added regressions for preview/control/unmount without randomUUID and a single environment start followed by same-Run behavior changes; RED→GREEN, full frontend53 + vue-tsc + Vite passed.
- User-facing flow now starts the environment directly with standing default; behavior selection exists only in the running simulation controls. Frozen preview/idempotent submission remains internal and inspection is optional.
- Actual HTTP browser Run `run_45048da064eb4ddcba2d`: one session submission, three task changes, episode0 throughout, qpos/history unchanged at application,170 snapshots and zero JS exceptions. Stopped and collected complete recording. Deployed with make up; no Git commits.
