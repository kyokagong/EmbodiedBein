# RoboTwin Five-Model Smoke Implementation Plan

> **For agentic workers:** Execute inline in dependency order. Do not create commits; the user explicitly prohibited git commits.

**Goal:** Make the five currently blocked RoboTwin SFT templates truthfully selectable and run one real RLinf smoke for each, with model files under `/data/kk/models`.

**Architecture:** Preserve the platform Job/Run path and RLinf ownership of forward, backward, optimizer and checkpoint. Apply a minimal patch to a pinned copy of the read-only RLinf tree when building model-specific runtime images. Derive template readiness from installed runtime/model/smoke evidence instead of permanently setting a boolean.

**Tech Stack:** FastAPI, SQLAlchemy, Vue 3, Docker, RLinf FSDP SFT, PyTorch, Hugging Face Hub.

**Spec:** `docs/design/08-robotwin-integration.md`; RLinf patch design: `docs/rlinf/01-robotwin-multimodel-sft.md`.

## Global Constraints

- Do not write `/data/kk/RLinf`; copy the pinned source into an image build stage and apply a platform-owned patch.
- Store models only below `/data/kk/models`, register their immutable manifest/hash, and mount them read-only into jobs.
- Do not replace LingBot-VLA-2 with V1, DM0.5 with DM0, G0 with G0.5, or XR1 with a padded 14D policy.
- RLinf owns the training loop, optimizer step and checkpoint.
- A template becomes ready only after its exact runtime and model pass the smoke contract.
- Do not perform git commits.

---

### Task 1: Readiness evidence and platform regression tests

**Files:**
- Modify: `backend/eb/domain/rlinf_form.py`
- Modify: `backend/eb/robotwin.py`
- Modify: `backend/eb/api/routes_train.py`
- Test: `backend/tests/unit/test_robotwin_train.py`
- Test: `frontend/tests/train-workbench.test.mjs`

**Interfaces:**
- Produces: structured template readiness with stable reason codes and one shared preview/submit check.

- [ ] Add failing tests proving model download alone does not make a template ready, and installed smoke evidence does.
- [ ] Add failing tests proving draft env/task and missing `incompatible_env_ids` are rejected before Job creation.
- [ ] Implement the smallest readiness resolver used by template listing, preview and submit.
- [ ] Display the structured reason in the existing training form without a new page.
- [ ] Run focused backend and frontend tests.

### Task 2: Pinned RLinf derived runtime

**Files:**
- Create: `deploy/patches/rlinf/manifest.json`
- Create: `deploy/patches/rlinf/robotwin-multimodel.patch`
- Modify: `deploy/images/rlinf.Dockerfile`
- Modify: `Makefile`
- Test: `backend/tests/unit/test_rlinf_adapter.py`

**Interfaces:**
- Consumes: read-only `/data/kk/RLinf` build context.
- Produces: model-specific image digests containing the same pinned RLinf patch.

- [ ] Add a failing image/adapter snapshot test for explicit runtime selection per model.
- [ ] Copy the RLinf source into the build image and apply the manifest-pinned patch.
- [ ] Keep π0.5 on the existing working image until the shared derived image passes its baseline smoke.
- [ ] Build and inspect the resulting image; record source and patch hashes.

### Task 3: Download and register exact models

**Files:**
- Modify: `backend/eb/api/routes_model.py`
- Modify: `backend/eb/model_worker.py`
- Test: `backend/tests/unit/test_model_download.py`

**Interfaces:**
- Produces: complete ModelVersion records whose paths are under `/data/kk/models`.

- [ ] Inventory the existing LingBot-VLA-2 download and create/verify its manifest.
- [ ] Resolve immutable official revisions for GR00T-N1.7, G0, DM0.5 and Xiaomi-Robotics-1.
- [ ] Download each exact artifact with resume support and sufficient-space checks.
- [ ] Verify required files and hashes, then register the exact architecture/version/embodiment profile.
- [ ] Reject incomplete or version-substituted directories.

### Task 4: LingBot-VLA-2 RLinf SFT adapter and smoke

**Patch targets:**
- `rlinf/config.py`
- `rlinf/models/__init__.py`
- `rlinf/workers/sft/fsdp_vla_sft_worker.py`
- `rlinf/models/embodiment/lingbotvla_v2/`
- `rlinf/data/datasets/lingbotvla_v2/`
- `examples/sft/config/model/lingbotvla_v2.yaml`
- `examples/sft/config/robotwin_sft_lingbotvla_v2.yaml`

- [ ] Add a failing contract test for exact V2 model/processor, real RoboTwin batch, loss and gradient.
- [ ] Implement the minimal V2 builder, loader and `ForwardType.SFT` wrapper around the official loss.
- [ ] Run one optimizer step, save/reload checkpoint, persist the receipt, and enable only this template.

### Task 5: GR00T-N1.7 RLinf SFT adapter and smoke

**Patch targets:**
- `rlinf/models/embodiment/gr00t/gr00t_n1d7/gr00t_action_model.py`
- `rlinf/models/embodiment/gr00t/gr00t_n1d7/robotwin_modality_config.py`
- `rlinf/data/datasets/gr00t/`
- `examples/sft/config/model/gr00t_n1d7.yaml`
- `examples/sft/config/robotwin_sft_gr00t_n1d7.yaml`

- [ ] Add a failing test showing current N1.7 rejects `ForwardType.SFT`.
- [ ] Add the RoboTwin ALOHA processor identity, loader and official supervised loss path.
- [ ] Run one optimizer step, save/reload checkpoint, persist the receipt, and enable only this template.

### Task 6: OpenDM DM0.5 RLinf SFT adapter and smoke

**Patch targets:**
- `rlinf/models/embodiment/opendm_dm05/`
- `rlinf/data/datasets/opendm_dm05/`
- `examples/sft/config/model/opendm_dm05.yaml`
- `examples/sft/config/robotwin_sft_dm05.yaml`

- [ ] Add a failing test proving `dexbotic_dm0` is not accepted for `opendm_dm05`.
- [ ] Wrap the official DM0.5 model/reader/loss while leaving optimization and checkpoint in RLinf.
- [ ] Run one optimizer step, save/reload checkpoint, persist the receipt, and enable only this template.

### Task 7: Galaxea G0 action profile, adapter and smoke

**Patch targets:**
- `rlinf/models/embodiment/galaxea_g0/`
- `rlinf/data/datasets/galaxea_g0/`
- `examples/sft/config/model/galaxea_g0.yaml`
- `examples/sft/config/robotwin_sft_g0.yaml`

- [ ] Pin the historical G0 code/weight pair and write a failing joint-order/action-profile round-trip test.
- [ ] Implement the evidence-backed RoboTwin mapping and official G0 action-expert loss wrapper.
- [ ] Run one optimizer step, save/reload checkpoint, persist the receipt, and enable only this template.

### Task 8: Xiaomi-Robotics-1 codec, adapter and smoke

**Patch targets:**
- `rlinf/models/embodiment/xiaomi_xr1/`
- `rlinf/data/datasets/xiaomi_xr1/`
- `examples/sft/config/model/xiaomi_xr1.yaml`
- `examples/sft/config/robotwin_sft_xr1.yaml`

- [ ] Add a failing numerical test for joint14/FK/TCP to official 60D relative action and inverse execution mapping.
- [ ] Implement the codec, official model/loss wrapper and chunk mask without zero-padding semantics.
- [ ] Run one optimizer step, save/reload checkpoint, persist the receipt, and enable only this template.

### Task 9: End-to-end platform verification

**Files:**
- Update: `docs/progress/2026-09-16-robotwin-platform.md`

- [ ] Rebuild and restart the platform using `make up` and build the required RLinf images.
- [ ] Verify all five templates expose exact readiness evidence and compatible registered models.
- [ ] Submit each smoke through `/api/v1/train`, not by bypassing the platform with a host script.
- [ ] Confirm five terminal successful Runs, finite loss, optimizer update and reloadable checkpoint receipts.
- [ ] Run focused tests, full backend/frontend checks, and record remaining performance limitations.

