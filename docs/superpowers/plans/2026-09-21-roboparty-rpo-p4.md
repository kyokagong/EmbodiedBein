# RoboParty RPO V2 WBC P0–P4 implementation plan

**Goal:** Add RPO V2 as a managed 23-DoF embodiment, train its walk-plus-upper-body policy through RLinf, and produce verifiable ONNX/Isaac/MuJoCo evaluation artifacts.

**Constraints:** Keep external repositories read-only, store managed robot assets under `/data/kk/assets`, models under `/data/kk/models`, execute jobs through ebd containers, and do not create Git commits.

## Scope and checks

1. **P0 — identity and contract**
   - Add `ee_rpo_v2_wbc_23dof_v1` with actor/critic/history dimensions and 23-joint codec.
   - Add a small asset verifier that records pinned source revisions, hashes URDF/MJCF/meshes, checks joint order, and records hardware mismatches as blockers.
   - Seed RPO embodiment, asset row, environment, and WBC task without claiming hardware verification.

2. **P1/P3 — RLinf runtime**
   - Add a managed RPO extension copied into a dedicated RLinf image; do not patch `/data/kk/RLinf` or downloaded sources.
   - Add the RLinf PPO recipe, environment bridge, upper-body reference generator, curriculum/reward settings, terminal-observation semantics, asymmetric critic, normalization, checkpoint resume, and smoke receipts.
   - Validate the pure observation/history/terminal logic outside Isaac; keep GPU smoke as an explicit runtime check.

3. **P2 — platform path**
   - Add the RPO template to the existing train template/render/submit path.
   - Mount the frozen asset bundle read-only and write run outputs/model checkpoints only to platform roots.
   - Reuse existing Job/Run/collector/model lineage flow, adding only RPO package metadata.

4. **P4 — export and evaluation**
   - Add RPO export and Isaac/MuJoCo evaluation modes to the existing adapter path.
   - Require measured ONNX I/O, golden numerical traces, per-case completion accounting, and a gate report before qualification.

5. **Verification**
   - Run focused tests after each slice, then backend unit tests, contract validation, Dockerfile/config static checks, and any available container smoke.
   - Report GPU/image/hardware checks as pending if the required local runtime or verified V2 hardware profile is absent.

