# LIBERO π0.5 runtime verification — 2026-09-09

This records implementation evidence, not a qualification or benchmark claim. External RLinf/OpenPI/LeRobot repositories were read-only; no git operations were used. GPU training was submitted through the platform and executed by ebd.

## Verified image and inputs

- Final validated image: `eb/rlinf:libero-pi05`, `sha256:0b13e47916c6867d91aab776b580c2043c77b21f71147ed376047e0038c326a2`. Earlier digests below are retained as chronological evidence for the fixes they introduced.
- Core versions: `rlinf-openpi==0.1.1`, `rlinf-libero==0.1.3`, `lerobot==0.4.3`, `transformers==4.53.2`; the fork's `transformers_replace` files are installed in the image.
- LIBERO assets: `RLinf/LIBERO-assets`, revision `3ba78404b48e8c70fa4ac9782d5aac8e5b46d55f`, 586 files, 422,320,979 bytes. Image `/opt/eb-runtime-assets.json` records each file's SHA-256 and size. Build downloads use a persistent cache, one thread and a request-rate delay.
- Offline PaliGemma tokenizer: 4,264,023 bytes, SHA-256 `8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6`; stored under `/opt/eb-openpi-cache/big_vision/`.
- The dedicated LIBERO image clears the inherited Isaac entrypoint. It does not launch Isaac or set `ACCEPT_EULA`.
- Parent: public `RLinf/RLinf-Pi05-LIBERO-SFT`, revision `45ccfcc4e28634f1576ebf78cab0fbe2fd82432d`. This parent is already SFT, not an untrained base. It has compatible OpenPI weights and `physical-intelligence/libero/norm_stats.json`.

## Runtime fixes and tests

The runner composes and resolves the actual Hydra configuration and persists the input receipt and resolved configuration. All nine LIBERO template configurations composed successfully. Evaluation has a separate runner because the upstream training entry does not honor evaluation-only intent; evaluation cannot publish a training checkpoint.

The owned SFT dataloader reads official LeRobot v3 directly with PyAV, maps both image columns/state/action/task into the OpenPI input schema, and creates a 10-action horizon. It uses the parent's normalization and pi05 transforms without applying a second action-delta conversion. The real dataset (273,465 frames) plus downloaded parent normalization passed an offline sample check: actions `(10,32)`, state `(32,)`, prompt `(200,)`, finite actions and both cameras. This does **not** establish a frozen train/held-out split or normalization provenance beyond the source parent.

Initial direct OpenPI source had a legacy LeRobot import reached indirectly by checkpoint loading. The pinned RLinf fork supplies a lazy import. Both that fork and the direct source have a separate SFT/FSDP incompatibility: joint VLM/expert forward directly calls child projections and MLPs, bypassing `GemmaDecoderLayer.forward`, while the default expert-only policy wraps `GemmaDecoderLayer`. A real first forward failed with zero-sized parameter storage. The fork has no `replace_gemma_decoder_layers` method, so changing the package alone does not solve this.

`libero_fsdp.py` configures RLinf's existing FSDP wrap policy around the children actually called, leaving the tied token embedding/lm_head in the root unit. It does not rewrite upstream layers. The meta audit finds 340 FSDP units, no mixed dtype/requires-grad units and no DecoderLayer wrapper. The actual GPU test below verifies the forward/backward behavior that a structural audit alone cannot prove. This custom policy currently applies to SFT; PPO uses the upstream policy and requires its own empirical verification.

- `backend/tests/unit/test_rlinf_entry.py`: 2 passed (publication completeness/latest numeric step, evaluation isolation).
- `backend/tests/runtime/libero_loader_smoke.py`: synthetic v3 loader passed.
- `backend/tests/runtime/libero_fsdp_smoke.py`: pinned-fork structural audit passed.
- `backend/tests/runtime/libero_env_smoke.py`: with `--network none`, `MUJOCO_GL=egl`, `PYOPENGL_PLATFORM=egl` and no GPU exposure, real LIBERO spatial task 0 reset, fixed initial state, one seven-dimensional zero action and both 64×64 RGB cameras passed. Reward 0, done false; this is functionality evidence only.

## Real SFT result

Platform run `run_273f2363c6d14d79af45` (`trn_36f8b09445584b158bb2`) used four RTX 4090 GPUs, microbatch 1, global batch 4, 96 GiB host-memory limit, two steps and save interval 2. The VLM was frozen and the action expert fine-tuned: 3,616,757,520 total parameters, 430,098,464 trainable. It is not full-parameter training.

Both optimizer steps completed. Observed step losses were 0.0079 and 0.0818; gradient norms 0.129 and 1.21. The two-step loop including saving took approximately 55 seconds. The container exited 0 and was not OOM-killed. Its actual output mount is `/var/lib/eb/runs/run_273f2363c6d14d79af45/output`.

The runner published `output/model` and atomically wrote `checkpoint-manifest.json` last: schema `eb/checkpoint/v1`, status complete, step 2, seven files totaling 8,526,579,376 bytes. The package contains `model_state_dict/full_weights.pt` and parent normalization/assets, with per-file SHA-256 and source-checkpoint/run provenance. Parent safetensors are not copied into the new model package.

`backend/tests/runtime/libero_handoff_smoke.py` checked this actual published checkpoint through read-only mmap against the PPO meta architecture: all 813 checkpoint entries match expected names/shapes, with no unexpected or mismatched entries. Only eight new `value_head.mlp.*` tensors are absent. Upstream `get_model` supports this package layout and loads with `strict=False`, allowing the critic to initialize separately. This is architecture/load-format evidence, not a completed PPO update.

A CPU parent-model load passed with maximum RSS 16,232,020 KiB (~15.48 GiB). A single post-training GPU snapshot showed about 7,123–7,128 MiB on GPUs 1–3; GPU 0 showed 12,990 MiB including unrelated existing usage. These are observations, not precise GPU peaks. Four concurrent initializations plus Ray overhead justify testing with more than 64 GiB host memory.

## PPO limits

Measured PPO architecture: 3,619,512,081 total parameters, 432,853,025 trainable. Full unsharded mixed-dtype weights occupy about 6.970 GiB. Frozen weights have no gradient/Adam state. The synchronous runner waits for rollout completion before actor training; configured offloading unloads rollout weights afterward and actor optimizer/weights around synchronization. An estimate that allocates optimizer state for all 3.6B parameters or simply adds all active role populations is not faithful to this recipe.

These measurements support a bounded platform smoke, not an empirical claim that PPO fits or converges. Actual PPO rollout/update, evaluation completion, pipeline handoff, frozen held-out splits and per-task qualification remain to be verified separately.

### Downstream startup findings and bounded-memory retry

The first pipeline PPO attempt failed because the code snapshot excluded the valid `rlinf/envs/venv` package. The snapshot filter was corrected in the data plane. The resulting immutable snapshot `58fec4588349d5adcf8141469f0270e4ce86e59c25d4d8d39ce94603c7800118` passed the same offline real environment reset test. Offline MP4 creation/finalization also passed without extra dependencies.

The next attempt (`run_61ae205ef16747e29428`) hit actual host memory exhaustion during initialization, before a PPO update. Host MemAvailable fell to 845,576 KiB with no swap; memory PSI full avg10 reached 97.98. Docker reported `OOMKilled=true` while other ranks remained alive; the cgroup reported `oom_kill=1` and current memory 78,365,597,696 bytes. The parent cancelled the run through the platform. EGL destructor exceptions were observed during this episode and are not evidence of an independent rendering root cause.

The runner now bounds model-loading concurrency without changing the model or PPO loss:

- LIBERO PPO uses native placement `actor:0-3`, `env,rollout:0` for a four-GPU request. This retains four training ranks and one rollout model replica instead of four.
- `LiberoPpoActor.model_provider_func` loads CPU models one rank at a time with distributed barriers. All ranks complete loading before entering upstream collective FSDP wrapping. Rollout/environment initialization and training otherwise retain the upstream implementation.
- Hydra requires deleting the prior combined-role dictionary before adding the new mapping; all nine actual template compositions passed with the final override generation.
- `libero_ppo_loading_smoke.py` passed with four real gloo ranks, non-overlapping model loads and verified weights on every rank. Seven form/runner unit tests passed.
- New complete image: `sha256:57d390a00dfba950573496d72e2223b3cbfd75409f7afc2f2766bdca28b87ea9`. Assets and pinned core versions are unchanged.

The 72 GiB retry `run_26ca5f6650ea4f9e90fe` completed all four sequential actor initializations and entered collective FSDP initialization at 19:06:22 UTC. The NCCL local-GPU-0 warnings did not indicate a barrier hang: each actor sees its distinct physical GPU as local device 0. Sampled job memory remained around 49 GiB, with no observed OOM; this is not an exact peak. Upstream model construction moves each completed model to CUDA before returning, so serialization bounds overlapping host initialization rather than retaining four completed CPU copies.

The first environment interaction then failed on an upstream missing import: `EnvWorker.compute_bootstrap_rewards` references `add_time_limit_bootstrap_rewards`, which already exists in `rlinf.utils.metric_utils`. The owned `LiberoEnvWorker` now binds that existing helper in its worker process; PPO and evaluation both use it. No reward algorithm or external source was rewritten. `libero_bootstrap_smoke.py` reproduces the original NameError, then verifies the real method adds the discounted value only at the first valid timeout and preserves the original rewards.

Evaluation config validation also reproduced a missing `rollout.model.model_type`: the upstream evaluation preset only specifies rollout path/precision. The runner now merges the actor architecture into rollout before resolving and recording its config, preserving explicit rollout overrides. `libero_eval_config_smoke.py` passed actual upstream validation for all four suites without creating training workers.

SFT now shares the same sequential model-loading helper; the four-rank gloo smoke passed after extracting it into `libero_fsdp.py`. Actual SFT under the new 72 GiB limit still requires a platform run. The complete image containing these fixes is `sha256:640d2e53ea2820eb69da4de9766f0876d2572993e77aea1df74035f54e893cd4`. Runner files are copied after stable dependency layers so code fixes no longer reinstall those packages.

A completed PPO optimizer update, evaluation result and full pipeline success remain unverified; initialization success and component tests do not establish them.

### First completed rollout and training-batch repair

The next platform retry `run_dd61a1646aec451ebd66` on image `640d2e53…` completed four actor initializations and one real rollout epoch in 62.02 seconds. The repaired timeout bootstrap path did not fail. Sampled memory reached approximately 49.9 GiB with no cgroup OOM events. Actor training then failed before its optimizer step: the generic batch shuffle indexed a one-row tensor using the twenty-step rollout permutation.

The actual upstream `EmbodiedTrajectoryBuilder` reproduces this failure on CPU. `EnvWorker.bootstrap_step` contributes one all-true `valid_step_mask`, while LIBERO does not provide optional per-step validity masks; the resulting field has shape `[1, B, 10]`. PPO does not consume this field in its loss, but its generic dictionary shuffle assumes every tensor is time-major. The owned actor receive boundary now removes only this all-true single bootstrap row. It leaves actual done/loss masks, rewards, and any nontrivial or full-length validity mask unchanged. `libero_batch_smoke.py` verifies the original failure and repaired upstream shuffle with preserved reward ordering. Complete image: `sha256:3083a1987b3749b022c23f4fcd9dfd131e9084f03f3d882a330b6d2436862e50`.

### Training-phase memory pressure and allocator cleanup

Run `run_e90c050a50ed45bea9ba` passed the repaired batch shuffle, completed its rollout in 61.85 seconds, ran actor training on all four GPUs and reached checkpoint saving at step 1. It nevertheless exited 255 and is not a healthy PPO result. Ray's memory monitor killed three required channel actors and the environment actor when usage reached 68.60–69.93 GiB of the 72 GiB cgroup limit (95% safety threshold). Kernel OOM counters stayed zero. Ray's peak table showed four actor training processes at 10.39–11.49 GiB RSS each, the offloaded rollout model at 11.01 GiB, and four environment children around 1.68 GiB each. The required-worker failure eventually propagated, so a log-only success override was not added.

An actual CPU load of the reused SFT `.pt` package measured freed-memory retention in glibc arenas. With the live model retained, RSS fell from 11,012,000 KiB to 8,356,084 KiB after `gc.collect()` and `malloc_trim(0)`, reclaiming 2.53 GiB. After releasing the CPU parameter storage to meta, the same trim reduced RSS from 3,264,432 KiB to 1,049,972 KiB (another 2.11 GiB). The bounded probe used no GPU, eight CPU quota, a 24 GiB limit, and finished in 80.8 seconds. This is allocator evidence, not a full GPU peak measurement.

The owned runner now trims freed CPU arenas after sequential initialization, actor parameter onload/offload and rollout model reload/offload. Live allocations and training math remain unchanged; Ray's safety threshold and the job memory budget were not relaxed. The shared four-rank gloo loader regression passed with cleanup enabled. Complete image: `sha256:51d26a688325acea6c3a06c2327ff8ac2b5defa4f26ca39ef753432032cf85fc`. A fresh SFT72 → two-epoch PPO → eval run is needed to verify the full lifecycle, including environment/channel survival across PPO epochs.

### SFT72 passed; separate checkpoint-save peak found

The fresh full-chain SFT `run_be000fb3359744a5a112` completed two optimizer steps and saving under 72 GiB, exited 0 and was not OOM-killed. Its two-step loop took 55 seconds; final displayed loss was 0.16 and gradient norm 1.66. The pipeline automatically bound its published model into PPO `run_a8de9323222b4e4cbc3f`.

PPO's first rollout completed in 61.13 seconds. Actor training remained around 52.8 GiB of **cgroup total** memory, with no Ray worker-kill records or memory-limit events at the sampled training times. One exact live sample was 56,666,263,552 bytes current, anon 33,704,026,112, file 22,206,205,952 and inactive_file 60,723,200 bytes; low/high/max/oom/oom_kill were all zero. These are not Docker stats cache-adjusted values. This host does not expose `memory.peak`; an exact total peak was not captured, and the sample does not establish saving headroom.

At checkpoint step 1 (20:16:37 UTC), memory rose beyond the limit. The platform recorded abort reason `oom` and cleaned up the remaining workers; this was not a manual cancellation. Persisted files contain all four native DCP shards plus `.metadata`, but no `full_weights.pt`, isolating the failed phase to the subsequent full-weight collection. Epoch 2 and independent evaluation did not run.

The owned actor now invokes native DCP saving with full weights disabled, waits for its temporary model/optimizer CPU dictionaries to leave scope, trims freed CPU arenas on every rank, and only then uses the same native full-state gather and rank-0 `torch.save`. Optimizer, scheduler and RNG saving remain in DCP; previous offload flags are restored. `libero_checkpoint_smoke.py` exercises two real Gloo ranks and native DCP/model/optimizer save/load, with cleanup ordered between phases. Image: `sha256:0b13e47916c6867d91aab776b580c2043c77b21f71147ed376047e0038c326a2`. Actual two-epoch PPO/save validation remains necessary.

### Two epochs executed; save memory margin remains insufficient at 72 GiB

Retry `run_ac55cc7dd32741ddbc24` completed its first rollout (62.15 seconds), training and step-1 save; the same environment/rollout workers then completed a second rollout (52.66 seconds) and second training epoch. Raw full-weight files exist for both steps, each 8,537,595,564 bytes. This establishes the cross-epoch call path, but **not a healthy completed run**: Ray killed three workers during the second save, including the rollout worker, and the subsequent execution-time collection propagated that failure. Exit was 255 and final model publication did not occur.

Five-second samples are in `/tmp/eb-ppo-ac55-memory.txt`; matching dirty/writeback samples are in `/tmp/eb-ppo-ac55-writeback.txt`. Before any Ray kill, the 20:40:46 UTC sample recorded cgroup total 77,307,359,232 bytes (72.00 GiB), total minus inactive-file 76,719,575,040 bytes (71.45 GiB), and anonymous plus shared memory 73,498,001,408 bytes (68.45 GiB). These are sampled quantities under an enforced limit, not an unconstrained exact peak. The cgroup `max` counter ended at 298,284, while kernel `oom`/`oom_kill` stayed zero. Ray recorded three kills; one reported 68.62/72.00 GiB usage against its 95% threshold. Saving grew anonymous memory by roughly 16 GiB; dirty/writeback pages were not the principal growth.

Shared-memory attribution showed actor processes holding most of the roughly 22–23.6 GB shared accounting; `/dev/shm` filesystem use was only 384 MiB and Ray's plasma process had negligible proportional shared memory. Large actor mappings included deleted `/dev/zero` regions. Their precise allocator ownership is not established, so no speculative allocator change was applied.

A guarded 76 GiB budget would put Ray's 95% threshold at 72.2 GiB, approximately 0.75 GiB above the largest sampled working-memory quantity before worker kills. This supports a bounded retry if the host admission guard still leaves its 4 GiB reserve, but does not prove sufficient peak headroom. The serializer and safety thresholds remain unchanged pending that actual validation.

### Healthy two-epoch PPO and complete step-2 publication at 76 GiB

The guarded retry `run_25bdb3d2dde942cb9409` (pipeline run `pr_daf7506388ed4e57b4ce`) used the same image `0b13e479…`, four actor GPUs, one rollout/environment placement, four environments, two epochs and 76 GiB host memory. It reused the successful SFT72 model. Both rollouts completed (61.40 and 51.89 seconds), both actor-training epochs completed, and both checkpoint saves completed. Final displayed actor gradient norm was 75.334 and critic value loss 0.171. The container **exited 0, OOMKilled=false**, with zero Ray worker-kill records in the five-second samples and no kernel OOM events. This verifies the environment and communication workers survived the inter-epoch transition and both saves for this bounded recipe.

The runner atomically published a complete step-2 manifest for `run_25bdb3d2dde942cb9409`. Full weights are 8,537,595,564 bytes, SHA-256 `62cda34b5b42bd239b7b782a999c2a723e9531e5943d7205e5ec49d75f36cf69`; the package includes the parent LIBERO norm statistics. Independent evaluation is a separate downstream check and this two-epoch smoke is not a convergence or held-out qualification claim.

`/tmp/eb-ppo-25bd-memory.txt` contains 192 usable five-second samples. Maximum sampled **cgroup total** was exactly 76 GiB, including file/cache/shared-memory accounting; maximum sampled total minus inactive-file was 70.324 GiB. Minimum sampled host MemAvailable was 12,084,444 KiB (11.525 GiB), above the unchanged 4 GiB admission reserve. The final sampled `memory.events.max` was 194,738, indicating substantial limit-driven reclaim; `oom=0`, `oom_kill=0` and Ray kill count stayed zero. The host still lacks `memory.peak`, so these are sampled measurements, not an unconstrained exact peak. Successful execution with reclaim does not establish comfortable headroom for larger batches, environment counts or different workloads.

### Independent evaluation handoff completed

The pipeline started evaluation only after the complete PPO step-2 manifest passed collection. Evaluation run `run_9cc87be2f4e749cab872` consumed `/var/lib/eb/runs/run_25bdb3d2dde942cb9409/output/model`, rather than the SFT parent or a guessed latest directory. It used one GPU, four environments and the evaluation-only runner; no actor or optimizer worker was created.

The run completed one Spatial rollout epoch in 38.12 seconds and exited 0. Four completed trajectories were reported. The normalized report is `ev_194156f0ce904c0ca492`: `success_once=1.0`, `success_at_end=1.0`, `return=1.0`, and Wilson 95% interval `[0.5101, 1.0]` for `n=4`. Its qualification remains `not_qualified`; four smoke trajectories do not satisfy the standard four-suite protocol, held-out evaluation, multi-seed comparison, sim2sim or sim2real release gates.

The platform validation pipeline is complete as an execution check: pinned LeRobot v3 input → π0.5 SFT → two-epoch online PPO → checkpoint-bound independent evaluation. The user will initiate any subsequent experiments through the UI; no additional training is required for platform delivery.
