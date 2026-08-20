# 2026-08-21 make-features-real 验收证据

实现分支：`feat/make-features-real`。下列为当时能在本机直接跑出的命令摘要，不含编造指标。

## 1. GPU / make up / doctor

`make up` 后 ebd 使用 `runtime: nvidia`。不要 bind-mount `libnvidia-ml.so.1`（与 nvidia-container-runtime hook 冲突，create 失败 `device or resource busy`）。只挂 `/usr/bin/nvidia-smi`。

```
$ docker exec eb-ebd-1 nvidia-smi -L
GPU 0: NVIDIA GeForce RTX 4090 (UUID: GPU-1b9c7093-3204-5084-3854-7ed540d2b1c5)
GPU 1: NVIDIA GeForce RTX 4090 (UUID: GPU-16478714-67b9-4a90-0c34-5d0e2a063322)
GPU 2: NVIDIA GeForce RTX 4090 (UUID: GPU-4e370b48-f7d4-89fa-546c-93aa18971480)
GPU 3: NVIDIA GeForce RTX 4090 (UUID: GPU-6758c666-e8ed-e972-401a-1af38348ff40)

$ curl --unix-socket /run/eb/ebd.sock http://ebd/v1/doctor
gpus.length == 4
memory_total_mb ≈ 24564
has_dri == true

$ curl -fsS http://127.0.0.1:8000/healthz
{"ok":true,"ebd":{"ok":true,"service":"ebd"}}
```

## 2. 无镜像 → infra_failed

单元测试 `tests/unit/test_orchestrator_fail.py`：`image_missing` → job `infra_failed`，run `aborted`，metrics 不含 `mean_reward`。ebd 已删除 `stub_finished`。

`ingest_tfevents` 不再把宿主历史 logs 写成 `succeeded` Job。bootstrap 会把 metrics 含 `stub` / `image missing` 的骨架 run 改成 `infra_failed`。

平台提交缺失镜像：`make smoke-job` 把 `image` 改成不存在的 tag 即可（`POST /api/v1/jobs`）。

## 3. Isaac 镜像与隔离

`eb/isaac-lab:2.3.2` 从本机 `isaac-lab-template`（`sha256:78d68e90…`）派生，WORKDIR `/workspace/isaaclab_extension_template`。`make images` 在缺少 template 时失败，不回落 busybox。

源仓隔离由 `eb.worktree` 的 `git clone --local` + overlay 保证；单测 `test_worktree_leaves_source_clean` 通过。本机 `/data/kk/Berkeley-Humanoid-Lite` 在平台训练前已有未提交改动（`.gitignore` / `scripts/` 等），不是本次 worktree 写入。

短训：UI `/train/submit` 默认 `num_envs=64 max_iterations=5`，或 `make smoke-job`。

## 4. 评估解析

`parse_play` 对真实 `eval_model_1700_seed20260808.json` 产出 `success_rate` 与 `per_seed`。两次相同输入得到相同向量（纯函数）。

## 5. 6 策略 ONNX 实测

`tests/unit/test_onnx_io.py` 对 checkpoints 下 6 个 ONNX 测最后一维：

| 文件 | act 实测 |
|---|---|
| policy_biped_25hz_a/b, policy_biped_50hz | 12 |
| policy_humanoid | 22 |
| policy_humanoid_legs | 12（文件名 hint 会指向 humanoid，G3 必须用实测维） |
| policy_video | 12 |

`refresh_measured_onnx_io` 在每次 seed 时覆盖库里的文件名 hint。

## 6. 扫描 / Episode / Range

```
$ curl --unix-socket /run/eb/ebd.sock -d '{"path":"/data/kk/berkeley_ego_data/manifests"}' .../v1/scan
files=17 bytes=1737803
content_hash=3d8205abb4c4023e61274bf49f05de78541c855932530ffdf452b1c11bd0e03e
kinds.ego_jsonl=9

$ curl --unix-socket /run/eb/ebd-http.sock -H 'Range: bytes=0-3' http://ebd/media/aa/testrange
HTTP/1.1 206 Partial Content
content-range: bytes 0-3/16
```

无 mp4 时 Episode 页显示「缺媒体」。tools 镜像无 pinocchio 时 retarget 退出码 2 + `ik_infeasible`。

## 7. RLinf / .ebp / robot-agent / Agent

- `deploy/images/rlinf.Dockerfile` `FROM rlinf-berkeley-openpi:local`，Ray 端口 6390。
- `bundle.build` 打 tar.zst；签名要求 G4+sim2sim。坏 YAML 被 `g4_export` 拒绝。
- robot-agent 子命令 `pull` / `serve` / `rollback`；promote 带 `X-Confirm` 仍 403。
- 禁止动作磁带 `llm_calls==0` 且 `blocked`。`EB_AGENT_WEEKLY_GPU_HOURS=0.01` 时 costly submit 被拒。

## 8. Page.vue

`frontend/src/router/index.ts` 生产路由不再引用 `Page.vue`。

## 单元测试

```
$ cd backend && uv run pytest tests/unit tests/agent tests/adapters -q
44 passed
```

Rust workspace（docker rust:1.83）：eb-runner / eb-tsdb / eb-scanner / eb-artifact 单测通过。

流水线：`pipeline_next` 单测覆盖 wait/submit/succeeded/failed；`POST /pipelines/{id}/start` 后调度器逐步提交 Job。
