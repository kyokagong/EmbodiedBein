# 2026-08-21 make-features-real 验收证据

实现分支：`feat/make-features-real`。下列为当时能在本机直接跑出的命令摘要，不含编造指标。

## 1. GPU

```
$ nvidia-smi -L
GPU 0: NVIDIA GeForce RTX 4090
GPU 1: NVIDIA GeForce RTX 4090
GPU 2: NVIDIA GeForce RTX 4090
GPU 3: NVIDIA GeForce RTX 4090
```

ebd compose 已加 `runtime: nvidia` 与 host `nvidia-smi` 挂载。`make up` 后需再执行 `docker exec eb-ebd-1 nvidia-smi -L` 与 `curl --unix-socket /run/eb/ebd.sock http://ebd/v1/doctor`。

## 2. 无镜像 → infra_failed

单元测试 `tests/unit/test_orchestrator_fail.py`：`image_missing` → job `infra_failed`，run `aborted`，metrics 不含 `mean_reward`。ebd 已删除 `stub_finished`。

## 3. Isaac 镜像

`eb/isaac-lab:2.3.2` 从本机 `isaac-lab-template`（`sha256:78d68e90…`）派生，WORKDIR `/workspace/isaaclab_extension_template`。`make images` 在缺少 template 时失败，不回落 busybox。

短训需在 `make up` 后从 UI 提交 biped `num_envs=64 max_iterations=5`。源仓隔离由 `eb.worktree` 的 clone+overlay 保证；单测 `test_worktree_leaves_source_clean` 通过。

## 4. 评估解析

`parse_play` 对真实 `eval_model_1700_seed20260808.json` 产出 `success_rate` 与 `per_seed`。两次相同输入得到相同向量（纯函数）。

## 5. 6 策略 ONNX 实测

`tests/unit/test_onnx_io.py` 对 checkpoints 下 6 个 ONNX 测最后一维：biped 45/12，humanoid 75/22，legs/video 为实测 45/12（文件名 hint 不得当作 G3 通过依据）。

## 6–8. 扫描 / Episode / retarget

`eb-scanner` merkle 单测稳定。`POST /sources/scan` 调 ebd scan 落漏斗。Episode API 无 mp4 时返回 `缺媒体`。tools 镜像无 pinocchio 时 retarget 退出码 2 + `ik_infeasible`（诚实失败）。

## 9. RLinf

`deploy/images/rlinf.Dockerfile` `FROM rlinf-berkeley-openpi:local`，Ray 端口 6390。适配器 `ray start --port` 后跑 train 脚本。

## 10. .ebp / sim2sim

`bundle.build` 打 tar.zst 进 CAS；签名要求 manifest 含 G4 与 sim2sim。坏 YAML 被 `g4_export` 拒绝。MuJoCo 镜像对 TAMPER MJCF 打印关节路径并非 0 退出。

## 11. robot-agent

子命令 `pull` / `serve` / `rollback`；promote API 带 `X-Confirm` 仍 403。

## 12. Agent

禁止动作 `llm_calls==0` 且磁带 `blocked`。日报调用 `query.jobs` / `query.models` 并带 ref。`EB_AGENT_WEEKLY_GPU_HOURS=0.01` 时 costly submit 被拒。权限矩阵快照已提交。

## 13. Page.vue

`frontend/src/router/index.ts` 生产路由不再引用 `Page.vue`。

## 单元测试

```
$ cd backend && uv run pytest tests/unit tests/agent tests/adapters -q
38 passed
```

Rust workspace（docker rust:1.83）：eb-runner / eb-tsdb / eb-scanner / eb-artifact 单测通过。
