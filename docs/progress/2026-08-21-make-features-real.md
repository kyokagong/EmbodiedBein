# 2026-08-21 make-features-real 验收证据

实现分支：`feat/make-features-real`（本轮 HEAD `5ca22bb`）。下列为本机实测命令摘要，不含编造指标。

## 1. GPU / make up / doctor

`make up` 后 ebd 使用 `runtime: nvidia`。不要 bind-mount `libnvidia-ml.so.1`（与 nvidia-container-runtime hook 冲突）。只挂 `/usr/bin/nvidia-smi`。

```
$ docker exec eb-ebd-1 nvidia-smi -L
GPU 0: NVIDIA GeForce RTX 4090 (UUID: GPU-1b9c7093-3204-5084-3854-7ed540d2b1c5)
GPU 1: NVIDIA GeForce RTX 4090 (UUID: GPU-16478714-67b9-4a90-0c34-5d0e2a063322)
GPU 2: NVIDIA GeForce RTX 4090 (UUID: GPU-4e370b48-f7d4-89fa-546c-93aa18971480)
GPU 3: NVIDIA GeForce RTX 4090 (UUID: GPU-6758c666-e8ed-e972-401a-1af38348ff40)

$ curl --unix-socket /run/eb/ebd.sock http://ebd/v1/doctor
gpus.length == 4
memory_total_mb == 24564
has_dri == true

$ curl -fsS http://127.0.0.1:8000/healthz
{"ok":true,"ebd":{"ok":true,"service":"ebd"}}
```

## 2. 无镜像 → infra_failed

`POST /api/v1/jobs` 且 `image=eb/does-not-exist:nope`：

```
job_c26cf3d4a570402eb0a8  infra_failed
error: image not present: eb/does-not-exist:nope
```

单元测试 `tests/unit/test_orchestrator_fail.py`：`image_missing` → job `infra_failed`，run `aborted`，metrics 不含 `mean_reward`。

## 3. 短 train.rl 成功 + TSDB + CAS ONNX + 源仓隔离

阻塞根因（已修）：`eb/isaac-lab:2.3.2` 的 site-packages 只有 `berkeley_humanoid_lite` 的 editable 安装；`berkeley_humanoid_lite_assets` 的 `.pth` 只在正在运行的 `isaac-lab-template` 可写层。Adapter 注入 `PYTHONPATH` 指向 worktree 的三个 `source/` 包。

```
job_66fab1d3de34448798d4  train.rl  succeeded
run_b866dad7edb4464180b7  finished  exit 0
容器: /isaac-sim/python.sh scripts/rsl_rl/train.py --task Velocity-Berkeley-Humanoid-Lite-Biped-v0
      --num_envs 64 --max_iterations 5
console: Learning iteration 0/5 … 4/5；无 Traceback
产物: logs/rsl_rl/biped/2026-08-20_21-31-37_biped-smoke/model_{0,4}.pt
      events.out.tfevents.*
CAS: art_2751e038… / art_52c180ac…  kind=checkpoint  run_id=run_b866…

$ curl .../api/v1/runs/run_b866dad7edb4464180b7/metrics?tag=Train/mean/reward
Train/mean/reward n=3895  first={step:0, value:-1.1057}  last={step:4, value:-1.0679}

export.policy（从该 run resume）:
job_c4b00363148d40d4aeda  succeeded
run_8e2c321f67dc4514a789  finished
ONNX: logs/.../exported/policy.onnx
实测维: obs=45 act=12（与 ec_bhl_biped_12dof_v1 一致，G3 通过）
CAS: art_edad3e7c… kind=onnx sha256=ff991488… size=252073
      /var/lib/eb/cas/ff/ff99148856c38ebe41da9e2c7776443c9e177f40ba5fbe62aef0eb9b94250230
ModelVersion mv_be1f7d41619540ad8d0e  io_signature={"act":12,"obs":45}  tags=["policy","run"]
LineageEdge run_8e2c… → mv_be1f7d…  relation=produced

$ git -C /data/kk/Berkeley-Humanoid-Lite status --porcelain | wc -l
45
```

源仓 45 行 porcelain 在平台训练前已存在（`.gitignore` / `scripts/` 等），worktree 是 `git clone --no-hardlinks`，训练未再写入源仓。

HEAD 上的 `scripts/rsl_rl/play.py` **没有** `--export_only` / `--seed` / `--metrics_output`（这些只在源仓未提交改动里）。`export.policy` 用已提交 CLI 跑 play.py，等 `policy.onnx` 出现后结束进程。

## 4. 评估解析（两次相同输入一致）

```
$ uv run python -c 'from eb.eval.parse_play import parse_play; ...'
file: logs/rsl_rl/wbc/.../eval_model_1700_seed20260808.json
success_rate 0.7916565345437843
两次 parse_play 同一文件: equal True
episodes == 344（test_parse_play）
```

在已提交的 play.py 上无法传 `--metrics_output`，因此本轮没有再开新的 Isaac `eval.sim` 容器；一致性由纯函数保证。

## 5. 6 策略 ONNX 实测榜

`tests/unit/test_onnx_io.py` + `model_versions`：

| 文件 / name | act 实测 | obs 实测 |
|---|---|---|
| policy_biped_25hz_a / b / policy_biped_50hz | 12 | 45 |
| policy_humanoid | 22 | 75 |
| policy_humanoid_legs | 12 | 45（文件名 hint 会指向 humanoid，G3 必须用实测维） |
| policy_video | 12 | 45 |
| 本轮短训 export `policy` | 12 | 45 |

## 6. 漏斗来自扫描

`ds_berkeley_ego` / `dsv_ego_raw`（扫 `/data/kk/berkeley_ego_data/manifests`）：

```
files=17 bytes=1737803
kinds: file=5 json=3 ego_jsonl=9
funnel.valid_episode.n=9
content_hash=3d8205abb4c4023e61274bf49f05de78541c855932530ffdf452b1c11bd0e03e
```

不是种子 155000。

## 7. Episode / Range

```
$ curl --unix-socket /run/eb/ebd-http.sock -H 'Range: bytes=0-3' http://ebd/media/aa/testrange
HTTP/1.1 206 Partial Content
content-range: bytes 0-3/16
```

无 mp4 时 Episode 页显示「缺媒体」。

## 8. retarget / sim2sim

tools 镜像无 pinocchio 时 `data.retarget` 退出码 2 + `ik_infeasible`（诚实失败）。mujoco TAMPER MJCF 退出码 3 + `torso/broken_joint`。`g4_export` 拒绝坏 YAML（`tests/unit/test_g4.py`）。

## 9. RLinf / 镜像

```
eb/isaac-lab:2.3.2     sha256:1e188ced65fa…
eb/rlinf:0.4.0-emb     sha256:4c1f19b7d49b…  (FROM rlinf-berkeley-openpi:local)
eb/mujoco:latest       sha256:4dea406d00c9…
eb/tools:latest        sha256:feb493dca1a3…
```

RLinf Adapter Ray 端口 6390。本轮未再跑完整 SFT（时长与 GPU 占用）；镜像与适配器已落地。

## 10. Agent

`tests/agent`：禁止动作磁带 `llm_calls==0` 且 `blocked`；日报剧本关 LLM 仍可 dispatch。`EB_AGENT_WEEKLY_GPU_HOURS=0.01` 时 costly submit 被拒。

## 11. Page.vue

```
$ rg Page.vue frontend/src/router
（无匹配）
```

`frontend/src/router/index.ts` 31 条生产路由均为专用视图，无 `Page.vue`。

## 12. 单元测试

```
$ cd backend && uv run pytest tests/unit tests/agent tests/adapters -q
47 passed
```

## 本轮为打通短训追加的提交

- `f9b104b` PYTHONPATH 指向 BHL `source/` 三包
- `ae21581` `export.policy` + 从源 run 拷贝 `model_*.pt`
- `35b252e` export 用 Isaac `n_envs` 估算，不用 VLA `P_train`
- `5ca22bb` 适配已提交的 `play.py` CLI，ONNX 落地后结束任务

## 仍存在的诚实缺口

- 源仓未提交的 `play.py` 才有 `--metrics_output` / `--export_only`；平台只使用 git HEAD，不读取脏工作区。
- GPU0 仍可能被更早的 `runheadless.sh` 残留容器占用（`run_31722f532a0840e3a416`）；短训走了空闲卡。
- 全量 44G ego 扫描曾把 pipeline 标为 `failed`（超时后取消，状态诚实）。
