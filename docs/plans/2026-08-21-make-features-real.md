# 全部功能真实可用 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 v0.1 骨架升级为设计验收意义上可使用的系统：训练、评估、数据、策略包、遥测、Agent 全部走真实容器/真实解析/真实 UI，禁止 stub 成功、假指标、JSON 模板页充当完成态。

**Architecture:** 控制面 Python 只做决策与元数据；数据面 `ebd` 按设计拆 crate 后负责 Docker 生命周期、worktree/overlay、tfevents→TSDB、CAS、扫描、媒体 Range。任务镜像从本机已验证容器派生（Isaac 权威环境是 `isaac-lab-template`），不进 compose。前端 31 页全部换成专用视图，数据来自 API/WS/TSDB，不再用 `Page.vue` 作为交付物。

**Tech Stack:** Python 3.11 FastAPI / SQLAlchemy 2 / Pydantic v2 / uv；Rust axum 0.8 workspace（eb-core, eb-probe, eb-runner, eb-collector, eb-tsdb, eb-scanner, eb-artifact, eb-cli, ebd）；Vue 3 + Vite + ECharts + Vue Flow；PostgreSQL 16；Redis 6380；本地 CAS `/var/lib/eb/cas`；Docker + NVIDIA runtime。

## Global Constraints

- 设计真源：`docs/design/`。冲突改代码不改设计。
- 外部仓只读：`/data/kk/Berkeley-Humanoid-Lite`、`/data/kk/RLinf`、`/data/kk/berkeley_ego_data`。
- 仅 `ebd` 持有 `docker.sock`；控制面禁止 `subprocess` 跑训练。
- Redis **6380**。任务容器不进 `make up`。
- ToolBus 永久禁止：`deploy.release`、`contract.create`、`contract.modify`、`gate.override`，以及 shell / 文件系统 / 出网 / SQL。
- 默认账号 `kk` / `embodied`；API Key `eb-dev-key`。
- 契约 ID 与维数不得改：humanoid 75/22/25、biped 45/12/25、wbc 85/22/50、fixedhand 10+img/10/20。
- Isaac 训练工作目录：`/workspace/isaaclab_extension_template`。权威容器名：`isaac-lab-template`。
- 本计划**禁止**下列完成态：`stub_finished` 标 `succeeded`、伪造 `Train/mean_reward`、busybox 任务镜像、`Page.vue` JSON 回显充当功能页、用文件名猜测 ONNX 维并宣称 G3 通过、剧本只返回一段中文而不调工具。

## 完成定义（DoD）

每一阶段结束必须能在本机演示，而不是「代码看起来像」。总 DoD：

1. `make up` 后 `/healthz` 且 `ebd doctor` 列出真实 GPU（本机 4×4090）。
2. UI 提交 `train.rl`（biped，短迭代）→ 容器真正跑 → Run 页曲线来自 TSDB → overlay 里 ONNX 入库 → 上游 `git status` 干净。
3. 镜像不存在时 Job 为 `infra_failed`，**永不** `succeeded`。
4. 同一冻结 EvalSuite 跑两次，逐 seed 成功率一致；6 个 checkpoints ONNX 上基线榜。
5. 漏斗页展示真实拒绝计数；Episode 预览能同步视频与动作；`data.retarget` 产出可过 G2 的轨迹或门禁明确失败原因（禁止假成功）。
6. 从一个 checkpoint 打出 `.ebp`，G4+sim2sim 不过不能签名；robot-agent 能拉包、缓存、上报真实协议点。
7. Agent 7 个剧本逐步调工具；禁止动作拦截并进审计；磁带 `llm_calls==0`；关 LLM 仍可用。
8. 31 个路由都有专用视图或设计系统组件，仓库内 `Page.vue` 不再被任何生产路由引用。

## 文件地图（开始前锁定）

新建：

- `rust/crates/eb-probe/` `eb-runner/` `eb-collector/` `eb-tsdb/` `eb-scanner/` `eb-artifact/` `eb-cli/`
- `backend/eb/api/jobs.py` `runs.py` `models.py` `datasets.py` `evals.py` `bundles.py` `agent.py` `ws.py`（从 `routes.py` 拆出）
- `backend/eb/adapters/data_scan.py` `data_convert.py` `data_retarget.py`
- `backend/eb/eval/parse_play.py`
- `backend/eb/onnx_io.py`
- `backend/eb/harness/tape.py` `budget.py`
- `backend/tests/agent/tapes/*.jsonl`
- `frontend/src/components/`（Table, Kpi, Chart, LogPane, GatePanel）
- `frontend/src/views/` 替换所有 Page 路由
- `docs/adr/0001-isaac-image.md` `0002-cas-layout.md`
- `deploy/images/isaac.Dockerfile`（真实 FROM）`rlinf.Dockerfile` `mujoco.Dockerfile` `tools.Dockerfile`

修改：

- `rust/Cargo.toml` workspace members
- `rust/crates/ebd/src/main.rs` 改为组装子 crate
- `deploy/docker-compose.yml` ebd `runtime: nvidia` 或 `gpus: all`
- `deploy/ebd.Dockerfile` 加入 nvidia-smi 可见性
- `backend/eb/orchestrator/service.py` 删除 stub 成功分支
- `backend/eb/ingest.py` 停止用文件名当 I/O；ONNX 走实测
- `frontend/src/router/index.ts` 去掉 `Page.vue`
- `frontend/src/api/client.ts` 浏览器走 Cookie，去掉写死 Bearer

---

### Task 1: 禁止 stub 成功 + 诚实失败

**Files:**
- Modify: `rust/crates/ebd/src/main.rs`（`submit_run` / `try_docker_run`）
- Modify: `backend/eb/orchestrator/service.py`
- Test: `backend/tests/unit/test_orchestrator_fail.py`（可 mock ebd 客户端）

**Interfaces:**
- Consumes: `EbdClient.submit_run` → JSON `{container_id, status, reason?}`
- Produces: `status` 仅允许 `started` | `image_missing` | `docker_error`；**删除** `stub_finished`

- [ ] **Step 1: 写失败测试**

```python
# backend/tests/unit/test_orchestrator_fail.py
import pytest
from eb.domain.fsm import JobStatus
from eb.orchestrator.service import Orchestrator

class FakeEbd:
    async def submit_run(self, payload):
        return {"container_id": None, "status": "image_missing", "reason": "image not present: eb/isaac-lab:2.3.2"}

@pytest.mark.asyncio
async def test_missing_image_is_infra_failed(db_session, queued_job):
    orch = Orchestrator(ebd=FakeEbd())
    run = await orch.admit_and_start(db_session, queued_job, gpu_ids=[0])
    assert queued_job.status == JobStatus.INFRA_FAILED.value
    assert run.status == "aborted"
    assert run.abort_reason == "image_missing"
    assert "mean_reward" not in (run.metrics_summary or {})
```

若尚无 `db_session` fixture，用内存 sqlite 不适用（JSONB/PG）。改为对 `admit_and_start` 抽一层纯函数 `apply_ebd_result(job, run, result: dict) -> None` 做单测，不碰 DB。

- [ ] **Step 2: 跑测试确认失败**

```bash
cd backend && uv run pytest tests/unit/test_orchestrator_fail.py -v
```

Expected: FAIL（`apply_ebd_result` 不存在，或仍把 stub 标 succeeded）

- [ ] **Step 3: 实现**

`apply_ebd_result`：

```python
def apply_ebd_result(job, run, result: dict) -> None:
    status = result.get("status")
    if status == "started":
        run.container_id = result["container_id"]
        run.status = RunStatus.RUNNING.value
        job.status = JobStatus.RUNNING.value
        return
    run.status = RunStatus.ABORTED.value
    run.abort_reason = status or "docker_error"
    job.status = JobStatus.INFRA_FAILED.value
    job.error = result.get("reason") or status
    run.metrics_summary = {}
```

Rust `submit_run`：镜像 `docker image inspect` 失败返回 `{"status":"image_missing",...}`，HTTP 200（业务错误，不是传输错误）。禁止写 `metrics`。

- [ ] **Step 4: 再跑测试 PASS；手动提交无镜像 Job，UI/API 为 `infra_failed`**
- [ ] **Step 5: Commit** `fix: never mark stub runs as succeeded`

---

### Task 2: ebd 看到真实 GPU

**Files:**
- Modify: `deploy/docker-compose.yml`（ebd service）
- Modify: `deploy/ebd.Dockerfile`（可选：不装驱动，挂宿主 `nvidia-smi`）
- Modify: `rust/crates/ebd/src/main.rs` `nvidia_smi_gpus`（随后迁到 `eb-probe`）
- Test: 本机 `docker exec eb-ebd-1 nvidia-smi -L`

**Interfaces:**
- Produces: `HostProfile.gpus: Vec<GpuInfo>` 非空；`GpuInfo.run_id` 由 runner 回填（Task 4）

- [ ] **Step 1: 在 ebd 服务加 NVIDIA**

```yaml
ebd:
  runtime: nvidia
  environment:
    NVIDIA_VISIBLE_DEVICES: all
    NVIDIA_DRIVER_CAPABILITIES: compute,utility,graphics
  devices:
    - /dev/dri:/dev/dri
```

若 compose 版本不认 `runtime: nvidia`，改用：

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

host network 的 ebd 仍要能执行 `/usr/bin/nvidia-smi`：把宿主二进制和必要 `.so` 挂进去，或在镜像装 `libnvidia-container` 工具。优先：

```yaml
volumes:
  - /usr/bin/nvidia-smi:/usr/bin/nvidia-smi:ro
  - /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1:ro
```

- [ ] **Step 2: `make up` 后验证**

```bash
docker exec eb-ebd-1 nvidia-smi -L
curl --unix-socket /run/eb/ebd.sock http://ebd/v1/doctor
```

Expected: 4 行 GPU；JSON `gpus.length == 4`，`memory_total_mb` 约为 24576。

- [ ] **Step 3: 调度器删除「无 GPU 则虚拟 4 卡」作为默认路径**

`backend/eb/scheduler/loop.py`：`doctor.gpus` 为空时**不要**再 `free = range(4)`。改为本 tick 跳过准入并打 error 日志。测试环境可用 `EB_FAKE_GPUS=0,1,2,3` 显式打开，生产默认关。

- [ ] **Step 4: Commit** `feat: expose host GPUs to ebd doctor`

---

### Task 3: 按设计拆 Rust crate（行为不变）

**Files:**
- Create: `rust/crates/eb-probe/src/lib.rs`（从 `doctor` / `nvidia_smi_gpus` 搬）
- Create: `rust/crates/eb-runner/src/lib.rs`（空壳 + `RunRequest` 类型）
- Create: `rust/crates/eb-tsdb/src/lib.rs`
- Create: `rust/crates/eb-collector/src/lib.rs`
- Create: `rust/crates/eb-scanner/src/lib.rs`
- Create: `rust/crates/eb-artifact/src/lib.rs`
- Create: `rust/crates/eb-cli/src/main.rs`（从 `ctl.rs` 搬）
- Modify: `rust/Cargo.toml` members
- Modify: `rust/crates/ebd` 只 `serve` 组装

**Interfaces:**

```rust
// eb-core 已有 GpuInfo, HostProfile, sha256_hex
// eb-probe
pub fn doctor() -> HostProfile;

// eb-runner
pub struct RunRequest {
    pub run_id: String,
    pub image: String,
    pub command: Vec<String>,
    pub env: Vec<(String, String)>,
    pub gpus: Vec<u32>,
    pub shm_gb: u32,
    pub devices: Vec<String>,
    pub mounts: Vec<Mount>, // src, dst, "ro"|"rw"
    pub workdir: String,
    pub network: Option<String>,
}
pub enum StartError { ImageMissing(String), Docker(String) }
pub async fn start(req: &RunRequest) -> Result<String, StartError>; // container_id
pub async fn cancel(run_id: &str) -> Result<(), String>;
pub fn ps() -> Vec<ContainerInfo>; // label eb.run_id

// eb-tsdb
pub fn append(root: &Path, run_id: &str, tag: &str, step: i64, value: f32) -> std::io::Result<()>;
pub fn query(root: &Path, run_id: &str, tag: &str) -> std::io::Result<Vec<(i64, f32)>>;

// eb-artifact
pub fn put(root: &Path, bytes: &[u8]) -> PathBuf; // cas/ab/abcd...
pub fn get(root: &Path, sha256: &str) -> Option<PathBuf>;
```

- [ ] **Step 1: 先加空 crate + `cargo test --workspace` 保持 ebd 行为**
- [ ] **Step 2: 逐个搬函数，每搬一个 `cargo test`**
- [ ] **Step 3: 提交 `Cargo.lock`**（`docker run rust:1.83-bookworm` 生成后拷出）
- [ ] **Step 4: Commit** `refactor: split ebd into workspace crates`

验收：`ebctl health` 与拆分前相同；无功能回归。

---

### Task 4: 真实 runner（GPU/shm/mount/日志泵/退出码）

**Files:**
- Modify: `rust/crates/eb-runner/src/lib.rs`
- Modify: `backend/eb/store/ebd.py`（payload 增加 mounts/env/shm/gpus）
- Modify: `backend/eb/adapters/base.py` `RenderedJob` 补 `mounts`、`outputs`、`metrics_glob`
- Modify: `backend/eb/orchestrator/service.py` 启动后进入 RUNNING，**另协程/tick 等退出**

**Interfaces:**
- Consumes: Task 3 `start` / `cancel` / `ps`
- Produces: 容器 label `eb.run_id=<id>`；日志文件 `/var/lib/eb/runs/<run_id>/console.log`；退出后 ebd `GET /v1/runs/{id}` 返回 `{status: exited, exit_code}`

- [ ] **Step 1: 单元测试命令拼装（Rust）**

```rust
#[test]
fn docker_args_include_gpu_and_shm() {
    let args = build_docker_args(&RunRequest {
        run_id: "run_x".into(),
        image: "eb/isaac-lab:2.3.2".into(),
        command: vec!["python".into(), "scripts/rsl_rl/train.py".into()],
        env: vec![("ACCEPT_EULA".into(), "Y".into())],
        gpus: vec![0],
        shm_gb: 2,
        devices: vec!["/dev/dri".into()],
        mounts: vec![Mount{src:"/wt".into(), dst:"/workspace/isaaclab_extension_template".into(), mode:"ro".into()}],
        workdir: "/workspace/isaaclab_extension_template".into(),
        network: None,
    });
    let s = args.join(" ");
    assert!(s.contains("--gpus"));
    assert!(s.contains("device=0"));
    assert!(s.contains("--shm-size"));
    assert!(s.contains("2g"));
    assert!(s.contains("--label"));
    assert!(s.contains("eb.run_id=run_x"));
}
```

- [ ] **Step 2: 实现 `build_docker_args` + `start`；镜像缺失映射 `StartError::ImageMissing`**
- [ ] **Step 3: 日志泵**：`docker logs -f` 写入 `console.log`，直到容器退出
- [ ] **Step 4: 控制面 Scheduler tick** 调 `ebd GET /v1/runs/{id}`，`exit_code==0` → COLLECTING_OUT，否则 FAILED/`abort_reason=exit_{code}`
- [ ] **Step 5: 重启 reconcile**：api 启动时问 `ebctl ps`，按 label 把 RUNNING 且容器已死的 Job 收口为 `infra_failed`
- [ ] **Step 6: Commit** `feat: real docker runner with logs and exit codes`

禁止：容器还在跑就标 succeeded。

---

### Task 5: 真实 Isaac 镜像（从 isaac-lab-template 派生）

**Files:**
- Modify: `deploy/images/isaac.Dockerfile`
- Create: `docs/adr/0001-isaac-image.md`
- Modify: `Makefile` `images` 目标

**事实：** BHL 权威环境是正在运行的容器 `isaac-lab-template`（Isaac Lab + RSL-RL，workdir `/workspace/isaaclab_extension_template`）。不要从 busybox 或凭空 NGC tag 开始。

- [ ] **Step 1: 记录 digest**

```bash
docker inspect isaac-lab-template --format '{{.Image}}'
docker inspect "$(docker inspect isaac-lab-template --format '{{.Image}}')" --format '{{.RepoDigests}} {{.Config.Cmd}} {{.Config.WorkingDir}}'
```

写入 ADR：镜像 ID、Isaac Lab 版本、RSL-RL 版本、EULA 环境变量。

- [ ] **Step 2: 打标签（不重新发明栈）**

```bash
docker tag "$(docker inspect isaac-lab-template --format '{{.Image}}')" eb/isaac-lab:2.3.2
```

Dockerfile 用于可重复构建时写成：

```dockerfile
# deploy/images/isaac.Dockerfile
# 构建机必须已有 isaac-lab-template 同源镜像。
ARG BASE=isaac-lab-template
FROM ${BASE}
ENV ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES
WORKDIR /workspace/isaaclab_extension_template
```

`make images`：若本机无 BASE 则失败并打印「先有 isaac-lab-template」，**禁止**回落到 busybox。

- [ ] **Step 3: 手工短训（不经过平台）**

```bash
docker run --rm --gpus '"device=0"' --shm-size 2g --device /dev/dri \
  -e ACCEPT_EULA=Y \
  -v /data/kk/Berkeley-Humanoid-Lite:/workspace/isaaclab_extension_template:ro \
  -w /workspace/isaaclab_extension_template \
  eb/isaac-lab:2.3.2 \
  python scripts/rsl_rl/train.py \
    --task Velocity-Berkeley-Humanoid-Lite-Biped-v0 \
    --headless --device cuda:0 --num_envs 64 --max_iterations 3 --seed 0 --logger tensorboard
```

Expected: 进程退出码 0 或至少出现 tfevents；**若因只读挂载写 logs 失败，这是 Task 6 要解决的，本步用一个临时 rw 目录验证 GPU/EGL 通。**

- [ ] **Step 4: Commit** `feat: pin real Isaac image as eb/isaac-lab:2.3.2`

---

### Task 6: worktree 快照 + overlay 写盘

**Files:**
- Create: `backend/eb/worktree.py`
- Modify: runner mounts
- Test: `backend/tests/unit/test_worktree.py`

**Interfaces:**

```python
def prepare_worktree(repo: Path, commit: str | None, run_id: str) -> Path:
    """Create git worktree under /var/lib/eb/worktrees/<run_id>, submodule update --init --recursive.
    Never writes inside the original repo."""

def overlay_mounts(worktree: Path, run_root: Path, writable_paths: list[str]) -> list[tuple[str,str,str]]:
    """Return (src, dst, mode). repo -> ro; each writable path bind from run_root/<path> rw."""
```

- [ ] **Step 1: 测试 worktree 不污染源仓**

```python
def test_worktree_leaves_source_clean(tmp_path, monkeypatch):
    # 用一个迷你 git repo 代替 BHL
    ...
    prepare_worktree(src, commit=None, run_id="run_t")
    assert (src / ".git").exists()
    # git -C src status --porcelain 为空
```

- [ ] **Step 2: 实现**：`git worktree add --detach /var/lib/eb/worktrees/<run_id> <commit-or-HEAD>`；`run_root=/var/lib/eb/runs/<run_id>` 预创建 `logs` `outputs` `configs`
- [ ] **Step 3: Isaac 可写路径**按设计：`logs`、`outputs`、`configs`（以及 BHL 实际写的 `logs/rsl_rl`）
- [ ] **Step 4: 验收**

```bash
git -C /data/kk/Berkeley-Humanoid-Lite status --porcelain
```

Expected: 空。一次平台训练后仍为空。

- [ ] **Step 5: Commit** `feat: isolate runs with git worktree and overlay mounts`

---

### Task 7: 平台跑通第一次真实 train.rl

**Files:** 串联 Task 1–6；修改 `frontend/src/views/Submit.vue` 默认 `max_iterations: 5`, `num_envs: 64` 作冒烟（生产默认仍 4096，表单可改）

- [ ] **Step 1: UI 提交 biped 短任务，带 `resources.gpus=1`**
- [ ] **Step 2: `docker ps --filter label=eb.run_id` 能看到容器**
- [ ] **Step 3: Job 最终 `succeeded` 或 `failed` 来自真实 exit_code，不是 stub**
- [ ] **Step 4: Commit**（若还需修适配器 argv）

冒烟失败时只修真实原因（EGL、shm、只读、EULA），禁止再加 stub。

---

### Task 8: TSDB + collector（Python 不再解析热路径 tfevents）

**Files:**
- `rust/crates/eb-tsdb` append/query（length-prefixed 或 JSONL 均可，选定一种写 ADR `0002`）
- `rust/crates/eb-collector`：watch `/var/lib/eb/runs/<id>/logs/**/events.out.tfevents.*`，解析后 `tsdb.append`
- `backend/eb/api` `GET /runs/{id}/metrics` 改 `EbdClient.tsdb_query`
- 删除或降级 `eb/ingest.py` 里给**新 Run**写 `series` 的路径；历史导入可一次性离线脚本，跑完归档

**Interfaces:**

```
GET http://ebd/v1/runs/{id}/metrics?tag=Train/mean_reward
→ {"tag":"...","points":[{"step":0,"value":1.2}, ...]}
```

- [ ] **Step 1: Rust 测试 roundtrip 100 点**
- [ ] **Step 2: collector 集成：短训过程中 `ebctl` 或 curl 能查到新增点**
- [ ] **Step 3: 控制面 metrics API 只转发 ebd**
- [ ] **Step 4: Commit** `feat: rust tsdb and tfevents collector`

---

### Task 9: Run 详情真实曲线与日志

**Files:**
- Modify: `frontend/src/views/RunDetail.vue`
- Create: `frontend/src/components/Chart.ts`（ECharts 折线）
- Create: `frontend/src/components/LogPane.vue`
- Modify: `backend/eb/api` WS `/ws/runs/{id}` 推 `{v:1,type:"metrics"|"log",...}`
- Modify: `frontend/package.json` 加 `echarts`

禁止继续只用 SVG sparkline 作为完成态。

- [ ] **Step 1: REST 拉全量 series 画图**
- [ ] **Step 2: 读 `GET /api/v1/runs/{id}/logs?tail=200`（api 读 `/var/lib/eb/runs/.../console.log` 或 ebd 代理）**
- [ ] **Step 3: WS 订阅增量；协议字段 `v,type,ts,seq,payload`**
- [ ] **Step 4: Commit** `feat: run page charts and live logs`

---

### Task 10: 产物认领 + ONNX 实测 G3

**Files:**
- `rust/crates/eb-artifact` CAS `cas/ab/<sha256>`
- `backend/eb/onnx_io.py` 用 `onnx` 库读 graph input/output 最后一维
- `backend/eb/ingest.py` 与认领共用 `onnx_io.read_signature(path) -> {obs, act}`
- `backend/eb/orchestrator` 在 COLLECTING_OUT 扫描 `outputs` glob：`**/exported/policy.onnx`、`**/model_*.pt`
- 模型列表：`kind=policy` 默认；`kind=checkpoint` 折叠

```python
# backend/tests/unit/test_onnx_io.py
def test_biped_onnx_dims():
    p = Path("/data/kk/Berkeley-Humanoid-Lite/checkpoints/policy_biped_25hz_a.onnx")
    sig = read_signature(p)
    assert sig["act"] == 12
    # obs 以文件实测为准，再与契约 45 比对；不匹配则 G3 fail，不许改契约
```

- [ ] **Step 1: 单测 6 个 checkpoints ONNX 的实测维**
- [ ] **Step 2: G3 只用实测维；去掉 infer_contract 作为通过依据（文件名仅作 UI hint）**
- [ ] **Step 3: 认领写入 Artifact.sha256 + ModelVersion.artifact_id + LineageEdge(run→model)**
- [ ] **Step 4: Commit** `feat: CAS claim and measured ONNX I/O`

---

### Task 11: 五元组 + 4 GPU 排队

**Files:**
- Modify: `backend/eb/domain/job.py` 增加只读计算属性或提交时写入 `spec.runtime.fingerprint`
- Modify: `backend/eb/orchestrator/service.py`
- Modify: `backend/eb/scheduler/loop.py` 用 doctor.gpus[].run_id 算 busy
- Modify: runner 启动时把占用 GPU 登记到 ebd 内存表，退出清除

五元组（设计 01.1）：`code_ref + config_hash + data_version + image_digest + seed`

```python
def fingerprint(spec: JobSpec, image_digest: str, commit: str) -> dict:
    return {
        "code_ref": commit,
        "config_hash": spec.canonical_hash(),
        "data_version": spec.inputs.datasets[0]["version"] if spec.inputs.datasets else None,
        "image_digest": image_digest,
        "seed": spec.params.get("seed"),
    }
```

`image_digest`：`docker image inspect --format '{{.Id}}'` 经 ebd 返回，禁止用浮动 tag 当指纹。

- [ ] **Step 1: 提交时 ebd 返回 digest，写入 `run.provenance` JSON**
- [ ] **Step 2: 单测 4 卡均 busy 时第 5 个 gpus=1 的 Job 保持 queued**
- [ ] **Step 3: Commit** `feat: pentuple provenance and real GPU admission`

---

### Task 12: 鉴权收口 + WebSocket 保活

**Files:**
- `frontend/src/api/client.ts`：浏览器 `credentials: include`，**删除常量 TOKEN**
- Showcase 继续调免鉴权 `GET /api/v1/shares/{token}`
- 登录页必须成功才能进 Shell；router guard
- `/ws/queue` 保持连接，tick 推队列快照

- [ ] **Step 1: 无 Cookie 访问 `/api/v1/contracts` → 401**
- [ ] **Step 2: 登录后 contracts 200**
- [ ] **Step 3: Commit** `fix: cookie session for web, no hardcoded API key`

---

### Task 13: P0 页面全部专用化（消灭生产路径 Page.vue）

**Files:** `frontend/src/views/*`、`frontend/src/components/*`、`frontend/src/router/index.ts`

本 Task 覆盖设计 09 章 **P0** 页：`/`、`/data`、`/data/:id`、`/data/:id/funnel`、`/data/:id/episodes/:ep`（Episode 可在 Task 18 做完预览，本 Task 至少真列表）、`/models`、`/models/:id`、`/contracts`、`/contracts/:id`、`/train/experiments`、`/train/runs/:id`、`/train/runs/compare`、`/train/submit`、`/train/queue`、`/eval/suites`、`/eval/reports/:id`、`/showcase/:token`。

每一页验收：打开后不是 `<pre>{{ JSON }}</pre>`。

- [ ] **Step 1: 抽 `EbTable` `EbKpi` `EbBadge`，样式跟 `docs/design/assets/eb/eb.css` 色板对齐（已有 CSS 变量）**
- [ ] **Step 2: 提交页：契约下拉、task 预填、G1 预检、equivalent-command、容量估算**
- [ ] **Step 3: 队列页：每 GPU 一条占用 + 队列表格 + WS**
- [ ] **Step 4: 总览：契约数、真实模型数、队列、doctor GPU**
- [ ] **Step 5: Showcase：快照由 `/shares` 创建时查询真实 counts + 一条 TSDB 缩略**
- [ ] **Step 6: Commit** `feat: replace P0 JSON pages with real views`

---

### Task 14: 评估闭环（play.py 真跑 + 解析）

**Files:**
- `backend/eb/eval/parse_play.py`
- Modify: `IsaacLabRslRlAdapter` EVAL_SIM 增加 `--metrics_output /mnt/run/outputs/metrics.json`
- `backend/eb/orchestrator` COLLECTING_OUT 调解析器写 `EvalReport`
- 种子套件已有 `suite_biped_locomotion_v2` seeds `[0..7]`

先导入仓库已有 json：

`/data/kk/Berkeley-Humanoid-Lite/logs/rsl_rl/wbc/.../eval_*.json`

解析器必须对真实文件产出 `success_rate` 与 per-seed 向量。再在线 `eval.sim`。

- [ ] **Step 1: 单测解析器对着一份现存 json**
- [ ] **Step 2: 短 play（num_envs=16, 1 seed）经平台出 EvalReport**
- [ ] **Step 3: 同一套件跑两次，per-seed 数组相等**
- [ ] **Step 4: 故意用错误 obs 维的 ONNX，G3 失败不得出报告成功**
- [ ] **Step 5: 排行榜 API 用真实 reports；UI 表 + baseline 标记 6 个 checkpoints 策略**
- [ ] **Step 6: Commit** `feat: real eval.sim reports and leaderboard`

失败视频墙：play 开启录像时把 mp4 进 CAS，报告页 `<video>`；无视频则显示「本 run 未开 camera」，不要假缩略图。

---

### Task 15: Run 对比 + 显著性

**Files:** `frontend/src/views/RunCompare.vue`；`GET /api/v1/runs/compare?ids=a,b`

- 参数 diff：对两个 JobSpec JSON 做扁平 key diff
- 曲线：同一 tag 叠加
- 显著性：对最后 N 个 `Train/mean_reward` 做 Welch t 或 CI 重叠；文案 **「差异不显著」** 必须在重叠时出现（设计 M2）

- [ ] **Step 1: 后端纯函数 `significant(a: list[float], b: list[float]) -> bool` 单测**
- [ ] **Step 2: 页面接两条真实 run id**
- [ ] **Step 3: Commit** `feat: run compare with significance label`

---

### Task 16: 数据六层 + Rust scanner

**Files:**
- 迁移：`dataset_layers` 或在 `dataset_versions.stats/funnel` 结构化为六层：source / raw / valid_episode / retargeted / canonical / slice
- `eb-scanner`：嗅探 ego jsonl / parquet / mp4；算 content_hash（目录 merkle：文件 sha256 排序拼接再 hash）
- `POST /sources/scan` 只调 ebd scan，Python 落库
- 漏斗：每一级 `n` + `rejected[]` `{reason, count}`
- 对 `berkeley_ego_data` **真扫**，禁止只展示种子 155000

- [ ] **Step 1: scanner 对小夹具目录 hash 稳定**
- [ ] **Step 2: 扫真实 ego 根，漏斗页数字来自扫描结果**
- [ ] **Step 3: `content_hash` 再扫一次不变**
- [ ] **Step 4: Commit** `feat: real dataset scan and funnel reasons`

---

### Task 17: Episode 预览（CAS Range + 动作曲线）

**Files:**
- ebd media：`Range` 请求返回 206
- `frontend/src/views/EpisodePreview.vue`：`<video>` + ECharts；时间轴共用 currentTime
- 样本索引 API：`GET /datasets/{id}/versions/{vid}/episodes/{ep}` → `{video_cas, action_series}`

没有视频文件时页面明确「缺媒体」，不要用纯色 canvas 冒充播放器。

- [ ] **Step 1: curl -H Range 测 ebd 206**
- [ ] **Step 2: 至少一条 ego 或评测录像可同步 scrub**
- [ ] **Step 3: Commit** `feat: episode preview with ranged media`

---

### Task 18: data.convert / slice / retarget 真实 Job

**Files:**
- `backend/eb/adapters/data_convert.py` `data_retarget.py`
- `deploy/images/tools.Dockerfile`：**禁止 busybox**。FROM 含 pinocchio/pin-pink 的环境，或基于 BHL 文档里已有的 teleop 容器。入口：`python -m eb_tools.retarget`
- 重定向核心：设计指向 `scripts/teleop/run_teleop.py` 的 Pinocchio IK。在 tools 镜像里调用该脚本（只读挂 BHL），输出轨迹到 overlay，G2 校验 action_dim=10、fps=20

若 IK 可行率过低：**Job `failed`**，漏斗记录拒绝原因 `ik_infeasible`，**不得**标 succeeded。这仍算功能「能正常使用」（门禁真实工作）。

- [ ] **Step 1: 手工 docker run tools 镜像对 1 条 ego episode 出关节轨迹**
- [ ] **Step 2: 平台 Job `data.retarget` 复用同一命令**
- [ ] **Step 3: `data.slice` 按查询写出新 dataset_version 与 content_hash**
- [ ] **Step 4: Commit** `feat: real data convert slice and retarget jobs`

---

### Task 19: RLinf 真实 SFT/RLFT

**Files:**
- `deploy/images/rlinf.Dockerfile`：从本机已有 RLinf 运行环境派生（检索 `docker ps` / 镜像名含 rlinf），`FROM` 钉 digest，Ray 端口 **6390+** 与 Redis 6380 隔离
- `backend/eb/adapters/rlinf.py`：训练前 `ray start --port ${ray_port}`；`cluster.component_placement` 与 `gpus` 一致
- 容量估算继续拒绝 OpenVLA 全参 4×24GB

- [ ] **Step 1: 手工容器内 `ray start --port 6390` + 最短 SFT/smoke**
- [ ] **Step 2: 平台提交 `train.sft`，TSDB 出现 RLinf 指标 tag**
- [ ] **Step 3: Commit** `feat: real RLinf adapter and image`

---

### Task 20: 策略包 .ebp + G4 + sim2sim

**Files:**
- `backend/eb/bundle/pack.py`：按设计 7.2 目录打 `tar.zst` → CAS
- `manifest.yaml` 字段与设计一致（contract_id、io_signature、gates_passed、provenance）
- G4：读 `policy_meta.yaml`/`deploy.yaml` 的 num_observations/num_actions/action_scale
- ONNX vs PyTorch：tools 镜像跑数值比对脚本，最大绝对误差超阈值 → 不得 `signed=True`
- `eval.sim2sim`：MuJoCo 镜像 **真实**（本机已有 mujoco python 环境则派生），故意改 overlay 里 MJCF 某关节 limit，门禁 path 含关节名
- 签名：HMAC 或 ed25519，密钥在 `deploy/secrets/bundle.key`（不进 git）；`X-Confirm: <bundle_id>` 仍要

- [ ] **Step 1: 从已认领 biped ONNX `bundle.build` 产出 .ebp，可 tar 解开看到 manifest**
- [ ] **Step 2: 坏 YAML 被 G4 拦**
- [ ] **Step 3: 坏 MJCF 被 sim2sim 拦并指出关节**
- [ ] **Step 4: Commit** `feat: real policy bundles with export and sim2sim gates`

---

### Task 21: robot-agent 真拉包与遥测

**Files:**
- `robot-agent/src/main.rs`：子命令 `pull` `serve` `rollback`
- 本地目录 `~/.eb/robot/<id>/current` `previous` `.ebp`
- `GET /api/v1/bundles/next` 返回已签名包的 CAS URL；agent 下载、校验 SIGNATURE、原子切换 symlink
- `POST /api/v1/deployments/{id}/telemetry` 写入 TSDB tag `robot.jitter_ms` `robot.temp_c` `robot.event`
- 前端 `/deploy/robots/:id`：放行步骤 + 曲线（真实点）。无真机时允许 **同一协议** 重放一段从真机或仿真采集的 udp/jsonl，但 ingest 路径必须与 live 相同。禁止前端 `Math.random()` 造温度。

- [ ] **Step 1: agent pull 一个 .ebp 到 current，断网后 rollback 切 previous**
- [ ] **Step 2: 上报 100 个点，机器人页能画出来**
- [ ] **Step 3: promote 仍 403；UI 按钮调 API 必须带人工 X-Confirm 且仍 403（设计：人按物理确认，平台 API 不自动放行）**
- [ ] **Step 4: Commit** `feat: robot-agent pull telemetry rollback`

---

### Task 22: 其余 P1/P2 页面

生产路由不得引用 `Page.vue`。本 Task：`/data/convert`、`/data/:id/slices`、`/models/:id/lineage`（Vue Flow）、`/assets`、`/assets/:id/viewer`（three.js 懒加载 URDF/MJCF）、`/train/pipelines`、`/eval/compare`、`/eval/failures`、`/eval/leaderboard`、`/deploy/bundles`、`/deploy/robots/:id`、`/agent/audit`、`/settings/*`。

流水线：最小 DAG 执行器——`Pipeline.spec.steps[]` 顺序提交 Job，下一步等上一步 SUCCEEDED。不要只存 JSON。

- [ ] **Step 1: grep 确认 router 无 `Page`**
- [ ] **Step 2: 每页用真实 API；空数据展示空态组件**
- [ ] **Step 3: Commit** `feat: remaining pages with real views`

---

### Task 23: Agent 真工具、真剧本、磁带、预算、LLM

**Files:**
- `backend/eb/agent/tools.py`：`query.jobs` 真查 DB；`capacity.estimate` 已真；`job.submit.train` 走 Orchestrator 且受审批
- `backend/eb/agent/playbooks.py`：每步 `await bus.call(...)`，禁止只拼中文
- `backend/eb/harness/tape.py`：jsonl `{ts, role, tool, args, result, llm_calls}`
- `backend/tests/agent/tapes/forbidden_release.jsonl` `daily_report.jsonl` `oom_suggest.jsonl`
- `backend/eb/harness/budget.py`：周 GPU·h，超限后 `call` 只允许 `effect=read`
- LLM：`backend/eb/agent/llm.py` 可配置 `EB_LLM_BASE_URL`；未配置则永不调用；配置后仅在规则+剧本都 miss 时调用
- MCP：stdio 或 HTTP 按注册表导出，权限=同一矩阵
- UI：工具轨迹、ref_id 链接到 `/train/runs/:id`、待审批、预算条

- [ ] **Step 1: `dispatch("写日报")` 实际 `query.jobs` + `query.models`，回复中的数字带 ref**
- [ ] **Step 2: 回放 forbidden tape，`llm_calls==0` 且审计有 blocked**
- [ ] **Step 3: 把 `EB_AGENT_WEEKLY_GPU_HOURS=0.01` 后 submit.train 被拒**
- [ ] **Step 4: 权限矩阵快照 `tests/agent/test_matrix_snapshot.py` 与 committed json 比对**
- [ ] **Step 5: Commit** `feat: agent toolbus playbooks tapes budget`

---

### Task 24: 插件真加载 + 搜索 + CI

**Files:**
- `backend/eb/pluginhost` 在 api lifespan `load("isaaclab", eb_plugin_isaaclab.setup)`
- 卸载 `dispose` 必须停 cron
- `GET /search?q=`：Postgres `ILIKE` 于 jobs.id/kind、model_versions.name、contracts.id
- `.github/workflows/ci.yml` 或本机 `make test`：ruff、pytest、适配器快照、`alembic upgrade head && downgrade -1`（compose 一次性）
- `deploy/api.Dockerfile` **停止编译 Rust**；不要拷贝 ebctl

- [ ] **Step 1: 卸载插件后对应 adapter 从 KIND_ADAPTER 消失（或不可选）**
- [ ] **Step 2: CI 绿**
- [ ] **Step 3: Commit** `feat: plugin host search and ci`

---

### Task 25: 全链路验收清单（人工，必须留证据）

在 `docs/progress/` 新增日期文档，贴命令输出摘要（不要假数据）：

1. `ebd doctor` 4 GPU  
2. 无镜像 Job → `infra_failed`  
3. 短 `train.rl` 成功，TSDB 有点，源仓 git clean，CAS 有 onnx  
4. 两次 `eval.sim` seed 一致  
5. 6 策略榜  
6. 漏斗来自扫描  
7. Episode 预览  
8. retarget 成功或真实 failed+原因  
9. RLinf smoke  
10. .ebp 签名；坏 MJCF 被拦  
11. robot-agent rollback 离线成功  
12. Agent 禁止动作 + 日报 ref + 磁带  
13. `rg Page.vue frontend/src/router` 无生产引用  

全部勾完才允许宣称「功能可正常使用」。

---

## 明确永远不做

- Kubernetes、Kafka、Celery、ClickHouse、多租户、OAuth
- 把 Isaac/RLinf 镜像打进 `compose up`
- 给平台 Agent 开 shell
- 用 stub/假 reward/JSON 页/busybox 冒充里程碑完成
- 为迁就实现修改四个契约 YAML 的维数与关节名

## 建议工期

按一人专职、本机已有 `isaac-lab-template` 与 4×4090：Task 1–7 约 1.5–2 周；8–13 约 1.5 周；14–15 约 1 周；16–19 约 2–3 周；20–21 约 2 周；22–24 约 1.5 周。合计约 10–12 周。RLinf 镜像与重定向 IK 是最大不确定项，只允许延长，不允许用 mock 填坑。
