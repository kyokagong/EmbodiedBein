# Python 文件能力下沉 Rust 实施方案

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> 实施结果见 [进度与验证](../progress/2026-10-28-python-rust-data-plane-migration.md)。按用户要求不执行 Git commit/push。

**Goal:** 分阶段将 Python 控制面中的批量文件校验、产物复制与发布、文件扫描、媒体读取等机械操作交给 Rust，使 Python 保留业务决策、元数据和算法语义。

**Architecture:** 复用现有 `API → HTTP-over-UDS → ebd` 链路，在 `eb-artifact`、`eb-scanner`、`eb-probe`、`eb-tsdb` 内补齐能力，由 ebd 装配内部接口。第一阶段只接通批量 manifest 校验和单文件 CAS 发布，不新增常驻服务，不改变 Job/Run 生命周期。后续模块独立交付，不以一次大规模重写作为前置条件。

**Tech Stack:** Python 3.11、FastAPI、SQLAlchemy 2、httpx；Rust 2021、axum 0.8、Tokio、SHA-256；现有 Docker Compose、pytest、cargo test。任务镜像中的 Python 版本与科学计算依赖保持各 recipe 的既定要求。

**Spec:** [系统架构](../design/02-system-architecture.md)、[数据与采集](../design/03-data-collection.md)、[模型与训练](../design/04-model-training.md)、[评估与交付](../design/05-evaluation-delivery.md)、[Agent 与安全](../design/06-agent-security.md)。这些文件仍是设计真源；本文只规定实现顺序，不修改设计约束。

## Global Constraints

- Python 控制面负责鉴权、领域校验、元数据、准入、编排、adapter、门禁、Agent 和页面 API；领域模块最终应保持纯逻辑。
- Rust 负责文件扫描、hash、CAS、物理产物校验与发布、宿主事实探测，以及日志、指标、媒体和大文件通道。
- 部署仍只有现有 ebd/ebctl 数据面二进制，不新增常驻微服务；只有 ebd 持有 `docker.sock`。
- 执行与测试容器化；平台部署入口为 `make up`。任务仍经 Job/Run/ebd，任务容器不挂载 ebd UDS 或 Docker socket。
- `/data/kk/Berkeley-Humanoid-Lite`、`/data/kk/RLinf`、`/data/kk/berkeley_ego_data` 保持只读；读白名单与写根分别校验。
- 已发布接口、数据/模型版本、Suite 与 Bundle 不原地修改；数据库只引用已完成发布的文件。
- 文件完整性、Job 执行完成、模型评估合格是不同结果，不能互相替代。
- JobSpec/plan/兼容性指纹、G1–G5、恢复资格、训练真实更新数和验收判据继续由 Python 定义；不借本次迁移改写历史 hash 协议。
- 不增加平台 Agent 的原始文件、shell、网络、SQL 或门禁豁免能力；内部 ebd 接口不直接注册到 ToolBus。
- 第一阶段不修改公开 API、数据库 schema、前端请求格式或任务镜像；后续任务确需这些变更时单独列明。
- 无实测不承诺 Rust 带来固定倍数提速；收益同时评估 API 响应、重复 IO、内存占用和职责一致性。

## Review Focus

以下五类情况分别由后文指定任务锁定行为，不能只验证成功路径：

1. 同长度篡改、校验期间替换文件、符号链接换入：必须拒绝或始终操作已安全打开且内容符合预期的文件；见任务 1、2。
2. 媒体与 Replay 的体积上限、零字节包成员不同：必须保持各调用者既有限制，不因统一接口放宽公共下载；见任务 1、3、6。
3. 发布失败、重复发布、PG 登记失败、与删除并发：不能覆盖已有对象、提前成功或自动删除共享 CAS；见任务 2、3。
4. 大文件校验阻塞心跳、日志或实时画面：文件 IO 必须离开异步执行线程，不能长时间持有全局 storage lock；见任务 1、2、9。
5. 历史模型、旧 Run 路径、冻结计划及断连下载：保持既有引用与业务判定，拒绝越界并释放流连接；见任务 3、5、6、9。

## 1. 源码依据与范围

本方案基于 2026-10-28 的静态调用链检查。实现时先重新检查工作区、函数调用者和镜像复制清单，不能依赖本文行号判断代码未变化。

| 能力 | 当前入口与问题 | 本方案安排 |
|---|---|---|
| 文件完整性 | [UFO verify_index](../../backend/eb/orchestrator/ufo_native.py)、[P-BFM verify_output_index](../../backend/eb/pbfm_evaluation.py) 逐文件调用 ebd scan；[checkpoints](../../backend/eb/checkpoints.py) 部分直接 hash | 第一阶段批量校验；checkpoint 业务规则不迁 |
| 模型/权重发布 | [collector](../../backend/eb/orchestrator/collect.py) 使用 `read_bytes/write_bytes` 复制 ONNX/checkpoint 到 CAS | 第一阶段单文件流式发布 |
| 恢复与目录复制 | [worktree](../../backend/eb/worktree.py) 直接复制旧 RSL 恢复输入；[P-BFM checkpoint](../../backend/eb/pbfm_training_checkpoint.py) 的目录发布实际位于任务运行链路 | 第二阶段先处理控制面恢复复制，原生运行时目录发布后置 |
| 扫描与磁盘 | [domain.scan](../../backend/eb/domain/scan.py)、[workbench.dir_size](../../backend/eb/workbench.py)、[domain.disk](../../backend/eb/domain/disk.py)、[dataset tree](../../backend/eb/api/routes_data.py)、[legacy ingest](../../backend/eb/ingest.py) 直接遍历/探测 | 第二阶段下沉事实采集，保留 Python 解释与登记 |
| 媒体 | [评估视频](../../backend/eb/api/routes.py)、[数据集媒体](../../backend/eb/api/routes_data.py)、[本体资产](../../backend/eb/api/embodiment_routes.py) 分别使用 FileResponse | 第二阶段统一通道；现有 FileResponse 已分块，不能宣称此前整视频入内存 |
| Bundle | [pack.py](../../backend/eb/bundle/pack.py) 在 API 内存中组装 tar、压缩；同时包含 ONNX/门禁逻辑 | 后置独立任务，只迁机械打包；模型检查保留 Python 工具任务 |
| 证据索引/遥测 | [评估进度](../../backend/eb/api/routes_eval.py) 重读 JSONL；[遥测](../../backend/eb/api/routes.py) 直接写旧 TSDB | 测量后按需实施，不与第一阶段捆绑 |

现有 Rust 也有需要补齐的地方：`eb-artifact::put_file()` 整文件读入内存且尚无生产调用；`eb-scanner::scan()` 不是 manifest 校验协议；新 `run_output` 已支持安全打开和流式 Range，旧 `/media` 仍整文件读取。复用代码前先核对具体行为。

### 方案取舍

| 方案 | 代价与收益 | 结论 |
|---|---|---|
| 仅把 Python 整文件复制改成流式 | 改动最小，可先降低内存；文件安全、发布与重复 RPC 仍分散 | 可作为独立紧急修复，不作为本方案最终边界 |
| 按能力逐步迁入现有 Rust 模块 | 每项可独立验证，保留业务协议和现有服务形态 | 本方案采用 |
| 一次迁移 collector、调度器、门禁和算法运行时 | 改动横跨事务、协议和训练依赖，缺少收益证据 | 不采用 |

### 阶段与出口

- **阶段 A：任务 1–3。** 批量校验和单文件发布接入真实调用者；可独立验收，不等后续任务。
- **阶段 B：任务 4–7。** 扫描、恢复文件准备、媒体、Bundle；每项单独验收，未完成项明确保留。
- **阶段 C：任务 8。** 证据查询与遥测，只有测量或实际使用证明需要时启动。
- **任务 9** 是各阶段共用的验证与交付出口，不要求为了文件改造重新训练模型。

## 2. 第一阶段内部接口

以下内部接口已实现，实际验证与保留范围见进度文档。只经控制 UDS 调用，不暴露公网，不接受平台 Agent 的任意文件操作。

### 2.1 受管根与文件记录

`ArtifactRoot = {namespace: "run" | "model_run", run_id: str, path: str}`。

- `run` 映射为 `${EB_DATA_ROOT}/runs/<run_id>/<path>`；`model_run` 映射为 `${EB_MODELS_ROOT}/runs/<run_id>/<path>`。
- `run_id` 复用仓库 ID 校验；`path` 允许空字符串表示根，非空时必须为规范相对路径。拒绝绝对路径、反斜杠、NUL、空路径段、`.`、`..`、符号链接和特殊文件。
- ebd 根据 namespace 自行解析根；不接受任意宿主绝对源路径或任意目标路径。第一阶段不覆盖外部仓导入、独立 `mdl_*` 或数据集根。
- `ArtifactFile = {path: str, bytes: int, sha256: str}`；SHA 为 64 位小写十六进制，bytes 必须为非负整数，拒绝布尔值冒充整数。
- 文件路径长度上限 1024 个 UTF-8 字节；集合内规范路径唯一。共享定义放在 `eb-artifact`，Python 仍在信任边界检查响应。

Rust 共享类型固定为 `VerifyLimits { max_files: usize, max_file_bytes: u64 }`、`FileExpectation { bytes: u64, sha256: Option<String> }`、`Verification { file_count: usize, total_bytes: u64 }`、`Publication { sha256: String, size_bytes: u64, storage: PathBuf }`；`ArtifactError` 区分参数、禁止、缺失、冲突、超限、空间不足和其他 IO 错误，由 ebd 映射 HTTP。

### 2.2 批量校验

`POST /v1/artifacts/verify`

```json
{
  "root": {"namespace": "run", "run_id": "run_<existing_id>", "path": "outputs/native"},
  "files": [{"path": "checkpoint/model/model.safetensors", "bytes": 1024, "sha256": "<64 lowercase hex>"}],
  "max_files": 200,
  "max_file_bytes": 2147483648
}
```

成功返回 `{schema_version: "eb/artifact-verification/v1", verified: true, file_count: int, total_bytes: int}`。任一文件失败时整次请求失败，不返回部分成功作为可登记结果。

- 默认限制沿用现有通用校验的 200 文件、单文件 2 GiB；硬上限为 10000 文件、单文件 1 TiB，调用者只能显式收紧或在该范围内选用既有 recipe 上限。
- 不因统一接口禁止所有零字节文件：空 `__init__.py` 等例外由 Python 原有规则决定；Rust 仍核对实际零字节内容 hash。
- HTTP JSON 请求体上限为 16 MiB，客户端使用 UTF-8 JSON，覆盖 10000 个合法长路径成员；测试合法长清单与请求体超限。参数超限在读取文件前拒绝，实际大小/hash 错误在文件检查时拒绝。
- 仅校验声明文件，不自动证明整个目录没有额外成员。现有 checkpoint 文件集合、`.saving`、必需分片等判断第一阶段仍保留；任务 4 再接入目录事实查询。
- 使用安全文件描述符访问与有界缓冲区 hash；校验不可代替生产者完成协议。正在写入的原生 staging 不能被业务层当作已发布状态。

### 2.3 单文件 CAS 发布

`POST /v1/artifacts/publish-file`

请求：`{root: ArtifactRoot, path: str, expected_bytes: int, expected_sha256: str | null}`。没有生产者 hash 的遗留产物允许 null，由 Rust 在流式复制时计算；有 manifest 时必须传入预期 hash。

成功返回 `{schema_version: "eb/artifact-publication/v1", sha256: str, size_bytes: int, storage: str}`，storage 只能是 `${EB_DATA_ROOT}/cas/<前两位>/<sha256>`。

- 目标由 ebd 决定；源文件只读打开，按块复制到目标文件系统内的独占 staging，校验源稳定性、大小与预期 hash 后发布。
- 完成文件 flush/fsync、只读处理及目录持久化；发布不得覆盖已有对象。相同内容重复发布返回相同引用；已有目标损坏则失败，不能静默替换。
- 同文件系统内完成最终原子发布；跨文件系统源先流式复制至 CAS 所在文件系统。失败不产生可登记的最终引用。
- 不删除源文件，不发布目录，不签名、不创建 ModelVersion、不判定门禁。

### 2.4 Python 方法、错误与并发

在现有 `EbdClient` 增加：

```python
async def verify_artifacts(self, root: dict, files: list[dict], *, max_files: int = 200,
                           max_file_bytes: int = 2 * 1024**3) -> dict: ...
async def publish_file(self, root: dict, path: str, *, expected_bytes: int,
                       expected_sha256: str | None = None) -> dict: ...
```

错误体统一为 `{error: {code: str, detail: str, path: str | null}}`；path 仅含相对路径，不泄露宿主秘密。400 为参数错误，403 为根/路径禁止，404 为缺失，409 为内容变化、摘要不符或发布冲突，413 为请求/文件限制，507 为 ENOSPC，其他 IO 失败为 500。Python 保留实际错误原因，不把传输失败解释为校验通过，也不自动回退绕过 Rust 校验。

hash/复制使用 `spawn_blocking`；第一版最多同时运行两个重 IO 请求，单请求顺序处理文件。该值是初始实现选择，不是吞吐承诺；压测后再调整。不为此新增环境变量或任务队列。全局 storage lock 不覆盖大文件读取和复制；发布、删除的冲突保护必须覆盖实际文件生命周期，不能只依赖启动时的路径检查。

## 3. 文件落点

| 文件 | 责任与改动范围 |
|---|---|
| `rust/crates/eb-artifact/src/lib.rs` | 文件记录、校验、流式单文件 CAS 发布、明确错误；替代现有无错误传播的生产用法 |
| `rust/crates/eb-artifact/Cargo.toml`、`rust/Cargo.lock` | 复用已在 workspace/lock 中的 serde、sha2 等依赖；必要依赖调整在容器内更新锁文件 |
| `rust/crates/ebd/src/artifacts.rs`（新增） | ArtifactRoot 解析、请求限制、并发与 HTTP 错误映射 |
| `rust/crates/ebd/src/main.rs` | 注册两个内部路由，保存必要的受管根；不在 main 中堆入文件算法 |
| `backend/eb/store/ebd.py` | 两个批量/发布客户端方法及响应校验 |
| `backend/eb/orchestrator/ufo_native.py`、`backend/eb/pbfm_evaluation.py` | 替换逐文件 scan，保留业务规则和有效上限 |
| `backend/eb/orchestrator/collect.py`、`backend/eb/checkpoints.py` | 接入批量验证与 CAS 引用；保留模型解析、G3、登记与血缘 |
| 后续 `ebd/src/storage.rs`、`eb-scanner`、`eb-probe`、`worktree.py`、相关 API | 只在对应任务执行时修改，不提前搭建空接口 |

## 4. 阶段 A：批量校验与单文件发布

### Task 1：Rust 批量校验与 Python 客户端

**Files:** 修改上述 artifact crate、Cargo/lock、ebd main、EbdClient；新增 `ebd/src/artifacts.rs`、`backend/tests/unit/test_artifact_io.py`。Rust 行为测试放在所属模块内。

**Interfaces:** 消费 ArtifactRoot/ArtifactFile；产出第 2 节 verify 接口与 `EbdClient.verify_artifacts()`。

- [x] 先写失败检查：`verify_manifest_rejects_same_size_mutation`、`verify_manifest_rejects_link_swap_and_escape`、`verify_manifest_keeps_empty_file_and_size_bounds`。Python 检查一个文件集合只发一次 POST，超限、非 2xx 和畸形成功响应均失败。
- [x] 运行聚焦测试确认失败来自缺失的接口或行为；不能用缺依赖/镜像作为有效失败证据。
- [x] 实现 `verify_manifest(root: &Path, files: &[ArtifactFile], limits: VerifyLimits) -> Result<Verification, ArtifactError>`；根到子成员都按安全描述符访问，路径预检查不能成为打开文件时的唯一保护。
- [x] 接入路由与客户端；检查同长度改写、重复成员、缺文件、零字节、2 GiB 默认限制与显式 Replay 上限。对超大文件上限用稀疏元数据夹具，不在单测实际分配或 hash 多 GiB。
- [x] 增加 `artifact_io_does_not_stall_live_heartbeat`：用阻塞 IO 夹具验证 Tokio 计时器/现有 live lease 路径仍可推进。通过本阶段聚焦命令。

**出口：** 客户端一次请求完成声明文件集合的校验，错误不被吞掉；尚未切换业务调用者。

### Task 2：流式 CAS 发布

**Files:** 修改 `eb-artifact/src/lib.rs`、`ebd/src/artifacts.rs`、`backend/eb/store/ebd.py`、`test_artifact_io.py`；必要时同步 Cargo/lock。

**Interfaces:** 消费任务 1 的根解析与错误；产出第 2 节 publish 接口与 `EbdClient.publish_file()`。

- [x] 写 `publish_file_is_streamed_and_idempotent`、`publish_file_never_overwrites_corrupt_cas`、`publish_file_failure_keeps_source_and_no_final_object`、`publish_file_rejects_source_mutation`，先确认失败。
- [x] 实现 `publish_file(root: &Path, relative: &str, cas_root: &Path, expected: FileExpectation) -> Result<Publication, ArtifactError>`；有界缓冲区完成复制和 hash，禁止以 `fs::read` 加内存整包方式实现。
- [x] 验证重复请求、相同 hash 的并发发布、不同内容冲突、源删除/替换、跨文件系统来源与注入 ENOSPC/写失败；中断 staging 可清理，源和已发布对象必须保留。
- [x] 与 `storage.rs` 补齐并发删除测试：被复制的源与待发布目标不能在保护区间被清理；保护粒度限制到本操作，不能阻塞无关 live heartbeat。
- [x] 通过 Rust/客户端聚焦检查。不要为了兼容旧 `put()` 返回值而在新接口吞掉写错误；旧 helper 的移除以实际调用点为准。

**出口：** ebd 可发布单文件并返回真实 CAS 引用；尚不负责业务登记。

### Task 3：接入 collector、UFO/P-BFM 与已有 checkpoint 校验

**Files:** 修改 `orchestrator/ufo_native.py`、`pbfm_evaluation.py`、`orchestrator/collect.py`、`checkpoints.py`；检查其所有调用者。扩展 `test_pbfm_publication.py`、`test_pbfm_depth_probe_index.py`、`test_pbfm_data_evaluation.py`、`test_train_resume.py`、`test_checkpoint_eval.py`；新增 `backend/tests/unit/test_artifact_collection.py`。

**Interfaces:** 使用任务 1、2 的 EbdClient 方法；保持现有 `verify_index()`/`verify_output_index()` 的业务入口及返回语义。同步 checkpoint 读取函数不隐式发异步 IO：保留其有界元数据校验，在异步 orchestrator 层调用 Rust 完成大文件验证。

在 `backend/eb/store/ebd.py` 定义 `artifact_root(root: Path) -> dict`，仅将现有配置下的 Run/model-run 绝对路径转换为第 2 节引用；其他根显式拒绝，不回退到任意路径 scan。测试以隔离配置根构造夹具，而非扩大生产白名单。

- [x] 先写失败测试：N 文件只调用一次批量验证；P-BFM 空包成员可用；大 Replay 不放宽媒体默认限制；校验失败不能创建成功模型或报告。
- [x] 将逐文件 scan 替换为 batch；保留 Python 对 producer、plan、必需文件、rank/step/Replay、loader/finite 证据、恢复资格和评估语义的判断。第一阶段不声称 checkpoint `_tree()` 已全部下沉。
- [x] 将通用 collector 的 ONNX/checkpoint 整文件复制替换为 publish。没有生产者摘要时让 Rust 计算；ONNX 的 `read_signature()` 读取发布后的不可变 CAS 文件，使 G3 与登记引用对应同一内容。
- [x] 验证 CAS 已发布但 G3/PG 提交失败时：Job 按现有失败流程收尾，不发布成功模型，不自动删除可能共享的 CAS；重试幂等复用。继续使用现有删除预约/输入保护，不建立第二套删除事实表。
- [x] 将连接失败、超时和畸形协议响应接入现有基础设施失败处理，与内容/hash 不符分别记录；不能把 ebd 不可达写成模型质量不合格，也不能因新增异常未被 collector 捕获而遗留无解释状态。
- [x] 检查所有经 `verify_index()` 的 native、partial、live、probe、prompt 调用者；原有业务限制不变。纯 Python/任务镜像内共享的验证 helper 暂时保留，不能让任务容器访问 ebd。
- [x] 跑阶段 A 聚焦回归与 API 返回值兼容检查；记录仍存在的直接文件处理，不顺手扩大到 ONNX 算法或模型下载重写。

**出口：** 实际生产调用链已使用 Rust batch/CAS；公开 API、plan hash、模型资格和历史引用格式保持兼容。

## 5. 阶段 B：逐项收敛剩余控制面 IO

### Task 4：目录事实、扫描与磁盘探测

**Files:** `eb-scanner/src/lib.rs`、`eb-probe/src/lib.rs`、`ebd/src/storage.rs`/`main.rs`、`store/ebd.py`、`domain/scan.py`、`domain/disk.py`、`domain/urdf.py`、`workbench.py`、`ingest.py`、`api/embodiment_routes.py`、`api/routes_data.py`、`api/routes_model.py`；测试复用 `test_scan.py`、`test_disk.py`、`test_ingest.py`，新增 `test_storage_inspection.py`。

**Interfaces:** 增加 `EbdClient.inspect_storage(root: dict, path: str, *, mode: str, cursor: str | None = None, limit: int = 50) -> dict`。mode 固定为 `inventory`、`size`、`disk`；inventory 返回稳定排序的相对路径、类型、大小、mtime 与游标，分页上限 1000。size/disk 不计算文件 hash。

小型文本通过 `EbdClient.read_storage_text(root: dict, path: str, *, max_bytes: int = 1024**2) -> str` 获取，内部 `POST /v1/storage/read-text` 同样只接受已解析的只读根；硬上限 16 MiB，超限或解码失败明确报错，禁止截断后继续解析。此接口仅支持文本元数据，不用于模型、视频或 Replay。

- [x] 先锁定旧扫描结果、目录排序/分页、缺失目录、不可读项、跳过缓存目录和磁盘阈值行为。新测试证明遍历和 statvfs 由 Rust 执行。
- [x] 扩展只读根：受管 run/model/data/assets，以及现有配置明确允许的外部源；复用已配置源根，不把 `/data/kk` 整体视为任意读权限。`deploy/docker-compose.yml` 给 ebd 显式传递与 API 一致的 datasets/assets/source 配置，保持写根单独限制。
- [x] Rust 返回文件事实；Python 仍解析有界 URDF 文本、推导本体/任务兼容性并写 PG。需要读取的小型源码/URDF复用安全有界读取；不为了下沉遍历重写 XML 或训练注册语义。
- [x] 将 `domain.scan` 改为消费 inventory 和有界文本的纯函数，`domain.disk` 消费 `{free_bytes,total_bytes}`。保留现有 10 GiB 余量、10% 警告和 2.5 归档倍率；API 调用者改为先取事实再执行规则。
- [x] 接入目录大小、dataset tree、legacy ingest 枚举；保留导入资格、删除 tombstone 和模型登记逻辑。将 native checkpoint 的实际成员集合检查改为消费 Rust inventory，不修改其算法语义。
- [x] 运行扫描、数据视图、导入、checkpoint、模型 API 回归；核对原返回字段和历史默认目录仍可解析。

**出口：** 已迁移入口不再直接遍历宿主目录；尚未迁移的入口列出清单。磁盘查询只是事实，不引入磁盘预算租约系统。

### Task 5：RSL 恢复文件准备

**Files:** `worktree.py`、`orchestrator/service.py`、`store/ebd.py`、`ebd/src/storage.rs`、`test_worktree.py`、`test_orchestrator_fail.py`，以及 Rust storage 测试。

**Interfaces:** `EbdClient.prepare_resume_files(source_run_id: str, target_run_id: str, files: list[dict]) -> dict`；每项含 `source_path`、`target_path` 两个 Run 内相对路径与 `expected_bytes`、`expected_sha256`。有生产者清单必须绑定原 hash；遗留文件允许 hash 为 null，由 Rust 在复制时计算并写入准备结果。源恢复目录与 checkpoint 选择继续由 Python 根据现有规则决定，Rust 不自行挑最大 step。

- [x] 写缺 checkpoint、源仍活动、目标路径越界、目标已存在、复制失败与重复准备测试，先确认失败。
- [x] 通过任务 4 获取候选事实，保持旧 RSL 选择规则；Rust 只将显式文件复制到目标 Run 的独占临时目录，验证后发布。
- [x] 保留旧 `.pt/.yaml/.pkl/.json` 与 `params` 配置覆盖范围、`--load_run/--checkpoint` 参数和失败原因；不覆盖已准备好的不同内容。
- [x] 运行恢复、orchestrator 失败与删除并发测试。P-BFM/UFO 原生 trainer 的保存、barrier、loader、Replay 复制暂不迁移；不得把本任务写成原生恢复全链路下沉。

**出口：** 控制面旧 RSL 恢复路径不再自行复制权重；原生任务运行时保持原协议。

### Task 6：媒体、模型资产与通用文件流

**Files:** `ebd/src/run_output.rs`、`main.rs`、`storage.rs`、`api/routes.py`、`api/routes_data.py`、`api/embodiment_routes.py`、`api/routes_ufo_native.py`；复用 `test_ufo_output_access.py`，新增 `test_media_proxy.py`。

**Interfaces:** 在现有媒体 UDS 增加受管根文件读取，使用任务 4 的只读 root 命名空间；Python 增加 `stream_storage_file(request, root: dict, path: str)`，继续由现有公开路由调用。不返回 ebd 直连地址。

- [x] 固定公开 GET/HEAD、Range/206/416、Content-Type、长度、缓存与错误行为；包含空文件、断连、嵌套资产、符号链接替换与私有 Replay 拒绝测试。
- [x] 复用 `run_output` 的 fd 安全打开和 64 KiB 分块读取，替换旧 `/media` 的整文件读取；只抽出实际复用的 primitive，不先构建通用文件服务框架。
- [x] Python 先按 Dataset/Embodiment/EvalReport 和已验证索引确定可访问文件，再转发 Range/HEAD；保持 native 的推理包/公开证据白名单，训练状态与优化器文件继续不可公开。
- [x] 统一代理连接释放：下载完成、异常和客户端断开均关闭上游 response/client；不在长下载期间持有数据库事务。
- [x] 回归媒体、资产预览、UFO/P-BFM 文件索引与浏览器播放/下载。不重写实时仿真 worker，不把实时仿真控制升级为真机控制。

**出口：** 指定媒体入口已统一字节通道；鉴权和业务可见性仍在 Python，公开 URL 保持兼容。

### Task 7：通用 Bundle 的机械打包

**Files:** `bundle/pack.py`、`api/routes.py`、`adapters/mujoco.py`、`orchestrator/collect.py`、`eb-artifact`、`ebd/src/artifacts.rs`；需要新增独立 Python `backend/eb/bundle/worker.py` 进行模型检查，并同步 `deploy/images/tools.Dockerfile`。测试新增 `test_bundle_job.py`，复用 `test_g4.py`、`test_onnx_io.py`、Task 6 的下载检查。

**Interfaces:** 新流程复用已存在的 `POST /api/v1/jobs` 与 `JobKind.BUNDLE_BUILD`；冻结成员引用/内容 hash，Python 工具任务产生模型检查结果及明确成员清单，控制面确认业务条件后调用 Rust `pack_bundle(root: ArtifactRoot, members: list[dict]) -> Publication` 完成打包。

- [x] 先补 `bundle.build` 真实 Job 测试；当前通用 ToolsAdapter 入口不能仅凭名称视为已实现。产物缺失、ONNX 检查跳过/失败、G4 失败都不能提升交付资格。
- [x] ONNX 图解析、数值推理与门禁判定继续使用 Python；工具镜像复用现有部署镜像已固定且验证兼容的 onnx/onnxruntime 版本并补 CPU probe，缺依赖不能被当作数值检查通过。工具任务不获得 ebd socket。Rust 只按显式清单归档、压缩、hash 和发布，不读取任意宿主文件。
- [x] 工具任务仅从声明的只读挂载读取输入，在自身 Run 的 `outputs/bundle-staging` 生成待打包成员与清单；权重准备使用流式复制。Rust pack 的 ArtifactRoot 指向该目录，不能把不同根的宿主绝对路径夹进 members。
- [x] 流式生成 tar.zst；逐项验证归档路径和成员 hash，不允许压缩失败后将未压缩 tar 当成同一种成功包。必须保留既有包成员名和消费者兼容性测试；若需新增归档依赖，仅在本任务加入并锁定。
- [x] collector 在模型证据与包完整性均通过后登记 Bundle/Artifact/血缘，签名和真机资格仍由原规则控制；完成打包不能自动签名。
- [x] 老 `/bundles/build` 是同步接口，本任务不偷偷改变响应为 Job：先作为明确保留的兼容路径，不宣称全部打包 IO 已下沉。新流程通过现有 Job 查询展示结果；移除或版本化老入口另立 API 兼容任务。
- [x] 核对 `frontend/src/views/Submit.vue` 的通用 Job 提交与 `Bundles.vue` 列表即可；不额外新增一套 Bundle 工作流页面。只有实际调用格式改变时才调整这些现有页面并运行前端检查。

**出口：** 新通用打包流程经 Job 运行，机械打包下沉；旧同步入口的保留范围明确记录，不能借此绕过 native 仿真专用交付规则。

## 6. 阶段 C：有证据后再做

### Task 8：证据索引与遥测，独立选择

当前结论：本轮没有满足启动条件的测量或真实遥测证据，保持未开始。robot-agent 的 serve 仅合成 100 条固定温度示例，不算真实现场生产者。

启动依据：阶段验证发现 JSONL/episode 索引重读占据请求主要耗时或响应体持续过大；或者已有真实端侧遥测生产者需要接入。没有这些证据就保留此任务未开始，不新增空服务或索引数据库。

**Files:** 按实际选项修改 `eb-collector`/`eb-tsdb`、`ebd/src/main.rs`、`store/ebd.py`、`api/routes_eval.py`、`api/routes_data.py`、`api/routes.py`。新增 `test_evidence_query.py` 或 `test_telemetry_io.py`，不用为未实施选项预建文件。

**Interfaces:** 证据查询为 `EbdClient.evidence_page(run_id, path, *, cursor, limit) -> dict`，返回有界原始记录与续读游标；遥测为 `EbdClient.append_telemetry(deployment_id, points) -> dict`。路径和 tag 受限，不新增任意 SQL/任意文件 API。

- [ ] 为选中能力先写失败测试：尾行半写、重复/未知 case、记录损坏或文件截断；或者 tag 冲突、非有限值、重复请求和旧 tag 读取。冻结预期行为后实现。
- [ ] Rust 提供有界/增量读取或批量追加；Python 继续根据冻结 Suite 判定 case 有效性、成功率和验收资格。图表生成与训练分析仍在 Python 任务内。
- [ ] 保留原始证据，不用下采样数据覆盖原数据；不把多卡采样量当优化器更新数。若需要持久游标，先证明只读重建能够处理重启/截断，再增加最少状态。
- [ ] 对比迁移前后的请求体、读取字节、延迟和内存，记录继续或停止结论。现有 `ingest_tfevents()` 已停用、`parse_tfevents()` 未接业务调用，不为遗留 helper 再移植解析器。

**出口：** 只交付被证据证明需要的选项。日志/指标全量重读问题已有部分位于 Rust 内，相关优化单列，不计为 Python 下沉成果。

## 7. 验证、部署与回退

### Task 9：各阶段共同执行的交付检查

**Files:** 新增 `docs/progress/2026-10-28-python-rust-data-plane-migration.md` 记录实际完成范围和证据；必要时更新 `.agents/tools.md` 的已有命令说明。不要把未执行计划写成完成进度，不修改设计真源以适配实现。

**Interfaces:** 不新增业务协议；汇总各任务结果、未迁移路径、构建版本、性能数据与运行限制。

- [x] **第一轮：边界与协议。** 核对所有调用者、类型、默认上限、空文件例外、公开 API 和历史 hash；确认任务容器/平台 Agent 没有获得新增底层权限。
- [x] **第二轮：失败与并发。** 执行同长度篡改、替换/链接、重复发布、损坏目标、写盘失败、PG 登记失败、删除竞态、失败/取消/重试及旧 Run 回归。
- [x] **第三轮：用户入口与证据。** 使用既有冻结模型/媒体和隔离夹具核对注册、查询、下载、Range、UI 错误可见性与资源释放；不另开训练验证文件操作。
- [x] 运行下述容器化命令并记录实际输出；测试未覆盖的 runtime/集成脚本不能由全量 pytest 推导通过。
- [x] 使用隔离文件根和数据库测试容器化 HTTP-over-UDS 链路，至少覆盖真实 Rust hash、复制与原子发布；Python mock 测试不能替代这一项。禁止对现有已发布对象做破坏性测试。
- [x] 对 1/100/1000 文件清单记录 RPC 数、总读取字节、耗时、API/ebd 峰值 RSS，以及并行健康/心跳请求延迟；单文件至少比较 64 MiB 和 512 MiB。分别记录冷/热缓存和磁盘环境，不给虚构的提速倍数。
- [x] 第一阶段硬出口：N 文件只有一次 verify RPC；大文件复制没有按文件大小分配整块内存；请求失败不登记成功产物；内容与历史业务结论一致；无关实时 lease 不被重 IO 阻塞。吞吐没有改善时先分析重复读取和磁盘瓶颈，不盲目增加并发。

### 容器化验证命令

以下从仓库根执行，保留为验证指引；本轮实际命令、镜像及结果以进度文档为准。`eb/test:local` 基于 `eb-api:latest`，先核对基础镜像并按需要构建；不以运行中的生产容器执行破坏性测试。

```bash
# 测试基础镜像与后端测试镜像：构建不等于部署
docker compose -f deploy/docker-compose.yml --env-file deploy/.env build api
docker build -f deploy/test.Dockerfile -t eb/test:local .

# 第一阶段新增 Rust 接口、流式发布、媒体/锁回归
docker run --rm -v "$PWD/rust:/workspace:ro" -v eb-cargo-cache:/usr/local/cargo \
  -e CARGO_TARGET_DIR=/tmp/target -w /workspace rust:1.83-bookworm \
  cargo test -p eb-artifact -p ebd --locked

# 第一阶段 Python：新增测试必须在对应任务创建后才加入执行
docker run --rm -v "$PWD:/workspace:ro" -v /data/kk:/data/kk:ro eb/test:local \
  python -m pytest -q -p no:cacheprovider \
  tests/unit/test_artifact_io.py tests/unit/test_artifact_collection.py \
  tests/unit/test_pbfm_publication.py tests/unit/test_pbfm_depth_probe_index.py \
  tests/unit/test_pbfm_data_evaluation.py tests/unit/test_train_resume.py \
  tests/unit/test_checkpoint_eval.py tests/unit/test_deletion.py \
  tests/unit/test_ufo_training_checkpoint.py tests/unit/test_pbfm_checkpoint.py

# 第二阶段按实际修改选择测试文件，不执行不存在或未涉及的选项
docker run --rm -v "$PWD:/workspace:ro" -v /data/kk:/data/kk:ro eb/test:local \
  python -m pytest -q -p no:cacheprovider \
  tests/unit/test_scan.py tests/unit/test_disk.py tests/unit/test_ingest.py \
  tests/unit/test_data_view.py tests/unit/test_worktree.py \
  tests/unit/test_orchestrator_fail.py tests/unit/test_ufo_output_access.py

# 阶段完成后的现有后端/Rust 总检查
make test
git diff --check
```

新增依赖需要在可写的临时/工作副本中用容器更新 `rust/Cargo.lock`，再运行上述只读挂载加 `--locked` 的命令。任务 4/5 增加 scanner/probe/runner 变更时相应扩展 Rust 聚焦包；任务 6/7/8 将各自新增的测试明确加入当轮命令。前端实际变化时使用[工具手册](../../.agents/tools.md)中的 Node 测试、类型检查和 Vite 构建命令。

### 部署与回退边界

1. 先完成隔离验证，再按 `make up` 部署。新增 ebd 路由先于切换 Python 调用者交付；可以先部署仍兼容旧调用者的 Rust，再部署 API。不在活动任务期间临时重启 ebd 验证方案。
2. 第一阶段不需要 PG schema 迁移。CAS 布局、Artifact.storage、模型/数据 ID 和历史 plan hash 保持原样；如实现发现必须改 schema，先补独立迁移设计和空库/旧库检查，不能悄悄扩大本阶段。
3. 回退以已保留的镜像 digest/构建产物为依据，不依赖 Git 提交，不覆盖发布资产。API 回退到旧调用者时可继续使用新 Rust 的旧接口；完整旧 ebd 回退需先确保 API 不再调用新接口。
4. 不因新接口失败自动落回 Python 并跳过完整性检查。失败保持可定位状态；已发布但未被 PG 引用的 CAS 留待既有引用/清理流程审查，不能立即删除。
5. 文档必须列出仍保留的旧同步 Bundle API、任务容器内 checkpoint 发布、未迁移的直接 IO 和未执行的阶段 C。部分完成不能宣称整个控制面已纯化。
6. 修改任务 worker 或镜像 COPY 文件时才重建对应任务镜像，并按原有 receipt/快照协议验证；不为本方案统一升级所有训练镜像。

## 8. 方案复核记录与当前状态

- **边界复核：** 文件 SHA、流式复制和路径打开下沉；计划指纹、训练/恢复语义、门禁与事务保留 Python。ONNX/HDF5/LeRobot/仿真依赖继续在受管 Python 工具任务中使用。
- **可达性复核：** 第一阶段覆盖真实 collector 和 UFO/P-BFM 调用者；不将未接通的 Rust CAS helper、旧 tools 占位命令或停止使用的 Python tfevents helper 当成已完成链路。
- **失败与兼容复核：** 明确默认媒体限制与 Replay 扩大上限、零字节例外、CAS 重复/损坏、PG 与文件发布先后、历史 hash、旧同步 Bundle 返回值及任务容器权限。
- **验证复核：** 单测、真实 UDS 文件链路、容量/心跳测量和用户入口检查分别记录；没有新训练或模型质量验收要求，不借测试通过提升模型资格。

当前状态：任务 1–7 已实施，任务 8 启动条件未满足、保持未开始；任务 9 已完成隔离验证、release 测量、make up 部署与真实只读入口验收。完整 runtime pytest 的 Torch 依赖和三项旧基线限制见进度文档；未新开训练、未执行 Git 提交。
