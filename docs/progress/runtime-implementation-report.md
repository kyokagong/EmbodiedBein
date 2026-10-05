# Runtime foundation implementation — 2026-09-09

本次落实现行 `docs/design/02-system-architecture.md` 的内容快照与任务资源/网络边界。没有执行 Git，没有写入三个外部仓库；测试与编译均在 Docker 容器执行。

## 实现

- `ebd POST /v1/snapshots` 在阻塞工作线程中复制当前代码，生成逐文件 SHA-256 清单以及清单内容哈希，发布到 `snapshots/<hash>/code`。控制面通过 UDS 调用，不运行训练或 Git subprocess。
- 复制前后重新遍历并比较内容清单；文件读取期间检查大小/修改时间；拒绝特殊文件与所有符号链接，并核对打开文件的真实路径。没有 hardlink，不修改源目录。文件与目录移除写权限后原子重命名发布，任务 bind 为只读。已冻结的受管路径按哈希复核后复用。
- 排除 VCS、缓存、虚拟环境；仅源根目录排除 logs/outputs/checkpoints/datasets/target，保留例如 `rlinf/data/datasets` 的实际 Python 源模块。
- 允许读取配置的 BHL/RLinf 仓根或平台 data_root 内的目录；其他源返回 403。Git revision 不再被解析；`commit` 兼容字段只接受 `sha256:<hash>`，旧 Git revision 明确报错。旧探测的 dirty 为未知、head 为未知，不假装干净或解析成功。
- Orchestrator 先执行 G1、容量与 renderer 校验，再冻结代码并写入 `content_snapshot` code_refs。支持 `submit(commit=False)`，由调用方原子提交关联记录。in_image 本体无需默认 BHL 快照。
- 写目录只创建于 Run 区，拒绝绝对路径、`..` 和 symlink 越界，不再修改快照。数据转换自带 mount 保留。
- CPU/RAM 从 JobSpec 传到 ebd，Docker 实际使用 `--cpus` / `--memory`。任务网络显式 bridge（独立 network namespace）或 none；host 被拒绝。Rust 校验 CPU、RAM、shm 正值且 shm 不超过 RAM。

## 验证

- 旧 Python 快照回归实际失败：调用 Git 被测试禁止、overlay 越界未拒绝。替换测试不执行任何 Git。
- Python 容器针对 worktree 与 orchestrator 检查：6 passed，覆盖 UDS 请求、未知 Git 状态、overlay 越界、不修改快照、资源/网络传递、数据 mount 保留及启动失败状态。
- Rust 容器 `cargo test -p eb-runner -p ebd --locked`：3 passed（快照复制/复用/哈希固定/源变更/拒绝 symlink/保留 nested datasets；Docker 参数；非法资源与 host 网络）。nested datasets 回归先看到失败后修复。
- 实际构建 ebd 后在无 docker.sock 的隔离测试容器启动：UDS 快照请求返回 sha256 和 owned path；`/etc` 请求返回 403；`HEAD` 请求返回 400。没有启动训练任务。

## 保留边界

- 所有 symlink 都明确拒绝，包括指向源内的链接；需要链接资产时应单独纳管并加入明确依赖版本，不静默跟随。
- root 数据目录排除项是显式策略；特殊布局应扩展版本化输入，而不是把大型权重或数据混进代码快照。当前没有快照垃圾回收和容量预估。
- 普通主机管理员仍可 chmod 修改平台文件；这里保证平台运行流程与任务只读 mount 不修改冻结代码，并在再次使用时校验内容，不宣称抵抗主机 root。
- 同一新哈希并发发布竞争可能返回可重试错误；未加入分布式锁。没有新增资源租约、重启容器恢复或训练依赖就绪检测。
- 新 ebd/API 协议需要一起部署；本次只完成代码、容器编译和隔离 UDS 验证，不代表线上部署或 LIBERO 训练成功。
