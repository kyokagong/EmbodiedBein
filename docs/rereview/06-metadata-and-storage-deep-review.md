# 06 · 元数据体系与磁盘存储深度详规

本规范定义了 EmbodiedBein 的存储架构与元数据规范：**以宿主机磁盘为唯一真源，PostgreSQL 管理目录元数据，训练容器按需安全挂载。**

---

## 1. 核心架构原则

1. **磁盘即真源 (Disk as Ground Truth)**：所有本体代码、数据集、预训练模型和训练产物直接驻留在宿主机文件系统上。平台不搞「将数据拷入 CAS 内容寻址池再分发」的中间层。
2. **只读保护与隔离可写 (Read-Only Isolation)**：外部业务代码仓与已有数据集在容器内一律为只读（`ro`）挂载；训练容器的唯一可写业务目录严格限定在该次任务专属的 `/var/lib/eb/runs/<train_id>/`。
3. **权限自洽 (UID/GID Self-Consistency)**：容器写入的文件必须保证宿主机普通开发用户（如 `kk`）拥有完整的读写、移动和删除权限，杜绝 `root:root` 锁死磁盘目录。

---

## 2. 宿主机磁盘标准布局

```
/data/kk/                                # 外部开发目录 (平台只读挂载，严禁写入)
├── Berkeley-Humanoid-Lite/             # 本体代码与 3D 资产
├── RLinf/                              # RLinf 训练框架源码与配置
└── berkeley_ego_data/                  # 本地预存数据集

/var/lib/eb/                            # 平台统一数据根目录 (EB_DATA_ROOT)
├── datasets/                           # 平台下载或导入的数据集目录
│   └── <dataset_id>/                   # 某具体数据集文件树
├── models/                             # 平台纳管的模型权重目录
│   └── <model_id>/                     # 某基座或认领的模型权重文件
├── runs/                               # 训练运行时独立隔离区 (任务唯一可写)
│   └── <train_id>/                     
│       ├── spec.json                   # 任务冻结配置快照
│       ├── hydra_resolved.yaml         # 最终生效的 Hydra 配置
│       ├── logs/stdout.log             # 容器实时终端日志
│       ├── tb/                         # TensorBoard tfevents 标量数据
│       └── output/                     # 训练产物 (checkpoints, rollout mp4)
├── cache/                              # 跨任务共享缓存区
│   ├── ov_data/                        # Isaac Sim / Nucleus 资产缓存
│   ├── huggingface/                    # Hugging Face 模型/数据下载缓存
│   └── torch/                          # PyTorch Hub 缓存
└── tmp/                                # 数据集下载解压中转区
```

---

## 3. 训练容器挂载安全矩阵

| 宿主机路径 | 容器内挂载点 | 挂载权限 | 用途与安全规则 |
|---|---|---|---|
| 本体登记路径 (`root_path`) | `/workspace/berkeley` | **`ro` (只读)** | 机器人物理模型与环境注册代码，严禁容器修改 |
| RLinf 框架路径 | `/workspace/rlinf` | **`ro` (只读)** | 训练算法与配置脚本，严禁容器修改 |
| 数据集路径 (`dest_path`) | `/workspace/data/<ds_id>` | **`ro` (只读)** | 输入离线轨迹与视频特征，保障数据不被污染 |
| 基座模型路径 (`path`) | `/workspace/models/<mdl_id>`| **`ro` (只读)** | 初始预训练权重输入 |
| 任务产物目录 `/runs/<id>/output` | `/workspace/output` | **`rw` (可写)** | 唯一的 Checkpoint、视频和评估输出目录 |
| 任务指标目录 `/runs/<id>/tb` | `/workspace/tb` | **`rw` (可写)** | TensorBoard 事件写入 |
| 平台共享缓存 `/cache/ov_data` | `/root/.local/share/ov/data`| **`rw` (可写)** | Isaac Lab 仿真资产本地缓存，加速冷启动 |

---

## 4. 权限与安全机制 (UID/GID & Path Allowlist)

### 4.1 路径白名单防护 (Path Allowlist)
为了防止恶意用户将宿主机敏感系统目录（如 `/etc`、`/root`、`/var/run/docker.sock`）登记为本体或数据集并挂载进容器，系统在所有涉及路径输入的接口上强制执行白名单前缀校验：
- **默认允许前缀**：
  - `/data/kk`
  - `/var/lib/eb`
- **规则**：登记路径必须以白名单前缀开头，且经过 `os.path.realpath` 规范化，严禁 `../` 路径越权。

### 4.2 容器 UID/GID 映射
`ebd` 在调用 Docker Engine 创建训练容器时，支持注入当前宿主机的运行用户 ID：
```bash
docker run --user 1000:1000 ...
```
或者在启动脚本中预设 `umask 0002`，确保任务生成在 `/var/lib/eb/runs/<id>/output` 下的所有权重文件，在宿主机上均可被普通用户 `kk` 正常操作与备份。

---

## 5. 完整数据库结构 (PostgreSQL DDL)

```sql
-- 1. 系统设置表 (单行全局配置)
CREATE TABLE system_settings (
    id INT PRIMARY KEY DEFAULT 1,
    allowed_path_prefixes JSONB NOT NULL DEFAULT '["/data/kk", "/var/lib/eb"]',
    rlinf_repo_path VARCHAR(512) NOT NULL DEFAULT '/data/kk/RLinf',
    default_train_image VARCHAR(256) NOT NULL DEFAULT 'eb/rlinf:0.4.0-emb',
    hf_token VARCHAR(256),
    disk_alert_threshold_gb INT DEFAULT 20,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 2. 本体表
CREATE TABLE embodiments (
    id VARCHAR(64) PRIMARY KEY,
    display_name VARCHAR(128) NOT NULL,
    root_path VARCHAR(512) NOT NULL,
    status VARCHAR(32) NOT NULL DEFAULT 'ready',
    error_message TEXT,
    default_urdf VARCHAR(256),
    urdf_variants JSONB NOT NULL DEFAULT '[]',
    joint_tree JSONB NOT NULL DEFAULT '{}',
    gym_ids JSONB NOT NULL DEFAULT '[]',
    interface_snapshot JSONB,
    scanned_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 3. 数据集表
CREATE TABLE datasets (
    id VARCHAR(64) PRIMARY KEY,
    name VARCHAR(128) NOT NULL,
    source_kind VARCHAR(32) NOT NULL,
    source_uri TEXT NOT NULL,
    dest_path VARCHAR(512) NOT NULL,
    format VARCHAR(32) NOT NULL DEFAULT 'unknown',
    status VARCHAR(32) NOT NULL DEFAULT 'ready',
    total_bytes BIGINT DEFAULT 0,
    download_progress JSONB DEFAULT '{}',
    summary_stats JSONB DEFAULT '{}',
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 4. 镜像表
CREATE TABLE sim_images (
    id VARCHAR(64) PRIMARY KEY,
    name VARCHAR(256) NOT NULL,
    tag VARCHAR(64) NOT NULL DEFAULT 'latest',
    digest VARCHAR(128),
    size_bytes BIGINT DEFAULT 0,
    role VARCHAR(32) NOT NULL DEFAULT 'train',
    is_default BOOLEAN DEFAULT FALSE,
    status VARCHAR(32) NOT NULL DEFAULT 'ready',
    pull_progress JSONB DEFAULT '{}',
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    CONSTRAINT uq_sim_images UNIQUE (name, tag)
);

-- 5. 训练任务表
CREATE TABLE train_jobs (
    id VARCHAR(64) PRIMARY KEY,
    name VARCHAR(128) NOT NULL,
    status VARCHAR(32) NOT NULL DEFAULT 'queued',
    template_id VARCHAR(64) NOT NULL,
    image_digest VARCHAR(128) NOT NULL,
    embodiment_id VARCHAR(64) NOT NULL REFERENCES embodiments(id),
    dataset_ids JSONB DEFAULT '[]',
    base_model_id VARCHAR(64),
    gpus_allocated JSONB DEFAULT '[]',
    ray_port INT DEFAULT 6390,
    params JSONB NOT NULL DEFAULT '{}',
    hydra_overrides JSONB NOT NULL DEFAULT '[]',
    run_dir VARCHAR(512) NOT NULL,
    auto_register_latest BOOLEAN DEFAULT FALSE,
    metrics_summary JSONB DEFAULT '{}',
    error_message TEXT,
    started_at TIMESTAMP WITH TIME ZONE,
    finished_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 6. 模型表
CREATE TABLE models (
    id VARCHAR(64) PRIMARY KEY,
    name VARCHAR(128) NOT NULL,
    kind VARCHAR(32) NOT NULL DEFAULT 'trained',
    path VARCHAR(512) NOT NULL,
    framework VARCHAR(32) NOT NULL DEFAULT 'pytorch',
    size_bytes BIGINT DEFAULT 0,
    embodiment_id VARCHAR(64) REFERENCES embodiments(id),
    train_job_id VARCHAR(64) REFERENCES train_jobs(id),
    parent_model_id VARCHAR(64) REFERENCES models(id),
    metrics_at_checkpoint JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 7. 血缘边表
CREATE TABLE lineage_edges (
    id BIGSERIAL PRIMARY KEY,
    src_type VARCHAR(32) NOT NULL,
    src_id VARCHAR(64) NOT NULL,
    dst_type VARCHAR(32) NOT NULL,
    dst_id VARCHAR(64) NOT NULL,
    relation VARCHAR(32) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    CONSTRAINT uq_lineage_edge UNIQUE (src_type, src_id, dst_type, dst_id, relation)
);
```

---

## 6. 磁盘空间监控与垃圾清理 (GC)

具身强化学习高频训练会产生海量 Checkpoint 与日志文件。平台内置磁盘清理助手：
1. **水位监测**：总览页实时计算 `/var/lib/eb` 的占用空间，当剩余空间不足 20GB 时展示橙色预警横幅。
2. **安全 GC 策略**：
   - 允许用户在设置页点击「清理历史未被模型纳管的中间 Checkpoints」，一键删除已完成或失败任务中未经注册的中间 step 权重文件，仅保留最终日志与指标，瞬间释放上百 GB 磁盘。
