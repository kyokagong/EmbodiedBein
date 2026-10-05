# 可配置宿主存储与现有元数据迁移

部署存储由 `deploy/.env` 的三个变量控制，容器内保持相同绝对路径：

| 变量 | 当前值 | 内容 |
|---|---|---|
| `EB_HOST_STATE_DIR` | `/data/kk/embodiedbein` | Run、日志、CAS、TSDB、快照 |
| `EB_HOST_MODELS_DIR` | `/data/kk/models` | 下载模型、训练模型 |
| `EB_HOST_DATA_DIR` | `/data/kk/data` | 平台下载与验证的数据集 |

2026-09-10 已将原 `/var/lib/eb` 状态目录迁至 `/data/kk/embodiedbein`，将 LIBERO 数据集迁至 `/data/kk/data/ds_8aecbcf58ea74fda9ca3`，并把两条已登记模型复制到 `/data/kk/models` 的受管目录。数据库内 Job、Run、Pipeline、Artifact、Dataset、Model、训练及评估 JSON 元数据中的旧路径在单个事务内完成改写，事务后未剩余 `/var/lib/eb` 引用。

外部仓库 `/data/kk/Berkeley-Humanoid-Lite`、`/data/kk/RLinf`、`/data/kk/berkeley_ego_data` 仍只读。Compose 只给三个配置目录增加写挂载，不授予整个 `/data/kk` 写权限。

验证：Backend `147 passed`；Rust workspace 全部通过；Compose 展开后的环境变量与 bind mount 指向三个配置目录。模型副本按下载凭证及 checkpoint manifest 的文件大小和 SHA-256 校验通过。

后续训练提交复核发现编排器仍只允许状态根作为写挂载，导致 `/data/kk/models/runs/<run_id>` 在预览成功后被提交门禁拒绝。门禁已统一为仅允许 `EB_HOST_STATE_DIR`、`EB_HOST_MODELS_DIR`、`EB_HOST_DATA_DIR` 三个受管根写入；外部源码根仍只读。
