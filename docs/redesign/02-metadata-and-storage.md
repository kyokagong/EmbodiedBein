# 02 · 元数据与磁盘

两条原则：

1. **文件在磁盘上，元数据在库里。** 训练不把数据「拷进 CAS 再下发」，而是把已有目录挂进容器。
2. **元数据是目录，不是文件的第二份副本。** 丢了库可以按约定路径再扫；丢了盘，库里的记录只表示「曾经有过」。

## 1. 谁是真源

| 对象 | 真源 | 库里记什么 |
|---|---|---|
| 本体代码与 3D | 你指定的目录（如 BHL 仓） | id、display_name、root_path、扫描时间、资产索引 |
| 数据集 | 你指定的目标目录 | id、source、dest_path、字节数、格式、状态 |
| 模型权重 | 你指定的或训练输出目录 | id、kind（base/run）、path、父模型、train_id |
| 训练过程 | `/var/lib/eb/runs/<train_id>/` | 状态、镜像、挂载表、Hydra 覆盖、GPU |
| 镜像 | 宿主机 Docker | name:tag、digest、size、pull 状态、用途 |
| Isaac Lab 任务 | 本体/RLinf 仓里的 gym id 与 yaml | 扫描缓存（可重建） |

哈希（sha256）是**可选完整性字段**，用来发现「目录被改过了」，不是寻址键。内容寻址仓库不再作为主存储。

MinIO 本版可以继续在 compose 里占位，训练主路径不依赖它。

## 2. 磁盘布局

平台可写根目录仍建议 `/var/lib/eb`（可用 `EB_DATA_ROOT` 改）。外部三个目录继续只读 bind。

```
/var/lib/eb/
├── datasets/<dataset_id>/     # 下载或拷贝后的数据（用户也可登记已有路径，不强制搬进来）
├── models/<model_id>/         # 登记的基座；训练认领的产物也可落在这里
├── runs/<train_id>/           # 一次训练唯一可写业务目录
│   ├── spec.json              # 冻结：镜像、挂载、Hydra overrides、等价命令
│   ├── hydra.yaml             # 提交时生成的配置快照
│   ├── logs/stdout.log
│   ├── tb/                    # tensorboard / tfevents
│   └── output/                # ckpt、视频、RLinf results
├── cache/{hf,torch,pip}/      # 跨任务共享缓存（可写，带锁）
└── tmp/                       # 下载中转

# 外部只读（compose bind，平台不写）
/data/kk/Berkeley-Humanoid-Lite
/data/kk/RLinf
/data/kk/berkeley_ego_data
```

**允许数据/模型不搬进 `/var/lib/eb`。** 元数据里的 `path` 可以是任意宿主机目录（需在允许前缀内）。训练挂载的就是这个 path。`/var/lib/eb/datasets` 只是默认落点。

允许前缀默认：

- `/var/lib/eb`
- `/data/kk`

可在设置里追加。前缀外的路径拒绝登记，防止把 `/` 或别人的家目录挂进容器。

## 3. 训练时怎么挂

一次 RLinf 训练的典型挂载（只读除非标明 rw）：

| 宿主机 | 容器内 | 模式 |
|---|---|---|
| 本体 `root_path` | `/workspace/berkeley` | ro |
| RLinf 仓（设置里的栈路径） | `/workspace/rlinf` | ro |
| 数据集 path | `/workspace/data/<dataset_id>` | ro |
| 基座模型 path | `/workspace/models/<model_id>` | ro |
| `/var/lib/eb/runs/<id>/output` | `/workspace/output` | **rw** |
| `/var/lib/eb/runs/<id>/tb` | `/workspace/tb` | **rw** |
| `/var/lib/eb/cache/hf` | `/root/.cache/huggingface` | rw |

容器**没有**把外部仓的 `logs/`、`configs/` 打成可写。RLinf 的 `runner.logger.log_path` 必须被 overrides 成 `/workspace/output`。这是现行隔离设计里唯一必须留下的部分：否则并发训练会写脏 BHL/RLinf 仓。

## 4. 元数据对象（库表意图）

表名可与现行 ORM 对齐改造，语义按下表，而不是继续用 `embodiment_envs` / `experiments` 当核心。

### 4.1 Embodiment

```yaml
id: emb_berkeley_humanoid_lite
display_name: Berkeley Humanoid Lite
root_path: /data/kk/Berkeley-Humanoid-Lite
status: ready | scanning | error
scanned_at: 2026-08-28T12:00:00+08:00
assets:
  - kind: urdf
    path: source/berkeley_humanoid_lite_assets/.../xxx.urdf
  - kind: mesh
    path: .../meshes/base.stl
gym_ids:
  - Isaac-Berkeley-Humanoid-Lite-v0
interface_snapshot:          # 派生，可空
  obs_dim: 85
  action_dim: 22
  policy_hz: 50
  source: scanned            # 不是手写 YAML 真源
error: null
```

### 4.2 Dataset

```yaml
id: ds_ego_raw
name: berkeley_ego_data
source:
  kind: path | http | huggingface
  uri: /data/kk/berkeley_ego_data
dest_path: /data/kk/berkeley_ego_data    # 已在本地则 dest=source
status: ready | downloading | error
bytes: 44000000000
format: mixed | lerobot_v2 | hdf5 | unknown
preview: files | episodes | none
```

### 4.3 TrainImage / SimStack（环境模块）

```yaml
# 镜像
name: eb/rlinf:0.4.0-emb
digest: sha256:...
size_bytes: ...
status: ready | missing | pulling
role: train | sim | tools
notes: 内含 Isaac Lab + RLinf

# 仿真栈（本版固定一条）
id: isaaclab
display_name: Isaac Lab
default_image: eb/rlinf:0.4.0-emb
```

本版不建模「多种仿真器插件」。第二条仿真栈出现之前，环境页就是 Isaac Lab + 镜像表。

### 4.4 TrainJob（训练任务，UI 主对象）

```yaml
id: trn_01J...
name: bhl-wbc-ppo-4gpu
status: queued | running | succeeded | failed | cancelled
image: eb/rlinf:0.4.0-emb
embodiment_id: emb_berkeley_humanoid_lite
template: berkeley_wbc_ppo_mlp
hydra_overrides:
  - runner.max_epochs=200
  - env.train.total_num_envs=1024
  - cluster.component_placement.actor,env,rollout=0,1,2,3
mounts: [...]
resources: { gpus: 4, shm_gb: 32 }
metrics_summary: { reward: 12.4, step: 180 }
error: null
run_path: /var/lib/eb/runs/trn_01J...
```

内部若仍拆 Job/Run：UI 和 API 对用户只暴露 TrainJob；重试产生新 attempt 可在详情里看到，不必先建 Experiment。

现行 `experiments` 表本版可以停用。

### 4.5 Model

```yaml
id: mdl_pi05_base
name: pi0.5 base
kind: base | trained
path: /var/lib/eb/models/mdl_pi05_base
parent_id: null              # trained 时指向基座
train_id: null               # trained 时指向 TrainJob
dataset_ids: []
embodiment_id: emb_berkeley_humanoid_lite
```

血缘边至少三类：`trained_from`（模型→模型）、`produced_by`（模型→训练）、`used`（训练→数据/本体/镜像）。

## 5. 下载与扫描也是任务，但不是「训练」

数据下载、镜像 pull、本体扫描会跑一段时间。它们走同一套容器/进程调度，但**不出现在训练列表**。

- 下载：环境模块或数据页上看进度
- pull：环境页镜像行内进度
- 扫描：本体详情「扫描中」

训练列表只放 RLinf（以及将来若加回来的训练类）任务。

## 6. 备份

备份 = Postgres dump + `/var/lib/eb` 目录。外部 `/data/kk/*` 由你自己的盘管。平台不把外部仓复制进 CAS。
