# 12 · 本体与本体环境

> 本章是「契约 → 本体环境」替换后的领域真源。  
> 02 / 09 / 11 的契约段落以本章为准；旧词 `EmbodimentContract` / `ec_*` / `contract_id` 仅为兼容别名。  
> 实现未完成前，代码仍可能使用旧名；冲突时改代码，不改回本章。

**状态**：设计已定（2026-08-24）  
**不写** `/data/kk/Berkeley-Humanoid-Lite`、`/data/kk/RLinf`、`/data/kk/berkeley_ego_data`

---

## 0. 为什么换掉「契约」

旧名 Embodiment Contract 只锁住了 obs / act / Hz。真实工作是在管：

1. **本体资产**：URDF / MJCF / USD / mesh  
2. **本体定义**：关节顺序、限位、控制律、接口数字  
3. **基本执行逻辑**：代码从哪来、Gym / Hydra 入口怎么指到这份身体  

「契约」不像在管这些；接 RLinf、接第二台机器人时更对不上。平台仍做**门禁与可视化治理**，不把上游仓搬进 EmbodiedBein。

已拍板：

| 项 | 决定 |
|---|---|
| 分层 | 本体 + 本体环境 两层；Job 只引用本体环境 |
| 任务 | 独立实体。提交 = 本体环境 × 任务 × 训练栈 × 镜像 |
| 版本 | 草稿可改；发布后该版本不可变；改数字就克隆 `_v2` |
| 文档 | 本章为真源；02 / 09 / 11 / index 打补丁；其余章加对照 |
| 来源 | 五种混合来源；BHL 是 `git_external` 范本 |
| BHL | **纳管，不迁仓、不改文件** |

---

## 1. 实体与命名

平台不再把 Embodiment Contract 当作一等实体。拆成三样：

| 中文 | 英文 | ID | 职责 | 谁引用 |
|---|---|---|---|---|
| **本体** | `Embodiment` | `emb_` | 机器人品牌/家族：来源、3D 资产、代码入口 | 本体环境、资产版本、机型 |
| **本体环境** | `EmbodimentEnv` | `ee_` | 一种可跑接口：关节、obs/act/Hz、控制律、仿真/真机绑定 | Job、ModelVersion、DatasetVersion、Bundle |
| **任务** | `Task` | `tsk_` | 在该接口上要做的事：Velocity、Reach、LIBERO-10… | 训练/评估 Job（与环境和栈一起选） |

G1–G5 仍在，比对对象改成本体环境。Agent 不能改已发布版本、不能豁免门禁。

**不要和 RLinf 的 `env/` 混用。** Hydra 里的 `env/isaaclab_berkeley_reach.yaml` 是任务在某个栈上的入口，登记在 `Task.stack_bindings.rlinf`，不是本体环境。

数据域已有的 `Scenario`（reach/push 场景编译）也不是 Task。两边可以共用 `task_key`（如 `reach`）；Scenario 管漏斗，Task 管怎么启动一次训练/评估。

`RobotAsset` / `Calibration` / `RobotInstance` 仍在本体域，父级从「某个契约」改为「某个本体」。G5 比对的自由度来自**该次部署选用的本体环境**。

### 1.1 引用约束（替换 02 章①②）

1. 任何 Job 必须引用恰好一个**已发布**的 `embodiment_env_id`。训练/评估类 Job 还须引用恰好一个与它兼容的 `task_id`。  
2. `ModelVersion.embodiment_env_id` 必填。没有本体环境的权重不能进注册表，只能当裸 Artifact。  
3. 已发布本体环境不可原地改 `spec_hash`；要改数字就克隆 `_v2`。从未发布的草稿可删。  
4. PolicyBundle 仍须通过门禁且 artifact 校验和一致（原④）。  
5. Deployment 仍须签名 Bundle + 已标定 RobotInstance + Approval（原⑤）。

### 1.2 ID 与兼容别名

| 前缀 | 规则 | 例 |
|---|---|---|
| `emb_` | 人工语义名，稳定 | `emb_berkeley_humanoid_lite` |
| `ee_` | 人工语义名 + `_vN` | `ee_bhl_humanoid_22dof_v1` |
| `tsk_` | 人工语义名 | `tsk_bhl_reach` |
| `ec_*` | **只读别名** → 对应 `ee_*` | 旧 Job / OpenAPI 不立刻断 |

新写入一律用 `ee_` / `embodiment_env_id`。读取 `contract_id` 与 `ec_*` 仍认。

### 1.3 BHL 在平台里的形状（仓内一字不改）

```
emb_berkeley_humanoid_lite          source.kind = git_external
│                                   path = /data/kk/Berkeley-Humanoid-Lite
├── 资产指针 → 仓内 urdf / mjcf / usd / meshes（不是副本）
├── ee_bhl_humanoid_22dof_v1        75 / 22 @ 25 Hz     ← 原 ec_bhl_humanoid_22dof_v1
├── ee_bhl_biped_12dof_v1           45 / 12 @ 25 Hz
├── ee_bhl_wbc_22dof_v1             85 / 22 @ 50 Hz（双 action term）
└── ee_bhl_fixedhand_10dof_v1       10+img / 10 @ 20 Hz（concat: false）

tsk_bhl_velocity_humanoid   兼容 ee_bhl_humanoid_22dof_v1
                            Isaac: Velocity-Berkeley-Humanoid-Lite-v0
tsk_bhl_velocity_biped      兼容 ee_bhl_biped_12dof_v1
                            Isaac: Velocity-Berkeley-Humanoid-Lite-Biped-v0
tsk_bhl_wbc_tracking        兼容 ee_bhl_wbc_22dof_v1
                            Isaac: Tracking-Berkeley-Humanoid-Lite-WholeBody-v0
                            RLinf: examples/embodiment/config/berkeley_wbc_ppo_mlp.yaml
                                   （defaults 已 include env/isaaclab_berkeley_wbc）
tsk_bhl_reach               兼容 ee_bhl_fixedhand_10dof_v1
                            Isaac: Isaac-Berkeley-Humanoid-Lite-FixedHand-Reach-v0
                            RLinf: env/isaaclab_berkeley_reach.yaml
tsk_bhl_push                兼容 ee_bhl_fixedhand_10dof_v1
                            Isaac: Isaac-Berkeley-Humanoid-Lite-FixedHand-Push-v0
```

四份接口 YAML **已经在 EmbodiedBein `contracts/`**，本来就不在 BHL 里。改名前缀只动平台文件。BHL 的 `tasks/manipulation/contracts.py` 也不动；平台环境快照去对齐它。

---

## 2. BHL 只纳管、不迁仓

`/data/kk/Berkeley-Humanoid-Lite` 不搬家、不改文件、不强制加 `eb.embodiment.yaml`。平台登记「这是谁、接口数字、训练时只读挂哪」。执行逻辑和资产继续以 BHL 为真源。

硬约束不变：compose bind `/data/kk`；平台**不得写入** BHL / RLinf / ego_data。`play.py` 写 `configs/`、`train.py` 写 `logs/` → 只落到 `/var/lib/eb/runs/<run_id>/`。跑完 `git -C /data/kk/Berkeley-Humanoid-Lite status` 必须干净。

纳管输入（不是搬迁清单）：分支 `feat/isaac5` @ `6ff398f`；子模块 assets `v2025.09.03`、lowlevel 在 `feat/isaac5`。commit 以提交 Job 时冻结值为准。

### 2.1 两套真源

| 东西 | 真源 | 平台 |
|---|---|---|
| Isaac 任务、obs 拼接、奖励、Gym id | BHL `source/berkeley_humanoid_lite/` | 只读挂进容器 |
| URDF / MJCF / USD / mesh | BHL `source/berkeley_humanoid_lite_assets/` | 扫描路径与哈希；3D 预览读文件 |
| 真机 lowlevel | BHL `source/berkeley_humanoid_lite_lowlevel/` | 部署 Job 只读引用 |
| 关节 / obs / act / Hz / G1–G5 | **平台**本体环境 YAML | 治理快照；与 BHL 对不上则门禁失败，不改 BHL |
| 任务入口 | **平台** Task 表 | 指向 BHL 已有 `gym.register` 或 RLinf 已有 `env/*.yaml` |

训练用 git worktree 快照到 `/var/lib/eb/repos/bhl/<commit>`。这是平台缓存，不是把 BHL 迁走。禁止用脏工作树开训。

### 2.2 规范加在平台；BHL 豁免清单文件

新的 `git_external` / `git_snapshot` **建议**仓根有 `eb.embodiment.yaml`。BHL 没有也可以纳管。扫描器认：

| 规范项 | BHL 已有落点 | 平台要求 |
|---|---|---|
| 可解析的机器人描述 | `…/urdf/*.urdf`、`mjcf/*.xml`、`usd/` | 资产扫描 + 关节名集合 |
| 可发现的任务入口 | 四处 `gym.register` | 登记为 Task 草稿，人确认后发布 |
| 接口数字可核对 | `env_cfg`、`play.py` 导出 YAML、`contracts.py` | 发布环境时跑断言；对不上拒绝发布 |
| 可复现 | git commit + 子模块 sha | Job 冻结 `code_refs.embodiment` |

### 2.3 明确不做

- 不把 BHL 拷进 `EmbodiedBein/robots/` 或 MinIO 当新真源  
- 不在 BHL 插入 `eb.embodiment.yaml`、不改 `env_cfg.py`、不重排关节  
- 不把四份平台 YAML 写回 BHL  
- 不把 RLinf 的 `examples/embodiment/config/env/*.yaml` 搬进平台仓  

接口要演进：人在 **BHL 仓**改代码并提交 → 平台扫到新 commit → 克隆 `ee_*_v2` 草稿 → 核对后发布。

---

## 3. 五种来源

来源写在**本体**上。同一本体一种来源；换来源等于换本体（或先归档再新建）。本体环境始终是平台治理对象。

| `source.kind` | 范本 | 代码 | 资产 | 平台可写 | 默认能力 |
|---|---|---|---|---|---|
| `git_external` | BHL、RLinf 仓 | 外部 git，只读 | 仓内相对路径 | 否（含 `/data/kk/*`） | 预览；任务绑定齐全则可训练 |
| `git_snapshot` | 自有新 git | clone 到 `/var/lib/eb/repos/<emb>/` | 同上 | 只写平台缓存 | 同上 |
| `managed_bundle` | 只丢了 URDF 的新机器 | 无 | 上传 CAS | 是 | **默认可预览、不可训练** |
| `in_image` | 镜像内 LIBERO / ManiSkill | 镜像 FS | 镜像内 | 否 | 训练钉 digest |
| `plugin` | `eb_plugin_*` | 插件包 | Manifest | 随插件版本 | 插件声明能力矩阵 |

能力徽章（列表/详情必须有）：`preview` / `train` / `eval` / `deploy`。没有 Task 栈绑定就不要点亮 `train`。`managed_bundle` 新建只有 `preview`——这是设计，不是缺陷。

### 3.1 `git_external`

```yaml
id: emb_berkeley_humanoid_lite
source:
  kind: git_external
  path: /data/kk/Berkeley-Humanoid-Lite
  writable: false
  default_ref: HEAD          # 提交 Job 时解析并冻结成 commit
  submodules:
    - source/berkeley_humanoid_lite_assets
    - source/berkeley_humanoid_lite_lowlevel
  worktree_cache: /var/lib/eb/repos/bhl/<commit>
```

接入步骤：路径存在且是 git → 记录 commit 与子模块 sha（不改远程）→ 扫资产哈希 → 扫 `gym.register` 成 Task 草稿 → 挂上平台已有四份 `ee_*` → 跑关节名 ⊆ URDF。失败则环境保持草稿，不去改 BHL。

`/data/kk/RLinf` 同样 `git_external`，但登记的是**训练栈代码**，不是第二个本体。本体仍是 BHL；Task 的 RLinf 绑定指向 RLinf 仓内已有 yaml。两个外部仓都只读，与 `compose.berkeley-openpi.yaml` 的双 bind 一致；平台用冻结 commit 的快照，不用活目录。

### 3.2 其余四种（摘要）

- **git_snapshot**：平台 clone；YAML 不 push 回远程；UI 改接口 = 克隆 `ee_*_v2`，下载 YAML 给人去原仓开 PR。  
- **managed_bundle**：zip 进 CAS；YAML 在 UI 编；没有栈绑定就禁用「提交训练」。v1 不承诺通用 `Isaac-LoadUrdf-v0`。  
- **in_image**：Job 钉 digest。平台不把镜像内文件抽成可编辑 YAML。BHL 的 Reach/WBC **不要**标 `in_image`（逻辑在 BHL）。  
- **plugin**：v1 不阻塞；BHL 不靠插件也能纳管。

接入向导按 `kind` 分支。失败要可定位：不可读、不是 git、子模块未 init、URDF 解析失败、Gym id 为零（允许，但不点亮 `train`）。

---

## 4. 工作台 CRUD、YAML、3D

导航「契约」改为「本体」。`/contracts`、`/contracts/:id` 301 到本体列表 / 对应环境。`ec_*` 仍能打开。`/assets` 并进本体详情。

### 4.1 路由

| 路由 | 内容 |
|---|---|
| `/embodiments` | 按本体聚合的卡片：来源、环境个数、能力徽章 |
| `/embodiments/new` | 接入向导 |
| `/embodiments/:id` | 概览、3D、资产表、来源、环境子卡片 |
| `/embodiments/:id/envs/:eeId` | 观测条带、关节树、频率、G 矩阵、YAML |
| `/train/submit` | 本体 → 已发布环境 → 兼容任务 → 栈/镜像 |

可点击全套 CRUD 原型：[13-embodiment-crud-prototype.html](13-embodiment-crud-prototype.html)（本体 / 环境 / 任务的增删改查 + 五种来源向导 + 提交四元组）。

BHL 四套接口**不要**再平铺成四个互不相关的列表卡片。

### 4.2 CRUD 规则

**本体**：向导创建；显示名/标签可改；`source.kind` 与 `git_external.path` 不可改；可改 `default_ref`、可只读重扫；无被引用的已发布环境才能删，否则归档；BHL 种子不可删。

**本体环境**：必须挂在某个本体下（空白草稿或克隆）。草稿可改可删。发布：断言全过且 `joint_order` ⊆ 资产关节名，然后冻 `spec_hash`。已发布只能「克隆为 `_v2`」。废弃后旧 Run 仍可复现，新 Job 默认不选。

**任务**：扫描结果先草稿，人确认绑定后发布。绑定 = `compatible_env_ids` + 各栈入口。不在平台编奖励或观测拼接。

保存草稿只写 PostgreSQL，不写 `/data/kk/*`。提供「下载 YAML」给人开 PR。

旧 11.3「不可在 UI 新建或改数字」作废，改为：**不可改已发布数字**。

### 4.3 YAML 编辑（仅草稿）

已发布：11.3 的可视化 + YAML 只读折叠 + `spec_hash`。

草稿：左表单右 YAML 同一份 spec。

| 区 | 表单 |
|---|---|
| 关节 | `joint_order` 拖拽；从 URDF 勾选子集（biped 去掉臂） |
| 观测 | 条带调 dim、增删 term；`concat:false` 单独图像卡 |
| 动作 | 单 term 或 WBC 式 term 列表 |
| 控制 | Hz、dt、decimation、kp/kd、scale |
| 绑定 | 仿真/真机端口；不写回 BHL |

发布前：原自洽断言；关节名 ⊆ URDF/MJCF；WBC 双 term 维数之和 = dof；fixed-hand `concat:false` 且图像 shape 完整；`git_external` 资产哈希与索引一致。

### 4.4 3D

本体页 **P0** 主视图，不是 09 里那个 P2 `/assets/:id/viewer`。

| 格式 | 展示 | 禁止 |
|---|---|---|
| URDF + STL | 浏览器 three.js 加载运动学 + 视觉 mesh | 不把 mesh 拷进前端仓 |
| MJCF | 一致性徽章；形体外观仍走 URDF/STL | v1 不嵌 MuJoCo WASM |
| USD | 卡片：路径、大小、sha256 | **不在浏览器解析 USD** |
| 缺 mesh | 文案 + URDF collision 盒线框 | 不用纯色块冒充机器人 |

媒体：`GET /api/v1/embodiments/{id}/assets/file?path=` → ebd 从只读根流式输出。路径必须在该本体资产根下，否则 403。

与环境联动：选 biped 只高亮 12 腿关节、臂半透明；humanoid/WBC 22 轴；fixed-hand 10 臂。草稿拖拽顺序，3D 序号跟着变。滑条范围 = URDF limit ∩ YAML `limits`，冲突标红。可切 visual / collision。

可选非阻塞：`asset.preview` 用 Isaac 对 USD 拍静帧进 CAS。无镜像就写「无预览」，不许假图。

---

## 5. 与 RLinf 的交互

一次可提交的 RLinf Job 是四元组，缺一不可：

**已发布本体环境 × 兼容任务 × 栈=`rlinf` × 镜像 digest**

平台不根据 YAML 生成假的 `env/*.yaml`。Hydra 入口来自 **Task.stack_bindings**，文件仍在 RLinf 仓。

### 5.1 UI 路径（BHL Reach）

1. 本体 `emb_berkeley_humanoid_lite`  
2. 环境 `ee_bhl_fixedhand_10dof_v1`  
3. 任务 `tsk_bhl_reach`  
4. 栈 RLinf → `eb/rlinf:0.4.0-emb@sha256:…`（只写 tag 则提交时冻结 digest）  
5. seed、卡数、数据集/基座 → G1 → 提交  

同一环境换 Isaac 只换栈和镜像。WBC：环境 `ee_bhl_wbc_22dof_v1`，任务绑定 `berkeley_wbc_ppo_mlp`（其 defaults 已 include `env/isaaclab_berkeley_wbc`，Gym id `Tracking-Berkeley-Humanoid-Lite-WholeBody-v0`）。

### 5.2 JobSpec

`contract_id` → `embodiment_env_id`（读入仍认旧字段）。训练/评估必带 `task_id`。RLinf+BHL 要**两份**代码冻结：

```yaml
spec:
  kind: train.rlft                 # 或 train.sft
  embodiment_env_id: ee_bhl_wbc_22dof_v1
  task_id: tsk_bhl_wbc_tracking
  stack: rlinf
  image: eb/rlinf:0.4.0-emb@sha256:…
  code_refs:
    embodiment:
      kind: git_external
      path: /data/kk/Berkeley-Humanoid-Lite
      commit: 6ff398f
    stack:
      kind: git_external           # 代码已烤进镜像则 kind: in_image，只钉 digest
      path: /data/kk/RLinf
      commit: "…"
  params: {}                       # config_name 默认来自 Task，可覆盖
  runtime:
    ray: { port: 6390 }
  resources: { gpus: 2, shm_gb: 32 }
```

`params.task` / `params.config_name` 不再当人填的真源。`overrides[]` 仍可透传 Hydra，但不得改观测维、动作维、关节顺序。

任务绑定示例：

```yaml
id: tsk_bhl_wbc_tracking
compatible_env_ids: [ee_bhl_wbc_22dof_v1]
stack_bindings:
  isaaclab_rslrl:
    gym_id: Tracking-Berkeley-Humanoid-Lite-WholeBody-v0
    workdir: /workspace/isaaclab_extension_template
  rlinf:
    config_path: examples/embodiment/config
    config_name: berkeley_wbc_ppo_mlp
    script: examples/embodiment/train.py    # 以 RLinf 仓实际入口为准，Adapter 钉死
    embodied_path: examples/embodiment
```

LIBERO 等：本体 `in_image`，`code_refs.embodiment` 可空，只钉镜像。不把 LIBERO 挂到 BHL 名下。

### 5.3 容器挂载

| 层 | 宿主机 | 容器 | 权限 |
|---|---|---|---|
| 栈代码 | `/var/lib/eb/repos/rlinf/<commit>` 或镜像内 `/opt/RLinf` | `/workspace/rlinf` | ro |
| 本体 | `/var/lib/eb/repos/bhl/<commit>` | `/workspace/berkeley` | ro |
| 本 Run | `/var/lib/eb/runs/<run_id>/workspace` | 覆盖 outputs / results / log_path | rw |
| 数据/权重 | CAS | `/mnt/data`、`/mnt/models` | ro |
| 缓存 | HF / torch | 共享 rw + flock | rw |

环境变量：`PYTHONPATH` 含 BHL 三个 `source/*` 包（与现 Isaac 适配器同一理由：只读仓不能 `pip install -e`）；`EMBODIED_PATH=/workspace/rlinf/examples/embodiment`；`MUJOCO_GL=egl`；`EB_EMBODIMENT_ENV_ID` / `EB_TASK_ID` 给日志。

启动：`ray start --head --port 6390` → `python <script> --config-path … --config-name <Task 绑定>`。禁止 Ray 用 6379。`runner.logger.log_path` 指向可写层；强制 tensorboard 便于采集。

Isaac 栈同一四元组，无「栈仓」挂载：只挂 BHL 快照到 `/workspace/isaaclab_extension_template`。

### 5.4 G1 增量与 Adapter 禁区

在原 G1 上增加：环境已发布且 `train` 亮；`task_id` 兼容；Task 有 `stack_bindings.rlinf`；镜像能力矩阵匹配；embodiment commit 干净或只用缓存；SFT 数据集 `compatible_envs`；基座 `embodiment_env_id` 相同；overrides 不得改接口数字。

Adapter **禁止**：把平台 YAML 渲成 RLinf env yaml 覆盖仓内文件；写两个外部仓；容器内对外部仓 `pip install -e`；Ray 走平台 Redis。

运行时观测对不上 `ee_*` → Run 失败 + 门禁失败。修复在上游仓或发 `_v2`，Adapter 不改数字。

### 5.5 失败归因

| 现象 | 归因 |
|---|---|
| `ModuleNotFoundError: berkeley_humanoid_lite` | 快照 / PYTHONPATH / 子模块 |
| Gym id 未注册 | Task 绑定 id 在该 BHL commit 不存在；列出扫描到的 register diff |
| Hydra 找不到 `env/isaaclab_berkeley_wbc` | RLinf commit 无该 yaml |
| Ray 连上平台 Redis | 未注入 6390 |
| 维数与 `ee_*` 不符 | G3，禁止注册模型 |
| `Read-only file system` | `writable_paths` 漏了；禁止 chmod 外部仓 |

Run 详情的等价 docker 命令必须能看出：两个 commit、两处 ro 挂载、Ray 端口、config-name 来自任务。

---

## 6. REST、schema、状态机

### 6.1 资源树（相对 `/api/v1`）

```
embodiments/
  POST                    接入（body 含 source.kind）
  GET                     列表
  GET    {id}
  PATCH  {id}             显示名/标签/default_ref；禁改 kind/path
  POST   {id}/scan        只读重扫
  POST   {id}/archive
  GET    {id}/assets
  GET    {id}/assets/file?path=   → ebd 流式
  GET    {id}/preview             3D 载荷（关节树 + mesh URL）
  POST   {id}/envs                新建草稿环境
embodiment-envs/
  GET    {id}
  PATCH  {id}             仅 draft
  PUT    {id}/yaml        仅 draft
  POST   {id}/publish
  POST   {id}/clone       产生 _vN+1 草稿
  POST   {id}/deprecate
  GET    {id}/validate    断言 + 与资产关节名对照
  GET    {id}/refs        Job / 模型 / 数据集引用方
tasks/
  GET · POST
  GET/PATCH {id}
  POST {id}/publish
contracts/                兼容：GET 列表/详情代理到 embodiment-envs
  {id}/validate           同 ee 的 validate
```

鉴权与现 API 相同。Agent 无 `publish` / 改 `active` / `gate.override`。

### 6.2 表（示意；Alembic 落地）

```sql
CREATE TABLE embodiments (
  id            text PRIMARY KEY,           -- emb_berkeley_humanoid_lite
  display_name  text NOT NULL,
  source        jsonb NOT NULL,             -- kind/path/image/plugin
  status        text NOT NULL DEFAULT 'active',  -- active | archived
  created_at    timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE embodiment_envs (
  id              text PRIMARY KEY,         -- ee_bhl_humanoid_22dof_v1
  embodiment_id   text NOT NULL REFERENCES embodiments(id),
  spec            jsonb NOT NULL,
  spec_hash       text NOT NULL,
  status          text NOT NULL DEFAULT 'draft',  -- draft | active | deprecated
  obs_dim         int GENERATED ALWAYS AS ((spec->'observation'->>'dim')::int) STORED,
  action_dim      int GENERATED ALWAYS AS ((spec->'action'->>'dim')::int) STORED,
  policy_hz       int GENERATED ALWAYS AS ((spec->'control'->>'policy_hz')::int) STORED,
  aliases         text[] NOT NULL DEFAULT '{}'    -- 含 ec_* 
);

-- 过渡期可保留 embodiment_contracts 为 embodiment_envs 的视图
-- jobs.contract_id 重命名为 embodiment_env_id；过渡列双写

CREATE TABLE tasks (
  id                   text PRIMARY KEY,
  spec                 jsonb NOT NULL,      -- compatible_env_ids, stack_bindings
  status               text NOT NULL DEFAULT 'draft',
  spec_hash            text NOT NULL
);

CREATE TABLE robot_assets (
  id              text PRIMARY KEY,
  embodiment_id   text NOT NULL REFERENCES embodiments(id),
  version_label   text NOT NULL,            -- bhl-hw-v1.2
  paths           jsonb NOT NULL,           -- urdf/mjcf/usd 相对路径
  content_hash    text NOT NULL,
  UNIQUE (embodiment_id, version_label)
);
```

`jobs.embodiment_env_id` NOT NULL REFERENCES `embodiment_envs(id)`。训练/评估 Job 的 spec 内 `task_id` 由应用层校验兼容性。

### 6.3 本体环境状态

`draft` → `active`（publish，spec 冻结）→ `deprecated`。  
`draft` 可回编辑、可删。`active` 只能 clone 或 deprecate。历史 Run 永远指向创建时的 `ee_*`。

---

## 7. 门禁与 Agent

G1–G5 语义不变，主语从契约改成本体环境：

| 卡口 | 时机 | 对象 |
|---|---|---|
| G1 | 提交 Job | env × task × 数据集 × 基座 × 镜像能力 × 双 code_ref |
| G2 | 数据集发版 | 样本维 / fps / 关节名 ⊆ 目标 `ee_*` |
| G3 | 模型注册 | ONNX I/O vs `ee_*` |
| G4 | 导出签名 | 部署 YAML 全字段 vs `ee_*` |
| G5 | 上机 | 固件关节数 vs 选用的 `ee_*`；标定未过期 |

工具禁区改名（结构禁止，不是审批）：

| 工具 | 状态 |
|---|---|
| `embodiment_env.publish` / 改 `active` spec | 永久禁止 |
| `embodiment_env.create_draft` / `clone` | 允许（若开放）；发布仍须人 |
| `embodiment.scan` | 允许（只读） |
| `gate.override` | 永久禁止 |
| `deploy.release` | 永久禁止 |

「改契约」三条红线改为「改已发布本体环境」。草稿不算考卷。

---

## 8. 词表对照

| 旧（仍可能出现在 01–11 未改段落） | 新 |
|---|---|
| Embodiment Contract / 契约 | 本体环境 `EmbodimentEnv` |
| `ec_bhl_*` | `ee_bhl_*`（`ec_*` 只读别名） |
| `contract_id` | `embodiment_env_id` |
| `/contracts` | `/embodiments` |
| 「契约约束一切」 | 「已发布本体环境约束接口数字」 |
| 四个契约卡片 | 一个本体卡片 + 四个环境 |
| `code_ref`（单仓） | `code_refs.embodiment` + `code_refs.stack` |
| `gate_contract_basic` | `gate_env_basic`（旧 id 别名） |
| Agent `contract.create/modify` | `embodiment_env.publish` / 改 active |

「OpenAPI 契约」「JobSpec 是控制面与执行层之间的唯一契约」这类**软件接口**含义保留，不改成本体环境。

---

## 9. 实现对照（设计约束，不是进度）

v0.1 骨架仍是 `contracts/*.yaml` + 表 `embodiment_contracts` + 路由 `/contracts`。落地本章时：

1. 种子：一个 `emb_berkeley_humanoid_lite` + 四份 YAML 改挂 `ee_*`，`ec_*` 进 `aliases`。  
2. 扫描器只读 BHL，不写。  
3. 前端：列表改聚合；详情保留观测条带并加 3D；草稿编辑。  
4. JobSpec 双字段读、新字段写。  
5. RLinf Adapter：双挂载 + PYTHONPATH + Ray 6390 + config-name 来自 Task。  
6. Agent 禁区换名，语义更严（禁止发布，不是禁止看）。

Isaac / RLinf 大镜像、真机、LLM 仍按 10 章路线，不因本章提前绑进 `make up`。

---

## 10. 五轮反思（写进真源的遗漏检查）

1. **BHL 仓内也有一份 `contracts.py`。** 那是 Python 常量，不是平台实体。平台对齐它，不替换它，不要求改名。  
2. **一个 `code_ref` 不够。** RLinf+BHL 必须双冻结，否则「复现五元组」在 VLA 路径上是假的。Isaac 路径 `code_refs.stack` 可空。  
3. **Task 和 Scenario 重名风险。** 文档与 API 禁止用 `env_id` 指本体环境（与 Gym/Hydra env 撞车）；本体环境固定 `embodiment_env_id` / `ee_`。  
4. **3D 若承诺 USD 浏览器预览会烂尾。** v1 只渲 URDF+STL；USD 卡片 + 可选 Isaac 截图。  
5. **`managed_bundle` 若默认可训，G1 会在容器里才爆。** 能力徽章是产品约束。  
6. **补一轮：脏工作树。** 活目录 bind 看似简单，但 `play.py` 会写仓。G1 拒绝脏树，只用 `/var/lib/eb/repos/…/<commit>`。  
7. **再一轮：overrides 改维数。** 必须在 G1 拦，否则发布冻结形同虚设。  
8. **Agent 若连草稿都不能建，接入新本体只能人手。** 允许 clone/草稿，禁止 publish。  
9. **WBC / concat:false 仍须 schema 一等表达**，不能在「可视化 CRUD」里退化成单向量。  
10. **RLinf 自带 LIBERO 不是 BHL 的第五个环境。** `in_image` 另立本体，避免「所有具身都叫 berkeley」。

---

## 11. 刻意不做

- 把 BHL/RLinf 迁进平台仓或改其目录约定  
- 平台生成 RLinf `env/*.yaml` 当执行真源  
- 已发布环境原地改数字  
- 浏览器解析 USD / 嵌入完整 Isaac 视口  
- Agent 发布环境或豁免门禁  
- 为迁就实现改回「契约」一词  
- 把任务大镜像打进 `make up`
