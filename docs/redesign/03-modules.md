# 03 · 五模块详规

每个模块回答三件事：干什么、页面上有什么、元数据怎么记。实现细节以能支撑原型和排期为准，不把现行 G1–G5 / 漏斗 / Bundle 搬回来。

---

## A. 本体

### 目的

登记机器人代码目录，扫出 3D 资产并展示。本体通过**代码在磁盘上的路径**接入，不是填一份 obs/act YAML。

### 接入

唯一必填：`root_path`（可选 display_name；缺省用目录名）。

支持：

- 已存在的本地目录（主路径）。BHL 就是这种方式。
- 不支持本版：上传 zip、平台代 clone、在平台里改上游代码。

点「扫描」后平台做：

1. 在允许前缀内解析绝对路径，拒绝写入探测（只读 walk）
2. 找出 `*.urdf` / `*.mjcf` / `*.usd` / 常见 mesh（stl/obj/dae）
3. 粗提取 URDF 的 link/joint 树，供 3D 查看器
4. 正则/AST 粗扫 `gym.register` 与 RLinf `env/*.yaml` 里的 `init_params.id`
5. 写 `assets[]`、`gym_ids[]`、可选 `interface_snapshot`

扫失败不删登记，状态 `error` + 原因。可重扫。

### 页面

**列表**：卡片。名称、路径、资产数、可训 gym 数、状态。入口「登记本体」。

**接入页**：路径输入、允许前缀提示、扫描日志。成功跳详情。

**详情**：

- 左：3D（沿用现有 UrdfStage；无 URDF 则空态「有代码、无网格」）
- 右：路径、扫描时间、资产树、gym id 列表
- 不提供「发布本体环境」「草稿 YAML」「G 矩阵编辑」

### 与训练的关系

训练表单选一个 `ready` 的本体 → 只读挂载 `root_path`。gym id 作为「仿真任务」下拉的数据源之一（另一源是 RLinf 仓里的 env yaml）。

---

## B. 数据

### 目的

从指定来源取数，落到指定目录，并能查看里面有什么。

本版**不做**六层漏斗、许可门禁、切片构造器、转换向导。若目录恰好是 LeRobot / 带视频的 episode，详情里用现有预览能力；否则就是文件浏览器。

### 来源

| kind | uri 例子 | 行为 |
|---|---|---|
| `path` | `/data/kk/berkeley_ego_data` | 只登记，不拷贝（默认）。可选「复制到 dest」 |
| `huggingface` | `repo/id` + revision | 下载到 dest |
| `http` | URL（目录清单或归档） | 下载/解压到 dest |

`dest_path` 默认 `/var/lib/eb/datasets/<id>`，可改到允许前缀内任意目录。

### 页面

**列表**：名称、来源、路径、大小、状态（ready / downloading）。入口「接入数据」。

**接入页**：来源类型、URI、目标目录、开始。进度条来自下载任务，不进训练列表。

**详情**：

- 元数据头（路径、字节、格式猜测）
- 文件树（目录可展开）
- 若识别为 episode 集：列表 + 点开视频/曲线（沿用 EpisodePreview 的能力，降级为「能播则播」）
- 没有漏斗图、没有许可徽章作为主信息

### 与训练的关系

表单可多选 ready 数据集 → 只读挂载。RLinf SFT 类模板需要数据；纯仿真 PPO（WBC）可以不选。

---

## C. 环境

### 目的

管**训练用的仿真运行时**。本版就是 Isaac Lab，外加镜像管理。

这不是现行的「本体环境」。Isaac Lab 回答「用哪套仿真器、哪张镜像跑」；本体回答「机器人代码在哪」。

### 仿真栈

本版写死一条记录：Isaac Lab。页面展示：

- 默认训练镜像（通常是带 Isaac Lab 的 RLinf 镜像）
- 从已登记本体 + RLinf 仓扫描到的 gym id / env yaml（只读缓存）
- 说明：RLinf 训练容器里跑 Isaac Lab，不要求用户先理解两套栈

不提供「新建仿真器」。

### 镜像管理

对宿主机 Docker 镜像做目录，不在平台里造镜像（`make images` / 外部构建仍是运维动作）。

页面表格列：`name:tag`、digest 短码、大小、状态（本机有 / 缺失 / 拉取中）、角色（train / sim / tools）。

动作：

- 刷新（docker images）
- 拉取（对缺失项；长任务，行内进度）
- 设为 Isaac Lab 默认训练镜像

本版预置关注：

- `eb/rlinf:0.4.0-emb`（主训练镜像）
- `eb/isaac-lab:2.3.2`（若仍单独存在，标 sim）
- 不再把 mujoco / tools 作为环境页重点，设置里可以看见

现行 `/system/images` 返回四条字符串，需要改成带 digest/状态的真实探测（ebd 已持有 docker.sock）。

### 与训练的关系

新建训练时镜像下拉只来自环境页 `ready` 的 train 角色镜像。提交时把 digest 写入 TrainJob，避免 tag 漂。

---

## D. 训练

### 目的

适配 RLinf：填配置、开训、看当前与历史任务（日志、指标、结果）。

这是本版的产品中心。原「实验」页应变成这个列表，而不是改个标题继续空表。

### 列表（默认页）

一行一个训练任务。列：

- 名称 / id
- 状态（queued / running / succeeded / failed / cancelled）
- 模板（如 `berkeley_wbc_ppo_mlp`）
- 本体
- 镜像
- GPU
- 最近指标（若有）
- 起止时间

顶栏：「新建训练」。过滤：状态、本体。不按 Experiment 分组。

队列不再单独成导航：`queued` + `running` 就是队列。总览上的 GPU 占用与此同源。

### 新建：RLinf 表单

不要把 Hydra 全树铺开。三层：

1. **模板**（决定 defaults 组合）
2. **常用字段**（人真正会改的）
3. **高级 overrides**（自由 YAML / 多行 `a.b=c`）

本版内置模板（均来自现有 RLinf 仓，路径可配）：

| 模板 id | 源 yaml | 用途 |
|---|---|---|
| `berkeley_wbc_ppo_mlp` | `examples/embodiment/config/berkeley_wbc_ppo_mlp.yaml` | 全身跟踪 PPO |
| `berkeley_reach_sft` | `examples/sft/config/berkeley_reach_sft_openpi_pi05.yaml` | Reach SFT |
| `collect_berkeley_reach` | `examples/embodiment/config/collect_berkeley_reach.yaml` | 采集（若本版要收数据；可后做） |

常用字段分组：

**运行**

- 任务名称
- 镜像（默认环境页的 train 镜像）
- GPU 数（1–4），生成 `cluster.component_placement`
- 超时小时

**代码与仿真**

- 本体（必选）
- 仿真任务：gym id 或 `env/isaaclab_berkeley_wbc.yaml` 这类 RLinf env
- 栈代码路径：默认 `/data/kk/RLinf`（设置可改）

**数据与模型**（按模板显示）

- 数据集（SFT 必选，PPO 可选）
- 基座模型路径 / 已登记模型
- `runner.ckpt_path` / `actor.model.model_path`

**算法（暴露最少集）**

- `runner.max_epochs`
- `env.train.total_num_envs`
- `actor.optim.lr`
- `runner.save_interval`
- `runner.logger.experiment_name`
- seed

其余保持模板默认。预览区实时显示：

- 将生成的 Hydra overrides
- 将使用的挂载表
- 等价 `docker run`（只读，便于你在宿主机复现）

提交后进列表为 queued；调度到卡后 running。

Isaac Lab + RSL-RL 的 `train.rl` **本版表单不提供**。那条路径代码可留，但不在 UI 出现，避免双栈并列。

### 详情

沿用并加强现有 Run 页，作为训练任务详情：

- 状态、GPU、耗时、镜像 digest、本体路径
- 指标图（tfevents → 现有 TSDB/Chart）
- 日志（stdout，可过滤）
- 产物：`output/` 文件列表；ckpt 可「登记为模型」
- 冻结的 hydra.yaml 与 spec.json
- 动作：取消（running）、重试（用同一 spec 新建一条）、登记产物

不在本版做多任务显著性对比页；若要比，让用户开两个详情。对比页可后做。

### 失败怎么说

把容器退出码、OOM、镜像不存在、路径不存在，直接写在任务 `error` 上。不要用「infra_failed」这类内部词当唯一文案。镜像缺失时链到环境页对应行。

---

## E. 模型

### 目的

登记两类权重，并记下它们怎么来的：

- **原始 / 基座**：你放在磁盘上的（π₀.₅、已有 ONNX 等）
- **训练产物**：某次 TrainJob 的 output

### 接入基座

表单：名称、`path`（允许前缀内）、可选关联本体。不强制测 IO 维、不强制 ModelCard 齐套。能扫到文件大小即可。

### 训练产物

训练详情点「登记为模型」：指定 `output/` 下某个子目录或文件，写入 `kind=trained`、`train_id`、`parent_id`。

也可以约定：训练成功且模板声明了 `outputs.checkpoints` 时自动登记 latest。本版建议**手动登记**，避免 60 个 ckpt 全进列表；列表提供「仅 latest」过滤即可后补。

### 页面

**列表**：名称、kind 徽章（基座 / 训练）、路径、来源训练、本体。

**详情**：路径与文件、血缘（父模型 ← 本模型 ← 子模型，中间标注 train_id）、用于哪些后续训练。

3D 资产不属于模型模块（在本体）。

### 与训练的关系

表单选基座 → 只读挂载。新产物登记后可被下一次训练选中，血缘自然延长。

---

## F. 总览与设置（附属，不是第六业务模块）

**总览**：GPU 占用、running 训练、磁盘水位（`/var/lib/eb` + 可选登记路径合计）、最近失败。

**设置**：允许路径前缀、RLinf 仓路径、默认镜像、数据根。不做 Agent 预算、门禁豁免、通知渠道。
