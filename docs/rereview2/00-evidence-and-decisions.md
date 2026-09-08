# 00 · 事实、差距与设计裁决

## 1. 证据口径

- **现场观察**：2026-09-08 UTC，对 `http://1.94.0.106:31800/` 的 GET 请求与前端静态 bundle 核查；首页响应 Date 为 08:08:39 UTC。
- **代码事实**：本机文件内容静态审查，不代表安装环境可运行；指纹见 [local-sources.json](evidence/local-sources.json)。未执行 git，也未导入外部机器人项目触发初始化。
- **历史实测**：`docs/progress/2026-08-21-make-features-real.md` 的 BHL 短训、TSDB、ONNX 证据。本轮未重跑，不将其扩大为 RLinf 全链路验收。
- **设计提议**：其余明确的新增实体、状态、接口、性能目标。

不沿用旧审查中的百分比成熟度、收益百分比与“100%可靠”措辞；这些没有统一测量方法。

## 2. 线上当前状态

| 检查 | 观察 | 能说明什么 / 不能说明什么 |
|---|---|---|
| 首页、`/healthz` | HTTP 200；api/ebd 健康返回 true | 服务可访问；不证明所有计算任务可执行 |
| JS `/assets/index-BWAkfJ98.js` | 总览、本体、数据、环境、训练、模型六项 | 当前是收敛后的工作台；不是早期 13 项导航 |
| `/api/v1/train/templates` | 3 项：BHL WBC PPO、Reach SFT、Reach 采集 | 当前表单仍绑定 BHL |
| `/api/v1/embodiments` | 1 个 BHL，8 个环境记录含 `ec_*` 与 `ee_*` | 要处理历史别名，不能当成 8 个不同机器人形态 |
| `/api/v1/train` | `items=[]` | 新训练工作台暂无记录；不能推断历史 Run 全为空 |
| `/api/v1/datasets` | 1 个 ego 数据集；统计含 17 文件、9 个 `ego_jsonl`；版本 bytes=1,737,803，而顶层 total_bytes=0 | 字段口径分裂；9 不能直接解释为 9 条可训练机器人 episode |
| `/api/v1/envs/doctor` | 4 张 4090，每张 24564 MB；主机内存 128644 MB；Vulkan 项 false 而总 healthy true | GPU 可见；不能证明训练镜像内相机渲染健康 |

访问来源：[当前页面](http://1.94.0.106:31800/)。未提交表单、没有调用采集/训练/拉取镜像端点。观察值是时间点快照，不是资源预留。

## 3. 当前实现真正的断点

| 优先级 | 文件 / 证据 | 断点与后果 | 本轮设计落点 |
|---|---|---|---|
| P0 | `backend/eb/api/routes_train.py:submit_train` | dataset/base model 记录到 TrainJob，却未填入 JobSpec.inputs；血缘有边并不证明数据真的被使用 | 解析版本→冻结 inputs→挂载→loader 回执，04 |
| P0 | 同文件、`adapters/rlinf.py`、`orchestrator/collect.py` | UI 的 `runs/trn_*/output/checkpoints`、实际 `runs/run_*/output`、旧收集器 `outputs/logs` 不一致 | Run 是执行路径真源，输出 manifest，04/08 |
| P0 | `domain/rlinf_form.py` | Hydra 只校验 `=`；高级项可与强制项冲突；三模板固定 BHL | 容器内 compose/resolve + 受保护字段；能力目录，04 |
| P0 | `adapters/rlinf.py` | 用户配置拼接进 `bash -lc` 字符串 | 有参数数组的受控 launcher，拒绝 shell 注入，04 |
| P0 | `api/routes_data.py` | API 中 create_task 下载，HF 同步调用；无持久恢复；未指定 dataset repo 类型；下载成功即 ready | 受 ebd 管理的数据 Job、验证后发布，02 |
| P0 | `api/routes_data.py`、`paths.py` | 读路径与写目标同属允许前缀，允许读取不代表允许写外部仓；HTTP 重定向无限制 | 只读源登记、平台写根、下载出网策略，01/02 |
| P1 | `domain/data_view.py` | 依据文件名找视频；将 ego_jsonl 文件数作为 episode 回退 | 格式专用索引、未知统计为 null，02/07 |
| P1 | `pipelines.py` | 只有顺序成功推进；无类型化产物引用/可靠扇出/发布提交 | 在原有引擎上补 Step、幂等与发布屏障，01/03 |
| P1 | `domain/rlinf_form.py:ray_port_for` | CRC32 % 50 不保证端口不碰撞 | 默认任务独立网络；不能把 head 端口散列当隔离，04 |
| P1 | `api/routes_env.py:env_doctor` | API 容器中看文件存在，且 Vulkan 不计入 healthy | 按镜像/能力运行 probe，03/04 |
| P1 | `store/models.py` | ModelVersion 与 WeightModel 两条登记轨道；TrainJob 与 Job 状态同步不完整 | 兼容映射、单一执行状态真源，08 |
| P1 | `orchestrator/service.py` | 运行流程仍可能调用代码仓快照相关 git 功能 | 本轮不触发 Job；后续采用内容快照路径满足无 git 工作流，01 |

以上是静态调用链结论；没有通过恶意输入或破坏性动作验证安全缺陷。新增数据接口与旧 `/datasets` 列表/详情分属不同文件，是口径分裂，不应误报成相同 method/path 的重复路由。

## 4. 周边项目复用清单

| 本地项目 | 已查到的能力 | 复用方式 / 限制 |
|---|---|---|
| `/data/kk/leisaac` | SO101 单/双臂、LeKiwi 资产；遥操、状态机、Mimic、HDF5→LeRobot v2/v3 | 作为采集工具容器；默认 HDF5 全尝试存档，不能直接把成功集当完整采集日志 |
| `/data/kk/RLinf` | 模型注册、SFT worker、IsaacLab 三个任务、BHL 严格采集器 | 复用已有 builder/loader/训练入口；G1、SO101 的 RL 环境桥接仍要做 |
| `/data/kk/IsaacLab` | G1 flat/rough locomotion、若干上肢/全身操作与 Mimic 配置 | 配置存在是候选能力，安装镜像、资产可用和动作语义另验 |
| `/data/kk/Berkeley-Humanoid-Lite` | 本体资产、既有训练、数据验证和 RLinf 评估脚本 | 保留当前短训路径与四种接口版本，不把 BHL 特例推广给其他机型 |
| `/data/kk/lerobot` | SO101/G1 设备实现、数据格式和加载器 | 接入已有端侧工具与数据，不让 Web 平台接管实时遥操 |
| `/data/kk/openpi`、`lingbot-vla-v2` | VLA 相关源码；后者含独立 policy server | 作为模型依赖候选，优先走 RLinf 已有集成；未验证所有独立路径 |
| `/data/kk/berkeley_ego_data` | 已有 ego 来源、manifests | 场景/语言/人类动作输入，不伪造机器人 action 标签 |

`/data/kk` 还有 verl/vllm 等项目，不因存在就纳入平台运行时。本轮不扫描模型大权重、不对 44GB 数据做全量哈希。

## 5. 旧设计逐项裁决

| 旧决策 | 本轮裁决 | 理由 |
|---|---|---|
| 单机 Compose，Python 控制面，Rust 数据面，ebd 唯一持 docker.sock | 保留 | 通用机器人不要求分布式平台 |
| redesign/rereview 不做评估/部署 | 恢复设计范围 | 用户本轮明确要求评估和 sim2real |
| 本体环境仅扫描派生、提交不阻断 | 界面可自动填写，兼容校验仍必须执行 | 同维度不同关节、相机、频率会导致错误动作 |
| 所有数据必须走 L0→L5 | 保留语义分层，允许原生机器人数据跳过 ego 编译/重定向 | 三类来源不是同一生产过程 |
| 目录可变且只记 path/mtime | 原始路径可登记；正式训练只消费冻结版本 | 只读挂载防容器写入，不能阻止宿主文件变化 |
| 全部产物强制复制 CAS / 全面抛弃 CAS | 保留已有 CAS，新增大数据采用不可变受管目录+manifest | 兼容历史，避免数据重复存储 |
| SFT 一律 actor_rollout placement | 从实际入口所需 actor 策略推导 | 本地 train_vla_sft 读取的是 actor，不是任意旧字符串 |
| 动态 head 端口即可避免冲突 | 独立网络命名空间；特殊 host 模式用完整端口租约 | head 以外还有 worker/dashboard 等端口 |
| 真机自动放行、失败后立刻切包 | 保持平台放行拒绝；现场端侧在安全状态切换 | 当前硬约束、机器人实时安全和既有 403 行为都保留 |
| 原型数字可展示成熟度 | 只有运行证据才能展示实测 | 历史原型不是现场数据 |

## 6. 三种方案的取舍

**推荐：共用资产与 Job/Run，按接口配置机器人、模型和工具能力。** 三类机器人都走数据、训练、评估、交付阶段；场景特有步骤通过现有 adapter 和参数配置组合。需要补齐数据版本与执行证据，但保留大部分基础设施。

备选一：每个机器人独立一套页面和任务类型。短期容易展示，随后数据格式、失败恢复和评估标准会复制三遍，不满足用户的通用方案。

备选二：重新引入通用工作流平台、任意插件 DSL、多机训练集群和独立数据湖。当前 4 卡单机没有这类必要性；先用有限管线模板+有界扇出，后续出现明确瓶颈再评估替换。
