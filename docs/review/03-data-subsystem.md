# 03 · 数据管理子系统深度审查报告

- **对照设计**：`docs/design/03-data.html`
- **实现范围**：`rust/crates/eb-scanner/`、`backend/eb/domain/data_view.py`、`frontend/src/views/Dataset*`、`frontend/src/views/Funnel.vue`、`frontend/src/views/EpisodePreview.vue`、`frontend/src/views/DataConvert.vue`
- **成熟度评分**：**40%**

---

## 1. 数据现状与业务背景对照

根据设计调研，当前具身数据现状具有极高的特殊性：
- 原始素材：**44 GB / 18,974 个文件**（Ego4D clips, TACO raw pkl/npy, annotations）。
- 场景产出：仅有 **35 个可用场景**（reach 29 / push 6），且原始数据中**没有一条机器人关节轨迹**。
- 许可限制：包含 Ego4D 与 TACO 的 NC（非商业）传染性标记。

因此，该模块的核心使命是：**「让漏斗透明可审计，并打通直接产出 10 维动作轨迹的高效采集路径」**。

---

## 2. 六层数据模型落地核查

```
设计六层模型：
[ L0: DataSource ] ──► [ L1: RawArchive ] ──► [ L2: CanonicalEvent ]
                                                      │
[ L5: DataSlice ]  ◄── [ L4: DatasetVersion ] ◄── [ L3: Scenario (35条) ]
```

| 层级 | 实体定义 | 规范职责 | 当前实现状态 | 差距分析 |
|---|---|---|---|---|
| **L0** | `DataSource` | 外部源声明：路径、许可标签、凭证、下载范围、状态（如 HOI4D 403 标记） | ❌ 仅作为 `Dataset.meta.sources` JSON 字段粗略记录 | 缺少独立的外部源台账、许可传染性规则链与凭证管理 |
| **L1** | `RawArchive` | 原始音视频与位姿归档：不可变校验和对账、只读挂载 | ⚠️ `eb-scanner` 可遍历并统计字节与哈希 | 44GB 全量哈希因耗时长易触发超时，增量缓存状态不完整 |
| **L2** | `CanonicalEvent` | 跨源规范事件层：统一 verb, goal_text, [start_s, end_s]（约 192k 条） | ❌ 未落库为实体，缺少 `data.canonicalize` 转换任务 | 无法在统一事件层执行基于动作词与目标的跨数据源搜索 |
| **L3** | `Scenario` | 仿真场景模板：objects、初始区域、instruction 模板、success 判据（35 条） | ⚠️ 种子数据中有 JSONL 记录 | 缺少场景目录管理页与按 task_family (reach/push) 结构化展示 |
| **L4** | `DatasetVersion` | 可训练数据集：LeRobot v2.1 / RLDS 格式，不可变 content_hash，G2 校验 | ⚠️ 数据库有表并记录 content_hash | 缺少真实已转换的 LeRobot parquet 产物与 norm stats 自动计算 |
| **L5** | `DataSlice` | 训练切片：train/val/ood 划分、按物体/任务过滤的物化视图 | ❌ `DatasetSlices.vue` 为初版静态预览 | 无法保存切片定义并在任务提交时作为 `inputs.datasets` 引用 |

---

## 3. 格式嗅探与扫描性能审查 (`eb-scanner`)

### 3.1 8 级特征优先级嗅探实现
`eb-scanner/src/lib.rs` 实现了按优先级短路匹配特征文件的逻辑：
1. `meta/info.json + data/chunk-*.parquet` → `LeRobot`
2. `*.tfrecord + dataset_info.json` → `RLDS`
3. `scenario_catalog.jsonl + canonical_events.jsonl` → `EB Scenario Catalog` (当前命中 4 版本)
4. `v2/annotations/fho_*.json` → `Ego4D` (当前命中)
5. `hand_poses/**/*.pkl + object_poses/**/*.npy` → `TACO raw` (当前命中 2,317 序列)
6. `*.mp4 + *.jsonl manifest` → `Clip 集合` (命中 35 段切片)

### 3.2 扫描性能与超时问题
- **实测表现**：在 `/data/kk/berkeley_ego_data` 上，对 17 个 manifest 文件扫描耗时 < 1 秒，输出 `files=17, valid_episode=9`。
- **瓶颈与风险**：当针对 44GB 全量目录发起哈希扫描时，IO 吞吐受限于磁盘随机读，导致控制面超时被判定为 `failed`。
- **改进方案**：引入增量扫描策略（仅基于 `(path, size, mtime)` 校验），未变动文件跳过哈希计算；将超大目录扫描转为异步后台 Job 执行并持久化中间进度。

---

## 4. 漏斗审计与 Episode 预览器审查

### 4.1 漏斗审计 (`Funnel.vue` / `data_view.py`)
- **优点**：明确将「扫描 17/9」与「设计输入的 Ego4D 编译 155,369→35」区分为不同数据集，避免了数字虚构。
- **差距**：
  1. 缺少逐级拒绝原因的明细下钻（例如：手部跟踪丢失、物体不可见、动作时长不足）。
  2. 缺少设计 03.4 中的**规则敏感性滑块**（模拟「若放宽手部位姿误差阈值，可增加 N 个可用场景」）。

### 4.2 Episode 预览器 (`EpisodePreview.vue`)
- **优点**：支持 HTML5 视频流播放与 action 曲线的 Playhead 时间指针同步；在媒体缺失时展示「缺媒体」空态，避免伪造播放器。
- **差距**：
  1. 缺少叙述时间轴（Narration Timeline）同步滚动联动。
  2. 缺少 TACO 3D 物体位姿与手部关键点（MANO 21 点）的三维同步回放。
  3. 缺少多视角摄像头（Top/Wrist 摄像头画面）的分屏同步播放。

---

## 5. 数据来源战略优先级落实分析

设计明确指出了四条并行数据获取路径的优先级：

```
① 脚本专家采集 (Isaac 环境已就绪，天生带 10 维动作，最高优先级)
② 在线 Rollout 回流 (RLinf DAgger online_lerobot，中期优先级)
③ 遥操作示教 (Pinocchio IK，依赖硬件)
④ 人类 Ego 数据重定向 (研究性质，高不确定性，容忍失败)
```

### 现状评估：
- **专家采集 (`data.collect`)**：当前平台尚未建立提交专家策略一键 Rollout 批量产出 LeRobot 数据集的自动化工作流。
- **数据重定向 (`data.retarget`)**：`eb/tools` 容器镜像中由于缺失 `pinocchio` / `pink` 依赖库，执行重定向 Job 时返回 exit code 2 并记录 `ik_infeasible`（诚实失败）。
- **策略建议**：优先落地「脚本专家在 IsaacLab 中自动采集并写入 LeRobot v2 格式」的完整流水线；重定向作为独立研究插件推进，不阻塞数据主线。

---

## 6. 改进计划与行动项

1. **[P1] 补齐六层数据模型持久化**：补充 `data_sources`, `canonical_events`, `scenarios`, `data_slices` 数据表与 CRUD 接口。
2. **[P2] 专家采集流水线 (`data.collect`)**：在 `Submit.vue` 中提供针对固定环境的脚本专家数据采集 Job 模板，输出标准 LeRobot 格式。
3. **[P2] 漏斗下钻与敏感性分析**：在 `Funnel.vue` 中接入各阶段样本下钻弹窗与阈值模拟滑块。
4. **[P2] Episode 预览器三联联动**：升级 `EpisodePreview.vue`，实现「视频画面 + 动作曲线 + 语言指令时间轴」严格三轴同步。
