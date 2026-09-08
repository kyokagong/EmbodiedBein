# 02 · 领域模型与本体/本体环境/任务审查报告

- **对照设计**：`docs/design/02-domain-model.html`、`12-embodiment-environment.md`、`13-embodiment-crud-prototype.html`
- **实现范围**：`backend/eb/domain/`、`backend/eb/api/embodiment_routes.py`、`contracts/`、`frontend/src/views/Embodiment*`、`frontend/src/views/EnvDetail.vue`、`frontend/src/views/TaskList.vue`
- **成熟度评分**：**75%**

---

## 1. 实体演进与分层模型审查

设计在 2026-08-24 明确将旧的单一「契约 (EmbodimentContract)」分拆为三层一等实体：

```
[ 本体 Embodiment (emb_*) ] ── 1:N ──► [ 本体环境 EmbodimentEnv (ee_*) ]
                                              ▲
                                              │ 兼容引用 (compatible_env_ids)
[ 任务 Task (tsk_*) ] ─────────────────────────┘
```

### 1.1 实体对照表

| 实体 | 前缀 | 设计职责 | 实现状态 | 遗留问题 / 风险 |
|---|---|---|---|---|
| **Embodiment** | `emb_` | 机器人品牌/家族元数据、5 种来源定义 (`source.kind`)、3D 资产与机型台账 | ✅ 已建立数据表与路由，支持来源元数据编辑与只读重扫 | 种子 BHL 来源正常；但未完整覆盖其他 4 种来源向导（如 `managed_bundle` 上传 CAS） |
| **EmbodimentEnv** | `ee_` | 接口数字（obs/act/Hz）、观测 layout、控制律、仿真与真机绑定、G1~G5 断言 | ✅ 4 份 BHL 规范 YAML 均落库自洽，支持草稿保存与发布冻结 `spec_hash` | 历史模型/实验表中部分关联字段仍保留 `contract_id`，未完全清洗为外键约束 |
| **Task** | `tsk_` | 独立任务定义（Velocity, Reach, Push, WBC），声明兼容环境与各栈入口 (`stack_bindings`) | ✅ 已支持任务列表与各栈绑定 (Isaac Gym ID, RLinf config) | 任务与数据域 `Scenario` 之间的关联映射尚未完全打通 |

---

## 2. 四套本体环境与「不规则点」实现核查

设计强调：模型必须能够准确表达 WBC 的双 action term 与 fixed-hand 的多模态非拼接观测，绝不能退化为单一平铺向量。

| 环境 ID | 自由度 (DOF) | 观测 (Obs) | 动作 (Act) | 控制频率 (Hz) | 不规则特性 | 实现核查 |
|---|---|---|---|---|---|---|
| `ee_bhl_humanoid_22dof_v1` | 22 | 75 | 22 | 25 | 标准运控，单一动作向量，scale 0.25 | ✅ Layout 顺序与张量维度完全一致 |
| `ee_bhl_biped_12dof_v1` | 12 | 45 | 12 | 25 | 真机 C++ 默认目标 (`N_JOINTS=12`) | ✅ 与 BHL 12 关节顺序完全对应 |
| `ee_bhl_wbc_22dof_v1` | 22 | 85 | 22 (10+12) | 50 | 多 `+10` 维臂参考观测；**双 action term**（臂 `to_limits` 10 + 腿 `position` 12）；`command_dim=13` | ✅ 动作 terms 拆分为 arm/leg 两组独立配置 |
| `ee_bhl_fixedhand_10dof_v1` | 10 | 10 + 图像 | 10 | 20 | 观测**不拼接 (`concat: false`)**；图像 `[224,224,3]` 独立模态；动作为**绝对**关节位置 | ✅ `concat: false` 生效，图像模态独立呈现 |

---

## 3. 五大校验卡口 (G1 ~ G5) 落实度审查

```
[ 提交任务 ] ──► [ G1: 任务/契约匹配 ] ──► (400 阻断)
[ 数据发版 ] ──► [ G2: 动作维与频率 ] ──► (标记不兼容)
[ 模型注册 ] ──► [ G3: ONNX 形状反查 ] ──► (拒绝注册)
[ 导出签名 ] ──► [ G4: YAML 全字段比对 ] ──► (拒绝签名)
[ 真机部署 ] ──► [ G5: 固件关节数/延迟 ] ──► (阻断部署)
```

| 卡口 | 触发时机 | 校验逻辑 | 当前实现深度 | 差距与改进 |
|---|---|---|---|---|
| **G1** | 任务提交时 | `task_id` 兼容性、`spec.params.task` 与 `sim_bindings` 比对、环境自洽断言 | `eb/gates/engine.py` 已实现，Submit 页提供「G1 预检」按钮并展示字段级 diff | 需在后端 `orchestrator.submit` 强制做前置同步阻断，不可仅依赖前端预检 |
| **G2** | 数据集发版 | 样本 action 维度、state 维度、fps 与环境比对 | `eb/domain/data_view.py` 提供静态兼容性列表计算 | 未实现对已录制 Parquet 文件的逐行真实抽样校验 |
| **G3** | 模型注册 | 从 `.onnx` 读取真实输入输出张量维度并比对环境 | 已通过 `eb/onnx_io.py` 实测 6 个遗留 ONNX 维度，不匹配时判定为 `failed` | 已具备实测能力；需扩展至 PyTorch `.pt` 文件的 TorchScript 符号表读取 |
| **G4** | 策略包导出/签名 | 部署 YAML 的 `num_observations`, `num_actions`, `action_scale` 与环境全字段比对 | `eb/bundle/pack.py` 已在构建 Bundle 时进行字段比对并生成 `gates_passed` | 签名接口需核验 `G4` 与 `sim2sim` 双通过，缺少 ONNX vs PyTorch 数值自检 |
| **G5** | 上机部署前 | 机器人固件上报关节数 ↔ 契约；标定有效性；延迟 ≤ `max_policy_latency_ms` | `deployments/{id}/promote` 固定返回 403 阻断放行 | 符合「人工按按钮」红线；需在 UI 呈现 G5 的各项检查清单（标定天数、推理延迟） |

---

## 4. 3D 可视化与工作台交互审查

### 4.1 3D 查看器 (`EmbodimentDetail.vue` / `UrdfStage.vue`)
- **优点**：使用 Three.js + STLLoader，能够从只读根目录经 `/api/v1/embodiments/{id}/assets/file` 流式拉取 URDF 与 STL 网格并渲染机器人骨架与外观。
- **差距**：
  1. 缺少与本体环境选中的联动高亮（例如：选中 Biped 时应将双臂半透明化，仅高亮 12 个腿部关节）。
  2. 缺少关节滑块（Joint Sliders）调试控制器，用户无法在浏览器中拖动关节查看各轴运动限位。
  3. 缺少 Visual Mesh 与 Collision Mesh 之间的快速切换开关。
  4. USD 资产目前仅展示文件大小和哈希卡片，符合「浏览器不解析 USD」的设计，但缺少 Isaac 离线渲染的静态缩略图展示。

### 4.2 草稿编辑与发布流 (`EnvDetail.vue`)
- **现状**：已实现「克隆为草稿 → 编辑 YAML → 提交自洽校验 → 发布为不可变版本」的完整状态转移；已发布版本加锁只读。
- **差距**：当前编辑草稿采用整个 YAML JSON 文本框直接编辑，尚未实现设计 12.5 中的「左侧可视化表单（关节拖拽、观测 term 增删）与右侧 YAML 同步双向绑定」的高保真交互。

---

## 5. 改进计划与行动项

1. **[P0] 任务提交多 JobKind 表单化**：在 `Submit.vue` 中支持 `train.rl`, `train.sft`, `eval.sim`, `eval.sim2sim`, `export.policy` 等主流任务类型的表单切换与预检。
2. **[P1] 3D 查看器交互增强**：在 `UrdfStage.vue` 中增加关节角度调节滑块面板，并实现与当前本体环境关节列表的联动高亮。
3. **[P1] 双向草稿编辑表单**：在 `EnvDetail.vue` 中为草稿环境提供可视化字段编辑组件（关节排序拖拽、观测 term 维度增减）。
4. **[P2] 彻底清理旧数据字段**：统一数据库与 API 序列化，将所有遗留 `contract_id` / `ec_*` 收口为 `embodiment_env_id` / `ee_*`。
