# 04 · 模型注册与容量估算深度审查报告

- **对照设计**：`docs/design/04-model.html`
- **实现范围**：`backend/eb/domain/`、`backend/eb/capacity/`、`backend/eb/onnx_io.py`、`frontend/src/views/Model*`、`frontend/src/views/Lineage.vue`
- **成熟度评分**：**60%**

---

## 1. A/B 两类模型分类治理审查

设计将模型清晰划分为特征差异巨大的两类，并赋予不同的治理规则：

```
A 类（低层运控 MLP · ~10⁵ 参数）:
  · 输入 obs 75/45/85 维向量，输出 act 22/12 维关节偏移
  · 治理重点：checkpoint 数量爆炸（top-k + last 保留）与 ONNX 维度严格比对 (G3)

B 类（具身 VLA 大模型 · 3B ~ 7B 参数）:
  · 输入 state[10] + camera[224,224,3] + 语言指令，输出 action chunk
  · 治理重点：4×24 GB 显存硬约束、基座权重谱系追溯、推理延迟预算
```

### 1.1 现状对照表

| 治理项 | A 类运控模型 (RSL-RL MLP) | B 类具身大模型 (VLA / OpenPI) |
|---|---|---|
| **注册表落库** | ✅ 6 个历史 checkpoints ONNX 实测入库，短训导出的 `policy.onnx` 成功挂载 | ⚠️ 种子库预置了 `mf_openpi` 族，但未纳管真实的 π₀.₅ base HF 权重 |
| **权重产物保留策略** | ⚠️ `ingest.py` 扫 `.pt` 范围过宽（一度将几十个中间 ckpt 全部注册） | ❌ 尚未建立基于 validation loss 的 `last_and_best (k=3)` 自动清洗 |
| **IO 签名反查 (G3)** | ✅ `onnx_io.py` 读真实 onnx input/output shape 并与 4 契约比对 | ❌ 未支持读取 HuggingFace config.json 或 PyTorch VLA 模型的输入模态声明 |
| **推理延迟画像** | ❌ 缺少在目标工控机 / 上位机 CPU 上的实测推理延迟记录 | ❌ 缺少单步推理 p50/p95 延迟对控制周期（如 20Hz 需 ≤15ms）的门禁校验 |

---

## 2. 三层容量估算器闭环审查 (`eb.capacity`)

设计参考 `llmfit` 理念，确立了 **「实测覆盖估算、必须自带估算凭据、硬件探测是一等产物」** 的三层容量估算架构：

```
L1 实测 (measured)  ──► 本机历史 NVML 峰值 (p99 × 1.10) ──► 最高置信度 (覆盖绝大多数重复任务)
L2 解析 (analytic)  ──► 按栈公式 (上界 × 1.25) ────────► 新配置 / 改 batch 档位
L3 目录 (estimated) ──► 骨干参数粗筛 (上界 × 1.40) ──────► 仅用于警告，不用于放行
```

### 2.1 实现情况核查
- **L2 解析公式**：`backend/eb/capacity/estimator.py` 已经完整实现了 IsaacRL 与 VLA 训练的显存解析公式。
  - **实测用例**：OpenVLA 全参微调在 4×24GB 环境下，估算器准确输出 `verdict: "no_fit"` 并拒绝提交；单测已覆盖。
- **L1 实测回填缺口**：
  - 目前训练运行结束后，`eb-collector` 采样到的 NVML 显存峰值仅写入 TSDB / Run 指标摘要，**尚未自动回填至 `MemProfile` 经验库并按 `(model_arch, num_envs, batch_size)` 配置指纹建立索引**。
  - 导致每次相同任务提交时，系统依然回退到 L2 解析公式估算，未能发挥 L1 实测积累的优势。

---

## 3. 模型卡 (ModelCard) 规范与血缘审查

### 3.1 ModelCard 必填项落地
根据设计 04.3，`ModelVersion` 必须包含 10 项核心元数据：

| 字段 | 必填 | 现状 | 存在差距 |
|---|---|---|---|
| `embodiment_env_id` | 是 | ✅ 必填 | 严格外键约束生效，无环境模型不可进注册表 |
| `family / arch` | 是 | ✅ 已记录 | `rsl_rl_mlp` / `openpi` / `openvla` 枚举 |
| `base_model` | 条件 | ⚠️ 字段存在 | 未在 UI 强制校验微调模型的基座指向 |
| `produced_by (run_id)` | 是 | ✅ 已记录 | 训练产出模型已自动绑定 `run_id` |
| `training_data` | 是 | ❌ 缺失 | 外部导入模型未记录数据集版本哈希列表 |
| `license / license_flags` | 是 | ⚠️ 元数据有 | 未实现 `nc_tainted`（非商业传染）向下游自动传播 |
| `io_signature` | 自动 | ✅ 实测读出 | ONNX 真实输入输出形状已提取 |
| `mem_profile` | 自动 | ⚠️ 部分记录 | 仅记录静态公式预估，无真实峰值回填 |
| `intended_use / limitations` | 建议 | ❌ 缺失 | 前端未提供面向团队的适用场景与失效模式文本域 |

### 3.2 模型血缘图 (`Lineage.vue`)
- **现状**：采用 `@vue-flow/core` 实现了 DAG 节点与连线渲染，能够展示 `run → model` 的产出边。
- **差距**：血缘图缺少更早期的输入节点展示（如 `DataSource → DatasetVersion → Run → ModelVersion → PolicyBundle` 的全链路穿透），节点不可直接点击下钻。

---

## 4. 本体资产一致性校验 (`asset.validate`)

设计 04.7 明确指出：**URDF、MJCF 与 USD 三种资产之间的静默漂移是导致「仿真正常、真机抽搐」的最大时间黑洞**。

### 现状与缺口：
1. **资产列表页 (`AssetList.vue`)**：目前仅展示各本体环境绑定的 `mjcf`, `usd`, `isaac_task` 路径字符串。
2. **缺乏数值比对**：尚未实现 `asset.validate` 任务，无法自动比对 URDF、MJCF、USD 中的关节名集合、关节限位 (`limits`)、默认位置 (`default_joint_pos`) 与刚体质量分布差异。

---

## 5. 改进计划与行动项

1. **[P0] 建立 L1 实测显存回填机制**：在 Run 完成认领产物时，将实际 NVML 峰值显存以配置指纹提取并落库为 `MemProfile`，供下一次调度优先匹配。
2. **[P1] 完善 ModelCard 必填审核与展示**：在 `ModelDetail.vue` 中补齐模型卡全字段展示（来源、数据、显存画像、适用场景说明）。
3. **[P1] 实现资产一致性自动化校验**：在 `eb/adapters/mujoco.py` 中增加解析 URDF 与 MJCF 关节限位差异的工具函数，在本体资产页输出红绿比对徽章。
4. **[P2] 全链路血缘 DAG 穿透**：重构 `Lineage.vue`，支持从数据集、任务、模型到部署包的双向血缘探索。
