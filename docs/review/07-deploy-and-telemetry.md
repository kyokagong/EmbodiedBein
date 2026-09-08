# 07 · 部署流水线与真机遥测深度审查报告

- **对照设计**：`docs/design/07-deploy.html`
- **实现范围**：`backend/eb/bundle/`、`robot-agent/`、`backend/eb/api/routes.py`、`frontend/src/views/Bundles.vue`、`frontend/src/views/RobotDetail.vue`
- **成熟度评分**：**35%**

---

## 1. 架构红线与平台职责边界审查

设计在 07.1 明确划定了最具防御性的架构红线：

```
[ 平台职责 (非实时) ]               [ 机器人端职责 (实时 · 自洽) ]
  · 策略包打包与签名               · 25 Hz 策略推理 / 1000 Hz 关节控制
  · 放行审计与签名验证             · 硬件 E-Stop 保护回路
  · 遥测异步归档与分析             · 离线本地策略缓存与安全限幅
         │                                   │
         ├─── ① 包拉取 (离线/幂等/人工) ─────┤
         └─── ② 遥测上报 (异步/批量/可丢) ───┘
```

### 审查结论：
- **绝对隔离原则落实良好**：平台代码中没有任何向机器人直接发送实时控制指令、关节力矩或占用网络控制回路的逻辑，完全符合「平台的故障不传导给机器人」的设计红线。

---

## 2. 策略包 (PolicyBundle) 规范审查

策略包是训练与真机之间的唯一交付物，打包格式为 `.ebp` (tar.zst)：

```
pb_<id>/
├── manifest.yaml          # 核心元数据（能不能装、从哪来、怎么安全地跑）
├── policy.onnx            # 或 policy.pt
├── policy_meta.yaml       # 兼容已有 C++ 控制器
├── normalization.npz      # 观测归一化参数
├── gate_report.json       # 门禁通过证明
└── SIGNATURE              # sha256 校验和清单 + HMAC 签名
```

### 现状与差距分析：
1. **打包代码 (`eb/bundle/pack.py`)**：
   - 已实现基本的 `.ebp` 目录组装、Manifest 写入及与当前本体环境的 G4 字段校验。
2. **缺失环节**：
   - **缺少数值一致性自检**：设计要求在导出时必须在 CPU 上实测 1000 次，且 ONNX vs PyTorch 的最大绝对推理误差必须 `≤ 1e-4`。当前打包尚未包含此数值自检。
   - **缺少 HMAC 签名保护**：当前 `SIGNATURE` 仅为简单 sha256 记录，未对私钥做 HMAC 签名防护。

---

## 3. 三段放行流水线与分级放行审查

```
[ ① 导出与自检 (G4) ] ──► [ ② sim2sim 门禁 (MuJoCo) ] ──► [ ③ 分级真机放行 (S0→S1→S2) ]
                                                                      │
                                                       (物理确认 · 平台 403 阻断 Agent)
```

### 3.1 分级放行机制核查
- **设计规范**：
  - **S0（系留/空载）**：吊带保护，小动作幅度测试。
  - **S1（平地/限速）**：速度 ≤ 0.3 m/s，人工手持急停。
  - **S2（全功率放行）**：正常作业。
- **现状**：
  - `POST /api/v1/releases/{id}/promote` 固定返回 HTTP 403 错误（`"S0/S1/S2 promotion is human-only and requires physical confirmation"`），结构性杜绝了 Agent 越权放行。
  - **不足**：前端 `Bundles.vue` 与 `RobotDetail.vue` 尚未提供人工确认的多级放行 UI 面板与审计留痕表单。

---

## 4. 端侧 `robot-agent` 与遥测链路审查

### 4.1 端侧 Agent 现状 (`robot-agent/`)
- 当前 `robot-agent` 仓库仅为一个基础的 Rust 骨架，仅能打印自身 ID，尚未完整实现 `pull`（拉取并校验策略包）、`serve`（本地缓存执行）、`rollback`（离线一键回滚）等子命令。

### 4.2 遥测接收与时序分析 (`/deployments/{id}/telemetry`)
- **现状**：API 端点已支持接收遥测 JSON 点并将其追加写入 `/var/lib/eb/tsdb/<did>/<tag>.jsonl`。
- **差距**：
  1. 缺少控制环周期抖动（Jitter p99）、CAN 错误率、电机温度、电流峰值的统计计算。
  2. 前端缺少真机运行时间线与事件回放组件。

---

## 5. 改进计划与行动项

1. **[P1] 补齐导出数值一致性自检**：在 `eb/bundle/pack.py` 中引入 ONNXRuntime 与 PyTorch 对随机输入的绝对误差比对，误差超过 `1e-4` 拒绝打包。
2. **[P2] 前端分级放行与签署界面**：在 `Bundles.vue` 中提供带有物理确认免责声明的人工签名与 S0/S1/S2 放行流。
3. **[P2] 完善端侧 `robot-agent` 基础功能**：实现端侧策略包本地解压、哈希自检与离线软链接回滚机制。
4. **[P3] 遥测时序看板**：在 `RobotDetail.vue` 中接入 Jitter 曲线与电机温度热力图展示。
