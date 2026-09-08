# 08 · Agent 体系与 ToolBus 深度审查报告

- **对照设计**：`docs/design/08-agent.html`
- **实现范围**：`backend/eb/harness/`、`backend/eb/agent/`、`frontend/src/views/Agent.vue`、`frontend/src/views/AgentAudit.vue`
- **成熟度评分**：**50%**

---

## 1. Agent 定位与安全边界审查

设计在 08.1 确立了 Agent 的首要前提：**「Agent 只是一个受限的特殊用户 (`actor_type=agent`)，没有任何后门」**。

### 1.1 永久禁止集 (Hard Blocklist) 落实核查

| 禁止动作 | 对应接口/工具 | 实现状态 | 验证结果 |
|---|---|---|---|
| **改写或发布本体环境** | `contract.modify` / `contract.create` | ✅ 硬阻断 | ToolBus 判定为 `forbidden`，规则路径直接拦截 |
| **绕过或豁免门禁** | `gate.override` | ✅ 硬阻断 | ToolBus 判定为 `forbidden`，不可调用 |
| **直接放行真机部署** | `deploy.release` / `promote` | ✅ 硬阻断 | API 固定返回 403，ToolBus 禁止该工具 |
| **直接执行 Shell 命令** | `shell.exec` | ✅ 未提供 | 平台未注册任何 Shell 交互工具 |
| **直接访问宿主文件系统** | `fs.write` / `fs.delete` | ✅ 未提供 | 平台未注册任何宿主文件系统工具 |
| **直接执行裸 SQL 查询** | `db.execute` | ✅ 未提供 | 平台未向 Agent 暴露 SQL 连接 |

---

## 2. ToolBus 七道处理管道审查

设计 08.4 规定 ToolBus 调用必须经过严格的七道防护处理：

```
[ 工具调用请求 ]
  │
  ├─► ① Schema 校验 (Pydantic v2 参数强校验)
  ├─► ② 权限白名单 (Effect: read / write / dangerous / forbidden)
  ├─► ③ 幂等去重 (Idempotency Key 缓存防重发)
  ├─► ④ 预算扣减 (周 GPU·h 账本硬顶，超支自动降级为只读)
  ├─► ⑤ Dry-Run 预检 (危险写操作前置试跑)
  ├─► ⑥ 核心执行与失败补偿
  └─► ⑦ 审计入库与磁带录制 (Tape Recording, 支持断言 llm_calls == 0)
```

### 现状与差距分析：

1. **已实现环节**：
   - **Schema 与白名单**：`backend/eb/harness/toolbus.py` 实现了基于装饰器的工具注册，支持自动提取权限矩阵 (`GET /api/v1/agent/tools`)，并对 `forbidden` 工具直接阻断。
   - **单测覆盖**：单测覆盖了禁止动作拦截，验证了在无 LLM 介入下的规则阻断输出。
2. **缺失环节**：
   - **预算硬顶 (Budget Hard Cap)**：虽有环境变量 `EB_AGENT_WEEKLY_GPU_HOURS` 配置，但尚未在 ToolBus 中实现实时的 GPU 算力消耗累加与超额阻断。
   - **速率限制令牌桶**：未启用按分钟/小时的工具调用频率限制。
   - **磁带录制与回放测试 (Tape Fixtures)**：目前仅有普通的 Playbook 单元测试，尚未建立设计中完整的「输入-输出磁带录制，离线重放断言 `llm_calls == 0`」的回归测试夹具。

---

## 3. 七大剧本 (Playbooks) 实现审查

设计定义了 7 个高频确定性剧本，遵循 **「规则优先 → 剧本编排 → LLM 兜底」** 的决策顺序：

| 剧本名称 | 设计职责 | 实现现状 | 差距与评估 |
|---|---|---|---|
| **1. 日报生成 (`daily_digest`)** | 汇总过去 24 小时的训练与评估进展 | ⚠️ 规则路径已注册 | 目前主要拼接简单字符串，未通过 ToolBus 动态聚合指标与失败列表 |
| **2. 失败诊断 (`diagnose_train_fail`)** | 提取 Run 失败退出码与末尾异常日志 | ⚠️ 规则路径已注册 | 缺少对 OOM、显示设备缺失、契约不匹配的自动化模式匹配分类 |
| **3. 资源巡检 (`inspect_resources`)** | 检查 4 GPU 占用、排队长度与磁盘水位 | ⚠️ 规则路径已注册 | 仅返回静态描述，未调用 `ebd doctor` 的实时数据 |
| **4. 降配重试 (`retry_with_downscale`)** | OOM 时自动将 `num_envs` 减半并重新提交 | ❌ 尚未实现 | 未能自动组装新 JobSpec 提交调度器 |
| **5. 候选策略对比 (`compare_candidates`)** | 比较同一任务下的最优与基线策略 | ❌ 尚未实现 | 缺少调用 `runs.compare` 自动生成对比结论的逻辑 |
| **6. 放行预检 (`preflight_release`)** | 检查 G4、sim2sim 门禁与标定有效期 | ❌ 尚未实现 | 缺少一键检查策略包是否满足 S0 放行条件的预检工具 |
| **7. 数据集健康度 (`dataset_health`)** | 检查数据漏斗各层留存与格式合规性 | ❌ 尚未实现 | 缺少自动扫描并输出数据健康度报告的编排 |

---

## 4. 控制台 UI 与交互审查 (`Agent.vue`)

### 现状与不足：
- **当前实现**：`Agent.vue` 目前仅为一个基础的消息气泡列表 + 输入框，能够发送文本并展示剧本回复。
- **设计差距**：
  1. **缺少工具调用轨迹 (Tool Call Trace Tree)**：无法在界面上展开查看 Agent 在幕后依次调用了哪些工具及其入参和返回值。
  2. **缺少待审批卡片 (Pending Approval Card)**：当 Agent 发起需人工确认的操作时，界面无法弹出带有参数确认和拒绝按钮的交互卡片。
  3. **缺少数字可溯源链接 (`ref_id`)**：Agent 回复中出现的每一个数字（如「成功率 82%」）未挂载可点击跳转至对应 Run 或 Report 的超链接。
  4. **缺少预算仪表盘**：未在控制台顶部显示本周已消耗的 Agent GPU 预算进度条。

---

## 5. 改进计划与行动项

1. **[P1] 升级 3 个核心只读剧本**：重构 `daily_digest`, `diagnose_train_fail`, `inspect_resources`，使其通过 ToolBus 真实调用 `runs.list`, `runs.logs`, `resources.gpus` 提取真实数据。
2. **[P2] 完善 ToolBus 预算与速率扣减**：在 ToolBus 执行器中接入 GPU·h 账本，超限时自动将写任务拒绝并记录审计。
3. **[P2] 重构 Agent 控制台 UI**：在 `Agent.vue` 中增加工具执行步骤折叠面板与数字引用 (`ref#`) 溯源跳转。
4. **[P2] 建立场景磁带回放夹具**：在 `backend/tests/agent/` 中构建磁带录制回放框架，保障关掉 LLM 时规则路径 100% 稳定运行。
