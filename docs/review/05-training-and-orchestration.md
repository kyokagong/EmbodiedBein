# 05 · 任务编排与训练可视化深度审查报告

- **对照设计**：`docs/design/05-training.html`
- **实现范围**：`backend/eb/adapters/`、`backend/eb/orchestrator/`、`backend/eb/scheduler/`、`backend/eb/worktree.py`、`frontend/src/views/Submit.vue`、`frontend/src/views/Queue.vue`、`frontend/src/views/RunDetail.vue`、`frontend/src/views/RunCompare.vue`
- **成熟度评分**：**60%**

---

## 1. 训练栈适配器 (Adapters) 审查

适配器是系统与底层训练代码解耦的核心屏障，负责将统一的 `JobSpec` 翻译为**命令行、挂载表、环境变量表**。

```
[ JobSpec (意图) ] ──► [ Adapter (翻译) ] ──► { argv[], mounts[], env{}, writable_paths[] }
```

### 1.1 适配器支持情况对照

| 适配器 | 支持的 JobKind | 核心翻译参数与陷阱防范 | 实现状态 | 差距与风险 |
|---|---|---|---|---|
| **`isaaclab_rslrl`** | `train.rl`, `eval.sim`, `export.policy` | `--task`, `--num_envs`, `--max_iterations`, `--headless`, `--device cuda:0`, 日志目录打洞 | ✅ 生产快照测试覆盖，短训 `biped` 已跑通 | 依赖 `python.sh` 与正确注入 `PYTHONPATH`；上游 `play.py` 缺乏 CLI 选项支持 |
| **`rlinf`** | `train.sft`, `train.rlft`, `serve.policy` | Hydra 覆盖、Ray 独立端口 `6390` 避让、`actor/rollout` 双路径注入 | ⚠️ 适配器代码与镜像已就绪 | 尚未在 4×4090 宿主上完成多卡 SFT/RLFT 的真实长周期冒烟压测 |
| **`mujoco`** | `eval.sim2sim` | 纯 CPU 执行，`play_mujoco.py` 调度 | ⚠️ 适配器已就绪 | 缺乏对 `play_mujoco.py` 输出轨迹与关节误差的自动化数值解析 |
| **`tools`** | `data.compile`, `bundle.build`, `report.render` | 轻量工具容器，无 GPU 需求 | ⚠️ 基础命令存在 | `data.retarget` 因缺少 `pinocchio` 库处于已知失败状态 |
| **`lerobot`** | `data.convert` | LeRobot 数据转换脚本调用 | ⚠️ 极薄实现 | 缺少转换进度抽取与 norm stats 计算对接 |

---

## 2. 工作区隔离与写路径打洞审查

设计通过三层挂载体系解决了「在只读仓库上运行会修改仓库的脚本」的经典矛盾：

```
① 只读层 (ro)  ──► /var/lib/eb/repos/bhl/<commit> (git 隔离，多 Run 共享)
② 覆盖层 (rw)  ──► /var/lib/eb/runs/<id>/workspace/ (logs, configs, outputs 独占可写)
③ 共享层 (rw/ro)► /var/lib/eb/cache/ (hf, torch, cas 跨任务高速缓存)
```

### 审查结论：
- **验证事实**：经实测，在运行 `train.rl` 与 `export.policy` 后，宿主 `/data/kk/Berkeley-Humanoid-Lite` 的 `git status` 保持绝对干净，所有生成的日志与权重均被隔离在 `/var/lib/eb/runs/<run_id>/` 下。
- **不足之处**：
  1. 跨设备 clone 无法使用 hardlink，当前采用 `cp -a` 复制子模块，在仓库变大时开销上升。
  2. 尚未实现基于磁盘使用率的 worktree 快照 LRU 自动清理（设计建议保留最近 10 个 commit）。

---

## 3. 单机调度器与资源准入审查 (`eb.scheduler`)

### 3.1 4×4090 真实资源纳管
- **硬件对接**：`ebd doctor` 能够直接调用 NVML 获取 4 张 RTX 4090 的真实 index、温度、显存占用，并输出 `has_dri=true`。
- **准入控制**：调度器按整卡 (Exclusive) 模式分配 GPU，当 4 卡满载时，后续提交的第 5 个任务会保持在 `queued` 状态等待前序任务释放。

### 3.2 缺失的调度策略：
1. **优先级与抢占 (Preemption)**：`JobSpec.policy.priority (0-100)` 尚未在调度队列中起效，当前为严格 FIFO 顺序。
2. **老化机制 (Aging)**：低优先级任务长时间等待未获动态提权。
3. **超时强杀 (Timeout Kill)**：`JobSpec.policy.timeout_h` 尚未配置监控定时器，死锁任务会永久占用 GPU。
4. **失败自动重试**：尚未支持 `retry_on: [infra_failed, oom]` 的自动重试策略（如 OOM 时自动减半 `num_envs`）。

---

## 4. 训练可视化与对比页面审查

### 4.1 Run 详情页 (`RunDetail.vue`)
- **现状**：能够通过 WebSocket 实时订阅 `/ws/runs/{id}`，渲染 TSDB 中的指标曲线（如 `Train/mean_reward`）并显示 `console.log` 尾部文本。
- **差距与改进**：
  1. **缺少多图表分类网格**：设计要求将 L0 过程指标（Loss, Entropy, SPS, GPU 显存）与 L1 行为指标（终止原因分布、关节跟踪误差）并列展示。
  2. **缺少终止原因堆叠面积图**：无法直观展示超时、倾倒、触限、自碰撞在训练过程中的比例演变。
  3. **缺少「复制等价命令」按钮**：无法一键复制 Adapter 实际生成的 Docker 完整命令以供本地离线复现。

### 4.2 Run 对比页 (`RunCompare.vue`)
- **现状**：实现了基于 `spec_diff` 的参数对比、双曲线叠加及简单的 `significant` 显著性判断。
- **差距**：
  1. 仅支持两个 Run 的简单对比，尚未支持 3~8 个 Run 的批量对比网格。
  2. 缺少多目标 Pareto 前沿图表（例如：X 轴训练时长 vs Y 轴最终回报率）。

---

## 5. 改进计划与行动项

1. **[P0] 升级 Run 详情页可视化**：
   - 增加「复制等价命令」按钮，从 `/api/v1/jobs/{id}/equivalent-command` 获取命令。
   - 重构 ECharts 图表，支持多曲线网格排列与平滑度调节。
2. **[P1] 补齐调度器高级策略**：
   - 实现任务执行超时定时器（超过 `timeout_h` 触发 `cancel_run`）。
   - 实现优先级排序与抢占凭据释放。
3. **[P1] RLinf 适配器端到端压测**：
   - 在真实宿主上启动一次短周期的 RLinf VLA SFT 容器任务，验证 Ray 端口 6390 独立性与多卡 FSDP 稳定性。
4. **[P2] 完善 Run 对比工具**：
   - 支持多任务勾选批量对比，提供参数差异高亮与 Pareto 散点图。
