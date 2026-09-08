# 06 · 评估体系、门禁与排行榜深度审查报告

- **对照设计**：`docs/design/06-evaluation.html`
- **实现范围**：`backend/eb/gates/`、`backend/eb/eval/`、`frontend/src/views/Eval*`、`frontend/src/views/Leaderboard.vue`
- **成熟度评分**：**35%**

---

## 1. 五层指标体系建设审查

设计建立了从训练过程到真机部署的五层严密指标体系：

```
L0 过程指标 (tfevents, 训练中)  ──► 判断是否停训重来 (Reward, Loss, KL, SPS)
L1 行为指标 (tfevents/eval, 仿真)──► 解释为什么变好/变坏 (终止原因分布, 跟踪误差, 动作抖动)
L2 任务指标 (EvalSuite, 评估)   ──► 决策依据与发版门禁 (Success Rate, Wilson 95% 置信区间, OOD 泛化)
L3 跨域指标 (sim2sim 诊断)       ──► 上机前防线 (发散步数 t*, 观测漂移热力图 |Δμ|/σ)
L4 部署指标 (真机遥测)          ──► 真机稳定运行 (控制环抖动 Jitter, 电机温度, CAN 丢帧)
```

### 1.1 现状与差距分析

| 指标层级 | 规范要求 | 当前实现状态 | 差距与评估 |
|---|---|---|---|
| **L0 过程** | Loss、Entropy、KL、SPS、GPU 显存曲线 | ✅ TSDB 已采集 `Train/mean_reward` 等核心标量 | 缺少 KL 散度与梯度范数的统一命名空间抽取 |
| **L1 行为** | 终止原因分布（超时/倾倒/触限/自碰撞）、动作抖动 | ❌ 未在训练中结构化解析终止事件 | 缺少动作平滑度（一阶差分能量）的自动计算 |
| **L2 任务** | 成功率、完成步数、4 类 OOD 泛化、**Wilson 95% 区间** | ⚠️ 数据库存储 `success_rate` 浮点数 | 缺少样本量不足时的 Wilson 置信区间计算，未区分 OOD 子集 |
| **L3 跨域** | 轨迹发散步数 `t*`、观测漂移热力图 `\|Δμ\|/σ` | ❌ **完全未实现** | 无法在上机前定位到底是哪一维观测引起了 sim2real gap |
| **L4 部署** | 控制环 Jitter p99、推理延迟、电机温度 | ⚠️ 遥测接收接口存在 | 缺少时序指标计算引擎，前端无聚合展示 |

---

## 2. 评估套件 (EvalSuite) 与 Seed 注册表审查

### 2.1 设计要求
- 评估套件必须**冻结 (Frozen)**，固定 64~256 个 seed 集合，确保跨模型比较面对绝对一致的初始物理分布。
- 强制标注样本量与置信区间，当两个模型的成功率置信区间重叠时，系统必须显式输出「<span class="badge b-amber">差异不显著</span>」，杜绝偶然性误导。

### 2.2 实现情况核查
- **种子数据**：已预置 `suite_biped_locomotion_v2`（8 seeds, frozen）。
- **执行瓶颈**：由于上游 BHL 仓库 HEAD 上的 `play.py` 缺乏 `--metrics_output`、`--seed`、`--export_only` 命令行参数，平台无法直接下发新的现场容器化 `eval.sim`。当前评估主要依靠解析已有的离线 `eval_*.json` 文件。
- **UI 呈现 (`EvalReport.vue`)**：目前仅为 19 行简陋模板，缺少成功率矩阵与置信区间上下界展示。

---

## 3. L3 跨域一致性诊断审查 (sim2sim)

设计 06.3 中最具工程价值的工具是 **「观测项分布漂移热力图」**：
- 对契约中每一个观测 term 分别统计 `Isaac→MuJoCo`, `MuJoCo→Real`, `Isaac→Real` 的标准化差异 `|Δμ|/σ`。
- **作用**：若 `projected_gravity` 在真机上漂移达 0.94σ，直接锁定问题出在 IMU 标定；若 `joint_vel_rel` 偏差大，直接锁定电机模型。这彻底改变了过去「上机抽搐全凭猜」的研发困境。

### 现状：
- 当前后端仅有 MuJoCo 适配器的命令行拼装，**尚未实现三域观测对齐采样与 Wasserstein / 均值漂移计算引擎**；前端亦无热力图组件。

---

## 4. 失败案例库 (FailureCase) 与排行榜审查

### 4.1 失败视频墙与案例库 (`EvalFailures.vue`)
- **设计**：在评估过程中自动捕获失败 episode 的视频片段与末尾轨迹，打上原因标签（如 `foot_slip`, `arm_limit`），并支持一键将失败样本回流至训练集（DAgger 闭环）。
- **现状**：`EvalFailures.vue` 目前仅为简单表格，无视频墙，无失败聚类聚类视图。

### 4.2 排行榜 (`Leaderboard.vue`)
- **现状**：按 `success_rate` 倒序展示 EvalReport 列表。
- **差距**：缺少基线策略（Baseline）对比列、缺少置信区间标尺、缺少按本体环境与任务族的筛选切换。

---

## 5. 改进计划与行动项

1. **[P1] 落地 Wilson 95% 置信区间计算**：在 `backend/eb/eval/` 中引入标准 Wilson 二项分布置信区间算法，并在 `EvalReport.vue` 与 `Leaderboard.vue` 中强制并列显示 `82% [71%, 89%] (n=64)`。
2. **[P1] 构建 L3 观测项漂移诊断工具**：在 `eb/eval/cross_domain.py` 中实现三域观测均值漂移计算，并在评估详情页提供热力图呈现。
3. **[P2] 完善评估报告与失败样本回放**：在 `EvalReport.vue` 中实现多 Seed 成功率矩阵表与失败视频抽样抽屉。
4. **[P2] 推动上游评估脚本对接**：为 BHL 适配器提供外部挂载包装脚本，模拟 `--metrics_output` 行为以打通现场评估闭环。
