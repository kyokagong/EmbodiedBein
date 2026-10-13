# 单卡教师 → 四卡 BFM 执行记录

> 已被用户最新指令替代：禁止继续教师训练。T3–T5的教师依赖分支不再执行，历史证据保留。后续按 `2026-10-09-bfm-reference-reassessment.md` 审查原始参考；此前教师追加预算问题不再适用。

依据：设计17、设计16，以及用户“按照建议实施，如果1卡可以的话，直接进行4卡训练和后续的评估”。本计划继承四卡训练计划的 Task2–5。

- [x] T1 固定教师 contract / 输入绑定 / actor 与 critic / 奖励 / terminal 行为；最小行为测试先 RED 后 GREEN。
- [x] T2 部署独立教师模板与任务镜像，平台 API preview，单卡短训，回读有限权重/normalizer/完整 checkpoint。
- [ ] T3 单卡有限探索（含短训合计 ≤1h，禁止自动重试）；完整片段独立 seeds 跟踪评估、视频和逐 case 指标。
- [ ] T4 仅资格合格后采集训练轨迹，派生 qualified DatasetVersion，补齐 source-group split 和权利/receipt 绑定。
- [ ] T5 返回原计划：单 learner 四卡采样容量探测，保持余量；1000 次 BFM joint updates；独立可视化评估和报告。

Pre-flight: T1 产生参考残差策略，T3 必须用同一 547/905 codec 与 normalization；T4 产生的是实际 BFM codec 375/733 数据，不能复制教师 observation。T5 必须消费 qualified prior，不能消费 reference-only 包。

Ruling: 用户最新消息已授权推荐的单卡一小时预算及条件成立后的四卡流程，不再重复询问；资格不成立则记录失败原因，不放宽门槛。
Ruling: 保留当前共享未提交工作区、无 Git 提交，技能中的 commit/worktree 建议不能覆盖用户要求。

T3执行结果：有限训练416 round成功，25 tensors有限，样本5,111,808；12case计划预算内完成4case、1case通过，评估按6分钟上限终止，单卡各Run总3564.1s。完整独立覆盖未完成，T3不能勾选完整通过；T4/T5条件未成立。报告与视频见`docs/progress/2026-10-09-rpo-tracking-teacher-evaluation.md`。

Final review: fresh read-only reviewer identified startup independence and nonfinite receipt issues; both fixed RED→GREEN, full related suite77 passed. No declined substantive finding; remaining runtime qualification/4GPU capabilities explicitly unverified. No Git commit.
