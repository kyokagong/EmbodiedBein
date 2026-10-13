# BFM 行走闭环执行计划

**Goal:** 修正BFM接入遗漏，按固定更新间隔独立评估，并在确认的预算内训练行走模型。
**Architecture:** 设计19；复用RLinf worker、checkpoint恢复、RPO评估和Job/Run，不添加外部trainer或常驻服务。本人顺序实施；禁止Git提交。
**Budget:** 长训用户答复待定。预算前只做代码/测试与有限接入smoke，不执行依赖该答复的长训。

## Review Focus

- mixed-z缓冲区和rollout同步遗漏：更新后池非空、从池实际采样，state回读不丢失。
- 8更新/轮预算错误：125轮=1000更新、恢复起点与实际计数一致，禁止超单段预算。
- 修改恢复终止元数据破坏指纹：同配置可恢复，UTD/数据/拓扑变化必须拒绝。
- validation冒充acceptance：种子/用途冻结隔离，16例不能出合格状态。
- 错Suite/profile/缺case或视频：在API/adapter/collector拒绝，不补造证据。

## Task 1 · 修复算法闭环和诊断

Files: `bfm_zero_worker.py`, `bfm_zero_learning.py`, `bfm_zero_policy.py`（仅必要处）、runtime `bfm_zero_learning_smoke.py`。

- [x] 回读既有1000更新checkpoint确认池计数0，核对固定上游源码。
- [x] 先加实际worker断言：mixed-z池填充、rollout采样匹配、所有诊断有限、多更新准确平均和计数。
- [x] 运行确认遗漏可复现；修正更新写池，保留完整模型buffer同步；复用现有loss诊断。
- [x] 容器内检查更新和checkpoint/RNG回读；不增加teacher或velocity reward。

## Task 2 · 更新预算与恢复

Files: `rpo_bfm.py`, `domain/rlinf_form.py`, `api/routes_train.py`, `adapters/rlinf.py`, `checkpoints.py`, `bfm_zero_worker.py`, unit预算/恢复测试。

- [x] 测试1/8更新、单Run新增上限1000、累计目标上限、错误恢复counter拒绝、其他模板不能使用BFM旋钮。
- [x] 冻结实际update目标与单位，scheduler constant，恢复不改变compatibility；终止条件由现有runner轮数负责。
- [x] focused平台测试和Torch测试通过，明确trainer_state的环境/rollout缺口。

## Task 3 · 独立快评和新验收组

Files: `rpo_bfm.py`, `bootstrap.py`, `api/routes_eval.py`, `adapters/rlinf.py`, `rpo_flat.py`、评估平台测试。

- [x] 增加冻结16 seed validation和100 seed acceptance Suite，复用4卡评估；保持旧profile/Suite兼容。
- [x] API、adapter、collector共用seed/profile绑定校验；validation不出qualification pass，acceptance保持原物理判据。
- [x] 测试篡改seed/profile、计划hash、套件使用错误与缺失case。

## Task 4 · 镜像与实际小样本

Files: `Makefile`新增版本target，progress台账/证据。

- [x] 新training/eval镜像固定digest，不覆盖v6/v8；容器聚焦测试。
- [x] Doctor核对空闲GPU/预算，API单卡小样本训练；回读完整checkpoint及所有optimizer真实计数。
- [x] 同镜像同配置完整恢复，验证新增updates、源只读；独立16 case快评、曲线及实际录像检查。
- [x] 三轮复核后四卡短段，核实3个sampler实收latent池、更新吞吐及独立快评。
- [ ] 四卡训练实际峰值：监控提前超时，只覆盖加载；下一配方扩容前重新测量，不能据此认定显存/RAM余量。


## Task 5 · 确认预算内分段训练与交付

- [ ] 按设计19追加复核，长训前派生上游学习语义对齐配方（显式权重初始化、replay逐更新norm、独立FB pessimism、0.8 relabel、模块LR、batch/clip与专家连续提示/初态核对），以新的小样本验证；不把bringup命名为上游复现。
- [ ] 在用户确认的预算内逐1000真实更新运行，发布后快评并写continue/stop/wait决定。
- [ ] 停止判据触发时只做有证据的有限调整，变配方新spec；保持完整失败证据。
- [ ] 冻结候选，运行四卡新acceptance，保存全部case、图表和录像索引；不达到判据如实失败。
- [ ] 更新progress/README并报告实际结果及未覆盖能力，不宣称真机或完整BFM通用能力通过。
