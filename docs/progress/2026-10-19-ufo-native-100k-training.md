# RPO UFO：累计100000次原生agent更新续训

**后续用户已明确取消分阶段训练。** 30,000分段Job已取消，分段控制器退出；以[单次20000→100000执行记录](2026-10-19-ufo-native-direct-100k.md)为当前方案。本文以下保留历史方案及停止原因，不再执行分段接续。

## 20,000步后用户纠正：继续到100,000

用户指出页面在20000显示成功，要求重新分析并严格推进到100000。实查时控制器正在完成20000段的MuJoCo评估，随后确实因为额外的显存余量启发式检查停止；并非100000目标已经完成，也不是实际OOM。20000段全部模型/五Adam/四rank满Replay及六原片验证通过，最低物理显存余量908MiB，主任务仍0/4成功。0.3m/s两后端均存活20秒但有物理越限，不能算任务通过。

本次按用户再次明确的计数目标修订**控制脚本自行附加的**1GiB预留规则：不足1GiB仍记录告警，不再单凭此启发式条件终止已被平台准入的同配置实验。14992及20000两个区间均未发生OOM，实际峰值约23172MiB；Torch expandable_segments未在完整区间兑现1GiB余量，不把该探测说成优化成功。没有修改平台准入、独占GPU租约、质量判据或 `gate.override`；实际OOM/NaN/Inf、rank/collective/timeout失败、不完整checkpoint/媒体及磁盘/总看门狗仍停止。原停止决定和revision1控制器文件完整保留，下面历史协议中的1GiB硬停止条款仅描述revision1，现由[修订依据](evidence/2026-10-19-ufo-native-100k/continuation-review-v2.json)替代。

修复同时将 `experiment_target_updates=100000` 显式冻结到后续Job计划及preview hash，继承时不改变模型/数据/算法身份；各段 `target_updates` 仍按既有正常结束→独立快评→完整状态恢复执行。目标不是从20000新增100000，而是再训练80000更新到累计100000。页面现在分别显示整体目标、本段终点，并从真实父子恢复关系展示后续任务链接；历史20000成功状态不被覆盖成运行或改写为100000。

修订控制器为 `operator-v2`，复用20000段已完成证据，从 `trn_1d0a497175cf4ddd93c9` 恢复到30000，继续40000…100000。3项完整控制循环测试实际遍历20000→30000→…→100000、训练失败不再提交下一段、媒体缺失不误报完成；原8项计数/预算/取消检查仍通过。后端104项及前端19项行为检查、Vue类型检查通过，`make up` 发布。全程无git提交。真实越过20000的运行核查另保存在 `continuation-v2-startup-verification.json`，该文件生成前不能宣称已完成衔接。

用户在上一轮结束后明确要求“这一轮要训练到10万step”。这里沿用已核实的step语义：累计原生agent联合更新100000次，不是100000个环境transition。源checkpoint为14992更新，新增目标85008更新。此前六小时截止时间只属于历史实验，本轮按累计目标执行；另设12小时异常看门狗，包含排队、保存和证据处理。

## 当前状态

已经通过平台API创建完整状态恢复训练：TrainJob `trn_1d0a497175cf4ddd93c9`，Job `job_cd9895abec1142c7b259`，Run `run_23bdb3114ef545b3b2e7`，首段目标20000更新。实际进度及最终状态以[持续更新状态](evidence/2026-10-19-ufo-native-100k/STATUS.md)为准；不能将“已启动”当成“已达到100000”。

启动核查已看到原生CSV达到15024实际更新（之后继续增长），四卡进入真实梯度更新，瞬时显存余量约2650MiB；四rank均通过过滤后的进程环境确认启用新CUDA分配设置。该数值只反映启动和初始更新，首段结束的全区间峰值、优化器回读及六例独立快评仍待完成。[启动证据](evidence/2026-10-19-ufo-native-100k/startup-verification.json)。

源为 `trn_44b8b94a98034ce7aa99` / `run_3709a8fc84e645f38c3b` / 模型 `mv_cc729adbbe2543f080ca`，累计14992更新、5242880全局环境步。完整checkpoint位于 `/data/kk/embodiedbein/runs/run_3709a8fc84e645f38c3b/outputs/native/checkpoint`。该段训练成功，全部5组Adam真实step均为14992，四rank Replay已满且可回读，六例评估与录像完整；主任务成功0/4。原停止原因是最低显存余量924MiB，低于已有1024MiB门槛，并非完成六小时预算。[原始停止决定](evidence/2026-10-19-ufo-native-2048/b262144-u14992/decision.json)保持不变。

## 冻结配置与续训协议

沿用4张GPU、每rank2048环境（总8192）、Replay262144/rank、batch1024/rank（global4096）、原生每批16更新以及全部学习率。UFO、RLinf、机器人资产、训练数据及本体/环境版本均继承源身份，不重新重定向或引入validation数据。

完整恢复模型、全部5组Adam、原生计数和四rank Replay。环境、rollout和各rank RNG仍没有原生精确恢复能力，继续如实标注 `not_available`，不称exact resume。

仅增加显式显存分配设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，尝试减少缓存分配碎片；其作用是待实测验证的资源假设，不宣称已定位内存泄漏。该设置通过 `native_cuda_allocator=expandable_segments` 进入恢复preview和冻结plan，再由adapter写入任务环境；仅接受此白名单值，未指定的历史Job行为不变。优化设置属于执行资源条件，不改变原生CLI、模型身份或checkpoint兼容性指纹；整个Job预览hash及Run计划hash包含此设置。仍使用原冻结任务镜像 `sha256:7217526a893b2293817b0210117f0b9427be055a63103ecd6d1682be23648af1`，未覆盖既有镜像标签或修改外部代码。

首段到20000，随后通常到30000、40000、50000、60000、70000、80000、90000、100000。每段不超过源5400秒墙钟，吞吐不足时只能缩小更新区间，最终累计目标仍封顶100000。按目前实测吞吐估计剩余约6–8小时，包含各段启动、保存、prompt/export及双后端录像评估；实际耗时以Run记录为准。累计global环境步受每段原生保存边界影响，按实际计数报告，不把update数乘环境数伪造样本量。

每段正常结束保存全rankcheckpoint、释放GPU，完成prompt/export及mjlab/mujoco六个固定case，核对全部原片hash、HTTP/Range及逐帧解码，保存原生训练时序分析和跨checkpoint趋势，再决定下一段。站立和0.3m/s为四例主任务，0.7m/s两例仅诊断。用户授权至100000更新的有界观察窗口内，零成功如实记录；质量阈值不下调，正式acceptance仍待确认。

硬停止条件继承：NaN/Inf、OOM、任一rank/collective/timeout失败、不完整或身份变化checkpoint、必需媒体缺失，以及段末最低显存余量低于1024MiB。不重复下调显存门槛；首段显存优化未达到门槛时保全证据并停止。磁盘预算400GB，新段前360GB储备检查，宿主剩余不足200GB停止提交；12小时异常截止只取消本控制器当前Job，不影响其他作业。无自动失败训练重试。

控制容器 `eb-ufo100k-operator-20261019` 仅用API编排，不持有GPU或docker.sock，代码只读挂载；训练和评估全部经过Job/Run/ebd。

## 检查与三轮复核

1. 设计/语义/隔离：重读设计25及训练闭环；源模型、五Adam、满Replay和分组数据身份沿用现有证据。新增参数仅影响CUDA分配，保持原生算法、数据split和只读源。没有重复执行已通过的112/512短训。
2. 计数/资源/恢复/失败：先验证新增检查出现5项预期失败，再实现；聚焦后端96项检查通过。计数控制器8项检查覆盖100000封顶、16对齐、预算缩段、达到目标不再训练、极短预算拒绝及取消只针对本Job。首段真实显存余量与完整恢复仍需实际Run验证。API部署通过 `make up`，healthz及ebd子状态均正常；新文件和训练模块ruff通过。路由文件整体ruff已有格式/长行错误未批量改动。
3. 图表/媒体/证据：旧14992段六例媒体完整并保留原停止记录；新段必须重新产生对应checkpoint的六例媒体及训练时序分析，不能用旧视频放行。控制器在证据/资源不满足时停止。当前首段运行中，未宣称新段评估或100000目标完成。

[协议](evidence/2026-10-19-ufo-native-100k/protocol.json)、[新会话预算](evidence/2026-10-19-ufo-native-100k/session.json)、[源验证](evidence/2026-10-19-ufo-native-100k/source-verification.json)、[控制脚本hash](evidence/2026-10-19-ufo-native-100k/operator/manifest.json)、[状态](evidence/2026-10-19-ufo-native-100k/STATUS.md)。本次未执行git暂存、提交或推送。
