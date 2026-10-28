# P-BFM故障恢复执行计划

设计：[26恢复](../../design/26-pbfm-rpo-recovery.md)。当前工作区原有差异全部保留；本轮修改基线/tmp/pbfm-recovery-before，不提交/暂存Git。主Agent实施，最后按executing-plans做一次独立只读复核。

- [x] 读取真实Job/Run/首个错误：23168日志、rank1地形非有限；19984完整周期保存已发布。原Job失败、四GPU已释放，不改状态。
- [x] CPU核验19984所有文件hash、父索引/receipt/manifest及原生8optimizer/4Replay；保全来源与资源证据。
- [x] 先RED后GREEN实现受约束完整周期恢复：新mode/整数代次、冻结源/父plan/receipt/路径/镜像；预览/提交/排队重验，历史plan不依赖当前runtime文件变化而失效。
- [ ] 捕获异常的实际物理与传感器帧；按用户答复处理训练终止约束。不得盲目填补NaN、伪造successor或放宽独立质量判据。需要的修复有最小复现与回归。
- [x] 派生独立恢复镜像；容器内原生计数、实际loader、边界/失配/失败检查，相关PBFM与前端回归，CPU脚本不冒充GPU证据。
- [ ] 一次独立复核后make up部署，通过平台提交唯一恢复Job，确认四rank参数817370e8…、全部optimizer19984、首次更新20000。4×2048，训练上限30小时，到100000结束后导出与独立评估/视频，不自动追加训练。
- [ ] 真实越过23168失败点及首个周期29968，检查异常计数/资源/课程；最终完整状态、32例/64原片与页面播放三轮复核，报告实际新增80016及丢弃未保存3184，未验证项明确。

台账：根因尚在定位；不能仅凭最后错误把NaN归因为ray未命中，MJLab原生miss分支本来返回有限值。训练只有time_out终止已由实际版本源码证实。用户已明确“保留原生终止，仅修复数值错误”。

源核验complete：8,886,062,250字节全hash、模型817370e8…、8optimizer19984、四Replay64槽，均在原producer CPU loader读取通过。

恢复实现检查：显式新mode/整数19984、父失败终态/实际Run退出、原子索引/receipt/manifest与payload、真实旧身份投影/新镜像、预览/排队重验已实现；入口2项RED→GREEN，完整PBFM257 passed/1 skipped。原生CPU循环确认80016新增、5001行、U100000/L12822528/G51290112及29968…99968周期。CPU异常帧复现能区分finite physics/nonfinite sensor，保留NPZ和MJB，原错误继续抛出，不填NaN或改变终止。

恢复阶段镜像c95f1d36…严格继承原920414父层，runtime receipt/实际预览plan在任务镜像中校验通过；API/前端make up部署和90前端/类型/构建通过。此镜像增加恢复和异常现场捕获，尚未宣称原始NaN根因已修复。预览已通过，正在提交单个带捕获的恢复Job；复现失败后先读真实现场再定最小修复，无自动训练重试。

已提交job_03fb1164877844009315/run_a984bbd051b346a39e1f；四GPU/2048环境、同参数与全部8optimizer19984实际启动审计通过，浏览器14图719指标/80016进度通过。额外恢复拒绝边界7 tests通过（包含运行中/成功父Job、未退出Run、其他来源和未保存计数）。

Ruling: 在没有真实异常帧时，先用不改变原生计算的捕获镜像复现，数值修复与最终独立复核仍待完成；不把诊断Job启动算修复完成。若复现失败，先读现场再派生新镜像，禁止自动训练重试。代价：复现期间可能再次丢失未保存的更新。

末尾闭环已配置新的HTTP-only观察器eb-pbfm-recovery-final-actions，仅精确U100000/L12822528/G51290112、完整有限状态/原预览/镜像均匹配才提交最终export/evals；本轮Job唯一幂等键，无训练提交入口。7个失败边界与末尾动作幂等离线检查通过，fixture全在临时目录，与真实观测分离。随后CPU完整字节/ORT/64视频/32NPZ及真实浏览器核验会话等待结果；任何失败停止，不提前标接入或质量通过。

CPU射线边界假设检查56448条原生射线有限、未复现；不能据此证明实际GPU异常已解决。失败Run823行曲线/原始CSV已归档，窗口统计与无NaN日志不代替物理状态有限性。

Final: fixed 最终浏览器错误指向旧Job及旧父来源 — browser-binding-check实际解析2份Job/2份父来源 RED→GREEN，改用本轮receipt/冻结recovery来源；无启动浏览器/GPU的自检通过。

Final: fixed c95历史计划受当前runtime内容变化影响 — test_frozen_recovery_survives_later_runtime_changes_with_exact_producer_receipt_pair RED→GREEN，按精确c95/d3c764c7…对保留历史receipt，错误镜像/receipt组合拒绝；新预览仍冻结当前runtime。完整suite264/264通过；前两次全量测试分别因缺仓库合同/补丁挂载和只读宿主状态根失败，已修正测试容器挂载并隔离EB_DATA_ROOT=/tmp后通过，未放宽产品检查。

历史兼容派生v2镜像30188223…，只改pbfm_recovery.py，v1 c95与原9204均保留；原生数值循环、终止和当前GPU Job不变。新API make up部署后health/ebd正常，原Job/Run/旧镜像仍可见并持续更新；新v2只做预览，未启动第二个训练。

Final: Ruling: 数值根因与修复效果仍无真实帧证据 — 保留pending，不把CPU边界或重启当修复；代价是本轮不能提前宣称故障已解决。
Final: Ruling: 捕获器实际GPU异常的完整性待验证 — 合成检查只证明调用路径与基本格式，真实失败后再核验是否足以定位；代价是可能需要补充现场采集。
Final: Ruling: 最终100000状态/32例64视频/页面播放未完成 — 由已冻结末尾流程实际执行后验收，不提前标pass；代价是仍需等待计算及审核。
Final: Ruling: 独立 reviewer未重复9GB源回读或执行测试 — 使用本轮已留存原生CPU审计与父Agent实际264项回归/镜像核验结果，明确复核证据范围；代价是该部分缺少独立重复运行。

同一次修复复核补充了未知producer/空receipt组合，观察到误匹配None的RED，再加显式非空检查GREEN；全量264/264与Ruff再次通过。为保留v2预览证据，最终候选派生v3 5290928b…，API默认指向v3，c95现有训练、v2候选和9204全保留。v3审计只见pbfm_recovery.py变化，真实c95历史plan及候选Task runtime计划核验通过，数值循环与终止仍未改变，没有启动v2/v3 GPU训练。
