# P-BFM训练页面闭环执行计划

> **For agentic workers:** 使用superpowers:executing-plans，由主Agent逐项实施，最后独立代码复核。用户已要求按既有设计实施并补齐页面；沿用当前工作区，保留所有已有改动，禁止Git提交。

**Goal:** 在现有训练页面提交P-BFM四卡分段训练，查看真实进度、指标与结果，并从模型页面追溯已登记模型及其训练/评估。

**Architecture:** 复用现有受保护的training_validation提交/预览/恢复和Job/Run/ebd。训练查询为这些原始Job提供只读训练记录，不创建或回填伪TrainJob；普通模板及Profile的发布状态不变。沿用原生collector登记的ModelVersion和现有P-BFM模型评估面板。

**Tech Stack:** FastAPI/SQLAlchemy/Pydantic、Vue3/TypeScript/ECharts，容器pytest/Node测试/类型检查/Vite；make up部署。

**Spec:** [设计26§12](../../design/26-pbfm-rpo-rlinf-integration.md)、[四卡派生设计](../../design/26-pbfm-rpo-4x2048-profile.md)、[工作台设计](../../design/07-workbench-api.md)。本次补齐已批准设计的用户页面，不改变算法、训练阶段、已发布版本或质量门槛。

## Global Constraints

- 仍为真实RLinf四rank/E2048/C131072，112→368→512；续段须对应完整独立validation，512不自动追加。
- 不能将asset.validate原Job重写成train.rl或创建假的TrainJob；只读视图必须标识来源，排除A0、失败注入、导出、评估和其它Job。
- 普通模板仍未就绪、Profile仍draft；仅页面显式提供已支持的四卡分段路径，不能普遍绕过ready或Profile约束。
- 当前训练完成不代表质量通过。已有512模型必须显示累计512与独立评估0/16、0/16，保留真实失败及缺失成功类别。
- 观测读取不重复扫描数GB checkpoint/媒体；完整来源和文件校验仍在明确的preview/submit/admission执行。
- 本轮不重复GPU训练。复用实际512/368/112记录验证可见性，提交/防重复/过期预览在隔离测试覆盖；浏览器检查真实页面与真实预览，测试提交不产生额外GPU作业。
- 不改任务镜像8a1，不修改外部只读仓、已发布接口/数据/模型/Suite，不提交Git。

## Review Focus

- job_*被误认成普通TrainJob、A0或其它实验混入列表；未知记录明确404。
- 普通模板关闭时，只有服务器声明的P-BFM四卡路径可提交，不能启用任意参数或其它未就绪模板。
- running/preparing无实测更新时不得按时间伪造进度；恢复段显示源计数和本段终点，不乘rank/optimizer。
- 历史模型只关联其真实producer，不因缺TrainJob丢链接，不将运行成功显示为质量成功。
- 页面轮询/切路由/重复点击/指标量大/媒体错误：清理请求与定时器、保留幂等、默认图表数量有界，缺证据明确展示。

## Task 1：训练记录与模型的只读平台投影

**Files:** 新建backend/eb/pbfm_workbench.py与backend/tests/unit/test_pbfm_workbench.py；修改backend/eb/api/routes_train.py和routes.py。

**Interfaces:** classifier只接受受保护training_validation Job；serializer返回record_kind=pbfm_validation、原job_id/run_id、native_rl显示类型、规范化只读params、真实summary、状态/来源/模型与时间。resume_candidate只读DB摘要给下一固定阶段和报告，真正完整校验仍走原预览/提交。

- [x] RED：受保护Job列表/详情、无Run排队、真实完成512模型关联、A0/其它Job/错误producer排除；不触碰session.add/flush/commit。
- [x] GREEN：训练列表合并真实Job视图，详情/checkpoints/evidence/resume-candidate支持这些原Job；不修改正常TrainJob分支。模型列表/详情关联原producer页面。
- [x] 验证：容器运行新测试及相关P-BFM/模型血缘回归；历史数据缺字段明确返回不可用，不扫描大文件。

## Task 2：页面提交、进度、指标及结果

**Files:** TrainNew.vue、TrainList.vue、TrainJobDetail.vue、ModelList.vue/ModelDetail.vue与PbfmPanel.vue（按真实缺口修改）；frontend/tests/pbfm-workbench.test.mjs及既有P-BFM回归。

**Interfaces:** /train/templates仅P-BFM附加受控页面能力元数据，普通ready不变。新建页复用/pbfm/runtime-probe/preview与submit，固定training_validation四卡输入，模型自动登记；响应job_id跳转/train/job_*。续训页从只读candidate取对应源/报告，原接口执行完整检查。

- [x] RED：关闭普通模板仍可使用明确四卡路径；草稿Profile仅该路径可选；固定E/C/seed/阶段、不混PPO/SFT字段；真实preview_hash与稳定Idempotency-Key提交、切换后旧preview无效。
- [x] GREEN：训练列表轮询、真实更新进度、结果模型链接；详情显示初始化/实际更新/保存结果、主指标图和按需单指标、资源/日志/曲线、独立质量结论及续段入口。已有P-BFM模型可回到原始训练页面。
- [x] 验证：容器Node行为/渲染测试、vue-tsc和Vite构建；正常UFO/PPO/SFT及模型血缘行为不回归，未出现的成功媒体不伪造。

## Task 3：部署与实际页面核对

**Files:** docs/progress/2026-10-22-pbfm-workbench.md及索引。

- [x] 独立代码复核；修复实际发现的Important问题并跑相关回归，保留已有工作区差异。
- [x] 经make up部署含最新前端的API镜像；不重新构建或改动8a1任务镜像。
- [x] 真实API核对112/368/512列表/详情/模型关联、32例评估和图表媒体、Profile/ready未提升。浏览器核对训练新建/历史详情/模型页，真实预览和提交回调由受控测试覆盖，禁止为页面验证额外跑112训练。
- [x] 写明实际验证与未验证项、可直接访问页面、无新增GPU训练/无Git提交，更新计划状态。

完成记录：2026-10-22（Asia/Shanghai），303后端/88前端检查、类型/构建、一次独立复核及实际API/Chromium核对通过。测试/部署/证据与执行裁定见[进度记录](../../progress/2026-10-22-pbfm-workbench.md)。用户禁止Git提交，使用实施前文件快照复核，本计划不包含任何提交步骤。
