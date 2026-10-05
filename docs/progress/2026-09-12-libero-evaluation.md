# LIBERO 独立评估实现进度（2026-09-12）

设计目标见 `docs/design/05-evaluation-delivery.md`，本文仅记录代码已具备的能力和未验收项，不改设计要求。

## 已实现

- 独立 `/evals/jobs`、`/evals/previews`、`/evals/checkpoints` API；新评估直接创建 `eval.sim` Job，训练 API 拒绝评估模板，训练列表不再显示评估。历史 TrainJob 保留审计，旧链接转评估详情。
- 选定受管训练 Run 和 step；ebd 将 `full_weights.pt` 流式 SHA-256 冻结进只读 CAS。评估计划冻结套件、smoke/standard case 列表、权重摘要、源推理参数及镜像 ID；通用 Job 提交经过相同验证。
- 镜像内使用已锁定 `RLinf/LIBERO-assets` 版本；RLInf runner 禁用自动重置，按 task/init-state 写逐条 JSONL。完成时核对 case 数量、重复、结果和上游汇总；失败、取消、重启后的异常终态归档部分报告。新旧报告有效性明确区分。
- 评估任务、创建、实测 case 进度、报告、逐 task/case、录像、失败库、配对描述和榜单在评估菜单下。榜单只收完整 standard 报告；没有阈值时资格保持 `not_requested`。

## 尚未达到设计验收的部分

- LeRobot `main` 的原生 `lerobot-eval` 路线、LeRobot 原生 checkpoint loader 和两 runner 等价性未实现。当前只有 RLInf π0.5 `.pt` 路线；LeRobot 文档提供协议参考，示教 Dataset 没有被当成仿真成功率真值。
- 资产目前随镜像构建锁定并有文件摘要清单，但尚无可操作的资产准备状态机、运行前完整文件校验回执或镜像内 GPU 冒烟凭证。
- 标准模式是单套件 100 case；四套件 400 case、三 seed 联合任务和宏平均展示未实现。不得把单套件报告显示为完整 LIBERO 总分。
- 视频是 RLInf 现有的每批次录像，非逐 episode 裁切；没有预算上限、保留策略、失败标签人审和审核后数据回流。缺失录像标 unavailable，不据此补造视频。
- 异常退出的部分报告使用已写的终态案例；未写的 case 标 not_started，无法区分尚未开始与进行中被中断。部分成功率只描述已完成样本，不用于榜单。
- 未验证 GPU 容器实际执行、Docker 镜像内资产/依赖、Rust 编译以及视频播放。当前环境缺少 cargo/rustc，Docker socket 不可访问，因此不能宣称端到端可用。

## 静态与单元验证

`backend/.venv/bin/pytest -q backend/tests`、`python -m compileall -q backend/eb backend/tests`、`frontend/node_modules/.bin/vue-tsc --noEmit` 和 `npm run build`。运行验收仍需按设计文档的 smoke → 单套件标准 → 四套件顺序在受管容器中完成。

## 实现反思记录

1. **归属检查**：只改导航会继续留下 `/train` 入口和 TrainJob 记录，因此同时在 API、查询与通用 Job 准入处收口；历史行不删除。
2. **输入检查**：单靠 step/文件大小不能防止源权重变化。增加 ebd 流式摘要冻结和镜像 ID 固定；源推理配置从训练产物读取，缺少 PPO 配置时明确拒绝。
3. **输出检查**：汇总成功率可能掩盖缺少的 episode，完成归档必须逐 case 对账；异常终态另存 partial，榜单过滤。视频是批次录像，页面明确告知这一点，文件缺失时显示不可用。
4. **再检查边界**：没有实际 GPU/镜像运行凭证，不把单元测试、LeRobot 文档或旧的 3/4 报告当成已通过真实 benchmark 的证据。
