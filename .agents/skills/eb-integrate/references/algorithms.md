# 算法、模板与训练栈

完整路径从仓库根目录算；下文省略前缀的 Python 模块相对 `backend/eb/`，`routes_train.py` 位于其中的 `api/`；测试位于 `backend/tests/unit/`。首先区分“现有算法的新参数”“新增算法实现”“新增训练栈”。

| 改动 | 优先复用 |
|---|---|
| 已支持参数/任务 | 现有模板及允许编辑的 EnvironmentProfile；不另写训练器 |
| RLinf 新算法/模型桥 | `domain/rlinf_form.py`、`adapters/rlinf.py`、`runners/`、`deploy/patches/rlinf/` |
| 不同训练栈 | `adapters/base.py` 的 `RenderedJob`、`adapters/__init__.py` 的分发与统一 Job/Run |

## 端到端核对

1. 读对应设计和外部实现（只读），确认真实入口、算法状态、loss/update、环境 reset/step、批量与 rollout 语义。SONIC/BFM/AMP 的设计文件存在不代表所有路线已完成；对照最新进度和实际代码。
2. 沿 `routes_train.py → _build_spec → domain/rlinf_form.py → get_adapter → render → orchestrator/service.py → ebd/eb-runner` 追踪。现有 RLinf 路由包含 `params.stack=rlinf` 分支，不能只按 JobKind 猜 adapter。
3. 新模板需明确接口/Task、config、entrypoint、允许的 override、必需数据/基座模型、资源预算。更新能力/来源目录与 Profile 绑定，仅在需要可选择该功能时改前端；预览/提交复用解析，禁止任意 Hydra 键穿透安全/路径设置。
4. 运行实现进入任务镜像：平台 runner/config 或 `deploy/patches/rlinf/` 的受管补丁；不编辑 `/data/kk/RLinf`。通过 Dockerfile 记录源码内容快照、补丁 hash 和镜像 digest，核对实际镜像中的代码版本。任务镜像单独构建，`make up` 不会替你更新它。
5. adapter 返回命令数组、只读源 mounts、受管 writable_paths、设备/GPU/shm 和输出协议；CPU/主存预算来自 `JobSpec.resources`，沿 orchestrator 传给 ebd，`RenderedJob` 没有 CPU/主存字段。复用 ebd 的快照、执行、日志、取消与超时；不在控制面 `subprocess` 跑训练，不创建第二个调度器。
6. 补齐指标、checkpoint、恢复状态、完成 receipt 与 `orchestrator/collect.py` 收集校验，再连接模型版本和独立评估。既要测试成功，也要验证半写产物、容器失败/取消不会发布模型。

“只新增一个 adapter 文件”只适用于已有完整训练栈协议的局部扩展。新模板、接口语义或算法状态确有需求时，应同步改现有领域 schema/注册表；不为每个算法新增 JobKind 或平行业务系统。

## 递进验证

- CPU：模板/override 校验、渲染快照、绑定失配、输出/恢复协议测试。
- 目标镜像：依赖和 config 解析、真实数据/模型 loader、环境 reset/step。
- 有预算的短训：确实发生梯度更新、日志可见、checkpoint 回读/恢复、取消收尾；之后独立评估实际导出产物。
- 正式训练：固定 seed/预算/输入与验收条件；新奖励/动作语义不能用旧结果背书。

例如 AMP 必须连通 motion prior、actor/critic 观测、判别器更新及状态恢复；仅写 `algorithm=amp` 或添加 UI 名称不够。参考 `backend/tests/unit/test_rlinf_form.py`、`test_rlinf_adapter.py`、`test_rlinf_entry.py`、`test_rlinf_patch_contract.py`、`test_rpo_amp_platform.py` 及相关 runtime 检查。
