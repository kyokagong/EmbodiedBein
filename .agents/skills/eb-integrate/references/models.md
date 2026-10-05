# 模型来源、训练产物与能力

本页完整路径从仓库根目录算；省略前缀的 Python 模块相对 `backend/eb/`，`routes_*.py` 位于其中的 `api/`；测试位于 `backend/tests/unit/`。API 均带 `/api/v1` 前缀。

## 区分三类对象

- `WeightModel`：本地/下载权重的工作台记录，不自动代表已冻结、兼容或合格。
- `ModelVersion` + `Artifact`：带接口、内容与来源证据的模型版本；核对具体发布分支。
- checkpoint：某次 Run 的中间/终态产物；包含哪些状态决定能否推理、warm start 或 resume。

## 接入已有权重

1. 核实来源/许可证、revision/hash、framework、结构、观测与动作语义；列全权重、config、tokenizer/processor、normalization、动作头及依赖。
2. 本地通过 `POST /weights`（`name`、`path`、可选 `embodiment_id`、`framework`）登记白名单内路径；不写外部模型目录。HF 先查 `GET /weights/sources`，`POST /weights/download` 固定 40 位 revision；只接受 `model_sources.py` 注册且本体匹配的来源。
3. 新来源要同时核对 `backend/eb/model_sources.py` 的 required_files、`model_worker.py` 的验证/下载、`adapters/data_convert.py` 的 Job 渲染及 `routes_model.py` 的 receipt 对账。新增列表项不能替代所需 loader/codec/镜像适配。
4. 在目标任务容器内加载真实小样本，验证输出维度、物理单位、action chunk 消费、夹爪方向、图像预处理和 norm；再绑定训练/评估模板。相同 shape 不代表兼容。

只有 HF URL 时先确认 `repo_id`；`GET /weights/hub-revision?repo_id=...` 可查询公开仓库 revision，但查询成功不代表该来源已允许下载或模型已兼容。

## 训练产物与恢复

检查 `backend/eb/checkpoints.py`、`orchestrator/collect.py`、`api/routes_train.py`。不同路径不完全相同：普通权重可能只产生 Artifact；LIBERO 可经 `/train/{tid}/register-model` 显式登记完整 checkpoint；RoboTwin/RPO 有专用 model package/自动发布逻辑。先确认实际分支再判断“成功无模型”。

新模型族要定义 checkpoint 完成标记、文件清单、内容 hash、推理配置、模型卡和父模型/数据/代码/镜像/Run 血缘。写一半的 checkpoint 不得登记。恢复必须检查 optimizer、scheduler、RNG、scaler 及算法特有状态（例如 AMP 判别器、normalizer）；只有权重时标明 warm start，不能冒充精确恢复。环境/rollout 无法恢复也要写明。

## 评估与交付出口

冻结模型和实际推理包后，走 `api/routes_eval.py` 的候选、预览、提交；RPO 使用 `/evals/rpo/preview`，通用路径使用 `/evals/previews`，不要混用 schema。缺失/重复 case、receipt hash 失配或缺少必要输入时不能提升为合格。

训练 reward、文件存在、容器 exit 0、smoke 通过都不代表模型能力达标。导出的 ONNX/engine 还需独立闭环评估；Orin 性能必须有目标实机证据，真机激活不属于平台工具。

参考测试：`backend/tests/unit/test_model_download.py`、`test_checkpoint_eval.py`、`test_train_resume.py`、`test_robotwin_model_package.py`、`test_eval_receipt.py`。完成报告列模型/Run/Artifact/报告引用以及尚不支持的能力。
