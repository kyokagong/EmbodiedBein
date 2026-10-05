# 数据、转换与采集接入

本页完整路径从仓库根目录算；链路中的 Python 模块省略 `backend/eb/` 前缀，单独的路由文件位于其中的 `api/`。API 均带 `/api/v1` 前缀；下方测试文件统一位于 `backend/tests/unit/`。

## 先判断路径

| 需求 | 当前入口与限制 |
|---|---|
| 本地目录纳管 | `POST /datasets`，`source_kind=path`、`source_uri`、`embodiment_env_id`、`usage=train|eval`；随后 `POST /datasets/{id}/validate`。普通路径默认 `validate_lerobot_v3`，其他格式需专用分支/转换器 |
| HF 数据 | `source_kind=huggingface`，固定 40 位 revision；登记后仍需 validate 触发 Job；只支持公开数据，拒绝内联 token |
| HTTP 下载 | 当前 `routes_data.py` 明确拒绝；不能按设计或表单字段宣称可用 |
| LIBERO 测试资产 | `libero_benchmark` 的固定仓库/revision/接口及 `usage=eval`；不是训练示教 |
| RoboTwin | `GET /dataset-sources/robotwin` → `POST /datasets/robotwin` → validate；原生 embodiment 与平台目标接口分别核对 |
| AMP 动作先验 | 专用 RPO 接口及 `purpose=motion_prior`；使用 `rpo_amp_convert.py` 和 motion worker，不能混作 LeRobot 示教 |

请求字段以 `backend/eb/api/routes_data.py` 的模型为准。目录登记并不复制/冻结源内容；源仍可变时，正式训练前必须建立设计要求的冻结版本，不能把外部目录引用当作完整不可变数据发布能力。

## 新格式或新来源

1. 确认格式/版本、许可证、来源 revision、任务与接口、fps/时间戳、相机、state、requested/applied action、单位和坐标系；取一个真实小样本。
2. 追踪 `routes_data.py → adapters/data_convert.py → data_worker.py / data_validation.py / robotwin_worker.py / rpo_amp_convert.py → orchestrator/collect.py`。扫描/大字节热路径查 Rust scanner/collector；格式解析放受管任务 worker，不在请求中解大文件。
3. 复用已有 operation。新格式才补转换器与校验器，明确输入/输出 schema、版本、误差、receipt、失败状态；输出写受管 staging/目录，禁止在外部源旁写索引或缓存。
4. 原始数据、规范化版本、训练 Slice 分开。冻结成员及 hash，按 session/祖先分组切分防泄漏；记录缺帧、未来帧、时间同步、mask 和 norm。确认目标 loader 实际读到预期样本。
5. 下载/解压必须有限额、源版本检查、路径穿越/符号链接防护与取消恢复语义。pickle/自定义 loader 只在无凭据、无出网、受限挂载的容器执行；不因扩展名可信而在 API 解包。
6. 采集用有次数/时间/字节上限的 recipe 和 shard Job，记录失败 attempt 分母。预算耗尽不是采集目标完成，仿真数据不冒充真机数据。

## 验收与证据

登记 → validate Job → receipt/质量统计 → 版本/Slice → 目标 loader 回读 → 小规模训练消费。区分 `registered`、`validated`、`training_ready`、`evaluation_ready`，不要手改状态强行训练。接口失配、eval 数据用于训练、源变更、坏媒体或错误 shape 应明确失败。

现有参考测试：`backend/tests/unit/test_data_validation.py`、`test_dataset_usage.py`、`test_robotwin_data.py`、`test_libero_asset_worker.py`、`test_rpo_amp_convert.py`。来源许可未知、重定向未验证、冻结/断点能力未实现都要记为未完成项。
