# 本体、接口、任务与环境

本页完整路径从仓库根目录算；上下文中的单个模块文件名在 `backend/eb/` 对应模块查找。API 均带 `/api/v1` 前缀。

## 现有落点

| 对象 | 代码 |
|---|---|
| 本体登记、扫描、预览、接口草稿/发布、Task | `backend/eb/api/embodiment_routes.py` |
| 本体/接口/任务校验 | `backend/eb/domain/{embodiment,contract,task}.py` |
| 接口样本与种子 | `contracts/ee_*.yaml`、`backend/eb/bootstrap.py` |
| 环境版本与执行绑定 | `backend/eb/environment_catalog.py`、`backend/eb/domain/environment_profile.py`、`backend/eb/api/routes_environments.py` |
| 路径与资产 | `backend/eb/paths.py`、`backend/eb/settings.py`、`rust/crates/eb-scanner/` |

## 操作顺序

1. 核实 URDF/USD/MJCF 及 mesh/texture 依赖、revision、单位、关节名称与顺序、控制器/标定来源。外部源只读；需修补的资产生成受管新版本，记录来源和变换。
2. 核对 API、ebd、任务容器能看到正确路径。路径白名单、Compose 挂载、任务 mounts 是三处检查；只改白名单不能让容器自动看到目录，禁止把 `/` 加入白名单。
3. 用 `POST /embodiments` 登记（先读 `EmbodimentIn`），`POST /embodiments/{id}/scan` 和预览核对资产。UI 自动推导的 capabilities 仅是线索，不证明可训练。
4. 用 `POST /embodiments/{id}/envs` 创建接口草稿，body 为 `{"spec": ...}`。在 `domain/contract.py` 对照实际字段：`joint_order`、obs layout/history/modalities、action terms、频率/decimation、limits/gains、sim/real bindings。多模态和复合动作参考对应现有 contract，不从 BHL 向量维度复制推断。
5. `GET /embodiment-envs/{id}` 查看 self_check；发布还检查资产关节。由有权限的人类用户执行发布；平台内 Agent 永久禁止 `embodiment_env.publish`。已发布接口用 clone 派生，不修改原 YAML/DB 内容来重解释旧 Run。
6. 定义 Task 的成功/失败/超时和适用接口，再关联训练模板或冻结 EvalSuite。`EnvironmentProfile` 是用途为 train/eval 的执行绑定与设置版本，不能替代本体接口；名称或说明文字变化不代表运行参数生效。
7. 若需开箱即用，再增加 contract 与幂等种子，检查 bootstrap 是否会覆盖已发布对象。模板、Profile、Task、Suite 的 ID/接口/用途必须一致。

示例：接 SO101 的观测/动作协议必须来自其真实实现；BHL fixedhand 的 10 维 Reach 协议不能用改名代替。没有真机时不创建在线 Robot，不宣称部署或硬件验证通过。

## 最小验收

资产完整可预览；关节顺序、拼接维度/历史、复合动作和频率自检通过；错误接口被拒绝；已发布对象不能原地改。接训练还需实际 reset/step、短训与模型回读证据，接评估还需独立 Suite。

复用 `backend/tests/unit/test_domain.py`、`test_contract_view.py`、`test_environment_profiles.py`、`test_train_bindings.py`。设计要求的单位/frame/时序语义超过现有 schema 校验时，记录并补实现，不能把 shape 通过当作语义通过。
