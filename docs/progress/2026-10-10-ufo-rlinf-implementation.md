# UFO → RLinf 实现与容器验证台账

日期：2026-10-10，Asia/Shanghai。依据：[实施计划](../plans/2026-10-10-ufo-rlinf-bfm-integration.md)、[设计20](../design/20-ufo-bfm-rlinf-integration.md)、[设计21](../design/21-training-runtime-contracts.md)。

**当前交付：B分支任务1–7的代码及容器验证；阶段资格未完成，能力保持 `adapter_ready`。** 用户明确选择“仅实现和容器验证，暂不启动GPU资格或长训”，随后要求离开期间按推荐方式自主实现、不再询问。本次未运行GPU probe、训练、物理环境、独立策略评估或长训，未执行 `make up`、部署或Git提交/暂存/推送。

没有新建实际Job、Run、DatasetVersion、ModelVersion、EvalSuite或受管训练Artifact。下列本地开发测试证据不能冒充这些实体，也不能据此称 `smoke_passed`、`validated` 或正常行走。

## 计划逐项对照

“代码/容器完成”不等于包含真实运行的阶段完成。

| 任务 | 实现及容器验证 | 仍缺的阶段证据 |
|---|---|---|
| 1 执行绑定 | typed/frozen binding、批准recipe、未知runtime/provider/stack拒绝；preview/submit共用validator；底层adapter限制用途/目标/资源，保护seed；原始数据与新Profile冻结；旧入口兼容 | 部署后OpenAPI/schema与真实API Job |
| 2 agent镜像 | UFO受管源码树/blob校验、实读RLinf身份、agent-only镜像；真实CPU更新差分，六Adam/四target/norm/pool/梯度/参数；固定GPU probe与collector回执 | GPU1024/3/batch128差分、真实峰值和受管probe Artifact |
| 3 原始状态/buffer | 登记父数据ID转换API/固定Job、派生版本/lineage；root/q/dq/世界速度；codec重编码核验；8/250连续窗口、祖先split/invalid隔离、官方/G1各0.5；CPU replay保留行为动作/z/term/trunc/版本、采样状态恢复 | 正式数据转换Job、全量有效窗口与来源抽样证明、物理FK/速度 |
| 4 Isaac RSI/aux | 新Profile、train-only初态、root-link速度写入；部分reset、重复读、terminal与新episode分离；八项raw aux；固定环境probe/资产挂载/回执 | 真实RSI/FK/速度/terminated/timeout/aux数值；现probe强制timeout不能替代全部物理检查 |
| 5 RLinf worker | 原生agent独占六optimizer；直接Worker RPC，无FSDP第二optimizer；真实RLinf Trajectory经模拟Channel/RPC；actor/B/norm/pool同步、四计数、warmup0、16块/目标截断、真实Hydra/RLinf配置校验、18项LR/grad/delta | 实际Ray/Channel/GPU rollout→learner闭环、采样/同步时序与预算退出 |
| 6 checkpoint/导出/收集 | native完整trainer_state、atomic staging/fsync/rename；严格文件/hash/key/六optimizer/四target/RNG/replay/expert/pool/rank；下一真实原生更新恢复等价；发布→消费→实际export CLI、Torch/ONNX actor/tracking/command golden trace；旧DCP回归 | GPU/CUDA RNG恢复、真实训练模型收集/lineage、推理包物理评估；初始化0步基线Job尚无运行证据 |
| 7 页面/资格 | recipe/Profile入口、冻结身份、真实联合更新横轴、warmup0/恢复累计目标、实际 `train/ufo/...` tag；沿用共享独立评估/缺媒体或损坏视频incomplete；页面/类型/Vite检查 | 初始化validation16基线→16→24实际更新恢复→held-out tracking/command快评/曲线/全部录像；API取消/容器退出/GPU租约释放；未放行B |
| 8 MJLab/C | **未实施**，遵循任务7→8依赖；无MJLab/native codec/接口或C接入声明 | B资格通过后实施C；当前依赖未满足，不是已有C运行失败 |
| 9 UFO四卡 | **未实施**，仅批准single_learner；单卡census/容量估算不等于四卡资格；旧BFM四卡/并行评估单元回归 | 已验证分支1 learner+3 sampler、同步/恢复、组合峰值/吞吐/评估等价 |
| 10 长训/验收 | **未执行**；无自动续训/预算追加/行走结论 | 拓扑资格、总预算、每1000更新快评决策、独立acceptance100及全部媒体 |
| 可选A | 按计划默认排除 | 另行选择才实施，不能替代RLinf主线 |

步骤状态：任务1步骤1–4代码/容器完成、步骤5开发证据已记录；任务2步骤1–3及步骤4 CPU部分完成，GPU未执行；任务3步骤1–3代码/容器完成、步骤4仅fixture重编码/采样、步骤5无正式DatasetVersion；任务4步骤1–3代码/模拟检查完成，步骤4/5真实物理未执行；任务5步骤1–4代码/CPU边界完成，步骤5无实际训练通信证据；任务6步骤1–5代码/CPU恢复/ONNX完成，GPU/平台注册未执行；任务7步骤1测试完成但部署未执行，步骤2–6真实资格未执行；任务8–10/A未执行。没有将整项阶段勾选为验收完成。

## 冻结身份与配方

- Recipe=`rpo_bfm_ufo_agent_rlinf_isaac_v1`；algorithm/provider/runtime/simulator=`bfm_fb_cpr_aux/ufo/rlinf/isaaclab`；primary unit=`learner_joint_update`；checkpoint=`ufo_agent_state_v1`；topology=`single_learner`。
- 最终本地不可变Docker Image ID=`sha256:c3fc0ea183cb299584c703597584abb381634e737e9734b14f2c1a2b4d7a7af8`；别名 `eb/rlinf:rpo-bfm-ufo-agent-v1`，无registry RepoDigest；renderer实际使用Image ID，不代表已发布到镜像仓。
- UFO commit=`0b89378de02f77712072d983bc43ec2a46455dc0`，root tree=`3a6b8b0d6a04287f37c1d7834e51561a8e014fc9`，content=`sha256:2191cf26c77262f7ef2e3525ca23878a8f08c46fd02a20b7776a791f8f5e7582`。
- RLinf实读commit=`018e6a9e0da60f96eeda8b11008543911027d414`，content=`sha256:dbaa8bed56be412f6d012ee34030822941945786cede6bf9bc3de0fec5445d48`；UFO patch=`sha256:f99aa821d9ccdf436f91010a7b175209a2bda0f32f40afe8e0a6b49ddaf7d5b5`；base BFM patch=`sha256:df5e404b711161d7068894890d1a90bdcf61f064c8c9bfea60210899ecf787a5`。
- Python3.11.16、Torch2.7.0+cu128、Gymnasium1.2.0、Pydantic2.11.5、safetensors0.6.2、tensordict0.11.0；源码导入实测3.11，保留上游包3.10元数据差异，不改平台Python。源码/license/包版本见[镜像回执](artifacts/2026-10-10-ufo-rlinf/image-receipt.json)。
- v1输入state52/last_action23/history300/privileged358，actor375/critic733/z256/action23；保留原生preset loss/LR/pessimism/relabel/norm，compile/cudagraphs关闭，不重写loss/训练教师/加载G1权重。
- 运行1024宽/3层、batch128、16env、每轮4控制步、CPU replay100,000；warmup10,240 transition，首触发11,264，其后每1024最多16联合更新，绝对目标截断；sampling cap4096、runner3300s、Job1h/零重试。资格≤24，普通训练需smoke_passed；兼容指纹仅排除目标/用途，保留学习语义。
- CPU差分64宽/2层/batch16/z256，1真实联合更新、六Adam各step1、四target齐全、参数delta L1=269.39996670366077；FP32 `atol=1e-6/rtol=1e-5`，ONNX `atol=1e-5/rtol=1e-4`，不计GPU资格。

完整1024/3网络CPU census：六optimizer参数分别16,737,604 / 171,520 / 2,785,281 / 5,268,467 / 16,214,854 / 16,214,854，总57,392,580；model state426,931,560字节（含targets197,355,328），rollout229,576,232；100k CPU ring703,800,000。单卡估算约10.2GB依据census和既有Isaac/context估算，**不是GPU实测峰值/准入上限**；expert/通信/保存/渲染组合峰值待测。

## 实施裁定与偏差

1. 保留当前脏工作区，不创建干净worktree：必需BFM前置尚未提交。保留他人已有改动，本次不修改设计真源、不做Git写操作。
2. CPU fixture用2层替代计划1层：固定UFO ResidualForwardMap embedding要求hidden_layers≥2（nn_models.py:453,484）。保留上游实现，GPU仍1024/3。
3. 从**冻结且保留raw root/q/FK数组的官方/标定G1混合父版本**派生RobotState，复用既有转换，不重复解包pickle/重做标定；仅52+358版本拒绝。重编码52+358并对照FK/root-link frame，保留世界平移/yaw/valid/group/train/validation/test/许可。首期仅一个已注册混合父版本，不拼任意路径/多个未经批准父版本；全量正式数据尚未验证。
4. 新初态/aux派生Profile，旧发布接口/Profile/Job/DCP语义保留。恢复明确trainer_state：环境不可恢复，重建episode并清history/context，不标exact。
5. 大文件hash在生产者/任务容器：控制面候选/收集/评估预览检查manifest/binding/路径/大小/声明hash，不解析GB replay/optimizer；实际保存/发布/恢复/导出默认完整hash，恢复在读张量/更新前拒绝损坏。等长损坏测试证实元数据预览通过而实际加载拒绝。校验发生于任务启动后，可能已占GPU；预览不声称全部负载已核验。
6. B未有真实资格，因此C/新四卡未实施。旧BFM四卡运行证据不转移为native UFO资格；可选A未选择。

## 三轮复核与独立审查

1. **设计/本体/数据隔离：** 核对身份保护、source/license、v1/history/action、祖先split、train-only8/250；修复test split兼容、root-link/COM速度混用、raw父重编码遗漏。资产仓只读。
2. **GPU/step/恢复失败：** 核对一个learner/六Adam、warmup0、1024触发、16→24截断及qualified24→1024门禁、target/norm/pool/RNG/replay；修复专家设备迁移，补缺失/额外/损坏状态、越界/符号链接故障矩阵。真实GPU取消/租约未验证。
3. **图表/媒体/证据：** 修复实际train/ufo tag，验证CLI→发布→消费/ONNX；共享缺媒体/损坏视频incomplete及validation/acceptance隔离回归。无本次策略录像/曲线/case结果，不借历史录像、不声称质量改善。

按executing-plans要求进行独立代码审查及修复复核。审查发现Dockerfile覆盖基镜像PYTHONPATH造成RoboLab不可见、环境probe缺URDF子挂载，均修复并重建。最终精确镜像CPU确认Isaac/RoboLab可定位、只读URDF alias存在。最终小范围复核未发现新阻断，独立checkpoint回归4项通过；审查未启动GPU/平台Job。

## 最终容器验证

所有Python/Node执行均在容器内；native `CUDA_VISIBLE_DEVICES=`、network none，仓库只读挂载。以下都是开发证据，不是正式训练Run/Artifact。

| 检查 | 实际结果 | 原始证据 |
|---|---|---|
| API/adapter/recipe/恢复/DCP/数据/Profile/容量16文件 | 140 passed | [backend.txt](artifacts/2026-10-10-ufo-rlinf/backend.txt) |
| 原生worker/replay/raw/Isaac边界/checkpoint/ONNX与CLI | 40 passed | [native.txt](artifacts/2026-10-10-ufo-rlinf/native.txt) |
| 共享入口/pool/既有四卡/并行评估 | 34 passed、3 skipped（缺Torch） | [shared.txt](artifacts/2026-10-10-ufo-rlinf/shared.txt) |
| 跳过项在任务镜像补跑、共享评估与视频解码/损坏拒绝 | 10 passed，覆盖上述3 skipped | [media.txt](artifacts/2026-10-10-ufo-rlinf/media.txt) |
| Node与vue-tsc | 14 passed、类型检查exit0 | [frontend.txt](artifacts/2026-10-10-ufo-rlinf/frontend.txt) |
| 正式api Dockerfile web阶段 | Vite exit0、725模块，已有chunk-size警告 | [web-build.txt](artifacts/2026-10-10-ufo-rlinf/web-build.txt) |
| 新增实现模块Ruff | All checks passed | [ruff.txt](artifacts/2026-10-10-ufo-rlinf/ruff.txt) |
| 最终镜像/source/CPU差分 | build exit0；image内source与代码一致 | [image-receipt.json](artifacts/2026-10-10-ufo-rlinf/image-receipt.json) · [image-build.txt](artifacts/2026-10-10-ufo-rlinf/image-build.txt) |
| 完整网络CPU census及模块/URDF路径 | CPU真实分配/可定位，无物理启动 | [model-census.txt](artifacts/2026-10-10-ufo-rlinf/model-census.txt) · [image-paths.txt](artifacts/2026-10-10-ufo-rlinf/image-paths.txt) |
| diff空白/Make默认入口 | git diff --check exit0、DEFAULT_GOAL=env | 无Git写操作 |

命令和测试清单见[commands.md](artifacts/2026-10-10-ufo-rlinf/commands.md)。合成fixture/模拟Channel/Env/测试专用API settings只是隔离依赖，不能当正式数据/部署证明。pytest asyncio配置提示、ONNX弃用提示不影响通过。

RED→GREEN实际观测包括binding缺失、source contract、replay接口、seed边界、qualified恢复门禁和probe挂载；不声称所有新增检查先观测RED，容量census等属于补证据。

## 后续门禁与预算

本次GPU/长训预算=0、无正式联合更新；CPU fixture不计训练进度。未预提交后续Job或启用pipeline续训。

下一阶段先部署/schema核验、正式raw数据Job，固定GPU agent/环境probe及实测准入，然后冻结B资格预算，初始化16例基线→单卡16→24→独立快评/全部录像/取消释放。证据齐全后提升B，再依计划实施C、选分支验证四卡/较小拓扑，另行冻结总预算和1000更新区间决策。正常行走须独立acceptance62000–62099至少80/100且满足原速度/跌倒/NaN/限位/媒体判据。
