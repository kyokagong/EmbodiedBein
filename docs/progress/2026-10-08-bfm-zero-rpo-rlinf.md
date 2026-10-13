# BFM-Zero / RPO / RLinf 执行记录

> 2026-10-09补充：真实物理回放发现当日诊断motion converter误解Lab源wxyz四元数。旧Run/数据/模型保留作生命周期追溯，新训练已拒绝旧转换版本；修正后数据和2更新RLinf smoke见[后续实测](2026-10-09-bfm-data-replay-and-100style-trial.md)。本文件的成功不表示旧prior姿态或行走质量正确。

计划：[2026-10-08 执行计划](../superpowers/plans/2026-10-08-bfm-zero-rpo-rlinf.md)。设计：[14 · 接入设计](../design/14-bfm-zero-rpo-rlinf-integration.md)。日期按 Asia/Shanghai。

## 当前状态

Task 0–6 的 smoke 范围已完成。BFM 已通过平台 RLinf 训练/续训、ModelVersion 发布和带 command-to-latent 的 ONNX 导出 smoke。所有改动留在当前工作区，未执行 Git commit。

## Pre-flight 与裁定

- Task 0→1→2→3：关节/body/codec 来自同一资产探针；实测 24 body、P=358 后才冻结合同。
- Task 1→4→5：motion view 的 manifest/asset/codec/split hash 从转换、准入到实际 loader 一致；禁止把 AMP 74 维窗口当作 BFM 全 body 状态。
- Task 2→3→5：完整模型、六 optimizer、四类 target 与 replay 清单统一；恢复标记 trainer_state，不声称物理逐步 exact。
- Ruling：用户已要求根据现有设计制作并执行计划，视为本轮实施授权，不再重复请求设计/执行确认。用户禁止 Git 提交优先于技能的提交/工作树台账建议；在当前非 main 分支保留现有修改，以文件台账和运行结果记录进度。成本：最终审查必须基于未提交差异。
- Ruling：诊断数据仅允许受限接入 smoke；不升级成正式训练合格数据，不放宽设计中的 P4/P5 验收。成本：冒烟成功不能推导模型会走路或正式可替换 Flat。

## Task 0：输入与运行时

- 初始差异：设计文档 14 与 design/README 已修改，`scripts/` 未跟踪；本轮不覆盖该目录。
- 当前分支 `feat/make-features-real`。目标 RPO 基础镜像 `eb/rlinf:rpo-v2-wbc` 已存在；测试镜像 `eb/test:local` 已存在。
- 容器探测：PyTorch `2.7.0+cu128`，4 张 RTX 4090，各空闲显存约 24.84 GB；这是实际容器证据，不是宿主推断。
- 已有 motion v4 `ds_518dad79f5d3482e8db2` 为诊断来源，FK 已记录通过，步行动态跟踪未通过；正式来源权利链仍有限制。
- 基线容器测试：Flat/RLinf 表单、adapter 和平台测试 **23 passed**；日志 `/tmp/bfm-baseline-tests.log`。
- 真实 Isaac 两环境 source probe passed：24 个 body，root=`base_link`，P=358；原生关节顺序与 Flat 相同，50/200 Hz。回执 `/tmp/bfm-zero-probe/source-probe.json`，日志同目录 `source-probe.log`。
- 动作范围核对：v4 左/右膝相对默认姿态最大偏移约 1.897/1.475 rad，不能统一继承 G1 gain=5。初始 BFM gain 按各关节 `ceil(1.1×最大偏移/0.25)` 且至少 1 冻结；此覆盖仅针对诊断片段，不宣称覆盖全部 Flat 行为。

## Task 1：codec 与 motion

- RED：新合同/manifest 两项测试失败，原因均为 `eb.rpo_bfm` 尚不存在；日志 `/tmp/bfm-task1-red.log`。
- Ruling：本轮 smoke 采用明确的新平地/零延迟环境 revision，历史按帧、从旧到新、零填充；不沿用 G1 的字段排序历史，也不声称正式 Flat 分布对照。零延迟让已命令/已施加动作一致，后续延迟随机化须派生修订并保留实际延迟历史。成本：当前 smoke 不验证延迟鲁棒性。
- GREEN：合同/manifest/可信 pickle 校验 **3 passed**；Torch codec 和专家连续窗/hash/RNG 行为检查通过。
- 环境 probe 初次没有成功回执，但 Isaac 关闭后容器 exit=0；显式异常输出定位为默认 plane 依赖远程 `default_environment.usd`。改为复用现有 AMP 的本地 procedural flat mesh，仍是几何平地；保留首次失败证据，下条记录了完整回执的重跑结果。
- 重跑 GREEN：真实两环境接口/部分 reset/纯观测读取 probe passed。随后从固定 A1/36_11 源派生 2 clips、691 帧受控 NPZ；完整 body 姿态来自 Isaac articulation，不从 AMP 特征猜造。初次本地转换 manifest SHA `e64bc9f8f0008b2712aa6530959ae77babe836cba35b147773b63f18cff17d43`；Task 4 的正式 DatasetVersion 是后续受管版本，hash 见下文，仍标 `smoke_ready=true, training_ready=false`。
- 进一步真实 golden probe passed：两 clip 的在线 state 最大误差 0，privileged 姿态最大误差 `7.152557e-7`；低 root 的 terminated+truncated 和另一个环境的时间截断末态在 reset 前保留。回执 `/tmp/bfm-zero-probe/interface-probe.json`、日志 `interface-golden.log`。此对照验证静态姿态/坐标，不把有限差分速度或动态可跟踪质量说成通过。
- Task 1 complete（tests：3 个领域测试；Torch codec、expert sampler 行为；真实 Isaac codec/末态/golden；来源转换与 loader 回读）。当时的 `/tmp` 视图只是开发证据；Task 4 已经经平台转换 Job 登记派生 DatasetVersion。

## Task 2：完整 BFM 算法

- Ruling：网络与 normalizer 直接复用固定上游最小源码包（base/nn_filters/nn_models/normalizers），保留许可证和文件 hash；学习 loss 拆成可求导函数，由 RLinf worker 负责 backward/optimizer/target。成本：仓库包含上游网络模块中的未使用类，但避免重新实现网络导致结构/初始化差异。
- smoke 配置保留 Actor/F/Q 双 embedding 差异、B/D、双成员 Forward、双 critic、四类 target、六组优化器；research profile 在 meta device 上核得 online `435,116,484`、target `400,806,352` 参数。
- RLinf 镜像行为脚本：codec、网络 shape、手算 FB loss、terminated discount、目标网络软更新、所有 online 模块梯度、连续八帧 expert latent、混合 latent、prompt 退化拒绝、CPU replay 淘汰和 RNG/sample ID 恢复均通过。
- command 前端补漏采用 TDD：先加入行为检查并观察 `ImportError: _CommandLatentGraph`（RED），再实现 ONNX 前的 frozen-reference graph；与 `command_to_latent` 在命令边界、站立、直行和混合样本对照通过（GREEN）。
- 当前没有将固定 batch 的 loss/所有梯度/一步参数变化与上游 FBcprAuxAgent 做完整逐值 oracle 对照；smoke 通过手算 FB loss 和实际 RLinf 更新验证，不把它表述成论文训练数值等价。

## Task 3：RLinf worker 与受管镜像

- 构建 `eb/rlinf:rpo-v2-bfm-zero`，镜像 digest：`sha256:eb0c3ae265d5258a864115a35f322ae27a8f3ba56d1c6940961e9828f6e6a67f`。
- RLinf 源码内容 hash：`sha256:dbaa8bed56be412f6d012ee34030822941945786cede6bf9bc3de0fec5445d48`；BFM patch hash：`sha256:d5b43b38a652e544d2ceee47634caf6e58b76ab15b5ff9409ad436c2f7fa100e`。训练镜像回执固定这两项；上游目录只读。
- 专用配置单 learner、一 action chunk、关闭 compile/graph；真实 RLinf run 日志含 `bfm/update_count`、`target_update_count` 和六组件 loss/grad metrics。
- Task 3 完成于平台运行证据与 runtime 行为检查；不宣称多 learner/多卡语义已验证。

## Task 4：数据、平台准入、checkpoint 与发布

- 正式平台转换生成 DatasetVersion `ds_e263952190bd4b068a99`：2 clips / 691 frames / 2.17 MB；manifest SHA `f591cd728de51ead67441591b72f7afafac1b2a2937dc77b7bcdf34211b03359`，asset SHA `b58868a782d8863ac38059eb4eab89d27ee1d70565bf58f5a055f44510c8e842`，codec SHA `fff6cf3e8f05c88fc7b552f65e5089dc47315c343bd9a205a1f59539cb2811c4`。准入为 `smoke_ready=true`、`training_ready=false`。
- 首次收集暴露 checkpoint package root 被误当成 output root；改正 collector 路径并用真实布局 fixture 验证。另将 actor obs signature 从误用 critic 733 修正为 actor 375、latent 256、critic_obs 733。旧 ModelVersion 不原地修改，以后续续训版本承载修复。
- 原始平台训练 Run `run_a466952b6f094f68b386` 完成 step 2；resume Run `run_f892448d06fb4c798563` 从 step 2 到 step 3；签名修复后 resume Run `run_405eeaf7cdf64c11a5f9` 到 step 4 并发布 `mv_1b91db2dc1494191aa06`。
- trainer state 恢复测试包含 BFM replay/更新计数和平台完整续训。replay 独立行为测试比较恢复前后 sample IDs；平台确有下一次更新，但未比较重启前后整次 optimizer 更新逐位相等，所以恢复模式只称 `trainer_state`，不称 exact resume。

## Task 5：actor 与连续 command-to-latent 导出

- 首次 ONNX export 暴露动作最后维度符号化；保留失败 Job `job_41dc8c2e532e472cbc71`，修复固定 23 维并通过 ONNXRuntime 检查。后续第一次平台成功导出后，反思发现仅保存 stand/walk 固定 z 不能覆盖命令控制；补出冻结 β/参考的 `command-to-latent.onnx`，将它作为独立 CAS Artifact 链接到 ModelVersion card。
- 最终平台导出 Job `job_4056f902e2fe42899d38` / Run `run_c235a1d8aedf48ffbc8e` 成功；导出 ModelVersion `mv_289c26f293424413a467`，actor Artifact `art_ae6d7e2aa42542d2a960`，command frontend Artifact `art_aaaabc4fce7f45dea3bd`。readiness=`diagnostic_smoke`。
- Actor ONNX：`observation[375], z[256] → action[23]`，batch `[1,4]`，torch max abs `1.043081283569336e-6`，joint target max error `1.1473894119262695e-6 rad`；normalization 已嵌入。
- 命令 ONNX：`command[3] → z[256]`，冻结 691 个 train references（上限 4096），batch `[1,6]`，Torch parity max abs `1.5497207641601562e-6`；prompt 文件含 goal、tracking、stand、walk 四类。不同 command 和 latent 已用实际 ONNXRuntime 验证。
- command ONNX Artifact `art_aaaabc4fce7f45dea3bd` hash `d1762f35f0843652fe62a551c07f0fddb66ecd4d92e748e88e3cabd0862b36f8`，CAS 大小 724,704 bytes；ModelVersion card 通过 `command_latent_artifact_id` 单独引用，不误注册成 actor。collector 对真实导出通过，tampered command hash 被拒绝。
- 额外负例 smoke：错误 asset hash 在导出时被拒绝；零/退化 latent 被 prompt API 拒绝。尚未测端侧完整周期时延、GPU/RAM 峰值和连续命令下闭环步态质量。

## Task 6：最终反思与回归

1. 数据/物理：只使用 RPO 原生交错 23 关节序；body=24、P=358；静态在线/专家 state 对齐，但诊断 clips 动态质量和授权不足，不能长训。
2. 算法：核对六组件、双 ensemble、四 target、pairwise FB batch、terminated discount；target 不进 optimizer。未做与上游完整更新的一步数值对照，记录为待补证据。
3. Reset/恢复/失败：部分 reset 和 reset 前末态由真实 Isaac probe 覆盖；错误 motion hash、shape 和缺状态被拒绝；checkpoint trainer state 有 replay/optimizer/target/RNG，仿真在新 episode 重启。未声称整次更新确定性 exact。
4. 模板/镜像/发布/提示：preview/Job 经过现有 Job→Run→ebd；只读 motion/asset；image、source、patch 和 ONNX/prompt/command-reference 都有 hash；导出 collector 登记 actor 与独立 command Artifact 血缘。
5. 资格边界：smoke 只证明管线和数学/导出接口能运行。100-case Flat、held-out tracking/goal、prompt switching、few-shot latent adaptation、sim2sim、目标板和真机均未通过/未做。

## 多轮反思

1. 输入复核：原 Flat 基线、原生交错关节序、body/P 未冻结、诊断数据状态均进入计划；没有把历史 AMP 训练成功当作动作合格。
2. 物理/数据复核：本地平地依赖、真实 link-origin 速度、逐关节增益、专家/在线坐标、纯观测历史、终止与截断末态逐项获得探针证据；诊断来源的动态质量与权利链限制保留。

## 验证命令与限制

- 聚焦容器回归：`62 passed, 1 warning`（passlib 的 Python `crypt` 弃用警告）。
- Ruff：相关改动文件通过 `--ignore E501`；长行规则按现有代码基线忽略。
- 真实 RLinf 镜像行为：`codec,networks,losses,replay` 通过；用登记 motion view 的 `inference,worker` 通过；新增 command graph 对照通过。
- 真实平台：BFM 初始训练、两次 trainer-state resume、actor+command-to-latent export 及模型/Artifact 登记成功；API healthz 返回顶层与 ebd 均为 ok。
- 导出 collector 对实际 Run 的 manifest/hash/ONNX 检查通过；复制到临时目录后改坏 command-to-latent hash，验证 collector 拒绝该产物。
- `ebctl ps` 确认本轮相关训练、续训和导出容器均已 exited；无遗留运行任务。API healthz 与 ebd health 均为 ok。
- `git diff --check` 通过。未执行完整 `make test`（Rust/全后端）或 100-case 评估；这轮交付按用户要求以聚焦测试和 smoke 为验收。
