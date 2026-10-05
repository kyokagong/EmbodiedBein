# RoboTwin 全量评估操作指南

## 当前固定范围

平台按 `docs/design/09-robotwin-evaluation.md` 使用 RLInf 驱动 RoboTwin。当前全量 Suite 为
`suite_robotwin_rlinf_full_clean_v1`，固定以下输入：

- RLInf revision：`018e6a9e0da60f96eeda8b11008543911027d414`
- RoboTwin revision：`0008ae6800df9f75fc8de7098bacb01735fd8fd2`
- 官方资产 revision：`981c92aa34d8f94d4cff47e0d5bc2f7d4e0af042`
- 22 个 RLInf 已支持任务、5000 个 success-seed case、`clean` domain
- ALOHA/AgileX joint14 控制器、三相机与 joint14 observation

这是当前固定 RLInf 协议的全量范围，不等同于 RoboTwin 主仓所有 50 个任务，也不标记为官方完整榜单。

## 已下载数据

评估所需文件已经下载并验证，无需再次下载：

| 内容 | 路径 | 状态 |
|---|---|---|
| RoboTwin 仿真资产（约 9 GB） | `/data/kk/data/robotwin-assets/981c92aa34d8f94d4cff47e0d5bc2f7d4e0af042` | `verified` |
| ALOHA LeRobot v2.1 clean 示教集（2500 episodes，约 2.4 GB） | `/data/kk/data/ds_ab3ed1ef3aef493987ab/content` | `validated` |
| 22 任务评估 seed manifest | `/data/kk/RLinf/rlinf/envs/sim/robotwin/seeds/eval_seeds.json` | SHA-256 固定 |

闭环评估读取仿真资产和 seed manifest；LeRobot 示教集用于训练，不作为评估 case。

## 页面操作

1. 打开 **评估 → 新建评估**。
2. 基准选择 **RoboTwin**。
3. 模型版本选择状态为 `smoke_passed` 或 `validated` 的模型。
4. 套件选择 **RoboTwin RLInf full clean · 22 tasks / 5000 cases**。
5. 首次先选择 **冒烟**并点击 **检查输入与计划**。冒烟会覆盖前 5 个不同任务，每个任务 1 个 case。
6. 检查预览中的模型版本、5000-case Suite、镜像 digest、RLInf/RoboTwin revision、GPU、只读挂载和 argv，然后提交。
7. 冒烟报告为 `complete` 后，新建相同 Suite 的 **标准**评估。标准档执行全部 5000 case。
8. 在 **评估任务**查看 `recorded / planned`；完成后从任务详情进入 `EvalReport`，检查完整性、每任务成功率、超时和失败证据。

`validation`、`standard` 和 `acceptance` 当前都运行冻结的 5000 case；用途不同：validation 用于模型选择，standard 用于固定协议报告，acceptance 用于候选冻结后的验收。只有 `acceptance` 才应进入发布门禁。

## 执行行为

- 平台按任务分片，每个分片最多 8 个 case，避免在单 GPU/8 CPU 主机上同时创建数百个仿真进程。
- 单卡标准评估会反复初始化任务和模型，预计长时间占用 GPU；请在资源窗口内提交，并以页面的 `recorded / 5000` 判断进度。
- 所有分片仍属于同一个 Job、Run、plan hash 和 EvalReport；模型包只读，外部 RLInf/RoboTwin 仓不会被修改。
- 中断时保留已完成 case，并生成 `incomplete` 报告；重新提交会生成新 Job，不会把两次运行拼成一个有效报告。
- `Job=succeeded` 表示 5000 个计划 case 已完整对账；模型资格仍由报告中的 `qualification` 单独表示。

## 当前模型限制

- π0.5：跨任务闭环 smoke `job_6383dfdaefb64a22a576` 已完成 5/5，有效证据完整；该模型 5 个 case 均超时、成功率为 0，表示管线可运行，但当前 checkpoint 在这些任务上未成功。
- LingBot-VLA-2、GR00T-N1.7、DM0.5：缺各自已验证的 closed-loop policy bridge，页面保持阻断。
- Xiaomi-Robotics-1：60D EEF controller/IK bridge 未验证，页面保持阻断。
- 星海图 G0：缺固定模型包与 RoboTwin action profile，暂不显示为可评估模型。
