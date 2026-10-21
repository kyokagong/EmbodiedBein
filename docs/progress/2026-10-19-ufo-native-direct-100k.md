# UFO单次续训：20000 → 100000

**最新状态：已正常完成累计100000更新**，单次训练约4小时38分35秒；最终双后端六例均运行满20秒且未跌倒，但物理越限和跟踪未达标。完整结果和可观看视频见[100000更新可视化评估](2026-10-20-ufo-native-100k-visual-eval.md)。下文启动阶段计数是历史记录。

用户明确纠正：“直接一步就是20000->100000，不要分阶段训练”。这取代此前按10000更新区间结束/评估/恢复的执行方式。本轮只创建一个训练Job，起点20000、终点累计100000（新增80000），中途不停止训练、不安排独立validation、不创建下一段训练Job。UFO原生训练内tracking反馈保持原生算法行为。

原30,000分段训练 `trn_f80d9689108741feb594` / `job_a98c881712354ebe9811` 已通过平台取消，两个旧分段控制容器均退出。被取消Run不用于新恢复输入。源仍为 `trn_1d0a497175cf4ddd93c9` / `run_23bdb3114ef545b3b2e7` 的完整20000 checkpoint：`/data/kk/embodiedbein/runs/run_23bdb3114ef545b3b2e7/outputs/native/checkpoint`。

单次任务已创建：TrainJob `trn_8f34166b7d454a43bf63`，Job `job_483cd6f03c6d4b3280f4`，Run `run_55de123b5d4a4039bfe1`。实际冻结计划的 `start_updates=20000`、`target_updates=100000`、`experiment_target_updates=100000`，并非名称100k而本段终点30000。[训练页面](http://1.94.0.106:31800/train/trn_8f34166b7d454a43bf63)。

真实HTTP浏览器6项核查通过：旧20000计数保持、整体100000目标、单次任务入口、目标100000页面跳转、不分段提示以及无运行时异常。[浏览器记录](evidence/2026-10-19-ufo-native-direct-100k/direct-browser.json)、[实际单次页面截图](evidence/2026-10-19-ufo-native-direct-100k/direct100000-page.png)。

真实启动核查已通过：四rank初始权重指纹均匹配20000源rank0，原生CSV实际更新达到20256、全局环境步7946240，随后继续增长；当前唯一活动训练容器为上述Run。训练终点仍为100000，尚未宣称目标已完成。[实际计数与源指纹](evidence/2026-10-19-ufo-native-direct-100k/startup-verification.json)。

配置保持4×2048环境、Replay262144/rank、batch1024/rank及全部学习率、数据split和UFO/RLinf/资产/本体身份；恢复模型、全部五组Adam、四rank Replay与原生计数。环境/RNG/latent/专家优先级仍按原生能力重新初始化，不称exact resume。任务镜像仍为 `sha256:7217526a893b2293817b0210117f0b9427be055a63103ecd6d1682be23648af1`，未修改外部源或覆盖已使用镜像。

旧分段源的单Job墙钟只有5400秒，单次80000更新按实测约需5小时。增加显式 `native_new_budget=true` 与 `timeout_seconds=21600`：新预算进入preview hash和冻结plan，记录原预算/新预算/源TrainJob；默认恢复仍不能静默扩大预算，schema与任务运行时上限仍为6小时。此次新预算由用户明确的单次训练请求授权，不复用旧六小时实验截止时间。

原生环境步终点重新核算为48783360；checkpoint和Replay保存间隔均设到这一终点，沿用已验证的最终全rank `SAVE_AND_STOP` 分支，没有平台自建训练循环或中途杀容器保存。CUDA分配设置继承；1GiB额外脚本余量仅保留诊断，平台准入、实际OOM/NaN/Inf/rank/collective/timeout及完整性检查仍执行。训练失败保留源20000状态，不自动启动新阶段。

CPU监控容器只提交这一笔恢复训练，通过API→Job→Run→ebd执行。达到100000正常结束并完整回读后，再生成训练时序报告、prompt/export和双后端六例最终评估及录像；不声称正式acceptance通过。最终录像和结果只对应100000 checkpoint，不能以20000录像替代。

检查：先复现旧预算拒绝单次长训以及未显式设时限仍被接受的问题，再实现新预算；聚焦后端107项检查通过。单次客户端8项检查覆盖准确终点、显式6小时预算、完整Replay恢复声明和各失败终态拒绝。Vite构建随 `make up` 发布，用户页面明确“本任务直接训练至整体目标，中途不分段停止”。三轮复核分别核对设计/数据隔离、真实计数/预算/完整恢复与取消、最终媒体生成与未验收状态。实际训练是否越过20000、冻结计划及运行ID以[实时状态](evidence/2026-10-19-ufo-native-direct-100k/STATUS.md)和[运行核查](evidence/2026-10-19-ufo-native-direct-100k/startup-verification.json)为准，文件产生前不宣布已完成真实更新。

[单次协议](evidence/2026-10-19-ufo-native-direct-100k/protocol.json)、[源五Adam/Replay验证](evidence/2026-10-19-ufo-native-direct-100k/source-verification.json)、[代码hash](evidence/2026-10-19-ufo-native-direct-100k/operator/manifest.json)。未执行git暂存、提交或推送。
