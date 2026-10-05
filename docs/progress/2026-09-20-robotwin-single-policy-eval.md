# RoboTwin 单 Policy 评估实现与实测

日期：2026-09-20

本次按 `docs/rlinf/02-robotwin-single-policy-eval.md` 完成单容器、单 Ray、单 Policy Worker、多 Env Worker 的评估路径。RLInf 以固定源码哈希和补丁哈希构建进任务镜像；运行时源码快照只作为收据挂载，不再覆盖镜像内补丁。

实现后的固定档位为 1 GPU/4 环境、2 GPU/8 环境和 4 GPU/16 环境。Policy 固定为一份，推理 micro-batch 为 4、最大在途 batch 为 1；环境保持离屏渲染，关闭视频。tail wave 保持每个 Env Worker 的 4 个槽位不变，以占位种子维持 Ray、Vulkan 和 SAPIEN 生命周期，只为 active slot 写入证据。

实测结果：

- `job_4400265322c54a3088e8`：1 GPU、4 槽、5 case、两轮 wave，状态 `succeeded`，5/5 有效，`model_load_count=1`，容器实测内存 18.24 GiB。
- `job_4c0c6692b02e42a28458`：4 GPU、16 槽、5 case，状态 `succeeded`，证据按 rank 分布为 2/1/1/1，5/5 有效，`model_load_count=1`，容器实测内存 24.14 GiB。
- 两次运行均得到 2 成功、3 超时，证明多卡分片未改变这 5 个固定 case 的评估结果。

回归：后端 282 passed、1 skipped；RoboTwin 前端检查 4 passed；Vite 生产构建通过。未启动 5000 case 全量评估，避免在功能验收阶段消耗完整评估时长；页面可直接选择全量 profile 后提交。
