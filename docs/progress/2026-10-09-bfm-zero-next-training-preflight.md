# BFM 下一阶段训练：输入与资源摸底

日期按 Asia/Shanghai：2026-10-09。目标：[设计 16](../design/16-bfm-zero-data-and-4gpu-training.md)，[执行计划](../superpowers/plans/2026-10-09-bfm-data-four-gpu-training.md)。用户确认 1000 step 为 1000 次 learner 更新，并说明目前没有其他数据，要求研究补充采样方法。

## 实测

- API / ebd 健康。Doctor 识别 4 张 RTX 4090，每卡 total=24564 MiB、free=24080 MiB，暂无关联 Run；主机 available RAM=106750 MiB。Doctor 的 Vulkan ICD 路径项为 false，不能据此推断渲染完全不可用；录像阶段须核对实际 ICD/渲染。
- 现有 BFM 数据 `ds_e263952190bd4b068a99`：2 clips、691 帧、`training_ready=false`。未改资格、未改源文件、未启动 1000 更新任务。
- 本地只读 `rpo_lab` 原始目录实际有 30 个文件。离线限资源容器读取：26 个使用 pickle，4 个使用 joblib；全部为 23 关节形状。共 20635 原始帧、188.5367 秒，源 FPS 为 120/50，不能用总原始帧数冒充 50 Hz 数据量。
- 按原时间轴重采样至 50 Hz，共 9431 帧。采用前轮真实 Isaac probe 的关节限位、冻结的 PD 目标范围，以及 converter 已有的 50 rad/s 静态速度筛查上限，共 4352 帧通过三项组合条件；18 个文件具有至少一个连续 9 帧窗口。该筛查尚未验证 root/body/FK、接触或真实跟踪，也未独立验证动作使用资格，不能称合格 DatasetVersion。
- 不少侧移、转向和跑步参考超出当前 gain 对应的 PD 目标范围，不能直接全量塞入原 BFM 接口。未裁剪 q、未放宽限位或改 action gain。
- [完整盘点与各文件 SHA/静态结果](2026-10-09-bfm-motion-inventory.json)已保存。JSON 保留首次 pickle 错误及 joblib 格式识别结果，避免把序列化差异误称数据损坏。
- Flat 教师候选 `mv_784eb4ce52d240719aba` 的标准报告 `ev_86e1aa59a8e943059896`：43/100、qualification=fail。没有将该教师或已有评估轨迹当作合格 BFM 专家。

## 当前差距

1. 数据需要新的冻结选择/FK/动力学验证与来源资格；用途（研究/可商用）问题已向用户提出，未推断答案。
2. BFM 当前 `build_overrides` 限制 gpus=1、batch=8；RPOBFMActor 要求 actor world_size=1。四卡可以设计为单 learner + 多采样 rank，但不是已验证的能力。
3. 当前 BFM 准入检查 smoke_ready，不是正式训练资格验证。正式 recipe 必须在 API、中央输入绑定和 adapter 共同落实 training_ready 与预算限制，保留受限 smoke 用途，不能利用已有检查宽松直接长训。
4. 未测每卡 BFM env 上限、组合峰值或 H2D 吞吐。原 Flat 2048 env/卡的结果不作为本轮 BFM 实测。
5. actor+command ONNX 导出 smoke 已完成，但独立 BFM 录像评估桥/冻结 suite/报告收集尚未完成；不能借 Flat 的推理接口直接验收。

本轮执行的是容器化只读盘点、静态筛查、实际 API/资源核对和原始来源文档检索。容器检查确认所有 30 个源文件 hash 未改变，9431/4352/18 的汇总和文档链接一致，`git diff --check` 通过。尚无新的模型、正式数据或评估结果。未进行 Git commit。
