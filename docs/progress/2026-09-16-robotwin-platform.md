# RoboTwin 页面、后台与数据落地记录（2026-09-16）

设计依据为 `docs/design/08-robotwin-integration.md`。本记录只说明实现与实测结果，不改变设计。

## 已完成

- 在既有数据模块中增加固定 revision 的 RoboTwin 目录、受管下载、失败重试、安全解压、内容探测和来源回执；耗时工作仍由 `ebd` 创建 `eb/tools:latest` 任务容器。
- 目录列出官方 50 项任务及 ALOHA/AgileX、Piper、Franka-Panda、ARX-X5、UR5-WSG 五种原生本体。除 ALOHA 外，本体接口尚无实值契约，页面显示但阻断下载和训练。
- 下载并登记 ALOHA LeRobot v2.1：Dataset `ds_ab3ed1ef3aef493987ab`，固定 revision `981c92aa34d8f94d4cff47e0d5bc2f7d4e0af042`，落地目录 `/data/kk/data/ds_ab3ed1ef3aef493987ab/content`。
- 下载包为 2,395,762,459 字节，SHA-256 `98a1d4bfb51d6e90469c9ef8d55e004a2cba01d20374adb68b554e48e9fe2cd8`；展开后 2,400,063,470 字节、2,500 episodes、548,893 frames、15 Hz。
- 内容探测确认 LeRobot v2.1、14 维 state、14 维 action、`cam_high`、`cam_left_wrist`、`cam_right_wrist` 三路视频；许可证为 MIT。数据记录已是 `validated`，来源 revision、文件摘要和落地路径均进入回执。
- 训练仍统一走 RLinf。六个 RoboTwin SFT 模板中，π0.5、LingBot-VLA-2、原力灵机 DM0.5、GR00T-N1.7、Xiaomi-Robotics-1 已有可提交的真实 worker；星海图 G0 继续显示具体阻断原因，不提交占位训练。
- π0.5 预览要求基座模型包含权重以及 `physical-intelligence/robotwin/norm_stats.json`，不会复用 LIBERO 归一化统计。
- 已下载并登记 RLinf π0.5 RoboTwin ALOHA SFT 基座：Model `mdl_6bde5878fb8c49dfa888`，固定 revision `fa8df6ed103db0f5549c122f3a17c00ba6426c98`，路径 `/data/kk/models/mdl_6bde5878fb8c49dfa888`，校验体积 8,529,288,523 字节；三个 safetensors 分片及 RoboTwin normalization 均完成 SHA-256 校验。
- 前端在数据接入、数据详情、数据列表、训练新建、模型列表中显示来源、本体、兼容性、训练就绪状态和六模型能力矩阵。
- 除星海图 G0 外，目标模型的公开权重均已下载并登记到 `/data/kk/models`。其中 LingBot-VLA-2 为 `mdl_cdb98e9fb8e047c990d4`，DM0.5 为 `mdl_9293dccffa114be8931d`，GR00T-N1.7 为 `mdl_1a4053f309844e43be9f`，Xiaomi-Robotics-1 为 `mdl_840294c597c94d928052`；π0.5 为 `mdl_6bde5878fb8c49dfa888`。LingBot 依赖的 Qwen3-VL tokenizer/config 也已登记为 `mdl_021d69c8adcd4c838fa8`。
- DM0.5 使用独立镜像 `eb/rlinf:robotwin-dm05` 和 OpenDM worker，平台单步烟测 `trn_9c1642f6cc56483cb9f9` / `run_cbff881cf3bb4ddfa294` 成功：`loss=0.449`、`grad_norm=1.53`，完整 checkpoint 与发布模型均已生成。
- LingBot-VLA-2 使用独立镜像 `eb/rlinf:robotwin-lingbot-vla2`、官方 V2 builder/processor/RoboTwin mapping，并由 RLInf 负责优化与 checkpoint。平台四卡单步烟测 `trn_f119eb129c2a4c40befc` / `run_756390204cbc4e0e986d` 成功：`loss=0.315`、`vla_loss=0.314`、`grad_norm=1.45`；发布权重 24,906,149,789 字节，checkpoint manifest 状态为 `complete`。
- GR00T-N1.7 使用独立镜像 `eb/rlinf:robotwin-gr00t-n1.7`、N1.7 modality/tag、三相机 PyAV loader 和本地 Cosmos-Reason2。平台四卡单步烟测 `trn_b62371c3913f4038a472` / `run_25cc110b66794d1e8f3d` 成功：`loss=1.49`、`grad_norm=0.403`；发布目录 13,837,944,619 字节，训练模型登记为 `mdl_454ac2ef3089426ebdb3`。
- Xiaomi-Robotics-1 使用独立镜像 `eb/rlinf:robotwin-xiaomi-xr1`、官方 XR-1 loss/processor、ALOHA URDF 正运动学和 60 维相对 EEF action view。平台四卡单步烟测 `trn_2471794e0ede4a4db1b0` / `run_0306fb4976414dabb462` 成功：`loss=3.18`、`loss_freq=2.47`、`loss_l1=0.439`、`loss_mse=0.878`、`loss_score=0.106`、`grad_norm=30.7`；发布目录 11,004,773,078 字节，训练模型登记为 `mdl_d3f8fdc3833041d29e95`。

## 多轮复核结论

1. 首轮修复了原始 HDF5 包误走 LeRobot 校验、失败任务无法重试、列表不对账及既有 LeRobot v3 校验回归。
2. 运行态发现草稿接口把数据探测也拦住；G1 现只允许 `data.convert` 在草稿接口执行，训练、评估和部署仍要求 active。
3. 实际下载核实了压缩包大小、MIT 许可、包内唯一 LeRobot 根、真实 15 Hz、14 维双臂通道和三相机，未从文件名推断这些值。
4. 模型复核确认 RoboTwin π0.5 使用独立 OpenPI config 和 norm 路径；页面列出的跨本体能力不等于直接动作兼容。
5. 最终范围复核后，只把完成真实 RLInf 优化步和 checkpoint 的 π0.5、LingBot-VLA-2、DM0.5、GR00T-N1.7、Xiaomi-Robotics-1 标为可用；未把四种未探测本体、G0 或 BHL 迁移标为可用。
6. 模型下载实测发现长连接会中断；下载 worker 现对大权重使用固定 Range 并发分段，对普通 Hub 传输进行自动续传，失败记录也可从原目录继续下载。
7. LingBot 平台烟测依次暴露并修复了 Ray/protobuf 版本冲突、官方 joint mapping 被错误转换为 dict、以及 norm stats 相对路径依赖；最终用同一页面提交链路验证数据、模型、四卡训练、指标和 checkpoint 发布。
8. GR00T 烟测用 8 帧 shard 降低协议验证等待时间，并确认 24 GB GPU 上需要四卡 FSDP；XR-1 烟测补齐官方 `liger-kernel`，在 meta 初始化后重建非持久化 rotary/Beta 张量，并避免直接读取的 sink embedding 被单独切分。

## 保留门禁

ALOHA 本体环境和多任务 Task 已按已验证的 ALOHA 14 维 joint-position 契约发布，π0.5、LingBot-VLA-2、DM0.5、GR00T-N1.7、Xiaomi-Robotics-1 可从页面提交。其余本体仍保持门禁，防止“维度相同”被误当作动作语义一致。

G0 权重尚未落地，且没有公开的 G0 RoboTwin action profile，不能借用 G0.5；其模板继续保持阻断。

全量 RoboTwin 仓约 1.53 TB，当前磁盘不适合一次性镜像全部数据。本次只下载设计中的首期 ALOHA v2.1 训练包；其余任务/场景按页面选择和受管预算逐项下载。
