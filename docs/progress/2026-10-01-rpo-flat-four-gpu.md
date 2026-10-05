# RPO-Flat：step-3000 后四卡扩容训练

用户授权从 `trn_78b1ffc69ca8457c9e05` 的 step-3000 checkpoint 新增 2000 轮 PPO 更新，4 张卡，每卡 2048 个环境。仍走 API → Job/Run → ebd → RLinf，不提交代码。

## 配置和初始化语义

- 源模型 `mv_56c5185022c14fd6aded`，源 Run `run_3e1d769a5dab4e28b367`，已发布完整策略和 8 个 empirical-normalization buffers。
- 按设计 `04-model-training.md` §6，改变 GPU/world size 和 batch 属于 `warm_start`：保留全部模型与归一化，重新创建 optimizer、scheduler、rank RNG、环境和 rollout。没有放宽原 `trainer_state` 兼容指纹。
- 新 Run 从 step 0 训练到 2000；源 3000 + 新增 2000 = 累计 5000 轮。模型卡同时记录本 Run step 和累计 updates，不把新 Run step 2000 伪装成恢复 step 5000。
- actor/env/rollout 都放在 GPU 0–3；训练环境总数 8192，每卡 2048；eval 总数 32。
- 每轮 24 步，196608 transitions；global batch 49152，micro batch 3072，每 rank 4 次梯度累积。每 epoch 4 个 minibatch，5 个 PPO epoch，合计 20 次 optimizer update。
- 新增交互 393216000 transitions。源 normalization count 147456000；全局统计无重复计数时最终应为 540672000。
- 初始 actor/value LR 使用源 DCP 实际保存的 `3.3750000000000014e-5`。这来自容器内读取 `fsdp_checkpoint.optimizers.param_groups.0.lr` 与 scheduler `_last_lr`，不是训练末尾日志中的平均 LR。Adam 动量仍按 warm-start 语义重置。
- 官方环境、完整奖励、指令分布、随机化、网络和 PPO 其余参数沿用现有 Flat recipe。预算 32 CPUs、96 GiB RAM、32 GiB shm。

## 核查和反思

1. 原恢复 API 和 adapter 两层指纹都冻结拓扑；不能只删一个限制，称为相同 trainer-state 恢复。复用已发布模型输入和新 Run，明确 warm start。
2. 显存估算先前把总环境数当单卡环境数；四卡 Flat 按 2048/卡估算，避免错误拒绝，也不降低实际显存预算。
3. 多卡 learner 必须共享全局归一化统计、KL 和 adaptive LR；单独更新各 rank 的统计会让 rollout 策略分叉。保存必须 barrier 后仅 rank 0 写完成清单。
4. 源包摘要在任务容器启动时检查；strict 模型加载和 8 个有限 norm buffers 必须保留。摘要或结构不一致直接失败，不从随机初始化继续。
5. 训练 success 与模型 qualification 分开；step-3000 的 19/100 通过率是基线，不因扩大环境数推断新模型合格。
6. 原 RLinf DCP 的 RNG 使用共同 key；多卡补充 `actor/rng/rank_N.pt` 独立保存和恢复，保存/加载失败在全部 rank 同步。候选与 adapter 都要求四份文件，历史缺文件的多卡冒烟不再宣称完整可恢复；单卡旧 checkpoint 保持兼容。

## 验证与执行证据

- 初步容器测试：Flat batch/warm-start 校验、原 resume、RLinf form 和 capacity，共 52 passed。
- Vue 类型检查通过。实际 task 镜像回读已发布 step-3000 包：normalization count 147456000；改变权重 SHA 时拒绝加载。
- 最终聚焦容器回归：89 passed；四个真实 CPU FSDP/Gloo rank 的统计、KL/LR 和 rank-0 回执检查通过；现有单 rank patched-policy smoke 通过。
- 四卡 warm-start 冒烟：Train `trn_8a7363c6120c4f5ab5a7`，Job `job_b2ad2bfa21cf4fc2ae91`，Run `run_c3ad9b457143443499e7`，完成 2 轮并成功发布。四个 EnvGroup 日志分别确认 2048 个环境。源包 SHA 验证与加载回执 `loaded/world_size=4`；checkpoint 有 4 个 DCP 分片，全部模型张量有限；actor/critic normalization count 均为 147849216，增加量 393216 恰好为 8192×24×2。
- 同拓扑恢复冒烟：Train `trn_4f3d88f093054902b340`，Job `job_aabe413e59134fefb093`，Run `run_b48a557057f64d9b9325`，从上述 step 2 恢复到 step 3，成功发布。回读 Adam step=60、scheduler last_epoch=3、normalization count=148045824，回执 `loaded`。没有再次注入初始 warm-start 权重。
- 第二轮审查补充：暖启动来源与本次加载模式分开，允许新 Run 后续进行同拓扑 trainer-state 恢复；通用 Job/ToolBus 入口共用中央 DB 绑定检查，拒绝伪造模型 ID、源 step 或权重路径。两项均有行为测试。
- 首轮任务镜像 `sha256:39d61a91ca775c05ef06a6ea2a8b68b7037fca60182b4165c2fcfdce3587381b`，上述恢复仅验证模型/optimizer/scheduler/norm；没有独立 rank RNG，不能作为最终完整四卡恢复证据。最终镜像 `sha256:923a4d446116ba2738896ec658402bcb32e6973e5224aae1f5d0d1bb9e37da58` 增加独立 RNG；源 RLinf 内容 hash 保持 `sha256:dbaa8bed56be412f6d012ee34030822941945786cede6bf9bc3de0fec5445d48`。
- 首次正式任务：Train `trn_bb12a04005fb4a42b1a4`，Job `job_f0014434495e46edb53a`，Run `run_50fdd64b98ec41b38052`。在加载前失败，无训练更新：后续修复把 active warm-start receipt 的 dict 误转为 bool，入口因此拒绝。已修正，并补充 actual adapter JSON 回执值和 ckpt_path 的行为检查；失败记录保留，重新创建正式任务。
- 最终镜像四卡冒烟：Train `trn_3c462145da424f5fa34a`，Job `job_e38ee47655e4409db6ed`，Run `run_38b35834d29c44d9bf1b`，完成 2 轮成功发布；normalization count=147849216，全部模型张量有限，完成清单含全部 4 份 rank RNG，CPU 回读确认每份都有 CPU/NumPy/Python/CUDA 状态。不同 seed 的四 rank 精确 roundtrip、单 rank 写失败/缺文件/损坏文件的全 rank 拒绝均在 CPU Gloo 检查通过。
- 最终镜像恢复冒烟：Train `trn_ef5baf072fbe46059af1`，Job `job_247da038a9924d0ca3b0`，Run `run_34e3f4b5f4814e36b0d5`，从 step 2 恢复到 step 3 成功。回执 loaded，清单再次包含四份 RNG，回读 Adam step=60、scheduler=3、normalization count=148045824。
- 正式训练重新启动：Train `trn_9884cea30175494baf70`，Job `job_1457133f34f248a29161`，Run `run_0ebf15f1328e484b9ee2`，2026-10-01 17:03:37（Asia/Shanghai）启动，状态 running，实际 GPU [0,1,2,3]。目标 max_epochs=2000，save_interval=50，模型名称 `RPO-Flat cumulative 5000 4GPU`。
- 正式 Run 的 warm-start 回执已 loaded/world_size=4，原源模型 `mv_56c5185022c14fd6aded`、step 3000、权重 SHA `bf7a263873970be6b7c774aac46e8631bc8b7497a10f5e8fe19d3e3710d6a30e`、初始 norm count 147456000；未使用任何冒烟模型。
- 2026-10-02 回查：正式训练于 2026-10-02 04:11:21（Asia/Shanghai）成功完成新增 2000 轮，发布模型 `mv_784eb4ce52d240719aba`；累计 5000 轮。权重 SHA `49783611974c1915dbcfe352ba95fdc88a19faafb6c68a6b11708f0832a61c28`，容器回读全部张量有限，actor/critic norm count 均为 540672000。API `/train/{id}/checkpoints` 返回完整已发布 step-2000；`resume-candidate` 返回 available=true，含四份 rank RNG，总大小 23576913 bytes。用户截图中为空的 checkpoint 在本次 API 回查时已可用。
- 用户要求全部评估案例录像：首轮标准 100 例 Job `job_4c8eb0d5d91d48fcb4b4`、Run `run_41ecd4da95804964ab1c`；固定 seeds 50000–50099，record_video=true，官方完整速度/heading/转向指令与原冻结通过判据。计划 hash `sha256:faa9972a7f8bedd39c87f329a083875ef32f3801616fed6278f7a225cdd1ddc8`。ONNX export/v3 已 verified，关节目标最大误差 3.147125244140625e-5 rad。首轮因 CPU 配额瓶颈被下述新任务替代。
- 全录像评估原 4 CPU 容器持续满额，整例运行缓慢。ONNX 推理单独测量约 0.05 ms，无需改推理代码。通过平台补跑同模型 seed 50003 的 16 CPU 诊断：Job `job_946c86c130b546dca85c`、Run `run_21864623f3ef4b5ea4d9`、报告 `ev_aee9252fc34844809606`；完整 1000 步通过，4/16 CPU 的全部指令、动作与物理轨迹逐记录一致，XY/yaw RMSE 分别为 0.16783490316752958 / 0.13000102009061376。
- 原任务在 15 例已完成且有录像后由平台取消，保留原 Run 的全部已有证据，不用于 100 例资格判定。复用原冻结 JobSpec 通过平台 `/jobs` 创建新任务，CPU 配额增加至 32，GPU=1、memory=24 GiB、shm=8 GiB，其余参数及计划 hash 不变：Job `job_9045ebfba3b543f9a82e`、Run `run_d886ddc9b1834ddd9994`，2026-10-02 19:05:12（Asia/Shanghai）启动，从全部 100 例重新运行。

## 累计 5000 轮模型：100 例全录像评估结果

- 正式评估于 2026-10-02 21:29:17（Asia/Shanghai）完成。报告 `ev_86e1aa59a8e943059896`，模型 `mv_784eb4ce52d240719aba`，Job `job_9045ebfba3b543f9a82e`，Run `run_d886ddc9b1834ddd9994`。
- [网页报告与逐案例录像](http://1.94.0.106:31800/eval/reports/ev_86e1aa59a8e943059896)：在“实际案例”点击“播放录像”，播放器位于案例列表下方；失败案例库也可打开失败录像。
- 100/100 独立案例完成，重复和无效记录均为 0，运行异常 0；43 例通过、32 例物理硬越限、25 例速度/转向跟踪误差未达标。跌倒 0、NaN 0。报告 validity=complete，qualification=fail；Job succeeded 只表示运行和证据收集成功，未达到至少 80/100 的模型验收目标。
- 68 例走满 20 秒，32 例硬越限提前终止；77409 个实际采样步骤。按实际步骤加权的平移向量 RMSE=0.19924265256225457 m/s、转向 RMSE=0.1537594330465954 rad/s；汇总均值不能替代逐案例判据。
- 同套件 step-3000 基线报告 `ev_8d9bc9895eb54b358970` 为 19/100，硬越限 58 例，平移/yaw RMSE 为 0.24356945828503707 / 0.2736601228131389。本次 43/100 有改善，仍不合格。
- 实际指令覆盖前进、后退、横向运动、转向和站立：观察到 vx 范围 [-0.59185, 0.99188] m/s、vy [-0.48842, 0.49434] m/s、yaw [-1.57000, 1.54350] rad/s；28 例出现 standing flag，共 13909 个 standing 步骤、13895 个全零速度指令步骤。沿用官方每 10 秒重采样和 20% 站立采样配置，未改成固定站立/0.3m/s 直行。
- 每例一份 H.264 MP4，320×240、10 fps；100 段总计 15864968 bytes。提前终止案例保留相应短录像，走满的录像为 20.1 秒。所有录像位于 Run 的 `output/video/eval/seed_<seed>/0.mp4`，与 case、seed、真实轨迹一一对应。
- 容器化 Chromium/Playwright 实际检查：100 个“播放录像”按钮、100 个登录后 `Range: bytes=0-1023` 请求均返回 206/video/mp4，100 个视频均可解码并实际播放，画面尺寸与实际轨迹时长全部通过。检查脚本 `/tmp/verify-rpo-flat-videos.cjs`，页面截图 `/tmp/rpo-flat-5000-standard100-browser.png`。
- CPU 配额扩容复核：原 4 CPU 任务的全部 15 个已完成案例，与 32 CPU 新任务的 11021 个指令/动作/物理轨迹记录逐条一致。最终模型、官方环境、指令范围和冻结通过判据保持原绑定；此次仅复用平台 Job 资源配置与现有录像页面，没有修改发布模型或套件，没有 git 提交。

## 三个原生高清录像案例

- 用户要求提高录像清晰度，新增 `video_quality=hd` 的 RPO-Flat 评估选项，预览冻结该选项并核对运行回执。相机原生 1920×1080、25 fps，H.264/yuv420p、CRF 18；50 Hz 控制每两步录一帧，包含初始及终止画面。关闭编码器宏块自动缩放，避免 1080 行变为 1088。标准录像仍为 320×240、10 fps；旧计划缺字段时保持兼容，旧镜像命令不追加新选项。
- 沿用模型 `mv_784eb4ce52d240719aba`、源 Run `run_0ebf15f1328e484b9ee2`、checkpoint SHA `sha256:7eeb8a262f0a7e344c4c2d5960876afc521c310742db22024951a014269c7617`。新评估镜像 `sha256:3c887aab31c40764bd46dd595767551f1207a1b2ea1058a8a1de75b112fda85b`，生产模型镜像仍绑定 `sha256:923a4d446116ba2738896ec658402bcb32e6973e5224aae1f5d0d1bb9e37da58`。没有改写历史模型、套件或报告。
- 平台 `/evals/rpo/preview` → `/evals/rpo/jobs` 创建 Job `job_c5d4478a1de643d0a7e8`、Run `run_e90e7d34a77f4aaab2c2`，GPU=1、CPU=32、RAM=24 GiB、shm=8 GiB。计划 hash `sha256:4ed6e898b92fa4ceb74be343f6b04e47291e34623f99691296b1d35de52dcef9`；2026-10-02 22:56:32 至 23:03:52（Asia/Shanghai）。采用既有 smoke 的固定 seeds 60000、60001、60002，每例上限 20 秒，官方速度、侧移、heading/转向指令及通过判据保持原套件定义。
- [高清评估报告及三段录像](http://1.94.0.106:31800/eval/reports/ev_aaae9e15a2b044038d9c)。3/3 案例完整收集，2 例通过、1 例物理硬越限，跌倒/NaN/基础设施异常均为 0。validity=complete，qualification=not_requested：三个可视化案例不替代前述 100 例验收，也不改变此前 43/100 的结果。

| seed | 结果 | 物理步骤 | XY RMSE (m/s) | yaw RMSE (rad/s) | 录像时长 | 录像帧数 |
|---|---|---:|---:|---:|---:|---:|
| 60000 | 通过 | 1000 | 0.1487486194 | 0.0976960550 | 20.04 秒 | 501 |
| 60001 | 物理硬越限 | 317 | 0.1304774697 | 0.2065686254 | 6.40 秒 | 160 |
| 60002 | 通过 | 1000 | 0.1786587626 | 0.1001303197 | 20.04 秒 | 501 |

- 三段真实 MP4 均回读验证 H.264、1920×1080、25 fps，总计 23876021 bytes；初始/终止帧使录像略长于实际物理时间。抽查原生画面，机器人全身、关节和足部完整可见。
- 容器化 Chromium 在报告页面找到 3 个播放按钮，3 个媒体 Range 请求全部返回 206/video/mp4，三段视频均实际解码并播放，分辨率和时长通过。验证脚本 `/tmp/verify-rpo-flat-hd3.cjs`；页面截图 `/tmp/rpo-flat-5000-hd3-browser.png`，原生帧 `/tmp/rpo-flat-hd-seed60000.png` 等。
- 聚焦 CPU 容器检查最初 16 passed、1 skipped（Torch），真实任务镜像补跑录像循环等 6 passed；增加旧镜像兼容检查后，平台/adapter 的 12 项检查通过。真实编码小样确认没有宏块缩放。API 实测拒绝未启用录像的 HD 请求（400）和非法 quality（422）。通过 `make up` 部署控制面及兼容修正，未执行 git 提交。
