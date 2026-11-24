# MotionDecode 原始数据接入（2026-11-10）

2026-11-11 终态补充：92567文件、147887248552 bytes已全部下载并通过文件hash检查；1个上游CSV仅有表头，原始版本保持invalid。按用户后续授权，精确排除该文件的有效版本已全量复核并发布，表格/动作视频与浏览器验收见 [11-11有效版本记录](2026-11-11-motiondecode-valid-view.md)。下文保留首次接入与下载启动时的历史快照，不作为最新终态。

## 范围与状态

用户确认本次仅下载公开数据、浏览和校验，不启动训练。来源：[CMRobot/MotionDecode](https://huggingface.co/datasets/CMRobot/MotionDecode)，固定 revision `6380f7420c83b20b6ad50124bc6cb829a7c0b178`。冻结清单包含 **92,567 文件，共 147,887,248,552 bytes（147.9 GB / 137.7 GiB）**；HF usedStorage 为 309,010,785,082 bytes，不等于该 revision 的实际文件量。

- Dataset：`ds_0d2332ff326e46ecb405`
- 落地：`/data/kk/data/MotionDecode/6380f7420c83b20b6ad50124bc6cb829a7c0b178`
- Job 类型：`data.convert/acquire_motiondecode`，4 CPU / 8 GiB / 0 GPU，72 小时硬上限；重试记录见下。
- 当前 Job：`job_7b95c61fd0c34d8da3da`，Run：`run_444fbee9664745eebb90`。
- 执行镜像：`sha256:8b73bd743187701605b177ccc66074aeb4ca35eca800acd3602119290d894eb3`。
- 工作台：`/data/ds_0d2332ff326e46ecb405`。
- 本记录初始状态：已登记、已提交下载；**尚未确认全量完成，不宣称校验通过**。实际状态通过 Dataset / Job 查询。

## 来源语义与许可（第一轮复核）

真实 CSV 有 36 列：root position **m**、root quaternion **wxyz**、29 个具名关节 **rad**。数据卡称 root position 为 mm，与字段不一致；保留原始 CSV，索引标明差异，不缩放猜测。120 Hz 仅为来源声明，CSV 无逐帧时间戳，统计不填写已验证 fps。

`LICENSE` 是空文件；实际条款见固定版本 `LICENSE.md`：学术研究、个人学习和非商业原型，保留 ChingMu 来源署名；商业分发、转售等须书面授权。Dataset 标记 `chingmu-terms (non-commercial)`，不开放商业交付。

平台 Job 当前必须绑定本体接口，故增加 `emb_unitree_g1_motion_reference` / `ee_g1_motiondecode_reference_29dof_v1` **草稿**，仅描述来源字段。没有仿真/真机绑定；不发布，不宣称可用于 G1/BHL/RPO 控制。`training_ready=false`、`evaluation_ready=false`。

## 下载、校验与失败（第二轮复核）

复用 HF 登记 → validate → Job/Run → ebd 任务容器。16 个下载线程，固定 commit、完整且不重复的文件集合，400 GiB 总量上限、20 GiB 下载空间余量。来源清单写在数据根 `.embodiedbein/source-manifest.json`；HF 缓存保留，失败重试可以复用文件与 partial。离线重索引无网络、源目录只读，不因来源盘不足 20 GiB 而拒绝。

逐文件验证大小与 HF LFS SHA256 / Git blob SHA1，同时记录本地 SHA256；CSV 校验严格表头、36 列、非空、全量有限值及四元数范数（容差 0.01）。坏数值不重下载同一份固定数据；异常逐文件保留，不能静默跳过后宣称全量通过。

Run 输出：`file-index.jsonl`（全部逐文件结果）、`download-progress.json`（实时进度）、`invalid-files.json`（完整异常列表）、`dataset-index.json`（最终动作索引与范围）。任一文件失败使最终验证无效；失败/取消不发布训练版本。总文件清单哈希与真实累计帧数仅在最终报告形成。

### 实际下载故障与修复

首试 `job_1cfc5025d91a4d309c9a` / `run_70705e3d2a0942449318` 与诊断重试 `job_1d2da7c7a337413d8680` / `run_b850ea6b18224aa6a466` 均由操作者 API 取消，保留失败日志。已在实际任务容器复现：平台配置 `hf-mirror.com` 经代理的 HEAD 返回 **308 → huggingface.co**，响应没有 HF commit/ETag/size；huggingface_hub 不跨域跟随该 HEAD，遂报 `FileMetadataError`。官方源在同一任务环境返回带固定 commit/ETag/size 的合法响应。

修复限定 MotionDecode adapter 使用 `https://huggingface.co`，保留既有代理，不改变全局镜像源，也不降低文件哈希/版本校验。新增镜像源配置下的 adapter 回归检查。

## 浏览与证据（第三轮复核）

工作台显示真实进度、用途边界、单位及许可；原始文件树在下载期间也可浏览、访问已落地 CSV，最终索引通过既有 episode 分页浏览，CSV 经 ebd 文件通道访问。文件树仍沿用既有每页前 80 项、未提供翻页控件的限制；全部动作可通过最终 episode 分页访问。当前不生成或伪造机器人动作录像。

运行证据：修复后的 Job 已真实下载 **800 / 92,567** 文件、**954,173,249 bytes**，**0 invalid**；例如 `BM_Standing_High_Jump_00001.csv` 校验为 **4,740 帧 × 36 列**，SHA256 与上游一致。此为阶段快照，不代表全量完成。最终部署重启后同一 Run 持续运行、API/ebd 健康。

验证证据：

- 后端聚焦回归 **42 passed**（MotionDecode worker/路径与哈希/非有限值/空文件/四元数、源接口隔离、固定 CPU Job、离线重索引；并覆盖已有 LeRobot/RoboTwin/LIBERO/容量）。
- 新增 Python worker/测试 ruff 检查通过；已有大型文件中存在其他 lint 项，不整文件重排。
- 前端 **90 passed**、`vue-tsc --noEmit` 通过；Vite 镜像构建通过。
- `make up` 成功；`/healthz` 的 API 与 ebd 均 ok；草稿接口 self_check 通过。
- 实测文件树 HTTP 200、CSV Range HTTP 206 / `text/csv`，数据页面 HTTP 200；原始数据可在下载期间浏览。
- 工具镜像的小 fixture 在源只读挂载与 `--network none` 下重索引通过，未写来源目录。
- 独立只读审查：完整清单约束及离线重索引空间门限已修复。

未验证：全量下载终态、全量校验报告、真实浏览器点击、坐标系/真实采样率/物理可行性、train/eval split 与目标 loader。不把上述代码与小测试当作全量验收。下载与逐文件校验作为持久 Job 继续运行；平台仅在完整报告且 Job 成功后显示 validated。
