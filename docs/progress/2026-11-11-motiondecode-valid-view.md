# MotionDecode 有效版本与数据/动作浏览（2026-11-11）

用户授权排除已确认的空文件，保留原始数据，校验其余数据并展示数值表、动作视频。本次不训练，不发布机器人控制接口。

## 数据版本与原始保留

- 原始 Dataset：`ds_0d2332ff326e46ecb405`，原始目录 `/data/kk/data/MotionDecode/6380f7420c83b20b6ad50124bc6cb829a7c0b178`，原失败记录保留。
- 有效 Dataset：`ds_90145a61d8c24298814e`，页面 `/data/ds_90145a61d8c24298814e`。
- DatasetVersion：`dsv_ffbf02d7130842e59833` / `valid-v1`。
- 派生 Job：`job_e673930b233a4b0d803a`；Run：`run_3ad95efc306f4f979a8d`，**succeeded / finished**。
- 原始文件 **92,567** 个全部重新核对 hash，匹配此前冻结逐文件校验证据；以只读挂载复用源文件，没有复制147.9GB或改写来源。
- 有效成员：**92,558 条 motion，428,058,265 帧**；含元数据共92,566文件、147,887,247,582 bytes。
- 选择清单 `included-motions.jsonl` 的 SHA256：`07b45735bbf48f44672fabc4a058f92df7a8f58b73b2c9f13e0291fcf319d382`；发布前由ebd再次核对。

唯一排除：`samples/5.Dance/5.3.Pop Dance/Summer_Wind/Summer_Wind.csv`，970 bytes，仅表头，无动作帧，SHA256 `5cea3e9e5684860ce1b45c799dac91d69b015b8e3df5c1575b2c47966b7d38c4`。排除同时要求精确路径、大小、hash与原校验错误一致；其他异常依然失败。原文件保留，可在父数据查看。

新版本 `validated`、`preview_ready=true`；`training_ready=false`、`evaluation_ready=false`、`compatible_envs=[]`。未证明目标训练loader、真实时间轴、动力学或模型质量，不以浏览能力替代训练验收。许可沿用 ChingMu 非商业条款。

## 表格与视频实现

- 按路径/名称/任务搜索92558个有效动作，服务端分页，不将全量CSV发送到浏览器。
- 预览 Job 只读挂载确切源CSV及冻结G1资产，每次读取前核对成员hash。
- 数值表保持36个原始列，每页100源帧，包含绝对源帧序号；CSV解析和分页JSON生成在任务容器执行，API由ebd提供有界字节服务。
- G1模型取已有UFO `ufo/data/robots/g1` 的21MiB资产，经ebd内容快照冻结；29关节按名字映射，根位置m、四元数wxyz，不凭位置猜关节顺序。
- 使用 `mj_forward` 做运动学姿态回放，固定来源声明120Hz→视频30fps（每4源帧取一帧），每段最多7200源帧/60秒，可继续下一段。CSV无时间戳，页面/receipt明确标注声明频率，不宣称动力学或真机能力。
- 复用已有MuJoCo/OSMesa/imageio/ffmpeg运行时，不新增依赖。Job为0GPU、2CPU、4GiB、network=none；输出MP4和分页表格有逐文件大小/hash、源帧映射及资产快照证据。
- 镜像 `eb/motiondecode-preview:latest`；执行时冻结digest。前端显示真实排队/运行/失败，过期响应、卸载与换数据集时停止轮询；缓存损坏可重建。

## 运行与复核证据

第一轮（来源/隔离）：精确排除、原始hash全量匹配、逻辑派生/父子血缘及不可变版本发布均通过真实Job。

第二轮（失败/恢复）：独立复核发现并修复派生timeout/no-Run取消后不能重试、成功但损坏预览缓存不能重建两项问题；坏缓存仅在确定内容失败时重建，暂时5xx/连接故障不重复渲染。

第三轮（展示/媒体）：多帧合成fixture在2CPU配额下完成真实G1映射、OSMesa渲染、H264编解码及表格数值检查；16视频帧共2.72秒（包含初始化）。真实数据页面验收通过，见下。

首批真实预览 `job_2a1432a1dc5f48019ee2`、`job_b0f0d97e5c6549aa968a` 发现软件渲染器启动66线程、编码器54线程，均受2CPU限额拖慢。经平台API取消后，固定`LP_NUM_THREADS=2`、编码2线程，关闭阴影/反射与离屏多采样；不改CSV姿态。保留旧Run记录。

真实浏览器还复现文件流路由在rollback后读取过期Run.id导致MissingGreenlet，已按既有文件流模式在rollback前冻结存储引用，新增真实SQLAlchemy Run过期回归（视频和表格）。修复部署后无需重新渲染，现有缓存正常服务。

最终验证：后端相关完整回归 **95 passed**；前端全套 **99 passed**、Vue类型检查通过；新增Python代码ruff通过；make up与Vite构建成功。

### 成功的真实预览与页面验收

| 内容 | Job / Run | 结果 |
|---|---|---|
| Standing_High_Jump_00001，完整4740源帧 | `job_cae4cb151b0e4c909bdc` / `run_db0d2de1abca47a4977f` | succeeded；1185视频帧，640×480、30fps、39.5秒；表格48页，末页4700–4739 |
| 最长Tour_Guide_Role_00067，源帧42116–49315 | `job_b8864684145244f7befe` / `run_c513ec93c2064e64aef8` | succeeded；1800视频帧，60秒；全部49316源帧表格494页，末页49300–49315 |

预览镜像：`sha256:b9d6b5f8ec722556872d9e65b013498bfdae59b49a2c76c6d8e0364f79ada789`。
G1资产快照：`sha256:0c3b078bb23b2080f0ac3843ba5f4d55932ed17f5ad7e87600bf52974d5924ea`。

- 两个MP4均通过任务内完整帧数/首帧形状解码检查、ebd文件hash核验和公开API HTTP206 / `video/mp4` Range访问。
- 数值表首帧36列逐列等于原始CSV解析值，短/长动作最后一页均到达真实末帧；被排除文件在有效动作查询中为0，原970byte源文件和hash仍保持不变。
- 现有Chromium容器中实际打开有效版本，搜索动作、生成/复用缓存、视频解码为640×480 / 39.5秒，播放时间推进，表格翻至第2页首帧100。测试：`backend/tests/runtime/check_motiondecode_browser.mjs`。
- 浏览器截图及JSON在 `/tmp/eb-motiondecode-browser-evidence/`（临时证据目录，测试镜像无CJK字体，截图中文字不完整；DOM文本及真实值已验证）。永久视频/表格/receipt保留在上述Run的outputs，API可访问。
- 重复创建有效视图返回相同Dataset与唯一valid-v1；原地reindex被409拒绝，空文件预览被400拒绝。

使用边界：未预渲染92558个视频，首次选择其他动作时按需生成缓存；未测试每个视频、未训练、未进行动力学/真机或目标loader验收。全部原始数据的hash与已有全帧结构证据已复核；该边界不影响已验证的查询、分页原始表格与运动学回放。
