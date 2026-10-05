# LIBERO → π0.5 SFT → PPO → 评估操作说明

## 已准备的输入

- 数据：`LIBERO v3 · pinned`，ID `ds_398a3043320d4ba69566`，来源`lerobot/libero`，固定版本`a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4`。
- 基座：`mdl_c01022c5c05e463d91e0`，来源`RLinf/RLinf-Pi05-LIBERO-SFT`，固定版本`45ccfcc4e28634f1576ebf78cab0fbe2fd82432d`，已由平台下载并检查文件大小与SHA256。
- 该基座已做上游LIBERO SFT；本平台执行继续微调，不将其当成未见过LIBERO任务的通用基座。
- 任务镜像：`eb/rlinf:libero-pi05`，已验证摘要`sha256:0b13e47916c6867d91aab776b580c2043c77b21f71147ed376047e0038c326a2`。

管理员只需在依赖或runner变化后执行`make images-rlinf`重建任务镜像，再用`make up`更新平台。任务镜像不会随`make up`隐式重建，避免每次部署重复下载大型依赖。

## 页面操作

入口：`http://1.94.0.106:31800/`。以下步骤全部由页面提交，训练容器由ebd启动。

1. **环境**（`/envs`）：确认`eb/rlinf:libero-pi05`状态为`ready`，并核对摘要前缀`0b13e47916c6`。该状态由ebd实时只读检查本机Docker镜像，不是页面静态占位。
2. **数据**（`/data`）：打开`LIBERO v3 · pinned`，确认ID为`ds_398a3043320d4ba69566`、revision为上述固定SHA、状态为已验证；抽查任务标签、双摄像头视频与episode分页。结构检查与视频/loader检查不是不可变版本发布。
3. **模型**（`/models`）：确认`RLinf-Pi05-LIBERO-SFT`（`mdl_c01022c5c05e463d91e0`）已下载完成。下载中或失败的模型不能用于训练。
4. **训练 → 新建**（`/train/new`）：选择`LIBERO π0.5 SFT`，机器人/本体选择`LIBERO Franka`，数据和基座分别选择上述条目，GPU选4。默认冻结视觉语言模型，仅训练动作专家。填写你实际实验的`max_steps`、batch size、学习率与保存间隔；不要沿用验收用的2步参数开展正式实验。
5. **预览**：核对固定输入、只读挂载、镜像、4张GPU、SFT 72GiB内存以及实际Hydra覆盖参数。预览无错误后，可以点“提交作业”只运行SFT；进入训练详情查看真实Run日志、指标和完整checkpoint。
6. **完整管线**：同一表单填写“管线后续 PPO max_epochs”，点“创建 SFT → spatial PPO → 独立评估管线”，跳转`/train/pipelines`。再次检查SFT 72GiB、PPO 76GiB、评估64GiB及步骤绑定，再点“开始新运行”。创建管线不会直接开跑，只有这个按钮会提交运行。
7. **运行与失败恢复**：管线页按Run展示每阶段状态和实际模型输入。失败时填写新的内存上限后点“从失败阶段继续”；系统创建新PipelineRun、复用连续成功前缀，不修改旧记录，也不恢复失败PPO的优化器中间状态。
8. **其他LIBERO套件**：训练模板也提供object、goal、10的PPO和独立评估。单独运行时选择已完整登记的检查点目录。
9. **模型产出**：在成功训练详情中查看完整检查点，再显式登记模型。评估报告保存真实指标，作业成功不会自动通过真机发布资格。

## 运行约束和验收边界

- 当前数据标注10Hz，仿真策略接口20Hz；页面保留G2不兼容提示。SFT loader按数据自身时间读取动作，不偷偷插值或把它声明成真机动作兼容。
- `lerobot/libero`是LeRobot文档推荐的视频版数据，元数据为1693个episode、273465帧、40个任务、状态8维、动作7维；平台固定revision并使用PyAV读取，不依赖运行时联网。
- LIBERO动作模式必须与模型一致；当前配方使用7维相对末端动作和10步action chunk。更换绝对动作模型时必须创建新的本体环境契约和归一化资产，不能只改页面名称。
- 现有公开基座归一化资产被原样继承；不宣称已经形成独立、无泄漏的训练/验证数据切分。
- 四张4090的显存各自独立。调度根据实时剩余显存与解析估算准入；估算不是实测峰值，不会终止其他进程释放GPU。
- 任务全部经ebd启动容器。平台代码不写外部RLinf/OpenPI仓库，不执行Git。
- 正式评估应覆盖Spatial、Object、Goal、Long四套任务，每任务10个episode，并用3个seed平均；平台的小样本运行检查不能作为论文级结果或发布门禁。
- 用`make up`部署平台；任务镜像用`make images-rlinf`另行构建。实际验收和剩余差距见`2026-09-09-platform-implementation.md`。

## 失败续跑与资源调整

管线页的“从失败阶段继续”新建执行快照，复用连续成功的训练阶段及其完整检查点。可选内存预算只修改新运行的首个待执行阶段，例如`76 GiB`；旧运行与失败日志保留。留空时沿用原配置。它不是优化器状态恢复：失败的PPO阶段从已成功的上游模型重新开始。

当前四卡PPO使用四个actor rank、一份rollout，CPU权重串行加载；训练容器默认预算76GiB。调度会检查主机实际可用内存，保留4GiB并预留运行中作业预算；不足时等待，不终止外部任务。长训练默认每40轮及最终轮保存checkpoint，避免每轮全量汇聚带来的内存和I/O成本。

评估成功率采用“已完成轨迹至少成功一次”；缺少完成轨迹数时不计算置信区间。没有实测跨域统计时显示尚未评估，不自动通过 sim2sim/sim2real 门禁。

参考：[LeRobot LIBERO数据、控制模式和评估协议](https://huggingface.co/docs/lerobot/main/en/libero)、[RLinf公开π0.5 LIBERO SFT模型](https://huggingface.co/RLinf/RLinf-Pi05-LIBERO-SFT)。

## 平台运行检查证据

已完成的最小端到端检查为`pr_daf7506388ed4e57b4ce`：SFT Run `run_be000fb3359744a5a112`、PPO Run `run_25bdb3d2dde942cb9409`、独立评估Run `run_9cc87be2f4e749cab872`、评估报告`ev_194156f0ce904c0ca492`。报告只含Spatial的4条轨迹并保持`not_qualified`。这些记录用于证明平台管线、模型交接和结果采集可运行；后续正式训练由使用者在页面自行创建和启动。
