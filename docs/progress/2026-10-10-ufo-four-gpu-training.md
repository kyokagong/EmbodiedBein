# UFO/RLinf GPU资格与四卡训练台账

计划：`docs/plans/2026-10-10-ufo-rlinf-bfm-integration.md`。日期2026-10-10，Asia/Shanghai。

用户在CPU实现交付后明确要求继续并完成GPU四卡训练，撤销此前仅容器验证的GPU限制。仍不执行Git提交/暂存/推送，不训练教师，不修改历史数据/模型/Suite/外部源码。

本轮有界目标：B单卡16→24资格、四卡16更新资格、通过后首个四卡1000新增联合更新区间；1个配置，总训练/评估/排队/录像墙钟上限6h，单Job不超过既有冻结上限，磁盘开发预算≤20GB（不计原有受管资产），每次validation16例全量录像、初始与24及四卡资格/1000阶段分别保留证据；没有授权无限追加或多个长训配置。任何必需证据缺失/非有限/恢复错误停止等待证据，infra failure仅一次有因修复重跑；task failure不换seed隐藏。

Ruling: 本次“完成四卡训练”按计划首个有界1000更新区间解释，不自动扩展到20,000更新/24h建议预算；若错则少训练而非超预算。先过单卡与拓扑资格，不直接用旧BFM四卡模型/证据为native UFO背书。

四卡采样上限沿现有runner为每Run4096轮×768transition=3,145,728；16资格、恢复24与新增1000区间合计不超过10,000,000 transition（通常目标计数会更早结束）。qualification≤24、每Run新增≤1000、3300秒runner安全边界、1小时Job边界同时生效。总墙钟含本次排障，最迟12:15 UTC结束本轮，不用重新计时规避6h。

在获得四卡24/1024快评前冻结best规则：四卡24完整validation为首个best；后续只在同Suite、已通过任务9的并行规模完整validation成功数增加，或成功数相同且median存活至少增加10步时替换，否则保留best，latest单独记录。最初候选4GPU×4env未通过等价检查，按任务9降至4GPU×1env（详见条目18），在1000区间开始前冻结；不改变成功/存活改善门槛。单卡等价检查不用于挑较好结果；acceptance仅评一次冻结best，不参与选择或调参。

四张4090（各24564MiB）初始空闲24080MiB；主存106422/128644MiB可用，无活动平台Job。物理UUID：

| index | UUID |
|---|---|
| 0 | GPU-1b9c7093-3204-5084-3854-7ed540d2b1c5 |
| 1 | GPU-16478714-67b9-4a90-0c34-5d0e2a063322 |
| 2 | GPU-4e370b48-f7d4-89fa-546c-93aa18971480 |
| 3 | GPU-6758c666-e8ed-e972-401a-1af38348ff40 |

执行入口均API→Job/Run→ebd。`make up`已部署原生入口；API启动期间历史ingest读取约74GB，等待启动完成后健康检查通过，未绕过扫描/准入。

## 实际运行

1. GPU原生差分：Job `job_a3d022789b8249b2b333`，Run `run_acabbff468324fd48426`，镜像 `sha256:c3fc0ea183cb299584c703597584abb381634e737e9734b14f2c1a2b4d7a7af8`，固定15分钟/1GPU。Job succeeded、Run finished，collector核验通过。1024宽/3层/batch128，六Adam各step1、四target/norm/pool/梯度及参数数值差分通过FP32 atol1e-6/rtol1e-5，z_pool128、参数delta L1=16458.890768822966。仅1次原生GPU更新资格probe，不能视为已完成物理学习资格。
2. 正式raw派生父版本 `ds_27fbaa34696e4f26b2c7`。API先暴露CPU转换漏入容量白名单，被误估3.3B模型63.5GB拒绝；最小测试 `test_ufo_raw_state_conversion_is_cpu_data_work` RED→GREEN，仅增加批准operation分类，不降低GPU训练准入。继续后发现DataConvertAdapter新分支缺JobKind导入；`test_ufo_raw_conversion_renders_registered_parent_readonly` RED→GREEN，旧转换回归。两次提交事务失败未产生训练Run，孤立空输出目录未删除。

3. Raw派生Job `job_256428195721453c83e0` / Run `run_b498435e0d4f477196ff` succeeded/finished；正式数据 `ds_92718168bfec4da6a01d` validated，manifest `7e47773bf53588ee200cccc5bbde3999df172a11593d7359cbeca2a00ac4c5d8`。官方/G1的8步窗口5206/6716，250步窗口1279/931，保留父split/valid、raw/FK重编码通过；training_ready仍false，等待真实学习资格。
4. Isaac环境probe Job `job_2546bf7dc9ab4700b82d` / Run `run_411c175692ca4243bae9` succeeded/finished；2env、参考部分reset/重复观测/timeout terminal/8raw aux实际通过，collector核验。GPU/source与RSI依赖齐全，静态recipe更新dependency_ready，尚非smoke_passed。

Ruling: 基线目标0仅允许从头qualification，普通training和所有resume仍须正向更新；原生learner不调用optimizer，现有有界runner在首轮warmup后保存step0并正常结束，然后独立评估初始模型。无需新增任意脚本/模型注册后门。API/domain及真实native CPU检查RED→GREEN（88/26 passed）。任务镜像34046999…在c3fc0e…基础上只改变零目标边界，source/agent/math/RSI模块不变、CPU差分重新通过；已通过GPU差分的源码身份仍相同，新镜像实际运行身份另记。

当前尚未完成：单卡baseline/16→24/独立快评/取消释放、native四卡实现/资格/1000区间。后续继续追加实际ID、输入hash、计数/媒体及决定，不提前提升smoke状态。共享data_convert/estimator全Ruff发现51项已有样式问题（未重排整个文件），运行错误F项检查通过，保留原日志。

5. 初始化基线TrainJob `trn_9d5c725048a9462bb559` / Job `job_ea8717f6fd30482f8a0e` / Run `run_9bf6863d73764bc2b560`，实际GPU0、镜像34046999…。真实RPC送CPU action后native Physics.step增益仍从CPU输入创建，与已移至GPU的action冲突，环境子进程失败。通过API取消：Job cancelled / Run aborted(cancelled)，四卡显存回落各1MiB、Doctor run_id均null；无完整checkpoint，不发布模型。用meta设备在CPU单元容器执行同一Physics.step复现RED，改为在已转换action设备创建gain；修正测试夹具的eb/裸模块重复类导入后42项native回归、88项API门禁回归与新增模块Ruff通过。最终新镜像 `sha256:f58d4d9a4d386e8c5ee5326a2f831d79e11e7f1f914d94fdedd0502606116153`，CPU原生差分重新通过。此为本次infra失败的一次有因修复重跑，旧Run保留。

6. 基线一次重跑TrainJob `trn_11d6596436254d2eb0aa` / Job `job_a0a3bb14ec574a9b8c6b` / Run `run_035c3a0c72174bc3b5ae` succeeded/finished，f58d4d…镜像、GPU0。实际1 sampling round / 64 transition / 4 control steps / 0 learner_joint_updates，六optimizer保持0，完整step0原子checkpoint与模型包发布，ModelVersion `mv_f6638c27bb594e6d8166`，446416116字节；无训练更新，不称短训成功。GPU释放。RSI Profile hash=`sha256:0d96a4a6210cdf665de469ea0188b1131abd75e9788db8381e3aa8ca3c9e0a1b`。

7. 基线独立validation Job `job_a4936e4ef7c34fe19be2` / Run `run_2469b9c7933a4e41ab5c` succeeded/finished，Report `ev_7eb664a31e9c45f3b2af`；16完整case/16全部录像逐段解码hash，0/16成功，7fall/9hard-limit，nan0，metric_steps188，velocity RMSE=.823493683，yaw RMSE=4.447926885。固定Suite `suite_rpo_v2_bfm_isaac_velocity_validation_v2`，plan=`sha256:d265296c77b314e9fc488505454c819a05952c4f64defc9346bbf64084942ec9`。没有成功录像，因为确实无成功案例。
8. 首次单卡16 TrainJob `trn_e02c37f425344cce9680` / Job `job_d5fa207efcf84bb8a99e` / Run `run_9d7b61b5a3634c42a2aa` succeeded/finished，完整step16模型 `mv_ed1740e866524f4e9d35`，984522901字节，f58d4d镜像。恢复preview通过；第一次resume因缺接口强制Idempotency-Key返回422未创建Run，补稳定幂等键后TrainJob `trn_8b7b5dfe0cd144fba62f` / Job `job_759c25886bd74fbbaf3a` / Run `run_fd841fe6c3254ba5bcd5`在加载checkpoint后、传回sampler状态时exit255。真实GPU全部释放；源完整16模型未改，未生成24模型。

根因：weights_only安全checkpoint把NumPy MT19937 uint32状态转为Torch uint32，Ray的legacy pickle不能回读该storage；CPU最小pickle与真实save/load/三rank回传均复现相同UntypedStorage.dtype异常。只在load出口把sampler rng_numpy转回NumPy，不绕过hash/optimizer/RNG检查；真实pickle回传及下一个NumPy序列RED→GREEN，49项原生回归通过。新单卡镜像 `sha256:6f6601aed8cf15499fe74e60da98096a2a44dda59c43d8d252de2639f4025d01`，CPU原生差分再次通过。

Ruling: 修复改变执行镜像，不把旧16 checkpoint跨镜像伪装完整续训；重新在新镜像资格16→24，同算法/source/raw/seed/预算，成本是一次额外16资格。初始化模型和基线保留，算法数学/初始化/命令物理模块未改变；必须记录各镜像实际生产身份。资格的不同失败路径各允许一次有因修复重跑；Task10单个训练区间infra失败最多一次，禁止同错误无限重跑。

Ruling: 当前四卡选B/Isaac，C/MJLab尚未实施/测试，不宣称已blocked或sim2sim通过；设计20允许保留B，用户本轮目标是四卡训练，额外启动C会增加第二分支与未冻结依赖成本。C仍为计划未完成项。

四卡代码：新recipe=`rpo_bfm_ufo_agent_rlinf_isaac_4gpu_v1`，one_learner_three_samplers，192env=3×64，batch128/L4/100000CPU replay/10240warmup/1024阈值/16更新块不变；GPU0 learner、1–3 sampler，CPU24/RAM64GiB/shm32GiB，每Job1h，不按四卡汇总VRAM。三rank实际actor/B/norm/pool字节hash与version同步，每rank独立seed与episode命名空间、完整RNG/专家state保存/恢复；实现/CPU检查通过尚非GPU资格。

反思1发现旧共享地形按batch首seed变化，不能拿单卡/四卡平均值证明等价；派生UFO validation/acceptance新Suite，固定官方网格seed64000与每case唯一slot，case startup/reset/command/noise/interval保持独立，旧Suite/模型不修改；实际单卡与4×4env等价尚待运行，容差预冻结为每case速度/yaw RMSE≤.005、存活差≤1step且成功/失败判据一致。

反思2发现父split中官方30clip全部train，validation12/test6均G1。新增固定held-out tracking诊断选validation有250帧的首尾clip，只调用原生tracking_inference/B与mean actor，写入raw参考初态后实际物理步进和录像，不用train回放冒充泛化。共享场景seed63000、两个独立参考case，无官方held-out诊断可报告，不宣称覆盖官方来源泛化。媒体/trace严格hash、逐段decode、源权重/数据/asset/plan绑定；故障/媒体缺失不得提升B。

9. 最终单卡资格镜像 `sha256:5abec37ade5f55fd043f9aceff9fc8b32ed7ff422b08c80879857bbd745af015`（6f6601仅构建、未GPU执行；5ab追加逐joint-update TensorBoard）。TrainJob `trn_cf11fe1489df421cad97` / Job `job_27a2093cec934d50b631` / Run `run_a332de80b40d486bb6f1` succeeded/finished，真实16更新；同镜像恢复TrainJob `trn_42ff0ff051b9412abf2e` / Job `job_f316b7c22dbe48f0b00c` / Run `run_c564fc226e764c7f8f49` succeeded/finished，完整累计24模型 `mv_ed390ad35b9641ab9065`，991732327字节。API实际FB原始曲线8点正好17–24，非16更新块的最后loss。单卡模型保存了actor Python与各sampler实际使用的NumPy/CPU/CUDA状态；固定上游sampler不使用Python随机数，四卡下仍补齐独立Python状态（inner rollout/v2），单卡5ab不原地更换，跨镜像完整恢复继续拒绝。

10. 24模型独立command validation Job `job_8e0331bb0d0640609c39` / Run `run_cce868f58bc647958939` succeeded/finished，Report `ev_618382c74bd2497b8ac4`；同旧validation Suite，实际16完整case/16录像解码hash、全部Range媒体接口可访问。0/16成功，14fall/2hard-limit、nan0，metric_steps156，velocity RMSE=.871994469、yaw=2.629687017。与step0相比平移误差和存活未改善；仅24次更新，仍处于预先冻结的2000更新热身观察范围，不能凭loss宣称行走质量。

11. 基线held-out诊断v1 Job `job_fafffd3049e043898cf3` / Run `run_b7f12740329e4ea0bb30`，GPU1、60d81b镜像。08:14:11→08:19:50进程exit0但没有receipt，collector严判invalid_output，未发布假通过。CPU复核确认first dance subject1无250帧有效窗口，last walk subject5有16；App.close掩盖异常。派生固定诊断v2：metadata必须valid_frames==frames再选validation首尾，原始split/数据不改；选择dance subject2与walk subject4，均400全有效帧，各151个250帧窗口。追加明确阶段日志/traceback，保留失败记录，不通过选择train片段补造泛化。

反思3追加四卡所有sampler的Python状态、env/rank联合episode命名空间和实际同步hash持久凭证，collector先核验再发布模型；控制面只用无Torch的checkpoint元数据helper。单卡/四卡真实Hydra验证暴露nested主配置searchpath不能继承，改为同一主配置+受保护的four-recipe默认placement/192env/3个禁用eval占位，避免复制整套配置。新旧API61passed/3skip、原生42passed（controller async在API容器检查）、前端15passed+类型检查、原生Ruff通过。新四卡镜像 `sha256:60fbb44a17c3247f7ed83104466e4a652dd06d8712b22ffb1ea202ea6d03cf2b`，仍无真实四卡训练证据，待B诊断闭环后继续。

12. v2基线诊断 Job `job_b04cdba9e693458fa509` / Run `run_83536222320e456588bb` 实际打印app/数据/完整checkpoint加载成功后，因直接赋值原生frozen config.device失败；exit0仍被collector严判invalid_output，未发布媒体或质量通过。读取原生BaseModel.to确认它已设置model.device，仅删除重复cfg/device赋值，训练60f镜像不改；派生独立诊断镜像 `sha256:502b71dea5bd4e849640ed26bdbe9ede6e45a0431e4c45be36a2a37fc083f677`。固定输入再运行基线 Job `job_2a4a09172f4d45ab9f9e` / Run `run_dbcbb5f241274cb2927d` 与24模型 Job `job_2d3c0b3d6f2948798f55` / Run `run_b4d148df898f4d0da6ce`，均进行中，仍使用v2同两clip及共享scene seed，不更换任务失败case。

最终独立只读审查实际复现三项页面遗漏：四卡默认env数误发为用户override、EvalNew遗漏rpo_bfm benchmark、EvalReport遗漏实际BFM receipt。真实Vue setup四项RED→GREEN，17行为检查+类型检查通过，另API门禁27passed/2skip（原生部分已有真实镜像检查）、tracking原生3passed。UFO页面只发目标/用途/seed，验证/验收默认对应角色并强制录像；报告显示实际分母和物理失败指标，不误报跨域证据。

单卡完整真实曲线：[1–24更新](artifacts/2026-10-10-ufo-four-gpu/single16-24.png)、[统计](artifacts/2026-10-10-ufo-four-gpu/single16-24-statistics.json)、[1–16原始数值](artifacts/2026-10-10-ufo-four-gpu/single16-metrics.json)、[17–24原始数值](artifacts/2026-10-10-ufo-four-gpu/single24-metrics.json)。虚线标恢复点；不足50点不补造50点平滑。完整[24模型case](artifacts/2026-10-10-ufo-four-gpu/single24-cases.json)和[媒体HTTP/Range核验](artifacts/2026-10-10-ufo-four-gpu/single24-media.json)另存；训练loss不作为独立任务成功率。

13. 502b镜像基线诊断实际完成参考初态写入/断言、物理场景与初始录像，首步前触发inference tensor在inference_mode之外原地修改的RuntimeError，Job failed / Run aborted(invalid_output)；不是模型物理任务失败。24模型同入口诊断通过API取消，未冒称完成。原生CPU mean actor复现同异常，动作改为普通functional masked_fill后可写且活动case动作完全一致，4项tracking检查与Ruff通过。新诊断镜像 `sha256:64c9803986b1a5c836d58fee5017d6a67e8695e022849ed46bb31ace167a4acd`，4卡训练仍固定60f。

实际Kit日志显示前6分钟主要为RtPso shader编译，旧4CPU作业占用约400%，RSS最高观察10.37GiB/16GiB；两条新诊断只提高CPU预算至8，物理条件/数据/模型/场景seed/参考case不变，RAM16GiB/单GPU/15分钟/不自动重试。Job `job_0e75e9a62b5a4b62a036`（基线）和 `job_6bcd826717df409a931a`（24模型）已正式提交，结果尚待核验。不同首次暴露的入口错误均先保留日志、做最小有因修复，再同输入重跑一次；不得把重复Run当额外训练更新。

14. 两个64c诊断均succeeded/finished：基线 Run `run_e4344963240b40b7aab6`，两clip各13步hard_limit，q RMSE=.354963376/.358291079rad；24模型 Run `run_09afed547f884c4ca9be`，10/11步hard_limit，RMSE=.591917911/.639610338rad。每例初态断言、物理trace、录像解码/hash通过，4个video Artifact及CAS Range访问实际成功；全部为失败录像，无成功录像。基线/24使用相同validation参考，未训练held-out或更新norm，不证明跟踪质量改善。完整[基线](artifacts/2026-10-10-ufo-four-gpu/tracking-baseline-job.json)、[24模型](artifacts/2026-10-10-ufo-four-gpu/tracking-step24-job.json)、[可访问媒体](artifacts/2026-10-10-ufo-four-gpu/tracking-media.json)。两个作业正常结束且四卡租约释放。单卡recipe提升smoke_passed，只表示闭环可执行，不表示acceptance通过。

15. 四卡TrainJob `trn_c9d5f444cefd40279916` / Job `job_bda0912026e24d25ad0b` / Run `run_9d4e95269c63447cba7c` 已正式开始，固定60f镜像、Profile `rpo_bfm_ufo_isaac_4gpu_rsi_v1` hash=`sha256:c3a549ee6ee14088a1b000d4674923827b02b05a52f5e876050514b31e4f6c61`、raw ds927/seed42/目标16。Doctor实际四张卡都租给同一Run，GPU0 learner、1–3 sampler，每rank64env，共192；首轮实际768 transition/4控制步，0联合更新属warmup，三rank实际model/B/norm/pool同步已进入采样，尚未宣称资格通过。资源只读采样另存，未启动1000区间。

16. 四卡16资格succeeded/finished，ModelVersion `mv_68d203fe4fad47869afd`，1,025,843,105字节；15轮/60控制步/11,520 transition，六Adam全部16，next_trigger12288。同步凭证为最后采样前version0/10,752transition，不把它误称final16；完整checkpoint另存actor16。通过严格同60f指纹恢复 TrainJob `trn_446e8ffb9c2246b08488` / Job `job_bc04cfcaf47c4ef9ba96` / Run `run_df1eb07702d24e1a8b64` succeeded/finished，新增8更新，ModelVersion `mv_6a87104807ab4cd5b9a6`，1,031,250,738字节；累计16轮/64控制步/12,288 transition，六Adam全部24，next_trigger13312。恢复后实际3rank hash均bbf4ce61…/version16/pool及norm完整同步，比初始4c32e17d…发生变化；完整checkpoint24与采样前16分开标记。环境未持久化，恢复mode为trainer_state而非exact。两次正常释放四卡。

5秒左右只读采样的组合显存最大值：16阶段 learner2365MiB，sampler4432/4479/4432MiB；24恢复 learner2403MiB，sampler4424/4471/4424MiB。容器主存观察最大27.95/26.98GiB（含cgroup内存，不等同独立进程RSS峰值），低于64GiB预算，显存/主存均保留超过15%余量。只有每sampler64env为实测档，不称128–1024上限已验证；保存I/O及原始[16资源](artifacts/2026-10-10-ufo-four-gpu/four16-resources.log)/[24资源](artifacts/2026-10-10-ufo-four-gpu/four24-resources.log)保留。

17. 固定四卡24模型进行16例等价评估：4×4 Job `job_7309fd4449dd4ee38ff4` / Run `run_bde01506fca54353acb8` 因rank2机器人USD缺defaultPrim而exit1，未生成完整报告；其余3rank实际12例/12录像保留，不补造4例。单卡1×1 Job `job_5f23f4b16b574d549115` / Run `run_15c1cb0c71e34713aea9` 继续，不将部分结果当资格。明确根因是Isaac AssetConverterBase默认目录由秒级时间和Python random.randrange生成，固定terrain seed64000令所有进程后缀都为5650；同时两进程写 `/tmp/IsaacLab/usd_20261010_093640_5650/rpo.usd`，真实残留USD仅1.5KiB且无defaultPrim。进程限额未证明为根因，不因early fork warning提高资源限额。

最小修正是每worker指定Job输出内独立robot-usd目录，固定source/权重/资产/物理/RNG/case不变，避免共享临时路径。派生评估镜像 `sha256:c2b36fe55f84880c28382e37801e4689abbfaed04ee507b0d4cbf78ae2d22563`，保留训练/完整恢复60f镜像；API27passed/2skip、原生分片/RNG/录像11passed（3个async在API容器覆盖）。重新冻结评估Plan后仅一次有因修复重跑4×4；最终对照报告明确各自评估镜像、同weights/config/motion/asset/ONNX与逐case初态和数值，不虚称两个执行镜像相同。

18. 缓存修复后4×4完成16例，但seed61009 yaw RMSE差0.009876094超过预冻结0.005（速度差0.000233613、步数/终止原因一致）；所有物理初态相同，trace前两步动作/z/状态一致，第三步首先在物理步进出现分歧。保留完整报告，判该规模未通过数值等价，不降低容差、不改模型/物理参数。按计划使用较小规模4×1：Job `job_d3b979e8117b40b6bf9e` / Run `run_342622ae715541939a48` succeeded/finished，Report `ev_d9c5cd3ab4164a828eda`。全部16例速度/yaw/存活/结果与单卡逐点完全一致，policy.onnx、command-to-latent.onnx、prompts.npz的SHA亦完全一致；不同评估镜像明确分列，不假称可执行文件相同。16录像实际解码/hash及Range访问全部通过，0成功/15fall/1hard-limit/nan0、median存活10步、velocity=.710715908、yaw=2.103974780。原始[不合格4env对照](artifacts/2026-10-10-ufo-four-gpu/equivalence-four-env-failed.json)、[通过1env对照](artifacts/2026-10-10-ufo-four-gpu/equivalence-oneenv.json)。

长训与最终评估现固定4GPU×1env评估规模，训练仍1 learner+3 sampler×64env。API预览/提交及RLinf渲染均拒绝UFO每卡4env，native eval缺stack不得回退默认Isaac；行为检查RED→GREEN，57passed/2skip（Torch已由原生容器覆盖），相关新模块Ruff通过。四卡recipe提升smoke_passed只表示流程资格，不表示正常行走或80/100验收。24模型作为初始best，latest独立记录；仅在固定改善规则满足时替换。

19. 首个1000新增联合更新区间正式启动：TrainJob `trn_85c95a1795dc477ca3b1` / Job `job_350923f15dee4449b44a` / Run `run_c8518b07e7914044bb68`。从完整四卡24模型以trainer_state恢复，目标1024（新增1000而非总1000/六倍optimizer计数），固定60f训练镜像、同Profile/raw/seed42/LR/batch/UTD，不跨配方恢复。Doctor四卡均租给该Run，4×1后续快评规模已冻结。训练尚未完成，不提前宣称行走验收。

资格吞吐的实测范围：single16为176轮/11264 transition/native wall383.885s，four16为15轮/11520/native wall76.390s，约29.34→150.80 transition/s、5.14倍采样吞吐；环境数量/拓扑不同且此为含warmup的资格区间，不能当DDP线性加速或稳态updates/s。四卡16 draw/new=.17778、留存1、最大policy年龄16；恢复24 draw/new=.25、留存1、最大年龄24。原始字段wall_time_s为每Run learner初始化后的时间，恢复会重置，不当成全程连续墙钟。

20. 首个1000新增更新区间 succeeded/finished，完整累计1024模型 `mv_3f0cf0900e254d68ba81`，1,485,286,344字节。完整checkpoint六Adam均1024，累计100 sampling rounds/76,800 transitions/next_trigger77,824；本区间新增84轮/64,512 transitions/每环境名义336控制步。环境在恢复时重建，不能把名义步数描述为同一个物理episode连续恢复。三sampler最后采样前同步为version1016/round99/76,032 transitions、共同hash `025171a1091e6c715562e534d8514cac8f709afd959f75f2419d01ecdffe2410`；最终actor1024由完整checkpoint单独核验。

保存[完整checkpoint元数据](artifacts/2026-10-10-ufo-four-gpu/four1024-checkpoint.json)、[1000点原始指标](artifacts/2026-10-10-ufo-four-gpu/four1024-metrics.json)、[区间曲线](artifacts/2026-10-10-ufo-four-gpu/four24-1024.png)、[统计](artifacts/2026-10-10-ufo-four-gpu/four24-1024-statistics.json)与[全程实际UTC事件曲线](artifacts/2026-10-10-ufo-four-gpu/four-full-wall.png)。UTC曲线取TensorBoard事件wall_time，分别标记Run与恢复间隔，不将每Run重置的wall_time_s拼成连续时间。训练保存完成后Doctor确认四卡释放。

21. latest1024独立validation Job `job_c368e63485ea4d259084` / Run `run_b22819ef25624bbca629` succeeded/finished，Report `ev_cc413786b6d142688919`。固定c2b评估镜像/4×1规模/16个validation seed，0/16成功，median存活10步，velocity RMSE=.771277027、yaw=4.625967876、nan0、metric_steps203；2例fall、15例hard-limit，其中1例同时触发，两类计数不可简单相加成独立case总数。16录像逐段解码/hash与HTTP Range均通过，无成功录像。

**验收前冻结决策（10:38 UTC）：** best=`mv_6a87104807ab4cd5b9a6`（四卡step24，validation `ev_d9c5cd3ab4164a828eda`）；latest=`mv_3f0cf0900e254d68ba81`（step1024，validation `ev_cc413786b6d142688919`）。两者成功数均0、中位存活均10，未满足同成功数下至少改善10步的规则，不替换best。首个1000新增更新预算到此结束，不自动追加区间；loss不能证明正常行走。接下来仅对冻结best运行一次固定100例acceptance，结果不反向用于选择模型。

22. 唯一最终acceptance Job `job_143ce491763341f4852e` / Run `run_595e2a732c854948be71` 正在执行，固定best step24、Suite `suite_rpo_ufo_isaac_velocity_acceptance_v1`、seed62000–62099、4×1、plan=`sha256:f2ff9057460309be9b705e8cb78e3b5570fd21203e154267af8549de210a8e25`。此前一次缺preview_hash的400请求未创建Job/Run，补正确plan.hash后才提交本Job，不算第二次验收。另排队latest1024同held-out参考诊断 Job `job_23d88152e30f491eb2a8`，不训练、不更新norm、不参与best选择。

区间时序分析：1000个joint-update点覆盖25–1024，所有98类API数值有限。首/末50个实际更新均值，FB loss 91,156.59→21,794.85，D loss 1.731→.287，actor loss1.710→−2.307，critic loss .908→3.423，aux critic1.683→1.737；F/B/D/actor/critic/aux六组梯度与参数变化逐次记录，LR固定（B/D=1e−5，其余3e−4），未追加原生关闭的梯度裁剪。actor梯度均值6.549→9.983、critic6.936→12.655，损失下降并未带来独立行走成功。

采样诊断64个实际记录点：饱和率最大45.590%、末44.661%，低于90%停止调整阈值；末replay draw/new=1.70667、留存1、大小76,800、采样到的policy年龄8–1024 updates，旧数据保留且policy age增长。累计真实expert抽样官方64,784/G1 66,288；此为训练split抽样，不是held-out回放。训练环境episode_len首/末50个记录均值19.02→21.86步，command validation median10→10，不互相替代。训练记录了policy动作绝对值/饱和率，但未独立记录requested/applied两路完整动作时序；此项保留为证据缺口，不用均值冒充逐步物理动作对照。

1000区间135个约5秒间隔只读资源样本：GPU最大观察2405/4424/4471/4424MiB，容器内存最大29.85/64GiB，末非零I/O2.93GB读/3.32GB写；不是精确峰值或独立RSS。native训练区间wall_time_s末401.115s，含采样/同步/更新、不含容器启动和最终保存收集；1000/401.115≈2.49 updates/s仅此口径，UTC全程另存。组合容量仍有15%以上余量，只实测每sampler64env，不追加更大规模探测。

三轮区间复核：①确认RPO23DOF、v1 codec/动作增益、train与validation/test隔离，官方来源无held-out覆盖，G1未加载教师权重；②同60f完整trainer_state恢复，六Adam与1024计数一致，环境无法exact恢复，训练正常退出后独立评估，预算无自动延长；③raw+50点曲线、真实UTC恢复间隔、16完整case/视频解码/hash/HTTP访问和ONNX导出数值验核齐全，以上requested/applied训练时序缺口明确保留。**decision=budget_end**：剩余新增训练更新预算0，后续只完成验收和本轮held-out诊断；不是continue，不宣称质量pass。

23. 唯一100例acceptance物理运行完成，ebd凭证为exit0/oom=false/c2b镜像。但控制面将Run标记aborted(invalid_output)，Job failed：原有manifest固定1MiB上限不能容纳100份完整物理设置，真实manifest约2.6MiB（16例约429KiB）。没有重跑100例或修改输出。新增容量回归实际RED→GREEN：仅aggregate manifest按已核验冻结case数分配64KiB/case、最低1MiB；100例最多6.25MiB，其他凭证/视频/trace边界与symlink/hash/seed/物理判据不变，超过预算仍拒绝。相关13passed/3skip，运行错误Ruff通过，`make up`部署后实际容器源码检查确认修正。

开发维护在API容器调用既有`claim_outputs`重验同一个Run，先核对ebd真实exit0/无OOM/固定c2b镜像，再登记Report `ev_c2af011ac8044093a22b`与血缘；断言前后Job failed、Run aborted、abort_reason及原错误均保持不变，没有SQL改状态、重放物理或新增放行接口。补收集完整validity=complete、qualification=fail：**0/100成功**，91例fall/9例hard-limit、nan0、metric_steps997、velocity RMSE=.630573059、yaw=2.217759413。此报告明确对应冻结best step24，不能称latest1024的acceptance。实际[补收集凭证](artifacts/2026-10-10-ufo-four-gpu/four-best-acceptance-recollection.json)与API报告/全部媒体索引另存；进程成功、收集故障和质量失败分开报告。

24. latest1024 held-out Job `job_23d88152e30f491eb2a8` / Run `run_5bd8826d40ee4fc7b2eb` succeeded/finished。同64c诊断镜像、同两个validation clip、同共享场景seed63000，实际3/6步hard-limit，q RMSE=.236516114/.249786367rad；更短存活窗口的RMSE不能当成跟踪改善。两例初态物理写入/断言、逐步trace、录像解码/hash均通过，两个CAS Artifact `art_69b573b778b84014ba70` / `art_717638d7cee34ecfbdb8` Range访问均206；无成功录像。[原始Run与血缘](artifacts/2026-10-10-ufo-four-gpu/tracking-step1024-run.json)、[逐步曲线](artifacts/2026-10-10-ufo-four-gpu/tracking-step1024.png)、[媒体](artifacts/2026-10-10-ufo-four-gpu/tracking-step1024-media.json)。training环境存活、held-out tracking与command validation/acceptance分列，未用loss替代任务结果。

最终验收[正式报告](artifacts/2026-10-10-ufo-four-gpu/four-best-acceptance-report.json)、[100例](artifacts/2026-10-10-ufo-four-gpu/four-best-acceptance-cases.json)、[100录像访问](artifacts/2026-10-10-ufo-four-gpu/four-best-acceptance-media.json)、[全部初态/trace/video/export实际SHA复核](artifacts/2026-10-10-ufo-four-gpu/four-best-acceptance-artifact-checks.json)、[图表](artifacts/2026-10-10-ufo-four-gpu/four-best-acceptance.png)齐全。Wilson95%区间0–3.7%，median存活10步；无成功录像。最终policy.onnx/command-to-latent.onnx/prompts.npz与选best的validation推理包SHA完全相同。资源采样最大观察GPU8559/8463/8430/8367MiB、主存56.36/96GiB，仍有15%以上余量；[实际进程时间](artifacts/2026-10-10-ufo-four-gpu/final-run-timings.json)训练695.286s（含启动/保存，约1.44新增update/s）、验收498.984s（含启动/录像，约.200case/s）、held-out297.006s，全部exit0/无OOM，不把native401.115s口径与容器全程混用。

页面入口：[1024模型validation](http://127.0.0.1:31800/eval/reports/ev_cc413786b6d142688919)、[冻结best24最终acceptance](http://127.0.0.1:31800/eval/reports/ev_c2af011ac8044093a22b)、[1024训练](http://127.0.0.1:31800/train/trn_85c95a1795dc477ca3b1)。需既有登录；真实浏览器session cookie媒体206验证已保留，没有为video取消鉴权。

25. 最后一轮磁盘复核发现预算统计遗漏：此前只计受管产物，未计结束容器可写层。按inode去重，已知本轮Run/output/Model/raw产物13.732GiB；23个已知容器可写层另27,856,732,170字节，加latest诊断约236MB，已知范围约39.9GiB，**超过本轮20GB预算**；还未包含共享镜像/build/cache，不能宣称整机峰值合规。此前8.086GiB只是较早且较窄范围，不作为最终预算结论。此次遗漏如实记录，训练预算已结束，不再追加运行。

清理前逐个断言24个本轮容器已停止，存档退出码/时间/镜像/持久挂载；只`docker rm`这些停止容器，不用force，不删镜像/卷/宿主数据/模型/输出，也不触碰其他作业。回收临时层后受管产物保留约13.732GiB；历史超限事实保持在[最终磁盘审计](artifacts/2026-10-10-ufo-four-gpu/disk-audit-final.json)。删除后全部100录像再次通过API Range访问，原始证据、模型与Job/Run失败历史保持可读。Doctor实际四张卡均run_id=null、各空闲24080MiB；本轮在11:01 UTC附近收尾，早于12:15截止，剩余墙钟不转为额外训练授权。

## 本轮交付边界

| 计划范围 | 实际状态 |
|---|---|
| Task1–6：原生agent、raw/RSI、RLinf worker/通信、完整训练状态与推理包/API页面 | 已实现；容器行为检查、真实原生GPU差分、生产Job及严格同配方恢复证据保留 |
| Task7：B单卡基线、16→24、独立command与held-out诊断、取消释放 | 完成所选资格闭环；task quality仍失败，环境状态不支持exact恢复 |
| Task8：C/MJLab、跨后端sim2sim | 未实施/未测试；保留B，不能称C blocked或sim2sim通过 |
| Task9：四卡1learner+3samplers、16→24、资源/同步/评估等价 | 所选64env/sampler与4×1评估资格完成；4×4数值等价失败，更大采样/评估档未验证 |
| Task10：首个1000新增更新、独立快评、三轮复核、冻结best与唯一最终acceptance | 本轮有界执行完成；累计1024、latest验证0/16，best24验收0/100/fail；收集容量故障已修复补收集，原Job失败保留，磁盘历史超限明确披露 |
| Optional A / 教师 / 真机 | 未执行；没有教师训练或真机放行 |

最终结论：**四卡训练与恢复执行完成，正常行走验收失败，整体计划尚有C/MJLab及证据缺口。** 原始任务镜像、权重、split和门槛不变；不提交/暂存/推送Git，不以smoke_passed、进程exit0或loss下降宣称模型质量合格。后续若要继续学习，需要另行明确预算及调整/资格链，本轮不自动继续。
