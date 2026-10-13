# 四卡1000次learner更新及可视化评估

设计16/18；用户授权四卡训练与评估，不提交Git、不训练教师。继续已有工作区，旧数据/镜像/模型不覆盖。1000为learner联合更新，不是控制步。

## Tasks

- [x] 0：核对真实4×4090/24GiB、当前G1 codec资格及现有单卡限制。现有两秒数据training_ready=false。
- [x] 1：用户确认多G1动作和冻结分组，并补充必须包含官方RPO30片段；已实际创建70片段混合视图。
- [x] 2：平台拓扑固定actor:GPU0单learner、env/rollout:GPU1–3；修正单learner batch整除和每rollout rank独立RNG；保留单卡smoke兼容。
- [x] 3：CPU测试及小规模真实四卡完整更新；以512→1024→2048→4096→8192 env/rank递增，记录采样峰值/显存余量、RSS/shm、吞吐和完整checkpoint。不得直接套Flat每卡2048。峰值是1秒间隔采样值，不是硬件连续峰值。
- [x] 4：训练资格/实验用途/预算在API与adapter一致冻结；预览后新Run执行1000联合更新，有限超时、定期完整checkpoint；不改旧training_ready。
- [x] 5：导出绑定actor/normalization/command→z/train-only参考；使用相同BFM codec/history推理，物理step、独立种子、冻结case/阈值、逐例录像/指标；不能用参考FK回放代替策略评估。
- [x] 6：回读1000 updater计数、采样量、完整模型/优化器/target/replay状态；实际播放录像、完整case分母、模型qualification分别报告。多轮复核与progress证据，未达到也如实输出。

## 关键接口

API冻结数据purpose/用途→RLinf单actor拓扑→3个env/rollout rank；replay能覆盖完整联合rollout。导出/评估使用本次训练的数据/codec/权重hash，模型资格与Job成功分开。

## Rulings

数据范围尚待可选澄清；设计16要求正式训练ready数据，不利用已有smoke-only检查宽松直接冒充正式1000实验。不停止独立实现工作；保留用户已确认本地非商业/不分发用途。单learnerbatch/FB不按4个actor切分。

Ruling：等待可选澄清期间按推荐路线准备40个原始motion/subject的固定片段池，每片段400帧/8秒；同一个take的所有subject与切片同组，15个take按代码固定train/validation/test，先冻结选择再看质量结果。未达到参考资格不能自动进入正式训练。

Ruling：发现现有checkpoint只保存actor上的rollout对象，没有保存3个实际sampler rank状态；新回执明确rollout=not_available，保留model/optimizer/target/replay恢复而不宣称exact resume。

Recovery：一次镜像构建误用了旧Make target，新镜像短暂被指向G1 v3标签。原digest仍存在，已立即恢复旧标签到0403a695...并将新digest9f29da62...保存为独立four-gpu-v1；核验两tag身份正确。新增独立Make target避免再用旧标签。

Evidence：四卡拓扑/容量/entry回归45项通过；新分组/标定/拓扑5项通过。四卡模型固定单learner、3个采样rank，rank seed由42+1009×rank派生。旧数据和外部只读仓未写入。

Evidence：G1 raw pool ds_1c715fe844b442f3834d，job_61c98aee68994ff08a3f/run_d71151f21a434ae48da9，40×400帧完成。混合实际Isaac FK view ds_de556887d6c6457fa0d8，job_161f09c1cf23446ba266/run_85291c0c94344ca195b9，70片段25431帧，manifest 8b02d8b1d54a7eb9f846fa2a48cebf6135b904fe2f16433e203d46e91f8332fe；train52/18231/13645有效帧/11922窗口，val12/4800/4159/3873，test6/2400/2145/1954。train G1 8take、val3、test4；官方未知祖先1组train。独立FK最大0.99694mm，真实loader batch128及RNG复现通过。全部有效帧19949，其中17741帧超当前PD目标范围；保持现有冻结codec，报告能力风险，不以此替代实际BFM策略评估。

Ruling：正式batch128明确冻结；smoke可8或128且≤2更新。新增统一API/adapter预算资格检查，不依赖宽松历史smoke检查启动1000。原始motion权利未知的官方片段不因合并升级商业/分发资格。无PD或教师资格前置。

Evidence：四卡512/rank batch8原probe trn_ac25bd32e1104fcaa63c/run_d2be253d5bd840a2812d成功；batch128512 trn_0a4b2c743f60401c86b2/run_8bad1e14345d44379f79成功；batch1281024 trn_207fecf497fd46d58e2c/run_03bb017e88904c86b6ac成功；batch1282048 trn_12e37093c8d44f06832e/job_67088b37ca86426ba3a0已提交。全程GPU采样/tmp/eb-bfm-capacity-gpu.csv、Docker memory/CPU采样/tmp/eb-bfm-capacity-containers.jsonl。模型Torch峰约2938MiB，整卡1024 env/rank峰约5413MiB。

Evidence：four-gpu-v2 digest 0c459ca9cd8ccc09ec462cfb39b8d6baba5e05cf731c2096e313bf1555d35e1c；独立eval-v1 digest3ac65a5c29308e0ae5e428553329788c9da7ec90971ae868f86fcbc03dc10f83。评估新增native375/733、3Dcommand→256z→23Dactor，复用既有物理case/录像/指标循环，冻结专属BFM suite。单元桥2项、评估adapter+Flat回归10项通过（1项Torch循环因基础CPU测试镜像无Torch跳过）；先实际诊断4command录像，后100独立seed案例。尚未实际评估，不宣称可播放或合格。

Evidence：batch1282048成功/run_83bdc850c0b34172b5ba，Torch2-update full state和发布checkpoint通过。512/1024/2048整卡采样峰约4.85/5.42/5.87GiB（原CSV数值为MiB）；learner整卡≤4188MiB。原估计把所有卡都叠加learner+env，现四卡按角色用max(8GiB,(5+0.0015×env/rank)×1.2)保守外推，后续必须继续实际探测。replay100000约束整轮env总数×4≤100000，8192/rank=98304transitions尚能覆盖。

Reflection：跨70片段分析发现官方有效参考中有1帧根角速度24.53rad/s，超过本轮20rad/s时序阈值；不能修改已发布混合v1。派生mixed-pool-v2，把同一根速度/角速度mask也用于官方副本，并重新计算窗口/hash；原v1按历史recipe保持可读与原资格，不假改历史数据。新转换与正式训练将使用four-gpu-v3 digest222bf7153a5e2360e097e7aafaeba7b5b6e3e74e37ded87b864f99fb5d7c271b。使用同一v3镜像进行评估，避免额外镜像变体。

Reflection：复用Flat物理循环时_numpy原只接受Torch；新增375维NumPy路径已RED复现AttributeError→GREEN修复。Torch评估镜像执行真实共享循环/录帧/BFM桥8项通过；加入mixed资格后10项通过，无Torch跳过。新评估adapter与Flat回归10项通过；v3重点54项通过。

Reflection：历史BFM source_job没有embodiment code snapshot，只有motion/asset manifest绑定。新BFM训练冻结资产content snapshot且adapter实际从snapshot挂载；完整标准评估要求它。历史smoke允许仅诊断，明确source_asset_snapshot=not_available_for_legacy_smoke，并给本次评估重新冻结当前同manifest资产。不原地补写历史Job，不宣称它曾冻结整个资产代码。

Evidence：mixed v2 ds_27fbaa34696e4f26b2c7/job_5729e5c3b94f4215a105；manifest252a2080470a3b3b029c9e17f9108eddd4bb3d388234fbc797165a972bab4195，training_ready=true。来源仍为40个G1+30个官方RPO；过滤官方1个角速度异常有效帧，不修改v1。

Evidence：4096/rank实际mixed-v2完整更新成功 trn_d1a4f554034f42d99436/run_4cf26a87bc3146a495f7，batch128、2次更新、replay98304；8192/rank trn_f5750052833242019e0d/job_2da7220591de44668903已提交。8192档每联合rollout98304，占固定replay100000的98.3%，不将更高env静默截断。

Evidence：8192/rank实际成功/run_5d9c973ae92f475eb844，2联合更新、196608累计transitions，replay100000，正式数据mixed v2，Torch allocated2936.3/reserved3074MiB；采样卡全局1秒采样峰8196/8175/8160MiB，余量远超2GiB/10%，运行内存已观测14.84GiB/64GiB，shm368KiB/32GiB。采用此冻结profile的最大已验证安全档；不声称是硬件绝对上限。保持一个learner+三张采样卡，固定replay/每轮采样量约束与过低UTD会如实报告。

Evidence：修复后四指令诊断job_11d7b6184d794f679343/run_299713aee0d14a40be53成功收集，ev_94e37cefef2b4371a244；旧2-update探测模型0/4通过，4硬限位、0NaN、0fall、总31物理指标步，RMSE线速度0.801366/yaw2.930666，qualification=not_requested。这是评估桥验证，不能当最终1000模型结果；4视频已产出，尚须实际播放。

Evidence：正式本地研究1000更新已preview并提交 trn_3593c876d82e489a9475/job_3732ab6715164a319c69，mixed-v2 ds_27fbaa34696e4f26b2c7，镜像v6，单learner/3sampler×8192，batch128，save_interval50，从头初始化，bfm_experiment_purpose=local_research。预期98,304,000 transitions/1000 learner更新/每env4000控制步；最终必须实际回读，而非按预算宣称完成。尚未完成1000更新和最终100case评估。

Reflection：标准评估若沿用原BFM stage-A cfg会只有平地/无噪声。专属Suite派生v2，恢复原Flat70/30地形和startup事件，原生BFM q±0.03rad/dq±1.75rad/s/gravity±0.05/omega÷4±0.05的官方物理幅度噪声（取RPOFlatEnvCfg.__post_init__最终值），同episode/step缓存测量，使历史保存策略真实见过的噪声状态。BFM保持原生gain/零延迟，不称旧Flat完全同接口。原v1仅兼容已冻结smoke，不接收完整标准验收。

Reflection：首轮实际录像诊断 job_3b9c18876a4049cfa09b/run_0a811a5b31614dc89370发现平地TerrainImporter无terrain_levels，4case均infra_failure/0物理步；不是模型评估失败。RED复现→记录缺失索引为null→GREEN修复。v6 digest f585a558eb21b09d835bf8df428d49d824f5d269564705ff886dccb0101b7270；45重点测试通过（基础镜像1个Torch循环跳过），实际Torch镜像12项全部通过。用新Run重跑，保留原始失败与旧Suite。

Final review：一次独立只读全链路审查未发现当前正式Run的阻断错误，复现了共享form三项准入缺口：max_steps可缩短预算、rollout_epoch可溢出整轮replay、正式预算仍可默认batch8。已在有效四卡配置RED复现三项→固定max_steps=-1/rollout_epoch=1、预算>2要求batch128→13项聚焦测试GREEN。当前Run参数原本正确，保持运行且不修改。旧单卡≤2更新batch8兼容。

Evidence：正式Run为run_d8c1001d92f4450a85eb，实际第50次checkpoint于2026-10-09T13:09:31Z写入完整回执；actor weights435419333 bytes、DCP895884315 bytes、BFM state700216393 bytes。environment/rollout覆盖仍not_available，不能宣称整个四卡管线精确恢复。旧诊断四段MP4已实际解码并查看画面，分别6/2/2/2帧；这是探测模型早期硬限位终止的视频，不是最终1000模型结果。

Units correction：前述512/1024/2048粗略整卡数字不应把MiB直接写成GiB；最终容量报告统一采用原CSV的MiB，并按每Run时间窗分组。

Reflection：独立审查后自主复核更新预算，发现小环境数/大batch会因首轮replay不足而跳过learner更新；有效四卡env6/batch128/1000预算RED复现，补齐joint rollout≥batch检查，平台/表单/容量/评估回归51项GREEN。此处只补齐共享控制面准入，当前24576env Run不受影响，任务镜像v6与预算保持原样。

Reflection：核对实际镜像的TerrainImporter发现单环境列索引默认0；继续核对冻结资产GRAVEL与TerrainGenerator证明curriculum=False，每单元按seed随机采样70/30子地形，因此列0不等于固定平地。比例是生成概率，不声称最终100案例恰好70/30；最终回执仍须核对实际配置。

Evidence：片段级再核对，70列入片段中G1全部40可采样、官方28/30可采样；rpo_127_06与rpo_A1-_Stand_stageii无连续有效窗口。train52列入、50实际可采样，不假称坏片段强制参与专家更新。容量完整Run内存采样峰8192档41.23GiB，而14.84GiB仅早期时点；正式500更新内存41.77GiB/64GiB，非持续泄漏证据。第300真实保存的168个优化器step均300，replay累计29491200，四组target与在线参数均存在差异。

Evidence：正式训练2026-10-09T16:08:31Z成功完成，模型mv_41f640b997cc499f84ea。1000个实际update_count为1..1000，98304000累计transition，实际完整checkpoint updater=1000、168个优化器step均1000、312权重张量有限、replay100000/total_seen98304000，四组target与在线参数不同。全记录标量有限。正式内存采样峰43.13GiB/64、整卡0/1/2/3峰4400/8506/8539/8492MiB，最终Torchallocated3101.65MiB。

Evidence：最终模型四指令HD job_7e84cc833463480d90e5/run_89b1ac7c822644a5b03d成功收集ev_cf4ce9c4410d492ea03d；0/4、4fall、0NaN、0hardlimit、25物理步，velocity RMSE0.796570/yaw3.081174。4×1920×1080/25fps视频实际逐帧解码17帧并查看；倾角0.8246–0.9074rad超0.8阈值，物理时长0.12–0.14sec，短视频如实保留。

Ruling：标准100case的4CPU/320×240 Run job_cea93ddd93b04b72a56b/run_98267a8cf6644ddd9ba1首例初始化约8分钟、后续地形/仿真创建约30–40秒/例。保留6个已执行案例及原Run，使用平台取消API终止，改为已有HD/32CPU资源档的新100case Job。新预览校验模型权重、Suite spec、全部100种子与旧计划完全一致，验收标准不改变；仅视频质量/资源和新plan hash变化，不合并旧6case到新100case分母。新plan hash d4ed50adea9d1c7ada46b08e93456cc203a13029d98761ccb85d99ca7426de23。

Final evidence：训练1000次联合更新与98304000 transition已真实回读；最终模型v2单卡100例0/100，66fall/37硬限位、0NaN，100HD视频全部解码。按用户新增四卡共享env要求，另执行[并行计划](2026-10-09-bfm-parallel-evaluation.md)，经过真实相机兼容修复派生v8，四卡×4env完成100/100，`job_6bd4c6d60651403cb95c` / `run_b78c49332c394751adc2` / `ev_0358de6a86dc42c0b781`。0/100、66fall/36硬限位、0NaN/infra，线速度RMSE0.830548/yaw3.924208，qualification=fail；100段1920×1080/25fps实际录像全部解码360帧，每条trace重新计算与聚合一致。控制器失败如实报告，本轮目标是按预算完成训练与评估，不假称模型合格、不自动追加训练。执行未提交Git。
