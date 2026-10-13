# BFM-Zero 四卡 1000 次更新与可视化评估

状态：四卡RLinf训练1000次联合更新、四卡每卡4env的完整100-case物理评估及可视化核验均已完成；模型0/100通过，qualification=fail，尚不能稳定直立行走。执行计划见[训练台账](../superpowers/plans/2026-10-09-bfm-four-gpu-1000-eval.md)及[并行评估台账](../superpowers/plans/2026-10-09-bfm-parallel-evaluation.md)，设计见[16](../design/16-bfm-zero-data-and-4gpu-training.md)与[18](../design/18-bfm-reference-prior-and-g1-rpo-retarget.md)。不训练教师，不提交Git。

## 冻结数据

最终混合视图 `ds_27fbaa34696e4f26b2c7`：官方 RPO 30 个片段，加上 40 个经过标定的 G1→RPO 片段。共 25431 帧，19948 有效帧；manifest SHA256 `252a2080470a3b3b029c9e17f9108eddd4bb3d388234fbc797165a972bab4195`。官方片段仅进入 train，原始动作祖先不明，不能声称跨来源完全独立。G1 按 take 冻结 train/validation/test，所有 subject、切片同组。

| 分组 | 片段 | 帧 | 有效帧 | 连续九帧窗口 |
|---|---:|---:|---:|---:|
| train | 52 | 18231 | 13644 | 11922 |
| validation | 12 | 4800 | 4159 | 3873 |
| test | 6 | 2400 | 2145 | 1954 |

G1 分组分别为 8/3/4 个独立 take；官方 30 片段作为一个未知祖先组。独立 FK 最大误差 0.99694 mm。实际训练 loader 仅使用 train，RNG 恢复检查通过。数据 `training_ready=true` 仅代表本地研究行为先验资格，`controller_evaluation_pass=false`。不因合并提升商业/分发权利；用户限定本地非商业研究、无派生数据分发。

70 个片段是纳入视图的数量，不等于全部具备可采样窗口。G1 40/40 可用，官方 28/30 可用；`rpo_127_06`、`rpo_A1-_Stand_stageii` 没有连续有效九帧窗口，不能强行贡献专家序列。因此 train 52 个列入片段中有 50 个实际具备专家窗口。见[片段资格聚合](artifacts/2026-10-09-bfm-four-gpu-1000-eval/usable-clips.json)。

## 容量与训练

通过现有 API → Job/Run → ebd → RLinf 容器执行。GPU0 单 learner，GPU1–3 各 8192 环境，总计 24576；batch128，4 控制步/联合 rollout，1 learner 联合更新/round，replay100000。每轮 98304 transition，已达到固定 replay 的 98.3%，不静默截掉整轮。512/1024/2048/4096/8192 档均实际通过两次联合更新和完整 checkpoint；8192 是此冻结 profile 的最大已验证档，不是硬件绝对最大值。未测试 learner 卡兼任第四个 sampler。

8192 探测时三张采样卡一秒间隔采样显存峰为 8196/8175/8160 MiB，每卡总 24564 MiB；learner Torch allocated 峰 2936.3 MiB。完整 Run 的 Docker memory 采样峰为 41.23 GiB/64 GiB，包含 checkpoint 阶段；14.84 GiB 只是较早运行时点，不能称为全程峰值。shm 当时读数 368 KiB/32 GiB。峰值均为采样值，不是连续硬件峰值。详情见[容量原始聚合](artifacts/2026-10-09-bfm-four-gpu-1000-eval/capacity.json)。

| 每采样卡 env | GPU0峰 MiB | GPU1/2/3峰 MiB | 容器内存采样峰 GiB | 两次更新/完整保存 |
|---:|---:|---|---:|---|
| 512 | 4188 | 5208 / 5255 / 5208 | 26.42 | 通过 |
| 1024 | 4180 | 5366 / 5413 / 5366 | 27.95 | 通过 |
| 2048 | 4188 | 5870 / 5853 / 5870 | 29.82 | 通过 |
| 4096 | 4188 | 6540 / 6605 / 6560 | 34.64 | 通过 |
| 8192 | 4188 | 8196 / 8175 / 8160 | 41.23 | 通过 |

训练 `trn_3593c876d82e489a9475`，Job `job_3732ab6715164a319c69`，Run `run_d8c1001d92f4450a85eb`，2026-10-09 12:56:39 UTC 提交。数据/资产 content snapshot/codec/配置固定，从头初始化，不使用教师或 warm start。镜像 `eb/rlinf:rpo-v2-bfm-four-gpu-v6`，digest `f585a558eb21b09d835bf8df428d49d824f5d269564705ff886dccb0101b7270`。每 50 次更新保存 checkpoint，预算为 1000 次 learner 联合更新；预期采样 98304000 transition，最终须实际回读。

模型为已设计的 bringup 网络 profile：主体宽度 1024、3 个残差块，backward 256/1，discriminator 1024/3，latent256；不是原论文 2048/6 的 research profile。375 维历史 actor 输入、733 维 critic 输入、23 维动作保持原生 BFM codec。大采样量配合每轮仅一次 batch128 联合更新，UTD 很低；1000 更新是本轮有限预算，不保证已收敛。

`env/reward` / `env/return` 来自 BFM 原生辅助负代价：关节限位、施加力矩平方、动作变化平方，各乘控制周期。它不是速度指令跟踪奖励；`env/success_once` 也不替代本专属 Suite 的独立成功判据。最终控制能力使用实际物理评估的 RMSE、完整时长及硬限位等指标判断。

第 50 次真实 checkpoint 于 13:09:31 UTC 完成；实际 updater=50、replay size100000、total_seen4915200、312 个权重张量均有限、target 参数存在。checkpoint 保存模型/优化器/target/replay及 actor RNG；environment 和三个实际采样 rank 的 rollout 状态不可用，因此不是完整管线精确恢复。

最终训练于 2026-10-09 16:08:31.820 UTC 成功结束，作业起止约 3 小时 11 分 45 秒，包含启动/同步/保存，约 8544 transition/s。模型 `mv_41f640b997cc499f84ea`。第 1000 次实际回读：updater1000、168 个优化器 step 张量全部1000、replay size100000/total_seen98304000，312 个权重张量全部有限；四组 target 均与在线参数存在差异。全部1000个更新计数严格为1..1000，全部1000轮各收集98304 transition，全部记录标量有限。见[训练聚合](artifacts/2026-10-09-bfm-four-gpu-1000-eval/training-summary.json)、[完整指标](artifacts/2026-10-09-bfm-four-gpu-1000-eval/metrics.json)、[实际checkpoint回读](artifacts/2026-10-09-bfm-four-gpu-1000-eval/checkpoint-1000.json)。

最终权重 SHA256 `fa577939a78e4afa2ef9b2fc6f95fbfe7506e3cb04e431906769aa000ab797de`，resolved config SHA256 `82eb7377eecf9eb9b8db8f626654418c42013133cc6a18caa687e6e3760d2d3d`。正式 Run 整卡显存采样峰 GPU0/1/2/3 为 4400/8506/8539/8492 MiB，容器内存采样峰43.13 GiB/64 GiB，learner Torch allocated峰3101.65 MiB。这些与两次更新探测值分别记录，详见容量聚合。

每条采样 transition 对应的联合更新数约 `1.0173e-5`，每条采样对应的 replay 抽取行数约 `0.001302`。每次联合更新包含 discriminator、FB、两个 critic 和 actor 的优化步骤；不是五个步骤合计1000。最终平均 episode 长9.05控制步，不能作为稳定行走证据。

![1000次实际更新训练曲线](artifacts/2026-10-09-bfm-four-gpu-1000-eval/training-curves.png)

## 评估方法与桥接诊断

使用冻结 Suite `suite_rpo_v2_bfm_isaac_velocity_commands_v2`；command→latent→actor 导出、Torch/ONNX 一致性及来源 hash 绑定后，真实物理 step，记录轨迹和逐例录像。先做四个固定指令 smoke，再做 100 个独立种子的标准评估，分母始终为 100。标准案例每个 20 秒，恢复官方完整指令范围、70/30 平地/粗糙地形、startup 随机化和最终物理幅度传感噪声；BFM 原生 gain/零延迟保留，所以不冒称旧 Flat Suite 相同接口。

标准通过条件：完整持续 20 秒，无 fall/NaN/关节位置、速度、力矩硬限位，线速度向量 RMSE≤0.2、yaw RMSE≤0.3，至少 80/100 成功。Job 成功与模型 qualification 分开报告；验证/测试参考分组不进入 command→latent 参考库。

桥接诊断 `job_11d7b6184d794f679343` / `run_299713aee0d14a40be53` / `ev_94e37cefef2b4371a244` 使用旧的两次更新探测模型：4 case 真实执行，0/4 成功，4 硬限位，0 NaN，0 fall，31 物理指标步；线速度 RMSE0.801366，yaw RMSE2.930666，qualification=not_requested。四段 MP4 实际解码并查看，为 6/2/2/2 帧，早停导致视频很短。仅证明评估链路跑通，不是最终模型评估结果。

## 多轮复核

1. 保留原始参考、不训练教师；真实标定进入 solver，实际 Isaac FK 与独立 FK 交叉核对。
2. 混合 v1 发现官方一帧根角速度异常后，派生 v2 并重算 mask/hash/窗口，旧版本不覆盖。
3. 训练新增资产 content snapshot；标准评估要求真正使用此快照，而不是可变工作区。
4. 评估实际发现平地无 terrain_levels，保留失败 Run，修复缺失索引为 null 后新 Run 真实通过；恢复标准 terrain/noise，避免只评估平地。
5. 独立全链路只读审查未发现当前 Run 的直接阻断错误；三项准入缺口在有效配置 RED 复现，再锁定 max_steps=-1、rollout_epoch=1、正式 batch128。自主预算复核再补齐首轮联合采样必须≥batch的检查，避免小环境配置预热导致更新数不足。最终容器聚焦及平台/容量回归 51 项通过。本次 Run 原参数正确，保持运行。
6. 对照实际镜像 TerrainImporter、TerrainGenerator 和冻结 GRAVEL 配置：单环境列0虽然固定，官方 curriculum=False 仍令各单元按seed随机生成。70/30是子地形生成概率，不是宣称实际100案例恰好70/30。

未验证真实机器人控制；没有将参考几何合格、Job成功或两次更新的探测结果视为模型合格。

## 最终模型标准评估及四卡扩展

单卡v2完整标准：`job_cc6f579ccbf44019b36d` / `run_ce5af618e60f4c1a888f` / `ev_ca6d10446cac45ecbbc2`。2026-10-10 00:27:08至01:13:04（Asia/Shanghai）执行；100/100实际案例完成，0基础设施失败，成功0/100，fall66、hard_limit37、NaN0，两类失败可能重叠。实际470指标步，线速度向量RMSE0.838280m/s、yaw RMSE4.001406rad/s，qualification=fail。全部100段1920×1080/25fps真实MP4已逐帧解码共363帧，提前失败所以视频短；不延长录像制造20秒存活。证据见[登记与聚合](artifacts/2026-10-09-bfm-four-gpu-1000-eval/baseline-evaluation.json)、[录像逐段核验](artifacts/2026-10-09-bfm-four-gpu-1000-eval/standard-hd-video-validation.json)。

用户要求每卡一个进程、多独立env后，派生不可变Suite v3（共享官方地形网格，每batch首seed生成；startup/reset/command/noise/interval推扰按case独立随机流）。阈值和分母保持100/80，不声称v2逐例重新生成的相同随机轨迹。API新增1或4GPU、每卡1或4env；原v2保留。训练v6不覆盖；独立评估v7 digest `afdb9fb7522d783e5b90196a018eca33c431fbd19740f02c15975acceea4bff9`。单卡4env实际诊断`job_e1941cb298184cc095c8` / `run_4e69a55001cc453aab14`已启动。

该单卡4env诊断已成功收集`ev_89422b7f381245d8acfd`，实际7/6/6/6步、0/4成功、4fall、0NaN/硬限位；四路HD视频17帧已解码并查看画面，显存1秒采样峰6901MiB。随后四卡v7 Run `run_63811e3626984e9d9513`暴露installed usdrt仅支持cuda:0的Fabric相机pose查询，rank0输出12例、rank1–3受阻，已通过API取消并保留，不计入最终分母。

独立复核installed Camera/XformPrimView源码后派生v8，仅Camera视图走已有USD读写，CUDA相机缓冲/物理/渲染仍使用worker rank；同卡多env语义保持。相机兼容回归先RED再GREEN，9项Torch+22项平台/容量/历史Flat收集回归通过。v8 digest `b4c911f26e8051821f607d793102a7d174168e8903e19b565516e94055ef5ffd`。四卡真实录像诊断`job_049609766ed94326ae1b` / `run_7db63bf9039e4520adf9`已启动，优先验证实际非零GPU路径，再完整100例。原单卡结果及旧镜像不覆盖。

## 最终四卡结果（2026-10-10 Asia/Shanghai）

四卡录像诊断已成功收集`ev_18b77c2f90414165a0b2`，四个物理GPU各一个case，实际7/6/6/6步、四段1920×1080/25fps视频共17帧全部解码，并实际查看非零卡终态。确认相机兼容后执行完整标准评估：

- Job `job_6bd4c6d60651403cb95c`，Run `run_b78c49332c394751adc2`，EvalReport `ev_0358de6a86dc42c0b781`；模型始终为`mv_41f640b997cc499f84ea`。
- 2026-10-10 01:35:13至01:43:22（Asia/Shanghai），8分9秒，包含导出、启动、场景重建、录像和收集。
- 4个不同PID/不同真实GPU UUID；每卡一个仿真进程，共享已导出的模型/只读资产。每卡25种子，6批×4env加最后1env，最多16个独立case同时运行；全部100种子无遗漏、无重复。
- 完整100/100，0基础设施失败；Job succeeded，模型qualification=fail。失败案例仍计入100分母。

| 指标 | 四卡v3结果 | 验收条件 |
|---|---:|---:|
| 成功 | 0/100（0%） | 至少80/100 |
| 跌倒判据 | 66 | 成功案例不得触发 |
| 硬限位 | 36 | 成功案例不得触发 |
| NaN | 0 | 0 |
| 线速度XY向量RMSE | 0.830548 m/s | ≤0.20 |
| 世界Z转向速度RMSE | 3.924208 rad/s | ≤0.30 |
| 实际持续时长 | 平均0.094 s，最大0.22 s | 每例20 s |

fall与hard_limit可以重叠，不能把计数直接相加为案例数。36个硬限位案例包含15个关节位置、25个关节速度越限，4个两类重叠；无施加力矩越限。累计470物理指标步、10810个动作值，97.206%满足abs(action)≥0.99，平均绝对动作0.991535。观察到明显饱和，不能据此单独证明某个网络层、数据来源或超参数是唯一根因。倾斜fall判据不等于已经接触地面。

全部100条trace重新计算yaw对齐速度、RMSE、跌倒/机械限位；逐项验证actual_steps/metric_steps、z norm16、动作边界、初态与trace哈希、100个不同关节初态、每GPU实际设备及batch/种子覆盖。重新计算与平台聚合完全一致。100段真实HD视频逐帧解码共360帧，录像短是实际提前失败；不补帧、不伪造20秒稳定轨迹。详见[逐例结果与核验](artifacts/2026-10-09-bfm-four-gpu-1000-eval/parallel-evaluation.json)、[逐视频核验](artifacts/2026-10-09-bfm-four-gpu-1000-eval/parallel-hd-video-validation.json)。

四卡评估整卡显存1秒采样峰GPU0/1/2/3为9761/9634/9857/9762MiB（每卡24564MiB），容器主存采样峰57.49GiB/96GiB；不是连续硬件峰值。详见[评估容量](artifacts/2026-10-09-bfm-four-gpu-1000-eval/evaluation-capacity.json)。v2单卡完整评估45分56秒，v3四卡8分9秒，但共享网格/随机流协议不同，不声称相同逐轨迹的纯扩容基准。

![真实四卡评估结果](artifacts/2026-10-09-bfm-four-gpu-1000-eval/evaluation-results.png)

[四卡评估工作台](http://1.94.0.106:31800/eval/jobs/job_6bd4c6d60651403cb95c) · [最终报告](http://1.94.0.106:31800/eval/reports/ev_0358de6a86dc42c0b781)。逐例录像保存在`/data/kk/embodiedbein/runs/run_b78c49332c394751adc2/output/video/eval/seed_<seed>/0.mp4`，逐例物理状态在同Run的`output/traces/`。

## 结论与验证边界

已按授权预算完成整条RLinf训练→模型→四卡物理评估→真实录像→登记报告管线，未训练教师、未提交Git。静态先验资格、有限训练成功与实际控制质量分别判断；本模型未达到稳定站立/行走标准。可考虑下一轮减少采样量、提高每条采样对应的更新次数，并先排查动作饱和/FB loss/先验与PD动作范围差异；这是后续实验建议，本轮没有自动追加训练。

独立最终静态审查及实际四卡复核后，无当前执行阻断；相机兼容问题已真实重跑验证。P3暂缓：v3 code_trace逐文件清单少于v2，但镜像digest与资产content snapshot仍锁来源；未为此覆盖已运行镜像。所有案例在0.22秒内终止，真实20秒持稳、10秒command重采样、10–15秒推扰及完整运动提示能力未验证。真机未验证；参考几何/FK检查不代表足接触/足滑等动态质量合格。
