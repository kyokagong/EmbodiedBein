# BFM 数据回放、四元数修正与 100STYLE 小样本试验

设计依据：[14](../design/14-bfm-zero-rpo-rlinf-integration.md)、[16](../design/16-bfm-zero-data-and-4gpu-training.md)；执行计划：[数据/四卡/训练计划](../superpowers/plans/2026-10-09-bfm-data-four-gpu-training.md)。用户授权先验证本地候选，再试少量 100STYLE；禁止 Git 提交。本轮仍使用现有未提交工作区，不覆盖其他改动，不启动未具备数据资格的正式训练。

## 已执行的本地链路

固定 [30 文件 hash/loader 清单](../../backend/eb/runners/rpo_bfm_local_sources.json)，执行 API → data.convert Job → ebd → 离线 Isaac FK，保留完整时间轴；pickle/joblib 均先验证全部固定字节，再反序列化。50 Hz 原生 23 关节/24 body，不截断为旧 smoke 的 500 帧，不裁剪参考关节。保留机械限位、速度、PD 命令可达范围、姿态/高度 mask；只从连续至少 9 帧的区间取专家窗口。

| 事件 | Dataset / Job / Run | 结果 |
|---|---|---|
| 初次完整转换 | `ds_0ab724bb4d534b64807a` / `job_21cf124672304a72b63f` / `run_424295a8207d40a2bec9` | 30 clips、9431 帧，发现源四元数解释错误；保留旧版本 |
| 首次真实回放 | `job_439885c7148f4bd7979b` / `run_a638f355d8494164a6e4` | 97 区间×3 seed，2/291；站立首步终止促成根因调查，不能作为正确源数据质量结论 |
| 修正后完整转换 | `ds_1bfbc60982fd4e878fa0` / `job_b5e4b1d09302414aa2fd` / `run_8ebe75a6285749bd990e` | 新 converter `isaac-fk-local-pool-wxyz-v3`；30 clips、9431 帧；`training_ready=false` |
| 修正后真实回放 | `job_77fa81bca68b4188b20c` / `run_eb74ac8a8bbe46118af8` | 97 区间×3 seed，27/291；整库未通过动态质量 |

修正后的 manifest SHA256：`000df06fe71ec6a4547353517f9c6533b79fef4996939995603df381ecf5966b`。9431帧中4351帧满足静态mask，97个至少9帧区间共4203连续有效帧（84.06秒，不包含重复seed）。资产和 codec 沿用冻结版本：`b58868a782d8863ac38059eb4eab89d27ee1d70565bf58f5a055f44510c8e842` / `fff6cf3e8f05c88fc7b552f65e5089dc47315c343bd9a205a1f59539cb2811c4`。

回放每个有效区间只初始化一次根位姿/q/dq，后续使用真实 PD 动作和物理推进，不逐帧 teleport。记录完整/提前终止、q/root/姿态误差、硬越限、接触足速和轨迹。配方 `rpo-bfm-pd-replay-v1`：seeds 42/43/44，q RMSE≤0.15 rad、root RMSE≤0.15 m、姿态 RMSE≤0.25 rad、接触足速 RMS≤0.30 m/s、无提前终止/硬越限/非有限数。按既定阈值接收回执和轨迹 hash，不因 Job succeeded 设置训练资格。

## 四元数根因与旧版本边界

只读核查源 `scripts/tools/retarget/gmr_to_lab.py`：原始 GMR 输入 xyzw，**输出 Isaac Lab motion 是 wxyz**。原 BFM converter 错把 `rpo_lab` 输出再次按 xyzw 解释；A1 第一帧正确 upright≈0.9994，错误转换 upright≈−0.0279。26 个 pickle 与4 个 joblib 均符合 Lab wxyz 约定。

SciPy 入口改为 wxyz→xyzw，Slerp 出口转回 wxyz；重力/body FK 和速度由同一姿态生成。新增真实源四元数回归检查。旧 `isaac-fk-raw-resample-v1`/`isaac-fk-local-pool-v2` 仍可读取追溯，但新训练拒绝它们；API 和 adapter 都复核 manifest 的当前资格，不能只信历史 summary。未原地改旧数据、模型、契约或评估版本。

原 BFM smoke 证明采样/六组件更新/保存/恢复/导出生命周期曾跑通，**不能证明旧 prior 姿态正确或模型行为有效**；后续正式训练必须使用修正后新数据与新初始化，不能把旧 smoke 的模型作质量合格起点。

## 外部小样本

[100STYLE 作者](https://www.ianxmason.com/100style/)公开 CC BY 4.0 与署名要求；选实际存在的 Neutral_ID/FW/BW/SW 及 Frame_Cuts.csv，不获取用途限制数据或人体模型文件。4 个动作共24877原始帧、414.617秒（含首尾T-pose）；按作者裁切区间去除冗余后才处理。作者文件为60 Hz、厘米/Y-up、Xsens Y-X-Z intrinsic Euler，不能套 LAFAN 的骨架/Euler约定。

| 事件 | Job / Run | 结果 |
|---|---|---|
| 首批下载 | `job_7f23511ab5af4cbfa121` / `run_b23cdfecc54246f8a1e5` | 仅头部检查通过；后续严格解析发现 FW 响应截断，不可作可用原始源 |
| 首批映射 | `job_3cae7c26433242a1b030` / `run_d29be090f5144c96884c` | 拒绝不完整频道，失败证据保留 |
| 严格重新获取 | `job_a7b5ce77d92d4190a8c9` / `run_177aa6ead41243a6ab9a` | 4 个文件均完整帧/频道/有限数验证；固定每文件 SHA，约17.4 MB |

Google Drive 单次响应可能截断；严格验证失败时从固定 [Zenodo v1](https://zenodo.org/records/8127870) 的100STYLE.zip按字节范围提取对应文件，必须 HTTP206/正确Content-Range，拒绝完整下载回退；校验 ZIP CRC 和取得文件 SHA。未下载14.8GB处理后数据或完整1.5GB包，未声称验证整个 ZIP 的 MD5。

小样本映射复用已安装 MuJoCo/SciPy：固定 RPO MJCF、原生关节序和硬限位/现有PD目标范围；Xsens格式T-pose坐标、常量骨长及RPO零关节足部旋转作校准，不把动作首帧当T-pose。每个动作仅取作者裁切范围中部3秒、150帧@50Hz，用有界IK拟合肩/肘/腕/膝/踝位置及足部朝向；记录误差和触边，不以可求解当作可物理跟踪。hhtools要求Python≥3.12且Newton依赖未安装，当前BFM任务Python3.11；本轮不引入整个新框架。

| 事件 | Job / Run | 结果 |
|---|---|---|
| 完整源第一版IK | `job_955553d229184e569a63` / `run_d87ac8fb685e4be791b2` | 4×150帧，RMSE 6.70–9.27 cm，全部帧触边 |
| 第一版真实回放 | `job_9d81747ea77e42f4aa97` / `run_0c9fbf95a2354226997d` | MJCF/Isaac FK最大误差约0.996 mm；0/12通过，28–41步因倾斜或低高度终止 |
| 分段骨长/支撑足校准v2 | `job_e18be813b77f423eae31` / `run_4338e81c0fc544989a9e` | RMSE降至0.99–1.56 cm，最大误差≤2.79 cm；仍有触边 |
| v2真实回放 | `job_52c84287d0924cff9811` / `run_9ec6ce614bf6447793c4` | 0/12通过，30–64步因倾斜/低高度终止；没有放宽原阈值 |

v2校准测量RPO零姿态的髋/肩anchor及上/下臂、上/下腿各段长度，保留人体骨段方向但按机器人实际骨长构造目标；用Xsens踝高度5%分位拟合地面，逐帧以较低支撑足对齐RPO踝中心0.04m。与第一版均匀缩放分别保存，不修改旧输出。四个动作的IK位置误差已明显改善，但开环PD不能完成3秒回放，**不能把IK成功当作专家轨迹合格**。这些结果是数据资格诊断，不是模型100-case独立评估。

| 动作 | v1位置RMSE (cm) | v2位置RMSE (cm) | v2真实通过 |
|---|---:|---:|---:|
| 站立 ID | 9.27 | 1.56 | 0/3 |
| 前进 FW | 7.81 | 1.34 | 0/3 |
| 后退 BW | 7.10 | 0.99 | 0/3 |
| 侧移 SW | 6.70 | 1.19 | 0/3 |

原始文件是同一Neutral风格/同一演员；不能把切片或重复种子当新增原始动作。当前本地池与外部试验均无正式train/validation/test资格；本地池全train只作诊断，正式划分必须额外核验包括`_M`镜像在内的原始祖先，不能沿用文件名当独立祖先。

## 修正后RLinf冒烟

通过训练preview及现有train API执行：`trn_b54a3ed03ace480c851b` / `job_7f1f40b2e4e8444aadeb` / `run_02f2bc407a334d87a1b9`。绑定修正后本地池`ds_1bfbc60982fd4e878fa0`，从头初始化，单learner、4env、每轮4控制步、专家/replay batch8、bringup网络、2次联合更新、每步checkpoint。未调用原smoke恢复输入，未将此诊断预算计入正式1000次。

已完成：logger steps0/1，`train/bfm/update_count`为1/2；累计采样32 transition、replay.size=32；完整step-2 checkpoint 435419491 bytes。模型权重SHA：`6fb7efce35761f9edef7c1ed5d42ca23aa2959cced45f998383b7aabc9a733bf`。CPU回读`bfm/state.pt`确认updater.update_count=2、rollout/RNG/专家sampler状态存在；16个trainer张量和312个权重张量均finite。仅本轮保存状态可按其覆盖恢复，仍不称物理exact resume。

两步指标是生命周期证据，不是训练稳定或行走质量证据：FB loss仍约31.4万→25.6万，FB梯度范数约148.6万→179.2万；没有完整episode。正式运行前仍需在已固定算法/网络配方下验证数值稳定，不能直接将smoke成功推断为1000更新可收敛。

## 多轮复核与后续条件

1. 静态关节合格不等于姿态/动力学正确：真实回放发现四元数错误，派生修正版本，保留错误版本作追溯。
2. 下载成功不等于完整原始数据：严格解析发现响应截断，补完整频道检查与有界发布包回退；不截掉坏尾部冒称完整。
3. 源授权、IK可达、闭环质量各自验证：本地 upstream rights 未独立核验；本地整库真实跟踪未通过，保持 diagnostic。外部仅12秒映射试验，没有独立数据划分/完整行为资格。
4. 四卡只扩大在线采样，不能修复离线专家质量；计划Task1的数据条件尚未满足，正式1000更新、四卡容量和模型视频评估尚未启动。若需RLinf motion-tracking教师，另行明确预算，不挪用1000次BFM learner预算。

已执行容器回归**128项通过**，新增runner/parser/calibration lint通过；现有被触及文件整文件 Ruff检查含既有行长/导入问题，未扩大为全仓格式整理。未提交Git。

## 后续预算已确认

用户随后授权推荐的**单卡独立RLinf动作跟踪教师探索、最长一小时**，并允许条件通过后自动继续四卡 BFM 训练和评估。执行细节和真实Run见[教师进度](2026-10-09-rpo-tracking-teacher.md)。教师训练预算与1000次BFM联合更新分别记账；时间上限或未通过门限不能晋升数据，也不能先消耗正式四卡预算。

Task1b具体执行顺序：固定tracking observation/reward/reset与参考未来帧语义 → 测试/受管镜像/契约revision → RLinf有限短训与完整checkpoint → 独立tracking质量回放 → 新采集DatasetVersion（含真实obs/action/next_obs/祖先来源和资格） → 回到本计划的单learner四卡容量探测。不得借用43/100的旧Flat模型作合格教师，不能只重放IK轨迹，也不能挪用1000次BFM learner更新预算。

最终任务镜像：`eb/rlinf:rpo-v2-bfm-zero` digest `sha256:24fa5d252391c1c21fc2b49409ebad2dc07126a5c0ce8a428ecb7178861baba4`；RLinf源码/patch hash沿用此前冻结来源。本轮多次镜像构建和`make up`均完成，既有Job保持各自冻结digest；外部源全部只读。

交付前回读：相关5个最终Job均`succeeded`/Run`finished`，API与ebd健康；本地30源文件SHA全部不变。分别以旧smoke池与首轮错误池调用训练preview，均400并明确要求派生修正版本；未创建失败检查的训练Job。`git diff --check`通过。
