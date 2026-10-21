# 100000更新模型可视化评估

2026-10-20（Asia/Shanghai）。用户要求对完成100000更新的模型进行可视化评估。本次复用单次训练结束后已经通过Job/Run/ebd完成的最终独立评估，重新核验模型/计划、逐case指标、六段原片和媒体访问，生成同步观看版、指标图、交互索引和下载包。没有追加训练、重复仿真、修改模型或改变判据。

- [六画面原速同步视频，20秒](evidence/2026-10-20-ufo-native-100k-visual-eval/100000-step-six-view.mp4)
- [全部六原片与观看版下载包，约15.3MB](evidence/2026-10-20-ufo-native-100k-visual-eval/100000-step-evaluation-videos.zip)
- [交互式视频索引](evidence/2026-10-20-ufo-native-100k-visual-eval/index.html)
- [指标图](evidence/2026-10-20-ufo-native-100k-visual-eval/evaluation-summary.png) · [完整逐case结果、原片和来源](evidence/2026-10-20-ufo-native-100k-visual-eval/README.md)

六案例全部运行满1000个策略步/20秒，未触发跌倒判据。正式任务成功仍为0/6：站立及0.3m/s主任务0/4，0.7m/s两例仅诊断；六例均被原冻结物理越限规则判为 `physical_limit`，速度跟踪也不满足原阈值。不把不跌倒当作任务成功，没有成功录像可展示。

| 后端 | 站立速度RMSE | 0.3m/s速度RMSE | 0.7m/s速度RMSE | 不跌倒 | 任务通过 |
|---|---:|---:|---:|---|---|
| mjlab | 0.907m/s | 1.740m/s | 1.838m/s | 3/3满20秒 | 0/3 |
| MuJoCo | 0.963m/s | 2.010m/s | 1.976m/s | 3/3满20秒 | 0/3 |

冻结RMSE阈值：站立0.15m/s、前行0.2m/s。站立两例关节位置没有越限，但速度分别有2/34个物理子步越限；物理越限判据优先返回，不能据此忽略也不满足的速度跟踪和漂移判据。0.3m/s关节位置越限占比为1.4125%/10.3875%，关节速度越限0.8625%/6.9875%。0.7m/s位置越限为33.3%/49.2125%，仅作诊断。

对比20000 checkpoint，站立从mjlab0.82秒、MuJoCo0.54秒提高到20秒，但每个checkpoint重新推导任务latent，图中比较是模型与各自任务latent组合的描述性结果，不能只归因于权重。六例是固定seed的小规模validation，不是正式acceptance或泛化证据。

训练实际达到100000原生agent更新，全局环境步48783360；五组Adam真实step全部100000，四rank完整Replay及源哈希验证通过。单次20000→100000训练由2026-10-19 19:27:13至2026-10-20 00:05:48（Asia/Shanghai），约4小时38分35秒。最新训练模型 `mv_2ea1dadb1b9c4e88b535`，prompt `mv_a4e828b0395e4c9eb873`，推理包 `mv_5458bbf626b249058b3f`。完整checkpoint：`/data/kk/embodiedbein/runs/run_55de123b5d4a4039bfe1/outputs/native/checkpoint`。

- mjlab：Job `job_6bacc9c4d15145ff9f70` / Run `run_e0879423386144f9abf8` / [工作台评估](http://1.94.0.106:31800/eval/reports/ev_080aeb66a89c4b9ebe29)。
- MuJoCo：Job `job_811b701ef334486994d0` / Run `run_17679eedd41c4e2cbc5b` / [工作台评估](http://1.94.0.106:31800/eval/reports/ev_fc2933ef457644deae2f)。

三轮复核：①训练/模型/prompt/导出/评估计划和seed身份绑定；②100000计数、五Adam/Replay、实际Job成功与四GPU空闲；③18个索引文件鉴权HTTP大小/hash、六视频Range206、六原片逐帧解码、六画面派生片500帧/25fps/20秒、12成员压缩包CRC与解压逐字节一致。所有原片480×480、25fps，观看版仅排版和标注，未抽帧、重复帧、插帧或变速；静态图和10秒视频帧已视觉核查。38个证据文件SHA已保存。正式验收仍未执行，未Git提交。
