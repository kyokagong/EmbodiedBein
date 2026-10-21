# 10000更新模型可视化评估

2026-10-19（Asia/Shanghai），按用户要求重新执行一轮带录像的评估，未启动新训练或修改模型。

实际使用训练模型 `mv_83ddd78dabd3480a8172` 的10000更新冻结产物、原周期评估的任务latent（seed71000、16384样本），推理包 `mv_5c07067b95ba4a56afbc`。通过API预览冻结同一smoke-v2、Job/Run/ebd执行，站立与0.3m/s为主任务、0.7m/s为诊断；不使用正式acceptance资格。

- [mjlab三案例8倍慢放观看版，22.40秒](evidence/2026-10-19-ufo-native-10000-visual-eval/preview-mjlab-8x.mp4)
- [MuJoCo三案例8倍慢放观看版，15.68秒](evidence/2026-10-19-ufo-native-10000-visual-eval/preview-mujoco-8x.mp4)
- [全部视频下载包，约1.57MB](evidence/2026-10-19-ufo-native-10000-visual-eval/10000-step-evaluation-videos.zip)
- [六段原速原片、完整指标、回执与校验索引](evidence/2026-10-19-ufo-native-10000-visual-eval/README.md)

| 后端 | 站立终止时间 | 0.3m/s终止时间 | 0.7m/s终止时间 | 结果 |
|---|---:|---:|---:|---|
| mjlab | 0.24s | 1.22s | 1.32s | 3例均fall |
| MuJoCo | 0.22s | 0.82s | 0.86s | 3例均fall |

六例planned/completed/valid均完整，invalid/infra_failed/cancelled均0。原速录像480×480、25fps，触发跌倒判据即终止，部分视频仅0.24秒；视频帧采样使片长可能比仿真终止时间多一个帧间隔。观看版保持原帧顺序、每帧重复8次，画面持续标注8x SLOW；没有插帧或补造动作，不能用观看版时长表示存活时间。所有原片原字节不变，没有成功案例或成功录像。

mjlab：Job `job_06cafa2d6c3f4eedba45` / Run `run_cd72760d8fbe471d81df` / 报告 `ev_2ccf168d4855487288fb`。

MuJoCo：Job `job_0414e09abd5b45cd908f` / Run `run_69887e13817245bcaa6f` / 报告 `ev_217f1b29a5b243a799d3`。

三轮复核已完成：模型/latent/接口与冻结计划绑定；实际执行/计数与资源回收；全部case/视频/图表和媒体入口。18个索引成员的实际鉴权HTTP GET大小/hash通过，六视频Range206通过；六段原片和两个观看版在隔离CPU容器逐帧解码通过，并检查非空画面。29个本目录证据文件SHA通过，12成员视频包CRC和解压内容逐字节通过。原件仍在受管Run，工作区副本可直接观看下载。

07:43:36 CST外部复核API/ebd健康、无活动任务容器，四GPU lease均null/free均24080MiB。未改原评估Suite、设计25、外部源码、模型或训练数据，未部署、未Git提交。本轮结果仍为仿真诊断，模型尚未获得正式验收资格。
