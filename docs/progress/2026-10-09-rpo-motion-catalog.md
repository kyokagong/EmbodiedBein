# 本地 RPO 动作目录

日期：2026-10-09（Asia/Shanghai）。只读源目录：`/data/kk/assets/roboparty/rpo-v2/robolab/data/motions/rpo_lab`，revision=`6b1c3d9988497c8961dcba77892de32edc1770e1`。

本目录按文件名解释动作；未进行本轮逐片段可视化，数字编号不强行标注动作，`M` 保留为变体标记。时长采用原始帧数/FPS；50 Hz 候选帧采用此前的静态关节限位、PD 目标范围与速度筛查。候选帧可能分散，不能把其合计时长视为一条完整连续轨迹。连续窗口至少9帧，只满足专家8帧序列加下一帧的最低采样条件，不代表动力学合格。

| 文件 | 按文件名分类 | 原始时长（秒） | 静态候选帧 / 50 Hz 总帧 | 有连续9帧窗口 |
|---|---|---:|---:|---|
| `114_08.pkl` | 编号片段，待确认动作 | 13.40 | 272 / 670 | 有 |
| `114_09.pkl` | 编号片段，待确认动作 | 15.63 | 134 / 782 | 有 |
| `127_04.pkl` | 编号片段，待确认动作 | 3.72 | 128 / 186 | 有 |
| `127_06.pkl` | 编号片段，待确认动作 | 1.44 | 0 / 72 | 无 |
| `36_01.pkl` | 编号片段，待确认动作 | 4.64 | 222 / 232 | 有 |
| `36_11.pkl` | 编号片段，待确认动作 | 34.73 | 1489 / 1737 | 有 |
| `A1-_Stand_stageii.pkl` | 站立 | 3.81 | 191 / 191 | 有 |
| `B10_-__Walk_turn_left_45_stageii.pkl` | 步行左转45° | 5.98 | 249 / 299 | 有 |
| `B11_-__Walk_turn_left_135_stageii.pkl` | 步行左转135° | 6.49 | 250 / 325 | 有 |
| `B13_-__Walk_turn_right_90_stageii.pkl` | 步行右转90° | 5.62 | 175 / 281 | 有 |
| `B14_-__Walk_turn_right_45_t2_stageii.pkl` | 步行右转45° | 5.80 | 187 / 290 | 有 |
| `B15_-__Walk_turn_around_stageii.pkl` | 步行转身 | 7.51 | 245 / 376 | 有 |
| `B4_-_Stand_to_Walk_backwards_stageii.pkl` | 站立→后退 | 4.61 | 120 / 230 | 有 |
| `B5_-__Walk_backwards_stageii.pkl` | 后退行走 | 8.13 | 182 / 406 | 有 |
| `B9_-__Walk_turn_left_90_stageii.pkl` | 步行左转90° | 5.03 | 205 / 252 | 有 |
| `C12_-_run_turn_left_45_stageii.pkl` | 跑步左转45° | 2.20 | 0 / 110 | 无 |
| `C17_-_run_change_direction_stageii.pkl` | 跑步改变方向 | 3.05 | 0 / 153 | 无 |
| `C1_-_stand_to_run_stageii.pkl` | 站立→跑步 | 4.09 | 120 / 205 | 有 |
| `C3_-_run_stageii.pkl` | 跑步 | 2.72 | 0 / 136 | 无 |
| `C4_-_run_to_walk_a_stageii.pkl` | 跑步→步行 | 3.85 | 49 / 193 | 有 |
| `C5_-_walk_to_run_stageii.pkl` | 步行→跑步 | 3.57 | 56 / 178 | 有 |
| `move_back.pkl` | 后退 | 6.24 | 0 / 312 | 无 |
| `move_l.pkl` | 左移 | 5.60 | 59 / 280 | 有 |
| `move_r.pkl` | 右移 | 5.00 | 8 / 250 | 无 |
| `run_start_180_R_001__A345.pkl` | 起跑（180_R/M变体） | 3.02 | 0 / 151 | 无 |
| `run_start_180_R_001__A345_M.pkl` | 起跑（180_R/M变体） | 3.02 | 0 / 151 | 无 |
| `run_stop_180_R_001__A345.pkl` | 停跑（180_R/M变体） | 4.03 | 0 / 202 | 无 |
| `run_stop_180_R_001__A345_M.pkl` | 停跑（180_R/M变体） | 4.03 | 0 / 202 | 无 |
| `turn_l.pkl` | 左转 | 6.80 | 6 / 340 | 无 |
| `turn_r.pkl` | 右转 | 4.78 | 5 / 239 | 无 |

共30文件、20635原始帧、188.54秒；50 Hz重采样共9431帧，静态候选4352帧，18文件具有至少一个连续9帧窗口。26个文件使用pickle，4个run_start/run_stop文件使用joblib，均已在隔离容器读回。

对应[原始hash与筛查回执](2026-10-09-bfm-motion-inventory.json)、[后续训练设计](../design/16-bfm-zero-data-and-4gpu-training.md)。当前未确认原始动作权利链或动力学质量，未发布为正式训练数据。

