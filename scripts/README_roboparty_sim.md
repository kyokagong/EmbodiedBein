# RoboParty 独立仿真预览

使用 MuJoCo + ONNX Runtime CPU 运行当前 RPO-Flat 780→23 策略。入口、镜像和推理逻辑独立，不导入 `eb`，不调用平台 API、数据库、Job 或 Compose。可以把这四个文件复制到其他目录单独使用。

## 启动

在 Linux 本地桌面或远程桌面的终端执行；窗口需要 X11/XWayland 的 `DISPLAY`。首次启动自动构建独立 Docker 镜像，需要联网下载依赖，后续启动复用镜像。GUI 默认采用 Mesa 软件渲染，物理和 ONNX 推理均使用 CPU。

```bash
bash /data/kk/EmbodiedBein/scripts/roboparty_sim.sh
```

默认读取本次训练的导出目录及现有 RoboParty 描述目录。更换模型、迁移到 Orin 或复制到独立机器后，用以下环境变量指定本地文件：

```bash
RPO_MODEL_DIR=/absolute/path/exported \
RPO_DESCRIPTION_DIR=/absolute/path/rpo_description \
bash /absolute/path/roboparty_sim.sh
```

模型目录需包含 `policy.onnx`、`export-manifest.json`、`golden-traces.npz`。描述目录需保留 `mjcf/rpo.xml` 及相邻的完整 `meshes/`，不能只复制 XML。导出清单必须声明 RPO-Flat 原生关节顺序和内嵌归一化；脚本启动时检查哈希、I/O 和参考输出。

## 按键

先点击仿真窗口，使其获得键盘焦点。

| 按键 | 操作 |
|---|---|
| W / S | 前后速度指令增加 / 减少 0.1 m/s |
| A / D | 左右速度指令增加 / 减少 0.1 m/s |
| Q / E | 左右转向指令增加 / 减少 0.2 rad/s |
| X | 三个速度指令归零，策略继续维持站立 |
| P | 暂停 / 继续物理仿真 |
| R | 回到初始站姿，速度、指令、上一动作和历史观测清零，并继续仿真 |
| Esc / 关闭窗口 | 退出 |

速度指令逐次增减并保持，松开按键不会自动归零；使用 X 停止。按键范围沿用训练配置：vx∈[-0.6,1.0]、vy∈[-0.5,0.5]、wz∈[-1.57,1.57]。状态每秒输出到终端。摔倒后自动暂停，按 R 重置。

## 无窗口检查和可调参数

```bash
# 校验键盘、坐标系、关节映射、参考输出、历史和重置
bash /data/kk/EmbodiedBein/scripts/roboparty_sim.sh --self-test

# 10 秒零指令闭环物理测试；不创建显示窗口
bash /data/kk/EmbodiedBein/scripts/roboparty_sim.sh --headless --seconds 10

# 前进及转向指令；无窗口模式按仿真时间运行，不限速
bash /data/kk/EmbodiedBein/scripts/roboparty_sim.sh \
  --headless --seconds 10 --command 0.3 0 0.2

# 固定 5ms 执行器延迟；仍保持 50Hz 策略
bash /data/kk/EmbodiedBein/scripts/roboparty_sim.sh --actuator-delay-ms 5
```

`--physics-dt` 默认 0.001 秒，范围 [0.0001,0.02] 秒，必须整除 0.02 秒。`--seconds` 限制累计执行的仿真时间，暂停不计时，GUI 默认持续运行。`--actuator-delay-ms` 默认 0，可在 [0,100]ms 设置固定延迟，按物理步长取整；它不是训练时随机延迟的完整复现。`--build` 强制重建镜像；Python 文件每次以只读方式挂载，修改后无需重建依赖。

退出码：0 表示达到指定时长或正常关闭，1 表示摔倒或运行错误，2 表示参数/启动条件错误，130 表示 Ctrl+C。正常退出会输出 `simulated_seconds`、`fallen` 和 `finite`。

## 与训练环境的关系

- 23 个关节和电机按名称映射；单帧 78 维、10 帧历史按旧到新排列。首帧及重置后的首帧重复填满历史；历史保存未裁剪的策略动作。
- 原始历史观测在 ±100 内裁剪后交给 ONNX，由图内执行归一化。动作目标为默认角度 + 0.25 × clip(action,±100)。不再次归一化、不追加 tanh。
- PD 默认姿态、刚度、阻尼与 RoboLab `RPO_CFG` 对齐；腿/躯干力矩上限 120Nm，踝/手臂 27Nm。只在内存中覆盖电机控制范围，资产和模型以只读方式挂载。
- MuJoCo 平地预览使用名义参数及上游 MJCF 碰撞设置（机器人内部自碰撞关闭）；不复现 Isaac 的碎石地形、域随机化、随机控制延迟和关节速度约束。该功能用于人工部署前观察，不能替代 Isaac 正式评估或证明模型已通过验收。

## 已执行的验证

在本机 x86_64 容器中通过模型哈希和参考输出检查、基座坐标系/关节映射/历史/重置自检，以及三组各 10 秒的闭环：零指令；`0.3,0,0.2` 加 5ms 延迟；`-0.2,0.1,-0.2`。三组均未摔倒、无数值异常。只读离线容器中的 Xvfb 实际窗口测试确认移动按键进入物理循环，停止、暂停/继续、重置、Esc 退出均生效。非法周期和越界指令返回错误。实际桌面显示和 Orin 端性能仍需在目标机器检查。

上游资产及参数来源：RoboParty RoboLab（BSD-3-Clause）。MuJoCo 查看器接口见 [官方文档](https://mujoco.readthedocs.io/en/3.3.7/python.html#passive-viewer)。
