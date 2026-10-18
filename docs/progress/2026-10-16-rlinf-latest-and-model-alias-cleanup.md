# 2026-10-16 · RLinf 日志保留最新实验与多余模型路径清理

用户明确授权 `/data/kk/RLinf/logs` 只保留最新实验，并确认重复模型只删除多余副本、保留仍被调用的路径。本轮仅对该日志目录执行授权清理，RLinf 源码及其他外部仓检查点不在范围内。未修改生产代码，未暂存或提交代码。

## 实际结果

- 删除 33 个旧 RLinf 实验目录，仅保留 `/data/kk/RLinf/logs/20260402-02:57:10-maniskill_ppo_openvla`，以目录中的实验时间戳确定最新实验。
- 删除多余模型目录 `/data/kk/models/pi05_base`，保留被 LeRobot 调用的 `/data/kk/models/lerobot/pi05_base`。
- 删除多余模型目录 `/data/kk/models/XiaomiRobotics/Xiaomi-Robotics-1-5B`，保留已登记模型 `/data/kk/models/mdl_840294c597c94d928052`。
- 保留 `/data/kk/models/Qwen3-VL-4B-Instruct` 与 `/data/kk/models/Qwen/Qwen3-VL-4B-Instruct`，两条路径分别被 LingBot 与 Xiaomi 调用；其共享权重状态不变。

本轮前后文件系统可用空间净增加 **30,420,963,328 bytes，约 28.33 GiB**，完成时 `df -h /data/kk` 显示约 **793 GiB 可用**。两份多余模型权重在上轮已变为硬链接，本轮删除别名不会再次释放对应大文件；主要释放量来自旧实验。可用空间差值也会受同期其他磁盘写入影响。

逐目录清单、保留文件 SHA256 与检查结果：[执行记录](artifacts/2026-10-16-rlinf-latest-and-model-alias-cleanup.json)。

## 多轮复核

1. 重新盘点 34 个日志目录及实时平台元数据，无活动平台 Job；宿主打开文件检查未发现日志目录使用者。RLinf 源仓根的代码引用不等于消费旧日志，代码快照规则明确排除根目录 `logs`；另核对具体实验目录名及路径不存在平台引用。
2. 确认最新目录无符号链接依赖旧实验，删除前记录全部文件 SHA256、inode、大小和 mtime。确认要删模型副本与保留主副本为同一 inode，并重算完整权重 SHA256。
3. 容器内持有平台存储 advisory lock，仅允许日志和模型目录写入；逐项保存删除清单。删除后最新目录的全部文件哈希及清单一致，pi05/Xiaomi 主副本哈希一致、链接数均为 1；两条 Qwen 路径的文件清单一致。
4. 相关平台表与血缘逐字段不变，原 72 份总结 Markdown 的 SHA256 全部不变；api、ebd 健康。未执行镜像构建、拉取或删除。

本轮未重跑训练或完整模型加载。旧实验日志和检查点未备份，删除后无法恢复；其余平台依赖保留组未动。
