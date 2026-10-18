# 2026-10-16 · 历史目录清理与重复权重去重

按用户确认，清理空间评估的前三项，并对重复权重去重、保留原调用路径。未暂存或提交代码，未修改生产代码；原有工作区改动保留。

## 执行结果

已删除以下三个目录，约释放 56.9 GiB：

- `/data/kk/embodiedbein/smokes/dm05-rlinf-run6`。
- `/data/kk/embodiedbein/snapshots/5750e388eb2c6c416a11212e7f54e75234e4b8c12c3fe606743cd7bd8043e6cb`。
- `/data/kk/verl/mem_snapshots`。

执行前重新检查实时平台引用、终态任务和宿主打开文件；无引用或活动使用者。维护脚本在容器内运行，仅写指定目录的父目录及模型目录；外部源通过只读挂载保护。检查及操作期间持有与平台存储提交/删除共用的 advisory lock。保存被删目录的文件清单和元数据，不备份被删大文件。

重复权重完整 SHA256 校验一致后，通过临时硬链接及原子替换合并存储。目录、配置、文件名和全部八条原路径保留，四对文件各共享一个 inode、链接数均为 2；未引入符号链接。

| 权重 | 两条保留路径 | 减少独立副本占用 |
|---|---|---:|
| pi05 | `/data/kk/models/pi05_base/model.safetensors`；`/data/kk/models/lerobot/pi05_base/model.safetensors` | 13.47 GiB |
| Qwen，两份 shard | `/data/kk/models/Qwen3-VL-4B-Instruct/model-0000{1,2}-of-00002.safetensors`；`/data/kk/models/Qwen/Qwen3-VL-4B-Instruct/model-0000{1,2}-of-00002.safetensors` | 8.27 GiB |
| Xiaomi | `/data/kk/models/XiaomiRobotics/Xiaomi-Robotics-1-5B/model_states.pt`；`/data/kk/models/mdl_840294c597c94d928052/model_states.pt` | 9.52 GiB |

去重减少独立副本分配量合计 **31.264 GiB**。pi05、Xiaomi 保持 0644；Qwen 采用原已登记副本的严格权限 0600，原另一份的 0644 收紧为 0600，没有放宽权限。现有 LingBot 和 Xiaomi 调用镜像均以 root 运行，两种镜像内的原路径读取检查均通过。

实际文件系统可用空间从 736,660,955,136 bytes 增至 831,258,046,464 bytes，增加 **94,597,091,328 bytes，约 88.10 GiB**；完成时 `df -h /data/kk` 显示剩余约 **775 GiB、使用率 78%**。以实际可用空间差值为准，不将文件逻辑长度相加当作释放量。

清单、SHA256、inode/权限和验证结果：[执行记录](artifacts/2026-10-16-space-cleanup-and-weight-dedup.json)。

## 多轮复核

1. **范围与引用**：重新读取全部相关平台表与血缘，确认三处目标无引用且无活动平台 Job；严格保留同级有效的 `dm05-view` 数据集。
2. **文件安全与去重**：拒绝符号链接、内容变化、意外硬链接和扩展属性；替换前校验内容及 inode/mtime/ctime。小文件自检覆盖原子替换、不同内容拒绝、严格权限和符号链接拒绝。
3. **结果与可用性**：替换后再次完整校验四份共享权重 SHA256；八条原路径可读，safetensors 头与 PyTorch archive 检查通过，无残留临时链接。
4. **保护对象**：所有读取的平台表逐字段一致，`dm05-view` 的 64 帧 JSONL 可解析，原 71 份总结文件 SHA256 全部不变。Docker 镜像列表完全一致；RLinf 日志、BHL VLA 输出和 verl 检查点目录占用与清理前一致。api、ebd 均健康。

没有清理 Docker 镜像、构建缓存、外部仓检查点或上轮保留的 10 组依赖；没有重启服务或新跑训练。

未做完整模型加载、推理或 GPU 训练验证。删除的历史实验文件不可恢复；共享权重后，任一路径原地写入会影响另一条路径，后续更新须写新文件并替换路径或派生新版本。
