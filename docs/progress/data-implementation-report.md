# 数据接入实现记录（2026-09-09）

对照现行 `docs/design/03-data-collection.md`，本轮完成的是 LeRobot v3.0 的来源登记、持久 Job 驱动的结构检查和共享媒体索引，不是完整发布流水线。

- `POST /datasets` 验证本地目录白名单，登记为 registered/not_validated；不在 FastAPI 递归统计目录、下载 HF 或读取 Parquet。
- `POST /datasets/{id}/validate` 使用指定 embodiment_env_id 提交现有 data.convert Job；operation=validate_lerobot_v3。ebd 动态运行 tools 容器，本地源只读，报告独立输出。GET 依据持久 Job/Run 和原子报告协调状态，关闭浏览器不会取消作业。
- HF 只接受公开 dataset、40 位 revision，明确 repo_type=dataset，文件范围限定 meta/data/videos，已知总量限 100 GiB，保留 20 GiB 磁盘空间。目标必须为受管新/空目录。HTTP 和内联 token 暂拒绝。
- 工具镜像固定 pyarrow 19.0.1、huggingface-hub 0.34.4。API 不增加这些解析依赖。
- v3 检查 info、必需 features、episode metadata、数据分片列/行数、episode/frame 对应关系、全局区间、视频存在与有限 offset。100000 episode 上限；JSON 报告 64 MiB 上限。
- episode stream 直接使用 metadata 的 chunk/file 和 from/to_timestamp，两个 episode 可以引用同一个 MP4。删除按文件名猜测和 API 复制截断视频到 CAS 的路径。详情页选择相机时限制播放时间区间，明确只是浏览器诊断预览。
- `require_validated_dataset(row)` 为训练入口提供结构检查前置条件。报告 scope=lerobot_v3_structure，training_ready=false；index_hash 仅覆盖语义索引，不冒充 payload content_hash。

## 已运行验收

在只读挂载仓库的 `eb-tests-rereview2` 容器：

```
docker exec eb-tests-rereview2 python -m pytest tests/unit/test_data_validation.py tests/unit/test_data_view.py -q -p no:cacheprovider
```

结果：8 passed。用真实 Parquet 合成两个 episode，共享一个视频文件；检查不同时间区间、缺失视频拒绝、ready 不能通过验证门禁。视频内容为结构用 fixture，未进行解码验收。

F 类静态检查发现 routes.py 原有 891 行 eb_tsdb 未使用；新数据模块未发现 F 类问题。前端构建和整体接口回归由主任务汇总。

## 明确未完成

没有把外部目录冻结成不可变 DatasetVersion/DataSlice；没有 payload 全量 hash、发布/GC、license 血缘、无泄漏 split、train-only norm、视频解码、逐帧动作语义/时间对齐、目标模型真实 loader smoke。本轮未启动真实 HF 大数据下载、生产 Job 或训练。未知质量不会自动变成 trainable。

容量目前为单 Job 预检，尚无跨作业磁盘租约和运行中写入硬限；并发大下载仍需这些能力。失败任务重提/断点恢复 UI 尚未实现，API 重启持久性依赖现有 Job/Run 生命周期。索引暂存在 Dataset JSON，详情仅显示前100条；大规模 episode 游标分页待单独实现。结构检查并不保证浏览器帧精确，也不代表实际消费模型的 action/camera/norm 契约相容。

## 真实 LIBERO v3 数据复核与展示修复（2026-09-09 续）

真实源 `/var/lib/eb/datasets/ds_8aecbcf58ea74fda9ca3` 位于 API 的 `/var/lib/eb` bind mount。离线 tools 容器只读重索引成功：1693 episodes、273465 frames、40 tasks、未索引任务0。任务标签来自 `meta/tasks.parquet` 的 pandas index 字符串，逐帧 `task_index` 聚合到所属 episode；不再假设 episode metadata 自带 tasks 列。

详情和列表返回精简统计，不附 episode 全集；新增 `GET /datasets/{id}/episodes?cursor=0&limit=50`（上限100）和前端翻页。真实 summary JSON 从1,058,396字节降至4,415字节。详情采用索引的 episodes_count、action/state features 和 fps，不再错误显示0 episodes或 no_action_dim。

LIBERO数据是10Hz，环境 policy_hz为20，G2明确显示“不兼容：fps: 10.0 ≠ 20”。维度为action7/state8。即使维度和频率匹配也只显示待验收，不能把结构校验当作动作语义/loader完整验收。本轮不重采样动作、不修改训练共享guard。

新增 `POST /datasets/{id}/validate?reindex=true`：仅对终态创建新持久Job，HF采用local-only保留来源revision而不再次下载；成功后协调最新报告。已完成报告不会在每次GET时重新读入和提交。工具镜像已重建，离线验证报告在 `/tmp/eb-reindex-review/dataset-index.json`；生产端到端重索引待新API部署后执行。

```
docker exec eb-tests-rereview2 python -m pytest tests/unit/test_data_validation.py tests/unit/test_data_view.py -q -p no:cacheprovider
```

9 passed；覆盖真实task_index形式、分页、精简摘要及10Hz/20Hz不兼容。新数据模块F类检查通过。

### 生产 API / 持久 Job 复验

API镜像 `772c17fb59a5` 上提交 `POST /datasets/ds_398a3043320d4ba69566/validate?reindex=true`，生成 `job_a9398ac0bb114decbff1` / `run_318b78b67c7447a4bd80`，随后GET协调为validated、error=null。

实际HTTP详情8977字节，summary_stats不含episodes全集；overview episodes=1693、action7/state8、fps10；G2“不兼容：fps: 10.0 ≠ 20”。分页首屏50条、next_cursor=50，第二屏首episode=50。tasks_count=40、tasks_unknown_episodes=0，首任务文字恢复。quality含fps_mismatch和interface_not_verified，不含no_action_dim。上述断言在运行中的API容器通过httpx实际调用并全部通过。

此次上线镜像先于最后一处adapter只读挂载修正；local-only worker已跳过下载且仅读取源，但源挂载模式仍来自旧adapter。源码现已使local-only挂载为ro，需主任务最后一次make up生效；工具镜像为 `4c252a426f1d`。本轮仍未把10Hz动作重采样成20Hz。
