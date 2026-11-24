# Python 文件能力下沉 Rust：实施进度与验证

日期：2026-10-28。对应[实施方案](../plans/2026-10-28-python-rust-data-plane-migration.md)。设计真源仍为 docs/design/；本轮没有修改设计、数据库 schema、训练协议或 Git 提交。

## 已交付范围

| 任务 | 结果 | 边界 |
|---|---|---|
| 1 批量校验 | 已实现 | 一份 manifest 一次 UDS POST；默认 200 文件/2 GiB，上限 10000/1 TiB；空文件例外由 Python 判断 |
| 2 单文件 CAS | 已实现 | 1 MiB 有界复制/hash、源摘要复读、只读/fsync、同 CAS 文件系统 renameat2 NOREPLACE；不覆盖损坏对象 |
| 3 调用者 | 已实现 | UFO/P-BFM 批量校验；通用 ONNX/checkpoint 使用 Rust 发布引用和大小，ONNX 从 CAS 解析；失败登记回滚 |
| 4 目录事实 | 已实现所列异步入口 | eb-scanner 提供 inventory/size/text/hash，eb-probe 提供磁盘事实，ebd 装配 HTTP；domain scan/disk/URDF 仅解释输入 |
| 5 RSL 恢复 | 已实现 | Python 保留 mtime/step 选择；Rust 检查活动容器、显式文件并原子发布恢复目录，重复准备核对内容 |
| 6 媒体 | 已实现 | 旧 CAS、数据媒体、本体资产、评估视频/native 流复用 64 KiB fd 通道；GET/HEAD/Range、取消关闭、读事务释放 |
| 7 Bundle Job | 已实现 | 既有 JobKind/ToolsAdapter、CPU worker、collector、Rust tar.zst/CAS 贯通；safety/G3/G4/数值结果绑定，保持 unsigned/can_sign=false |
| 8 索引/遥测 | 条件评估后未启动 | 没有 JSONL 瓶颈或真实端侧遥测生产者的启动证据；robot-agent serve 是固定温度合成示例，不是现场数据 |
| 9 验证/交付 | 已验证并部署，结果见下 | 文件完整性不提升模型/真机资格 |

只读根为配置中的 state/models/data/assets、Run 及明确允许的 source；写入仍限制在受管 Run/CAS。Compose 显式传递 datasets/assets/source 配置，未增加平台 Agent 权限。

inventory 目录优先、路径稳定排序；单页最多 1000 项，整树最多 100000 项。小文本最大 16 MiB，超限/无效 UTF-8 明确失败。磁盘 10 GiB 余量、10% 警告、归档 2.5 倍保持原规则。

## 三轮复核与独立审查

1. **边界与协议：** 核对真实调用者、媒体/Replay 上限、空 __init__.py、旧路径/历史指纹、外部仓只读、Job/Run/ebd 与镜像 COPY。任务内 checkpoint 校验不新增控制面依赖。
2. **失败与并发：** 同长度改写、链接/越界、重复发布、损坏 CAS、ENOSPC 传播、有界复制、删除 pin、恢复冲突与重试；真实 PG 验证后一份产物 RPC/G3 失败会撤销前一份模型登记。日志后置失败也回滚，未引用 CAS 不自动删除。
3. **用户入口与证据：** 私有 Replay 拒绝、Range/HEAD/416/空文件、响应前取消/下载中断、所有 stream_output 调用者的读事务；沿用 Submit/Bundle 列表和下载链接，无新训练。

独立审查发现 6 项 Important 和 1 项 MIME 兼容问题；后者属于任务 6 明确要求，升级为必须修复。全部经过失败测试→修复→回归：登记事务、发布大小、Bundle safety、畸形 receipt、发送阶段取消、遗漏流调用者事务及 WEBM/MKV/大写扩展名。

最终并发回归还实际触发了硬链接清理改变 ctime、导致相同 CAS 对象被误判变更的问题。确定性测试证明原成功对象仍有 2 个链接；改为私有 CAS staging 的原子不覆盖重命名，成功对象不再等待 staging unlink，并 fsync 两个目录。没有覆盖既有对象。

## 实际验证

| 检查 | 结果 |
|---|---|
| Python unit/adapters/agent | 1306 passed、16 skipped、3 deselected；排除项先在未修改 HEAD 归档复现 |
| Rust workspace | 80 passed，含确定性 CAS 生命周期回归 |
| 真实 release ebd UDS | 3 passed：实际 hash/复制/发布、损坏 CAS、inventory/text/hash、Range/HEAD/416/空文件、旧 CAS、tar.zst 和性能 |
| CPU Bundle Job + 隔离 PG | 3 passed：真实工具容器、解压包成员/safety/G3/G4/数值结果、幂等登记、两类真实 PG 回滚 |
| 工具镜像 | ONNX 1.16.1、ORT 1.22.1、NumPy 1.26.4、PyYAML 6.0.2；CPUExecutionProvider probe 通过 |
| 前端 | 90 Node tests passed、vue-tsc 与 Vite build 通过（既有大 chunk 提示仍保留） |
| make test | 实际失败：既有 tests/runtime/test_pbfm_state.py 收集时缺少 torch；Rust 步骤因此未执行，已另行完整验证 |
| git diff --check | 已执行，无空白错误 |

已在 HEAD 归档复现的三个排除项：

- test_pbfm_longrun.py::test_real_long_builder_preserves_native_source_and_expands_only_its_scoped_budget：尝试写只读宿主 Run 目录。
- test_rpo_amp_patch.py::test_amp_patch_applies_after_rpo_patch_and_keeps_python_parseable。
- test_rpo_flat_runtime.py::test_flat_patch_chain：与上一项涉及外部 RLinf 当前源码的既有 patch 链不匹配。

不向 API 镜像安装 Torch 掩盖 runtime 依赖。16 个 skip 保留原条件。未运行浏览器交互式验收、GPU 训练或真机测试；不能从这些检查推导专项 runtime 全部通过。本轮日志保留于 .superpowers/sdd/2026-10-28-python-rust-data-plane-migration/，机器可读测量见[原始 JSON](2026-10-28-python-rust-data-plane-benchmark.json)。

## release 性能记录

环境为隔离容器 **overlayfs**，Rust **release**。FastAPI/uvicorn 测量壳使用真实 EbdClient/UDS，未包含业务 DB/鉴权。RSS 为请求期间 /proc 采样峰值，不是内核精确高水位；读取字节为逻辑量，不是物理块 IO。

advisory-cold 通过 fsync + POSIX_FADV_DONTNEED 建议性驱逐，没有清空全机缓存，不能保证等价真实冷盘。逐文件基线顺序发 scan，不能当作历史并发校验器的精确耗时；每文件 4096 字节。

| 文件数 | 缓存 | 逐文件/批量 RPC | 逐文件 s | 批量 s | 逻辑读取字节（各自） |\n|---|---|---|---|---|---|\n| 1 | advisory-cold | 1/1 | 0.0335 | 0.0095 | 4096 |\n| 1 | warm | 1/1 | 0.0086 | 0.0101 | 4096 |\n| 100 | advisory-cold | 100/1 | 0.2815 | 0.0215 | 409600 |\n| 100 | warm | 100/1 | 0.2538 | 0.0259 | 409600 |\n| 1000 | advisory-cold | 1000/1 | 2.1511 | 0.1413 | 4096000 |\n| 1000 | warm | 1000/1 | 2.0499 | 0.1302 | 4096000 |

legacy-python-copy **不含完整性 hash/fsync**；新发布包含复制 hash、源复读、fsync/不覆盖发布。两者不能用于宣称同等安全语义的固定提速；新发布逻辑读源两遍。

| MiB | 缓存 | 操作 | s | API 峰值 MiB | ebd 峰值 MiB | 健康请求最大 ms | 逻辑读取字节 |\n|---|---|---|---|---|---|---|---|\n| 64 | advisory-cold | batch-verify | 0.1003 | 77.78 | 6.92 | 1.890 | 67108864 |\n| 64 | advisory-cold | legacy-python-copy | 0.2699 | 141.71 | 6.92 | 1.771 | 67108864 |\n| 64 | advisory-cold | publish-file | 0.2978 | 77.78 | 6.92 | 2.784 | 134217728 |\n| 64 | warm | batch-verify | 0.0635 | 77.78 | 6.92 | 4.480 | 67108864 |\n| 64 | warm | legacy-python-copy | 0.1953 | 141.71 | 6.92 | 3.584 | 67108864 |\n| 64 | warm | publish-file | 0.2365 | 77.78 | 6.92 | 3.327 | 134217728 |\n| 512 | advisory-cold | batch-verify | 0.6954 | 77.78 | 7.04 | 1.921 | 536870912 |\n| 512 | advisory-cold | legacy-python-copy | 1.4895 | 589.53 | 7.04 | 3.053 | 536870912 |\n| 512 | advisory-cold | publish-file | 2.0237 | 77.78 | 7.81 | 2.964 | 1073741824 |\n| 512 | warm | batch-verify | 0.4050 | 77.78 | 7.81 | 2.949 | 536870912 |\n| 512 | warm | legacy-python-copy | 1.3760 | 589.53 | 7.95 | 3.064 | 536870912 |\n| 512 | warm | publish-file | 1.7474 | 77.78 | 7.95 | 3.067 | 1073741824 |

本夹具证明批量化减少 RPC，大文件发布的 API/Rust RSS 不随文件体积线性增长，独立健康请求继续响应。真实 live lease 的并行推进另有阻塞 IO/计时器与既有 lease 回归，不把健康请求冒充完整 live/GPU 压测。debug 也测过，未用作生产吞吐结论。

## 保留范围与限制

- 旧同步 POST /bundles/build 返回形式与实现保留；历史内存打包、ORT 缺失时跳过数值检查、压缩失败回退 raw tar 的行为不进入新 Job 流程，不能声称全部 Bundle IO 已迁移。
- 小目录 prepare_run_dirs/overlay_mounts、同步 adapter/checkpoints 旧枚举/小元数据、部分 collector 候选枚举与 bounded JSON/hash 仍为 Python。native 直接异步收集/提示/基线/周期恢复已传 Rust inventory，共享同步 resume_checkpoint/checkpoint_model 仍能本地校验。
- 任务内原生 checkpoint 保存/barrier/loader/Replay、ONNX/HDF5/仿真/训练仍为 Python，保持原协议，不计为下沉成果。
- inventory 游标无持久快照，大型树分页可重复遍历；冻结资格仍由 Python 终态/清单协议判断。100000 项明确失败，未声称超大树都有吞吐提升。
- 不破坏生产 DB/已发布资产；ENOSPC 用受控 writer，不实际填满宿主磁盘。没有新增数据库迁移、服务、索引数据库或遥测接口。

## 实施取舍与成本

1. 复用当前 codex 工作区，不新建/合并分支；改动共用当前 checkout，保持未提交。
2. 保留忽略目录日志/ledger，不随普通 Git 文件保存；维护者结果另存本页/原始 JSON。
3. 复用 Rust 镜像与 Cargo target/cache；缓存不作证据，结果来自新命令。
4. 发布源摘要复读处理同 tick 同长度改写；额外一次顺序读取。
5. 不为 API 安装 Torch；完整 runtime 收集限制单独报告。
6. inventory 增加 recursive/exclude_dirs 有界内部字段；内部协议扩大，公共 API 不变。
7. 只读 hash-file 供 legacy/USD 使用，不扩 Run/CAS 写根；同步及任务内校验保留范围见上。
8. tar 禁用未用 xattr、jobserver 锁 0.1.32 适配 Rust 1.83；不保存宿主扩展属性，成员用显式确定性 header。
9. 工具镜像复用固定版本，移除基础镜像已有的冗余 pip apt 安装，并复用仓库 Aliyun pip 镜像站应对超时；构建依赖镜像站。公开 Cargo 缓存导入 BuildKit，未复制凭据。

## 部署与回退

已在两次切换前核对平台队列与运行任务为空，通过 make up 部署。tools 独立构建并切换已验证的本地标签，不纳入 make up。API/ebd/PostgreSQL/Redis 均 healthy。最终镜像：

| 服务 | 部署镜像 |
|---|---|
| API | sha256:aa262ff7c8885b53e8398e4bb7cc0fe31ad91b24bde9eb1033ec8fc7e9f43aee |
| ebd | sha256:5fb947ebbd674cb262bc43a5789528b45ed66de0956a0c3c49374478642e3481 |
| tools | sha256:1b7ea415881c204f3274b5611273d5f43f770815f0f2aa828e5ffdbdca84e933 |

只读真实入口：health/SPA 200；BHL 预览返回 26 关节/26 网格；既有 URDF GET/HEAD Range 均 206，Content-Type application/xml、Content-Range bytes 0-31/33612；已有数据目录返回 total=5、page_items=2；既有已登记评估 MP4 返回 206/64 字节、video/mp4。没有创建生产验收 Job。

初次部署验收捕获 BHL 预览误触发整仓 inventory 的 413。已补失败测试并改为只查询 URDF 引用的网格目录及相邻 USD 目录，没有提高遍历上限；1306 项控制面回归与真实入口再次验证通过。最终真实 CPU Job 集成使用上述 ebd/tools 镜像及真正 image_info，没有模拟镜像查询。

| 服务 | 原镜像 | 已保留标签 |
|---|---|---|
| API | sha256:d11842ea3f7b8f8f4acd64b385a639b35227c6080c1f5b5bcc05677fc4baf484 | eb/rollback:data-plane-api-20261028 |
| ebd | sha256:02a7e15d33fc8c29a9aa62c88d6883ed84224f374229f708796ffb2f2eef93e0 | eb/rollback:data-plane-ebd-20261028 |
| tools | sha256:15d1db5f84b4935edccb031e63270f423945ecd8c425de1799c77e8edc597c11 | eb/rollback:data-plane-tools-20261028 |

回退先恢复旧 API；它可继续使用新 ebd 旧接口。完整回退按保留镜像重标部署标签、使用原 Compose up -d --no-build，不覆盖发布资产、不依赖 Git。PG schema 未变，无 downgrade。新接口失败不自动退回 Python 绕过校验；未引用 CAS 交由既有清理流程审查。
