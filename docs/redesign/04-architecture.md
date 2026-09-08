# 04 · 架构（本版）

控制面 / 数据面 / 容器化执行 **继续沿用**现行 01 章的分层，本版只砍职责、改存储叙事。

## 1. 还要哪些进程

| 进程 | 本版职责 | 相对现行 |
|---|---|---|
| api（FastAPI + SPA） | 元数据 CRUD、提交训练、读状态 | 路由按五模块拆；去掉评估/部署/Agent 页面依赖 |
| ebd | docker 生命周期、日志、tfevents、镜像探测与 pull、目录扫描 | **主存储不再写 CAS**；扫描结果回元数据 |
| postgres | 全部元数据 | 表语义按 02 章演进 |
| redis 6380 | 可继续占位 | 本版不把 Streams 当验收项 |
| minio | 可继续占位 | 训练主路径不依赖 |
| 任务容器 | 只跑 RLinf（及下载/扫描若走容器） | 不进 compose |

硬约束不变：控制面禁止 `subprocess` 跑训练；只有 ebd 持有 `docker.sock`；不写三个外部仓。

## 2. 提交一次训练（简化时序）

```
浏览器  POST /train
  → api 校验：本体 ready、镜像 digest 存在、路径在允许前缀、GPU 数 ≤ 空闲+可排队
  → 写 TrainJob(queued) + 冻结 spec.json / hydra.yaml
  → 调度器占卡后调用 ebd.create_container(image, mounts, argv, gpus)
  → 容器内：ray + python RLinf，log_path=/workspace/output
  → ebd 采集 stdout 与 tfevents → api/WS 推到详情页
  → 退出码 0 → succeeded；产物仍在 run_path/output，等用户登记为模型
```

不再经过「本体环境 G1–G5 卡口」。路径不存在、镜像 missing 在提交时或调度时失败并写清原因。

## 3. Adapter

RLinf Adapter 仍是唯一知道命令行的地方。输入改为：

```
TrainSpec = {
  template, hydra_overrides,
  image, gpus,
  embodiment_path, rlinf_path,
  data_mounts[], model_mounts[],
  output_path
}
```

输出仍是 argv + mounts + env。禁止把 Hydra 知识散落到 Vue。表单只产 overrides 列表。

RLinf 仓升级时：改模板清单和 Adapter 快照测试，不改领域表。

## 4. 调度

保持整卡、单机、最多 4 并发。本版要能：

- queued 按提交时间（可选优先级，默认 FIFO）
- running 可取消
- 超时杀容器
- 镜像 missing 不要占着卡空转：调度前再查一次 digest

不把抢占、多用户配额当本版验收。

## 5. 前端

Vue 3 继续。Shell 主导航六项：总览、本体、数据、环境、训练、模型。设置放侧栏底部。删除或隐藏：任务、实验、提交任务、队列、评估、排行榜、部署、Agent、Showcase。

训练详情保留 WebSocket 日志/指标。新建训练是本版前端的重点工作量。

## 6. 和现行代码的迁移含义（设计层，不是排期）

| 现行 | 本版 |
|---|---|
| `embodiment_envs` + YAML | 停止作为提交前置；可只读兼容一段时间 |
| `tasks` | 数据并入扫描缓存，表可弃 |
| `experiments` | UI 弃用 |
| `jobs.kind` 多种 | UI 只提交 `train.rlinf` |
| Artifact CAS | 改为 path 登记 |
| `/train/experiments` | 重做成 `/train` 列表 |
| `/system/images` 假数据 | ebd 实探 |

这些是实现阶段的对照，不要求本设计文档写出迁移 SQL。
