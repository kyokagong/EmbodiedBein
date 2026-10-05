# 工作台前端实现与验证（2026-09-09）

本次对照现行 `docs/design/07-workbench-api.md`，在既有页面和 API 上增量实施；没有修改设计文件，没有运行任何 Git 命令，没有写外部仓库。

## 已实现

- `frontend/src/views/TrainNew.vue`：从 `/train/templates` 读取 kind、本体/环境/任务及必需输入元数据，只有模板明确给出本体 ID 才自动选择；显示 SFT、在线 PPO、独立评估语义。LIBERO 模板由后端统一注册，无前端重复模板配置。
- SFT 数据与基座模型按 `requires_dataset` / `requires_base_model` 校验；超参数留空传 null，包含 max_steps、micro_batch_size、global_batch_size，移除原先 1024 环境、1e-3 学习率等表单硬编码默认值。
- 预览展示后端实际解析返回的完整配置、argv、挂载及错误；300ms 防抖、取消旧请求并检查响应代次。修改后旧预览立即失效，预览失败/未完成不能提交。移除后端未实现的自动登记 checkpoint 开关。
- `frontend/src/views/TrainJobDetail.vue`：采用 `/train/{id}` 返回的真实 `run_id`、status 和 run_dir；显示输入 ID 和错误。详情轮询完成后再安排下一次 8 秒刷新，切路由/卸载取消旧请求，旧响应不能污染新作业。取消和登记错误可见。
- 指标保留实际原始标签，SFT 标题说明 loss；不会把 reward 回退为别的指标。每条曲线显示采样数和最新值，没有指标时显示空态。日志/指标/产物各请求错误分别报告。
- `frontend/src/views/Shell.vue`：八个一级入口为总览、机器人、数据、训练、模型、评估、交付、环境。数据/训练子导航暴露现有流水线页，评估子导航暴露套件、对比、失败和排行榜；全部复用既有路由。

## 可复跑检查

使用原有 `deploy/api.Dockerfile` 的 web stage，未在宿主机运行 npm：

```bash
docker build --target web -f deploy/api.Dockerfile -t eb-workbench-web-check .
docker run --rm --network none -v /data/kk/EmbodiedBein/frontend/src:/web/src:ro eb-workbench-web-check npx --no-install vue-tsc --noEmit
docker run --rm --network none -v /data/kk/EmbodiedBein/frontend/tests:/web/tests:ro -v /data/kk/EmbodiedBein/frontend/src:/web/src:ro eb-workbench-web-check node --test tests/train-workbench.test.mjs
```

结果：生产构建通过；Vue TypeScript 检查通过；3 个 Node 原生回归测试通过（空值/必需输入、过期预览、路由切换及指标标签）。Vite 保留现有大 chunk 提醒，没有因此增加依赖或拆包配置。

## 实际边界

本报告仅说明前端检查，不代表 LIBERO GPU 训练、数据/模型资产可用性或机器人现场能力已验收。后端负责模板兼容性、输入路径/版本、真实挂载以及执行准入。前端没有制造样本、指标或成功状态；没有新增采集/评估/交付占位页面。当前仍通过 8 秒轮询读取状态；未新增需要服务端事件游标协议的 WS 重连系统。
