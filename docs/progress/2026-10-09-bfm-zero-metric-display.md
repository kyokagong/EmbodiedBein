# BFM-Zero smoke 指标展示排障

日期按 Asia/Shanghai：2026-10-09。依据：[训练设计](../design/04-model-training.md)、[BFM 接入设计](../design/14-bfm-zero-rpo-rlinf-integration.md)。

## 根因与现场

页面：`http://194.0.0.106:31800/train/trn_6eff1218bcf74e67b2d7`；Run：`run_405eeaf7cdf64c11a5f9`。

指标没有丢失。`GET /api/v1/runs/{run_id}/metrics` 返回 33 项实测标量，每项只有一个采样点（logger step 3）。页面沿用接口的字母序，把 17 项 `time/` 指标排在 `train/bfm/` 之前；BFM loss 在很下方，且标题仍显示通用 reward / success。

此 Run 从 checkpoint step 3 恢复到 step 4，实际仅新增一轮：4 个环境各采样 4 步，共 16 条 transition。当前 Run 不包含源 Run 的历史点，所以无法形成趋势曲线。最近一轮 `env/num_trajectories=0`，没有完整 episode 结束，因此没有 episode reward / success 汇总；验证采样关闭，也没有独立评估结果。

| 指标 | 实测原值 |
|---|---|
| FB loss | 229991.625 |
| Critic loss | 59.7772102355957 |
| Aux critic loss | 30.019397735595703 |
| Discriminator loss | 8.550698280334473 |
| Actor loss | -0.014269933104515076 |
| 累计更新次数 | 4 |
| 本轮 transition | 16 |
| Replay 条数 | 64 |

原始证据：Run 目录 `console.log`、resolved config 与 `output/tensorboard/`；指标收集和 API 均正常，无需修改 Rust collector 或训练逻辑。

## 修改

训练详情复用 `EbKpi`，将 BFM 五类 loss、累计更新次数、本轮采样与 replay 数值显示在曲线上方。卡片保留六位有效数字，下方继续显示原始数值。曲线顺序改为 loss → 其他训练指标 → 采样/环境指标 → 耗时；保留全部实测名称和值。补充单点提示、当前 Run 的范围和完整 episode 尚未结束的说明。

缺失值显示“— / 尚无实测值”，不补零、不拼接历史 Run、不生成 reward 或 success。其他训练模板不显示 BFM 卡片。

## 验证与复核

- RED：新增排序与 BFM 最新值检查在旧组件上失败；另发现三个既有 TrainNew 测试仍使用旧表单约定。两个 fixture 补齐当前必填的模型名/训练环境；PPO 测试按已有 API 行为检查显式提交字段（兼容 max_steps 到 max_epochs 的转换位于服务端），未修改 TrainNew 或 API 行为。
- GREEN：隔离 Node 容器运行全部前端测试，`20 passed`；`vue-tsc --noEmit` 通过；Compose API 镜像的 Vite 生产构建通过。构建仍有既有大 chunk 提示。
- 真实 API 数据 + 实际 Vue/EbKpi 模板 SSR 检查通过：八个关键值可见、33 项曲线保留、五类 loss 先于 17 项耗时、单点/episode 说明可见。Chart 的现有单采样点行为测试通过。开发回执：`/tmp/bfm-ui-check/receipt.json`，渲染文本：`/tmp/bfm-ui-check/rendered.html`。Chart 在该 SSR 检查中是占位组件，不将 SSR 当作浏览器 canvas 验证。
- 已通过 Compose 重建并更新 API 服务。镜像 `sha256:57579d62b64edec550fc46e13f54d7825b0689fa49cae20f84e7486afac14686`；HTTP 实际返回 `/assets/TrainJobDetail-CR5DEMSI.js`，包含数值卡片与说明；API/ebd 健康均为 ok。部署后重查 33 项单点指标，FB loss、更新次数、transition 和已结束轨迹数均保持原值。`git diff --check` 通过。
- 第一轮复核收集链：原始日志 → TensorBoard → TSDB API → 页面，未发现指标丢失。
- 第二轮复核恢复语义：step 4 是累计更新计数，当前 Run 只有一个新增日志点；不把源 Run 的数据悄悄拼入。
- 第三轮复核展示边界：空值、零、负 loss、多点数据、路由切换与非 BFM 模板有行为检查；短 smoke 没有步态质量/正式通过率证据。

仅修复展示，未重新训练、未进行 Git commit；未做真实浏览器交互/响应式布局检查或步态质量评估。
