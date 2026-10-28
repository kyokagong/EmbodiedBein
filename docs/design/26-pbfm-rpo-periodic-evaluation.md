# P-BFM 停止训练后的周期 checkpoint 评估

2026-10-23。用户将本轮目标改为：6万更新附近的最新完整 checkpoint 保存后先停止，再评估并查看视频。本轮选择现有冻结周期 U59968，不新增训练、不修改原100000目标或训练终止语义。训练 Job 保持真实取消状态，未保存更新不算进候选。

复用既有模型登记、策略导出和独立评估生命周期，新增 operator 专用周期 checkpoint 预览/登记入口。仅接受已取消且 ebd 确认退出的 P-BFM 长训 Run；步数由受管 recovery/index.json 选择，客户端不能传路径或改 producer。校验冻结训练 plan、原始镜像、原子 index/receipt/manifest、全 rank 模型/optimizer/Replay 原生 loader 与有限性证明、实际计数和全部 payload hash。大文件 hash 仍由 ebd 执行。运行中、半写、缺分片、篡改、错误来源或错误步数均拒绝。

登记为原 Run 的独立不可变 ModelVersion 和两个 pinned Artifact（推理网络与完整恢复状态），模型卡明确周期 checkpoint、U59968、原100000未完成、实际镜像和源 Run。登记幂等并串行化同一来源与步数，不改变父 Job/Run 的状态、配置、原始文件或已有版本。已停止的 Run 不再执行周期清理；后续恢复必须创建新 Run。训练页 checkpoint 列表显示已登记的周期状态及模型链接。

导出和评估沿用原模型的冻结镜像，通过原 Job/Run/ebd 入口运行。周期模型只在再次校验来源、模型卡/Artifact/index/manifest 一致后允许使用，普通模型仍要求成功 producer。原独立 validation Suite 不变：16个 case，normal/occluded 共32次，20秒 horizon，逐次 RGB/深度视频共64份、trace、深度数组与图表。真实成功/失败如实展示，不降低阈值；本轮评估不是正式 acceptance。核对原片可解码、时长、case/hash 和页面播放，报告与10000基线的同条件指标对比。
