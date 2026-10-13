# BFM 四卡共享场景评估执行计划

用户新增要求：四卡同时评估，每卡一个仿真进程、多个独立 env。训练已完成，模型固定 mv_41f640b997cc499f84ea，不继续训练、不训练教师、不提交 Git。

1. 派生不可变的并行 Suite v3；沿用 v2 的100案例、20秒、指令/噪声/地形分布、80/100阈值。明确共享地形网格、每案例独立 startup/reset/command/noise RNG，区别于逐例重建 v2。保留已有结果。
2. 扩展现有 BFM 评估 API/adapter：1或4GPU、每卡1或4env，仍走API→Job/Run→ebd→任务容器。新评估镜像独立版本，不覆盖训练v6。
3. 单进程批量真实物理 step；共享模型/资产/渲染调用。各案例独立轨迹、初态、视频、计数；已终止env冻结并退出指标统计，不自动带入下一案例。每批重新初始化环境及参数。
4. 容器内行为检查：批量模型输入、RNG隔离、一个案例早停而其他继续、种子分片恰好覆盖且无重复、全100分母和证据聚合。
5. 先单卡4env真实诊断及显存探测，再通过平台启动四卡各4env的100案例。当前单卡Run完成或取消后释放GPU；不直接修改活动Job资源。
6. 回读4个worker及全部100案例、验证录像和最终标准结果，更新报告。协议差异明确记录，不冒称v2逐例重建的相同随机轨迹。

每批地形使用首个case的seed生成共享官方GRAVEL网格，每个env使用不同地形单元；70/30是生成概率。初态参数/command RNG/传感噪声按case seed独立管理；存储真实terrain seed/row/column。判据复用原物理指标函数，不复制业务生命周期。

## 执行台账

- [x] 1–3：v3不可变Suite/API/adapter/collector及批量物理runner完成，新镜像v7独立构建。训练v6保留。
- [x] 4：先观测3项RED（模块缺失及N>1合同），补齐实现后8项Torch实际镜像回归通过，包括单例早停、其他env继续、冻结且不重置；20项平台/容量回归通过。共享状态helper原fixture一参数签名失配已修复。
- [x] 5a：单卡4env `job_e1941cb298184cc095c8` / `run_4e69a55001cc453aab14`成功收集`ev_89422b7f381245d8acfd`。1真实GPU UUID、1worker、4初态/轨迹/HD录像，7/6/6/6独立物理步，0/4成功、4fall、0NaN/硬限位。视频17帧全部解码；显存1秒采样峰6901MiB/24564MiB。
- [x] 5b：四卡4env/卡标准完成。原v7 `job_0c6012119ee24d408680`受非零GPU相机限制后取消保留；修复派生v8正式`job_6bd4c6d60651403cb95c`成功。最终预览hash `836b9b9f450ce2cfb8baac2a4e233cc446e4997ceab0d3a68b9898991fc92a0c`，同最终权重/数据/资产/codec，100独立种子、64CPU/96GiB内存/32GiBshm。
- [x] 6：四worker/不同GPU UUID/不同PID、每卡25种子/7批、完整100唯一案例逐项核验；全部轨迹重新计算与平台结果一致；100段真实HD录像360帧全部解码并查看终态。已更新报告、曲线及容量证据。

Ruling：原单卡v2完整100例已完成并保留，不取消/不合并旧部分案例；新v3结果分别报告。Vulkan/CUDA用同worker rank索引并关闭多GPU渲染，不采用可能造成Vulkan与CUDA索引错位的逐进程CUDA mask。

Review：独立最终只读审查核对实际AppLauncher与材料事件源码，尚无P1/P2阻断。P3暂缓：v3 code_trace仅列四个runner，缺少v2的codec/artifact/源环境逐文件hash；实际image digest及资产content snapshot仍固定来源。当前模型均在0.14秒内终止，10秒command重采样与10–15秒推扰没有被真实物理smoke触发，保留验证限制。

Reflection/真实兼容修复：v7四卡rank0已输出12case，但rank1–3相机reset卡于installed usdrt `SelectPrims`只支持cuda:0的Fabric pose view。不是模型失败或显存OOM；可选telemetry transmitter的fork失败也不是此根因。已通过API取消保留Run及部分case，不进入新100分母。固定源源码与独立复核证明Camera init结束前尚无pose查询；新增专属Camera子类只把view._use_fabric设False，走已有USD getter/setter并保留同rank CUDA pose/image buffers及物理/渲染设备。相机设备保持cuda:3、USD flag改变的RED→GREEN及原回归共9项通过；镜像派生v8。先实际验证GPU1–3录像，再完整100例。

Evidence：v8四卡录像诊断`job_049609766ed94326ae1b` / `run_7db63bf9039e4520adf9`成功收集`ev_18b77c2f90414165a0b2`。四个真实rank/物理GPU各1case，实际7/6/6/6步；4fall、0NaN/硬限位、0infra，四段HD录像逐帧解码17帧并查看非零卡终态图。证明USD路径恢复实际GPU1–3相机，模型诊断仍0/4。v8完整100-case四卡各4env已重新preview并提交，hash `836b9b9f450ce2cfb8baac2a4e233cc446e4997ceab0d3a68b9898991fc92a0c`，不沿用旧v7计划hash。

Final evidence：`run_b78c49332c394751adc2`，`ev_0358de6a86dc42c0b781`，Asia/Shanghai 2026-10-10 01:35:13→01:43:22，8分9秒。100/100实际完成、0infra、0NaN，0/100通过，66fall/36硬限位（可重叠）；速度向量RMSE0.8305478707502073、yaw3.924208344666101，平均物理时长0.094s/最大0.22s，qualification=fail。470步/10810动作值中97.206% abs(action)≥.99，硬限位类型15关节位置、25关节速度、0力矩（4重叠）。已实际核对source权重/config/motion hash、train-only参考、z norm16、每步动作边界、机械阈值/跌倒及100不同关节初态；独立重算完全一致。100HD视频360帧全部解码。显存1s采样峰0/1/2/3为9761/9634/9857/9762MiB，主存峰57.49GiB/96；已停止本任务只读容量监控。

Completion：9项Torch相机/批量/RNG/早停检查、22项平台/容量/旧Flat回归通过，真实四卡完整验证完成。P3清单仍为source逐文件code_trace缩减和实际长程command/interval未触发；来源由immutable image digest+asset snapshot绑定，不升级为20秒控制或真机验证。模型质量失败已经写入progress，计划完成不意味着模型验收通过。
