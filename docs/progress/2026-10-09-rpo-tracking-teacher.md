# RLinf 闭环跟踪教师执行证据

> 本文保留历史接入记录。用户最新指令禁止继续教师训练，当前执行[原始参考重审](2026-10-09-bfm-reference-reassessment.md)。此前教师预算不是当前待执行授权。

最终状态见[训练与录像诊断报告](2026-10-09-rpo-tracking-teacher-evaluation.md)：416 PPO round/5,111,808 transitions；25 tensors有限；单卡累计59分24秒。独立验证4/12case完成、1/4通过，余8case未执行，未达到正式行走prior资格，未启动四卡BFM1000。

用户已授权设计17与执行计划：单卡教师探索累计最多一小时；条件合格后自动继续原计划四卡 BFM1000 更新和录像评估。不提交 Git，不使用旧诊断 prior 绕过门禁。

T1：新增独立 native23 547/905 教师 contract/template/profile/task；PPO 复用原 ELU 512/256/128、empirical normalization、KL worker。未来1/5/10帧、root位置/姿态/速度误差与residual动作控制已实现。输入只读并绑定retarget receipt/NPZ/asset hash，教师 checkpoint 归属独立模型族。

验证：缺失教师模块测试 RED→GREEN；源 Run 实际是 `finished`，来源 Job 须 `succeeded`，对应测试 RED→GREEN；容量误走大型VLA默认导致63.5GiB/no_fit，补入小型PPO分类，对应测试 RED→GREEN。相关容器回归 **69 passed**。任务镜像CPU策略测试：547/905维度、采样和重算logprobs相等、初始deterministic residual零、std .1、value有限，全部通过。

当前未证明：Isaac短训、教师稳定性、完整片段独立跟踪资格、合格采集prior与四卡正式训练。T2正在执行，不能根据代码存在或单元测试晋升数据。

首个物理 smoke：Train `trn_e1e618aa3ef44aa0af48` / Job `job_1aa2851578434fa8a574` / Run `run_3a4a77fe5e42499bbcf5`，64env、两轮。首步暴露 venv 通道传入 CPU action 与 CUDA reference 相加的问题，已通过平台取消，未产生更新/checkpoint。动作入口补齐 `.to(self.device)`，属于接入错误而非教师质量失败。正式有限探索镜像同步加入教师专属2900秒训练循环预算退出并保存，预留启动/smoke/独立录像时间。

修复后 smoke：Train `trn_fda71c5fc883470da235` / Job `job_56b1f284bc754aae9227` / Run `run_18aa2835090546a88ffd`，64env×24steps×2，`succeeded`、step2完整包。normalizer count=3072，存在有限PPO loss/KL/reward指标。相关容器回归增至 **75 passed**。独立验证实现包含完整四片段×3新seed（142/143/144）、原跟踪门限、逐例误差/终止原因、实际轨迹与四份录像；尚待真实执行。

smoke模型回读：25 tensors全部finite，actor/critic统计分别547/905维且count3072，最小std=.096266，权重SHA `0dd92aecd74998d0abcd745b2052dfadb967ddd2697a7cfdd13f030d568efe46`。归一化包为独立`eb/rpo-tracking-normalization/v1`。

T3有限训练已提交：Train `trn_61ab30c9a68a4ad39266` / Job `job_5f8348f9c42942be9f2e`，单GPU、512env、24step/round、PPO5epochs、micro768/global3072/four minibatches、最多5000轮，每100轮checkpoint；教师同步循环2900秒正常保存退出。Job hard timeout1h、max_retries0。此处轮数是教师PPO轮数，不能计作BFM1000联合更新。

执行技能要求的独立只读审查：发现并修复两项重要问题，1）原验证三 seed 只影响 reset，startup仍42；教师现各held-out seed独立子进程新建单环境，12case/12录像并在收集端复核，2）非有限误差导致strict JSON丢失失败回执；现不可用数字null、finite/pass false，inactive observation不再进入策略。对应新增用例RED→GREEN，完整相关集 **77 passed**、新增runner lint通过。既有诊断PD回放的三个reset seed共享一个startup42，不能据此宣称三个独立物理配置；其既有“不合格”结论仍成立。

反思1：新增模板必须覆盖API、adapter、容量准入、收集、normalization、镜像和独立评估；已查明两处接入缺口并修复。反思2：参考截断与跌倒必须分开；只有正常参考尾端truncation允许完成，跌倒或提前终止保持失败。反思3：教师三秒片段结果不能直接证明正式prior的多样性或祖先分组独立；仍须执行原设计16的数据资格流程。
