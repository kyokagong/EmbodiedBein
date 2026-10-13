# UFO 框架与RLinf接入分析进度

2026-10-10（Asia/Shanghai）。本次按用户要求分析并写设计，不改产品代码、不部署、不启动训练、不提交Git。

设计输出：[20 · UFO/BFM/RLinf](../design/20-ufo-bfm-rlinf-integration.md)、[21 · 多框架运行契约](../design/21-training-runtime-contracts.md)。

## 实际调查

- 官方GitHub API查询main，固定UFO revision `0b89378de02f77712072d983bc43ec2a46455dc0`；与先前G1资产来源revision相同，但先前并未接UFO训练。
- 容器内只读取源码/文档，取149个文本文件共1,136,137bytes，重点18个文件保留SHA256；未导入或执行UFO模块，未下载额外动作库。[源码审计回执](artifacts/2026-10-10-ufo-design/source-audit.json)
- 实际读取现有 `eb/rlinf:rpo-v2-bfm-walking-v10` 镜像解释器/依赖：Python3.11.16、Torch2.7.0+cu128、Pydantic2.11.5、Gymnasium1.2.0。UFO声明Python3.10，兼容安装未验证；Torch系列并非已知冲突。
- 对照本地worker/policy/loss/codec/replay、API/template/adapter、checkpoint/collector、环境目录、TrainJob ORM与pipeline源码；保留已有工作区所有改动。

## 核实结果

UFO `fb` preset构建完整FBcprAuxAgent，具有六优化器、目标网络、连续专家z与参考状态reset。它采用MJLab，主分支的非G1路径仍实验性；没有自动G1→RPO映射。

当前接入已修复mixed-z池遗漏，不能再报告池仍为空；仍存在与UFO的显式初始化、LR/FB pessimism/relabel/norm/clip、warmup、replay/采样、连续提示、RSI和aux差异。当前RPO历史按时间帧排列，UFO按特征组且最新在前；同维度权重不直接兼容。现有观测视图去掉全局平移/heading，不能唯一逆推native motion library所需RobotState。

推荐RLinf worker调用固定UFO agent，先复用RPO Isaac，再接MJLab；不把UFO CLI子进程启动冒充RLinf算法接入。平台已有统一Job/Run/adapter基础，只需围绕recipe执行绑定、native产物/恢复和能力元组补边界，不推倒架构。

## 三轮设计复核

1. **算法、动作、数据**：发现RSI与history顺序/时间方向差异；新增raw状态版本前置条件；动作rescale/PD与G1 actuator不能照搬。验收保持现有门槛，无教师，无新增来源授权推断。
2. **运行、资源、恢复**：明确UFO六优化器所有权；RLinf与native不能重复梯度同步；区分local FB负样本目标与global batch。估算native5.12M RPO replay浮点下界约20.89GB，非实测；旧GPU证据不准入新配置。native保存缺RNG/z池等状态及原子性，需平台wrapper补齐。
3. **证据、依赖、顺序**：修正P2 reference reset的数据先决条件，raw train状态在P0准备；native MJLab转换在P3。区分native/tracking诊断与独立速度验收、同/跨runtime恢复；所有未运行能力标为设计目标。

## 验证与未验证

容器静态审查通过：两份设计13个本地/固定revision链接、18个重点源码hash及其中Python语法；Markdown fence闭合、无TODO/TBD、`git diff --check`通过。没有因文档变更重跑训练或后端全量测试。

尚未验证：UFO在现有镜像的实际import/update，UFOActorWorker/replay/env桥、原生RPO MJLab、native完整恢复、多learner、训练资源峰值、正常行走与真机。此前0/16和0/100结果保持原结论；本次没有新的模型质量结果。
