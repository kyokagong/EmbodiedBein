# 官方 RPO 参考重审与 UFO 重映射执行计划

> **For agentic workers:** 按顺序执行并记录证据；本计划是有限数据审查，不创建或训练教师，不提交Git。

**Goal:** 判断官方RPO参考作为BFM状态先验的实际条件，并给出UFO G1 LAFAN→RPO的无教师可行路径。

**Architecture:** 复用已冻结官方资产、修正后的BFM motion view和现有容器；离线只读审查产生报告，不修改原数据。UFO调查停在格式与重映射工具边界；实际G1映射须另有固定输入与运行证据。

**Tech Stack:** 当前BFM任务镜像中的NumPy/SciPy/MuJoCo；官方UFO/hhtools公开源码；Markdown。

**Spec:** `docs/design/18-bfm-reference-prior-and-g1-rpo-retarget.md`。

## Global Constraints

- 本阶段禁止训练教师，也不使用教师生成轨迹替代动捕参考。
- 最终训练仍为平台 API→Job/Run→ebd→RLinf；数据下载/转换/训练通过受管Job，不以审查脚本绕过。
- 本次CPU只读审查在容器内，源与已发布数据只读；报告写入当前仓库。
- 保留已有工作区与历史Run/数据，不提交Git；本阶段不晋升 `training_ready`。

## Review Focus

- 相同原始运动的rpo_gmr/rpo_lab和_M变体不增加独立来源数。
- 验证原始与转换后hash；先固定可信字节再读pickle，不碰外部目录缓存。
- PD范围、硬关节限位、姿态任务范围、数值异常分别统计。
- BM数组有body状态但缺body顺序时保留unknown，不能按名字猜语义。
- UFO原始聚合PKL不等于hhtools单片段PKL；不下载实际G1字节时不得声称完成映射。

## T1 · 改正设计与教师分支

- [x] 查论文/官方入口，确认observation-only原始参考及10秒切片，核对现有门槛与调用者。
- [x] 更新设计14/16引用、标记设计17及教师计划不再执行；保留历史结果。
- [x] 核查平台正在运行的任务；不发起教师、正式训练或数据晋升。

## T2 · 官方RPO只读数据审查

- [x] 临时审查脚本绑定源inventory及新view manifest hash；用assert核对source hash、数组shape/finite/quaternion和时间轴。
- [x] 在离线限CPU/内存容器回读30片段，分别统计当前mask、硬限位、速度和直立任务条件；抽帧对照冻结MJCF的FK，检查观测/速度一致性。
- [x] 固定并检查BM NPZ结构/有限值/频率；读取GMR与Lab关节名映射，比较对应原始姿态，识别变体祖先。
- [x] 回读现有MotionView连续采样；输出JSON及逐片段报告，不改旧mask、manifest或平台状态。

## T3 · UFO/hhtools接口核对及报告

- [x] 核对UFO新机器人文档、MotionLib字段、数据下载入口及预期hash；核对hhtoolsR2R加载/校准/输入schema。
- [x] 记录可复用工具、必要桥接、源/目标固定资产、Python和IK依赖、缺少实际G1数据/校准的限制。
- [x] 保存 `docs/progress/2026-10-09-bfm-reference-reassessment.md` 与审查JSON，更新设计/进度索引。
- [x] 三轮复核：论文方法一致性；数据/祖先/坐标证据；实施边界与未验证项；执行 `git diff --check`。

Ruling: 用户已明确同意重审直接参考并禁止教师，按本计划继续，不重复请求相同授权。现有教师预算提问不再适用于当前路线。
Ruling: 不在本阶段偷偷改变生产validation门槛。当前30片段DatasetVersion保持原状态；新规则的代码实施、派生数据、G1实际映射和四卡训练由本次证据决定下一阶段顺序。

Ruling: CPU复核发现MJCF与训练URDF硬限位不同，最终按URDF统计9276静态候选/9205连续有效帧；初次MJCF9430数不作为正式候选结论。26对GMR/Lab按官方elbow_yaw清零后匹配，BM身体速度需重建；实际G1映射未执行。

完成证据：报告与JSON已保存，Lab360帧/BM1045帧静态FK检查、原mask及内存候选mask连续采样断言通过；所有原始与已发布数据保持不变。生产新准入规则及正式四卡训练仍在下一阶段，不把本计划完成当作这些能力完成。
