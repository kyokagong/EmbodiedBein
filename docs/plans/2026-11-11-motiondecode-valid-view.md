# MotionDecode 有效版本与表格/动作视频

用户授权：保持所有原始数据不变，排除空 Summer_Wind.csv，让有效数据校验通过并可浏览表格、动作视频；不训练。

依据 docs/design/03-data-collection.md：派生不可变成员清单，保留来源/许可/排除证据；计算经过 Job/Run/ebd；原始根只读。

实现边界：
- derive_motiondecode_view：复用冻结的原逐文件校验 receipt，重新核对全部原始文件 hash。仅允许固定路径+970 bytes+已知 SHA256 的空文件排除；其他异常仍失败。新 Dataset / DatasetVersion 引用源 CSV，不复制147.9GB、不修改原始文件。
- motiondecode_preview：现有 UFO G1 MJCF/meshes经ebd内容快照冻结；按关节名映射 29关节，根位置m、四元数wxyz，mj_forward运动学回放；无动力学/训练验收声明。使用已有容器依赖、OSMesa CPU与imageio编码，网络none。
- 视频按需生成缓存，30fps，每次最多60秒（7200原帧），可分段遍历；时间基准明确标为来源声明120Hz。表格每页100帧、36原始列，worker生成分页JSON，API经ebd读取，浏览器不加载完整CSV。
- 固定 API：POST /datasets/{parent}/motiondecode/valid-view；GET episodes 增加q；POST /datasets/{view}/motiondecode/previews {path,start_frame}；GET同路径/{job_id} 返回状态、preview元数据；/files/{path} 提供经校验的receipt、分页表格与MP4。

步骤：
- [x] worker/排除规则/边界测试先失败后实现；实际原始目录只读哈希复核。
- [x] API/adapter/CPU容量/镜像/DatasetVersion/lineage接入。
- [x] 前端子任务独立实现搜索、表格分页、视频及错误/轮询/失效响应测试。
- [x] 容器测试与构建；make up部署；真实派生Job通过、预览Job/视频解码/表格数值与HTTP Range验证。
- [x] 独立审查与三轮证据复核，记录实际未验证项。

不自动将 raw/preview_ready 升格为 training_ready；不会删除、补造或重写空源文件，不预渲染92,558份视频。
