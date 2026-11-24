# MotionDecode 下载与原始数据接入

目标：按用户确认的范围，下载公开仓库全部文件到 `/data/kk/data`，在数据工作台浏览、校验；不启动训练。

依据：`docs/design/03-data-collection.md`。固定来源 `CMRobot/MotionDecode@6380f7420c83b20b6ad50124bc6cb829a7c0b178`。保留 ChingMu 非商业许可；CSV 表头的米、wxyz、29 关节顺序优先于卡片毫米描述。120 Hz 仅为来源声明，文件没有时间戳，不证明真实时间轴。

方案：复用 HF 数据登记、data.convert Job、ebd、现有工作台，增加专用原始 CSV 下载/校验 worker。Job 当前必须绑定本体接口，因此增加仅用于来源描述的 G1 reference-pose 草稿（不发布，不宣称控制器/仿真兼容）。不改生命周期、数据库或既有接口。

- [x] 新增真实 CSV fixture 测试：完整性、36 列、非有限数、四元数、坏表头、路径逃逸；验证失败后实现 worker。
- [x] worker 固定 revision/文件 manifest，限制 400 GiB 和 20 GiB 余量；重试复用已下载文件，完整哈希核对；生成逐 motion 索引、进度和失败证据，training_ready=false。
- [x] 接通 HF 路由、adapter、CPU 容量、tools 镜像与 UI 的原始数据说明、进度、CSV 下载；保留已有工作区改动。
- [x] 聚焦容器测试，构建镜像，make up；通过 API 创建来源草稿和 Dataset，提交持久 Job。
- [ ] 三轮复核设计/许可与字段、失败与重试、浏览与完整证据，记录实际下载/校验结果到 docs/progress。

风险：约 92,567 文件，HF 存储量约 309 GB；实际大小以冻结 manifest 为准。网络下载可能耗时数小时。不发布训练版本，不推导原始动作可直接用于 BHL/RPO。
