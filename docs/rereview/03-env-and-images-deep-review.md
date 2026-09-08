# 03 · 环境与镜像深度详规与架构设计

环境模块负责管理用于物理仿真的 **Isaac Lab 运行时**，以及宿主机上的 **Docker 训练镜像** 与 **GPU / 驱动兼容性诊断**。

---

## 1. 用户需求与设计原则

- **专一聚焦 Isaac Lab**：当前阶段仿真环境唯一绑定 Isaac Lab，不再抽象冗余的通用仿真器插件层，降低概念复杂度。
- **真实 Docker 镜像纳管**：由 `ebd` 守护进程（持有 `docker.sock`）实探宿主机 Docker 引擎，真实管理镜像的 Pull、Digest、大小与就绪状态。
- **环境诊断 (Environment Doctor)**：由于 Isaac Lab 强依赖 NVIDIA Driver、CUDA、EGL 无头渲染与 Vulkan ICD，环境页提供一键硬件兼容性自检，避免训练盲目启动后秒退。
- **镜像 Digest 锁定**：提交训练任务时固化 Docker Image Digest（`sha256:...`），杜绝因 `latest` 标签漂移导致的实验不可复现问题。

---

## 2. 仿真栈与镜像拓扑关系

```
┌─────────────────────────────────────────────────────────────┐
│                      环境模块 (Environment)                   │
│                                                             │
│  ┌─────────────────────────┐   ┌─────────────────────────┐  │
│  │ 仿真运行时: Isaac Lab   │   │ GPU & 驱动诊断 (Doctor) │  │
│  │ · 官方版本: 2.3.2 / 4.x │   │ · Driver: 550.90.07 (OK)│  │
│  │ · 默认训练镜像绑定      │   │ · CUDA: 12.4 (OK)       │  │
│  │ · Nucleus 共享资产缓存  │   │ · Vulkan / EGL: 就绪    │  │
│  └────────────┬────────────┘   └─────────────────────────┘  │
│               │                                             │
│               ▼ 调度使用                                    │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ 宿主机 Docker 镜像清单 (由 ebd 实时探测)               │  │
│  │ 1. eb/rlinf:0.4.0-emb   [sha256:a1b2...] Ready (默认) │  │
│  │ 2. eb/isaac-lab:2.3.2   [sha256:c3d4...] Ready (纯仿) │  │
│  │ 3. eb/rlinf:dev-nightly [--] Missing (可一键拉取)     │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. GPU / Vulkan / EGL 硬件与驱动兼容性诊断 (Doctor)

Isaac Lab 容器在无物理显示器的服务器（Headless Server）上运行时，对底层图形栈有极其严苛的要求。为保障训练可靠性，环境模块内置 **Environment Doctor**：

### 3.1 诊断检查项与标准
| 检查项 | 检测命令 / 途径 | 达标标准 | 不达标影响 |
|---|---|---|---|
| **NVIDIA Driver** | `nvidia-smi` API 查询 | Driver $\ge 535.54.03$ (推荐 550+) | Isaac Sim 无法启动，提示驱动过旧 |
| **CUDA Toolkit** | NVML / CUDA Runtime API | CUDA $\ge 12.1$ | PyTorch / RLinf 算子编译崩溃 |
| **NVIDIA Container Toolkit** | `docker info` Runtime 查询 | 包含 `nvidia` 运行时 | 容器内无法挂载 GPU 设备 |
| **Vulkan ICD / EGL** | 检查宿主机 `/usr/share/vulkan/icd.d/nvidia_icd.json` | 存在合法的 NVIDIA EGL/Vulkan ICD | 仿真器渲染相机图像时抛出 `VK_ERROR_INCOMPATIBLE_DRIVER` |
| **共享内存 /dev/shm** | 宿主机 `/dev/shm` 挂载大小 | 可用空间 $\ge 32\text{ GB}$ | 多并行环境进程间通信 OOM 崩溃 |

页面右上角提供 **「运行环境体检」** 按钮，3 秒内完成全面探测并输出绿色/黄色/红色诊断报告。

---

## 4. 镜像生命周期管理与实时拉取

### 4.1 镜像真实探测
后端 API 通过 UDS 向 `ebd` 发送指令，`ebd` 调用 Docker Engine API：
- `GET /images/json`：获取宿主机已有镜像的 Repository、Tag、Id (Digest)、Created、Size。
- 对比平台预置的训练镜像清单，精确标记镜像状态：
  - `ready`：宿主机已存在，Digest 完整。
  - `missing`：宿主机尚未拉取。
  - `pulling`：正在后台异步拉取。

### 4.2 镜像 Pull 任务与实时进度流
当用户在前端点击「拉取镜像」时：
1. `ebd` 发起 `POST /images/create?fromImage=eb/rlinf:0.4.0-emb`。
2. 解析 Docker Daemon 返回的 JSON Stream，提取分层下载状态：
   ```json
   {"status": "Downloading", "progressDetail": {"current": 450123440, "total": 1200000000}, "id": "layer_3a1b"}
   ```
3. 计算总体下载百分比与速度，通过 WebSocket / 轮询推送至环境页，表格行内实时呈现动态进度条。
4. 提供「取消拉取」操作，支持终止 Docker pull 任务。

---

## 5. Isaac Sim / Nucleus 资产缓存优化

Isaac Lab 运行时会高频加载官方机器人资产与环境 USD。若每次容器冷启动都重复从网络下载，会导致初始化耗时高达数分钟且容易因网络超时失败。

**平台级挂载规范**：
- 宿主机创建跨任务共享缓存目录：`/var/lib/eb/cache/ov_data`
- 在 `RlinfAdapter` 渲染容器启动参数时，固定添加只读/缓存挂载：
  ```
  -v /var/lib/eb/cache/ov_data:/root/.local/share/ov/data:rw
  -v /var/lib/eb/cache/isaac_nucleus:/root/Documents/Isaac:rw
  ```
大幅提升训练任务的容器启动速度（从 180s 降低至 5s）。

---

## 6. 元数据模型 (PostgreSQL Schema)

```sql
CREATE TABLE sim_images (
    id VARCHAR(64) PRIMARY KEY,                  -- 格式：img_xxx
    name VARCHAR(256) NOT NULL,                  -- 完整镜像名，如 "eb/rlinf:0.4.0-emb"
    tag VARCHAR(64) NOT NULL DEFAULT 'latest',   -- 镜像标签
    digest VARCHAR(128),                         -- sha256 镜像摘要
    size_bytes BIGINT DEFAULT 0,                 -- 镜像体积大小
    role VARCHAR(32) NOT NULL DEFAULT 'train',   -- train (主训练) | sim (纯仿真) | tools (工具)
    is_default BOOLEAN DEFAULT FALSE,            -- 是否为 Isaac Lab 默认训练镜像
    status VARCHAR(32) NOT NULL DEFAULT 'ready', -- ready | missing | pulling | error
    pull_progress JSONB DEFAULT '{}',            -- {layers, total_bytes, current_bytes, speed}
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    CONSTRAINT uq_image_name_tag UNIQUE (name, tag)
);
```

---

## 7. 页面交互规范

### 7.1 环境总览与仿真栈卡片
- **顶部卡片**：展示「Isaac Lab 仿真环境」，显示默认绑定镜像、绑定的资产缓存路径，以及「环境 Doctor 状态：健康（全部 5 项通过）」。

### 7.2 镜像管理表格
- **表格列**：
  - 镜像名称及 Tag（如 `eb/rlinf:0.4.0-emb`）
  - Digest 摘要（前 12 位短码，支持一键复制完整 SHA）
  - 体积大小（如 `18.4 GB`）
  - 角色（`Train 默认` 绿色徽章 / `Sim` / `Tools`）
  - 状态（`Ready` 绿色 / `Missing` 橙色 / `Pulling 72% 18MB/s` 动态进度）
  - 操作栏：
    - 对 `ready` 镜像：提供「设为默认」、「重新拉取」按钮。
    - 对 `missing` 镜像：提供高亮「拉取镜像」按钮。
- **右上角操作**：提供「刷新 Docker 镜像」、「手动注册镜像名」按钮。
