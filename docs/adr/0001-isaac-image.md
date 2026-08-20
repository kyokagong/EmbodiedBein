# ADR 0001: Isaac 任务镜像从 isaac-lab-template 派生

- Status: accepted
- Date: 2026-08-21

## 事实

本机构建机上的权威环境是正在运行的容器 `isaac-lab-template`：

| 项 | 值 |
|---|---|
| 容器名 | `isaac-lab-template` |
| 镜像 ID | `sha256:78d68e90393e499465062c8e4757b0b82199fd2ecee974caab370287937d94dc` |
| RepoDigests | 空（本地构建，未推 registry） |
| WorkingDir | `/workspace` |
| 扩展模板路径 | `/workspace/isaaclab_extension_template` |
| Isaac Lab | `ISAACLAB_PATH=/workspace/isaaclab`，资产根 `Isaac/5.1` |
| Isaac Sim | `ISAACSIM_ROOT_PATH=/isaac-sim` |
| EULA | 运行时注入 `ACCEPT_EULA=Y` `OMNI_KIT_ACCEPT_EULA=YES`；容器已有 `OMNI_KIT_ALLOW_ROOT=1` |
| 平台标签 | `eb/isaac-lab:2.3.2`（标签名按设计钉住，digest 才是指纹） |

RSL-RL 随该镜像内的 Isaac Lab 安装提供，不另起栈。

## 决策

- `make images` 必须能 `docker image inspect isaac-lab-template`，否则失败并打印「先有 isaac-lab-template」。
- **禁止**回落到 busybox 或凭空 NGC tag。
- `eb/isaac-lab:2.3.2` Dockerfile 清空 ENTRYPOINT：源镜像是 `runheadless.sh`（Isaac Sim Streaming），权威容器 `isaac-lab-template` 实际用 bash 跑 `python scripts/rsl_rl/train.py`。
- Job 指纹使用 `docker image inspect --format '{{.Id}}'`，不用浮动 tag。
- 镜像 site-packages 只有 `berkeley_humanoid_lite` 的 editable 安装；`berkeley_humanoid_lite_assets` 的 `.pth` 只存在于正在运行的 `isaac-lab-template` 可写层。Adapter 必须注入 `PYTHONPATH` 指向 worktree 的 `source/berkeley_humanoid_lite{,_assets,_lowlevel}`，否则 `train.py` 会 `ModuleNotFoundError`。
