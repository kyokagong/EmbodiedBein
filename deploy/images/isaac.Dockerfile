# 构建机必须已有 isaac-lab-template 同源镜像。禁止 busybox 回落。
ARG BASE=isaac-lab-template
FROM ${BASE}
ENV ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES
WORKDIR /workspace/isaaclab_extension_template
