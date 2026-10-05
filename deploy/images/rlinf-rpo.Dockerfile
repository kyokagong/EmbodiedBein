FROM eb/rlinf:libero-pi05

ARG RLINF_CONTENT_HASH
COPY --from=rlinf . /workspace/rlinf
COPY deploy/images/code_snapshot_hash.py /tmp/code_snapshot_hash.py
RUN apt-get update && apt-get install -y --no-install-recommends libsqlite3-0 && rm -rf /var/lib/apt/lists/*
RUN /usr/local/bin/python -m pip --python /workspace/isaaclab_extension_template/.venv/bin/python install onnx onnxruntime
ARG ISAACSIM_URDF_URL=https://d4i3qtqj3r0z5.cloudfront.net/isaacsim.asset.importer.urdf-2.4.31%2B107.3.3.lx64.r.cp311.zip
ARG ISAACSIM_URDF_SHA256=665cc13d48842ac6d0c203359819c1d5c767a1d920fb8a00179e147ec2fb5201
ARG OMNI_PIP_ARCHIVE_URL=http://d4i3qtqj3r0z5.cloudfront.net/omni.kit.pip_archive-d38fa9ecd1fb6df4.zip
ARG OMNI_PIP_ARCHIVE_SHA256=3a0d0623be2349a2861f566dc8d5bdb5040c11b0977bb255c6d522b4be7bd682
RUN python3 -c 'import sys, urllib.request; urllib.request.urlretrieve(sys.argv[1], "/tmp/isaacsim-urdf.zip")' "$ISAACSIM_URDF_URL" && \
    echo "$ISAACSIM_URDF_SHA256  /tmp/isaacsim-urdf.zip" | sha256sum -c - && \
    rm -rf /workspace/isaaclab_extension_template/.venv/lib/python3.11/site-packages/isaacsim/exts/isaacsim.asset.importer.urdf && \
    mkdir -p /workspace/isaaclab_extension_template/.venv/lib/python3.11/site-packages/isaacsim/exts/isaacsim.asset.importer.urdf && \
    python3 -m zipfile -e /tmp/isaacsim-urdf.zip /workspace/isaaclab_extension_template/.venv/lib/python3.11/site-packages/isaacsim/exts/isaacsim.asset.importer.urdf && \
    grep -Fq 'version = "2.4.31"' /workspace/isaaclab_extension_template/.venv/lib/python3.11/site-packages/isaacsim/exts/isaacsim.asset.importer.urdf/config/extension.toml && \
    rm /tmp/isaacsim-urdf.zip
RUN python3 -c 'import sys, urllib.request; urllib.request.urlretrieve(sys.argv[1], "/tmp/omni-kit-pip-archive.zip")' "$OMNI_PIP_ARCHIVE_URL" && \
    echo "$OMNI_PIP_ARCHIVE_SHA256  /tmp/omni-kit-pip-archive.zip" | sha256sum -c - && \
    mkdir -p /workspace/isaaclab_extension_template/.venv/lib/python3.11/site-packages/isaacsim/extscache/omni.kit.pip_archive-0.0.0+69cbf6ad.lx64.cp311 && \
    python3 -m zipfile -e /tmp/omni-kit-pip-archive.zip /workspace/isaaclab_extension_template/.venv/lib/python3.11/site-packages/isaacsim/extscache/omni.kit.pip_archive-0.0.0+69cbf6ad.lx64.cp311 && \
    rm /tmp/omni-kit-pip-archive.zip
COPY deploy/patches/rlinf/patch_rpo_wbc.py /tmp/patch_rpo_wbc.py
COPY deploy/patches/rlinf/patch_rpo_amp.py /tmp/patch_rpo_amp.py
COPY deploy/patches/rlinf/patch_rpo_flat.py /tmp/patch_rpo_flat.py
COPY backend/eb/runners /workspace/eb_runner
COPY backend/eb/rpo_wbc.py /workspace/eb_runner/rpo_wbc.py
RUN test "$(python3 /tmp/code_snapshot_hash.py /workspace/rlinf)" = "$RLINF_CONTENT_HASH" && \
    python3 /tmp/patch_rpo_wbc.py /workspace/rlinf && \
    python3 /tmp/patch_rpo_amp.py /workspace/rlinf && \
    python3 /tmp/patch_rpo_flat.py /workspace/rlinf && \
    grep -Fq '"isaacsim.asset.importer.urdf" = {version = "2.4.31", exact = true}' \
      /workspace/isaaclab/apps/isaaclab.python.kit \
      /workspace/isaaclab_extension_template/.venv/lib/python3.11/site-packages/isaaclab/apps/isaaclab.python.kit && \
    /workspace/isaaclab_extension_template/.venv/bin/python -c 'import sqlite3' && \
    PYTHONPATH=/workspace/rlinf:/workspace/eb_runner \
      /workspace/isaaclab_extension_template/.venv/bin/python -m py_compile \
      /workspace/eb_runner/rpo_wbc_extension.py \
      /workspace/eb_runner/rpo_amp_extension.py \
      /workspace/eb_runner/rpo_amp_motion.py \
      /workspace/eb_runner/rpo_amp_worker.py \
      /workspace/eb_runner/rpo_flat_extension.py \
      /workspace/eb_runner/rpo_flat_policy.py \
      /workspace/eb_runner/rpo_flat_worker.py \
      /workspace/eb_runner/rpo_flat_artifact.py \
      /workspace/rlinf/rlinf/models/embodiment/mlp_policy/mlp_policy.py && \
    printf '{"schema_version":"eb/rlinf-rpo-source/v1","rlinf_content_hash":"%s"}\n' \
      "$RLINF_CONTENT_HASH" > /workspace/rlinf/.eb-source-receipt.json && \
    rm -rf /workspace/rlinf/.git /tmp/code_snapshot_hash.py /tmp/patch_rpo_wbc.py /tmp/patch_rpo_amp.py /tmp/patch_rpo_flat.py
ENV PYTHONPATH=/workspace/isaaclab/source/isaaclab:/workspace/isaaclab/source/isaaclab_tasks:/workspace/rpo-assets/robolab:/workspace/eb_runner:/workspace/rlinf
ENTRYPOINT []
