FROM eb/rlinf:rpo-v2-wbc

ARG RLINF_CONTENT_HASH
ARG RLINF_PATCH_SHA256
COPY deploy/patches/rlinf/patch_rpo_bfm.py /tmp/patch_rpo_bfm.py
RUN python3 -c 'import json,sys; r=json.load(open("/workspace/rlinf/.eb-source-receipt.json")); assert r.get("rlinf_content_hash")==sys.argv[1], "base RLinf source receipt differs from current source"' "$RLINF_CONTENT_HASH"
COPY backend/eb/runners /workspace/eb_runner
COPY backend/eb/rpo_bfm.py /workspace/eb_runner/rpo_bfm.py
RUN python3 /tmp/patch_rpo_bfm.py /workspace/rlinf && \
    PYTHONPATH=/workspace/rlinf:/workspace/eb_runner \
      /workspace/isaaclab_extension_template/.venv/bin/python -m py_compile \
      /workspace/eb_runner/rpo_bfm.py \
      /workspace/eb_runner/rpo_bfm_extension.py \
      /workspace/eb_runner/rpo_bfm_codec.py \
      /workspace/eb_runner/rpo_bfm_motion.py \
      /workspace/eb_runner/rpo_bfm_artifact.py \
      /workspace/eb_runner/bfm_zero_policy.py \
      /workspace/eb_runner/bfm_zero_learning.py \
      /workspace/eb_runner/bfm_zero_worker.py \
      /workspace/rlinf/examples/embodiment/train_embodied_agent.py \
      /workspace/rlinf/rlinf/data/schema/embodied_types.py \
      /workspace/rlinf/rlinf/models/embodiment/mlp_policy/__init__.py && \
    printf '{"schema_version":"eb/rlinf-image-source/v1","rlinf_content_hash":"%s","patch_sha256":"%s"}\n' \
      "$RLINF_CONTENT_HASH" "$RLINF_PATCH_SHA256" > /workspace/rlinf/.eb-source-receipt.json && \
    rm -f /tmp/patch_rpo_bfm.py

ENV EB_RLINF_CONTENT_HASH=${RLINF_CONTENT_HASH}
ENV EB_RLINF_PATCH_SHA256=${RLINF_PATCH_SHA256}
ENV PYTHONPATH=/workspace/isaaclab/source/isaaclab:/workspace/isaaclab/source/isaaclab_tasks:/workspace/rpo-assets/robolab:/workspace/eb_runner:/workspace/rlinf
