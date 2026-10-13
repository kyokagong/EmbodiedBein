FROM eb/rlinf:rpo-v2-bfm-walking-v10
ARG RLINF_REVISION
ARG RLINF_CONTENT_HASH
COPY deploy/images/code_snapshot_hash.py deploy/images/ufo_source_receipt.py /tmp/
COPY --from=ufo-source / /opt/ufo/
COPY deploy/patches/rlinf/patch_rpo_ufo.py /tmp/patch_rpo_ufo.py
RUN /workspace/isaaclab_extension_template/.venv/bin/python /tmp/ufo_source_receipt.py "$RLINF_REVISION" "$RLINF_CONTENT_HASH"
COPY backend/eb/runners /workspace/eb_runner
COPY backend/tests/runtime/bfm_ufo_agent_smoke.py /workspace/tests/bfm_ufo_agent_smoke.py
COPY backend/tests/runtime/rpo_ufo_environment_smoke.py /workspace/tests/rpo_ufo_environment_smoke.py
ENV PYTHONPATH=/opt/ufo:${PYTHONPATH}
ENV PYTHONDONTWRITEBYTECODE=1
RUN /workspace/isaaclab_extension_template/.venv/bin/python /tmp/patch_rpo_ufo.py /workspace/rlinf
RUN CUDA_VISIBLE_DEVICES="" /workspace/isaaclab_extension_template/.venv/bin/python /workspace/tests/bfm_ufo_agent_smoke.py --device cpu --output /opt/ufo-agent-cpu-probe.json
