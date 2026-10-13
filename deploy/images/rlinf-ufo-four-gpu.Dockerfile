FROM eb/rlinf:rpo-bfm-ufo-agent-v1
COPY backend/eb/runners /workspace/eb_runner
COPY backend/tests/runtime/rpo_ufo_tracking_diagnostic.py /workspace/tests/rpo_ufo_tracking_diagnostic.py
COPY deploy/patches/rlinf/patch_rpo_ufo_four_gpu.py /tmp/
RUN /workspace/isaaclab_extension_template/.venv/bin/python /tmp/patch_rpo_ufo_four_gpu.py
RUN CUDA_VISIBLE_DEVICES="" /workspace/isaaclab_extension_template/.venv/bin/python /workspace/tests/bfm_ufo_agent_smoke.py --device cpu --output /opt/ufo-four-gpu-cpu-probe.json
