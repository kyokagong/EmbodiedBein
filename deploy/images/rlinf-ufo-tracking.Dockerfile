FROM eb/rlinf:rpo-bfm-ufo-agent-4gpu-v1
COPY backend/tests/runtime/rpo_ufo_tracking_diagnostic.py /workspace/tests/rpo_ufo_tracking_diagnostic.py
COPY backend/eb/runners/bfm_ufo_tracking.py /workspace/eb_runner/bfm_ufo_tracking.py
