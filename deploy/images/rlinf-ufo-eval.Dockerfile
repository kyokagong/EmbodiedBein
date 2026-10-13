FROM eb/rlinf:rpo-bfm-ufo-agent-4gpu-v1
COPY backend/eb/runners/rpo_bfm_parallel_eval.py /workspace/eb_runner/rpo_bfm_parallel_eval.py
