FROM eb/tools:rpo-g1-r2r-v7
COPY backend/eb/runners/g1_lafan_export.py backend/eb/runners/rpo_g1_r2r_trial.py backend/eb/runners/g1_lafan_pool.py /workspace/eb_runner/
RUN /workspace/isaaclab_extension_template/.venv/bin/python -m py_compile /workspace/eb_runner/g1_lafan_pool.py
