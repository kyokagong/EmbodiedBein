FROM eb/tools:rpo-g1-cpu-base-v1
COPY --from=eb/tools:rpo-g1-code-v1 /opt/hhtools/LICENSE /opt/hhtools/LICENSE
ADD https://raw.githubusercontent.com/Roboparty/human-humanoid-tools/1181cc98593dbef20b23f92ebbf9ad38ba6a1e7e/NOTICE /opt/hhtools/NOTICE
RUN /workspace/isaaclab_extension_template/.venv/bin/python -m pip install --no-cache-dir --no-deps --timeout 30 --retries 1 numpy==2.2.6
COPY backend/eb/runners/g1_lafan_export.py backend/eb/runners/g1_lafan_bridge.py backend/eb/runners/rpo_g1_r2r_trial.py backend/eb/runners/g1_rpo_calibration.py /workspace/eb_runner/
RUN /workspace/isaaclab_extension_template/.venv/bin/python -m py_compile /workspace/eb_runner/rpo_g1_r2r_trial.py
