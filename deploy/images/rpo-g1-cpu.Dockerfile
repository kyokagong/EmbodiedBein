FROM eb/tools:rpo-g1-compat-probe-v1
# shortcut: pinned CPU R2R subset on existing Python 3.11; use supported 3.12 for the full hhtools application.
RUN /workspace/isaaclab_extension_template/.venv/bin/python -m ensurepip \
    && /workspace/isaaclab_extension_template/.venv/bin/python -m pip install --no-cache-dir --no-deps --timeout 30 --retries 1 yourdfpy==0.0.60 lxml==6.1.3
RUN /workspace/isaaclab_extension_template/.venv/bin/python -c 'import hhtools.retarget.robot_to_robot; import hhtools.retarget.interaction_mesh.pipeline'
