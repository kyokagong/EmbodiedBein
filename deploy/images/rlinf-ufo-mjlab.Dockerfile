FROM eb/rlinf:rpo-bfm-ufo-compliance-v11 AS dependencies
SHELL ["/bin/bash", "-ic"]
# Preserve the proven learner ABI while adding the pinned native simulator.
RUN printf 'torch==2.7.0\nnumpy<2\n' > /tmp/mjlab-constraints.txt && \
    /workspace/isaaclab_extension_template/.venv/bin/python -m ensurepip && \
    /workspace/isaaclab_extension_template/.venv/bin/python -m pip install \
      -c /tmp/mjlab-constraints.txt mjlab==1.4.0 mujoco==3.8.0 mujoco-warp==3.8.0.3 warp-lang==1.12.0 && \
    /workspace/isaaclab_extension_template/.venv/bin/python -m pip install --force-reinstall --no-deps prompt-toolkit==3.0.52

FROM dependencies
RUN --mount=type=bind,source=backend/eb,target=/tmp/eb-src,readonly \
    --mount=type=bind,source=deploy/patches/rlinf,target=/tmp/patches,readonly \
    cp -a /tmp/eb-src/runners/. /workspace/eb_runner/ && \
    cp /tmp/eb-src/rpo_bfm.py /workspace/eb_runner/rpo_bfm.py && \
    PYTHONPATH=/tmp/patches /workspace/isaaclab_extension_template/.venv/bin/python /tmp/patches/patch_rpo_ufo_mjlab.py && \
    /workspace/isaaclab_extension_template/.venv/bin/python -c 'import mjlab, mujoco, torch; assert torch.__version__.startswith("2.7.")'
