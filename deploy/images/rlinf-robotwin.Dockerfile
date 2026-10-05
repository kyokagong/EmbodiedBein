FROM eb/rlinf:libero-pi05

RUN /usr/local/bin/python -m pip --python /workspace/isaaclab_extension_template/.venv/bin/python install --no-deps \
    sapien==3.0.0b1 mplib==0.2.1 && \
    /usr/local/bin/python -m pip --python /workspace/isaaclab_extension_template/.venv/bin/python install \
    transforms3d==0.4.2 'pyglet<2' toppra lxml open3d==0.18.0
RUN SAPIEN=/workspace/isaaclab_extension_template/.venv/lib/python3.11/site-packages/sapien/wrapper/urdf_loader.py && \
    sed -i -E 's/("r")(\))( as)/\1, encoding="utf-8") as/g; s/urdf_file\[:-4\] \+ "srdf"/urdf_file[:-4] + ".srdf"/' "$SAPIEN"

# RLInf is copied and patched at build time; task containers never overlay it.
ARG RLINF_CONTENT_HASH
ARG RLINF_PATCH_SHA256
COPY --from=rlinf . /workspace/rlinf
COPY deploy/images/code_snapshot_hash.py /tmp/code_snapshot_hash.py
COPY deploy/patches/rlinf/robotwin-single-policy-eval.patch /tmp/robotwin-single-policy-eval.patch
COPY deploy/patches/rlinf/test_robotwin_single_policy_eval.py /tmp/test_robotwin_single_policy_eval.py
RUN test "$(python3 /tmp/code_snapshot_hash.py /workspace/rlinf)" = "$RLINF_CONTENT_HASH" && \
    test "sha256:$(sha256sum /tmp/robotwin-single-policy-eval.patch | cut -d' ' -f1)" = "$RLINF_PATCH_SHA256" && \
    git -C /workspace/rlinf apply --check /tmp/robotwin-single-policy-eval.patch && \
    git -C /workspace/rlinf apply /tmp/robotwin-single-policy-eval.patch && \
    PYTHONPATH=/workspace/rlinf /workspace/isaaclab_extension_template/.venv/bin/python \
      /tmp/test_robotwin_single_policy_eval.py && \
    printf '{"schema_version":"eb/rlinf-image-source/v1","rlinf_content_hash":"%s","patch_sha256":"%s"}\n' \
      "$RLINF_CONTENT_HASH" "$RLINF_PATCH_SHA256" > /workspace/rlinf/.eb-source-receipt.json && \
    rm -rf /workspace/rlinf/.git /tmp/code_snapshot_hash.py \
      /tmp/robotwin-single-policy-eval.patch /tmp/test_robotwin_single_policy_eval.py
COPY backend/eb/runners /workspace/eb_runner
COPY backend/eb/domain/robotwin_eval_runtime.py /workspace/eb_runner/eb/domain/robotwin_eval_runtime.py
RUN PYTHONPATH=/workspace/eb_runner /workspace/isaaclab_extension_template/.venv/bin/python \
    -c 'from eb.domain.robotwin_eval_runtime import derive_robotwin_runtime; from robotwin_env import pad_worker_wave; assert derive_robotwin_runtime(1, 4)[0].policy_worker_count == 1; assert pad_worker_wave([{"seed":1}], [0], 4)[1] == [0, None, None, None]'
