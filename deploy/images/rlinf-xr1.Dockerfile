FROM eb/rlinf:libero-pi05
COPY --from=xr1 /xr1 /opt/xr1
RUN /usr/local/bin/python -m pip --python /workspace/isaaclab_extension_template/.venv/bin/python install \
      transformers==4.57.1 lightning==2.5.3 decord==0.6.0 mmengine==0.10.7 liger-kernel==0.6.5
COPY backend/eb/runners /workspace/eb_runner
ENV PYTHONPATH="/opt/xr1:/opt/eb-openpi:/workspace/eb_runner:/workspace/rlinf"
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
