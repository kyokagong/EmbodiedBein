FROM eb/rlinf:libero-pi05
COPY --from=gr00t / /opt/isaac-gr00t
RUN /usr/local/bin/python -m pip --python /workspace/isaaclab_extension_template/.venv/bin/python install \
      numpy==1.26.4 packaging==24.2 fsspec==2024.6.1 networkx==3.3 \
      datasets==3.6.0 diffusers==0.35.1 av==16.1.0 albumentations==1.4.18 \
      pyzmq==27.0.1 pyopengl==3.1.10 transformers==4.57.3 torchcodec==0.4.0 \
      peft==0.17.1 lmdb==1.7.5 dm-tree tyro==0.9.17 msgpack-numpy==0.4.8
COPY backend/eb/runners /workspace/eb_runner
ENV PYTHONPATH="/opt/isaac-gr00t:/opt/eb-openpi:/workspace/eb_runner:/workspace/rlinf"
ENTRYPOINT []
