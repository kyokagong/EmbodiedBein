FROM python:3.12-slim-bookworm AS tool_source

ARG HHTOOLS_REVISION=1181cc98593dbef20b23f92ebbf9ad38ba6a1e7e
RUN sed -i 's|http://deb.debian.org|https://deb.debian.org|g' /etc/apt/sources.list.d/debian.sources \
    && apt-get -o Acquire::ForceIPv4=true -o Acquire::https::Timeout=20 -o Acquire::Retries=0 update \
    && apt-get -o Acquire::ForceIPv4=true -o Acquire::https::Timeout=20 -o Acquire::Retries=0 install -y --no-install-recommends git libgl1 libegl1 \
    && rm -rf /var/lib/apt/lists/*
# Fetch only tool code/configuration, never bundled third-party motion assets.
RUN git init /opt/hhtools && cd /opt/hhtools \
    && git remote add origin https://github.com/Roboparty/human-humanoid-tools.git \
    && git config core.sparseCheckout true \
    && printf '/hhtools/\n/scripts/\n/configs/robots/\n/pyproject.toml\n/README.md\n/LICENSE\n/NOTICE\n' > .git/info/sparse-checkout \
    && git fetch --depth=1 --filter=blob:none origin "$HHTOOLS_REVISION" \
    && git checkout --detach FETCH_HEAD \
    && test "$(git rev-parse HEAD)" = "$HHTOOLS_REVISION" \
    && test ! -d assets/motions
FROM tool_source
RUN pip install --no-cache-dir --timeout 30 --retries 1 --index-url https://mirrors.aliyun.com/pypi/simple numpy==2.2.6 scipy==1.15.3 mujoco==3.3.5 '/opt/hhtools[robot]' \
    && pip freeze > /opt/runtime-requirements.txt
COPY backend/eb/runners/g1_lafan_bridge.py /opt/eb_runner/g1_lafan_bridge.py
ENV PYTHONPATH=/opt/eb_runner:/opt/hhtools
WORKDIR /opt/hhtools
CMD ["python", "-c", "import hhtools.retarget.robot_to_robot; print('hhtools R2R import ready')"]
