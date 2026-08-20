ARG BASE=rlinf-berkeley-openpi:local
FROM ${BASE}
ENV RAY_PORT=6390
WORKDIR /workspace/rlinf
