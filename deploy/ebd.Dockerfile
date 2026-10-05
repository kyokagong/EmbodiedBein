# syntax=docker/dockerfile:1
FROM rust:1.83-bookworm AS build
WORKDIR /src
COPY rust/ ./
RUN --mount=type=cache,id=eb-cargo-registry,target=/usr/local/cargo/registry \
    cargo build --release -p ebd -p eb-cli --locked --offline || cargo build --release -p ebd -p eb-cli --locked

FROM debian:bookworm-slim
# Host NVIDIA driver is bind-mounted / injected by nvidia-container-runtime.
# Do not install a driver inside this image; nvidia-smi must see the host GPUs.
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*
COPY --from=docker:27-cli /usr/local/bin/docker /usr/local/bin/docker
COPY --from=build /src/target/release/ebd /usr/local/bin/ebd
COPY --from=build /src/target/release/ebctl /usr/local/bin/ebctl
ENV EB_DATA_ROOT=/data/kk/embodiedbein
CMD ["ebd", "serve"]
