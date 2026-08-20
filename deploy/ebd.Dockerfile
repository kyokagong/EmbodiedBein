# syntax=docker/dockerfile:1
FROM rust:1.83-bookworm AS build
WORKDIR /src
COPY rust/ ./
RUN cargo build --release -p ebd

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*
COPY --from=docker:27-cli /usr/local/bin/docker /usr/local/bin/docker
COPY --from=build /src/target/release/ebd /usr/local/bin/ebd
COPY --from=build /src/target/release/ebctl /usr/local/bin/ebctl
ENV EB_DATA_ROOT=/var/lib/eb
CMD ["ebd", "serve"]
