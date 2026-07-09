FROM rust:1.89-bookworm AS builder

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends clang g++ pkg-config ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY Cargo.toml Cargo.lock build.rs ./
COPY src ./src
COPY cpp ./cpp

RUN cargo build --release --locked

FROM debian:bookworm-slim AS runtime

WORKDIR /app

ARG ONNXRUNTIME_VERSION=1.23.2

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl libgomp1 \
    && curl -fsSL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz" \
        | tar -xz -C /opt \
    && ln -s "/opt/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}" /opt/onnxruntime \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --uid 10001 --create-home mini-recsys \
    && mkdir -p /app/data /models \
    && chown -R mini-recsys:mini-recsys /app /models /opt/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}

COPY --from=builder /app/target/release/mini-recsys /usr/local/bin/mini-recsys
COPY assets ./assets

ENV PORT=3000 \
    DATA_DIR=/app/data \
    MODEL_PATH=/models/all-MiniLM-L6-v2.onnx \
    TOKENIZER_PATH=/models/tokenizer.json \
    CORS_ORIGIN=* \
    ORT_DYLIB_PATH=/opt/onnxruntime/lib/libonnxruntime.so \
    LD_LIBRARY_PATH=/opt/onnxruntime/lib

EXPOSE 3000

USER mini-recsys

ENTRYPOINT ["/usr/local/bin/mini-recsys"]
