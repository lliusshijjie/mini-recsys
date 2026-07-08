FROM rust:1.82-bookworm AS builder

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends clang g++ pkg-config ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY Cargo.toml build.rs ./
COPY src ./src
COPY cpp ./cpp

RUN cargo build --release

FROM debian:bookworm-slim AS runtime

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --uid 10001 --create-home mini-recsys \
    && mkdir -p /app/data /models \
    && chown -R mini-recsys:mini-recsys /app /models

COPY --from=builder /app/target/release/mini-recsys /usr/local/bin/mini-recsys
COPY assets ./assets

ENV PORT=3000 \
    DATA_DIR=/app/data \
    MODEL_PATH=/models/all-MiniLM-L6-v2.onnx \
    TOKENIZER_PATH=/models/tokenizer.json \
    CORS_ORIGIN=*

EXPOSE 3000

USER mini-recsys

ENTRYPOINT ["/usr/local/bin/mini-recsys"]
