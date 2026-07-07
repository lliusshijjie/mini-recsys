# Mini-RecSys

Mini-RecSys is a lightweight recommendation-system MVP built with a Rust/Axum
backend, a C++ HNSW vector-search layer, ONNX Runtime embeddings, Tantivy text
search, Sled persistence, and a Vite/React frontend.

The current backend implements an explainable Phase 1 recommendation pipeline:

```text
Recall -> Rank -> Rerank -> Explain
```

It is intentionally small and rule-driven. The project avoids large-scale model
training infrastructure while keeping a clear extension point for future
machine-learning ranking.

## Features

- Multi-source recall:
  - semantic recall from the HNSW vector index
  - popular-item recall
  - category-based recall from user interests and item categories
- Rule-based ranking with semantic score, category match, popularity,
  price affinity, and novelty.
- Ranking strategy extension point:
  - default: `fixed_weights`
  - reserved: `machine_learning_reserved`
- Lightweight reranking:
  - seen-item filtering through Bloom filters
  - category diversity in the top results
  - one exploration slot for a relevant non-top-scored item
- Explainable recommendation responses with `source`, `reason`, feature scores,
  and `ranking_strategy`.
- Hybrid search that combines vector search and Tantivy keyword search through
  Reciprocal Rank Fusion.

## Project Structure

```text
src/
  main.rs                 Axum server, route handlers, app state
  embedding.rs            ONNX Runtime embedding inference
  ffi.rs                  Rust bindings for the C++ HNSW layer
  hybrid.rs               RRF fusion for search results
  model.rs                User and item data models
  storage.rs              Sled persistence
  text_search.rs          Tantivy indexing and search
  recommendation/
    mod.rs                Recommendation module exports
    pipeline.rs           Recall, rank, rerank, explain orchestration
    recall.rs             Multi-source candidate generation
    rank.rs               Ranking strategies and scoring
    rerank.rs             Diversity and exploration reranking
    explain.rs            Source and reason labels
    features.rs           Ranking feature helpers
    types.rs              Pipeline input/output types
    tests.rs              Recommendation unit tests
cpp/                       C++17 HNSW/vector-search implementation
frontend/                  Vite/React UI
models/                    Local ONNX model and tokenizer files
data/                      Local Sled and index persistence
assets/                    Static assets
docs/                      Design notes and expansion plans
```

## Requirements

- Rust 1.75+
- A C++17 compiler
- Node.js 18+
- Local model files:
  - `models/all-MiniLM-L6-v2.onnx`
  - `models/tokenizer.json`

The current tokenizer/model setup is treated as English-only for this MVP.
User profile text, search input, item names, and recommendation explanations
should remain English. Search requests containing CJK text are rejected.

## Running Locally

Start the backend:

```bash
cargo run --release
```

Start the frontend:

```bash
cd frontend
npm install
npm run dev
```

The frontend development server proxies user-facing workflows to the backend
API during local development.

## API Overview

- `GET /recommend?uid=<user_id>`: returns explainable recommendations.
- `GET /search?query=<text>`: runs hybrid vector and keyword search.
- `POST /mark_seen`: records seen item IDs in the user's Bloom filter.
- `GET /health`: returns a basic service health response.

Recommendation items include ranking features and explanation fields such as:

```json
{
  "item_id": 1,
  "semantic_score": 0.82,
  "category_score": 1.0,
  "popularity": 0.45,
  "price_affinity": 0.91,
  "novelty": 0.55,
  "final_score": 0.73,
  "ranking_strategy": "fixed_weights",
  "source": "semantic+category",
  "reason": "semantic_match"
}
```

## Configuration

The ranking strategy can be selected with:

```bash
MINI_RECSYS_RANKING_STRATEGY=fixed_weights
MINI_RECSYS_RANKING_STRATEGY=machine_learning_reserved
```

`machine_learning_reserved` currently falls back to the same fixed-weight score.
It exists to keep the ranking strategy boundary explicit without introducing a
training pipeline yet.

## Development Checks

Run these before committing backend changes:

```bash
cargo fmt
cargo fmt --check
cargo clippy --fix --allow-dirty --allow-staged
cargo check
cargo test
```

Run this before committing frontend or API-response changes:

```bash
cd frontend
npm run build
```

Generated files under `frontend/dist/`, local data under `data/`, model files
under `models/`, and dependency directories should not be committed.

## Notes

This repository is currently scoped to a single-service MVP. Phase 2 feedback
loops and Phase 3 Kubernetes deployment work are documented under `docs/` but
are not required for running the current Phase 1 service.
