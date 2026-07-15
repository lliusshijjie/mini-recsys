# Repository Guidelines

## Project Structure & Module Organization

This is a hybrid recommendation-system demo with a Rust backend, C++ vector search, ONNX assets, persisted local data, and a Vite/React frontend.

- `src/`: Rust backend. `main.rs` starts Axum; `embedding.rs`, `text_search.rs`, `hybrid.rs`, `storage.rs`, and `ffi.rs` cover inference, search, fusion, persistence, and C++ bindings.
- `src/recommendation/`: recall, ranking, reranking, feature, and explanation pipeline code.
- `cpp/`: C++17 vector and HNSW code compiled from `build.rs`; exported C ABI declarations belong in `cpp/vector_ops.h`.
- `frontend/src/`: React UI; shared styles live in `frontend/src/index.css`.
- `models/` and `data/`: local ONNX/tokenizer artifacts and persisted indexes/databases; do not commit them.
- `docs/`, `scripts/`, `deploy/`, and `assets/`: notes, operational scripts, deployment assets, and static files.

## Build, Test, and Development Commands

- `cargo run --release`: build and run the backend.
- `cargo check`: fast Rust validation.
- `cargo test`: run Rust unit and integration tests.
- `cargo fmt` and `cargo fmt --check`: format and verify Rust style.
- `cargo clippy --fix --allow-dirty --allow-staged`: fix lint warnings.
- `cd frontend && npm install`: install dependencies.
- `cd frontend && npm run dev`: start the Vite dev server.
- `cd frontend && npm run build`: build the production frontend.

## Coding Style & Naming Conventions

Rust uses edition 2021 and default `rustfmt`. Use `snake_case` for functions, modules, and variables; `PascalCase` for types; and `SCREAMING_SNAKE_CASE` for constants. Use `//!` for module docs and `///` for public APIs.

Keep all `unsafe` and `extern "C"` declarations inside `src/ffi.rs`; expose safe wrappers elsewhere. C++ uses C++17 under `cpp/`. Comments and doc comments in Rust, C++, and `build.rs` must be English.

## Testing Guidelines

Place focused Rust unit tests in `#[cfg(test)] mod tests` near the code. Use `tests/` for integration tests as request/response behavior expands. Run `cargo test` after ranking, storage, embedding, or FFI changes. The frontend has no test framework; add one before adding tests.

## Commit & Pull Request Guidelines

Recent commits are concise and capability-focused, such as `Integrating Tantivy to enable hybrid search` and `add bloom filter`. Prefer short messages that name the changed behavior.

Pull requests should summarize backend/frontend impact, list verification commands, mention model or data prerequisites, and include screenshots for UI changes.

## Security & Configuration Tips

Do not commit `models/`, `data/`, `.env*`, `node_modules/`, or build outputs. Keep large ONNX/tokenizer files local and document required model versions in PR notes when relevant.
