# Repository Guidelines

## Project Structure & Module Organization

This repository is a hybrid recommendation-system demo with a Rust backend, C++ vector-search code, ONNX models, persisted data, and a Vite/React frontend.

- `src/`: Rust backend modules. Key files include `main.rs` for the Axum server, `embedding.rs` for ONNX inference, `text_search.rs` for Tantivy search, `hybrid.rs` for RRF fusion, `storage.rs` for Sled persistence, and `ffi.rs` for C++ bindings.
- `cpp/`: C++17 HNSW/vector operations compiled by `build.rs`.
- `frontend/`: React UI powered by Vite. Main app code is under `frontend/src/`.
- `models/`: local model artifacts such as `all-MiniLM-L6-v2.onnx` and `tokenizer.json`; these are ignored by git.
- `data/`: local persisted database and indexes; also ignored by git.
- `assets/`: static assets.

## Build, Test, and Development Commands

- `cargo run --release`: builds and runs the backend with C++ FFI and ONNX Runtime.
- `cargo check`: quickly validates Rust code without producing a release binary.
- `cargo test`: runs Rust unit and integration tests when present.
- `cargo fmt`: formats Rust code before committing.
- `cd frontend && npm install`: installs frontend dependencies.
- `cd frontend && npm run dev`: starts the Vite development server.
- `cd frontend && npm run build`: creates the production frontend bundle.

## Coding Style & Naming Conventions

### Tooling (run before committing)

```powershell
cargo fmt
cargo fmt --check
cargo clippy --fix --allow-dirty --allow-staged
cargo check
cargo test
```

- **`cargo fmt`**: default rustfmt rules; no project-local `rustfmt.toml` — do not hand-format.
- **`cargo clippy`**: fix warnings in changed code; do not suppress lints without a documented reason.
- **`cargo check` / `cargo test`**: required after changes to ranking, storage, embedding, or FFI.

### Comments

- **English only** for all comments and doc comments in Rust (`//`, `///`, `//!`), C++ (`//`, `///`), and `build.rs`.
- Use `//!` module-level docs at the top of each Rust source file.
- Use `///` for public API documentation; inline `//` for non-obvious logic only — avoid stating the obvious.
- Demo/user-facing strings (e.g. sample user names) may stay in any language; comments must not.

### Rust

- Edition **2021**; follow standard rustfmt output (4-space indent, trailing commas where fmt adds them).
- **Naming**: `snake_case` for functions, modules, variables; `PascalCase` for types/structs/enums; `SCREAMING_SNAKE_CASE` for constants.
- **Modules**: one concern per file under `src/`; match existing boundaries (`embedding`, `ffi`, `hybrid`, `model`, `recommendation`, `storage`, `text_search`).
- **Section headers** in large files (e.g. `main.rs`):

  ```rust
  // ============================================================================
  // Section Name
  // ============================================================================
  ```

- **FFI boundary**: all `unsafe` and `extern "C"` declarations live in `src/ffi.rs`; expose safe wrappers to the rest of the codebase. Business logic must not call `unsafe` directly.
- **Errors**: prefer `anyhow::Result` at boundaries; use `.context(...)` for error messages.
- **Tests**: unit tests in `#[cfg(test)] mod tests` at the bottom of the same file; integration tests in `tests/` when needed.

### C++

- **Standard**: C++17 (set in `build.rs`).
- **Location**: implementation under `cpp/`; headers in `cpp/vector_ops.h`.
- **Comments**: English only; use `///` for exported C API docs in the header.
- **Linkage**: `extern "C"` exports consumed by `src/ffi.rs`; keep the C ABI stable.

### Frontend (React / Vite)

- Components: `PascalCase` exports (e.g. `App.jsx`).
- Shared styles: `frontend/src/index.css`.
- API base URL and constants at the top of the file; use existing `axios` patterns for HTTP calls.

### General

- Minimize scope: match surrounding style; do not refactor unrelated code in the same change.
- Keep FFI declarations in `src/ffi.rs` and C++ details under `cpp/`.

## Testing Guidelines

Place Rust unit tests near the code under `#[cfg(test)]` modules, and use `tests/` for integration tests if request/response behavior grows. Run `cargo test` before changing ranking, storage, embedding, or FFI code. For frontend changes, add a test framework before introducing test files; currently only Vite build scripts are defined.

## Commit & Pull Request Guidelines

Recent history uses short imperative or descriptive commits, for example `Integrating Tantivy to enable hybrid search` and `add bloom filter`. Prefer concise messages that name the changed capability. Pull requests should describe backend/frontend impact, list verification commands, mention model or data prerequisites, and include screenshots for UI changes.

## Security & Configuration Tips

Do not commit `models/`, `data/`, `.env*`, `node_modules/`, or build outputs. Keep large ONNX/tokenizer files local and document any required model version in the PR description.
