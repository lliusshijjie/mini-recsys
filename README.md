# Mini-RecSys: AI-Powered Hybrid Recommendation System

A high-performance recommendation system featuring a **Rust** web server, an **ONNX-driven AI** embedding engine, a **Tantivy-powered** text search engine, an **HNSW-powered (C++)** vector search engine, and a **Vite/React** frontend.

## 🌟 Key Features

-   **Hybrid Search**: Combines **Semantic Vector Search** (ONNX + HNSW) and **Keyword Search** (Tantivy) via **RRF (Reciprocal Rank Fusion)** algorithm.
-   **Semantic Search**: Real-time semantic vector search using ONNX Runtime (BERT `all-MiniLM-L6-v2`).
-   **Full-Text Search**: High-performance inverted index search powered by [Tantivy](https://github.com/quickwit-oss/tantivy).
-   **Hybrid Architecture**: Blends Rust's safety, C++'s search performance, and Python-trained models' intelligence.
-   **Full Persistence**: 
    -   **Sled (KV Engine)**: Persists user/item metadata and popularity.
    -   **HNSW & Tantivy**: Both vector and text indices are persisted for sub-second startup response.
-   **Smart Lifecycle**: Automatic index hydration from Sled and graceful index saving on shutdown.

## 🏗️ System Architecture

```mermaid
graph TD
    A["Frontend: React"] -->|Search Query| B["Backend: Axum"]
    B -->|Tokenize| E["ONNX Runtime"]
    E -->|384D Vector| C["HNSW Engine (C++)"]
    B -->|Keyword Match| F["Tantivy Engine (Rust)"]
    C & F -->|RRF Fusion| G["Ranked Results"]
    B <-->|Metadata| D["Sled DB (Rust)"]
    D -.->|Hydrate| C
    D -.->|Hydrate| F
```

## 🚀 Getting Started

### Prerequisites

-   **Rust**: 1.75+
-   **C++ Compiler**: Support for C++17
-   **Node.js**: 18+
-   **Models**: Place `all-MiniLM-L6-v2.onnx` and `tokenizer.json` in `/models`.

### Installation

1.  **Initialize Backend**:
    ```bash
    cargo run --release
    ```
2.  **Initialize Frontend**:
    ```bash
    cd frontend && npm install && npm run dev
    ```

## 📊 Technical Components

-   **AI Embedding (`src/embedding.rs`)**: Uses `ort` crate to run BERT models. Implements Mean Pooling and L2 Normalization.
-   **Keyword Search (`src/text_search.rs`)**: Tantivy-based full-text indexing for precise term matching.
-   **Hybrid Logic (`src/hybrid.rs`)**: Implements Reciprocal Rank Fusion (RRF) to merge multiple search result streams.
-   **C++ Engine (`cpp/`)**: FFI-wrapped HNSW index for high-speed retrieval.
-   **Storage (`src/storage.rs`)**: ACID-compliant metadata storage.

---
**Mini-RecSys** - Intelligent recommendation through systems engineering.
