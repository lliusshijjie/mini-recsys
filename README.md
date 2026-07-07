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
    A["Frontend\nReact + Vite"]

    subgraph Backend["Backend: Axum + Tokio"]
        B1["GET /recommend"]
        B2["GET /search"]
        B3["POST /mark_seen"]
    end

    subgraph Engines["Search Engines"]
        E["ONNX Runtime\nBERT all-MiniLM-L6-v2\n→ 384D Vector"]
        C["HNSW Index (C++)\nhnswlib · ANN Search"]
        F["Tantivy\nInverted Index · Keyword Search"]
    end

    subgraph Processing["Result Processing"]
        BL["Bloom Filter\nSeen-Item Dedup"]
        SC["Coarse Ranking\nsim×0.7 + popularity×0.3"]
        RRF["RRF Merge\nscore = 1 / (60 + rank)"]
        FB["Popularity Fallback\n补足 Top-5"]
    end

    subgraph Storage["Storage: Sled KV DB"]
        D1["users_tree\nUser + Embedding"]
        D2["items_tree\nItem + Embedding"]
        D3["history_tree\nBloom Filter bytes"]
    end

    A -->|uid| B1
    A -->|query text| B2
    A -->|item_ids| B3

    B1 -->|user embedding| C
    C -->|Top-100 candidates| BL
    D3 -->|get_user_filter| BL
    BL -->|unseen candidates| SC
    SC -->|results < 5| FB
    D2 -->|hot items| FB
    SC --> A
    FB --> A

    B2 --> E
    E -->|384D query vector| C
    B2 -->|keywords| F
    C -->|Top-50 vector hits| RRF
    F -->|Top-50 keyword hits| RRF
    RRF --> A

    B3 -->|add item_id to filter| D3

    D2 & D1 -.->|startup hydration| C
    D2 -.->|startup hydration| F
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
