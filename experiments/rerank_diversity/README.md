# Rerank Diversity Experiments (MMR / DPP)

Standalone Rust and C++ kernels for diversity-aware reranking. Not wired into
the production `src/recommendation/rerank.rs` path yet; intended as a replacement
candidate for the current category-cap / exploration-slot heuristic.

## Algorithms

| Algorithm | Selection idea | Hot-path complexity |
| --- | --- | --- |
| **MMR** | Greedy: `λ·Rel - (1-λ)·max_sim_to_S`, with incremental `max_sim` updates | `O(N·K)` after `O(N²·D)` similarity build |
| **DPP** | Greedy MAP on `L_ij = q_i S_ij q_j` via incremental Cholesky | `O(N·K²)` after `O(N²)` kernel build |

## Layout

- `mmr.hpp` / `dpp.hpp` / `common.hpp` — C++ kernels
- `rust_rerank.rs` — Rust kernels
- `*_test.*` — unit + fixture checks
- `*_bench.*` — microbenchmarks
- `generate_fixture.ps1` — builds `data/candidates.csv` from `assets/products.json`
- `run.ps1` — generate fixture, test, bench

## Run

From repository root:

```powershell
.\experiments\rerank_diversity\run.ps1
```

## Validation against project data

`generate_fixture.ps1` samples the first 128 rows of `assets/products.json` and
synthesizes:

- a descending rank score (stand-in for `final_score`);
- a 32-d L2-normalized category embedding (same spirit as the demo catalog
  category anchors, not the full ONNX 384-d vectors).

Checks:

1. Toy near-duplicates: second pick should leave the clone and take the orthogonal item.
2. `λ = 1` MMR preserves score order.
3. On the products fixture, MMR/DPP top-10 category coverage is at least as high
   as pure score sort, while the first item remains the top-score item.

## Notes for later production swap

- Production rerank currently receives `RecommendedItem` without embeddings; a
  swap needs either item embeddings from `AppState.items` or a cheap feature
  similarity (category / brand rules).
- Keep K small (≈10) and N at post-rank candidate size (tens–low hundreds).
- Prefer the Rust kernel first inside `/recommend`; use C++ FFI only if endpoint
  profiling still shows a hotspot.
