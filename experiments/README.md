# Low-Level Recommendation Kernel Experiments

This directory contains standalone feasibility experiments for optimizing hot
candidate operations in the recommendation recall path. The code is intentionally
not wired into the production Rust crate, `build.rs`, or the C++ HNSW index.

## Experiments

### `recall_kernels/`

Compare Rust and C++ implementations of:

- seen filtering, duplicate merging, source-mask merging, and deterministic
  top-k selection;
- 64-byte aligned reusable workspaces for hot arrays;
- scalar versus AVX SIMD dot products for contiguous `f32` vectors.

```powershell
.\experiments\recall_kernels\run.ps1
```

### `rerank_diversity/`

MMR and greedy-MAP DPP diversity rerankers in Rust and C++, validated against a
fixture derived from `assets/products.json`. Intended as a future replacement
for the production category-cap rerank heuristic.

```powershell
.\experiments\rerank_diversity\run.ps1
```

## Latest Local Results

These results were collected on this Windows development machine with the
experiment script. They are local feasibility data, not `/recommend` endpoint
latency.

| Language | Case | Baseline avg_us | Optimized avg_us | Speedup |
| --- | ---: | ---: | ---: | ---: |
| Rust | candidate small | 81.182 | 5.989 | 13.6x |
| Rust | candidate medium | 1049.108 | 169.238 | 6.2x |
| Rust | candidate large | 12701.962 | 2044.198 | 6.2x |
| C++ | candidate small | 97.897 | 5.950 | 16.5x |
| C++ | candidate medium | 1392.550 | 71.917 | 19.4x |
| C++ | candidate large | 16704.200 | 1581.600 | 10.6x |
| Rust | dot384 | 4398.481 | 1272.615 | 3.5x |
| C++ | dot384 | 18017.900 | 1856.290 | 9.7x |

The candidate cases compare hash-based filtering and deduplication against a
generation-array workspace plus partial top-k selection. The dot-product case
compares scalar loops against AVX SIMD.

## Interpretation

The strongest current signal is not that C++ should replace Rust immediately.
It is that the data structure and memory-access pattern matter more than the
language boundary for candidate merge/filter/top-k work. Replacing per-request
hash tables with reusable dense workspaces gives a large win in both languages.

C++ is faster in the medium candidate case and competitive in the large case,
but Rust with the same generation-array strategy is already close enough to be a
credible production implementation path. For SIMD dot products, both languages
benefit substantially from contiguous memory and AVX. This is more relevant to
embedding-heavy batch work than to small per-user candidate post-processing.

Because an FFI boundary adds build, ABI, memory ownership, testing, and
deployment cost, these numbers do not justify integrating C++ kernels into the
production serving path yet. The next production-facing step should be to port
the proven Rust-native workspace strategy into the recall merge path and measure
endpoint-level p95 before adding FFI.

## Future Expansion Directions

1. Add aligned-versus-unaligned A/B benchmarks to isolate the direct value of
   64-byte alignment instead of mixing it with workspace reuse.
2. Add AVX2 and FMA variants with runtime CPU feature dispatch, while keeping a
   scalar fallback for portability.
3. Replay real recommendation debug snapshots so candidate id distributions,
   seen ratios, source overlap, and top-k sizes match production traffic.
4. Compare more top-k strategies: full sort, partial sort, selection plus final
   sort, binary heap, and fixed-size insertion buffers for small `k`.
5. Add compressed id remapping or segmented bitsets for production item ids that
   are sparse or exceed the dense workspace range.
6. Track memory footprint, allocation count, and cache behavior with profiler
   artifacts, not just wall-clock microbenchmarks.
7. Evaluate batch kernels separately from single-request kernels. SIMD and GPU
   work are more likely to pay off when many users or vectors are processed
   together.
8. Prototype FFI only after a Rust-native optimized implementation is measured
   inside the real `/recommend` path and still leaves a clear bottleneck.

## Production Guardrails

- Keep these experiments standalone until endpoint-level profiling proves that
  candidate post-processing is a dominant serving bottleneck.
- Do not introduce GPU work for per-request recall merge/filter/top-k; the
  transfer and scheduling overhead is unlikely to help small online requests.
- Prefer Rust-native optimized data structures first. Use C++ FFI only for a
  narrow, measured kernel with stable ownership and deterministic output.
- Treat benchmark numbers as hardware-, compiler-, and dataset-dependent. Rerun
  the script after changing CPU flags, compiler toolchains, or test data.
