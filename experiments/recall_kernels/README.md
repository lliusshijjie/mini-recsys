# Recall Kernel Experiments

Standalone CPU experiments for low-level recommendation candidate operations.
This folder is intentionally not wired into the main Rust crate, `build.rs`, or
the production C++ HNSW code.

The shared contract is:

1. filter seen item IDs,
2. merge duplicate candidate IDs,
3. keep the highest score per ID,
4. OR together recall source masks,
5. return deterministic score-descending top-k.

Run on Windows PowerShell:

```powershell
.\experiments\recall_kernels\run.ps1
```

The benchmark compares:

- `naive_hash`: HashMap/HashSet or unordered_map/unordered_set baseline.
- `generation_topk`: reusable generation arrays for seen/filter/dedup plus
  partial top-k selection.
- `dot_scalar`: scalar 384-dimension dot product loop.
- `dot_simd`: AVX dot product when the host CPU/compiler exposes AVX, otherwise
  scalar fallback.

The benchmark output is CSV:

```text
language,algorithm,case,candidates,max_id,seen,k,iterations,total_ms,avg_us,checksum
```

Treat the numbers as local feasibility data. They are not production endpoint
latency because they isolate candidate post-processing only.
