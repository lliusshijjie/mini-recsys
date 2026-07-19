#pragma once

// Greedy MAP DPP (Determinantal Point Process) with incremental Cholesky
//
// Idea:
//   Model a set Y with probability proportional to det(L_Y), where the kernel
//     L_ij = q_i * S_ij * q_j
//   mixes item quality q_i (from ranking score) and similarity S_ij.
//   Large det favors high-quality items that are also mutually diverse:
//   near-duplicate rows/columns shrink the volume of the parallelepiped.
//
// Iterative selection (Fast Greedy MAP):
//   1) Build L once from qualities and the similarity matrix.
//   2) Keep residual marginal gains d2[i] ≈ squared Cholesky diagonal for i.
//   3) Each round pick argmax_i d2[i] (largest log-det gain).
//   4) Append one Cholesky column and update all residual d2[j] -= e_j^2.
//   5) Repeat until K items are chosen.
//
// Complexity:
//   Build L: O(N^2). Selection with incremental Cholesky: O(N * K^2).
//   Much cheaper than recomputing det(L_Y) from scratch each trial O(N * K^4).

#include "common.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace rerank {

// Map ranking score in [0, 1] to a strictly positive quality.
inline float quality_from_score(float score, float theta) {
    const float s = std::min(1.0f, std::max(0.0f, score));
    return std::exp(0.5f * theta * s);
}

inline std::vector<float> build_dpp_kernel(
    const std::vector<Candidate>& candidates,
    const std::vector<float>& similarity,
    float theta) {
    const std::size_t n = candidates.size();
    std::vector<float> quality(n, 0.0f);
    for (std::size_t i = 0; i < n; ++i) {
        quality[i] = quality_from_score(candidates[i].score, theta);
    }

    std::vector<float> kernel(n * n, 0.0f);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            // Slight diagonal jitter keeps L numerically PSD under float noise.
            const float s = (i == j) ? (similarity[i * n + j] + 1e-5f) : similarity[i * n + j];
            kernel[i * n + j] = quality[i] * s * quality[j];
        }
    }
    return kernel;
}

inline std::vector<std::size_t> dpp_greedy(
    const std::vector<Candidate>& candidates,
    const std::vector<float>& kernel, // n*n row-major L
    std::size_t top_k) {
    const std::size_t n = candidates.size();
    if (n == 0 || top_k == 0) {
        return {};
    }
    top_k = std::min(top_k, n);

    std::vector<std::size_t> selected;
    selected.reserve(top_k);

    // cis[j * top_k + t] stores Cholesky coupling of item j to selected round t.
    std::vector<float> cis(n * top_k, 0.0f);
    std::vector<float> d2(n, 0.0f);
    for (std::size_t i = 0; i < n; ++i) {
        d2[i] = std::max(0.0f, kernel[i * n + i]);
    }

    for (std::size_t t = 0; t < top_k; ++t) {
        std::size_t best = n;
        float best_gain = -std::numeric_limits<float>::infinity();
        for (std::size_t i = 0; i < n; ++i) {
            if (d2[i] <= 0.0f) {
                continue;
            }
            if (d2[i] > best_gain ||
                (d2[i] == best_gain && (best == n || candidates[i].id < candidates[best].id))) {
                best_gain = d2[i];
                best = i;
            }
        }
        if (best == n) {
            break;
        }

        selected.push_back(best);
        const float sqrt_d = std::sqrt(std::max(d2[best], 1e-12f));

        for (std::size_t j = 0; j < n; ++j) {
            if (d2[j] <= 0.0f && j != best) {
                // Already eliminated; still need eis for bookkeeping of later rows
                // only when d2[j] > 0. Selected item itself is zeroed below.
            }
            float dot = 0.0f;
            for (std::size_t p = 0; p < t; ++p) {
                dot += cis[best * top_k + p] * cis[j * top_k + p];
            }
            const float e = (kernel[best * n + j] - dot) / sqrt_d;
            cis[j * top_k + t] = e;
            if (j == best) {
                continue;
            }
            d2[j] -= e * e;
            if (d2[j] < 1e-12f) {
                d2[j] = 0.0f;
            }
        }
        d2[best] = 0.0f;
    }

    return selected;
}

inline std::vector<std::size_t> dpp_rerank(
    const std::vector<Candidate>& candidates,
    const std::vector<float>& similarity,
    std::size_t top_k,
    float theta) {
    const std::vector<float> kernel = build_dpp_kernel(candidates, similarity, theta);
    return dpp_greedy(candidates, kernel, top_k);
}

} // namespace rerank
