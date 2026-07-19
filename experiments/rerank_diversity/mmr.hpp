#pragma once

// MMR (Maximal Marginal Relevance)
//
// Idea:
//   Trade off relevance and diversity while building the list greedily.
//   Score of a remaining candidate i given selected set S:
//     MMR(i) = λ * Rel(i) - (1-λ) * max_{j in S} Sim(i, j)
//   High λ keeps ranking order; low λ penalizes near-duplicates harder.
//
// Iterative selection:
//   1) Initialize max_sim[i] = 0 for all i (S is empty).
//   2) Pick argmax_i MMR(i); append it to S.
//   3) For every remaining candidate r, update
//        max_sim[r] = max(max_sim[r], Sim(r, picked))
//      so the expensive max-over-S scan is not repeated from scratch.
//   4) Repeat until |S| = K.
//
// Complexity (with precomputed similarity matrix):
//   Build S: O(N^2 * D) once (caller), then selection O(N * K).
//   Without a matrix, step 3 costs O(N * D) per pick → O(N * K * D).

#include "common.hpp"

#include <limits>
#include <vector>

namespace rerank {

inline std::vector<std::size_t> mmr_rerank(
    const std::vector<Candidate>& candidates,
    const std::vector<float>& similarity, // n*n row-major, cosine in [0, 1]
    std::size_t top_k,
    float lambda) {
    const std::size_t n = candidates.size();
    if (n == 0 || top_k == 0) {
        return {};
    }
    top_k = std::min(top_k, n);
    lambda = std::min(1.0f, std::max(0.0f, lambda));
    const float diversity = 1.0f - lambda;

    std::vector<std::size_t> selected;
    selected.reserve(top_k);
    std::vector<char> used(n, 0);
    std::vector<float> max_sim(n, 0.0f);

    for (std::size_t round = 0; round < top_k; ++round) {
        std::size_t best = n;
        float best_mmr = -std::numeric_limits<float>::infinity();

        for (std::size_t i = 0; i < n; ++i) {
            if (used[i]) {
                continue;
            }
            const float mmr = lambda * candidates[i].score - diversity * max_sim[i];
            if (mmr > best_mmr || (mmr == best_mmr && (best == n || candidates[i].id < candidates[best].id))) {
                best_mmr = mmr;
                best = i;
            }
        }

        if (best == n) {
            break;
        }

        used[best] = 1;
        selected.push_back(best);

        const float* row = &similarity[best * n];
        for (std::size_t i = 0; i < n; ++i) {
            if (used[i]) {
                continue;
            }
            if (row[i] > max_sim[i]) {
                max_sim[i] = row[i];
            }
        }
    }

    return selected;
}

} // namespace rerank
