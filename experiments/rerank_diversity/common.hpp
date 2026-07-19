#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace rerank {

constexpr std::size_t kEmbeddingDim = 32;

struct Candidate {
    std::uint64_t id = 0;
    float score = 0.0f;
    std::string category;
    std::vector<float> embedding; // L2-normalized, size = kEmbeddingDim
};

inline float dot(const std::vector<float>& a, const std::vector<float>& b) {
    float sum = 0.0f;
    const std::size_t n = a.size();
    for (std::size_t i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

inline void l2_normalize(std::vector<float>& v) {
    float norm_sq = 0.0f;
    for (float x : v) {
        norm_sq += x * x;
    }
    if (norm_sq <= 1e-12f) {
        return;
    }
    const float inv = 1.0f / std::sqrt(norm_sq);
    for (float& x : v) {
        x *= inv;
    }
}

// Precompute cosine similarity matrix for L2-normalized embeddings: S_ij = e_i · e_j.
// Complexity: O(N^2 * D).
inline std::vector<float> build_similarity_matrix(const std::vector<Candidate>& candidates) {
    const std::size_t n = candidates.size();
    std::vector<float> sim(n * n, 0.0f);
    for (std::size_t i = 0; i < n; ++i) {
        sim[i * n + i] = 1.0f;
        for (std::size_t j = i + 1; j < n; ++j) {
            const float s = std::max(0.0f, dot(candidates[i].embedding, candidates[j].embedding));
            sim[i * n + j] = s;
            sim[j * n + i] = s;
        }
    }
    return sim;
}

inline std::vector<Candidate> load_candidates_csv(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open fixture: " + path);
    }

    std::string line;
    if (!std::getline(in, line)) {
        throw std::runtime_error("empty fixture: " + path);
    }

    std::vector<Candidate> out;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        std::stringstream ss(line);
        std::string cell;
        Candidate c;
        c.embedding.assign(kEmbeddingDim, 0.0f);

        if (!std::getline(ss, cell, ',')) {
            continue;
        }
        c.id = static_cast<std::uint64_t>(std::stoull(cell));
        if (!std::getline(ss, cell, ',')) {
            continue;
        }
        c.score = std::stof(cell);
        if (!std::getline(ss, c.category, ',')) {
            continue;
        }
        for (std::size_t d = 0; d < kEmbeddingDim; ++d) {
            if (!std::getline(ss, cell, ',')) {
                throw std::runtime_error("bad embedding columns in fixture");
            }
            c.embedding[d] = std::stof(cell);
        }
        l2_normalize(c.embedding);
        out.push_back(std::move(c));
    }
    return out;
}

inline std::size_t unique_category_count(
    const std::vector<Candidate>& candidates,
    const std::vector<std::size_t>& indices) {
    std::vector<std::string> cats;
    cats.reserve(indices.size());
    for (std::size_t idx : indices) {
        cats.push_back(candidates[idx].category);
    }
    std::sort(cats.begin(), cats.end());
    cats.erase(std::unique(cats.begin(), cats.end()), cats.end());
    return cats.size();
}

inline std::vector<std::size_t> score_topk(const std::vector<Candidate>& candidates, std::size_t k) {
    std::vector<std::size_t> order(candidates.size());
    for (std::size_t i = 0; i < order.size(); ++i) {
        order[i] = i;
    }
    std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        if (candidates[a].score != candidates[b].score) {
            return candidates[a].score > candidates[b].score;
        }
        return candidates[a].id < candidates[b].id;
    });
    if (order.size() > k) {
        order.resize(k);
    }
    return order;
}

} // namespace rerank
