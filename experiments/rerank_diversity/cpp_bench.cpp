#include "dpp.hpp"
#include "mmr.hpp"

#include <chrono>
#include <iostream>
#include <string>
#include <vector>

using rerank::Candidate;
using rerank::build_similarity_matrix;
using rerank::dpp_rerank;
using rerank::kEmbeddingDim;
using rerank::l2_normalize;
using rerank::load_candidates_csv;
using rerank::mmr_rerank;

std::vector<Candidate> synth_candidates(std::size_t n) {
    static const char* categories[] = {
        "Books", "Electronics", "Home", "Clothing", "Sports", "Beauty"};
    std::vector<Candidate> out;
    out.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        Candidate c;
        c.id = static_cast<std::uint64_t>(i + 1);
        c.score = 1.0f - static_cast<float>(i) * 0.001f;
        c.category = categories[i % 6];
        c.embedding.assign(kEmbeddingDim, 0.0f);
        const std::size_t base = (i % 6) * 5;
        for (std::size_t d = 0; d < 5; ++d) {
            c.embedding[base + d] = 1.0f;
        }
        c.embedding[i % kEmbeddingDim] += 0.05f;
        l2_normalize(c.embedding);
        out.push_back(std::move(c));
    }
    return out;
}

template <typename Fn>
void bench(const std::string& name, int rounds, Fn&& fn) {
    for (int i = 0; i < 3; ++i) {
        fn();
    }
    const auto started = std::chrono::steady_clock::now();
    for (int i = 0; i < rounds; ++i) {
        fn();
    }
    const auto elapsed = std::chrono::steady_clock::now() - started;
    const double avg_us =
        std::chrono::duration<double, std::micro>(elapsed).count() / rounds;
    std::cout << "cpp," << name << "," << avg_us << "\n";
}

int main() {
    for (std::size_t n : {64ull, 128ull, 256ull}) {
        auto candidates = synth_candidates(n);
        auto sim = build_similarity_matrix(candidates);
        const int rounds = (n <= 128) ? 200 : 80;
        bench("mmr_n" + std::to_string(n) + "_k10", rounds, [&] {
            auto selected = mmr_rerank(candidates, sim, 10, 0.7f);
            (void)selected;
        });
        bench("dpp_n" + std::to_string(n) + "_k10", rounds, [&] {
            auto selected = dpp_rerank(candidates, sim, 10, 1.0f);
            (void)selected;
        });
    }

    try {
        auto candidates = load_candidates_csv("experiments/rerank_diversity/data/candidates.csv");
        auto sim = build_similarity_matrix(candidates);
        const auto n = candidates.size();
        bench("fixture_mmr_n" + std::to_string(n) + "_k10", 100, [&] {
            auto selected = mmr_rerank(candidates, sim, 10, 0.7f);
            (void)selected;
        });
        bench("fixture_dpp_n" + std::to_string(n) + "_k10", 100, [&] {
            auto selected = dpp_rerank(candidates, sim, 10, 1.0f);
            (void)selected;
        });
    } catch (const std::exception& ex) {
        std::cerr << "fixture bench skipped: " << ex.what() << "\n";
    }
    return 0;
}
