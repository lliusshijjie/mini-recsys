#include "dpp.hpp"
#include "mmr.hpp"

#include <cassert>
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
using rerank::score_topk;
using rerank::unique_category_count;

std::vector<Candidate> toy_candidates() {
    Candidate a;
    a.id = 1;
    a.score = 0.99f;
    a.category = "Books";
    a.embedding.assign(kEmbeddingDim, 0.0f);
    a.embedding[0] = 1.0f;

    Candidate b;
    b.id = 2;
    b.score = 0.98f;
    b.category = "Books";
    b.embedding.assign(kEmbeddingDim, 0.0f);
    b.embedding[0] = 0.98f;
    b.embedding[1] = 0.2f;

    Candidate c;
    c.id = 3;
    c.score = 0.90f;
    c.category = "Sports";
    c.embedding.assign(kEmbeddingDim, 0.0f);
    c.embedding[8] = 1.0f;

    std::vector<Candidate> items{a, b, c};
    for (auto& item : items) {
        l2_normalize(item.embedding);
    }
    return items;
}

void test_mmr_prefers_diverse_second_item() {
    auto candidates = toy_candidates();
    auto sim = build_similarity_matrix(candidates);
    auto selected = mmr_rerank(candidates, sim, 2, 0.5f);
    assert(selected.size() == 2);
    assert(candidates[selected[0]].id == 1);
    assert(candidates[selected[1]].id == 3);
}

void test_mmr_lambda_one_keeps_score_order() {
    auto candidates = toy_candidates();
    auto sim = build_similarity_matrix(candidates);
    auto selected = mmr_rerank(candidates, sim, 3, 1.0f);
    assert(selected.size() == 3);
    assert(candidates[selected[0]].id == 1);
    assert(candidates[selected[1]].id == 2);
    assert(candidates[selected[2]].id == 3);
}

void test_dpp_breaks_near_duplicates() {
    auto candidates = toy_candidates();
    auto sim = build_similarity_matrix(candidates);
    auto selected = dpp_rerank(candidates, sim, 2, 1.0f);
    assert(selected.size() == 2);
    assert(candidates[selected[0]].id == 1);
    assert(candidates[selected[1]].id == 3);
}

void test_fixture_from_products_improves_category_coverage() {
    const std::string path = "experiments/rerank_diversity/data/candidates.csv";
    std::vector<Candidate> candidates;
    try {
        candidates = load_candidates_csv(path);
    } catch (const std::exception& ex) {
        std::cerr << "skip fixture test: " << ex.what() << "\n";
        return;
    }
    assert(candidates.size() >= 32);

    auto sim = build_similarity_matrix(candidates);
    const std::size_t k = 10;
    auto baseline = score_topk(candidates, k);
    auto mmr = mmr_rerank(candidates, sim, k, 0.7f);
    auto dpp = dpp_rerank(candidates, sim, k, 1.0f);

    const auto base_cats = unique_category_count(candidates, baseline);
    const auto mmr_cats = unique_category_count(candidates, mmr);
    const auto dpp_cats = unique_category_count(candidates, dpp);

    assert(mmr.size() == k);
    assert(dpp.size() == k);
    assert(mmr_cats >= base_cats);
    assert(dpp_cats >= base_cats);
    assert(candidates[mmr[0]].id == candidates[baseline[0]].id);
    assert(candidates[dpp[0]].id == candidates[baseline[0]].id);

    std::cout << "fixture n=" << candidates.size()
              << " score_cats=" << base_cats
              << " mmr_cats=" << mmr_cats
              << " dpp_cats=" << dpp_cats << "\n";
}

int main() {
    test_mmr_prefers_diverse_second_item();
    test_mmr_lambda_one_keeps_score_order();
    test_dpp_breaks_near_duplicates();
    test_fixture_from_products_improves_category_coverage();
    std::cout << "cpp_rerank_test: ok\n";
    return 0;
}
