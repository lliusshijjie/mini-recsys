#include "cpp_kernels.hpp"

#include <cassert>
#include <vector>

int main() {
    Workspace workspace(128);
    assert(workspace.seen_alignment() % 64 == 0);
    assert(workspace.candidate_alignment() % 64 == 0);
    assert(workspace.score_alignment() % 64 == 0);
    assert(workspace.source_alignment() % 64 == 0);

    std::vector<Candidate> candidates = {
        {4, 0.40f, 0b0001},
        {2, 0.90f, 0b0010},
        {4, 0.80f, 0b0100},
        {7, 0.70f, 0b1000},
    };
    std::vector<uint32_t> seen_ids = {2};

    auto result = merge_filter_topk(candidates, seen_ids, 8, 2);

    assert(result.size() == 2);
    assert(result[0].id == 4);
    assert(result[0].source_mask == 0b0101);
    assert(result[1].id == 7);

    std::vector<float> left(384);
    std::vector<float> right(384);
    for (size_t i = 0; i < left.size(); ++i) {
        left[i] = static_cast<float>(i) * 0.001f;
        right[i] = 1.0f - static_cast<float>(i) * 0.0005f;
    }
    float scalar = dot_scalar(left.data(), right.data(), left.size());
    float simd = dot_simd(left.data(), right.data(), left.size());
    assert(std::abs(scalar - simd) < 0.001f);
    return 0;
}
