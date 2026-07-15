#include "cpp_kernels.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <cstdint>
#include <iostream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

struct Case {
    const char* name;
    size_t candidate_count;
    uint32_t max_id;
    size_t seen_count;
    size_t k;
    size_t iterations;
};

class Lcg {
public:
    explicit Lcg(uint64_t seed) : state_(seed) {}

    uint32_t next_u32() {
        state_ = state_ * 6364136223846793005ULL + 1ULL;
        return static_cast<uint32_t>(state_ >> 32);
    }

    float next_f32() {
        return static_cast<float>(next_u32()) / static_cast<float>(UINT32_MAX);
    }

private:
    uint64_t state_;
};

std::vector<Candidate> generate_candidates(size_t count, uint32_t max_id, uint64_t seed) {
    Lcg rng(seed);
    std::vector<Candidate> candidates;
    candidates.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        candidates.push_back(Candidate{
            rng.next_u32() % (max_id + 1),
            rng.next_f32(),
            static_cast<uint8_t>(1u << (rng.next_u32() % 4)),
        });
    }
    return candidates;
}

std::vector<uint32_t> generate_seen_ids(size_t count, uint32_t max_id, uint64_t seed) {
    Lcg rng(seed);
    std::vector<uint32_t> seen;
    seen.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        seen.push_back(rng.next_u32() % (max_id + 1));
    }
    return seen;
}

bool better(const Candidate& left, const Candidate& right) {
    if (left.score != right.score) {
        return left.score > right.score;
    }
    return left.id < right.id;
}

std::vector<Candidate> naive_merge_filter_topk(
    const std::vector<Candidate>& candidates,
    const std::vector<uint32_t>& seen_ids,
    uint32_t max_id,
    size_t k) {
    std::unordered_set<uint32_t> seen;
    seen.reserve(seen_ids.size() * 2);
    for (uint32_t id : seen_ids) {
        seen.insert(id);
    }

    std::unordered_map<uint32_t, Candidate> merged;
    merged.reserve(candidates.size());
    for (const Candidate& candidate : candidates) {
        if (candidate.id > max_id || seen.find(candidate.id) != seen.end()) {
            continue;
        }
        auto [iter, inserted] = merged.emplace(candidate.id, candidate);
        if (!inserted) {
            iter->second.source_mask |= candidate.source_mask;
            if (candidate.score > iter->second.score) {
                iter->second.score = candidate.score;
            }
        }
    }

    std::vector<Candidate> output;
    output.reserve(merged.size());
    for (const auto& entry : merged) {
        output.push_back(entry.second);
    }
    std::sort(output.begin(), output.end(), better);
    if (output.size() > k) {
        output.resize(k);
    }
    return output;
}

uint64_t checksum(const std::vector<Candidate>& items) {
    uint64_t acc = 0;
    for (const Candidate& item : items) {
        uint32_t score_bits = 0;
        static_assert(sizeof(score_bits) == sizeof(item.score), "score bit size mismatch");
        std::memcpy(&score_bits, &item.score, sizeof(score_bits));
        acc = (acc * 16777619ULL) ^ item.id ^ (static_cast<uint64_t>(item.source_mask) << 32)
              ^ (static_cast<uint64_t>(score_bits) << 1);
    }
    return acc;
}

template <typename Operation>
std::pair<double, uint64_t> measure(size_t iterations, Operation operation) {
    uint64_t checksum_acc = 0;
    auto started = std::chrono::steady_clock::now();
    for (size_t i = 0; i < iterations; ++i) {
        checksum_acc = checksum_acc * 1099511628211ULL + operation();
    }
    auto elapsed = std::chrono::steady_clock::now() - started;
    double total_ms = std::chrono::duration<double, std::milli>(elapsed).count();
    return {total_ms, checksum_acc};
}

void print_result(
    const char* language,
    const char* algorithm,
    const Case& test_case,
    double total_ms,
    uint64_t checksum_value) {
    double avg_us = total_ms * 1000.0 / static_cast<double>(test_case.iterations);
    std::cout << language << "," << algorithm << "," << test_case.name << ","
              << test_case.candidate_count << "," << test_case.max_id << ","
              << test_case.seen_count << "," << test_case.k << "," << test_case.iterations
              << "," << total_ms << "," << avg_us << "," << checksum_value << "\n";
}

void run_case(const Case& test_case) {
    auto candidates = generate_candidates(test_case.candidate_count, test_case.max_id, 0xC0FFEE);
    auto seen_ids = generate_seen_ids(test_case.seen_count, test_case.max_id, 0xBAD5EED);

    auto [naive_ms, naive_checksum] = measure(test_case.iterations, [&]() {
        return checksum(naive_merge_filter_topk(candidates, seen_ids, test_case.max_id, test_case.k));
    });

    Workspace workspace(test_case.max_id);
    auto [optimized_ms, optimized_checksum] = measure(test_case.iterations, [&]() {
        return checksum(workspace.merge_filter_topk(candidates, seen_ids, test_case.k));
    });

    print_result("cpp", "naive_hash", test_case, naive_ms, naive_checksum);
    print_result("cpp", "generation_topk", test_case, optimized_ms, optimized_checksum);
}

std::vector<float> generate_f32_values(size_t count, uint64_t seed) {
    Lcg rng(seed);
    std::vector<float> values;
    values.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        values.push_back(rng.next_f32() * 2.0f - 1.0f);
    }
    return values;
}

void print_dot_result(
    const char* language,
    const char* algorithm,
    const char* name,
    size_t dim,
    size_t vector_count,
    size_t iterations,
    double total_ms,
    uint64_t checksum_value) {
    double avg_us = total_ms * 1000.0 / static_cast<double>(iterations);
    std::cout << language << "," << algorithm << "," << name << ","
              << dim * vector_count << "," << dim << "," << vector_count << ","
              << dim << "," << iterations << "," << total_ms << "," << avg_us
              << "," << checksum_value << "\n";
}

void run_dot_case(const char* name, size_t dim, size_t vector_count, size_t iterations) {
    auto query = generate_f32_values(dim, 0x5151);
    auto vectors = generate_f32_values(dim * vector_count, 0x9191);

    auto [scalar_ms, scalar_checksum] = measure(iterations, [&]() {
        uint64_t bits = 0;
        for (size_t index = 0; index < vector_count; ++index) {
            float value = dot_scalar(query.data(), vectors.data() + index * dim, dim);
            uint32_t value_bits = 0;
            std::memcpy(&value_bits, &value, sizeof(value_bits));
            bits += value_bits;
        }
        return bits;
    });
    auto [simd_ms, simd_checksum] = measure(iterations, [&]() {
        uint64_t bits = 0;
        for (size_t index = 0; index < vector_count; ++index) {
            float value = dot_simd(query.data(), vectors.data() + index * dim, dim);
            uint32_t value_bits = 0;
            std::memcpy(&value_bits, &value, sizeof(value_bits));
            bits += value_bits;
        }
        return bits;
    });

    print_dot_result("cpp", "dot_scalar", name, dim, vector_count, iterations, scalar_ms, scalar_checksum);
    print_dot_result("cpp", "dot_simd", name, dim, vector_count, iterations, simd_ms, simd_checksum);
}

int main() {
    const std::vector<Case> cases = {
        {"small", 1000, 2000, 200, 100, 2000},
        {"medium", 10000, 20000, 2000, 100, 300},
        {"large", 100000, 200000, 20000, 100, 40},
    };

    std::cout << "language,algorithm,case,candidates,max_id,seen,k,iterations,total_ms,avg_us,checksum\n";
    for (const Case& test_case : cases) {
        run_case(test_case);
    }
    run_dot_case("dot384", 384, 20000, 300);
    return 0;
}
