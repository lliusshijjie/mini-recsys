#ifndef MINI_RECSYS_EXPERIMENT_CPP_KERNELS_HPP
#define MINI_RECSYS_EXPERIMENT_CPP_KERNELS_HPP

#include <algorithm>
#include <cstdint>
#include <immintrin.h>
#include <memory>
#include <new>
#include <numeric>
#include <vector>

struct Candidate {
    uint32_t id;
    float score;
    uint8_t source_mask;
};

template <typename T>
class AlignedBuffer {
public:
    explicit AlignedBuffer(size_t len)
        : len_(len),
          data_(static_cast<T*>(::operator new[](sizeof(T) * len_, std::align_val_t(64)))) {
        std::fill(data_, data_ + len_, T{});
    }

    ~AlignedBuffer() {
        ::operator delete[](data_, std::align_val_t(64));
    }

    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;

    T& operator[](size_t index) {
        return data_[index];
    }

    const T& operator[](size_t index) const {
        return data_[index];
    }

    T* data() {
        return data_;
    }

    const T* data() const {
        return data_;
    }

    T* begin() {
        return data_;
    }

    T* end() {
        return data_ + len_;
    }

private:
    size_t len_;
    T* data_;
};

class Workspace {
public:
    explicit Workspace(uint32_t max_id)
        : max_id_(max_id),
          seen_generation_(static_cast<size_t>(max_id) + 1),
          candidate_generation_(static_cast<size_t>(max_id) + 1),
          scores_(static_cast<size_t>(max_id) + 1),
          source_masks_(static_cast<size_t>(max_id) + 1),
          generation_(0) {}

    std::vector<Candidate> merge_filter_topk(
        const std::vector<Candidate>& candidates,
        const std::vector<uint32_t>& seen_ids,
        size_t k) {
        if (k == 0) {
            return {};
        }

        next_generation();
        touched_.clear();

        for (uint32_t id : seen_ids) {
            if (id <= max_id_) {
                seen_generation_[id] = generation_;
            }
        }

        for (const Candidate& candidate : candidates) {
            if (candidate.id > max_id_ || seen_generation_[candidate.id] == generation_) {
                continue;
            }

            if (candidate_generation_[candidate.id] != generation_) {
                candidate_generation_[candidate.id] = generation_;
                scores_[candidate.id] = candidate.score;
                source_masks_[candidate.id] = candidate.source_mask;
                touched_.push_back(candidate.id);
            } else {
                source_masks_[candidate.id] |= candidate.source_mask;
                if (candidate.score > scores_[candidate.id]) {
                    scores_[candidate.id] = candidate.score;
                }
            }
        }

        std::vector<Candidate> output;
        output.reserve(touched_.size());
        for (uint32_t id : touched_) {
            output.push_back(Candidate{id, scores_[id], source_masks_[id]});
        }

        topk_desc(output, k);
        return output;
    }

    uintptr_t seen_alignment() const {
        return reinterpret_cast<uintptr_t>(seen_generation_.data());
    }

    uintptr_t candidate_alignment() const {
        return reinterpret_cast<uintptr_t>(candidate_generation_.data());
    }

    uintptr_t score_alignment() const {
        return reinterpret_cast<uintptr_t>(scores_.data());
    }

    uintptr_t source_alignment() const {
        return reinterpret_cast<uintptr_t>(source_masks_.data());
    }

private:
    static bool better(const Candidate& left, const Candidate& right) {
        if (left.score != right.score) {
            return left.score > right.score;
        }
        return left.id < right.id;
    }

    static void topk_desc(std::vector<Candidate>& items, size_t k) {
        if (items.size() > k) {
            std::nth_element(items.begin(), items.begin() + static_cast<std::ptrdiff_t>(k), items.end(), better);
            items.resize(k);
        }
        std::sort(items.begin(), items.end(), better);
    }

    void next_generation() {
        ++generation_;
        if (generation_ == 0) {
            std::fill(seen_generation_.begin(), seen_generation_.end(), 0);
            std::fill(candidate_generation_.begin(), candidate_generation_.end(), 0);
            generation_ = 1;
        }
    }

    uint32_t max_id_;
    AlignedBuffer<uint32_t> seen_generation_;
    AlignedBuffer<uint32_t> candidate_generation_;
    AlignedBuffer<float> scores_;
    AlignedBuffer<uint8_t> source_masks_;
    std::vector<uint32_t> touched_;
    uint32_t generation_;
};

inline float dot_scalar(const float* left, const float* right, size_t len) {
    float total = 0.0f;
    for (size_t i = 0; i < len; ++i) {
        total += left[i] * right[i];
    }
    return total;
}

inline float dot_simd(const float* left, const float* right, size_t len) {
#if defined(__AVX__)
    size_t index = 0;
    __m256 sum = _mm256_setzero_ps();
    for (; index + 8 <= len; index += 8) {
        __m256 l = _mm256_loadu_ps(left + index);
        __m256 r = _mm256_loadu_ps(right + index);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(l, r));
    }

    alignas(32) float lanes[8];
    _mm256_store_ps(lanes, sum);
    float total = std::accumulate(lanes, lanes + 8, 0.0f);
    for (; index < len; ++index) {
        total += left[index] * right[index];
    }
    return total;
#else
    return dot_scalar(left, right, len);
#endif
}

inline std::vector<Candidate> merge_filter_topk(
    const std::vector<Candidate>& candidates,
    const std::vector<uint32_t>& seen_ids,
    uint32_t max_id,
    size_t k) {
    Workspace workspace(max_id);
    return workspace.merge_filter_topk(candidates, seen_ids, k);
}

#endif
