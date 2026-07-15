// vector_ops.cpp - C++ vector operations and HNSW index implementation
//
// This file contains:
// 1. Basic vector math (dot_product, cpp_add)
// 2. HNSW index wrapper (hnswlib)
// 3. Legacy brute-force search (search_top_k) for backward compatibility

#include "vector_ops.h"
#include "hnswlib/hnswlib.h"
#include "hnswlib/space_ip.h"  // InnerProductSpace
#include <algorithm>
#include <cstdio>
#include <memory>
#include <vector>
#include <mutex>
#include <shared_mutex>
#include <string>

// ============================================================================
// Global HNSW index
// ============================================================================

struct HnswIndexHandle {
    std::unique_ptr<hnswlib::InnerProductSpace> space;
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> index;
    int dim = 0;
    mutable std::shared_mutex mutex;
};

static HnswIndexHandle* g_hnsw_index = nullptr;
static std::mutex g_mutex;  // Thread safety

static int fill_search_results(
    std::priority_queue<std::pair<float, hnswlib::labeltype>>& result,
    int k,
    uint64_t* out_ids,
    float* out_scores
) {
    int count = 0;
    std::vector<std::pair<float, hnswlib::labeltype>> results;
    while (!result.empty()) {
        results.push_back(result.top());
        result.pop();
    }

    std::reverse(results.begin(), results.end());

    for (const auto& item : results) {
        if (count >= k) break;
        out_scores[count] = 1.0f - item.first;
        out_ids[count] = static_cast<uint64_t>(item.second);
        count++;
    }

    return count;
}

// ============================================================================
// Basic vector operations
// ============================================================================

extern "C" int cpp_add(int a, int b) {
    return a + b;
}

extern "C" float dot_product(const float* vec_a, const float* vec_b, int len) {
    float result = 0.0f;
    for (int i = 0; i < len; ++i) {
        result += vec_a[i] * vec_b[i];
    }
    return result;
}

// ============================================================================
// HNSW index implementation
// ============================================================================

extern "C" HnswIndexHandle* hnsw_create_index(int dim, int max_elements, int M, int ef_construction) {
    try {
        auto handle = std::make_unique<HnswIndexHandle>();
        handle->dim = dim;
        handle->space = std::make_unique<hnswlib::InnerProductSpace>(dim);
        handle->index = std::make_unique<hnswlib::HierarchicalNSW<float>>(
            handle->space.get(),
            max_elements,
            M,
            ef_construction
        );
        return handle.release();
    } catch (...) {
        return nullptr;
    }
}

extern "C" HnswIndexHandle* hnsw_load_or_create_index(
    const char* path,
    int dim,
    int max_elements,
    int* out_status
) {
    if (out_status != nullptr) {
        *out_status = -1;
    }

    try {
        auto handle = std::make_unique<HnswIndexHandle>();
        handle->dim = dim;
        handle->space = std::make_unique<hnswlib::InnerProductSpace>(dim);

        FILE* f = fopen(path, "rb");
        if (f != nullptr) {
            fclose(f);
            handle->index = std::make_unique<hnswlib::HierarchicalNSW<float>>(
                handle->space.get(),
                std::string(path)
            );
            if (out_status != nullptr) {
                *out_status = 0;
            }
        } else {
            handle->index = std::make_unique<hnswlib::HierarchicalNSW<float>>(
                handle->space.get(),
                max_elements,
                16,
                200
            );
            if (out_status != nullptr) {
                *out_status = 1;
            }
        }

        return handle.release();
    } catch (...) {
        if (out_status != nullptr) {
            *out_status = -1;
        }
        return nullptr;
    }
}

extern "C" int hnsw_index_add_item(HnswIndexHandle* handle, uint64_t id, const float* vector) {
    if (handle == nullptr || handle->index == nullptr || vector == nullptr) {
        return -1;
    }

    std::unique_lock<std::shared_mutex> lock(handle->mutex);
    try {
        handle->index->addPoint(vector, static_cast<hnswlib::labeltype>(id));
        return 0;
    } catch (...) {
        return -1;
    }
}

extern "C" void hnsw_index_set_ef(HnswIndexHandle* handle, int ef) {
    if (handle == nullptr || handle->index == nullptr) {
        return;
    }

    std::unique_lock<std::shared_mutex> lock(handle->mutex);
    handle->index->setEf(ef);
}

extern "C" int hnsw_index_search_knn(
    HnswIndexHandle* handle,
    const float* query,
    int k,
    uint64_t* out_ids,
    float* out_scores
) {
    if (handle == nullptr || handle->index == nullptr || query == nullptr || k <= 0) {
        return -1;
    }

    std::shared_lock<std::shared_mutex> lock(handle->mutex);
    try {
        auto result = handle->index->searchKnn(query, k);
        return fill_search_results(result, k, out_ids, out_scores);
    } catch (...) {
        return -1;
    }
}

extern "C" int hnsw_index_search_batch(
    HnswIndexHandle* handle,
    const float* queries,
    int query_count,
    int k,
    uint64_t* out_ids,
    float* out_scores
) {
    if (handle == nullptr || handle->index == nullptr || queries == nullptr || query_count < 0 || k <= 0) {
        return -1;
    }

    std::shared_lock<std::shared_mutex> lock(handle->mutex);
    try {
        for (int query_index = 0; query_index < query_count; ++query_index) {
            const float* query = queries + query_index * handle->dim;
            auto result = handle->index->searchKnn(query, k);
            int written = fill_search_results(
                result,
                k,
                out_ids + query_index * k,
                out_scores + query_index * k
            );
            for (int i = written; i < k; ++i) {
                out_ids[query_index * k + i] = 0;
                out_scores[query_index * k + i] = 0.0f;
            }
        }
        return query_count;
    } catch (...) {
        return -1;
    }
}

extern "C" uint64_t hnsw_index_get_count(HnswIndexHandle* handle) {
    if (handle == nullptr || handle->index == nullptr) {
        return 0;
    }

    std::shared_lock<std::shared_mutex> lock(handle->mutex);
    return static_cast<uint64_t>(handle->index->getCurrentElementCount());
}

extern "C" int hnsw_index_save(HnswIndexHandle* handle, const char* path) {
    if (handle == nullptr || handle->index == nullptr || path == nullptr) {
        return -1;
    }

    std::shared_lock<std::shared_mutex> lock(handle->mutex);
    try {
        handle->index->saveIndex(std::string(path));
        return 0;
    } catch (...) {
        return -1;
    }
}

extern "C" void hnsw_free_index(HnswIndexHandle* handle) {
    delete handle;
}

extern "C" int hnsw_init(int dim, int max_elements, int M, int ef_construction) {
    std::lock_guard<std::mutex> lock(g_mutex);

    if (g_hnsw_index != nullptr) {
        hnsw_free_index(g_hnsw_index);
    }

    g_hnsw_index = hnsw_create_index(dim, max_elements, M, ef_construction);
    return g_hnsw_index == nullptr ? -1 : 0;
}

extern "C" int hnsw_add_item(int id, const float* vector) {
    std::lock_guard<std::mutex> lock(g_mutex);
    return hnsw_index_add_item(g_hnsw_index, static_cast<uint64_t>(id), vector);
}

extern "C" void hnsw_set_ef(int ef) {
    std::lock_guard<std::mutex> lock(g_mutex);
    hnsw_index_set_ef(g_hnsw_index, ef);
}

extern "C" int hnsw_search_knn(const float* query, int k, int* out_ids, float* out_scores) {
    std::vector<uint64_t> wide_ids(k);
    int count = hnsw_index_search_knn(g_hnsw_index, query, k, wide_ids.data(), out_scores);
    if (count < 0) {
        return count;
    }
    for (int i = 0; i < count; ++i) {
        out_ids[i] = static_cast<int>(wide_ids[i]);
    }
    return count;
}

extern "C" void hnsw_destroy() {
    std::lock_guard<std::mutex> lock(g_mutex);

    if (g_hnsw_index != nullptr) {
        hnsw_free_index(g_hnsw_index);
        g_hnsw_index = nullptr;
    }
}

extern "C" int hnsw_get_count() {
    std::lock_guard<std::mutex> lock(g_mutex);
    return static_cast<int>(hnsw_index_get_count(g_hnsw_index));
}

extern "C" int hnsw_save_index(const char* path) {
    std::lock_guard<std::mutex> lock(g_mutex);
    return hnsw_index_save(g_hnsw_index, path);
}

extern "C" int hnsw_load_index(const char* path, int dim, int max_elements) {
    std::lock_guard<std::mutex> lock(g_mutex);

    if (g_hnsw_index != nullptr) {
        hnsw_free_index(g_hnsw_index);
    }

    int status = -1;
    g_hnsw_index = hnsw_load_or_create_index(path, dim, max_elements, &status);
    return g_hnsw_index == nullptr ? -1 : status;
}

// ============================================================================
// Legacy brute-force search
// ============================================================================

extern "C" int search_top_k(
    const float* query_vec,
    const float* item_matrix,
    const int* item_ids,
    int rows,
    int cols,
    int k,
    int* out_ids,
    float* out_scores
) {
    if (rows <= 0 || k <= 0) return 0;

    int actual_k = std::min(k, rows);

    std::vector<std::pair<float, int>> scores(rows);

    for (int i = 0; i < rows; ++i) {
        const float* row_ptr = item_matrix + i * cols;
        float score = dot_product(query_vec, row_ptr, cols);
        scores[i] = {score, item_ids[i]};
    }

    std::partial_sort(
        scores.begin(),
        scores.begin() + actual_k,
        scores.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; }
    );

    for (int i = 0; i < actual_k; ++i) {
        out_ids[i] = scores[i].second;
        out_scores[i] = scores[i].first;
    }

    return actual_k;
}
