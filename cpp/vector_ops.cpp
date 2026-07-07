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
#include <vector>
#include <mutex>

// ============================================================================
// Global HNSW index
// ============================================================================

// Index pointer using inner-product space.
// Inner product suits normalized vectors: distance = 1 - dot_product.
static hnswlib::HierarchicalNSW<float>* g_hnsw_index = nullptr;
static hnswlib::InnerProductSpace* g_space = nullptr;
static int g_dim = 0;
static std::mutex g_mutex;  // Thread safety

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

extern "C" int hnsw_init(int dim, int max_elements, int M, int ef_construction) {
    std::lock_guard<std::mutex> lock(g_mutex);

    // Tear down any existing index.
    if (g_hnsw_index != nullptr) {
        delete g_hnsw_index;
        g_hnsw_index = nullptr;
    }
    if (g_space != nullptr) {
        delete g_space;
        g_space = nullptr;
    }

    try {
        g_dim = dim;
        // Inner product space: for normalized vectors, distance = 1 - inner_product.
        // Lower distance means higher similarity.
        g_space = new hnswlib::InnerProductSpace(dim);

        // Create HNSW index.
        // M: max connections per layer (graph density).
        // ef_construction: dynamic list size during build (index quality).
        g_hnsw_index = new hnswlib::HierarchicalNSW<float>(
            g_space,
            max_elements,
            M,
            ef_construction
        );

        return 0;  // Success
    } catch (...) {
        return -1;  // Failure
    }
}

extern "C" int hnsw_add_item(int id, const float* vector) {
    std::lock_guard<std::mutex> lock(g_mutex);

    if (g_hnsw_index == nullptr) {
        return -1;  // Index not initialized
    }

    try {
        // Add vector; use id as the label.
        g_hnsw_index->addPoint(vector, static_cast<hnswlib::labeltype>(id));
        return 0;
    } catch (...) {
        return -1;
    }
}

extern "C" void hnsw_set_ef(int ef) {
    std::lock_guard<std::mutex> lock(g_mutex);

    if (g_hnsw_index != nullptr) {
        // ef: dynamic list size at query time.
        // Higher ef = better recall, slower queries.
        g_hnsw_index->setEf(ef);
    }
}

extern "C" int hnsw_search_knn(const float* query, int k, int* out_ids, float* out_scores) {
    std::lock_guard<std::mutex> lock(g_mutex);

    if (g_hnsw_index == nullptr) {
        return -1;
    }

    try {
        // Search k nearest neighbors.
        // Returns priority_queue<pair<distance, label>>.
        auto result = g_hnsw_index->searchKnn(query, k);

        int count = 0;
        // Results are ordered by distance descending; reverse them.
        std::vector<std::pair<float, hnswlib::labeltype>> results;
        while (!result.empty()) {
            results.push_back(result.top());
            result.pop();
        }

        // Reverse to ascending distance (most similar first).
        std::reverse(results.begin(), results.end());

        for (const auto& item : results) {
            // Inner product space: distance = 1 - inner_product.
            // similarity = 1 - distance = inner_product.
            float similarity = 1.0f - item.first;
            out_scores[count] = similarity;
            out_ids[count] = static_cast<int>(item.second);
            count++;
        }

        return count;
    } catch (...) {
        return -1;
    }
}

extern "C" void hnsw_destroy() {
    std::lock_guard<std::mutex> lock(g_mutex);

    if (g_hnsw_index != nullptr) {
        delete g_hnsw_index;
        g_hnsw_index = nullptr;
    }
    if (g_space != nullptr) {
        delete g_space;
        g_space = nullptr;
    }
    g_dim = 0;
}

extern "C" int hnsw_get_count() {
    std::lock_guard<std::mutex> lock(g_mutex);

    if (g_hnsw_index == nullptr) {
        return 0;
    }
    return static_cast<int>(g_hnsw_index->getCurrentElementCount());
}

extern "C" int hnsw_save_index(const char* path) {
    std::lock_guard<std::mutex> lock(g_mutex);

    if (g_hnsw_index == nullptr) {
        return -1;  // Index not initialized
    }

    try {
        g_hnsw_index->saveIndex(std::string(path));
        return 0;
    } catch (...) {
        return -1;
    }
}

extern "C" int hnsw_load_index(const char* path, int dim, int max_elements) {
    std::lock_guard<std::mutex> lock(g_mutex);

    // Tear down any existing index.
    if (g_hnsw_index != nullptr) {
        delete g_hnsw_index;
        g_hnsw_index = nullptr;
    }
    if (g_space != nullptr) {
        delete g_space;
        g_space = nullptr;
    }

    try {
        g_dim = dim;
        g_space = new hnswlib::InnerProductSpace(dim);

        // Try loading from file.
        FILE* f = fopen(path, "rb");
        if (f != nullptr) {
            fclose(f);
            // File exists; load index.
            g_hnsw_index = new hnswlib::HierarchicalNSW<float>(g_space, std::string(path));
            return 0;  // Loaded successfully
        } else {
            // File missing; create a new index.
            g_hnsw_index = new hnswlib::HierarchicalNSW<float>(g_space, max_elements, 16, 200);
            return 1;  // Created new index
        }
    } catch (...) {
        return -1;  // Failure
    }
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
