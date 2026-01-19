// vector_ops.cpp - C++ 向量运算与 HNSW 索引实现
//
// 本文件包含:
// 1. 基础向量运算 (dot_product, cpp_add)
// 2. HNSW 索引封装 (使用 hnswlib)
// 3. 旧版暴力搜索 (search_top_k) - 保持向后兼容

#include "vector_ops.h"
#include "hnswlib/hnswlib.h"
#include "hnswlib/space_ip.h"  // InnerProductSpace (内积空间)
#include <algorithm>
#include <vector>
#include <mutex>

// ============================================================================
// 全局 HNSW 索引
// ============================================================================

// 索引指针 - 使用内积空间 (Inner Product Space)
// 内积空间适合归一化向量的相似度计算: distance = 1 - dot_product
static hnswlib::HierarchicalNSW<float>* g_hnsw_index = nullptr;
static hnswlib::InnerProductSpace* g_space = nullptr;
static int g_dim = 0;
static std::mutex g_mutex;  // 线程安全

// ============================================================================
// 基础向量运算实现
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
// HNSW 索引实现
// ============================================================================

extern "C" int hnsw_init(int dim, int max_elements, int M, int ef_construction) {
    std::lock_guard<std::mutex> lock(g_mutex);
    
    // 清理旧索引
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
        // 使用内积空间 (Inner Product Space)
        // 对于归一化向量: distance = 1 - inner_product
        // 所以 distance 越小 = similarity 越高
        g_space = new hnswlib::InnerProductSpace(dim);
        
        // 创建 HNSW 索引
        // M: 每层的最大连接数 (影响图的密度)
        // ef_construction: 构建时的动态列表大小 (影响索引质量)
        g_hnsw_index = new hnswlib::HierarchicalNSW<float>(
            g_space, 
            max_elements, 
            M, 
            ef_construction
        );
        
        return 0;  // 成功
    } catch (...) {
        return -1;  // 失败
    }
}

extern "C" int hnsw_add_item(int id, const float* vector) {
    std::lock_guard<std::mutex> lock(g_mutex);
    
    if (g_hnsw_index == nullptr) {
        return -1;  // 索引未初始化
    }
    
    try {
        // 添加向量到索引
        // label 使用 id 作为标识符
        g_hnsw_index->addPoint(vector, static_cast<hnswlib::labeltype>(id));
        return 0;
    } catch (...) {
        return -1;
    }
}

extern "C" void hnsw_set_ef(int ef) {
    std::lock_guard<std::mutex> lock(g_mutex);
    
    if (g_hnsw_index != nullptr) {
        // ef: 查询时的动态列表大小
        // 更高的 ef = 更好的召回率，但查询更慢
        g_hnsw_index->setEf(ef);
    }
}

extern "C" int hnsw_search_knn(const float* query, int k, int* out_ids, float* out_scores) {
    std::lock_guard<std::mutex> lock(g_mutex);
    
    if (g_hnsw_index == nullptr) {
        return -1;
    }
    
    try {
        // 搜索 K 个最近邻
        // 返回 priority_queue<pair<distance, label>>
        auto result = g_hnsw_index->searchKnn(query, k);
        
        int count = 0;
        // 结果按距离从大到小排列，我们需要反转
        std::vector<std::pair<float, hnswlib::labeltype>> results;
        while (!result.empty()) {
            results.push_back(result.top());
            result.pop();
        }
        
        // 反转得到从小到大 (最相似在前)
        std::reverse(results.begin(), results.end());
        
        for (const auto& item : results) {
            // 对于内积空间: distance = 1 - inner_product
            // 所以 similarity = 1 - distance = inner_product
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

// ============================================================================
// 旧版暴力搜索 (Legacy Brute-force Search)
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
