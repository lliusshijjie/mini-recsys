#include "vector_ops.h"
#include <algorithm>
#include <vector>

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
    
    // 存储 (score, item_id) 对
    std::vector<std::pair<float, int>> scores(rows);
    
    for (int i = 0; i < rows; ++i) {
        // 指针偏移访问第 i 行向量:
        // item_matrix 是扁平化的一维数组，第 i 行的起始位置 = i * cols
        // 即 item_matrix + i * cols 指向第 i 个 Item 的向量首地址
        const float* row_ptr = item_matrix + i * cols;
        
        float score = dot_product(query_vec, row_ptr, cols);
        scores[i] = {score, item_ids[i]};
    }
    
    // 使用 partial_sort 只排序前 K 个，O(n * log(k))
    std::partial_sort(
        scores.begin(),
        scores.begin() + actual_k,
        scores.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; }  // 降序
    );
    
    // 写入输出数组
    for (int i = 0; i < actual_k; ++i) {
        out_ids[i] = scores[i].second;
        out_scores[i] = scores[i].first;
    }
    
    return actual_k;
}
