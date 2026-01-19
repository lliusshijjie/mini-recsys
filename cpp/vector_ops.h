#ifndef VECTOR_OPS_H
#define VECTOR_OPS_H

#ifdef __cplusplus
extern "C" {
#endif

int cpp_add(int a, int b);

float dot_product(const float* vec_a, const float* vec_b, int len);

/// 搜索与 query_vec 相似度最高的 Top K 个 Item
/// @param query_vec     查询向量 (长度为 cols)
/// @param item_matrix   扁平化的 Item 矩阵 (rows * cols)
/// @param item_ids      Item ID 数组 (长度为 rows)
/// @param rows          矩阵行数 (Item 数量)
/// @param cols          矩阵列数 (向量维度)
/// @param k             返回的 Top K 数量
/// @param out_ids       输出: Top K 的 Item ID (调用方分配, 长度 >= k)
/// @param out_scores    输出: Top K 的相似度分数 (调用方分配, 长度 >= k)
/// @return              实际返回的数量 (min(k, rows))
int search_top_k(
    const float* query_vec,
    const float* item_matrix,
    const int* item_ids,
    int rows,
    int cols,
    int k,
    int* out_ids,
    float* out_scores
);

#ifdef __cplusplus
}
#endif

#endif
