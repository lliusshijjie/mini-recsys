#ifndef VECTOR_OPS_H
#define VECTOR_OPS_H

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// 基础向量运算 (Basic Vector Operations)
// ============================================================================

int cpp_add(int a, int b);
float dot_product(const float* vec_a, const float* vec_b, int len);

// ============================================================================
// HNSW 索引操作 (HNSW Index Operations)
// ============================================================================

/// 初始化 HNSW 索引
/// 
/// @param dim              向量维度
/// @param max_elements     索引最大容量
/// @param M                每个节点的最大连接数 (影响精度和内存)
///                         - 推荐值: 16 (平衡), 32-64 (高精度)
///                         - 更高的 M = 更好的召回率，但更慢的构建速度和更多内存
/// @param ef_construction  构建时的搜索深度 (影响索引质量)
///                         - 推荐值: 200
///                         - 更高 = 更好的索引质量，但更慢的构建速度
/// @return                 0 成功, -1 失败
int hnsw_init(int dim, int max_elements, int M, int ef_construction);

/// 向索引添加单个向量
/// @param id      向量的唯一标识符
/// @param vector  向量数据指针 (长度为 dim)
/// @return        0 成功, -1 失败
int hnsw_add_item(int id, const float* vector);

/// 设置查询时的搜索深度
/// @param ef  查询时的搜索深度 (必须 >= k)
///            - 推荐值: 50-100
///            - 更高的 ef = 更好的召回率，但更慢的查询速度
void hnsw_set_ef(int ef);

/// 搜索最近邻
/// @param query       查询向量 (长度为 dim)
/// @param k           返回的最近邻数量
/// @param out_ids     输出: 最近邻的 ID (调用方分配, 长度 >= k)
/// @param out_scores  输出: 最近邻的距离/分数 (调用方分配, 长度 >= k)
/// @return            实际返回的数量, -1 表示失败
int hnsw_search_knn(const float* query, int k, int* out_ids, float* out_scores);

/// 销毁索引并释放内存
void hnsw_destroy();

/// 获取索引中的元素数量
int hnsw_get_count();

/// 保存索引到文件
/// @param path  保存路径
/// @return      0 成功, -1 失败
int hnsw_save_index(const char* path);

/// 从文件加载索引 (若文件不存在则创建新索引)
/// @param path          索引文件路径
/// @param dim           向量维度
/// @param max_elements  最大元素数量 (仅在创建新索引时使用)
/// @return              0 成功加载, 1 创建了新索引, -1 失败
int hnsw_load_index(const char* path, int dim, int max_elements);

// ============================================================================
// 旧版接口 (Legacy Interface - 保持向后兼容)
// ============================================================================

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

#endif // VECTOR_OPS_H
