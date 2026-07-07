#ifndef VECTOR_OPS_H
#define VECTOR_OPS_H

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Basic vector operations
// ============================================================================

int cpp_add(int a, int b);
float dot_product(const float* vec_a, const float* vec_b, int len);

// ============================================================================
// HNSW index operations
// ============================================================================

/// Initialize the HNSW index.
///
/// @param dim              Vector dimension.
/// @param max_elements     Maximum index capacity.
/// @param M                Max connections per node (affects accuracy and memory).
///                         - Recommended: 16 (balanced), 32-64 (high accuracy).
///                         - Higher M = better recall, slower build, more memory.
/// @param ef_construction  Search depth during build (affects index quality).
///                         - Recommended: 200.
///                         - Higher = better index quality, slower build.
/// @return                 0 on success, -1 on failure.
int hnsw_init(int dim, int max_elements, int M, int ef_construction);

/// Add a single vector to the index.
/// @param id      Unique vector identifier.
/// @param vector  Vector data pointer (length dim).
/// @return        0 on success, -1 on failure.
int hnsw_add_item(int id, const float* vector);

/// Set query-time search depth.
/// @param ef  Search depth at query time (must be >= k).
///            - Recommended: 50-100.
///            - Higher ef = better recall, slower queries.
void hnsw_set_ef(int ef);

/// Search nearest neighbors.
/// @param query       Query vector (length dim).
/// @param k           Number of neighbors to return.
/// @param out_ids     Output: neighbor IDs (caller-allocated, length >= k).
/// @param out_scores  Output: neighbor distances/scores (caller-allocated, length >= k).
/// @return            Number of results returned, or -1 on failure.
int hnsw_search_knn(const float* query, int k, int* out_ids, float* out_scores);

/// Destroy the index and free memory.
void hnsw_destroy();

/// Return the number of elements in the index.
int hnsw_get_count();

/// Save the index to a file.
/// @param path  Output path.
/// @return      0 on success, -1 on failure.
int hnsw_save_index(const char* path);

/// Load an index from file (creates a new one if the file is missing).
/// @param path          Index file path.
/// @param dim           Vector dimension.
/// @param max_elements  Max elements (used only when creating a new index).
/// @return              0 loaded, 1 created new index, -1 on failure.
int hnsw_load_index(const char* path, int dim, int max_elements);

// ============================================================================
// Legacy interface (backward compatible)
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
