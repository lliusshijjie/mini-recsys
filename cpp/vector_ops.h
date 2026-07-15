#ifndef VECTOR_OPS_H
#define VECTOR_OPS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

// ============================================================================
// Basic vector operations
// ============================================================================

int cpp_add(int a, int b);
float dot_product(const float* vec_a, const float* vec_b, int len);

// ============================================================================
// HNSW index operations
// ============================================================================

typedef struct HnswIndexHandle HnswIndexHandle;

HnswIndexHandle* hnsw_create_index(int dim, int max_elements, int M, int ef_construction);
HnswIndexHandle* hnsw_load_or_create_index(
    const char* path,
    int dim,
    int max_elements,
    int* out_status
);
int hnsw_index_add_item(HnswIndexHandle* handle, uint64_t id, const float* vector);
void hnsw_index_set_ef(HnswIndexHandle* handle, int ef);
int hnsw_index_search_knn(
    HnswIndexHandle* handle,
    const float* query,
    int k,
    uint64_t* out_ids,
    float* out_scores
);
int hnsw_index_search_batch(
    HnswIndexHandle* handle,
    const float* queries,
    int query_count,
    int k,
    uint64_t* out_ids,
    float* out_scores
);
uint64_t hnsw_index_get_count(HnswIndexHandle* handle);
int hnsw_index_save(HnswIndexHandle* handle, const char* path);
void hnsw_free_index(HnswIndexHandle* handle);

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
