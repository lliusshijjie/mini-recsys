//! FFI bindings and safe wrappers.
//!
//! This module is the boundary between Rust and C++.
//! All `unsafe` code lives here; business logic must not call unsafe directly.

use crate::model::Item;
use libc::{c_float, c_int};

// ============================================================================
// External C function declarations (raw FFI bindings)
// ============================================================================

extern "C" {
    // Basic math
    fn cpp_add(a: c_int, b: c_int) -> c_int;
    fn dot_product(vec_a: *const c_float, vec_b: *const c_float, len: c_int) -> c_float;

    // HNSW index operations
    fn hnsw_init(dim: c_int, max_elements: c_int, M: c_int, ef_construction: c_int) -> c_int;
    fn hnsw_add_item(id: c_int, vector: *const c_float) -> c_int;
    fn hnsw_set_ef(ef: c_int);
    fn hnsw_search_knn(
        query: *const c_float,
        k: c_int,
        out_ids: *mut c_int,
        out_scores: *mut c_float,
    ) -> c_int;
    fn hnsw_destroy();
    fn hnsw_get_count() -> c_int;
    fn hnsw_save_index(path: *const libc::c_char) -> c_int;
    fn hnsw_load_index(path: *const libc::c_char, dim: c_int, max_elements: c_int) -> c_int;

    // Legacy brute-force search
    fn search_top_k(
        query_vec: *const c_float,
        item_matrix: *const c_float,
        item_ids: *const c_int,
        rows: c_int,
        cols: c_int,
        k: c_int,
        out_ids: *mut c_int,
        out_scores: *mut c_float,
    ) -> c_int;
}

// ============================================================================
// Basic math safe wrappers
// ============================================================================

pub fn add(a: i32, b: i32) -> i32 {
    // SAFETY: cpp_add is a pure function; c_int is compatible with i32.
    unsafe { cpp_add(a, b) }
}

pub fn compute_dot_product(vec_a: &[f32], vec_b: &[f32]) -> Option<f32> {
    if vec_a.len() != vec_b.len() {
        return None;
    }
    let len = vec_a.len() as c_int;
    // SAFETY: slices are valid for the call; as_ptr() and len are correct.
    let result = unsafe { dot_product(vec_a.as_ptr(), vec_b.as_ptr(), len) };
    Some(result)
}

// ============================================================================
// HNSW index safe wrappers
// ============================================================================

/// HNSW index configuration.
pub struct HnswConfig {
    /// Vector dimension.
    pub dim: usize,
    /// Maximum number of elements.
    pub max_elements: usize,
    /// Max connections per node (affects accuracy and memory).
    /// Recommended: 16 (balanced), 32-64 (high accuracy).
    pub m: usize,
    /// Search depth during index build (affects index quality).
    /// Recommended: 200.
    pub ef_construction: usize,
    /// Search depth at query time (affects recall).
    /// Recommended: 50-100, must be >= k.
    pub ef_search: usize,
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            dim: 64,
            max_elements: 10000,
            m: 16,
            ef_construction: 200,
            ef_search: 50,
        }
    }
}

/// Initialize the HNSW index.
///
/// # Arguments
/// * `config` - HNSW configuration.
///
/// # Returns
/// * `Ok(())` - initialization succeeded.
/// * `Err(String)` - initialization failed.
pub fn init_hnsw_index(config: &HnswConfig) -> Result<(), String> {
    // SAFETY: all arguments are plain scalars; no pointer operations.
    let result = unsafe {
        hnsw_init(
            config.dim as c_int,
            config.max_elements as c_int,
            config.m as c_int,
            config.ef_construction as c_int,
        )
    };

    if result == 0 {
        // Set query-time search depth.
        unsafe { hnsw_set_ef(config.ef_search as c_int) };
        Ok(())
    } else {
        Err("Failed to initialize HNSW index".to_string())
    }
}

/// Add a single item to the index.
///
/// # Arguments
/// * `id` - item ID.
/// * `embedding` - item vector.
///
/// # Returns
/// * `Ok(())` - add succeeded.
/// * `Err(String)` - add failed.
pub fn add_item_to_hnsw(id: u64, embedding: &[f32]) -> Result<(), String> {
    // SAFETY: embedding is a valid slice for the duration of the call.
    let result = unsafe { hnsw_add_item(id as c_int, embedding.as_ptr()) };

    if result == 0 {
        Ok(())
    } else {
        Err(format!("Failed to add item {} to HNSW index", id))
    }
}

/// Search nearest neighbors with the HNSW index.
///
/// # Arguments
/// * `query` - query vector.
/// * `k` - number of neighbors to return.
///
/// # Returns
/// List of `(item_id, similarity_score)` sorted by similarity descending.
pub fn hnsw_search(query: &[f32], k: usize) -> Vec<(u64, f32)> {
    if k == 0 {
        return Vec::new();
    }

    let mut out_ids: Vec<c_int> = vec![0; k];
    let mut out_scores: Vec<f32> = vec![0.0; k];

    // SAFETY:
    // 1. query is a valid slice for the duration of the call.
    // 2. out_ids/out_scores are preallocated with sufficient capacity.
    let count = unsafe {
        hnsw_search_knn(
            query.as_ptr(),
            k as c_int,
            out_ids.as_mut_ptr(),
            out_scores.as_mut_ptr(),
        )
    };

    if count < 0 {
        return Vec::new();
    }

    (0..count as usize)
        .map(|i| (out_ids[i] as u64, out_scores[i]))
        .collect()
}

/// Destroy the HNSW index and free memory.
pub fn destroy_hnsw_index() {
    // SAFETY: no arguments; releases the global index only.
    unsafe { hnsw_destroy() };
}

/// Return the number of elements in the index.
pub fn get_hnsw_count() -> usize {
    // SAFETY: no arguments; return value is a plain scalar.
    unsafe { hnsw_get_count() as usize }
}

/// Save the index to a file.
pub fn save_hnsw_index(path: &str) -> Result<(), String> {
    use std::ffi::CString;
    let c_path = CString::new(path).map_err(|_| "Invalid path".to_string())?;

    // SAFETY: c_path is a valid null-terminated C string.
    let result = unsafe { hnsw_save_index(c_path.as_ptr()) };

    if result == 0 {
        Ok(())
    } else {
        Err("Failed to save HNSW index".to_string())
    }
}

/// Load an index (creates a new one if the file is missing).
/// Returns: Ok(true) = loaded, Ok(false) = created new index, Err = failure.
pub fn load_hnsw_index(
    path: &str,
    dim: usize,
    max_elements: usize,
    ef_search: usize,
) -> Result<bool, String> {
    use std::ffi::CString;
    let c_path = CString::new(path).map_err(|_| "Invalid path".to_string())?;

    // SAFETY: c_path is a valid null-terminated C string.
    let result = unsafe { hnsw_load_index(c_path.as_ptr(), dim as c_int, max_elements as c_int) };

    match result {
        0 => {
            // Loaded successfully; set ef_search.
            unsafe { hnsw_set_ef(ef_search as c_int) };
            Ok(true)
        }
        1 => {
            // Created a new index.
            unsafe { hnsw_set_ef(ef_search as c_int) };
            Ok(false)
        }
        _ => Err("Failed to load HNSW index".to_string()),
    }
}

// ============================================================================
// Legacy brute-force search
// ============================================================================

/// Recall phase: find top-k items most similar to the user (brute force).
pub fn recommend_recall(user_embedding: &[f32], items: &[Item], k: usize) -> Vec<(u64, f32)> {
    if items.is_empty() || k == 0 {
        return Vec::new();
    }

    let rows = items.len();
    let cols = user_embedding.len();

    let flat_matrix: Vec<f32> = items
        .iter()
        .flat_map(|item| item.embedding.iter().copied())
        .collect();

    let item_ids: Vec<c_int> = items.iter().map(|item| item.id as c_int).collect();

    let actual_k = k.min(rows);
    let mut out_ids: Vec<c_int> = vec![0; actual_k];
    let mut out_scores: Vec<f32> = vec![0.0; actual_k];

    // SAFETY: all pointer and length arguments have been validated.
    let count = unsafe {
        search_top_k(
            user_embedding.as_ptr(),
            flat_matrix.as_ptr(),
            item_ids.as_ptr(),
            rows as c_int,
            cols as c_int,
            actual_k as c_int,
            out_ids.as_mut_ptr(),
            out_scores.as_mut_ptr(),
        )
    };

    (0..count as usize)
        .map(|i| (out_ids[i] as u64, out_scores[i]))
        .collect()
}

// ============================================================================
// Unit tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpp_add() {
        assert_eq!(add(2, 3), 5);
        assert_eq!(add(-1, 1), 0);
    }

    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let result = compute_dot_product(&a, &b);
        assert!(result.is_some());
        assert!((result.unwrap() - 32.0).abs() < 1e-6);
    }

    #[test]
    fn test_hnsw_lifecycle() {
        // Initialize.
        let config = HnswConfig {
            dim: 3,
            max_elements: 100,
            m: 16,
            ef_construction: 100,
            ef_search: 50,
        };
        assert!(init_hnsw_index(&config).is_ok());

        // Add vectors.
        assert!(add_item_to_hnsw(1, &[1.0, 0.0, 0.0]).is_ok());
        assert!(add_item_to_hnsw(2, &[0.0, 1.0, 0.0]).is_ok());
        assert!(add_item_to_hnsw(3, &[0.5, 0.5, 0.0]).is_ok());

        assert_eq!(get_hnsw_count(), 3);

        // Search.
        let results = hnsw_search(&[1.0, 0.0, 0.0], 2);
        assert_eq!(results.len(), 2);
        // First result should be ID=1 (exact match).
        assert_eq!(results[0].0, 1);

        // Tear down.
        destroy_hnsw_index();
        assert_eq!(get_hnsw_count(), 0);
    }

    #[test]
    fn test_recommend_recall() {
        let user_emb = vec![1.0, 0.0, 0.0];
        let items = vec![
            Item::new(1, "A", vec![1.0, 0.0, 0.0]),
            Item::new(2, "B", vec![0.0, 1.0, 0.0]),
            Item::new(3, "C", vec![0.5, 0.5, 0.0]),
        ];

        let results = recommend_recall(&user_emb, &items, 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 1);
        assert!((results[0].1 - 1.0).abs() < 1e-6);
    }
}
