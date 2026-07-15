//! FFI bindings and safe wrappers.
//!
//! This module is the boundary between Rust and C++.
//! All `unsafe` code lives here; business logic must not call unsafe directly.

use crate::model::Item;
use libc::{c_float, c_int};
use std::ptr::NonNull;

#[repr(C)]
struct HnswIndexHandle {
    _private: [u8; 0],
}

// ============================================================================
// External C function declarations (raw FFI bindings)
// ============================================================================

#[allow(dead_code)]
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
    fn hnsw_create_index(
        dim: c_int,
        max_elements: c_int,
        m: c_int,
        ef_construction: c_int,
    ) -> *mut HnswIndexHandle;
    fn hnsw_load_or_create_index(
        path: *const libc::c_char,
        dim: c_int,
        max_elements: c_int,
        out_status: *mut c_int,
    ) -> *mut HnswIndexHandle;
    fn hnsw_index_add_item(handle: *mut HnswIndexHandle, id: u64, vector: *const c_float) -> c_int;
    fn hnsw_index_set_ef(handle: *mut HnswIndexHandle, ef: c_int);
    fn hnsw_index_search_knn(
        handle: *mut HnswIndexHandle,
        query: *const c_float,
        k: c_int,
        out_ids: *mut u64,
        out_scores: *mut c_float,
    ) -> c_int;
    fn hnsw_index_search_batch(
        handle: *mut HnswIndexHandle,
        queries: *const c_float,
        query_count: c_int,
        k: c_int,
        out_ids: *mut u64,
        out_scores: *mut c_float,
    ) -> c_int;
    fn hnsw_index_get_count(handle: *mut HnswIndexHandle) -> u64;
    fn hnsw_index_save(handle: *mut HnswIndexHandle, path: *const libc::c_char) -> c_int;
    fn hnsw_free_index(handle: *mut HnswIndexHandle);

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

pub struct HnswIndex {
    handle: NonNull<HnswIndexHandle>,
    dim: usize,
}

unsafe impl Send for HnswIndex {}
unsafe impl Sync for HnswIndex {}

impl HnswIndex {
    pub fn new(config: &HnswConfig) -> Result<Self, String> {
        let handle = unsafe {
            hnsw_create_index(
                config.dim as c_int,
                config.max_elements as c_int,
                config.m as c_int,
                config.ef_construction as c_int,
            )
        };
        let handle =
            NonNull::new(handle).ok_or_else(|| "Failed to create HNSW index".to_string())?;
        let index = Self {
            handle,
            dim: config.dim,
        };
        index.set_ef(config.ef_search);
        Ok(index)
    }

    pub fn load_or_create(
        path: &str,
        dim: usize,
        max_elements: usize,
        ef_search: usize,
    ) -> Result<(Self, bool), String> {
        use std::ffi::CString;
        let c_path = CString::new(path).map_err(|_| "Invalid path".to_string())?;
        let mut status = -1;
        let handle = unsafe {
            hnsw_load_or_create_index(
                c_path.as_ptr(),
                dim as c_int,
                max_elements as c_int,
                &mut status,
            )
        };
        let handle = NonNull::new(handle).ok_or_else(|| "Failed to load HNSW index".to_string())?;
        let index = Self { handle, dim };
        index.set_ef(ef_search);
        match status {
            0 => Ok((index, true)),
            1 => Ok((index, false)),
            _ => Err("Failed to load HNSW index".to_string()),
        }
    }

    pub fn add_item(&self, id: u64, embedding: &[f32]) -> Result<(), String> {
        if embedding.len() != self.dim {
            return Err(format!(
                "Embedding dimension {} does not match HNSW dimension {}",
                embedding.len(),
                self.dim
            ));
        }

        let result = unsafe { hnsw_index_add_item(self.handle.as_ptr(), id, embedding.as_ptr()) };
        if result == 0 {
            Ok(())
        } else {
            Err(format!("Failed to add item {} to HNSW index", id))
        }
    }

    pub fn set_ef(&self, ef_search: usize) {
        unsafe { hnsw_index_set_ef(self.handle.as_ptr(), ef_search as c_int) };
    }

    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(u64, f32)>, String> {
        if k == 0 {
            return Ok(Vec::new());
        }
        if query.len() != self.dim {
            return Err(format!(
                "Query dimension {} does not match HNSW dimension {}",
                query.len(),
                self.dim
            ));
        }

        let mut out_ids: Vec<u64> = vec![0; k];
        let mut out_scores: Vec<f32> = vec![0.0; k];
        let count = unsafe {
            hnsw_index_search_knn(
                self.handle.as_ptr(),
                query.as_ptr(),
                k as c_int,
                out_ids.as_mut_ptr(),
                out_scores.as_mut_ptr(),
            )
        };
        if count < 0 {
            return Err("HNSW search failed".to_string());
        }

        Ok((0..count as usize)
            .map(|index| (out_ids[index], out_scores[index]))
            .collect())
    }

    pub fn search_batch(
        &self,
        queries: &[Vec<f32>],
        k: usize,
    ) -> Result<Vec<Vec<(u64, f32)>>, String> {
        if queries.is_empty() || k == 0 {
            return Ok(vec![Vec::new(); queries.len()]);
        }
        if queries.iter().any(|query| query.len() != self.dim) {
            return Err("At least one query dimension does not match HNSW dimension".to_string());
        }

        let flat_queries: Vec<f32> = queries
            .iter()
            .flat_map(|query| query.iter().copied())
            .collect();
        let mut out_ids: Vec<u64> = vec![0; queries.len() * k];
        let mut out_scores: Vec<f32> = vec![0.0; queries.len() * k];
        let count = unsafe {
            hnsw_index_search_batch(
                self.handle.as_ptr(),
                flat_queries.as_ptr(),
                queries.len() as c_int,
                k as c_int,
                out_ids.as_mut_ptr(),
                out_scores.as_mut_ptr(),
            )
        };
        if count < 0 {
            return Err("HNSW batch search failed".to_string());
        }

        Ok((0..queries.len())
            .map(|query_index| {
                (0..k)
                    .filter_map(|rank| {
                        let offset = query_index * k + rank;
                        let item_id = out_ids[offset];
                        if item_id == 0 {
                            None
                        } else {
                            Some((item_id, out_scores[offset]))
                        }
                    })
                    .collect()
            })
            .collect())
    }

    pub fn count(&self) -> usize {
        unsafe { hnsw_index_get_count(self.handle.as_ptr()) as usize }
    }

    pub fn save(&self, path: &str) -> Result<(), String> {
        use std::ffi::CString;
        let c_path = CString::new(path).map_err(|_| "Invalid path".to_string())?;
        let result = unsafe { hnsw_index_save(self.handle.as_ptr(), c_path.as_ptr()) };
        if result == 0 {
            Ok(())
        } else {
            Err("Failed to save HNSW index".to_string())
        }
    }
}

impl Drop for HnswIndex {
    fn drop(&mut self) {
        unsafe { hnsw_free_index(self.handle.as_ptr()) };
    }
}

// ============================================================================
// Basic math safe wrappers
// ============================================================================

#[allow(dead_code)]
pub fn add(a: i32, b: i32) -> i32 {
    // SAFETY: cpp_add is a pure function; c_int is compatible with i32.
    unsafe { cpp_add(a, b) }
}

#[allow(dead_code)]
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
#[allow(dead_code)]
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
#[allow(dead_code)]
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
#[allow(dead_code)]
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
#[allow(dead_code)]
pub fn destroy_hnsw_index() {
    // SAFETY: no arguments; releases the global index only.
    unsafe { hnsw_destroy() };
}

/// Return the number of elements in the index.
#[allow(dead_code)]
pub fn get_hnsw_count() -> usize {
    // SAFETY: no arguments; return value is a plain scalar.
    unsafe { hnsw_get_count() as usize }
}

/// Save the index to a file.
#[allow(dead_code)]
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
#[allow(dead_code)]
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
#[allow(dead_code)]
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
    use std::sync::{Arc, Mutex};
    use std::thread;
    use std::time::Instant;

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
    fn hnsw_index_handle_supports_u64_ids_and_batch_search() {
        let config = HnswConfig {
            dim: 3,
            max_elements: 100,
            m: 16,
            ef_construction: 100,
            ef_search: 50,
        };
        let index = HnswIndex::new(&config).expect("handle index should initialize");
        let large_id = i32::MAX as u64 + 7;

        index.add_item(large_id, &[1.0, 0.0, 0.0]).unwrap();
        index.add_item(42, &[0.0, 1.0, 0.0]).unwrap();

        let single = index.search(&[1.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(single[0].0, large_id);

        let batch = index
            .search_batch(&[vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]], 1)
            .unwrap();
        assert_eq!(batch.len(), 2);
        assert_eq!(batch[0][0].0, large_id);
        assert_eq!(batch[1][0].0, 42);
        assert_eq!(index.count(), 2);
    }

    #[test]
    #[ignore]
    fn performance_matrix() {
        run_hnsw_performance_matrix();
    }

    fn run_hnsw_performance_matrix() {
        let dataset_sizes = parse_usize_list_env("MINI_RECSYS_PERF_DATASETS", &[10_000, 100_000]);
        let concurrency_levels = parse_usize_list_env("MINI_RECSYS_PERF_CONCURRENCY", &[1, 8, 32]);
        let query_count = env_usize("MINI_RECSYS_PERF_QUERIES", 256);
        let dim = env_usize("MINI_RECSYS_PERF_DIM", 384);
        let k = env_usize("MINI_RECSYS_PERF_K", 100);

        for dataset_size in dataset_sizes {
            let config = HnswConfig {
                dim,
                max_elements: dataset_size + 16,
                m: 16,
                ef_construction: 200,
                ef_search: k.max(100),
            };
            let index = Arc::new(HnswIndex::new(&config).expect("HNSW benchmark index"));
            let build_started = Instant::now();
            for item_id in 1..=dataset_size {
                let embedding = deterministic_unit_vector(item_id as u64, dim);
                index.add_item(item_id as u64, &embedding).unwrap();
            }
            let build_ms = build_started.elapsed().as_secs_f64() * 1000.0;
            assert_eq!(index.count(), dataset_size);

            for concurrency in &concurrency_levels {
                let samples = Arc::new(Mutex::new(Vec::with_capacity(query_count)));
                let started = Instant::now();
                let mut handles = Vec::new();
                let per_thread = query_count.div_ceil(*concurrency);

                for thread_index in 0..*concurrency {
                    let index = Arc::clone(&index);
                    let samples = Arc::clone(&samples);
                    let handle = thread::spawn(move || {
                        for query_offset in 0..per_thread {
                            let query_index = thread_index * per_thread + query_offset;
                            if query_index >= query_count {
                                break;
                            }
                            let query =
                                deterministic_unit_vector(10_000_000 + query_index as u64, dim);
                            let query_started = Instant::now();
                            let results = index.search(&query, k).unwrap();
                            assert!(!results.is_empty());
                            samples
                                .lock()
                                .unwrap()
                                .push(query_started.elapsed().as_micros() as u64);
                        }
                    });
                    handles.push(handle);
                }

                for handle in handles {
                    handle.join().unwrap();
                }

                let elapsed = started.elapsed().as_secs_f64();
                let mut samples = samples.lock().unwrap().clone();
                samples.sort_unstable();
                let qps = query_count as f64 / elapsed.max(0.001);
                println!(
                    "performance_matrix dataset={} dim={} concurrency={} queries={} k={} build_ms={:.2} qps={:.2} p50_us={} p95_us={} p99_us={}",
                    dataset_size,
                    dim,
                    concurrency,
                    query_count,
                    k,
                    build_ms,
                    qps,
                    percentile(&samples, 0.50),
                    percentile(&samples, 0.95),
                    percentile(&samples, 0.99)
                );
            }
        }
    }

    fn parse_usize_list_env(name: &str, default: &[usize]) -> Vec<usize> {
        std::env::var(name)
            .ok()
            .map(|value| {
                value
                    .split(',')
                    .filter_map(|part| part.trim().parse::<usize>().ok())
                    .collect::<Vec<_>>()
            })
            .filter(|values| !values.is_empty())
            .unwrap_or_else(|| default.to_vec())
    }

    fn env_usize(name: &str, default: usize) -> usize {
        std::env::var(name)
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(default)
    }

    fn deterministic_unit_vector(seed: u64, dim: usize) -> Vec<f32> {
        let mut state = seed ^ 0x9E37_79B9_7F4A_7C15;
        let mut vector = Vec::with_capacity(dim);
        for _ in 0..dim {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let value = ((state >> 32) as u32) as f32 / u32::MAX as f32;
            vector.push(value * 2.0 - 1.0);
        }
        let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
        vector.into_iter().map(|value| value / norm).collect()
    }

    fn percentile(sorted_samples: &[u64], percentile: f64) -> u64 {
        if sorted_samples.is_empty() {
            return 0;
        }
        let index = ((sorted_samples.len() - 1) as f64 * percentile).round() as usize;
        sorted_samples[index]
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
