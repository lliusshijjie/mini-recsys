//! FFI 接口定义与 Safe Wrapper
//!
//! 这个模块是 Rust 与 C++ 交互的边界层。
//! 所有的 `unsafe` 代码都集中在这里，业务层不应该直接接触 unsafe。

use crate::model::Item;
use libc::{c_float, c_int};

// ============================================================================
// 外部 C 函数声明 (Raw FFI Bindings)
// ============================================================================

extern "C" {
    // 基础运算
    fn cpp_add(a: c_int, b: c_int) -> c_int;
    fn dot_product(vec_a: *const c_float, vec_b: *const c_float, len: c_int) -> c_float;

    // HNSW 索引操作
    fn hnsw_init(dim: c_int, max_elements: c_int, M: c_int, ef_construction: c_int) -> c_int;
    fn hnsw_add_item(id: c_int, vector: *const c_float) -> c_int;
    fn hnsw_set_ef(ef: c_int);
    fn hnsw_search_knn(query: *const c_float, k: c_int, out_ids: *mut c_int, out_scores: *mut c_float) -> c_int;
    fn hnsw_destroy();
    fn hnsw_get_count() -> c_int;
    fn hnsw_save_index(path: *const libc::c_char) -> c_int;
    fn hnsw_load_index(path: *const libc::c_char, dim: c_int, max_elements: c_int) -> c_int;

    // 旧版暴力搜索
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
// 基础运算 Safe Wrapper
// ============================================================================

pub fn add(a: i32, b: i32) -> i32 {
    // SAFETY: cpp_add 是纯函数，c_int 与 i32 兼容
    unsafe { cpp_add(a, b) }
}

pub fn compute_dot_product(vec_a: &[f32], vec_b: &[f32]) -> Option<f32> {
    if vec_a.len() != vec_b.len() {
        return None;
    }
    let len = vec_a.len() as c_int;
    // SAFETY: 切片在调用期间有效，as_ptr() 返回有效指针，len 正确
    let result = unsafe { dot_product(vec_a.as_ptr(), vec_b.as_ptr(), len) };
    Some(result)
}

// ============================================================================
// HNSW 索引 Safe Wrapper
// ============================================================================

/// HNSW 索引配置
pub struct HnswConfig {
    /// 向量维度
    pub dim: usize,
    /// 最大元素数量
    pub max_elements: usize,
    /// 每个节点的最大连接数 (影响精度和内存)
    /// 推荐值: 16 (平衡), 32-64 (高精度)
    pub m: usize,
    /// 构建时的搜索深度 (影响索引质量)
    /// 推荐值: 200
    pub ef_construction: usize,
    /// 查询时的搜索深度 (影响召回率)
    /// 推荐值: 50-100, 必须 >= k
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

/// 初始化 HNSW 索引
/// 
/// # Arguments
/// * `config` - HNSW 配置参数
/// 
/// # Returns
/// * `Ok(())` - 初始化成功
/// * `Err(String)` - 初始化失败
pub fn init_hnsw_index(config: &HnswConfig) -> Result<(), String> {
    // SAFETY: 所有参数都是基本类型，无指针操作
    let result = unsafe {
        hnsw_init(
            config.dim as c_int,
            config.max_elements as c_int,
            config.m as c_int,
            config.ef_construction as c_int,
        )
    };

    if result == 0 {
        // 设置查询时的搜索深度
        unsafe { hnsw_set_ef(config.ef_search as c_int) };
        Ok(())
    } else {
        Err("Failed to initialize HNSW index".to_string())
    }
}

/// 向索引添加单个物品
/// 
/// # Arguments
/// * `id` - 物品 ID
/// * `embedding` - 物品向量
/// 
/// # Returns
/// * `Ok(())` - 添加成功
/// * `Err(String)` - 添加失败
pub fn add_item_to_hnsw(id: u64, embedding: &[f32]) -> Result<(), String> {
    // SAFETY: embedding 是有效切片，在调用期间不会被释放
    let result = unsafe { hnsw_add_item(id as c_int, embedding.as_ptr()) };

    if result == 0 {
        Ok(())
    } else {
        Err(format!("Failed to add item {} to HNSW index", id))
    }
}

/// 使用 HNSW 索引搜索最近邻
/// 
/// # Arguments
/// * `query` - 查询向量
/// * `k` - 返回的最近邻数量
/// 
/// # Returns
/// 返回 (item_id, similarity_score) 的列表，按相似度降序排列
pub fn hnsw_search(query: &[f32], k: usize) -> Vec<(u64, f32)> {
    if k == 0 {
        return Vec::new();
    }

    let mut out_ids: Vec<c_int> = vec![0; k];
    let mut out_scores: Vec<f32> = vec![0.0; k];

    // SAFETY:
    // 1. query 是有效切片，在调用期间有效
    // 2. out_ids/out_scores 已预分配足够空间
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

/// 销毁 HNSW 索引并释放内存
pub fn destroy_hnsw_index() {
    // SAFETY: 无需传递参数，仅释放全局索引
    unsafe { hnsw_destroy() };
}

/// 获取索引中的元素数量
pub fn get_hnsw_count() -> usize {
    // SAFETY: 无参数，返回值是基本类型
    unsafe { hnsw_get_count() as usize }
}

/// 保存索引到文件
pub fn save_hnsw_index(path: &str) -> Result<(), String> {
    use std::ffi::CString;
    let c_path = CString::new(path).map_err(|_| "Invalid path".to_string())?;
    
    // SAFETY: c_path 是有效的以 null 结尾的 C 字符串
    let result = unsafe { hnsw_save_index(c_path.as_ptr()) };
    
    if result == 0 {
        Ok(())
    } else {
        Err("Failed to save HNSW index".to_string())
    }
}

/// 加载索引 (若文件不存在则创建新索引)
/// 返回: Ok(true) = 已加载, Ok(false) = 创建了新索引, Err = 失败
pub fn load_hnsw_index(path: &str, dim: usize, max_elements: usize, ef_search: usize) -> Result<bool, String> {
    use std::ffi::CString;
    let c_path = CString::new(path).map_err(|_| "Invalid path".to_string())?;
    
    // SAFETY: c_path 是有效的以 null 结尾的 C 字符串
    let result = unsafe { 
        hnsw_load_index(c_path.as_ptr(), dim as c_int, max_elements as c_int) 
    };
    
    match result {
        0 => {
            // 加载成功，设置 ef_search
            unsafe { hnsw_set_ef(ef_search as c_int) };
            Ok(true)
        }
        1 => {
            // 创建了新索引
            unsafe { hnsw_set_ef(ef_search as c_int) };
            Ok(false)
        }
        _ => Err("Failed to load HNSW index".to_string()),
    }
}

// ============================================================================
// 旧版暴力搜索 (Legacy)
// ============================================================================

/// 召回阶段：从物品库中找出与用户最相似的 Top K 物品 (暴力搜索)
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

    // SAFETY: 所有指针和长度参数都经过验证
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
// 单元测试
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
        // 初始化
        let config = HnswConfig {
            dim: 3,
            max_elements: 100,
            m: 16,
            ef_construction: 100,
            ef_search: 50,
        };
        assert!(init_hnsw_index(&config).is_ok());

        // 添加向量
        assert!(add_item_to_hnsw(1, &[1.0, 0.0, 0.0]).is_ok());
        assert!(add_item_to_hnsw(2, &[0.0, 1.0, 0.0]).is_ok());
        assert!(add_item_to_hnsw(3, &[0.5, 0.5, 0.0]).is_ok());

        assert_eq!(get_hnsw_count(), 3);

        // 搜索
        let results = hnsw_search(&[1.0, 0.0, 0.0], 2);
        assert_eq!(results.len(), 2);
        // 第一个结果应该是 ID=1 (完全匹配)
        assert_eq!(results[0].0, 1);

        // 销毁
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
