//! FFI 接口定义与 Safe Wrapper

use crate::model::Item;
use libc::{c_float, c_int};

extern "C" {
    fn cpp_add(a: c_int, b: c_int) -> c_int;
    fn dot_product(vec_a: *const c_float, vec_b: *const c_float, len: c_int) -> c_float;
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

/// 召回阶段：从物品库中找出与用户最相似的 Top K 物品
/// 
/// # 内存布局说明
/// 将 Vec<Item> 的 embedding 扁平化为一维数组传给 C++：
/// ```text
/// Item[0].embedding: [e00, e01, ..., e0d]
/// Item[1].embedding: [e10, e11, ..., e1d]
/// ...
/// 扁平化后: [e00, e01, ..., e0d, e10, e11, ..., e1d, ...]
/// ```
/// 这里会发生一次内存拷贝（将分散的 Vec 合并为连续内存），
/// 这是必要的，因为 C++ 需要连续的内存布局。
pub fn recommend_recall(user_embedding: &[f32], items: &[Item], k: usize) -> Vec<(u64, f32)> {
    if items.is_empty() || k == 0 {
        return Vec::new();
    }

    let rows = items.len();
    let cols = user_embedding.len();

    // 扁平化 Item embeddings 为连续内存（发生一次拷贝）
    let flat_matrix: Vec<f32> = items
        .iter()
        .flat_map(|item| item.embedding.iter().copied())
        .collect();

    // 提取 Item IDs（转为 c_int）
    let item_ids: Vec<c_int> = items.iter().map(|item| item.id as c_int).collect();

    // 分配输出缓冲区
    let actual_k = k.min(rows);
    let mut out_ids: Vec<c_int> = vec![0; actual_k];
    let mut out_scores: Vec<f32> = vec![0.0; actual_k];

    // SAFETY:
    // 1. user_embedding 是有效切片，在调用期间不会被释放
    // 2. flat_matrix 是刚创建的连续内存，生命周期覆盖整个调用
    // 3. item_ids 同样是刚创建的 Vec，生命周期有效
    // 4. out_ids/out_scores 已预分配足够空间，C++ 写入不会越界
    // 5. rows/cols/k 正确反映实际长度
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

    // 组装结果
    (0..count as usize)
        .map(|i| (out_ids[i] as u64, out_scores[i]))
        .collect()
}

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
    fn test_recommend_recall() {
        let user_emb = vec![1.0, 0.0, 0.0];
        let items = vec![
            Item::new(1, "A", vec![1.0, 0.0, 0.0]),  // 完全匹配，score=1
            Item::new(2, "B", vec![0.0, 1.0, 0.0]),  // 正交，score=0
            Item::new(3, "C", vec![0.5, 0.5, 0.0]),  // 部分匹配，score=0.5
        ];

        let results = recommend_recall(&user_emb, &items, 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 1);  // Item 1 分数最高
        assert!((results[0].1 - 1.0).abs() < 1e-6);
    }
}
