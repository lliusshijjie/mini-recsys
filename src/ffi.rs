//! FFI 接口定义与 Safe Wrapper
//! 
//! 这个模块是 Rust 与 C++ 交互的边界层。
//! 所有的 `unsafe` 代码都集中在这里，业务层不应该直接接触 unsafe。

use libc::{c_float, c_int};

// ============================================================================
// 外部 C 函数声明 (Raw FFI Bindings)
// ============================================================================

extern "C" {
    /// 简单的加法函数 - 用于验证 FFI 流程
    fn cpp_add(a: c_int, b: c_int) -> c_int;

    /// 计算两个向量的点积
    fn dot_product(vec_a: *const c_float, vec_b: *const c_float, len: c_int) -> c_float;
}

// ============================================================================
// Safe Wrapper Functions
// ============================================================================

/// 安全的加法函数封装
/// 
/// # Arguments
/// * `a` - 第一个整数
/// * `b` - 第二个整数
/// 
/// # Returns
/// 两数之和
pub fn add(a: i32, b: i32) -> i32 {
    // SAFETY: cpp_add 是一个纯函数，不涉及指针操作，
    // c_int 与 i32 在当前平台上是兼容的。
    unsafe { cpp_add(a, b) }
}

/// 安全的点积计算封装
/// 
/// # Arguments
/// * `vec_a` - 第一个向量
/// * `vec_b` - 第二个向量
/// 
/// # Returns
/// * `Some(result)` - 点积结果
/// * `None` - 如果向量长度不匹配
/// 
/// # 内存安全说明
/// Rust 的 Vec<f32> 在调用期间保持有效（不会被 drop 或 reallocate），
/// 因此传递给 C++ 的指针在整个 FFI 调用期间都是有效的。
pub fn compute_dot_product(vec_a: &[f32], vec_b: &[f32]) -> Option<f32> {
    // 检查向量长度是否匹配
    if vec_a.len() != vec_b.len() {
        return None;
    }

    let len = vec_a.len() as c_int;

    // SAFETY:
    // 1. vec_a 和 vec_b 是有效的 Rust 切片，其底层内存在函数调用期间不会被释放
    // 2. as_ptr() 返回的指针指向切片的首元素，保证连续内存布局
    // 3. len 参数正确反映了切片的实际长度，C++ 不会越界访问
    // 4. C++ 函数声明参数为 const float*，不会修改 Rust 的内存
    let result = unsafe { dot_product(vec_a.as_ptr(), vec_b.as_ptr(), len) };

    Some(result)
}

// ============================================================================
// 单元测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpp_add() {
        // 测试基本加法
        assert_eq!(add(2, 3), 5);
        assert_eq!(add(-1, 1), 0);
        assert_eq!(add(0, 0), 0);
        assert_eq!(add(100, 200), 300);
    }

    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        
        // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        let result = compute_dot_product(&a, &b);
        assert!(result.is_some());
        assert!((result.unwrap() - 32.0).abs() < 1e-6);
    }

    #[test]
    fn test_dot_product_length_mismatch() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];
        
        // 长度不匹配应返回 None
        assert!(compute_dot_product(&a, &b).is_none());
    }
}
