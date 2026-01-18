// vector_ops.cpp - C++ 核心实现
// 包含向量运算的具体实现

#include "vector_ops.h"

/// 简单的加法函数 - 用于验证 FFI 流程是否正常工作
/// 这是一个最基础的 "Hello World" 级别的函数
extern "C" int cpp_add(int a, int b) {
    return a + b;
}

/// 计算两个向量的点积 (内积)
/// 
/// 内存布局：
/// ┌─────────────────────────────────────────┐
/// │ vec_a: [a0, a1, a2, ..., a_{len-1}]     │ <- Rust Vec<f32> 的底层内存
/// │ vec_b: [b0, b1, b2, ..., b_{len-1}]     │ <- Rust Vec<f32> 的底层内存
/// └─────────────────────────────────────────┘
/// 
/// 点积公式: result = Σ(a_i * b_i) for i in 0..len
///
extern "C" float dot_product(const float* vec_a, const float* vec_b, int len) {
    float result = 0.0f;
    
    // 简单的循环实现
    // 注意：输入指针是 const，确保不会修改 Rust 端的内存
    for (int i = 0; i < len; ++i) {
        result += vec_a[i] * vec_b[i];
    }
    
    return result;
}
