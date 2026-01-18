// vector_ops.h - C++ 头文件
// 定义对外暴露的 C ABI 兼容接口
//
// 注意：使用 extern "C" 确保函数名不会被 C++ 名称修饰 (name mangling)，
// 这样 Rust 才能通过 FFI 正确找到并调用这些函数。

#ifndef VECTOR_OPS_H
#define VECTOR_OPS_H

#ifdef __cplusplus
extern "C" {
#endif

/// 简单的加法函数 - 用于验证 FFI 流程
/// @param a 第一个整数
/// @param b 第二个整数
/// @return a + b 的结果
int cpp_add(int a, int b);

/// 计算两个向量的点积 (内积)
/// 
/// 内存布局说明：
/// - vec_a 和 vec_b 是由 Rust 分配的连续内存块
/// - len 表示向量的元素个数
/// - C++ 端仅读取数据，不修改也不释放内存
///
/// @param vec_a 第一个向量的首地址 (只读)
/// @param vec_b 第二个向量的首地址 (只读)
/// @param len 向量长度
/// @return 点积结果
float dot_product(const float* vec_a, const float* vec_b, int len);

#ifdef __cplusplus
}
#endif

#endif // VECTOR_OPS_H
