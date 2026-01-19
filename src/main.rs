//! Mini-RecSys - 混合 Rust/C++ 推荐系统 Demo
//! 
//! 这是程序的入口点，演示了 Rust 调用 C++ 函数的基本流程。

mod ffi;
mod model;

fn main() {
    println!("=== Mini-RecSys: Rust/C++ FFI Demo ===\n");

    // 测试 1: 简单加法 (Hello World)
    println!("📝 测试 1: C++ 加法函数");
    let a = 42;
    let b = 58;
    let sum = ffi::add(a, b);
    println!("   {} + {} = {}", a, b, sum);
    println!("   ✅ FFI 调用成功!\n");

    // 测试 2: 向量点积运算
    println!("📝 测试 2: 向量点积计算");
    let vec_a = vec![1.0_f32, 2.0, 3.0, 4.0];
    let vec_b = vec![5.0_f32, 6.0, 7.0, 8.0];
    
    println!("   向量 A: {:?}", vec_a);
    println!("   向量 B: {:?}", vec_b);
    
    match ffi::compute_dot_product(&vec_a, &vec_b) {
        Some(result) => {
            // 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
            println!("   点积结果: {}", result);
            println!("   ✅ 向量运算成功!\n");
        }
        None => {
            println!("   ❌ 向量长度不匹配!");
        }
    }

    println!("=== 项目初始化完成 ===");
    println!("🚀 Rust + C++ FFI 编译流程验证通过!");
}
