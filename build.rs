// build.rs - 构建脚本
// 
// cc crate 的作用：
// cc 是一个 Cargo 构建辅助 crate，它能够自动检测系统中可用的 C/C++ 编译器
// (如 gcc, clang, msvc) 并调用它们来编译 C/C++ 源文件。
// 编译产生的静态库会自动链接到最终的 Rust 可执行文件中。
//
// 工作流程：
// 1. cc::Build 配置编译器选项（如 C++ 标准、优化级别等）
// 2. 指定要编译的 .cpp 源文件
// 3. 调用 .compile() 生成静态库 (lib<name>.a 或 <name>.lib)
// 4. Cargo 自动将该静态库链接到 Rust 二进制文件

fn main() {
    // 使用 cc crate 编译 C++ 代码
    cc::Build::new()
        // 启用 C++17 标准
        .cpp(true)
        .flag_if_supported("-std=c++17")    // GCC/Clang
        .flag_if_supported("/std:c++17")    // MSVC
        // 添加头文件搜索路径
        .include("cpp")
        // 指定要编译的 C++ 源文件
        .file("cpp/vector_ops.cpp")
        // 编译并生成静态库，名称为 "vector_ops"
        // 这会生成 libvector_ops.a (Unix) 或 vector_ops.lib (Windows)
        .compile("vector_ops");

    // 告诉 Cargo 当 cpp/ 目录下的文件变化时重新编译
    println!("cargo:rerun-if-changed=cpp/");
}
