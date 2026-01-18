---
name: minirecsys
description: 推荐系统开发规范
---

# Project Context: Mini-RecSys (Hybrid Rust/C++)

## 1. 项目概述 (Project Overview)
这是一个用于学习目的的混合架构推荐系统 Demo。
- **核心目标**：深入理解 Rust 的所有权机制、并发模型以及 C++ 的内存管理和指针操作。
- **业务逻辑**：Rust 负责 Web 服务、数据管理和业务编排。
- **计算核心**：C++ 负责底层的向量相似度计算（Simulated High-Performance Engine）。
- **性能要求**：不追求极致的企业级性能，但追求逻辑正确性、内存安全和清晰的 FFI 边界设计。

## 2. 技术栈规范 (Tech Stack)

### Rust (Host)
- **Edition**: 2021
- **Web Framework**: Axum (追求模块化和人体工学)
- **Async Runtime**: Tokio
- **Serialization**: Serde (JSON)
- **FFI**: `libc` crate (用于定义 C 类型)
- **Build Tool**: Cargo (包含 `build.rs` 自定义构建脚本)

### C++ (Guest)
- **Standard**: C++17
- **Compiler**: GCC / Clang
- **Build System**: Makefile 或简单的 CMake (由 `build.rs` 调用)
- **External Interface**: `extern "C"` (纯 C ABI 兼容)

## 3. 代码风格与规范 (Coding Standards)

### 3.1 Rust 规范
- **错误处理**:
  - 严禁在 FFI 边界即使 panic。使用 `Result` 传播错误。
  - 仅在 `main` 初始化阶段或明确不可能发生的逻辑中使用 `.unwrap()`，其余情况尽量使用 `?` 操作符或 `expect("context")`。
- **Unsafe 使用**:
  - 所有的 `unsafe` 代码块必须伴随注释 `// SAFETY: ...`，详细解释为什么这里的指针解引用或类型转换是安全的（例如：生命周期由 Rust 保证，缓冲区长度已检查等）。
  - FFI 封装层必须保持独立，不要让 `unsafe` 逻辑泄露到业务层。
- **变量命名**: 遵循 Rust 官方规范 (Snake Case for functions/variables, Pascal Case for structs)。

### 3.2 C++ 规范
- **指针操作**:
  - 输入参数尽量使用 `const float*` 或 `const int*`，保证 C++ 端不会意外修改 Rust 的内存。
  - 在内部逻辑中，可以使用 `std::vector` 或 `std::algorithm` 来简化实现，但对外接口必须是原始指针。
- **内存管理**:
  - 遵循 "Who allocates, must free" 原则。
  - 本项目中，数据内存由 Rust 分配及持有，C++ 仅作为 View（借用）进行读取计算，不拥有所有权。

## 4. 架构设计与目录结构 (Architecture)

```text
/
├── Cargo.toml
├── build.rs          # 负责编译 C++ 代码并链接
├── src/
│   ├── main.rs       # 程序入口 & Web Server 启动
│   ├── ffi.rs        # FFI 接口定义与 Safe Wrapper (unsafe 代码主要在此)
│   ├── model.rs      # User/Item 结构体定义
│   └── service.rs    # 业务逻辑 (Recall -> Rank)
└── cpp/
    ├── vector_ops.h  # C++ 头文件
    └── vector_ops.cpp# C++ 核心实现 (Simd, Loop, Math)
```
## 5.FFI 交互规则 (Crucial)
AI 在生成代码时必须严格遵守以下内存交互协议：

### 5.1 数据布局 (Layout):

Rust 中的 Vec<f32> 传递给 C++ 时，使用 .as_ptr() 获取首地址，并显式传递长度 len。

所有的多维数组（如 Item 矩阵）在传递前必须在 Rust 端 Flatten（扁平化）为一维数组，C++ 端通过 index = i * dim + j 访问。

### 5.2 生命周期 (Lifetime):

Rust 必须保证在调用 C++ 函数期间，相关的数据（User 和 Item 列表）是被 Pin 住或未被 Drop 的。

### 5.3 类型对应:

Rust f32 <-> C++ float

Rust i32 <-> C++ int

Rust *const f32 <-> C++ const float*

## 6. 开发工作流指令 (Workflow Instructions)
**当你协助编写代码时，请遵循：**

### 6.1 优先定义接口：
先写 C++ 的 .h 文件和 Rust 的 ffi.rs 对应部分。

### 6.2 编写构建脚本：
确保 build.rs 能正确编译 C++ 代码并链接生成的静态/动态库。

### 6.3 实现核心逻辑：
编写 C++ 算法。

### 6.4 封装与测试：
在 Rust 中编写单元测试 (#[test]) 验证 FFI 调用的正确性，无需启动 Web Server 即可测试核心算法。

## 7. 学习增强 (Learning Enhancements)
**为了帮助用户学习：**

### 7.1 当生成复杂的指针操作代码时，请在注释中解释内存布局。

### 7.2 当使用 Rust 的 Arc<Mutex<...>> 时，解释为什么这里需要线程安全。

### 7.3 解释 build.rs 中 cc crate 的作用。