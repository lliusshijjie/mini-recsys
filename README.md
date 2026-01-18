# Mini-RecSys

A hybrid **Rust/C++** recommendation system demo for learning purposes. This project demonstrates practical FFI (Foreign Function Interface) usage between Rust and C++.

## 📋 Project Overview

Mini-RecSys is an educational project that showcases:
- **Rust and C++ integration** via FFI (Foreign Function Interface)
- **Vector operations** implemented in C++ and called from Rust
- **Basic recommendation system architecture** with user embeddings and item scoring
- **Modern async web framework** using Axum (web API support)

## 🏗️ Architecture

The project is structured into several key components:

### Core Components

1. **FFI Layer** (`src/ffi.rs`)
   - Safe wrapper functions for C++ interop
   - Vector dot product computation via C++
   - Simple arithmetic operations for testing

2. **Data Models** (`src/model.rs`)
   - `User`: User representation with embeddings
   - `Item`: Product representation with embeddings
   - Serialization support via `serde`

3. **Business Logic** (`src/service.rs`)
   - `RecommendationService`: Core recommendation engine
   - Similarity scoring using dot products
   - Top-K item recommendation

4. **C++ Backend** (`cpp/`)
   - `vector_ops.cpp/h`: Vector operation implementations
   - Efficient dot product calculation
   - FFI-compatible C functions

## 🚀 Quick Start

### Prerequisites

- **Rust**: 1.70+ (install from [rustup.rs](https://rustup.rs/))
- **C++ Compiler**:
  - Windows: MSVC (Visual Studio)
  - Linux: GCC or Clang
  - macOS: Clang

### Building

```bash
# Build the project
cargo build --release

# Run the demo
cargo run --release
```

### Testing

```bash
# Run all tests
cargo test

# Run with verbose output
cargo test -- --nocapture
```

## 📊 How It Works

### Recommendation Flow

1. **User Embedding**: Each user has a vector representation (embedding)
2. **Item Embeddings**: Each item also has a vector representation
3. **Similarity Scoring**: Compute dot product between user and item embeddings
4. **Ranking**: Sort items by similarity score
5. **Top-K Selection**: Return the top-K most similar items

### FFI Integration

The Rust code calls C++ functions through the FFI layer:

```rust
// Rust calls C++ dot product calculation
ffi::compute_dot_product(&user.embedding, &item.embedding)
```

This demonstrates safe Rust-C++ interop with proper error handling and memory management.

## 📦 Dependencies

### Runtime
- **axum**: Modern, modular web framework for building web APIs
- **tokio**: Async runtime for concurrent operations
- **serde/serde_json**: JSON serialization and deserialization
- **libc**: C type definitions for FFI

### Build
- **cc**: Compile and link C/C++ code with Rust

## 📚 Project Structure

```
Mini-RecSys/
├── src/
│   ├── main.rs           # Entry point, demo workflow
│   ├── ffi.rs            # Rust-C++ FFI interface
│   ├── model.rs          # Data models (User, Item)
│   └── service.rs        # Business logic (Recommendations)
├── cpp/
│   ├── vector_ops.h      # C++ header files
│   └── vector_ops.cpp    # C++ implementation
├── build.rs              # Build script for C++ compilation
├── Cargo.toml            # Rust project manifest
└── README.md             # This file
```

## 🔧 Build Configuration

The `build.rs` script automatically:
- Detects available C++ compiler (GCC, Clang, MSVC)
- Compiles C++ code with C++17 standard
- Links the static library to the Rust binary
- Watches for changes in `cpp/` directory

## 💡 Learning Objectives

This project is designed to teach:
1. **Rust FFI fundamentals** - Safe and unsafe code boundaries
2. **C++ interoperability** - How to expose C functions for FFI
3. **Build system integration** - Using cc crate for compilation
4. **Recommendation systems basics** - Vector embeddings and similarity scoring
5. **Systems programming** - Memory safety, performance, and concurrency

## 🎯 Use Cases

While educational, the architecture can be extended for:
- Production recommendation systems
- Collaborative filtering engines
- Content-based recommendation systems
- Vector similarity search applications

## 📝 Example Usage

```rust
use mini_recsys::model::{User, Item};
use mini_recsys::service::RecommendationService;

// Create users and items with embeddings
let user = User::new(1, vec![1.0, 2.0, 3.0, 4.0]);
let items = vec![
    Item::new(1, "Product A", vec![5.0, 6.0, 7.0, 8.0]),
    Item::new(2, "Product B", vec![2.0, 3.0, 4.0, 5.0]),
];

// Create recommendation service
let service = RecommendationService::new(items);

// Get top-3 recommendations
let recommendations = service.recommend(&user, 3);
for (item, score) in recommendations {
    println!("Item: {}, Score: {}", item.name, score);
}
```

## 🤝 Contributing

This is an educational project. Feel free to:
- Extend the recommendation algorithm
- Add more C++ vector operations
- Implement web API endpoints with Axum
- Add comprehensive test cases

## 📄 License

This project is provided for educational purposes.

## 🔗 Resources

- [Rust FFI Guide](https://doc.rust-lang.org/nomicon/ffi.html)
- [Axum Web Framework](https://github.com/tokio-rs/axum)
- [cc Crate Documentation](https://docs.rs/cc/)
- [Recommendation Systems Basics](https://en.wikipedia.org/wiki/Recommender_system)

---

**Happy Learning!** 🎓
