// build.rs - Cargo build script
//
// The cc crate detects available C/C++ compilers (gcc, clang, msvc, etc.)
// and compiles C/C++ sources into a static library linked into the Rust binary.
//
// Workflow:
// 1. cc::Build configures compiler flags (C++ standard, optimization, etc.)
// 2. Specify .cpp source files to compile
// 3. Call .compile() to produce a static library (lib<name>.a or <name>.lib)
// 4. Cargo links that library into the final Rust binary

fn main() {
    // Compile C++ via the cc crate.
    cc::Build::new()
        // Enable C++17.
        .cpp(true)
        .flag_if_supported("-std=c++17") // GCC/Clang
        .flag_if_supported("/std:c++17") // MSVC
        // Header search paths.
        .include("cpp")
        .include("cpp/hnswlib")
        // C++ sources.
        .file("cpp/vector_ops.cpp")
        // Build static library "vector_ops"
        // (libvector_ops.a on Unix, vector_ops.lib on Windows).
        .compile("vector_ops");

    // Rebuild when anything under cpp/ changes.
    println!("cargo:rerun-if-changed=cpp/");
}
