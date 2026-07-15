$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$Target = Join-Path $Root "target\recall_kernels"
New-Item -ItemType Directory -Force -Path $Target | Out-Null

Push-Location $Root
try {
    rustc --test experiments\recall_kernels\rust_kernels_test.rs -O -o "$Target\rust_kernels_test.exe"
    & "$Target\rust_kernels_test.exe"

    g++ -std=c++17 -O3 -march=native experiments\recall_kernels\cpp_kernels_test.cpp -o "$Target\cpp_kernels_test.exe"
    & "$Target\cpp_kernels_test.exe"

    rustc experiments\recall_kernels\rust_bench.rs -O -C target-cpu=native -o "$Target\rust_bench.exe"
    g++ -std=c++17 -O3 -march=native experiments\recall_kernels\cpp_bench.cpp -o "$Target\cpp_bench.exe"

    & "$Target\rust_bench.exe"
    & "$Target\cpp_bench.exe"
}
finally {
    Pop-Location
}
