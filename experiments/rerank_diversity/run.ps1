$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$Target = Join-Path $Root "target\rerank_diversity"
New-Item -ItemType Directory -Force -Path $Target | Out-Null

Push-Location $Root
try {
    & "$PSScriptRoot\generate_fixture.ps1"

    rustc --test experiments\rerank_diversity\rust_rerank_test.rs -O -o "$Target\rust_rerank_test.exe"
    & "$Target\rust_rerank_test.exe"

    g++ -std=c++17 -O3 -march=native `
        experiments\rerank_diversity\cpp_rerank_test.cpp `
        -I experiments\rerank_diversity `
        -o "$Target\cpp_rerank_test.exe"
    & "$Target\cpp_rerank_test.exe"

    rustc experiments\rerank_diversity\rust_bench.rs -O -C target-cpu=native -o "$Target\rust_bench.exe"
    g++ -std=c++17 -O3 -march=native `
        experiments\rerank_diversity\cpp_bench.cpp `
        -I experiments\rerank_diversity `
        -o "$Target\cpp_bench.exe"

    & "$Target\rust_bench.exe"
    & "$Target\cpp_bench.exe"
}
finally {
    Pop-Location
}
