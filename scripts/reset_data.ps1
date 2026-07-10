# Reset persisted mini-recsys data (Sled DB, HNSW index, Tantivy index).
# Usage: .\scripts\reset_data.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $Root "data"

$Targets = @(
    (Join-Path $DataDir "db"),
    (Join-Path $DataDir "index.bin"),
    (Join-Path $DataDir "tantivy_index")
)

foreach ($Target in $Targets) {
    if (Test-Path $Target) {
        Remove-Item -Recurse -Force $Target
        Write-Host "Removed $Target"
    } else {
        Write-Host "Skip missing $Target"
    }
}

Write-Host ""
Write-Host "Done. Restart the server to rebuild from assets/products.json:"
Write-Host "  cargo run --release"
