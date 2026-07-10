# Download ONNX Runtime for local Windows development (load-dynamic mode).
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Version = "1.23.2"
$DestDir = Join-Path $Root "libs\onnxruntime"
$DllPath = Join-Path $DestDir "onnxruntime.dll"

if (Test-Path $DllPath) {
    Write-Host "ONNX Runtime already present at $DllPath"
} else {
    $ZipPath = Join-Path $env:TEMP "onnxruntime-win-x64-$Version.zip"
    $Url = "https://github.com/microsoft/onnxruntime/releases/download/v$Version/onnxruntime-win-x64-$Version.zip"
    Write-Host "Downloading $Url"
    Invoke-WebRequest -Uri $Url -OutFile $ZipPath -UseBasicParsing
    $ExtractDir = Join-Path $env:TEMP "onnxruntime-win-x64-$Version"
    Expand-Archive -Path $ZipPath -DestinationPath (Split-Path $ExtractDir) -Force
    New-Item -ItemType Directory -Force -Path $DestDir | Out-Null
    Copy-Item (Join-Path $ExtractDir "lib\onnxruntime.dll") $DestDir -Force
    Write-Host "Installed $DllPath"
}

Write-Host ""
Write-Host "Run the server with:"
Write-Host "  `$env:ORT_DYLIB_PATH = `"$DllPath`""
Write-Host "  cargo run --release"
