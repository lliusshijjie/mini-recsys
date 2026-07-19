$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$ProductsPath = Join-Path $Root "assets\products.json"
$OutDir = Join-Path $PSScriptRoot "data"
$OutPath = Join-Path $OutDir "candidates.csv"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

if (-not (Test-Path $ProductsPath)) {
    throw "Missing $ProductsPath"
}

$Dim = 32
$Take = 128
$products = Get-Content -Raw -Path $ProductsPath | ConvertFrom-Json
if ($products.Count -lt 1) {
    throw "products.json is empty"
}

$categories = @{}
$categoryIndex = 0
$rows = New-Object System.Collections.Generic.List[string]
$header = "id,score,category," + ((0..($Dim - 1) | ForEach-Object { "e$_" }) -join ",")
$rows.Add($header) | Out-Null

$slice = $products | Select-Object -First $Take
$i = 0
foreach ($p in $slice) {
    if (-not $categories.ContainsKey($p.category)) {
        $categories[$p.category] = $categoryIndex
        $categoryIndex += 1
    }
    $catId = [int]$categories[$p.category]
    $emb = New-Object float[] $Dim
    $base = ($catId % 6) * 5
    for ($d = 0; $d -lt 5; $d++) {
        if (($base + $d) -lt $Dim) {
            $emb[$base + $d] = 1.0
        }
    }
    # Item-level perturbation from id keeps same-category sims < 1.
    $emb[[int]($p.id % $Dim)] += 0.05 + (($p.id % 17) * 0.001)

    $norm = 0.0
    foreach ($x in $emb) { $norm += ($x * $x) }
    $norm = [math]::Sqrt($norm)
    if ($norm -gt 1e-12) {
        for ($d = 0; $d -lt $Dim; $d++) { $emb[$d] = $emb[$d] / $norm }
    }

    # Descending synthetic rank score (stand-in for final_score).
    $score = [math]::Max(0.01, 1.0 - ($i * 0.004))
    $embText = ($emb | ForEach-Object { $_.ToString("G9", [cultureinfo]::InvariantCulture) }) -join ","
    $cat = ($p.category -replace ",", " ")
    $rows.Add("$($p.id),$($score.ToString("G9", [cultureinfo]::InvariantCulture)),$cat,$embText") | Out-Null
    $i += 1
}

$utf8NoBom = New-Object System.Text.UTF8Encoding $false
[System.IO.File]::WriteAllLines($OutPath, $rows, $utf8NoBom)
Write-Host "Wrote $($rows.Count - 1) candidates to $OutPath"
