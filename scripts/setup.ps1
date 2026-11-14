# 모델 받기(GitHub 릴리스 models-v1) → sha256 확인 → models\ 에 풀기 → 빌드(윈도우, DirectML).
#   powershell -ExecutionPolicy Bypass -File scripts\setup.ps1 [-Train] [-NoBuild] [-Cpu]
param([switch]$Train, [switch]$NoBuild, [switch]$Cpu)
$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')
$url = 'https://github.com/juyoung020/ondevice-scene-memory/releases/download/models-v1'
$assets = @('yolo26s-seg-onnx.tar.gz', 'objectsam-onnx.tar.gz', 'siglip2-onnx.tar.gz')
if ($Train) { $assets += 'train-weights.tar.gz' }

$dl = 'models\.dl'
New-Item -ItemType Directory -Force $dl | Out-Null
Invoke-WebRequest "$url/SHA256SUMS" -OutFile "$dl\SHA256SUMS" -UseBasicParsing
$sums = @{}
foreach ($l in Get-Content "$dl\SHA256SUMS") {
    $p = $l -split '\s+', 2
    $sums[$p[1].TrimStart('*')] = $p[0].ToLower()
}
foreach ($f in $assets) {
    $path = Join-Path $dl $f
    $ok = (Test-Path $path) -and ((Get-FileHash $path -Algorithm SHA256).Hash.ToLower() -eq $sums[$f])
    if (-not $ok) {
        Write-Host "받는 중: $f"
        $ProgressPreference = 'SilentlyContinue'
        Invoke-WebRequest "$url/$f" -OutFile $path -UseBasicParsing
    }
    $got = (Get-FileHash $path -Algorithm SHA256).Hash.ToLower()
    if ($got -ne $sums[$f]) { throw "sha256 다름: $f" }
    tar -xzf $path -C models
    if ($LASTEXITCODE -ne 0) { throw "풀기 실패: $f" }
    Write-Host "풀었음: $f"
}

if (-not $NoBuild) {
    if ($Cpu) { cargo build --release } else { cargo build --release --features directml }
    if ($LASTEXITCODE -ne 0) { throw 'cargo build 실패' }
    Get-Item target\release\ondevice.exe
}
