# 모델 받기(GitHub 릴리스 models-v1 · models-v2) → sha256 확인 → models\ 에 풀기 → 빌드(윈도우, DirectML).
#   powershell -ExecutionPolicy Bypass -File scripts\setup.ps1 [-Train] [-Yolo] [-Board] [-NoBuild] [-Cpu]
#   -Board: 보드 NPU 모델(models-v2 board-npu-v2)도 models\board\ 에 받음(보드로 옮겨 쓸 때)
param([switch]$Train, [switch]$Yolo, [switch]$Board, [switch]$NoBuild, [switch]$Cpu)
$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')
$rel = 'https://github.com/juyoung020/ondevice-scene-memory/releases/download'
$assets = @('models-v1/objectsam-onnx.tar.gz', 'models-v1/siglip2-onnx.tar.gz')
if ($Train) { $assets += 'models-v1/train-weights.tar.gz' }
if ($Yolo) { $assets += 'models-v1/yolo26s-seg-onnx.tar.gz' }
if ($Board) { $assets += 'models-v2/board-npu-v2.tar.gz' }

$dl = 'models\.dl'
New-Item -ItemType Directory -Force $dl | Out-Null
$ProgressPreference = 'SilentlyContinue'
$sums = @{}
foreach ($tag in ($assets | ForEach-Object { ($_ -split '/')[0] } | Sort-Object -Unique)) {
    Invoke-WebRequest "$rel/$tag/SHA256SUMS" -OutFile "$dl\SHA256SUMS.$tag" -UseBasicParsing
    foreach ($l in Get-Content "$dl\SHA256SUMS.$tag") {
        $p = $l -split '\s+', 2
        $sums["$tag/" + $p[1].TrimStart('*')] = $p[0].ToLower()
    }
}
foreach ($a in $assets) {
    $f = ($a -split '/')[1]
    $path = Join-Path $dl $f
    $ok = (Test-Path $path) -and ((Get-FileHash $path -Algorithm SHA256).Hash.ToLower() -eq $sums[$a])
    if (-not $ok) {
        Write-Host "받는 중: $a"
        Invoke-WebRequest "$rel/$a" -OutFile $path -UseBasicParsing
    }
    $got = (Get-FileHash $path -Algorithm SHA256).Hash.ToLower()
    if ($got -ne $sums[$a]) { throw "sha256 다름: $f" }
    tar -xzf $path -C models
    if ($LASTEXITCODE -ne 0) { throw "풀기 실패: $f" }
    Write-Host "풀었음: $f"
}

if (-not $NoBuild) {
    if ($Cpu) { cargo build --release } else { cargo build --release --features directml }
    if ($LASTEXITCODE -ne 0) { throw 'cargo build 실패' }
    Get-Item target\release\ondevice.exe
}
