#!/usr/bin/env bash
# 모델 받기(GitHub 릴리스 models-v1) → sha256 확인 → models/ 에 풀기 → 빌드.
#   bash scripts/setup.sh            # PC(리눅스·WSL): 분할 + 검색 모델, 기본(CPU ONNX) 빌드
#   bash scripts/setup.sh --board    # 보드(Pi 5 + DX-M1): + NPU 모델, dxrt 빌드
#   bash scripts/setup.sh --train    # + 학습용 가중치(train/)
#   bash scripts/setup.sh --no-build # 모델만
set -euo pipefail
cd "$(dirname "$0")/.."
URL=https://github.com/juyoung020/ondevice-scene-memory/releases/download/models-v1
ASSETS=(yolo26s-seg-onnx.tar.gz objectsam-onnx.tar.gz siglip2-onnx.tar.gz)
BOARD=0; BUILD=1
for a in "$@"; do
  case $a in
    --board) BOARD=1; ASSETS+=(board-npu.tar.gz) ;;
    --train) ASSETS+=(train-weights.tar.gz) ;;
    --no-build) BUILD=0 ;;
    *) echo "모르는 인자: $a" >&2; exit 2 ;;
  esac
done

mkdir -p models/.dl
cd models/.dl
curl -fsSL -o SHA256SUMS "$URL/SHA256SUMS"
for f in "${ASSETS[@]}"; do
  want=$(grep " \*\?$f\$" SHA256SUMS | cut -d' ' -f1)
  if [ ! -f "$f" ] || [ "$(sha256sum "$f" | cut -d' ' -f1)" != "$want" ]; then
    echo "받는 중: $f"
    curl -fL --retry 3 -o "$f" "$URL/$f"
  fi
  got=$(sha256sum "$f" | cut -d' ' -f1)
  [ "$got" = "$want" ] || { echo "sha256 다름: $f" >&2; exit 1; }
  tar -xzf "$f" -C ..
  echo "풀었음: $f"
done
cd ../..

if [ $BUILD = 1 ]; then
  [ -f "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"
  if [ $BOARD = 1 ]; then
    cargo build --release --no-default-features --features dxrt
  else
    cargo build --release
  fi
  ls -la target/release/ondevice
fi
