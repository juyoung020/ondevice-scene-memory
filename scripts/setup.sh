#!/usr/bin/env bash
# 모델 받기(GitHub 릴리스 models-v1 · models-v2) → sha256 확인 → models/ 에 풀기 → 빌드.
#   bash scripts/setup.sh            # PC(리눅스·WSL): 분할(ObjectSAM) + 사진·글 벡터(SigLIP2), 기본(CPU ONNX) 빌드
#   bash scripts/setup.sh --board    # 보드(Pi 5 + DX-M1): + NPU 모델(models-v2 board-npu-v2, ~/models 에도 복사), dxrt 빌드
#   bash scripts/setup.sh --train    # + 학습용 가중치(train/)
#   bash scripts/setup.sh --yolo     # + YOLO26s-seg ONNX(비교용)
#   bash scripts/setup.sh --no-build # 모델만
set -euo pipefail
cd "$(dirname "$0")/.."
REL=https://github.com/juyoung020/ondevice-scene-memory/releases/download
ASSETS=(models-v1/objectsam-onnx.tar.gz models-v1/siglip2-onnx.tar.gz)
BOARD=0; BUILD=1
for a in "$@"; do
  case $a in
    --board) BOARD=1; ASSETS+=(models-v2/board-npu-v2.tar.gz) ;;
    --train) ASSETS+=(models-v1/train-weights.tar.gz) ;;
    --yolo) ASSETS+=(models-v1/yolo26s-seg-onnx.tar.gz) ;;
    --no-build) BUILD=0 ;;
    *) echo "모르는 인자: $a" >&2; exit 2 ;;
  esac
done

mkdir -p models/.dl
cd models/.dl
for tag in $(printf '%s\n' "${ASSETS[@]}" | cut -d/ -f1 | sort -u); do
  curl -fsSL -o "SHA256SUMS.$tag" "$REL/$tag/SHA256SUMS"
done
for a in "${ASSETS[@]}"; do
  tag=${a%%/*}; f=${a#*/}
  want=$(grep " \*\?$f\$" "SHA256SUMS.$tag" | cut -d' ' -f1)
  if [ ! -f "$f" ] || [ "$(sha256sum "$f" | cut -d' ' -f1)" != "$want" ]; then
    echo "받는 중: $tag/$f"
    curl -fL --retry 3 -o "$f" "$REL/$tag/$f"
  fi
  got=$(sha256sum "$f" | cut -d' ' -f1)
  [ "$got" = "$want" ] || { echo "sha256 다름: $f" >&2; exit 1; }
  tar -xzf "$f" -C ..
  echo "풀었음: $f"
done
cd ../..

if [ $BOARD = 1 ]; then
  # 프로그램의 보드 기본 모델 폴더는 ~/models(ONDEVICE_MODELS 로 바꿈). 같은 이름의 다른 파일은 덮지 않는다
  mkdir -p "$HOME/models"
  for m in siglip2-vision_npu.dxnn objectsam-416_640.dxnn; do
    if [ ! -f "$HOME/models/$m" ]; then
      cp "models/board/$m" "$HOME/models/$m"; echo "복사: ~/models/$m"
    elif ! cmp -s "models/board/$m" "$HOME/models/$m"; then
      echo "~/models/$m 이 릴리스 파일과 다름(그대로 둠): 릴리스 판은 models/board/$m" >&2
    fi
  done
fi

if [ $BUILD = 1 ]; then
  [ -f "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"
  if [ $BOARD = 1 ]; then
    cargo build --release --no-default-features --features dxrt
  else
    cargo build --release
  fi
  ls -la target/release/ondevice
fi
