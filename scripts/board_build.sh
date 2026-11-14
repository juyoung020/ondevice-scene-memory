#!/usr/bin/env bash
# 보드(Raspberry Pi 5 + DX-M1)에서 한 번에: 저장소 받기(또는 최신으로) → ondevice 빌드(NPU dxrt, Cortex-A76 맞춤).
#   bash scripts/board_build.sh [브랜치=main] [저장소 폴더=~/ondevice]
# 끝나면 실행 파일: <저장소>/target/release/ondevice
set -euo pipefail
BRANCH=${1:-main}
DIR=${2:-$HOME/ondevice}
URL=https://github.com/juyoung020/ondevice-scene-memory.git

if [ -d "$DIR/.git" ]; then
  git -C "$DIR" fetch origin "$BRANCH"
  git -C "$DIR" checkout "$BRANCH"
  git -C "$DIR" reset --hard "origin/$BRANCH"
else
  git clone --branch "$BRANCH" "$URL" "$DIR"
fi
cd "$DIR"
[ -f "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"
# .cargo/config.toml 이 aarch64 에서 target-cpu=cortex-a76 을 건다
cargo build --release --no-default-features --features dxrt
ls -la target/release/ondevice
echo "빌드 끝: $(git rev-parse --short HEAD) ($BRANCH)"
