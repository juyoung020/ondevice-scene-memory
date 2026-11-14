#!/bin/bash
# 보드(Raspberry Pi 5 + DX-M1) 실행 파일을 PC(우분투/WSL)에서 교차 빌드하고 보드로 보낸다(원하면 바로 실행).
#   [PROFILE=dev-fast] [SEND=0(빌드만)] [PGO=gen|use:<profdata>|auto:<보드 녹화>] bash scripts/cross_board.sh [보드=kau@192.168.219.101] [보드에 둘 경로=~/ondevice-cross] [-- 보드에서 돌릴 인자…]
# 필요: sudo apt install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu; rustup target add aarch64-unknown-linux-gnu
# 처음 한 번 보드의 DXRT 런타임(libdxrt.so)을 ~/board-sysroot/lib 로 받는다(저장소 밖). 링크에 그것만 쓰고,
# libdxrt 가 부르는 onnxruntime 등은 보드에서 실행할 때 찾는다(--allow-shlib-undefined).
# PC 우분투 24.04 교차 glibc(2.39) ≤ 보드 Debian 13 glibc(2.41) 라 그대로 돈다. target-cpu=cortex-a76 은 아래 RUSTFLAGS 에 같이 건다(.cargo/config.toml 과 같음).
set -euo pipefail
BOARD=${1:-kau@192.168.219.101}
DEST=${2:-ondevice-cross}
shift $(( $# < 2 ? $# : 2 ))
[ "${1:-}" = "--" ] && shift
ROOT=${ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
SYS=${BOARD_SYSROOT:-$HOME/board-sysroot}
if [ ! -f "$SYS/lib/libdxrt.so" ]; then
  mkdir -p "$SYS/lib"
  scp -q "$BOARD:/usr/local/lib/libdxrt.so.3.4.0" "$SYS/lib/"
  ln -sf libdxrt.so.3.4.0 "$SYS/lib/libdxrt.so.3"
  ln -sf libdxrt.so.3 "$SYS/lib/libdxrt.so"
fi
cd "$ROOT"
[ -f "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"
export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc
export CC_aarch64_unknown_linux_gnu=aarch64-linux-gnu-gcc
export CXX_aarch64_unknown_linux_gnu=aarch64-linux-gnu-g++
# 이 환경 변수는 .cargo/config.toml 의 rustflags 를 대신하므로 target-cpu 를 함께 적음
export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_RUSTFLAGS="-C target-cpu=cortex-a76 -L $SYS/lib -C link-arg=-Wl,--allow-shlib-undefined"
PROFILE=${PROFILE:-release}
# PGO(프로필 기반 최적화, 계산 식은 그대로라 출력 값이 같다):
#   PGO=gen  → 계측 빌드(보드에서 돌리면 ~/pgo/*.profraw 가 생김, 별도 target 폴더)
#   PGO=use:<합친 .profdata> → 그 프로필로 최종 빌드
#   PGO=auto:<보드의 TUM 녹화 폴더> → 한 번에(계측 빌드 → 보드에서 녹화 한 벌 → 프로필 받아 합침 → 최종 빌드·보냄)
if [ "${PGO:-}" != "" ] && [ "${PGO%%:*}" = auto ]; then
  REC=${PGO#auto:}
  PD=$(find "$(rustc --print sysroot)" -name llvm-profdata -type f | head -1)
  [ -x "$PD" ] || { echo "llvm-profdata 없음: rustup component add llvm-tools"; exit 1; }
  PGO=gen SEND=0 bash "$0" "$BOARD" "$DEST"
  scp -q "${CARGO_TARGET_DIR:-target}/pgo-gen/aarch64-unknown-linux-gnu/$PROFILE/ondevice" "$BOARD:ondevice-pgo-gen"
  ssh "$BOARD" "rm -rf ~/pgo && mkdir -p ~/pgo && chmod +x ~/ondevice-pgo-gen && cd ~/ondevice-scene-memory &&     ~/ondevice-pgo-gen --slam '--tum $REC --rate 0 --block --config docs/rtabmap_pi.ini --send-every 2' --no-drop --every 1       --seg ~/models/objectsam-416_640.dxnn --embed ~/models/siglip2-vision_npu.dxnn --vocab vocab/indoor.voc --max-depth 3.5 --out /tmp/pgo-out > /dev/null 2>&1;     rm -rf /tmp/pgo-out ~/ondevice-pgo-gen"
  rm -rf /tmp/ondevice-pgo && mkdir -p /tmp/ondevice-pgo && scp -q "$BOARD:pgo/*.profraw" /tmp/ondevice-pgo/
  "$PD" merge -o /tmp/ondevice-pgo/merged.profdata /tmp/ondevice-pgo/*.profraw
  PGO=use:/tmp/ondevice-pgo/merged.profdata exec bash "$0" "$BOARD" "$DEST"
fi
case "${PGO:-}" in
  gen) CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_RUSTFLAGS+=" -C profile-generate=/home/${BOARD%%@*}/pgo"; export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-target}/pgo-gen ;;
  use:*) CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_RUSTFLAGS+=" -C profile-use=${PGO#use:} -C llvm-args=-pgo-warn-missing-function=false" ;;
esac
# mold 가 있으면 링크를 mold 로(빠름)
if command -v mold >/dev/null; then CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_RUSTFLAGS+=" -C link-arg=-fuse-ld=mold"; fi
if command -v sccache >/dev/null; then export RUSTC_WRAPPER=sccache; fi
time cargo build --profile $PROFILE --target aarch64-unknown-linux-gnu --no-default-features --features dxrt
BIN=${CARGO_TARGET_DIR:-target}/aarch64-unknown-linux-gnu/$PROFILE/ondevice
[ "${SEND:-1}" = 0 ] && { echo "빌드만: $BIN"; exit 0; }
scp -q "$BIN" "$BOARD:$DEST" && ssh "$BOARD" chmod +x "$DEST"
echo "보냄: $BOARD:$DEST ($(git rev-parse --short HEAD))"
if [ $# -gt 0 ]; then
  ssh "$BOARD" "cd ~/ondevice-scene-memory && ~/$DEST $*"
fi
