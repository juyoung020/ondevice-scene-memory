#!/bin/bash
# 보드(Pi 5)에서 프로필 기반 최적화(PGO) 빌드. LLVM 이 실제 실행 경로로 코드 배치·인라인을 고른다(계산 식은 그대로라 출력 값이 같다).
# 쓰기: scripts/pgo.sh <녹화 폴더(TUM 형식)> [추가 ondevice 인자]
#   1) 계측 빌드 → 2) 녹화 한 벌을 돌려 프로필 모음 → 3) 프로필을 합쳐 최종 빌드(target/release/ondevice)
# 필요: rustup component add llvm-tools (llvm-profdata). 기본 특징(--features dxrt 등)은 FEAT 로: FEAT="--no-default-features --features dxrt"
set -euo pipefail
REC=${1:?녹화 폴더}; shift
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PROF=${PROF:-/tmp/ondevice-pgo}
FEAT=${FEAT:-}
SYSROOT=$(rustc --print sysroot)
PROFDATA=$(find "$SYSROOT" -name llvm-profdata -type f | head -1)
[ -x "$PROFDATA" ] || { echo "llvm-profdata 없음: rustup component add llvm-tools"; exit 1; }
rm -rf "$PROF"; mkdir -p "$PROF"
cd "$ROOT"
RUSTFLAGS="-C target-cpu=native -C profile-generate=$PROF" cargo build --release $FEAT --target-dir target/pgo-gen
INI=${INI:-$ROOT/docs/rtabmap_pi.ini}
./target/pgo-gen/release/ondevice --slam "--tum $REC --rate 0 --block --config $INI" --no-drop --out "$PROF/out" "$@" > /dev/null
"$PROFDATA" merge -o "$PROF/merged.profdata" "$PROF"/*.profraw
RUSTFLAGS="-C target-cpu=native -C profile-use=$PROF/merged.profdata -C llvm-args=-pgo-warn-missing-function" cargo build --release $FEAT
echo "PGO 빌드: $ROOT/target/release/ondevice (프로필 $PROF/merged.profdata)"
