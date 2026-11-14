#!/bin/bash
# aarch64(보드) 판을 PC 에서 교차 빌드하고 qemu-aarch64 로 시험을 돌린다: 손으로 짠 NEON 판이 스칼라 판과 비트까지 같은지,
# 그리고 aarch64 + cortex-a76 빌드에서 나머지 시험이 모두 맞는지.
# 필요(우분투/WSL): sudo apt install gcc-aarch64-linux-gnu qemu-user; rustup target add aarch64-unknown-linux-gnu
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc
export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_RUNNER="qemu-aarch64 -cpu max -L /usr/aarch64-linux-gnu"
export CC_aarch64_unknown_linux_gnu=aarch64-linux-gnu-gcc
cargo test --release --target aarch64-unknown-linux-gnu -p common -p slam -p rsdriver -p tools "$@"
