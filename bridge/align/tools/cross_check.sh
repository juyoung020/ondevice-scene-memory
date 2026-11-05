#!/usr/bin/env bash
# WSL(Ubuntu 24.04)에서: x86-64 GCC 빌드·시험, aarch64(Cortex-A76) 크로스 빌드·qemu 시험, NEON 명령 확인.
#   bash bridge/align/tools/cross_check.sh            (필요: g++ g++-aarch64-linux-gnu qemu-user cmake ninja-build)
# 빌드 산출물은 ~/build/align 아래.
set -euo pipefail
SRC="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$HOME/build/align"
mkdir -p "$OUT"

echo "== x86-64 GCC =="
cmake -S "$SRC" -B "$OUT/x86" -G Ninja -DCMAKE_BUILD_TYPE=Release -DFA_TOOLS=OFF >/dev/null
ninja -C "$OUT/x86" >/dev/null
"$OUT/x86/test_align" > "$OUT/x86_test.txt" || true
tail -1 "$OUT/x86_test.txt"

echo "== aarch64 크로스(-mcpu=cortex-a76) =="
cmake -S "$SRC" -B "$OUT/a64" -G Ninja -DCMAKE_BUILD_TYPE=Release -DFA_TOOLS=OFF \
  -DCMAKE_TOOLCHAIN_FILE="$SRC/cmake/aarch64-linux-gnu.cmake" >/dev/null
ninja -C "$OUT/a64" >/dev/null
QEMU_LD_PREFIX=/usr/aarch64-linux-gnu qemu-aarch64 "$OUT/a64/test_align" > "$OUT/a64_test.txt" || true
tail -1 "$OUT/a64_test.txt"
echo "이식본과 다른 화소가 0 인 줄(정확한 식): $(grep -v '빠른 식' "$OUT/a64_test.txt" | grep -c '다른 화소 0 ') / $(grep -v '빠른 식' "$OUT/a64_test.txt" | grep -c '다른 화소')"

echo "== aarch64 핵심 루프의 NEON 명령(fast_align.cpp.o) =="
OBJ=$(find "$OUT/a64" -name 'fast_align.cpp.o' | head -1)
aarch64-linux-gnu-objdump -d --no-show-raw-insn "$OBJ" > "$OUT/a64_fast_align.S"
for ins in 'fdiv\s+v[0-9]+\.4s' 'fmla\s+v[0-9]+\.4s' 'fcvtzs\s+v[0-9]+\.4s' 'ucvtf\s+v[0-9]+\.4s|scvtf\s+v[0-9]+\.4s' 'fmul\s+v[0-9]+\.4s'; do
  printf '  %-36s %s\n' "$ins" "$(grep -cE "$ins" "$OUT/a64_fast_align.S")"
done
