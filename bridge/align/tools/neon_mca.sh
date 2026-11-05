#!/usr/bin/env bash
# aarch64 빌드(cross_check.sh 다음)의 벡터 계산 루프를 llvm-mca(Neoverse-N1 모형 = Cortex-A76 에서 나온 코어)로 재서
# 깊이 화소당 주기를 낸다. 루프 = fdiv v.4s 를 품은 뒤로 가는 분기 구간 중 명령 100 개 이하(가장 안쪽).
# 화소 수: fdiv 1 개 = 4 화소(빠른 식), 그 밖에는 fdiv 2 개(X/Z, Y/Z) = 4 화소.
#   bash bridge/align/tools/neon_mca.sh        (필요: llvm-18)
set -euo pipefail
BIN="$HOME/build/align/a64/bench_align"
MCA=${MCA:-llvm-mca-18}
DIS="$HOME/build/align/a64_bench.S"
aarch64-linux-gnu-objdump -d --no-show-raw-insn -C "$BIN" > "$DIS"

loops() {  # $1 = 함수 이름 일부
  awk -v pat="$1" '
    /^[0-9a-f]+ <.*>:$/ { infn = index($0, pat) > 0; n = 0; next }
    infn && /^ +[0-9a-f]+:/ {
      a = $1; sub(":", "", a); addr[n] = strtonum("0x" a); line = $0; sub(/^ +[0-9a-f]+:[ \t]+/, "", line); ins[n] = line; n++
      if (match(line, /^b(\.[a-z]+)?[ \t]+([0-9a-f]+)/, m) || match(line, /^(cbnz|cbz)[ \t]+[a-z0-9]+,[ \t]*([0-9a-f]+)/, m)) {
        t = strtonum("0x" m[2])
        if (t < addr[n-1]) {
          s = -1; has = 0
          for (i = 0; i < n; i++) if (addr[i] == t) s = i
          if (s >= 0 && n - s <= 100) {
            for (i = s; i < n; i++) if (ins[i] ~ /fdiv[ \t]+v/) has++
            if (has > 0) { printf "@@LOOP %d\n", has; for (i = s; i < n; i++) { x = ins[i]; sub(/ <.*>/, "", x); print x } }
          }
        }
      }
    }' "$DIS"
}

measure() {
  local name="$1" pat="$2"
  loops "$pat" | awk -v name="$name" -v mca="$MCA" '
    /^@@LOOP/ { if (body != "") run(); fd = $2; body = ""; next }
    { body = body $0 "\n" }
    END { if (body != "") run() }
    function run(   f, cmd, l, cyc, it, ins, px, key) {
      f = "/tmp/mca_loop.s"; printf "%s", body > f; close(f)
      cmd = mca " -mtriple=aarch64 -mcpu=neoverse-n1 -iterations=500 " f " 2>/dev/null"
      cyc = 0; it = 1; ins = 0
      while ((cmd | getline l) > 0) {
        if (l ~ /^Iterations:/) { split(l, a, ":"); it = a[2] + 0 }
        if (l ~ /^Instructions:/) { split(l, a, ":"); ins = a[2] + 0 }
        if (l ~ /^Total Cycles:/) { split(l, a, ":"); cyc = a[2] + 0 }
      }
      close(cmd)
      px = (fd == 1) ? 4 : fd * 2
      key = sprintf("%d/%d", ins / it, fd)
      if (!(key in seen)) { seen[key] = 1
        printf "  %-34s 명령 %3d, fdiv %d, 화소 %d: 반복당 %5.1f 주기 → 화소당 %.2f 주기\n", name, ins / it, fd, px, cyc / it, cyc / it / px }
    }'
}

echo "llvm-mca(Neoverse-N1 모형), 벡터 계산 루프(흩뿌리기 제외). 같은 이름의 여러 줄 = 왜곡·표 모양·빠른 식 변형"
measure "librealsense NEON get_texture_map" "main::{lambda()#1}>::_M_invoke"
measure "fast_align Sdk(번호까지 계산)" "Impl::sdk_row<true>"
