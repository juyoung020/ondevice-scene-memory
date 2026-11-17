#!/usr/bin/env bash
# 보드에서 물체 연결 · 병합 전(c745d30) · 후(이 판) 견주기(docs/object_merge.md "보드에서 잴 것").
#   bash scripts/board_merge_check.sh [TUM 폴더=~/om/data/tum/rgbd_dataset_freiburg1_desk] [횟수=2]
# 1. 속도 재기(maps_bench.rs, 물체 50 · 200 · 1,000 개의 관측 1 회 · 병합 1 회): 후 = 이 저장소, 전 = 예전 maps.rs · embed.rs 에 같은 재기 파일을 붙인 임시 작업 트리
# 2. 같은 녹화를 최대 속도로 전 · 후 번갈아(board_run.sh --tum, RATE=0): 물체 · 사라짐 · 병합 수 · kf/s · 물체 ms/장 · 병합 최대 ms
# 결과: /tmp/merge_check/ (요약 report.txt). 전 판 빌드는 따로 대상 폴더라 처음 한 번 오래 걸린다
set -euo pipefail
cd "$(dirname "$0")/.."
DATA=${1:-$HOME/om/data/tum/rgbd_dataset_freiburg1_desk}
N=${2:-2}
BASE=c745d30
OUTD=/tmp/merge_check
OLD=/tmp/om-merge-base
FEAT="--release --no-default-features --features dxrt -p ondevice --bin ondevice"
mkdir -p "$OUTD"
R=$OUTD/report.txt
: > "$R"
[ -f "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"

# 후 판
cargo build $FEAT
cp target/release/ondevice "$OUTD/ondevice_new"
cargo test $FEAT bench_memory -- --ignored --nocapture 2>&1 | grep '^bench' | sed 's/^/new /' | tee -a "$R"

# 전 판: 예전 소스 + 같은 재기 파일
if [ ! -d "$OLD" ]; then git worktree add --detach "$OLD" "$BASE"; fi
cp pipeline/src/maps_bench.rs "$OLD/pipeline/src/"
grep -q 'path = "maps_bench.rs"' "$OLD/pipeline/src/maps.rs" || printf '\n#[cfg(test)]\n#[path = "maps_bench.rs"]\nmod bench;\n' >> "$OLD/pipeline/src/maps.rs"
( cd "$OLD" && cargo build $FEAT && cp target/release/ondevice "$OUTD/ondevice_old" \
  && cargo test $FEAT bench_memory -- --ignored --nocapture 2>&1 | grep '^bench' | sed 's/^/old /' ) | tee -a "$R"

# 같은 녹화 전 · 후 번갈아
for r in $(seq 1 "$N"); do for b in old new; do
  O=$OUTD/${b}_$r
  RATE=0 BIN=$OUTD/ondevice_$b OUT=$O bash scripts/board_run.sh --tum "$DATA" --serve-exit > "$O.log" 2>&1 || true
  s=$(tail -1 "$O.log")
  g() { echo "$s" | grep -o "\"$1\":[0-9.]*" | tail -1 | cut -d: -f2; }
  echo "$b $r objects=$(g objects) gone=$(g gone) merged=$(g merged) kf_per_s=$(g kf_per_s) obj_ms=$(g obj) merge_max_ms=$(echo "$s" | grep -o '"max":{"merge":[0-9.]*' | cut -d: -f3)" | tee -a "$R"
done; done
echo "끝: $R"
