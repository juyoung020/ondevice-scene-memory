#!/usr/bin/env bash
# PC(WSL) 바이트 비교 세트: 구조를 바꾼 판이 결과를 바꾸지 않았는지 본다(설계 4.3 절).
#   bash scripts/pc_bytes.sh run <ondevice 실행 파일> <결과 폴더> [POOL=3]   세트 전부를 한 판으로 돌려 결과 폴더에
#   bash scripts/pc_bytes.sh mem <ondevice 실행 파일> <결과 폴더> [POOL=n]   기억 폴더 셋만(풀 크기 비교)
#   bash scripts/pc_bytes.sh cmp <결과 폴더 A> <결과 폴더 B>                  두 판 비교(실행 번호 · 시간 칸만 뺌, 한쪽에만 있는 것은 건너뜀)
# fr1_floor 는 다시 잡기 예산(장면당 20 ms, 벽시계)이 PC 부하에 따라 걸려 같은 판끼리도 가끔 다르다: 다르면 기준 판도 다시 돌려 본다
# 장면 스트림은 sha256 만 남긴다(장면당 약 1.5 MB).
# 세트: 카메라 전용 TUM fr1 desk · room · 360 · rpy · floor(위치 프로그램 단독, 최대 속도), 합성 IMU 재생 desk · room · floor,
#       fr1_desk 기억 폴더(분할 대역, --serve 끝 저장 + 디스크 기록), fr3 긴 사무실 기억 폴더(--every 2, --serve 끝 저장).
# 데이터 자리(바꿀 수 있음): DESK · SHAKE(room 360 rpy floor 폴더가 든 곳) · LONG · SYN(합성 IMU csv 폴더) · MODELS · SEGC(분할 대역 폴더)
set -uo pipefail
# 장면당 시간 예산(벽시계)을 꺼 PC 부하와 상관없이 같은 결과(slam 의 budget_ok)
export ONDEVICE_NO_BUDGET=1
ROOT=$(cd "$(dirname "$0")/.." && pwd)
DESK=${DESK:-$HOME/unify/data/rgbd_dataset_freiburg1_desk}
SHAKE=${SHAKE:-$HOME/data/tum_shake}
LONG=${LONG:-/mnt/c/ondevice/data/tum/rgbd_dataset_freiburg3_long_office_household}
SYN=${SYN:-$HOME/imub}
MODELS=${MODELS:-/mnt/c/ondevice/models}
SEGC=${SEGC:-/dev/shm/int_segcache}
INI=$ROOT/docs/rtabmap_pi.ini

seq_dir() { if [ "$1" = desk ]; then echo "$DESK"; else echo "$SHAKE/rgbd_dataset_freiburg1_$1"; fi; }

# 기억 폴더 하나: <실행 파일> <폴더> <TUM> [인자…]
mem_run() {
  local B=$1 O=$2 D=$3; shift 3
  rm -rf "$O"; mkdir -p "$O"
  ( cd "$ROOT" && "$B" --slam "--tum $D --rate 0 --block --config $INI" --no-drop --surface --max-depth 3.5 \
      --seg "$MODELS/ObjectSAM-416_640.onnx" --seg-cache "$SEGC" --embed off --out "$O/mem" ${POOL:+--pool $POOL} "$@" > "$O/log.txt" 2> "$O/err.txt" )
  echo "$(basename "$O"): exit $? files $(find "$O/mem" -type f | wc -l)"
}

mem_all() {
  local B=$1 R=$2
  mkdir -p "$R" "$SEGC"
  local P=$((18700 + RANDOM % 200))
  mem_run "$B" "$R/desk_serve" "$DESK" --serve "$P --bind 127.0.0.1" --serve-exit
  mem_run "$B" "$R/desk_disk" "$DESK"
  mem_run "$B" "$R/long_serve" "$LONG" --every 2 --serve "$((P + 1)) --bind 127.0.0.1" --serve-exit
}

run_all() {
  local B=$1 R=$2
  mkdir -p "$R" "$SEGC"
  for s in desk room 360 rpy floor; do
    local O=$R/cam_$s; mkdir -p "$O"
    "$B" slam --tum "$(seq_dir $s)" --config "$INI" --rate 0 --map-inline --out "$O/stream.bin" --poses "$O/poses.txt" --final-poses "$O/final.txt" --stats "$O/stats.csv" > "$O/log.txt" 2>&1
    echo "cam_$s: exit $?"; sha256sum < "$O/stream.bin" > "$O/stream.sha"; rm -f "$O/stream.bin"
  done
  for s in desk room floor; do
    local O=$R/imu_$s; mkdir -p "$O"
    "$B" slam --tum "$(seq_dir $s)" --config "$INI" --rate 0 --map-inline --imu-csv "$SYN/syn_$s.csv" --out "$O/stream.bin" --poses "$O/poses.txt" --final-poses "$O/final.txt" --stats "$O/stats.csv" > "$O/log.txt" 2>&1
    echo "imu_$s: exit $?"; sha256sum < "$O/stream.bin" > "$O/stream.sha"; rm -f "$O/stream.bin"
  done
  mem_all "$B" "$R"
}

# 기억 폴더를 견줄 꼴로: 실행 번호 폴더(scene/<번호>) 이름과 view.json · octree.json 의 run 값만 RUN 으로
norm_mem() {
  local d=$1 o=$2
  rm -rf "$o"; cp -r "$d" "$o"
  local r; r=$(ls "$o/scene" 2>/dev/null | head -1)
  if [ -n "$r" ]; then mv "$o/scene/$r" "$o/scene/RUN"; sed -i -E "s/$r/RUN/g; s/\"run\":[0-9]+/\"run\":RUN/g" "$o/view.json" "$o/scene/RUN/octree.json" 2>/dev/null; fi
  sed -i -E 's/"run":[0-9]+/"run":RUN/g' "$o/view.json" 2>/dev/null
}

cmp_all() {
  local A=$1 B=$2 bad=0 T=/dev/shm/pc_bytes_cmp.$$
  mkdir -p "$T"
  for d in "$A"/cam_* "$A"/imu_*; do
    [ -d "$d" ] || continue
    local n; n=$(basename "$d"); local e=$B/$n
    [ -d "$e" ] || continue
    local r="same"
    for f in stream.sha poses.txt final.txt; do cmp -s "$d/$f" "$e/$f" || r="DIFF($f)"; done
    # 통계: 시간 칸(read_ms odom_ms slam_ms send_ms = 3 – 6 번째) 뺌
    cmp -s <(cut -d, -f1-2,7- "$d/stats.csv") <(cut -d, -f1-2,7- "$e/stats.csv") || r="$r DIFF(stats)"
    [ "$r" = same ] || bad=1
    echo "$n: $r"
  done
  for d in "$A"/desk_* "$A"/long_*; do
    [ -d "$d" ] || continue
    local n; n=$(basename "$d")
    [ -d "$B/$n/mem" ] || { echo "$n: MISSING"; bad=1; continue; }
    norm_mem "$d/mem" "$T/a"; norm_mem "$B/$n/mem" "$T/b"
    local c f; c=$(diff -rq "$T/a" "$T/b" | wc -l); f=$(find "$T/a" -type f | wc -l)
    [ "$c" = 0 ] || bad=1
    echo "$n: files $f, differ $c"
    [ "$c" = 0 ] || diff -rq "$T/a" "$T/b" | head -5
  done
  rm -rf "$T"
  [ $bad = 0 ] && echo "ALL SAME" || echo "SOME DIFFER"
  return $bad
}

case ${1:-} in
  run) run_all "$2" "$3" ;;
  mem) mem_all "$2" "$3" ;;
  cmp) cmp_all "$2" "$3" ;;
  *) echo "run <실행 파일> <결과 폴더> | cmp <A> <B>"; exit 2 ;;
esac
