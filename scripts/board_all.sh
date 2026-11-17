#!/usr/bin/env bash
# 보드(Pi 5 + D435 + ICM-42688-P + DX-M1)에서 한 번에: 동기화 · 빌드 → 모델 확인 → NPU 단독 · 끝 알림 시험 → 사진 벡터 함께 올리기 수 →
# 녹화 재생 → 실시간 가만히 60 초 → 열 시험 10 분 → 흔들기 60 초(사람) → 요약 → 실시간 시연(뷰어 :8091)을 띄워 둠.
# 항목 하나가 실패해도 다음으로 넘어가고 요약에 표시한다. 사람이 할 일은 전원 켜기와 흔들기(알림이 뜨면)뿐.
#
#   PC 에서: ssh -t kau@192.168.219.101 'bash -lc "cd ~/ondevice-scene-memory && git fetch -q origin opt/integrate && git checkout -q opt/integrate && git reset -q --hard origin/opt/integrate && bash scripts/board_all.sh"'
#   보드에서: bash scripts/board_all.sh [브랜치=opt/integrate]
#
# 환경 변수(기본값)
#   OUT=~/board_all/<시각>   결과 폴더(요약 summary.md, 단계별 로그 · TSV)
#   STILL_S=60 THERMAL_S=600 SHAKE_S=60   실시간 길이(초). 0 이면 그 단계 건너뜀
#   SHAKE_WAIT=60            흔들기 알림 뒤 기다리는 최대 초(엔터를 누르면 바로 시작)
#   EMBED_JOBS="3 4 6 8"     사진 벡터 함께 올리기 수 후보
#   RUNS="~/imucal/run_173135 ~/imucal/run_174258"   녹화 재생(각 폴더에 imu0.csv)
#   TUM=~/om/data/tum/rgbd_dataset_freiburg1_desk     사진 벡터 · 물체 병합 재생용 녹화
#   VIEW=1                   실시간 단계에 화면 뷰어(Chromium 키오스크)도 띄움(보드 화면이 있을 때)
#   DEMO=1                   끝에 실시간 시연을 띄워 둠
set -uo pipefail
BRANCH=${1:-opt/integrate}
REPO=$HOME/ondevice-scene-memory
# 저장소를 새 판으로 바꾸는 동안 이 파일이 바뀌어도 끝까지 같은 내용으로 돌게 사본에서
if [ -z "${BOARD_ALL_COPY:-}" ]; then
  cp "$0" "/tmp/board_all.$$.sh"
  BOARD_ALL_COPY=1 exec bash "/tmp/board_all.$$.sh" "$@"
fi
OUT=${OUT:-$HOME/board_all/$(date +%m%d_%H%M)}
STILL_S=${STILL_S:-60}; THERMAL_S=${THERMAL_S:-600}; SHAKE_S=${SHAKE_S:-60}; SHAKE_WAIT=${SHAKE_WAIT:-60}
EMBED_JOBS=${EMBED_JOBS:-"3 4 6 8"}
RUNS=${RUNS:-"$HOME/imucal/run_173135 $HOME/imucal/run_174258"}
TUM=${TUM:-$HOME/om/data/tum/rgbd_dataset_freiburg1_desk}
VIEW=${VIEW:-1}; DEMO=${DEMO:-1}
M=${MODELS:-$HOME/models}
SEG=$M/objectsam-416_640.dxnn; EMB=$M/siglip2-vision_npu.dxnn
mkdir -p "$OUT"
R=$OUT/results.tsv
: > "$R"
[ -f "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT/board_all.log"; }
# step <이름> <명령…>: 실행해 성공 · 실패를 결과 표에 남기고 계속
step() {
  local name=$1; shift
  log ">>> $name"
  local c0=$SECONDS
  if "$@" >> "$OUT/board_all.log" 2>&1; then echo -e "$name\tOK\t$((SECONDS - c0)) s" >> "$R"; log "    OK"
  else echo -e "$name\tFAIL($?)\t$((SECONDS - c0)) s" >> "$R"; log "    FAIL"; fi
}
BIN=$REPO/target/release/ondevice

# 1. 동기화 · 빌드(scripts/board_build.sh: 가져오기 → 브랜치 맞춤 → NPU dxrt 빌드)
build() { bash "$REPO/scripts/board_build.sh" "$BRANCH" "$REPO"; }
step "동기화 · 빌드($BRANCH)" build
cd "$REPO" || exit 1
git rev-parse --short HEAD > "$OUT/commit.txt" 2>/dev/null

# 2. 모델 sha256(docs/models.md 의 models-v2 값)
models() {
  local ok=0
  while read -r sum f; do
    local got; got=$(sha256sum "$M/$f" 2>/dev/null | cut -d' ' -f1)
    if [ "$got" = "$sum" ]; then echo "같음 $f"; else echo "다름 $f: $got"; ok=1; fi
  done <<'S'
b3579a4bfa144aec88f49e26491c39575142d444eacdc0b7c7b4e5a91e405178 objectsam-416_640.dxnn
2c9f0bcc3a27a1224016300547bd624292ecd942d45ef8f817bc2a07dbe5b00d siglip2-vision_npu.dxnn
S
  return $ok
}
step "모델 sha256(models-v2)" models

# 3. NPU 단독: 모델마다 기다림 스레드 길로 1 · 2 · 3 장 함께(처리량 · 지연). MegaLoc · XFeat 은 파일이 있으면
npu_solo() {
  for f in "$SEG" "$EMB" "$M"/megaloc*.dxnn "$M"/xfeat*.dxnn; do
    [ -f "$f" ] || continue
    echo "== $(basename "$f")"
    "$BIN" npu-test --model "$f" --n 200 --jobs 1,2,3 --wake thread | tee -a "$OUT/npu_solo.jsonl"
  done
}
step "NPU 단독(분할 · 사진 벡터 · MegaLoc · XFeat)" npu_solo

# 4. NPU 끝 알림 시험: 콜백 → 루프 깨움이 비트 같고 끝까지 도는지(통과하면 --npu-wake callback 을 기본으로 바꿀 근거)
npu_cb() {
  dxrt-cli --help > "$OUT/dxrt_cli_help.txt" 2>&1 || true
  "$BIN" npu-test --model "$SEG" --n 1000 --jobs 1,2,4 --wake thread,callback | tee "$OUT/npu_cb.jsonl"
  grep -q '"callback_pass":true' "$OUT/npu_cb.jsonl"
}
step "NPU 콜백 시험" npu_cb

# 5. 사진 벡터 함께 올리기 수(--embed-jobs): 같은 녹화를 최대 속도로, 사진 벡터 처리량 · kf/s
embed_jobs() {
  [ -d "$TUM" ] || { echo "녹화 없음: $TUM"; return 1; }
  for k in $EMBED_JOBS; do
    local d=$OUT/embed_k$k
    RATE=0 OUT=$d IMU= bash scripts/board_run.sh --tum "$TUM" --embed-jobs "$k" --serve-exit > "$d.log" 2>&1
    echo -e "$k\t$(grep -o 'embed: [0-9]* vectors.*' "$d.log" | tail -1)\t$(grep -o '"kf_per_s":[0-9.]*' "$d.log" | tail -1)" | tee -a "$OUT/embed_jobs.tsv"
  done
}
step "사진 벡터 함께 올리기 수($EMBED_JOBS)" embed_jobs

# 6. 녹화 재생: 카메라 전용 · IMU, 그대로 · 번짐(찍은 속도 = 실시간과 같은 타이밍) + 물체 병합 속도(maps_bench)
replay() {
  local CAL=$REPO/docs/imu_cam_calib.yaml ok=0
  echo -e "조건\t장면\t놓침\t새 지도\t다시 붙기 초\t못 붙음\t튐\t겹 2cm 넘는 칸 %\t위치 추적 ms 평균 · 95%" > "$OUT/replay.tsv"
  mkdir -p /dev/shm/board_all
  for RUN in $RUNS; do
    [ -f "$RUN/imu0.csv" ] || { echo "건너뜀(imu0.csv 없음): $RUN"; ok=1; continue; }
    for MD in cam imu; do for S in plain blur; do
      local n; n=$(basename "$RUN")_${MD}_$S; local d=$OUT/replay/$n; mkdir -p "$d"
      local st=/dev/shm/board_all/$n.bin
      local mode; mode=$([ $MD = imu ] && echo "--imu-csv $RUN/imu0.csv" || echo "--stress-gyro $RUN/imu0.csv")
      local stress; stress=$([ $S = blur ] && echo "--stress-blur 0.02" || echo "")
      "$BIN" slam --rsdrec "$RUN" --config docs/rtabmap_pi.ini --rate 1 --out "$st" --poses "$d/poses.txt" --final-poses "$d/final.txt" --stats "$d/stats.csv" $mode $stress > "$d/log.txt" 2>&1
      "$REPO/target/release/slam_eval" --poses "$d/poses.txt" --final "$d/final.txt" --stats "$d/stats.csv" --imu "$RUN/imu0.csv" --calib "$CAL" --stream "$st" --tag "$n" > "$d/eval.json" 2>/dev/null
      rm -f "$st"
      if command -v jq > /dev/null; then
        jq -r --arg n "$n" '[$n, .frames, .lost_events, .new_maps, (.rejoin_mean_s // "-"), .unjoined, (.jump_events // "-"), (.layer2_pct // "-"), "\(.odom_ms) · \(.odom_p95_ms)"] | @tsv' "$d/eval.json" >> "$OUT/replay.tsv"
      else echo -e "$n\t$(cat "$d/eval.json")" >> "$OUT/replay.tsv"; fi
    done; done
  done
  cargo test --release --no-default-features --features dxrt -p ondevice bench_memory -- --ignored --nocapture 2>&1 | grep '^bench' > "$OUT/merge_bench.txt"
  return $ok
}
step "녹화 재생(카메라 전용 · IMU, 그대로 · 번짐) + 물체 병합 속도" replay

# 실시간 한 번: <이름> <초>. 5 초마다 온도 · 클록 · 스로틀 · NPU 상태 · CPU % · RSS · 깨어남 · kf/s 를 TSV 로
NPU_STAT() { dxrt-cli -s 2>/dev/null | grep -iE 'temp|clock|freq|mhz|volt' | tr -s ' \n\t' ' ' | cut -c1-200; }
live() {
  local name=$1 secs=$2 d=$OUT/$1
  [ "$secs" -gt 0 ] || { echo "건너뜀"; return 0; }
  [ -e /dev/i2c-1 ] || sudo dtparam i2c_arm=on 2>/dev/null
  mkdir -p "$d"
  local frames=$((secs * 30))
  OUT=$d/mem SLAM_EXTRA="--stats $d/stats.csv --max-frames $frames" bash scripts/board_run.sh --serve-exit > "$d/log.txt" 2> "$d/err.txt" &
  local sh=$! pid=""
  for _ in $(seq 1 100); do pid=$(pgrep -n -x ondevice) && break; sleep 0.2; done
  local kp=""
  if [ "$VIEW" = 1 ] && [ -n "${DISPLAY:-:0}" ]; then DISPLAY=${DISPLAY:-:0} bash scripts/board_view.sh 8091 > "$d/view.log" 2>&1 & kp=$!; fi
  local hz; hz=$(getconf CLK_TCK)
  echo -e "초\tCPU °C\tARM Hz\tthrottled\tNPU\tCPU %\tRSS MB\t깨어남/s\t스레드\tkf\tkf/s" > "$d/samples.tsv"
  local t0=$SECONDS cpu0=0 ctx0=0 kf0=0
  while kill -0 "$sh" 2>/dev/null; do
    sleep 5
    [ -n "$pid" ] && [ -d /proc/$pid ] || break
    local cpu ctx kf rss thr
    cpu=$(awk '{print $14+$15}' /proc/$pid/stat 2>/dev/null || echo 0)
    ctx=$(cat /proc/$pid/task/*/status 2>/dev/null | awk '/^voluntary_ctxt/{s+=$2} END{print s+0}')
    rss=$(awk '/^VmRSS/{printf "%.0f", $2/1024}' /proc/$pid/status 2>/dev/null)
    thr=$(ls /proc/$pid/task 2>/dev/null | wc -l)
    kf=$(grep -o '^kf [0-9]*' "$d/log.txt" | tail -1 | cut -d' ' -f2); kf=${kf:-0}
    echo -e "$((SECONDS - t0))\t$(vcgencmd measure_temp 2>/dev/null | tr -dc '0-9.')\t$(vcgencmd measure_clock arm 2>/dev/null | cut -d= -f2)\t$(vcgencmd get_throttled 2>/dev/null | cut -d= -f2)\t$(NPU_STAT)\t$(( (cpu - cpu0) * 100 / (hz * 5) ))\t$rss\t$(( (ctx - ctx0) / 5 ))\t$thr\t$kf\t$(awk "BEGIN{printf \"%.1f\", ($kf - $kf0) / 5}")" >> "$d/samples.tsv"
    cpu0=$cpu; ctx0=$ctx; kf0=$kf
  done
  wait "$sh"; local rc=$?
  [ -n "$kp" ] && pkill -f "user-data-dir=$HOME/.cache/ondevice-view" 2>/dev/null
  grep '^{"keyframes"' "$d/log.txt" | tail -1 > "$d/summary.json"
  return $rc
}

# 7. 실시간 가만히
step "실시간 가만히 ${STILL_S} 초" live still "$STILL_S"
# 8. 열 시험: 실시간 전체 부하(뷰어 포함) 10 분 연속
step "열 시험 ${THERMAL_S} 초" live thermal "$THERMAL_S"
# 9. 흔들기: 사람이 필요 — 화면 · 파일 · 터미널로 알리고 기다렸다가 시작
shake() {
  local msg="지금 장비를 손으로 빠르게 흔들어 주세요(${SHAKE_S} 초). 준비되면 엔터(또는 ${SHAKE_WAIT} 초 뒤 자동 시작)"
  echo "$msg" > "$HOME/board_all_SHAKE_NOW.txt"
  wall "$msg" 2>/dev/null || true
  [ -n "${DISPLAY:-}" ] && command -v zenity > /dev/null && (zenity --info --text "$msg" --timeout "$SHAKE_WAIT" > /dev/null 2>&1 &)
  printf '\n\n=====================================================\n %s\n=====================================================\n' "$msg" > /dev/tty 2>/dev/null || echo "$msg"
  read -r -t "$SHAKE_WAIT" < /dev/tty 2>/dev/null || sleep "$SHAKE_WAIT"
  echo "흔들기 시작" > "$HOME/board_all_SHAKE_NOW.txt"
  live shake "$SHAKE_S"
  local rc=$?
  rm -f "$HOME/board_all_SHAKE_NOW.txt"
  return $rc
}
[ "$SHAKE_S" -gt 0 ] && step "흔들기 ${SHAKE_S} 초(사람)" shake

# 10. 요약
{
  # 첫 줄: 스로틀(펌웨어) 한 번도 안 걸렸는지 · 비상 열 보호가 몇 번 걸렸는지
  thr=$(cat "$OUT"/*/samples.tsv 2>/dev/null | awk -F'\t' 'NR>1 && $4 != "" && $4 != "0x0" && $4 != "throttled" {print $4}' | sort -u | tr '\n' ' ')
  hits=$(cat "$OUT"/*/summary.json 2>/dev/null | grep -o '"hits":[0-9]*' | cut -d: -f2 | awk '{s+=$1} END{print s+0}')
  peak=$(cat "$OUT"/*/samples.tsv 2>/dev/null | awk -F'\t' 'NR>1 && $2+0 > m {m=$2+0} END{print m+0}')
  echo "# 보드 통합 측정 $(date '+%Y-%m-%d %H:%M') ($(cat "$OUT/commit.txt" 2>/dev/null))"
  echo
  echo "**스로틀: $([ -z "$thr" ] && echo '한 번도 안 걸림' || echo "걸림($thr)") · 비상 열 보호: ${hits} 번 · 최고 CPU ${peak} °C**"
  echo
  echo "## 단계"
  echo "| 단계 | 결과 | 시간 |"; echo "|---|---|---|"
  awk -F'\t' '{printf "| %s | %s | %s |\n", $1, $2, $3}' "$R"
  echo
  echo "## NPU 끝 알림(콜백 시험)"
  echo '```'; cat "$OUT/npu_cb.jsonl" 2>/dev/null; echo '```'
  grep -q '"callback_pass":true' "$OUT/npu_cb.jsonl" 2>/dev/null && echo "콜백 길 통과: --npu-wake callback 을 기본으로 바꿀 수 있음(기다림 스레드 2 개가 없어짐)." || echo "콜백 길 미통과: 기본 thread 그대로."
  echo
  echo "## NPU 단독"; echo '```'; cat "$OUT/npu_solo.jsonl" 2>/dev/null; echo '```'
  echo "## 사진 벡터 함께 올리기 수"; echo '```'; cat "$OUT/embed_jobs.tsv" 2>/dev/null; echo '```'
  echo "## 녹화 재생"; echo '```'; column -t -s $'\t' "$OUT/replay.tsv" 2>/dev/null; echo '```'
  echo "## 물체 병합 속도(maps_bench)"; echo '```'; cat "$OUT/merge_bench.txt" 2>/dev/null; echo '```'
  for n in still thermal shake; do
    [ -f "$OUT/$n/samples.tsv" ] || continue
    echo "## 실시간 $n"
    echo '```'
    grep -o '"kf_per_s":[0-9.]*\|"scene_ms":{[^}]*}\|"live":[0-9]*\|"loop":{[^}]*}' "$OUT/$n/summary.json" 2>/dev/null
    awk -F'\t' 'NR>1 {n++; t+=$2; if ($2>m) m=$2; c+=$6; r=$7; w+=$8; k+=$11} END{if(n) printf "평균 %.1f °C(최고 %.1f), CPU %.0f %%, RSS 끝 %s MB, 깨어남 %.0f /s, kf/s %.1f\n", t/n, m, c/n, r, w/n, k/n}' "$OUT/$n/samples.tsv"
    echo '```'
  done
  echo "열 시험 시간별 표(그래프용): $OUT/thermal/samples.tsv"
} > "$OUT/summary.md"
cat "$OUT/summary.md"

# 11. 실시간 시연을 띄워 둠(뷰어 http://<보드>:8091, 보드 화면에도)
if [ "$DEMO" = 1 ]; then
  nohup bash scripts/board_run.sh > "$OUT/demo.log" 2>&1 &
  [ "$VIEW" = 1 ] && (DISPLAY=${DISPLAY:-:0} nohup bash scripts/board_view.sh 8091 > "$OUT/demo_view.log" 2>&1 &)
  log "실시간 시연을 띄워 둠: http://$(hostname -I | cut -d' ' -f1):8091 (끄기: pkill -x ondevice)"
fi
log "결과: $OUT/summary.md"
