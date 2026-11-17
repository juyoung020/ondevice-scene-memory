#!/usr/bin/env bash
# 보드(Pi 5 + D435 + ICM-42688-P) IMU 모드 실측을 한 번에: 빌드 → 보정 녹화 재생(카메라 전용 · IMU, 그대로 · 번짐) → 실시간(가만히 · 흔들기) → 요약 파일.
#   bash scripts/board_imu_measure.sh            (저장소 폴더에서, 보드에서)
# 환경 변수
#   BUILD=0        빌드 건너뜀(기본 1: scripts/board_build.sh 와 같은 release 빌드)
#   RUNS="..."     재생할 rsd_rec 녹화 폴더(각 폴더에 imu0.csv, 기본 ~/imucal/run_173135 ~/imucal/run_174258)
#   LIVE_S=60      실시간 한 번 길이(초): 가만히 한 번, 흔들기 한 번(흔들기 차례에 장비를 손으로 빠르게 흔든다)
#   OUT=~/imumeas/<시각>  결과 폴더. 큰 장면 스트림은 /dev/shm 에 쓰고 재고 나서 지운다
# 결과: $OUT/summary.tsv(조건마다 한 줄: 놓침 · 새 지도 · 다시 붙기 · 튐 · 겹 어긋남 · 위치 추적 ms · 깊이 정합 시간) 와 조건별 로그 · 통계 CSV · eval.json.
# 실시간 칸의 물체 기억 kf/s · 화면 지연은 로그의 상태 줄(kf/s)과 뷰어 ?metrics=1 로 본다(로그는 $OUT/live_*.log).
set -uo pipefail
cd "$(dirname "$0")/.."
REPO=$(pwd)
BIN=${BIN:-$REPO/target/release}
OUT=${OUT:-$HOME/imumeas/$(date +%m%d_%H%M)}
RUNS=${RUNS:-"$HOME/imucal/run_173135 $HOME/imucal/run_174258"}
LIVE_S=${LIVE_S:-60}
mkdir -p "$OUT" /dev/shm/imumeas
if [ "${BUILD:-1}" = 1 ]; then
    cargo build --release -p ondevice 2>&1 | tail -2
fi
CAL=$REPO/docs/imu_cam_calib.yaml
echo -e "조건\t장면\t놓침\t새 지도\t다시 붙기 초(평균)\t못 붙음\t튐\t겹 2cm 넘는 칸 %\t위치 추적 ms 평균 · 95%\t깊이 정합" > "$OUT/summary.tsv"
row() { # row <이름> <폴더>: eval.json + 로그 → 한 줄
    local n=$1 d=$2
    local geo
    command -v jq > /dev/null || { echo -e "$n	$(cat "$d/eval.json")" >> "$OUT/summary.tsv"; return; }
    geo=$(grep -h "깊이 서브맵 정합:" "$d/log.txt" 2>/dev/null | sed 's/.*정합: //' | tr '\t' ' ')
    jq -r --arg n "$n" --arg g "${geo:--}" '[$n, .frames, .lost_events, .new_maps, (.rejoin_mean_s // "-"), .unjoined, (.jump_events // "-"), (.layer2_pct // "-"), "\(.odom_ms) · \(.odom_p95_ms)", $g] | @tsv' "$d/eval.json" >> "$OUT/summary.tsv"
}
# 1) 보정 녹화 재생(찍은 속도 그대로 = 실시간과 같은 타이밍, IMU 는 같은 재생 시계로)
for R in $RUNS; do
    [ -f "$R/imu0.csv" ] || { echo "건너뜀(imu0.csv 없음): $R"; continue; }
    for M in cam imu; do
        for S in plain blur; do
            n=$(basename "$R")_${M}_$S
            d=$OUT/$n; mkdir -p "$d"
            st=/dev/shm/imumeas/$n.bin
            mode=$([ $M = imu ] && echo "--imu-csv $R/imu0.csv" || echo "--stress-gyro $R/imu0.csv")
            stress=$([ $S = blur ] && echo "--stress-blur 0.02" || echo "")
            "$BIN/ondevice" slam --rsdrec "$R" --config docs/rtabmap_pi.ini --rate 1 --out "$st" \
                --poses "$d/poses.txt" --final-poses "$d/final.txt" --stats "$d/stats.csv" $mode $stress > "$d/log.txt" 2>&1
            "$BIN/slam_eval" --poses "$d/poses.txt" --final "$d/final.txt" --stats "$d/stats.csv" --imu "$R/imu0.csv" --calib "$CAL" --stream "$st" --tag "$n" > "$d/eval.json"
            rm -f "$st"
            row "$n" "$d"
        done
    done
done
# 2) 실시간(D435 + IMU): 가만히 · 흔들기 LIVE_S 초씩. 통계 CSV 로 놓침 · 새 지도 · 위치 추적 시간
[ -e /dev/i2c-1 ] || sudo dtparam i2c_arm=on
for S in still shake; do
    d=$OUT/live_$S; mkdir -p "$d"
    echo ">>> 실시간 $S ${LIVE_S} 초 $([ $S = shake ] && echo '— 지금 장비를 손으로 빠르게 흔들기')"
    IMU=/dev/i2c-1 SLAM_EXTRA="--stats $d/stats.csv" timeout -s INT "$LIVE_S" bash scripts/board_run.sh > "$d/log.txt" 2>&1
    pkill -x ondevice 2>/dev/null; sleep 2
    "$BIN/slam_eval" --stats "$d/stats.csv" --tag "live_$S" > "$d/eval.json"
    row "live_$S" "$d"
    grep -E "kf/s" "$d/log.txt" | tail -3 > "$d/kfs.txt"
done
column -t -s $'\t' "$OUT/summary.tsv"
echo "결과: $OUT"
