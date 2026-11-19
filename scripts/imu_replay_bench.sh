#!/bin/bash
# 재생 벤치: 녹화(TUM · rsd_rec) + IMU(imu0.csv)를 최대 속도로 재생해 카메라 전용 · IMU 모드를 같은 잣대(slam_eval)로 잰다.
#   scripts/imu_replay_bench.sh <태그> <cam|imu> [ondevice slam 추가 인자 ...]
# 환경 변수
#   B      빌드 폴더(기본 ~/tgt-imu-dev/release: ondevice · slam_eval · imu_synth), SLAM 위치 프로그램만 다른 빌드로(전후 비교)
#   SEQS   "이름=폴더 ..."(기본 TUM fr1 desk room 360 rpy floor)
#   IMUDIR 시퀀스별 IMU 파일 폴더(<이름>.csv, 없으면 imu_synth 로 정답에서 만듦: TUM fr1 정답 → 영상 시계 차이 SYNTH_OFFSET 기본 0.015 초)
#   OUT    결과 폴더(기본 ~/imubench/<태그>/<모드>), 큰 장면 스트림은 /dev/shm 에 쓰고 재고 나서 지움
# 끊김 · 흐림 시험은 추가 인자로: --stress-gap 4:0.6 --stress-imu-gap · --stress-blur 0.02 (흐림 각속도는 같은 IMU 파일, 카메라 전용도 같은 영상)
set -e
T=$1; M=$2; shift 2
B=${B:-~/tgt-imu-dev/release}
SLAM=${SLAM:-$B/ondevice}
REPO=$(cd "$(dirname "$0")/.." && pwd)
CAL=$REPO/docs/imu_cam_calib.yaml
SEQS=${SEQS:-"desk=$HOME/unify/data/rgbd_dataset_freiburg1_desk room=$HOME/data/tum_shake/rgbd_dataset_freiburg1_room 360=$HOME/data/tum_shake/rgbd_dataset_freiburg1_360 rpy=$HOME/data/tum_shake/rgbd_dataset_freiburg1_rpy floor=$HOME/data/tum_shake/rgbd_dataset_freiburg1_floor"}
IMUDIR=${IMUDIR:-$HOME/imub/synth}
OUT=${OUT:-$HOME/imubench/$T/$M}
mkdir -p "$OUT" "$IMUDIR" /dev/shm/imub
for kv in $SEQS; do
  s=${kv%%=*}; D=${kv#*=}
  I=$IMUDIR/$s.csv
  [ -f "$I" ] || "$B/imu_synth" "$D/groundtruth.txt" "$I" --calib "$CAL" --offset ${SYNTH_OFFSET:-0.015} 2>> "$OUT/synth.log"
  O=$OUT/$s; mkdir -p "$O"
  ST=/dev/shm/imub/$T.$M.$s.bin
  if [ "$M" = imu ]; then MODE="--imu-csv $I"; else MODE="--stress-gyro $I"; fi
  "$SLAM" slam --tum "$D" --rate 0 --out "$ST" \
    --poses "$O/poses.txt" --final-poses "$O/final.txt" --stats "$O/stats.csv" $MODE "$@" > "$O/log.txt" 2>&1
  GT=$D/groundtruth.txt
  "$B/slam_eval" --gt "$GT" --poses "$O/poses.txt" --final "$O/final.txt" --stats "$O/stats.csv" --imu "$I" --calib "$CAL" --stream "$ST" --tag "$T/$M/$s" > "$O/eval.json"
  rm -f "$ST"
  echo "$(cat "$O/eval.json")"
done
