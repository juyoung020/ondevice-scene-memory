#!/usr/bin/env bash
# 적외선 스테레오 + IMU 녹화(ORB-SLAM3 등 스테레오-관성 위치 추적 비교용, EuRoC 꼴):
#   bash scripts/rec_ir.sh <이름> [초=90] [레이저 on|off]
# 결과 ~/rec/<이름>: rsd_rec 폴더(rgb · depth · calib.json …) + mav0/cam0 · cam1(적외선 왼쪽 · 오른쪽) + mav0/imu0/data.csv + ir_times.txt
set -euo pipefail
cd "$(dirname "$0")/.."
name=$1 secs=${2:-90} em=${3:-on}
d=$HOME/rec/$name
[ -e "$d" ] && { echo "이미 있음: $d"; exit 1; }
mkdir -p "$d"
target/release/imu_rec /dev/i2c-1 "$d/imu0.csv" $((secs + 4)) & ip=$!
target/release/rsd_rec "$d" $((secs * 30)) --ir --emitter "$em"
wait $ip
mkdir -p "$d/mav0/imu0" && cp "$d/imu0.csv" "$d/mav0/imu0/data.csv"
du -sh "$d"; wc -l "$d/ir_times.txt" "$d/imu0.csv"
