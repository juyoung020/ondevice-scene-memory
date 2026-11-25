#!/bin/bash
# WSL 에서: 원본 ORB-SLAM3 소스의 ORBextractor.cc 를 끌어와 시험 도구를 빌드한다
set -e
ORB=${ORB:-$HOME/slam-bench/src/ORB_SLAM3}
OUT=${OUT:-$HOME/orbnpu_out}
mkdir -p "$OUT"
D=$(cd "$(dirname "$0")" && pwd)
g++ -O3 -march=native -std=c++17 -I"$ORB/include" -I"$ORB/src" -I"$ORB" -I"$ORB/include/CameraModels" \
  "$D/orbnpu_bench.cc" -o "$OUT/orbnpu_bench" $(pkg-config --cflags --libs opencv4)
echo "$OUT/orbnpu_bench"
# 궤적 비교용 LD_PRELOAD 판
g++ -O2 -shared -fPIC -std=c++17 -I"$ORB/include" -I"$ORB" "$D/orb3_replay.cc" -o "$OUT/liborb3_replay.so" $(pkg-config --cflags opencv4)
echo "$OUT/liborb3_replay.so"
