#!/bin/bash
# 섞음 경로(NPU S 후보 + 정확 점수) 비교: run_hyb.sh <mapdir> <n> <기준.bin> [옵션들...]
set -e
cd ~/orbnpu_out
S=${SEQ:-$HOME/unify/data/rgbd_dataset_freiburg1_desk}
A=${ASSOC:-$HOME/slam-bench/src/ORB_SLAM3/Examples/RGB-D/associations/fr1_desk.txt}
T=/mnt/c/ondevice-orb-npu/tools/orbnpu
md=$1; n=$2; ref=$3; shift 3
for o in "$@"; do
  echo "== $o ($md)"
  ./orbnpu_bench maps "$S" "$A" "$n" "$md" "q_$o.bin" "$o"
  ~/sfm/bin/python "$T/cmp_kp.py" "$ref" "q_$o.bin"
done
