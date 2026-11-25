#!/bin/bash
# 전체 장면을 8 비트 흉내 지도로 조금씩 만들어(지도는 바로 지움) 원본 · lazy · lazyB 덤프를 만든다
set -e
cd ~/orbnpu_out
S=${SEQ:-$HOME/unify/data/rgbd_dataset_freiburg1_desk}
A=${ASSOC:-$HOME/slam-bench/src/ORB_SLAM3/Examples/RGB-D/associations/fr1_desk.txt}
T=/mnt/c/ondevice-orb-npu/tools/orbnpu
N=${N:-573}; STEP=${STEP:-48}
[ -s full_orig.bin ] || ./orbnpu_bench dump "$S" "$A" "$N" full_orig.bin
rm -f full_lazy.bin full_lazyB.bin
for ((s = 0; s < N; s += STEP)); do
  e=$((s + STEP < N ? s + STEP : N))
  rm -rf mtmp
  ~/sfm/bin/python "$T/orb_maps.py" run mos mtmp "$e" --fq --start=$s
  START=$s ./orbnpu_bench maps "$S" "$A" "$e" mtmp full_lazy.bin lazy
  START=$s ./orbnpu_bench maps "$S" "$A" "$e" mtmp full_lazyB.bin lazyB
  echo "장면 $e/$N"
done
rm -rf mtmp
~/sfm/bin/python "$T/cmp_kp.py" full_orig.bin full_lazy.bin
~/sfm/bin/python "$T/cmp_kp.py" full_orig.bin full_lazyB.bin
