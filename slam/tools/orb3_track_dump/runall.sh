# runall.sh 태그 [장면이름...]  (DUMP=덤프실행파일)
cd ~/opt/orb3_track_bg; O=${ORB:-~/slam-bench/src/ORB_SLAM3}; P=~/slam-bench/install
export LD_LIBRARY_PATH=$O/lib:$O/Thirdparty/g2o/lib:$O/Thirdparty/DBoW2/lib:$P/lib
R=~/tgt-orbtrack/release/examples/orb3rs_track_replay; D=./${DUMP:-track_dump}
tag=$1; shift; want=" $* "
out=~/orbtrack_out/result_$tag.txt; : > $out
run() {
  [ "$want" = "  " ] || [[ "$want" == *" $1 "* ]] || return 0
  echo "== $1" >> $out
  timeout 2400 $D $O/Vocabulary/ORBvoc.txt $2 $3 ~/orbtrack_out/$1.bin 100000 8 1 2>&1 | tail -1 >> $out
  ls -la ~/orbtrack_out/$1.bin | awk "{print \$5}" >> $out
  $R ~/orbtrack_out/$1.bin >> $out 2>&1
  rm -f ~/orbtrack_out/$1.bin
}
run ds ds.yaml ~/rec_dl/ds_1118_121652/cam
for s in desk2 room 360 floor rpy; do run tum_$s ${YDIR:-$O}/Examples/RGB-D/TUM1.yaml ~/data/tum_shake/rgbd_dataset_freiburg1_$s; done
echo done >> $out
