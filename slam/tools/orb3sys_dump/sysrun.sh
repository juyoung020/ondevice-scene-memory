#!/bin/bash
# 녹화 전체 재생: bash sysrun.sh rs|cpp 출력.tsv [최대장수] [설정] [녹화폴더]
# (examples/orb3_sys_run, CARGO_TARGET_DIR=~/tgt-orbrs 로 빌드해 둔 것)
ulimit -c 0
S=$HOME/slam-bench/orb3-nopango
export LD_LIBRARY_PATH=$S/lib:$S/Thirdparty/DBoW2/lib:$S/Thirdparty/g2o/lib:$HOME/slam-bench/install/lib
exec timeout ${SYSRUN_TIMEOUT:-3600} $HOME/tgt-orbrs/release/examples/orb3_sys_run "$1" $S/Vocabulary/ORBvoc.txt "${4:-$HOME/opt/orb3_ba_bg/ds.yaml}" "${5:-$HOME/rec_dl/ds_1118_121652/cam}" "${3:-100000}" "$2"
