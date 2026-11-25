#!/bin/bash
# 원본 기준 빌드(~/opt/orb3_refid: 번호 순 비교자 + -ffp-contract=off)를 실행 때 라이브러리만 바꿔 끼워 재생: bash sysrun_ref.sh 출력.tsv [장수]
ulimit -c 0
S=$HOME/opt/orb3_refid/ORB_SLAM3
export LD_LIBRARY_PATH=$S/lib:$S/Thirdparty/DBoW2/lib:$S/Thirdparty/g2o/lib:$HOME/slam-bench/install/lib
exec timeout ${SYSRUN_TIMEOUT:-3600} $HOME/tgt-orbrs/release/examples/orb3_sys_run cpp $HOME/slam-bench/orb3-nopango/Vocabulary/ORBvoc.txt "${3:-$HOME/opt/orb3_ba_bg/ds.yaml}" "${4:-$HOME/rec_dl/ds_1118_121652/cam}" "${2:-100000}" "$1"
