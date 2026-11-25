#!/bin/bash
# 통합 흐름 대조 덤프(원본 = 번호 순 비교자 + -ffp-contract=off 기준 빌드 ~/opt/orb3_refid)
# build: bash run.sh build
# 전역 BA 결과 넣기: bash run.sh run 출력.bin yaml 장면폴더
# 새 점 넣기:       ORBSYS_CNMP_EVERY=10 bash run.sh run 출력.bin yaml 장면폴더
# 옮겨 붙이기 계산:  bash run.sh graft /tmp/graft.bin 50000
D=$(dirname "$(readlink -f "$0")"); OUT=${ORBSYS_OUT:-$HOME/opt/orb3_sys}; mkdir -p $OUT
S=$HOME/opt/orb3_refid/ORB_SLAM3; P=$HOME/slam-bench/install
case $1 in
 build)
  INC="-I$S -I$S/include -I$S/include/CameraModels -I$S/Thirdparty/Sophus -I$S/Thirdparty -I/usr/include/eigen3 -I$P/include $(pkg-config --cflags opencv4)"
  LIB="-L$S/lib -lORB_SLAM3 -L$S/Thirdparty/DBoW2/lib -lDBoW2 -L$S/Thirdparty/g2o/lib -lg2o $(pkg-config --libs opencv4) -Wl,--disable-new-dtags,-rpath,$S/lib:$S/Thirdparty/DBoW2/lib:$S/Thirdparty/g2o/lib:$P/lib -L$P/lib"
  g++ -include $S/include/OrderById.h -O2 -std=c++17 -march=native -ffp-contract=off -DCOMPILEDWITHC11 -DHAVE_GLEW -DHAVE_EIGEN -D_LINUX_ $INC "$D/flow_dump.cc" -o $OUT/flow_dump $LIB \
    -lpango_display -lpango_core -lpango_windowing -lpango_opengl -lGLEW -lGL -lboost_serialization
  g++ -O3 -std=c++17 -DNDEBUG -march=native -ffp-contract=off -I$S/Thirdparty/Sophus -isystem /usr/include/eigen3 "$D/graft_probe.cc" -o $OUT/graft_probe ;;
 run) ulimit -c 0; LD_LIBRARY_PATH=$P/lib timeout ${ORBSYS_TIMEOUT:-2400} $OUT/flow_dump $HOME/slam-bench/orb3-nopango/Vocabulary/ORBvoc.txt "$3" "$4" "$2" 0-100000 2>&1 | tail -1 ;;
 graft) $OUT/graft_probe "$2" "$3" ;;
esac
