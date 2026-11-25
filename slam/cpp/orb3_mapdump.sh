#!/bin/bash
# 시험 도구 빌드 · 실행: orb3_mapdump.sh build | run 이름 yaml 장면폴더 [장면수] | probe | pg [그래프수] [씨앗]
# 원본 = 기준 빌드(-ffp-contract=off) ~/opt/orb3_ref/ORB_SLAM3

O=${ORB_REF:-$HOME/opt/orb3_ref/ORB_SLAM3}; P=$HOME/slam-bench/install
D=$(dirname "$(readlink -f "$0")"); OUT=${ORBMAP_OUT:-$HOME/orbmap_out}; mkdir -p $OUT
export LD_LIBRARY_PATH=$O/lib:$O/Thirdparty/g2o/lib:$O/Thirdparty/DBoW2/lib:$P/lib
case $1 in
 build) g++ -O3 -march=native -ffp-contract=off -std=c++17 -DCOMPILEDWITHC11 -DNDEBUG -DHAVE_EIGEN -DHAVE_GLEW -D_LINUX_ -I$O -I$O/include -I$O/include/CameraModels -I$O/Thirdparty/Sophus -I/usr/include/eigen3 -I$P/include $(pkg-config --cflags opencv4) \
    "$D/orb3_mapdump.cc" -o $OUT/orb3_mapdump -L$O/lib -lORB_SLAM3 -L$O/Thirdparty/g2o/lib -lg2o $(pkg-config --libs opencv4) -L$P/lib -lpango_display -lpango_core -lpango_windowing -lpango_opengl -lGLEW -lGL -ldl -lboost_serialization -Wl,--allow-shlib-undefined ;;
 probe) g++ -O3 -march=native -ffp-contract=off -std=c++17 -DCOMPILEDWITHC11 -DNDEBUG -DHAVE_EIGEN -DHAVE_GLEW -D_LINUX_ -I$O -I$O/include -I$O/include/CameraModels -I$O/Thirdparty/Sophus -I/usr/include/eigen3 -I$P/include $(pkg-config --cflags opencv4)     "$D/orb3_sim3pg_probe.cc" -o $OUT/orb3_sim3pg_probe -Wl,--no-as-needed -L$O/lib -lORB_SLAM3 -L$O/Thirdparty/g2o/lib -lg2o $(pkg-config --libs opencv4) -L$P/lib -lpango_display -lpango_core -lpango_windowing -lpango_opengl -lGLEW -lGL -ldl -lboost_serialization -Wl,--allow-shlib-undefined ;;
 pg) $OUT/orb3_sim3pg_probe $OUT/pg.bin ${2:-300} ${3:-1}; ls -la $OUT/pg.bin ;;
 run) ulimit -c 0; timeout ${ORBMAP_TIMEOUT:-600} $OUT/orb3_mapdump ${ORB_VOC:-$HOME/slam-bench/src/ORB_SLAM3/Vocabulary/ORBvoc.txt} "$3" "$4" $OUT/$2.bin ${5:-100000} 2> $OUT/$2.log >/dev/null; tail -32 $OUT/$2.log; ls -la $OUT/$2.bin ;;
esac
