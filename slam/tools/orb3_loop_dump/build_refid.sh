# 기준 빌드(-ffp-contract=off, ~/opt/orb3_ref) 에 링크: bash build_ref.sh 인자...
set -e
S=~/opt/orb3_refid/ORB_SLAM3
INC="-I$S -I$S/include -I$S/include/CameraModels -I$S/Thirdparty/Sophus -I$S/Thirdparty -I/usr/include/eigen3 -I$HOME/slam-bench/install/include $(pkg-config --cflags opencv4)"
LIB="-L$S/lib -lORB_SLAM3 -L$S/Thirdparty/DBoW2/lib -lDBoW2 -L$S/Thirdparty/g2o/lib -lg2o $(pkg-config --libs opencv4) -Wl,--disable-new-dtags,-rpath,$S/lib:$S/Thirdparty/DBoW2/lib:$S/Thirdparty/g2o/lib:$HOME/slam-bench/install/lib -L$HOME/slam-bench/install/lib"
g++ -include $S/include/OrderById.h -O2 -std=c++17 -march=native -ffp-contract=off -DCOMPILEDWITHC11 $INC "$@" $LIB
