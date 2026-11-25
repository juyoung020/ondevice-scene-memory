# 기준 빌드(-ffp-contract=off) + 번호 순 비교자(OrderById) + 공시성 정렬 동률 번호 순: ~/opt/orb3_refid
set -e
D=~/opt/orb3_refid; mkdir -p $D
[ -d $D/ORB_SLAM3 ] || cp -a ~/opt/orb3_ref/ORB_SLAM3 $D/
cd $D/ORB_SLAM3
P="/mnt/c/Users/user one/AppData/Local/Temp/rl/det.patch"
# 패치에서 OrderById.h · OrderById.cc · CMake -include · KeyFrame.cc 정렬만 골라 적용
python3 - "$P" <<'PY'
import sys,re
t=open(sys.argv[1]).read()
files=re.split(r'(?m)^--- a/',t)
keep=[f for f in files if f.startswith(('include/OrderById.h','src/OrderById.cc','src/KeyFrame.cc','CMakeLists.txt'))]
open('/tmp/refid.patch','w').write(''.join('--- a/'+f for f in keep))
PY
patch -p1 -N --dry-run < /tmp/refid.patch >/dev/null && patch -p1 -N < /tmp/refid.patch
rm -rf build lib && mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-ffp-contract=off" -DCMAKE_C_FLAGS="-ffp-contract=off" -DCMAKE_PREFIX_PATH=$HOME/slam-bench/install >/dev/null && make -j8 2>&1 | tail -2
ls -la $D/ORB_SLAM3/lib
