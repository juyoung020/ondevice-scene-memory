# go.sh 태그 [장면...]: 저장소 판으로 재생 도구 · 덤프 도구를 다시 빌드하고 기준 원본(~/opt/orb3_refid)으로 비교
# TAGS 로 덤프 태그를 고름(기본: Tracking 단계 + 판단)
set -e
ulimit -c 0
S=/mnt/c/ondevice-orb-track/slam
REF=$HOME/opt/orb3_refid/ORB_SLAM3
(cd $S && ORB_SLAM3_DIR=$HOME/slam-bench/orb3-nopango CARGO_TARGET_DIR=$HOME/tgt-orbtrack cargo build --release --example orb3rs_track_replay 2>&1 | grep -E "^error" -A9 || true)
mkdir -p ~/opt/orb3_track_bg && cd ~/opt/orb3_track_bg
cp $S/tools/orb3_track_dump/{track_dump.cc,build.sh,runall.sh} .
ORB=$REF ORB_INC=$REF/include/OrderById.h OUT=track_dump_ref bash build.sh 2>&1 | grep -E " error|error:" || true
YDIR=$HOME/slam-bench/orb3-parb DUMP_TAGS=${TAGS:-CRLF,TRKF,TWMM,SLPT,TLMP,SINI,NNKF,CNKF,ULKF,ULMP} ORB=$REF DUMP=track_dump_ref bash runall.sh "$@"
cat ~/orbtrack_out/result_$1.txt
