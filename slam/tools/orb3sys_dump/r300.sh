cd /mnt/c/ondevice-orbrs && export CARGO_TARGET_DIR=~/tgt-orbrs ORB_SLAM3_DIR=~/slam-bench/orb3-nopango && cargo build --release -p slam --example orb3_sys_run 2>&1 | grep -E "^error" -A8 | head -20
T=/mnt/c/ondevice-orbrs/slam/tools/orb3sys_dump
cd ~/orbsys_run && bash $T/sysrun.sh rs rs300.tsv ${1:-300} >/dev/null 2>&1; bash $T/sysrun_ref.sh ref300.tsv ${1:-300} > /dev/null 2>&1
python3 $T/trajcmp.py rs300.tsv ref300.tsv | sed -n 2,3p
