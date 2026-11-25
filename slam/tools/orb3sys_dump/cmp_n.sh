cd /mnt/c/ondevice-orbrs && export CARGO_TARGET_DIR=~/tgt-orbrs ORB_SLAM3_DIR=~/slam-bench/orb3-nopango && cargo build --release -p slam --example orb3_sys_run 2>&1 | grep -E "^error" -A6 | head
T=/mnt/c/ondevice-orbrs/slam/tools/orb3sys_dump; N=${1:-900}
cd ~/orbsys_run && bash $T/sysrun.sh rs rsN.tsv $N >/dev/null 2>&1; [ -n "$2" ] && bash $T/sysrun_ref.sh refN_$N.tsv $N >/dev/null 2>&1
paste rsN.tsv refN_$N.tsv | awk -F"\t" '{d=0; for(k=1;k<=15;k++) if($k!=$(k+17)) d=1; if($17!=$34 && !r){print "ref_kf first diff", NR, $17, $34; r=1} if(d){print "first diff", NR; exit}}'
paste rsN.tsv refN_$N.tsv | awk -F"\t" 'NR>=695 && NR<=709 {print NR, $2, $17, "|", $19, $34}' | tr "\n" ";"
