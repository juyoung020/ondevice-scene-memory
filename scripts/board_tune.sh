#!/bin/bash
# 보드(Raspberry Pi 5 + D435 + DX-M1) 실행 전 리눅스 설정. sudo 로 돌린다. 재부팅하면 모두 처음 상태로 돌아간다.
# 쓰기: sudo scripts/board_tune.sh [--no-desktop] [--no-remote] [--hugepages N] [--user 이름]
#       sudo scripts/board_tune.sh --revert      # 이 스크립트가 바꾼 것을 처음 값으로(재부팅 없이)
#   - CPU 주파수 관리자 performance(코어 전부): 주파수 오르내림으로 생기는 장면 시간 흔들림 없앰
#   - D435(USB 8086:0b07) USB 절전 끄기: 스트림 끊김·다시 잡기 없앰
#   - 큰 페이지 N 개(선택), swappiness 10
#   - 쓰지 않는 서비스 멈춤(블루투스·프린터·모뎀·avahi), --no-remote 면 원격 화면(rustdesk·wayvnc)도, --no-desktop 이면 그래픽 화면도
#   - 실시간 권한: --user 의 rtprio 99·memlock 무제한(ondevice --rt auto --mlock 이 쓰도록, 다시 로그인 후)
#   - 인터럽트: USB(xhci, 카메라) → 코어 0, NPU(dx-dma) → 코어 1. 계산 스레드(ondevice --threads 기본 배치)는 코어 2·3
# 바꾸기 전 값은 /run/ondevice-tune 에 남겨 --revert 가 그대로 되돌린다(/run 은 재부팅하면 비므로 재부팅 = 되돌림).
set -u
NODESK=0; NOREMOTE=0; HP=0; USR=${SUDO_USER:-}; REVERT=0
while [ $# -gt 0 ]; do
  case $1 in
    --no-desktop) NODESK=1 ;;
    --no-remote) NOREMOTE=1 ;;
    --hugepages) HP=$2; shift ;;
    --user) USR=$2; shift ;;
    --revert) REVERT=1 ;;
  esac; shift
done
[ "$(id -u)" = 0 ] || { echo "sudo 로 실행"; exit 1; }
ST=/run/ondevice-tune

if [ $REVERT = 1 ]; then
  [ -d $ST ] || { echo "되돌릴 기록 없음($ST)"; exit 0; }
  while read -r f v; do echo "$v" > "$f" 2>/dev/null; done < $ST/files
  while read -r s; do systemctl start "$s" 2>/dev/null && echo "다시 켬: $s"; done < $ST/services
  [ -f $ST/graphical ] && systemctl isolate graphical.target
  rm -f /etc/udev/rules.d/99-d435-power.rules /etc/security/limits.d/99-ondevice.conf
  udevadm control --reload 2>/dev/null
  rm -rf $ST
  echo "되돌림: 주파수 관리자 $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor), 인터럽트 · 서비스 처음 값"
  exit 0
fi

mkdir -p $ST; touch $ST/files $ST/services
# 파일 값 바꾸기: 처음 바꿀 때의 값만 기록
set_file() {
  local f=$1 v=$2
  [ -w "$f" ] || return 1
  grep -q "^$f " $ST/files || echo "$f $(cat "$f" | sed 's/.*\[\(.*\)\].*/\1/')" >> $ST/files
  echo "$v" > "$f"
}
stop_service() {
  if systemctl is-active -q "$1"; then systemctl stop "$1" && echo "$1" >> $ST/services && echo "멈춤: $1"; fi
}

for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do set_file "$g" performance; done
echo "주파수 관리자: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null)"

for d in /sys/bus/usb/devices/*; do
  if [ -f "$d/idVendor" ] && [ "$(cat $d/idVendor)" = 8086 ]; then
    set_file "$d/power/control" on; set_file "$d/power/autosuspend_delay_ms" -1
    echo "USB 절전 끔: $d ($(cat $d/idProduct))"
  fi
done
cat > /etc/udev/rules.d/99-d435-power.rules <<'R'
ACTION=="add", SUBSYSTEM=="usb", ATTR{idVendor}=="8086", TEST=="power/control", ATTR{power/control}="on"
R
udevadm control --reload 2>/dev/null

[ -f /sys/kernel/mm/transparent_hugepage/enabled ] && set_file /sys/kernel/mm/transparent_hugepage/enabled madvise
[ "$HP" -gt 0 ] && set_file /proc/sys/vm/nr_hugepages "$HP" && echo "큰 페이지 $(cat /proc/sys/vm/nr_hugepages) 개"
set_file /proc/sys/vm/swappiness 10

for s in bluetooth cups cups-browsed ModemManager avahi-daemon triggerhappy; do stop_service "$s"; done
if [ $NOREMOTE = 1 ]; then for s in rustdesk wayvnc wayvnc-control; do stop_service "$s"; done; fi
if [ $NODESK = 1 ]; then touch $ST/graphical; systemctl isolate multi-user.target && echo "그래픽 화면 멈춤"; fi

if [ -n "$USR" ]; then
  cat > /etc/security/limits.d/99-ondevice.conf <<L
$USR - rtprio 99
$USR - memlock unlimited
L
  echo "실시간 권한: $USR (다시 로그인하면 적용)"
fi

for irq in $(grep -E 'xhci' /proc/interrupts | awk -F: '{print $1}'); do set_file /proc/irq/${irq// /}/smp_affinity 1 && echo "IRQ $irq(USB) → 코어 0"; done
for irq in $(grep -E 'dx-dma' /proc/interrupts | awk -F: '{print $1}'); do set_file /proc/irq/${irq// /}/smp_affinity 2 && echo "IRQ $irq(NPU) → 코어 1"; done
echo "끝(되돌리기: sudo $0 --revert 또는 재부팅)"
