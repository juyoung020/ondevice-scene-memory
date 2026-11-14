#!/bin/bash
# 보드(Raspberry Pi 5 + D435 + DX-M1) 실행 전 리눅스 설정. sudo 로 한 번 돌린다(재부팅하면 풀리는 것은 다시).
# 쓰기: sudo scripts/board_tune.sh [--no-desktop] [--hugepages N] [--user 이름]
#   - CPU 주파수 관리자 performance(코어 전부): 주파수 오르내림으로 생기는 장면 시간 흔들림 없앰
#   - D435(USB 8086:0b07) USB 절전 끄기: 스트림 끊김·다시 잡기 없앰
#   - 투명 큰 페이지 madvise, 큰 페이지 N 개(선택): 깊이·지도 큰 버퍼의 TLB 놓침 줄임
#   - 쓰지 않는 서비스 멈춤(블루투스·프린터·모뎀·avahi), --no-desktop 이면 그래픽 화면도
#   - 실시간 권한: --user 의 rtprio 99·memlock 무제한(ondevice --rt auto --mlock 이 쓰도록, 다시 로그인 후)
#   - USB(xhci) 인터럽트를 코어 0 에, 나머지 코어는 계산에(ondevice --threads 기본 배치와 맞춤)
set -u
NODESK=0; HP=0; USR=${SUDO_USER:-}
while [ $# -gt 0 ]; do
  case $1 in
    --no-desktop) NODESK=1 ;;
    --hugepages) HP=$2; shift ;;
    --user) USR=$2; shift ;;
  esac; shift
done
[ "$(id -u)" = 0 ] || { echo "sudo 로 실행"; exit 1; }

for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo performance > "$g" 2>/dev/null; done
echo "주파수 관리자: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null)"

for d in /sys/bus/usb/devices/*; do
  if [ -f "$d/idVendor" ] && [ "$(cat $d/idVendor)" = 8086 ]; then
    echo on > "$d/power/control" 2>/dev/null; echo -1 > "$d/power/autosuspend_delay_ms" 2>/dev/null
    echo "USB 절전 끔: $d ($(cat $d/idProduct))"
  fi
done
# 새로 꽂아도 절전 끔
cat > /etc/udev/rules.d/99-d435-power.rules <<'R'
ACTION=="add", SUBSYSTEM=="usb", ATTR{idVendor}=="8086", TEST=="power/control", ATTR{power/control}="on"
R
udevadm control --reload 2>/dev/null

echo madvise > /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null
[ "$HP" -gt 0 ] && echo "$HP" > /proc/sys/vm/nr_hugepages && echo "큰 페이지 $(cat /proc/sys/vm/nr_hugepages) 개"
sysctl -q vm.swappiness=10

for s in bluetooth cups cups-browsed ModemManager avahi-daemon triggerhappy; do systemctl stop "$s" 2>/dev/null && echo "멈춤: $s"; done
[ $NODESK = 1 ] && systemctl isolate multi-user.target && echo "그래픽 화면 멈춤(다시: systemctl isolate graphical.target)"

if [ -n "$USR" ]; then
  cat > /etc/security/limits.d/99-ondevice.conf <<L
$USR - rtprio 99
$USR - memlock unlimited
L
  echo "실시간 권한: $USR (다시 로그인하면 적용)"
fi

for irq in $(grep -E 'xhci|usb' /proc/interrupts | awk -F: '{print $1}'); do echo 1 > /proc/irq/$irq/smp_affinity 2>/dev/null && echo "IRQ $irq → 코어 0"; done
echo 끝
