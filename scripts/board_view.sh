#!/bin/bash
# 보드(Pi 5) 화면(HDMI, DISPLAY :0)에 뷰어를 띄운다: 같은 보드에서 도는 ondevice --serve 의 서버를 Chromium 앱 창(키오스크)으로.
#   [Q="metrics=1"(2 초마다 fps · 위치→화면 지연을 서버 로그로)] bash scripts/board_view.sh [포트=8080] [코어=0] [디버깅 포트=9222]
# 보드에서 실행(ssh 로 부르면 DISPLAY=:0 로 화면에 뜸). GPU: V3D 로 WebGL(--use-gl=egl), 화면 맞춤 그리기 그대로(화질 그대로).
# CPU: 브라우저 전체를 코어 하나(기본 0: --threads auto 의 위치 · 카메라 받기 · 분할 코어와 겹치지 않는 쪽)에 묶고 nice 10 —
# 위치·NPU 를 굶기지 않게. 디버깅 포트로 PC 에서 프레임 시간을 잰다: ssh -L 9222:localhost:9222 <보드>
set -euo pipefail
PORT=${1:-8080}; CORE=${2:-0}; DBG=${3:-9222}
export DISPLAY=${DISPLAY:-:0}
B=$(command -v chromium || command -v chromium-browser)
for i in $(seq 1 60); do curl -s -o /dev/null "http://localhost:$PORT/api/mode" && break; sleep 0.5; done
exec taskset -c "$CORE" nice -n 10 "$B" --app="http://localhost:$PORT/${Q:+?$Q}" --kiosk --noerrdialogs --disable-infobars \
  --no-first-run --disable-features=Translate --password-store=basic --user-data-dir="$HOME/.cache/ondevice-view" \
  --use-gl=egl --enable-gpu-rasterization --ignore-gpu-blocklist --disable-background-timer-throttling \
  --remote-debugging-port="$DBG"
