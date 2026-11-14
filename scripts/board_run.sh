#!/usr/bin/env bash
# 보드(Raspberry Pi 5 + D435 + DX-M1) 실시간 실행: D435 → 위치 추적 → 물체 기억 → 뷰어(:8091) 한 프로세스.
#   bash scripts/board_run.sh [--tum <TUM 폴더>] [ondevice 에 더 넘길 인자...]
# 기본 모델(MODELS=~/models):
#   - 분할: objectsam-416_640.dxnn (클래스 없는 "object", NPU). 이름은 SigLIP2 사진 벡터 → 낱말 표(vocab/names.voc)로 붙는다
#   - 사진 벡터: siglip2-vision_v8.dxnn (NPU). 분할과 같은 NPU 에서 함께 돈다(분할은 --npu-jobs 장까지 겹쳐 올림)
#   - 뷰어 검색 글 인코더: $MODELS/siglip2 (CPU ONNX)
# 실시간 설정: --rt auto(역할별 SCHED_FIFO) · --mlock. 권한은 한 번 `sudo scripts/board_tune.sh --user $USER` 후 다시 로그인.
# 빌드: bash scripts/board_build.sh <브랜치> (.cargo/config.toml 이 target-cpu=cortex-a76, Cargo.toml 이 fat LTO)
set -euo pipefail
cd "$(dirname "$0")/.."
M=${MODELS:-$HOME/models}
SEG=${SEG:-$M/objectsam-416_640.dxnn}
EMB=${EMB:-$M/siglip2-vision_v8.dxnn}
OUT=${OUT:-data/memory/live}
SRC="--realsense --config docs/rtabmap_pi.ini --send-every 2 --reset-file $OUT/reset.req"
if [ "${1:-}" = "--tum" ]; then SRC="--tum $2 --config docs/rtabmap_pi.ini --send-every 2"; shift 2; fi
[ "$(ulimit -r)" = 0 ] && echo "board_run: 실시간 권한 없음 — sudo scripts/board_tune.sh --user $USER 후 다시 로그인(그대로 진행)" >&2
rm -rf "$OUT"; mkdir -p "$OUT"
exec target/release/ondevice --slam "$SRC" \
    --every 1 --live 1 --seg "$SEG" --embed "$EMB" --vocab vocab/names.voc \
    --max-depth 3.5 --out "$OUT" --rt auto --mlock \
    --serve "8091 --bind 0.0.0.0 --search-model $M/siglip2 --vocab vocab/names.voc" "$@"
