#!/usr/bin/env bash
# 보드(Raspberry Pi 5 + D435 + DX-M1) 실시간 실행: D435 → 위치 추적 → 물체 기억 → 뷰어(:8091) 한 프로세스.
#   bash scripts/board_run.sh [--tum <TUM 폴더>] [ondevice 에 더 넘길 인자...]
# 기본 모델(MODELS=~/models):
#   - 분할: objectsam-416_640.dxnn (클래스 없는 "object", NPU). 이름은 SigLIP2 사진 벡터 → 낱말 표(vocab/names.voc)로 붙는다
#   - 사진 벡터: siglip2-vision_v8.dxnn (NPU). 분할과 같은 NPU 에서 함께 돈다(분할은 --npu-jobs 장까지 겹쳐 올림)
#   - 뷰어 검색 글 인코더: $MODELS/siglip2 (CPU ONNX)
# 실시간 설정: --rt auto(장면 받는 스레드만 SCHED_FIFO, 코어 3) · --mlock. 추론 스레드와 DXRT 런타임 스레드는 보통 스케줄.
#   권한은 `sudo scripts/board_tune.sh --user $USER` 후 다시 로그인.
# 빌드: bash scripts/board_build.sh <브랜치> (.cargo/config.toml 이 target-cpu=cortex-a76, Cargo.toml 이 fat LTO)
set -euo pipefail
cd "$(dirname "$0")/.."
M=${MODELS:-$HOME/models}
SEG=${SEG:-$M/objectsam-416_640.dxnn}
EMB=${EMB:-$M/siglip2-vision_v8.dxnn}
OUT=${OUT:-data/memory/live}
SRC="--realsense --config docs/rtabmap_pi.ini --send-every 2 --reset-file $OUT/reset.req"
if [ "${1:-}" = "--tum" ]; then SRC="--tum $2 --config docs/rtabmap_pi.ini --send-every 2"; shift 2; fi
RT=${RT:-auto}
# NPU 코어 묶기(DX-M1 코어 0·1·2, all = 셋 다): 분할과 사진 벡터를 다른 코어에 두어 줄 서지 않고 함께 돌게
SEG_NPU=${SEG_NPU:-all}
EMB_NPU=${EMB_NPU:-all}
rm -rf "$OUT"; mkdir -p "$OUT"
exec target/release/ondevice --slam "$SRC" \
    --every 1 --live 1 --seg "$SEG" --embed "$EMB" --vocab vocab/names.voc \
    --max-depth 3.5 --out "$OUT" --rt "$RT" --mlock --seg-npu "$SEG_NPU" --embed-npu "$EMB_NPU" \
    --serve "8091 --bind 0.0.0.0 --search-model $M/siglip2 --vocab vocab/names.voc" "$@"
