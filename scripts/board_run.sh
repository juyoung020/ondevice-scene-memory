#!/usr/bin/env bash
# 보드(Raspberry Pi 5 + D435 + DX-M1) 실시간 실행: D435 → 위치 추적 → 물체 기억 → 뷰어(:8091) 한 프로세스.
#   bash scripts/board_run.sh [--tum <TUM 폴더>(녹화 속도 그대로, RATE=0 이면 최대 속도)] [ondevice 에 더 넘길 인자...]
# 기본 모델(MODELS=~/models):
#   - 분할: objectsam-416_640.dxnn (클래스 없는 "object", NPU). 이름은 SigLIP2 사진 벡터 → 실내 낱말 표(vocab/indoor.voc, 104 개)로 붙는다.
#     전체 표(names.voc, 1,225 개)는 실제 녹화(어둡고 어수선한 사진)에서 몫이 비슷한 이름에 갈려 문턱을 거의 넘지 못했다
#   - 사진 벡터: siglip2-vision_npu.dxnn (NPU 만, 입력은 32×32 조각 64 개). 분할과 같은 NPU 에서 함께 돈다(분할은 --npu-jobs 장까지 겹쳐 올림)
#   - 뷰어 검색 글 인코더: $M/siglip2(text_model_fp16.onnx 의 가중치를 Rust 로 계산, tokenizer.json)
# 실시간 설정: --rt auto(장면 받는 스레드만 SCHED_FIFO, 코어 3) · --mlock. 추론 스레드와 DXRT 런타임 스레드는 보통 스케줄.
#   권한은 `sudo scripts/board_tune.sh --user $USER` 후 다시 로그인.
# 빌드: bash scripts/board_build.sh <브랜치> (.cargo/config.toml 이 target-cpu=cortex-a76, Cargo.toml 이 fat LTO)
set -euo pipefail
cd "$(dirname "$0")/.."
M=${MODELS:-$HOME/models}
SEG=${SEG:-$M/objectsam-416_640.dxnn}
EMB=${EMB:-$M/siglip2-vision_npu.dxnn}
OUT=${OUT:-data/memory/live}
SRC="--realsense --config docs/rtabmap_pi.ini --send-every 2 --reset-file $OUT/reset.req"
if [ "${1:-}" = "--tum" ]; then SRC="--tum $2 --rate ${RATE:-1} --config docs/rtabmap_pi.ini --send-every 2"; shift 2; fi
RT=${RT:-auto}
VOCAB=${VOCAB:-vocab/indoor.voc}
# NPU 코어 묶기(DX-M1 코어 0·1·2, all = 셋 다). 2026-11-15 fr1_desk 에서 all/all 과 01/2 는 같은 속도(NPU 는 분할 9 ms · 사진 13 ms 로 한가하고, 시간은 CPU 쪽)
SEG_NPU=${SEG_NPU:-all}
EMB_NPU=${EMB_NPU:-all}
# 분할을 NPU 에 함께 올릴 장 수. 2026-11-15 fr1_desk: 1 · 2 · 4 장에서 분할 지연 41 · 53 · 62 ms, 장면 처리 속도는 같음
NPU_JOBS=${NPU_JOBS:-1}
rm -rf "$OUT"; mkdir -p "$OUT"
exec "${BIN:-target/release/ondevice}" --slam "$SRC" \
    --every 1 --live 1 --seg "$SEG" --embed "$EMB" --vocab "$VOCAB" \
    --max-depth 3.5 --out "$OUT" --rt "$RT" --mlock --seg-npu "$SEG_NPU" --embed-npu "$EMB_NPU" --npu-jobs "$NPU_JOBS" \
    --serve "8091 --bind 0.0.0.0 --search-model $M/siglip2 --vocab vocab/names.voc" "$@"
