#!/bin/bash
# PC(우분투/WSL)에서 한 프로그램(ondevice)을 뷰어와 함께 돌린다: 위치(slam) + 물체 기억 + 뷰어 서버가 한 프로세스.
#   bash scripts/run.sh tum <TUM 폴더> [기억 폴더=data/memory/run] [더 줄 인자…]   재생(찍은 속도, 밀리면 버림)
#   bash scripts/run.sh replay <TUM 폴더> [기억 폴더] [인자…]                       결정적 재생(--no-drop, 최대 속도)
#   bash scripts/run.sh d435 [기억 폴더] [인자…]                                      D435 실시간
# 뷰어: http://localhost:${PORT:-8080}. 기억 폴더는 메모리에 두고 끝에 디스크에 쓴다(SAVE_EVERY=초 이면 그 간격으로도, 뷰어 저장 버튼도).
# BIN=<실행 파일>(기본 target/release/ondevice), SEG=<분할 모델>(기본 models/ObjectSAM-416_640.onnx), EMB=<SigLIP2 영상 인코더>, VOCAB=<이름 표>(기본 vocab/indoor.voc)
# 물체는 ObjectSAM 으로 자르고 이름은 SigLIP2 사진 벡터가 가장 닮은 실내 낱말로 붙인다. COCO 이름(YOLO)은 SEG=models/yolo26s-seg.onnx 로 직접, INI=<RTAB-Map 설정>(기본 docs/rtabmap_pi.ini)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
BIN=${BIN:-$ROOT/target/release/ondevice}
SEG=${SEG:-$ROOT/models/ObjectSAM-416_640.onnx}
EMB=${EMB:-$ROOT/models/siglip2-base-patch32-256/vision_model.onnx}
VOCAB=${VOCAB:-$ROOT/vocab/indoor.voc}
INI=${INI:-$ROOT/docs/rtabmap_pi.ini}
PORT=${PORT:-8080}
[ -x "$BIN" ] || (cd "$ROOT" && cargo build --release)
mode=${1:?tum | replay | d435}; shift
case $mode in
  tum|replay) SRC=${1:?TUM 폴더}; shift; SLAM="--tum $SRC --config $INI" ;;
  d435) SLAM="--realsense --config $INI --send-every 2" ;;
  *) echo "모름: $mode (tum | replay | d435)"; exit 2 ;;
esac
OUT=${1:-$ROOT/data/memory/run}; [ $# -gt 0 ] && shift
case $mode in
  tum) SLAM="$SLAM --rate 1"; EXTRA=(--live 1) ;;
  replay) SLAM="$SLAM --rate 0 --block"; EXTRA=(--no-drop) ;;
  d435) EXTRA=(--live 1) ;;
esac
SAVE=(); [ -n "${SAVE_EVERY:-}" ] && SAVE=(--save-every "$SAVE_EVERY")
echo "뷰어: http://localhost:$PORT  기억 폴더: $OUT"
exec "$BIN" --slam "$SLAM" --seg "$SEG" --embed "$EMB" --vocab "$VOCAB" --out "$OUT" --serve "$PORT" "${SAVE[@]}" "${EXTRA[@]}" "$@"
