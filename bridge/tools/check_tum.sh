#!/usr/bin/env bash
# TUM 폴더 하나로 연결 프로그램 경로를 오프라인 경로와 견줌(docs/bridge.md "검증").
#   check_tum.sh <TUM 폴더> <출력 폴더> [최대 거리 m(기본 3.5)]
# 환경 변수: BRIDGE(rtab_bridge), MEMMAP·ATE·SURFCMP(pipeline 바이너리), INI(RTAB-Map 설정), SEG(분할 모델 .onnx)
#   1) 기준: 같은 설정 + 모든 장면을 지도 노드로(Rtabmap/CreateIntermediateNodes true) 끝까지 돌린 뒤 최적화된 위치 → memmap --tum --gt
#   2) 스트림: rtab_bridge(보드와 같은 설정) | memmap --stream -
#   3) 같은 위치 다른 입력: 스트림으로 보낸 위치 파일 → memmap --tum --gt (입력 경로만 다름)
#   4) 위치 오차(ate, 정답 기준), 물체 수, 표면(surfcmp --align, 두 방향)
# 출력 폴더에 이미 결과가 있으면 그 단계는 건너뜀.
set -euo pipefail
SEQ=$1
OUT=$2
MAXD=${3:-3.5}
: "${BRIDGE:?}" "${MEMMAP:?}" "${ATE:?}" "${SURFCMP:?}" "${INI:?}" "${SEG:?}"
mkdir -p "$OUT"
MM_ARGS=(--depth kinect --surface --seg "$SEG" --every 2 --max-depth "$MAXD")

if [ ! -f "$OUT/offline.log" ]; then
  "$BRIDGE" --tum "$SEQ" --config "$INI" --Rtabmap/CreateIntermediateNodes true --out /dev/null \
    --final-poses "$OUT/ref_final.txt" 2> "$OUT/ref_bridge.log"
  "$MEMMAP" --tum "$SEQ" --gt "$OUT/ref_final.txt" "${MM_ARGS[@]}" --out "$OUT/offline" > "$OUT/offline.log" 2>&1
fi
if [ ! -f "$OUT/stream.log" ]; then
  "$BRIDGE" --tum "$SEQ" --config "$INI" --poses "$OUT/sent.txt" --final-poses "$OUT/final.txt" --stats "$OUT/stats.csv" 2> "$OUT/bridge.log" \
    | "$MEMMAP" --stream - "${MM_ARGS[@]}" --out "$OUT/stream" > "$OUT/stream.log" 2> "$OUT/stream.err"
fi
if [ ! -f "$OUT/offline_sent.log" ]; then
  "$MEMMAP" --tum "$SEQ" --gt "$OUT/sent.txt" "${MM_ARGS[@]}" --out "$OUT/offline_sent" > "$OUT/offline_sent.log" 2>&1
fi

GT="$SEQ/groundtruth.txt"
echo "== 위치 오차(정답 기준)"
echo "기준(전 장면 노드, 끝에 최적화): $("$ATE" "$OUT/ref_final.txt" "$GT")"
echo "스트림으로 보낸 위치(그때그때):   $("$ATE" "$OUT/sent.txt" "$GT")"
echo "스트림 실행의 지도 노드(끝):     $("$ATE" "$OUT/final.txt" "$GT")"
echo "memmap 이 받은 위치(poses_est):  $("$ATE" "$OUT/stream/poses_est.txt" "$GT")"
echo "보낸 위치 대 memmap 받은 위치:    $("$ATE" "$OUT/stream/poses_est.txt" "$OUT/sent.txt" 0.001)"
echo "== 물체·장면(키프레임, 물체, kf/s)"
for k in offline offline_sent stream; do
  L=$(tail -1 "$OUT/$k.log")
  echo "$k: $(echo "$L" | grep -o '"keyframes":[0-9]*') $(echo "$L" | grep -o '"objects":[0-9]*') $(echo "$L" | grep -o '"kf_per_s":[0-9.]*') $(echo "$L" | grep -o '"stream_lost":[0-9]*' || true)"
done
grep -h "수평 맞춤\|스트림 끝\|새 지도" "$OUT/stream.err" || true
echo "== 표면: 스트림 → 같은 위치 오프라인, 반대"
"$SURFCMP" "$OUT/stream" "$OUT/offline_sent" --align
"$SURFCMP" "$OUT/offline_sent" "$OUT/stream" --align
echo "== 표면: 스트림 → 기준, 반대"
"$SURFCMP" "$OUT/stream" "$OUT/offline" --align
"$SURFCMP" "$OUT/offline" "$OUT/stream" --align
echo "== 연결 프로그램"
grep -v WARN "$OUT/bridge.log" | tail -9
