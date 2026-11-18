#!/usr/bin/env bash
# 루프 · 재위치 NPU 모델(MegaLoc · XFeat) 보드 확인. PC(WSL)에서 부르면 기준 출력을 만들고 보드로 올린 뒤 보드에서 잰다.
#   bash scripts/board_lc_check.sh            (저장소 폴더에서, PC WSL)
# 환경 변수
#   BOARD=kau@192.168.219.101   보드
#   MODELS=models               PC 의 모델 폴더(아래 파일들, 만드는 법 docs/npu_models.md)
#       megaloc_168x224.dxnn · megaloc_168x224.onnx(그 원본, PC 실수) · megaloc_fp32.onnx(손질 전 원본 계산) · megaloc_head_pca1024.bin
#       xfeat_480x640.dxnn · xfeat_480x640.onnx(그 원본, u8 입력, PC 실수)
#       objectsam-416_640.dxnn · siglip2-vision_npu.dxnn(함께 올리기 시험용, 보드 ~/models 에 이미 있으면 그것)
#   IMG=사진 한 장(640×480 PNG, 기본 TUM fr1_desk 첫 장)   STRESS_S=60(함께 돌리기 초)
#   CARGO_TARGET_DIR(PC 빌드 폴더, 기본 /home/rea/tgt-lc). 보드용 lc_bench(dxrt)는 PC 에서 교차 빌드(scripts/cross_board.sh 와 같은 설정)
# 하는 일
#   PC : lc_bench(onnx) 빌드 → 입력 바이트(prep-in) → PC 실수 ONNX 출력 저장(기준 둘: 원본 실수 · NPU 원본 ONNX 실수)
#   보드: 파일 · 교차 빌드한 lc_bench(dxrt) 올리기 · sha256 맞춤 → parse_model · run_model 단독 시간 → PC 기준과 출력 · 기술자 코사인
#         → 네 모델(ObjectSAM · SigLIP2 · MegaLoc · XFeat) 동시 적재 + 동시 실행 STRESS_S 초(앞뒤 dxrt-cli 상태 · 온도 · 오류 기록)
# 결과: 보드 ~/lccheck/<시각>/ 와 PC 같은 이름 폴더(results/), summary.txt
set -uo pipefail
cd "$(dirname "$0")/.."
REPO=$(pwd)
BOARD=${BOARD:-kau@192.168.219.101}
MODELS=${MODELS:-models}
IMG=${IMG:-$(ls /mnt/c/ondevice/data/tum/rgbd_dataset_freiburg1_desk/rgb/*.png | head -1)}
STRESS_S=${STRESS_S:-60}
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-/home/rea/tgt-lc}
STAMP=$(date +%m%d_%H%M)
LOCAL=$REPO/results/lccheck_$STAMP
mkdir -p "$LOCAL"
log() { echo "$*" | tee -a "$LOCAL/summary.txt"; }

# ---------- PC: 기준 ----------
cargo build --release -p slam --features onnx --example lc_bench 2>&1 | tail -1
LB=$CARGO_TARGET_DIR/release/examples/lc_bench
"$LB" prep-in megaloc "$IMG" "$LOCAL/ml_in.bin"
"$LB" prep-in xfeat8 "$IMG" "$LOCAL/xf_in.bin"
"$LB" npu-check "$MODELS/megaloc_fp32.onnx" "$LOCAL/ml_in.bin" "$LOCAL/ref_ml" --save > /dev/null
"$LB" npu-check "$MODELS/megaloc_168x224.onnx" "$LOCAL/ml_in.bin" "$LOCAL/ref_ml_npu32" --save > /dev/null
"$LB" npu-check "$MODELS/xfeat_480x640.onnx" "$LOCAL/xf_in.bin" "$LOCAL/ref_xf" --save > /dev/null
( cd "$MODELS" && sha256sum megaloc_168x224.dxnn xfeat_480x640.dxnn megaloc_head_pca1024.bin ) > "$LOCAL/sha256.txt"
log "PC 기준 출력: $LOCAL"

# ---------- 보드용 lc_bench(dxrt, aarch64) 교차 빌드 ----------
SYS=${BOARD_SYSROOT:-$HOME/board-sysroot}
(
  export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc CC_aarch64_unknown_linux_gnu=aarch64-linux-gnu-gcc
  export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_RUSTFLAGS="-C target-cpu=cortex-a76 -L $SYS/lib -C link-arg=-Wl,--allow-shlib-undefined"
  cargo build --release --target aarch64-unknown-linux-gnu -p slam --features dxrt --example lc_bench 2>&1 | tail -1
)
LBA=$CARGO_TARGET_DIR/aarch64-unknown-linux-gnu/release/examples/lc_bench

# ---------- 보드로 ----------
R=lccheck/$STAMP
ssh "$BOARD" "mkdir -p ~/$R ~/models" || { log "보드 연결 실패: $BOARD"; exit 1; }
scp -q "$MODELS/megaloc_168x224.dxnn" "$MODELS/xfeat_480x640.dxnn" "$MODELS/megaloc_head_pca1024.bin" "$BOARD:models/"
scp -q -r "$LOCAL/ml_in.bin" "$LOCAL/xf_in.bin" "$LOCAL/ref_ml" "$LOCAL/ref_ml_npu32" "$LOCAL/ref_xf" "$LOCAL/sha256.txt" "$BOARD:$R/"
scp -q "$LBA" "$BOARD:$R/lc_bench"

ssh "$BOARD" bash -s "$R" "$STRESS_S" > "$LOCAL/board.txt" 2>&1 <<'EOS'
set -uo pipefail
R=$HOME/$1; S=$2
cd ~/models && sha256sum -c "$R/sha256.txt" && echo "sha256 같음"
LB=$R/lc_bench; chmod +x $LB
echo "== 장치 상태(앞)"; dxrt-cli -s 2>&1 | tee "$R/status_before.txt"
for m in megaloc_168x224 xfeat_480x640; do
    echo "== parse_model $m"; parse_model -m ~/models/$m.dxnn 2>&1 | tail -25
    echo "== run_model $m(NPU 계산 · 지연)"; run_model -m ~/models/$m.dxnn -s -v -l 100 2>&1 | tail -12
done
echo "== MegaLoc: PC 원본 실수 대비"; $LB npu-check ~/models/megaloc_168x224.dxnn "$R/ml_in.bin" "$R/ref_ml" --head ~/models/megaloc_head_pca1024.bin --runs 50
echo "== MegaLoc: PC NPU 원본 ONNX(실수) 대비"; $LB npu-check ~/models/megaloc_168x224.dxnn "$R/ml_in.bin" "$R/ref_ml_npu32" --head ~/models/megaloc_head_pca1024.bin --runs 1
echo "== XFeat: PC 실수(u8 입력 ONNX) 대비"; $LB npu-check ~/models/xfeat_480x640.dxnn "$R/xf_in.bin" "$R/ref_xf" --runs 50
# 네 모델 동시 적재 + 동시 실행: 입력은 0 으로 채운 바이트(크기만 맞춤)
head -c $((640*640*3)) /dev/zero > "$R/osam_in.bin"
head -c $((64*3072*4)) /dev/zero > "$R/sig_in.bin"
echo "== 네 모델 동시 ${S} 초"; DXRT_TASK_MAX_LOAD=2 timeout $((S + 60)) $LB npu-stress "$S" \
    ~/models/objectsam-416_640.dxnn:"$R/osam_in.bin" ~/models/siglip2-vision_npu.dxnn:"$R/sig_in.bin" \
    ~/models/megaloc_168x224.dxnn:"$R/ml_in.bin" ~/models/xfeat_480x640.dxnn:"$R/xf_in.bin" 2>&1; echo "끝 코드 $?"
echo "== 장치 상태(뒤)"; dxrt-cli -s 2>&1 | tee "$R/status_after.txt"
dmesg 2>/dev/null | grep -i -E "dx|npu" | tail -10
EOS
scp -q -r "$BOARD:$R/." "$LOCAL/board/" 2>/dev/null
grep -E "sha256|코사인|ms/번|번/초|적재|오류|끝 코드|Inference|FPS|latency|Temperature|temp|Memory|memory" "$LOCAL/board.txt" | tee -a "$LOCAL/summary.txt"
log "전체 기록: $LOCAL/board.txt"
