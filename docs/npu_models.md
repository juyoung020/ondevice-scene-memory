# 보드 NPU 모델(.dxnn) 만들기

보드(Pi 5 + DX-M1, DXRT 3.4)는 파일 형식 6 ~ 8 을 연다. 컴파일러는 **DX-COM 2.4.1**(형식 8)을 쓴다(2.5.0 은 형식 9 라 열리지 않음). 데스크톱 WSL 의 `~/dx241` 가상 환경에 설치되어 있다.

두 모델 모두 NPU 작업 하나로만 돈다(CPU 작업 없음, `parse_model -m <파일>` 로 확인). 모델 안에 CPU 작업이 남으면 DXRT 가 ONNX Runtime 으로 그 부분을 돌리며 CPU 를 쓴다.

ONNX 손질은 `tools` 의 `npu_prep`(Rust)으로 한다.

```bash
cargo build --release -p tools --bin npu_prep
P=target/release/npu_prep
```

## 분할: objectsam-416_640.dxnn

입력 `images` u8 [1,640,640,3](레터박스, 회색 114, RGB). 출력은 머리 9 개(보폭 8 · 16 · 32 마다 상자 거리 [4,n,n] · 점수 로짓 [1,n,n] · 마스크 계수 [32,n,n])와 원형 `output1` [32,160,160]. 점수 문턱 → 상위 300 → 상자 해독은 `pipeline/src/seg.rs` 의 `heads_to_det` 가 한다(원래 모델 안 후처리와 같은 값).

```bash
# 1) 머리 출력에서 끊기(모델 안 상위 300 고르기를 뺌). DX-COM 의 --compile_output_nodes 는 뒤를 CPU 작업으로 남겨서 쓰지 않는다
N=$(for l in 0 1 2; do for c in 2 3 4; do printf "/model.23/one2one_cv$c.$l/one2one_cv$c.$l.2/Conv_output_0,"; done; done)output1
$P cut models/ObjectSAM-416_640.onnx osam_heads.onnx "$N"
# 2) 컴파일(보정: TUM fr1_desk 4 장마다 1 장 + fr3_long_office 17 장마다 1 장, 305 장을 calib/ 에)
dxcom -m osam_heads.onnx -c osam.json -o out
```

`osam.json`:

```json
{"inputs":{"images":[1,3,640,640]},"calibration_num":300,"calibration_method":"ema",
 "default_loader":{"dataset_path":"calib","file_extensions":["png"],
  "preprocessings":[{"resize":{"mode":"pad","size":640,"pad_location":"edge","pad_value":[114,114,114]}},
   {"div":{"x":255.0}},{"convertColor":{"form":"BGR2RGB"}},{"transpose":{"axis":[2,0,1]}},{"expandDim":{"axis":0}}]}}
```

확인(2026-11-15, fr1_desk 304 장, PC ONNX 대비): 물체별 마스크 IoU 평균 0.861, 0.5 이상 89.6 %. NPU 계산 9 ms(`run_model -s -v`).

## 사진 벡터: siglip2-vision_npu.dxnn

`docs/performance_log.md` 의 SigLIP2 절(토큰 고정 · 회전 · GELU 나누기)로 만든 영상 인코더 ONNX(`vs3.onnx`)에서 시작한다.

```bash
# 1) 조각 나누기 합성곱(32×32, 보폭 32)을 행렬곱으로(입력 patches [1,64,3072]), 끝의 Reshape·Transpose·Gather 제거
$P siglip-patches vs3.onnx vs3_patches.onnx
# 2) 보정 자료: 물체 사진(png) 200 장 → [200,64,3072] (embed.rs 와 같은 전처리)
$P patches-calib <물체 사진 폴더> patches_calib.npy 200
# 3) 컴파일: dxcom 명령의 자료 읽기는 사진만 받아서, DX-COM 의 파이썬 API dx_com.compile 에 patches_calib.npy 를
#    한 장씩 내주는 dataloader 를 넘긴다(calibration_method="ema", calibration_num=200, quant_diagnosis=True)
```

확인: 컴파일러 진단 코사인 0.9907(이전 판 0.973). 보드에서 PC 기준 벡터와 코사인 평균 0.928(사진 40 장, `ondevice --embed-check`). NPU 계산 13.5 ms.

프로그램은 입력 이름이 `patches` 이면 조각 배치로 전처리한다(`embed.rs` `preprocess_into`).
