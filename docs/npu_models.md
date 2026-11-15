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
# 2) 보정 자료: 물체 사진(png) 200 장 → [200,1,64,3072] (embed.rs 와 같은 전처리)
$P patches-calib <물체 사진 폴더> patches_calib.npy 200
# 3) 컴파일: dxcom 명령의 자료 읽기는 사진만 받아서, DX-COM 의 파이썬 API dx_com.compile 에 patches_calib.npy 를
#    한 장씩 내주는 dataloader 를 넘긴다(calibration_method="ema", calibration_num=200, quant_diagnosis=True)
```

확인: 컴파일러 진단 코사인 0.9907(이전 판 0.973). 보드에서 PC 기준 벡터와 코사인 평균 0.928(사진 40 장, `ondevice --embed-check`). NPU 계산 13.5 ms.

프로그램은 입력 이름이 `patches` 이면 조각 배치로 전처리한다(`embed.rs` `preprocess_into`).

## 글 인코더(뷰어 검색): Rust

뷰어 검색의 SigLIP2 글 인코더는 `viewer/sgview/src/textnet.rs` 가 계산한다. `text_model_fp16.onnx` 의 가중치(f16)를 그대로 읽어 토큰 표 + 위치 → 블록 12 개(LayerNorm · 12 머리 주의 · GELU tanh) → 마지막 토큰 LayerNorm → head 를 f32 로 돈다. 보드 빌드(`dxrt`)는 이 인코더를 쓰고, PC 빌드(`onnx`)는 ONNX Runtime 이 더 빨라 그쪽을 쓴다. 토크나이저는 `tokenizers`(정규식 fancy-regex, Rust)다.

| 확인 (2026-11-15) | 값 |
|---|---|
| Rust ↔ ONNX Runtime, 글 1,329 개(이름 1,225 + 한·영 질의 104, PC) | 코사인 최소 0.9999985, 최대 차 1.2e-2 |
| 토큰(fancy-regex ↔ 파이썬 tokenizers) | 1,329 개 모두 같음 |
| PC(Ryzen 9 9950X) 글 하나 | Rust 75.7 ms · ONNX Runtime(스레드 4) 34.1 ms |
| 보드 질의 104 개(한·영): 이름 1,225 개 중 1 위 · 상위 5 개 | ONNX Runtime 과 104 / 104 같음, 상위 5 개 100 % 같음 |
| 보드 글 하나(다른 실행이 함께 돌 때, 부하 평균 15) | 약 75 ms(이전 ONNX Runtime CPU 0.45 초). 가중치 읽기 9.5 초는 뷰어가 뜰 때 한 번 |

행렬곱은 열을 64 개씩 나눠 스레드에 주고, 한 덩어리의 64 토큰 × 64 열 합을 L1 에 둔 채 가중치를 한 번만 읽는다(보드 캐시 2 MB 에 가중치 340 MB 가 들지 않아 토큰마다 가중치를 다시 읽으면 메모리에 막힌다). 원소마다 더하는 순서는 편향 → 입력 순서 그대로다.

NPU 판(`.dxnn`)도 만들어 보드에서 쟀다: 토큰 고정 · 회전 · GELU 나누기를 넣은 모델(DX-COM 2.4.1, 컴파일러 진단 QSNR 30 dB)이 보드에서는 ONNX 와 코사인 평균 0.60, 서로 다른 글끼리 코사인 0.97(출력이 거의 한 방향으로 모임), 질의 104 개의 첫 이름이 하나도 맞지 않았다. CPU 쪽 입력(토큰 표 + 위치, 회전)은 보정 자료와 차 1e-3 안이라 NPU 실행 결과가 진단과 다른 것이다. 그래서 글 인코더는 CPU(Rust)로 돈다.

## PC 모델 실행기: ONNX Runtime 유지

순수 Rust 실행기 tract 0.21 을 같은 ONNX 로 견줬다(PC, 입력 고정, 10 번 중앙값).

| 모델 | ONNX Runtime 스레드 1 / 8 | tract | 출력 |
|---|---|---|---|
| ObjectSAM 분할 640×640 | 48.8 / 25.0 ms | 297 ms | 코사인 1.000000, 최대 차 2.4e-4 |
| SigLIP2 영상 인코더 | 59.6 / 26.3 ms | 읽지 못함(모양식 `floor(height/32)`) | — |
| SigLIP2 글 인코더(fp16) | 34 ms | 읽지 못함(동적 배치 차원) | — |

tract 는 분할에서 6 배 느리고 SigLIP2 둘을 읽지 못해 PC 는 ONNX Runtime 을 쓴다. 보드는 분할 · 사진 벡터가 NPU(DXRT), 글 벡터가 Rust 라 ONNX Runtime 을 넣지 않는다.

## DXRT 설정

DXRT 3.4 C API(`/usr/local/include/dxrt/dxrt_c_api.h`)와 런타임 소스(`~/dx-all-suite/dx-runtime/dx_rt/lib`)에서 바꿀 수 있는 것 전부와 고른 값. 설정은 `common::dxrt::Config` 하나에 모았고 `ondevice` 명령줄로 받는다.

| 설정 | 하는 일 | 라이브러리 기본 | 우리 값 |
|---|---|---|---|
| `bound_option`(`--seg-npu` · `--embed-npu`) | 모델을 NPU 코어 0 · 1 · 2 중 어디에 묶을지(`all` · `0` · `1` · `2` · `01` · `12` · `02`) | 셋 다 | 셋 다(fr1_desk 에서 `01`/`2` 와 같은 속도, NPU 는 한가하고 시간은 CPU 쪽) |
| `buffer_count`(`--npu-buffers N`) | 모델마다 잡는 입출력 버퍼 수 = 함께 올릴 수 있는 추론 수 상한. 환경 변수 `DXRT_TASK_MAX_LOAD` 로도(1 – 100) | 6 | 6(`--npu-jobs` 최대 8 보다 작으면 올리기에서 기다림) |
| `use_ort`(`--npu-ort`) | 모델 안 CPU 작업을 DXRT 안의 ONNX Runtime 으로 돌림 | ORT 가 있으면 켬 | 끔(두 모델 모두 NPU 작업 하나) |
| `device_id` | 장치 고르기 | 자동 | 자동(DX-M1 하나) |
| NFH 일꾼 스레드(`--npu-nfh 입력,출력`, 환경 변수 `NFH_INPUT_WORKER_THREADS` · `NFH_OUTPUT_WORKER_THREADS`) | 입력을 NPU 배치로, 출력을 호스트 배치로 바꾸는 스레드 수 | ARM 1, 2 | 기본 |
| `NFH_ASYNC`(설정 항목 11) | 형식 바꾸기를 따로 스레드에서 | 켬 | 켬 |
| `NFH_ACCELERATION` · `CPU_OP_ACCELERATION` | 형식 바꾸기 NEON 가속, CPU 작업 XNNPACK | 빌드 옵션 | 보드 라이브러리에 빌드되어 있지 않음(`dxrun --accel-nfh` 없음) |
| `DYNAMIC_CPU_THREAD`(환경 변수 `DXRT_DYNAMIC_CPU_THREAD=ON`) | CPU 작업이 밀릴 때 스레드를 늘림 | 끔 | 끔(CPU 작업 없음) |
| `CUSTOM_INTRA_OP_THREADS` · `CUSTOM_INTER_OP_THREADS` | DXRT 안 ONNX Runtime 스레드 수 | 1 · 1 | 쓰지 않음(CPU 작업 없음) |
| 프로파일러(`--npu-profile`, 항목 `PROFILER` + `PROFILER_SAVE_DATA`) | 작업마다 시간을 재 끝날 때 작업 폴더에 `profiler.json` | 끔 | 끔(잴 때만) |
| `SHOW_PROFILE` · `SHOW_MODEL_INFO` · `TASK_FLOW` · `SHOW_THROTTLING` · `DEBUG` | 표준 출력에 보이기 | 끔 | 끔 |
| `SERVICE`(환경 변수 `DXRT_USE_SERVICE`) | `dxrtd` 서비스를 거쳐 여러 프로세스가 NPU 를 나눠 씀 | 빌드값(켬, 보드에 `dxrtd` 가 돎) | 기본 |
| 비동기 실행 | `dxrt_engine_run_async` · `dxrt_engine_wait`(작업 번호), 콜백 `dxrt_engine_register_callback` | — | 분할 `--npu-jobs` 와 사진 벡터가 올리기 · 받기를 나눠 씀 |
| 사용자 출력 버퍼 | `run` · `run_async` 의 출력 포인터에 바로 받기 | — | 분할 · 사진 벡터 모두 받을 쪽 버퍼에 바로(`run_into` · `wait_into`) |
| 클럭 · 전압 · 온도 | `dxrt_device_get_npu_clock` · `_voltage` · `_temperature`(읽기만). 바꾸기는 펌웨어 설정 JSON(`dxrt-cli -C`)뿐 | 1000 MHz · 750 mV | 그대로(보드 온도 기록 `~/temp.log` 의 npu 값) |
| 시간 재기 | `dxrt_engine_get_npu_inference_time` · `_latency_mean` 등, `dxrt_profiler_get_job_metrics` | — | 쓰지 않음(프로그램이 단계 시간을 따로 남김) |
