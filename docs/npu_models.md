# 보드 NPU 모델(.dxnn) 만들기

보드(Pi 5 + DX-M1, DXRT 3.4)는 파일 형식 6 ~ 8 을 연다. 컴파일러는 **DX-COM 2.4.1**(형식 8)을 쓴다(2.5.0 은 형식 9 라 열리지 않음). 데스크톱 WSL 의 `~/dx241` 가상 환경에 설치되어 있다.

모든 모델이 NPU 작업 하나로만 돈다(CPU 작업 없음, `parse_model -m <파일>` 로 확인). 모델 안에 CPU 작업이 남으면 DXRT 가 ONNX Runtime 으로 그 부분을 돌리며 CPU 를 쓴다.

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

## 루프 · 재위치: megaloc_168x224.dxnn · xfeat_480x640.dxnn

쓰임과 정확도는 [loop_closure_npu.md](loop_closure_npu.md). 내보내기 · 손질 · 보정은 저장소 밖 일회성 파이썬(데스크톱 WSL `/home/rea/port/lc`, `~/sfm` 가상 환경의 torch, CPU 만)이고, 컴파일은 `~/dx241/bin/python`(DX-COM 2.4.1 의 파이썬 API `dx_com.compile`, 보정 자료를 한 장씩 내주는 dataloader, `calibration_method="ema"`, `quant_diagnosis=True`).

보정 사진 320 장(평가 녹화 fr1 360 · rpy · desk · room 은 뺌): 우리 D435 녹화 walk1 80 · rec 20 · fr3 office(D435) 40 · digiteo2(D435) 30, TUM fr3 long_office 60 · fr1 desk2 50 · fr1 floor 40. 녹화마다 고르게 건너뛰며 고름.

### MegaLoc(gmberton/MegaLoc, MIT): 입력 조각 [1,192,588] → f [1,192,256] · s [1,192,64] · t [1,256]

가중치는 `torch.hub.load('gmberton/MegaLoc','get_trained_model')`(Hugging Face `gberton/MegaLoc`, 약 870 MB). 모델을 같은 계산의 손으로 짠 판으로 옮겨(원본과 기술자 코사인 1.000000000, 최대 차 4e-7) 그 판을 내보낸다.

1. 조각 합성곱(14×14, 보폭 14)을 행렬곱으로: 입력 = 224×168 사진의 조각 192 개(조각 안 순서 채널 · 세로 · 가로, 값 /255). ImageNet 평균 · 표준편차는 행렬곱 무게 · 편향에 접는다. 위치 임베딩은 168×224 로 미리 보간(원본 `interpolate_pos_encoding`, 쌍삼차 + 0.1 오프셋)해 상수, CLS 토큰과 더해 앞에 붙인다. LayerScale 은 proj · fc2 무게에 접는다.
2. SALAD 의 세 MLP(군집 특징 768→512→256, 점수 768→512→64, 토큰 768→512→256)까지 모델 안. Sinkhorn · 군집 가중합 · 정규화 · 마지막 선형(16,640 → 8,448)은 Rust(`place.rs`). 마지막 선형 무게는 `megaloc_head.bin`(MLH1).
3. 8 비트로 깨지는 까닭: 블록 8 의 MLP 가 토큰 1~2 개(첫 조각 토큰은 늘, 나머지 하나는 사진마다 다른 자리)의 채널 38 · 60 · 293 · 415 에 약 400 을 쓰고 잔차가 끝까지 ±450 범위가 된다(보통 값 ±2). 손질 없이 8 비트 흉내 코사인 −0.02.
4. 손질(실수 결과는 원본과 코사인 0.998 – 0.9996):
   - 첫 조각 토큰 고정: 블록 8 주의 뒤로 흐름에서 빼고, 블록 9 – 11 에서 다른 토큰이 보는 K · V 와 끝 값은 보정 자료 평균 상수(SigLIP2 와 같은 방법).
   - 블록 8 · 11 의 fc1 출력을 10 으로 자름(다른 자리의 큰 값 토큰이 생기지 않게).
   - 잔차를 평균 0 으로 유지(잔차에 쓰는 층의 출력 채널 평균을 뺌) → LayerNorm 의 배율 · 편향을 다음 선형 층에 접음 → 잔차 채널을 직교 행렬 Q(Q·1 = 1, 무작위 씨앗 0)로 돌림. 평균 0 이라 돌린 뒤에도 LayerNorm(배율 없음)이 그대로 맞는다.
   - GELU 나누기: `GELU(min(f,2)) · W + ReLU(f−2) · W`.
   - LayerNorm · GELU 는 `F.layer_norm`(배율 없음) · `F.gelu`(erf)를 opset 13 으로 내보낸 분해형으로 둔다. 손으로 쓴 RMSNorm 꼴은 DX-COM 이 알아보지 못해 블록 전체를 CPU 작업으로 남겼다.
5. 컴파일: 보정 자료 = 위 320 장의 조각 [320,192,588].

⟨ML_COMPILE⟩

### XFeat(verlab/accelerated_features, Apache-2.0): 입력 u8 [1,1,480,640] → feats [1,64,60,80] · kpts [1,65,60,80] · heat [1,1,60,80]

가중치 `xfeat.pt`(6 MB). 모델 = 원본 `XFeatModel.forward` 의 InstanceNorm 뒤 전부(원본 출력과 최대 차 9e-5).

1. BatchNorm(배율 없음)을 합성곱에 접고, 키포인트 머리의 8×8 펼치기(unfold) + 1×1 합성곱을 8×8 보폭 8 합성곱 하나로 바꾼다. 쌍선형 올리기는 배율 2 · 4 고정 Resize.
2. 입력: 합성곱이 첫 연산인 입력을 DX-COM 은 u8 로 받는다(컴파일 로그 `NPU input : uint8`, 형식 FORMATTER). 그래서 CPU 가 정규화된 회색 x 를 `q = clamp(round(32·x + 112), 0, 255)` 로 넣고, 모델 첫머리에서 `(q − 112)/32` 로 되돌린다(정규화 범위 −3.5 ~ 4.47, 보정 사진에서 이 밖 화소 0.04 %). `.dxnn` 의 입력 기록은 `UINT8 [1,480,640,1]`, scale 없음.
3. 컴파일: 보정 자료 = 320 장의 q [320,1,480,640].

확인: DX-COM 진단 코사인 0.9916(QSNR 17.8 dB, 무게만 0.9970 · 활성값만 0.9938). NPU 작업 하나(CPU 작업 없음). 파일 1.0 MB.

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

메모리: 가중치는 첫 검색 때 읽고(검색하지 않는 뷰어는 0), 행렬은 f16 그대로 둔 채 곱할 때 64 열씩 f32 로 푼다(값이 같아 결과 비트 동일). 토큰 표(256,000 × 768, 393 MB)는 모델 파일에 둔 채 질의가 쓰는 행만 읽는다. 보드에서 인코더를 올린 프로세스 최대 RSS 0.40 GB(토크나이저 포함, 이전 f32 판은 가중치만 약 0.73 GB). 질의 104 개 결과는 바꾸기 전과 같다(ONNX 와 코사인 최소 0.9999989).

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
| 비동기 실행 | `dxrt_engine_run_async` · `dxrt_engine_wait`(작업 번호), 콜백 `dxrt_engine_register_callback` | — | 분할(`--npu-jobs`)은 읽기 스레드가 올리고 받기 스레드가 기다림, 사진 벡터는 `--embed-jobs` 장(기본 3)까지 함께 |
| 사용자 출력 버퍼 | `run` · `run_async` 의 출력 포인터에 바로 받기. `wait` 는 출력 포인터가 있으면 텐서마다 memcpy, NULL 이면 건너뜀(3.4.0 디스어셈블) | — | 올릴 때 받을 버퍼를 넘기고 `wait` 는 NULL(복사 없음) |
| 클럭 · 전압 · 온도 | `dxrt_device_get_npu_clock` · `_voltage` · `_temperature`(읽기만). 바꾸기는 펌웨어 설정 JSON(`dxrt-cli -C`)뿐 | 1000 MHz · 750 mV | 그대로(보드 온도 기록 `~/temp.log` 의 npu 값) |
| 시간 재기 | `dxrt_engine_get_npu_inference_time` · `_latency_mean` 등, `dxrt_profiler_get_job_metrics` | — | 쓰지 않음(프로그램이 단계 시간을 따로 남김) |
