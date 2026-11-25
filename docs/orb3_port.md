# ORB-SLAM3 → Rust · NPU 옮기기

(이 브랜치 `opt/orb3-rs-npu` 는 "포터 NPU" 절만 쓴다. 나머지 절은 `opt/orb3-rust` 판이 본문.)

## 포터 NPU

ORB 추출 중 NPU 로 옮길 수 있는 부분을 찾고, 원본과 결과를 비교한다. 시험 도구는 `tools/orbnpu/`(파이썬 · C++ 시험 도구), 출력은 WSL `~/orbnpu_out`.

### 도구

- `orbnpu_bench.cc`(build.sh): 원본 `ORBextractor.cc` 를 그대로 끌어와 `time`(단계별 시간) · `dump`(원본 점 · 기술자) · `mosaic`(8 층 피라미드를 1280×1120 한 장 u8 로 붙임 + layout.txt) · `maps`(NPU/ONNX 가 낸 지도로 나머지 CPU 단계를 돌려 같은 형식으로 덤프). `maps ... cpuS` 는 S 만 CPU 로 계산(지도 경로 자체 검증).
- `orb_maps.py`: 모자이크 → FAST 점수 S(9/16 연속 호 최소 · 최대를 합성곱 · ReLU 로만, 값 [0,127]) · 7×7 가우시안 블러 두 지도. `export`(ONNX opset 13) · `run`(실수) · `run --fq`(8 비트 흉내: 텐서별 u8 비대칭, 무게 채널별 int8).
- `compile_dx.py` · `cc.sh`: DX-COM 2.4.1 파이썬 API(`~/dx241/bin/python`, CPU 보정 ema 64 장)로 `.dxnn`.
- `cmp_kp.py`: 덤프 둘 비교(같은 자리 점 비율 · 방향 차 · 응답 차 · 기술자 해밍).
- `orb3_replay.cc`: `LD_PRELOAD` 로 `ORBextractor::operator()` 를 덤프 재생으로 바꿔 원본 rgbd_tum 궤적을 그대로 비교.

### ① CPU 시간 비중 (PC WSL, fr1_desk 300 장, 1 스레드 원본, 점 평균 1004)

| 단계 | ms/장 | 비중 | NPU 후보 |
|---|---|---|---|
| 피라미드(resize) | 0.53 | 9 % | 가능(쉬움)하나 모자이크 입력을 CPU 가 만들어야 해 이득 작음 |
| FAST(칸별, 두 문턱) | 2.77 | 45 % | **S 지도** — 문턱 비교 · 비최대 억제 · 점 수집은 CPU 에 남음 |
| 사분 트리 분배 | 0.39 | 6 % | 불가(분기 · 정렬) |
| 방향(IC_Angle) | 0.40 | 6 % | 모멘트 지도 시험: 8 비트 오차 커 기각(아래) |
| 블러 7×7 | 1.28 | 21 % | **blur 지도** |
| 기술자 rBRIEF | 0.73 | 12 % | 점마다 회전된 무늬 → CPU |
| 합 | 6.12 | | NPU 로 갈 수 있는 몫 최대 ≈ 66 % |

보드(Cortex-A76) 비중은 보드 확인 필요. NPU 를 쓰면 CPU 에는 "지도 읽기 + 문턱/억제" 가 새로 생기므로 실제 절약은 보드 실측 필요.

### ② · ④ PC 비교 (fr1_desk 앞 10 · 20 장, 원본 덤프 기준)

| 판 | 같은 자리 점 | 방향 | 기술자 해밍 평균 | 기술자 완전 같음 |
|---|---|---|---|---|
| S 는 CPU, 블러 실수(f32_20, 20 장) | 100 % | 같음 | 0.22/256 | 81.1 % |
| S 만 CPU 지도 경로(cpuS20) | 100 % | 같음 | 0 | 100 % (경로 비트 동일) |
| ONNX 실수 S · blur(f32, 10 장) | 97.5 % | 같음 | 0.32 | 74.4 % |
| 8 비트 흉내(fq, 10 장) | 87.5 % | 같음 | 2.01 | 27.3 % |
| 방향 모멘트까지 지도(mix, 이전 모델) | 100 % | 평균 2.4° 차 · 95 % 7.2° | 8.51 | 16.8 % |

- 실수 S 가 97.5 % 인 것은 ONNX 실수 합성곱이 정수 차이를 아주 약간 흔들어 FAST 점수 동점이 갈리는 곳(비최대 억제 · 사분 트리 응답 순위)에서 다른 점을 고르기 때문.
- 8 비트에서 S 가 크게 흔들림(87.5 %): S 는 정수 0..127 이라 u8 표현은 충분하지만 중간 텐서(차이 ±127, ReLU 합) 범위가 자료별 보정이라 1 단위 반올림이 섞임. 다음: 모든 중간 텐서의 scale 을 1 로 고정(정수 그대로)하는 보정 자료 · 모델 손질이 필요 — DX-COM 이 텐서 scale 을 고정할 수 있는지가 관건.
- 방향을 NPU 로 옮기면 기술자가 크게 바뀌어(해밍 8.5) 기각, 방향은 CPU 에 남긴다.
- 최종 궤적(`orb3_replay.cc`)은 전체 573 장 지도 덤프가 필요해 아직 못 돌림(남은 일).

### ③ DX-Compiler

- `orb_maps.onnx`(입력 [1,1,1120,1280] f32, 출력 S · blur) 를 DX-COM 2.4.1 로 컴파일: 97 층, 데스크톱 CPU 보정에서 **40 분 넘게** 걸림(첫 시도는 진단 옵션 포함 · 호출 시간 제한으로 끊김). 결과는 아래 "컴파일 결과".
- 보드에서 돌린 결과는 보드 확인 필요.

### ⑤ Rust 호출 자리(설계)

- 실행기: `slam/src/nnrun.rs` 의 `Net` 그대로 — `.dxnn` 이면 DXRT(`common::dxrt::Engine`, 비동기 `submit`/`wait`), PC 는 ONNX Runtime. 입력은 모자이크 u8(`wants_u8`), 출력 `out(buf, "S")` · `out(buf, "blur")`.
- `orb3rs::extract`(opt/orb3-rust 몫)에 맞물릴 자리: `Extractor::extract` 안에서 `compute_pyramid` 뒤 →
  1. 층들을 layout(orbnpu_bench mosaic 과 같은 배치)으로 모자이크에 복사 → `Net::submit`.
  2. NPU 가 도는 동안 CPU 는 할 일 없음(같은 장면 안 의존) — 이득은 이전 장면의 나머지 단계와 겹치는 파이프라인(장면 k 지도 NPU ‖ 장면 k−1 분배 · 기술자 CPU)에서 나옴.
  3. `level_keys` 의 `fast(...)` 를 `fast_from_score(S 층 조각, 문턱)`(점수 ≥ 문턱 · 3×3 비최대 억제, 원본 FAST 와 같은 순서로 수집)으로, `gaussian7` 을 blur 층 조각 복사로 바꾸는 선택 경로. 나머지(분배 · 방향 · rBRIEF)는 그대로.
- 개선은 기본값 규칙: 8 비트에서 원본과 같을 때만 기본으로 넣는다 → 아래 후보 거르개 방식으로 충족.

### 8 비트에서 원본과 비트 같게: NPU S 는 후보 거르개로만 (11-25)

- 8 비트 흉내 S 와 정확한 FAST 점수의 차(`err_s.py`, 3 장 모든 층): **최대 1, 늘 −1 쪽**(실수 ONNX S 는 차 0). 그래서 칸 문턱 th 마다 `S_npu ≥ th + 1 − margin`(margin 2) 인 화소만 CPU 가 정확한 점수를 계산(`orbnpu_bench maps ... lazy`). 후보 밖 화소는 정확한 점수 ≤ th 라 구석이 아니므로 결과가 원본과 같다.
- fr1_desk 10 장, 8 비트 흉내 지도: **같은 자리 점 100 % · 응답 · 방향 · 기술자 전부 같음**(블러 CPU, `lazy`). 블러까지 NPU(`lazyB`)면 기술자 해밍 평균 2.09/256 → 블러는 CPU.
- 정확 계산 화소는 전체의 1.78 %(최소 문턱 기준으로 전부 다듬으면 6.1 %).
- 실수 ONNX 판 97.5 % 의 원인: 모델이 S 를 127 에서 잘라 큰 응답끼리 동점 → 사분 트리 순위 변화. 후보 거르개에선 CPU 가 정확 점수를 다시 내므로 무관.
- C++ `orbnpu_bench` 지도 경로는 `Mat::at` 확인용이라 느림(4.3 ms/장). 속도는 Rust `fast_npu` 로 잰다.
- 보드 확인 필요: 실제 `.dxnn` S 오차가 margin 2 안인지.

### Rust 호출 코드 (`slam/src/orb3npu.rs`)

- `Mosaic::new/pack/level_scores`(모델 입력 1280×1120, orbnpu_bench 와 같은 배치), `fast_npu`(`orb3rs::extract::fast` 와 같은 출력), `MARGIN = 2`.
- PC 하네스 `tools/orbnpu/rs_harness.sh` + `harness_main.rs`: 저장소 밖 `~/orbnpu_rs` 에 opt/orb3-rust 의 extract.rs 를 읽기 전용으로 가져와 fast 자리만 갈고리로 바꾸고 이 브랜치 `nnrun.rs`(ONNX) · `orb3npu.rs` 로 돌려 원본 덤프와 비교. `NOISE=1` 은 S 를 화소마다 0/1 낮춤(8 비트 최악 흉내).

### 컴파일 결과

- 11-25: `~/orbnpu_out/dx_maps/orb_maps.dxnn` 생성(59.7 MB, 진단 없이 약 45 분, 오류 없음). 크기가 큰 것은 1280×1120 전 해상도 중간 텐서 32 채널이 많아서로 보임 — 보드 메모리 · 속도는 보드 확인 필요.
- PC 에는 NPU 가 없어 `.dxnn` 실행 결과 비교는 못 함. 8 비트 결과 추정은 fq 흉내 + 후보 거르개(위).
- 이 모델은 blur 출력도 내지만 블러는 CPU 로 하기로 해 쓰지 않는다. S 만 내는 판으로 다시 컴파일하면 더 작아질 것(약 45 분).

### 전체 573 장 · 궤적 (11-25)

- `full_fq.sh`: 8 비트 흉내 지도를 48 장씩 만들고 바로 지우며 덤프. **lazy(S 후보 거르개 · 블러 CPU): 점 575,781 개 전부 같은 자리, 기술자 100 % 같음** — 덤프 바이트 동일. lazyB(블러 NPU): 점 100 % · 기술자 해밍 평균 2.33/256(같음 25 %).
- `traj.sh`(LD_PRELOAD 재생, rgbd_tum nopango 판, fr1_desk ATE RMSE): 원본 덤프 1.76 · 1.72 · 1.72 cm, lazy 1.72 cm(같은 입력이라 차이는 ORB-SLAM3 스레드 흔들림), lazyB 1.73 · 1.70 cm — 블러를 NPU 로 해도 궤적은 흔들림 범위 안. 그래도 비트 같음 원칙대로 블러는 CPU.

### Rust 하네스 결과 (PC, 11-25)

- `run_rs.sh 20 orig20.bin`: Rust CPU fast · Rust ONNX(nnrun) S + `fast_npu` · 같은 + `NOISE=1` 모두 원본 덤프와 **바이트 동일**(20 장, 점 20,079 개).
- CPU 시간(100 장, 1 스레드, 부하 15 의 WSL): 추출 6.80 → 5.23 ms/장(FAST 칸 검사가 후보만 계산). 층 병렬: 2.3–2.5 → 1.9 ms/장. 더하는 CPU 일: 모자이크 복사 · S 꺼내기(f32 → u8).
- PC ONNX 실행은 770 ms/장(전 해상도 32 채널 중간 텐서 · CPU 실행) — 의미 있는 시간은 NPU 쪽: **보드 확인 필요**. NPU 시간이 장면 간격보다 짧고 비동기로 겹쳐야 이득(줄어드는 CPU 1.6 ms/장).
- 남은 일: 보드에서 `.dxnn` S 오차(≤ margin 2) · 시간 확인, 무거우면 모델 줄이기(S 를 칸 문턱 후보 여부 비트만 내게: 정밀도 요구가 더 낮음), opt/orb3-rust 통합 때 `level_keys` 의 `fast` 두 줄을 `fast_npu` 로.
