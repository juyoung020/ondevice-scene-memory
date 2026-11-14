# RealSense 카메라 입력과 깊이→컬러 정렬

카메라 입력부: RealSense **D435**(IMU 없는 모델, USB ID 8086:0b07) RGB-D → (깊이를 컬러 화면에 정렬) → `slam_bridge` 위치 → memmap. 이 문서는 카메라 설정, 정렬 방법과 측정, 카메라를 처음 꽂았을 때 확인 순서를 적는다. 코드는 `bridge/align/`(정렬 라이브러리 `fast_align`, RealSense 도구 `rs_info`·`rs_rec`·`rs_align_check`), librealsense 는 v2.59.1. 도구와 정렬은 D400 계열(D435·D455)에 같이 쓴다.

## D435 보정값과 깊이 범위

- **보드에서 읽은 값**(2026-11-06, `rtab_bridge --realsense` 첫 줄): 펌웨어 5.16.0.1, 컬러 640×480 fx 614.20 · fy 614.69 · cx 324.95 · cy 241.56, 컬러 왜곡 모델 Inverse Brown-Conrady, 깊이 단위 1 mm(1000/m). USB 3(5 Gbps)으로 붙었다.
- **깊이 기준 카메라** = 왼쪽 IR. 두 IR 사이 기준선 50 mm. 깊이 내부 보정값은 Brown-Conrady 이고 계수가 모두 0(왜곡 없음).
- **컬러 카메라**(OV2740, 롤링 셔터)는 깊이 기준 옆에 있다(`rs_info` 의 `depth_to_color`). D435 컬러 왜곡 계수는 0 이라 왜곡 모델 이름과 상관없이 왜곡 없는 투영과 같다(`rs_info` 의 `coeffs` 로 확인).
- **깊이와 컬러의 화소 밀도**: 640×480 에서 깊이 fx 는 약 383(시야 87°), 컬러 fx 는 614(시야 69°)라 **컬러가 각도당 화소가 더 많다** → 정렬은 깊이 화소마다 사각형을 채우는 경우(아래 "SIMD 규칙").
- **최소 거리(Min-Z)**: Intel 사양은 최대 해상도(1280×720)에서 약 28 cm, 권장 범위 0.3–3 m. Min-Z 는 깊이 초점거리(화소)에 비례한다(Min-Z ≈ fx × 기준선 / 126 화소): 640×480(fx≈383) 약 0.15 m, 848×480(fx≈425) 약 0.17 m 가 이론값이다. 책상 물체를 가까이(0.3–1 m) 찍기에 맞고, 3 m 넘는 벽은 깊이 잡음이 커서 memmap `--max-depth 3` 안팎으로 자른다. 실제 값은 벽에 다가가며 `rs_rec` 의 "가운데 깊이"가 사라지는 거리로 잰다(아래 확인 순서 4).
- IMU 가 없다. 구성은 RGB-D 만 쓴다.

## 권장 스트림 설정

| 항목 | 값 | 이유 |
|---|---|---|
| 컬러 | RGB8 640×480 @30 | memmap·RTAB-Map 입력 크기. RTAB-Map 보드 설정(`rtabmap_pi.ini`)이 30 장/초 기준 |
| 깊이 | Z16 640×480 @30 (848×480 도 시험) | 컬러와 같은 크기면 정렬이 가장 가볍다(PC 0.6 ms). 848×480 은 Min-Z 가 비슷하고 가로 시야가 넓지만 정렬이 1.6 배. 두 해상도의 시야각은 `rs_info --depth 848x480` 의 `fov_deg` 로 비교해 정한다 |
| 레이저(이미터) | 켬 | 무늬 없는 벽·책상 면에도 깊이가 나온다. 점무늬가 컬러에 찍히는지는 녹화 `rgb/` 로 확인(찍히면 RTAB-Map 특징점이 점무늬에 붙음) |
| 프리셋 | 기본(Default) | 채움과 정확도 균형. 경계 잡음이 크면 `--preset high_accuracy`(구멍이 늘고 틀린 깊이가 준다)와 비교 |
| 자동 노출 | 깊이·컬러 켬, 컬러 노출 우선(auto exposure priority) 끔 | 어두운 곳에서도 30 장/초 유지(노출 우선이 켜져 있으면 프레임 수가 떨어짐) |
| 시각 | 전역 시각(global time, 기본 켬) | 깊이·컬러·PC 시계를 같은 축으로 |
| 처음 30 장 | 버림 | 자동 노출이 자리 잡을 때까지 |

`rs_rec` 가 위 값을 그대로 건다(`--emitter`, `--preset`, `--depth`, `--color`, `--fps`, `--skip` 으로 바꿈).

## 정렬: rs2::align 이 실제로 하는 일

librealsense v2.59.1 의 `rs2::align(RS2_STREAM_COLOR)` 은 빌드에 따라 세 구현 중 하나를 쓴다(`src/proc/align.cpp` 의 `create_align`):

| 구현 | 쓰는 곳 | 규칙 |
|---|---|---|
| SSE(`src/proc/sse/sse-align.cpp`) | Windows SDK 설치본, x86 리눅스 | 아래 "SIMD 규칙" |
| NEON(`src/proc/neon/neon-align.cpp`) | aarch64 소스 빌드(`BUILD_WITH_NEON` 기본 켬, 2024-10 PR #13389 로 들어옴) | SIMD 규칙(SSE 를 옮긴 것, 곱·더하기를 fma 로) |
| 일반(`align_images` + rsutil) | 그 밖(NEON 없는 빌드, 옛 버전) | 깊이 화소의 두 모서리를 옮겨 그 사각형을 채움, 사각형이 화면을 벗어나면 통째로 버림, 컬러 왜곡 적용 |

**SIMD 규칙**: 깊이 화소마다 왼쪽 위 모서리 (x−0.5, y−0.5)를 깊이 단위로 3D 로 올리고 깊이→컬러 변환 후 컬러로 투영, (u+0.5) 를 버림한 화소에 원래 깊이 값을 쓴다. 겹치면 가까운(작은) 값. 깊이의 각도당 화소 수가 컬러보다 적을 때(또는 640×480→640×360)만 오른쪽 아래 모서리까지 옮겨 사각형을 채운다. D435 의 640×480 깊이 → 640×480 컬러는 컬러가 더 촘촘해 **사각형을 채운다**(D455 처럼 깊이가 더 촘촘하면 모서리 하나만). 컬러 왜곡은 Modified Brown-Conrady 일 때만 적용하므로 Inverse Brown-Conrady 왜곡은 무시된다(D435 는 계수 0 이라 영향 없음).

Pi 5 에서 "정렬만 수십 ms" 라는 보고(Pi 4 에서 3 장/초, `align.process` 0.1–0.2 초)는 일반 구현의 비용이다. 일반 구현을 옮겨 PC 코어 하나에서 재면 640×480 에 6.7 ms(실제 SDK 는 화소마다 rsutil API 함수를 6 번 불러 더 느림)이고, NEON 이 있는 v2.59.1 소스 빌드는 SIMD 규칙(PC 0.9 ms)을 쓴다. 보드에서 어느 구현인지는 도구들이 SDK 로그 한 줄("Using NEON-optimized align implementation")을 잡아 보여 준다.

## fast_align

`bridge/align/include/fast_align/fast_align.hpp` — `DepthAligner(깊이 보정, 컬러 보정, 깊이→컬러, 깊이 단위, Options)` 를 한 번 만들고 장면마다 `process(깊이 u16, 출력 u16)`. 보정값 구조체는 `rs2_intrinsics`·`rs2_extrinsics` 와 같은 배치라 그대로 복사해 넘긴다.

- **방식**: `Method::Sdk`(기본) = SIMD 규칙, `Method::Generic` = 일반 구현 규칙. 각 방식은 SDK 와 같은 순서의 float 연산을 쓴다(x86 은 SSE 의 곱·더하기 순서, aarch64 는 NEON 의 fma 순서, 나머지는 `-ffp-contract=off`·MSVC `/fp:precise`).
- **빠르게 한 것**: 모서리 역투영 방향을 열마다 x, 줄마다 y 로만 둔다(깊이 왜곡이 없을 때, SDK 는 화소마다 float 4 개 표 = 4.9 MB). 줄 하나씩 좌표 계산 → 바로 흩뿌리기(SDK 는 영상 전체 int2 배열을 쓰고 다시 읽음). 계산 루프는 분기 없는 식이라 자동 벡터화되고(MSVC·GCC x86, aarch64 NEON 확인), 모서리 하나 경우는 컬러 화소 번호와 화면 밖 판정까지 벡터 루프에서 낸다. 줄 단위 여러 스레드(`threads`): 깊이 줄 덩어리를 짝수·홀수 두 번에 나눠 돌리고, 동시에 돈 덩어리가 쓴 컬러 줄 범위가 겹치면 한 스레드로 다시 해서 결과가 늘 같다.
- **선택**: `exact = false` 는 나눗셈 하나(1/Z)로 계산해 반올림 경계의 화소 약 0.002 % 가 옆 화소로 옮겨진다. `color_distortion = true` 는 컬러 왜곡 계수가 있는 카메라(D455 등)에서 그 왜곡을 rsutil 식으로 적용해 깊이를 실제 컬러 화소 위치에 놓는다(SDK 와 결과가 달라짐). `force_rect` 는 늘 사각형을 채운다.

### 정확성

`bridge/align/tests/test_align.cpp`(옮겨 온 librealsense 코드 `tests/rs_ref.hpp`, Apache-2.0 출처 주석)와 `tools/align_sdk_check.cpp`(이 PC 의 실제 SDK):

| 비교 | 결과 |
|---|---|
| fast_align Sdk ↔ 옮겨 온 SSE 경로, 8 가지 보정(D455 가정 640/848 깊이, 1280×720 컬러, 왜곡 없음·MBC·깊이 왜곡·640×360·사각형 경우) × 방·무작위 깊이 × 스레드 1·3·4 | 다른 화소 **0** |
| fast_align Generic ↔ 옮겨 온 일반 경로, 같은 조합 | 다른 화소 **0** |
| aarch64(-mcpu=cortex-a76) 빌드를 qemu 로: fast_align ↔ 옮겨 온 NEON 경로 | 다른 화소 **0**(96 / 96 경우) |
| fast_align Sdk ↔ **실제 rs2::align**(Windows SDK 2.59.1, software_device, 640·848 깊이 각 120 장) | 다른 화소 **0 / 7,372.8 만** |
| 빠른 식(exact=false) ↔ rs2::align | 0.0021–0.0023 % 다름 |

합성 장면 정답(컬러 화소 중심이 보는 면의 깊이)과 비교, 컬러 왜곡 없음:

| 장면 | Sdk 안쪽 값 맞음 / 구멍 | Generic 안쪽 값 맞음 / 구멍 | 경계 화소 중 맞음(Sdk / Generic) |
|---|---|---|---|
| 정면 평면 2 m | 모든 값 2000 | 모든 값 2000 | – |
| 계단(1 m 판 / 2 m 벽) | 1.000 / 0 | 1.000 / 0 | 0.78 / 1.00 |
| 방(벽·바닥·상자) 640 깊이 | 0.997 / 0 | 0.997 / 0 | 0.72 / 0.78 |
| 방 848 깊이 | 0.997 / 0 | 0.997 / 0 | 0.63 / 0.79 |

뒤쪽 깊이가 앞을 덮은 화소는 모든 경우 0. SIMD 규칙은 왼쪽 위 모서리를 반올림해 경계가 반 화소쯤 밀린다(경계 화소 정답률이 Generic 보다 낮은 이유).

컬러 왜곡 계수가 있는 카메라(D455 공개 계수를 가정한 방): Sdk(= rs2::align) 안쪽 값 맞음 **0.894**, `color_distortion` 켠 Sdk 0.997, Generic 0.997. 컬러 왜곡을 무시하면 같은 광선이 평균 2.6 화소, 가장자리에서 4.9 화소 옮겨진다(640×480 화소의 78 % 에서 1 화소 넘게). 실제 계수로 다시 잰다(`rs_align_check` 의 "컬러 왜곡 적용" 줄).

### 시간

PC(Ryzen 9 9950X, 다른 작업이 CPU 를 약 55 % 쓰는 중) 코어 하나에 묶고 20 번 평균 × 21 회 중앙값, D455 가정 보정값의 방 장면(잡음·구멍 3 %). 전체 표는 `performance_log.md`.

| 깊이 → 컬러 640×480, ms/장면 | 640×480 깊이 | 848×480 깊이 |
|---|---|---|
| 실제 rs2::align.process(Windows SDK, SSE) | 0.92 | 1.25 |
| fast_align Sdk, 스레드 1 (MSVC) | 0.61–0.70 | 0.99–1.08 |
| fast_align Sdk, 스레드 4 | 0.27–0.28 | 0.38 |
| 옮겨 온 일반 경로(NEON 없는 빌드의 규칙) | 6.7 | 9.5 |

GCC(x86, WSL)로 빌드하면 한 스레드는 옮겨 온 SSE 경로보다 느리다(0.79 대 0.58 ms, SSE2 기본에서 32 비트 곱·고르기를 흉내 내는 비용). aarch64 의 계산 루프는 llvm-mca(Neoverse-N1 모형, A76 계열)로 SDK NEON 7.26 주기/화소, fast_align 7.52(빠른 식 6.64)로 거의 같다.

**Pi 5 환산**(가정: A76 2.4 GHz 코어 = 이 PC 코어의 2–3 배 시간, `rtabmap_tuning.md` 와 같은 가정): rs2::align NEON 경로 640×480 약 1.8–2.8 ms/장면(30 장/초에 코어 하나의 5–8 %), fast_align 한 스레드 1.2–2.1 ms, 스레드 4 개 0.6–0.9 ms(다른 일과 코어를 나누면 늘어남), NEON 없는 일반 구현 13–20 ms(실제 SDK 는 API 호출 비용이 더해져 수십 ms). 즉 v2.59.1 소스 빌드에서는 정렬이 병목이 아니고, fast_align 은 같은 결과를 프레임 할당 없이, 원하는 스레드에서, 필요하면 여러 코어로 낸다. 보드 실측으로 바꾼다(아래 "Pi 5 에서").

## 카메라 확인 순서(Windows PC)

빌드(한 번, 이미 해 둠):

```
cmake -S bridge/align -B bridge/align/build -G "Visual Studio 17 2022" -A x64 -DREALSENSE_ROOT=C:/librealsense2/v2.59.1
cmake --build bridge/align/build --config Release
```

카메라를 **USB3 포트**(파란 포트, `rs_info` 의 `"usb": "3.2"`)에 꽂고 `bridge/align/build/Release` 에서:

1. `rs_info > rs_info.json` — 장치·펌웨어·USB, 스트림 목록, IMU, 640×480 깊이·컬러 보정값과 외부 변환, `sdk_align_fills_rect`(false 면 모서리 하나 규칙). `rs_info --depth 848x480 > rs_info_848.json` 로 시야각 비교. 펌웨어가 `recommended_firmware` 보다 낮으면 RealSense Viewer 로 올린다.
2. `rs_rec C:/ondevice/data/d435/desk01 20` — 20 초 녹화(레이저 켬, 기본 프리셋). 1 초마다 rs2::align · fast_align 시간, 가운데 깊이, 유효 화소 비율을 찍고, 끝에 rs2::align 과 다른 화소 수와 memmap 용 변환 명령을 찍는다. 쓰기가 밀린 장면이 있으면 숫자로 보인다.
3. `rs_align_check C:/ondevice/data/d435/desk01` — 녹화된 원래 깊이로 fast_align(정확한 식·빠른 식·컬러 왜곡)과 rs2::align(software_device 로 다시)을 돌려 녹화된 rs2::align 결과와 일치 비율, 둘 다 값 있는데 다른 화소, 한쪽만 값 있는 화소, 경계 화소 차이, 코어 하나 시간을 낸다. 정확한 식은 다른 화소 0 이어야 한다(끝 코드 0).
4. 최소 거리: `rs_rec C:/ondevice/data/d435/minz 15` 를 켜고 벽을 정면으로 1 m 에서 천천히 다가가며 "가운데 깊이 … (유효 …%)" 가 0 으로 떨어지는 거리를 적는다. 레이저 끔(`--emitter off`)과 비교.
5. memmap 입력: 2 번 끝에 찍힌 `rgbd2tum <녹화> <녹화>_freiburg3 fx fy cx cy 1000` 으로 TUM fr3 형식(5000/m)으로 바꿔 `memmap --tum` 과 `rtabmap-rgbd_dataset` 에 넣는다.

녹화 폴더: `rgb/`(컬러), `depth/`(fast_align 정렬, 컬러 크기), `depth_raw/`(정렬 전), `depth_rs/`(rs2::align), 각 목록 `rgb.txt`·`depth.txt`·`depth_raw.txt`·`depth_rs.txt`, 짝 `association.txt`, `calib.json`(보정값·설정·SDK 정렬 구현·변환 명령), `rec_stats.json`(장면 수·시간·rs2::align 과 다른 화소). 깊이 PNG 는 카메라 단위 그대로(1 mm = 1): 단위 바꾸기와 fr3 보정값으로 다시 그리기는 `rgbd2tum` 한 곳에서 한다(memmap 의 `tum.rs` 는 폴더 이름으로 fr1/2/3 보정값을 고르고 5000/m 로 읽음).

## Pi 5 에서

- librealsense v2.59.1 소스 빌드(`-DFORCE_RSUSB_BACKEND=ON -DBUILD_WITH_NEON=ON -DCMAKE_BUILD_TYPE=Release`, NEON 은 aarch64 기본값) 뒤 `cmake -S bridge/align -B build -G Ninja && ninja -C build` 로 같은 도구가 빌드된다(`find_package(realsense2)`).
- 보드에서 `bench_align --core 3` 와 `rs_align_check <녹화> --core 3 --threads 3` 로 PC 환산 대신 실측을 남긴다. 도구가 찍는 SDK 정렬 구현 줄이 "NEON-optimized" 인지 확인한다.
- PC 에서 aarch64 확인: `bash bridge/align/tools/cross_check.sh`(WSL, 크로스 빌드·qemu 시험·NEON 명령 수), `bash bridge/align/tools/neon_mca.sh`(llvm-mca 주기).

## 참고

- librealsense v2.59.1: https://github.com/realsenseai/librealsense/tree/v2.59.1 (`src/proc/align.cpp`, `src/proc/sse/sse-align.cpp`, `src/proc/neon/neon-align.cpp`, `src/rs.cpp`, `CMake/unix_config.cmake`)
- NEON 정렬 PR #13389(2024-10): https://github.com/realsenseai/librealsense/pull/13389
- Pi 4 에서 align 3 장/초: https://github.com/IntelRealSense/librealsense/issues/12928 , `align.process` 0.1–0.2 초: https://github.com/IntelRealSense/librealsense/issues/11758
- D435 사양(Min-Z ~28 cm, 0.3–3 m): https://www.intelrealsense.com/depth-camera-d435/
- 컬러 왜곡 계수 예(D455): https://dev.classmethod.jp/articles/correct-distortion-realsense-d455
