# rsdriver — librealsense2 없이 D435 를 직접 읽는 Rust 드라이버

RealSense **D435**(USB 8086:0b07)를 리눅스 V4L2(uvcvideo)와 UVC 확장 단위(XU)로 직접 다룬다. librealsense v2.59.1(보드와 같은 판, Apache-2.0) 중 우리가 쓰는 경로(`slam/src/io/realsense.rs`, `bridge/src/realsense_source.cpp`)만 같은 계산으로 옮겼다. 정렬은 `tools::fast_align`(rs2::align 과 비트 단위로 같음)을 쓴다. 리눅스 전용(x86_64·aarch64), 의존성은 `libc` 와 `tools`.

## 하는 일

| 단계 | librealsense 에서 옮긴 곳 | rsdriver |
|---|---|---|
| 장치 찾기 | `backend-v4l2.cpp` | `/sys/class/video4linux/video*` → USB 장치(idVendor·idProduct)별로 묶음. 인터페이스 0 = 깊이, 3 = 컬러, 같은 인터페이스의 META 노드 = 메타데이터 |
| 시리얼·펌웨어 | `d400-device.cpp` `get_fw_details` | HWM `GVD`: 시리얼 = 48 부터 6 바이트 16진, 펌웨어 = 12 부터 4 바이트 거꾸로. `Config.serial` 은 이 시리얼(또는 USB 시리얼)과 맞춤 |
| HW 모니터 | `hw-monitor.cpp`, `uvc-option.cpp` | 깊이 XU(unit 3) 제어 1 에 1024 바이트 SET_CUR → GET_CUR, 응답 길이 = 1020 번째 u32 + 4 |
| 깊이 내부 보정 | `d400-private.cpp` | 펌웨어 5.11.9.5 이상이면 `RECPARAMSGET`(0x7E) 새 표에서 해상도를 찾고, 없으면 계수표(`GETINTCAL` 25)의 rect_params. 왜곡 Brown-Conrady, 계수 0 |
| 컬러 내부 보정 | 〃 `get_d400_color_stream_intrinsic` | RGB 표(`GETINTCAL` 32)의 정규화 행렬을 보정 해상도 화면비로 맞춤, 왜곡 Inverse Brown-Conrady + 표의 계수 |
| 깊이→컬러 | 〃 `get_d400_color_stream_extrinsic`, `pose.h` | `rotation_matrix_rect`·`translation_rect × −0.001` = 컬러→깊이, 그 역(Rᵀ, Rᵀ·(−t)) |
| 깊이 단위 | `ds-options.cpp` `depth_scale_option` | 고급 모드(`UAMG`)이고 펌웨어 5.6.3.0 이상이면 `GET_ADV`(depth table) 의 µm 값 → `(float)(1e-6 × (float)µm)`, 아니면 0.001 |
| 스트림 | `backend-v4l2.cpp` | 깊이 `Z16`, 컬러 `YUYV`, `VIDIOC_S_PARM` 1/fps, mmap 버퍼 4 개(메타 8 개), 메타 노드 `UVCH` |
| 제어 | `backend-v4l2.cpp` `set_pu`, `ds-options.cpp` | 컬러 노출 우선 끔(`EXPOSURE_AUTO_PRIORITY` 0), 노출 고정이면 `EXPOSURE_AUTO`=수동(1)·`EXPOSURE_ABSOLUTE`=ms×10·`GAIN`, 아니면 `EXPOSURE_AUTO`=조리개 우선(3). 이미터 = 깊이 XU 제어 2 에 1 바이트 |
| 메타데이터 | `metadata.h`, `d400-color.cpp` | UVC 머리 PTS(µs) = 장면 시각, `md_capture_timing`(장면 번호·센서 시각), `md_rgb_control`(실제 노출·감도) |
| 짝 맞추기 | `sync.cpp` `timestamp_composite_matcher` | 장치 시각 차 < (1000/fps)/2 ms 면 한 묶음, 아니면 이른 쪽을 버림 |
| YUYV → RGB8 | `color-formats-converter.cpp`, `neon/image-neon.cpp` | 빌드 대상의 librealsense 규칙: x86 = SSE 규칙, aarch64 = NEON 규칙(i16 감김 포함) |
| 정렬 | `proc/align.cpp` | `tools::fast_align::DepthAligner` |
| 처음 장면 버림 | `realsense.rs` | `Config.warmup`(기본 30 묶음) |

복사는 한 번: 장면 버퍼는 mmap 한 V4L2 버퍼 그대로 짝이 맞을 때까지 들고 있고, 컬러는 그 버퍼에서 바로 RGB8 로 바꿔 호출자 버퍼에 쓴다. 깊이는 줄 간격이 빽빽하면 mmap 버퍼를 그대로 정렬 입력으로 쓴다.

## API

```rust
use rsdriver::camera::{Camera, Config, Output};
let mut cam = Camera::open(Config { width: 640, height: 480, fps: 30, serial: String::new(),
                                    exposure_ms: 0.0, gain: -1, ..Default::default() })?;
let k = cam.calib.color;          // fx fy ppx ppy (Intrinsics = rs2_intrinsics 배치)
let per_m = cam.depth_per_m();    // 1 / 깊이 단위
let mut out = Output::default();  // 장면마다 다시 씀
let info = cam.next_into(&mut out)?;  // out.rgb(w*h*3 RGB8), out.aligned(w*h, 컬러 격자 깊이, 카메라 단위)
// info.t_ms(시스템 시계 ms), info.color_hw_ms, info.wait_ms, info.align_ms, info.color_meta.actual_exposure·gain
println!("{}", cam.describe());
```

`slam` 의 `RealSenseSource` 가 쓰는 값과의 대응: `rs2_get_frame_timestamp`×1e-3 → `info.t_ms × 1e-3`, 컬러 보정 → `cam.calib.color`, `depth_per_m` → `cam.depth_per_m()`, 정렬 깊이 → `out.aligned`, 설명 줄 → `cam.describe()`(실제 노출·감도는 `info.color_meta`).

시각: `color_hw_ms` 는 librealsense 하드웨어 시각(UVC PTS µs × 0.001, 32 비트 감김을 풂)과 같은 값이다. `t_ms` 는 첫 묶음에서 시스템 시계와의 차를 한 번 정해 더한 값이다(librealsense 전역 시각은 차이를 계속 선형 회귀로 고친다).

## 도구

- `rsd_info [--size 640x480] [--serial S] [--dump 폴더]` — 보정값·깊이 단위를 JSON 으로(f32 는 16진 비트도). `--dump` 는 HWM 원래 바이트(`gvd`·`coeff`·`rgb`·`new`·`depth_table` .bin)를 저장. 카메라가 없으면 끝 코드 2.
- `rsd_rec <폴더> [장면=60] [--size WxH] [--fps 30] [--exposure ms] [--gain g] [--emitter on|off] [--skip 30]` — 정렬 확인 도구(`tools` 의 `rs_align_check`)가 읽는 폴더 꼴(`rgb/` `depth/` `depth_raw/` `depth_rs/` + 목록, `calib.json`)에 원본 `yuyv/*.bin` 과 `frames.tsv`(장치 시각·순번·장면 번호·노출·감도·YUYV/Z16/RGB/정렬 FNV 해시·대기·정렬 시간)를 더 쓴다. `depth_rs/` 에는 rsdriver 정렬 결과가 들어가 `rs_align_check` 를 그대로 돌릴 수 있다.

## 같은 결과 확인 (PC, 카메라 없이)

`cargo test`(`tests/librealsense_same.rs`): 기대값은 WSL 의 librealsense 2.59.1 라이브러리 함수를 직접 불러 만든 것이다(`get_d400_intrinsic_by_resolution`, `try_get_d400_intrinsic_by_resolution_new`, `get_d400_color_stream_intrinsic`, `get_d400_color_stream_extrinsic` + `from_pose`·`inverse`, `calc_crc32`, `unpack_yuy2(RGB8)`; NEON 은 `image-neon.cpp` 본문을 aarch64 로 빌드해 qemu 로).

| 비교 | 결과 |
|---|---|
| 보정표 60 벌(계수표·RGB 표·새 표 있음/없음, 기본값 가지, 화면비 두 경우, 보정 해상도 0) × 14 해상도(848×100·256×144·목록 밖 1024×768 포함): 깊이·컬러 내부 보정, 컬러→깊이, 깊이→컬러 | **비트 단위 같음**(86,468 바이트, x86 와 aarch64 qemu). 저장소에는 6 벌(`tests/data`, 16 KB) |
| 깊이 단위 11 값, CRC32 6 버퍼 | 같음 |
| YUYV → RGB8, 모든 (Y, U, V) 1,677 만 조합 | SSE 규칙 = x86 `unpack_yuy2` 와 같음, NEON 규칙 = `image-neon.cpp` aarch64 와 같음. 두 라이브러리 빌드끼리는 36.5 만 바이트 다름(U 끝값에서 NEON i16 감김), 일반 규칙은 4,362 만 바이트 다름 |
| ioctl 번호·제어 번호·`v4l2_buffer` 배치 | 리눅스 헤더와 같음(x86_64·aarch64) |
| HWM 명령 바이트, 메타데이터 해석, 짝 규칙 | 단위 시험 |

정렬 결과는 같은 보정값 비트를 `tools::fast_align` 에 넣으므로 `tools` 의 확인(실제 D435 깊이 80 장에서 rs2::align 과 다른 화소 0)을 그대로 따른다.

빌드: WSL 에서 `CARGO_TARGET_DIR=~/build/rsdriver cargo test --release`, aarch64 는 `--target aarch64-unknown-linux-gnu`(링커 `aarch64-linux-gnu-gcc`, 실행 `qemu-aarch64 -L /usr/aarch64-linux-gnu`).

## 보드 시험 순서 (Pi 5, D435 를 꽂고)

카메라 흐름은 한 프로그램만 열 수 있어 librealsense 와 rsdriver 를 차례로 돌려 견준다. 저장소 `~/om` 기준. librealsense 쪽은 `ondevice` 를 `--features slam-realsense` 로 빌드해 `--driver librealsense` 로 고르고, `rs_align_check` 는 `tools` 를 `--features realsense` 로 빌드한 것.

1. 노드 확인: `v4l2-ctl --list-devices`, `ls /sys/class/video4linux/*/device -l` — D435 인터페이스 0·3 에 영상 노드와 메타 노드가 하나씩. 메타 노드가 없으면 커널 uvcvideo 메타데이터 지원부터 본다(그때 시각은 V4L2 버퍼 시각으로 대신함).
2. 보정값: `rsd_info --dump calib_raw > rsd.json` 과 `ondevice slam --realsense --driver librealsense --out /dev/null --max-frames 30` 첫 줄의 보정값 — fx·fy·cx·cy·깊이 단위가 같아야 한다(`rsd.json` 의 `bits` 를 f32 로 바꿔 비교). 해상도마다 `--size 848x480` 도. `calib_raw/*.bin` 은 PC 로 가져와 `tests/data` 에 실제 D435 표로 추가한다(`depth_intrinsics`·`color_intrinsics` 기대값은 같은 파일로 PC 의 librealsense 를 불러 만듦).
3. 정렬·깊이: `rsd_rec /tmp/rsd 90` → `rs_align_check /tmp/rsd` — rsdriver 의 원래 깊이와 보정값으로 실제 rs2::align 을 다시 돌린 결과와 `depth_rs/`(rsdriver 정렬)의 다른 화소가 0 이어야 한다.
4. 컬러 변환: 보드 librealsense 가 NEON 빌드인지 확인(librealsense 판 실행 로그의 "Using NEON-optimized align implementation" 줄). 보드에서 `cargo test --release`(NEON 규칙 해시 시험)가 통과하면 같은 YUYV 에서 같은 RGB.
5. 시각·짝: `frames.tsv` 에서 `color_ts_us − depth_ts_us` 가 ±16.7 ms 안, 장면 번호가 1 씩 늘고 빠진 곳 수가 librealsense 판(`--driver librealsense --stats`)과 비슷한지. 30 장/초 유지(`wait_ms` 중앙 ≈ 33).
6. 제어: `rsd_rec /tmp/e 60 --exposure 10 --gain 64` → `frames.tsv` 노출 100(0.1 ms 단위)·감도 64, 같은 설정으로 `ondevice slam --realsense --driver librealsense --exposure 10 --gain 64` 첫 줄의 "컬러 노출 10 ms 감도 64(고정)" 과 같음. 돌리는 동안 다른 창에서 `v4l2-ctl -d <컬러 노드> -C auto_exposure,exposure_time_absolute,gain,exposure_dynamic_framerate` 로 1·100·64·0 확인. `--emitter off` 로 녹화하면 `depth_raw/` 의 IR 점무늬 채움이 줄어든다.
7. 시간: `frames.tsv` 의 `align_ms`·`wait_ms` 를 `ondevice slam --realsense --driver librealsense --stats` 의 `wait`·`align` 과 견준다.
