# 보드에 올리기 (Pi 5 · Debian 13 trixie arm64)

보드에서 도는 것은 실행 파일 `ondevice` 하나다: 위치 프로그램(`slam/`, Rust, [bridge.md](bridge.md)) · 물체 기억 · 뷰어 서버가 한 프로세스에서 돈다. 저장소 맨 위 Cargo 작업 공간에서 빌드하며 결과는 `target/release/` 에 생긴다. D435 입력은 rsdriver(V4L2 · UVC 직접, Rust)다. 실행 파일이 링크하는 것은 DXRT 런타임(libdxrt)과 C 표준 라이브러리뿐이다.

```bash
# 보드 실행 파일(D435 입력은 rsdriver)
cargo build --release --no-default-features --features dxrt      # target/release/ondevice

# 실시간: D435 → 위치 추적 → 물체 기억(NPU 분할) → 뷰어(:8091), 한 프로세스
# 기본 모델: ~/models/objectsam-416_640.dxnn(분할) + siglip2-vision_npu.dxnn(이름) + vocab/indoor.voc, 검색 글 인코더 ~/models/siglip2(text_model_fp16.onnx 가중치를 Rust 로, scripts/board_run.sh 와 같음)
target/release/ondevice --slam "--realsense --config docs/rtabmap_pi.ini --send-every 2" \
    --every 1 --live 1 --max-depth 3.5 --out data/memory/live --serve "8091 --bind 0.0.0.0"
```

- NPU 는 한 번에 한 프로세스만: 새로 돌리기 전에 `pkill -f ondevice`(다른 NPU 프로그램도 멈춤).
- 코어 묶기: `--threads` 로 역할마다(예 `--threads auto` 기본, 위치 프로그램 풀을 따로 묶을 때 `slam=0:2-3`).

## PC 에서 교차 빌드해 보드로 보내기

보드에서 빌드(약 4 – 5 분) 대신 PC(우분투 · WSL)에서 aarch64 로 빌드해 `scp` 로 보낸다. 처음 한 번 보드의 `/usr/local/lib/libdxrt.so.3.4.0` 을 `~/board-sysroot/lib` 로 받아 링크에 쓴다(스크립트가 알아서 받음, 저장소 밖). libdxrt 가 부르는 onnxruntime 은 보드에서 실행할 때 찾는다. PC 교차 glibc 2.39 ≤ 보드 2.41.

```bash
sudo apt install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu mold   # mold · sccache 는 있으면 씀
rustup target add aarch64-unknown-linux-gnu
bash scripts/cross_board.sh                         # release(fat LTO) → kau@192.168.219.101:~/ondevice-cross
PROFILE=dev-fast bash scripts/cross_board.sh        # 빠른 확인용: LTO 끔 · 나눠 컴파일 · 증분
bash scripts/cross_board.sh kau@192.168.219.101 ondevice-cross -- --help   # 보내고 바로 실행
```

`dev-fast` 는 확인용이고, 측정 · 배포는 `release` 빌드로 한다.

## 보드에서 처음 할 확인

1. `ondevice slam --realsense --out /dev/null --max-frames 300 --stats /tmp/rs.csv` — 첫 줄의 카메라 이름·펌웨어·보정값, 끝의 `읽기`(카메라 대기 + 정렬)·`위치추적` ms.
2. 위 보정값 줄의 `왜곡 모델`: D435 컬러는 왜곡 계수가 0 이라 왜곡 없는 모델로 넘긴다.
3. 코어 묶기 예(코어 4 개): `--threads` 기본(auto)은 본 스레드·읽기를 앞 코어 둘에, 점 · 다시 쌓기 · 기록을 나머지에 두고 위치 프로그램 풀은 코어 전부. 숫자는 [bridge.md](bridge.md) "시간".
4. 카메라 장치 권한: 일반 사용자가 `/dev/video*` 를 열 수 있게 `video` 묶음에 든다(`sudo usermod -aG video $USER`). D435 는 USB 3 포트(파란색)에 꽂는다.

## 실행 전 설정(scripts/board_tune.sh)

`sudo scripts/board_tune.sh --no-remote --user $USER`: CPU 주파수 관리자 performance, D435 USB 절전 끔, swappiness 10, 쓰지 않는 서비스(블루투스 · 프린터 · 모뎀 · avahi)와 원격 화면(rustdesk · wayvnc) 멈춤, 실시간 권한(다시 로그인 후), 인터럽트 USB(카메라) → 코어 0 · NPU(dx-dma) → 코어 1. 바꾸기 전 값은 `/run/ondevice-tune` 에 남는다.

- 되돌리기: `sudo scripts/board_tune.sh --revert`(기록한 값으로 되쓰고 멈춘 서비스를 다시 켬, 규칙 파일 지움) 또는 재부팅.
- 원격 화면이 필요하면 `--no-remote` 를 빼고 돌린다.

## RTAB-Map 설정 값

`docs/rtabmap_pi.ini` 는 위치 프로그램의 특징 · 최적화기를 적는다: `Vis/FeatureType=6`, `Kp/DetectorStrategy=6`(GFTT/BRIEF), `Optimizer/Strategy=2`(GTSAM). RTAB-Map 0.24.2 로 잰 키프레임 위치 오차(한 번씩, Windows):

| 조합 | 사무실 | fr1 책상 |
|---|---|---|
| BRIEF + GTSAM (ini) | 2.1 cm | 2.8 cm |
| BRIEF + g2o | — | 4.7 cm |
| ORB + GTSAM | 3.6 cm | 4.1 cm |
| ORB + g2o | 3.7 cm | 10.0 cm |

## 보드 실행 파일이 링크하는 것

`readelf -d ondevice | grep NEEDED`(2026-11-15): `libdxrt.so`(DEEPX NPU 런타임, C++) · `libc.so.6` · `libm.so.6` · `libgcc_s.so.1`(Rust 표준 라이브러리가 쓰는 C 런타임). 나머지(D435 rsdriver, 위치 프로그램, 분할 후처리, 글 인코더 · 토크나이저, 뷰어 서버)는 모두 Rust 다.

- `libdxrt.so` 는 NPU 를 쓰는 유일한 길이다(커널 드라이버 · `dxrtd` 서비스와 이야기함). 이 라이브러리가 자기 안에서 `libonnxruntime.so` · `libstdc++` 를 불러오지만, 우리 모델은 NPU 작업만 있어 그 ONNX Runtime 은 모델을 돌리지 않는다(`use_ort` 끔).
- 스크립트(`scripts/*.sh`)는 bash 로 빌드 · 실행 인자만 이어 준다. DX-COM(모델 컴파일러, 파이썬)은 PC 에서 `.dxnn` 을 만들 때만 쓴다.
