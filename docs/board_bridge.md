# 보드에 연결 프로그램 올리기 (Pi 5 · Debian 13 trixie arm64)

연결 프로그램(`bridge/`, [bridge.md](bridge.md))에 필요한 것: RTAB-Map 0.24.2 라이브러리, librealsense2(D435), OpenCV, CMake. memmap 은 지금처럼 `cargo build --release --no-default-features --features dxrt`.

## 패키지로 받을 수 있는 것 (2026-11-05 packages.debian.org 확인)

| 필요한 것 | trixie arm64 패키지 | 판 |
|---|---|---|
| RTAB-Map | 없음(모든 판 검색 결과 없음) → 소스 빌드 | — |
| librealsense2 | 없음(Debian 검색 결과 없음). RealSense 공식 apt 저장소는 `librealsense.intel.com/Debian/apt-repo` 가 403 → 소스 빌드 | — |
| OpenCV | `libopencv-dev` | 4.10.0 |
| PCL(RTAB-Map 필수) | `libpcl-dev` | 1.15.0 |
| g2o(그래프 최적화) | `libg2o-dev` | 0~20230806 |
| GTSAM | `libgtsam-dev` | 4.2.0, CMake 설정이 패키지에 없는 `libCppUnitLite.a` 를 가리켜 RTAB-Map 설정 단계가 멈춤(같은 패키지인 Ubuntu 24.04 에서 재현) → `-DWITH_GTSAM=OFF` |
| OpenCV xfeatures2d(BRIEF 기술자) | 없음(`xfeatures2d.hpp` 를 담은 패키지 없음) | — |

### 빌드마다 달라지는 RTAB-Map 기본값

RTAB-Map 은 빌드에 들어간 라이브러리에 따라 기본값을 고른다. `docs/rtabmap_pi.ini` 는 이 둘을 적지 않아 기본값을 따른다.

| 파라미터 | Windows 0.24.2 실행 파일(조정 기록을 잰 것) | Debian·Ubuntu 패키지로 빌드 |
|---|---|---|
| `Vis/FeatureType`, `Kp/DetectorStrategy` | 6 = GFTT/BRIEF (xfeatures2d 있음) | 8 = GFTT/ORB |
| `Optimizer/Strategy` | 2 = GTSAM | 1 = g2o |

같은 PC·같은 설정(`rtabmap_pi.ini`)에서 이 둘만 바꿔 Windows 도구로 잰 키프레임 위치 오차(한 번씩):

| 조합 | 사무실 | fr1 책상 |
|---|---|---|
| BRIEF + GTSAM (Windows 기본) | 2.1 cm | 2.8 cm |
| BRIEF + g2o | — | 4.7 cm |
| ORB + GTSAM | 3.6 cm | 4.1 cm |
| ORB + g2o (Debian 빌드 기본) | 3.7 cm | 10.0 cm |

사무실은 특징 종류(BRIEF → ORB)가 차이를 만들고 최적화기는 거의 같다. fr1 책상은 같은 조합도 실행마다 4–10 cm 로 흔들린다(연결 프로그램으로 두 번: 4.6 · 10.4 cm). 보드는 WSL 빌드와 같은 ORB + g2o 가 되고, 연결 프로그램의 측정값은 [bridge.md](bridge.md) "검증". BRIEF 를 쓰려면 OpenCV 를 contrib(xfeatures2d) 와 함께 소스 빌드해 RTAB-Map 을 그 OpenCV 로 다시 빌드한다.

## 설치 순서

```bash
sudo apt install -y build-essential cmake ninja-build git pkg-config \
  libopencv-dev libpcl-dev libg2o-dev libsqlite3-dev libeigen3-dev libsuitesparse-dev \
  libusb-1.0-0-dev libudev-dev libssl-dev
```

### librealsense2 v2.59.1 (RSUSB 백엔드: 커널 패치 없이 libusb 로 카메라를 엶)

```bash
git clone --depth 1 --branch v2.59.1 https://github.com/realsenseai/librealsense.git ~/src/librealsense
cmake -S ~/src/librealsense -B ~/build/librealsense -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=$HOME/opt/librealsense -DFORCE_RSUSB_BACKEND=ON \
  -DBUILD_EXAMPLES=OFF -DBUILD_GRAPHICAL_EXAMPLES=OFF -DBUILD_TOOLS=OFF -DBUILD_UNIT_TESTS=OFF \
  -DBUILD_PYTHON_BINDINGS=OFF -DCHECK_FOR_UPDATES=OFF
ninja -C ~/build/librealsense -j 3 && ninja -C ~/build/librealsense install
# 일반 사용자로 카메라 열기
sudo cp ~/src/librealsense/config/99-realsense-libusb.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
```

- PC(WSL, 12 작업)에서 1 분 15 초, 컴파일 하나 최대 0.8 GB. 보드는 `-j 3`(8 GB 안).
- 카메라 확인용 도구가 필요하면 `-DBUILD_TOOLS=ON`(`rs-enumerate-devices`).
- D435 는 USB 3 포트(파란색)에. `rs-enumerate-devices` 의 `Usb Type Descriptor` 가 3.2 인지 확인.

### RTAB-Map 0.24.2 (화면 없이 라이브러리만)

```bash
git clone --depth 1 --branch 0.24.2 https://github.com/introlab/rtabmap.git ~/src/rtabmap
cmake -S ~/src/rtabmap -B ~/build/rtabmap -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=$HOME/opt/rtabmap -DWITH_QT=OFF -DBUILD_APP=OFF -DBUILD_TOOLS=OFF \
  -DBUILD_EXAMPLES=OFF -DWITH_PYTHON=OFF -DWITH_GTSAM=OFF
ninja -C ~/build/rtabmap -j 3 && ninja -C ~/build/rtabmap install
```

- `BUILD_TOOLS=OFF`: 도구 중 `rtabmap-euroc_dataset` 이 배포판 yaml-cpp 와 링크 오류. 연결 프로그램은 핵심 라이브러리만 쓴다.
- 설정 단계는 PATH 를 훑으니 길면 `PATH=/usr/bin:/bin` 으로 줄여 돌린다.

### 연결 프로그램

```bash
cd ~/om   # 저장소
cmake -S bridge -B ~/build/bridge -G Ninja -DCMAKE_PREFIX_PATH="$HOME/opt/rtabmap;$HOME/opt/librealsense"
ninja -C ~/build/bridge
# 설정 출력에 "RealSense 입력 켬 (librealsense2 2.59.1)" 이 보여야 카메라 입력이 들어감
```

## 보드에서 처음 할 확인

1. `~/build/bridge/rtab_bridge --realsense --out /dev/null --max-frames 300 --stats /tmp/rs.csv` — 첫 줄의 카메라 이름·펌웨어·보정값, 끝의 `읽기`(카메라 대기 + 정렬)·`위치추적` ms.
2. 위 보정값 줄의 `왜곡 모델`: D435 컬러는 왜곡 계수가 0 이라 왜곡 없는 모델로 넘긴다.
3. 코어 묶기 예(코어 4 개): 연결 프로그램 `taskset -c 0,1`, memmap `taskset -c 2,3`. 숫자는 [bridge.md](bridge.md) "시간".
