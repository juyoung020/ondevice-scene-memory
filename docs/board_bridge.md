# 보드에 올리기 (Pi 5 · Debian 13 trixie arm64)

보드에서 도는 것은 실행 파일 `ondevice` 하나다: 위치 프로그램(`slam/`, Rust, [bridge.md](bridge.md)) · 물체 기억 · 뷰어 서버가 한 프로세스에서 돈다. 저장소 맨 위 Cargo 작업 공간에서 빌드하며 결과는 `target/release/` 에 생긴다. D435 입력은 rsdriver(V4L2 직접, Rust 만)가 기본이고, librealsense2 판(`slam-realsense`)은 견줄 때 넣는다.

```bash
# 보드 실행 파일(D435 입력은 rsdriver)
cargo build --release --no-default-features --features dxrt      # target/release/ondevice
# librealsense2 판 입력도 넣을 때(~/opt/librealsense, 아래 "librealsense2 v2.59.1" 순서로 설치)
RUSTFLAGS="-L $HOME/opt/librealsense/lib -C link-arg=-Wl,-rpath,$HOME/opt/librealsense/lib" \
  cargo build --release --no-default-features --features dxrt,slam-realsense

# 실시간: D435 → 위치 추적 → 물체 기억(NPU 분할) → 뷰어(:8091), 한 프로세스
M=~/dx-all-suite_my/workspace/res/models
target/release/ondevice --slam "--realsense --config docs/rtabmap_pi.ini --send-every 2 --reset-file data/memory/live/reset.req" \
    --every 1 --live 1 --seg $M/yolo26-s-seg_640x640.dxnn --max-depth 3.5 --out data/memory/live --serve "8091 --bind 0.0.0.0"
```

- NPU 는 한 번에 한 프로세스만: 새로 돌리기 전에 `pkill -f ondevice`(다른 NPU 프로그램도 멈춤).
- 코어 묶기: `--threads` 로 역할마다(예 `--threads auto` 기본, 위치 프로그램 풀을 따로 묶을 때 `slam=0:2-3`).

## 위치 프로그램을 C++ 원본과 견주기(보드 ARM)

위치 프로그램은 PC(x86)에서 C++ 원본 `rtab_bridge` 와 스트림 · 위치 파일 · 판단 바이트가 같다(시험 입력 18 개 × 설정 2 개). 보드 원본은 OpenCV · Eigen 이 ARM NEON 곱셈-덧셈 합침을 써서, 같은 비교를 보드에서 따로 한다(남은 일). 견주는 법: 같은 녹화를 `rtab_bridge --tum <폴더> --rate 0 --block --config docs/rtabmap_pi.ini --out a.mmf --poses a.txt` 와 `ondevice slam` (같은 인자, `--out b.mmf --poses b.txt`)로 돌려 두 파일의 md5 를 본다. `bridge/` 의 C++ 원본은 이 비교에만 쓰고, 아래는 그 빌드 순서다. 필요한 것: RTAB-Map 0.24.2 라이브러리(GTSAM 4.2, OpenCV 4.10.0 + contrib), librealsense2, CMake.

## 패키지로 받을 수 있는 것 (2026-11-05 packages.debian.org 확인)

| 필요한 것 | trixie arm64 패키지 | 판 |
|---|---|---|
| RTAB-Map | 없음(모든 판 검색 결과 없음) → 소스 빌드 | — |
| librealsense2 | 없음(Debian 검색 결과 없음). RealSense 공식 apt 저장소는 `librealsense.intel.com/Debian/apt-repo` 가 403 → 소스 빌드 | — |
| OpenCV | `libopencv-dev` 4.10.0(contrib 없음) → contrib 와 함께 소스 빌드 | — |
| PCL(RTAB-Map 필수) | `libpcl-dev` | 1.15.0 |
| g2o(그래프 최적화) | `libg2o-dev` | 0~20230806 |
| GTSAM | `libgtsam-dev` 4.2.0 은 CMake 설정이 패키지에 없는 `libCppUnitLite.a` 를 가리켜 RTAB-Map 설정 단계가 멈춤(같은 패키지인 Ubuntu 24.04 에서 재현) → 4.2 소스 빌드 | — |
| OpenCV xfeatures2d(BRIEF 기술자) | 없음(`xfeatures2d.hpp` 를 담은 패키지 없음) → OpenCV 4.10.0 + contrib 소스 빌드 | — |

### 특징·최적화기는 ini 에 적음

RTAB-Map 은 빌드에 들어간 라이브러리에 따라 기본값을 고른다(xfeatures2d 가 있으면 GFTT/BRIEF, 없으면 GFTT/ORB; GTSAM 이 있으면 GTSAM, 없으면 g2o). `docs/rtabmap_pi.ini` 는 `Vis/FeatureType=6`, `Kp/DetectorStrategy=6`(GFTT/BRIEF), `Optimizer/Strategy=2`(GTSAM) 을 적어 어느 빌드든 같은 조합을 쓰고, 보드는 아래 순서로 OpenCV contrib 와 GTSAM 을 넣어 빌드한다. 없는 조합을 고르면 RTAB-Map 이 경고를 내고 다른 것으로 바꾸니 실행 로그를 본다.

같은 PC·같은 설정에서 이 둘만 바꿔 Windows 도구로 잰 키프레임 위치 오차(한 번씩):

| 조합 | 사무실 | fr1 책상 |
|---|---|---|
| BRIEF + GTSAM (ini) | 2.1 cm | 2.8 cm |
| BRIEF + g2o | — | 4.7 cm |
| ORB + GTSAM | 3.6 cm | 4.1 cm |
| ORB + g2o (배포판 패키지만으로 빌드한 기본값) | 3.7 cm | 10.0 cm |

연결 프로그램(WSL, 세 번씩, 끝 그래프의 지도 노드 오차)은 [bridge.md](bridge.md) "특징·최적화기 비교".

## 설치 순서

```bash
sudo apt install -y build-essential cmake ninja-build git pkg-config \
  libpcl-dev libg2o-dev libsqlite3-dev libeigen3-dev libsuitesparse-dev \
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

### OpenCV 4.10.0 + contrib (xfeatures2d)

```bash
git clone --depth 1 --branch 4.10.0 https://github.com/opencv/opencv.git ~/src/opencv
git clone --depth 1 --branch 4.10.0 https://github.com/opencv/opencv_contrib.git ~/src/opencv_contrib
cmake -S ~/src/opencv -B ~/build/opencv -G Ninja -DCMAKE_BUILD_TYPE=Release   -DCMAKE_INSTALL_PREFIX=$HOME/opt/opencv -DOPENCV_EXTRA_MODULES_PATH=$HOME/src/opencv_contrib/modules   -DBUILD_LIST=core,imgproc,imgcodecs,features2d,flann,calib3d,xfeatures2d,video,videoio,highgui,photo,stitching   -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_DOCS=OFF -DBUILD_opencv_apps=OFF   -DBUILD_opencv_python3=OFF -DBUILD_JAVA=OFF -DWITH_QT=OFF -DWITH_GTK=OFF -DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF   -DWITH_OPENCL=OFF -DWITH_IPP=OFF -DWITH_ADE=OFF -DOPENCV_ENABLE_NONFREE=OFF
ninja -C ~/build/opencv -j 3 install
```

- 모듈 목록에 `stitching` 이 있어야 RTAB-Map 설정 단계가 OpenCV 를 받아들인다(RTAB-Map 이 찾는 구성 요소).
- PC(WSL, 24 작업)에서 1 분 26 초. BRIEF 는 nonfree 가 아니라 `OPENCV_ENABLE_NONFREE=OFF` 그대로.

### GTSAM 4.2

```bash
git clone --depth 1 --branch 4.2 https://github.com/borglab/gtsam.git ~/src/gtsam
cmake -S ~/src/gtsam -B ~/build/gtsam -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$HOME/opt/gtsam   -DCMAKE_INSTALL_RPATH=$HOME/opt/gtsam/lib   -DGTSAM_BUILD_TESTS=OFF -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF -DGTSAM_BUILD_TIMING_ALWAYS=OFF -DGTSAM_BUILD_PYTHON=OFF   -DGTSAM_INSTALL_MATLAB_TOOLBOX=OFF -DGTSAM_BUILD_DOCS=OFF -DGTSAM_USE_SYSTEM_EIGEN=ON -DGTSAM_WITH_TBB=OFF   -DGTSAM_BUILD_WITH_MARCH_NATIVE=OFF -DGTSAM_BUILD_UNSTABLE=ON
ninja -C ~/build/gtsam -j 3 install
```

- `CMAKE_INSTALL_RPATH`: `libgtsam.so` 가 같이 설치되는 `libmetis-gtsam.so` 를 찾게 함(없으면 실행 때 "libmetis-gtsam.so: cannot open").
- PC 에서 약 1 분.

### RTAB-Map 0.24.2 (화면 없이 라이브러리만)

```bash
git clone --depth 1 --branch 0.24.2 https://github.com/introlab/rtabmap.git ~/src/rtabmap
PATH=/usr/bin:/bin cmake -S ~/src/rtabmap -B ~/build/rtabmap -G Ninja -DCMAKE_BUILD_TYPE=Release   -DCMAKE_INSTALL_PREFIX=$HOME/opt/rtabmap   -DOpenCV_DIR=$HOME/opt/opencv/lib/cmake/opencv4 -DGTSAM_DIR=$HOME/opt/gtsam/lib/cmake/GTSAM -DWITH_GTSAM=ON   -DCMAKE_INSTALL_RPATH="$HOME/opt/rtabmap/lib;$HOME/opt/gtsam/lib;$HOME/opt/opencv/lib"   -DWITH_QT=OFF -DBUILD_APP=OFF -DBUILD_TOOLS=OFF -DBUILD_EXAMPLES=OFF -DWITH_PYTHON=OFF -DBUILD_TESTING=OFF
ninja -C ~/build/rtabmap -j 3 && ninja -C ~/build/rtabmap install
```

- `CMAKE_INSTALL_RPATH`: 설치한 `librtabmap_core.so` 가 소스 빌드 GTSAM·OpenCV 를 잡게 함. 배포판 `libgtsam` 이 깔려 있으면 그쪽(TBB 할당자 빌드)을 잡아 연결 프로그램 링크가 `gtsam::NonlinearFactor::rekey` 정의 없음으로 멈춘다.
- `BUILD_TOOLS=OFF`: 도구 중 `rtabmap-euroc_dataset` 이 배포판 yaml-cpp 와 링크 오류. 연결 프로그램은 핵심 라이브러리만 쓴다.
- 설정 단계는 PATH 를 훑으니 `PATH=/usr/bin:/bin` 으로 줄여 돌린다.

### `rtab_bridge`

```bash
cd ~/om   # 저장소
cmake -S bridge -B ~/build/bridge -G Ninja   -DCMAKE_PREFIX_PATH="$HOME/opt/rtabmap;$HOME/opt/librealsense;$HOME/opt/gtsam;$HOME/opt/opencv"   -DOpenCV_DIR=$HOME/opt/opencv/lib/cmake/opencv4   -DCMAKE_EXE_LINKER_FLAGS="-Wl,-rpath-link,$HOME/opt/gtsam/lib -Wl,-rpath,$HOME/opt/gtsam/lib"
ninja -C ~/build/bridge
# ldd ~/build/bridge/rtab_bridge 에서 libgtsam·libopencv_core 가 ~/opt 쪽인지 확인
# 설정 출력에 "RealSense 입력 켬 (librealsense2 2.59.1)" 이 보여야 카메라 입력이 들어감
```

## 보드에서 처음 할 확인

1. `ondevice slam --realsense --out /dev/null --max-frames 300 --stats /tmp/rs.csv` — 첫 줄의 카메라 이름·펌웨어·보정값, 끝의 `읽기`(카메라 대기 + 정렬)·`위치추적` ms.
2. 위 보정값 줄의 `왜곡 모델`: D435 컬러는 왜곡 계수가 0 이라 왜곡 없는 모델로 넘긴다.
3. 코어 묶기 예(코어 4 개): `--threads` 기본(auto)은 본 스레드·읽기를 앞 코어 둘에, 점 · 다시 쌓기 · 기록을 나머지에 두고 위치 프로그램 풀은 코어 전부. 숫자는 [bridge.md](bridge.md) "시간".
