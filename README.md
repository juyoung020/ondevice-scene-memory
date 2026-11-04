# 물체 공간 기억 — 손카메라로 훑으면 보드가 방을 3D로 기억한다

[변경 기록(CHANGELOG)](CHANGELOG.md)

USB 카메라를 손에 들고 방을 한 바퀴 훑으면, 라즈베리파이 5와 DEEPX DX-M1 NPU 보드가 다음 세 가지를 실시간으로 만들어 기억합니다.

- **3D 지도:** 방 전체의 점 지도와 바닥 점유 지도
- **물체 목록:** 무엇이 어디에 있는지(3D 위치, 사진, 마지막으로 본 시각)
- **씬그래프:** 방 → 물체, 물체 ↔ 물체 관계(위·옆)

계산은 모두 보드 안에서 합니다. 입력은 **일반 RGB 카메라 한 대**이고, 깊이와 카메라 위치도 영상에서 추정합니다.

![손카메라로 책상 주위를 돌며 3D 지도·물체·방이 쌓이는 모습](docs/cover-slam.gif)

*공개 영상(TUM RGB-D 사무실)을 넣은 모습. 아래는 3D 지도와 손에 든 카메라(파란 원뿔), 가운데는 찾은 물체, 위는 방입니다. 고화질: [cover-slam.mp4](docs/cover-slam.mp4)*

## 되는 것

- 물체 찾기(COCO 80종: 의자·컵·노트북·책 …) — 보드 NPU
- 화면 전체 깊이 추정 — 보드 NPU
- 카메라 위치 추정(영상만으로) — 보드 NPU 특징점 + CPU
- 3D 지도, 물체 기억(옮겨짐·사라짐 표시)
- 방 나누기, 방↔물체·물체↔물체 연결
- 보드 웹캠 실시간, PC·폰 브라우저 뷰어

다음 단계: 물체 검색("텀블러 어디 있어?"), 방 이름 붙이기, IMU 더한 위치 추정, 웹캠 보정, 직접 학습한 분할 모델(ObjectSAM).

## 빠른 시작 — PC(Windows)

보드 없이 PC에서 같은 과정을 그대로 돌려 볼 수 있습니다. 그래픽카드가 있으면 더 빠릅니다(RTX 5070 Ti에서 22.8 키프레임/초).

**1. 모델 받기** — `models/` 폴더에 둡니다.

| 파일 | 하는 일 | 받는 곳 |
|---|---|---|
| `yolo26s-seg.onnx` | 물체 찾기(분할) | Hugging Face `prithivMLmods/YOLO26-ONNX` |
| `yolo26s-depth.onnx` | 깊이 추정 | [mobilint/YOLO26s-depth](https://huggingface.co/mobilint/YOLO26s-depth/resolve/main/yolo26s-depth.onnx) |
| `superpoint.onnx` | 위치 추정용 특징점 | [EclipseAidge/SuperPoint](https://huggingface.co/EclipseAidge/SuperPoint/resolve/main/superpoint.onnx) |

**2. 시험 영상 받기** — [TUM RGB-D](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download)에서 `freiburg1_desk`(책상, 23초) 또는 `freiburg3_long_office_household`(사무실, 87초)를 받아 `data/tum/`에 풉니다.

**3. 빌드하고 실행**

```bash
cargo build --release --features directml --manifest-path pipeline/Cargo.toml
cargo build --release --manifest-path viewer/sgview/Cargo.toml

# 영상을 실제 속도로 넣으며 기억을 쌓는다(끝나면 처음부터 반복)
pipeline/target/release/memmap --tum data/tum/rgbd_dataset_freiburg1_desk --out data/memory/desk \
    --every 3 --depth models/yolo26s-depth.onnx --pose vo --live 1 --loop

# 다른 창에서 뷰어 → 브라우저로 http://127.0.0.1:8091
viewer/sgview/target/release/sgview data/memory/desk --port 8091 --bind 127.0.0.1
```

`--live`를 빼면 영상을 최대 속도로 한 번 돌리고 끝에 요약(속도, 물체 수, 위치 오차)을 출력합니다.

## 보드에서 실행 — Raspberry Pi 5 + DX-M1

모델 세 개를 모두 NPU에서 돌립니다. 모델은 DX-M1 공식 모델 묶음에 들어 있습니다.

```bash
cargo build --release --no-default-features --features dxrt,cam --manifest-path pipeline/Cargo.toml
M=~/dx-all-suite_my/workspace/res/models
NPU="--seg $M/yolo26-s-seg_640x640.dxnn --depth $M/yolo26-depth-s_768x768.dxnn --pose vo --superpoint $M/superpoint_480x640.dxnn"

# 시험 영상으로
pipeline/target/release/memmap --tum data/tum/rgbd_dataset_freiburg1_desk --out data/memory/live --every 3 $NPU --live 1 --loop
# 웹캠으로(실시간, 모델 셋을 NPU 에서 동시에, 찍은 영상은 data/rec/cam1 에 남김)
pipeline/target/release/memmap --cam /dev/video0 --out data/memory/live $NPU --live 1 --npu-par --seg-every 2 \
    --scene-vox 0.02 --max-depth 3 --rec data/rec/cam1
# 남긴 영상을 같은 조건으로 다시(PC·보드 모두)
pipeline/target/release/memmap --cam data/rec/cam1 --out data/memory/replay $NPU --npu-par --seg-every 2

# 뷰어: 같은 네트워크의 PC·폰에서 http://<보드 주소>:8091
sgview data/memory/live --port 8091 --bind 0.0.0.0
```

- **NPU는 한 번에 한 프로그램만** 쓰세요. 두 개가 동시에 쓰면 NPU 연결이 끊겨 재부팅해야 할 수 있습니다. 새로 돌리기 전에 `pkill -f 'memmap --'`.
- 웹캠은 첫 장면을 "바닥에서 1.3 m 높이, 수평으로 앞을 봄"으로 놓고 시작합니다. 카메라 보정값이 다르면 `--k fx,fy,cx,cy`로 줍니다(기본 450,450,320,240).

## 주요 옵션

| 옵션 | 뜻 |
|---|---|
| `--tum <폴더>` / `--cam <장치 또는 녹화 폴더>` | 입력: TUM 영상, 웹캠, 또는 `--rec` 로 남긴 녹화 |
| `--rec <폴더>` | 웹캠 영상을 그대로 남김(나중에 `--cam <폴더>` 로 다시) |
| `--npu-par` | 모델 셋을 NPU 에서 동시에(보드에서 약 2 배 빠름) |
| `--seg-every N` | 물체 분할은 N 장마다(위치 추정은 매 장) |
| `--scene-vox` / `--max-depth` | 3D 지도 칸 크기(m, 기본 0.01) / 지도에 넣을 최대 거리(m, 기본 4) |
| `--out <폴더>` | 기억을 저장할 폴더(뷰어가 이 폴더를 읽음) |
| `--depth kinect \| <모델>` | 깊이: `kinect`면 TUM의 실측 깊이(비교 기준), 모델 파일이면 딥러닝 추정 |
| `--pose vo` | 카메라 위치를 영상만으로 추정. 빼면 TUM의 정답 위치를 씀 |
| `--every N` | N장마다 한 장씩 처리(기본 5) |
| `--live 1` / `--loop` | 실제 속도로 재생하며 뷰어에 계속 갱신 / 끝나면 반복 |
| `--conf 0.35` | 물체로 인정할 확신도 |
| `--snaps 150,500` | 해당 키프레임 시점의 기억을 따로 저장(비교용) |

## 뷰어 보는 법

- **마우스:** 끌면 돌리기, 휠은 확대·축소, 오른쪽 끌기는 이동. 왼쪽 위 `Top view`(위에서 보기)·`Fit`(전체 맞추기).
- **화면 구성:** 바닥의 흰색·검은색 칸이 점유 지도(갈 수 있는 곳·막힌 곳), 색 점이 3D 지도, 파란 원뿔이 지금 카메라입니다.
- **물체 색:** 초록 = 지금 보임, 주황 = 옮겨짐, 회색 = 그 자리에서 사라짐.
- **오른쪽 패널에서 켜고 끄기:** `3D map points`(지도), `Object name labels`(이름표), `Show graph layers`(위쪽 물체·방 층), `Objects as`(점 또는 표면).
