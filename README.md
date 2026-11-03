# 온디바이스 AI 기말 프로젝트 — 물체 공간 기억

손에 든 USB 카메라로 공간을 훑으면, 라즈베리파이 5 + DEEPX DX-M1 보드 안에서 본 물체를 **경량 3D 씬그래프**로 기억한다.
나중에 "텀블러"를 검색하면 3D 좌표, 주변 관계(책상 위·노트북 옆), 마지막으로 본 시각과 사진을 돌려준다. 모든 계산은 보드 안에서 하고, LLM 은 쓰지 않는다.

![손카메라로 책상 주위를 돌며 3D 점 지도·물체 노드·엣지·방 노드가 쌓이는 모습](docs/cover-slam.gif)

*TUM RGB-D fr3/long_office_household(22 m), 위치는 카메라만으로 추정(SuperPoint 특징점 + 랜드마크 평균, 정답 대비 오차 0.17 m), 깊이는 Kinect 실측. 약 2.4배속. 아래 = 3D 점 지도와 손에 든 카메라(파란 원뿔)·궤적, 가운데 = 물체 노드층(사진·상태색), 위 = 방 노드(물체 수), 회색 선 = 방→물체 엣지. 분할 COCO YOLO26s-seg. 고화질 [cover-slam.mp4](docs/cover-slam.mp4).*

## 흐름

```
USB 카메라 30fps ─▶ ① 카메라 포즈 추적
                         │ 키프레임
                         ▼
                   ② NPU: YOLO-seg ─ 깊이 추정 ─ CLIP 임베딩
                         ▼
                   ③ 3D 매핑 (CPU): 마스크 + 깊이 → 물체 3D 점 → 같은 물체 합치기
                         ▼
                   ④ 씬그래프 저장 → 검색 + 웹 뷰어
```

## 하드웨어

| 부품 | 사양 |
|---|---|
| 보드 | Raspberry Pi 5 (8 GB) |
| NPU | DEEPX DX-M1 (4 GB) |
| 카메라 | USB RGB 카메라 (깊이·IMU 없음), 손에 들고 움직임 |

## 폴더

| 폴더 | 내용 | 담당 |
|---|---|---|
| `docs/` | 설계·일정·측정 결과 ([plan.md](docs/plan.md)) | 같이 |
| `train/` | 분할 모델 학습 (PC). `train/ObjectSAM` = YOLO26s-seg C++/CUDA 학습기(원본 juyoung020/ObjectSAM 을 기록째로 합침, [docs/YOLO26_PORT.md](train/ObjectSAM/docs/YOLO26_PORT.md)) | A |
| `compile/` | ONNX → `.dxnn` 변환 설정, 양자화 보정 이미지 목록 | A |
| `pipeline/` | **지도·물체 기억 파이프라인(Rust)**: 영상+포즈 → 분할 → 누적 점유 지도·3D 점 지도·물체 트래커 → 방·엣지 → `view.json`. PC 는 ONNX Runtime, Pi5 는 DX-M1 NPU(DXRT C API) | B |
| `board/` | 보드에서 도는 코드: 카메라, 위치 추적, NPU 실행, 3D 매핑, 씬그래프, 검색 | B |
| `viewer/` | 3D 지도·씬그래프 웹 뷰어 | B |
| `tools/` | 속도·정확도 측정 스크립트 | 같이 |
| `models/` | `.pt` `.onnx` `.dxnn` (git 밖) | |
| `data/` | 촬영 영상·라벨·기억 저장본 (git 밖) | |

A = 모델·컴파일러, B = 보드·앱 (수업 지정 분담).

## 실행 (pipeline)

```bash
# PC (Windows, ONNX Runtime + 그래픽카드 DirectML) — 보드와 같은 구성: 분할·깊이·SuperPoint 모두 모델, 위치는 카메라만으로
#   models/yolo26s-seg.onnx, models/yolo26s-depth.onnx(HF mobilint/YOLO26s-depth), models/superpoint.onnx(HF EclipseAidge/SuperPoint)
cargo build --release --features directml --manifest-path pipeline/Cargo.toml
pipeline/target/release/memmap --tum data/tum/rgbd_dataset_freiburg1_desk --out data/memory/pc_fr1 --every 3 \
    --depth models/yolo26s-depth.onnx --pose vo [--live 1 --loop]
#   Kinect 실측 깊이·정답 위치로: --depth kinect (--pose 생략)

# Pi5 + DX-M1 (NPU, DXRT 3.4) — 모델 셋 모두 NPU
cargo build --release --no-default-features --features dxrt,cam --manifest-path pipeline/Cargo.toml
M=~/dx-all-suite_my/workspace/res/models
pipeline/target/release/memmap --tum <seq> --out data/memory/tum_pi --seg $M/yolo26-s-seg_640x640.dxnn \
    --depth $M/yolo26-depth-s_768x768.dxnn --pose vo --superpoint $M/superpoint_480x640.dxnn

# 보기
viewer/sgview/target/release/sgview data/memory/tum_rust --port 8091 --bind 127.0.0.1
```

TUM fr3/long_office_household(516 키프레임, 깊이는 Kinect 실측) 측정:

| | 키프레임/초 | 분할 ms | 지도·물체·기록 ms | 물체 |
|---|---|---|---|---|
| PC (RTX 5070 Ti, ONNX CPU) | 16.7 | 55 | 17 | 68 |
| Pi 5 + DX-M1 NPU | 11.8 | 70 (NPU 54) | 36 | 74 |

카메라만으로 위치(분할·딥러닝 깊이·SuperPoint, TUM fr1/desk 9.4 m):

| | 키프레임/초 | 위치 오차(ATE) |
|---|---|---|
| PC (RTX 5070 Ti, ONNX DirectML) | 22.8 | 0.57 m (Kinect 깊이면 0.15 m) |
| Pi 5 + DX-M1 NPU | 5.1–6.3 | 0.30 m (Kinect 깊이면 0.08 m) |

PC 와 보드의 SuperPoint 는 가중치 출처가 달라(PC rpautrat, 보드 DeepX 공식) 위치 오차가 다르다. PC 그림: [docs/results/pc_fr1desk_onnx.png](docs/results/pc_fr1desk_onnx.png).

분할은 작업 스레드에서 한 장 앞서 돌려 지도 갱신과 겹친다. 같은 시점 비교: [docs/results/pc_vs_pi.png](docs/results/pc_vs_pi.png), 속도 개선 기록: [docs/performance_log.md](docs/performance_log.md). 앞으로의 정확도·속도 평가는 [docs/evaluation_plan.md](docs/evaluation_plan.md).

## 보드 접속

```bash
ssh a2024130045@kau
source ~/dx-all-suite_my/dx-runtime/venv-dx-runtime/bin/activate
```

측정 방법은 `Desktop\kau-2026-2\온디바이스 AI\기말프로젝트\벤치마킹 가이드.md`.

## 규칙

- 경로에 공백·한글을 넣지 않는다 (DX 툴체인이 깨짐). 그래서 이 폴더가 `C:\ondevice` 다.
- 가중치·데이터는 git 에 넣지 않는다.
