# 계획

## 목표

손으로 움직이는 RGB 카메라 하나로, 보드 안에서 끝까지(카메라 → NPU 추론 → 3D 매핑 → 씬그래프 → 검색) 물체 공간 기억을 만든다.

## 핵심 고민: 일반 카메라로 포즈와 3D 지도를 어떻게 얻나

[pose_and_map.md](pose_and_map.md) 에 후보·비교 방법 정리.

## 부품

| 단계 | 방법 | 돌리는 곳 |
|---|---|---|
| 카메라 포즈 | 고민 중 — pose_and_map.md | CPU + NPU |
| 깊이 | **YOLO26-depth** (DEEPX 모델 동물원에 n/s/m/l/x, 768×768, NYU 실내, max 10 m) | NPU |
| 물체 분할 | **ObjectSAM 을 YOLO26-seg 로 다시 학습** (클래스 없이 물체만, 벽·천장·바닥 제외) + 우리 데이터로 미세조정 | NPU |
| 물체 벡터 | CLIP 계열 영상 인코더, 물체 부분만 잘라 벡터 1개 (DEEPX 튜토리얼 T22 CLIP) | NPU |
| 글 검색 벡터 | 같은 CLIP 의 글 인코더 | CPU 또는 NPU |
| 3D 매핑 | 마스크 픽셀 × 깊이 → 3D 점, 3D 겹침 + 벡터 코사인으로 같은 물체 합치기 | CPU |
| 저장 | SQLite | CPU |
| 보기 | 웹 뷰어 | CPU |

## 씬그래프

| 층 | 담는 것 |
|---|---|
| 물체 노드 | id, 클래스, 3D 중심·박스, 대표 벡터, 본 횟수, 마지막으로 본 시각, 썸네일 |
| 관계 선 | 위에 / 안에 / 옆에 — 박스 위치로 기하 계산 |
| 키프레임 노드 | 카메라 포즈, 시각, 그때 보인 물체 id |

## 질의 (LLM 없음)

- 글: "텀블러" → 글 벡터 → 코사인 top-k → 좌표 + 관계 + 썸네일 + 마지막 본 시각
- 공간: "책상 위 물체" → 관계 선 따라감
- 뷰어: 3D 지도 위 물체를 누르면 위 정보

## 잴 것

| 항목 | 방법 |
|---|---|
| 카메라 포즈 오차 (ATE) | 정답 포즈 있는 공개 데이터 + ArUco 마커 |
| 물체 위치 오차 (cm) | 줄자로 잰 실제 위치와 비교 |
| 놓친 물체 / 중복 노드 수 | 장면마다 실제 물체 목록과 비교 |
| 검색 top-1 / top-5 정답률 | 질의 목록 고정 |
| 단계별 ms, 전체 FPS | Sync vs Async |
| 양자화 손실 | PC FP32 vs NPU INT8 (같은 검증셋) |
| 모델 크기 | `.onnx` vs `.dxnn` |
| 장시간 구동 | 연속 구동 온도·FPS (`dxtop`) |

## 역할

- A (모델·컴파일러): YOLO-seg 학습, 모델 변환, INT8 보정, 양자화 손실 측정
- B (보드·앱): 카메라, 포즈 추적, Async 실행기, 3D 매핑·씬그래프, 검색, 뷰어, 속도 측정

## 참고

기존 개인 프로젝트 `juyoung020/object-memory-robot` 의 물체 기억(같은 물체 합치기 규칙, 이름 + 벡터 검색, sgview 뷰어)을 GPU 없이 옮긴다.

## 분할 모델: ObjectSAM → YOLO26

- 지금 ObjectSAM v21L = YOLOv8s-seg(1클래스, 640, 1,180만 변수). CropFormer Hornet-L 선생이 COCO 11.8만 장에 만든 라벨로 증류, 직접 짠 C++/CUDA 학습기(YOLOv8s-seg 전용).
- 예전 ObjectSAM v1.0 = YOLO26n-seg(416, FastSAM-s 라벨, Ultralytics 학습). 성능 차이(uHumans2 F1 42.9 vs 66.6)는 라벨·크기·입력이 다 달라 구조 비교가 아니다.
- YOLO26 으로 가는 이유: DEEPX 가 공식 지원(dx_app 예제, modelzoo `yolo26-seg`, 튜토리얼 T21 에 YOLO26s 검출·포즈·분할·깊이 동시 실행), NMS·DFL 없는 머리라 CPU 후처리가 가볍다(DEEPX 파이프라인은 머리 해독을 CPU 에서 ONNX Runtime 으로 돎).
- 할 것: v21L 의 CropFormer 라벨을 YOLO-seg 형식으로 바꿔 Ultralytics 로 YOLO26-seg(n·s) 학습 → 같은 uHumans2 벤치마크로 v21L 과 비교 → .dxnn 변환해 NPU ms·INT8 손실 측정.
- YOLO27 은 Ultralytics 문서에 미리보기만 있고 가중치·코드 미공개(2026-10-29), DEEPX 지원 없음.

### 보드 실측 (DEEPX dx-benchmark, RPi5 + DX-M1, dx-all-suite v2.4.0, 2026-07-22)

| 모델 (640) | 1장 지연 ms (단일 코어) | 최대 처리량 FPS (3코어 비동기) | 전체 파이프라인 FPS |
|---|---|---|---|
| YOLO26n-seg | 35.5 | 95.7 | 55.1 |
| **YOLO26s-seg** | 43.6 | **95.7** | 55.2 |
| YOLO26m-seg | 56.8 | 80.0 | 54.8 |
| YOLO26l-seg | 65.0 | 64.3 | 53.1 |

(ORT OFF 기준. ORT ON 이면 처리량 67 FPS — Pi5 에선 머리 해독을 ORT 대신 C++ 로 하는 쪽이 +42 %)
- Pi5 에선 n 과 s 처리량이 같다(호스트 CPU 가 한계) → **s 를 고른다**(COCO 마스크 mAP 33.9 → 40.0 을 공짜로).
- YOLO26s-seg vs YOLOv8s-seg(지금 ObjectSAM): 마스크 mAP 40.0 vs 36.8, 연산 34.5 vs 42.6 GFLOPs, 변수 10.4 M vs 11.8 M (Ultralytics 표).
- 깊이(YOLO26-depth-s 768, 37.7 GMACs)·CLIP 은 이 벤치마크에 없음 → 보드에서 `dxrun --use-ort -t 3 -v` 로 직접 잰다.

### 최적화 손잡이

| 손잡이 | 효과 / 근거 |
|---|---|
| 머리 해독 ORT 끄기(C++ 후처리) | Pi5 분할 처리량 67 → 96 FPS (dx-benchmark) |
| 비동기 + 3코어 + 버퍼 수 조정 | s-seg 단일 동기 22 FPS → 비동기 96 FPS, 최적 버퍼 6~9 |
| 양자화 프로필 c-lite / q-lite(100장) / q-pro(1024장) | 정확도 ↔ 변환 시간. 보정 사진은 우리 카메라 사진으로 |
| 입력 크기(640 → 더 작게) | NPU 시간 ≈ 화소 수. 다시 변환 필요, mAP 와 함께 곡선으로 |
| 키프레임만 NPU | 추적은 매 프레임 CPU, 분할·깊이·CLIP 은 키프레임만 |
| 카메라 YUYV 로 받기 | Pi5 는 JPEG 하드웨어 해독이 없어 MJPEG 해독이 CPU 를 먹음 |

### 학습기 (C++/CUDA, 2026-10-29)

`train/ObjectSAM` (juyoung020/ObjectSAM, `yolo26` 브랜치)에 YOLO26s-seg 를 넣었다. 자세히 `train/ObjectSAM/docs/YOLO26_PORT.md`.
- PyTorch 와 정방향·역방향 정합성 통과(변수 11,434,463 일치)
- 학습 속도 46.4 ms/스텝 = 345 영상/초 (640, 배치 16, 윈도우 RTX 5070 Ti) — 지금 YOLOv8s-seg 학습기 37.9 ms
- 본 학습(COCO + CropFormer 라벨)은 리눅스에서
