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
