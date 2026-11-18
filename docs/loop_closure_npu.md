# 루프 클로저 · 재위치: MegaLoc(후보) + XFeat(검증)을 NPU 로

후보 찾기 = MegaLoc(168×224, DINOv2-B + SALAD), 검증 = XFeat(480×640) + 깊이 + IMU 중력 4 자유도. 신경망 본체만 NPU(DX-M1, INT8 `.dxnn`), 그 앞뒤(전처리 · 집계 · 점 뽑기 · 기술자 · 짝 · RANSAC)는 모두 Rust. 런타임에 파이썬 없음. 앱(`app.rs` · `klt.rs`)에는 아직 잇지 않았다 — 잇는 자리는 7 절.

## 1. NPU 메모리 · 크기 (네 모델 함께 올릴 때)

⟨MEM⟩

## 2. 결론

⟨CONCL⟩

## 3. 만든 것

| 파일 | 하는 일 |
|---|---|
| `slam/src/place.rs` | MegaLoc 앞뒤: OpenCV INTER_AREA 와 같은 축소(640×480 → 224×168) · 14×14 조각 [192, 588] · Sinkhorn(쓰레기 칸, 로그 공간 3 회) 집계 · 군집 정규화 · 투영(원본 마지막 선형 층 또는 주성분 접은 판) · 정규화, 키프레임 기술자 표 `PlaceDb`(연속 행렬, 내적 상위 k, NEON) |
| `slam/src/xfeat.rs` | XFeat 앞뒤: 채널 평균 + InstanceNorm, NPU 입력 u8 변환, softmax(65) → 8×8 펼치기 → 5×5 NMS(문턱 0.05) → 점수(점 확률 × 신뢰도 쌍선형) → 상위 1000 → 쌍삼차 기술자(PyTorch `grid_sample` 과 같은 식), 상호 최근접 짝(NEON 내적) |
| `slam/src/lcverify.rs` | 검증: 깊이 3D-3D 두 짝 가설(중력 정렬 뒤 방위 + 이동) RANSAC 200 회(결정적 LCG) → 4 자유도 GN → 6 자유도 재투영 GN(문턱 8 → 4 → 2 화소) → 맞은 점 ≥ 20 |
| `slam/src/nnrun.rs` | 실행기 하나: `.dxnn` = DXRT(`common::dxrt` 재사용, 비동기 올리기 · 받기, 받을 버퍼 미리 넘김), `.onnx` = ONNX Runtime(PC). 출력은 이름으로 찾음 |
| `slam/examples/lc_bench.rs` | 파이썬 기준과 비교 · TUM 재생 벤치 · 보드 확인(`npu-check` · `npu-stress` · `prep-in`) |
| `scripts/board_lc_check.sh` | 보드 확인 한 번에(올리기 · sha256 · 단독 시간 · PC 출력과 코사인 · 네 모델 동시 1 분) |
| `slam/Cargo.toml` | 기능 `onnx`(ort) · `dxrt`(common/dxrt) |

모델 만드는 법(내보내기 · 손질 · 보정 · 컴파일)은 [npu_models.md](npu_models.md) "루프 · 재위치".

## 4. 포팅 확인: 원본 파이썬(FP32, PC) ↔ Rust, 같은 입력

⟨PORT⟩

## 5. 정확도: 실수 → NPU INT8

⟨INT8⟩

## 6. 검증 방식 비교 (번짐 게이트 15 화소 안: 흐림 0 · 5 · 10 · 15)

⟨VERIFY⟩

## 7. 앱에 잇는 자리 (통합 단계에서)

⟨WIRE⟩

## 8. 실행 빈도와 NPU 부하

⟨LOAD⟩

## 9. 보드에서 할 일

⟨BOARD⟩

## 10. 한계

⟨LIMIT⟩
