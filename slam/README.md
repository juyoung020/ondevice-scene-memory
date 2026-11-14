# slam — 위치 프로그램 `slam_bridge`

RTAB-Map 0.24.2 에서 우리 설정(`docs/rtabmap_pi.ini`)이 지나가는 경로 — F2F 위치 추적, 루프 클로저, GTSAM 그래프 최적화 — 를 Rust 로 옮긴 크레이트다. 실행 파일 `slam_bridge` 는 RGB-D 입력(TUM 폴더 또는 RealSense D435)을 받아 장면마다 위치를 구하고, 위치 + 컬러 + 컬러에 맞춘 깊이를 memmap 으로 보낸다(`memmap --stream -`).

- 명령줄과 출력은 C++ 원본 `rtab_bridge`(`bridge/src`)와 같다. PC(x86)에서 같은 입력·설정이면 스트림, 보낸 위치(`--poses`), 최종 지도 위치(`--final-poses`), 장면별 판단(`--stats`)이 바이트까지 같다.
- 속도는 원본의 약 2.2 – 2.3 배(PC WSL, fr1 책상 268 장/초, 복도 232 장/초). 본 스레드 계산이 장면당 2.8 – 3.1 ms 다.
- 인자·스트림 형식·좌표 약속: [docs/bridge.md](../docs/bridge.md). 보드 설치: [docs/board_bridge.md](../docs/board_bridge.md).
- 보드 실시간 스크립트는 지금 `rtab_bridge` 를 쓴다. 보드(ARM)에서 원본과 같은 바이트 비교를 마치면 `slam_bridge` 로 바꾼다.

## 모듈

| 모듈 | 하는 일 | 원본 |
|---|---|---|
| `features` | 장면 → GFTT 특징점 · BRIEF 32 바이트 · 3D 점 | RTAB-Map `Features2d.cpp`, `RegistrationVis.cpp`, OpenCV GFTT · BRIEF |
| `odom` | 장면-키프레임 위치 추적(속도 추측, 놓침, 키프레임 바꾸기) | `OdometryF2F.cpp`, `Odometry.cpp` |
| `register` (+ `rtf` · `cvmath` · `epnp` · `pnp` · `ba`) | 특징 짝짓기 → PnP RANSAC → 두 장면 묶음 조정. 위치 곱셈은 RTAB-Map `Transform`(float 3×4)과 같게 | `RegistrationVis.cpp`, `solvepnp.cpp`, OpenCV `epnp.cpp` · `lapack.cpp`, g2o `optimizeBA` |
| `loopclose` | 지도 갱신 `Rtabmap::process`: 노드 기억, 단어 사전, 루프 가설, 근접 찾기, 지도 보정 | `Rtabmap.cpp`, `Memory.cpp`, `VWDictionary.cpp` |
| `loopclose/flann` | 단어 사전 찾기 색인(rtflann KD-트리 4 개, L1, checks 32) | RTAB-Map `FlannIndex.cpp` · rtflann |
| `loopclose/mem` | 노드(Signature) · 링크 · 단어 사전 | `Memory.cpp`, `Signature.cpp`, `VWDictionary.cpp` |
| `loopclose/bayes` | 루프 가설 베이즈 필터 | `BayesFilter.cpp`, `bayes/*.cpp` |
| `loopclose/nnkd` | 가까운 노드 찾기 kd-나무(잎 15) | FLANN 1.9 `KDTreeSingleIndex`, PCL `KdTreeFLANN` |
| `loopclose/refdump` | 원본 덤프와 단계별 비트 비교(시험용) | — |
| `graph` (+ `geom` · `solve`) | 그래프 최적화(GTSAM 가우스-뉴턴, Pose3 지수·로그 사상) | `OptimizerGTSAM.cpp`, GTSAM 4.2 |
| `graph/ccolamd` | 변수 순서 | CCOLAMD(GTSAM 4.2 `3rdparty/CCOLAMD`) |
| `graph/exact` | 선형 풀이를 원본 덧셈 순서 그대로(소거 트리 → 접합 트리 → 클리크 Cholesky) | GTSAM 4.2 `GaussianFactorGraph::optimize`, Eigen 3.4 |
| `io` (+ `io/realsense`) | TUM 재생, RealSense 입력(`realsense` 기능), 읽기 스레드(위치 추적 특징점도 여기서 뽑음) | `bridge/src` 의 입력 코드, OpenCV `cvtColor` |
| `stream` | 장면 스트림 쓰기 스레드 | `bridge/src/stream_writer.cpp` |
| `params` | 설정: 기본값 ← ini `[Core]` ← 명령줄 `--Group/Name 값` | RTAB-Map `Parameters.h` |
| `main` | `slam_bridge`: 명령줄, 장면마다의 판단(다시 잡기 · 새 지도 · 맡아 둠 · 그래프 블록), 끝 요약 | `bridge/src/main.cpp` |

난수(`rand()` · `std::random_device`)는 프로세스 하나에 하나인 `register::SharedRng` 를 모든 모듈이 원본과 같은 순서로 쓴다.

## 검증

기록은 [docs/performance_log.md](../docs/performance_log.md) 의 다음 절에 있다.

- "SLAM Rust 이식: 단어 사전(FLANN) · 입출력" — 단어 사전, 입출력, 특징점, 루프 클로저 단계, 그래프 최적화를 조각마다 원본과 비트 비교.
- "SLAM Rust 이식: 위치 추적(OdometryF2F + RegistrationVis)" — 장면마다 위치 · 공분산까지 비트 비교.
- "SLAM Rust 이식: 연결 프로그램 전체(slam_bridge ↔ rtab_bridge)" — 녹화 셋 전체 스트림 바이트 비교와 시간.
- "SLAM Rust 이식: 루프 클로저 검증 넓히기 · 속도" — 반복 · 새 지도 상황과 `getGraph`.
- "Rust 위치 프로그램(slam_bridge) 통합 재검증 · 읽기 스레드 특징점 · kd-나무 짓기" — 18 개 입력(원래 녹화 3 개 + 장면 빠짐 · 검은 장면 · 다시 잡기 · 새 지도 · 리셋 상황 15 개)에서 기본 설정과 정확한 kNN 모두 18/18 바이트 동일.

비교 기준은 C++ `rtab_bridge` 에 `std::random_device` 를 고정하는 LD_PRELOAD 심을 넣은 실행이다. 바이트 동일은 PC(x86, SSE, 곱셈-덧셈 합침 없음) 원본 기준이고, 보드 원본(ARM NEON)과의 비교는 보드에서 한다.

## 빌드 · 실행

```bash
# TUM 재생만
cargo build --release --manifest-path slam/Cargo.toml
# RealSense 입력 포함(librealsense2 C API 를 링크, 라이브러리 경로는 docs/board_bridge.md)
cargo build --release --features realsense --manifest-path slam/Cargo.toml

# TUM 재생 → memmap
slam/target/release/slam_bridge --tum data/d435/rgbd_dataset_freiburg3_office_sync --config docs/rtabmap_pi.ini \
  | pipeline/target/release/memmap --stream - --every 2 --seg models/yolo26s-seg.onnx --max-depth 3.5 --out data/memory/office

# 카메라 → memmap(실시간)
slam_bridge --realsense --config docs/rtabmap_pi.ini --send-every 2 --reset-file data/memory/live/reset.req \
  | memmap --stream - --every 1 --live 1 --seg <분할 모델> --max-depth 3.5 --out data/memory/live
```

시험용 `--reset-at N` 은 N 번째 장면에서 리셋 파일이 생긴 것처럼 처음부터 다시 시작한다. `--db` 는 받아서 넘어간다(데이터베이스 저장은 `rtab_bridge` 에만 있음).

## 기본 꺼진 선택

모두 원본과 바이트가 달라지는 선택이라 기본값은 원본 그대로다.

| 선택 | 켜는 법 | 효과 |
|---|---|---|
| `Rust/DictChecks` | `--Rust/DictChecks 16`(기본 32) | 단어 사전 찾기가 빨라지지만(긴 사무실 73 → 60 · 54 ms, checks 16 · 8) 루프 가설이 13 → 8 · 7 번으로 줄어 루프를 놓친다 |
| `Slam/FeatureBoxF32` | `--Slam/FeatureBoxF32 true` | 특징점 상자 합을 f32 로: 특징점 13 – 17 % 빠름, 점 순서가 바뀜(점 집합은 거의 같음) |
| `fast_solver` | 코드 안 `OptimizerParams` 칸 | 그래프 최적화를 원본과 다른 근사 풀이(`graph/solve`)로 1.5 – 2.5 배 빠르게, 값은 마지막 몇 ulp 다름 |
| `Vis/BundleAdjustment` | `--Vis/BundleAdjustment 0`(기본 1, 원본 기본값) | 위치 추적의 두 장면 묶음 조정을 끔: 1.0 · 1.5 ms/장(책상 · 복도), ATE 0.0792 → 0.0506 m · 0.386 → 0.351 m |
| `Slam/BAEarlyStop` | `--Slam/BAEarlyStop 1e-3`(기본 0 = 끔) | 번들 조정 반복의 상대 chi2 감소가 이 값보다 작으면 멈춤: BA 0.5 → 0.2 ms/장, ATE desk 0.0792 → 0.0784 · 복도 0.386 → 0.280 · 긴 사무실 0.0434 → 0.0405 m |
| `Slam/BAAnalyticJacobian` | `--Slam/BAAnalyticJacobian true` | 두 카메라 사이 변의 야코비안을 해석식으로: 시간 · ATE 같음, 비트만 다름(쓸 이유 없음) |
| `Vis/CorNNType` | `--Vis/CorNNType 3`(기본 1, kd-나무) | 정확한 kNN 짝짓기: 시간 · ATE 같음, 원본 같은 설정과도 바이트 동일 |

## 출처와 라이선스

이 크레이트는 저장소와 같이 PolyForm Noncommercial 1.0.0 이다. 아래 원본의 알고리즘과 연산 순서를 옮겼고, 각 원본의 라이선스와 저작권 표시를 따른다. 모듈 머리말에 원본 파일 · 함수를 적었다.

- RTAB-Map 0.24.2 (introlab/rtabmap, 안의 rtflann 포함) — BSD-3-Clause
- OpenCV 4.10 (GFTT · BRIEF · cvtColor · EPnP · SVD) — Apache-2.0
- GTSAM 4.2 (borglab/gtsam) — BSD
- CCOLAMD (SuiteSparse, GTSAM `3rdparty` 판) — BSD-3-Clause
- FLANN 1.9 · PCL `KdTreeFLANN` — BSD
- g2o (`optimizeBA` 경로) — BSD. 6×6 선형 풀이(`ba.rs` `chol6_solve`)는 알고리즘 설명(up-looking Cholesky)에서 직접 쓴 독자 구현이다
- Eigen 3.4 (덧셈 순서를 따름) — MPL-2.0
- librealsense2 (`realsense` 기능에서 링크) — Apache-2.0
