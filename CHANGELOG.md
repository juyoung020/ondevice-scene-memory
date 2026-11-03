# Changelog

이 프로젝트의 주요 변경 사항을 기록한다.
형식은 [Keep a Changelog](https://keepachangelog.com/ko/1.1.0/)를, 버전은 [Semantic Versioning](https://semver.org/lang/ko/)을 따른다.

## [Unreleased]

### Fixed
- VIO 궤적의 세계 z 가 중력과 58.7° 어긋나 물체가 비스듬히 최대 2.7 m 높이까지 쌓이던 문제: IMU 가속도로 위 방향을 재서 수평을 맞춤(`run_memory --pose vio`). 정면에서 물체가 바닥 높이에 놓임. 정면 캡처 `docs/results/advio15_vio_front.png`.
- VIO 경로가 IMU 수평면에서 13.4° 기울어 평지 보행인데 높이가 0.76 m 오르내리던 문제(GT 0.18 m): 기본 수평 기준을 경로 평면으로(`--level traj`, IMU 는 부호만).

### Changed
- 물체 점 칸 3 cm → 2 cm, 물체당 점 상한 4,000 → 12,000, 뷰어 점 크기 1.6 → 1.1 (색이 뭉개져 보이던 문제 완화).

## [0.2.0] - 2026-11-03

### Added
- 손카메라 영상+포즈로 점유 지도를 갱신하고 물체를 3D로 기억하는 추론 파이프라인 `tools/run_memory.py` (YOLO26s-seg + YOLO26s-depth, 사라짐 판정, 시간대별 스냅샷).
- 씬그래프 뷰어 `viewer/sgview` (Rust + three.js, object-memory-robot 에서 가져와 윈도우 MSVC 빌드).
- 뷰어: 손에 든 카메라 표시(폰 + 시야 원뿔 + 바닥 고리), 기억이 `handheld` 일 때 LIMO 대신.
- ADVIO(아이폰 카메라+IMU) → EuRoC 형식 변환 `tools/advio_to_euroc.py` 와 ORB-SLAM3 단안-관성 VIO 연결(`--pose vio`).
- 결과 캡처 `docs/results/`: ARKit/VIO 시간대별·3D 화면, 지도/물체 분리, 분할 모델 비교.

### Changed
- 물체 점: 여러 관측에서 지지받은 칸만 남기기, 3D 겹침으로 같은 물체 연관(물체 68 → 45).
- 바닥 높이를 키프레임마다 카메라 높이로 추정, 사라짐 판정 엄격히(관측 5+·깊이 일치·연속 8).
- 점유 지도를 키프레임마다 칸당 한 번만 세기.

### Fixed
- ADVIO 영상 시각이 IMU 보다 0.318 s 늦던 문제(영상 회전↔자이로 맞춤으로 측정) — VIO 가 매번 초기화 실패하던 원인. IMU 잡음 10배로 추적 안정.
  결과: VIO ATE 0.64 m / 경로 17 m (ARKit 같은 구간 0.64 m), 1,551 프레임 중 1,165 추적.

## [0.1.0] - 2026-10-29

### Added
- 프로젝트 계획·하드웨어 결정(일반 USB 카메라 + 납땜 없는 Qwiic IMU, Pi5 + DX-M1), 포즈·지도 방법 문서.
- `train/ObjectSAM`: YOLO26s-seg C++/CUDA 학습기(그래프·Attention·그룹 conv·E2E 손실), Ultralytics 정합성 통과, 46.4 ms/스텝(B16 640).

### Fixed
- ModelCfg 새 필드 순서로 기존 초기화가 깨지던 회귀.

[Unreleased]: https://github.com/juyoung020/ondevice-scene-memory/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/juyoung020/ondevice-scene-memory/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/juyoung020/ondevice-scene-memory/releases/tag/v0.1.0
