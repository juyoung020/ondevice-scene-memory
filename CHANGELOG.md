# Changelog

이 프로젝트의 주요 변경 사항을 모두 이 파일에 기록한다.

형식은 [Keep a Changelog 1.1.0](https://keepachangelog.com/ko/1.1.0/)을 따르고, 버전은 [Semantic Versioning 2.0.0](https://semver.org/lang/ko/)을 따른다.

## [Unreleased]

### Added
- 뷰어 리셋 아이콘(오른쪽 아래): 위치를 다시 잡고 지도를 처음부터(`rtab_bridge --reset-file`).
- 위치를 놓치면 같은 지도에서 바로 다시 잡는다(`rtab_bridge --reloc-wait`).
- `--tum` 재생이 폴더의 `calibration.txt` 보정값을 읽는다(ETH3D 영상).

### Changed
- 뷰어 기본 지도를 표면에서 점 지도로(더 빨리 갱신됨, 표면은 설정에서 켬).
- 연결 프로그램 → memmap 장면 넘기기의 복사를 줄임(결과 값은 같음).

### Fixed
- NPU 없는 PC 에서 CPU 분할이 코어 둘에 묶여 지도 갱신이 느리던 문제.
- 위치를 놓친 장면마다 새 지도를 열어 지도가 조각나던 문제.
- 위치를 놓친 뒤 틀린 위치로 지도를 쌓아 천장·벽이 엉뚱한 곳에 찍히던 문제.
- 카메라 입력에서 memmap 이 밀릴 때 큐에 든 장면의 컬러가 다음 장면으로 덮이던 문제.

## [0.4.0] - 2026-11-11

### Added
- Rust 파이프라인 `pipeline/`(memmap): 같은 코드가 PC(ONNX Runtime)와 보드(Pi5 + DX-M1 NPU)에서 돈다.
- 연결 프로그램 `bridge/`(C++ `rtab_bridge`): RealSense D435 또는 TUM 폴더 → RTAB-Map 위치 추적·루프 클로저 → memmap 장면 스트림.
- memmap 장면 스트림 입력(`--stream`)과 루프 클로저 뒤 지도·물체 다시 쌓기.
- 사진 텍스처를 입힌 표면 지도(`--surface`, TSDF 서브맵).
- Potree 방식 3D 점 지도(`--points`).
- 보드 실시간: 밀리면 최신 장면으로 건너뛰고, 밀릴 때만 분할을 생략한다(`--seg-skip`).
- 물체 분할 단계를 겹쳐 돌리고 NPU 에 4 장까지 함께 올린다(`--npu-jobs`).
- 스레드 우선순위·코어 배치(`--threads`).
- 방 종류 이름(물체 구성으로 사무실·주방·거실 …).
- 따로 생긴 같은 물체를 합치는 병합 후처리.
- 웹캠 입력·녹화·재생(`--cam`, `--rec`), 실제 속도 재생과 반복(`--live`, `--loop`).
- 카메라만으로 위치 추정(`--pose vo`·`dvo`), 외부 위치로 돌리기(`--gt`).
- 뷰어: 발표용 화면, High/Low 화질, 설정 패널·단축키, 손카메라 따라가기, 이름표.
- 깊이→컬러 정렬 라이브러리 `bridge/align` 과 RealSense 도구 `rs_info`·`rs_rec`·`rs_align_check`.
- 평가·확인 도구 `ate`·`surfcmp`·`bench`·`objdup`·`objcrop`·`meshshot`·`rgbd2tum`.
- 문서: 동작 원리, 연결 프로그램, 보드 설치, 카메라 입력, RTAB-Map 조정, 성능 기록, 평가 계획.

### Changed
- **호환 변경:** 분할 기본 모델을 COCO YOLO26s-seg 로.
- **호환 변경:** 표면 지도 칸 기본값 2 → 3 cm.
- 카메라를 RealSense D435, 위치 추정을 RTAB-Map 으로 정리.
- 지도를 조각 파일로 내보내 뷰어가 바뀐 조각만 받는다.
- 뷰어가 Pi 5 화면에서도 지도가 바뀌는 중에 부드럽게 돈다.
- 물체 병합·사라짐 판정을 다듬어 중복 물체가 줄었다.
- 지도 작업(점·텍스처·다시 쌓기·기록)을 본 스레드 밖으로 옮겨 장면 처리가 덜 막힌다.
- README·문서를 지금 구성으로 다시 쓰고 표지를 다시 찍음.

### Removed
- **호환 변경:** 파이썬 코드 전부(`tools/*.py`, `train/ObjectSAM` 의 파이썬). 실행은 `pipeline/` 으로.

### Fixed
- 그대로 있는 물체를 '사라짐'으로 잘못 표시하던 문제.
- 한 방이 여러 방으로 쪼개지던 문제.
- 위치 추정이 한 번 실패하면 지도가 어긋나던 문제.
- 가만히 둔 카메라의 위치가 조금씩 흘러가던 문제.
- 보드 화면 뷰어가 끊기고 멈추던 문제.
- 뷰어가 파일을 읽는 중일 때 윈도우에서 재생이 멈추던 문제.
- 뷰어 화질·표시 오류(점 지도 비침, 절반 해상도로 남음, 엉뚱한 삼각형, 점 수 "undefined").

## [0.3.0] - 2026-11-03

### Added
- TUM RGB-D 공개 영상 입력.
- 깊이 모델 비교(Depth Anything V2, YOLO26s-depth).
- 방 전체 3D 점 지도와 뷰어 표시.
- 씬그래프: 방 나누기, 방→물체·물체 위/옆 연결.
- 카메라 실제 자세 표시.
- 물체를 표면(그물)으로 보는 선택지.

### Changed
- **호환 변경:** 분할 기본 모델을 ObjectSAM v1.0 으로.
- 물체 점을 여러 장면에서 확인된 것만 남기도록.
- 뷰어의 지나온 길을 가는 선으로.
- 키프레임당 5.0 s → 0.5 s 로 빨라짐.

### Fixed
- 실시간에서 3D 지도 조각이 몇 초씩 안 나가던 문제(이제 1 초마다).
- VIO 경로가 중력과 어긋나 물체가 비스듬히 쌓이던 문제.
- 칸막이 같은 얇은 벽을 물체로 넣던 문제.

## [0.2.0] - 2026-11-03

### Added
- 영상과 위치로 점유 지도를 만들고 물체를 3D 로 기억하는 파이프라인.
- 씬그래프 웹 뷰어 `viewer/sgview`, 손에 든 카메라 표시.
- ADVIO 영상 변환과 ORB-SLAM3 VIO 연결.
- 결과 캡처 `docs/results/`.

### Changed
- 같은 물체를 3D 겹침으로 합치기.
- 바닥 높이 추정과 사라짐 판정을 더 엄격히.

### Fixed
- 실시간에서 3D 지도 조각이 몇 초씩 안 나가던 문제(이제 1 초마다).
- ADVIO 영상과 IMU 시각이 0.318 s 어긋나 VIO 가 시작되지 않던 문제.

## [0.1.0] - 2026-10-29

### Added
- 프로젝트 계획과 하드웨어 결정(USB 카메라, Pi5 + DX-M1).
- YOLO26s-seg C++/CUDA 학습기 `train/ObjectSAM`.

### Fixed
- 실시간에서 3D 지도 조각이 몇 초씩 안 나가던 문제(이제 1 초마다).
- 모델 설정 필드 순서로 초기화가 깨지던 문제.

[Unreleased]: https://github.com/juyoung020/ondevice-scene-memory/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/juyoung020/ondevice-scene-memory/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/juyoung020/ondevice-scene-memory/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/juyoung020/ondevice-scene-memory/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/juyoung020/ondevice-scene-memory/releases/tag/v0.1.0
