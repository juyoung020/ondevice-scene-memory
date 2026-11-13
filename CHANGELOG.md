# Changelog

이 프로젝트의 주요 변경 사항을 모두 이 파일에 기록한다.

형식은 [Keep a Changelog 1.1.0](https://keepachangelog.com/ko/1.1.0/)을 따르고, 버전은 [Semantic Versioning 2.0.0](https://semver.org/lang/ko/)을 따른다.

## [Unreleased]

### Added
- 라이선스: PolyForm Noncommercial 1.0.0(상업적 이용 금지), `LICENSE.md`.
- 물체 이름을 사진으로 붙이기 `memmap --vocab vocab/names.voc`: 이름 1,221 개(LVIS · COCO 등) 중 사진이 가장 닮은 이름을 쓴다(tv → monitor · laptop computer).
- 이름 글 벡터를 미리 만들어 두는 `sgview --encode-vocab <이름 목록> <파일>` 과 그 파일을 읽는 `sgview --vocab`.
- 분할 모델의 클래스 이름 `memmap --seg-names <이름,…>`: 이름 없이 물체만 자르는 1 클래스 모델(ObjectSAM)은 `--seg-names object` 로 쓰고 이름은 `--vocab` 이 붙인다.

### Changed
- 카메라를 가만히 두는 동안(1 cm · 1° 안)은 다시 쌓기용 장면을 2 초에 한 장만 보관한다.
- 검색에서 질의와 같은 뜻으로 읽힌 이름은 사진 비교의 경쟁 상대에서 뺀다.

## [0.8.0] - 2026-11-13

### Added
- 보드에서 아무 말로 물체 찾기: `memmap --embed` 가 NPU 모델(.dxnn)을 받는다(SigLIP2 영상 인코더, 사진 한 장 18 ms).
- 사진 벡터 확인 `memmap --embed-check <모델> <사진 폴더> <벡터 파일>`: PC 의 ONNX 벡터와 보드 NPU 벡터의 코사인·추론 시간.
- 컬러 노출 고정 `rtab_bridge --exposure <ms> [--gain 0..128]`.
- IMU(ICM-42688-P) 연결 확인 도구 `bridge/tools/imu_probe.cpp`.
- 뷰어 수평 버튼(리셋 왼쪽): 카메라를 수평으로 든 채 누르면 그 자세가 수평이 되게 쌓인 지도·물체를 돌려 다시 쌓는다(`GET /api/level`, `level.req`).

## [0.7.0] - 2026-11-12

### Added
- 위치를 놓친 뒤 새로 찍은 구간을 버리지 않고 맡아 두었다가, 아는 곳과 다시 이어지면 지도·물체에 넣는다(스트림 상태 8, 그래프 노드의 맡아 둠 칸, 결과 `stream_held`).

## [0.6.0] - 2026-11-12

### Added
- 글로 물체 찾기: 뷰어 검색창에 아무 말이나 치면(의자, something to drink) 닮은 물체를 점수 순으로 보여 주고 지도에 표시한다(`sgview --search-model`, `GET /api/search`).
- 물체 사진 벡터(`memmap --embed`, SigLIP2): `objects/O<id>_emb.bin`, `view.json` 물체의 `emb`.
- 분할 모델이 낼 수 있는 이름 목록 `vocab.json`.

### Changed
- 물체 병합이 사진 벡터도 본다: 한 물체가 두 이름으로 따로 남으면(인형이 의자로도 잡힘) 사진이 닮고 자리가 겹칠 때 하나로 합친다.

## [0.5.0] - 2026-11-12

### Added
- 방 사이 문(위치·폭)과 방↔방 연결(`view.json` 의 `doors`).
- 방 이름에 종류별 확률과 근거 물체(`view.json` 의 `rooms[].probs`·`evidence`).
- 위치를 놓치면 같은 지도에서 바로 다시 잡는다(`rtab_bridge --reloc-wait`).
- 뷰어 리셋 아이콘(오른쪽 아래): 위치를 다시 잡고 지도를 처음부터(`rtab_bridge --reset-file`).
- `--tum` 재생이 폴더의 `calibration.txt` 보정값을 읽는다(ETH3D 영상).

### Changed
- **호환 변경:** 방 나누기를 Hydra room finder 의 2D 판으로 바꿈(문처럼 좁은 목에서 끊음).
- **호환 변경:** 방 번호가 1 부터이고 지도가 자라도 유지된다. 같은 종류의 방은 번호가 붙는다(kitchen 2).
- 방은 벽으로만 나뉜다(책상·의자 같은 낮은 가구는 방 안).
- 3D 지도는 점 지도 하나이고 옵션 없이 늘 만든다. 뷰어도 점 지도로 열린다.
- 장면 넘기기와 보관 장면 압축이 가벼워짐(결과 값은 같음).

### Removed
- **호환 변경:** 표면 지도(`--surface`, TSDF 서브맵·사진 텍스처)와 옵션 `--points`, 도구 `surfcmp`·`meshshot`.

### Fixed
- 가구 때문에 한 방이 여러 방으로 쪼개지던 문제.
- NPU 없는 PC 에서 CPU 분할이 코어 둘에 묶여 지도 갱신이 느리던 문제.
- 위치를 놓친 장면마다 새 지도를 열어 지도가 조각나던 문제.
- 위치를 놓친 뒤 틀린 위치로 지도를 쌓아 천장·벽이 엉뚱한 곳에 찍히던 문제.
- memmap 이 밀릴 때 큐에 든 장면의 컬러가 다음 장면으로 덮이던 문제.
- 카메라를 켠 직후의 어두운 첫 장면 때문에 위치를 바로 놓치던 문제.

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
- ADVIO 영상과 IMU 시각이 0.318 s 어긋나 VIO 가 시작되지 않던 문제.

## [0.1.0] - 2026-10-29

### Added
- 프로젝트 계획과 하드웨어 결정(USB 카메라, Pi5 + DX-M1).
- YOLO26s-seg C++/CUDA 학습기 `train/ObjectSAM`.

### Fixed
- 모델 설정 필드 순서로 초기화가 깨지던 문제.

[Unreleased]: https://github.com/juyoung020/ondevice-scene-memory/compare/v0.8.0...HEAD
[0.8.0]: https://github.com/juyoung020/ondevice-scene-memory/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/juyoung020/ondevice-scene-memory/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/juyoung020/ondevice-scene-memory/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/juyoung020/ondevice-scene-memory/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/juyoung020/ondevice-scene-memory/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/juyoung020/ondevice-scene-memory/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/juyoung020/ondevice-scene-memory/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/juyoung020/ondevice-scene-memory/releases/tag/v0.1.0
