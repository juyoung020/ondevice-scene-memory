# Changelog

이 프로젝트의 주요 변경 사항을 기록한다.
형식은 [Keep a Changelog](https://keepachangelog.com/ko/1.1.0/)를, 버전은 [Semantic Versioning](https://semver.org/lang/ko/)을 따른다.

## [Unreleased]

### Added
- 실시간 재생 `run_memory --live 1 [--loop]`: 영상 촬영 시각에 맞춰 키프레임마다 기억을 기록(처리가 밀리면 장면을 건너뜀), 뷰어가 파일 변화를 읽어 궤적·지도·물체가 자라는 모습을 생중계. 끝나면 처음부터 반복.
- TUM 실측 깊이 기준판 `--depth kinect`: 매핑이 정상(주사위·책·잡지가 사진과 같은 자리)으로 나와, 딥러닝 깊이 판의 어긋남이 깊이 배율 탓임을 확인. 같은 장면 물체 COCO+Kinect 69 / COCO+DA2 114.
- 분할 점검 도구 `tools/seg_overlay.py`, 비교 `docs/results/seg_overlay_compare.png`(ObjectSAM v1.0 은 장당 38–66 조각·SigLIP 이름 확신 0.04–0.2, COCO 는 5–12 개로 깔끔).
- README 표지 `docs/cover-graph.gif`·`docs/cover-graph.mp4`: 실제 속도 재생(2배속)에서 3D 점 지도·궤적·물체 노드·엣지·방 노드(이름·물체 수)가 자라는 모습.
- 뷰어: 'Object name labels' 따로 끄기(방 이름표는 남김), 방 이름표에 물체 수.

### Changed
- **호환 변경:** 분할 기본값 `--seg` 를 다시 COCO YOLO26s-seg 로(ObjectSAM v1.0 은 재학습 전까지 선택 사항).
- 3D 점 지도 1 cm(3 화소 간격, 최대 150만 점), 지도·물체 점 색은 가장 가까이서 본 관측의 색(평균 색은 뿌옇게 번짐).

### Fixed
- 뷰어가 읽는 동안 기억 파일 교체가 윈도우 파일 잠금에 막혀 재생이 죽던 문제: 임시 파일에 쓰고 잠기면 잠깐 기다렸다 교체.

## [0.3.0] - 2026-11-03

### Added
- TUM RGB-D 입력 `run_memory --tum`(모션캡처 정답 포즈, 경로 평면으로 수평 맞춤). 시험 장면 fr3/long_office_household.
- 깊이 모델 비교 도구 `tools/depth_compare.py`(TUM Kinect 정답 20장): Depth Anything V2 Metric-Indoor-Small(2,480만) 상대 오차 18.8 %, 장면별 배율을 맞추면 5.0 % / YOLO26s-depth(1,320만) 27.4 %, 9.2 %. 두 모델 모두 장면마다 절대 배율이 흔들림(DA2 0.71–1.71, YOLO 0.42–1.02) — 지도·물체가 어긋나는 주원인. 그림 `docs/results/depth_compare.png`, `docs/results/tum_loh_depth_viewer.png`.
- `run_memory --depth <Hugging Face 폴더>`: Depth Anything V2 등 HF 깊이 모델 사용.
- 장면 전체 3D 점 지도 `scene_points.ply`(벽·책상·바닥, 2 cm, 2 키프레임 이상에서 본 칸, 최대 40만 점), 뷰어 '3D map points' 층. 실제 카메라 시점 비교 `docs/results/tum_scene_map.png`.
- 씬그래프 층 `tools/scene_graph.py`: 점유 지도의 빈 공간을 구역으로 나눠 방 노드, 키프레임 SigLIP 2 벡터 평균으로 방 종류 제로샷 이름, 엣지 contains(방→물체)·on(위·아래)·near(물체마다 가까운 2 개). 캡처 `docs/results/tum_loh_graph*.png`.
- 카메라 실제 자세를 `view.json` `cam`(위치 + 회전)에 기록, 뷰어가 높이·기울기 그대로 그림.
- 물체 표면 그물(`--mesh 1`, 가장 크게 본 장면의 마스크+깊이)과 뷰어 'Objects as points/surface' 선택. 깊이 배율이 장면마다 달라 그물이 늘어나므로 기본은 점. 비교 `docs/results/tum_osam_points_vs_surface.png`.

### Changed
- **호환 변경:** 분할 기본값 `--seg` 를 COCO YOLO26s-seg 에서 ObjectSAM v1.0(종류 없음, 416)으로. 이름은 SigLIP 2(`models/siglip2-base-patch32-256`) 배치 분류(물체 55·구조물 10 이름), 구조물 이름이거나 화면 35 % 넘게 덮는 덩어리는 버림, 같은 물체 판정은 3D 겹침(이름은 투표). TUM 책상: 물체 137 — 종이 54·폴더 19 등 책상·바닥 조각이 아직 많이 남음.
- 물체 점: 보여 주는 칸 1.2 cm, 여러 관측 지지는 4 cm 굵은 칸으로 셈(고운 칸으로 세면 깊이 흔들림에 점이 다 버려짐), 물체당 최대 4만 점, 뷰어 점을 둥글게.
- 뷰어: 내 경로를 굵은 원 줄 대신 가는 선 + 0.4 m 간격 점으로.
- 속도: SigLIP 배치, 물체 점 정리는 기록할 때만 — TUM 5.0 → 0.5 s/키프레임(RTX).

### Fixed
- VIO 세계 z 가 중력과 58.7° 어긋나 물체가 비스듬히 최대 2.7 m 높이로 쌓이던 문제: 수평 맞춤 추가. 기본 기준은 걸어간 경로 평면(`--level traj`; IMU 평균 중력과 13.4° 차, 경로 평면 쪽이 GT 높이 변화 0.18 m 에 맞음). 정면 캡처 `docs/results/advio15_vio_front.png`.
- 칸막이 같은 얇고 넓은 수직면 덩어리를 물체로 넣던 문제 일부: `is_wall`(두께 < 8 cm, 폭 > 0.8 m 수직면 제외).

## [0.2.0] - 2026-11-03

### Added
- 손카메라 영상+포즈로 점유 지도를 갱신하고 물체를 3D로 기억하는 추론 파이프라인 `tools/run_memory.py`(YOLO26s-seg + YOLO26s-depth, 사라짐 판정, 시간대별 스냅샷).
- 씬그래프 뷰어 `viewer/sgview`(Rust + three.js, object-memory-robot 에서 가져와 윈도우 MSVC 빌드), 손에 든 카메라 표시(폰 + 시야 원뿔 + 바닥 고리).
- ADVIO(아이폰 카메라+IMU) → EuRoC 형식 변환 `tools/advio_to_euroc.py` 와 ORB-SLAM3 단안-관성 VIO 연결(`--pose vio`).
- 결과 캡처 `docs/results/`: ARKit/VIO 시간대별·3D 화면, 지도/물체 분리, 분할 모델 비교.

### Changed
- 물체 점: 여러 관측에서 지지받은 칸만 남기기, 3D 겹침으로 같은 물체 연관(물체 68 → 45).
- 바닥 높이를 키프레임마다 카메라 높이로 추정, 사라짐 판정 엄격히(관측 5+·깊이 일치·연속 8).
- 점유 지도를 키프레임마다 칸당 한 번만 세기.

### Fixed
- ADVIO 영상 시각이 IMU 보다 0.318 s 늦던 문제(영상 회전↔자이로 맞춤으로 측정) — VIO 가 매번 초기화 실패하던 원인. IMU 잡음 10배로 추적 안정.
  결과: VIO ATE 0.64 m / 경로 17 m(ARKit 같은 구간 0.64 m), 1,551 프레임 중 1,165 추적.

## [0.1.0] - 2026-10-29

### Added
- 프로젝트 계획·하드웨어 결정(일반 USB 카메라 + 납땜 없는 Qwiic IMU, Pi5 + DX-M1), 포즈·지도 방법 문서.
- `train/ObjectSAM`: YOLO26s-seg C++/CUDA 학습기(그래프·Attention·그룹 conv·E2E 손실), Ultralytics 정합성 통과, 46.4 ms/스텝(B16 640).

### Fixed
- ModelCfg 새 필드 순서로 기존 초기화가 깨지던 회귀.

[Unreleased]: https://github.com/juyoung020/ondevice-scene-memory/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/juyoung020/ondevice-scene-memory/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/juyoung020/ondevice-scene-memory/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/juyoung020/ondevice-scene-memory/releases/tag/v0.1.0
