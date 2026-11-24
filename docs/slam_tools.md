# 가져다 쓸 SLAM · 재위치 도구 비교 — 2026-11-25

우리 센서(D435 적외선 스테레오 · 컬러 · 깊이 + 외부 ICM-42688-P)에서 [`pose_issues.md`](pose_issues.md)의 문제점을 품 적게 푸는 기존 도구. 가성비 기준(무료 · CPU · 붙이기 쉬움 우선), 연산 장치 제약은 두지 않음. 조사만 했고 아직 돌려 보지 않음.

## 문제점별 대응

◎ 해결 · △ 부분 · ✕ 안 됨 · ? 확인 못 함 · — 해당 없음

| 문제점 | ORB-SLAM3 스테레오 | ORB-SLAM3 스테레오+IMU | cuVSLAM 스테레오 | RTAB-Map 원본 + 스테레오 추적 | Basalt · OpenVINS |
|---|---|---|---|---|---|
| 놓친 뒤 기존 지도에 재위치 | ◎ | ◎ | △ | △ | ✕ |
| 다중 지도 합치기 | ◎ | ◎ | ? | ◎ | ✕ |
| 중력 · 기울기 표류 | △ | ◎ | △ | ◎ | ◎ |
| 루프 보정 0 | ◎ | ◎ | ◎ | ◎ | ✕ |
| 같은 물체 중복 | △ | △ | △ | △ | ✕ |
| 번짐 · 롤링 셔터 | ◎ | ◎ | ◎ | ◎ | ◎ |
| 카메라–IMU 시각차 | — | ✕ | — | ◎ | OpenVINS ◎ · Basalt ✕ |
| 녹화 빈틈 · USB 끊김 | △ | ✕ | ? | △ | ✕ |
| 먼 깊이 | △ | △ | ? | △ | △ |

- 재위치: ORB-SLAM3 는 회전 · 크기 불변 ORB 낱말로 전체 키프레임에서 찾아 PnP 검증. RTAB-Map 원본은 우리 포팅판과 같은 구조라 BRIEF 변별 약점이 남음. cuVSLAM 은 장소 인식 기반 `localize_in_map` 을 우리가 불러 줘야 함.
- 합치기: ORB-SLAM3 Atlas 는 놓치면 새 지도를 열고 모든 지도를 상대로 장소 인식을 계속, 맞으면 합쳐 다시 최적화.
- 중력: 스테레오 전용은 추정기에 중력이 없음 → 시작 때 IMU 중력으로 지도 전체를 한 번 돌림(우리가 붙임).
- 물체 중복: 루프 · 합치기 뒤 바뀐 키프레임 자세를 물체 지도에 다시 반영하는 연결은 어느 도구든 우리가 만듦.
- 번짐 · 롤링 셔터는 입력을 적외선(글로벌 셔터)으로 바꿔서 풀림. 빈틈은 녹화기(램 디스크 · SSD)로 고침.
- 먼 깊이: ORB-SLAM3 는 기준선 약 40 배(D435 약 2 m)보다 먼 점을 삼각측량으로 다룸. 물체 지도 깊이 오차는 D455 로만 줄어듦.

## 추천

1. **ORB-SLAM3 적외선 스테레오 + Atlas**(IMU 는 중력 맞추기에만). 재위치 · 합치기 · 루프 보정 내장 → `pose_issues.md` "다시 붙게 하는 순서" 1–8 을 만들 필요 없음. 시각차 · IMU 재보정 없음, 보정은 카메라 공장값. 무료(GPLv3) · CPU. 우리가 만들 것: 시작 중력 맞추기, 바뀐 키프레임 자세를 물체 지도에 반영. 소스 WSL `~/slam-bench/src/ORB_SLAM3`.
2. **cuVSLAM 적외선 스테레오**. 설치 가장 쉬움, 다음 날 재위치(초기 추정 없음) 강점, NVIDIA GPU 필수(가벼움), 합치기 · 빈틈 뒤 동작 미확인.
3. **RTAB-Map 원본**. 중력 · 시각차는 잘 메우나 재위치 변별 약점 그대로, "RTAB-Map 통째 교체" 방침과 부딪힘.

Basalt · OpenVINS 는 추적 전용(재위치 · 합치기 · 루프 없음). 학습 기반(DROID · MASt3R-SLAM · VGGT-SLAM · Gaussian Splatting)은 큰 GPU · 지도 저장/다음 날 재위치 없음으로 제외. Spectacular AI 는 D435(i 없음) 미지원.

| 후보 | 비용 · 하드웨어 | 근거 |
|---|---|---|
| [ORB-SLAM3](https://arxiv.org/pdf/2007.11898) | 무료(GPLv3) · CPU | [제3자 표](https://arxiv.org/pdf/2610.07217) EuRoC 3.5 cm · TUM-VI 방 0.9 cm, [Atlas 저장](https://www.mintlify.com/UZ-SLAMLab/ORB_SLAM3/guides/map-save-load), IMU 쓰면 [초기화 실패 보고](https://github.com/UZ-SLAMLab/ORB_SLAM3/issues/69) |
| [cuVSLAM](https://github.com/nvidia-isaac/cuVSLAM) | 무료([NVIDIA 기기 한정](https://raw.githubusercontent.com/NVlabs/PyCuVSLAM/main/LICENSE)) · NVIDIA GPU | [논문](https://arxiv.org/html/2506.04359v3) EuRoC 0.054, [IMU 융합은 스테레오 품질을 올리지 않음](https://forums.developer.nvidia.com/t/cuvslam-imu-fusion-is-not-improving-performance/343427) |
| [RTAB-Map 원본](https://arxiv.org/pdf/2403.06341) | 무료(BSD) · CPU | 자체 추적이 [D435i 에서 자주 놓침](https://answers.ros.org/question/376539/rtabmap-visual-odometry-getting-lost-too-often-using-d435i) |
| [Basalt](https://arxiv.org/pdf/1904.06504.pdf) · OpenVINS | 무료 · CPU | Basalt [지도 오프라인](https://gitlab.com/VladyslavUsenko/basalt/-/issues/69), OpenVINS [2025 비교](https://arxiv.org/pdf/2507.11241) 평균 오차 최저 |

## 센서에서 바꿀 것

- 녹화 · 추적 입력에 적외선 좌우 추가, 시험 녹화는 프로젝터 끔(점 무늬가 추적 방해)
- IMU 는 추적에서 빼고 중력 맞추기에만
- 녹화는 램 디스크 · SSD 에
- D455 교체는 물체 지도의 먼 깊이 오차가 문제로 남을 때

## 시험

- 녹화: 적외선 좌우(프로젝터 끔, 640×480, 30 fps, 노출 상한 4–8 ms) + IMU EuRoC 꼴 + 컬러 · 깊이, 빈틈 0.
  - A: 2–3 초 정지 → 방 한 바퀴 → 처음 자세로 끝, 중간에 렌즈를 3–5 초 가렸다가 다른 방향에서 뗌
  - B: 다음 날 같은 방, 다른 시작 위치
  - 지금 있는 적외선 녹화: GitHub 릴리즈 `recordings-v2`(`ir_loop_off` · `ir_loop_on` 방 한 바퀴, `ir_off1/2`, `ir_on1`)
- ORB-SLAM3 스테레오를 WSL 별도 폴더에 빌드해 A 를 돌리고 Atlas 저장 → B 를 그 Atlas 로 시작.
- 판정: 끝–시작 위치 · 각도 차, 천장 면 기울기(목표 3° 미만), 가린 뒤 같은 지도로 돌아왔는지 · 새 지도가 합쳐졌는지, B 가 A 지도에 붙었는지.
