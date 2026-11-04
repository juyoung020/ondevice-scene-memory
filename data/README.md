# 데이터

영상 파일은 저장소에 올리지 않는다. 필요할 때 아래 주소에서 받아 표의 위치에 푼다.

## 공개 데이터셋

| 이름 | 쓰는 곳 | 크기 | 받는 곳 | 넣을 위치 |
|---|---|---|---|---|
| TUM RGB-D fr1/desk | 위치 추정·지도 시험(책상, 23 초, 손에 든 Kinect) | 328 MB | https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk.tgz | `data/tum/rgbd_dataset_freiburg1_desk/` |
| TUM RGB-D fr3/long_office_household | 위치 추정·지도·표지(사무실 22 m, 87 초) | 1.4 GB | https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_long_office_household.tgz | `data/tum/rgbd_dataset_freiburg3_long_office_household/` |
| TUM-VI room1 (512, 16 bit) | 카메라+IMU 위치 추정 시험 | 1.6 GB | https://cvg.cit.tum.de/tumvi/exported/euroc/512_16/dataset-room1_512_16.tar | `data/tum/room1.tar` |
| Digiteo seq2 (RealSense D435i, 연구소 복도 58 m 한 바퀴, 흑백 컬러, 정답 위치) | D435 깊이 + SLAM 시험 | 3.2 GB | https://doi.org/10.17632/tb9g7th9yz.2 | `data/d435/Digiteo_seq_2/` → `rgbd2tum` 으로 `data/d435/rgbd_dataset_freiburg3_digiteo2/` |
| ADVIO 15 (아이폰 카메라+IMU, 사무실 52 초) | 0.2.0 의 VIO·ARKit 시험 | 77 MB | https://github.com/AaltoVision/ADVIO (advio-15.zip) | `data/advio/advio-15/` |

받는 법(예: 책상):

```bash
mkdir -p data/tum && cd data/tum
curl -LO https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk.tgz
tar -xzf rgbd_dataset_freiburg1_desk.tgz
```

TUM RGB-D 전체 목록: https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download

## 우리가 찍은 것

| 이름 | 내용 | 있는 곳 |
|---|---|---|
| `rec/<날짜_시각>/` | 웹캠(ABKO APC930) 녹화, `--rec` 로 남김. `rgb.txt` + `rgb/*.jpg` | 보드 `~/om/data/rec/`, PC `data/rec/` |
| `pc_test/run1.avi` | PC 웹캠 첫 시험 영상 | PC `data/pc_test/` |

녹화에는 주변 사람이 찍혀 있을 수 있어 저장소에 올리지 않는다.

## 다시 만들 수 있는 것

| 이름 | 내용 |
|---|---|
| `memory/<이름>/` | `memmap --out` 실행 결과(view.json·지도·물체 파일) |
| `euroc/advio15_*` | ADVIO 15 를 EuRoC 형식으로 바꾼 것(0.2.0 때 ORB-SLAM3 입력) |
