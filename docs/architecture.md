# 동작 원리

## 처리 흐름

```
카메라 한 장
  ├─ NPU ① 분할 (YOLO26s-seg)   → 물체 마스크·이름
  ├─ NPU ② 깊이 (YOLO26s-depth) → 화소마다 거리
  └─ NPU ③ 특징점 (SuperPoint)  → 앞 장면과 맞출 점
        ▼ CPU (Rust)
  위치 추정: 특징점(SuperPoint) 또는 RGB-D 직접 방식(웹캠 데모) → 카메라 위치·방향
        ▼
  누적 지도: 깊이를 카메라 위치로 옮겨 3D 점·점유 지도에 더함(새 점만)
            --surface 면 TSDF(3 cm 칸, 표면 근처 블록만)에 누적 평균 → Surface Nets 로 면(바뀐 96 cm 구역만)
  물체 트래커: 같은 자리·같은 이름이면 같은 물체로 합침, 안 보이면 "사라짐"
  병합 후처리(15 장마다): 따로 생긴 두 물체를 3D IoU·겹침 비율·이름 분포·색 분포로 판단해 합침
        ▼
  씬그래프: 빈 공간으로 방 나누기 → 방→물체, 물체 위·옆 관계, 방 종류는 방 안 물체 구성으로(예: 노트북·키보드·마우스 → office)
        ▼
  기억 폴더(view.json · 점 파일 · 물체 사진) → 웹 뷰어
```

- 작업 스레드가 다음 장면의 읽기와 모델 실행을 한 장 앞서 처리하고, 본 스레드가 위치·지도·트래커를 갱신한다. 실시간 모드에서는 파일 쓰기와 PLY 짜기·PNG 압축을 기록 스레드가 맡는다.
- `--fast-pose`: 작업 스레드는 카메라 장면을 모두 본 스레드로 보내 위치부터 갱신하고(직접 방식, 마지막 깊이 장면 기준), 모델 스레드가 쉬고 있으면 그 장면을 맡긴다. 모델 결과(깊이·분할)는 늦게 도착해 이미 구한 위치로 지도·물체에 더해지고 새 기준이 된다.
- 병렬 계산(직접 방식 점 합산, 표면 지도, 특징점 짝 맞추기, 점 목록)은 상주 스레드 풀(rayon).
- 모델 셋은 공통 인터페이스(`Segmenter`·`DepthNet`·`FeatNet`)로 묶여, 모델 파일이 `.onnx` 면 PC(ONNX Runtime), `.dxnn` 이면 보드 NPU(DXRT C API)에서 돈다.
- 기록은 바뀐 것만: 3D 지도는 새 칸만 조각 파일로, 물체 사진은 더 잘 보였을 때만, 물체 점 파일은 점이 5 % 넘게 늘었을 때만, 카메라 위치(pose.json)는 장면마다, 지도·물체 목록(view.json)은 0.4 초마다, 실시간 3D 지도 조각·표면은 1 초마다.
- 병합 기준(HOV-SG·ConceptGraphs 방식): 4 cm 칸 IoU ≥ 0.5, 또는 작은 쪽 기준 겹침 ≥ 0.6 이면서 이름 투표 코사인 ≥ 0.5 이거나 색 히스토그램 코사인 ≥ 0.8.
- 위치 추정과 지도 방법의 자세한 내용: [pose_and_map.md](pose_and_map.md).
- RTAB-Map 위치를 장면마다 받는 연결 프로그램(`bridge/`, 깊이 카메라 → RTAB-Map → `memmap --stream`): [bridge.md](bridge.md), 보드 설치 [board_bridge.md](board_bridge.md).

## 폴더

| 폴더 | 내용 |
|---|---|
| `pipeline/` | 기억을 만드는 프로그램(Rust, memmap) |
| `pipeline/src/vo.rs` | 위치 추정 |
| `pipeline/src/maps.rs` | 누적 지도·물체 트래커 |
| `pipeline/src/tsdf.rs` | 표면 지도(TSDF 융합·Surface Nets) |
| `pipeline/src/dvo.rs` | 특징점 없는 RGB-D 직접 위치 추정 |
| `pipeline/src/graph.rs` | 방 나누기·씬그래프 연결 |
| `pipeline/src/seg.rs` | 모델 실행(ONNX·NPU) |
| `pipeline/src/stream.rs` | 장면 스트림 입력(`--stream`, 연결 프로그램이 보냄) |
| `bridge/` | 연결 프로그램 `rtab_bridge`(C++, RealSense·TUM → RTAB-Map → 장면 스트림) |
| `viewer/sgview/` | 웹 뷰어(Rust 서버 + three.js) |
| `train/ObjectSAM/` | 자체 분할 모델 학습기(C++/CUDA) |
| `docs/` | 설계·측정 결과·평가 계획 |
| `models/`, `data/` | 모델·영상·기억 저장본(git 밖) |

## 개발 환경

- 저장소 경로는 공백·한글 없이(`C:\ondevice`). DEEPX 도구가 이런 경로에서 깨진다.
- 빌드 기능: PC `--features directml`(그래픽카드) 또는 기본(CPU), 보드 `--no-default-features --features dxrt,cam`.
- 평가 표: `pipeline/target/release/bench [--quick]` — TUM 책상·사무실을 정해진 조건으로 돌려 위치 오차·실패·속도·물체 수를 한 표로(보드는 `--depth <.dxnn> --extra "--seg <.dxnn>"`). 표면 지도 비교: `surfcmp <A> <B>`.
- 다른 카메라 RGB-D 를 TUM 형식으로: `rgbd2tum <원본> <출력(이름에 freiburg3)> fx fy cx cy <깊이 1 m 값>` — TUM fr3 보정값 640×480 으로 다시 그려 RTAB-Map `rtabmap-rgbd_dataset` 과 memmap 이 같이 읽음. 다른 SLAM 위치로 돌리기: `memmap --tum <폴더> --gt <위치 파일>`, 정렬 후 위치 오차: `ate <추정> <정답>`(memmap 은 `poses_est.txt`·`poses_map.txt` 를 남김), 세계 좌표가 다른 두 실행의 표면 비교: `surfcmp <A> <B> --align`.
