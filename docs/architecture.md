# 동작 원리

## 처리 흐름

```
RealSense D435 (컬러 + 깊이, 640×480@30)
        ▼ rtab_bridge (C++)
  RTAB-Map: 장면마다 위치 추적, 루프 클로저로 지도 노드 위치 다시 맞춤(BRIEF + GTSAM)
  → 장면(컬러 + 컬러에 맞춘 깊이 + 위치 + 기준 노드) · 그래프 블록(움직인 노드)을 스트림으로
        ▼ memmap --stream (Rust)
  NPU 분할 (YOLO26s-seg)         → 물체 마스크·이름
  표면 지도: 20 s 단위 TSDF 서브맵(기준 노드에 붙임) → 루프 클로저 때 서브맵만 옮겨 합침 → 메시 + 사진 텍스처
  점 지도(점 작업 스레드): Potree 방식 옥트리 노드
  점유 지도: 바닥 5 cm 칸
  물체 트래커: 같은 자리·같은 이름이면 같은 물체로 합침, 안 보이면 "사라짐"
  병합 후처리(15 장마다): 따로 생긴 두 물체를 3D IoU·겹침 비율·이름 분포·색 분포로 판단해 합침
  루프 클로저 뒤 다시 쌓기(뒤 스레드): 보관한 장면으로 점유 지도·점 지도·물체를 새 위치로
        ▼
  씬그래프: 빈 공간으로 방 나누기 → 방→물체, 물체 위·옆 관계, 방 종류는 방 안 물체 구성으로(예: 노트북·키보드·마우스 → office)
        ▼
  기억 폴더(view.json · 표면 구역 · 점 노드 · 물체 사진) → 웹 뷰어(변화를 SSE 로 밀어 줌)
```

- 본 스레드는 위치·트래커·서브맵 쌓기만 하고, 점 지도·텍스처·다시 쌓기·파일 쓰기는 각자 스레드에서 돈다(`--threads` 로 우선순위·코어 배치).
- 처리가 0.2 s 넘게 밀리면 쌓인 장면을 건너뛰고 최신 장면부터 처리한다. 그래프 블록은 버리지 않는다(`--no-drop` 이면 모두 처리).
- 분할 모델은 공통 인터페이스(`Segmenter`)로, 모델 파일이 `.onnx` 면 PC(ONNX Runtime), `.dxnn` 이면 보드 NPU(DXRT C API)에서 돈다.
- 스트림 입력의 분할은 세 단계(전처리 → 추론 → 후처리)를 스레드 셋이 장면을 한 장씩 차례로 넘기며 겹쳐 돌린다: 추론이 도는 동안 앞 장면의 마스크 해독과 다음 장면의 읽기·전처리가 돈다. 보드 NPU 에는 장면을 4 장까지 함께 올려(`--npu-jobs`, DXRT 비동기 실행) NPU 코어 여럿이 나눠 돌고, 결과는 올린 순서대로 받아 장면 순서는 그대로다(`--seg-pipe off` 는 한 스레드에서 차례로).
- 병합 기준(HOV-SG·ConceptGraphs 방식): 4 cm 칸 IoU ≥ 0.5, 또는 작은 쪽 기준 겹침 ≥ 0.6 이면서 이름 투표 코사인 ≥ 0.5 이거나 색 히스토그램 코사인 ≥ 0.8.
- 연결 프로그램·스트림 형식·서브맵·루프 클로저: [bridge.md](bridge.md), 보드 설치 [board_bridge.md](board_bridge.md).

## 폴더

| 폴더 | 내용 |
|---|---|
| `pipeline/` | 기억을 만드는 프로그램(Rust, memmap) |
| `pipeline/src/stream.rs` | 장면 스트림 입력(`--stream`) |
| `pipeline/src/kfstore.rs` | 다시 쌓기용 장면 보관(압축) |
| `pipeline/src/submap.rs` | 표면 서브맵 |
| `pipeline/src/tsdf.rs` | 표면 지도(TSDF 융합·Surface Nets) |
| `pipeline/src/texture.rs` | 표면 사진 텍스처 |
| `pipeline/src/maps.rs` | 누적 지도·물체 트래커 |
| `pipeline/src/graph.rs` | 방 나누기·씬그래프 연결 |
| `pipeline/src/seg.rs` | 분할 모델 실행(ONNX·NPU) |
| `bridge/` | 연결 프로그램 `rtab_bridge`(C++, RealSense·TUM → RTAB-Map → 장면 스트림) |
| `viewer/sgview/` | 웹 뷰어(Rust 서버 + three.js) |
| `train/ObjectSAM/` | 자체 분할 모델 학습기(C++/CUDA) |
| `docs/` | 설계·측정 결과·평가 계획 |
| `models/`, `data/` | 모델·영상·기억 저장본(git 밖) |

## 개발 환경

- 저장소 경로는 공백·한글 없이(`C:\ondevice`). DEEPX 도구가 이런 경로에서 깨진다.
- 빌드 기능: PC `--features directml`(윈도우 그래픽카드) · `--features cuda`(리눅스 NVIDIA) 또는 기본(CPU), 보드 `--no-default-features --features dxrt,cam`.
- 평가 표: `pipeline/target/release/bench [--quick]` — TUM 책상·사무실을 정해진 조건으로 돌려 위치 오차·실패·속도·물체 수를 한 표로. 표면 지도 비교: `surfcmp <A> <B>`.
- 다른 카메라 RGB-D 를 TUM 형식으로: `rgbd2tum <원본> <출력(이름에 freiburg3)> fx fy cx cy <깊이 1 m 값>` — TUM fr3 보정값 640×480 으로 다시 그려 RTAB-Map `rtabmap-rgbd_dataset` 과 memmap 이 같이 읽음. 다른 SLAM 위치로 돌리기: `memmap --tum <폴더> --gt <위치 파일>`, 정렬 후 위치 오차: `ate <추정> <정답>`(memmap 은 `poses_est.txt`·`poses_map.txt` 를 남김), 세계 좌표가 다른 두 실행의 표면 비교: `surfcmp <A> <B> --align`.
